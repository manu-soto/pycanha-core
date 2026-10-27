// GeometryModel::mesh_parts / material_table — the GMM entry points feeding
// the radiative raytracer (rigid-part split + per-face material tables).

#include <spdlog/spdlog.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/gmm/materials/optical_material.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/gmm/scene/geometry.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"
#include "pycanha-core/gmm/scene/resolve.hpp"
#include "pycanha-core/radiative/materials.hpp"
#include "pycanha-core/radiative/scene_part.hpp"
#include "pycanha-core/utils/logger.hpp"
#include "pycanha-core/utils/parallel_for.hpp"

namespace pycanha::gmm {

namespace {

// One rigid part under assembly. Its pieces carry GLOBAL face offsets, so a
// part's face ids are a subset of the model mesh's rather than a numbering of
// its own.
struct PartBuild {
    std::vector<TriMeshD> pieces;
    std::vector<pycanha::MeshIndex> face_offsets;
    CoordinateTransformation transform;  // part -> world
    radiative::PartKind kind = radiative::PartKind::Spacecraft;
    bool encountered = false;
};

// A cutter reaching across a rigid-part boundary is subtracted at resolution
// time, so the hole travels with the part it was cut into and moving the parts
// apart will not close it. Legitimate for a fixed configuration, wrong for an
// articulated one, and invisible unless said out loud.
void report_cross_part_cutters(const detail::ResolvedTarget& target) {
    const bool crosses = std::ranges::any_of(
        target.cutters, [&target](const detail::ResolvedCutter& cutter) {
            return cutter.part != target.part;
        });
    if (crosses) {
        SPDLOG_LOGGER_WARN(
            pycanha::get_logger(),
            "mesh_parts: '{}' is cut by a cutter belonging to another rigid "
            "part; the subtraction is baked in, so moving the parts apart "
            "will not close the hole",
            target.item.get().name());
    }
}

}  // namespace

std::vector<radiative::ScenePart> GeometryModel::mesh_parts(
    std::span<const std::string> split) const {
    // Part 0 is the remainder (world frame, identity transform); one part per
    // split name follows in the given order.
    std::vector<PartBuild> parts(split.size() + 1);
    parts[0].encountered = true;

    std::unordered_map<const Geometry*, std::size_t> frame_breaks;
    for (std::size_t idx = 0; idx < split.size(); ++idx) {
        const auto target = find(split[idx]);
        if (target == nullptr) {
            throw std::invalid_argument(
                "GeometryModel::mesh_parts: unknown split name '" + split[idx] +
                "'");
        }
        if (!frame_breaks.emplace(target.get(), idx).second) {
            throw std::invalid_argument(
                "GeometryModel::mesh_parts: duplicate split name '" +
                split[idx] + "'");
        }
        parts[idx + 1].kind = radiative::PartKind::Articulated;
    }

    // The same walk the model mesh is built from. Resolving each part on its
    // own would give a cut group inside it a different cutter set from the one
    // it gets in the model, and the parts would stop partitioning the model's
    // faces.
    const auto targets =
        detail::collect_targets(*_root, _root->transform(), frame_breaks);

    // Meshed as resolve_subtree() meshes the model: an uncut item fills its
    // cached mesh, which mutates the item, so those stay on this thread; the
    // boolean cuts, where the time goes, run across workers, each writing only
    // its own piece, so the worker count cannot change the result.
    std::vector<TriMeshD> meshes(targets.size());
    std::vector<std::size_t> cut_targets;
    for (std::size_t index = 0; index < targets.size(); ++index) {
        if (targets[index].cutters.empty()) {
            meshes[index] = detail::mesh_target(targets[index]);
        } else {
            cut_targets.push_back(index);
        }
    }
    const unsigned workers =
        cut_targets.size() > 1U
            ? std::min<unsigned>(std::thread::hardware_concurrency(),
                                 static_cast<unsigned>(cut_targets.size()))
            : 1U;
    utils::parallel_for_index(
        cut_targets.size(), workers, [&](std::size_t index) {
            meshes[cut_targets[index]] =
                detail::mesh_target(targets[cut_targets[index]]);
        });

    pycanha::MeshIndex face_offset = 0;
    for (std::size_t index = 0; index < targets.size(); ++index) {
        const auto& target = targets[index];
        PartBuild& part = parts[target.part];
        part.encountered = true;
        part.transform = target.part_to_root;
        report_cross_part_cutters(target);

        part.pieces.push_back(std::move(meshes[index]));
        part.face_offsets.push_back(face_offset);
        face_offset += part.pieces.back().nf();
    }

    for (std::size_t idx = 0; idx < split.size(); ++idx) {
        if (!parts[idx + 1].encountered) {
            throw std::invalid_argument(
                "GeometryModel::mesh_parts: split name '" + split[idx] +
                "' contributes no geometry and cannot form a rigid part");
        }
    }

    std::vector<radiative::ScenePart> result;
    result.reserve(parts.size());
    for (std::size_t idx = 0; idx < parts.size(); ++idx) {
        if (idx == 0 && !split.empty() && parts[idx].pieces.empty()) {
            continue;  // omit an empty remainder part
        }
        result.push_back(radiative::ScenePart{
            .mesh = detail::concatenate_at(parts[idx].pieces,
                                           parts[idx].face_offsets)
                        .cast<float>(),
            .transform = parts[idx].transform,
            .kind = parts[idx].kind,
            .part_id = static_cast<std::uint32_t>(result.size())});
    }
    return result;
}

radiative::MaterialTable GeometryModel::material_table() const {
    const TriMeshF& model_mesh = mesh();
    const auto num_faces = static_cast<Eigen::Index>(model_mesh.nf());

    radiative::MaterialTable table;
    table.face_material = Eigen::VectorXi::Constant(num_faces, -1);
    table.face_active =
        Eigen::Matrix<bool, Eigen::Dynamic, 1>::Constant(num_faces, false);

    std::vector<const OpticalMaterial*> unique_materials;
    std::unordered_map<const OpticalMaterial*, int> material_row;
    const auto row_of = [&](const std::shared_ptr<OpticalMaterial>& material) {
        if (material == nullptr) {
            return -1;
        }
        const auto [it, inserted] = material_row.emplace(
            material.get(), static_cast<int>(unique_materials.size()));
        if (inserted) {
            unique_materials.push_back(material.get());
        }
        return it->second;
    };

    // Ranges are processed in order and never overlap: an item reserves the
    // faces it owns whether or not a cut left any triangles on them.
    for (const auto& range : model_mesh.primitives) {
        const auto node_it =
            _by_id.find(static_cast<std::uint64_t>(range.geometry_id));
        const auto item =
            node_it == _by_id.end()
                ? nullptr
                : std::dynamic_pointer_cast<GeometryItem>(node_it->second);
        if (item == nullptr) {
            SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                               "material_table: mesh range without a "
                               "registered GeometryItem (id {})",
                               static_cast<std::uint64_t>(range.geometry_id));
            continue;
        }

        const ThermalMesh& thermal_mesh = item->thermal_mesh();
        const int side1_row = row_of(thermal_mesh.get_side1_optical());
        const int side2_row = row_of(thermal_mesh.get_side2_optical());
        if (side1_row < 0 || side2_row < 0) {
            SPDLOG_LOGGER_WARN(
                pycanha::get_logger(),
                "material_table: item '{}' has no optical "
                "material on side {} (faces treated as "
                "blackbody)",
                item->name(),
                side1_row < 0 ? (side2_row < 0 ? "1/2" : "1") : "2");
        }

        const auto first = static_cast<Eigen::Index>(range.first_face_id);
        const auto last = std::min(
            static_cast<Eigen::Index>(range.last_face_id), num_faces - 2);
        for (Eigen::Index face = first; face <= last; face += 2) {
            table.face_material[face] = side1_row;
            table.face_material[face + 1] = side2_row;
            // The raytracer table is radiative-only: a conductive-only side is
            // inactive here even though it still carries a node.
            table.face_active[face] = thermal_mesh.is_radiative_active(1U);
            table.face_active[face + 1] = thermal_mesh.is_radiative_active(2U);
        }
    }

    table.properties.resize(static_cast<Eigen::Index>(unique_materials.size()),
                            6);
    for (std::size_t row = 0; row < unique_materials.size(); ++row) {
        const auto& props = unique_materials[row]->get_th_optical_properties();
        for (int dof = 0; dof < 6; ++dof) {
            table.properties(static_cast<Eigen::Index>(row), dof) =
                static_cast<float>(props.at(static_cast<std::size_t>(dof)));
        }
    }
    return table;
}

}  // namespace pycanha::gmm
