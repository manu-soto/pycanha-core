#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/primitives/cube.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/gmm/scene/geometry.hpp"
#include "pycanha-core/gmm/scene/geometry_group.hpp"
#include "pycanha-core/gmm/scene/geometry_group_cutted.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"
#include "pycanha-core/radiative/scene_part.hpp"

namespace {

using pycanha::gmm::CoordinateTransformation;
using pycanha::gmm::GeometryGroup;
using pycanha::gmm::GeometryItem;
using pycanha::gmm::GeometryModel;
using pycanha::gmm::Rectangle;
using pycanha::gmm::ThermalMesh;
using pycanha::gmm::TriMeshF;
using pycanha::radiative::PartKind;
using pycanha::radiative::ScenePart;

[[nodiscard]] std::shared_ptr<GeometryItem> make_panel(
    const std::string& name, CoordinateTransformation transform = {}) {
    // Two dir1 face pairs -> two face pairs (4 faces) per panel.
    ThermalMesh thermal_mesh{{0.0, 0.5, 1.0}, {0.0, 1.0}};
    return std::make_shared<GeometryItem>(
        name, Rectangle({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}),
        std::move(thermal_mesh), std::move(transform));
}

// Per-triangle face ids of a mesh, sorted (multiset semantics).
[[nodiscard]] std::vector<pycanha::MeshIndex> sorted_face_ids(
    const TriMeshF& mesh) {
    std::vector<pycanha::MeshIndex> ids;
    ids.reserve(static_cast<std::size_t>(mesh.face_ids.rows()));
    for (pycanha::Index row = 0; row < mesh.face_ids.rows(); ++row) {
        ids.push_back(mesh.face_ids(row));
    }
    std::ranges::sort(ids);
    return ids;
}

}  // namespace

TEST_CASE("mesh_parts: single part when no split", "[gmm][parts]") {
    GeometryModel model("scene");
    model.add(make_panel("body"));
    model.add(make_panel(
        "lid", CoordinateTransformation::from_translation({0.0, 0.0, 1.0})));

    const auto parts = model.mesh_parts();
    const TriMeshF& full = model.mesh();

    REQUIRE(parts.size() == 1);
    REQUIRE(parts[0].kind == PartKind::Spacecraft);
    REQUIRE(parts[0].part_id == 0);
    REQUIRE(parts[0].transform.is_identity());
    REQUIRE(parts[0].mesh.vertices.rows() == full.vertices.rows());
    REQUIRE(parts[0].mesh.triangles.rows() == full.triangles.rows());
    REQUIRE(parts[0].mesh.vertices.cwiseEqual(full.vertices).all());
    REQUIRE(parts[0].mesh.triangles.cwiseEqual(full.triangles).all());
    REQUIRE(parts[0].mesh.face_ids.cwiseEqual(full.face_ids).all());
}

TEST_CASE("mesh_parts: split group into its own part", "[gmm][parts]") {
    GeometryModel model("scene");
    model.add(make_panel("body"));
    const CoordinateTransformation wing_tf =
        CoordinateTransformation::from_translation({5.0, 0.0, 0.0});
    model.add(std::make_shared<GeometryGroup>(
        "wing", std::vector<std::shared_ptr<pycanha::gmm::Geometry>>{},
        wing_tf));
    model.add(make_panel("wing_panel"), "wing");

    const std::vector<std::string> split{"wing"};
    const auto parts = model.mesh_parts(split);
    const TriMeshF& full = model.mesh();

    REQUIRE(parts.size() == 2);
    REQUIRE(parts[0].kind == PartKind::Spacecraft);
    REQUIRE(parts[1].kind == PartKind::Articulated);
    REQUIRE(parts[1].part_id == 1);

    // Face ids stay global: the parts' id sets are disjoint and their union
    // is exactly the full model's.
    auto combined = sorted_face_ids(parts[0].mesh);
    const auto wing_ids = sorted_face_ids(parts[1].mesh);
    combined.insert(combined.end(), wing_ids.begin(), wing_ids.end());
    std::ranges::sort(combined);
    REQUIRE(combined == sorted_face_ids(full));

    // The wing part is in its local frame (untranslated vertices) with the
    // translation carried by part.transform (== wing's world transform).
    REQUIRE(parts[1].mesh.vertices.col(0).maxCoeff() <= 1.0F);
    REQUIRE(parts[1]
                .transform.apply({0.0, 0.0, 0.0})
                .isApprox(pycanha::Point3D{5.0, 0.0, 0.0}));
    // The remainder part is in the world frame already.
    REQUIRE(parts[0].transform.is_identity());
}

TEST_CASE("mesh_parts: nested articulated group", "[gmm][parts]") {
    GeometryModel model("scene");
    const CoordinateTransformation parent_tf =
        CoordinateTransformation::from_translation({0.0, 3.0, 0.0});
    const CoordinateTransformation wing_tf =
        CoordinateTransformation::from_translation({5.0, 0.0, 0.0});
    model.add(std::make_shared<GeometryGroup>(
        "assembly", std::vector<std::shared_ptr<pycanha::gmm::Geometry>>{},
        parent_tf));
    model.add(
        std::make_shared<GeometryGroup>(
            "wing", std::vector<std::shared_ptr<pycanha::gmm::Geometry>>{},
            wing_tf),
        "assembly");
    model.add(make_panel("wing_panel"), "wing");

    const std::vector<std::string> split{"wing"};
    const auto parts = model.mesh_parts(split);

    REQUIRE(parts.size() == 1);  // empty remainder is omitted
    REQUIRE(parts[0].kind == PartKind::Articulated);
    REQUIRE(parts[0].part_id == 0);
    // Part transform composes parent x wing; vertices stay wing-local.
    REQUIRE(parts[0]
                .transform.apply({0.0, 0.0, 0.0})
                .isApprox(pycanha::Point3D{5.0, 3.0, 0.0}));
    REQUIRE(parts[0].mesh.vertices.col(0).maxCoeff() <= 1.0F);
    REQUIRE(parts[0].mesh.vertices.col(1).maxCoeff() <= 1.0F);
}

TEST_CASE("mesh_parts: unknown split name throws", "[gmm][parts]") {
    GeometryModel model("scene");
    model.add(make_panel("body"));

    const std::vector<std::string> unknown{"no_such_group"};
    REQUIRE_THROWS_AS(model.mesh_parts(unknown), std::invalid_argument);

    const std::vector<std::string> duplicated{"body", "body"};
    REQUIRE_THROWS_AS(model.mesh_parts(duplicated), std::invalid_argument);
}

TEST_CASE("mesh_parts: item inside a split group not duplicated",
          "[gmm][parts]") {
    GeometryModel model("scene");
    model.add(make_panel("body"));
    model.add(std::make_shared<GeometryGroup>("wing"));
    model.add(make_panel("wing_panel"), "wing");

    const std::vector<std::string> split{"wing"};
    const auto parts = model.mesh_parts(split);
    const TriMeshF& full = model.mesh();

    const auto total_triangles =
        std::accumulate(parts.begin(), parts.end(), pycanha::Index{0},
                        [](pycanha::Index acc, const ScenePart& part) {
                            return acc + part.mesh.triangles.rows();
                        });
    REQUIRE(total_triangles == full.triangles.rows());
}

TEST_CASE("mesh_parts: item as split target", "[gmm][parts]") {
    GeometryModel model("scene");
    model.add(make_panel("body"));
    model.add(make_panel(
        "probe", CoordinateTransformation::from_translation({2.0, 0.0, 0.0})));

    const std::vector<std::string> split{"probe"};
    const auto parts = model.mesh_parts(split);

    REQUIRE(parts.size() == 2);
    REQUIRE(parts[1].kind == PartKind::Articulated);
    // The item's own transform moves to the part transform; the part mesh is
    // expressed in the item frame.
    REQUIRE(parts[1]
                .transform.apply({0.0, 0.0, 0.0})
                .isApprox(pycanha::Point3D{2.0, 0.0, 0.0}));
    REQUIRE(parts[1].mesh.vertices.col(0).maxCoeff() <= 1.0F + 1e-6F);
}

TEST_CASE("mesh_parts: cut items mesh the same as the model mesh",
          "[gmm][parts][cutting]") {
    // Many cut groups, so their cuts run across workers, between uncut
    // panels, inside and outside a split group: the parts must still hold the
    // model mesh's own triangles, bit for bit, whichever worker made them.
    using pycanha::gmm::Cube;
    using pycanha::gmm::GeometryGroupCutted;
    constexpr int cut_groups = 24;
    GeometryModel model("scene");
    std::vector<std::shared_ptr<pycanha::gmm::Geometry>> wing_members;
    for (int index = 0; index < cut_groups; ++index) {
        const double z = static_cast<double>(index);
        ThermalMesh thermal_mesh{{0.0, 0.25, 0.5, 0.75, 1.0},
                                 {0.0, 0.25, 0.5, 0.75, 1.0}};
        auto plate = std::make_shared<GeometryItem>(
            "plate_" + std::to_string(index),
            Rectangle({0.0, 0.0, z}, {1.0, 0.0, z}, {0.0, 1.0, z}),
            std::move(thermal_mesh));
        const double offset = 0.05 * static_cast<double>(index % 7);
        auto cutter = std::make_shared<GeometryItem>(
            "cutter_" + std::to_string(index),
            Cube({0.3 + offset, 0.4, z}, {0.35, 0.3, 0.5}), ThermalMesh{});
        auto cut = std::make_shared<GeometryGroupCutted>(
            "cut_" + std::to_string(index),
            std::vector<std::shared_ptr<pycanha::gmm::Geometry>>{plate},
            std::vector<std::shared_ptr<GeometryItem>>{cutter});
        if (index % 3 == 0) {
            wing_members.push_back(cut);
        } else {
            model.add(cut);
        }
        model.add(make_panel(
            "panel_" + std::to_string(index),
            CoordinateTransformation::from_translation({2.0, 0.0, z})));
    }
    model.add(std::make_shared<GeometryGroup>("wing", wing_members));

    const TriMeshF& full = model.mesh();
    const auto parts = model.mesh_parts();
    REQUIRE(parts.size() == 1);
    REQUIRE(parts[0].mesh.vertices.rows() == full.vertices.rows());
    REQUIRE(parts[0].mesh.triangles.rows() == full.triangles.rows());
    REQUIRE(parts[0].mesh.vertices.cwiseEqual(full.vertices).all());
    REQUIRE(parts[0].mesh.triangles.cwiseEqual(full.triangles).all());
    REQUIRE(parts[0].mesh.face_ids.cwiseEqual(full.face_ids).all());

    // Split off the group: the two parts partition the same faces.
    const std::vector<std::string> split{"wing"};
    const auto split_parts = model.mesh_parts(split);
    REQUIRE(split_parts.size() == 2);
    std::vector<pycanha::MeshIndex> ids = sorted_face_ids(split_parts[0].mesh);
    const auto wing_ids = sorted_face_ids(split_parts[1].mesh);
    ids.insert(ids.end(), wing_ids.begin(), wing_ids.end());
    std::ranges::sort(ids);
    REQUIRE(ids == sorted_face_ids(full));
}
