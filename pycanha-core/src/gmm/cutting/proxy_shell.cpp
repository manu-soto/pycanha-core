#include "pycanha-core/gmm/cutting/proxy_shell.hpp"

#include <manifold/common.h>
#include <manifold/manifold.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"

namespace pycanha::gmm::cutting {
namespace {

using Edge = std::array<std::uint64_t, 2>;

struct EdgeHash {
    [[nodiscard]] std::size_t operator()(const Edge& edge) const noexcept {
        return static_cast<std::size_t>((edge[0] * 1315423911ULL) + edge[1]);
    }
};

[[nodiscard]] Edge normalized_edge(std::uint64_t lhs, std::uint64_t rhs) {
    return lhs < rhs ? Edge{lhs, rhs} : Edge{rhs, lhs};
}

[[nodiscard]] manifold::MeshGL64 build_proxy_meshgl(const TriMeshD& mesh,
                                                    const ProxyMeta& meta) {
    std::vector<Vector3D> vertex_normals(
        static_cast<std::size_t>(mesh.vertices.rows()), Vector3D::Zero());
    std::unordered_map<Edge, int, EdgeHash> edge_use_counts;
    std::vector<Edge> directed_edges;
    edge_use_counts.reserve(
        static_cast<std::size_t>(mesh.triangles.rows() * 3));
    directed_edges.reserve(static_cast<std::size_t>(mesh.triangles.rows() * 3));

    for (Eigen::Index tri_idx = 0; tri_idx < mesh.triangles.rows(); ++tri_idx) {
        const auto triangle = mesh.triangles.row(tri_idx);
        const Vector3D p0 = mesh.vertices.row(triangle[0]).transpose();
        const Vector3D p1 = mesh.vertices.row(triangle[1]).transpose();
        const Vector3D p2 = mesh.vertices.row(triangle[2]).transpose();
        const Vector3D normal = (p1 - p0).cross(p2 - p0);
        if (normal.norm() > LENGTH_TOL) {
            vertex_normals[static_cast<std::size_t>(triangle[0])] += normal;
            vertex_normals[static_cast<std::size_t>(triangle[1])] += normal;
            vertex_normals[static_cast<std::size_t>(triangle[2])] += normal;
        }

        const auto a = static_cast<std::uint64_t>(triangle[0]);
        const auto b = static_cast<std::uint64_t>(triangle[1]);
        const auto c = static_cast<std::uint64_t>(triangle[2]);
        ++edge_use_counts[normalized_edge(a, b)];
        ++edge_use_counts[normalized_edge(b, c)];
        ++edge_use_counts[normalized_edge(c, a)];
        directed_edges.push_back({a, b});
        directed_edges.push_back({b, c});
        directed_edges.push_back({c, a});
    }

    for (Vector3D& normal : vertex_normals) {
        if (normal.norm() <= LENGTH_TOL) {
            normal = Vector3D::UnitZ();
        } else {
            normal.normalize();
        }
    }

    manifold::MeshGL64 proxy;
    proxy.numProp = 3U;
    proxy.vertProperties.reserve(
        static_cast<std::size_t>(mesh.vertices.rows() * 2 * 3));
    for (Eigen::Index vertex_idx = 0; vertex_idx < mesh.vertices.rows();
         ++vertex_idx) {
        const Vector3D vertex = mesh.vertices.row(vertex_idx).transpose();
        const Vector3D offset =
            0.5 * meta.thickness *
            vertex_normals[static_cast<std::size_t>(vertex_idx)];
        const Vector3D outer = vertex + offset;
        proxy.vertProperties.insert(proxy.vertProperties.end(),
                                    {outer.x(), outer.y(), outer.z()});
    }
    for (Eigen::Index vertex_idx = 0; vertex_idx < mesh.vertices.rows();
         ++vertex_idx) {
        const Vector3D vertex = mesh.vertices.row(vertex_idx).transpose();
        const Vector3D offset =
            0.5 * meta.thickness *
            vertex_normals[static_cast<std::size_t>(vertex_idx)];
        const Vector3D inner = vertex - offset;
        proxy.vertProperties.insert(proxy.vertProperties.end(),
                                    {inner.x(), inner.y(), inner.z()});
    }

    const auto inner_offset = static_cast<std::uint64_t>(mesh.vertices.rows());
    const std::uint32_t inner_original_id = meta.outer_original_id + 1U;
    const std::uint32_t wall_original_id = meta.outer_original_id + 2U;

    std::size_t outer_triangles = 0U;
    std::size_t inner_triangles = 0U;
    std::size_t wall_triangles = 0U;

    for (Eigen::Index tri_idx = 0; tri_idx < mesh.triangles.rows(); ++tri_idx) {
        const auto triangle = mesh.triangles.row(tri_idx);
        proxy.triVerts.insert(proxy.triVerts.end(),
                              {static_cast<std::uint64_t>(triangle[0]),
                               static_cast<std::uint64_t>(triangle[1]),
                               static_cast<std::uint64_t>(triangle[2])});
        proxy.faceID.push_back(mesh.face_ids(tri_idx));
        ++outer_triangles;
    }

    for (Eigen::Index tri_idx = 0; tri_idx < mesh.triangles.rows(); ++tri_idx) {
        const auto triangle = mesh.triangles.row(tri_idx);
        proxy.triVerts.insert(
            proxy.triVerts.end(),
            {static_cast<std::uint64_t>(triangle[0]) + inner_offset,
             static_cast<std::uint64_t>(triangle[2]) + inner_offset,
             static_cast<std::uint64_t>(triangle[1]) + inner_offset});
        proxy.faceID.push_back(mesh.face_ids(tri_idx));
        ++inner_triangles;
    }

    for (const Edge edge : directed_edges) {
        if (edge_use_counts[normalized_edge(edge[0], edge[1])] != 1) {
            continue;
        }

        const auto a = edge[0];
        const auto b = edge[1];
        proxy.triVerts.insert(proxy.triVerts.end(), {b, a, a + inner_offset});
        proxy.triVerts.insert(proxy.triVerts.end(),
                              {b, a + inner_offset, b + inner_offset});
        proxy.faceID.push_back(0U);
        proxy.faceID.push_back(0U);
        wall_triangles += 2U;
    }

    proxy.runIndex = {
        0U, static_cast<std::uint64_t>(outer_triangles * 3U),
        static_cast<std::uint64_t>((outer_triangles + inner_triangles) * 3U)};
    proxy.runOriginalID = {meta.outer_original_id, inner_original_id};
    if (wall_triangles > 0U) {
        proxy.runIndex.push_back(static_cast<std::uint64_t>(
            (outer_triangles + inner_triangles + wall_triangles) * 3U));
        proxy.runOriginalID.push_back(wall_original_id);
    }
    proxy.tolerance = meta.thickness * 0.5;
    return proxy;
}

}  // namespace

manifold::Manifold build_primitive_proxy(const TriMeshD& triangulated_primitive,
                                         const ProxyMeta& meta) {
    manifold::MeshGL64 proxy_mesh =
        build_proxy_meshgl(triangulated_primitive, meta);
    proxy_mesh.Merge();
    manifold::Manifold proxy(proxy_mesh);
    if (proxy.Status() != manifold::Manifold::Error::NoError) {
        throw std::runtime_error(
            "Failed to build Manifold proxy shell (status=" +
            std::to_string(static_cast<int>(proxy.Status())) + ")");
    }
    return proxy;
}

}  // namespace pycanha::gmm::cutting
