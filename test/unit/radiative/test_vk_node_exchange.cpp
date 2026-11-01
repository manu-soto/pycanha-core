// Vulkan-only: NodeExchangeAccumulator sums the exchange trace to nodes on
// the device. Its result must be what a consumer of ExchangeAccumulator gets
// by summing the face result with aggregate_matrix - the same combination of
// each face pair's two directions - up to the float rounding of the kernel's
// scaled deposits; its Dense and Tiled node layouts hold the same integer
// cells and must agree bit for bit.

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"
#include "pycanha-core/radiative/accumulators.hpp"
#include "pycanha-core/radiative/aggregate.hpp"
#include "pycanha-core/radiative/device.hpp"
#include "pycanha-core/radiative/materials.hpp"
#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/scene.hpp"
#include "pycanha-core/radiative/settings.hpp"
#include "radiative/vk_test_hooks.hpp"
#include "scene_fixtures.hpp"

namespace rad = pycanha::radiative;
using pycanha::NodeNum;
using radiative_fixtures::csr_bit_identical;
using radiative_fixtures::csr_value;
using radiative_fixtures::gray_row;
using radiative_fixtures::make_materials;

namespace {

constexpr std::size_t cells_per_side = 16;
constexpr std::size_t num_pairs = 2 * cells_per_side * cells_per_side;
constexpr auto num_faces = static_cast<std::uint32_t>(2 * num_pairs);
constexpr std::array<float, 2> absorptivity = {0.5F, 0.85F};

// Graded cuts: the cells grow along each side (1/256 to 31/256), so the faces
// of a node differ in area and so do their emission shares and blend weights.
[[nodiscard]] std::vector<double> graded_cuts() {
    std::vector<double> cuts(cells_per_side + 1);
    for (std::size_t i = 0; i < cuts.size(); ++i) {
        const double t =
            static_cast<double>(i) / static_cast<double>(cells_per_side);
        cuts[i] = t * t;
    }
    return cuts;
}

// Two meshed unit plates half a unit apart, facing each other; the open
// sides let rays escape, so the space column is filled too.
[[nodiscard]] std::unique_ptr<pycanha::gmm::GeometryModel> make_plates() {
    using pycanha::gmm::GeometryItem;
    using pycanha::gmm::Rectangle;
    using pycanha::gmm::ThermalMesh;
    const ThermalMesh mesh(graded_cuts(), graded_cuts());
    auto model = std::make_unique<pycanha::gmm::GeometryModel>("plates");
    model->add(std::make_shared<GeometryItem>(
        "plate_a", Rectangle({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}),
        mesh));
    model->add(std::make_shared<GeometryItem>(
        "plate_b", Rectangle({0.0, 0.0, 0.5}, {0.0, 1.0, 0.5}, {1.0, 0.0, 0.5}),
        mesh));
    return model;
}

[[nodiscard]] int pair_material(std::size_t index) {
    return index % 3 == 0 ? 1 : 0;
}

// Nodes of unequal size, each gathering scattered faces of both plates (so a
// node also exchanges with itself), the two sides of a pair in different
// nodes, every 11th face in no node; the numbers are not contiguous.
[[nodiscard]] std::vector<NodeNum> node_numbers() {
    std::vector<NodeNum> nodes(num_faces);
    for (std::uint32_t face = 0; face < num_faces; ++face) {
        if (face % 11 == 5) {
            nodes[face] = pycanha::gmm::NO_NODE;
            continue;
        }
        const std::uint32_t pair_index = face / 2;
        nodes[face] = static_cast<NodeNum>((10 * ((pair_index * 7) % 61)) +
                                           (3 * (face % 2)) + 1);
    }
    return nodes;
}

// The face result summed to nodes as a consumer of ExchangeAccumulator does
// it, plus what the node result reports beside the couplings.
struct Summed {
    std::vector<NodeNum> nodes;
    rad::SparseMatrix couplings;
    // Per node pair: the two forward estimates' sum, both directions, before
    // they are combined (the scale a weight's error multiplies).
    rad::SparseMatrix forward;
    double intra_node_total = 0.0;
    // Per node: space, inactive, lost, unassigned.
    std::vector<std::array<double, 4>> buckets;
};

// `face` must have kept its full matrix: the unassigned column is the
// forward estimate, A_i eps_i B_ij over the faces j in no node.
[[nodiscard]] Summed sum_to_nodes(const rad::RadiativeScene& scene,
                                  const rad::ExchangeResult& face,
                                  std::span<const NodeNum> node_of) {
    Summed out;
    out.nodes = rad::aggregate_nodes(node_of);
    const rad::AggregateResult aggregated =
        rad::aggregate_matrix(face.factors, node_of);
    out.couplings = aggregated.matrix;
    out.intra_node_total = aggregated.intra_node_total;
    out.buckets.assign(out.nodes.size(), {0.0, 0.0, 0.0, 0.0});
    const auto slot = [&](NodeNum node) {
        return static_cast<std::size_t>(
            std::ranges::lower_bound(out.nodes, node) - out.nodes.begin());
    };
    const std::span<const double> areas = scene.face_areas();
    // Both triangles of A_i eps_i B_ij: aggregate_matrix folds each entry to
    // its node pair, so the two directions add up.
    Eigen::VectorXd emissive_area(num_faces);
    for (std::uint32_t row = 0; row < num_faces; ++row) {
        emissive_area(row) =
            areas[row] *
            absorptivity[static_cast<std::size_t>(pair_material(row / 2))];
    }
    const rad::SparseMatrix forward =
        emissive_area.asDiagonal() * (*face.full_factors);
    out.forward = rad::aggregate_matrix(forward, node_of).matrix;
    for (std::uint32_t row = 0; row < num_faces; ++row) {
        if (node_of[row] == pycanha::gmm::NO_NODE) {
            continue;
        }
        std::array<double, 4>& bucket = out.buckets[slot(node_of[row])];
        for (std::size_t k = 0; k < 3; ++k) {
            bucket[k] += csr_value(face.factors, row,
                                   static_cast<Eigen::Index>(num_faces + k));
        }
        const double emissive =
            areas[row] *
            absorptivity[static_cast<std::size_t>(pair_material(row / 2))];
        for (rad::SparseMatrix::InnerIterator entry(*face.full_factors, row);
             entry; ++entry) {
            if (entry.col() < static_cast<Eigen::Index>(num_faces) &&
                node_of[static_cast<std::size_t>(entry.col())] ==
                    pycanha::gmm::NO_NODE) {
                bucket[3] += emissive * entry.value();
            }
        }
    }
    return out;
}

// Float rounding of the scaled deposits, never Monte-Carlo noise: both
// results come from the same rays. `forward` is the pair's two forward
// estimates summed: both sides compute the blend weight to about 1e-7 in
// absolute terms (the face side from a table rounded to 24 bits, the kernel
// with float pow), which on a pair whose combined value is far below its
// forward estimates (one direction weighted near 0) is more than 2e-6 of it.
[[nodiscard]] bool close(double lhs, double rhs, double forward = 0.0) {
    constexpr double relative = 2e-6;
    constexpr double weight = 5e-7;
    constexpr double absolute = 1e-16;  // the lost column is ~1e-12 a node
    return std::abs(lhs - rhs) <=
           (relative * std::max(std::abs(lhs), std::abs(rhs))) +
               (weight * forward) + absolute;
}

void require_matches(const rad::NodeExchangeResult& node, const Summed& face) {
    REQUIRE(node.nodes == face.nodes);
    const auto nodes = static_cast<Eigen::Index>(face.nodes.size());
    REQUIRE(node.factors.rows() == nodes);
    REQUIRE(node.factors.cols() == nodes + rad::num_node_virtual_columns);
    REQUIRE(face.couplings.nonZeros() > 0);
    for (Eigen::Index row = 0; row < nodes; ++row) {
        // Upper triangle only, the diagonal left out.
        for (rad::SparseMatrix::InnerIterator entry(node.factors, row); entry;
             ++entry) {
            REQUIRE(entry.col() > row);
        }
        for (Eigen::Index col = row + 1; col < nodes; ++col) {
            const double expected = csr_value(face.couplings, row, col);
            const double actual = csr_value(node.factors, row, col);
            const double forward = csr_value(face.forward, row, col);
            if (!close(actual, expected, forward)) {
                INFO("node pair " << row << ", " << col << ": " << actual
                                  << " against " << expected << ", forward "
                                  << forward);
                REQUIRE(close(actual, expected, forward));
            }
        }
        const auto& bucket = face.buckets[static_cast<std::size_t>(row)];
        bool buckets_close = true;
        for (std::size_t k = 0; k < 4; ++k) {
            const double actual = csr_value(
                node.factors, row, nodes + static_cast<Eigen::Index>(k));
            INFO("node " << row << ", virtual column " << k << ": " << actual
                         << " against " << bucket[k]);
            CHECK(close(actual, bucket[k]));
            buckets_close = buckets_close && close(actual, bucket[k]);
        }
        REQUIRE(buckets_close);
    }
    REQUIRE(face.intra_node_total > 0.0);
    REQUIRE(close(node.intra_node_total, face.intra_node_total));
}

struct Bench {
    std::unique_ptr<pycanha::gmm::GeometryModel> model = make_plates();
    rad::Device device = rad::Device::create();
    rad::RadiativeScene scene;
    std::vector<NodeNum> nodes = node_numbers();

    Bench() : scene(device, model->mesh_parts(), materials()) {}

    [[nodiscard]] static rad::MaterialTable materials() {
        const std::array<std::array<float, 6>, 2> rows{
            gray_row(absorptivity[0]), gray_row(absorptivity[1])};
        std::vector<int> pair_rows(num_pairs);
        for (std::size_t index = 0; index < num_pairs; ++index) {
            pair_rows[index] = pair_material(index);
        }
        return make_materials(rows, pair_rows);
    }
};

// Two batches with different seeds into each accumulator.
template <class Accumulator>
void trace_two_batches(rad::RadiativeScene& scene, Accumulator& acc,
                       std::span<const std::uint32_t> emitters) {
    rad::TraceSettings settings;
    settings.rays_per_face = 2'000;
    settings.seed = 3;
    scene.accumulate_exchange(acc, settings, emitters);
    settings.seed = 4;
    scene.accumulate_exchange(acc, settings, emitters);
}

[[nodiscard]] rad::AccumConfig with_triangulation(rad::TriangulationMode mode,
                                                  bool keep_full) {
    rad::AccumConfig config;
    config.triangulation.mode = mode;
    config.triangulation.keep_full_matrix = keep_full;
    return config;
}

}  // namespace

TEST_CASE("radiative vulkan: node rows match the face result summed to nodes",
          "[radiative][gpu][vulkan][nodes]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
        return;
    }
    Bench bench;
    REQUIRE(bench.scene.num_faces() == num_faces);
    std::vector<std::uint32_t> subset;
    for (std::uint32_t face = 0; face < num_faces; face += 3) {
        subset.push_back(face);
    }
    for (const rad::TriangulationMode mode :
         {rad::TriangulationMode::RayDensity, rad::TriangulationMode::None}) {
        for (const bool all_faces : {true, false}) {
            const std::span<const std::uint32_t> emitters =
                all_faces ? std::span<const std::uint32_t>{}
                          : std::span<const std::uint32_t>(subset);
            INFO("mode " << static_cast<int>(mode) << ", all faces "
                         << all_faces);
            rad::ExchangeAccumulator face(bench.scene, rad::Band::IR,
                                          with_triangulation(mode, true));
            trace_two_batches(bench.scene, face, emitters);
            const Summed summed =
                sum_to_nodes(bench.scene, face.result(), bench.nodes);

            rad::NodeExchangeAccumulator dense(bench.scene, rad::Band::IR,
                                               bench.nodes,
                                               with_triangulation(mode, false));
            trace_two_batches(bench.scene, dense, emitters);
            const rad::NodeExchangeResult reference = dense.result();
            require_matches(reference, summed);
            REQUIRE(reference.stats.rays_per_face == 4'000);

            // Tiled node rows: the same integer cells, block by block.
            for (const std::uint32_t tile_rows : {7U, 50U, 1000U}) {
                rad::AccumConfig config = with_triangulation(mode, false);
                config.layout = rad::AccumLayout::Tiled;
                config.tile_rows = tile_rows;
                rad::NodeExchangeAccumulator tiled(bench.scene, rad::Band::IR,
                                                   bench.nodes, config);
                trace_two_batches(bench.scene, tiled, emitters);
                const rad::NodeExchangeResult result = tiled.result();
                REQUIRE(csr_bit_identical(result.factors, reference.factors));
                REQUIRE(result.intra_node_total == reference.intra_node_total);
            }
        }
    }
}

TEST_CASE("radiative vulkan: node rows refuse what they cannot reproduce",
          "[radiative][gpu][vulkan][nodes]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
        return;
    }
    Bench bench;
    const auto make = [&](const rad::AccumConfig& config) {
        return rad::NodeExchangeAccumulator(bench.scene, rad::Band::IR,
                                            bench.nodes, config);
    };
    REQUIRE_THROWS_AS(
        make(with_triangulation(rad::TriangulationMode::ConstrainedLeastSquares,
                                false)),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        make(with_triangulation(rad::TriangulationMode::RayDensity, true)),
        std::invalid_argument);
    const std::vector<NodeNum> short_nodes(num_faces - 1, 1);
    REQUIRE_THROWS_AS(
        rad::NodeExchangeAccumulator(bench.scene, rad::Band::IR, short_nodes),
        std::invalid_argument);

    rad::TraceSettings settings;
    settings.rays_per_face = 100;
    const std::vector<std::uint32_t> first = {0, 1, 2, 3};
    const std::vector<std::uint32_t> other = {0, 1, 2, 4};
    const std::vector<std::uint32_t> twice = {0, 1, 1, 2};
    rad::NodeExchangeAccumulator acc(bench.scene, rad::Band::IR, bench.nodes);
    REQUIRE_THROWS_AS(bench.scene.accumulate_exchange(acc, settings, twice),
                      std::invalid_argument);
    bench.scene.accumulate_exchange(acc, settings, first);
    // The same set in another order is the same set.
    const std::vector<std::uint32_t> reordered = {3, 2, 1, 0};
    bench.scene.accumulate_exchange(acc, settings, reordered);
    REQUIRE_THROWS_AS(bench.scene.accumulate_exchange(acc, settings, other),
                      std::invalid_argument);
    // A reset accumulator takes a new set.
    acc.reset();
    bench.scene.accumulate_exchange(acc, settings, other);
    REQUIRE(acc.result().stats.rays_per_face == 100);
}

TEST_CASE("radiative vulkan: one face a node gives the face rows' balances",
          "[radiative][gpu][vulkan][nodes]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
        return;
    }
    // Every face its own node: each deposit's share is exactly 1 and the
    // fixed-point scale is the face accumulator's, so the virtual columns
    // hold the face rows' cells themselves (the couplings still differ by
    // the float blend weight).
    Bench bench;
    std::vector<NodeNum> nodes(num_faces);
    for (std::uint32_t face = 0; face < num_faces; ++face) {
        nodes[face] = static_cast<NodeNum>(face);
    }
    rad::ExchangeAccumulator face(
        bench.scene, rad::Band::IR,
        with_triangulation(rad::TriangulationMode::RayDensity, true));
    trace_two_batches(bench.scene, face, {});
    const Summed summed = sum_to_nodes(bench.scene, face.result(), nodes);
    rad::NodeExchangeAccumulator acc(bench.scene, rad::Band::IR, nodes);
    trace_two_batches(bench.scene, acc, {});
    const rad::NodeExchangeResult result = acc.result();
    require_matches(result, summed);
    const auto n = static_cast<Eigen::Index>(num_faces);
    for (Eigen::Index row = 0; row < n; ++row) {
        for (std::size_t k = 0; k < 3; ++k) {
            const double actual = csr_value(result.factors, row,
                                            n + static_cast<Eigen::Index>(k));
            const double expected =
                summed.buckets[static_cast<std::size_t>(row)][k];
            INFO("face " << row << ", virtual column " << k << ": " << actual
                         << " against " << expected);
            REQUIRE(std::abs(actual - expected) <= 1e-12 * std::abs(expected));
        }
    }
}

TEST_CASE(
    "radiative vulkan: node rows read back whole where blocks cannot "
    "be compacted",
    "[radiative][gpu][vulkan][nodes]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
        return;
    }
    // Launches of at most 5 emitter rows: every Tiled block of 50 node rows
    // is then refused by the compaction and read back whole (absorb_block,
    // whose host mirror is only allocated here).
    Bench bench;
    rad::NodeExchangeAccumulator dense(bench.scene, rad::Band::IR, bench.nodes);
    trace_two_batches(bench.scene, dense, {});
    const rad::NodeExchangeResult reference = dense.result();
    rad::detail::lower_max_dispatch_emitters(bench.device, 5U);
    rad::AccumConfig config;
    config.layout = rad::AccumLayout::Tiled;
    config.tile_rows = 50;
    rad::NodeExchangeAccumulator tiled(bench.scene, rad::Band::IR, bench.nodes,
                                       config);
    trace_two_batches(bench.scene, tiled, {});
    const rad::NodeExchangeResult result = tiled.result();
    REQUIRE(reference.factors.nonZeros() > 0);
    REQUIRE(csr_bit_identical(result.factors, reference.factors));
    REQUIRE(result.intra_node_total == reference.intra_node_total);
}
