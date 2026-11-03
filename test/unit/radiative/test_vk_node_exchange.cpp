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
                                                  bool area_weights,
                                                  bool keep_full) {
    rad::AccumConfig config;
    config.triangulation.mode = mode;
    config.triangulation.area_weights = area_weights;
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
    struct Mode {
        rad::TriangulationMode mode;
        bool area_weights;
    };
    for (const Mode mode : {Mode{rad::TriangulationMode::RayDensity, false},
                            Mode{rad::TriangulationMode::RayDensity, true},
                            Mode{rad::TriangulationMode::None, false}}) {
        for (const bool all_faces : {true, false}) {
            const std::span<const std::uint32_t> emitters =
                all_faces ? std::span<const std::uint32_t>{}
                          : std::span<const std::uint32_t>(subset);
            INFO("mode " << static_cast<int>(mode.mode) << ", area weights "
                         << mode.area_weights << ", all faces " << all_faces);
            rad::ExchangeAccumulator face(
                bench.scene, rad::Band::IR,
                with_triangulation(mode.mode, mode.area_weights, true));
            trace_two_batches(bench.scene, face, emitters);
            const Summed summed =
                sum_to_nodes(bench.scene, face.result(), bench.nodes);

            rad::NodeExchangeAccumulator dense(
                bench.scene, rad::Band::IR, bench.nodes,
                with_triangulation(mode.mode, mode.area_weights, false));
            trace_two_batches(bench.scene, dense, emitters);
            const rad::NodeExchangeResult reference = dense.result();
            require_matches(reference, summed);
            REQUIRE(reference.stats.rays_per_face == 4'000);

            // Tiled node rows: the same integer cells, block by block.
            for (const std::uint32_t tile_rows : {7U, 50U, 1000U}) {
                rad::AccumConfig config =
                    with_triangulation(mode.mode, mode.area_weights, false);
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
                                false, false)),
        std::invalid_argument);
    REQUIRE_THROWS_AS(make(with_triangulation(
                          rad::TriangulationMode::RayDensity, false, true)),
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
        with_triangulation(rad::TriangulationMode::RayDensity, false, true));
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

namespace {

// Absorptance derivatives. Two traces with the same seed follow the same
// paths when no choice along them depends on a: no Russian roulette
// (energy_threshold 0), and on the faces whose a moves no transmission and a
// specular share of the reflectance that the shifted rows keep exactly (s = d,
// in dyadic values, so the kernel's choice is u < 1/2 in all three traces).
// Their central difference is then the derivative of the very estimator the
// kernel scores, up to its float rounding and the step's second order.
// Material 0 (group 0) reflects half specularly, material 2 (group 1) is
// diffuse, material 1 (no group) also transmits.
constexpr std::array<std::array<float, 3>, 3> optics = {
    {{0.5F, 0.25F, 0.0F}, {0.75F, 0.0625F, 0.0625F}, {0.375F, 0.0F, 0.0F}}};
constexpr std::array<int, 3> material_group = {0, -1, 1};
constexpr float absorptance_step = 1.0F / 1024.0F;

[[nodiscard]] int optics_material(std::size_t pair_index) {
    return static_cast<int>(pair_index % 3);
}

// The rows with material `shifted`'s a moved by `step`, its specular share of
// the reflectance kept and its transmittance fixed; both bands alike.
[[nodiscard]] rad::MaterialTable optics_materials(int shifted, float step) {
    std::array<std::array<float, 6>, 3> rows{};
    for (std::size_t m = 0; m < rows.size(); ++m) {
        float a = optics[m][0];
        float s = optics[m][1];
        const float t = optics[m][2];
        if (static_cast<int>(m) == shifted) {
            const float share = s / (1.0F - a - t);
            a += step;
            s = share * (1.0F - a - t);
        }
        rows[m] = {a, s, t, a, s, t};
    }
    std::vector<int> pair_rows(num_pairs);
    for (std::size_t index = 0; index < num_pairs; ++index) {
        pair_rows[index] = optics_material(index);
    }
    return make_materials(rows, pair_rows);
}

[[nodiscard]] std::vector<std::int32_t> material_groups() {
    std::vector<std::int32_t> groups(num_faces);
    for (std::uint32_t face = 0; face < num_faces; ++face) {
        groups[face] =
            material_group[static_cast<std::size_t>(optics_material(face / 2))];
    }
    return groups;
}

[[nodiscard]] rad::NodeExchangeResult trace_without_roulette(
    rad::RadiativeScene& scene, std::span<const NodeNum> nodes,
    std::span<const std::int32_t> groups) {
    rad::NodeExchangeAccumulator acc(scene, rad::Band::IR, nodes, {}, groups);
    rad::TraceSettings settings;
    settings.rays_per_face = 1'000;
    settings.energy_threshold = 0.0F;
    for (const std::uint32_t seed : {5U, 6U}) {
        settings.seed = seed;
        scene.accumulate_exchange(acc, settings, {});
    }
    return acc.result();
}

}  // namespace

TEST_CASE(
    "radiative vulkan: absorptance derivatives match a same-seed central "
    "difference",
    "[radiative][gpu][vulkan][nodes]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
        return;
    }
    const std::unique_ptr<pycanha::gmm::GeometryModel> model = make_plates();
    rad::Device device = rad::Device::create();
    rad::RadiativeScene scene(device, model->mesh_parts(),
                              optics_materials(-1, 0.0F));
    const std::vector<NodeNum> nodes = node_numbers();
    const std::vector<std::int32_t> groups = material_groups();

    const rad::NodeExchangeResult central =
        trace_without_roulette(scene, nodes, groups);
    REQUIRE(central.absorptance_derivatives.size() == 2);
    // The groups leave the factors as a trace without them makes them.
    const rad::NodeExchangeResult plain =
        trace_without_roulette(scene, nodes, {});
    REQUIRE(plain.absorptance_derivatives.empty());
    REQUIRE(csr_bit_identical(central.factors, plain.factors));

    const Eigen::Index rows = central.factors.rows();
    const Eigen::Index cols = central.factors.cols();
    for (std::size_t m = 0; m < optics.size(); ++m) {
        const int group = material_group[m];
        if (group < 0) {
            continue;
        }
        const auto shifted = static_cast<int>(m);
        scene.update_materials(optics_materials(shifted, absorptance_step));
        const rad::NodeExchangeResult up =
            trace_without_roulette(scene, nodes, {});
        scene.update_materials(optics_materials(shifted, -absorptance_step));
        const rad::NodeExchangeResult down =
            trace_without_roulette(scene, nodes, {});
        scene.update_materials(optics_materials(-1, 0.0F));
        // The step as the float rows hold it.
        const double span =
            static_cast<double>(optics[m][0] + absorptance_step) -
            static_cast<double>(optics[m][0] - absorptance_step);

        const rad::SparseMatrix& derivative =
            central.absorptance_derivatives[static_cast<std::size_t>(group)];
        REQUIRE(derivative.rows() == rows);
        REQUIRE(derivative.cols() == cols);
        std::size_t compared = 0;
        double largest = 0.0;
        for (Eigen::Index row = 0; row < rows; ++row) {
            // The node columns' upper triangle and the buckets.
            for (Eigen::Index col = row + 1; col < cols; ++col) {
                const double expected = (csr_value(up.factors, row, col) -
                                         csr_value(down.factors, row, col)) /
                                        span;
                const double actual = csr_value(derivative, row, col);
                const double value = csr_value(central.factors, row, col);
                // Float rounding of the deposits and shares, amplified by
                // a / (2 step), and the step's second order: both well
                // below 1e-3 of the entry.
                const double tolerance =
                    2e-3 * (std::abs(expected) + std::abs(value));
                if (std::abs(actual - expected) > tolerance) {
                    INFO("group " << group << ", node " << row << ", column "
                                  << col << ": " << actual << " against "
                                  << expected << " (entry " << value << ")");
                    REQUIRE(std::abs(actual - expected) <= tolerance);
                }
                compared += value != 0.0 ? 1 : 0;
                largest = std::max(largest, std::abs(expected));
            }
        }
        REQUIRE(compared > 1'000);
        REQUIRE(largest > 0.0);
    }
}

TEST_CASE("radiative vulkan: absorptance derivatives add up across groups",
          "[radiative][gpu][vulkan][nodes]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
        return;
    }
    // Each material's group split in two by face parity (four groups, the
    // most a trace takes): the halves' derivatives sum to the whole's, ray
    // for ray, since a path's score is a sum over its faces.
    const std::unique_ptr<pycanha::gmm::GeometryModel> model = make_plates();
    rad::Device device = rad::Device::create();
    rad::RadiativeScene scene(device, model->mesh_parts(),
                              optics_materials(-1, 0.0F));
    const std::vector<NodeNum> nodes = node_numbers();
    const std::vector<std::int32_t> whole = material_groups();
    std::vector<std::int32_t> halves(num_faces);
    for (std::uint32_t face = 0; face < num_faces; ++face) {
        halves[face] =
            whole[face] < 0
                ? -1
                : (2 * whole[face]) + static_cast<std::int32_t>(face % 2);
    }
    const rad::NodeExchangeResult by_whole =
        trace_without_roulette(scene, nodes, whole);
    const rad::NodeExchangeResult by_halves =
        trace_without_roulette(scene, nodes, halves);
    REQUIRE(by_halves.absorptance_derivatives.size() ==
            static_cast<std::size_t>(rad::max_absorptance_groups));
    REQUIRE(csr_bit_identical(by_whole.factors, by_halves.factors));
    for (std::size_t group = 0; group < 2; ++group) {
        const rad::SparseMatrix& sum = by_whole.absorptance_derivatives[group];
        const rad::SparseMatrix& first =
            by_halves.absorptance_derivatives[2 * group];
        const rad::SparseMatrix& second =
            by_halves.absorptance_derivatives[(2 * group) + 1];
        REQUIRE(sum.nonZeros() > 0);
        for (Eigen::Index row = 0; row < sum.rows(); ++row) {
            for (Eigen::Index col = row + 1; col < sum.cols(); ++col) {
                const double expected = csr_value(sum, row, col);
                const double parts =
                    csr_value(first, row, col) + csr_value(second, row, col);
                const double scale = std::abs(csr_value(first, row, col)) +
                                     std::abs(csr_value(second, row, col)) +
                                     std::abs(expected);
                INFO("group " << group << ", node " << row << ", column " << col
                              << ": " << parts << " against " << expected);
                REQUIRE(std::abs(parts - expected) <= (1e-5 * scale) + 1e-18);
            }
        }
    }
}

TEST_CASE("radiative vulkan: absorptance groups refuse what they cannot trace",
          "[radiative][gpu][vulkan][nodes]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
        return;
    }
    Bench bench;
    const auto make = [&](std::span<const std::int32_t> groups,
                          const rad::AccumConfig& config) {
        return rad::NodeExchangeAccumulator(bench.scene, rad::Band::IR,
                                            bench.nodes, config, groups);
    };
    std::vector<std::int32_t> groups(num_faces, -1);
    groups[0] = 0;
    REQUIRE_NOTHROW(make(groups, {}));
    const std::vector<std::int32_t> short_groups(num_faces - 1, 0);
    REQUIRE_THROWS_AS(make(short_groups, {}), std::invalid_argument);
    groups[1] = rad::max_absorptance_groups;
    REQUIRE_THROWS_AS(make(groups, {}), std::invalid_argument);
    groups[1] = -2;
    REQUIRE_THROWS_AS(make(groups, {}), std::invalid_argument);
    groups[1] = 1;
    rad::AccumConfig tiled;
    tiled.layout = rad::AccumLayout::Tiled;
    tiled.tile_rows = 50;
    REQUIRE_THROWS_AS(make(groups, tiled), std::invalid_argument);
    // No face in a group: nothing to trace, any layout.
    const std::vector<std::int32_t> none(num_faces, -1);
    rad::NodeExchangeAccumulator acc = make(none, tiled);
    rad::TraceSettings settings;
    settings.rays_per_face = 100;
    bench.scene.accumulate_exchange(acc, settings, {});
    REQUIRE(acc.result().absorptance_derivatives.empty());
}

TEST_CASE(
    "radiative vulkan: rays per face sum to the face result with area "
    "weights",
    "[radiative][gpu][vulkan][nodes]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
        return;
    }
    Bench bench;
    // Two ray levels, a face of every third pair given 4x the rays of the
    // others, and a face of every 13th not traced at all.
    constexpr std::uint64_t low = 1'000;
    constexpr std::uint64_t high = 4'000;
    std::vector<std::uint64_t> rays(num_faces, low);
    std::vector<std::uint32_t> many;
    std::vector<std::uint32_t> few;
    for (std::uint32_t face = 0; face < num_faces; ++face) {
        if (face % 13 == 7) {
            rays[face] = 0;
        } else if ((face / 2) % 3 == 1) {
            rays[face] = high;
            many.push_back(face);
        } else {
            few.push_back(face);
        }
    }
    rad::TraceSettings settings;
    settings.seed = 3;
    // The face rows: the larger batch first (it sets their fixed-point
    // scale), each row normalised by its own rays, the directions of a pair
    // weighed by area.
    rad::ExchangeAccumulator face(
        bench.scene, rad::Band::IR,
        with_triangulation(rad::TriangulationMode::RayDensity, true, true));
    settings.rays_per_face = high;
    bench.scene.accumulate_exchange(face, settings, many);
    settings.rays_per_face = low;
    bench.scene.accumulate_exchange(face, settings, few);
    const Summed summed = sum_to_nodes(bench.scene, face.result(), bench.nodes);

    // The node rows, the batches in the other order: the allocation fixes
    // the weights and the scale, not the first batch.
    const auto trace = [&](rad::NodeExchangeAccumulator& acc) {
        settings.rays_per_face = low;
        bench.scene.accumulate_exchange(acc, settings, few);
        settings.rays_per_face = high;
        bench.scene.accumulate_exchange(acc, settings, many);
        return acc.result();
    };
    rad::NodeExchangeAccumulator dense(
        bench.scene, rad::Band::IR, bench.nodes,
        with_triangulation(rad::TriangulationMode::RayDensity, true, false), {},
        rays);
    const rad::NodeExchangeResult reference = trace(dense);
    require_matches(reference, summed);
    REQUIRE(reference.stats.rays_per_face == low);

    rad::AccumConfig config =
        with_triangulation(rad::TriangulationMode::RayDensity, true, false);
    config.layout = rad::AccumLayout::Tiled;
    config.tile_rows = 7;
    rad::NodeExchangeAccumulator tiled(bench.scene, rad::Band::IR, bench.nodes,
                                       config, {}, rays);
    const rad::NodeExchangeResult result = trace(tiled);
    REQUIRE(csr_bit_identical(result.factors, reference.factors));
}

TEST_CASE("radiative vulkan: rays per face alike give the uniform trace",
          "[radiative][gpu][vulkan][nodes]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
        return;
    }
    Bench bench;
    std::vector<std::uint32_t> subset;
    std::vector<std::uint64_t> rays(num_faces, 0);
    for (std::uint32_t face = 0; face < num_faces; face += 3) {
        subset.push_back(face);
        rays[face] = 2'000;
    }
    rad::TraceSettings settings;
    settings.rays_per_face = 2'000;
    settings.seed = 5;
    for (const bool area_weights : {false, true}) {
        INFO("area weights " << area_weights);
        const rad::AccumConfig config =
            with_triangulation(area_weights ? rad::TriangulationMode::RayDensity
                                            : rad::TriangulationMode::None,
                               area_weights, false);
        rad::NodeExchangeAccumulator uniform(bench.scene, rad::Band::IR,
                                             bench.nodes, config);
        bench.scene.accumulate_exchange(uniform, settings, subset);
        rad::NodeExchangeAccumulator allocated(bench.scene, rad::Band::IR,
                                               bench.nodes, config, {}, rays);
        bench.scene.accumulate_exchange(allocated, settings, subset);
        const rad::NodeExchangeResult a = uniform.result();
        const rad::NodeExchangeResult b = allocated.result();
        REQUIRE(csr_bit_identical(b.factors, a.factors));
        REQUIRE(b.intra_node_total == a.intra_node_total);
        REQUIRE(b.stats.rays_per_face == a.stats.rays_per_face);
    }
}

TEST_CASE("radiative vulkan: rays per face refuse what they cannot trace",
          "[radiative][gpu][vulkan][nodes]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
        return;
    }
    Bench bench;
    const rad::AccumConfig area =
        with_triangulation(rad::TriangulationMode::RayDensity, true, false);
    const auto make = [&](std::span<const std::uint64_t> rays,
                          const rad::AccumConfig& config) {
        return rad::NodeExchangeAccumulator(bench.scene, rad::Band::IR,
                                            bench.nodes, config, {}, rays);
    };
    const std::vector<std::uint64_t> short_rays(num_faces - 1, 100);
    REQUIRE_THROWS_AS(make(short_rays, area), std::invalid_argument);
    const std::vector<std::uint64_t> none(num_faces, 0);
    REQUIRE_THROWS_AS(make(none, area), std::invalid_argument);
    std::vector<std::uint64_t> rays(num_faces, 0);
    rays[0] = 100;
    rays[1] = 100;
    rays[2] = 400;
    // RayDensity weighing each face by its own rays: not node rows' weights.
    REQUIRE_THROWS_AS(
        make(rays, with_triangulation(rad::TriangulationMode::RayDensity, false,
                                      false)),
        std::invalid_argument);

    rad::NodeExchangeAccumulator acc = make(rays, area);
    rad::TraceSettings settings;
    settings.rays_per_face = 100;
    const std::vector<std::uint32_t> first = {0, 1};
    const std::vector<std::uint32_t> wrong_rays = {2};
    const std::vector<std::uint32_t> not_allocated = {3};
    REQUIRE_THROWS_AS(
        bench.scene.accumulate_exchange(acc, settings, wrong_rays),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        bench.scene.accumulate_exchange(acc, settings, not_allocated),
        std::invalid_argument);
    bench.scene.accumulate_exchange(acc, settings, first);
    REQUIRE_THROWS_AS(bench.scene.accumulate_exchange(acc, settings, first),
                      std::invalid_argument);
    // Face 2 not traced yet.
    REQUIRE_THROWS_AS(acc.result(), std::logic_error);
    settings.rays_per_face = 400;
    bench.scene.accumulate_exchange(acc, settings, wrong_rays);
    REQUIRE(acc.result().stats.rays_per_face == 100);
    // A reset accumulator traces the allocation again.
    acc.reset();
    REQUIRE_THROWS_AS(acc.result(), std::logic_error);
    bench.scene.accumulate_exchange(acc, settings, wrong_rays);
    settings.rays_per_face = 100;
    bench.scene.accumulate_exchange(acc, settings, first);
    REQUIRE_NOTHROW(acc.result());
}
