// Vulkan-only: every kernel launch gives each emitting face one workgroup row
// in the dispatch's Y dimension, and a device caps those rows (NVIDIA at
// 65535), so longer emitter lists are traced in batches. No test scene comes
// near a real limit, so it is lowered through a test hook, and each kernel
// must then reproduce bit for bit what it computes in one launch: the faces
// come from the emitters buffer, the random streams are keyed on (face,
// global ray index, seed) and the cells are integer atomics.

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"
#include "pycanha-core/radiative/accumulators.hpp"
#include "pycanha-core/radiative/device.hpp"
#include "pycanha-core/radiative/materials.hpp"
#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/scene.hpp"
#include "pycanha-core/radiative/settings.hpp"
#include "radiative/vk_test_hooks.hpp"
#include "scene_fixtures.hpp"

namespace rad = pycanha::radiative;
using radiative_fixtures::csr_bit_identical;
using radiative_fixtures::gray_row;
using radiative_fixtures::make_materials;

namespace {

// 25 x 25 face pairs per plate: 2500 faces, so every lowered limit below
// splits each launch into several batches, the last one partial.
constexpr std::size_t cells_per_side = 25;
constexpr std::size_t num_pairs = 2 * cells_per_side * cells_per_side;
constexpr auto num_faces = static_cast<std::uint32_t>(2 * num_pairs);
// Tiled blocks larger than a batch, so a block is itself split while its
// accumulator rows stay relative to the block's row_offset.
constexpr std::uint32_t tile_rows = 1500;

[[nodiscard]] std::vector<double> uniform_cuts() {
    std::vector<double> cuts(cells_per_side + 1);
    for (std::size_t i = 0; i < cuts.size(); ++i) {
        cuts[i] = static_cast<double>(i) / static_cast<double>(cells_per_side);
    }
    return cuts;
}

// Two meshed unit plates half a unit apart, facing each other (the
// parallel-plates fixture, subdivided).
[[nodiscard]] std::unique_ptr<pycanha::gmm::GeometryModel>
make_meshed_plates() {
    using pycanha::gmm::GeometryItem;
    using pycanha::gmm::Rectangle;
    using pycanha::gmm::ThermalMesh;
    const ThermalMesh mesh(uniform_cuts(), uniform_cuts());
    auto model = std::make_unique<pycanha::gmm::GeometryModel>("plates");
    model->add(std::make_shared<GeometryItem>(
        "plate_a", Rectangle({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}),
        mesh));
    model->add(std::make_shared<GeometryItem>(
        "plate_b", Rectangle({0.0, 0.0, 0.5}, {0.0, 1.0, 0.5}, {1.0, 0.0, 0.5}),
        mesh));
    return model;
}

struct Traced {
    rad::VfResult vf;
    rad::VfResult vf_tiled;
    rad::ExchangeResult exchange;
    std::uint64_t exchange_conservation_error = 0;
    rad::SolarResult solar;
};

[[nodiscard]] Traced trace_all(rad::RadiativeScene& scene) {
    // 2000 rays per face is one ray chunk per batch of 1000 emitters but two
    // in a single launch over all 2500, so the chunking differs as well.
    rad::TraceSettings settings;
    settings.rays_per_face = 2'000;
    settings.seed = 20261004;

    Traced traced;
    {
        rad::VfAccumulator acc(scene);
        scene.accumulate_vf(acc, settings);
        traced.vf = acc.result();
    }
    {
        rad::VfAccumulator acc(
            scene, rad::AccumConfig{.layout = rad::AccumLayout::Tiled,
                                    .tile_rows = tile_rows,
                                    .sparse_threshold = 0.0,
                                    .triangulation = {}});
        scene.accumulate_vf(acc, settings);
        traced.vf_tiled = acc.result();
    }
    {
        rad::ExchangeAccumulator acc(scene, rad::Band::IR);
        scene.accumulate_exchange(acc, settings);
        traced.exchange = acc.result();
        traced.exchange_conservation_error = acc.conservation_error();
    }
    {
        // Oblique, so plate A is partly lit through the gap and the gray
        // surfaces add a reflected component on top of the direct one.
        const rad::SolarState sun{.direction = {0.6, 0.3, -1.0},
                                  .irradiance = 1361.0};
        rad::SolarAccumulator acc(scene);
        scene.accumulate_solar(sun, acc, settings);
        traced.solar = acc.result();
    }
    return traced;
}

void require_vf_identical(const Traced& traced, const Traced& reference) {
    REQUIRE(csr_bit_identical(traced.vf.vf, reference.vf.vf));
    REQUIRE(traced.vf.row_sums.cwiseEqual(reference.vf.row_sums).all());
    REQUIRE(csr_bit_identical(traced.vf_tiled.vf, reference.vf_tiled.vf));
}

void require_exchange_and_solar_identical(const Traced& traced,
                                          const Traced& reference) {
    REQUIRE(
        csr_bit_identical(traced.exchange.factors, reference.exchange.factors));
    REQUIRE(traced.exchange_conservation_error == 0);
    REQUIRE(traced.solar.direct.cwiseEqual(reference.solar.direct).all());
    REQUIRE(traced.solar.total.cwiseEqual(reference.solar.total).all());
}

// Guards against comparing empty results with each other.
void require_nontrivial(const Traced& reference) {
    REQUIRE(reference.vf.vf.nonZeros() > 0);
    REQUIRE(reference.exchange.factors.nonZeros() > 0);
    REQUIRE(reference.solar.total.sum() > 0.0);
    REQUIRE((reference.solar.total - reference.solar.direct).sum() > 0.0);
}

void require_rejected(rad::Device& device, std::uint32_t limit) {
    REQUIRE_THROWS_AS(rad::detail::lower_max_dispatch_emitters(device, limit),
                      std::invalid_argument);
}

}  // namespace

TEST_CASE("radiative vulkan: batched emitter launches match one launch",
          "[radiative][gpu][vulkan]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
        return;
    }
    const auto model = make_meshed_plates();
    const std::array<std::array<float, 6>, 1> rows{gray_row(0.5F)};
    const std::vector<int> pair_rows(num_pairs, 0);
    rad::Device device = rad::Device::create();
    rad::RadiativeScene scene(device, model->mesh_parts(),
                              make_materials(rows, pair_rows));
    REQUIRE(scene.num_faces() == num_faces);

    // Vulkan guarantees at least 65535 rows, far above this scene, so the
    // reference is one launch per ray chunk.
    REQUIRE(rad::detail::max_dispatch_emitters(device) >= 65535U);
    const Traced reference = trace_all(scene);
    require_nontrivial(reference);

    // 1000 divides the face count, 997 does not; each step only lowers it.
    for (const std::uint32_t limit : {1000U, 997U}) {
        rad::detail::lower_max_dispatch_emitters(device, limit);
        REQUIRE(rad::detail::max_dispatch_emitters(device) == limit);
        const Traced traced = trace_all(scene);
        require_vf_identical(traced, reference);
        require_exchange_and_solar_identical(traced, reference);
    }
}

// Tiled exchange blocks are compacted on the device before the readback
// (SceneImpl::compact_block, kernels/compact.slang). Here a row is 2503 cells,
// ten workgroup chunks of the compaction kernel; small odd tiles give many
// blocks, an emitter subset leaves rows and blocks without emitters, and the
// second batch merges into rows the first one filled. Each must equal the
// Dense layout bit for bit.
TEST_CASE("radiative vulkan: compacted Tiled exchange blocks match Dense",
          "[radiative][gpu][vulkan][tiled]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
        return;
    }
    const auto model = make_meshed_plates();
    const std::array<std::array<float, 6>, 1> rows{gray_row(0.5F)};
    const std::vector<int> pair_rows(num_pairs, 0);
    rad::Device device = rad::Device::create();
    rad::RadiativeScene scene(device, model->mesh_parts(),
                              make_materials(rows, pair_rows));
    std::vector<std::uint32_t> subset;
    for (std::uint32_t face = 0; face < num_faces; face += 3) {
        subset.push_back(face);
    }
    for (const bool all_faces : {true, false}) {
        const std::span<const std::uint32_t> emitters =
            all_faces ? std::span<const std::uint32_t>{}
                      : std::span<const std::uint32_t>(subset);
        rad::TraceSettings settings;
        settings.rays_per_face = 500;
        rad::ExchangeAccumulator dense(scene, rad::Band::IR);
        settings.seed = 3;
        scene.accumulate_exchange(dense, settings, emitters);
        settings.seed = 4;
        scene.accumulate_exchange(dense, settings, emitters);
        const rad::ExchangeResult reference = dense.result();
        REQUIRE(reference.factors.nonZeros() > 0);
        for (const std::uint32_t block_rows : {7U, 256U, 1500U}) {
            rad::ExchangeAccumulator tiled(
                scene, rad::Band::IR,
                rad::AccumConfig{.layout = rad::AccumLayout::Tiled,
                                 .tile_rows = block_rows,
                                 .sparse_threshold = 0.0,
                                 .triangulation = {}});
            settings.seed = 3;
            scene.accumulate_exchange(tiled, settings, emitters);
            settings.seed = 4;
            scene.accumulate_exchange(tiled, settings, emitters);
            REQUIRE(csr_bit_identical(tiled.result().factors,
                                      reference.factors));
            REQUIRE(tiled.conservation_error() == 0);
        }
    }
}

TEST_CASE("radiative vulkan: the emitter-limit hook only lowers the limit",
          "[radiative][gpu][vulkan]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
        return;
    }
    rad::Device device = rad::Device::create();
    // Lowered first: some devices report UINT32_MAX, which has no value
    // above it to try.
    rad::detail::lower_max_dispatch_emitters(device, 1000U);
    // Raising the limit could exceed what the device accepts, and zero
    // would never launch anything.
    require_rejected(device, 1001U);
    require_rejected(device, 0U);
    REQUIRE(rad::detail::max_dispatch_emitters(device) == 1000U);
}
