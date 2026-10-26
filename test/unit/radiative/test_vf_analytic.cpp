#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/radiative/accumulators.hpp"
#include "pycanha-core/radiative/device.hpp"
#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/scene.hpp"
#include "pycanha-core/radiative/settings.hpp"
#include "scene_fixtures.hpp"

namespace rad = pycanha::radiative;
using radiative_fixtures::csr_bit_identical;
using radiative_fixtures::csr_value;
using radiative_fixtures::make_parallel_plates;

namespace {

// Closed-form view factor between identical, directly opposed parallel
// rectangles of size a x b at distance c (standard catalog formula).
[[nodiscard]] double parallel_plates_vf(double a, double b, double c) {
    const double x = a / c;
    const double y = b / c;
    const double x2 = x * x;
    const double y2 = y * y;
    const double term_log =
        0.5 * std::log((1.0 + x2) * (1.0 + y2) / (1.0 + x2 + y2));
    const double term_x =
        x * std::sqrt(1.0 + y2) * std::atan(x / std::sqrt(1.0 + y2));
    const double term_y =
        y * std::sqrt(1.0 + x2) * std::atan(y / std::sqrt(1.0 + x2));
    return 2.0 / (std::numbers::pi * x * y) *
           (term_log + term_x + term_y - (x * std::atan(x)) -
            (y * std::atan(y)));
}

// The explicit space column closes every emitted row to exactly one.
void require_closed_row(const rad::VfResult& result, Eigen::Index row,
                        Eigen::Index space_col, double expected_space) {
    REQUIRE(csr_value(result.vf, row, space_col) ==
            Catch::Approx(expected_space).margin(1e-12));
    REQUIRE(result.row_sums(row) == Catch::Approx(1.0).margin(1e-12));
}

// Bit-exact equality of two VF results (integer counting cells make this a
// hard guarantee, not a tolerance check).
void require_bit_identical(const rad::VfResult& result,
                           const rad::VfResult& reference) {
    REQUIRE(csr_bit_identical(result.vf, reference.vf));
    REQUIRE(result.row_sums.size() == reference.row_sums.size());
    REQUIRE(result.row_sums.cwiseEqual(reference.row_sums).all());
}

}  // namespace

TEST_CASE("radiative vf: parallel plates match the analytic value",
          "[radiative][gpu][vf]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
        return;
    }
    const auto model = make_parallel_plates(1.0);
    rad::Device device = rad::Device::create();
    rad::RadiativeScene scene(device, model->mesh_parts(),
                              model->material_table());
    rad::VfAccumulator acc(scene);

    rad::TraceSettings settings;
    settings.rays_per_face = 20'000;
    settings.seed = 7;
    const std::vector<std::uint32_t> emitters{0};
    scene.accumulate_vf(acc, settings, emitters);
    const rad::VfResult result = acc.result();

    const double expected = parallel_plates_vf(1.0, 1.0, 1.0);
    const double vf = csr_value(result.vf, 0, 2);
    // 4-sigma tolerance of the binomial estimate at 20k rays.
    const double tolerance =
        4.0 * std::sqrt(expected * (1.0 - expected) / 20'000.0);
    REQUIRE(vf == Catch::Approx(expected).margin(tolerance));
    // Every first hit from plate A lands on plate B's facing side (face 2)
    // or scores the virtual space column: the row closes to exactly one.
    require_closed_row(result, 0, static_cast<std::int32_t>(scene.num_faces()),
                       1.0 - vf);
    REQUIRE(result.stats.rays_per_face == 20'000);
    REQUIRE(result.stats.total_rays == 20'000);
    REQUIRE(result.stats.max_stderr > 0.0);
}

TEST_CASE("radiative vf: same seed reproduces bit-identical results",
          "[radiative][gpu][vf]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
        return;
    }
    const auto model = make_parallel_plates(1.0);
    rad::Device device = rad::Device::create();
    rad::RadiativeScene scene(device, model->mesh_parts(),
                              model->material_table());

    rad::TraceSettings settings;
    settings.rays_per_face = 5'000;
    settings.seed = 42;

    rad::VfAccumulator first(scene);
    scene.accumulate_vf(first, settings);
    rad::VfAccumulator second(scene);
    scene.accumulate_vf(second, settings);

    require_bit_identical(second.result(), first.result());
}

TEST_CASE("radiative vf: identity instances match the monolithic scene",
          "[radiative][gpu][vf][scene]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
        return;
    }
    const auto model = make_parallel_plates(1.0);
    rad::Device device = rad::Device::create();

    rad::TraceSettings settings;
    settings.rays_per_face = 5'000;
    settings.seed = 3;
    const std::vector<std::uint32_t> emitters{0};

    // Same geometry once as one part, once split into two rigid parts with
    // identity placements: the counting must be bit-identical because the
    // RNG is keyed on (face, ray index, seed), not on scene structure.
    rad::RadiativeScene monolithic(device, model->mesh_parts(),
                                   model->material_table());
    const std::vector<std::string> split{"plate_b"};
    rad::RadiativeScene instanced(device, model->mesh_parts(split),
                                  model->material_table());

    rad::VfAccumulator acc_mono(monolithic);
    monolithic.accumulate_vf(acc_mono, settings, emitters);
    rad::VfAccumulator acc_inst(instanced);
    instanced.accumulate_vf(acc_inst, settings, emitters);

    require_bit_identical(acc_inst.result(), acc_mono.result());
}

TEST_CASE("radiative vf: tiled layout is bit-identical to dense",
          "[radiative][gpu][vf]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
        return;
    }
    const auto model = make_parallel_plates(1.0);
    rad::Device device = rad::Device::create();
    rad::RadiativeScene scene(device, model->mesh_parts(),
                              model->material_table());

    rad::TraceSettings settings;
    settings.rays_per_face = 5'000;
    settings.seed = 11;

    rad::VfAccumulator dense(scene);
    scene.accumulate_vf(dense, settings);
    const rad::VfResult reference = dense.result();

    // Several tile sizes, including 1 row and num_faces - 1 (a block split
    // that exercises the row_offset bookkeeping hardest).
    for (const std::uint32_t tile_rows : {1U, 2U, 3U}) {
        rad::VfAccumulator tiled(
            scene, rad::AccumConfig{.layout = rad::AccumLayout::Tiled,
                                    .tile_rows = tile_rows,
                                    .sparse_threshold = 0.0,
                                    .triangulation = {}});
        scene.accumulate_vf(tiled, settings);
        require_bit_identical(tiled.result(), reference);
    }
}

TEST_CASE("radiative vf: batches accumulate", "[radiative][gpu][vf]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
        return;
    }
    const auto model = make_parallel_plates(1.0);
    rad::Device device = rad::Device::create();
    rad::RadiativeScene scene(device, model->mesh_parts(),
                              model->material_table());
    rad::VfAccumulator acc(scene);

    rad::TraceSettings settings;
    settings.rays_per_face = 2'000;
    settings.seed = 1;
    scene.accumulate_vf(acc, settings);
    settings.seed = 2;  // a new batch must use a fresh seed
    scene.accumulate_vf(acc, settings);

    const rad::VfResult result = acc.result();
    REQUIRE(result.stats.rays_per_face == 4'000);
    // The explicit space column closes every emitted row exactly.
    REQUIRE(result.row_sums(0) == Catch::Approx(1.0).margin(1e-12));
}

TEST_CASE("radiative vf: Sobol emission is unbiased and beats random sampling",
          "[radiative][gpu][vf][qmc]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
        return;
    }
    const auto model = make_parallel_plates(1.0);
    rad::Device device = rad::Device::create();
    rad::RadiativeScene scene(device, model->mesh_parts(),
                              model->material_table());
    const double expected = parallel_plates_vf(1.0, 1.0, 1.0);
    const std::vector<std::uint32_t> emitters{0};
    constexpr std::uint32_t rays = 1U << 14U;
    constexpr int seeds = 8;

    // Mean square error over independent seeds (independent scramblings for
    // Sobol), for each sampler.
    auto mse = [&](rad::Sampler sampler) {
        double sum = 0.0;
        for (int seed = 1; seed <= seeds; ++seed) {
            rad::VfAccumulator acc(scene);
            rad::TraceSettings settings;
            settings.rays_per_face = rays;
            settings.seed = static_cast<std::uint32_t>(seed);
            settings.sampler = sampler;
            scene.accumulate_vf(acc, settings, emitters);
            const rad::VfResult result = acc.result();
            const double error = csr_value(result.vf, 0, 2) - expected;
            sum += error * error;
            // The space column still closes the row exactly.
            REQUIRE(result.row_sums(0) == Catch::Approx(1.0).margin(1e-12));
        }
        return sum / seeds;
    };
    const double binomial_variance = expected * (1.0 - expected) / rays;
    const double sobol = mse(rad::Sampler::Sobol);
    const double random = mse(rad::Sampler::Random);
    // Unbiased: the Sobol RMS error is within the random sampling's spread.
    REQUIRE(std::sqrt(sobol) < 3.0 * std::sqrt(binomial_variance));
    // Lower error: at 16k rays on this smooth integrand the scrambled net's
    // variance is far below the binomial one (about 40x fewer rays for the
    // same error measured offline); a factor of 4 in variance leaves margin.
    REQUIRE(sobol < 0.25 * binomial_variance);
    REQUIRE(random > sobol);
}

TEST_CASE("radiative vf: Sobol emission is reproducible and seeded",
          "[radiative][gpu][vf][qmc]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
        return;
    }
    const auto model = make_parallel_plates(1.0);
    rad::Device device = rad::Device::create();
    rad::RadiativeScene scene(device, model->mesh_parts(),
                              model->material_table());

    rad::TraceSettings settings;
    settings.rays_per_face = 5'000;
    settings.seed = 42;
    settings.sampler = rad::Sampler::Sobol;
    rad::VfAccumulator first(scene);
    scene.accumulate_vf(first, settings);
    const rad::VfResult first_result = first.result();
    rad::VfAccumulator second(scene);
    scene.accumulate_vf(second, settings);
    require_bit_identical(second.result(), first_result);

    // Another seed is another scrambling: a different estimate.
    settings.seed = 43;
    rad::VfAccumulator third(scene);
    scene.accumulate_vf(third, settings);
    REQUIRE_FALSE(csr_bit_identical(third.result().vf, first_result.vf));

    // The default sampler after a Sobol trace still gives the Random result.
    settings.seed = 42;
    settings.sampler = rad::Sampler::Random;
    rad::VfAccumulator random_after(scene);
    scene.accumulate_vf(random_after, settings);
    rad::RadiativeScene fresh(device, model->mesh_parts(),
                              model->material_table());
    rad::VfAccumulator random_fresh(fresh);
    fresh.accumulate_vf(random_fresh, settings);
    require_bit_identical(random_after.result(), random_fresh.result());
}
