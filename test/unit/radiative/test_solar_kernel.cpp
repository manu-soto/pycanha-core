#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>

#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"
#include "pycanha-core/radiative/accumulators.hpp"
#include "pycanha-core/radiative/device.hpp"
#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/scene.hpp"
#include "pycanha-core/radiative/settings.hpp"
#include "scene_fixtures.hpp"

namespace rad = pycanha::radiative;
using radiative_fixtures::gray_row;
using radiative_fixtures::make_materials;
using radiative_fixtures::make_parallel_plates;

namespace {

constexpr double solar_constant = 1361.0;

// One unit plate at z = 0, side 1 (face 0) facing +z.
[[nodiscard]] std::unique_ptr<pycanha::gmm::GeometryModel> make_single_plate() {
    using pycanha::gmm::GeometryItem;
    using pycanha::gmm::Rectangle;
    using pycanha::gmm::ThermalMesh;
    auto model = std::make_unique<pycanha::gmm::GeometryModel>("plate");
    model->add(std::make_shared<GeometryItem>(
        "plate", Rectangle({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}),
        ThermalMesh{}));
    return model;
}

// The mirror-bench source and 45-degree mirror WITHOUT the catcher (which
// would shade a sun coming from +x): faces 0/1 source, 2/3 mirror.
[[nodiscard]] std::unique_ptr<pycanha::gmm::GeometryModel>
make_source_and_mirror() {
    using pycanha::gmm::GeometryItem;
    using pycanha::gmm::Rectangle;
    using pycanha::gmm::ThermalMesh;
    auto model = std::make_unique<pycanha::gmm::GeometryModel>("sun_mirror");
    model->add(std::make_shared<GeometryItem>(
        "source", Rectangle({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}),
        ThermalMesh{}));
    model->add(std::make_shared<GeometryItem>(
        "mirror",
        Rectangle({-0.914214, -1.5, 0.585786}, {-0.914214, 2.5, 0.585786},
                  {1.914214, -1.5, 3.414214}),
        ThermalMesh{}));
    return model;
}

// One sun angle of the cosine-law scene: a flat black plate under a
// parallel sun deposits the same cos(theta) at every sample, so the only
// error left is f32 geometry round-off.
void check_cosine_angle(rad::RadiativeScene& scene,
                        const rad::TraceSettings& settings, double angle_deg) {
    const double theta = angle_deg * std::numbers::pi / 180.0;
    // Sun -> scene: tilted away from -z by theta.
    const rad::SolarState sun{
        .direction = {std::sin(theta), 0.0, -std::cos(theta)},
        .irradiance = solar_constant};
    rad::SolarAccumulator acc(scene);
    scene.accumulate_solar(sun, acc, settings);
    const rad::SolarResult result = acc.result();

    const double expected = solar_constant * std::cos(theta);
    REQUIRE(result.direct(0) ==
            Catch::Approx(expected).margin(1e-3 * solar_constant));
    REQUIRE(result.total(0) == result.direct(0));
    // The back side never sees the sun.
    REQUIRE(result.direct(1) == 0.0);
    REQUIRE(result.total(1) == 0.0);
}

// Bit-equality of two solar results (integer cells make this exact).
void require_bit_identical(const rad::SolarResult& result,
                           const rad::SolarResult& reference) {
    REQUIRE(result.direct.cwiseEqual(reference.direct).all());
    REQUIRE(result.total.cwiseEqual(reference.total).all());
}

}  // namespace

TEST_CASE("radiative solar: direct flux follows the cosine law",
          "[radiative][gpu][solar]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
        return;
    }
    const auto model = make_single_plate();
    const std::array<std::array<float, 6>, 1> rows{gray_row(1.0F)};
    const std::array<int, 1> pair_rows{0};
    rad::Device device = rad::Device::create();
    rad::RadiativeScene scene(device, model->mesh_parts(),
                              make_materials(rows, pair_rows));

    rad::TraceSettings settings;
    settings.rays_per_face = 5'000;
    settings.seed = 3;

    for (const double angle_deg : {0.0, 45.0, 60.0}) {
        check_cosine_angle(scene, settings, angle_deg);
    }
}

TEST_CASE("radiative solar: occlusion shadows completely",
          "[radiative][gpu][solar]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
        return;
    }
    const auto model = make_parallel_plates(1.0);
    const std::array<std::array<float, 6>, 1> rows{gray_row(1.0F)};
    const std::array<int, 2> pair_rows{0, 0};
    rad::Device device = rad::Device::create();
    rad::RadiativeScene scene(device, model->mesh_parts(),
                              make_materials(rows, pair_rows));

    const rad::SolarState sun{.direction = {0.0, 0.0, -1.0},
                              .irradiance = solar_constant};
    rad::TraceSettings settings;
    settings.rays_per_face = 5'000;
    settings.seed = 7;
    rad::SolarAccumulator acc(scene);
    scene.accumulate_solar(sun, acc, settings);
    const rad::SolarResult result = acc.result();

    // Plate B sits exactly between plate A and the sun: every shadow ray
    // from A hits it, so A gets zero — the eclipse mechanism.
    REQUIRE(result.direct(0) == 0.0);
    // B's top side (face 3) is in full sun; its bottom side faces away.
    REQUIRE(result.direct(3) ==
            Catch::Approx(solar_constant).margin(1e-3 * solar_constant));
    REQUIRE(result.direct(2) == 0.0);
}

TEST_CASE("radiative solar: overlapping occluders shade their union",
          "[radiative][gpu][solar]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
        return;
    }
    // A unit plate at z = 0 under a sun from +z, and above it three strips
    // across it that overlap: x in [0, 0.5] at z = 1, [0.25, 0.75] at z = 2
    // and [0.1, 0.4] at z = 3. A shadow ray from x < 0.75 meets one to three
    // of them; the plate keeps its sun on x in [0.75, 1] only.
    using pycanha::gmm::GeometryItem;
    using pycanha::gmm::Rectangle;
    using pycanha::gmm::ThermalMesh;
    auto model = std::make_unique<pycanha::gmm::GeometryModel>("stack");
    model->add(std::make_shared<GeometryItem>(
        "plate", Rectangle({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}),
        ThermalMesh{}));
    const std::array<std::array<double, 3>, 3> strips{
        {{0.0, 0.5, 1.0}, {0.25, 0.75, 2.0}, {0.1, 0.4, 3.0}}};
    for (std::size_t index = 0; index < strips.size(); ++index) {
        const auto& [x0, x1, z] = strips.at(index);
        model->add(std::make_shared<GeometryItem>(
            "strip_" + std::to_string(index),
            Rectangle({x0, -0.5, z}, {x1, -0.5, z}, {x0, 1.5, z}),
            ThermalMesh{}));
    }
    const std::array<std::array<float, 6>, 1> rows{gray_row(1.0F)};
    const std::array<int, 4> pair_rows{0, 0, 0, 0};
    rad::Device device = rad::Device::create();
    rad::RadiativeScene scene(device, model->mesh_parts(),
                              make_materials(rows, pair_rows));

    const rad::SolarState sun{.direction = {0.0, 0.0, -1.0},
                              .irradiance = solar_constant};
    rad::TraceSettings settings;
    settings.rays_per_face = 20'000;
    settings.seed = 5;
    rad::SolarAccumulator acc(scene);
    scene.accumulate_solar(sun, acc, settings);
    const rad::SolarResult result = acc.result();

    const double expected = 0.25 * solar_constant;
    const double sigma =
        solar_constant *
        std::sqrt(0.25 * 0.75 / static_cast<double>(settings.rays_per_face));
    REQUIRE(result.direct(0) == Catch::Approx(expected).margin(5.0 * sigma));
    // The top strip is in full sun; the plate's underside never sees it.
    REQUIRE(result.direct(6) == Catch::Approx(0.3 * 2.0 * solar_constant)
                                    .margin(1e-3 * 0.6 * solar_constant));
    REQUIRE(result.direct(1) == 0.0);
}

TEST_CASE("radiative solar: mirrors add a reflected component",
          "[radiative][gpu][solar]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
        return;
    }
    const auto model = make_source_and_mirror();
    const std::array<std::array<float, 6>, 2> rows{
        gray_row(1.0F),  // source: black
        std::array<float, 6>{1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F}};  // mirror
    const std::array<int, 2> pair_rows{0, 1};
    rad::Device device = rad::Device::create();
    rad::RadiativeScene scene(device, model->mesh_parts(),
                              make_materials(rows, pair_rows));

    // Horizontal sun from +x: the source plate is edge-on (zero direct),
    // but the 45-degree mirror folds the beam straight down onto it.
    const rad::SolarState sun{.direction = {-1.0, 0.0, 0.0},
                              .irradiance = solar_constant};
    rad::TraceSettings settings;
    settings.rays_per_face = 20'000;
    settings.seed = 11;
    rad::SolarAccumulator acc(scene);
    scene.accumulate_solar(sun, acc, settings);
    const rad::SolarResult result = acc.result();

    REQUIRE(result.direct(0) == 0.0);
    // A perfect flat mirror conserves the parallel beam's flux density, so
    // the source sits under one full reflected sun (the area weighting of
    // the deposits is what makes this come out right).
    REQUIRE(result.total(0) ==
            Catch::Approx(solar_constant).margin(0.1 * solar_constant));
    // The mirror itself absorbs nothing in the solar band.
    REQUIRE(result.direct(2) == 0.0);
    REQUIRE(result.total(2) == 0.0);
}

TEST_CASE("radiative solar: determinism, additivity and sun consistency",
          "[radiative][gpu][solar]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
        return;
    }
    const auto model = make_parallel_plates(1.0);
    const std::array<std::array<float, 6>, 1> rows{gray_row(0.5F)};
    const std::array<int, 2> pair_rows{0, 0};
    rad::Device device = rad::Device::create();
    rad::RadiativeScene scene(device, model->mesh_parts(),
                              make_materials(rows, pair_rows));

    const rad::SolarState sun{.direction = {0.3, -0.2, -1.0},
                              .irradiance = solar_constant};
    rad::TraceSettings settings;
    settings.rays_per_face = 5'000;
    settings.seed = 19;

    rad::SolarAccumulator first(scene);
    scene.accumulate_solar(sun, first, settings);
    rad::SolarAccumulator second(scene);
    scene.accumulate_solar(sun, second, settings);
    const rad::SolarResult run_a = first.result();
    require_bit_identical(second.result(), run_a);

    // Batches with fresh seeds refine the same estimate.
    settings.seed = 20;
    scene.accumulate_solar(sun, first, settings);
    const rad::SolarResult combined = first.result();
    REQUIRE(combined.stats.rays_per_face == 10'000);
    REQUIRE(combined.direct(3) ==
            Catch::Approx(run_a.direct(3))
                .margin((4.0 * run_a.stats.max_stderr) + 1e-6));

    // A different sun in the same accumulator would be a meaningless sum.
    const rad::SolarState other_sun{.direction = {0.0, 0.0, -1.0},
                                    .irradiance = solar_constant};
    REQUIRE_THROWS_AS(scene.accumulate_solar(other_sun, first, settings),
                      std::invalid_argument);
}
