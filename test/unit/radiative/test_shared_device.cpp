// Several scenes on one Device. Each acceleration-structure build takes its
// scratch memory from the Device's allocator, next to the buffers of the
// other scenes alive on that Device, and the build needs the scratch address
// aligned to the device's minAccelerationStructureScratchOffsetAlignment. A
// misaligned scratch builds a wrong BVH without any error: a scene built
// while a larger scene was alive lost occluders and lit fewer faces. A scene
// must trace the same, bit for bit, whatever else lives on its Device.

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
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
#include "scene_fixtures.hpp"

namespace rad = pycanha::radiative;
using radiative_fixtures::gray_row;
using radiative_fixtures::make_materials;

namespace {

constexpr std::size_t fins_per_side = 8;
constexpr std::size_t fin_cells = 10;

[[nodiscard]] std::vector<double> uniform_cuts(std::size_t cells) {
    std::vector<double> cuts(cells + 1);
    for (std::size_t i = 0; i < cuts.size(); ++i) {
        cuts[i] = static_cast<double>(i) / static_cast<double>(cells);
    }
    return cuts;
}

// A 2 x 2 plate of plate_cells x plate_cells face pairs at z = 0 (side 1 up),
// shaded by a grid of 8 x 8 small vertical fins of 10 x 10 face pairs each,
// turned at different angles. Face pairs: the plate's first, then the fins'.
[[nodiscard]] std::unique_ptr<pycanha::gmm::GeometryModel> make_shaded_plate(
    std::size_t plate_cells) {
    using pycanha::gmm::GeometryItem;
    using pycanha::gmm::Rectangle;
    using pycanha::gmm::ThermalMesh;
    auto model = std::make_unique<pycanha::gmm::GeometryModel>("shaded");
    model->add(std::make_shared<GeometryItem>(
        "plate", Rectangle({0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 2.0, 0.0}),
        ThermalMesh(uniform_cuts(plate_cells), uniform_cuts(plate_cells))));
    const double step = 2.0 / static_cast<double>(fins_per_side);
    for (std::size_t i = 0; i < fins_per_side; ++i) {
        for (std::size_t j = 0; j < fins_per_side; ++j) {
            const double x0 = (static_cast<double>(i) + 0.25) * step;
            const double y0 = (static_cast<double>(j) + 0.2) * step;
            const double angle =
                0.6 * (static_cast<double>(i) - static_cast<double>(j)) /
                static_cast<double>(fins_per_side);
            const double dx = 0.5 * step * std::cos(angle);
            const double dy = 0.5 * step * std::sin(angle);
            model->add(std::make_shared<GeometryItem>(
                "fin_" + std::to_string(i) + "_" + std::to_string(j),
                Rectangle({x0, y0, 0.0}, {x0 + dx, y0 + dy, 0.0},
                          {x0, y0, 0.6 * step}),
                ThermalMesh(uniform_cuts(fin_cells), uniform_cuts(fin_cells))));
        }
    }
    return model;
}

// White plate (absorptivity 0.25), darker fins (0.6), all diffuse.
[[nodiscard]] rad::MaterialTable shaded_plate_materials(
    std::size_t plate_cells) {
    const std::array<std::array<float, 6>, 2> rows{gray_row(0.25F),
                                                   gray_row(0.6F)};
    const std::size_t plate_pairs = plate_cells * plate_cells;
    const std::size_t fin_pairs =
        fins_per_side * fins_per_side * fin_cells * fin_cells;
    std::vector<int> pair_rows(plate_pairs + fin_pairs, 1);
    for (std::size_t p = 0; p < plate_pairs; ++p) {
        pair_rows[p] = 0;
    }
    return make_materials(rows, pair_rows);
}

[[nodiscard]] rad::SolarResult trace_sun(rad::RadiativeScene& scene) {
    const rad::SolarState sun{.direction = {-0.4, -0.3, -0.866},
                              .irradiance = 1361.0};
    rad::TraceSettings settings;
    settings.rays_per_face = 3'000;
    settings.seed = 7;
    rad::SolarAccumulator acc(scene);
    scene.accumulate_solar(sun, acc, settings);
    return acc.result();
}

void require_bit_identical(const rad::SolarResult& result,
                           const rad::SolarResult& reference) {
    REQUIRE(result.direct.size() == reference.direct.size());
    REQUIRE(result.direct.cwiseEqual(reference.direct).all());
    REQUIRE(result.total.cwiseEqual(reference.total).all());
}

}  // namespace

TEST_CASE("radiative: a scene traces the same whatever else is on its Device",
          "[radiative][gpu][solar]") {
    if (!rad::is_available()) {
        SUCCEED("no RT-capable GPU device: skipped");
        return;
    }
    constexpr std::size_t small_cells = 250;
    constexpr std::size_t large_cells = 300;
    const auto small = make_shaded_plate(small_cells);
    const auto large = make_shaded_plate(large_cells);

    // The small scene alone on a Device of its own: the reference.
    rad::SolarResult reference;
    {
        rad::Device device = rad::Device::create();
        rad::RadiativeScene scene(device, small->mesh_parts(),
                                  shaded_plate_materials(small_cells));
        reference = trace_sun(scene);
    }
    REQUIRE(reference.total.sum() > 0.0);

    SECTION("built while a larger scene is alive on the same Device") {
        rad::Device device = rad::Device::create();
        rad::RadiativeScene larger(device, large->mesh_parts(),
                                   shaded_plate_materials(large_cells));
        rad::RadiativeScene scene(device, small->mesh_parts(),
                                  shaded_plate_materials(small_cells));
        REQUIRE(larger.num_faces() > scene.num_faces());
        require_bit_identical(trace_sun(scene), reference);
    }
    SECTION("built before a larger scene on the same Device") {
        rad::Device device = rad::Device::create();
        rad::RadiativeScene scene(device, small->mesh_parts(),
                                  shaded_plate_materials(small_cells));
        rad::RadiativeScene larger(device, large->mesh_parts(),
                                   shaded_plate_materials(large_cells));
        REQUIRE(larger.num_faces() > scene.num_faces());
        require_bit_identical(trace_sun(scene), reference);
    }
}
