#include <catch2/catch_test_macros.hpp>

#include "pycanha-core/radiative/radiative.hpp"

namespace rad = pycanha::radiative;

TEST_CASE("radiative types: trace settings defaults", "[radiative]") {
    const rad::TraceSettings settings;
    REQUIRE(settings.rays_per_face == 10'000);
    REQUIRE(settings.seed == 0);
    REQUIRE(settings.max_bounces == 10'000);
}

TEST_CASE("radiative types: accumulator config defaults", "[radiative]") {
    const rad::AccumConfig config;
    REQUIRE(config.layout == rad::AccumLayout::Dense);
    REQUIRE(config.tile_rows == 0);
    REQUIRE(config.sparse_threshold == 0.0);
}

TEST_CASE("radiative types: an untraced result carries an empty matrix",
          "[radiative]") {
    const rad::VfResult result;
    REQUIRE(result.vf.rows() == 0);
    REQUIRE(result.vf.cols() == 0);
    REQUIRE(result.vf.nonZeros() == 0);
}

TEST_CASE("radiative types: results carry statistics", "[radiative]") {
    const rad::VfResult vf_result;
    REQUIRE(vf_result.stats.total_rays == 0);
    const rad::ExchangeResult exchange_result;
    REQUIRE(exchange_result.band == rad::Band::IR);
    const rad::SolarResult solar_result;
    REQUIRE(solar_result.direct.size() == 0);
}
