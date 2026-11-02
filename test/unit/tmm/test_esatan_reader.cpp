#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>  // NOLINT(misc-include-cleaner)
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/io/esatan.hpp"
#include "pycanha-core/solvers/settings.hpp"
#include "pycanha-core/solvers/steady_state.hpp"
#include "pycanha-core/thermaldata/data_model.hpp"
#include "pycanha-core/thermaldata/dense_time_series.hpp"
#include "pycanha-core/thermaldata/named_constants.hpp"
#include "pycanha-core/thermaldata/thermaldata.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"

namespace {

// The iteration limit and the convergence tolerance [K] of a solver.
pycanha::solvers::Settings iteration_settings(int max_iterations,
                                              double tolerance) {
    pycanha::solvers::Settings settings;
    settings.max_iterations = max_iterations;
    settings.tolerance = tolerance;
    return settings;
}

std::filesystem::path get_esatan_data_dir() {
    const std::filesystem::path this_file(__FILE__);
    const std::filesystem::path test_root =
        this_file.parent_path().parent_path().parent_path();
    return test_root / "data" / "esatan";
}

std::filesystem::path get_reference_tmd_path() {
    return get_esatan_data_dir() / "DISCTR_TRANSIENT.TMD";
}

std::filesystem::path get_steady_tmd_path() {
    return get_esatan_data_dir() / "DISCTR_STEADY.TMD";
}

// Trims ESATAN space padding from a fixed-width character value. Numeric
// values are right-justified (leading spaces), text values are left-justified
// (trailing spaces), so strip both sides.
std::string_view trim_spaces(std::string_view value) {
    while (!value.empty() && (value.front() == ' ')) {
        value.remove_prefix(1);
    }
    while (!value.empty() && (value.back() == ' ')) {
        value.remove_suffix(1);
    }
    return value;
}

std::unordered_map<pycanha::Index, pycanha::Index> build_column_lookup(
    const std::vector<pycanha::Index>& node_numbers) {
    std::unordered_map<pycanha::Index, pycanha::Index> lookup;
    lookup.reserve(node_numbers.size());

    for (std::size_t i = 0; i < node_numbers.size(); ++i) {
        lookup.emplace(node_numbers[i], static_cast<pycanha::Index>(i));
    }

    return lookup;
}

void require_monotonic_times(const pycanha::DenseTimeSeries& series) {
    for (pycanha::Index i = 1; i < series.num_timesteps(); ++i) {
        REQUIRE(series.times()(i) >= series.times()(i - 1));
    }
}

void require_default_transient_series(
    const pycanha::DataModel& model,
    const std::vector<pycanha::Index>& node_numbers) {
    for (const auto attribute :
         {pycanha::DataModelAttribute::T, pycanha::DataModelAttribute::C,
          pycanha::DataModelAttribute::QA, pycanha::DataModelAttribute::QE,
          pycanha::DataModelAttribute::QI, pycanha::DataModelAttribute::QR,
          pycanha::DataModelAttribute::QS}) {
        const auto& series = model.get_dense_attribute(attribute);
        REQUIRE(series.num_timesteps() > 1);
        REQUIRE(series.num_columns() ==
                static_cast<pycanha::Index>(node_numbers.size()));
        require_monotonic_times(series);
    }
}
}  // namespace

TEST_CASE("ESATANReader can import real TMD and model is solvable",
          "[tmm][esatan]") {
    const std::filesystem::path reference_tmd_path = get_reference_tmd_path();
    REQUIRE(std::filesystem::exists(reference_tmd_path));

    auto model = std::make_shared<pycanha::ThermalMathematicalModel>(
        "esatan-real-file-smoke-test");
    pycanha::ESATANReader reader(*model);
    REQUIRE_NOTHROW(reader.read_tmd(reference_tmd_path.string()));

    REQUIRE(model->nodes().get_num_nodes() > 0);

    pycanha::solvers::steady::SteadyState solver(model);
    solver.set_settings(iteration_settings(2, 1.0));

    REQUIRE_NOTHROW(solver.initialize());
    REQUIRE(solver.initialized());
    REQUIRE_NOTHROW(solver.solve());
}

TEST_CASE("read_tmd_transient imports default transient node attributes",
          "[tmm][esatan][thermaldata]") {
    const std::filesystem::path reference_tmd_path = get_reference_tmd_path();
    REQUIRE(std::filesystem::exists(reference_tmd_path));

    pycanha::ThermalData thermal_data;
    const auto node_numbers = pycanha::read_tmd_transient(
        reference_tmd_path.string(), thermal_data, "transient");

    REQUIRE(!node_numbers.empty());
    REQUIRE(thermal_data.models().has_model("transient"));
    require_default_transient_series(
        thermal_data.models().get_model("transient"), node_numbers);
}

TEST_CASE(
    "read_tmd_transient temperature matches steady-state import at the "
    "first timestep",
    "[tmm][esatan][thermaldata]") {
    const std::filesystem::path reference_tmd_path = get_reference_tmd_path();
    REQUIRE(std::filesystem::exists(reference_tmd_path));

    auto model = std::make_shared<pycanha::ThermalMathematicalModel>(
        "esatan-transient-cross-check");
    pycanha::ESATANReader reader(*model);
    REQUIRE_NOTHROW(reader.read_tmd(reference_tmd_path.string()));

    pycanha::ThermalData thermal_data;
    const auto node_numbers = pycanha::read_tmd_transient(
        reference_tmd_path.string(), thermal_data, "transient");
    const auto column_lookup = build_column_lookup(node_numbers);
    const auto& temperature_series =
        thermal_data.models().get_model("transient").T();

    for (pycanha::Index i = 0; i < model->nodes().get_num_nodes(); ++i) {
        pycanha::Node node = model->nodes().get_node_from_idx(i);
        const auto iterator = column_lookup.find(node.get_node_num());
        REQUIRE(iterator != column_lookup.end());
        REQUIRE(temperature_series.values()(0, iterator->second) ==
                Catch::Approx(node.get_T()));
    }
}

TEST_CASE("read_tmd_transient preserves inactive nodes in returned columns",
          "[tmm][esatan][thermaldata]") {
    const std::filesystem::path reference_tmd_path = get_reference_tmd_path();
    REQUIRE(std::filesystem::exists(reference_tmd_path));

    auto model = std::make_shared<pycanha::ThermalMathematicalModel>(
        "esatan-inactive-node-check");
    pycanha::ESATANReader reader(*model);
    REQUIRE_NOTHROW(reader.read_tmd(reference_tmd_path.string()));

    pycanha::ThermalData thermal_data;
    const auto node_numbers = pycanha::read_tmd_transient(
        reference_tmd_path.string(), thermal_data, "transient");

    REQUIRE(node_numbers.size() == 103U);
    REQUIRE(model->nodes().get_num_nodes() == 102);

    const auto& temperature_series =
        thermal_data.models().get_model("transient").T();
    REQUIRE(temperature_series.num_columns() ==
            static_cast<pycanha::Index>(node_numbers.size()));
}

TEST_CASE("read_tmd_transient can load only requested attributes",
          "[tmm][esatan][thermaldata]") {
    const std::filesystem::path reference_tmd_path = get_reference_tmd_path();
    REQUIRE(std::filesystem::exists(reference_tmd_path));

    pycanha::ThermalData thermal_data;
    const auto node_numbers = pycanha::read_tmd_transient(
        reference_tmd_path.string(), thermal_data, "single",
        /*overwrite=*/false, {pycanha::DataModelAttribute::T});

    REQUIRE(!node_numbers.empty());
    const auto& model_data = thermal_data.models().get_model("single");
    REQUIRE(model_data.T().num_timesteps() > 0);
    REQUIRE(model_data.C().num_timesteps() == 0);
    REQUIRE(model_data.QA().num_timesteps() == 0);
    REQUIRE(model_data.QE().num_timesteps() == 0);
    REQUIRE(model_data.QI().num_timesteps() == 0);
    REQUIRE(model_data.QR().num_timesteps() == 0);
    REQUIRE(model_data.QS().num_timesteps() == 0);
}

TEST_CASE("read_tmd_transient rejects unsupported JAC attribute",
          "[tmm][esatan][thermaldata]") {
    const std::filesystem::path reference_tmd_path = get_reference_tmd_path();
    REQUIRE(std::filesystem::exists(reference_tmd_path));

    pycanha::ThermalData thermal_data;
    REQUIRE_THROWS_AS(
        pycanha::read_tmd_transient(
            reference_tmd_path.string(), thermal_data, "jac",
            /*overwrite=*/false, {pycanha::DataModelAttribute::JAC}),
        std::invalid_argument);
}

TEST_CASE("read_tmd_transient throws before writing when overwrite is disabled",
          "[tmm][esatan][thermaldata]") {
    const std::filesystem::path reference_tmd_path = get_reference_tmd_path();
    REQUIRE(std::filesystem::exists(reference_tmd_path));

    pycanha::ThermalData thermal_data;
    pycanha::DataModel existing_model({1});
    existing_model.QS().resize(1, 1);
    existing_model.QS().times()(0) = 42.0;
    existing_model.QS().values()(0, 0) = 99.0;
    thermal_data.models().add_model("case", std::move(existing_model));

    REQUIRE_THROWS_AS(pycanha::read_tmd_transient(reference_tmd_path.string(),
                                                  thermal_data, "case"),
                      std::runtime_error);

    const auto& model_data = thermal_data.models().get_model("case");
    REQUIRE(model_data.T().num_timesteps() == 0);
    REQUIRE(model_data.C().num_timesteps() == 0);
    REQUIRE(model_data.QS().num_timesteps() == 1);
    REQUIRE(model_data.QS().num_columns() == 1);
    REQUIRE(model_data.QS().times()(0) == Catch::Approx(42.0));
    REQUIRE(model_data.QS().values()(0, 0) == Catch::Approx(99.0));
}

TEST_CASE("read_tmd_transient overwrites existing series when requested",
          "[tmm][esatan][thermaldata]") {
    const std::filesystem::path reference_tmd_path = get_reference_tmd_path();
    REQUIRE(std::filesystem::exists(reference_tmd_path));

    pycanha::ThermalData thermal_data;
    pycanha::DataModel existing_model({1});
    existing_model.QS().resize(1, 1);
    thermal_data.models().add_model("case", std::move(existing_model));

    const auto node_numbers = pycanha::read_tmd_transient(
        reference_tmd_path.string(), thermal_data, "case",
        /*overwrite=*/true);

    REQUIRE(thermal_data.models().has_model("case"));

    const auto& overwritten_series =
        thermal_data.models().get_model("case").QS();
    REQUIRE(overwritten_series.num_timesteps() > 1);
    REQUIRE(overwritten_series.num_columns() ==
            static_cast<pycanha::Index>(node_numbers.size()));
}

TEST_CASE("read_tmd_transient imports user-defined named constants",
          "[tmm][esatan][thermaldata]") {
    const std::filesystem::path reference_tmd_path = get_reference_tmd_path();
    REQUIRE(std::filesystem::exists(reference_tmd_path));

    pycanha::ThermalData thermal_data;
    REQUIRE_NOTHROW(pycanha::read_tmd_transient(reference_tmd_path.string(),
                                                thermal_data, "transient"));

    const auto& constants =
        thermal_data.models().get_model("transient").constants();

    const pycanha::Index num_timesteps = constants.num_timesteps();
    REQUIRE(num_timesteps == 101);
    const pycanha::Index last = num_timesteps - 1;

    // $REAL constants.
    REQUIRE(constants.real_names() ==
            std::vector<std::string>{"TIMECT", "TIME_REAL_CONST_1",
                                     "TIME_REAL_CONST_2"});
    REQUIRE(constants.real_values().rows() == num_timesteps);
    REQUIRE(constants.real_values().cols() == 3);
    REQUIRE(constants.real_values()(0, 0) == Catch::Approx(0.0));
    REQUIRE(constants.real_values()(0, 1) == Catch::Approx(-1.0));
    REQUIRE(constants.real_values()(0, 2) == Catch::Approx(0.0));
    REQUIRE(constants.real_values()(last, 1) == Catch::Approx(10000.0));
    REQUIRE(constants.real_values()(last, 2) == Catch::Approx(10001.0));

    // $INTEGER constants (faithful int64 storage).
    REQUIRE(constants.int_names() ==
            std::vector<std::string>{"TIME_INT_CONST_1", "TIME_INT_CONST_2"});
    REQUIRE(constants.int_values().rows() == num_timesteps);
    REQUIRE(constants.int_values().cols() == 2);
    REQUIRE(constants.int_values()(0, 0) == std::int64_t{-1});
    REQUIRE(constants.int_values()(0, 1) == std::int64_t{0});
    REQUIRE(constants.int_values()(last, 0) == std::int64_t{10001});
    REQUIRE(constants.int_values()(last, 1) == std::int64_t{0});

    // $CHARACTER constants (contiguous fixed-width storage).
    REQUIRE(constants.char_names() ==
            std::vector<std::string>{"TIME_CHAR_CONST_1", "TIME_CHAR_CONST_2"});
    REQUIRE(constants.char_width() > 0);
    REQUIRE(trim_spaces(constants.char_value(0, 0)) == "XXX_XXX_XXX_XXX");
    REQUIRE(trim_spaces(constants.char_value(last, 0)) == "10000.000");
    REQUIRE(trim_spaces(constants.char_value(last, 1)) == "10001.000");

    // Constants share the node-series time axis.
    REQUIRE(constants.times().size() ==
            thermal_data.models().get_model("transient").T().num_timesteps());
}

TEST_CASE("read_tmd_transient imports named constants from a steady file",
          "[tmm][esatan][thermaldata]") {
    const std::filesystem::path steady_tmd_path = get_steady_tmd_path();
    REQUIRE(std::filesystem::exists(steady_tmd_path));

    pycanha::ThermalData thermal_data;
    REQUIRE_NOTHROW(pycanha::read_tmd_transient(steady_tmd_path.string(),
                                                thermal_data, "steady"));

    const auto& constants =
        thermal_data.models().get_model("steady").constants();

    // Steady-state is a single-timestep transient holding the converged
    // snapshot values at time 0.
    REQUIRE(constants.num_timesteps() == 1);
    REQUIRE(constants.real_names() ==
            std::vector<std::string>{"TIMECT", "TIME_REAL_CONST_1",
                                     "TIME_REAL_CONST_2"});
    REQUIRE(constants.real_values().rows() == 1);
    REQUIRE(constants.real_values()(0, 1) == Catch::Approx(0.0));
    REQUIRE(constants.real_values()(0, 2) == Catch::Approx(1.0));
    REQUIRE(constants.int_values()(0, 0) == std::int64_t{1});
    REQUIRE(constants.int_values()(0, 1) == std::int64_t{0});
    REQUIRE(trim_spaces(constants.char_value(0, 0)) == "0.000");
    REQUIRE(trim_spaces(constants.char_value(0, 1)) == "1.000");
}
