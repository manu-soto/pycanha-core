#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include "pycanha-core/config.hpp"
#include "pycanha-core/solvers/crank_nicolson.hpp"
#include "pycanha-core/solvers/settings.hpp"
#include "pycanha-core/thermaldata/data_model.hpp"
#include "pycanha-core/thermaldata/dense_time_series.hpp"
#include "pycanha-core/thermaldata/thermaldata.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "solver_test_models.hpp"

namespace {

constexpr double init_temp = 273.15;
constexpr int num_nodes = 5;
constexpr double tol_temp = 1e-2;
constexpr int num_time_steps = 10;
constexpr double tol_time = 1e-6;

// Expected times recorded in the solver output table.
constexpr std::array<double, num_time_steps + 1> times = {
    0.0000000,  10000.0000, 20000.0000, 30000.0000, 40000.0000, 50000.0000,
    60000.0000, 70000.0000, 80000.0000, 90000.0000, 100000.0};

// Transient expected temperatures (time steps x nodes)
constexpr std::array<std::array<double, num_nodes>, num_time_steps + 1>
    expected_temps = {{
        {273.14999, 273.14999, 273.14999, 273.14999, 3.14999},
        {259.03552, 283.85105, 258.98241, 262.06791, 3.14999},
        {247.56014, 291.67014, 247.37629, 253.45623, 3.14999},
        {237.98527, 297.25685, 237.62266, 246.62735, 3.14999},
        {229.83503, 301.16946, 229.26392, 241.11244, 3.14999},
        {222.78667, 303.85891, 221.98896, 236.58283, 3.14999},
        {216.61234, 305.67267, 215.57742, 232.80415, 3.14999},
        {211.14591, 306.86934, 209.86801, 229.60718, 3.14999},
        {206.26295, 307.63674, 204.73939, 226.86828, 3.14999},
        {201.86811, 308.10888, 200.09819, 224.49601, 3.14999},
        {197.88691, 308.38019, 195.87117, 222.42185, 3.14999},
    }};

constexpr std::array<int, num_nodes> node_ids = {10, 15, 20, 25, 99};

[[nodiscard]] bool has_expected_output_shape(
    const pycanha::DenseTimeSeries& output_table, bool print_diffs) {
    const auto output_rows =
        static_cast<std::size_t>(output_table.num_timesteps());
    const auto output_cols =
        static_cast<std::size_t>(output_table.num_columns());

    if (output_rows == times.size() && output_cols == num_nodes) {
        return true;
    }

    if (print_diffs) {
        std::cout << "Unexpected output table shape: " << output_rows << "x"
                  << output_cols << " (expected " << times.size() << "x"
                  << num_nodes << ")\n";
    }
    return false;
}

[[nodiscard]] bool resolve_node_column_indices(
    const pycanha::DataModel& output_model,
    std::array<Eigen::Index, num_nodes>& node_column_indices,
    bool print_diffs) {
    for (std::size_t i = 0; i < node_ids.size(); ++i) {
        const auto iterator =
            std::find(output_model.node_numbers().begin(),
                      output_model.node_numbers().end(), node_ids.at(i));
        if (iterator == output_model.node_numbers().end()) {
            if (print_diffs) {
                std::cout << "Node " << node_ids.at(i)
                          << " is missing from the model output.\n";
            }
            return false;
        }

        node_column_indices.at(i) = static_cast<Eigen::Index>(
            std::distance(output_model.node_numbers().begin(), iterator));
    }

    return true;
}

[[nodiscard]] bool compare_output_row(
    const pycanha::DenseTimeSeries& output_table, std::size_t time_idx,
    const std::array<Eigen::Index, num_nodes>& node_column_indices,
    bool print_diffs) {
    const auto row = static_cast<Eigen::Index>(time_idx);
    const double computed_time = output_table.times()(row);
    const double expected_time = times.at(time_idx);
    bool row_within_tol = true;

    if (std::fabs(computed_time - expected_time) > tol_time) {
        row_within_tol = false;
        if (print_diffs) {
            std::cout << "Time index " << time_idx
                      << ": Computed time = " << computed_time
                      << " s, Expected time = " << expected_time << " s\n";
        }
    }

    for (std::size_t node_idx = 0; node_idx < node_ids.size(); ++node_idx) {
        const auto column = node_column_indices.at(node_idx);
        const double computed_temp = output_table.values()(row, column);
        const double expected_temp = expected_temps.at(time_idx).at(node_idx);
        const double temp_diff = std::fabs(computed_temp - expected_temp);

        if (temp_diff <= tol_temp) {
            continue;
        }

        row_within_tol = false;
        if (!print_diffs) {
            continue;
        }

        std::cout << "t=" << expected_time << " s, Node "
                  << node_ids.at(node_idx) << ": Computed = " << computed_temp
                  << " K, Expected = " << expected_temp
                  << " K, Diff = " << temp_diff << " K\n";
    }

    return row_within_tol;
}

std::shared_ptr<pycanha::ThermalMathematicalModel> make_model() {
    auto model =
        std::make_shared<pycanha::ThermalMathematicalModel>("test_model");

    auto node_10 = pycanha::Node(10);
    auto node_15 = pycanha::Node(15);
    auto node_20 = pycanha::Node(20);
    auto node_25 = pycanha::Node(25);
    auto env_node = pycanha::Node(99);

    // Set initial temperatures
    node_10.set_T(init_temp);
    node_15.set_T(init_temp);
    node_20.set_T(init_temp);
    node_25.set_T(init_temp);
    env_node.set_T(3.15);

    // Set thermal capacities
    node_10.set_C(2.0e5);
    node_15.set_C(2.0e5);
    node_20.set_C(2.0e5);
    node_25.set_C(2.0e5);

    // Set dissipation
    node_15.set_qi(500.0);

    // Set node types
    env_node.set_type(pycanha::BOUNDARY_NODE);

    // Add nodes to the model
    model->add_node(node_10);
    model->add_node(node_15);
    model->add_node(node_20);
    model->add_node(node_25);
    model->add_node(env_node);

    // Add conductive couplings
    model->add_conductive_coupling(10, 15, 0.1);
    model->add_conductive_coupling(20, 25, 0.1);

    // Add radiative couplings
    model->add_radiative_coupling(10, 99, 1.0);
    model->add_radiative_coupling(15, 25, 0.2);
    model->add_radiative_coupling(15, 99, 0.8);
    model->add_radiative_coupling(20, 99, 1.0);
    model->add_radiative_coupling(25, 99, 0.8);

    return model;
}

bool compare_temps(const pycanha::ThermalMathematicalModel& model,
                   bool print_diffs = false) {
    const auto& thermal_data = model.thermal_data();
    if (!thermal_data.models().has_model("crank_nicolson")) {
        if (print_diffs) {
            std::cout << "Thermal data model 'crank_nicolson' not found.\n";
        }
        return false;
    }

    const auto& output_model =
        thermal_data.models().get_model("crank_nicolson");
    const auto& output_table = output_model.T();
    if (!has_expected_output_shape(output_table, print_diffs)) {
        return false;
    }

    std::array<Eigen::Index, num_nodes> node_column_indices{};
    if (!resolve_node_column_indices(output_model, node_column_indices,
                                     print_diffs)) {
        return false;
    }

    bool all_within_tol = true;
    for (std::size_t time_idx = 0; time_idx < times.size(); ++time_idx) {
        if (!compare_output_row(output_table, time_idx, node_column_indices,
                                print_diffs)) {
            all_within_tol = false;
        }
    }

    return all_within_tol;
}

using pycanha::solvers::AccelerateOrdering;
using pycanha::solvers::Backend;
using pycanha::solvers::PardisoLevel;
using pycanha::solvers::PardisoOrdering;
using pycanha::solvers::Settings;
using pycanha::solvers::Storage;
using pycanha::solvers::transient::CrankNicolson;
namespace models = pycanha::test_models;

// The reference run: 100 000 s in steps of 1000 s, an output every 10 000 s.
void configure(CrankNicolson& solver, Settings settings) {
    settings.max_iterations = 100;
    settings.tolerance = 1e-6;
    solver.set_settings(settings);
    solver.set_simulation_time(0.0, 100000.0, 1000.0, 10000.0);
}

void reset_model_temps(pycanha::ThermalMathematicalModel& model) {
    model.nodes().set_T(10, init_temp);
    model.nodes().set_T(15, init_temp);
    model.nodes().set_T(20, init_temp);
    model.nodes().set_T(25, init_temp);
}

// The temperatures after 20 steps of 5 s on a radiative plate.
Eigen::VectorXd plate_transient(Settings settings) {
    const auto model = models::make_plate(20, /*radiation=*/true);
    settings.max_iterations = 50;
    settings.tolerance = 1e-10;
    CrankNicolson solver(model);
    solver.set_settings(settings);
    solver.set_simulation_time(0.0, 100.0, 5.0, 100.0);
    solver.initialize();
    solver.solve();
    REQUIRE(solver.converged());
    return models::temperatures(model);
}

// The same transient reusing factors with a loose relative residual: its
// largest difference with a direct solve.
double loose_reuse_difference(double reuse_tolerance,
                              const Eigen::VectorXd& reference) {
    Settings settings;
    settings.reuse_factors = true;
    settings.reuse_tolerance = reuse_tolerance;
    return models::max_difference(plate_transient(settings), reference);
}

}  // namespace

TEST_CASE("CrankNicolson solves a simple model", "[solver][transient]") {
    auto model = make_model();

    CrankNicolson solver(model);
    configure(solver, {});
    solver.initialize();
    REQUIRE(solver.initialized());
    solver.solve();

    // In case of error, set print_diffs to true to see detailed comparison
    REQUIRE(compare_temps(*model, false));

    // Re-run to verify no errors (like mem-leaks) on multiple initializations
    solver.deinitialize();
    reset_model_temps(*model);
    configure(solver, {});
    solver.initialize();
    solver.solve();
    REQUIRE(compare_temps(*model, false));

    // Another solver instance on the same model.
    CrankNicolson solver2(model);
    reset_model_temps(*model);
    configure(solver2, {});
    solver2.initialize();
    solver2.solve();
    REQUIRE(compare_temps(*model, false));
}

TEST_CASE("CrankNicolson gives the same transient with every configuration",
          "[solver][transient]") {
    struct Option {
        Backend backend;
        std::optional<PardisoLevel> level;
        std::optional<PardisoOrdering> ordering;
        bool reuse;
        Storage storage = Storage::SPARSE;
        std::optional<AccelerateOrdering> accelerate_ordering = std::nullopt;
    };
    const auto option = GENERATE(
        Option{Backend::EIGEN, std::nullopt, std::nullopt, false},
        Option{Backend::PARDISO, PardisoLevel::TWO_LEVEL, std::nullopt, false},
        Option{Backend::PARDISO, PardisoLevel::ONE_LEVEL,
               PardisoOrdering::NESTED_DISSECTION, false},
        Option{Backend::PARDISO, PardisoLevel::ONE_LEVEL,
               PardisoOrdering::MIN_DEGREE, false},
        Option{Backend::PARDISO, std::nullopt,
               PardisoOrdering::NESTED_DISSECTION, true},
        Option{Backend::PARDISO, std::nullopt, PardisoOrdering::MIN_DEGREE,
               true},
        Option{Backend::EIGEN, std::nullopt, std::nullopt, false,
               Storage::DENSE},
        Option{Backend::LAPACK, std::nullopt, std::nullopt, false,
               Storage::DENSE},
        Option{Backend::LAPACK, std::nullopt, std::nullopt, true,
               Storage::DENSE},
        Option{Backend::ACCELERATE, std::nullopt, std::nullopt, false,
               Storage::SPARSE, AccelerateOrdering::METIS},
        Option{Backend::ACCELERATE, std::nullopt, std::nullopt, false,
               Storage::SPARSE, AccelerateOrdering::AMD});
    const bool needs_mkl =
        option.backend == Backend::PARDISO || option.backend == Backend::LAPACK;
    if (needs_mkl && !pycanha::MKL_ENABLED) {
        SKIP("MKL not built");
    }
    if (option.backend == Backend::ACCELERATE &&
        !pycanha::solvers::accelerate_available()) {
        SKIP("Accelerate not available");
    }
    const bool fixed = GENERATE(false, true);
    auto model = make_model();
    CrankNicolson solver(model);
    Settings settings;
    settings.fixed_couplings = fixed;
    settings.storage = option.storage;
    settings.backend = option.backend;
    settings.pardiso.level = option.level;
    settings.pardiso.ordering = option.ordering;
    settings.reuse_factors = option.reuse;
    settings.accelerate.ordering = option.accelerate_ordering;
    configure(solver, settings);
    solver.initialize();
    REQUIRE(solver.initialized());
    solver.solve();
    REQUIRE(compare_temps(*model, false));
}

TEST_CASE("CrankNicolson with Eigen AMD gives the same transient",
          "[solver][transient]") {
    auto model = make_model();
    CrankNicolson solver(model);
    Settings settings;
    settings.backend = Backend::EIGEN;
    settings.eigen.ordering = pycanha::solvers::EigenOrdering::AMD;
    configure(solver, settings);
    solver.initialize();
    solver.solve();
    REQUIRE(compare_temps(*model, false));
}

TEST_CASE(
    "CrankNicolson reusing factors keeps a direct solve's answer at a "
    "loose iterative step",
    "[solver][transient][reuse]") {
    if (!pycanha::MKL_ENABLED) {
        SKIP("MKL not built");
    }
    // A relative residual of 1e-1 (iparm[3] = 11) stays in each change,
    // which shrinks as the inner iterations converge.
    const Eigen::VectorXd reference = plate_transient(Settings{});
    REQUIRE(loose_reuse_difference(1e-1, reference) < 1e-7);
}

TEST_CASE("CrankNicolson refuses unavailable configurations",
          "[solver][transient]") {
    const auto model = make_model();
    CrankNicolson solver(model);
    const auto refused = [&solver]() {
        try {
            solver.initialize();
        } catch (const std::invalid_argument&) {
            return !solver.initialized();
        }
        return false;
    };

    Settings cholesky;
    cholesky.factorization = pycanha::solvers::Factorization::CHOLESKY;
    configure(solver, cholesky);
    REQUIRE(refused());

    solver.set_settings(Settings{});  // no time step
    REQUIRE(refused());

    // Refused on every platform: Accelerate cannot reuse factors.
    Settings accelerate_reuse;
    accelerate_reuse.backend = Backend::ACCELERATE;
    accelerate_reuse.reuse_factors = true;
    configure(solver, accelerate_reuse);
    REQUIRE(refused());

    if (pycanha::MKL_ENABLED) {
        Settings reuse;
        reuse.reuse_factors = true;
        reuse.pardiso.level = PardisoLevel::TWO_LEVEL;
        configure(solver, reuse);
        REQUIRE(refused());
    }
}

TEST_CASE("CrankNicolson validates the simulation time",
          "[solver][transient]") {
    CrankNicolson solver(make_model());
    REQUIRE_THROWS_AS(solver.set_simulation_time(2.0, 1.0, 1.0, 0.0),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(solver.set_simulation_time(0.0, 1.0, 0.0, 0.0),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(solver.set_simulation_time(0.0, 1.0, 1.0, -0.5),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(solver.solve(), std::invalid_argument);
}

TEST_CASE("CrankNicolson freezes time and output settings while initialized",
          "[solver][transient]") {
    auto model = make_model();
    CrankNicolson solver(model);
    configure(solver, {});
    solver.initialize();
    REQUIRE_THROWS_AS(solver.set_simulation_time(0.0, 50000.0, 1000.0, 0.0),
                      std::runtime_error);
    REQUIRE_THROWS_AS(
        solver.set("output_name",
                   pycanha::solvers::SettingValue{std::string("other")}),
        std::runtime_error);
    REQUIRE_NOTHROW(
        solver.set("tolerance", pycanha::solvers::SettingValue{1e-5}));
    solver.solve();
    REQUIRE(compare_temps(*model, false));
}
