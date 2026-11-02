#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "pycanha-core/config.hpp"
#include "pycanha-core/parameters/formulas.hpp"
#include "pycanha-core/solvers/settings.hpp"
#include "pycanha-core/solvers/steady_state.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/nodes.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "solver_test_models.hpp"

namespace {

using pycanha::solvers::Backend;
using pycanha::solvers::Factorization;
using pycanha::solvers::Settings;
using pycanha::solvers::Storage;
using pycanha::solvers::steady::SteadyState;
using pycanha::test_models::Model;
namespace models = pycanha::test_models;

constexpr double solve_tolerance = 1e-11;

// A parameter of the radiative test model, the formulas it enters and its
// value.
struct Parameter {
    std::string name;
    double value;
    std::vector<std::array<std::string, 2>> formulas;
};

// The small radiative model (four diffusive nodes, a boundary node at
// 3.15 K) with parameters in an internal conductive coupling, an internal
// and a boundary radiative coupling, a heat load, the boundary temperature,
// and one parameter in two entities with factors other than one.
std::vector<Parameter> radiative_parameters() {
    return {
        {.name = "g", .value = 0.1, .formulas = {{"GL(10,15)", "g"}}},
        {.name = "r", .value = 0.2, .formulas = {{"GR(15,25)", "r"}}},
        {.name = "rb", .value = 1.0, .formulas = {{"GR(10,99)", "rb"}}},
        {.name = "q", .value = 500.0, .formulas = {{"QI15", "q"}}},
        {.name = "tb", .value = 150.0, .formulas = {{"T99", "tb"}}},
        {.name = "s",
         .value = 1.0,
         .formulas = {{"GL(20,25)", "0.1 * s * s"}, {"QI20", "20 * s"}}},
    };
}

Model make_parametric_radiative_model() {
    auto model = models::make_small_radiative_model();
    for (const auto& parameter : radiative_parameters()) {
        model->parameters().add_parameter(parameter.name, parameter.value);
        for (const auto& [entity, expression] : parameter.formulas) {
            static_cast<void>(
                model->formulas().add_parameter_formula(entity, expression));
        }
        model->formulas().parameters_with_derivatives().add_parameter(
            parameter.name);
    }
    model->formulas().apply_formulas();
    return model;
}

void solve(SteadyState& solver, const Model& model, double start) {
    models::reset_temperatures(model, start);
    solver.solve();
    REQUIRE(solver.converged());
}

void configure(SteadyState& solver, Settings settings) {
    settings.max_iterations = 200;
    settings.tolerance = solve_tolerance;
    solver.set_settings(settings);
}

Eigen::VectorXd diffusive_temperatures(const Model& model) {
    return models::temperatures(model).head(
        model->nodes().get_num_diff_nodes());
}

// Central differences of the diffusive temperatures, the parameter moved by
// a relative step.
Eigen::MatrixXd central_differences(const Model& model) {
    const auto parameters = radiative_parameters();
    SteadyState solver(model);
    configure(solver, {});
    solver.initialize();
    const auto nd = model->nodes().get_num_diff_nodes();
    Eigen::MatrixXd differences(nd,
                                static_cast<Eigen::Index>(parameters.size()));
    for (std::size_t column = 0; column < parameters.size(); ++column) {
        const auto& parameter = parameters[column];
        const double step = 1e-4 * parameter.value;
        std::array<Eigen::VectorXd, 2> sides;
        for (std::size_t side = 0; side < 2; ++side) {
            model->parameters().set_parameter(
                parameter.name, parameter.value + ((side == 0) ? step : -step));
            model->formulas().apply_formulas();
            solve(solver, model, 250.0);
            sides.at(side) = diffusive_temperatures(model);
        }
        model->parameters().set_parameter(parameter.name, parameter.value);
        model->formulas().apply_formulas();
        differences.col(static_cast<Eigen::Index>(column)) =
            (sides[0] - sides[1]) / (2.0 * step);
    }
    return differences;
}

double max_relative_difference(const Eigen::MatrixXd& value,
                               const Eigen::MatrixXd& reference) {
    double worst = 0.0;
    for (Eigen::Index column = 0; column < reference.cols(); ++column) {
        const double scale =
            std::max(reference.col(column).cwiseAbs().maxCoeff(), 1e-300);
        worst = std::max(
            worst,
            (value.col(column) - reference.col(column)).cwiseAbs().maxCoeff() /
                scale);
    }
    return worst;
}

struct Configuration {
    std::string name;
    Settings settings;
};

// Every storage and backend of this build, with and without factor reuse
// where it applies.
std::vector<Configuration> configurations(bool radiation) {
    std::vector<Configuration> list;
    const auto add = [&list](std::string name, Storage storage, Backend backend,
                             std::optional<Factorization> factorization,
                             bool reuse) {
        Settings settings;
        settings.storage = storage;
        settings.backend = backend;
        settings.factorization = factorization;
        if (reuse) {
            settings.reuse_factors = true;
        }
        list.push_back({.name = std::move(name), .settings = settings});
    };
    add("sparse Eigen", Storage::SPARSE, Backend::EIGEN, std::nullopt, false);
    add("dense Eigen", Storage::DENSE, Backend::EIGEN, std::nullopt, false);
    if (!radiation) {
        add("sparse Eigen Cholesky", Storage::SPARSE, Backend::EIGEN,
            Factorization::CHOLESKY, false);
    }
    if (pycanha::MKL_ENABLED) {
        add("PARDISO", Storage::SPARSE, Backend::PARDISO, std::nullopt, false);
        add("PARDISO reuse", Storage::SPARSE, Backend::PARDISO, std::nullopt,
            true);
        add("LAPACK", Storage::DENSE, Backend::LAPACK, std::nullopt, false);
        add("LAPACK reuse", Storage::DENSE, Backend::LAPACK, std::nullopt,
            true);
    }
    if (pycanha::solvers::accelerate_available()) {
        add("Accelerate", Storage::SPARSE, Backend::ACCELERATE, std::nullopt,
            false);
    }
    return list;
}

}  // namespace

TEST_CASE("SteadyState jacobian matches central differences",
          "[solver][steady][jacobian]") {
    const auto model = make_parametric_radiative_model();
    const Eigen::MatrixXd reference = central_differences(model);

    SteadyState solver(model);
    configure(solver, {});
    solver.initialize();
    solve(solver, model, 250.0);
    const Eigen::MatrixXd jacobian = solver.jacobian();

    const auto parameters = radiative_parameters();
    REQUIRE(jacobian.rows() == 4);
    REQUIRE(jacobian.cols() == static_cast<Eigen::Index>(parameters.size()));
    REQUIRE(solver.derivative_parameter_names().size() == parameters.size());
    for (std::size_t column = 0; column < parameters.size(); ++column) {
        REQUIRE(solver.derivative_parameter_names()[column] ==
                parameters[column].name);
    }
    // Central differences with a 1e-4 relative step: truncation ~1e-8.
    CAPTURE(jacobian, reference);
    REQUIRE(max_relative_difference(jacobian, reference) < 1e-6);
}

TEST_CASE("SteadyState jacobian is the same in every configuration",
          "[solver][steady][jacobian]") {
    const bool radiation = GENERATE(false, true);
    const auto model = radiation ? make_parametric_radiative_model()
                                 : models::make_plate(6, /*radiation=*/false);
    if (!radiation) {
        model->parameters().add_parameter("k", 0.5);
        model->parameters().add_parameter("q", 2.0);
        model->parameters().add_parameter("te", 293.15);
        static_cast<void>(
            model->formulas().add_parameter_formula("GL(8,9)", "k"));
        static_cast<void>(model->formulas().add_parameter_formula("QI15", "q"));
        static_cast<void>(model->formulas().add_parameter_formula(
            "T" + std::to_string(models::boundary_node), "te"));
        for (const char* name : {"k", "q", "te"}) {
            model->formulas().parameters_with_derivatives().add_parameter(name);
        }
        model->formulas().apply_formulas();
    }

    SteadyState reference_solver(model);
    configure(reference_solver, {});
    reference_solver.initialize();
    solve(reference_solver, model, 250.0);
    const Eigen::MatrixXd reference = reference_solver.jacobian();

    for (const auto& configuration : configurations(radiation)) {
        for (const bool fixed : {false, true}) {
            INFO(configuration.name << (fixed ? ", fixed couplings" : "")
                                    << (radiation ? ", radiation" : ""));
            SteadyState solver(model);
            Settings settings = configuration.settings;
            settings.fixed_couplings = fixed;
            configure(solver, settings);
            solver.initialize();
            solve(solver, model, 250.0);
            REQUIRE(max_relative_difference(solver.jacobian(), reference) <
                    1e-9);
        }
    }
}

TEST_CASE("SteadyState jacobian of a conductive link is exact",
          "[solver][steady][jacobian]") {
    // One node, a load Q and a link k to a boundary at Tb: T = Tb + Q / k,
    // dT/dk = -Q / k^2, dT/dQ = 1 / k, dT/dTb = 1, and the capacity does not
    // enter.
    auto model = std::make_shared<pycanha::ThermalMathematicalModel>("link");
    pycanha::Node node(1);
    node.set_T(0.0);
    node.set_C(1.0);
    model->add_node(node);
    pycanha::Node sink(2);
    sink.set_type(pycanha::BOUNDARY_NODE);
    sink.set_T(1.0);
    model->add_node(sink);
    model->add_conductive_coupling(1, 2, 1.0);
    for (const auto& [name, value, entity] :
         std::array<std::tuple<std::string, double, std::string>, 4>{{
             {"k", 2.0, "GL(1,2)"},
             {"Q", 3.0, "QI1"},
             {"Tb", 5.0, "T2"},
             {"C", 7.0, "C1"},
         }}) {
        model->parameters().add_parameter(name, value);
        static_cast<void>(
            model->formulas().add_parameter_formula(entity, name));
        model->formulas().parameters_with_derivatives().add_parameter(name);
    }
    model->formulas().apply_formulas();

    SteadyState solver(model);
    configure(solver, {});
    solver.initialize();
    solver.solve();
    REQUIRE(solver.converged());
    REQUIRE(model->nodes().get_T(1) == Catch::Approx(5.0 + (3.0 / 2.0)));
    const Eigen::MatrixXd jacobian = solver.jacobian();
    REQUIRE(jacobian(0, 0) == Catch::Approx(-3.0 / 4.0).epsilon(1e-12));
    REQUIRE(jacobian(0, 1) == Catch::Approx(1.0 / 2.0).epsilon(1e-12));
    REQUIRE(jacobian(0, 2) == Catch::Approx(1.0).epsilon(1e-12));
    REQUIRE(jacobian(0, 3) == Catch::Approx(0.0).margin(1e-15));
}

TEST_CASE("SteadyState linear response is the jacobian of a load",
          "[solver][steady][jacobian]") {
    const auto model = make_parametric_radiative_model();
    SteadyState solver(model);
    configure(solver, {});
    solver.initialize();
    solve(solver, model, 250.0);
    const Eigen::MatrixXd jacobian = solver.jacobian();

    // QI15 = q: one watt more on node 15 is the column of q.
    const auto nd = model->nodes().get_num_diff_nodes();
    const auto row_15 = *model->nodes().get_idx_from_node_num(15);
    Eigen::MatrixXd heat = Eigen::MatrixXd::Zero(nd, 2);
    heat(row_15, 0) = 1.0;
    heat(row_15, 1) = -2.5;
    const Eigen::MatrixXd response = solver.linear_response(heat);
    REQUIRE(response.rows() == nd);
    REQUIRE(response.cols() == 2);
    REQUIRE((response.col(0) - jacobian.col(3)).cwiseAbs().maxCoeff() <
            1e-12 * jacobian.col(3).cwiseAbs().maxCoeff());
    REQUIRE((response.col(1) + (2.5 * response.col(0))).cwiseAbs().maxCoeff() <
            1e-12 * response.col(0).cwiseAbs().maxCoeff());
    REQUIRE_THROWS_AS(solver.linear_response(Eigen::MatrixXd::Zero(nd + 1, 1)),
                      std::invalid_argument);
}

TEST_CASE("SteadyState jacobian leaves the next solve as it was",
          "[solver][steady][jacobian]") {
    const auto model = make_parametric_radiative_model();
    SteadyState solver(model);
    configure(solver, {});
    solver.initialize();
    solve(solver, model, 250.0);
    const Eigen::VectorXd first = diffusive_temperatures(model);
    static_cast<void>(solver.jacobian());
    solve(solver, model, 250.0);
    REQUIRE(models::max_difference(diffusive_temperatures(model), first) <
            1e-9);
}

TEST_CASE("SteadyState jacobian needs a converged solve and a parameter",
          "[solver][steady][jacobian]") {
    const auto model = make_parametric_radiative_model();
    SteadyState solver(model);
    REQUIRE_THROWS_AS(solver.jacobian(), std::logic_error);

    Settings settings;
    settings.max_iterations = 1;
    settings.tolerance = solve_tolerance;
    solver.set_settings(settings);
    solver.initialize();
    models::reset_temperatures(model, 250.0);
    solver.solve();
    REQUIRE_FALSE(solver.converged());
    REQUIRE_THROWS_AS(solver.jacobian(), std::logic_error);
    REQUIRE_THROWS_AS(solver.linear_response(Eigen::MatrixXd::Zero(
                          model->nodes().get_num_diff_nodes(), 1)),
                      std::logic_error);

    solver.deinitialize();
    configure(solver, {});
    solver.initialize();
    solve(solver, model, 250.0);
    model->parameters().add_parameter("unused", 1.0);
    model->formulas().parameters_with_derivatives().add_parameter("unused");
    REQUIRE_THROWS_AS(solver.jacobian(), std::invalid_argument);
    for (const auto& parameter : radiative_parameters()) {
        static_cast<void>(
            model->formulas().parameters_with_derivatives().remove_parameter(
                parameter.name));
    }
    static_cast<void>(
        model->formulas().parameters_with_derivatives().remove_parameter(
            "unused"));
    REQUIRE_THROWS_AS(solver.jacobian(), std::invalid_argument);
}

TEST_CASE("SteadyState jacobian and linear response take zero columns",
          "[solver][steady][jacobian]") {
    // A parameter that does not enter a steady state (a capacity) has a zero
    // column, and a load case may be zero: with reuse_factors, PARDISO's
    // iteration on the factors fails on a zero right-hand side.
    const auto model = make_parametric_radiative_model();
    model->parameters().add_parameter("c", 2.0e5);
    static_cast<void>(model->formulas().add_parameter_formula("C10", "c"));
    model->formulas().parameters_with_derivatives().add_parameter("c");
    model->formulas().apply_formulas();
    const auto nd = model->nodes().get_num_diff_nodes();
    const auto row_15 = *model->nodes().get_idx_from_node_num(15);

    for (const auto& configuration : configurations(/*radiation=*/true)) {
        INFO(configuration.name);
        SteadyState solver(model);
        configure(solver, configuration.settings);
        solver.initialize();
        solve(solver, model, 250.0);
        const Eigen::MatrixXd jacobian = solver.jacobian();
        REQUIRE(jacobian.col(jacobian.cols() - 1).isZero(0.0));
        REQUIRE_FALSE(jacobian.col(3).isZero(0.0));
        Eigen::MatrixXd heat = Eigen::MatrixXd::Zero(nd, 2);
        heat(row_15, 1) = 1.0;
        const Eigen::MatrixXd response = solver.linear_response(heat);
        REQUIRE(response.col(0).isZero(0.0));
        REQUIRE((response.col(1) - jacobian.col(3)).cwiseAbs().maxCoeff() <
                1e-9 * jacobian.col(3).cwiseAbs().maxCoeff());
    }
}
