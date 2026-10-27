#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>

#include "pycanha-core/parameters/formulas.hpp"
#include "pycanha-core/solvers/crank_nicolson.hpp"
#include "pycanha-core/solvers/jacobian_propagation.hpp"
#include "pycanha-core/solvers/settings.hpp"
#include "pycanha-core/thermaldata/data_model.hpp"
#include "pycanha-core/thermaldata/dense_matrix_time_series.hpp"
#include "pycanha-core/thermaldata/dense_time_series.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"

// A node without capacity (an arithmetic node) between a diffusive node and a
// boundary: it must be in balance at the end of every step, as in ESATAN's
// SLCRNC, and not alternate around its balance as the trapezoidal rule makes
// it do.

namespace {

constexpr double diffusive_capacity = 100.0;
constexpr double diffusive_start = 300.0;
constexpr double arithmetic_start = 250.0;  // its balance is 266.67 K
constexpr double sink_temperature = 200.0;
constexpr double dissipation = 10.0;
constexpr double inner_conductance = 2.0;
constexpr double outer_conductance = 1.0;
constexpr double time_step = 10.0;
constexpr std::size_t num_steps = 3;
constexpr double end_time = static_cast<double>(num_steps) * time_step;

using StepTemperatures = std::array<std::array<double, 2>, num_steps>;

// Node 1 (diffusive, 10 W) - GL 2 - node 2 (C = 0) - GL k - node 3 (200 K).
// With a parameter, GL(2,3) is the formula "k".
std::shared_ptr<pycanha::ThermalMathematicalModel> make_model(
    double outer, bool with_parameter) {
    auto model =
        std::make_shared<pycanha::ThermalMathematicalModel>("arithmetic_node");
    pycanha::Node diffusive(1);
    diffusive.set_T(diffusive_start);
    diffusive.set_C(diffusive_capacity);
    diffusive.set_qi(dissipation);
    pycanha::Node arithmetic(2);
    arithmetic.set_T(arithmetic_start);
    arithmetic.set_C(0.0);
    pycanha::Node sink(3);
    sink.set_type(pycanha::BOUNDARY_NODE);
    sink.set_T(sink_temperature);
    model->add_node(diffusive);
    model->add_node(arithmetic);
    model->add_node(sink);
    model->add_conductive_coupling(1, 2, inner_conductance);
    model->add_conductive_coupling(2, 3, outer);
    if (with_parameter) {
        model->parameters().add_parameter("k", outer);
        static_cast<void>(
            model->formulas().add_parameter_formula("GL(2,3)", "k"));
        model->formulas().apply_formulas();
        model->formulas().parameters_with_derivatives().add_parameter("k");
    }
    return model;
}

pycanha::solvers::Settings tight_settings() {
    pycanha::solvers::Settings settings;
    settings.max_iterations = 50;
    settings.tolerance = 1.0e-10;
    return settings;
}

double step_end(std::size_t step) {
    return static_cast<double>(step + 1) * time_step;
}

// Temperatures of nodes 1 and 2 at the end of every step: Crank-Nicolson on
// node 1, node 2 in balance at the end of the step (the start of the first
// step keeps node 2 where it was set).
StepTemperatures reference() {
    StepTemperatures out{};
    double t1 = diffusive_start;
    double t2 = arithmetic_start;
    const double c_over = 2.0 * diffusive_capacity / time_step;
    // Node 2's balance: T2 = (2 T1 + 200) / 3, so node 1's net heat flow at the
    // end of the step is 10 - (2/3) (T1 - 200).
    const double gl_eff = inner_conductance * outer_conductance /
                          (inner_conductance + outer_conductance);
    for (auto& temperatures : out) {
        const double start_rate = dissipation - (inner_conductance * (t1 - t2));
        // c_over (t1n - t1) = 10 - gl_eff (t1n - 200) + start_rate
        const double t1n = ((c_over * t1) + dissipation +
                            (gl_eff * sink_temperature) + start_rate) /
                           (c_over + gl_eff);
        const double t2n = ((inner_conductance * t1n) +
                            (outer_conductance * sink_temperature)) /
                           (inner_conductance + outer_conductance);
        temperatures = {t1n, t2n};
        t1 = t1n;
        t2 = t2n;
    }
    return out;
}

Eigen::Index row_at(const pycanha::DenseTimeSeries& table, double time) {
    for (Eigen::Index row = 0; row < table.num_timesteps(); ++row) {
        if (std::abs(table.times()(row) - time) < 1.0e-9) {
            return row;
        }
    }
    throw std::runtime_error("output time not found");
}

Eigen::Index row_at(const pycanha::DenseMatrixTimeSeries& table, double time) {
    for (Eigen::Index row = 0; row < table.num_timesteps(); ++row) {
        if (std::abs(table.time_at(row) - time) < 1.0e-9) {
            return row;
        }
    }
    throw std::runtime_error("output time not found");
}

// Nodes 1 and 2 at the end of every step of a CrankNicolson run.
StepTemperatures crank_nicolson_run(double outer) {
    auto model = make_model(outer, /*with_parameter=*/false);
    pycanha::solvers::transient::CrankNicolson solver(model);
    solver.set_settings(tight_settings());
    solver.set_simulation_time(0.0, end_time, time_step, time_step);
    solver.initialize();
    solver.solve();
    const pycanha::DenseTimeSeries& table = solver.output_model().T();
    StepTemperatures out{};
    for (std::size_t step = 0; step < num_steps; ++step) {
        const Eigen::Index row = row_at(table, step_end(step));
        out.at(step) = {table.values()(row, 0), table.values()(row, 1)};
    }
    return out;
}

}  // namespace

TEST_CASE("CrankNicolson keeps a node without capacity in balance",
          "[solver][transient]") {
    const StepTemperatures computed = crank_nicolson_run(outer_conductance);
    const StepTemperatures expected = reference();
    for (std::size_t step = 0; step < num_steps; ++step) {
        const double t1 = computed.at(step).at(0);
        const double t2 = computed.at(step).at(1);
        REQUIRE(t1 == Catch::Approx(expected.at(step).at(0)).margin(1.0e-6));
        REQUIRE(t2 == Catch::Approx(expected.at(step).at(1)).margin(1.0e-6));
        // Node 2's net heat flow at the end of the step.
        const double net = (inner_conductance * (t1 - t2)) -
                           (outer_conductance * (t2 - sink_temperature));
        REQUIRE(std::abs(net) < 1.0e-6);
    }
}

TEST_CASE("JacobianPropagation follows the balance of a node without capacity",
          "[solver][transient][jacobian]") {
    // dT/dk against a central difference of two CrankNicolson runs.
    constexpr double h = 1.0e-4;
    const StepTemperatures plus = crank_nicolson_run(outer_conductance + h);
    const StepTemperatures minus = crank_nicolson_run(outer_conductance - h);

    auto model = make_model(outer_conductance, /*with_parameter=*/true);
    pycanha::solvers::sensitivity::JacobianPropagation solver(model);
    solver.set_settings(tight_settings());
    solver.set_simulation_time(0.0, end_time, time_step, time_step);
    solver.initialize();
    solver.solve();
    const pycanha::DenseMatrixTimeSeries& jacobian =
        solver.output_model().jacobian();
    for (std::size_t step = 0; step < num_steps; ++step) {
        const Eigen::Index row = row_at(jacobian, step_end(step));
        for (std::size_t node = 0; node < 2; ++node) {
            const double central =
                (plus.at(step).at(node) - minus.at(step).at(node)) / (2.0 * h);
            const double propagated =
                jacobian.at(row)(static_cast<Eigen::Index>(node), 0);
            REQUIRE(propagated ==
                    Catch::Approx(central).epsilon(1.0e-5).margin(1.0e-7));
        }
    }
}
