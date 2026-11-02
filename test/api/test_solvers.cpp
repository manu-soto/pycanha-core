#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "pycanha-core/parameters/formulas.hpp"
#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/solvers/crank_nicolson.hpp"
#include "pycanha-core/solvers/jacobian_propagation.hpp"
#include "pycanha-core/solvers/settings.hpp"
#include "pycanha-core/solvers/solver_registry.hpp"
#include "pycanha-core/solvers/steady_state.hpp"
#include "pycanha-core/thermaldata/data_model.hpp"
#include "pycanha-core/thermaldata/thermaldata.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/tmm/thermalmodel.hpp"

namespace {

// The iteration limit and the convergence tolerance [K] of a solver.
pycanha::solvers::Settings iteration_settings(int max_iterations,
                                              double tolerance) {
    pycanha::solvers::Settings settings;
    settings.max_iterations = max_iterations;
    settings.tolerance = tolerance;
    return settings;
}

void populate_parameterized_steady_model(pycanha::ThermalModel& tm) {
    pycanha::Node plate(1);
    plate.set_T(300.0);
    plate.set_C(100.0);

    pycanha::Node sink(2);
    sink.set_type(pycanha::BOUNDARY_NODE);
    sink.set_T(250.0);

    tm.tmm().add_node(plate);
    tm.tmm().add_node(sink);
    tm.tmm().add_conductive_coupling(1, 2, 0.0);

    tm.parameters().add_parameter("heater_power", 50.0);
    tm.parameters().add_parameter("conductance", 5.0);
    static_cast<void>(
        tm.formulas().add_parameter_formula("QI1", "heater_power"));
    static_cast<void>(
        tm.formulas().add_parameter_formula("GL(1,2)", "conductance"));
    tm.formulas().apply_formulas();
}

void populate_transient_model(pycanha::ThermalModel& tm) {
    pycanha::Node plate(1);
    plate.set_T(300.0);
    plate.set_C(10.0);
    plate.set_qi(25.0);

    pycanha::Node sink(2);
    sink.set_type(pycanha::BOUNDARY_NODE);
    sink.set_T(250.0);

    tm.tmm().add_node(plate);
    tm.tmm().add_node(sink);
    tm.tmm().add_conductive_coupling(1, 2, 5.0);
}

void populate_jacobian_model(pycanha::ThermalModel& tm) {
    pycanha::Node diffusive_node(1);
    diffusive_node.set_T(0.0);
    diffusive_node.set_C(1.0);
    diffusive_node.set_qi(1.0);

    pycanha::Node boundary_node(2);
    boundary_node.set_type(pycanha::BOUNDARY_NODE);
    boundary_node.set_T(1.0);

    tm.tmm().add_node(diffusive_node);
    tm.tmm().add_node(boundary_node);
    tm.tmm().add_conductive_coupling(1, 2, 1.0);

    tm.parameters().add_parameter("k", 1.0);
    tm.parameters().add_parameter("C", 1.0);
    static_cast<void>(tm.formulas().add_parameter_formula("GL(1,2)", "k"));
    static_cast<void>(tm.formulas().add_parameter_formula("C1", "C"));
    tm.formulas().apply_formulas();
}

}  // namespace

TEST_CASE("SolverRegistry keeps a stable, model-owned solver surface",
          "[api][solvers]") {
    pycanha::ThermalModel tm("solvers_demo");
    populate_parameterized_steady_model(tm);

    auto& registry = tm.solvers();
    REQUIRE(&registry == &tm.tmm().solvers());
    REQUIRE(&registry.steady().steady_state() ==
            &tm.solvers().steady().steady_state());
    REQUIRE(&registry.transient().crank_nicolson() ==
            &tm.solvers().transient().crank_nicolson());
    REQUIRE(&registry.sensitivity().jacobian_propagation() ==
            &tm.solvers().sensitivity().jacobian_propagation());

    // Solvers are configured first, then initialized explicitly. That keeps
    // the lifecycle visible to readers and mirrors how the production API is
    // meant to be used from bindings.
    auto& steady = registry.steady().steady_state();
    steady.set_settings(iteration_settings(50, 1.0e-9));
    steady.initialize();
    steady.solve();

    REQUIRE(steady.initialized());
    REQUIRE(steady.converged());
    REQUIRE(tm.tmm().nodes().get_T(1) == Catch::Approx(260.0));

    steady.deinitialize();
    steady.initialize();
    steady.solve();
    REQUIRE(tm.tmm().nodes().get_T(1) == Catch::Approx(260.0));
    steady.deinitialize();
}

TEST_CASE("SteadyState follows parameter changes between solves",
          "[api][solvers]") {
    pycanha::ThermalModel tm("steady_sweep");
    populate_parameterized_steady_model(tm);

    auto& steady = tm.solvers().steady().steady_state();
    steady.initialize();
    steady.solve();
    REQUIRE(steady.converged());
    REQUIRE(tm.tmm().nodes().get_T(1) == Catch::Approx(260.0));

    // Same values: the factors of the previous solve are reused.
    steady.solve();
    REQUIRE(steady.num_factorizations() == 0);
    REQUIRE(tm.tmm().nodes().get_T(1) == Catch::Approx(260.0));

    tm.parameters().set_parameter("conductance", 10.0);
    tm.formulas().apply_formulas();
    steady.solve();
    REQUIRE(steady.converged());
    REQUIRE(tm.tmm().nodes().get_T(1) == Catch::Approx(255.0));
    steady.deinitialize();
}

TEST_CASE("Transient solvers write reusable output models", "[api][solvers]") {
    SECTION(
        "CrankNicolson stores its output in ThermalData and exposes a "
        "shortcut") {
        pycanha::ThermalModel tm("transient_demo");
        populate_transient_model(tm);

        auto& transient = tm.solvers().transient().crank_nicolson();
        transient.set_settings(iteration_settings(20, 1.0e-9));
        transient.set_simulation_time(0.0, 2.0, 0.1, 0.5);
        transient.initialize();
        transient.solve();
        transient.deinitialize();

        REQUIRE(tm.thermal_data().models().size() == 1);
        const auto model_names = tm.thermal_data().models().model_names();
        REQUIRE(model_names.size() == 1);
        REQUIRE(&transient.output_model() ==
                &tm.thermal_data().models().get_model(model_names.front()));
        REQUIRE(transient.output_model().T().num_timesteps() > 1);
        REQUIRE(transient.output_model().T().num_columns() == 2);
    }

    SECTION("JacobianPropagation honors the registered derivative parameters") {
        pycanha::ThermalModel tm("jacobian_demo");
        populate_jacobian_model(tm);
        tm.formulas().parameters_with_derivatives().add_parameter("C");
        tm.formulas().parameters_with_derivatives().add_parameter("k");

        auto& jacobian_solver =
            tm.solvers().sensitivity().jacobian_propagation();
        jacobian_solver.set_settings(iteration_settings(50, 1.0e-9));
        jacobian_solver.set_simulation_time(0.0, 5.0, 0.01, 0.1);
        jacobian_solver.initialize();
        jacobian_solver.solve();
        jacobian_solver.deinitialize();

        REQUIRE(jacobian_solver.derivative_parameter_names().size() == 2);
        REQUIRE(jacobian_solver.derivative_parameter_names().at(0) == "C");
        REQUIRE(jacobian_solver.derivative_parameter_names().at(1) == "k");
        REQUIRE(jacobian_solver.output_model().jacobian().cols() == 2);
        REQUIRE(jacobian_solver.output_model().jacobian().rows() == 1);
    }
}
