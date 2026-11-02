#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <string>
#include <vector>

#include "pycanha-core/parameters/formulas.hpp"
#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/solvers/crank_nicolson.hpp"
#include "pycanha-core/solvers/jacobian_propagation.hpp"
#include "pycanha-core/solvers/settings.hpp"
#include "pycanha-core/solvers/solver_registry.hpp"
#include "pycanha-core/thermaldata/lookup_table.hpp"
#include "pycanha-core/thermaldata/thermaldata.hpp"
#include "pycanha-core/tmm/node.hpp"
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

void populate_transient_demo(pycanha::ThermalModel& tm) {
    pycanha::Node sink(100);
    sink.set_type(pycanha::BOUNDARY_NODE);
    sink.set_T(250.0);

    pycanha::Node plate(1);
    plate.set_T(300.0);
    plate.set_C(20.0);
    plate.set_qi(25.0);

    // Insert the boundary node first so the output model can show that thermal
    // data follows internal node ordering, not insertion order.
    tm.tmm().add_node(sink);
    tm.tmm().add_node(plate);
    tm.tmm().add_conductive_coupling(1, 100, 5.0);
}

void solve_transient(pycanha::ThermalModel& tm) {
    auto& solver = tm.solvers().transient().crank_nicolson();
    solver.set_settings(iteration_settings(20, 1.0e-9));
    solver.set_simulation_time(0.0, 2.0, 0.1, 0.5);
    solver.initialize();
    solver.solve();
    solver.deinitialize();
}

void populate_jacobian_demo(pycanha::ThermalModel& tm,
                            const std::vector<std::string>& derivative_order) {
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

    for (const auto& parameter_name : derivative_order) {
        tm.formulas().parameters_with_derivatives().add_parameter(
            parameter_name);
    }
}

Eigen::MatrixXd solve_jacobian(pycanha::ThermalModel& tm) {
    auto& solver = tm.solvers().sensitivity().jacobian_propagation();
    solver.set_settings(iteration_settings(50, 1.0e-9));
    solver.set_simulation_time(0.0, 5.0, 0.01, 0.1);
    solver.initialize();
    solver.solve();
    solver.deinitialize();

    const auto last_row = solver.output_model().jacobian().num_timesteps() - 1;
    return solver.output_model().jacobian().at(last_row);
}

}  // namespace

TEST_CASE("ThermalData keeps transient models separate from user tables",
          "[api][thermaldata]") {
    pycanha::ThermalModel tm("thermaldata_demo");
    populate_transient_demo(tm);

    Eigen::VectorXd x(2);
    x << 0.0, 1.0;
    pycanha::LookupTableVec1D::MatrixType y(2, 1);
    y << 0.1, 0.2;

    tm.thermal_data().tables().add_table("heater_schedule",
                                         pycanha::LookupTableVec1D(x, y));
    REQUIRE(tm.thermal_data().tables().has_table("heater_schedule"));
    REQUIRE_FALSE(tm.thermal_data().models().has_model("crank_nicolson"));

    solve_transient(tm);

    auto& transient = tm.solvers().transient().crank_nicolson();
    REQUIRE(tm.thermal_data().models().size() == 1);
    const auto model_names = tm.thermal_data().models().model_names();
    REQUIRE(model_names.size() == 1);
    REQUIRE(&transient.output_model() ==
            &tm.thermal_data().models().get_model(model_names.front()));
    REQUIRE(tm.thermal_data().tables().has_table("heater_schedule"));
    REQUIRE(transient.output_model().T().num_timesteps() > 1);
    REQUIRE(transient.output_model().T().num_columns() == 2);
    REQUIRE(transient.output_model().node_numbers().size() == 2);
    REQUIRE(transient.output_model().node_numbers().at(0) == 1);
    REQUIRE(transient.output_model().node_numbers().at(1) == 100);
}

TEST_CASE("Jacobian columns follow the derivative parameter order",
          "[api][thermaldata]") {
    pycanha::ThermalModel first("jacobian_columns_first");
    populate_jacobian_demo(first, {"C", "k"});
    const auto& first_matrix = solve_jacobian(first);

    pycanha::ThermalModel second("jacobian_columns_second");
    populate_jacobian_demo(second, {"k", "C"});
    const auto& second_matrix = solve_jacobian(second);

    REQUIRE(first.solvers()
                .sensitivity()
                .jacobian_propagation()
                .derivative_parameter_names()
                .at(0) == "C");
    REQUIRE(first.solvers()
                .sensitivity()
                .jacobian_propagation()
                .derivative_parameter_names()
                .at(1) == "k");
    REQUIRE(second.solvers()
                .sensitivity()
                .jacobian_propagation()
                .derivative_parameter_names()
                .at(0) == "k");
    REQUIRE(second.solvers()
                .sensitivity()
                .jacobian_propagation()
                .derivative_parameter_names()
                .at(1) == "C");

    REQUIRE(first_matrix.rows() == 1);
    REQUIRE(first_matrix.cols() == 2);
    REQUIRE(first_matrix(0, 0) ==
            Catch::Approx(second_matrix(0, 1)).margin(5.0e-6));
    REQUIRE(first_matrix(0, 1) ==
            Catch::Approx(second_matrix(0, 0)).margin(5.0e-6));
}
