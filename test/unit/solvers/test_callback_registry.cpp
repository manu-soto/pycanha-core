#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pycanha-core/solvers/callback_registry.hpp"
#include "pycanha-core/solvers/crank_nicolson.hpp"
#include "pycanha-core/solvers/settings.hpp"
#include "pycanha-core/solvers/solver.hpp"
#include "pycanha-core/solvers/solver_registry.hpp"
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

void require_callback_model_refs(pycanha::CallbackContext& context,
                                 const pycanha::ThermalModel& tm) {
    REQUIRE(&context.tm() == &tm);
    REQUIRE(&context.tmm() == &tm.tmm());
}

void require_callback_time(const pycanha::CallbackContext& context) {
    REQUIRE(context.time() == Catch::Approx(4.5));
}

void require_const_callback_view(const pycanha::CallbackContext& context,
                                 const pycanha::ThermalModel& tm) {
    const auto& const_context = std::as_const(context);
    REQUIRE(&const_context.tm() == &tm);
    REQUIRE(&const_context.tmm() == &tm.tmm());
    REQUIRE(const_context.time() == Catch::Approx(4.5));
}

void require_solver_guard(pycanha::CallbackContext& context) {
    const auto& const_context = std::as_const(context);
    REQUIRE_THROWS_AS(context.solver(), std::runtime_error);
    REQUIRE_THROWS_AS(const_context.solver(), std::runtime_error);
}

void invoke_all_callbacks(pycanha::CallbackRegistry& callbacks) {
    callbacks.invoke_solver_loop();
    callbacks.invoke_time_change();
    callbacks.invoke_after_timestep();
}

}  // namespace

TEST_CASE("ThermalModel callbacks expose root model solver and time",
          "[solver][callbacks]") {
    pycanha::ThermalModel tm("callback_model");

    auto node_1 = pycanha::Node(1);
    auto node_2 = pycanha::Node(2);

    node_1.set_T(300.0);
    node_1.set_C(100.0);
    node_1.set_qi(25.0);
    node_2.set_T(250.0);
    node_2.set_type(pycanha::BOUNDARY_NODE);

    tm.tmm().add_node(node_1);
    tm.tmm().add_node(node_2);
    tm.tmm().add_conductive_coupling(1, 2, 5.0);

    pycanha::ThermalModel* seen_tm = nullptr;
    pycanha::ThermalMathematicalModel* seen_tmm = nullptr;
    std::string seen_solver_name;
    std::vector<double> seen_times;

    tm.callbacks().after_timestep = [&](pycanha::CallbackContext& context) {
        seen_tm = &context.tm();
        seen_tmm = &context.tmm();
        seen_solver_name = context.solver().name();
        seen_times.push_back(context.time());
    };

    auto& solver = tm.solvers().transient().crank_nicolson();
    solver.set_settings(iteration_settings(20, 1.0e-9));
    solver.set_simulation_time(0.0, 20.0, 1.0, 5.0);

    solver.initialize();
    solver.solve();
    solver.deinitialize();

    REQUIRE(seen_tm == &tm);
    REQUIRE(seen_tmm == &tm.tmm());
    REQUIRE(seen_solver_name == "CrankNicolson");
    REQUIRE_FALSE(seen_times.empty());
    REQUIRE(seen_times.back() > 0.0);
}

TEST_CASE("CallbackRegistry solver-loop invocation exposes context guards",
          "[solver][callbacks]") {
    pycanha::ThermalModel tm("direct_callback_model");

    REQUIRE(tm.tmm().python_callbacks_active);

    tm.tmm().time = 4.5;

    int solver_loop_calls = 0;

    tm.callbacks().solver_loop = [&](pycanha::CallbackContext& context) {
        ++solver_loop_calls;
        require_callback_model_refs(context, tm);
        require_callback_time(context);
        require_const_callback_view(context, tm);
        require_solver_guard(context);
    };

    tm.callbacks().invoke_solver_loop();

    REQUIRE(solver_loop_calls == 1);
}

TEST_CASE("CallbackRegistry time callbacks expose the current time",
          "[solver][callbacks]") {
    pycanha::ThermalModel tm("direct_time_callback_model");
    tm.tmm().time = 4.5;

    int time_change_calls = 0;
    int after_timestep_calls = 0;
    double seen_time_change = 0.0;
    double seen_after_timestep = 0.0;

    tm.callbacks().time_change = [&](const pycanha::CallbackContext& context) {
        ++time_change_calls;
        seen_time_change = context.time();
    };
    tm.callbacks().after_timestep =
        [&](const pycanha::CallbackContext& context) {
            ++after_timestep_calls;
            seen_after_timestep = context.time();
        };

    tm.callbacks().invoke_time_change();
    tm.callbacks().invoke_after_timestep();

    REQUIRE(time_change_calls == 1);
    REQUIRE(after_timestep_calls == 1);
    REQUIRE(seen_time_change == Catch::Approx(4.5));
    REQUIRE(seen_after_timestep == Catch::Approx(4.5));
}

TEST_CASE("CallbackRegistry active flag suppresses direct invocations",
          "[solver][callbacks]") {
    pycanha::ThermalModel tm("inactive_direct_callback_model");

    int solver_loop_calls = 0;
    int time_change_calls = 0;
    int after_timestep_calls = 0;

    tm.callbacks().solver_loop = [&](pycanha::CallbackContext&) {
        ++solver_loop_calls;
    };
    tm.callbacks().time_change = [&](const pycanha::CallbackContext&) {
        ++time_change_calls;
    };
    tm.callbacks().after_timestep = [&](const pycanha::CallbackContext&) {
        ++after_timestep_calls;
    };

    tm.callbacks().active = false;
    invoke_all_callbacks(tm.callbacks());

    REQUIRE(solver_loop_calls == 0);
    REQUIRE(time_change_calls == 0);
    REQUIRE(after_timestep_calls == 0);
}
