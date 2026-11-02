#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <stdexcept>
#include <utility>

#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/parameters/formulas.hpp"
#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/solvers/settings.hpp"
#include "pycanha-core/solvers/solver_registry.hpp"
#include "pycanha-core/solvers/steady_state.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/tmm/thermalmodel.hpp"

namespace {

constexpr double boundary_temperature = 250.0;
constexpr double heater_power = 50.0;
constexpr double contact_conductance = 5.0;
constexpr double expected_plate_temperature =
    boundary_temperature + (heater_power / contact_conductance);

void populate_minimal_model(pycanha::ThermalModel& tm) {
    pycanha::Node plate(1);
    plate.set_T(300.0);
    plate.set_C(100.0);

    pycanha::Node sink(2);
    sink.set_T(boundary_temperature);
    sink.set_type(pycanha::BOUNDARY_NODE);

    tm.tmm().add_node(plate);
    tm.tmm().add_node(sink);
    tm.tmm().add_conductive_coupling(1, 2, 0.0);

    tm.parameters().add_parameter("heater_power", heater_power);
    tm.parameters().add_parameter("contact_conductance", contact_conductance);

    static_cast<void>(tm.formulas().add_formula("QI1", "heater_power"));
    static_cast<void>(
        tm.formulas().add_formula("GL(1,2)", "contact_conductance"));
    tm.formulas().apply_formulas();
}

}  // namespace

TEST_CASE("ThermalModel exposes stable subsystem ownership",
          "[api][overview]") {
    SECTION(
        "The convenience constructor keeps one persistent instance per "
        "subsystem") {
        // The public story starts at ThermalModel. Readers should be able to
        // take stable references once and use them throughout a workflow.
        pycanha::ThermalModel tm("overview_root");

        auto& tmm = tm.tmm();
        auto& gmm = tm.gmm();
        auto& parameters = tm.parameters();
        auto& formulas = tm.formulas();
        auto& thermal_data = tm.thermal_data();
        auto& solvers = tm.solvers();
        auto& callbacks = tm.callbacks();

        REQUIRE(&tmm == &tm.tmm());
        REQUIRE(&gmm == &tm.gmm());
        REQUIRE(&parameters == &tm.parameters());
        REQUIRE(&formulas == &tm.formulas());
        REQUIRE(&thermal_data == &tm.thermal_data());
        REQUIRE(&solvers == &tm.solvers());
        REQUIRE(&callbacks == &tm.callbacks());

        // The ThermalMathematicalModel keeps the same parameter-facing
        // subsystems, so user code can hop between tm and tmm without ending up
        // on detached copies.
        REQUIRE(&parameters == &tmm.parameters());
        REQUIRE(&formulas == &tmm.formulas());
        REQUIRE(&thermal_data == &tmm.thermal_data());

        // SolverRegistry lazily owns persistent solver instances.
        REQUIRE(&solvers.steady().steady_state() ==
                &tm.solvers().steady().steady_state());

        const auto& const_tm = std::as_const(tm);
        REQUIRE(&const_tm.tmm() == &tmm);
        REQUIRE(&const_tm.gmm() == &gmm);
        REQUIRE(&const_tm.parameters() == &parameters);
        REQUIRE(&const_tm.formulas() == &formulas);
        REQUIRE(&const_tm.thermal_data() == &thermal_data);
        REQUIRE(&const_tm.solvers() == &solvers);
        REQUIRE(&const_tm.callbacks() == &callbacks);
    }

    SECTION(
        "The explicit constructor path accepts pre-built wrapper components") {
        // Bindings can pre-create the low-level model and then hand it to the
        // root owner without losing identity.
        auto tmm = std::make_shared<pycanha::ThermalMathematicalModel>(
            "explicit_model");
        auto gmm =
            std::make_shared<pycanha::gmm::GeometryModel>("explicit_geometry");

        pycanha::ThermalModel tm("wrapper_root", tmm, gmm);

        REQUIRE(&tm.tmm() == tmm.get());
        REQUIRE(&tm.gmm() == gmm.get());
        REQUIRE(tm.solvers().tmm_ptr().get() == tmm.get());
        REQUIRE(&tm.parameters() == &tm.tmm().parameters());
    }

    SECTION("The explicit constructor rejects missing core objects") {
        auto tmm = std::make_shared<pycanha::ThermalMathematicalModel>(
            "explicit_model");
        auto gmm =
            std::make_shared<pycanha::gmm::GeometryModel>("explicit_geometry");

        REQUIRE_THROWS_AS(pycanha::ThermalModel("missing_tmm", nullptr, gmm),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(pycanha::ThermalModel("missing_gmm", tmm, nullptr),
                          std::invalid_argument);
    }
}

TEST_CASE("A minimal ThermalModel solves steady state end to end",
          "[api][overview]") {
    // This is the smallest useful story for a new user: build a model entirely
    // in memory, bind symbolic names through parameters and formulas, solve,
    // then read the resulting temperature back from the model.
    pycanha::ThermalModel tm("overview_model");
    populate_minimal_model(tm);

    auto& solver = tm.solvers().steady().steady_state();
    pycanha::solvers::Settings settings;
    settings.max_iterations = 50;
    settings.tolerance = 1.0e-9;
    solver.set_settings(settings);

    solver.initialize();
    REQUIRE(solver.initialized());

    solver.solve();
    REQUIRE(solver.converged());

    REQUIRE(tm.tmm().nodes().get_T(2) == Catch::Approx(boundary_temperature));
    REQUIRE(tm.tmm().nodes().get_T(1) ==
            Catch::Approx(expected_plate_temperature).margin(1.0e-6));

    solver.deinitialize();
}
