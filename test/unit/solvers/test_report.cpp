#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "pycanha-core/config.hpp"
#include "pycanha-core/solvers/crank_nicolson.hpp"
#include "pycanha-core/solvers/model_changed_error.hpp"
#include "pycanha-core/solvers/report.hpp"
#include "pycanha-core/solvers/settings.hpp"
#include "pycanha-core/solvers/steady_state.hpp"
#include "solver_test_models.hpp"

namespace {

using pycanha::solvers::Backend;
using pycanha::solvers::Origin;
using pycanha::solvers::Settings;
using pycanha::solvers::SolverReport;
using pycanha::solvers::SolveStatus;
using pycanha::solvers::steady::SteadyState;
using pycanha::solvers::transient::CrankNicolson;
namespace models = pycanha::test_models;

constexpr int plate_size = 4;
// The tolerance of the factor reuse report test [K].
constexpr double reuse_tolerance_check = 1e-6;
constexpr double time_step = 10.0;

// A radiative plate whose iterations can never meet the tolerance.
Settings unreachable_tolerance() {
    Settings settings;
    settings.tolerance = 1e-12;
    settings.max_iterations = 1;
    return settings;
}

// A transient of `steps` time steps of 10 s on a radiative plate.
std::unique_ptr<CrankNicolson> plate_transient(const models::Model& model,
                                               const Settings& settings,
                                               int steps) {
    auto solver = std::make_unique<CrankNicolson>(model);
    solver->set_settings(settings);
    solver->set_simulation_time(0.0, time_step * steps, time_step, 0.0);
    return solver;
}

bool has_text(const std::string& text, const std::string& part) {
    return text.contains(part);
}

}  // namespace

TEST_CASE("A converged steady state is reported", "[solver][report]") {
    const auto model = models::make_plate(plate_size, /*radiation=*/true);
    SteadyState solver(model);
    solver.initialize();
    solver.solve();
    const SolverReport& report = solver.report();

    REQUIRE((report.converged() && solver.converged()));
    REQUIRE(report.iterations == solver.iterations());
    REQUIRE(report.factorizations == solver.num_factorizations());
    REQUIRE((report.solves == report.iterations && report.steps == 0));
    REQUIRE((report.memory.total() > 0 && report.solve_time > 0.0));
}

TEST_CASE("A linear steady state solves twice with one factorization",
          "[solver][report]") {
    const auto model = models::make_plate(plate_size, /*radiation=*/false);
    SteadyState solver(model);
    solver.initialize();
    solver.solve();
    const SolverReport& report = solver.report();

    REQUIRE(report.factorizations == 1);
    REQUIRE(report.solves == 2);
    REQUIRE(report.reuse_rate() == 0.5);
}

TEST_CASE("A steady state that does not converge warns once",
          "[solver][report]") {
    const auto model = models::make_plate(plate_size, /*radiation=*/true);
    SteadyState solver(model);
    solver.set_settings(unreachable_tolerance());
    solver.initialize();
    solver.solve();
    const SolverReport& report = solver.report();

    REQUIRE(report.status == SolveStatus::NOT_CONVERGED);
    REQUIRE_FALSE(solver.converged());
    REQUIRE(
        (report.nonconverged_steps == 1 && report.nonconverged.size() == 1));
    REQUIRE(report.warnings.size() == 1);
    REQUIRE(has_text(report.warnings.front(), "did not converge"));
}

TEST_CASE("Non-converged time steps warn once and are summarised",
          "[solver][report]") {
    constexpr int steps = 10;
    const auto model = models::make_plate(plate_size, /*radiation=*/true);
    auto solver = plate_transient(model, unreachable_tolerance(), steps);
    solver->initialize();
    solver->solve();
    const SolverReport& report = solver->report();

    REQUIRE((report.steps == steps && report.nonconverged_steps == steps));
    REQUIRE(report.status == SolveStatus::NOT_CONVERGED);
    REQUIRE_FALSE(solver->converged());
    // The first non-converged step and the summary.
    REQUIRE(report.warnings.size() == 2);
    REQUIRE(has_text(report.warnings.back(), "10 of 10 time steps"));
}

TEST_CASE("The non-converged list is capped but every step is counted",
          "[solver][report]") {
    constexpr int steps = 150;
    const auto model = models::make_plate(plate_size, /*radiation=*/true);
    auto solver = plate_transient(model, unreachable_tolerance(), steps);
    solver->initialize();
    solver->solve();
    const SolverReport& report = solver->report();

    REQUIRE(report.nonconverged_steps == steps);
    REQUIRE(report.nonconverged.size() == SolverReport::max_listed_steps);
    REQUIRE(report.nonconverged.front().step == 1);
    REQUIRE(report.nonconverged.front().time == time_step);
    REQUIRE(report.nonconverged.front().node > 0);
}

TEST_CASE("A converged transient counts steps, iterations and solves",
          "[solver][report]") {
    constexpr int steps = 5;
    const auto model = models::make_plate(plate_size, /*radiation=*/true);
    auto solver = plate_transient(model, Settings{}, steps);
    solver->initialize();
    solver->solve();
    const SolverReport& report = solver->report();

    REQUIRE((report.converged() && solver->converged()));
    REQUIRE(report.steps == steps);
    REQUIRE(report.iterations >= steps);
    REQUIRE((report.solves == report.iterations &&
             report.factorizations == report.solves));
    REQUIRE(report.memory.output > 0);
}

TEST_CASE("A transient reusing factors counts its first factorization",
          "[solver][report]") {
    if (!pycanha::MKL_ENABLED) {
        SKIP("MKL not built");
    }
    constexpr int steps = 5;
    const auto model = models::make_plate(plate_size, /*radiation=*/true);
    Settings settings;
    settings.reuse_factors = true;
    auto solver = plate_transient(model, settings, steps);
    solver->initialize();
    solver->solve();
    const SolverReport& report = solver->report();

    REQUIRE(report.factorizations >= 1);
    REQUIRE(report.factorizations < report.solves);
    REQUIRE(report.reuse_rate() > 0.0);
}

TEST_CASE("initialize() clears the report and keeps its own warnings",
          "[solver][report]") {
    const auto model = models::make_plate(plate_size, /*radiation=*/true);
    SteadyState solver(model);
    Settings settings;
    settings.time_step = 1.0;  // not used by a steady state: one WARN
    solver.set_settings(settings);
    solver.initialize();
    solver.solve();
    REQUIRE(solver.report().status == SolveStatus::CONVERGED);

    solver.deinitialize();
    solver.initialize();
    REQUIRE(solver.report().status == SolveStatus::NOT_RUN);
    REQUIRE(solver.report().warnings.size() == 1);
    solver.solve();
    REQUIRE(solver.report().warnings.size() == 1);
}

TEST_CASE("A failed solve is reported as FAILED", "[solver][report]") {
    const auto model = models::make_floating_model(/*radiation=*/false);
    SteadyState solver(model);
    solver.initialize();
    solver.solve();
    REQUIRE(solver.report().status == SolveStatus::FAILED);
    REQUIRE_FALSE(solver.converged());
}

TEST_CASE("A solve stopped by a model change is reported as FAILED",
          "[solver][report]") {
    const auto model = models::make_plate(plate_size, /*radiation=*/false);
    SteadyState solver(model);
    solver.initialize();
    model->add_conductive_coupling(1, plate_size * plate_size, 1.0);
    REQUIRE_THROWS_AS(solver.solve(), pycanha::solvers::ModelChangedError);
    REQUIRE(solver.report().status == SolveStatus::FAILED);
}

TEST_CASE("The report prints and converts to a dict", "[solver][report]") {
    constexpr int steps = 3;
    const auto model = models::make_plate(plate_size, /*radiation=*/true);
    auto solver = plate_transient(model, unreachable_tolerance(), steps);
    solver->initialize();
    solver->solve();
    const SolverReport& report = solver->report();
    const std::string text = report.to_string();
    const auto dict = report.to_dict();

    REQUIRE(has_text(text, "NOT_CONVERGED (3 of 3 time steps)"));
    REQUIRE(has_text(text, "Non-converged time steps (3)"));
    REQUIRE(std::get<std::int64_t>(dict.at("nonconverged_steps")) == steps);
    REQUIRE(std::get<std::vector<std::int64_t>>(dict.at("nonconverged.step"))
                .size() == steps);
    REQUIRE(std::get<std::string>(dict.at("status")) == "NOT_CONVERGED");
}

TEST_CASE("The report estimates the error of factor reuse",
          "[solver][report]") {
    if (!pycanha::MKL_ENABLED) {
        SKIP("MKL not built");
    }
    const auto model = models::make_plate(plate_size, /*radiation=*/true);
    const auto solve_with = [&model](bool reuse) {
        models::reset_temperatures(model, 250.0);
        SteadyState solver(model);
        Settings settings;
        settings.reuse_factors = reuse;
        settings.tolerance = reuse_tolerance_check;
        solver.set_settings(settings);
        solver.initialize();
        solver.solve();
        return solver.report();
    };
    const SolverReport reuse = solve_with(true);
    // reuse_tolerance (1e-2) times a last change below the tolerance.
    REQUIRE(reuse.reuse_error > 0.0);
    REQUIRE(reuse.reuse_error <= 1e-2 * reuse_tolerance_check);
    REQUIRE(has_text(reuse.to_string(), "reuse error"));
    REQUIRE(solve_with(false).reuse_error == 0.0);
}

TEST_CASE("explain() gives every value with its origin and reason",
          "[solver][explain]") {
    const auto model = models::make_plate(plate_size, /*radiation=*/true);
    SteadyState solver(model);
    solver.set("tolerance", 1e-4);
    const std::string text = solver.explain();

    REQUIRE(has_text(text, "SteadyState settings"));
    REQUIRE(has_text(text, "1e-04"));
    REQUIRE(has_text(text,
                     "AUTO: radiative couplings make the matrix "
                     "unsymmetric"));
    REQUIRE(solver.origin("tolerance") == Origin::USER);
    REQUIRE(solver.origin("max_iterations") == Origin::DEFAULT);
}

TEST_CASE("explain() shows the PARDISO iparm entries of factor reuse",
          "[solver][explain]") {
    if (!pycanha::MKL_ENABLED) {
        SKIP("MKL not built");
    }
    const auto model = models::make_plate(plate_size, /*radiation=*/true);
    Settings settings;
    settings.reuse_factors = true;
    auto solver = plate_transient(model, settings, 1);
    const std::string text = solver->explain();

    REQUIRE(has_text(text, "reuse_factors needs the one-level factorization"));
    REQUIRE(has_text(text, "iparm[3]"));
    REQUIRE(has_text(text, "relative residual of 1e-2"));
    REQUIRE(has_text(text, "minimum degree ordering"));
}

TEST_CASE("explain() shows the iparm[3] decade of reuse_tolerance",
          "[solver][explain]") {
    if (!pycanha::MKL_ENABLED) {
        SKIP("MKL not built");
    }
    // 3e-3 stops at the decade below it: 1e-3, iparm[3] = 31.
    const auto model = models::make_plate(plate_size, /*radiation=*/true);
    Settings settings;
    settings.reuse_factors = true;
    settings.reuse_tolerance = 3e-3;
    auto solver = plate_transient(model, settings, 1);
    const std::string text = solver->explain();

    REQUIRE(has_text(text, "31"));
    REQUIRE(has_text(text, "relative residual of 1e-3"));
}

TEST_CASE("explain() lists contradictions and ignored settings",
          "[solver][explain]") {
    const auto model = models::make_plate(plate_size, /*radiation=*/true);
    SteadyState solver(model);
    Settings settings;
    settings.backend = Backend::EIGEN;
    settings.reuse_factors = true;
    settings.end_time = 1.0;
    solver.set_settings(settings);
    const std::string text = solver.explain();

    REQUIRE(has_text(text, "Error: reuse_factors = true needs backend"));
    REQUIRE(has_text(text, "Set but not applied to SteadyState"));
    REQUIRE(has_text(text, "end_time"));
}
