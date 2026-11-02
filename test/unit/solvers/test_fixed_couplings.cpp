#include <Eigen/Core>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <memory>
#include <vector>

#include "pycanha-core/config.hpp"
#include "pycanha-core/solvers/crank_nicolson.hpp"
#include "pycanha-core/solvers/report.hpp"
#include "pycanha-core/solvers/settings.hpp"
#include "pycanha-core/solvers/steady_state.hpp"
#include "pycanha-core/thermaldata/data_model.hpp"
#include "pycanha-core/thermaldata/dense_time_series.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "solver_test_models.hpp"

namespace {

using pycanha::solvers::Backend;
using pycanha::solvers::Settings;
using pycanha::solvers::SolverMemory;
using pycanha::solvers::Storage;
using pycanha::solvers::steady::SteadyState;
using pycanha::solvers::transient::CrankNicolson;
namespace models = pycanha::test_models;

constexpr int plate_size = 6;
// The callbacks below change the model from this time on [s].
constexpr double change_time = 45.0;

struct Storing {
    Storage storage;
    Backend backend;
};

// Each storage with the library of this build.
Storing storing(Storage storage) {
    if (!pycanha::MKL_ENABLED) {
        return {.storage = storage, .backend = Backend::EIGEN};
    }
    return {.storage = storage,
            .backend =
                storage == Storage::DENSE ? Backend::LAPACK : Backend::PARDISO};
}

// Every storage and library of this build.
std::vector<Storing> every_storing() {
    std::vector<Storing> all = {
        {.storage = Storage::SPARSE, .backend = Backend::EIGEN},
        {.storage = Storage::DENSE, .backend = Backend::EIGEN}};
    if (pycanha::MKL_ENABLED) {
        all.push_back(
            {.storage = Storage::SPARSE, .backend = Backend::PARDISO});
        all.push_back({.storage = Storage::DENSE, .backend = Backend::LAPACK});
    }
    if (pycanha::solvers::accelerate_available()) {
        all.push_back(
            {.storage = Storage::SPARSE, .backend = Backend::ACCELERATE});
    }
    return all;
}

Settings settings_for(Storing storing, bool fixed) {
    Settings settings;
    settings.storage = storing.storage;
    settings.backend = storing.backend;
    settings.fixed_couplings = fixed;
    settings.max_iterations = 50;
    settings.tolerance = 1e-10;
    return settings;
}

void change_coupling(pycanha::ThermalMathematicalModel* model) {
    if (model->time > change_time) {
        model->conductive_couplings().set_coupling_value(1, 2, 5.0);
    }
}

void change_capacity(pycanha::ThermalMathematicalModel* model) {
    if (model->time > change_time) {
        model->nodes().set_C(1, 50.0);
    }
}

// The temperatures of every output of a 100 s transient on a new plate,
// with the time-change callback given (or none).
Eigen::MatrixXd run_transient(
    const Settings& settings,
    void (*time_change)(pycanha::ThermalMathematicalModel*)) {
    const auto model = models::make_plate(plate_size, /*radiation=*/false);
    if (time_change != nullptr) {
        model->c_extern_callback_transient_time_change = time_change;
        model->c_callbacks_active = true;
    }
    CrankNicolson solver(model);
    solver.set_settings(settings);
    solver.set_simulation_time(0.0, 100.0, 10.0, 10.0);
    solver.initialize();
    solver.solve();
    REQUIRE(solver.converged());
    return solver.output_model().T().values();
}

double max_difference(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
    return (a - b).cwiseAbs().maxCoeff();
}

}  // namespace

TEST_CASE("fixed_couplings is off unless set, also after auto_configure()",
          "[solver][fixed]") {
    const auto model = models::make_plate(plate_size, /*radiation=*/true);
    SteadyState solver(model);
    REQUIRE_FALSE(solver.resolved_settings().fixed_couplings);
    static_cast<void>(solver.auto_configure());
    REQUIRE_FALSE(solver.resolved_settings().fixed_couplings);

    Settings settings;
    settings.fixed_couplings = true;
    solver.set_settings(settings);
    static_cast<void>(solver.auto_configure());
    REQUIRE(solver.resolved_settings().fixed_couplings);
    REQUIRE(solver.explain().contains("fixed_couplings"));
}

TEST_CASE("A conduction transient with fixed_couplings factorizes once",
          "[solver][fixed][transient]") {
    const auto made = GENERATE(from_range(every_storing()));
    INFO(pycanha::solvers::to_string(made.storage)
         << " " << pycanha::solvers::to_string(made.backend));
    const auto model = models::make_plate(plate_size, /*radiation=*/false);
    CrankNicolson solver(model);
    solver.set_settings(settings_for(made, /*fixed=*/true));
    solver.set_simulation_time(0.0, 100.0, 10.0, 20.0);
    solver.initialize();
    solver.solve();
    REQUIRE(solver.converged());
    REQUIRE(solver.report().factorizations == 1);
    // Every iteration of every step solved with those factors.
    REQUIRE(solver.report().solves == solver.report().iterations);
}

// What the promise saves: the matrix is not read again during the solve. A
// broken promise gives wrong results; this only shows that the solver trusts
// it.
TEST_CASE("fixed_couplings does not read the couplings again during a solve",
          "[solver][fixed][transient]") {
    const auto made = storing(GENERATE(Storage::SPARSE, Storage::DENSE));
    INFO(pycanha::solvers::to_string(made.storage));
    const Eigen::MatrixXd unchanged =
        run_transient(settings_for(made, /*fixed=*/false), nullptr);
    // Without the promise the change is followed...
    REQUIRE(max_difference(run_transient(settings_for(made, /*fixed=*/false),
                                         &change_coupling),
                           unchanged) > 1e-3);
    // ...with it the matrix keeps the couplings of the start.
    REQUIRE(max_difference(run_transient(settings_for(made, /*fixed=*/true),
                                         &change_coupling),
                           unchanged) < 1e-10);
}

TEST_CASE("Without fixed_couplings a callback may change a capacity",
          "[solver][fixed][transient]") {
    const auto made = storing(GENERATE(Storage::SPARSE, Storage::DENSE));
    INFO(pycanha::solvers::to_string(made.storage));
    const Eigen::MatrixXd unchanged =
        run_transient(settings_for(made, /*fixed=*/false), nullptr);
    REQUIRE(max_difference(run_transient(settings_for(made, /*fixed=*/false),
                                         &change_capacity),
                           unchanged) > 1e-3);
}

TEST_CASE("A fixed dense run keeps the estimated memory",
          "[solver][fixed][memory]") {
    const bool radiation = GENERATE(false, true);
    const auto model = models::make_plate(plate_size, radiation);
    SteadyState solver(model);
    solver.set_settings(settings_for(storing(Storage::DENSE), /*fixed=*/true));
    const SolverMemory estimate = solver.estimate_memory();
    solver.initialize();
    solver.solve();
    REQUIRE(solver.converged());
    REQUIRE(solver.report().memory.total() == estimate.total());
}
