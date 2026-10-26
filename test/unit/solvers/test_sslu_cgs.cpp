#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <memory>
#include <stdexcept>

#include "pycanha-core/solvers/linear_solver.hpp"
#include "pycanha-core/solvers/sslu.hpp"
#include "pycanha-core/solvers/sslu_cgs.hpp"
#include "pycanha-core/tmm/conductivecouplings.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "solver_test_models.hpp"

namespace {

using pycanha::IterativeSolverType;
using pycanha::test_models::Model;
namespace models = pycanha::test_models;

}  // namespace

TEST_CASE("SSLU_CGS defaults", "[solver][sslu_cgs]") {
    const auto model = models::make_small_radiative_model();
    const pycanha::SSLU_CGS solver(model);
    REQUIRE(solver.solver_type == IterativeSolverType::MIN_DEGREE);
    REQUIRE(solver.pardiso_iparm_3 == 61);
}

TEST_CASE("SSLU_CGS rejects a solve without an iterative step",
          "[solver][sslu_cgs]") {
    const auto model = models::make_small_radiative_model();
    pycanha::SSLU_CGS solver(model);
    solver.pardiso_iparm_3 = 0;
    REQUIRE_THROWS_AS(solver.initialize(), std::invalid_argument);
    REQUIRE_FALSE(solver.solver_initialized);
}

#if PYCANHA_USE_MKL

namespace {

Eigen::VectorXd solve_with_sslu(const Model& model, double abstol) {
    pycanha::SSLU solver(model);
    solver.max_iters = 100;
    solver.abstol_temp = abstol;
    solver.initialize();
    solver.solve();
    REQUIRE(solver.solver_converged);
    return models::temperatures(model);
}

}  // namespace

TEST_CASE("SSLU_CGS matches SSLU on a radiative plate", "[solver][sslu_cgs]") {
    constexpr double abstol = 1e-8;
    const auto model = models::make_plate(20, /*radiation=*/true);
    const auto reference = solve_with_sslu(model, abstol);

    // The iteration stops at a relative residual of 10^-L (iparm_3 = 10 L +
    // 1), and that residual stays in the converged temperatures.
    const auto type = GENERATE(IterativeSolverType::MIN_DEGREE,
                               IterativeSolverType::ONE_LEVEL);
    const int iparm_3 = GENERATE(61, 91);
    models::reset_temperatures(model, 250.0);
    pycanha::SSLU_CGS solver(model);
    solver.solver_type = type;
    solver.pardiso_iparm_3 = iparm_3;
    solver.max_iters = 100;
    solver.abstol_temp = abstol;
    solver.initialize();
    REQUIRE_FALSE(solver.uses_cholesky());
    solver.solve();
    REQUIRE(solver.solver_converged);
    REQUIRE(models::max_difference(models::temperatures(model), reference) <
            1e-6);
    // One factorisation, the other passes iterate on it.
    REQUIRE(solver.num_factorizations() == 1);
}

TEST_CASE("SSLU_CGS solved for the change keeps SSLU's answer at a loose "
          "iterative step",
          "[solver][sslu_cgs]") {
    // A relative residual of 1e-4 (iparm_3 = 41) on the temperatures leaves
    // errors far above abstol; on the change it shrinks with every pass.
    constexpr double abstol = 1e-9;
    const auto model = models::make_plate(20, /*radiation=*/true);
    const auto reference = solve_with_sslu(model, abstol);

    models::reset_temperatures(model, 250.0);
    pycanha::SSLU_CGS solver(model);
    solver.pardiso_iparm_3 = 41;
    solver.max_iters = 100;
    solver.abstol_temp = abstol;
    REQUIRE(solver.correction_form);
    solver.initialize();
    solver.solve();
    REQUIRE(solver.solver_converged);
    REQUIRE(models::max_difference(models::temperatures(model), reference) <
            1e-7);
}

TEST_CASE("SSLU gives the same answer solved for the change or not",
          "[solver][sslu_cgs]") {
    constexpr double abstol = 1e-10;
    const auto model = models::make_plate(20, /*radiation=*/true);
    const auto solve_sslu = [&](bool correction) {
        models::reset_temperatures(model, 250.0);
        pycanha::SSLU solver(model);
        solver.correction_form = correction;
        solver.max_iters = 100;
        solver.abstol_temp = abstol;
        solver.initialize();
        solver.solve();
        REQUIRE(solver.solver_converged);
        return models::temperatures(model);
    };
    const auto direct = solve_sslu(false);
    REQUIRE(models::max_difference(solve_sslu(true), direct) < 1e-9);
}

TEST_CASE("SSLU_CGS uses Cholesky and CG on a conduction model",
          "[solver][sslu_cgs]") {
    const auto model = models::make_plate(15, /*radiation=*/false);
    const auto reference = solve_with_sslu(model, 1e-6);
    models::reset_temperatures(model, 293.15);

    pycanha::SSLU_CGS solver(model);
    solver.initialize();
    REQUIRE(solver.uses_cholesky());
    solver.solve();
    REQUIRE(solver.solver_converged);
    REQUIRE(models::max_difference(models::temperatures(model), reference) <
            1e-9);
}

TEST_CASE("SSLU_CGS never solves with factors of an older matrix",
          "[solver][sslu_cgs]") {
    // Pass 2 sees a doubled coupling and is solved with CG on the pass-1
    // factors, which PARDISO keeps. Pass 3 sees the same values as pass 2
    // but must not take the solve-only shortcut: the factors still belong to
    // the pass-1 matrix.
    const auto model = models::make_plate(12, /*radiation=*/false);
    const int node_1 = 1;
    const int node_2 = 2;
    const double original =
        model->conductive_couplings().get_coupling_value(node_1, node_2);
    model->conductive_couplings().set_coupling_value(node_1, node_2,
                                                     2.0 * original);
    const auto reference = solve_with_sslu(model, 1e-9);
    model->conductive_couplings().set_coupling_value(node_1, node_2, original);
    models::reset_temperatures(model, 293.15);

    models::install_callback(model, models::double_coupling_once, node_1,
                             node_2);
    pycanha::SSLU_CGS solver(model);
    solver.max_iters = 3;
    solver.abstol_temp = 0.0;
    solver.initialize();
    solver.solve();
    model->c_callbacks_active = false;

    REQUIRE(solver.num_factorizations() == 1);
    REQUIRE(models::max_difference(models::temperatures(model), reference) <
            1e-4);
}

#else

TEST_CASE("SSLU_CGS refuses to initialize without MKL", "[solver][sslu_cgs]") {
    const auto model = models::make_small_radiative_model();
    pycanha::SSLU_CGS solver(model);
    REQUIRE_THROWS_AS(solver.initialize(), std::invalid_argument);
    REQUIRE_FALSE(solver.solver_initialized);
}

#endif
