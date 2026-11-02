#pragma once

#include <memory>
#include <string_view>

#include "pycanha-core/solvers/solver.hpp"

namespace pycanha::solvers::detail {
class LinearizedNetwork;
}  // namespace pycanha::solvers::detail

namespace pycanha::solvers::steady {

/// Steady state of the network.
/**
 * Newton linearisation of the radiation, assembled into a sparse matrix
 * whose pattern is built once in initialize(), or into a dense one, and solved
 * with a direct factorization (see Settings: storage, backend, factorization,
 * reuse_factors, pardiso.*, eigen.*). Each iteration refactorizes only when
 * the matrix values differ from those of the last factorization, so the second
 * iteration of a linear model, or a new solve of an unchanged model, costs
 * one solve with the existing factors. With fixed_couplings the couplings
 * are read once per solve() and the matrices are not compared. With
 * reuse_factors each iteration solves for the temperature change,
 * A dT = b - A T, so the residual of the iteration on old factors stays in a
 * change that shrinks as the solve converges.
 *
 * A singular matrix (a group of diffusive nodes without a path to a boundary
 * node) is reported as an ERROR and leaves the temperatures of that solve
 * untouched; a solve that does not converge within max_iterations logs a
 * WARN and keeps the temperatures of its last iteration.
 */
class SteadyState : public Solver {
  public:
    explicit SteadyState(std::shared_ptr<ThermalMathematicalModel> tmm_shptr);
    ~SteadyState() override;

    SteadyState(const SteadyState&) = delete;
    SteadyState& operator=(const SteadyState&) = delete;
    SteadyState(SteadyState&&) = delete;
    SteadyState& operator=(SteadyState&&) = delete;

    void initialize() override;
    void solve() override;
    void deinitialize() override;

    [[nodiscard]] std::string_view name() const noexcept override {
        return "SteadyState";
    }

    /// The last initialize() chose a Cholesky factorization.
    [[nodiscard]] bool uses_cholesky() const noexcept { return _symmetric; }
    /// Numerical factorizations done by the last solve().
    [[nodiscard]] int num_factorizations() const noexcept {
        return _report.factorizations;
    }
    /// Iterations done by the last solve().
    [[nodiscard]] int iterations() const noexcept { return _report.iterations; }

  protected:
    void prepare(bool log_warnings) override;
    [[nodiscard]] SolverMemory memory_now() const override;

  private:
    // A = -K and b = -Qd of the linearised system K T = Qd, so the diagonal
    // is positive, which Cholesky needs.
    std::unique_ptr<detail::LinearizedNetwork> _network;
    VectorXd _b;
    // A T, subtracted from b when reuse_factors solves for the change.
    VectorXd _a_times_t;

    VectorXd _t_cubed_domain;
    VectorXd _t_fourth_boundary;
    VectorXd _radiation_linear_term;

    bool _symmetric = false;
    bool _radiation = false;

    void assemble_rhs();
    [[nodiscard]] bool factorize_and_solve_step();
    void release() noexcept;
};

}  // namespace pycanha::solvers::steady
