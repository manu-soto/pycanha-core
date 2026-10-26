#pragma once

#include <array>
#include <memory>
#include <vector>

#include "pycanha-core/solvers/linear_solver.hpp"
#include "pycanha-core/solvers/solver.hpp"

namespace pycanha {

/// Base of the steady-state solvers.
/**
 * Holds what SSLU and SSLU_CGS share: the Newton linearisation of the
 * network, assembled in place into a matrix whose pattern is built once in
 * initialize(), and the solver loop. Each pass refactorises only when the
 * matrix values differ from those of the last factorisation, so the second
 * pass of a linear model, or a new solve of an unchanged model, costs one
 * solve with the existing factors.
 */
class SteadyStateSolver : public Solver {
    friend class SSLU;
    friend class SSLU_CGS;

  public:
    explicit SteadyStateSolver(
        std::shared_ptr<ThermalMathematicalModel> tmm_shptr);
    ~SteadyStateSolver() override;

    SteadyStateSolver(const SteadyStateSolver&) = delete;
    SteadyStateSolver& operator=(const SteadyStateSolver&) = delete;
    SteadyStateSolver(SteadyStateSolver&&) = delete;
    SteadyStateSolver& operator=(SteadyStateSolver&&) = delete;

    /// With the MKL and ACCELERATE engines, factorise models without
    /// radiative couplings with Cholesky instead of LU (faster, less memory).
    /// SSLU_CGS only uses MKL. With Eigen,
    /// Cholesky is chosen with DirectSolverType::LDLT instead.
    bool allow_cholesky = true;

    /// Solve each pass for the temperature change, A dT = b - A T, instead of
    /// the temperatures, A T = b. The same equations, so a direct solve gives
    /// the same answer to rounding. With the iterative step of SSLU_CGS its
    /// relative tolerance (pardiso_iparm_3) then bounds the error of the
    /// change, which shrinks as the passes converge, instead of the error of
    /// the temperatures: a looser tolerance, fewer CGS iterations a pass, and
    /// still the direct solver's answer.
    bool correction_form = true;

    /// The last initialize() chose a Cholesky-type factorisation.
    [[nodiscard]] bool uses_cholesky() const noexcept { return _symmetric; }
    /// Numerical factorisations done by the last solve().
    [[nodiscard]] int num_factorizations() const noexcept {
        return _num_factorizations;
    }

  protected:
    void restart_solve() override;

    [[nodiscard]] bool has_radiation() const noexcept;
    /// Builds the pattern and analyses it. options.symmetric_positive_definite
    /// selects the upper-triangle (Cholesky) form. Throws
    /// std::invalid_argument when the options are not available. Leaves
    /// solver_initialized false if the analysis fails.
    void initialize_linearised(const LinearSolverOptions& options);
    void solve_linearised();
    void release_linearised() noexcept;

  private:
    // A = -K and b = -Qd of the linearised system K T = Qd, so the diagonal
    // is positive, which Cholesky needs.
    SpMatRow _a_matrix;
    VectorXd _b;
    VectorXd _factorized_values;
    bool _has_factors = false;

    VectorXd _t_cubed_domain;
    VectorXd _t_fourth_boundary;
    VectorXd _radiation_linear_term;
    VectorXd _row_sums;
    VectorXd _ones_domain;
    VectorXd _ones_boundary;

    std::vector<int> _upper_kl_indices;
    std::vector<int> _lower_kl_indices;
    std::vector<int> _upper_kr_indices;
    std::vector<int> _lower_kr_indices;
    std::vector<int> _diagonal_indices;

    // Non-zeros of KLdd, KLdb, KRdd and KRdb seen by initialize().
    std::array<Index, 4> _initialized_non_zeros{};
    int _num_factorizations = 0;
    bool _symmetric = false;
    bool _radiation = false;

    std::unique_ptr<SparseLinearSolver> _linear_solver;

    [[nodiscard]] bool model_structure_unchanged() const noexcept;
    void build_pattern();
    void assemble_matrix();
    void assemble_rhs();
    [[nodiscard]] bool factorize_and_solve_step();
};

}  // namespace pycanha
