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

    /// SSLU_CGS: factorise again at the next pass when the last pass needed
    /// more than this many CGS/CG iterations on the previous factors. 0 keeps
    /// PARDISO's own rule: refactorise only when the iteration fails.
    ///
    /// The iteration uses the last factorisation as its preconditioner, so
    /// every pass pays for how far the matrix has moved since then. Over a
    /// sequence of solves whose temperatures drift (a heater control loop
    /// calling solve() for each new power), the radiative 4 sigma T^3 terms
    /// drift with them, and with PARDISO's rule the first factors are kept to
    /// the end: on large radiative models every pass then iterates longer
    /// than one new factorisation takes.
    int cgs_refactor_iterations = 0;

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
    // The last iterative pass went past cgs_refactor_iterations.
    bool _refactor_next = false;

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
