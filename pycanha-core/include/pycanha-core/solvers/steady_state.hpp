#pragma once

#include <Eigen/Core>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

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
 *
 * After a converged solve(), jacobian() and linear_response() answer what
 * the steady state does when a parameter or a load changes, to first order,
 * from one factorization of the network at the converged temperatures.
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

    /// Derivatives of the diffusive temperatures of the last solve() with
    /// respect to the parameters selected in
    /// Formulas::parameters_with_derivatives() [K per unit of the parameter]:
    /// one row per diffusive node, in the model's order, and one column per
    /// derivative_parameter_names() entry.
    /**
     * The network is linearized at the converged temperatures, which gives
     * the exact tangent of the energy balance, factorized once, and solved
     * for the change of the balance each parameter makes through the
     * ParameterFormula entities it enters: conductive and radiative
     * couplings, the heat loads (QI, QS, QA, QE, QR) and the temperatures
     * of boundary nodes. Capacities, and the temperatures of diffusive
     * nodes (a start value), do not enter a steady state. A coupling that a
     * callback sets from the temperatures is taken at its converged value:
     * its own temperature dependence is not in the tangent.
     *
     * Throws std::logic_error when the solver is not initialized or the last
     * solve() did not converge, std::invalid_argument when no derivative
     * parameter is selected or a selected one enters no ParameterFormula,
     * and std::runtime_error when the factorization fails.
     */
    [[nodiscard]] Eigen::MatrixXd jacobian();
    /// The parameters of the columns of the last jacobian().
    [[nodiscard]] const std::vector<std::string>& derivative_parameter_names()
        const noexcept {
        return _derivative_parameter_names;
    }
    /// First-order change of the diffusive temperatures [K] for heat [W]
    /// added to the diffusive nodes, at the temperatures of the last
    /// solve(): one row per diffusive node, in the model's order, and one
    /// column per load case. The same factorization as jacobian(), one solve
    /// per column. Throws as jacobian(), and std::invalid_argument when
    /// `heat` does not have one row per diffusive node.
    [[nodiscard]] Eigen::MatrixXd linear_response(const Eigen::MatrixXd& heat);

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
    // The network holds the factors of the tangent at the temperatures of
    // the last converged solve(); solve() and initialize() clear it.
    bool _tangent_factorized = false;
    std::vector<std::string> _derivative_parameter_names;

    void assemble_rhs();
    [[nodiscard]] bool factorize_and_solve_step();
    // Linearizes the network at the converged temperatures and factorizes
    // it, once per solve(). Throws as jacobian().
    void factorize_tangent();
    void release() noexcept;
};

}  // namespace pycanha::solvers::steady
