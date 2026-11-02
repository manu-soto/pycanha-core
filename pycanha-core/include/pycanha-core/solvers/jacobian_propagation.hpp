#pragma once

#include <Eigen/Core>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/parameters/entity.hpp"
#include "pycanha-core/solvers/crank_nicolson.hpp"

namespace pycanha::solvers::sensitivity {

/// Crank-Nicolson transient that also propagates the derivatives of the
/// diffusive temperatures with respect to the parameters selected in
/// Formulas::parameters_with_derivatives().
/**
 * The derivatives are stored as the JAC attribute of the output model, at the
 * same output times as the temperatures. The sensitivity solve uses the
 * factors of the last step's matrix, so reuse_factors does not apply.
 * initialize() throws std::invalid_argument when no derivative parameter is
 * selected.
 */
class JacobianPropagation : public transient::CrankNicolson {
  public:
    explicit JacobianPropagation(
        std::shared_ptr<ThermalMathematicalModel> tmm_shptr);
    ~JacobianPropagation() override = default;

    JacobianPropagation(const JacobianPropagation&) = delete;
    JacobianPropagation& operator=(const JacobianPropagation&) = delete;
    JacobianPropagation(JacobianPropagation&&) = delete;
    JacobianPropagation& operator=(JacobianPropagation&&) = delete;

    void initialize() override;
    void solve() override;

    [[nodiscard]] std::string_view name() const noexcept override {
        return "JacobianPropagation";
    }

    /// Parameters the parameter formulas depend on, in first-use order.
    [[nodiscard]] const std::vector<std::string>& parameter_names()
        const noexcept {
        return _parameter_names;
    }

    /// Parameters whose derivatives are propagated (the columns of JAC).
    [[nodiscard]] const std::vector<std::string>& derivative_parameter_names()
        const noexcept {
        return _derivative_parameter_names;
    }

  protected:
    void prepare(bool log_warnings) override;
    [[nodiscard]] SolverMemory memory_now() const override;
    // The sensitivity products read the matrix of the last solve.
    [[nodiscard]] bool keeps_matrix_after_solve() const noexcept override {
        return true;
    }

  private:
    using DenseJacobian = Eigen::MatrixXd;

    // Throws std::invalid_argument when no derivative parameter is selected.
    void collect_parameter_names();
    void fill_matrices(const Entity& entity, Index parameter_index,
                       double derivative_value);
    void build_mk();
    void build_mq();
    void build_mc();
    void solve_jacobian_step();
    void save_jacobian_data();

    std::vector<std::string> _parameter_names;
    std::vector<std::string> _derivative_parameter_names;

    std::vector<SpMatRow> _d_kl_dd_matrices;
    std::vector<SpMatRow> _d_kl_db_matrices;
    std::vector<SpMatRow> _d_kr_dd_matrices;
    std::vector<SpMatRow> _d_kr_db_matrices;

    DenseJacobian _d_capacity_matrix;
    DenseJacobian _d_heat_flux_matrix;
    DenseJacobian _m_k;
    DenseJacobian _m_q;
    DenseJacobian _m_c;
    DenseJacobian _mt;
    DenseJacobian _mb;
};

}  // namespace pycanha::solvers::sensitivity
