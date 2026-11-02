#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "pycanha-core/solvers/settings.hpp"
#include "pycanha-core/solvers/solver.hpp"

namespace pycanha {
class DataModel;
}  // namespace pycanha

namespace pycanha::solvers::detail {
class LinearizedNetwork;
}  // namespace pycanha::solvers::detail

namespace pycanha::solvers::transient {

/// Crank-Nicolson transient with the radiation linearised at every inner
/// iteration, solved with a sparse or dense direct factorization.
/**
 * The time and output settings (start_time, end_time, time_step,
 * output_interval, output_attributes, output_name) are read by initialize(),
 * which also creates the output model in the thermal data; time_step is
 * required. The factorization is LU, or Cholesky for a model without
 * radiative couplings (see Settings: factorization). Every inner iteration
 * reads the couplings and the capacities again, so formulas and callbacks
 * may change them during the run; with fixed_couplings they are read once
 * per solve() and the matrix is factorized again only when the temperatures
 * change it. With reuse_factors each inner iteration solves for the
 * temperature change, so the residual of the iteration on old factors does
 * not stay in the temperatures. A step whose inner
 * iterations do not converge within max_iterations keeps its last iteration
 * and the run goes on: the first such step logs a WARN, the end of the run
 * one summary WARN, and report() lists them.
 */
class CrankNicolson : public Solver {
  public:
    explicit CrankNicolson(std::shared_ptr<ThermalMathematicalModel> tmm_shptr);
    ~CrankNicolson() override;

    CrankNicolson(const CrankNicolson&) = delete;
    CrankNicolson& operator=(const CrankNicolson&) = delete;
    CrankNicolson(CrankNicolson&&) = delete;
    CrankNicolson& operator=(CrankNicolson&&) = delete;

    void initialize() override;
    void solve() override;
    void deinitialize() override;

    [[nodiscard]] std::string_view name() const noexcept override {
        return "CrankNicolson";
    }

    /// Sets start_time, end_time, time_step and output_interval [s] at once.
    /// Throws std::invalid_argument for an end before the start, a time step
    /// that is not positive or a negative output interval.
    void set_simulation_time(double start_time, double end_time,
                             double time_step, double output_interval);

    /// The output model of the last initialize().
    [[nodiscard]] DataModel& output_model();
    [[nodiscard]] const DataModel& output_model() const;

  protected:
    CrankNicolson(std::shared_ptr<ThermalMathematicalModel> tmm_shptr,
                  SolverType type);

    void prepare(bool log_warnings) override;
    [[nodiscard]] SolverMemory memory_now() const override;
    // Whether the matrix is used after the solve of a step (its products).
    [[nodiscard]] virtual bool keeps_matrix_after_solve() const noexcept {
        return false;
    }

    // Time steps and outputs from the resolved settings.
    void compute_output_schedule();
    // Creates the output model in the thermal data.
    void create_output_model();
    // Bytes of the output model the settings ask for.
    [[nodiscard]] std::int64_t output_bytes() const;
    void save_output_data();
    void outputs();
    void outputs_first_last();
    // Throws std::invalid_argument when the solver is not initialised;
    // otherwise rewinds the time to start_time.
    void restart_solve();

    void build_capacities();
    void build_conductance_matrix();
    void build_heat_flux();
    void store_heat_flux_at_n0();
    void solve_step();
    void release() noexcept;

    // Two times closer than this are the same output time [s].
    static constexpr double eps_time = 1.0e-6;

    double time = 0.0;
    int time_iter = 0;
    int num_time_steps = 0;
    int num_outputs = 0;
    int wait_n_dtimes = 0;
    int idata_out = 0;

    // -K + 2C/dt and its factors.
    std::unique_ptr<detail::LinearizedNetwork> _network;
    bool _symmetric = false;
    VectorXd _rhs;

    VectorXd _heat_flux_n0;
    VectorXd _capacities;
    VectorXd _capacities_inverse;
    // 2C/dt
    VectorXd _capacity_diagonal;

    VectorXd _t3_domain;
    VectorXd _t4_domain;
    VectorXd _t4_boundary;

    VectorXd _radiation_linear_term;
    VectorXd _kt_q_n0;
    // (A + 2C/dt) T, subtracted from the right-hand side when reuse_factors
    // solves for the change.
    VectorXd _a_times_t;
    VectorXd _ones_domain;
    VectorXd _ones_boundary;
};

}  // namespace pycanha::solvers::transient
