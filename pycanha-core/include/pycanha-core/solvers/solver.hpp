#pragma once

#include <Eigen/Sparse>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/solvers/report.hpp"
#include "pycanha-core/solvers/settings.hpp"
#include "pycanha-core/tmm/couplingmatrices.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"

namespace pycanha::solvers {

using SpMatRow = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using SpVec = Eigen::SparseVector<double>;
using VectorXd = Eigen::VectorXd;
using WrappVectorXd = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>;

/// Base of every solver type.
/**
 * A solver works on one model with one Settings. initialize() resolves the
 * settings (see resolve()), warns once about the set fields that do not
 * apply, and builds everything the solver needs; solve() runs it. The INIT
 * settings are frozen while the solver is initialised; the LIVE ones are read
 * again at every solve().
 *
 * solve() throws ModelChangedError when the model's nodes or coupling
 * patterns changed since initialize(), checked at the start and after every
 * callback. An iteration loop that stops at max_iterations logs one WARN
 * (and, in a transient, one summary WARN at the end); report() has the
 * details.
 */
class Solver {
  public:
    virtual ~Solver() = default;
    Solver(const Solver&) = delete;
    Solver& operator=(const Solver&) = delete;
    Solver(Solver&&) = delete;
    Solver& operator=(Solver&&) = delete;

    virtual void initialize() = 0;
    virtual void solve() = 0;
    virtual void deinitialize() = 0;

    [[nodiscard]] SolverType solver_type() const noexcept {
        return _solver_type;
    }
    /// "SteadyState", "CrankNicolson", "JacobianPropagation".
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;

    /// A copy, with solver_type set to this solver's type.
    [[nodiscard]] Settings settings() const { return _settings; }
    /// Copies `settings` and sets its solver_type to this solver's. Throws
    /// std::runtime_error when an INIT field changes while the solver is
    /// initialised.
    void set_settings(const Settings& settings);
    /// Changes one setting by name; see Settings::set and set_settings().
    void set(std::string_view name, const SettingValue& value);
    /// Unsets one setting by name, so that it takes its default.
    void unset(std::string_view name);
    /// The settings resolved for the current model, as initialize() would.
    [[nodiscard]] ResolvedSettings resolved_settings() const;

    [[nodiscard]] bool initialized() const noexcept { return _initialized; }
    /// The last solve() converged in every iteration loop.
    [[nodiscard]] bool converged() const noexcept { return _converged; }

    /// What the last solve() did; initialize() clears it.
    [[nodiscard]] const SolverReport& report() const noexcept {
        return _report;
    }

    /// Every setting that applies, with its effective value, where it comes
    /// from and why; the PARDISO iparm entries the settings give; the set
    /// fields that do not apply, the warnings and the contradictions (listed,
    /// not thrown).
    [[nodiscard]] std::string explain() const;

    /// DEFAULT, USER, or AUTO_CONFIGURE for a value auto_configure() set and
    /// nothing changed since; DEFAULT for a field that does not apply.
    /// Throws std::invalid_argument for an unknown name.
    [[nodiscard]] Origin origin(std::string_view name) const;

    /// The memory this solver needs for the current model and settings.
    /**
     * Without an initialized solver it builds the matrix pattern and runs the
     * analysis of the factorization (see SolverMemory::exact for which
     * figures are known and which predicted), and a following initialize() with
     * the same settings and model structure reuses that work. On an initialized
     * solver it gives the current figures. Throws std::invalid_argument for
     * contradictory settings and std::runtime_error when the analysis fails.
     */
    [[nodiscard]] SolverMemory estimate_memory();

    /// Chooses the storage, backend, factorization, reuse_factors, threads,
    /// pardiso.ordering, pardiso.level and eigen.ordering for the current
    /// model, clears pardiso.iparm, and returns the new settings.
    /**
     * The other settings are kept. The candidates, in order of preference:
     *
     * | Solver              | # | With MKL                     | Without MKL  |
     * |---------------------|---|------------------------------|--------------|
     * | SteadyState         | 1 | DENSE LAPACK, reuse (rule S) | DENSE EIGEN  |
     * | SteadyState         | 2 | PARDISO TWO_LEVEL ND         | EIGEN COLAMD |
     * | SteadyState         | 3 | PARDISO ONE_LEVEL MD         | EIGEN AMD    |
     * | CrankNicolson       | 1 | DENSE LAPACK, reuse (rule T) | DENSE EIGEN  |
     * | CrankNicolson       | 2 | PARDISO ONE_LEVEL MD, reuse  | EIGEN COLAMD |
     * | CrankNicolson       | 3 | PARDISO TWO_LEVEL ND         | EIGEN AMD    |
     * | JacobianPropagation | 1 | DENSE LAPACK (rule T)        | DENSE EIGEN  |
     * | JacobianPropagation | 2 | PARDISO TWO_LEVEL ND         | EIGEN COLAMD |
     * | JacobianPropagation | 3 | PARDISO ONE_LEVEL MD         | EIGEN AMD    |
     *
     * Candidates 2 and 3 are SPARSE; ND is NESTED_DISSECTION, MD MIN_DEGREE
     * and "reuse" reuse_factors = true. Candidate 1 exists only when its rule
     * holds, with n the diffusive nodes and a coupling a conductive or
     * radiative link between two of them:
     * - rule S: n <= 5000 and at least 100 couplings per node on average;
     * - rule T: n <= 2000 and at least 20 % of the n x n matrix filled.
     * Dense storage wins where nodes see hundreds of others: it needs no
     * analysis, and the sparse factors of such a matrix are nearly dense
     * anyway. Rule T is stricter because a sparse transient does its
     * analysis once for all its steps.
     *
     * Every candidate sets factorization = AUTO (Cholesky without radiative
     * couplings, except with sparse Eigen; LU otherwise) and, with MKL,
     * threads = 0. reuse_factors iterates on the previous factors (PARDISO
     * CGS, or CG for Cholesky; LAPACK FGMRES) and refactorizes only when that
     * fails. It needs MKL and never applies to JacobianPropagation, whose
     * sensitivity solve needs the factors of the current matrix. A steady
     * state without it still solves an unchanged matrix with the existing
     * factors. The PARDISO minimum degree analysis is done once, but can
     * take long on models where a few nodes are coupled to many.
     *
     * Without max_memory the first candidate is taken. With max_memory
     * [bytes], a candidate whose estimate_memory() exceeds it is skipped, and
     * when none fits the smallest is taken with a WARN. Throws
     * std::runtime_error on an initialized solver.
     */
    Settings auto_configure(
        std::optional<std::int64_t> max_memory = std::nullopt);
    /// max_memory as "16 GB" (see parse_memory_size()).
    Settings auto_configure(std::string_view max_memory);

  protected:
    using Clock = std::chrono::steady_clock;

    Solver(std::shared_ptr<ThermalMathematicalModel> tmm_shptr,
           SolverType type);

    class FormulaExecutionGuard {
      public:
        explicit FormulaExecutionGuard(Solver& solver);
        ~FormulaExecutionGuard();

        FormulaExecutionGuard(const FormulaExecutionGuard&) = delete;
        FormulaExecutionGuard& operator=(const FormulaExecutionGuard&) = delete;
        FormulaExecutionGuard(FormulaExecutionGuard&&) = delete;
        FormulaExecutionGuard& operator=(FormulaExecutionGuard&&) = delete;

      private:
        Solver& _solver;
    };

    std::shared_ptr<ThermalMathematicalModel> _tmm_shptr;
    ThermalMathematicalModel& tmm;
    ThermalNetwork& tnw;
    CouplingMatrices& ltcs;
    CouplingMatrices& rtcs;

    SpMatRow& KLdd;
    SpMatRow& KLdb;
    SpMatRow& KLbb;

    SpMatRow& KRdd;
    SpMatRow& KRdb;
    SpMatRow& KRbb;

    WrappVectorXd T;
    WrappVectorXd Td;
    WrappVectorXd Tb;

    WrappVectorXd Qd;
    WrappVectorXd Qb;

    WrappVectorXd C;
    WrappVectorXd Cd;
    WrappVectorXd Cb;

    const SpVec& QI_sp;
    const SpVec& QS_sp;
    const SpVec& QA_sp;
    const SpVec& QE_sp;
    const SpVec& QR_sp;

    VectorXd Q;

    Index N = 0;
    Index nd = 0;
    Index nb = 0;

    VectorXd dTd;
    double max_dT = 0.0;
    int solver_iter = 0;

    VectorXd Td_solver;

    /// The settings resolved by initialize(); its LIVE fields are refreshed
    /// at the start of every solve().
    ResolvedSettings _resolved;
    bool _initialized = false;
    bool _converged = false;

    /// Resolves the settings for the current model, warns about the set
    /// fields that do not apply (with log_warnings), sizes the shared
    /// vectors and records the model's structure versions. Throws
    /// std::invalid_argument for contradictory settings.
    void initialize_common(bool log_warnings = true);

    /// Builds what initialize() needs that depends only on the settings and
    /// the model's structure (sizes, patterns, the analysis of the
    /// factorization) and calls mark_prepared() when it succeeds.
    /// estimate_memory() runs it without the warnings.
    virtual void prepare(bool log_warnings) = 0;
    /// The memory of what prepare() and initialize() built.
    [[nodiscard]] virtual SolverMemory memory_now() const = 0;
    /// Bytes of the vectors every solver sizes in initialize_common().
    [[nodiscard]] std::int64_t common_bytes() const noexcept;
    void mark_prepared();
    /// The last prepare() was not followed by initialize() and still matches
    /// the settings and the model structure.
    [[nodiscard]] bool preparation_matches() const;

    /// Logs a WARN prefixed with the solver name and keeps it in the report.
    void warn(const std::string& message);

    void begin_initialize_report();
    void end_initialize_report(Clock::time_point start);
    /// Clears the counters of the report, keeping initialize()'s warnings;
    /// the status is FAILED until finish_solve_report().
    void begin_solve_report();
    void record_linear_solve(Clock::time_point start, bool factorized);
    /// The iteration loop of the steady state, or of time step `step`
    /// ending at `time` [s]. A loop that did not converge is listed (up to
    /// SolverReport::max_listed_steps) and the first one logs a WARN.
    void record_steady_iterations(int iterations, bool converged);
    void record_step_iterations(int step, double time, int iterations,
                                bool converged);
    /// Sets the status, the solve time and the memory; a transient with
    /// non-converged steps logs the summary WARN. converged() follows.
    void finish_solve_report(Clock::time_point start, bool failed = false);

    SolverReport _report;
    // True from a successful prepare() until initialize() completes or the
    // solver is released.
    bool _prepared = false;
    /// Refreshes the LIVE fields of the resolved settings; checks the model.
    void refresh_live_settings();

    void callback_solver_loop();
    void callback_transient_time_change();
    void callback_transient_after_timestep();

    // Throws ModelChangedError when nodes were added, removed, retyped or
    // reordered, or couplings inserted or removed, since initialize(): the
    // solver's sizes, patterns and index tables no longer match the model and
    // it must be initialised again. `callback` names the callback that has
    // just run, empty at the start of solve().
    void check_model_unchanged(std::string_view callback = {}) const;

    [[nodiscard]] bool temperature_convergence_check();

    /// The model has radiative couplings (between diffusive nodes or to
    /// boundary nodes).
    [[nodiscard]] bool has_radiation() const noexcept;

  private:
    void prepare_formulas_for_execution();
    void release_formulas_after_execution() noexcept;
    void record_nonconverged(int step, double time, int iterations);
    void record_reuse_error() noexcept;
    [[nodiscard]] SolverMemory estimate_memory_quietly();
    [[nodiscard]] std::string_view auto_reason(std::string_view name) const;

    // A value auto_configure() set, with the rule that chose it.
    struct AutoValue {
        std::string name;
        SettingValue value;
        std::string reason;
    };

    SolverType _solver_type;
    Settings _settings;
    Settings _prepared_settings;
    std::vector<AutoValue> _auto_values;
    std::vector<std::string> _initialize_warnings;
    // The model's structure versions when initialize() last ran.
    StructureVersions _initialized_structure;
};

}  // namespace pycanha::solvers
