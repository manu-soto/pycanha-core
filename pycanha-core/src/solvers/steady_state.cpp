#include "pycanha-core/solvers/steady_state.hpp"

#include <spdlog/spdlog.h>

#include <memory>
#include <utility>

#include "factorizer.hpp"
#include "linearized_network.hpp"
#include "memory.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/solvers/report.hpp"
#include "pycanha-core/solvers/settings.hpp"
#include "pycanha-core/solvers/solver.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/utils/logger.hpp"
#include "pycanha-core/utils/profiling.hpp"

namespace pycanha::solvers::steady {

SteadyState::SteadyState(std::shared_ptr<ThermalMathematicalModel> tmm_shptr)
    : Solver(std::move(tmm_shptr), SolverType::STEADY_STATE) {}

SteadyState::~SteadyState() = default;

void SteadyState::initialize() {
    const auto start = Clock::now();
    begin_initialize_report();
    if (preparation_matches()) {
        // estimate_memory() built the pattern and ran the analysis.
        initialize_common();
    } else {
        prepare(/*log_warnings=*/true);
        if (!_prepared) {
            return;
        }
    }
    _network->allocate();
    if (const auto hint = _network->hint(_resolved)) {
        warn(*hint);
    }
    SPDLOG_LOGGER_INFO(get_logger(), "{} initializing ({})", name(),
                       detail::describe(_resolved, _symmetric));
    _prepared = false;
    _initialized = true;
    end_initialize_report(start);
}

void SteadyState::prepare(bool log_warnings) {
    release();
    // Throws before anything is built when the settings contradict.
    initialize_common(log_warnings);
    _symmetric = _resolved.factorization == Factorization::CHOLESKY;
    _radiation = has_radiation();
    _network =
        detail::make_network(_resolved,
                             {.conductive = &KLdd,
                              .conductive_boundary = &KLdb,
                              .radiative = &KRdd,
                              .radiative_boundary = &KRdb},
                             {.symmetric = _symmetric,
                              .radiation = _radiation,
                              .skip_unchanged = true,
                              .keep_matrix = false,
                              .fixed_couplings = _resolved.fixed_couplings});

    _b = VectorXd::Zero(nd);
    _a_times_t = VectorXd::Zero(nd);
    _t_cubed_domain = VectorXd::Zero(nd);
    _t_fourth_boundary = VectorXd::Zero(nb);
    _radiation_linear_term = VectorXd::Zero(nd);

    if (!_network->analyze()) {
        SPDLOG_LOGGER_ERROR(get_logger(), "{}: analysis failed ({}).", name(),
                            _network->error_message());
        release();
        return;
    }
    mark_prepared();
}

SolverMemory SteadyState::memory_now() const {
    using detail::bytes_of;
    SolverMemory memory;
    if (_network) {
        memory.factors = _network->factor_bytes();
        memory.exact = _network->factor_bytes_exact();
        memory.matrix = _network->matrix_bytes();
    }
    memory.matrix += bytes_of(_b) + bytes_of(_a_times_t) +
                     bytes_of(_t_cubed_domain) + bytes_of(_t_fourth_boundary) +
                     bytes_of(_radiation_linear_term) + common_bytes();
    return memory;
}

void SteadyState::assemble_rhs() {
    PYCANHA_PROFILE_SCOPE("Steady state assemble rhs");
    Q.setZero();
    Q += QI_sp;
    Q += QS_sp;
    Q += QA_sp;
    Q += QE_sp;
    Q += QR_sp;

    _b = Q.head(nd);
    _b.noalias() += KLdb * Tb;
    if (_radiation) {
        _t_fourth_boundary = STF_BOLTZ * Tb.array().square().square();
        _b.noalias() += KRdb * _t_fourth_boundary;
        _b += _radiation_linear_term;
    }
}

bool SteadyState::factorize_and_solve_step() {
    PYCANHA_PROFILE_SCOPE("Steady state factorize and solve");
    const auto start = Clock::now();
    const detail::SolveOutcome outcome = _network->solve(_b, Td_solver);
    SPDLOG_LOGGER_DEBUG(get_logger(), "{} iteration {}: factors {}.", name(),
                        solver_iter + 1,
                        outcome.factorized ? "computed" : "reused");

    if (!outcome.succeeded) {
        if (_network->zero_pivot()) {
            SPDLOG_LOGGER_ERROR(
                get_logger(),
                "{}: zero pivot in the {} factorization ({}). The matrix is "
                "singular: a group of diffusive nodes has no conductive or "
                "radiative path to a boundary node.",
                name(), _symmetric ? "Cholesky" : "LU",
                _network->error_message());
        } else {
            SPDLOG_LOGGER_ERROR(get_logger(), "{}: {}.", name(),
                                _network->error_message());
        }
        return false;
    }

    // PARDISO replaces a pivot below 1e-13 relative to the matrix by that
    // value. A matrix diagonally dominant by columns never needs it unless
    // it is singular, and the solution is then meaningless.
    if (outcome.factorized && _network->perturbed_pivots() > 0) {
        SPDLOG_LOGGER_ERROR(
            get_logger(),
            "{}: {} pivots perturbed in the LU factorization. The matrix is "
            "singular: a group of diffusive nodes has no conductive or "
            "radiative path to a boundary node.",
            name(), _network->perturbed_pivots());
        _network->discard_factors();
        return false;
    }
    record_linear_solve(start, outcome.factorized);
    return true;
}

void SteadyState::solve() {
    if (!_initialized) {
        SPDLOG_LOGGER_ERROR(
            get_logger(),
            "{} has not been initialized. Call initialize() before solve().",
            name());
        return;
    }
    const auto start = Clock::now();
    begin_solve_report();
    refresh_live_settings();
    _network->apply_live_settings(_resolved);
    _network->begin_solve();
    SPDLOG_LOGGER_INFO(get_logger(), "{} solving...", name());

    const FormulaExecutionGuard formula_execution(*this);

    _converged = false;
    int iterations_done = 0;
    bool converged = false;

    for (solver_iter = 0; solver_iter < _resolved.max_iterations;
         ++solver_iter) {
        PYCANHA_PROFILE_SCOPE("Steady state iteration");
        iterations_done = solver_iter + 1;

        if (_radiation) {
            _t_cubed_domain = (4.0 * STF_BOLTZ) * Td.array().cube();
        }
        {
            PYCANHA_PROFILE_SCOPE("Steady state linearize");
            // No diagonal of its own: the steady state solves -K T = -Qd.
            _network->linearize(_t_cubed_domain, Td, VectorXd{},
                                _radiation_linear_term);
        }
        assemble_rhs();
        if (_resolved.reuse_factors) {
            // A dT = b - A T: the same iteration, solved for the change, so
            // the residual of an iteration on old factors stays in a change
            // that shrinks as the solve converges. The product follows
            // linearize(), so the matrix is the current one. The iteration
            // starts from x: zero, closer to this change than the last one.
            _network->multiply(Td, _a_times_t);
            _b -= _a_times_t;
            Td_solver.setZero();
        }

        // A zero residual is solved by a zero change, and PARDISO's
        // iteration on old factors fails on a zero right-hand side.
        const bool settled = _resolved.reuse_factors && _b.isZero(0.0);
        if (!settled && !factorize_and_solve_step()) {
            finish_solve_report(start, /*failed=*/true);
            return;
        }

        if (_resolved.reuse_factors) {
            dTd = Td_solver;
            Td += dTd;
        } else {
            dTd = Td_solver - Td;
            Td = Td_solver;
        }
        max_dT = dTd.cwiseAbs().maxCoeff();

        this->callback_solver_loop();

        if (max_dT < _resolved.tolerance) {
            SPDLOG_LOGGER_INFO(get_logger(),
                               "{} converged. Num. iters: {}. Max. dT = {} K. "
                               "Factorizations: {}.",
                               name(), iterations_done, max_dT,
                               _report.factorizations);
            converged = true;
            break;
        }
    }

    record_steady_iterations(iterations_done, converged);
    finish_solve_report(start);
}

void SteadyState::deinitialize() { release(); }

void SteadyState::release() noexcept {
    _network.reset();
    _prepared = false;
    _symmetric = false;
    _initialized = false;
}

}  // namespace pycanha::solvers::steady
