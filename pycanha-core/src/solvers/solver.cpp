#include "pycanha-core/solvers/solver.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "auto_configure.hpp"
#include "factorizer.hpp"
#include "memory.hpp"
#include "pycanha-core/config.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/parameters/formulas.hpp"
#include "pycanha-core/solvers/model_changed_error.hpp"
#include "pycanha-core/solvers/report.hpp"
#include "pycanha-core/solvers/settings.hpp"
#include "pycanha-core/tmm/nodes.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"
#include "pycanha-core/utils/logger.hpp"
#include "pycanha-core/utils/profiling.hpp"

namespace pycanha::solvers {

namespace {

class ActiveSolverCallbackScope {
  public:
    ActiveSolverCallbackScope(ThermalMathematicalModel& tmm, Solver& solver)
        : _tmm(tmm) {
        _tmm.set_current_callback_solver(std::addressof(solver));
    }

    ~ActiveSolverCallbackScope() { _tmm.set_current_callback_solver(nullptr); }

    ActiveSolverCallbackScope(const ActiveSolverCallbackScope&) = delete;
    ActiveSolverCallbackScope& operator=(const ActiveSolverCallbackScope&) =
        delete;
    ActiveSolverCallbackScope(ActiveSolverCallbackScope&&) = delete;
    ActiveSolverCallbackScope& operator=(ActiveSolverCallbackScope&&) = delete;

  private:
    ThermalMathematicalModel& _tmm;
};

// The parts that differ, as "nodes", "nodes and conductive couplings" or
// "nodes, conductive couplings and radiative couplings".
std::string describe_changes(const StructureVersions& before,
                             const StructureVersions& now) {
    std::vector<std::string_view> parts;
    if (before.nodes != now.nodes) {
        parts.emplace_back("nodes");
    }
    if (before.conductive != now.conductive) {
        parts.emplace_back("conductive couplings");
    }
    if (before.radiative != now.radiative) {
        parts.emplace_back("radiative couplings");
    }
    std::string text;
    for (std::size_t part = 0; part < parts.size(); ++part) {
        if (part > 0) {
            text += part + 1 == parts.size() ? " and " : ", ";
        }
        text += parts[part];
    }
    return text;
}

std::string describe_not_applied(const std::vector<NotApplied>& ignored) {
    std::string text;
    for (const NotApplied& field : ignored) {
        if (!text.empty()) {
            text += "; ";
        }
        text += field.name + " (" + field.reason + ")";
    }
    return text;
}

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         start)
        .count();
}

// Adds the time spent in its scope to a total [s].
class ScopeTimer {
  public:
    explicit ScopeTimer(double& total)
        : _total(total), _start(std::chrono::steady_clock::now()) {}
    ~ScopeTimer() { _total += seconds_since(_start); }

    ScopeTimer(const ScopeTimer&) = delete;
    ScopeTimer& operator=(const ScopeTimer&) = delete;
    ScopeTimer(ScopeTimer&&) = delete;
    ScopeTimer& operator=(ScopeTimer&&) = delete;

  private:
    double& _total;
    std::chrono::steady_clock::time_point _start;
};

constexpr int explain_name_width = 19;
constexpr int explain_value_width = 21;

std::string explain_line(std::string_view name, std::string_view value,
                         std::string_view reason) {
    return std::format("\n  {:<{}}{:<{}}  {}", name, explain_name_width, value,
                       explain_value_width, reason);
}

}  // namespace

Solver::Solver(std::shared_ptr<ThermalMathematicalModel> tmm_shptr,
               SolverType type)
    : _tmm_shptr(std::move(tmm_shptr)),
      tmm(*_tmm_shptr),
      tnw(tmm.network()),
      ltcs(tnw.conductive_matrices()),
      rtcs(tnw.radiative_matrices()),
      KLdd(ltcs.sparse_dd),
      KLdb(ltcs.sparse_db),
      KLbb(ltcs.sparse_bb),
      KRdd(rtcs.sparse_dd),
      KRdb(rtcs.sparse_db),
      KRbb(rtcs.sparse_bb),
      T(nullptr, Index{0}),
      Td(nullptr, Index{0}),
      Tb(nullptr, Index{0}),
      Qd(nullptr, Index{0}),
      Qb(nullptr, Index{0}),
      C(nullptr, Index{0}),
      Cd(nullptr, Index{0}),
      Cb(nullptr, Index{0}),
      QI_sp(tmm.nodes().qi_vector),
      QS_sp(tmm.nodes().qs_vector),
      QA_sp(tmm.nodes().qa_vector),
      QE_sp(tmm.nodes().qe_vector),
      QR_sp(tmm.nodes().qr_vector),
      _solver_type(type) {
    _settings.solver_type = type;
}

void Solver::set_settings(const Settings& settings) {
    Settings replacement = settings;
    replacement.solver_type = _solver_type;
    check_settings_change(_settings, replacement,
                          resolve_context(_solver_type, tmm), _initialized);
    _settings = std::move(replacement);
}

void Solver::set(std::string_view name, const SettingValue& value) {
    Settings replacement = _settings;
    replacement.set(name, value);
    set_settings(replacement);
}

void Solver::unset(std::string_view name) {
    Settings replacement = _settings;
    replacement.unset(name);
    set_settings(replacement);
}

ResolvedSettings Solver::resolved_settings() const {
    return resolve(_settings, resolve_context(_solver_type, tmm));
}

Solver::FormulaExecutionGuard::FormulaExecutionGuard(Solver& solver)
    : _solver(solver) {
    _solver.prepare_formulas_for_execution();
}

Solver::FormulaExecutionGuard::~FormulaExecutionGuard() {
    _solver.release_formulas_after_execution();
}

void Solver::initialize_common(bool log_warnings) {
    SPDLOG_LOGGER_DEBUG(pycanha::get_logger(), "{} initializing...", name());

    // Nodes appended since the last coupling access have not grown the
    // matrices yet.
    tnw.synchronize_structure();
    _resolved = resolve(_settings, resolve_context(_solver_type, tmm));
    if (log_warnings) {
        if (!_resolved.not_applied.empty()) {
            warn("settings not applied: " +
                 describe_not_applied(_resolved.not_applied));
        }
        for (const auto& warning : _resolved.warnings) {
            warn(warning);
        }
    }
    _initialized_structure = tmm.structure_versions();

    KLdd.makeCompressed();
    KLdb.makeCompressed();
    KLbb.makeCompressed();

    KRdd.makeCompressed();
    KRdb.makeCompressed();
    KRbb.makeCompressed();

    auto& nodes = tmm.nodes();
    N = nodes.num_nodes();
    nd = KLdd.cols();
    nb = KLdb.cols();

    dTd = VectorXd::Zero(nd);
    Td_solver = VectorXd::Zero(nd);
    Q = VectorXd::Zero(N);

    new (&T) WrappVectorXd(nodes.T_vector.data(), N);
    new (&Td) WrappVectorXd(nodes.T_vector.data(), nd);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    new (&Tb) WrappVectorXd(nodes.T_vector.data() + nd, nb);
    new (&C) WrappVectorXd(nodes.C_vector.data(), N);
    new (&Cd) WrappVectorXd(nodes.C_vector.data(), nd);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    new (&Cb) WrappVectorXd(nodes.C_vector.data() + nd, nb);
    new (&Qd) WrappVectorXd(Q.data(), nd);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    new (&Qb) WrappVectorXd(Q.data() + nd, nb);
}

void Solver::refresh_live_settings() {
    check_model_unchanged();
    // The INIT fields cannot have changed while the solver is initialised,
    // so resolving again only picks up the LIVE ones.
    _resolved = resolve(_settings, resolve_context(_solver_type, tmm));
}

void Solver::check_model_unchanged(std::string_view callback) const {
    const StructureVersions now = tmm.structure_versions();
    if (now == _initialized_structure) {
        return;
    }
    std::string message = std::string(name()) + ": the model's " +
                          describe_changes(_initialized_structure, now) +
                          " changed ";
    if (callback.empty()) {
        message += "after initialize()";
    } else {
        message += "in the ";
        message += callback;
        message += " callback";
    }
    message += "; call initialize() again";
    throw ModelChangedError(message);
}

void Solver::callback_transient_time_change() {
    PYCANHA_PROFILE_SCOPE("Callback Time Change");
    {
        const ScopeTimer timer(_report.callback_time);
        const ActiveSolverCallbackScope active_solver_scope(tmm, *this);
        _tmm_shptr->callback_transient_time_change();
    }
    check_model_unchanged("time change");
}

void Solver::callback_solver_loop() {
    PYCANHA_PROFILE_SCOPE("Callback Solver Loop");
    {
        const ScopeTimer timer(_report.callback_time);
        const ActiveSolverCallbackScope active_solver_scope(tmm, *this);
        _tmm_shptr->callback_solver_loop();
    }
    check_model_unchanged("solver loop");
}

void Solver::callback_transient_after_timestep() {
    PYCANHA_PROFILE_SCOPE("Callback Timestep");
    {
        const ScopeTimer timer(_report.callback_time);
        const ActiveSolverCallbackScope active_solver_scope(tmm, *this);
        _tmm_shptr->callback_transient_after_timestep();
    }
    check_model_unchanged("after time step");
}

void Solver::prepare_formulas_for_execution() {
    tmm.formulas().validate_for_execution();
    tmm.formulas().compile_formulas();
    tmm.formulas().lock_parameters_for_execution();
}

void Solver::release_formulas_after_execution() noexcept {
    tmm.formulas().unlock_parameters();
}

std::int64_t Solver::common_bytes() const noexcept {
    return detail::bytes_of(dTd) + detail::bytes_of(Td_solver) +
           detail::bytes_of(Q);
}

void Solver::mark_prepared() {
    _prepared = true;
    _prepared_settings = _settings;
}

bool Solver::preparation_matches() const {
    return _prepared && tmm.structure_versions() == _initialized_structure &&
           changed_init_fields(_prepared_settings, _settings,
                               resolve_context(_solver_type, tmm))
               .empty();
}

void Solver::warn(const std::string& message) {
    SPDLOG_LOGGER_WARN(get_logger(), "{}: {}", name(), message);
    _report.warnings.push_back(message);
}

void Solver::begin_initialize_report() {
    _report = SolverReport{};
    _report.solver = std::string(name());
}

void Solver::end_initialize_report(Clock::time_point start) {
    _report.initialize_time = seconds_since(start);
    _initialize_warnings = _report.warnings;
}

void Solver::begin_solve_report() {
    SolverReport fresh;
    fresh.solver = std::string(name());
    fresh.status = SolveStatus::FAILED;
    fresh.initialize_time = _report.initialize_time;
    fresh.warnings = _initialize_warnings;
    _report = std::move(fresh);
}

void Solver::record_linear_solve(Clock::time_point start, bool factorized) {
    _report.linear_time += seconds_since(start);
    ++_report.solves;
    if (factorized) {
        ++_report.factorizations;
    }
}

void Solver::record_steady_iterations(int iterations, bool converged) {
    _report.iterations = iterations;
    _report.max_step_iterations = iterations;
    _report.max_dT = max_dT;
    record_reuse_error();
    if (!converged) {
        record_nonconverged(0, 0.0, iterations);
    }
}

void Solver::record_step_iterations(int step, double time, int iterations,
                                    bool converged) {
    ++_report.steps;
    _report.iterations += iterations;
    _report.max_step_iterations =
        std::max(_report.max_step_iterations, iterations);
    _report.max_dT = max_dT;
    record_reuse_error();
    if (!converged) {
        // The report counts time steps from 1.
        record_nonconverged(step + 1, time, iterations);
    }
}

// The iteration on old factors stops at a relative residual of
// reuse_tolerance, which stays in the last change of each loop.
void Solver::record_reuse_error() noexcept {
    if (_resolved.reuse_factors) {
        _report.reuse_error =
            std::max(_report.reuse_error, _resolved.reuse_tolerance * max_dT);
    }
}

void Solver::record_nonconverged(int step, double time, int iterations) {
    Index max_index = 0;
    const double change =
        dTd.size() > 0 ? dTd.cwiseAbs().maxCoeff(&max_index) : 0.0;
    const auto node = tmm.nodes().get_node_num_from_idx(max_index);
    const NonConvergedStep entry{.step = step,
                                 .time = time,
                                 .iterations = iterations,
                                 .max_dT = change,
                                 .node = node.has_value() ? *node : -1};
    ++_report.nonconverged_steps;
    if (_report.nonconverged.size() < SolverReport::max_listed_steps) {
        _report.nonconverged.push_back(entry);
    }
    if (_report.nonconverged_steps > 1) {
        return;
    }
    if (_solver_type == SolverType::STEADY_STATE) {
        warn(
            std::format("did not converge after {} iterations: max. dT = "
                        "{:.3g} K at node {}",
                        iterations, entry.max_dT, entry.node));
    } else {
        warn(
            std::format("time step {} (t = {} s) did not converge after {} "
                        "iterations: max. dT = {:.3g} K at node {}. Later "
                        "non-converged steps are only counted; see report()",
                        step, time, iterations, entry.max_dT, entry.node));
    }
}

void Solver::finish_solve_report(Clock::time_point start, bool failed) {
    if (failed) {
        _report.status = SolveStatus::FAILED;
    } else {
        _report.status = _report.nonconverged_steps == 0
                             ? SolveStatus::CONVERGED
                             : SolveStatus::NOT_CONVERGED;
    }
    if (_report.steps > 0 && _report.nonconverged_steps > 0) {
        warn(std::format("{} of {} time steps did not converge; see report()",
                         _report.nonconverged_steps, _report.steps));
    }
    _converged = _report.status == SolveStatus::CONVERGED;
    _report.solve_time = seconds_since(start);
    _report.memory = memory_now();
}

std::string_view Solver::auto_reason(std::string_view name) const {
    const auto found = std::ranges::find(_auto_values, name, &AutoValue::name);
    if (found == _auto_values.end() || _settings.get(name) != found->value) {
        return {};
    }
    return found->reason;
}

Origin Solver::origin(std::string_view name) const {
    const Origin settings_origin =
        explain_settings(_settings, resolve_context(_solver_type, tmm))
            .resolved.origin(name);
    if (settings_origin == Origin::USER && !auto_reason(name).empty()) {
        return Origin::AUTO_CONFIGURE;
    }
    return settings_origin;
}

std::string Solver::explain() const {
    const SettingsExplanation explanation =
        explain_settings(_settings, resolve_context(_solver_type, tmm));
    const ResolvedSettings& resolved = explanation.resolved;
    std::string text = std::format("{} settings", name());
    for (const FieldExplanation& field : explanation.fields) {
        std::string reason;
        const std::string_view chosen = auto_reason(field.name);
        if (field.origin == Origin::USER && !chosen.empty()) {
            reason = std::format("auto_configure: {}", chosen);
        } else if (field.origin == Origin::USER) {
            reason = field.reason.empty() ? "set" : "set, " + field.reason;
        } else {
            reason = field.reason.empty() ? "default" : field.reason;
        }
        text += explain_line(field.name, field.value_text, reason);
    }
    if (resolved.backend == Backend::PARDISO) {
        const bool symmetric =
            resolved.factorization == Factorization::CHOLESKY;
        text += std::format("\nPARDISO matrix type {}, iparm (zero-based)",
                            detail::pardiso_matrix_type(symmetric));
        for (const detail::IparmEntry& entry :
             detail::pardiso_iparm(resolved, symmetric)) {
            text += explain_line(std::format("iparm[{}]", entry.index),
                                 std::format("{}", entry.value), entry.meaning);
        }
    }
    if (!resolved.not_applied.empty()) {
        text += std::format("\nSet but not applied to {}", name());
        for (const NotApplied& ignored : resolved.not_applied) {
            text += explain_line(ignored.name, "", ignored.reason);
        }
    }
    for (const std::string& warning : resolved.warnings) {
        text += std::format("\nWarning: {}", warning);
    }
    for (const std::string& contradiction : explanation.contradictions) {
        text += std::format("\nError: {}", contradiction);
    }
    return text;
}

SolverMemory Solver::estimate_memory_quietly() {
    if (!_initialized && !preparation_matches()) {
        prepare(/*log_warnings=*/false);
        if (!_prepared) {
            throw std::runtime_error(std::format(
                "{}: the analysis of the matrix failed; see the log", name()));
        }
    }
    return memory_now();
}

SolverMemory Solver::estimate_memory() {
    const SolverMemory memory = estimate_memory_quietly();
    SPDLOG_LOGGER_INFO(get_logger(), "{} memory estimate: {}", name(),
                       memory.to_string());
    return memory;
}

Settings Solver::auto_configure(std::string_view max_memory) {
    return auto_configure(parse_memory_size(max_memory));
}

Settings Solver::auto_configure(std::optional<std::int64_t> max_memory) {
    if (_initialized) {
        throw std::runtime_error(std::format(
            "{}: auto_configure() changes settings that initialize() reads; "
            "call deinitialize() first",
            name()));
    }
    if (max_memory.has_value() && *max_memory <= 0) {
        throw std::invalid_argument("max_memory must be positive");
    }
    // Nodes appended since the last coupling access have not grown the
    // matrices yet.
    tnw.synchronize_structure();
    const std::vector<detail::AutoCandidate> candidates =
        detail::auto_candidates(
            _solver_type,
            {.mkl = MKL_ENABLED, .accelerate = accelerate_available()},
            detail::network_shape(KLdd, KRdd));

    Settings base = _settings;
    detail::clear_auto_fields(base);
    const auto settings_for = [&base](const detail::AutoCandidate& candidate) {
        Settings settings = base;
        for (const detail::AutoField& field : candidate.fields) {
            settings.set(field.name, field.value);
        }
        return settings;
    };

    std::size_t chosen = 0;
    bool fits = true;
    std::int64_t smallest = std::numeric_limits<std::int64_t>::max();
    if (max_memory.has_value()) {
        fits = false;
        for (std::size_t index = 0; index < candidates.size(); ++index) {
            set_settings(settings_for(candidates[index]));
            const std::int64_t total = estimate_memory_quietly().total();
            if (total <= *max_memory) {
                chosen = index;
                fits = true;
                break;
            }
            if (total < smallest) {
                smallest = total;
                chosen = index;
            }
        }
    }
    set_settings(settings_for(candidates.at(chosen)));

    _auto_values.clear();
    std::ranges::transform(candidates.at(chosen).fields,
                           std::back_inserter(_auto_values),
                           [](const detail::AutoField& field) {
                               return AutoValue{.name = std::string(field.name),
                                                .value = field.value,
                                                .reason = field.reason};
                           });
    if (!fits) {
        SPDLOG_LOGGER_WARN(
            get_logger(),
            "{}: no configuration fits max_memory = {}; auto_configure() "
            "chose the smallest, which needs {}",
            name(), format_bytes(*max_memory), format_bytes(smallest));
    }
    const ResolvedSettings resolved = resolved_settings();
    SPDLOG_LOGGER_INFO(get_logger(), "{} auto_configure(): {}", name(),
                       detail::describe(resolved, resolved.factorization ==
                                                      Factorization::CHOLESKY));
    return settings();
}

bool Solver::has_radiation() const noexcept {
    return KRdd.nonZeros() != 0 || KRdb.nonZeros() != 0;
}

bool Solver::temperature_convergence_check() {
    PYCANHA_PROFILE_SCOPE("Convergence Check");
    dTd = Td_solver - Td;
    max_dT = dTd.cwiseAbs().maxCoeff();
    return max_dT < _resolved.tolerance;
}

}  // namespace pycanha::solvers
