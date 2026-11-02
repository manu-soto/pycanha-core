#include "pycanha-core/solvers/steady_state.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "factorizer.hpp"
#include "linearized_network.hpp"
#include "memory.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/parameters/entity.hpp"
#include "pycanha-core/parameters/formula.hpp"
#include "pycanha-core/parameters/formulas.hpp"
#include "pycanha-core/solvers/report.hpp"
#include "pycanha-core/solvers/settings.hpp"
#include "pycanha-core/solvers/solver.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/utils/logger.hpp"
#include "pycanha-core/utils/profiling.hpp"

namespace pycanha::solvers::steady {

namespace {

// What each selected parameter changes in the network: d(entity)/d(parameter)
// of every ParameterFormula, collected as JacobianPropagation does.
struct ParameterChanges {
    std::vector<std::string> names;
    // Per parameter: the upper triangle among the diffusive nodes, and the
    // diffusive x boundary block.
    std::vector<SpMatRow> conductive;
    std::vector<SpMatRow> conductive_boundary;
    std::vector<SpMatRow> radiative;
    std::vector<SpMatRow> radiative_boundary;
    // Diffusive nodes x parameters.
    Eigen::MatrixXd heat;
    // Boundary nodes x parameters.
    Eigen::MatrixXd boundary_temperature;
};

[[nodiscard]] bool is_heat_flux_entity(EntityType type) {
    return type == EntityType::qs || type == EntityType::qa ||
           type == EntityType::qe || type == EntityType::qi ||
           type == EntityType::qr;
}

[[nodiscard]] Index node_index(const Nodes& nodes, NodeNum node_num) {
    const auto resolved = nodes.get_idx_from_node_num(node_num);
    if (!resolved.has_value()) {
        throw std::invalid_argument(
            "A parameter formula references the missing node " +
            std::to_string(node_num));
    }
    return *resolved;
}

void add_change(const Entity& entity, Index column, double value,
                const Nodes& nodes, Index nd, ParameterChanges& changes) {
    const auto type = entity.type();
    const auto parameter = to_sizet(column);
    if (type == EntityType::gl || type == EntityType::gr) {
        const bool conductive = type == EntityType::gl;
        const Index first = node_index(nodes, entity.node_1());
        const Index second = node_index(nodes, entity.node_2());
        if (first < nd && second < nd) {
            auto& matrix = conductive ? changes.conductive[parameter]
                                      : changes.radiative[parameter];
            matrix.coeffRef(std::min(first, second), std::max(first, second)) +=
                value;
        } else if ((first < nd) != (second < nd)) {
            auto& matrix = conductive ? changes.conductive_boundary[parameter]
                                      : changes.radiative_boundary[parameter];
            const Index diffusive = (first < nd) ? first : second;
            const Index boundary = (first < nd) ? second : first;
            matrix.coeffRef(diffusive, boundary - nd) += value;
        }
        // Between two boundary nodes: no effect on the diffusive nodes.
        return;
    }
    const Index index = node_index(nodes, entity.node_1());
    if (is_heat_flux_entity(type)) {
        if (index < nd) {
            changes.heat(index, column) += value;
        }
        return;
    }
    if (type == EntityType::t && index >= nd) {
        changes.boundary_temperature(index - nd, column) += value;
    }
    // A capacity, or the start temperature of a diffusive node: no effect on
    // a steady state.
}

ParameterChanges collect_parameter_changes(ThermalMathematicalModel& model,
                                           Index nd, Index nb) {
    std::unordered_set<std::string> produced;
    for (const auto& formula : model.formulas().formulas()) {
        if (dynamic_cast<ParameterFormula*>(formula.get()) != nullptr) {
            const auto& dependencies = formula->parameter_dependencies();
            produced.insert(dependencies.begin(), dependencies.end());
        }
    }

    ParameterChanges changes;
    for (const auto& name :
         model.formulas().parameters_with_derivatives().parameter_names()) {
        if (!produced.contains(name)) {
            throw std::invalid_argument("Derivative parameter '" + name +
                                        "' is not produced by any "
                                        "ParameterFormula");
        }
        changes.names.push_back(name);
    }
    if (changes.names.empty()) {
        throw std::invalid_argument(
            "SteadyState::jacobian() requires at least one selected "
            "derivative parameter");
    }

    const std::size_t count = changes.names.size();
    changes.conductive.assign(count, SpMatRow(nd, nd));
    changes.conductive_boundary.assign(count, SpMatRow(nd, nb));
    changes.radiative.assign(count, SpMatRow(nd, nd));
    changes.radiative_boundary.assign(count, SpMatRow(nd, nb));
    changes.heat = Eigen::MatrixXd::Zero(nd, to_idx(count));
    changes.boundary_temperature = Eigen::MatrixXd::Zero(nb, to_idx(count));

    std::unordered_map<std::string, Index> columns;
    for (std::size_t column = 0; column < count; ++column) {
        columns.emplace(changes.names[column], to_idx(column));
    }

    for (const auto& formula : model.formulas().formulas()) {
        auto* parameter_formula =
            dynamic_cast<ParameterFormula*>(formula.get());
        if (parameter_formula == nullptr) {
            continue;
        }
        parameter_formula->calculate_derivatives();
        const auto* derivatives = formula->get_derivative_values();
        if ((derivatives == nullptr) || derivatives->empty()) {
            continue;
        }
        const auto& dependencies = formula->parameter_dependencies();
        if (dependencies.size() != derivatives->size()) {
            throw std::invalid_argument(
                "Formula derivative metadata size mismatch for " +
                formula->entity().string_representation());
        }
        for (std::size_t dependency = 0; dependency < dependencies.size();
             ++dependency) {
            const auto found = columns.find(dependencies[dependency]);
            if (found != columns.end()) {
                add_change(formula->entity(), found->second,
                           derivatives->at(dependency), model.nodes(), nd,
                           changes);
            }
        }
    }
    return changes;
}

// dR_i += sum_j dG_ij (x_j - x_i) over the diffusive nodes j (the upper
// triangle `domain`) and the boundary nodes (`boundary`).
void add_coupling_change(const SpMatRow& domain, const SpMatRow& boundary,
                         const Eigen::VectorXd& x_domain,
                         const Eigen::VectorXd& x_boundary,
                         Eigen::VectorXd& change) {
    const auto symmetric = domain.selfadjointView<Eigen::Upper>();
    const Eigen::VectorXd row_sums =
        (symmetric * Eigen::VectorXd::Ones(domain.rows())) +
        (boundary * Eigen::VectorXd::Ones(boundary.cols()));
    change.noalias() += symmetric * x_domain;
    change.noalias() += boundary * x_boundary;
    change -= row_sums.cwiseProduct(x_domain);
}

}  // namespace

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
    _derivative_parameter_names.clear();
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
    _tangent_factorized = false;
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

void SteadyState::factorize_tangent() {
    if (!_initialized) {
        throw std::logic_error(std::string(name()) +
                               ": initialize() and solve() before jacobian() "
                               "or linear_response()");
    }
    if (!_converged) {
        throw std::logic_error(std::string(name()) +
                               ": the last solve() did not converge, so its "
                               "temperatures are not a steady state");
    }
    if (_tangent_factorized) {
        return;
    }
    PYCANHA_PROFILE_SCOPE("Steady state tangent");
    // The last iteration linearized the network at the temperatures before
    // its change: linearize again at the converged ones, which gives the
    // exact tangent A = -dR/dT, and factorize it (no iteration on older
    // factors).
    if (_radiation) {
        _t_cubed_domain = (4.0 * STF_BOLTZ) * Td.array().cube();
    }
    _network->linearize(_t_cubed_domain, Td, VectorXd{},
                        _radiation_linear_term);
    _network->discard_factors();
    VectorXd rhs = VectorXd::Ones(nd);
    VectorXd solution = VectorXd::Zero(nd);
    if (!_network->solve(rhs, solution).succeeded) {
        throw std::runtime_error(std::string(name()) +
                                 ": the factorization at the converged "
                                 "temperatures failed (" +
                                 _network->error_message() + ")");
    }
    _tangent_factorized = true;
}

Eigen::MatrixXd SteadyState::jacobian() {
    PYCANHA_PROFILE_SCOPE("Steady state jacobian");
    factorize_tangent();
    ParameterChanges changes = collect_parameter_changes(tmm, nd, nb);
    const auto count = to_idx(changes.names.size());

    const VectorXd t4_domain = STF_BOLTZ * Td.array().square().square();
    const VectorXd t4_boundary = STF_BOLTZ * Tb.array().square().square();
    const VectorXd t3_boundary = (4.0 * STF_BOLTZ) * Tb.array().cube();

    // dR/dp: the couplings' change with x = T (conductive) or sigma T^4
    // (radiative), the heat loads, and (GL_ib + 4 sigma GR_ib T_b^3) dT_b for
    // a boundary temperature.
    Eigen::MatrixXd rhs(nd, count);
    for (Index column = 0; column < count; ++column) {
        const auto parameter = to_sizet(column);
        VectorXd change = changes.heat.col(column);
        add_coupling_change(changes.conductive[parameter],
                            changes.conductive_boundary[parameter], Td, Tb,
                            change);
        add_coupling_change(changes.radiative[parameter],
                            changes.radiative_boundary[parameter], t4_domain,
                            t4_boundary, change);
        const VectorXd boundary_change =
            changes.boundary_temperature.col(column);
        change.noalias() += KLdb * boundary_change;
        change.noalias() += KRdb * t3_boundary.cwiseProduct(boundary_change);
        rhs.col(column) = change;
    }

    Eigen::MatrixXd derivatives = Eigen::MatrixXd::Zero(nd, count);
    if (!_network->solve(rhs, derivatives)) {
        throw std::runtime_error(std::string(name()) +
                                 ": the derivative solve failed (" +
                                 _network->error_message() + ")");
    }
    _derivative_parameter_names = std::move(changes.names);
    return derivatives;
}

Eigen::MatrixXd SteadyState::linear_response(const Eigen::MatrixXd& heat) {
    PYCANHA_PROFILE_SCOPE("Steady state linear response");
    factorize_tangent();
    if (heat.rows() != nd) {
        throw std::invalid_argument(std::string(name()) +
                                    ": linear_response() needs one row per " +
                                    "diffusive node (" + std::to_string(nd) +
                                    "), not " + std::to_string(heat.rows()));
    }
    Eigen::MatrixXd rhs = heat;
    Eigen::MatrixXd response = Eigen::MatrixXd::Zero(nd, heat.cols());
    if (!_network->solve(rhs, response)) {
        throw std::runtime_error(std::string(name()) +
                                 ": the response solve failed (" +
                                 _network->error_message() + ")");
    }
    return response;
}

void SteadyState::deinitialize() { release(); }

void SteadyState::release() noexcept {
    _network.reset();
    _prepared = false;
    _symmetric = false;
    _tangent_factorized = false;
    _initialized = false;
}

}  // namespace pycanha::solvers::steady
