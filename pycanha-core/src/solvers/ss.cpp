#include "pycanha-core/solvers/ss.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "pycanha-core/config.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/solvers/linear_solver.hpp"
#include "pycanha-core/solvers/solver.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/utils/SparseUtils.hpp"
#include "pycanha-core/utils/logger.hpp"
#include "pycanha-core/utils/profiling.hpp"

namespace pycanha {

namespace {

// Position in the value array of the stored entry (row, col) of a compressed
// row-major matrix.
int value_index(const SpMatRow& matrix, Index row, Index col) {
    const std::span<const int> outer(matrix.outerIndexPtr(),
                                     to_sizet(matrix.outerSize() + 1));
    const std::span<const int> inner(matrix.innerIndexPtr(),
                                     to_sizet(matrix.nonZeros()));
    const auto first = inner.begin() + outer[to_sizet(row)];
    const auto last = inner.begin() + outer[to_sizet(row + 1)];
    const auto found = std::lower_bound(first, last, static_cast<int>(col));
    PYCANHA_ASSERT(found != last && *found == col,
                   "Entry missing from the solver matrix pattern");
    return static_cast<int>(found - inner.begin());
}

// For every stored value of source, in storage order, the position in
// target's value array of the same entry, or of its transpose.
std::vector<int> map_values(const SpMatRow& source, const SpMatRow& target,
                            bool transposed) {
    std::vector<int> indices;
    indices.reserve(to_sizet(source.nonZeros()));
    for (Index row = 0; row < source.outerSize(); ++row) {
        for (SpMatRow::InnerIterator it(source, row); it; ++it) {
            indices.push_back(transposed ? value_index(target, it.col(), row)
                                         : value_index(target, row, it.col()));
        }
    }
    return indices;
}

// dest[indices[k]] += scale * source[k]
void scatter_add(std::span<double> dest, std::span<const double> source,
                 const std::vector<int>& indices, double scale) {
    for (std::size_t k = 0; k < indices.size(); ++k) {
        dest[to_sizet(indices[k])] += scale * source[k];
    }
}

std::span<double> mutable_values_of(SpMatRow& matrix) {
    return {matrix.valuePtr(), to_sizet(matrix.nonZeros())};
}

std::span<const double> values_of(const SpMatRow& matrix) {
    return {matrix.valuePtr(), to_sizet(matrix.nonZeros())};
}

}  // namespace

SteadyStateSolver::SteadyStateSolver(
    std::shared_ptr<ThermalMathematicalModel> tmm_shptr)
    : Solver(std::move(tmm_shptr)) {}

SteadyStateSolver::~SteadyStateSolver() = default;

void SteadyStateSolver::restart_solve() {
    SPDLOG_LOGGER_DEBUG(pycanha::get_logger(), "Re-starting solve...");
    SPDLOG_LOGGER_ERROR(pycanha::get_logger(), "Not implemented yet.");
}

bool SteadyStateSolver::has_radiation() const noexcept {
    return KRdd.nonZeros() != 0 || KRdb.nonZeros() != 0;
}

bool SteadyStateSolver::model_structure_unchanged() const noexcept {
    return KLdd.rows() == nd && KLdb.cols() == nb &&
           _initialized_non_zeros ==
               std::array<Index, 4>{KLdd.nonZeros(), KLdb.nonZeros(),
                                    KRdd.nonZeros(), KRdb.nonZeros()};
}

void SteadyStateSolver::initialize_linearised(
    const LinearSolverOptions& options) {
    // Throws before anything is built when the options are not available.
    _linear_solver = make_linear_solver(options);
    _symmetric = options.symmetric_positive_definite;
    _radiation = has_radiation();
    SPDLOG_LOGGER_INFO(
        get_logger(), "{} initializing ({} {}, {})", solver_name,
        to_string(options.engine),
        to_string(resolve_solver_type(options.engine, options.type)),
        _symmetric ? "Cholesky" : "LU");

    build_pattern();

    _b = VectorXd::Zero(nd);
    _factorized_values = VectorXd::Zero(_a_matrix.nonZeros());
    _t_cubed_domain = VectorXd::Zero(nd);
    _t_fourth_boundary = VectorXd::Zero(nb);
    _radiation_linear_term = VectorXd::Zero(nd);
    _row_sums = VectorXd::Zero(nd);
    _ones_domain = VectorXd::Ones(nd);
    _ones_boundary = VectorXd::Ones(nb);
    _initialized_non_zeros = {KLdd.nonZeros(), KLdb.nonZeros(), KRdd.nonZeros(),
                              KRdb.nonZeros()};
    _has_factors = false;

    _linear_solver->analyze_pattern(_a_matrix);
    if (!_linear_solver->succeeded()) {
        SPDLOG_LOGGER_ERROR(get_logger(), "{}: analysis failed ({}).",
                            solver_name, _linear_solver->error_message());
        release_linearised();
        return;
    }
    suggest_min_degree_if_dense(*_linear_solver, options, nd, solver_name);
    solver_initialized = true;
}

void SteadyStateSolver::build_pattern() {
    if (_symmetric) {
        // Upper triangle with the diagonal: KLdd is already stored that way.
        _a_matrix = KLdd;
        sparse_utils::add_zero_diag_square(_a_matrix);
        _upper_kl_indices = map_values(KLdd, _a_matrix, /*transposed=*/false);
        _lower_kl_indices.clear();
        _upper_kr_indices.clear();
        _lower_kr_indices.clear();
    } else {
        _a_matrix = KLdd.selfadjointView<Eigen::Upper>();
        if (_radiation) {
            _a_matrix += KRdd.selfadjointView<Eigen::Upper>();
        }
        sparse_utils::add_zero_diag_square(_a_matrix);
        _upper_kl_indices = map_values(KLdd, _a_matrix, /*transposed=*/false);
        _lower_kl_indices = map_values(KLdd, _a_matrix, /*transposed=*/true);
        _upper_kr_indices.clear();
        _lower_kr_indices.clear();
        if (_radiation) {
            _upper_kr_indices =
                map_values(KRdd, _a_matrix, /*transposed=*/false);
            _lower_kr_indices =
                map_values(KRdd, _a_matrix, /*transposed=*/true);
        }
    }

    _diagonal_indices.resize(to_sizet(nd));
    for (Index row = 0; row < nd; ++row) {
        _diagonal_indices[to_sizet(row)] = value_index(_a_matrix, row, row);
    }
    sparse_utils::set_to_zero(_a_matrix);
}

void SteadyStateSolver::assemble_matrix() {
    PYCANHA_PROFILE_SCOPE("Steady state assemble matrix");
    sparse_utils::set_to_zero(_a_matrix);
    const auto values = mutable_values_of(_a_matrix);
    const std::span<const double> diagonal_sums(_row_sums.data(), to_sizet(nd));

    if (_radiation) {
        const auto kr_values = values_of(std::as_const(KRdd));
        scatter_add(values, kr_values, _upper_kr_indices, -1.0);
        scatter_add(values, kr_values, _lower_kr_indices, -1.0);
        _row_sums.noalias() =
            KRdd.selfadjointView<Eigen::Upper>() * _ones_domain;
        _row_sums.noalias() += KRdb * _ones_boundary;
        for (std::size_t row = 0; row < _diagonal_indices.size(); ++row) {
            values[to_sizet(_diagonal_indices[row])] += diagonal_sums[row];
        }

        // Newton linearisation of sigma T^4: column j carries 4 sigma Tj^3.
        const Eigen::Map<const Eigen::VectorXi> columns(
            _a_matrix.innerIndexPtr(), _a_matrix.nonZeros());
        _a_matrix.coeffs() *= _t_cubed_domain(columns).array();
        _radiation_linear_term.noalias() = _a_matrix * Td;
    }

    const auto kl_values = values_of(std::as_const(KLdd));
    scatter_add(values, kl_values, _upper_kl_indices, -1.0);
    scatter_add(values, kl_values, _lower_kl_indices, -1.0);
    _row_sums.noalias() = KLdd.selfadjointView<Eigen::Upper>() * _ones_domain;
    _row_sums.noalias() += KLdb * _ones_boundary;
    for (std::size_t row = 0; row < _diagonal_indices.size(); ++row) {
        values[to_sizet(_diagonal_indices[row])] += diagonal_sums[row];
    }
}

void SteadyStateSolver::assemble_rhs() {
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
        _b += 0.75 * _radiation_linear_term;
    }
}

bool SteadyStateSolver::factorize_and_solve_step() {
    PYCANHA_PROFILE_SCOPE("Steady state factorize and solve");
    auto& linear_solver = *_linear_solver;
    const auto values = values_of(std::as_const(_a_matrix));
    const std::span<const double> factorized(_factorized_values.data(),
                                             values.size());
    bool factors_recomputed = true;

    if (_has_factors && std::ranges::equal(values, factorized)) {
        linear_solver.solve(_a_matrix, _b, Td_solver);
        factors_recomputed = false;
        SPDLOG_LOGGER_DEBUG(get_logger(),
                            "{} iteration {}: matrix unchanged, solved with "
                            "the existing factors.",
                            solver_name, solver_iter + 1);
    } else if (_has_factors && linear_solver.iterative()) {
        linear_solver.factorize_and_solve(_a_matrix, _b, Td_solver);
        factors_recomputed = linear_solver.factors_recomputed();
        SPDLOG_LOGGER_DEBUG(get_logger(),
                            "{} iteration {}: iterative step, factors {}.",
                            solver_name, solver_iter + 1,
                            factors_recomputed ? "recomputed" : "reused");
    } else {
        linear_solver.factorize(_a_matrix);
        if (linear_solver.succeeded()) {
            linear_solver.solve(_a_matrix, _b, Td_solver);
        }
        SPDLOG_LOGGER_DEBUG(get_logger(), "{} iteration {}: factorized.",
                            solver_name, solver_iter + 1);
    }

    if (!linear_solver.succeeded()) {
        if (linear_solver.zero_pivot()) {
            SPDLOG_LOGGER_ERROR(
                get_logger(),
                "{}: zero pivot in the {} factorization ({}). The matrix is "
                "singular: a group of diffusive nodes has no conductive or "
                "radiative path to a boundary node.",
                solver_name, _symmetric ? "Cholesky" : "LU",
                linear_solver.error_message());
        } else {
            SPDLOG_LOGGER_ERROR(get_logger(), "{}: {}.", solver_name,
                                linear_solver.error_message());
        }
        _has_factors = false;
        return false;
    }

    // PARDISO replaces a pivot below 1e-13 relative to the matrix by that
    // value. A matrix diagonally dominant by columns never needs it unless
    // it is singular, and the solution is then meaningless.
    if (factors_recomputed && linear_solver.perturbed_pivots() > 0) {
        SPDLOG_LOGGER_ERROR(
            get_logger(),
            "{}: {} pivots perturbed in the LU factorization. The matrix is "
            "singular: a group of diffusive nodes has no conductive or "
            "radiative path to a boundary node.",
            solver_name, linear_solver.perturbed_pivots());
        _has_factors = false;
        return false;
    }

    if (factors_recomputed) {
        _factorized_values = Eigen::Map<const VectorXd>(_a_matrix.valuePtr(),
                                                        _a_matrix.nonZeros());
        _has_factors = true;
        ++_num_factorizations;
    }
    return true;
}

void SteadyStateSolver::solve_linearised() {
    if (!solver_initialized) {
        SPDLOG_LOGGER_ERROR(
            get_logger(),
            "Solver has not been initialized. Please call initialize() before "
            "solve().");
        return;
    }
    if (!structure_unchanged_since_initialize()) {
        return;
    }
    if (!model_structure_unchanged()) {
        SPDLOG_LOGGER_ERROR(get_logger(),
                            "{}: nodes or couplings were added or removed "
                            "after initialize(). Call initialize() again.",
                            solver_name);
        return;
    }
    SPDLOG_LOGGER_INFO(get_logger(), "{} solving...", solver_name);

    const FormulaExecutionGuard formula_execution(*this);

    solver_converged = false;
    _num_factorizations = 0;

    for (solver_iter = 0; solver_iter < max_iters; ++solver_iter) {
        PYCANHA_PROFILE_SCOPE("Steady state iteration");

        if (_radiation) {
            _t_cubed_domain = (4.0 * STF_BOLTZ) * Td.array().cube();
        }
        assemble_matrix();
        assemble_rhs();
        if (correction_form) {
            // A dT = b - A T: the same pass, solved for the change
            if (_symmetric) {
                _b.noalias() -= _a_matrix.selfadjointView<Eigen::Upper>() * Td;
            } else {
                _b.noalias() -= _a_matrix * Td;
            }
        }

        if (!factorize_and_solve_step()) {
            return;
        }

        if (correction_form) {
            dTd = Td_solver;
            Td += dTd;
        } else {
            dTd = Td_solver - Td;
            Td = Td_solver;
        }
        max_dT = dTd.cwiseAbs().maxCoeff();

        this->callback_solver_loop();

        if (max_dT < abstol_temp) {
            SPDLOG_LOGGER_INFO(get_logger(),
                               "{} converged. Num. iters: {}. Max. dT = {} K. "
                               "Factorizations: {}.",
                               solver_name, solver_iter + 1, max_dT,
                               _num_factorizations);
            solver_converged = true;
            break;
        }
    }

    if (!solver_converged) {
        SPDLOG_LOGGER_ERROR(
            get_logger(),
            "{} did NOT converge after {} iterations. Max. dT = {} K.",
            solver_name, max_iters, max_dT);
    }
}

void SteadyStateSolver::release_linearised() noexcept {
    _linear_solver.reset();
    _has_factors = false;
    _symmetric = false;
    _a_matrix = SpMatRow();
    _factorized_values.resize(0);
    solver_initialized = false;
}

}  // namespace pycanha
