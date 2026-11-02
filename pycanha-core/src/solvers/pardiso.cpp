#include "pardiso.hpp"

#if PYCANHA_USE_MKL

#include <mkl_pardiso.h>
#include <mkl_service.h>
#include <mkl_types.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

#include "factorizer.hpp"
#include "pycanha-core/config.hpp"
#include "pycanha-core/globals.hpp"
#include "thread_scope.hpp"

namespace pycanha::solvers::detail {

// Eigen's index arrays are handed to PARDISO without a copy. The build links
// the LP64 interface, where MKL_INT is a 32-bit int like Eigen's default
// StorageIndex.
static_assert(
    std::is_same_v<PardisoSolver::SparseMatrix::StorageIndex, MKL_INT>,
    "Eigen StorageIndex and MKL_INT must be the same type");

namespace {

constexpr MKL_INT phase_analysis = 11;
constexpr MKL_INT phase_factorization = 22;
constexpr MKL_INT phase_factorization_solve = 23;
constexpr MKL_INT phase_solve = 33;
constexpr MKL_INT phase_release_all = -1;

// iparm entries (zero-based, as in the MKL C documentation).
constexpr std::size_t iparm_non_default = 0;
constexpr std::size_t iparm_ordering = 1;
constexpr std::size_t iparm_iterative = 3;
constexpr std::size_t iparm_perturbed_pivots = 13;
constexpr std::size_t iparm_factor_nnz = 17;
constexpr std::size_t iparm_mflops = 18;
constexpr std::size_t iparm_cgs_diagnostic = 19;
constexpr std::size_t iparm_two_level = 23;
constexpr std::size_t iparm_zero_based = 34;

constexpr MKL_INT ordering_parallel_metis = 3;
constexpr MKL_INT report_on = -1;
constexpr MKL_INT zero_pivot_error = -4;

constexpr MKL_INT one = 1;

constexpr std::size_t iparm_peak_analysis = 14;
constexpr std::size_t iparm_permanent_analysis = 15;
constexpr std::size_t iparm_factorization_memory = 16;
constexpr std::int64_t kilobyte = 1024;

}  // namespace

PardisoSolver::PardisoSolver(MatrixType type, MKL_INT iparm_3) : _type(type) {
    auto mtype = static_cast<MKL_INT>(_type);
    pardisoinit(static_cast<void*>(_pt.data()), &mtype, _iparm.data());

    _iparm.at(iparm_non_default) = 1;
    _iparm.at(iparm_ordering) = ordering_parallel_metis;
    _iparm.at(iparm_iterative) = pardiso_iterative_control(
        iparm_3, _type == MatrixType::SYMMETRIC_POSITIVE_DEFINITE);
    _iparm.at(iparm_factor_nnz) = report_on;
    _iparm.at(iparm_mflops) = 0;
    // Two-level factorisation. The classic one-level parallel factorisation
    // (iparm[23] = 0) of MKL 2025.3 livelocks with 4 or more threads on some
    // matrices of radiative models, on Windows and Linux alike: half of the
    // threads spin and the factorisation never ends. On the same matrices
    // the two-level factorisation of matrix types 1 and 2 completes.
    _iparm.at(iparm_two_level) = 1;
    _iparm.at(iparm_zero_based) = 1;
}

PardisoSolver::~PardisoSolver() { release(); }

void PardisoSolver::set_iparm(std::size_t index, MKL_INT value) {
    PYCANHA_ASSERT(!_analyzed && index != iparm_iterative,
                   "iparm can only be set before analyze_pattern(), and "
                   "iparm[3] only through the constructor");
    _iparm.at(index) = value;
}

MKL_INT PardisoSolver::iparm(std::size_t index) const {
    return _iparm.at(index);
}

void PardisoSolver::run_phase(MKL_INT phase, const SparseMatrix& matrix,
                              MKL_INT nrhs, double* b, double* x) {
    PYCANHA_ASSERT(matrix.isCompressed(),
                   "PARDISO needs a compressed (CSR) matrix");
    PYCANHA_ASSERT(
        phase == phase_analysis || matrix.nonZeros() == _analyzed_non_zeros,
        "The matrix pattern changed after analyze_pattern()");

    const ThreadScope threads(_threads);
    auto mtype = static_cast<MKL_INT>(_type);
    _last_phase = phase;
    pardiso(static_cast<void*>(_pt.data()), &one, &one, &mtype, &phase, &_size,
            matrix.valuePtr(), matrix.outerIndexPtr(), matrix.innerIndexPtr(),
            _perm.data(), &nrhs, _iparm.data(), &_message_level, b, x, &_error);
}

void PardisoSolver::analyze_pattern(const SparseMatrix& matrix) {
    release();
    _size = static_cast<MKL_INT>(matrix.rows());
    _perm.assign(to_sizet(matrix.rows()), 0);
    _analyzed_non_zeros = matrix.nonZeros();
    run_phase(phase_analysis, matrix, one, nullptr, nullptr);
    _analyzed = _error == 0;
}

void PardisoSolver::factorize(const SparseMatrix& matrix) {
    run_phase(phase_factorization, matrix, one, nullptr, nullptr);
    _recomputed = true;
    _has_factors = _error == 0;
}

void PardisoSolver::factorize_and_solve(const SparseMatrix& matrix,
                                        Eigen::VectorXd& b,
                                        Eigen::VectorXd& x) {
    // iparm[19] is only written when an iteration is attempted. Clear it so a
    // fresh factorisation does not report the previous call.
    _iparm.at(iparm_cgs_diagnostic) = 0;
    const bool had_factors = _has_factors;
    run_phase(phase_factorization_solve, matrix, one, b.data(), x.data());
    _recomputed = !had_factors || _iparm[iparm_cgs_diagnostic] <= 0;
    _has_factors = _error == 0;
}

void PardisoSolver::solve(const SparseMatrix& matrix, Eigen::VectorXd& b,
                          Eigen::VectorXd& x) {
    run_phase(phase_solve, matrix, one, b.data(), x.data());
}

void PardisoSolver::solve(const SparseMatrix& matrix, Eigen::MatrixXd& b,
                          Eigen::MatrixXd& x) {
    x.resize(b.rows(), b.cols());
    run_phase(phase_solve, matrix, static_cast<MKL_INT>(b.cols()), b.data(),
              x.data());
}

void PardisoSolver::release() noexcept {
    if (!_analyzed) {
        return;
    }
    auto mtype = static_cast<MKL_INT>(_type);
    MKL_INT release_error = 0;
    pardiso(static_cast<void*>(_pt.data()), &one, &one, &mtype,
            &phase_release_all, &_size, nullptr, nullptr, nullptr, _perm.data(),
            &one, _iparm.data(), &_message_level, nullptr, nullptr,
            &release_error);
    _analyzed = false;
    _analyzed_non_zeros = 0;
    _last_phase = 0;
    _has_factors = false;
    _recomputed = true;
    mkl_free_buffers();
}

Eigen::ComputationInfo PardisoSolver::info() const noexcept {
    constexpr MKL_INT inconsistent_input = -1;
    if (_error == 0) {
        return Eigen::Success;
    }
    if (_error == inconsistent_input) {
        return Eigen::InvalidInput;
    }
    return Eigen::NumericalIssue;
}

bool PardisoSolver::zero_pivot() const noexcept {
    return _error == zero_pivot_error;
}

std::string PardisoSolver::error_message() const {
    return "PARDISO error " + std::to_string(_error);
}

Eigen::Index PardisoSolver::perturbed_pivots() const noexcept {
    return _iparm[iparm_perturbed_pivots];
}

Eigen::Index PardisoSolver::factor_nonzeros() const noexcept {
    return _iparm[iparm_factor_nnz];
}

MKL_INT PardisoSolver::cgs_iterations() const noexcept {
    return _iparm[iparm_cgs_diagnostic];
}

bool PardisoSolver::factors_recomputed() const noexcept {
    return _last_phase != phase_factorization_solve || _recomputed;
}

bool PardisoSolver::reuses_factors() const noexcept {
    return _iparm[iparm_iterative] != 0;
}

std::int64_t PardisoSolver::factor_bytes() const noexcept {
    const auto peak_analysis =
        static_cast<std::int64_t>(_iparm[iparm_peak_analysis]);
    const auto factorization =
        static_cast<std::int64_t>(_iparm[iparm_permanent_analysis]) +
        static_cast<std::int64_t>(_iparm[iparm_factorization_memory]);
    return kilobyte * std::max(peak_analysis, factorization);
}

bool PardisoSolver::factor_bytes_exact() const noexcept {
    return _has_factors || _iparm[iparm_two_level] != 0;
}

}  // namespace pycanha::solvers::detail

#endif  // PYCANHA_USE_MKL
