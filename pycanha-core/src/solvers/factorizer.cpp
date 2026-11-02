#include "factorizer.hpp"

#include <Eigen/OrderingMethods>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/solvers/settings.hpp"

#if PYCANHA_USE_MKL
#include "pardiso.hpp"
#endif
#if PYCANHA_USE_ACCELERATE
#include "accelerate.hpp"
#endif

namespace pycanha::solvers::detail {

namespace {

using SparseMatrix = Factorizer::SparseMatrix;
using ColumnMajorMatrix = Eigen::SparseMatrix<double>;
using StorageIndex = ColumnMajorMatrix::StorageIndex;

constexpr auto value_bytes =
    static_cast<std::int64_t>(sizeof(double) + sizeof(StorageIndex));
constexpr auto index_bytes = static_cast<std::int64_t>(sizeof(StorageIndex));

// Strictly lower non-zeros of the Cholesky factor of the symmetric pattern of
// matrix + matrix^T, with row and column i moved to new_index[i]: for every
// column k, the rows above it are walked up the elimination tree.
std::int64_t cholesky_off_diagonal(const ColumnMajorMatrix& matrix,
                                   std::span<const StorageIndex> new_index) {
    const auto n = to_sizet(matrix.cols());
    // For every permuted column, the permuted rows above it (duplicates are
    // harmless), stored as compressed lists.
    std::vector<std::size_t> starts(n + 1, 0);
    const auto for_each_pair = [&matrix, new_index](auto&& action) {
        for (Index col = 0; col < matrix.outerSize(); ++col) {
            for (ColumnMajorMatrix::InnerIterator it(matrix, col); it; ++it) {
                const auto row = to_sizet(new_index[to_sizet(it.row())]);
                const auto column = to_sizet(new_index[to_sizet(col)]);
                if (row != column) {
                    action(std::min(row, column), std::max(row, column));
                }
            }
        }
    };
    for_each_pair([&starts](std::size_t /*above*/, std::size_t column) {
        ++starts[column + 1];
    });
    for (std::size_t k = 0; k < n; ++k) {
        starts[k + 1] += starts[k];
    }
    std::vector<std::size_t> rows(starts[n]);
    std::vector<std::size_t> fill(starts.begin(), starts.end() - 1);
    for_each_pair([&rows, &fill](std::size_t above, std::size_t column) {
        rows[fill[column]++] = above;
    });

    constexpr auto none = static_cast<std::size_t>(-1);
    std::vector<std::size_t> parent(n, none);
    std::vector<std::size_t> visited(n, none);
    std::int64_t count = 0;
    for (std::size_t k = 0; k < n; ++k) {
        visited[k] = k;
        for (std::size_t entry = starts[k]; entry < starts[k + 1]; ++entry) {
            for (std::size_t node = rows[entry]; visited[node] != k;
                 node = parent[node]) {
                if (parent[node] == none) {
                    parent[node] = k;
                }
                ++count;
                visited[node] = k;
            }
        }
    }
    return count;
}

// Eigen's sparse solvers read column-major matrices. The factorizer keeps a
// column-major copy of the pattern and, for every one of its values, the
// position of the same entry in the row-major matrix, so each factorization
// only gathers the values.
template <typename EigenFactorization, bool Cholesky>
class EigenFactorizer final : public Factorizer {
  public:
    explicit EigenFactorizer(std::string_view name) : _name(name) {}

    void analyze_pattern(const SparseMatrix& matrix) override {
        // Number the row-major values, convert, and read the numbers back.
        SparseMatrix numbered = matrix;
        const std::span<double> numbers(numbered.valuePtr(),
                                        to_sizet(numbered.nonZeros()));
        for (std::size_t k = 0; k < numbers.size(); ++k) {
            numbers[k] = static_cast<double>(k);
        }
        _column_major = numbered;
        const std::span<const double> sources(
            _column_major.valuePtr(), to_sizet(_column_major.nonZeros()));
        _source.resize(sources.size());
        for (std::size_t k = 0; k < sources.size(); ++k) {
            _source[k] = static_cast<std::size_t>(sources[k]);
        }
        _factorization.analyzePattern(_column_major);
        _info = Eigen::Success;
        _factorized = false;
        _off_diagonal = cholesky_off_diagonal(_column_major, new_index());
    }

    void factorize(const SparseMatrix& matrix) override {
        const std::span<const double> values(matrix.valuePtr(),
                                             to_sizet(matrix.nonZeros()));
        const std::span<double> gathered(_column_major.valuePtr(),
                                         to_sizet(_column_major.nonZeros()));
        for (std::size_t k = 0; k < gathered.size(); ++k) {
            gathered[k] = values[_source[k]];
        }
        _factorization.factorize(_column_major);
        _info = _factorization.info();
        _factorized = _info == Eigen::Success;
    }

    void solve(const SparseMatrix& /*matrix*/, Eigen::VectorXd& b,
               Eigen::VectorXd& x) override {
        x = _factorization.solve(b);
        _info = _factorization.info();
    }

    void solve(const SparseMatrix& /*matrix*/, Eigen::MatrixXd& b,
               Eigen::MatrixXd& x) override {
        x = _factorization.solve(b);
        _info = _factorization.info();
    }

    [[nodiscard]] bool succeeded() const noexcept override {
        return _info == Eigen::Success;
    }
    [[nodiscard]] bool zero_pivot() const noexcept override {
        return _info == Eigen::NumericalIssue;
    }
    [[nodiscard]] std::string error_message() const override {
        return std::format("Eigen {} failed (ComputationInfo {})", _name,
                           static_cast<int>(_info));
    }

    // LDLT stores the strictly lower factor, the diagonal and two
    // permutations; its pattern is the symbolic count. LU is predicted with
    // the same count on both triangles until its factorization is known.
    [[nodiscard]] std::int64_t factor_bytes() const noexcept override {
        const std::int64_t n = _column_major.cols();
        if constexpr (Cholesky) {
            return (_off_diagonal * value_bytes) +
                   (n * static_cast<std::int64_t>(sizeof(double))) +
                   (((3 * n) + 1) * index_bytes);
        } else {
            const std::int64_t entries =
                _factorized ? static_cast<std::int64_t>(_factorization.nnzL() +
                                                        _factorization.nnzU())
                            : (2 * _off_diagonal) + n;
            return (entries * value_bytes) + (2 * (n + 1) * index_bytes);
        }
    }
    [[nodiscard]] bool factor_bytes_exact() const noexcept override {
        return Cholesky || _factorized;
    }
    [[nodiscard]] std::int64_t matrix_copy_bytes() const noexcept override {
        return (_column_major.nonZeros() * value_bytes) +
               ((_column_major.outerSize() + 1) * index_bytes) +
               static_cast<std::int64_t>(_source.size() * sizeof(std::size_t));
    }

  private:
    // Where the analysis moves each row and column.
    [[nodiscard]] std::span<const StorageIndex> new_index() const {
        if constexpr (Cholesky) {
            const auto& indices = _factorization.permutationP().indices();
            return {indices.data(), to_sizet(indices.size())};
        } else {
            const auto& indices = _factorization.colsPermutation().indices();
            return {indices.data(), to_sizet(indices.size())};
        }
    }

    std::string_view _name;
    EigenFactorization _factorization;
    ColumnMajorMatrix _column_major;
    std::vector<std::size_t> _source;
    Eigen::ComputationInfo _info = Eigen::Success;
    bool _factorized = false;
    std::int64_t _off_diagonal = 0;
};

using EigenLuColamd = EigenFactorizer<
    Eigen::SparseLU<ColumnMajorMatrix, Eigen::COLAMDOrdering<int>>, false>;
using EigenLuAmd =
    EigenFactorizer<Eigen::SparseLU<ColumnMajorMatrix, Eigen::AMDOrdering<int>>,
                    false>;
using EigenLdlt =
    EigenFactorizer<Eigen::SimplicialLDLT<ColumnMajorMatrix, Eigen::Upper,
                                          Eigen::AMDOrdering<int>>,
                    true>;

std::unique_ptr<Factorizer> make_eigen(const ResolvedSettings& settings,
                                       bool symmetric) {
    if (symmetric) {
        return std::make_unique<EigenLdlt>("SimplicialLDLT");
    }
    if (settings.eigen.ordering == EigenOrdering::AMD) {
        return std::make_unique<EigenLuAmd>("SparseLU (AMD)");
    }
    return std::make_unique<EigenLuColamd>("SparseLU (COLAMD)");
}

constexpr int iparm_ordering = 1;
constexpr int iparm_iterative = 3;
constexpr int iparm_two_level = 23;

#if PYCANHA_USE_MKL
std::unique_ptr<Factorizer> make_pardiso(const ResolvedSettings& settings,
                                         bool symmetric) {
    const std::vector<IparmEntry> entries = pardiso_iparm(settings, symmetric);
    const auto iterative =
        std::ranges::find(entries, iparm_iterative, &IparmEntry::index);
    auto pardiso = std::make_unique<PardisoSolver>(
        symmetric ? PardisoSolver::MatrixType::SYMMETRIC_POSITIVE_DEFINITE
                  : PardisoSolver::MatrixType::STRUCTURALLY_SYMMETRIC,
        iterative == entries.end() ? 0 : iterative->value);
    for (const IparmEntry& entry : entries) {
        if (entry.index == iparm_iterative) {
            continue;  // passed to the constructor
        }
        if (entry.index < 0 ||
            std::cmp_greater_equal(entry.index, PardisoSolver::iparm_size)) {
            throw std::invalid_argument(
                std::format("PARDISO iparm index {} is out of range (0 to {})",
                            entry.index, PardisoSolver::iparm_size - 1));
        }
        pardiso->set_iparm(to_sizet(entry.index), entry.value);
    }
    pardiso->apply_live_settings(settings);
    return pardiso;
}
#endif

}  // namespace

void Factorizer::factorize_and_solve(const SparseMatrix& matrix,
                                     Eigen::VectorXd& b, Eigen::VectorXd& x) {
    factorize(matrix);
    if (succeeded()) {
        solve(matrix, b, x);
    }
}

std::unique_ptr<Factorizer> make_factorizer(const ResolvedSettings& settings,
                                            bool symmetric) {
    if (settings.backend == Backend::EIGEN) {
        return make_eigen(settings, symmetric);
    }
    if (settings.backend == Backend::ACCELERATE) {
#if PYCANHA_USE_ACCELERATE
        return make_accelerate_factorizer(settings.accelerate.ordering,
                                          symmetric);
#else
        throw std::invalid_argument(
            "backend = ACCELERATE needs a macOS build with Accelerate; use "
            "backend = EIGEN");
#endif
    }
#if PYCANHA_USE_MKL
    return make_pardiso(settings, symmetric);
#else
    throw std::invalid_argument(
        "backend = PARDISO needs a build with MKL; use backend = EIGEN");
#endif
}

namespace {

std::string describe_factorization(const ResolvedSettings& settings,
                                   bool symmetric) {
    const std::string_view form = symmetric ? "Cholesky" : "LU";
    if (settings.storage == Storage::DENSE) {
        return std::format(
            "{} dense, {}{}", to_string(settings.backend), form,
            settings.reuse_factors ? ", reusing factors (FGMRES)" : "");
    }
    if (settings.backend == Backend::EIGEN) {
        return std::format(
            "EIGEN {}, {}",
            symmetric ? "AMD" : to_string(settings.eigen.ordering), form);
    }
    if (settings.backend == Backend::ACCELERATE) {
        return std::format("ACCELERATE {}, {}",
                           to_string(settings.accelerate.ordering), form);
    }
    return std::format("PARDISO {} {}{}, {}", to_string(settings.pardiso.level),
                       to_string(settings.pardiso.ordering),
                       settings.reuse_factors ? ", reusing factors" : "", form);
}

}  // namespace

std::string describe(const ResolvedSettings& settings, bool symmetric) {
    return describe_factorization(settings, symmetric) +
           (settings.fixed_couplings ? ", fixed couplings" : "");
}

int pardiso_matrix_type(bool symmetric) noexcept { return symmetric ? 2 : 1; }

int reuse_digits(double reuse_tolerance) noexcept {
    // The decade at or below the tolerance; the margin keeps 1e-4 at 4
    // despite its rounding.
    constexpr double margin = 1e-9;
    constexpr int fewest = 1;
    constexpr int most = 15;
    if (reuse_tolerance <= 0.0) {
        return most;
    }
    const auto digits =
        static_cast<int>(std::ceil(-std::log10(reuse_tolerance) - margin));
    return std::clamp(digits, fewest, most);
}

int pardiso_iterative_control(int iparm_3, bool symmetric) noexcept {
    constexpr int decimal = 10;
    const int method = iparm_3 % decimal;
    if (method == 0) {
        return iparm_3;
    }
    const int method_for_type = symmetric ? 2 : 1;
    return (decimal * (iparm_3 / decimal)) + method_for_type;
}

std::vector<IparmEntry> pardiso_iparm(const ResolvedSettings& settings,
                                      bool symmetric) {
    constexpr int parallel_nested_dissection = 3;
    constexpr int minimum_degree = 0;

    const bool min_degree =
        settings.pardiso.ordering == PardisoOrdering::MIN_DEGREE;
    const bool two_level = settings.pardiso.level == PardisoLevel::TWO_LEVEL;
    std::map<int, IparmEntry> entries;
    entries[iparm_ordering] = {
        .index = iparm_ordering,
        .value = min_degree ? minimum_degree : parallel_nested_dissection,
        .meaning = min_degree ? "minimum degree ordering"
                              : "parallel nested dissection ordering"};
    // iparm[3] = 10 L + 1: iterate on the previous factors (CGS, or CG for
    // Cholesky) down to a relative residual of 1e-L before refactorizing.
    const int digits = reuse_digits(settings.reuse_tolerance);
    constexpr int decimal = 10;
    entries[iparm_iterative] = {
        .index = iparm_iterative,
        .value = settings.reuse_factors ? pardiso_iterative_control(
                                              (decimal * digits) + 1, symmetric)
                                        : 0,
        .meaning = settings.reuse_factors
                       ? std::format("iterate on the previous factors down "
                                     "to a relative residual of 1e-{} "
                                     "(reuse_tolerance)",
                                     digits)
                       : "no iteration on the previous factors"};
    entries[iparm_two_level] = {.index = iparm_two_level,
                                .value = two_level ? 1 : 0,
                                .meaning = two_level
                                               ? "two-level factorization"
                                               : "one-level factorization"};
    for (const auto& [index, value] : settings.pardiso.iparm) {
        entries[index] = {
            .index = index,
            .value = index == iparm_iterative
                         ? pardiso_iterative_control(value, symmetric)
                         : value,
            .meaning = "set by pardiso.iparm"};
    }
    std::vector<IparmEntry> result;
    result.reserve(entries.size());
    for (auto& [index, entry] : entries) {
        result.push_back(std::move(entry));
    }
    return result;
}

std::optional<std::string> min_degree_hint(const Factorizer& factorizer,
                                           const ResolvedSettings& settings,
                                           bool symmetric, Eigen::Index size) {
    if (settings.backend != Backend::PARDISO ||
        settings.pardiso.ordering == PardisoOrdering::MIN_DEGREE || size <= 0) {
        return std::nullopt;
    }
    // A factor of more than 5e7 non-zeros filling more than a quarter of the
    // dense matrix: many radiative couplings per node, where the minimum
    // degree ordering seems to factorize much faster than nested dissection.
    constexpr double large_factor = 5.0e7;
    constexpr double near_dense = 0.25;
    const auto factor = static_cast<double>(factorizer.factor_nonzeros());
    const auto n = static_cast<double>(size);
    const double dense = symmetric ? n * (n + 1.0) / 2.0 : n * n;
    const double fraction = factor / dense;
    if (factor > large_factor && fraction > near_dense) {
        return std::format(
            "the factor will have {:.3g} non-zeros, {:.0f} % of a dense "
            "matrix. For such nearly dense models (many radiative couplings "
            "per node) pardiso.ordering = MIN_DEGREE can factorize several "
            "times faster.",
            factor, 100.0 * fraction);
    }
    return std::nullopt;
}

}  // namespace pycanha::solvers::detail
