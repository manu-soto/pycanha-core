#include "accelerate.hpp"

#include "pycanha-core/solvers/settings.hpp"

#if PYCANHA_USE_ACCELERATE

#include <spdlog/spdlog.h>
#include <vecLib/Sparse/Solve.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "factorizer.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/utils/logger.hpp"

#endif

namespace pycanha::solvers {

bool accelerate_available() noexcept {
#if PYCANHA_USE_ACCELERATE
    // __builtin_available is only allowed as an if condition.
    if (__builtin_available(macOS 15.5, *)) {
        return true;  // NOLINT(readability-simplify-boolean-expr)
    }
#endif
    return false;
}

}  // namespace pycanha::solvers

#if PYCANHA_USE_ACCELERATE

namespace pycanha::solvers::detail {

namespace {

using SparseMatrix = Factorizer::SparseMatrix;

// Without a reportError callback Accelerate traps on a parameter error. With
// it, the error is logged and returned in the status of the factorization.
void report_error(const char* message) {
    SPDLOG_LOGGER_ERROR(get_logger(), "Accelerate: {}", message);
}

[[nodiscard]] std::string_view to_string(SparseStatus_t status) noexcept {
    switch (status) {
        case SparseStatusOK:
            return "OK";
        case SparseFactorizationFailed:
            return "factorization failed";
        case SparseMatrixIsSingular:
            return "matrix is singular";
        case SparseInternalError:
            return "internal error";
        case SparseParameterError:
            return "parameter error";
        case SparseStatusReleased:
            return "factorization released";
    }
    return "unknown status";
}

class AccelerateFactorizer final : public Factorizer {
  public:
    AccelerateFactorizer(SparseFactorization_t factorization,
                         SparseOrder_t order, bool symmetric, std::string name)
        : _factorization(factorization),
          _order(order),
          _symmetric(symmetric),
          _name(std::move(name)) {}

    ~AccelerateFactorizer() override { release(); }

    AccelerateFactorizer(const AccelerateFactorizer&) = delete;
    AccelerateFactorizer& operator=(const AccelerateFactorizer&) = delete;
    AccelerateFactorizer(AccelerateFactorizer&&) = delete;
    AccelerateFactorizer& operator=(AccelerateFactorizer&&) = delete;

    void analyze_pattern(const SparseMatrix& matrix) override {
        if (!matrix.isCompressed() || matrix.rows() != matrix.cols()) {
            throw std::invalid_argument(
                "Accelerate needs a square, compressed matrix");
        }
        release();
        _size = static_cast<int>(matrix.rows());
        const std::span<const int> starts(matrix.outerIndexPtr(),
                                          to_sizet(matrix.rows()) + 1);
        const std::span<const int> indices(matrix.innerIndexPtr(),
                                           to_sizet(matrix.nonZeros()));
        _column_starts.assign(starts.begin(), starts.end());
        _row_indices.assign(indices.begin(), indices.end());

        SparseSymbolicFactorOptions options{};
        options.control = SparseDefaultControl;
        options.orderMethod = _order;
        options.order = nullptr;
        options.ignoreRowsAndColumns = nullptr;
        options.malloc = std::malloc;
        options.free = std::free;
        options.reportError = report_error;
        _symbolic = SparseFactor(_factorization, structure(), options);
        _status = _symbolic.status;
        _analyzed = _status == SparseStatusOK;
    }

    void factorize(const SparseMatrix& matrix) override {
        if (!_analyzed) {
            _status = SparseParameterError;
            return;
        }
        if (to_sizet(matrix.nonZeros()) != _row_indices.size()) {
            throw std::invalid_argument(
                "Accelerate: the matrix pattern changed since "
                "analyze_pattern()");
        }
        // Accelerate only reads the values.
        const SparseMatrix_Double values = {
            .structure = structure(),
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
            .data = const_cast<double*>(matrix.valuePtr())};
        if (_factored) {
            SparseRefactor(values, &_numeric);
        } else {
            _numeric = SparseFactor(_symbolic, values);
            _factored = true;
        }
        _status = _numeric.status;
        if (_status != SparseStatusOK) {
            // The next factorize() starts from the symbolic analysis again.
            SparseCleanup(_numeric);
            _factored = false;
            return;
        }
        if (!_symmetric && _size > 0) {
            // Infinity norm, for the backward error of the solves.
            _matrix_norm =
                (matrix.cwiseAbs() * Eigen::VectorXd::Ones(matrix.cols()))
                    .maxCoeff();
        }
    }

    void solve(const SparseMatrix& matrix, Eigen::VectorXd& b,
               Eigen::VectorXd& x) override {
        if (!_factored) {
            _status = SparseFactorizationFailed;
            return;
        }
        x = b;
        const DenseVector_Double xb = {.count = _size, .data = x.data()};
        SparseSolve(_numeric, xb);
        check_backward_error(matrix, b, x);
    }

    void solve(const SparseMatrix& matrix, Eigen::MatrixXd& b,
               Eigen::MatrixXd& x) override {
        if (!_factored) {
            _status = SparseFactorizationFailed;
            return;
        }
        x = b;
        const DenseMatrix_Double xb = {
            .rowCount = _size,
            .columnCount = static_cast<int>(x.cols()),
            .columnStride = static_cast<int>(x.outerStride()),
            .attributes = {},
            .data = x.data()};
        SparseSolve(_numeric, xb);
        check_backward_error(matrix, b, x);
    }

    [[nodiscard]] bool succeeded() const noexcept override {
        return _status == SparseStatusOK;
    }
    [[nodiscard]] bool zero_pivot() const noexcept override {
        return _status == SparseMatrixIsSingular;
    }
    [[nodiscard]] std::string error_message() const override {
        return std::format("Accelerate {} failed ({})", _name,
                           to_string(_status));
    }

    // The factor storage and the factorization workspace the symbolic
    // analysis asks for.
    [[nodiscard]] std::int64_t factor_bytes() const noexcept override {
        if (!_analyzed) {
            return 0;
        }
        return static_cast<std::int64_t>(_symbolic.factorSize_Double +
                                         _symbolic.workspaceSize_Double);
    }
    // Cholesky does not pivot, so the analysis fixes its factor. The LU
    // pivots during the factorization, which may change it.
    [[nodiscard]] bool factor_bytes_exact() const noexcept override {
        return _symmetric;
    }
    // The copied index arrays.
    [[nodiscard]] std::int64_t matrix_copy_bytes() const noexcept override {
        return static_cast<std::int64_t>(
            (_column_starts.size() * sizeof(_column_starts.front())) +
            (_row_indices.size() * sizeof(_row_indices.front())));
    }

  private:
    SparseFactorization_t _factorization;
    SparseOrder_t _order;
    bool _symmetric;
    std::string _name;
    int _size = 0;
    // Accelerate's SparseMatrixStructure takes long column starts (not
    // int64_t, which is long long on macOS).
    std::vector<long> _column_starts;  // NOLINT(runtime/int)
    std::vector<int> _row_indices;
    SparseOpaqueSymbolicFactorization _symbolic{};
    SparseOpaqueFactorization_Double _numeric{};
    bool _analyzed = false;
    bool _factored = false;
    SparseStatus_t _status = SparseStatusOK;
    double _matrix_norm = 0.0;

    // Accelerate's LU does not report a singular matrix: with a zero pivot
    // it returns a status OK and a wrong solution. A backward error
    // |A x - b| / (|A| |x| + |b|) far above rounding marks it as singular.
    // Cholesky reports it itself.
    template <typename Dense>
    void check_backward_error(const SparseMatrix& matrix, const Dense& b,
                              const Dense& x) {
        constexpr double tolerance = 1e-8;
        if (_symmetric || x.size() == 0) {
            return;
        }
        const Dense residual = (matrix * x) - b;
        const double error = residual.cwiseAbs().maxCoeff();
        const double scale =
            (_matrix_norm * x.cwiseAbs().maxCoeff()) + b.cwiseAbs().maxCoeff();
        if (!std::isfinite(error) || error > tolerance * scale) {
            _status = SparseMatrixIsSingular;
        }
    }

    // The CSR arrays seen as CSC. LU: the transpose of the matrix, flagged
    // so. Cholesky: the upper triangle in rows is the lower one in columns.
    [[nodiscard]] SparseMatrixStructure structure() noexcept {
        SparseAttributes_t attributes{};
        attributes.transpose = !_symmetric;
        attributes.triangle = SparseLowerTriangle;
        // kind is a 2-bit field: SparseOrdinary (0) and SparseSymmetric (3)
        // fit, other SparseKind_t values do not, hence the warning.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wbitfield-enum-conversion"
        attributes.kind = _symmetric ? SparseSymmetric : SparseOrdinary;
#pragma clang diagnostic pop
        return {.rowCount = _size,
                .columnCount = _size,
                .columnStarts = _column_starts.data(),
                .rowIndices = _row_indices.data(),
                .attributes = attributes,
                .blockSize = 1};
    }

    void release() noexcept {
        if (_factored) {
            SparseCleanup(_numeric);
            _factored = false;
        }
        if (_analyzed) {
            SparseCleanup(_symbolic);
            _analyzed = false;
        }
    }
};

}  // namespace

std::unique_ptr<Factorizer> make_accelerate_factorizer(
    AccelerateOrdering ordering, bool symmetric) {
    const SparseOrder_t order = ordering == AccelerateOrdering::METIS
                                    ? SparseOrderMetis
                                    : SparseOrderAMD;
    std::string name = std::format("{} ({})", symmetric ? "Cholesky" : "LU",
                                   to_string(ordering));
    if (symmetric) {
        return std::make_unique<AccelerateFactorizer>(
            SparseFactorizationCholesky, order, /*symmetric=*/true,
            std::move(name));
    }
    if (__builtin_available(macOS 15.5, *)) {
        return std::make_unique<AccelerateFactorizer>(
            SparseFactorizationLU, order, /*symmetric=*/false, std::move(name));
    }
    throw std::invalid_argument(
        "Accelerate sparse LU needs macOS 15.5 or newer");
}

}  // namespace pycanha::solvers::detail

#endif  // PYCANHA_USE_ACCELERATE
