#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Sparse>
#include <cstddef>
#include <cstdint>
#include <format>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "dense_assembly.hpp"
#include "linearized_network.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/solvers/settings.hpp"

#if PYCANHA_USE_MKL
#include <mkl_lapack.h>
#include <mkl_rci.h>
#include <mkl_types.h>

#include <array>

#include "thread_scope.hpp"
#else
#include <stdexcept>
#endif

namespace pycanha::solvers::detail {

namespace {

using VectorRef = Eigen::Ref<Eigen::VectorXd>;
using ConstVectorRef = Eigen::Ref<const Eigen::VectorXd>;

// A pivot below this fraction of the largest diagonal entry marks the matrix
// as singular: a group of diffusive nodes without a path to a boundary node
// leaves a pivot of about 1e-16 of the others rather than an exact zero.
// PARDISO perturbs pivots below the same fraction.
constexpr double singular_pivot = 1e-13;

[[nodiscard]] double largest_diagonal(const DenseMatrix& matrix) {
    return matrix.diagonal().cwiseAbs().maxCoeff();
}

[[nodiscard]] bool has_tiny_pivot(double largest_diagonal,
                                  double smallest_pivot) {
    return !(smallest_pivot > singular_pivot * largest_diagonal);
}

// The factors of one matrix, in their own storage.
class DenseFactors {
  public:
    DenseFactors() = default;
    virtual ~DenseFactors() = default;
    DenseFactors(const DenseFactors&) = delete;
    DenseFactors& operator=(const DenseFactors&) = delete;
    DenseFactors(DenseFactors&&) = delete;
    DenseFactors& operator=(DenseFactors&&) = delete;

    virtual void allocate(Index size) = 0;
    // Factorizes the matrix; false when it is singular. Without keep_matrix
    // the factorization may take the matrix's storage instead of copying it
    // (LAPACK does), and the matrix holds anything afterwards.
    [[nodiscard]] virtual bool factorize(DenseMatrix& matrix,
                                         bool keep_matrix) = 0;
    virtual void solve(ConstVectorRef b, VectorRef x) const = 0;
    virtual void solve(const DenseMatrix& b, DenseMatrix& x) const = 0;
    virtual void set_threads([[maybe_unused]] int threads) noexcept {}
    [[nodiscard]] virtual std::int64_t bytes(Index size) const noexcept = 0;
    // "LAPACK dgetrf"
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
};

class EigenLu final : public DenseFactors {
  public:
    // Created with its size, never assigned: the sized constructors leave
    // members that an assignment would read uninitialised.
    void allocate(Index size) override {
        _lu = std::make_unique<Eigen::PartialPivLU<DenseMatrix>>(size);
    }
    bool factorize(DenseMatrix& matrix, bool /*keep_matrix*/) override {
        _lu->compute(matrix);
        return !has_tiny_pivot(
            largest_diagonal(matrix),
            _lu->matrixLU().diagonal().cwiseAbs().minCoeff());
    }
    void solve(ConstVectorRef b, VectorRef x) const override {
        x = _lu->solve(b);
    }
    void solve(const DenseMatrix& b, DenseMatrix& x) const override {
        x = _lu->solve(b);
    }
    // The factors and two permutations of int indices.
    [[nodiscard]] std::int64_t bytes(Index size) const noexcept override {
        return square_bytes(size) + (2 * static_cast<std::int64_t>(size) *
                                     static_cast<std::int64_t>(sizeof(int)));
    }
    [[nodiscard]] std::string_view name() const noexcept override {
        return "Eigen PartialPivLU";
    }

  private:
    std::unique_ptr<Eigen::PartialPivLU<DenseMatrix>> _lu;
};

class EigenLlt final : public DenseFactors {
  public:
    void allocate(Index size) override {
        _llt = std::make_unique<Eigen::LLT<DenseMatrix, Eigen::Lower>>(size);
    }
    bool factorize(DenseMatrix& matrix, bool /*keep_matrix*/) override {
        _llt->compute(matrix);
        if (_llt->info() != Eigen::Success) {
            return false;
        }
        const double smallest =
            _llt->matrixLLT().diagonal().cwiseAbs2().minCoeff();
        return !has_tiny_pivot(largest_diagonal(matrix), smallest);
    }
    void solve(ConstVectorRef b, VectorRef x) const override {
        x = _llt->solve(b);
    }
    void solve(const DenseMatrix& b, DenseMatrix& x) const override {
        x = _llt->solve(b);
    }
    [[nodiscard]] std::int64_t bytes(Index size) const noexcept override {
        return square_bytes(size);
    }
    [[nodiscard]] std::string_view name() const noexcept override {
        return "Eigen LLT";
    }

  private:
    std::unique_ptr<Eigen::LLT<DenseMatrix, Eigen::Lower>> _llt;
};

#if PYCANHA_USE_MKL

// LAPACK factorizes in place: the matrix is copied into the factor storage,
// or, when it is not needed afterwards, the two storages are swapped.
class LapackFactors : public DenseFactors {
  protected:
    void take(DenseMatrix& matrix, bool keep_matrix) {
        if (keep_matrix) {
            _factors = matrix;
        } else {
            _factors.swap(matrix);
        }
    }

    DenseMatrix _factors;
    int _threads = 0;
};

class LapackLu final : public LapackFactors {
  public:
    void allocate(Index size) override {
        _factors.resize(size, size);
        _pivots.resize(to_sizet(size));
    }
    bool factorize(DenseMatrix& matrix, bool keep_matrix) override {
        const double largest = largest_diagonal(matrix);
        take(matrix, keep_matrix);
        const auto size = static_cast<MKL_INT>(_factors.rows());
        MKL_INT info = 0;
        {
            const ThreadScope threads(_threads);
            dgetrf(&size, &size, _factors.data(), &size, _pivots.data(), &info);
        }
        return info == 0 &&
               !has_tiny_pivot(largest,
                               _factors.diagonal().cwiseAbs().minCoeff());
    }
    void solve(ConstVectorRef b, VectorRef x) const override {
        x = b;
        solve_in_place(x.data(), 1);
    }
    void solve(const DenseMatrix& b, DenseMatrix& x) const override {
        x = b;
        solve_in_place(x.data(), static_cast<MKL_INT>(x.cols()));
    }
    void set_threads(int threads) noexcept override { _threads = threads; }
    [[nodiscard]] std::int64_t bytes(Index size) const noexcept override {
        return square_bytes(size) +
               (static_cast<std::int64_t>(size) *
                static_cast<std::int64_t>(sizeof(MKL_INT)));
    }
    [[nodiscard]] std::string_view name() const noexcept override {
        return "LAPACK dgetrf";
    }

  private:
    void solve_in_place(double* x, MKL_INT columns) const {
        const auto size = static_cast<MKL_INT>(_factors.rows());
        const char no_transpose = 'N';
        MKL_INT info = 0;
        const ThreadScope threads(_threads);
        dgetrs(&no_transpose, &size, &columns, _factors.data(), &size,
               _pivots.data(), x, &size, &info);
    }

    std::vector<MKL_INT> _pivots;
};

class LapackCholesky final : public LapackFactors {
  public:
    void allocate(Index size) override { _factors.resize(size, size); }
    bool factorize(DenseMatrix& matrix, bool keep_matrix) override {
        const double largest = largest_diagonal(matrix);
        take(matrix, keep_matrix);
        const auto size = static_cast<MKL_INT>(_factors.rows());
        MKL_INT info = 0;
        {
            const ThreadScope threads(_threads);
            dpotrf(&lower, &size, _factors.data(), &size, &info);
        }
        return info == 0 &&
               !has_tiny_pivot(largest,
                               _factors.diagonal().cwiseAbs2().minCoeff());
    }
    void solve(ConstVectorRef b, VectorRef x) const override {
        x = b;
        solve_in_place(x.data(), 1);
    }
    void solve(const DenseMatrix& b, DenseMatrix& x) const override {
        x = b;
        solve_in_place(x.data(), static_cast<MKL_INT>(x.cols()));
    }
    void set_threads(int threads) noexcept override { _threads = threads; }
    [[nodiscard]] std::int64_t bytes(Index size) const noexcept override {
        return square_bytes(size);
    }
    [[nodiscard]] std::string_view name() const noexcept override {
        return "LAPACK dpotrf";
    }

  private:
    // The lower triangle; the matrix is symmetric, so either one would do.
    static constexpr char lower = 'L';

    void solve_in_place(double* x, MKL_INT columns) const {
        const auto size = static_cast<MKL_INT>(_factors.rows());
        MKL_INT info = 0;
        const ThreadScope threads(_threads);
        dpotrs(&lower, &size, &columns, _factors.data(), &size, x, &size,
               &info);
    }
};

constexpr auto double_bytes = static_cast<std::int64_t>(sizeof(double));

// FGMRES (MKL's reverse-communication solver) preconditioned with the factors
// of an earlier matrix: the dense counterpart of PARDISO's iteration on
// previous factors. It stops at a relative residual of `tolerance`
// (reuse_tolerance) against the initial one, so it never accepts the initial
// guess unchecked.
class FactorReuse {
  public:
    static constexpr MKL_INT max_iterations = 20;

    void allocate(Index size) { _work.resize(to_sizet(work_size(size))); }
    void set_tolerance(double tolerance) noexcept { _tolerance = tolerance; }

    [[nodiscard]] static std::int64_t bytes(Index size) noexcept {
        return work_size(size) * double_bytes;
    }

    // Solves matrix x = b from the x given; false when it does not converge.
    [[nodiscard]] bool solve(const DenseMatrix& matrix,
                             const DenseFactors& factors, Eigen::VectorXd& b,
                             Eigen::VectorXd& x, int threads) {
        const auto size = static_cast<MKL_INT>(matrix.rows());
        const std::span<double> work(_work);
        MKL_INT request = 0;
        const ThreadScope scope(threads);
        dfgmres_init(&size, x.data(), b.data(), &request, _ipar.data(),
                     _dpar.data(), work.data());
        if (request != 0) {
            return false;
        }
        _ipar[maximum_iterations] = max_iterations;
        _ipar[stop_at_maximum] = 1;
        _ipar[residual_test] = 1;
        _ipar[user_test] = 0;
        _ipar[preconditioned] = 1;
        _ipar[zero_norm_test] = 1;
        _ipar[restart] = max_iterations;
        _dpar[relative_tolerance] = _tolerance;
        dfgmres_check(&size, x.data(), b.data(), &request, _ipar.data(),
                      _dpar.data(), work.data());
        if (request == check_failed) {
            return false;
        }
        while (true) {
            dfgmres(&size, x.data(), b.data(), &request, _ipar.data(),
                    _dpar.data(), work.data());
            if (request != multiply_request &&
                request != precondition_request) {
                break;
            }
            const Eigen::Map<const Eigen::VectorXd> input(
                work.subspan(to_sizet(_ipar[input_position] - 1)).data(), size);
            Eigen::Map<Eigen::VectorXd> output(
                work.subspan(to_sizet(_ipar[output_position] - 1)).data(),
                size);
            if (request == multiply_request) {
                output.noalias() = matrix * input;
            } else {
                factors.solve(input, output);
            }
        }
        if (request != 0) {
            return false;
        }
        MKL_INT iterations = 0;
        dfgmres_get(&size, x.data(), b.data(), &request, _ipar.data(),
                    _dpar.data(), work.data(), &iterations);
        return request == 0;
    }

  private:
    // Indices into ipar and dpar (zero-based) and the request codes, from
    // MKL's FGMRES interface.
    static constexpr std::size_t maximum_iterations = 4;
    static constexpr std::size_t stop_at_maximum = 7;
    static constexpr std::size_t residual_test = 8;
    static constexpr std::size_t user_test = 9;
    static constexpr std::size_t preconditioned = 10;
    // FGMRES checks the norm of each new basis vector itself instead of
    // asking for it.
    static constexpr std::size_t zero_norm_test = 11;
    static constexpr std::size_t restart = 14;
    static constexpr std::size_t input_position = 21;
    static constexpr std::size_t output_position = 22;
    static constexpr std::size_t relative_tolerance = 0;
    static constexpr MKL_INT multiply_request = 1;
    static constexpr MKL_INT precondition_request = 3;
    static constexpr MKL_INT check_failed = -1100;

    // The size of tmp that FGMRES needs for a restart length m:
    // (2 m + 1) n + m (m + 9) / 2 + 1.
    [[nodiscard]] static std::int64_t work_size(Index size) noexcept {
        constexpr std::int64_t m = max_iterations;
        return (((2 * m) + 1) * static_cast<std::int64_t>(size)) +
               ((m * (m + 9)) / 2) + 1;
    }

    std::vector<double> _work;
    double _tolerance = 0.0;
    std::array<MKL_INT, 128> _ipar{};
    std::array<double, 128> _dpar{};
};

#endif

std::unique_ptr<DenseFactors> make_factors(const ResolvedSettings& settings,
                                           bool symmetric) {
    if (settings.backend == Backend::LAPACK) {
#if PYCANHA_USE_MKL
        if (symmetric) {
            return std::make_unique<LapackCholesky>();
        }
        return std::make_unique<LapackLu>();
#else
        throw std::invalid_argument(
            "backend = LAPACK needs a build with MKL; use backend = EIGEN");
#endif
    }
    if (symmetric) {
        return std::make_unique<EigenLlt>();
    }
    return std::make_unique<EigenLu>();
}

// The matrix as a dense n x n array, written by the assembly, and its
// factors in a second one. Fast for small models whose nodes are coupled to
// most of the others; the memory grows with n^2.
class DenseNetwork final : public LinearizedNetwork {
  public:
    DenseNetwork(const ResolvedSettings& settings,
                 const NetworkCouplings& couplings,
                 const NetworkOptions& options)
        : _options(options),
          _compare(options.skip_unchanged && !options.fixed_couplings),
          _size(couplings.conductive->rows()),
          _reuse(settings.reuse_factors),
          _factors(make_factors(settings, options.symmetric)),
          _assembly(make_dense_assembly(couplings, options)) {
#if PYCANHA_USE_MKL
        _factor_reuse.set_tolerance(settings.reuse_tolerance);
#endif
        apply_live_settings(settings);
    }

    bool analyze() override { return true; }

    void allocate() override {
        _matrix = DenseMatrix::Zero(_size, _size);
        if (_compare) {
            _factorized.resize(_size, _size);
        }
        _rewrite = false;
        _assembly->allocate();
        _factors->allocate(_size);
#if PYCANHA_USE_MKL
        if (_reuse) {
            _factor_reuse.allocate(_size);
        }
#endif
        _has_factors = false;
    }

    void begin_solve() override { _assembly->begin_solve(); }

    void linearize(const Eigen::VectorXd& t3, const Eigen::VectorXd& t,
                   const Eigen::VectorXd& diagonal,
                   Eigen::VectorXd& radiation) override {
        if (_assembly->assemble(_matrix, _rewrite, t3, t, diagonal,
                                radiation)) {
            _factors_current = false;
        }
        _rewrite = false;
    }

    void multiply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const override {
#if PYCANHA_USE_MKL
        const ThreadScope threads(_threads);
#endif
        y.noalias() = _matrix * x;
    }

    void multiply(const Eigen::MatrixXd& x, Eigen::MatrixXd& y) const override {
#if PYCANHA_USE_MKL
        const ThreadScope threads(_threads);
#endif
        y.noalias() = _matrix * x;
    }

    SolveOutcome solve(Eigen::VectorXd& b, Eigen::VectorXd& x) override {
        if (_has_factors &&
            (_factors_current || (_compare && _matrix == _factorized))) {
            _factors->solve(b, x);
            return {.succeeded = true, .factorized = false};
        }
        if (_has_factors && _reuse && iterate(b, x)) {
            return {.succeeded = true, .factorized = false};
        }
        if (_compare) {
            _factorized = _matrix;
        }
        // The next linearize() writes the matrix again; only keep_matrix
        // needs it until then.
        _has_factors = _factors->factorize(_matrix, _options.keep_matrix);
        _rewrite = !_options.keep_matrix;
        if (!_has_factors) {
            return {.succeeded = false, .factorized = true};
        }
        _factors_current = true;
        _factors->solve(b, x);
        return {.succeeded = true, .factorized = true};
    }

    bool solve(Eigen::MatrixXd& b, Eigen::MatrixXd& x) override {
        if (!_has_factors) {
            return false;
        }
        _factors->solve(b, x);
        return true;
    }

    void discard_factors() noexcept override { _has_factors = false; }

    void apply_live_settings(const ResolvedSettings& settings) override {
        _threads = settings.threads;
        _factors->set_threads(settings.threads);
    }

    // The only failure is a singular matrix.
    [[nodiscard]] bool zero_pivot() const noexcept override { return true; }
    [[nodiscard]] std::string error_message() const override {
        return std::format(
            "{} found a pivot below {} of the largest diagonal entry",
            _factors->name(), singular_pivot);
    }

    [[nodiscard]] std::int64_t factor_bytes() const noexcept override {
        return _factors->bytes(_size);
    }
    [[nodiscard]] bool factor_bytes_exact() const noexcept override {
        return true;
    }
    [[nodiscard]] std::int64_t matrix_bytes() const noexcept override {
        std::int64_t bytes = square_bytes(_size) + _assembly->bytes();
        if (_compare) {
            bytes += square_bytes(_size);
        }
#if PYCANHA_USE_MKL
        if (_reuse) {
            bytes += FactorReuse::bytes(_size);
        }
#endif
        return bytes;
    }

  private:
    // Iterates on the current factors; false when it does not converge.
    bool iterate(Eigen::VectorXd& b, Eigen::VectorXd& x) {
#if PYCANHA_USE_MKL
        return _factor_reuse.solve(_matrix, *_factors, b, x, _threads);
#else
        static_cast<void>(b);
        static_cast<void>(x);
        return false;
#endif
    }

    NetworkOptions _options;
    // skip_unchanged without fixed_couplings (a fixed assembly says itself
    // when the matrix changes).
    bool _compare;
    Index _size;
    bool _reuse;
    std::unique_ptr<DenseFactors> _factors;
    std::unique_ptr<DenseAssembly> _assembly;
    DenseMatrix _matrix;
    // The matrix last factorized, with _compare.
    DenseMatrix _factorized;
#if PYCANHA_USE_MKL
    FactorReuse _factor_reuse;
#endif
    int _threads = 0;
    bool _has_factors = false;
    // The factors belong to the matrix as it is now (not to an earlier one).
    bool _factors_current = false;
    // A factorization took the matrix storage: the next assemble() writes it
    // in full.
    bool _rewrite = false;
};

}  // namespace

std::unique_ptr<LinearizedNetwork> make_dense_network(
    const ResolvedSettings& settings, const NetworkCouplings& couplings,
    const NetworkOptions& options) {
    return std::make_unique<DenseNetwork>(settings, couplings, options);
}

}  // namespace pycanha::solvers::detail
