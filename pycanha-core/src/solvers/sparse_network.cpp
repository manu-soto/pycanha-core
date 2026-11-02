#include <Eigen/Core>
#include <Eigen/Sparse>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "factorizer.hpp"
#include "linearized_network.hpp"
#include "memory.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/solvers/settings.hpp"
#include "sparse_assembly.hpp"

namespace pycanha::solvers::detail {

namespace {

// The matrix in compressed row-major storage, with a pattern fixed at
// construction: both triangles, or the upper one in the Cholesky form. The
// assembly writes the values in place and says when they changed; the
// factorizer reads them without a copy.
class SparseNetwork final : public LinearizedNetwork {
  public:
    SparseNetwork(const ResolvedSettings& settings,
                  const NetworkCouplings& couplings,
                  const NetworkOptions& options)
        : _options(options),
          _compare(options.skip_unchanged && !options.fixed_couplings),
          _factorizer(make_factorizer(settings, options.symmetric)),
          _matrix(sparse_pattern(couplings, options)),
          _assembly(make_sparse_assembly(_matrix, couplings, options)) {}

    bool analyze() override {
        _factorizer->analyze_pattern(_matrix);
        _has_factors = false;
        return _factorizer->succeeded();
    }

    void allocate() override {}

    void begin_solve() override { _assembly->begin_solve(); }

    void linearize(const Eigen::VectorXd& t3, const Eigen::VectorXd& t,
                   const Eigen::VectorXd& diagonal,
                   Eigen::VectorXd& radiation) override {
        if (_assembly->assemble(_matrix, t3, t, diagonal, radiation)) {
            _factors_current = false;
        }
    }

    void multiply(const Eigen::VectorXd& x, Eigen::VectorXd& y) const override {
        if (_options.symmetric) {
            y.noalias() = _matrix.selfadjointView<Eigen::Upper>() * x;
        } else {
            y.noalias() = _matrix * x;
        }
    }

    void multiply(const Eigen::MatrixXd& x, Eigen::MatrixXd& y) const override {
        if (_options.symmetric) {
            y.noalias() = _matrix.selfadjointView<Eigen::Upper>() * x;
        } else {
            y.noalias() = _matrix * x;
        }
    }

    SolveOutcome solve(Eigen::VectorXd& b, Eigen::VectorXd& x) override {
        auto& factorizer = *_factorizer;
        SolveOutcome outcome{.succeeded = false, .factorized = true};
        if (_has_factors && (_factors_current || equals_factorized())) {
            factorizer.solve(_matrix, b, x);
            outcome.factorized = false;
        } else if (_has_factors && factorizer.reuses_factors()) {
            factorizer.factorize_and_solve(_matrix, b, x);
            outcome.factorized = factorizer.factors_recomputed();
        } else {
            factorizer.factorize(_matrix);
            if (factorizer.succeeded()) {
                factorizer.solve(_matrix, b, x);
            }
        }
        outcome.succeeded = factorizer.succeeded();
        if (!outcome.succeeded) {
            _has_factors = false;
        } else if (outcome.factorized) {
            _has_factors = true;
            _factors_current = true;
            if (_compare) {
                const std::span<const double> values(
                    _matrix.valuePtr(), to_sizet(_matrix.nonZeros()));
                _factorized_values.assign(values.begin(), values.end());
            }
        }
        return outcome;
    }

    bool solve(Eigen::MatrixXd& b, Eigen::MatrixXd& x) override {
        _factorizer->solve(_matrix, b, x);
        return _factorizer->succeeded();
    }

    void discard_factors() noexcept override { _has_factors = false; }

    void apply_live_settings(const ResolvedSettings& settings) override {
        _factorizer->apply_live_settings(settings);
    }

    [[nodiscard]] bool zero_pivot() const noexcept override {
        return _factorizer->zero_pivot();
    }
    [[nodiscard]] std::string error_message() const override {
        return _factorizer->error_message();
    }
    [[nodiscard]] Index perturbed_pivots() const noexcept override {
        return _factorizer->perturbed_pivots();
    }
    [[nodiscard]] std::optional<std::string> hint(
        const ResolvedSettings& settings) const override {
        return min_degree_hint(*_factorizer, settings, _options.symmetric,
                               _matrix.rows());
    }

    [[nodiscard]] std::int64_t factor_bytes() const noexcept override {
        return _factorizer->factor_bytes();
    }
    [[nodiscard]] bool factor_bytes_exact() const noexcept override {
        return _factorizer->factor_bytes_exact();
    }
    [[nodiscard]] std::int64_t matrix_bytes() const noexcept override {
        return _factorizer->matrix_copy_bytes() + bytes_of(_matrix) +
               bytes_of(_factorized_values) + _assembly->bytes();
    }

  private:
    // skip_unchanged: the matrix is the one last factorized.
    [[nodiscard]] bool equals_factorized() const {
        const std::span<const double> values(_matrix.valuePtr(),
                                             to_sizet(_matrix.nonZeros()));
        return _compare && std::ranges::equal(values, _factorized_values);
    }

    NetworkOptions _options;
    // skip_unchanged without fixed_couplings (a fixed assembly says itself
    // when the matrix changes).
    bool _compare;
    std::unique_ptr<Factorizer> _factorizer;
    SparseMatrix _matrix;
    std::unique_ptr<SparseAssembly> _assembly;
    // The values last factorized, with _compare.
    std::vector<double> _factorized_values;
    bool _has_factors = false;
    // The factors belong to the matrix as it is now (not to an earlier one).
    bool _factors_current = false;
};

}  // namespace

std::unique_ptr<LinearizedNetwork> make_sparse_network(
    const ResolvedSettings& settings, const NetworkCouplings& couplings,
    const NetworkOptions& options) {
    return std::make_unique<SparseNetwork>(settings, couplings, options);
}

std::unique_ptr<LinearizedNetwork> make_network(
    const ResolvedSettings& settings, const NetworkCouplings& couplings,
    const NetworkOptions& options) {
    if (settings.storage == Storage::DENSE) {
        return make_dense_network(settings, couplings, options);
    }
    return make_sparse_network(settings, couplings, options);
}

}  // namespace pycanha::solvers::detail
