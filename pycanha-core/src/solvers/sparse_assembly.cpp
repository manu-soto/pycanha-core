#include "sparse_assembly.hpp"

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "linearized_network.hpp"
#include "memory.hpp"
#include "pycanha-core/config.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/utils/SparseUtils.hpp"

namespace pycanha::solvers::detail {

namespace {

// Position in the value array of the stored entry (row, col) of a compressed
// row-major matrix.
int value_index(const SparseMatrix& matrix, Index row, Index col) {
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
std::vector<int> map_values(const SparseMatrix& source,
                            const SparseMatrix& target, bool transposed) {
    std::vector<int> indices;
    indices.reserve(to_sizet(source.nonZeros()));
    for (Index row = 0; row < source.outerSize(); ++row) {
        for (SparseMatrix::InnerIterator it(source, row); it; ++it) {
            indices.push_back(transposed ? value_index(target, it.col(), row)
                                         : value_index(target, row, it.col()));
        }
    }
    return indices;
}

std::span<const double> values_of(const SparseMatrix& matrix) {
    return {matrix.valuePtr(), to_sizet(matrix.nonZeros())};
}

std::span<double> mutable_values_of(SparseMatrix& matrix) {
    return {matrix.valuePtr(), to_sizet(matrix.nonZeros())};
}

// Where each coupling and each diagonal entry is in the value array of the
// pattern. The lower triangle is empty in the Cholesky form, and so are the
// radiative positions without radiation.
struct ValuePositions {
    std::vector<int> upper_kl;
    std::vector<int> lower_kl;
    std::vector<int> upper_kr;
    std::vector<int> lower_kr;
    std::vector<int> diagonal;

    [[nodiscard]] std::int64_t bytes() const noexcept {
        return bytes_of(upper_kl) + bytes_of(lower_kl) + bytes_of(upper_kr) +
               bytes_of(lower_kr) + bytes_of(diagonal);
    }
};

ValuePositions value_positions(const SparseMatrix& pattern,
                               const NetworkCouplings& couplings,
                               const NetworkOptions& options) {
    const SparseMatrix& conductive = *couplings.conductive;
    ValuePositions positions;
    positions.upper_kl = map_values(conductive, pattern, /*transposed=*/false);
    if (!options.symmetric) {
        positions.lower_kl =
            map_values(conductive, pattern, /*transposed=*/true);
        if (options.radiation) {
            positions.upper_kr = map_values(*couplings.radiative, pattern,
                                            /*transposed=*/false);
            positions.lower_kr = map_values(*couplings.radiative, pattern,
                                            /*transposed=*/true);
        }
    }
    positions.diagonal.resize(to_sizet(pattern.rows()));
    for (Index row = 0; row < pattern.rows(); ++row) {
        positions.diagonal[to_sizet(row)] = value_index(pattern, row, row);
    }
    return positions;
}

// The sums of the couplings of each row, boundary couplings included.
Eigen::VectorXd row_sums(const SparseMatrix& domain,
                         const SparseMatrix& boundary) {
    Eigen::VectorXd sums = domain.selfadjointView<Eigen::Upper>() *
                           Eigen::VectorXd::Ones(domain.cols());
    sums.noalias() += boundary * Eigen::VectorXd::Ones(boundary.cols());
    return sums;
}

void add_at(std::span<double> values, const std::vector<int>& positions,
            const Eigen::VectorXd& added) {
    for (std::size_t row = 0; row < positions.size(); ++row) {
        values[to_sizet(positions[row])] += added[to_idx(row)];
    }
}

// The couplings may change at any time (formulas, callbacks): every
// assemble() reads them again and writes the values through the position
// tables, and counts as a change.
class VaryingSparseAssembly final : public SparseAssembly {
  public:
    VaryingSparseAssembly(const SparseMatrix& pattern,
                          const NetworkCouplings& couplings,
                          const NetworkOptions& options)
        : _couplings(couplings),
          _options(options),
          _positions(value_positions(pattern, couplings, options)) {
        const Index nd = pattern.rows();
        _row_sums = Eigen::VectorXd::Zero(nd);
        _ones_domain = Eigen::VectorXd::Ones(nd);
        _ones_boundary =
            Eigen::VectorXd::Ones(couplings.conductive_boundary->cols());
    }

    bool assemble(SparseMatrix& matrix, const Eigen::VectorXd& t3,
                  const Eigen::VectorXd& t, const Eigen::VectorXd& diagonal,
                  Eigen::VectorXd& radiation) override {
        sparse_utils::set_to_zero(matrix);
        const std::span<double> values = mutable_values_of(matrix);
        if (_options.radiation) {
            const auto radiative = values_of(*_couplings.radiative);
            for (std::size_t k = 0; k < radiative.size(); ++k) {
                values[to_sizet(_positions.upper_kr[k])] = -radiative[k];
                values[to_sizet(_positions.lower_kr[k])] = -radiative[k];
            }
            _row_sums.noalias() =
                _couplings.radiative->selfadjointView<Eigen::Upper>() *
                _ones_domain;
            _row_sums.noalias() +=
                *_couplings.radiative_boundary * _ones_boundary;
            for (std::size_t row = 0; row < _positions.diagonal.size(); ++row) {
                values[to_sizet(_positions.diagonal[row])] =
                    _row_sums[to_idx(row)];
            }
            // Newton linearisation of sigma T^4: column j carries
            // 4 sigma Tj^3.
            const Eigen::Map<const Eigen::VectorXi> columns(
                matrix.innerIndexPtr(), matrix.nonZeros());
            matrix.coeffs() *= t3(columns).array();
            radiation.noalias() = matrix * t;
            radiation *= 0.75;
        } else {
            radiation.setZero();
        }

        const auto conductive = values_of(*_couplings.conductive);
        for (std::size_t k = 0; k < conductive.size(); ++k) {
            values[to_sizet(_positions.upper_kl[k])] -= conductive[k];
        }
        for (std::size_t k = 0; k < _positions.lower_kl.size(); ++k) {
            values[to_sizet(_positions.lower_kl[k])] -= conductive[k];
        }
        _row_sums.noalias() =
            _couplings.conductive->selfadjointView<Eigen::Upper>() *
            _ones_domain;
        _row_sums.noalias() += *_couplings.conductive_boundary * _ones_boundary;
        add_at(values, _positions.diagonal, _row_sums);
        if (diagonal.size() != 0) {
            add_at(values, _positions.diagonal, diagonal);
        }
        return true;
    }

    [[nodiscard]] std::int64_t bytes() const noexcept override {
        return _positions.bytes() + bytes_of(_row_sums) +
               bytes_of(_ones_domain) + bytes_of(_ones_boundary);
    }

  private:
    NetworkCouplings _couplings;
    NetworkOptions _options;
    ValuePositions _positions;
    Eigen::VectorXd _row_sums;
    Eigen::VectorXd _ones_domain;
    Eigen::VectorXd _ones_boundary;
};

// fixed_couplings: the couplings and the diagonal are trusted to stay as they
// are at the first assemble() of a solve(). The values are then
// base + g * t3[col], with base (conduction and diagonal) and g (radiation)
// built once in the order of the value array. Without radiation the matrix
// is written once per solve().
class FixedSparseAssembly final : public SparseAssembly {
  public:
    FixedSparseAssembly(const SparseMatrix& pattern,
                        const NetworkCouplings& couplings,
                        const NetworkOptions& options)
        : _couplings(couplings),
          _options(options),
          _positions(value_positions(pattern, couplings, options)),
          _base(to_sizet(pattern.nonZeros()), 0.0) {
        if (options.radiation) {
            _g.assign(to_sizet(pattern.nonZeros()), 0.0);
        }
    }

    void begin_solve() noexcept override { _built = false; }

    bool assemble(SparseMatrix& matrix, const Eigen::VectorXd& t3,
                  const Eigen::VectorXd& t, const Eigen::VectorXd& diagonal,
                  Eigen::VectorXd& radiation) override {
        if (!_built) {
            build(diagonal);
        }
        const std::span<double> values = mutable_values_of(matrix);
        if (!_options.radiation) {
            radiation.setZero();
            if (_written) {
                return false;
            }
            std::ranges::copy(_base, values.begin());
            _written = true;
            return true;
        }
        if (_written && t3 == _last_t3) {
            radiation = _last_radiation;
            return false;
        }
        _t3_t = t3.cwiseProduct(t);
        const std::span<const int> outer(matrix.outerIndexPtr(),
                                         to_sizet(matrix.rows() + 1));
        const std::span<const int> inner(matrix.innerIndexPtr(), values.size());
        for (Index row = 0; row < matrix.rows(); ++row) {
            double sum = 0.0;
            for (int k = outer[to_sizet(row)]; k < outer[to_sizet(row) + 1];
                 ++k) {
                const auto col = to_idx(inner[to_sizet(k)]);
                const double g = _g[to_sizet(k)];
                values[to_sizet(k)] = _base[to_sizet(k)] + (g * t3[col]);
                sum += g * _t3_t[col];
            }
            radiation[row] = 0.75 * sum;
        }
        _last_t3 = t3;
        _last_radiation = radiation;
        _written = true;
        return true;
    }

    [[nodiscard]] std::int64_t bytes() const noexcept override {
        return _positions.bytes() + bytes_of(_base) + bytes_of(_g) +
               bytes_of(_t3_t) + bytes_of(_last_t3) + bytes_of(_last_radiation);
    }

  private:
    // base and g from the couplings and the diagonal of this solve().
    void build(const Eigen::VectorXd& diagonal) {
        const std::span<double> base(_base);
        std::ranges::fill(base, 0.0);
        const auto conductive = values_of(*_couplings.conductive);
        for (std::size_t k = 0; k < conductive.size(); ++k) {
            base[to_sizet(_positions.upper_kl[k])] -= conductive[k];
        }
        for (std::size_t k = 0; k < _positions.lower_kl.size(); ++k) {
            base[to_sizet(_positions.lower_kl[k])] -= conductive[k];
        }
        add_at(
            base, _positions.diagonal,
            row_sums(*_couplings.conductive, *_couplings.conductive_boundary));
        if (diagonal.size() != 0) {
            add_at(base, _positions.diagonal, diagonal);
        }
        if (_options.radiation) {
            const std::span<double> g(_g);
            std::ranges::fill(g, 0.0);
            const auto radiative = values_of(*_couplings.radiative);
            for (std::size_t k = 0; k < radiative.size(); ++k) {
                g[to_sizet(_positions.upper_kr[k])] = -radiative[k];
                g[to_sizet(_positions.lower_kr[k])] = -radiative[k];
            }
            // Assigned, like the varying assembly: a radiative coupling of
            // a node with itself does not reach the matrix.
            const Eigen::VectorXd sums =
                row_sums(*_couplings.radiative, *_couplings.radiative_boundary);
            for (std::size_t row = 0; row < _positions.diagonal.size(); ++row) {
                g[to_sizet(_positions.diagonal[row])] = sums[to_idx(row)];
            }
        }
        _built = true;
        _written = false;
    }

    NetworkCouplings _couplings;
    NetworkOptions _options;
    ValuePositions _positions;
    std::vector<double> _base;
    std::vector<double> _g;
    Eigen::VectorXd _t3_t;
    // The temperatures and radiation term of the values written last.
    Eigen::VectorXd _last_t3;
    Eigen::VectorXd _last_radiation;
    bool _built = false;
    bool _written = false;
};

}  // namespace

SparseMatrix sparse_pattern(const NetworkCouplings& couplings,
                            const NetworkOptions& options) {
    const SparseMatrix& conductive = *couplings.conductive;
    SparseMatrix pattern;
    if (options.symmetric) {
        // Upper triangle with the diagonal: the conductive couplings are
        // already stored that way.
        pattern = conductive;
    } else {
        pattern = conductive.selfadjointView<Eigen::Upper>();
        if (options.radiation) {
            pattern += couplings.radiative->selfadjointView<Eigen::Upper>();
        }
    }
    sparse_utils::add_zero_diag_square(pattern);
    sparse_utils::set_to_zero(pattern);
    return pattern;
}

std::unique_ptr<SparseAssembly> make_sparse_assembly(
    const SparseMatrix& pattern, const NetworkCouplings& couplings,
    const NetworkOptions& options) {
    if (options.fixed_couplings) {
        return std::make_unique<FixedSparseAssembly>(pattern, couplings,
                                                     options);
    }
    return std::make_unique<VaryingSparseAssembly>(pattern, couplings, options);
}

}  // namespace pycanha::solvers::detail
