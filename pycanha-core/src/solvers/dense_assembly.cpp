#include "dense_assembly.hpp"

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "linearized_network.hpp"
#include "pycanha-core/globals.hpp"

namespace pycanha::solvers::detail {

namespace {

constexpr auto double_bytes = static_cast<std::int64_t>(sizeof(double));

// The sums of the couplings of each row, as the product of the stored upper
// triangle with a vector of ones counts them.
void add_to_sums(Eigen::VectorXd& sums, Index row, Index col, double value) {
    sums[row] += value;
    if (col != row) {
        sums[col] += value;
    }
}

void add_boundary_sums(Eigen::VectorXd& sums, const SparseMatrix& boundary) {
    for (Index row = 0; row < boundary.outerSize(); ++row) {
        for (SparseMatrix::InnerIterator it(boundary, row); it; ++it) {
            sums[row] += it.value();
        }
    }
}

// Copies the strict lower triangle to the upper one in square tiles that
// fit the L1 cache: a plain transposed copy would read a new cache line for
// every entry.
void mirror_lower_to_upper(DenseMatrix& matrix) {
    constexpr Index tile = 32;
    const Index size = matrix.rows();
    for (Index col_start = 0; col_start < size; col_start += tile) {
        const Index col_end = std::min(col_start + tile, size);
        for (Index row_start = 0; row_start < col_end; row_start += tile) {
            // Entry (i, j) above the diagonal takes entry (j, i).
            for (Index j = col_start; j < col_end; ++j) {
                const Index row_end = std::min(row_start + tile, j);
                for (Index i = row_start; i < row_end; ++i) {
                    matrix(i, j) = matrix(j, i);
                }
            }
        }
    }
}

// The couplings may change at any time (formulas, callbacks): every
// assemble() copies them again, whether their values changed or not, and
// counts as a change.
class VaryingDenseAssembly final : public DenseAssembly {
  public:
    VaryingDenseAssembly(const NetworkCouplings& couplings,
                         const NetworkOptions& options)
        : _couplings(couplings),
          _options(options),
          _size(couplings.conductive->rows()) {}

    void allocate() override { _row_sums.resize(_size); }

    bool assemble(DenseMatrix& matrix, bool rewrite, const Eigen::VectorXd& t3,
                  const Eigen::VectorXd& t, const Eigen::VectorXd& diagonal,
                  Eigen::VectorXd& radiation) override {
        clear_written_entries(matrix, rewrite);
        _row_sums.setZero();
        radiation.setZero();
        if (_options.radiation) {
            add_radiation(matrix, t3, t, radiation);
        } else {
            matrix.diagonal().setZero();
        }
        for (Index row = 0; row < _size; ++row) {
            for (SparseMatrix::InnerIterator it(*_couplings.conductive, row);
                 it; ++it) {
                matrix(row, it.col()) -= it.value();
                matrix(it.col(), row) -= it.value();
                add_to_sums(_row_sums, row, it.col(), it.value());
            }
        }
        add_boundary_sums(_row_sums, *_couplings.conductive_boundary);
        matrix.diagonal() += _row_sums;
        if (diagonal.size() != 0) {
            matrix.diagonal() += diagonal;
        }
        return true;
    }

    [[nodiscard]] std::int64_t bytes() const noexcept override {
        return static_cast<std::int64_t>(_size) * double_bytes;
    }

  private:
    // The entries outside the couplings and the diagonal stay zero, so only
    // the conductive entries need clearing before they are accumulated: the
    // radiative ones and the diagonal are assigned. A matrix to rewrite is
    // cleared in full.
    void clear_written_entries(DenseMatrix& matrix, bool rewrite) const {
        if (rewrite) {
            matrix.setZero();
            return;
        }
        for (Index row = 0; row < _size; ++row) {
            for (SparseMatrix::InnerIterator it(*_couplings.conductive, row);
                 it; ++it) {
                matrix(row, it.col()) = 0.0;
                matrix(it.col(), row) = 0.0;
            }
        }
    }

    // A_ij = -GR_ij 4 sigma Tj^3 off the diagonal and the row sums times
    // 4 sigma Ti^3 on it (the Newton linearisation of sigma T^4), with
    // 0.75 A t in `radiation`. The stored rows give the lower triangle
    // column by column; the upper one is its mirror until the columns are
    // scaled, which one vectorized pass does with the product.
    void add_radiation(DenseMatrix& matrix, const Eigen::VectorXd& t3,
                       const Eigen::VectorXd& t, Eigen::VectorXd& radiation) {
        const SparseMatrix& radiative = *_couplings.radiative;
        const std::span<const double> values(radiative.valuePtr(),
                                             to_sizet(radiative.nonZeros()));
        const std::span<const int> outer(radiative.outerIndexPtr(),
                                         to_sizet(_size + 1));
        const std::span<const int> inner(radiative.innerIndexPtr(),
                                         values.size());
        add_boundary_sums(_row_sums, *_couplings.radiative_boundary);
        // Spans and a local sum: through the members the compiler reloads
        // the sums after every write to the matrix.
        const std::span<double> sums(_row_sums.data(), to_sizet(_size));
        for (Index col = 0; col < _size; ++col) {
            const std::span<double> column(matrix.col(col).data(),
                                           to_sizet(_size));
            double sum = 0.0;
            for (int k = outer[to_sizet(col)]; k < outer[to_sizet(col) + 1];
                 ++k) {
                const auto row = to_sizet(inner[to_sizet(k)]);
                const double coupling = values[to_sizet(k)];
                sum += coupling;
                if (row != to_sizet(col)) {
                    sums[row] += coupling;
                    column[row] = -coupling;
                }
            }
            sums[to_sizet(col)] += sum;
        }
        mirror_lower_to_upper(matrix);
        matrix.diagonal() = _row_sums;
        for (Index col = 0; col < _size; ++col) {
            auto column = matrix.col(col);
            column *= t3[col];
            radiation.noalias() += column * t[col];
        }
        radiation *= 0.75;
        _row_sums.setZero();
    }

    NetworkCouplings _couplings;
    NetworkOptions _options;
    Index _size;
    Eigen::VectorXd _row_sums;
};

// fixed_couplings: the couplings and the diagonal are trusted to stay as they
// are at the first assemble() of a solve(). The radiation G (-GR off the
// diagonal, the row sums on it) is built once as an n x n array and the
// conduction as per-column lists of both triangles plus a diagonal (row sums
// and D). assemble() then writes every column once: G(:, j) 4 sigma Tj^3 and
// the conduction of column j. Without radiation the matrix is written once
// per solve().
class FixedDenseAssembly final : public DenseAssembly {
  public:
    FixedDenseAssembly(const NetworkCouplings& couplings,
                       const NetworkOptions& options)
        : _couplings(couplings),
          _options(options),
          _size(couplings.conductive->rows()) {
        for (Index row = 0; row < _size; ++row) {
            for (SparseMatrix::InnerIterator it(*couplings.conductive, row); it;
                 ++it) {
                _conduction_entries += it.col() != row ? 2 : 0;
            }
        }
    }

    void allocate() override {
        if (_options.radiation) {
            _g.resize(_size, _size);
        }
        _built = false;
    }

    void begin_solve() noexcept override { _built = false; }

    bool assemble(DenseMatrix& matrix, bool rewrite, const Eigen::VectorXd& t3,
                  const Eigen::VectorXd& t, const Eigen::VectorXd& diagonal,
                  Eigen::VectorXd& radiation) override {
        if (!_built) {
            build(diagonal);
        }
        if (!_options.radiation) {
            radiation.setZero();
            if (_written && !rewrite) {
                return false;
            }
            matrix.setZero();
            for (Index col = 0; col < _size; ++col) {
                add_conduction(matrix, col);
            }
            const bool changed = !_written;
            _written = true;
            return changed;
        }
        const bool same = _written && t3 == _last_t3;
        if (same && !rewrite) {
            radiation = _last_radiation;
            return false;
        }
        radiation.setZero();
        for (Index col = 0; col < _size; ++col) {
            const auto g = _g.col(col);
            matrix.col(col) = g * t3[col];
            radiation.noalias() += g * (t3[col] * t[col]);
            add_conduction(matrix, col);
        }
        radiation *= 0.75;
        _last_t3 = t3;
        _last_radiation = radiation;
        _written = true;
        return !same;
    }

    [[nodiscard]] std::int64_t bytes() const noexcept override {
        const auto n = static_cast<std::int64_t>(_size);
        std::int64_t bytes =
            (_conduction_entries *
             (static_cast<std::int64_t>(sizeof(int)) + double_bytes)) +
            ((n + 1) * static_cast<std::int64_t>(sizeof(int))) +
            (n * double_bytes);
        if (_options.radiation) {
            bytes += square_bytes(_size) + (2 * n * double_bytes);
        }
        return bytes;
    }

  private:
    // The lists, the diagonal and G from the couplings and the diagonal of
    // this solve().
    void build(const Eigen::VectorXd& diagonal) {
        const SparseMatrix& conductive = *_couplings.conductive;
        // Both triangles by column, rows in increasing order.
        _starts.assign(to_sizet(_size + 1), 0);
        for (Index row = 0; row < _size; ++row) {
            for (SparseMatrix::InnerIterator it(conductive, row); it; ++it) {
                if (it.col() != row) {
                    ++_starts[to_sizet(it.col()) + 1];
                    ++_starts[to_sizet(row) + 1];
                }
            }
        }
        for (std::size_t col = 0; col < to_sizet(_size); ++col) {
            _starts[col + 1] += _starts[col];
        }
        _rows.resize(to_sizet(_starts.back()));
        _values.resize(to_sizet(_starts.back()));
        std::vector<int> next(_starts.begin(), _starts.end() - 1);
        _diagonal = Eigen::VectorXd::Zero(_size);
        for (Index row = 0; row < _size; ++row) {
            for (SparseMatrix::InnerIterator it(conductive, row); it; ++it) {
                const Index col = it.col();
                add_to_sums(_diagonal, row, col, it.value());
                if (col == row) {
                    // As the varying assembly writes it: subtracted from
                    // both triangles, which meet on the diagonal.
                    _diagonal[row] -= 2.0 * it.value();
                    continue;
                }
                auto entry = to_sizet(next[to_sizet(col)]++);
                _rows[entry] = static_cast<int>(row);
                _values[entry] = -it.value();
                entry = to_sizet(next[to_sizet(row)]++);
                _rows[entry] = static_cast<int>(col);
                _values[entry] = -it.value();
            }
        }
        add_boundary_sums(_diagonal, *_couplings.conductive_boundary);
        if (diagonal.size() != 0) {
            _diagonal += diagonal;
        }
        if (_options.radiation) {
            build_radiation();
        }
        _built = true;
        _written = false;
    }

    void build_radiation() {
        const SparseMatrix& radiative = *_couplings.radiative;
        _g.setZero();
        Eigen::VectorXd sums = Eigen::VectorXd::Zero(_size);
        for (Index row = 0; row < _size; ++row) {
            for (SparseMatrix::InnerIterator it(radiative, row); it; ++it) {
                add_to_sums(sums, row, it.col(), it.value());
                if (it.col() != row) {
                    _g(row, it.col()) = -it.value();
                    _g(it.col(), row) = -it.value();
                }
            }
        }
        add_boundary_sums(sums, *_couplings.radiative_boundary);
        _g.diagonal() = sums;
    }

    void add_conduction(DenseMatrix& matrix, Index col) const {
        const std::span<double> column(matrix.col(col).data(), to_sizet(_size));
        for (int k = _starts[to_sizet(col)]; k < _starts[to_sizet(col) + 1];
             ++k) {
            column[to_sizet(_rows[to_sizet(k)])] += _values[to_sizet(k)];
        }
        column[to_sizet(col)] += _diagonal[col];
    }

    NetworkCouplings _couplings;
    NetworkOptions _options;
    Index _size;
    std::int64_t _conduction_entries = 0;
    DenseMatrix _g;
    std::vector<int> _starts;
    std::vector<int> _rows;
    std::vector<double> _values;
    Eigen::VectorXd _diagonal;
    // The temperatures and radiation term of the matrix written last.
    Eigen::VectorXd _last_t3;
    Eigen::VectorXd _last_radiation;
    bool _built = false;
    bool _written = false;
};

}  // namespace

std::unique_ptr<DenseAssembly> make_dense_assembly(
    const NetworkCouplings& couplings, const NetworkOptions& options) {
    if (options.fixed_couplings) {
        return std::make_unique<FixedDenseAssembly>(couplings, options);
    }
    return std::make_unique<VaryingDenseAssembly>(couplings, options);
}

}  // namespace pycanha::solvers::detail
