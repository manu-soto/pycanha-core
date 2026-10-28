#pragma once

// Machinery shared by the view-factor and the exchange assembly. Both turn an
// accumulated Nf x (Nf + num_virtual_columns) cell block into an
// upper-triangular sparse matrix by visiting every face pair together with
// its transpose, and that traversal is about the ACCESS PATTERN, not about
// what a cell means. Keeping it in one place is what stops the second
// assembly from re-deriving a walk that is written for cache behaviour and
// pinned by bit-identity tests.
//
// What is deliberately NOT here is everything that knows what a cell means:
// how a raw cell becomes a physical quantity, which per-face scalars weight
// the two directions against each other, and when an entry is negligible.
// Each assembly supplies that as an `Emit` object with two members:
//
//     emit.pair(row, column, forward_cell, backward_cell)
//         one face pair, column >= row, cells widened to u64. Called once
//         per pair where at least one direction is nonzero.
//     emit.bucket(row, column, cell)
//         one virtual bucket column, column >= faces, cell != 0. Buckets
//         have no transpose partner and are never combined.
//
// Both are called with ascending `column` within a row, which is what makes
// the emitted arrays a valid CSR image without a sort. Every work item owns
// a private output range, so `Emit` must index its state by the row it is
// handed and never share mutable state between rows.

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/settings.hpp"
#include "pycanha-core/utils/parallel_for.hpp"

namespace pycanha::radiative::detail {

// Tiled-layout row storage: the accumulated (column, cell) pairs of one
// matrix row, kept sorted by column. A sorted array rather than a hash map:
// a block's cells arrive in column order, so a new row is an append and a
// row an earlier batch filled is a linear merge, and the row is freed as one
// allocation. The hash map paid an allocation per entry on insert and a free
// per entry on destruction — minutes for the 2.6e8 entries of a 107 k-face
// scene. operator[] and erase keep the map-like use of the tests and of the
// Metal path; iteration yields (column, cell) pairs like the map's.
class HostCountRow {
  public:
    using value_type = std::pair<std::uint32_t, std::uint64_t>;

    [[nodiscard]] auto begin() const noexcept { return _entries.begin(); }
    [[nodiscard]] auto end() const noexcept { return _entries.end(); }
    [[nodiscard]] std::size_t size() const noexcept { return _entries.size(); }
    [[nodiscard]] bool empty() const noexcept { return _entries.empty(); }
    // Releases the storage, as clearing the map freed its nodes.
    void clear() noexcept { std::vector<value_type>().swap(_entries); }
    void reserve(std::size_t entries) { _entries.reserve(entries); }

    // The cell of `column`, inserted as 0 when absent (the order is kept).
    std::uint64_t& operator[](std::uint32_t column) {
        auto it = lower(column);
        if (it == _entries.end() || it->first != column) {
            it = _entries.insert(it, value_type{column, 0});
        }
        return it->second;
    }
    void erase(std::uint32_t column) {
        const auto it = lower(column);
        if (it != _entries.end() && it->first == column) {
            _entries.erase(it);
        }
    }

    // Adds a dense row's non-zero cells (index = column) with wrap, as the
    // map's += did: an entry an earlier batch made stays even if the sum
    // wraps to 0, and a zero cell adds nothing.
    template <typename Cell>
    void add_dense(std::span<const Cell> cells) {
        std::size_t nonzero = 0;
        for (const Cell cell : cells) {
            nonzero += cell != 0 ? 1U : 0U;
        }
        if (nonzero == 0) {
            return;
        }
        if (_entries.empty()) {
            _entries.reserve(nonzero);
            for (std::size_t col = 0; col < cells.size(); ++col) {
                if (cells[col] != 0) {
                    _entries.emplace_back(static_cast<std::uint32_t>(col),
                                          static_cast<std::uint64_t>(cells[col]));
                }
            }
            return;
        }
        std::vector<value_type> merged;
        merged.reserve(_entries.size() + nonzero);
        auto it = _entries.begin();
        for (std::size_t col = 0; col < cells.size(); ++col) {
            if (cells[col] == 0) {
                continue;
            }
            const auto column = static_cast<std::uint32_t>(col);
            while (it != _entries.end() && it->first < column) {
                merged.push_back(*it++);
            }
            if (it != _entries.end() && it->first == column) {
                merged.emplace_back(
                    column, it->second + static_cast<std::uint64_t>(cells[col]));
                ++it;
            } else {
                merged.emplace_back(column,
                                    static_cast<std::uint64_t>(cells[col]));
            }
        }
        merged.insert(merged.end(), it, _entries.end());
        _entries.swap(merged);
    }

  private:
    [[nodiscard]] std::vector<value_type>::iterator lower(
        std::uint32_t column) {
        return std::lower_bound(
            _entries.begin(), _entries.end(), column,
            [](const value_type& entry, std::uint32_t c) {
                return entry.first < c;
            });
    }

    std::vector<value_type> _entries;
};

// Matrix row stride: the real face columns plus the virtual
// space/inactive/lost bucket columns.
[[nodiscard]] std::size_t matrix_columns(std::size_t faces);

// Weight of the forward estimate when the two Monte-Carlo estimates of a
// face pair are combined:
//
//     X_i = 1/2 (1 + sign(Y) |Y|^n),   Y = (v - u) / (v + u)
//
// with u and v the two estimator variances up to a factor that cancels
// (A_i/N_i and A_j/N_j for both assemblies). |Y|^n is tabulated rather than
// handed to std::pow because the inner loop runs once per matrix cell: at
// 1e4 faces a pow per pair costs roughly ten times the memory traffic
// of the entire pass, which would turn a streaming, bandwidth-bound assembly
// into a compute-bound one. It also removes the only transcendental from an
// otherwise IEEE-exact pipeline, which is what keeps results reproducible
// between toolchains.
//
// The tabulation splits t = m * 2^e with m in [1, 2), so t^n = m^n * 2^(n e),
// and tabulates the two factors separately. A uniform grid in t would be bad
// exactly where it matters: for n < 1 the derivative of t^n is unbounded as
// t -> 0, so the small view factors would carry large relative error. m^n
// over [1, 2) is smooth with bounded derivative, so linear interpolation on
// it is accurate to about 1e-8 at 4096 nodes.
//
// Every entry is rounded to 24 significand bits. Two platforms' std::pow
// agree to within an ulp or so, far finer than that quantum, so the rounded
// tables come out identical and the assembled values then match bit for bit;
// the 6e-8 relative error the rounding costs sits orders of magnitude below
// the Monte-Carlo noise the weight is applied to.
class WeightTable {
  public:
    // `exponent` must be finite and > 0. n = 0 would make the weight
    // winner-takes-all AND break the equal-density case, where 0^n must
    // vanish so that both estimates get exactly one half.
    explicit WeightTable(double exponent);

    // u and v are the two variance proxies; both must be finite and >= 0.
    [[nodiscard]] double forward_weight(double u, double v) const noexcept;

    // Same weight computed straight from std::pow. Kept as the oracle the
    // tabulated path is tested against, and as the fallback the tuning knob
    // below can select.
    [[nodiscard]] static double naive_forward_weight(double u, double v,
                                                     double exponent) noexcept;

  private:
    [[nodiscard]] double unit_pow(double t) const noexcept;

    double _exponent;
    // m^n at 4096 + 1 nodes across m in [1, 2].
    std::vector<double> _mantissa;
    // 2^(n e) for every binary exponent a double in (0, 1) can carry.
    std::vector<double> _scale;
};

// Assembly knobs that are not part of the public API. They exist so a test
// can pin the fast path against the simple one: the blocked walk, the thread
// split and the tabulated weight are all written for speed, and each must
// produce exactly what its obvious counterpart produces.
struct AssemblyTuning {
    // Side of the square tile pairs the transposed walk holds in cache.
    // 0 selects the plain row-by-row walk instead (Dense only; the sparse
    // path is a linear merge in both cases).
    std::size_t tile = 128;
    // 0 asks for hardware_concurrency. Small matrices always run inline.
    unsigned threads = 0;
    // false selects WeightTable::naive_forward_weight.
    bool tabulated_weight = true;
};

// The chosen combination rule, resolved once so the pair loop only ever sees
// a weight lookup.
class Weighting {
  public:
    Weighting(const TriangulationConfig& config, const AssemblyTuning& tuning);

    [[nodiscard]] TriangulationMode mode() const noexcept { return _mode; }

    [[nodiscard]] double forward_weight(double u, double v) const noexcept {
        if (_tabulated) {
            return _table.forward_weight(u, v);
        }
        return WeightTable::naive_forward_weight(u, v, _exponent);
    }

  private:
    TriangulationMode _mode;
    double _exponent;
    bool _tabulated;
    // Built unconditionally: a few thousand pow calls cost microseconds, and
    // constructing it up front also validates the exponent for every mode
    // rather than only for the ones that go on to use it.
    WeightTable _table;
};

// The u = A_i/N_i each face's weight reads (TriangulationConfig::area_weights):
// A_i over the rays face i emitted, or A_i over the mean of the emitting
// faces' rays. Zero where the face emitted nothing, as the gate on the
// actual ratio expects. Ratios only matter relative to each other, but the
// mean is taken exactly so that a uniform trace gets the same doubles either
// way, bit for bit.
[[nodiscard]] std::vector<double> weight_ratios(
    std::span<const double> areas, std::span<const std::uint64_t> rays_per_row,
    bool area_weights);

using pycanha::utils::parallel_for_index;

// Threading a small matrix costs more than the pass itself, so the automatic
// choice stays serial until the cell count justifies it. An explicit request
// is always honoured — that is what lets a test pin the worker count and show
// the result does not depend on it.
[[nodiscard]] unsigned worker_count(const AssemblyTuning& tuning,
                                    std::size_t work_items, std::size_t cells);

// One stored matrix row, built in ascending column order.
struct RowEntries {
    std::vector<SparseIndex> columns;
    std::vector<double> values;
    // Per-entry variance of `values`, parallel to it. Filled only by the
    // constrained least-squares mode, which is the only thing that needs to
    // know how far an entry is allowed to move.
    std::vector<double> variances;

    void push(std::size_t column, double value) {
        columns.push_back(static_cast<SparseIndex>(column));
        values.push_back(value);
    }

    void push(std::size_t column, double value, double variance) {
        push(column, value);
        variances.push_back(variance);
    }
};

// Concatenates the per-row entries into one compressed matrix. The row
// offsets are known before anything is copied, so each row lands at a fixed
// offset and the copy parallelises without affecting the result. Building
// the CSR arrays directly avoids setFromTriplets, whose sort and per-entry
// allocation would dominate everything above it.
[[nodiscard]] SparseMatrix pack_rows(std::span<const RowEntries> rows,
                                     Eigen::Index cols, unsigned threads);

// Dense cells, copied out of the mapped GPU allocation. Reads from mapped
// device memory can traverse PCIe and may be uncached or write-combined,
// which is pathological for the transposed access the pair walk needs; one
// bulk sequential copy — the pattern that hardware is good at — buys back
// every later read.
template <typename Cell>
struct DenseCells {
    std::vector<Cell> cells;
    std::size_t cols = 0;

    [[nodiscard]] std::span<const Cell> row(std::size_t index) const {
        return std::span<const Cell>(cells).subspan(index * cols, cols);
    }
};

// Row-major CSR of the tiled layout's per-row maps, plus the transpose of
// its real-column block. Building the transpose once as a counting sort is
// linear in the stored entries; the alternative those hash maps invite — one
// lookup per column of every row — is quadratic in the face count.
struct SparseCells {
    std::vector<std::size_t> row_start;  // faces + 1
    std::vector<std::uint32_t> column;
    std::vector<std::uint64_t> value;
    // Transpose of the real-column block only; bucket columns have no
    // transpose partner and are never combined. `transposed_row` holds the
    // SOURCE row of each entry, so transposed row i lists every j with a
    // nonzero cell from j to i.
    std::vector<std::size_t> transposed_start;
    std::vector<std::uint32_t> transposed_row;
    std::vector<std::uint64_t> transposed_value;
};

[[nodiscard]] SparseCells build_sparse(std::span<const HostCountRow> host_rows,
                                       std::size_t faces, unsigned threads);

// Thrown by copy_dense when the mapped block is not faces x matrix_columns.
[[noreturn]] void throw_dense_size_mismatch();

template <typename Cell>
[[nodiscard]] DenseCells<Cell> copy_dense(std::span<const Cell> mapped,
                                          std::size_t faces) {
    DenseCells<Cell> cells;
    cells.cols = matrix_columns(faces);
    if (mapped.size() != faces * cells.cols) {
        throw_dense_size_mismatch();
    }
    cells.cells.assign(mapped.begin(), mapped.end());
    return cells;
}

// Visits the nonzero cells of one row in ascending column order. Both
// layouts present the same order, which is what makes every derived
// statistic bit-identical between them.
template <typename Cell, typename Visit>
void visit_row(const DenseCells<Cell>& cells, std::size_t row,
               const Visit& visit) {
    const std::span<const Cell> line = cells.row(row);
    for (std::size_t column = 0; column < line.size(); ++column) {
        if (line[column] != 0) {
            visit(column, static_cast<std::uint64_t>(line[column]));
        }
    }
}

template <typename Visit>
void visit_row(const SparseCells& cells, std::size_t row, const Visit& visit) {
    for (std::size_t at = cells.row_start[row]; at < cells.row_start[row + 1];
         ++at) {
        // A cell can accumulate back to zero (the exchange lost column
        // wraps), and the dense walk skips those, so this one must too.
        if (cells.value[at] != 0) {
            visit(static_cast<std::size_t>(cells.column[at]), cells.value[at]);
        }
    }
}

namespace pair_walk_detail {

// Copies the transposed block (rows [col_begin, col_end), columns
// [row_begin, row_end)) into a small [row][column] scratch buffer. Read in
// place it would stride a full matrix row per useful cell; here the source
// runs are contiguous and the scatter stays inside a buffer that fits in L2
// alongside the forward block.
template <typename Cell>
void gather_transposed(const DenseCells<Cell>& cells, std::size_t row_begin,
                       std::size_t height, std::size_t col_begin,
                       std::size_t width, std::span<Cell> scratch) {
    for (std::size_t column = 0; column < width; ++column) {
        const std::span<const Cell> source =
            cells.row(col_begin + column).subspan(row_begin, height);
        for (std::size_t row = 0; row < height; ++row) {
            scratch[(row * width) + column] = source[row];
        }
    }
}

template <typename Cell, typename Emit>
void diagonal_block(const DenseCells<Cell>& cells, std::size_t begin,
                    std::size_t end, const Emit& emit) {
    for (std::size_t row = begin; row < end; ++row) {
        const std::span<const Cell> forward = cells.row(row);
        for (std::size_t column = row; column < end; ++column) {
            const Cell forward_cell = forward[column];
            const Cell backward_cell = cells.row(column)[row];
            if (forward_cell == 0 && backward_cell == 0) {
                continue;
            }
            emit.pair(row, column, static_cast<std::uint64_t>(forward_cell),
                      static_cast<std::uint64_t>(backward_cell));
        }
    }
}

template <typename Cell, typename Emit>
void offset_block(const DenseCells<Cell>& cells, std::size_t row_begin,
                  std::size_t height, std::size_t col_begin, std::size_t width,
                  std::span<const Cell> scratch, const Emit& emit) {
    for (std::size_t row = 0; row < height; ++row) {
        const std::span<const Cell> forward =
            cells.row(row_begin + row).subspan(col_begin, width);
        const std::span<const Cell> backward =
            scratch.subspan(row * width, width);
        for (std::size_t column = 0; column < width; ++column) {
            if (forward[column] == 0 && backward[column] == 0) {
                continue;
            }
            emit.pair(row_begin + row, col_begin + column,
                      static_cast<std::uint64_t>(forward[column]),
                      static_cast<std::uint64_t>(backward[column]));
        }
    }
}

// Walks one row-tile against every tile to its right, then appends that
// tile's bucket columns. All output from a tile pair (I, J >= I) lands in row
// block I, so the row-tile index is a private, contiguous output range and
// the walk parallelises with no contention and no atomics.
template <typename Cell, typename Emit>
void row_tile(const DenseCells<Cell>& cells, std::size_t faces,
              std::size_t tile, std::size_t index, std::span<Cell> scratch,
              const Emit& emit) {
    const std::size_t row_begin = index * tile;
    const std::size_t row_end = std::min(row_begin + tile, faces);
    const std::size_t height = row_end - row_begin;
    for (std::size_t col_begin = row_begin; col_begin < faces;
         col_begin += tile) {
        const std::size_t width = std::min(col_begin + tile, faces) - col_begin;
        if (col_begin == row_begin) {
            diagonal_block(cells, row_begin, row_end, emit);
            continue;
        }
        gather_transposed(cells, row_begin, height, col_begin, width, scratch);
        offset_block(cells, row_begin, height, col_begin, width,
                     std::span<const Cell>(scratch).first(height * width),
                     emit);
    }
    for (std::size_t row = row_begin; row < row_end; ++row) {
        const std::span<const Cell> buckets = cells.row(row).subspan(faces);
        for (std::size_t offset = 0; offset < buckets.size(); ++offset) {
            if (buckets[offset] != 0) {
                emit.bucket(row, faces + offset,
                            static_cast<std::uint64_t>(buckets[offset]));
            }
        }
    }
}

// Merges the sorted row i of the cells with the sorted row i of their
// transpose, which is column i: linear in the stored entries, sequential in
// both streams, and a pair present in only one direction merges against an
// implicit zero — exactly the degenerate case the weight already handles.
template <typename Emit>
void sparse_row(const SparseCells& cells, std::size_t faces, std::size_t row,
                const Emit& emit) {
    std::size_t at = cells.row_start[row];
    const std::size_t row_end = cells.row_start[row + 1];
    while (at < row_end && cells.column[at] < row) {
        ++at;
    }
    std::size_t transposed = cells.transposed_start[row];
    const std::size_t transposed_end = cells.transposed_start[row + 1];
    while (transposed < transposed_end &&
           cells.transposed_row[transposed] < row) {
        ++transposed;
    }
    while (true) {
        // The face count doubles as the end sentinel: past it lie only the
        // bucket columns, which never take part in a merge.
        const std::size_t forward_column =
            (at < row_end && cells.column[at] < faces) ? cells.column[at]
                                                       : faces;
        const std::size_t backward_column =
            transposed < transposed_end ? cells.transposed_row[transposed]
                                        : faces;
        const std::size_t column = std::min(forward_column, backward_column);
        if (column >= faces) {
            break;
        }
        std::uint64_t forward_cell = 0;
        std::uint64_t backward_cell = 0;
        if (forward_column == column) {
            forward_cell = cells.value[at++];
        }
        if (backward_column == column) {
            backward_cell = cells.transposed_value[transposed++];
        }
        emit.pair(row, column, forward_cell, backward_cell);
    }
    for (; at < row_end; ++at) {
        emit.bucket(row, cells.column[at], cells.value[at]);
    }
}

}  // namespace pair_walk_detail

template <typename Cell, typename Emit>
void walk_dense_pairs(const DenseCells<Cell>& cells, std::size_t faces,
                      const AssemblyTuning& tuning, const Emit& emit) {
    // Tile 0 asks for the plain row-by-row walk: one tile spanning the whole
    // matrix degenerates to exactly that, transposed reads and all.
    const std::size_t tile =
        tuning.tile > 0 ? tuning.tile : std::max<std::size_t>(faces, 1);
    const std::size_t row_tiles = (faces + tile - 1) / tile;
    const unsigned threads =
        worker_count(tuning, row_tiles, faces * cells.cols);
    // A single row-tile means only the diagonal block runs, which reads the
    // transpose in place and never touches the scratch buffer.
    const std::size_t scratch_size = row_tiles > 1 ? tile * tile : 0;
    parallel_for_index(row_tiles, threads, [&](std::size_t index) {
        std::vector<Cell> scratch(scratch_size, 0);
        pair_walk_detail::row_tile(cells, faces, tile, index,
                                   std::span<Cell>(scratch), emit);
    });
}

template <typename Emit>
void walk_sparse_pairs(const SparseCells& cells, std::size_t faces,
                       const AssemblyTuning& tuning, const Emit& emit) {
    const unsigned threads =
        worker_count(tuning, faces, faces * matrix_columns(faces));
    parallel_for_index(faces, threads, [&](std::size_t row) {
        pair_walk_detail::sparse_row(cells, faces, row, emit);
    });
}

}  // namespace pycanha::radiative::detail
