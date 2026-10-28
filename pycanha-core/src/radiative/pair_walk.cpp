// Non-template part of the shared pair-walk machinery (see pair_walk.hpp).
//
// The weight table interpolates as a + f * (b - a). GCC and Clang default to
// contracting that into an FMA, which rounds once instead of twice — a
// different result from a compiler that does not contract, which would
// silently cost the reproducibility the assemblies are written to guarantee.
// The build therefore compiles this file with -ffp-contract=off; MSVC does
// not contract under its default /fp:precise, so no equivalent flag is
// needed there.

#include "pair_walk.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "csr_assembly.hpp"
#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/settings.hpp"

namespace pycanha::radiative::detail {

namespace {

// Nodes across m in [1, 2] for the tabulated m^n.
constexpr std::size_t mantissa_nodes = 1U << 12U;
// Smallest binary exponent a positive double can carry (the smallest
// subnormal is 2^-1074).
constexpr int min_binary_exponent = -1074;
// Significand bits kept in the tables (see the WeightTable comment).
constexpr int table_significand_bits = 24;

// Rounds to table_significand_bits using only exactly-representable
// operations, so the rounded value cannot depend on the platform.
[[nodiscard]] double quantize(double value) {
    if (!(value > 0.0) || !std::isfinite(value)) {
        return value;
    }
    int exponent = 0;
    const double fraction = std::frexp(value, &exponent);
    const double scaled = std::ldexp(fraction, table_significand_bits);
    return std::ldexp(std::round(scaled), exponent - table_significand_bits);
}

}  // namespace

std::size_t matrix_columns(std::size_t faces) {
    return faces + static_cast<std::size_t>(num_virtual_columns);
}

void throw_dense_size_mismatch() {
    throw std::invalid_argument(
        "pycanha::radiative: the dense cell buffer does not match the "
        "face count");
}

WeightTable::WeightTable(double exponent) : _exponent(exponent) {
    if (!std::isfinite(exponent) || exponent <= 0.0) {
        throw std::invalid_argument(
            "pycanha::radiative: the triangulation exponent must be finite "
            "and greater than zero");
    }
    _mantissa.resize(mantissa_nodes + 1);
    for (std::size_t node = 0; node <= mantissa_nodes; ++node) {
        const double mantissa = 1.0 + (static_cast<double>(node) /
                                       static_cast<double>(mantissa_nodes));
        _mantissa[node] = quantize(std::pow(mantissa, exponent));
    }
    _scale.resize(static_cast<std::size_t>(-min_binary_exponent) + 1);
    for (int binary = min_binary_exponent; binary <= 0; ++binary) {
        _scale[static_cast<std::size_t>(binary - min_binary_exponent)] =
            quantize(std::pow(2.0, exponent * static_cast<double>(binary)));
    }
}

double WeightTable::unit_pow(double t) const noexcept {
    if (!(t > 0.0)) {
        return 0.0;
    }
    if (t >= 1.0) {
        return 1.0;
    }
    if (_exponent == 1.0) {
        // Exact, and it is what makes exponent 1 reduce to inverse-variance
        // weighting itself rather than to an interpolation of it.
        return t;
    }
    int exponent = 0;
    const double fraction = std::frexp(t, &exponent);  // in [0.5, 1)
    const double mantissa = fraction * 2.0;            // in [1, 2)
    const double position =
        (mantissa - 1.0) * static_cast<double>(mantissa_nodes);
    const auto node = static_cast<std::size_t>(position);
    const double offset = position - static_cast<double>(node);
    const double lower = _mantissa[node];
    const double step = _mantissa[node + 1] - lower;
    const double mantissa_pow = lower + (offset * step);
    // t = mantissa * 2^(exponent - 1), so the binary factor is 2^(n(e - 1)).
    return mantissa_pow *
           _scale[static_cast<std::size_t>(exponent - 1 - min_binary_exponent)];
}

double WeightTable::forward_weight(double u, double v) const noexcept {
    const double sum = u + v;
    if (!(sum > 0.0)) {
        return 0.5;  // no sampling either way: nothing to prefer
    }
    const double ratio = (v - u) / sum;
    const double magnitude = unit_pow(std::abs(ratio));
    return 0.5 * (1.0 + (ratio < 0.0 ? -magnitude : magnitude));
}

double WeightTable::naive_forward_weight(double u, double v,
                                         double exponent) noexcept {
    const double sum = u + v;
    if (!(sum > 0.0)) {
        return 0.5;
    }
    const double ratio = (v - u) / sum;
    const double magnitude = std::pow(std::abs(ratio), exponent);
    return 0.5 * (1.0 + (ratio < 0.0 ? -magnitude : magnitude));
}

Weighting::Weighting(const TriangulationConfig& config,
                     const AssemblyTuning& tuning)
    : _mode(config.mode),
      _exponent(config.exponent),
      _tabulated(tuning.tabulated_weight),
      _table(config.exponent) {}

std::vector<double> weight_ratios(std::span<const double> areas,
                                  std::span<const std::uint64_t> rays_per_row,
                                  bool area_weights) {
    std::vector<double> ratio(areas.size(), 0.0);
    double rays_used = 0.0;
    if (area_weights) {
        // An integer sum and count, then one division: for N rays on each of
        // k faces this is (N k) / k, which is N exactly, so a uniform trace
        // divides by the same double as with the actual counts.
        std::uint64_t total = 0;
        std::uint64_t emitting = 0;
        for (const std::uint64_t rays : rays_per_row) {
            total += rays;
            emitting += rays > 0 ? 1U : 0U;
        }
        if (emitting > 0) {
            rays_used =
                static_cast<double>(total) / static_cast<double>(emitting);
        }
    }
    for (std::size_t face = 0; face < areas.size(); ++face) {
        if (rays_per_row[face] > 0) {
            ratio[face] =
                areas[face] / (area_weights
                                   ? rays_used
                                   : static_cast<double>(rays_per_row[face]));
        }
    }
    return ratio;
}

unsigned worker_count(const AssemblyTuning& tuning, std::size_t work_items,
                      std::size_t cells) {
    constexpr std::size_t parallel_cell_threshold = 1U << 18U;
    unsigned threads = tuning.threads;
    if (threads == 0) {
        if (cells < parallel_cell_threshold) {
            return 1;
        }
        threads = std::thread::hardware_concurrency();
    }
    threads = std::max(threads, 1U);
    return static_cast<unsigned>(
        std::min<std::size_t>(threads, std::max<std::size_t>(work_items, 1)));
}

SparseMatrix pack_rows(std::span<const RowEntries> rows, Eigen::Index cols,
                       unsigned threads) {
    std::vector<SparseIndex> row_starts;
    row_starts.reserve(rows.size() + 1);
    row_starts.push_back(0);
    std::size_t total = 0;
    for (const RowEntries& row : rows) {
        total += row.columns.size();
        check_sparse_capacity(total);
        row_starts.push_back(static_cast<SparseIndex>(total));
    }
    std::vector<SparseIndex> columns(total);
    std::vector<double> values(total);
    parallel_for_index(rows.size(), threads, [&](std::size_t row) {
        const auto at = static_cast<std::ptrdiff_t>(row_starts[row]);
        std::ranges::copy(rows[row].columns, columns.begin() + at);
        std::ranges::copy(rows[row].values, values.begin() + at);
    });
    return make_csr(static_cast<Eigen::Index>(rows.size()), cols, row_starts,
                    columns, values);
}

namespace {

void sort_sparse_rows(SparseCells& cells,
                      std::span<const HostCountRow> host_rows,
                      unsigned threads) {
    const std::size_t faces = host_rows.size();
    cells.row_start.assign(faces + 1, 0);
    for (std::size_t row = 0; row < faces; ++row) {
        cells.row_start[row + 1] = cells.row_start[row] + host_rows[row].size();
    }
    cells.column.resize(cells.row_start[faces]);
    cells.value.resize(cells.row_start[faces]);
    parallel_for_index(faces, threads, [&](std::size_t row) {
        using Entry = std::pair<std::uint32_t, std::uint64_t>;
        std::vector<Entry> entries(host_rows[row].begin(),
                                   host_rows[row].end());
        std::ranges::sort(entries, {}, &Entry::first);
        std::size_t at = cells.row_start[row];
        for (const auto& [column, cell] : entries) {
            cells.column[at] = column;
            cells.value[at] = cell;
            ++at;
        }
    });
}

void transpose_sparse_rows(SparseCells& cells, std::size_t faces) {
    std::vector<std::size_t> per_column(faces + 1, 0);
    for (const std::uint32_t column : cells.column) {
        if (column < faces) {
            ++per_column[static_cast<std::size_t>(column) + 1];
        }
    }
    cells.transposed_start.assign(faces + 1, 0);
    std::inclusive_scan(per_column.begin() + 1, per_column.end(),
                        cells.transposed_start.begin() + 1);
    cells.transposed_row.resize(cells.transposed_start[faces]);
    cells.transposed_value.resize(cells.transposed_start[faces]);
    std::vector<std::size_t> cursor(cells.transposed_start.begin(),
                                    cells.transposed_start.end() - 1);
    // Rows are visited in ascending order, so every transposed row comes out
    // sorted by source row without a second sort.
    for (std::size_t row = 0; row < faces; ++row) {
        for (std::size_t at = cells.row_start[row];
             at < cells.row_start[row + 1]; ++at) {
            const std::uint32_t column = cells.column[at];
            if (column >= faces) {
                continue;
            }
            const std::size_t target = cursor[column]++;
            cells.transposed_row[target] = static_cast<std::uint32_t>(row);
            cells.transposed_value[target] = cells.value[at];
        }
    }
}

}  // namespace

SparseCells build_sparse(std::span<const HostCountRow> host_rows,
                         std::size_t faces, unsigned threads) {
    if (host_rows.size() != faces) {
        throw std::invalid_argument(
            "pycanha::radiative: the tiled cell rows do not match the "
            "face count");
    }
    SparseCells cells;
    sort_sparse_rows(cells, host_rows, threads);
    transpose_sparse_rows(cells, faces);
    return cells;
}

}  // namespace pycanha::radiative::detail
