// Backend-agnostic VF assembly (pure CPU; see vf_assemble.hpp for why it
// does not live in either backend, and pair_walk.hpp for the traversal it
// shares with the exchange assembly).
//
// Two passes over the accumulated counts: a streaming row scan for closure
// and the per-entry statistics, then the pair walk that combines each face
// pair with its transpose into the upper triangle.
//
// Everything that accumulates a double does so in a fixed order — one whole
// row or one whole row-tile per worker, each writing its own output range —
// so the worker count and the schedule cannot change a bit of the result.
// The only reductions that cross workers are maxima, which are
// order-independent anyway.

#include "vf_assemble.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#include "closure_projection.hpp"
#include "pair_walk.hpp"
#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/settings.hpp"

namespace pycanha::radiative::detail {

namespace {

// Per-face precomputation. A_i/N_i is at once the variance proxy the weight
// is built from and the factor that turns a raw count into the extensive
// A_i * count / N_i, so hoisting it out of the pair loop removes a multiply
// and a divide from every cell and leaves the weight a function of two
// scalars that do not vary along the inner loop.
//
// A row that emitted nothing gets 0, which stands in for the infinite ratio
// that limit implies: every use of it either multiplies a count that is
// itself zero or is guarded explicitly.
[[nodiscard]] std::vector<double> slot_ratios(
    std::span<const double> areas,
    std::span<const std::uint64_t> rays_per_row) {
    std::vector<double> ratio(areas.size(), 0.0);
    for (std::size_t face = 0; face < areas.size(); ++face) {
        if (rays_per_row[face] > 0) {
            ratio[face] = areas[face] / static_cast<double>(rays_per_row[face]);
        }
    }
    return ratio;
}

// Per-row scan products, reduced afterwards in row order.
struct RowStats {
    double stderr_sum = 0.0;
    double stderr_max = 0.0;
    std::size_t entries = 0;
};

[[nodiscard]] double combine_pair(double forward, double backward,
                                  double forward_ratio, double backward_ratio,
                                  const Weighting& weighting) {
    if (weighting.mode() == TriangulationMode::None) {
        return forward;
    }
    // A row that emitted nothing has an infinite A/N, so Y -> -1 and the
    // weight passes entirely to the direction that does have data. The
    // degenerate limit is the right answer; only the arithmetic needs
    // guarding, which is why the ratio is stored as zero there.
    if (!(forward_ratio > 0.0)) {
        return backward;
    }
    if (!(backward_ratio > 0.0)) {
        return forward;
    }
    const double weight =
        weighting.forward_weight(forward_ratio, backward_ratio);
    return (weight * forward) + ((1.0 - weight) * backward);
}

// Thresholding compares the INTENSIVE max(F_ij, F_ji) = G / min(A_i, A_j),
// so the knob keeps its units and a pair goes only when both directions are
// negligible. A face with no area carries no geometry and cannot be hit, so
// there the guard keeps whatever arrived rather than dividing by zero.
[[nodiscard]] bool keep_pair(double value, double area_i, double area_j,
                             double threshold) {
    if (value == 0.0) {
        return false;
    }
    const double smaller = std::min(area_i, area_j);
    return !(smaller > 0.0) || (value / smaller) > threshold;
}

// --- pass A: closure, standard errors, optional full matrix ---------------

struct ScanInputs {
    std::size_t faces = 0;
    std::span<const std::uint64_t> rays_per_row;
    std::span<const double> ratio;
    double threshold = 0.0;
    bool keep_full_matrix = false;
};

struct ScanOutputs {
    std::span<double> row_sums;
    std::span<RowStats> stats;
    std::span<RowEntries> full_rows;  // empty unless keep_full_matrix
};

template <typename Cells>
void scan_rows(const Cells& cells, const ScanInputs& in, const ScanOutputs& out,
               unsigned threads) {
    parallel_for_index(in.faces, threads, [&](std::size_t row) {
        const std::uint64_t rays = in.rays_per_row[row];
        if (rays == 0) {
            return;
        }
        const auto rays_as_double = static_cast<double>(rays);
        const double ratio = in.ratio[row];
        double row_sum = 0.0;
        RowStats stats;
        visit_row(cells, row, [&](std::size_t column, std::uint64_t count) {
            const double vf = static_cast<double>(count) / rays_as_double;
            // Closure accounting runs over every column and BEFORE any
            // thresholding, so pruning tiny entries never corrupts it.
            row_sum += vf;
            if (column < in.faces) {
                // Binomial standard error of the per-entry estimate (real
                // face columns only).
                const double entry_stderr =
                    std::sqrt(vf * std::max(1.0 - vf, 0.0) / rays_as_double);
                stats.stderr_sum += entry_stderr;
                stats.stderr_max = std::max(stats.stderr_max, entry_stderr);
                ++stats.entries;
            }
            if (in.keep_full_matrix && vf > in.threshold) {
                out.full_rows[row].push(column,
                                        ratio * static_cast<double>(count));
            }
        });
        out.row_sums[row] = row_sum;
        out.stats[row] = stats;
    });
}

// --- pass B: triangulation into the upper triangle ------------------------

// The Emit the shared pair walk drives. Every member writes only into the
// row it is handed, so the walk can distribute rows however it likes.
class VfEmit {
  public:
    VfEmit(std::span<const double> areas, std::span<const double> ratio,
           std::span<const double> weight, const Weighting& weighting,
           double threshold, std::span<RowEntries> rows,
           std::span<double> residual)
        : _areas(areas),
          _ratio(ratio),
          _weight(weight),
          _weighting(&weighting),
          _threshold(threshold),
          _project(weighting.mode() ==
                   TriangulationMode::ConstrainedLeastSquares),
          _rows(rows),
          _residual(residual) {}

    void pair(std::size_t row, std::size_t column, std::uint64_t forward,
              std::uint64_t backward) const {
        const double forward_value = _ratio[row] * static_cast<double>(forward);
        const double backward_value =
            _ratio[column] * static_cast<double>(backward);
        _residual[row] =
            std::max(_residual[row],
                     raw_residual(row, column, forward_value, backward_value));
        if (_project) {
            // Nothing is dropped before the projection: an entry the
            // projection cannot see is one the closure it imposes would be
            // missing. The threshold runs afterwards instead.
            const BlueEstimate blue = blue_combine(
                forward_value, backward_value, _ratio[row], _ratio[column]);
            _rows[row].push(column, blue.value, blue.variance);
            return;
        }
        // The weight reads `_weight`, which is zero exactly where `_ratio`
        // is (weight_ratios), so the gates in combine_pair do not change.
        const double value =
            combine_pair(forward_value, backward_value, _weight[row],
                         _weight[column], *_weighting);
        if (keep_pair(value, _areas[row], _areas[column], _threshold)) {
            _rows[row].push(column, value);
        }
    }

    // The space/inactive/lost columns are the closure accounting: they have
    // no transpose partner to be combined with and no partner to be
    // negligible against, so they pass through untriangulated and
    // unthresholded. They are part of a row's closure, so the projection
    // gets to move them like any other unknown.
    void bucket(std::size_t row, std::size_t column, std::uint64_t cell) const {
        const double value = _ratio[row] * static_cast<double>(cell);
        if (_project) {
            _rows[row].push(column, value, value * _ratio[row]);
            return;
        }
        _rows[row].push(column, value);
    }

  private:
    // The disagreement of the two raw estimates, measured BEFORE they are
    // combined and only where both directions carry samples. It is the
    // winding/parity check: taken after combining it would be zero by
    // construction and would detect nothing.
    [[nodiscard]] double raw_residual(std::size_t row, std::size_t column,
                                      double forward, double backward) const {
        const double larger = std::max(forward, backward);
        if (_ratio[row] > 0.0 && _ratio[column] > 0.0 && larger > 0.0) {
            return std::abs(forward - backward) / larger;
        }
        return 0.0;
    }

    std::span<const double> _areas;
    std::span<const double> _ratio;
    std::span<const double> _weight;
    const Weighting* _weighting;
    double _threshold;
    bool _project;
    std::span<RowEntries> _rows;
    std::span<double> _residual;
};

// Closure targets for the projection: a row that emitted rays must account
// for exactly its own area across all columns. A row that emitted nothing
// carries no constraint, which is what the zero says.
[[nodiscard]] std::vector<double> closure_targets(
    std::span<const double> areas,
    std::span<const std::uint64_t> rays_per_row) {
    std::vector<double> targets(areas.size(), 0.0);
    for (std::size_t face = 0; face < areas.size(); ++face) {
        if (rays_per_row[face] > 0) {
            targets[face] = areas[face];
        }
    }
    return targets;
}

// Drops the negligible entries the projection was deliberately not allowed
// to drop earlier. Buckets are closure accounting and always survive.
void prune_rows(std::span<RowEntries> rows, std::span<const double> areas,
                double threshold, std::size_t faces, unsigned threads) {
    parallel_for_index(rows.size(), threads, [&](std::size_t row) {
        RowEntries kept;
        const RowEntries& entries = rows[row];
        for (std::size_t at = 0; at < entries.values.size(); ++at) {
            const auto column = static_cast<std::size_t>(entries.columns[at]);
            const double value = entries.values[at];
            if (column >= faces ||
                keep_pair(value, areas[row], areas[column], threshold)) {
                kept.push(column, value);
            }
        }
        rows[row] = std::move(kept);
    });
}

// Reduces the per-row scan products in row order, so the double sums do not
// depend on how the rows were distributed across workers.
void reduce_stats(std::span<const RowStats> rows, TraceStats& stats) {
    double stderr_sum = 0.0;
    std::size_t entries = 0;
    double stderr_max = 0.0;
    for (const RowStats& row : rows) {
        stderr_sum += row.stderr_sum;
        entries += row.entries;
        stderr_max = std::max(stderr_max, row.stderr_max);
    }
    stats.mean_stderr =
        entries > 0 ? stderr_sum / static_cast<double>(entries) : 0.0;
    stats.max_stderr = stderr_max;
}

}  // namespace

VfResult assemble_vf(CountSource counts, std::span<const double> areas,
                     std::span<const std::uint64_t> rays_per_row,
                     const AccumConfig& config, const AssemblyTuning& tuning) {
    const std::size_t faces = areas.size();
    if (rays_per_row.size() != faces) {
        throw std::invalid_argument(
            "pycanha::radiative: rays_per_row must have one entry per face "
            "face");
    }
    const std::size_t cols = matrix_columns(faces);
    const std::vector<double> ratio = slot_ratios(areas, rays_per_row);
    const Weighting weighting(config.triangulation, tuning);
    const bool keep_full = config.triangulation.keep_full_matrix;

    VfResult result;
    result.row_sums = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(faces));
    std::vector<RowStats> row_stats(faces);
    std::vector<RowEntries> rows(faces);
    std::vector<RowEntries> full_rows(keep_full ? faces : 0);
    std::vector<double> residual(faces, 0.0);

    const ScanInputs scan_in{.faces = faces,
                             .rays_per_row = rays_per_row,
                             .ratio = ratio,
                             .threshold = config.sparse_threshold,
                             .keep_full_matrix = keep_full};
    const ScanOutputs scan_out{
        .row_sums = std::span<double>(result.row_sums.data(), faces),
        .stats = row_stats,
        .full_rows = full_rows};
    const std::vector<double> weight = weight_ratios(
        areas, rays_per_row, config.triangulation.area_weights);
    const VfEmit emit(areas, ratio, weight, weighting, config.sparse_threshold,
                      rows, residual);
    const unsigned scan_threads = worker_count(tuning, faces, faces * cols);

    if (const auto* mapped =
            std::get_if<std::span<const std::uint32_t>>(&counts)) {
        const DenseCells<std::uint32_t> dense = copy_dense(*mapped, faces);
        scan_rows(dense, scan_in, scan_out, scan_threads);
        walk_dense_pairs(dense, faces, tuning, emit);
    } else {
        const SparseCells sparse =
            build_sparse(std::get<std::span<const HostCountRow>>(counts), faces,
                         scan_threads);
        scan_rows(sparse, scan_in, scan_out, scan_threads);
        walk_sparse_pairs(sparse, faces, tuning, emit);
    }

    if (config.triangulation.mode ==
        TriangulationMode::ConstrainedLeastSquares) {
        project_onto_closure(rows, closure_targets(areas, rays_per_row), faces);
        prune_rows(rows, areas, config.sparse_threshold, faces, scan_threads);
    }

    result.vf = pack_rows(rows, static_cast<Eigen::Index>(cols), scan_threads);
    if (keep_full) {
        result.full_vf =
            pack_rows(full_rows, static_cast<Eigen::Index>(cols), scan_threads);
    }
    reduce_stats(row_stats, result.stats);
    result.stats.reciprocity_residual =
        residual.empty() ? 0.0 : *std::ranges::max_element(residual);
    return result;
}

}  // namespace pycanha::radiative::detail
