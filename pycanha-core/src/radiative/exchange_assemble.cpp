// Backend-agnostic exchange assembly (pure CPU; see exchange_assemble.hpp
// for why it does not live in either backend, and pair_walk.hpp for the
// traversal it shares with the view-factor assembly).
//
// Two passes over the accumulated cells: a streaming row scan for the
// per-entry statistics, the signed lost-energy accounting and the optional
// raw matrix, then the pair walk that combines each face pair with its
// transpose into the upper triangle of H.
//
// Everything that accumulates a double does so in a fixed order — one whole
// row or one whole row-tile per worker, each writing its own output range —
// so the worker count and the schedule cannot change a bit of the result.

#include "exchange_assemble.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#include "closure_projection.hpp"
#include "pair_walk.hpp"
#include "pycanha-core/radiative/materials.hpp"
#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/settings.hpp"

namespace pycanha::radiative::detail {

namespace {

// Per-face precomputation, hoisted out of the pair loop so the inner loop
// only ever multiplies.
struct SlotScales {
    // A_i eps_i / N_i: turns a de-scaled cell energy straight into the
    // extensive H_ij. Zero when the row emitted nothing, when the face has
    // no area, or when it has no emissivity in this band.
    std::vector<double> emission;
    // A_i / N_i: the variance proxy the weight is built from. The exchange
    // kernel's per-cell variance carries a further eps_i eps_j, but that
    // factor appears in BOTH directions and cancels out of the weight, so
    // the proxy is the same one the view-factor assembly uses.
    std::vector<double> ratio;
    // What the weight reads: `ratio` itself, or A_i over the emitting faces'
    // mean rays (TriangulationConfig::area_weights, weight_ratios).
    std::vector<double> weight;
    // A_i eps_i: the emissive area a stored H is divided by to recover the
    // intensive B, and therefore what the threshold compares against.
    std::vector<double> emissive_area;
    // The band absorptivity itself, which the absolute variance needs even
    // though the weight does not.
    std::vector<double> emissivity;
};

[[nodiscard]] SlotScales slot_scales(
    std::span<const double> areas, std::span<const double> emissivity,
    std::span<const std::uint64_t> rays_per_row, bool area_weights) {
    const std::size_t faces = areas.size();
    SlotScales scales;
    scales.emission.assign(faces, 0.0);
    scales.ratio.assign(faces, 0.0);
    scales.emissive_area.assign(faces, 0.0);
    scales.emissivity.assign(emissivity.begin(), emissivity.end());
    for (std::size_t face = 0; face < faces; ++face) {
        scales.emissive_area[face] = areas[face] * emissivity[face];
        if (rays_per_row[face] > 0) {
            const auto rays = static_cast<double>(rays_per_row[face]);
            scales.emission[face] = scales.emissive_area[face] / rays;
            scales.ratio[face] = areas[face] / rays;
        }
    }
    scales.weight = weight_ratios(areas, rays_per_row, area_weights);
    return scales;
}

// Per-row scan products, reduced afterwards in row order.
struct RowStats {
    double stderr_sum = 0.0;
    double stderr_max = 0.0;
    std::size_t entries = 0;
    double lost_energy = 0.0;  // signed: Russian-roulette adjustments
};

// --- pass A: statistics, lost energy, optional raw matrix -----------------

struct ScanInputs {
    std::size_t faces = 0;
    std::size_t lost_column = 0;
    std::span<const std::uint64_t> rays_per_row;
    double inv_scale = 0.0;
    double threshold = 0.0;
    bool keep_full_matrix = false;
};

struct ScanOutputs {
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
        RowStats stats;
        visit_row(cells, row, [&](std::size_t column, std::uint64_t cell) {
            // The lost column is signed: Russian-roulette boost withdrawals
            // may push it (slightly) negative. The fixed-point value
            // converts FIRST (exact for full deposits), THEN divides by the
            // rays — the same rounding path as the vf counts, so blackbody
            // exchange factors match view factors to the bit.
            const double energy =
                column == in.lost_column
                    ? static_cast<double>(static_cast<std::int64_t>(cell)) *
                          in.inv_scale
                    : static_cast<double>(cell) * in.inv_scale;
            const double factor = energy / rays_as_double;
            if (column == in.lost_column) {
                stats.lost_energy += energy;
            }
            if (column < in.faces) {
                // Conservative per-entry standard error: a ray's deposit
                // into one cell is in [0, 1], so the Bernoulli bound
                // dominates the true variance.
                const double entry_stderr = std::sqrt(
                    factor * std::max(1.0 - factor, 0.0) / rays_as_double);
                stats.stderr_sum += entry_stderr;
                stats.stderr_max = std::max(stats.stderr_max, entry_stderr);
                ++stats.entries;
            }
            // Statistics come before thresholding; the threshold only prunes
            // what is stored.
            if (in.keep_full_matrix && std::abs(factor) > in.threshold) {
                out.full_rows[row].push(column, factor);
            }
        });
        out.stats[row] = stats;
    });
}

// --- pass B: triangulation into the upper triangle ------------------------

// The Emit the shared pair walk drives. Every member writes only into the
// row it is handed, so the walk can distribute rows however it likes.
class ExchangeEmit {
  public:
    ExchangeEmit(const SlotScales& scales, const Weighting& weighting,
                 double inv_scale, double threshold, std::size_t lost_column,
                 std::span<RowEntries> rows, std::span<double> residual)
        : _scales(&scales),
          _weighting(&weighting),
          _inv_scale(inv_scale),
          _threshold(threshold),
          _project(weighting.mode() ==
                   TriangulationMode::ConstrainedLeastSquares),
          _lost_column(lost_column),
          _rows(rows),
          _residual(residual) {}

    void pair(std::size_t row, std::size_t column, std::uint64_t forward_cell,
              std::uint64_t backward_cell) const {
        const double forward = _scales->emission[row] *
                               (static_cast<double>(forward_cell) * _inv_scale);
        const double backward =
            _scales->emission[column] *
            (static_cast<double>(backward_cell) * _inv_scale);
        // The gate is about SAMPLING, so it reads the ray-density proxy and
        // not the emission scale: a face with zero emissivity did emit rays
        // and its estimate is a perfectly good (identically zero) one.
        const bool has_forward = _scales->ratio[row] > 0.0;
        const bool has_backward = _scales->ratio[column] > 0.0;
        // Measured on the raw estimates, before they are combined, and only
        // where both directions carry samples. Taken after combining it
        // would be zero by construction and would detect nothing.
        const double larger = std::max(forward, backward);
        if (has_forward && has_backward && larger > 0.0) {
            _residual[row] =
                std::max(_residual[row], std::abs(forward - backward) / larger);
        }
        if (_project) {
            // Nothing is dropped before the projection: an entry the
            // projection cannot see is one the closure it imposes would be
            // missing. The threshold runs afterwards instead.
            const BlueEstimate blue = blue_combine(
                forward, backward, _scales->ratio[row], _scales->ratio[column]);
            // The exchange variance carries a further eps_i eps_j that the
            // ray-density proxy drops because it cancels out of the WEIGHT.
            // It does not cancel out of the absolute variance, which is what
            // decides how far this entry may move relative to the others.
            _rows[row].push(column, blue.value,
                            blue.variance * _scales->emissivity[row] *
                                _scales->emissivity[column]);
            return;
        }
        const double value =
            combine(row, column, forward, backward, has_forward, has_backward);
        if (keep(value, row, column)) {
            _rows[row].push(column, value);
        }
    }

    // The space/inactive/lost columns are the closure accounting: they have
    // no transpose partner to be combined with and no partner to be
    // negligible against, so they pass through untriangulated and
    // unthresholded. They carry the same A_i eps_i scaling as the real
    // columns, which is what keeps the stored matrix dimensionally uniform.
    void bucket(std::size_t row, std::size_t column, std::uint64_t cell) const {
        const double energy =
            column == _lost_column
                ? static_cast<double>(static_cast<std::int64_t>(cell)) *
                      _inv_scale
                : static_cast<double>(cell) * _inv_scale;
        const double value = _scales->emission[row] * energy;
        // A row whose emissive area is zero traced rays and filled cells,
        // but carries no energy: storing explicit zeros for it would be the
        // one place the matrix admits a structural nonzero that is not one.
        if (value == 0.0) {
            return;
        }
        if (_project) {
            // Energy that escapes to space is fully accounted there, so the
            // absorber-side emissivity of a bucket is one. The magnitude,
            // because the lost column can be negative.
            _rows[row].push(column, value,
                            std::abs(value) * _scales->ratio[row] *
                                _scales->emissivity[row]);
            return;
        }
        _rows[row].push(column, value);
    }

  private:
    [[nodiscard]] double combine(std::size_t row, std::size_t column,
                                 double forward, double backward,
                                 bool has_forward, bool has_backward) const {
        if (_weighting->mode() == TriangulationMode::None) {
            return forward;
        }
        // A row that emitted nothing has an infinite A/N, so Y -> -1 and the
        // weight passes entirely to the direction that does have data. The
        // degenerate limit is the right answer; only the arithmetic needs
        // guarding, which is why the ratio is stored as zero there.
        if (!has_forward) {
            return backward;
        }
        if (!has_backward) {
            return forward;
        }
        const double weight = _weighting->forward_weight(
            _scales->weight[row], _scales->weight[column]);
        return (weight * forward) + ((1.0 - weight) * backward);
    }

    // Thresholding compares the INTENSIVE max(B_ij, B_ji), which is the
    // stored H over the smaller of the two emissive areas, so the knob keeps
    // its units and a pair goes only when both directions are negligible.
    // Zero emissivity makes that divisor zero — a face that absorbs nothing
    // and emits nothing — and there the value is already exactly zero, which
    // the first test drops anyway.
    [[nodiscard]] bool keep(double value, std::size_t row,
                            std::size_t column) const {
        if (value == 0.0) {
            return false;
        }
        const double smaller = std::min(_scales->emissive_area[row],
                                        _scales->emissive_area[column]);
        return !(smaller > 0.0) || (std::abs(value) / smaller) > _threshold;
    }

    const SlotScales* _scales;
    const Weighting* _weighting;
    double _inv_scale;
    double _threshold;
    bool _project;
    std::size_t _lost_column;
    std::span<RowEntries> _rows;
    std::span<double> _residual;
};

// Drops the negligible entries the projection was deliberately not allowed
// to drop earlier. Buckets are closure accounting and always survive.
void prune_rows(std::span<RowEntries> rows,
                std::span<const double> emissive_area, double threshold,
                std::size_t faces, unsigned threads) {
    parallel_for_index(rows.size(), threads, [&](std::size_t row) {
        RowEntries kept;
        const RowEntries& entries = rows[row];
        for (std::size_t at = 0; at < entries.values.size(); ++at) {
            const auto column = static_cast<std::size_t>(entries.columns[at]);
            const double value = entries.values[at];
            if (value == 0.0) {
                continue;
            }
            if (column < faces) {
                const double smaller =
                    std::min(emissive_area[row], emissive_area[column]);
                if (smaller > 0.0 &&
                    !((std::abs(value) / smaller) > threshold)) {
                    continue;
                }
            }
            kept.push(column, value);
        }
        rows[row] = std::move(kept);
    });
}

// Reduces the per-row scan products in row order, so the double sums do not
// depend on how the rows were distributed across workers.
void reduce_stats(std::span<const RowStats> rows,
                  std::span<const std::uint64_t> rays_per_row,
                  TraceStats& stats) {
    double stderr_sum = 0.0;
    double lost_energy = 0.0;
    double emitted_energy = 0.0;
    std::size_t entries = 0;
    double stderr_max = 0.0;
    for (std::size_t row = 0; row < rows.size(); ++row) {
        stderr_sum += rows[row].stderr_sum;
        lost_energy += rows[row].lost_energy;
        entries += rows[row].entries;
        stderr_max = std::max(stderr_max, rows[row].stderr_max);
        emitted_energy += static_cast<double>(rays_per_row[row]);
    }
    stats.mean_stderr =
        entries > 0 ? stderr_sum / static_cast<double>(entries) : 0.0;
    stats.max_stderr = stderr_max;
    stats.lost_energy_fraction =
        emitted_energy > 0.0 ? lost_energy / emitted_energy : 0.0;
}

}  // namespace

std::vector<double> band_emissivity(const MaterialTable& materials, Band band) {
    const auto faces = static_cast<std::size_t>(materials.num_faces());
    // No material assigned means blackbody, matching the kernel's fallback.
    std::vector<double> emissivity(faces, 1.0);
    const Eigen::Index column = band == Band::Solar ? 3 : 0;
    for (std::size_t face = 0; face < faces; ++face) {
        const int material =
            materials.face_material(static_cast<Eigen::Index>(face));
        if (material >= 0) {
            emissivity[face] = materials.properties(material, column);
        }
    }
    return emissivity;
}

ExchangeResult assemble_exchange(const ExchangeInputs& in,
                                 const AccumConfig& config,
                                 const AssemblyTuning& tuning) {
    const std::size_t faces = in.rays_per_row.size();
    if (in.areas.size() != faces || in.emissivity.size() != faces) {
        throw std::invalid_argument(
            "pycanha::radiative: areas and emissivity must have one entry per "
            "face");
    }
    const std::size_t cols = matrix_columns(faces);
    const SlotScales scales =
        slot_scales(in.areas, in.emissivity, in.rays_per_row,
                    config.triangulation.area_weights);
    const Weighting weighting(config.triangulation, tuning);
    const bool keep_full = config.triangulation.keep_full_matrix;

    std::vector<RowStats> row_stats(faces);
    std::vector<RowEntries> rows(faces);
    std::vector<RowEntries> full_rows(keep_full ? faces : 0);
    std::vector<double> residual(faces, 0.0);

    const ScanInputs scan_in{
        .faces = faces,
        .lost_column = faces + static_cast<std::size_t>(lost_column_offset),
        .rays_per_row = in.rays_per_row,
        .inv_scale = in.fp_scale > 0.0 ? 1.0 / in.fp_scale : 0.0,
        .threshold = config.sparse_threshold,
        .keep_full_matrix = keep_full};
    const ScanOutputs scan_out{.stats = row_stats, .full_rows = full_rows};
    const ExchangeEmit emit(scales, weighting, scan_in.inv_scale,
                            config.sparse_threshold, scan_in.lost_column, rows,
                            residual);
    const unsigned scan_threads = worker_count(tuning, faces, faces * cols);

    // Nothing was ever traced: no scale was chosen, so there is nothing to
    // divide by and nothing to report.
    if (in.fp_scale > 0.0) {
        if (const auto* mapped =
                std::get_if<std::span<const std::uint64_t>>(&in.cells)) {
            const DenseCells<std::uint64_t> dense = copy_dense(*mapped, faces);
            scan_rows(dense, scan_in, scan_out, scan_threads);
            walk_dense_pairs(dense, faces, tuning, emit);
        } else {
            const SparseCells sparse =
                build_sparse(std::get<std::span<const HostCountRow>>(in.cells),
                             faces, scan_threads);
            scan_rows(sparse, scan_in, scan_out, scan_threads);
            walk_sparse_pairs(sparse, faces, tuning, emit);
        }
        if (config.triangulation.mode ==
            TriangulationMode::ConstrainedLeastSquares) {
            // A row that emitted rays must account for exactly the energy it
            // emitted, A_i eps_i, across all columns. A row that emitted
            // nothing — or that emits nothing because its emissivity is zero
            // — carries no constraint, which is what a zero target says.
            std::vector<double> targets(faces, 0.0);
            for (std::size_t face = 0; face < faces; ++face) {
                if (in.rays_per_row[face] > 0) {
                    targets[face] = scales.emissive_area[face];
                }
            }
            project_onto_closure(rows, targets, faces);
            prune_rows(rows, scales.emissive_area, config.sparse_threshold,
                       faces, scan_threads);
        }
    }

    ExchangeResult result;
    result.band = in.band;
    result.factors =
        pack_rows(rows, static_cast<Eigen::Index>(cols), scan_threads);
    if (keep_full) {
        result.full_factors =
            pack_rows(full_rows, static_cast<Eigen::Index>(cols), scan_threads);
    }
    reduce_stats(row_stats, in.rays_per_row, result.stats);
    result.stats.reciprocity_residual =
        residual.empty() ? 0.0 : *std::ranges::max_element(residual);
    return result;
}

std::uint64_t exchange_conservation_error(
    ExchangeCellSource cells, std::span<const std::uint64_t> rays_per_row,
    double fp_scale) {
    if (fp_scale <= 0.0) {
        return 0;
    }
    const std::size_t faces = rays_per_row.size();
    const std::size_t cols = matrix_columns(faces);
    const auto scale = static_cast<std::uint64_t>(fp_scale);
    // Everything wraps mod 2^64 — the identity the kernel maintains. The
    // virtual columns are part of the balance, so the full row accounts for
    // every parcel of emitted energy. Wrapping addition is associative, so
    // the summation order cannot matter and neither layout needs sorting.
    const auto row_error = [&](std::uint64_t balance, std::size_t row) {
        const std::uint64_t difference = balance - (rays_per_row[row] * scale);
        return std::min(difference, std::uint64_t{0} - difference);
    };
    std::uint64_t max_error = 0;
    if (const auto* mapped =
            std::get_if<std::span<const std::uint64_t>>(&cells)) {
        if (mapped->size() != faces * cols) {
            throw_dense_size_mismatch();
        }
        for (std::size_t row = 0; row < faces; ++row) {
            if (rays_per_row[row] == 0) {
                continue;
            }
            const std::span<const std::uint64_t> line =
                mapped->subspan(row * cols, cols);
            const std::uint64_t balance =
                std::accumulate(line.begin(), line.end(), std::uint64_t{0});
            max_error = std::max(max_error, row_error(balance, row));
        }
        return max_error;
    }
    const auto host_rows = std::get<std::span<const HostCountRow>>(cells);
    if (host_rows.size() != faces) {
        throw std::invalid_argument(
            "pycanha::radiative: the tiled cell rows do not match the "
            "face count");
    }
    for (std::size_t row = 0; row < faces; ++row) {
        if (rays_per_row[row] == 0) {
            continue;
        }
        std::uint64_t balance = 0;
        for (const auto& [column, cell] : host_rows[row]) {
            balance += cell;
        }
        max_error = std::max(max_error, row_error(balance, row));
    }
    return max_error;
}

}  // namespace pycanha::radiative::detail
