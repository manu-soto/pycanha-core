// White-box tests of the view-factor assembly. They drive assemble_vf with
// hand-built count matrices instead of a traced scene, so every semantic the
// triangulation depends on — which direction wins, when a pair is dropped,
// what closure means — is checked exactly and without a GPU.
//
// The counts are chosen so that every quantity is a dyadic rational: ray
// counts are powers of two and areas are multiples of 1/4, which makes the
// expected values exactly representable and lets these be equality checks
// rather than tolerance checks.

#include <bit>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/settings.hpp"
#include "radiative/pair_walk.hpp"
#include "radiative/vf_assemble.hpp"

namespace rad = pycanha::radiative;
namespace detail = pycanha::radiative::detail;

namespace {

// A count matrix in both layouts at once: the dense cell block the GPU
// buffer would hold, and the per-row maps the tiled path accumulates into.
// Writing through set() keeps the two in step, which is what makes the
// Dense/Tiled comparison meaningful.
class Counts {
  public:
    explicit Counts(std::size_t faces)
        : _slots(faces),
          _cols(faces + static_cast<std::size_t>(rad::num_virtual_columns)),
          _areas(faces, 1.0),
          _rays(faces, 0),
          _cells(faces * _cols, 0),
          _rows(faces) {}

    void set(std::size_t row, std::size_t column, std::uint32_t count) {
        _cells[(row * _cols) + column] = count;
        if (count == 0) {
            _rows[row].erase(static_cast<std::uint32_t>(column));
        } else {
            _rows[row][static_cast<std::uint32_t>(column)] = count;
        }
    }

    [[nodiscard]] std::uint32_t at(std::size_t row, std::size_t column) const {
        return _cells[(row * _cols) + column];
    }

    void set_area(std::size_t face, double area) { _areas[face] = area; }
    void set_rays(std::size_t face, std::uint64_t rays) { _rays[face] = rays; }

    [[nodiscard]] std::size_t faces() const { return _slots; }
    [[nodiscard]] std::size_t space_column() const { return _slots; }
    [[nodiscard]] double area(std::size_t face) const { return _areas[face]; }
    [[nodiscard]] std::uint64_t rays(std::size_t face) const {
        return _rays[face];
    }

    [[nodiscard]] rad::VfResult dense(
        const rad::AccumConfig& config,
        const detail::AssemblyTuning& tuning = {}) const {
        return detail::assemble_vf(std::span<const std::uint32_t>(_cells),
                                   _areas, _rays, config, tuning);
    }

    [[nodiscard]] rad::VfResult tiled(
        const rad::AccumConfig& config,
        const detail::AssemblyTuning& tuning = {}) const {
        return detail::assemble_vf(std::span<const detail::HostCountRow>(_rows),
                                   _areas, _rays, config, tuning);
    }

  private:
    std::size_t _slots;
    std::size_t _cols;
    std::vector<double> _areas;
    std::vector<std::uint64_t> _rays;
    std::vector<std::uint32_t> _cells;
    std::vector<detail::HostCountRow> _rows;
};

[[nodiscard]] double value_at(const rad::SparseMatrix& matrix, Eigen::Index row,
                              Eigen::Index column) {
    for (rad::SparseMatrix::InnerIterator entry(matrix, row); entry; ++entry) {
        if (entry.col() == column) {
            return entry.value();
        }
    }
    return 0.0;
}

[[nodiscard]] bool same_entries(const rad::SparseMatrix& lhs,
                                const rad::SparseMatrix& rhs) {
    using Iterator = rad::SparseMatrix::InnerIterator;
    if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols() ||
        lhs.nonZeros() != rhs.nonZeros()) {
        return false;
    }
    for (Eigen::Index row = 0; row < lhs.rows(); ++row) {
        Iterator left(lhs, row);
        Iterator right(rhs, row);
        for (; left && right; ++left, ++right) {
            if (left.col() != right.col() || left.value() != right.value()) {
                return false;
            }
        }
        if (left || right) {
            return false;
        }
    }
    return true;
}

// Enough faces to span several tiles at the sizes the tests sweep, with a
// deliberately non-emitting face (7) and ray counts that differ between
// rows so the two directions of a pair really are weighted differently.
// Every row closes exactly: whatever the real columns did not take goes to
// the space bucket.
[[nodiscard]] Counts make_scene(std::size_t faces = 40) {
    Counts counts(faces);
    std::uint32_t state = 20260807U;
    const auto next = [&state]() {
        state = (state * 1664525U) + 1013904223U;
        return (state >> 16U) % 50U;
    };
    for (std::size_t row = 0; row < faces; ++row) {
        counts.set_area(row, 0.5 + (0.25 * static_cast<double>(row % 5)));
        const std::uint64_t rays =
            row == 7 ? 0 : (std::uint64_t{1} << (14U + (row % 3)));
        counts.set_rays(row, rays);
        if (rays == 0) {
            continue;
        }
        std::uint64_t scored = 0;
        for (std::size_t column = 0; column < faces; ++column) {
            const std::uint32_t count = next();
            if (count < 40) {
                continue;  // most pairs never see each other
            }
            counts.set(row, column, count);
            scored += count;
        }
        counts.set(row, counts.space_column(),
                   static_cast<std::uint32_t>(rays - scored));
    }
    return counts;
}

[[nodiscard]] rad::AccumConfig ray_density(double exponent = 0.4) {
    rad::AccumConfig config;
    config.triangulation.mode = rad::TriangulationMode::RayDensity;
    config.triangulation.exponent = exponent;
    return config;
}

[[nodiscard]] rad::AccumConfig untriangulated() {
    rad::AccumConfig config;
    config.triangulation.mode = rad::TriangulationMode::None;
    return config;
}

[[nodiscard]] rad::AccumConfig least_squares() {
    rad::AccumConfig config;
    config.triangulation.mode = rad::TriangulationMode::ConstrainedLeastSquares;
    return config;
}

// Sum of a stored row over EVERY column it takes part in: the entries stored
// in the row itself plus, because only the upper triangle is kept, the
// entries stored above it at this row's column.
[[nodiscard]] double closure_of(const rad::SparseMatrix& matrix,
                                Eigen::Index face, Eigen::Index faces) {
    double total = 0.0;
    for (rad::SparseMatrix::InnerIterator entry(matrix, face); entry; ++entry) {
        total += entry.value();
    }
    for (Eigen::Index row = 0; row < face; ++row) {
        for (rad::SparseMatrix::InnerIterator entry(matrix, row); entry;
             ++entry) {
            if (entry.col() == face && face < faces) {
                total += entry.value();
            }
        }
    }
    return total;
}

}  // namespace

TEST_CASE("radiative assemble: only the upper triangle is stored",
          "[radiative][vf][assemble]") {
    const Counts counts = make_scene();
    const rad::VfResult result = counts.dense(ray_density());

    REQUIRE(result.vf.rows() == static_cast<Eigen::Index>(counts.faces()));
    REQUIRE(result.vf.cols() == static_cast<Eigen::Index>(counts.faces()) +
                                    rad::num_virtual_columns);
    for (Eigen::Index row = 0; row < result.vf.rows(); ++row) {
        for (rad::SparseMatrix::InnerIterator entry(result.vf, row); entry;
             ++entry) {
            REQUIRE(entry.col() >= row);
        }
    }
}

TEST_CASE("radiative assemble: the stored value is reciprocal by construction",
          "[radiative][vf][assemble]") {
    const Counts counts = make_scene();
    const rad::VfResult result = counts.dense(ray_density());

    // Reciprocity is structural: one stored number serves both directions,
    // so A_i F_ij and A_j F_ji cannot disagree by more than the round-off of
    // recovering them. That is a numerical statement, not the statistical
    // one the raw estimates would only satisfy on average.
    std::size_t checked = 0;
    for (Eigen::Index row = 0; row < result.vf.rows(); ++row) {
        for (rad::SparseMatrix::InnerIterator entry(result.vf, row); entry;
             ++entry) {
            if (entry.col() >= result.vf.rows()) {
                continue;  // bucket column, no partner
            }
            const auto column = static_cast<std::size_t>(entry.col());
            const auto face = static_cast<std::size_t>(row);
            const double forward = entry.value() / counts.area(face);
            const double backward = entry.value() / counts.area(column);
            REQUIRE(
                counts.area(face) * forward ==
                Catch::Approx(counts.area(column) * backward).epsilon(1e-15));
            ++checked;
        }
    }
    REQUIRE(checked > 0);
}

TEST_CASE("radiative assemble: closure is the raw estimate over all columns",
          "[radiative][vf][assemble]") {
    const Counts counts = make_scene();
    for (const rad::AccumConfig& config : {untriangulated(), ray_density()}) {
        const rad::VfResult result = counts.dense(config);
        for (std::size_t face = 0; face < counts.faces(); ++face) {
            const auto row = static_cast<Eigen::Index>(face);
            // Closure is computed from the raw counts before anything is
            // combined or pruned, so it is exactly 1 for every emitting row
            // and untouched by the triangulation mode.
            REQUIRE(result.row_sums(row) ==
                    (counts.rays(face) > 0 ? 1.0 : 0.0));
        }
    }
}

TEST_CASE("radiative assemble: bucket columns pass through untriangulated",
          "[radiative][vf][assemble]") {
    const Counts counts = make_scene();
    const rad::VfResult raw = counts.dense(untriangulated());
    const rad::VfResult combined = counts.dense(ray_density());
    const auto space = static_cast<Eigen::Index>(counts.space_column());

    for (std::size_t face = 0; face < counts.faces(); ++face) {
        const auto row = static_cast<Eigen::Index>(face);
        const double expected =
            counts.rays(face) == 0 ? 0.0
                                   : counts.area(face) *
                                         static_cast<double>(counts.at(
                                             face, counts.space_column())) /
                                         static_cast<double>(counts.rays(face));
        REQUIRE(value_at(raw.vf, row, space) == expected);
        REQUIRE(value_at(combined.vf, row, space) == expected);
    }
}

TEST_CASE("radiative assemble: equal ray densities average the two estimates",
          "[radiative][vf][assemble]") {
    Counts counts(2);
    counts.set_area(0, 1.0);
    counts.set_area(1, 1.0);
    counts.set_rays(0, 1024);
    counts.set_rays(1, 1024);
    counts.set(0, 1, 256);
    counts.set(1, 0, 512);

    // Identical areas and ray counts put Y at exactly zero, and 0^n vanishes
    // for every valid n, so both weights are one half.
    const rad::VfResult result = counts.dense(ray_density());
    REQUIRE(value_at(result.vf, 0, 1) == ((0.25 + 0.5) / 2.0));
}

TEST_CASE("radiative assemble: area weights do not follow the allocation",
          "[radiative][vf][assemble]") {
    // The same pair estimates with face 0 traced at four times the rays of
    // face 1, then at the same rays; the two directions disagree.
    //   forward  = A_0 F_01 = 1 * 0.25   = 0.25
    //   backward = A_1 F_10 = 3 * 0.0625 = 0.1875
    const auto pair = [](std::uint64_t rays_0) {
        Counts counts(2);
        counts.set_area(0, 1.0);
        counts.set_area(1, 3.0);
        counts.set_rays(0, rays_0);
        counts.set_rays(1, 1024);
        counts.set(0, 1, static_cast<std::uint32_t>(rays_0 / 4));
        counts.set(1, 0, 64);
        return counts;
    };
    // Exponent 1, so that the weight is exact (inverse-variance).
    rad::AccumConfig by_area = ray_density(1.0);
    by_area.triangulation.area_weights = true;
    // Areas 1 and 3: Y = 1/2, the forward weight 3/4, whatever the rays; on
    // the uniform trace the actual counts give the same. The allocated
    // trace's mean of 2560 rays makes A/N non-dyadic: compared as a real.
    const double expected = (0.75 * 0.25) + (0.25 * 0.1875);
    REQUIRE(value_at(pair(4096).dense(by_area).vf, 0, 1) ==
            Catch::Approx(expected).epsilon(1e-15));
    REQUIRE(value_at(pair(1024).dense(by_area).vf, 0, 1) == expected);
    REQUIRE(same_entries(pair(1024).dense(by_area).vf,
                         pair(1024).dense(ray_density(1.0)).vf));
    // A/N of 1/4096 against 3/1024: Y = 11/13, the forward weight 12/13.
    REQUIRE(value_at(pair(4096).dense(ray_density(1.0)).vf, 0, 1) ==
            Catch::Approx(((12.0 * 0.25) + 0.1875) / 13.0).epsilon(1e-15));
    // The default exponent bit for bit on the uniform trace as well.
    rad::AccumConfig by_area_default = ray_density();
    by_area_default.triangulation.area_weights = true;
    REQUIRE(same_entries(pair(1024).dense(by_area_default).vf,
                         pair(1024).dense(ray_density()).vf));
}

TEST_CASE(
    "radiative assemble: an exponent of one is inverse-variance weighting",
    "[radiative][vf][assemble]") {
    // X_i = 1/2 (1 + Y) collapses to v/(u+v), the minimum-variance weight.
    // Powers of two make the algebra exact, so this is an equality.
    const detail::WeightTable table(1.0);
    REQUIRE(table.forward_weight(1.0, 3.0) == 0.75);
    REQUIRE(table.forward_weight(2.0, 2.0) == 0.5);
    REQUIRE(table.forward_weight(3.0, 1.0) == 0.25);
    // Away from exactly representable ratios the two expressions round
    // differently, so compare them as the reals they are.
    for (const double v : {0.1, 0.7, 1.3, 9.0}) {
        REQUIRE(table.forward_weight(1.0, v) ==
                Catch::Approx(v / (1.0 + v)).epsilon(1e-15));
    }
}

TEST_CASE("radiative assemble: the tabulated weight tracks std::pow",
          "[radiative][vf][assemble]") {
    for (const double exponent : {0.2, 0.4, 0.75, 2.0}) {
        const detail::WeightTable table(exponent);
        for (int step = 0; step <= 400; ++step) {
            const double ratio = static_cast<double>(step) / 400.0;
            const double u = 1.0;
            const double v = (1.0 + ratio) / (1.0 - (0.5 * ratio));
            REQUIRE(table.forward_weight(u, v) ==
                    Catch::Approx(detail::WeightTable::naive_forward_weight(
                                      u, v, exponent))
                        .margin(1e-7));
        }
    }
}

TEST_CASE("radiative assemble: a non-emitting face still gets its coupling",
          "[radiative][vf][assemble]") {
    Counts counts(2);
    counts.set_area(0, 2.0);
    counts.set_area(1, 4.0);
    counts.set_rays(0, 0);  // never emits: the environment/collector case
    counts.set_rays(1, 1024);
    counts.set(1, 0, 256);

    // With no rays of its own face 0 has an infinite A/N, so the weight
    // passes entirely to the reverse direction and the coupling is sourced
    // from it alone. No division by zero and nothing dropped.
    const rad::VfResult result = counts.dense(ray_density());
    REQUIRE(value_at(result.vf, 0, 1) == 4.0 * 256.0 / 1024.0);
    REQUIRE(result.row_sums(0) == 0.0);
    REQUIRE(result.row_sums(1) == 0.25);
}

TEST_CASE("radiative assemble: thresholding happens after combining",
          "[radiative][vf][assemble]") {
    Counts counts(2);
    counts.set_area(0, 1.0);
    counts.set_area(1, 1.0);
    counts.set_rays(0, 1024);
    counts.set_rays(1, 1024);
    counts.set(0, 1, 1);    // F = 1/1024, far below the threshold
    counts.set(1, 0, 512);  // F = 1/2, far above it

    rad::AccumConfig config = ray_density();
    config.sparse_threshold = 0.01;
    const rad::VfResult result = counts.dense(config);

    // Thresholding each direction first would drop the forward estimate and
    // then combine the survivor against a zero, re-breaking the reciprocity
    // and biasing the entry low. Combining first keeps both.
    REQUIRE(result.vf.nonZeros() >= 1);
    REQUIRE(value_at(result.vf, 0, 1) ==
            ((0.5 * (1.0 / 1024.0)) + (0.5 * 0.5)));
}

TEST_CASE("radiative assemble: the full matrix is optional and untriangulated",
          "[radiative][vf][assemble]") {
    const Counts counts = make_scene();
    REQUIRE_FALSE(counts.dense(ray_density()).full_vf.has_value());

    rad::AccumConfig config = ray_density();
    config.triangulation.keep_full_matrix = true;
    const rad::VfResult kept = counts.dense(config);
    REQUIRE(kept.full_vf.has_value());
    if (!kept.full_vf.has_value()) {
        return;  // the REQUIRE above already failed; do not dereference
    }

    // Its upper triangle is exactly what an untriangulated run stores, and
    // its lower triangle carries the second, independent estimate of the
    // same coupling.
    const rad::VfResult raw = counts.dense(untriangulated());
    const rad::SparseMatrix& full = *kept.full_vf;
    for (Eigen::Index row = 0; row < raw.vf.rows(); ++row) {
        for (rad::SparseMatrix::InnerIterator entry(raw.vf, row); entry;
             ++entry) {
            REQUIRE(value_at(full, row, entry.col()) == entry.value());
        }
    }
    REQUIRE(value_at(full, 1, 0) == counts.area(1) *
                                        static_cast<double>(counts.at(1, 0)) /
                                        static_cast<double>(counts.rays(1)));
}

TEST_CASE("radiative assemble: the blocked walk matches the row-by-row walk",
          "[radiative][vf][assemble]") {
    const Counts counts = make_scene();
    const rad::AccumConfig config = ray_density();
    // Tile 0 is the plain row-by-row walk, which strides a full row per
    // transposed cell — the simple traversal the cache-blocked one has to
    // reproduce bit for bit.
    const rad::VfResult naive =
        counts.dense(config, detail::AssemblyTuning{.tile = 0});
    for (const std::size_t tile :
         {std::size_t{1}, std::size_t{7}, std::size_t{16}, std::size_t{128}}) {
        const rad::VfResult blocked =
            counts.dense(config, detail::AssemblyTuning{.tile = tile});
        REQUIRE(same_entries(blocked.vf, naive.vf));
        REQUIRE(blocked.stats.reciprocity_residual ==
                naive.stats.reciprocity_residual);
    }
}

TEST_CASE("radiative assemble: the sparse merge matches the dense walk",
          "[radiative][vf][assemble]") {
    const Counts counts = make_scene();
    for (const rad::AccumConfig& config : {untriangulated(), ray_density()}) {
        const rad::VfResult dense = counts.dense(config);
        const rad::VfResult tiled = counts.tiled(config);
        REQUIRE(same_entries(tiled.vf, dense.vf));
        REQUIRE(tiled.row_sums == dense.row_sums);
        REQUIRE(tiled.stats.reciprocity_residual ==
                dense.stats.reciprocity_residual);
        REQUIRE(tiled.stats.mean_stderr == dense.stats.mean_stderr);
    }
}

TEST_CASE("radiative assemble: the worker count cannot change the result",
          "[radiative][vf][assemble]") {
    const Counts counts = make_scene();
    const rad::AccumConfig config = ray_density();
    const rad::VfResult serial =
        counts.dense(config, detail::AssemblyTuning{.threads = 1});
    for (const unsigned threads : {2U, 3U, 8U}) {
        const detail::AssemblyTuning tuning{.tile = 7, .threads = threads};
        const rad::VfResult parallel = counts.dense(config, tuning);
        REQUIRE(same_entries(parallel.vf, serial.vf));
        REQUIRE(parallel.row_sums == serial.row_sums);
        REQUIRE(parallel.stats.mean_stderr == serial.stats.mean_stderr);
        REQUIRE(counts.tiled(config, tuning).row_sums == serial.row_sums);
    }
}

TEST_CASE("radiative assemble: the tabulated and naive weights agree in place",
          "[radiative][vf][assemble]") {
    const Counts counts = make_scene();
    const rad::AccumConfig config = ray_density();
    const rad::VfResult tabulated =
        counts.dense(config, detail::AssemblyTuning{.tabulated_weight = true});
    const rad::VfResult naive =
        counts.dense(config, detail::AssemblyTuning{.tabulated_weight = false});

    REQUIRE(tabulated.vf.nonZeros() == naive.vf.nonZeros());
    for (Eigen::Index row = 0; row < tabulated.vf.rows(); ++row) {
        rad::SparseMatrix::InnerIterator left(tabulated.vf, row);
        rad::SparseMatrix::InnerIterator right(naive.vf, row);
        for (; left && right; ++left, ++right) {
            REQUIRE(left.col() == right.col());
            REQUIRE(left.value() ==
                    Catch::Approx(right.value()).epsilon(1e-6).margin(1e-12));
        }
    }
}

TEST_CASE("radiative assemble: the weight table is the same on every platform",
          "[radiative][vf][assemble]") {
    // The tabulated |Y|^n is the only transcendental in an otherwise
    // IEEE-exact pipeline, and std::pow is not bit-specified between C
    // libraries. Rounding every table entry to 24 significand bits is what
    // makes two implementations agree — they differ by an ulp or so, which
    // is far finer than that quantum — but only a golden digest can show it,
    // because a single drifting bit anywhere in the table changes the
    // assembled matrix. Recompute this constant if the table's size,
    // quantization or construction ever changes; do NOT relax it.
    std::uint64_t digest = 1469598103934665603ULL;  // FNV-1a offset basis
    const auto absorb = [&digest](double value) {
        auto bits = std::bit_cast<std::uint64_t>(value);
        for (int byte = 0; byte < 8; ++byte) {
            digest = (digest ^ (bits & 0xFFU)) * 1099511628211ULL;
            bits >>= 8U;
        }
    };
    for (const double exponent : {0.4, 0.75, 2.0}) {
        const detail::WeightTable table(exponent);
        // Ratios spanning many binary exponents of |Y|, so the sweep reaches
        // both the mantissa table and the power-of-two scale table.
        for (int step = 1; step <= 2000; ++step) {
            const auto ratio = static_cast<double>(step);
            absorb(table.forward_weight(1.0, ratio / 2000.0));
            absorb(table.forward_weight(1.0, 1.0 + ratio));
            absorb(table.forward_weight(ratio, 1.0));
        }
    }
    REQUIRE(digest == 0xf27530a5a7be38a9ULL);
}

TEST_CASE("radiative assemble: an invalid exponent is rejected",
          "[radiative][vf][assemble]") {
    const Counts counts = make_scene(4);
    REQUIRE_THROWS_AS(counts.dense(ray_density(0.0)), std::invalid_argument);
    REQUIRE_THROWS_AS(counts.dense(ray_density(-1.0)), std::invalid_argument);
}

TEST_CASE("radiative assemble: least squares closes every row exactly",
          "[radiative][vf][assemble][closure]") {
    const Counts counts = make_scene();
    const rad::VfResult combined = counts.dense(ray_density());
    const rad::VfResult projected = counts.dense(least_squares());
    const auto faces = static_cast<Eigen::Index>(counts.faces());

    // Weighting each pair on its own leaves rows no longer summing to their
    // own area; the projection is the smallest weighted move that restores
    // that, and it restores it exactly rather than approximately.
    bool ray_density_missed = false;
    for (std::size_t face = 0; face < counts.faces(); ++face) {
        const auto row = static_cast<Eigen::Index>(face);
        if (counts.rays(face) == 0) {
            continue;  // emitted nothing, so there is nothing to close
        }
        if (closure_of(combined.vf, row, faces) !=
            Catch::Approx(counts.area(face)).epsilon(1e-9)) {
            ray_density_missed = true;
        }
        REQUIRE(closure_of(projected.vf, row, faces) ==
                Catch::Approx(counts.area(face)).epsilon(1e-9));
    }
    REQUIRE(ray_density_missed);
}

TEST_CASE("radiative assemble: least squares keeps reciprocity structural",
          "[radiative][vf][assemble][closure]") {
    const Counts counts = make_scene();
    const rad::VfResult projected = counts.dense(least_squares());
    // Closure is imposed on the SHARED entry, so it cannot pull the two
    // directions apart: only the upper triangle exists to be stored.
    for (Eigen::Index row = 0; row < projected.vf.rows(); ++row) {
        for (rad::SparseMatrix::InnerIterator entry(projected.vf, row); entry;
             ++entry) {
            REQUIRE(entry.col() >= row);
        }
    }
    // Closure is a statement about the raw estimate and is reported from it,
    // so the projection must not have moved what row_sums means.
    REQUIRE(projected.row_sums == counts.dense(ray_density()).row_sums);
}

TEST_CASE("radiative assemble: the least-squares correction is exact",
          "[radiative][vf][assemble][closure]") {
    // Two unit faces that see each other and space, with disagreeing
    // estimates: forward says 1/4 of row 0 reaches face 1, backward says 1/2
    // of row 1 reaches face 0. Equal ray densities put the unconstrained
    // combination at 3/8, which leaves row 0 over-full and row 1 short.
    Counts counts(2);
    counts.set_area(0, 1.0);
    counts.set_area(1, 1.0);
    counts.set_rays(0, 1024);
    counts.set_rays(1, 1024);
    counts.set(0, 1, 256);
    counts.set(0, counts.space_column(), 768);
    counts.set(1, 0, 512);
    counts.set(1, counts.space_column(), 512);

    const rad::VfResult projected = counts.dense(least_squares());
    // Hand-solved: with variances proportional to the estimates, the
    // two-equation dual has the coupling land on 5/13 and both space columns
    // on 8/13. Every row then closes to exactly one.
    REQUIRE(value_at(projected.vf, 0, 1) ==
            Catch::Approx(5.0 / 13.0).epsilon(1e-12));
    REQUIRE(value_at(projected.vf, 0, 2) ==
            Catch::Approx(8.0 / 13.0).epsilon(1e-12));
    REQUIRE(value_at(projected.vf, 1, 2) ==
            Catch::Approx(8.0 / 13.0).epsilon(1e-12));
}

TEST_CASE("radiative assemble: a non-emitting row carries no constraint",
          "[radiative][vf][assemble][closure]") {
    Counts counts(2);
    counts.set_area(0, 2.0);
    counts.set_area(1, 4.0);
    counts.set_rays(0, 0);  // never emits: the environment/collector case
    counts.set_rays(1, 1024);
    counts.set(1, 0, 256);
    counts.set(1, counts.space_column(), 768);

    // Row 0 emitted nothing, so there is no closure to impose on it and the
    // system must not acquire an equation for it. Row 1 still closes.
    const rad::VfResult projected = counts.dense(least_squares());
    REQUIRE(closure_of(projected.vf, 1, 2) ==
            Catch::Approx(counts.area(1)).epsilon(1e-12));
}

TEST_CASE("radiative assemble: a singular closure system is reported",
          "[radiative][vf][assemble][closure]") {
    // Two unequal faces in a closed enclosure: every ray of each lands on the
    // other and nothing reaches space. The two closure equations then both
    // constrain the single shared entry, to different values, so no matrix
    // satisfies them and the dual system is rank deficient.
    Counts counts(2);
    counts.set_area(0, 1.0);
    counts.set_area(1, 2.0);
    counts.set_rays(0, 1024);
    counts.set_rays(1, 1024);
    counts.set(0, 1, 1024);
    counts.set(1, 0, 1024);

    REQUIRE_THROWS_AS(counts.dense(least_squares()), std::runtime_error);
    // The same model is perfectly fine for the independent-pair mode.
    REQUIRE_NOTHROW(counts.dense(ray_density()));
}

TEST_CASE("radiative assemble: least squares is layout and thread invariant",
          "[radiative][vf][assemble][closure]") {
    const Counts counts = make_scene();
    const rad::AccumConfig config = least_squares();
    const rad::VfResult serial =
        counts.dense(config, detail::AssemblyTuning{.threads = 1});
    REQUIRE(same_entries(counts.tiled(config).vf, serial.vf));
    for (const unsigned threads : {2U, 3U, 8U}) {
        const detail::AssemblyTuning tuning{.tile = 7, .threads = threads};
        REQUIRE(same_entries(counts.dense(config, tuning).vf, serial.vf));
        REQUIRE(same_entries(counts.tiled(config, tuning).vf, serial.vf));
    }
}
