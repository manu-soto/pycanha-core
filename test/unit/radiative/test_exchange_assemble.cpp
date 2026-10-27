// White-box tests of the exchange assembly. They drive assemble_exchange
// with hand-built fixed-point cell matrices instead of a traced scene, so
// every semantic the triangulation depends on — which direction wins, what
// happens to a face that absorbs nothing, when a pair is dropped — is
// checked exactly and without a GPU.
//
// Ray counts, the fixed-point scale, areas and emissivities are all chosen
// so that every expected value is a dyadic rational, which makes these
// equality checks rather than tolerance checks. That needs the exponent to
// be 1: it is the one value for which the weight table is exact rather than
// interpolated.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "pycanha-core/radiative/materials.hpp"
#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/settings.hpp"
#include "radiative/exchange_assemble.hpp"
#include "radiative/pair_walk.hpp"

namespace rad = pycanha::radiative;
namespace detail = pycanha::radiative::detail;

namespace {

// The power of two an accumulator would have fixed on its first batch. One
// deposit unit is 1/fp_scale of a ray's energy.
constexpr double fp_scale = 1024.0;

// A fixed-point cell matrix in both layouts at once: the dense cell block
// the GPU buffer would hold, and the per-row maps the tiled path accumulates
// into. Writing through set() keeps the two in step, which is what makes the
// Dense/Tiled comparison meaningful.
class Cells {
  public:
    explicit Cells(std::size_t faces)
        : _slots(faces),
          _cols(faces + static_cast<std::size_t>(rad::num_virtual_columns)),
          _areas(faces, 1.0),
          _emissivity(faces, 1.0),
          _rays(faces, 0),
          _cells(faces * _cols, 0),
          _rows(faces) {}

    void set(std::size_t row, std::size_t column, std::uint64_t cell) {
        _cells[(row * _cols) + column] = cell;
        if (cell == 0) {
            _rows[row].erase(static_cast<std::uint32_t>(column));
        } else {
            _rows[row][static_cast<std::uint32_t>(column)] = cell;
        }
    }

    // Adds a map entry the dense block does not have. This is how a cell
    // that accumulated back to zero reaches the tiled path: absorb_block
    // only ever inserts nonzero cells, but the wrapping += that follows can
    // land on zero and the entry stays.
    void set_tiled_only(std::size_t row, std::size_t column,
                        std::uint64_t cell) {
        _rows[row][static_cast<std::uint32_t>(column)] = cell;
    }

    void set_slot(std::size_t face, double area, double emissivity,
                  std::uint64_t rays) {
        _areas[face] = area;
        _emissivity[face] = emissivity;
        _rays[face] = rays;
    }

    [[nodiscard]] std::size_t faces() const { return _slots; }
    [[nodiscard]] std::size_t space_column() const { return _slots; }
    [[nodiscard]] std::size_t lost_column() const {
        return _slots + static_cast<std::size_t>(rad::lost_column_offset);
    }
    [[nodiscard]] double area(std::size_t face) const { return _areas[face]; }
    [[nodiscard]] double emissivity(std::size_t face) const {
        return _emissivity[face];
    }
    [[nodiscard]] std::uint64_t rays(std::size_t face) const {
        return _rays[face];
    }

    [[nodiscard]] rad::ExchangeResult dense(
        const rad::AccumConfig& config,
        const detail::AssemblyTuning& tuning = {}) const {
        return detail::assemble_exchange(
            inputs(std::span<const std::uint64_t>(_cells)), config, tuning);
    }

    [[nodiscard]] rad::ExchangeResult tiled(
        const rad::AccumConfig& config,
        const detail::AssemblyTuning& tuning = {}) const {
        return detail::assemble_exchange(
            inputs(std::span<const detail::HostCountRow>(_rows)), config,
            tuning);
    }

    [[nodiscard]] std::uint64_t dense_error() const {
        return detail::exchange_conservation_error(
            std::span<const std::uint64_t>(_cells), _rays, fp_scale);
    }

    [[nodiscard]] std::uint64_t tiled_error() const {
        return detail::exchange_conservation_error(
            std::span<const detail::HostCountRow>(_rows), _rays, fp_scale);
    }

  private:
    [[nodiscard]] detail::ExchangeInputs inputs(
        detail::ExchangeCellSource cells) const {
        return detail::ExchangeInputs{.cells = cells,
                                      .areas = _areas,
                                      .emissivity = _emissivity,
                                      .rays_per_row = _rays,
                                      .fp_scale = fp_scale,
                                      .band = rad::Band::IR};
    }

    std::size_t _slots;
    std::size_t _cols;
    std::vector<double> _areas;
    std::vector<double> _emissivity;
    std::vector<std::uint64_t> _rays;
    std::vector<std::uint64_t> _cells;
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

[[nodiscard]] rad::AccumConfig ray_density(double exponent = 1.0) {
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

// Four faces whose A/N ratios put the weight at exactly 3/4 for the pair
// (0, 1) and exactly 1/2 for the pair (0, 2), plus face 3 with ZERO
// emissivity — a perfect reflector, which absorbs nothing (so no column
// deposits into it) while still emitting rays of its own. Every row balances
// exactly: whatever the real columns did not take goes to the space bucket,
// and row 2 carries a NEGATIVE lost cell, the Russian-roulette withdrawal
// that makes that column signed.
[[nodiscard]] Cells make_scene() {
    Cells cells(4);
    cells.set_slot(0, /*area=*/1.0, /*emissivity=*/1.0, /*rays=*/1024);
    cells.set_slot(1, 3.0, 0.5, 1024);
    cells.set_slot(2, 1.0, 0.25, 1024);
    cells.set_slot(3, 3.0, 0.0, 1024);

    cells.set(0, 1, 262144);                     // 256 energy units
    cells.set(0, 2, 65536);                      // 64
    cells.set(0, cells.space_column(), 720896);  // 704, closes row 0

    cells.set(1, 0, 131072);                     // 128
    cells.set(1, cells.space_column(), 917504);  // 896, closes row 1

    cells.set(2, 0, 262144);                                        // 256
    cells.set(2, cells.lost_column(), ~std::uint64_t{0} - 131071);  // -128
    cells.set(2, cells.space_column(), 917504);  // 896, closes row 2

    cells.set(3, 0, 262144);                     // 256
    cells.set(3, cells.space_column(), 786432);  // 768, closes row 3
    return cells;
}

// A wider scene for the invariance sweeps: enough faces to span several
// tiles, ray counts and areas that differ between rows so the two directions
// of a pair really are weighted differently, and one face that never emits.
[[nodiscard]] Cells make_wide_scene(std::size_t faces = 40) {
    Cells cells(faces);
    std::uint32_t state = 20260808U;
    const auto next = [&state]() {
        state = (state * 1664525U) + 1013904223U;
        return (state >> 16U) % 64U;
    };
    for (std::size_t row = 0; row < faces; ++row) {
        const std::uint64_t rays =
            row == 7 ? 0 : (std::uint64_t{1} << (10U + (row % 3)));
        cells.set_slot(row, 0.5 + (0.25 * static_cast<double>(row % 5)),
                       0.25 * static_cast<double>(1 + (row % 4)), rays);
        if (rays == 0) {
            continue;
        }
        std::uint64_t scored = 0;
        for (std::size_t column = 0; column < faces; ++column) {
            // A quarter of a deposit unit per count, so that even a row
            // where every column scores the maximum stays well inside the
            // energy its rays carried and the space bucket cannot wrap.
            const std::uint64_t cell =
                static_cast<std::uint64_t>(next()) * 256U;
            if (cell < 40U * 256U) {
                continue;  // most pairs never see each other
            }
            cells.set(row, column, cell);
            scored += cell;
        }
        cells.set(row, cells.space_column(),
                  (rays * static_cast<std::uint64_t>(fp_scale)) - scored);
    }
    return cells;
}

[[nodiscard]] std::vector<double> three_ones() {
    return std::vector<double>(3, 1.0);
}

// Assembles a three-face scene from whatever is handed in, so the validation
// tests can feed it deliberately inconsistent shapes.
rad::ExchangeResult assemble_three_faces(detail::ExchangeCellSource cells,
                                         std::span<const double> areas) {
    const std::vector<double> emissivity = three_ones();
    const std::vector<std::uint64_t> rays(3, 1024);
    return detail::assemble_exchange(
        detail::ExchangeInputs{.cells = cells,
                               .areas = areas,
                               .emissivity = emissivity,
                               .rays_per_row = rays,
                               .fp_scale = fp_scale,
                               .band = rad::Band::IR},
        rad::AccumConfig{});
}

}  // namespace

TEST_CASE("radiative exchange assemble: only the upper triangle is stored",
          "[radiative][exchange][assemble]") {
    const Cells cells = make_scene();
    const rad::ExchangeResult result = cells.dense(ray_density());

    REQUIRE(result.band == rad::Band::IR);
    REQUIRE(result.factors.rows() == static_cast<Eigen::Index>(cells.faces()));
    REQUIRE(result.factors.cols() == static_cast<Eigen::Index>(cells.faces()) +
                                         rad::num_virtual_columns);
    for (Eigen::Index row = 0; row < result.factors.rows(); ++row) {
        for (rad::SparseMatrix::InnerIterator entry(result.factors, row); entry;
             ++entry) {
            REQUIRE(entry.col() >= row);
        }
    }
}

TEST_CASE("radiative exchange assemble: the stored value is the weighted H",
          "[radiative][exchange][assemble]") {
    const Cells cells = make_scene();
    const rad::ExchangeResult result = cells.dense(ray_density());

    // Pair (0, 1): A/N is 1/1024 against 3/1024, so Y = 1/2 and the forward
    // estimate takes exactly three quarters of the weight.
    //   forward  = A_0 eps_0 B_01 = 1 * 1    * 256/1024 = 0.25
    //   backward = A_1 eps_1 B_10 = 3 * 0.5  * 128/1024 = 0.1875
    REQUIRE(value_at(result.factors, 0, 1) ==
            ((0.75 * 0.25) + (0.25 * 0.1875)));

    // Pair (0, 2): equal ray densities, so both weights are one half — and
    // here the two estimates agree exactly, so the combination is that value.
    //   forward  = 1 * 1    * 64/1024  = 0.0625
    //   backward = 1 * 0.25 * 256/1024 = 0.0625
    REQUIRE(value_at(result.factors, 0, 2) == 0.0625);

    // Buckets carry the same A eps scaling, which is what keeps the stored
    // matrix dimensionally uniform: 1 * 1 * 704/1024.
    REQUIRE(value_at(result.factors, 0,
                     static_cast<Eigen::Index>(cells.space_column())) ==
            0.6875);
    // 3 * 0.5 * 896/1024
    REQUIRE(value_at(result.factors, 1,
                     static_cast<Eigen::Index>(cells.space_column())) ==
            1.3125);
}

TEST_CASE("radiative exchange assemble: reciprocity is structural",
          "[radiative][exchange][assemble]") {
    const Cells cells = make_scene();
    const rad::ExchangeResult result = cells.dense(ray_density());

    // One stored number serves both directions, so A_i eps_i B_ij and
    // A_j eps_j B_ji cannot disagree by more than the round-off of recovering
    // them. That is a numerical statement, not the statistical one the raw
    // estimates would only satisfy on average.
    std::size_t checked = 0;
    for (Eigen::Index row = 0; row < result.factors.rows(); ++row) {
        for (rad::SparseMatrix::InnerIterator entry(result.factors, row); entry;
             ++entry) {
            if (entry.col() >= result.factors.rows()) {
                continue;  // bucket column, no partner
            }
            const auto face = static_cast<std::size_t>(row);
            const auto column = static_cast<std::size_t>(entry.col());
            const double emissive_i = cells.area(face) * cells.emissivity(face);
            const double emissive_j =
                cells.area(column) * cells.emissivity(column);
            const double forward = entry.value() / emissive_i;
            const double backward = entry.value() / emissive_j;
            REQUIRE(emissive_i * forward ==
                    Catch::Approx(emissive_j * backward).epsilon(1e-15));
            ++checked;
        }
    }
    REQUIRE(checked > 0);
}

TEST_CASE("radiative exchange assemble: a zero-emissivity face stores nothing",
          "[radiative][exchange][assemble]") {
    rad::AccumConfig config = ray_density();
    config.triangulation.keep_full_matrix = true;
    const Cells cells = make_scene();
    const rad::ExchangeResult result = cells.dense(config);

    // Face 3 absorbs nothing and emits nothing, so every coupling it takes
    // part in carries exactly zero heat — including its own row, whose rays
    // were traced but whose emissive power is zero. Nothing is stored, and
    // that is the physics rather than a loss.
    REQUIRE_FALSE(rad::SparseMatrix::InnerIterator(result.factors, 3));
    for (Eigen::Index row = 0; row < 3; ++row) {
        REQUIRE(value_at(result.factors, row, 3) == 0.0);
    }

    // The geometry behind those zeros is still available: the raw intensive
    // matrix keeps row 3 untouched, which is the only place it survives.
    REQUIRE(result.full_factors.has_value());
    if (!result.full_factors.has_value()) {
        return;  // the REQUIRE above already failed; do not dereference
    }
    const rad::SparseMatrix& full = *result.full_factors;
    REQUIRE(value_at(full, 3, 0) == 0.25);
    REQUIRE(value_at(full, 3,
                     static_cast<Eigen::Index>(cells.space_column())) == 0.75);
}

TEST_CASE("radiative exchange assemble: the full matrix is raw and intensive",
          "[radiative][exchange][assemble]") {
    const Cells cells = make_scene();
    REQUIRE_FALSE(cells.dense(ray_density()).full_factors.has_value());

    rad::AccumConfig config = ray_density();
    config.triangulation.keep_full_matrix = true;
    const rad::ExchangeResult kept = cells.dense(config);
    REQUIRE(kept.full_factors.has_value());
    if (!kept.full_factors.has_value()) {
        return;
    }
    const rad::SparseMatrix& full = *kept.full_factors;
    // Both triangles, each holding the plain deposit share of its own row:
    // B_01 = 256/1024 and B_10 = 128/1024, neither scaled by anything.
    REQUIRE(value_at(full, 0, 1) == 0.25);
    REQUIRE(value_at(full, 1, 0) == 0.125);
    // Combined with the areas and emissivities it reproduces the stored
    // triangle exactly, which is what makes the extensive form lossless.
    const double forward =
        cells.area(0) * cells.emissivity(0) * value_at(full, 0, 1);
    const double backward =
        cells.area(1) * cells.emissivity(1) * value_at(full, 1, 0);
    REQUIRE(value_at(kept.factors, 0, 1) ==
            ((0.75 * forward) + (0.25 * backward)));
}

TEST_CASE("radiative exchange assemble: buckets pass through untriangulated",
          "[radiative][exchange][assemble]") {
    const Cells cells = make_scene();
    const rad::ExchangeResult raw = cells.dense(untriangulated());
    const rad::ExchangeResult combined = cells.dense(ray_density());
    const auto space = static_cast<Eigen::Index>(cells.space_column());

    for (std::size_t face = 0; face < cells.faces(); ++face) {
        const auto row = static_cast<Eigen::Index>(face);
        REQUIRE(value_at(raw.factors, row, space) ==
                value_at(combined.factors, row, space));
    }
    // Untriangulated keeps the forward estimate as traced, so the pair (0, 1)
    // is A_0 eps_0 B_01 alone.
    REQUIRE(value_at(raw.factors, 0, 1) == 0.25);
}

TEST_CASE("radiative exchange assemble: the lost column stays signed",
          "[radiative][exchange][assemble]") {
    const Cells cells = make_scene();
    const rad::ExchangeResult result = cells.dense(ray_density());

    // Russian-roulette withdrawals wrap the cell, so it must be read back as
    // a signed integer; read as unsigned it would come out astronomically
    // large instead of slightly negative. Extensive: 1 * 0.25 * -128/1024.
    REQUIRE(value_at(result.factors, 2,
                     static_cast<Eigen::Index>(cells.lost_column())) ==
            -0.03125);
    // The fraction stays intensive and is computed from the raw cells, so
    // the extensive storage does not touch it: -128 units against the 4096
    // rays the scene emitted.
    REQUIRE(result.stats.lost_energy_fraction == -0.03125);
}

TEST_CASE("radiative exchange assemble: the residual precedes the combination",
          "[radiative][exchange][assemble]") {
    const Cells cells = make_scene();
    const rad::ExchangeResult result = cells.dense(ray_density());

    // Pair (0, 1) disagrees by |0.25 - 0.1875| / 0.25 = 0.25 before the two
    // are combined. Measured afterwards it would be identically zero and
    // would stop being the winding/parity check it exists to be.
    REQUIRE(result.stats.reciprocity_residual == 0.25);
    REQUIRE(cells.dense(untriangulated()).stats.reciprocity_residual == 0.25);
}

TEST_CASE("radiative exchange assemble: thresholding happens after combining",
          "[radiative][exchange][assemble]") {
    Cells cells(2);
    cells.set_slot(0, 1.0, 1.0, 1024);
    cells.set_slot(1, 1.0, 1.0, 1024);
    cells.set(0, 1, 1024);    // B = 1/1024, far below the threshold
    cells.set(1, 0, 524288);  // B = 1/2, far above it

    rad::AccumConfig config = ray_density();
    config.sparse_threshold = 0.01;
    const rad::ExchangeResult result = cells.dense(config);
    const rad::ExchangeResult kept = cells.dense(ray_density());

    // Thresholding each direction first would drop the forward estimate and
    // then combine the survivor against a zero, re-breaking the reciprocity
    // and biasing the entry low. Combining first keeps both.
    REQUIRE(value_at(result.factors, 0, 1) ==
            ((0.5 * (1.0 / 1024.0)) + (0.5 * 0.5)));
    // Statistics are computed before the threshold, so pruning cannot move
    // them.
    REQUIRE(result.stats.mean_stderr == kept.stats.mean_stderr);
    REQUIRE(result.stats.max_stderr == kept.stats.max_stderr);
    REQUIRE(result.stats.lost_energy_fraction ==
            kept.stats.lost_energy_fraction);
}

TEST_CASE("radiative exchange assemble: the layouts are bit-identical",
          "[radiative][exchange][assemble]") {
    const Cells cells = make_wide_scene();
    for (const rad::AccumConfig& config :
         {untriangulated(), ray_density(), ray_density(0.4)}) {
        const rad::ExchangeResult dense = cells.dense(config);
        const rad::ExchangeResult tiled = cells.tiled(config);
        REQUIRE(same_entries(tiled.factors, dense.factors));
        REQUIRE(tiled.stats.reciprocity_residual ==
                dense.stats.reciprocity_residual);
        REQUIRE(tiled.stats.mean_stderr == dense.stats.mean_stderr);
        REQUIRE(tiled.stats.lost_energy_fraction ==
                dense.stats.lost_energy_fraction);
    }
}

TEST_CASE("radiative exchange assemble: the blocked walk matches row-by-row",
          "[radiative][exchange][assemble]") {
    const Cells cells = make_wide_scene();
    const rad::AccumConfig config = ray_density(0.4);
    // Tile 0 is the plain row-by-row walk, which strides a full row per
    // transposed cell — the simple traversal the cache-blocked one has to
    // reproduce bit for bit.
    const rad::ExchangeResult naive =
        cells.dense(config, detail::AssemblyTuning{.tile = 0});
    for (const std::size_t tile :
         {std::size_t{1}, std::size_t{7}, std::size_t{16}, std::size_t{128}}) {
        const rad::ExchangeResult blocked =
            cells.dense(config, detail::AssemblyTuning{.tile = tile});
        REQUIRE(same_entries(blocked.factors, naive.factors));
        REQUIRE(blocked.stats.reciprocity_residual ==
                naive.stats.reciprocity_residual);
    }
}

TEST_CASE("radiative exchange assemble: the worker count cannot change it",
          "[radiative][exchange][assemble]") {
    const Cells cells = make_wide_scene();
    const rad::AccumConfig config = ray_density(0.4);
    const rad::ExchangeResult serial =
        cells.dense(config, detail::AssemblyTuning{.threads = 1});
    for (const unsigned threads : {2U, 3U, 8U}) {
        const detail::AssemblyTuning tuning{.tile = 7, .threads = threads};
        const rad::ExchangeResult parallel = cells.dense(config, tuning);
        REQUIRE(same_entries(parallel.factors, serial.factors));
        REQUIRE(parallel.stats.mean_stderr == serial.stats.mean_stderr);
        REQUIRE(
            same_entries(cells.tiled(config, tuning).factors, serial.factors));
    }
}

TEST_CASE("radiative exchange assemble: a cell that wrapped to zero is skipped",
          "[radiative][exchange][assemble]") {
    Cells cells = make_scene();
    // Row 1 never deposited on itself, so the dense block holds a plain zero
    // there and skips it. The tiled map still holds the key, so the sparse
    // walk has to skip it too or the two layouts stop agreeing.
    cells.set_tiled_only(1, 1, 0);

    REQUIRE(same_entries(cells.tiled(ray_density()).factors,
                         cells.dense(ray_density()).factors));
    REQUIRE(cells.tiled_error() == cells.dense_error());
}

TEST_CASE("radiative exchange assemble: energy conservation is exact",
          "[radiative][exchange][assemble]") {
    Cells cells = make_scene();
    REQUIRE(cells.dense_error() == 0);
    REQUIRE(cells.tiled_error() == 0);

    // A row whose cells no longer add up to the energy it emitted is what
    // the check exists to catch, in either direction of the wrap.
    cells.set(0, 1, 262144 + 64);
    REQUIRE(cells.dense_error() == 64);
    REQUIRE(cells.tiled_error() == 64);
    cells.set(0, 1, 262144 - 64);
    REQUIRE(cells.dense_error() == 64);
    REQUIRE(cells.tiled_error() == 64);
}

TEST_CASE("radiative exchange assemble: nothing traced yields an empty result",
          "[radiative][exchange][assemble]") {
    const std::vector<double> ones(3, 1.0);
    const std::vector<std::uint64_t> rays(3, 0);
    const std::vector<std::uint64_t> flat(
        3 * (3 + static_cast<std::size_t>(rad::num_virtual_columns)), 0);
    // fp_scale 0 is an accumulator that never ran a batch: no scale was ever
    // chosen, so there is nothing to divide by and nothing to report.
    const rad::ExchangeResult result = detail::assemble_exchange(
        detail::ExchangeInputs{.cells = std::span<const std::uint64_t>(flat),
                               .areas = ones,
                               .emissivity = ones,
                               .rays_per_row = rays,
                               .fp_scale = 0.0,
                               .band = rad::Band::Solar},
        rad::AccumConfig{});
    REQUIRE(result.band == rad::Band::Solar);
    REQUIRE(result.factors.nonZeros() == 0);
    REQUIRE(result.stats.lost_energy_fraction == 0.0);
    REQUIRE(detail::exchange_conservation_error(
                std::span<const std::uint64_t>(flat), rays, 0.0) == 0);
}

TEST_CASE("radiative exchange assemble: a mismatched cell buffer is rejected",
          "[radiative][exchange][assemble]") {
    const std::vector<std::uint64_t> too_small(3 * 3, 0);
    const std::vector<detail::HostCountRow> too_few_rows(2);
    REQUIRE_THROWS_AS(
        assemble_three_faces(std::span<const std::uint64_t>(too_small),
                             three_ones()),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        assemble_three_faces(
            std::span<const detail::HostCountRow>(too_few_rows), three_ones()),
        std::invalid_argument);
}

TEST_CASE(
    "radiative exchange assemble: mismatched per-face inputs are rejected",
    "[radiative][exchange][assemble]") {
    const std::vector<std::uint64_t> cells(
        3 * (3 + static_cast<std::size_t>(rad::num_virtual_columns)), 0);
    const std::vector<double> too_few_areas(2, 1.0);
    REQUIRE_THROWS_AS(assemble_three_faces(
                          std::span<const std::uint64_t>(cells), too_few_areas),
                      std::invalid_argument);
}

TEST_CASE("radiative exchange assemble: an invalid exponent is rejected",
          "[radiative][exchange][assemble]") {
    const Cells cells = make_scene();
    REQUIRE_THROWS_AS(cells.dense(ray_density(0.0)), std::invalid_argument);
    REQUIRE_THROWS_AS(cells.dense(ray_density(-1.0)), std::invalid_argument);
}

TEST_CASE("radiative exchange assemble: the band emissivity follows the kernel",
          "[radiative][exchange][assemble]") {
    rad::MaterialTable materials;
    materials.properties.resize(2, 6);
    materials.properties.row(0) << 0.8F, 0.0F, 0.0F, 0.3F, 0.0F, 0.0F;
    materials.properties.row(1) << 0.1F, 0.0F, 0.0F, 0.9F, 0.0F, 0.0F;
    materials.face_material.resize(3);
    materials.face_material << 0, 1, -1;
    materials.face_active.resize(3);
    materials.face_active.setConstant(true);

    const std::vector<double> infrared =
        detail::band_emissivity(materials, rad::Band::IR);
    const std::vector<double> solar =
        detail::band_emissivity(materials, rad::Band::Solar);
    REQUIRE(infrared[0] == Catch::Approx(0.8));
    REQUIRE(infrared[1] == Catch::Approx(0.1));
    REQUIRE(solar[0] == Catch::Approx(0.3));
    REQUIRE(solar[1] == Catch::Approx(0.9));
    // No material assigned is a blackbody, matching the kernel's fallback.
    REQUIRE(infrared[2] == 1.0);
    REQUIRE(solar[2] == 1.0);
}

TEST_CASE("radiative exchange assemble: the wide scene balances too",
          "[radiative][exchange][assemble]") {
    // The invariance sweeps only compare runs against each other, so a scene
    // whose rows did not balance would sail through them. It is the closure
    // projection that needs the scene to be physical, so pin it here.
    const Cells cells = make_wide_scene();
    REQUIRE(cells.dense_error() == 0);
    REQUIRE(cells.tiled_error() == 0);
}

TEST_CASE("radiative exchange assemble: least squares closes the energy",
          "[radiative][exchange][assemble][closure]") {
    rad::AccumConfig config;
    config.triangulation.mode = rad::TriangulationMode::ConstrainedLeastSquares;
    const Cells cells = make_wide_scene();
    const rad::ExchangeResult projected = cells.dense(config);
    const auto faces = static_cast<Eigen::Index>(cells.faces());

    // A row's closure target is the energy it actually emitted, A_i eps_i,
    // and the projection restores it exactly across every column the row
    // takes part in — its own entries plus the ones stored above it.
    for (std::size_t face = 0; face < cells.faces(); ++face) {
        if (cells.rays(face) == 0 || !(cells.emissivity(face) > 0.0)) {
            continue;  // emitted no energy, so there is nothing to close
        }
        const auto row = static_cast<Eigen::Index>(face);
        double total = 0.0;
        for (rad::SparseMatrix::InnerIterator entry(projected.factors, row);
             entry; ++entry) {
            total += entry.value();
        }
        for (Eigen::Index above = 0; above < row; ++above) {
            for (rad::SparseMatrix::InnerIterator entry(projected.factors,
                                                        above);
                 entry; ++entry) {
                if (entry.col() == row && row < faces) {
                    total += entry.value();
                }
            }
        }
        REQUIRE(total ==
                Catch::Approx(cells.area(face) * cells.emissivity(face))
                    .epsilon(1e-9));
    }
}

TEST_CASE("radiative exchange assemble: least squares is layout invariant",
          "[radiative][exchange][assemble][closure]") {
    rad::AccumConfig config;
    config.triangulation.mode = rad::TriangulationMode::ConstrainedLeastSquares;
    const Cells cells = make_wide_scene();
    const rad::ExchangeResult serial =
        cells.dense(config, detail::AssemblyTuning{.threads = 1});
    REQUIRE(same_entries(cells.tiled(config).factors, serial.factors));
    for (const unsigned threads : {2U, 3U, 8U}) {
        const detail::AssemblyTuning tuning{.tile = 7, .threads = threads};
        REQUIRE(
            same_entries(cells.dense(config, tuning).factors, serial.factors));
    }
}

TEST_CASE("radiative exchange assemble: a zero-emissivity row is unconstrained",
          "[radiative][exchange][assemble][closure]") {
    rad::AccumConfig config;
    config.triangulation.mode = rad::TriangulationMode::ConstrainedLeastSquares;
    // Face 3 emits rays but has no emissivity, so it transports no energy and
    // there is nothing to close. It must not acquire an equation, and it must
    // still store nothing.
    const Cells cells = make_scene();
    const rad::ExchangeResult projected = cells.dense(config);
    REQUIRE_FALSE(rad::SparseMatrix::InnerIterator(projected.factors, 3));
}

TEST_CASE("radiative tiled host rows: sorted merges match the map semantics",
          "[radiative][assembly]") {
    detail::HostCountRow row;
    // A new row: the block's non-zeros in column order.
    const std::vector<std::uint64_t> first{0, 5, 0, 7, 0, 1};
    row.add_dense(std::span<const std::uint64_t>(first));
    REQUIRE(row.size() == 3);
    // A second batch: adds to columns 1 and 5, inserts 0 and 4, and wraps
    // column 3 to zero (the entry stays, as the map's += left it).
    const std::vector<std::uint64_t> second{2, 1, 0, ~std::uint64_t{6}, 4, 9};
    row.add_dense(std::span<const std::uint64_t>(second));
    const std::vector<detail::HostCountRow::value_type> expected{
        {0, 2}, {1, 6}, {3, 0}, {4, 4}, {5, 10}};
    REQUIRE(std::vector<detail::HostCountRow::value_type>(row.begin(),
                                                           row.end()) ==
            expected);
    // u32 cells (the view-factor counts) and the map-like operations.
    const std::vector<std::uint32_t> counts{0, 0, 3};
    row.add_dense(std::span<const std::uint32_t>(counts));
    REQUIRE(row[2] == 3);
    row[7] += 1;
    row.erase(0);
    REQUIRE(row.size() == 6);
    REQUIRE(row.begin()->first == 1);
    row.clear();
    REQUIRE(row.empty());
}
