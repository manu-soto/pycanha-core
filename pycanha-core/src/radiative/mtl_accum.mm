#include "mtl_accum.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

#include "exchange_assemble.hpp"
#include "mtl_device.hpp"
#include "mtl_scene.hpp"
#include "pair_walk.hpp"
#include "pycanha-core/radiative/materials.hpp"
#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/scene.hpp"
#include "pycanha-core/radiative/settings.hpp"
#include "vf_assemble.hpp"

namespace pycanha::radiative::detail {

namespace {

// Largest power of two S with rays_per_face * S <= 2^58. Powers of two are
// exact in f32 and make to_fp(1.0) == S on the GPU, so a row's expected
// balance is the exact integer rays * S; the 2^58 budget leaves 32x
// cumulative-ray headroom before a u64 row balance could overflow, with a
// deposit resolution (1/S) still far below Monte-Carlo noise.
[[nodiscard]] double select_fp_scale(std::uint64_t rays_per_face) {
    constexpr int budget_bits = 58;
    const int rays_bits = rays_per_face > 1 ? std::bit_width(rays_per_face - 1) : 0;
    return std::ldexp(1.0, std::max(budget_bits - rays_bits, 0));
}

// Cumulative rays_per_face a fixed-point accumulator can absorb before its
// u64 balances could overflow.
[[nodiscard]] std::uint64_t max_cumulative_rays(double fp_scale) {
    return (std::uint64_t{1} << 63U) / static_cast<std::uint64_t>(fp_scale);
}

// Shared storage on unified memory: the CPU writes the cells directly, and
// because every submission is waited for there is nothing to flush or
// invalidate around it.
void clear_cells(const GpuBuffer& buffer) { std::memset(checked_mapped(buffer), 0, buffer.size); }

}  // namespace

VfAccumImpl::VfAccumImpl(SceneImpl& scene, const AccumConfig& config)
    : _scene(scene), _config(config) {
    const std::uint64_t faces = _scene.num_faces();
    std::uint64_t buffer_rows = faces;
    if (_config.layout == AccumLayout::Tiled) {
        if (_config.tile_rows == 0) {
            throw std::invalid_argument("pycanha::radiative: the Tiled layout needs tile_rows > 0");
        }
        _config.tile_rows =
            static_cast<std::uint32_t>(std::min<std::uint64_t>(_config.tile_rows, faces));
        buffer_rows = _config.tile_rows;
        _host_rows.resize(faces);
    }
    _counts = _scene.create_buffer(buffer_rows * matrix_columns(faces) * sizeof(std::uint32_t));
    _rays_per_row.assign(faces, 0);
    reset();
}

VfAccumImpl::~VfAccumImpl() { _scene.destroy_buffer(_counts); }

void VfAccumImpl::reset() {
    clear_block_scratch();
    for (auto& row : _host_rows) {
        row.clear();
    }
    std::ranges::fill(_rays_per_row, 0);
    _rays_per_face = 0;
    _total_rays = 0;
}

void VfAccumImpl::clear_block_scratch() { clear_cells(_counts); }

void VfAccumImpl::absorb_block(std::span<const std::uint32_t> block_emitters,
                               std::uint32_t row_offset) {
    const std::size_t cols = matrix_columns(_scene.num_faces());
    const std::span<const std::uint32_t> scratch(
        static_cast<const std::uint32_t*>(checked_mapped(_counts)),
        static_cast<std::size_t>(_config.tile_rows) * cols);
    for (const std::uint32_t face : block_emitters) {
        const std::size_t row = face - row_offset;
        auto& host_row = _host_rows[face];
        for (std::size_t col = 0; col < cols; ++col) {
            const std::uint32_t cell = scratch[(row * cols) + col];
            if (cell != 0) {
                host_row[static_cast<std::uint32_t>(col)] += cell;
            }
        }
    }
}

void VfAccumImpl::record_batch(std::span<const std::uint32_t> emitters,
                               std::uint64_t rays_per_face) {
    for (const std::uint32_t face : emitters) {
        _rays_per_row[face] += rays_per_face;
    }
    _rays_per_face += rays_per_face;
    _total_rays += rays_per_face * emitters.size();
}

VfResult VfAccumImpl::build_result() const {
    const std::size_t faces = _scene.num_faces();
    VfResult result;
    if (_config.layout == AccumLayout::Dense) {
        const std::span<const std::uint32_t> cells(
            static_cast<const std::uint32_t*>(checked_mapped(_counts)),
            faces * matrix_columns(faces));
        result = assemble_vf(cells, _scene.face_areas(), _rays_per_row, _config);
    } else {
        result = assemble_vf(std::span<const HostCountRow>(_host_rows), _scene.face_areas(),
                             _rays_per_row, _config);
    }
    result.stats.total_rays = _total_rays;
    result.stats.rays_per_face = _rays_per_face;
    // TODO(radiative): fill gpu_time from command-buffer GPU timestamps.
    return result;
}

ExchangeAccumImpl::ExchangeAccumImpl(SceneImpl& scene, Band band, const AccumConfig& config)
    : _scene(scene), _band(band), _config(config) {
    const std::uint64_t faces = _scene.num_faces();
    std::uint64_t buffer_rows = faces;
    if (_config.layout == AccumLayout::Tiled) {
        if (_config.tile_rows == 0) {
            throw std::invalid_argument("pycanha::radiative: the Tiled layout needs tile_rows > 0");
        }
        _config.tile_rows =
            static_cast<std::uint32_t>(std::min<std::uint64_t>(_config.tile_rows, faces));
        buffer_rows = _config.tile_rows;
        _host_rows.resize(faces);
    }
    _cells = _scene.create_buffer(buffer_rows * matrix_columns(faces) * sizeof(std::uint64_t));
    _rays_per_row.assign(faces, 0);
    reset();
}

ExchangeAccumImpl::~ExchangeAccumImpl() { _scene.destroy_buffer(_cells); }

void ExchangeAccumImpl::reset() {
    clear_block_scratch();
    for (auto& row : _host_rows) {
        row.clear();
    }
    std::ranges::fill(_rays_per_row, 0);
    _fp_scale = 0.0;
    _rays_per_face = 0;
    _total_rays = 0;
}

void ExchangeAccumImpl::clear_block_scratch() { clear_cells(_cells); }

void ExchangeAccumImpl::absorb_block(std::span<const std::uint32_t> block_emitters,
                                     std::uint32_t row_offset) {
    const std::size_t cols = matrix_columns(_scene.num_faces());
    const std::span<const std::uint64_t> scratch(
        static_cast<const std::uint64_t*>(checked_mapped(_cells)),
        static_cast<std::size_t>(_config.tile_rows) * cols);
    for (const std::uint32_t face : block_emitters) {
        const std::size_t row = face - row_offset;
        auto& host_row = _host_rows[face];
        // Cells add with wrap: the lost column may carry negative (wrapped)
        // Russian-roulette adjustments.
        for (std::size_t col = 0; col < cols; ++col) {
            const std::uint64_t cell = scratch[(row * cols) + col];
            if (cell != 0) {
                host_row[static_cast<std::uint32_t>(col)] += cell;
            }
        }
    }
}

float ExchangeAccumImpl::prepare_batch(std::uint64_t rays_per_face) {
    if (_fp_scale <= 0.0) {
        _fp_scale = select_fp_scale(rays_per_face);
    }
    if (_rays_per_face + rays_per_face > max_cumulative_rays(_fp_scale)) {
        throw std::invalid_argument("pycanha::radiative: cumulative rays_per_face exceeds the "
                                    "fixed-point accumulation range; reset the accumulator or use "
                                    "fewer rays");
    }
    return static_cast<float>(_fp_scale);
}

void ExchangeAccumImpl::record_batch(std::span<const std::uint32_t> emitters,
                                     std::uint64_t rays_per_face) {
    for (const std::uint32_t face : emitters) {
        _rays_per_row[face] += rays_per_face;
    }
    _rays_per_face += rays_per_face;
    _total_rays += rays_per_face * emitters.size();
}

ExchangeCellSource ExchangeAccumImpl::cells() const {
    if (_config.layout != AccumLayout::Dense) {
        return std::span<const HostCountRow>(_host_rows);
    }
    const std::size_t faces = _scene.num_faces();
    return std::span<const std::uint64_t>(static_cast<const std::uint64_t*>(checked_mapped(_cells)),
                                          faces * matrix_columns(faces));
}

ExchangeResult ExchangeAccumImpl::build_result() const {
    const std::vector<double> emissivity = band_emissivity(_scene.materials(), _band);
    ExchangeResult result = assemble_exchange(ExchangeInputs{.cells = cells(),
                                                             .areas = _scene.face_areas(),
                                                             .emissivity = emissivity,
                                                             .rays_per_row = _rays_per_row,
                                                             .fp_scale = _fp_scale,
                                                             .band = _band},
                                              _config);
    result.stats.total_rays = _total_rays;
    result.stats.rays_per_face = _rays_per_face;
    return result;
}

std::uint64_t ExchangeAccumImpl::conservation_error() const {
    return exchange_conservation_error(cells(), _rays_per_row, _fp_scale);
}

SolarAccumImpl::SolarAccumImpl(SceneImpl& scene) : _scene(scene) {
    const std::uint64_t faces = _scene.num_faces();
    _direct = _scene.create_buffer(faces * sizeof(std::uint64_t));
    _total = _scene.create_buffer(faces * sizeof(std::uint64_t));
    const std::span<const double> areas = _scene.face_areas();
    // A sun sample carries at most cos * A of the face it is drawn on, and
    // only the default emitters draw samples (a planet's faces never do), so
    // their area bounds every cell. A planet's area here drove the
    // fixed-point scale to 1: one deposit unit per square metre, and every
    // lit face got S A whatever its angle to the sun.
    double total_area = 0.0;
    for (const std::uint32_t face : _scene.default_emitters()) {
        total_area += areas[face];
    }
    _area_units = std::max<std::uint64_t>(1, static_cast<std::uint64_t>(std::ceil(total_area)));
    reset();
}

SolarAccumImpl::~SolarAccumImpl() {
    _scene.destroy_buffer(_direct);
    _scene.destroy_buffer(_total);
}

void SolarAccumImpl::reset() {
    clear_cells(_direct);
    clear_cells(_total);
    _sun_recorded = false;
    _fp_scale = 0.0;
    _rays_per_face = 0;
    _total_rays = 0;
}

SolarAccumImpl::BatchSetup SolarAccumImpl::prepare_batch(const SolarState& sun,
                                                         std::uint64_t rays_per_face) {
    const double norm = sun.direction.norm();
    if (!(norm > 0.0)) {
        throw std::invalid_argument(
            "pycanha::radiative: the sun direction must be a nonzero vector");
    }
    if (sun.irradiance < 0.0) {
        throw std::invalid_argument("pycanha::radiative: the solar irradiance cannot be negative");
    }
    const Vector3D direction = sun.direction / norm;
    if (!_sun_recorded) {
        _sun.direction = direction;
        _sun.irradiance = sun.irradiance;
        _sun_recorded = true;
    } else if (!(_sun.direction.array() == direction.array()).all() ||
               _sun.irradiance != sun.irradiance) {
        throw std::invalid_argument(
            "pycanha::radiative: all batches of a solar accumulator must use "
            "the same SolarState; use a fresh accumulator per sun snapshot");
    }
    if (_fp_scale <= 0.0) {
        _fp_scale = select_fp_scale(rays_per_face * _area_units);
    }
    if ((_rays_per_face + rays_per_face) * _area_units > max_cumulative_rays(_fp_scale)) {
        throw std::invalid_argument("pycanha::radiative: cumulative rays_per_face exceeds the "
                                    "fixed-point accumulation range; reset the accumulator or use "
                                    "fewer rays");
    }
    return BatchSetup{
        .sun_dir = {static_cast<float>(direction.x()), static_cast<float>(direction.y()),
                    static_cast<float>(direction.z())},
        .fp_scale = static_cast<float>(_fp_scale)};
}

void SolarAccumImpl::record_batch(std::uint64_t rays_per_face, std::size_t num_emitters) {
    _rays_per_face += rays_per_face;
    _total_rays += rays_per_face * num_emitters;
}

SolarResult SolarAccumImpl::build_result() const {
    const auto faces = static_cast<Eigen::Index>(_scene.num_faces());
    SolarResult result;
    result.direct = Eigen::VectorXd::Zero(faces);
    result.total = Eigen::VectorXd::Zero(faces);
    result.stats.total_rays = _total_rays;
    result.stats.rays_per_face = _rays_per_face;
    if (_rays_per_face == 0 || _fp_scale <= 0.0) {
        return result;
    }

    const std::span<const std::uint64_t> direct(
        static_cast<const std::uint64_t*>(checked_mapped(_direct)),
        static_cast<std::size_t>(faces));
    const std::span<const std::uint64_t> total(
        static_cast<const std::uint64_t*>(checked_mapped(_total)), static_cast<std::size_t>(faces));

    const std::span<const double> areas = _scene.face_areas();
    const auto rays = static_cast<double>(_rays_per_face);
    const double norm = (1.0 / _fp_scale) / rays;
    double stderr_sum = 0.0;
    double stderr_max = 0.0;
    std::size_t stderr_entries = 0;
    for (Eigen::Index face = 0; face < faces; ++face) {
        const auto index = static_cast<std::size_t>(face);
        const double area = areas[index];
        if (area <= 0.0) {
            continue;  // no geometry on this face, nothing was deposited
        }
        // Deposits already carry the emitting face's area, so the
        // accumulated cells ARE the absorbed watts (per unit irradiance).
        const double direct_watts = _sun.irradiance * static_cast<double>(direct[index]) * norm;
        const double total_watts = _sun.irradiance * static_cast<double>(total[index]) * norm;
        result.direct(face) = direct_watts;
        result.total(face) = total_watts;
        if (total_watts > 0.0) {
            // Advisory precision estimate: the Bernoulli bound on the flux
            // fraction (exact only without concentration above 1 sun),
            // scaled back to watts.
            const double flux_fraction = total_watts / (_sun.irradiance * area);
            const double entry_stderr =
                _sun.irradiance * area *
                std::sqrt(std::min(flux_fraction, 1.0) * std::max(1.0 - flux_fraction, 0.0) / rays);
            stderr_sum += entry_stderr;
            stderr_max = std::max(stderr_max, entry_stderr);
            ++stderr_entries;
        }
    }
    result.stats.mean_stderr =
        stderr_entries > 0 ? stderr_sum / static_cast<double>(stderr_entries) : 0.0;
    result.stats.max_stderr = stderr_max;
    return result;
}

}  // namespace pycanha::radiative::detail
