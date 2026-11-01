#include "vk_accum.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "exchange_assemble.hpp"
#include "pair_walk.hpp"
#include "pycanha-core/gmm/ids.hpp"
#include "pycanha-core/radiative/aggregate.hpp"
#include "pycanha-core/radiative/materials.hpp"
#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/scene.hpp"
#include "pycanha-core/radiative/settings.hpp"
#include "vf_assemble.hpp"
#include "vk_device.hpp"
#include "vk_scene.hpp"

namespace pycanha::radiative::detail {

namespace {

// Largest power of two S with rays_per_face * S <= 2^58. Powers of two are
// exact in f32 and make to_fp(1.0) == S on the GPU, so a row's expected
// balance is the exact integer rays * S; the 2^58 budget leaves 32x
// cumulative-ray headroom before a u64 row balance could overflow, with a
// deposit resolution (1/S) still far below Monte-Carlo noise.
[[nodiscard]] double select_fp_scale(std::uint64_t rays_per_face) {
    constexpr int budget_bits = 58;
    const int rays_bits =
        rays_per_face > 1 ? std::bit_width(rays_per_face - 1) : 0;
    return std::ldexp(1.0, std::max(budget_bits - rays_bits, 0));
}

// Cumulative rays_per_face a fixed-point accumulator can absorb before its
// u64 balances could overflow.
[[nodiscard]] std::uint64_t max_cumulative_rays(double fp_scale) {
    return (std::uint64_t{1} << 63U) / static_cast<std::uint64_t>(fp_scale);
}

// Usage for a device-local cell buffer: the kernels atomically add into it
// (storage), clear_buffer fills it (transfer dst) and read_back copies out
// of it (transfer src). The casts keep the signed enum constants out of the
// bitwise expression.
constexpr auto accum_usage =
    static_cast<VkBufferUsageFlags>(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) |
    static_cast<VkBufferUsageFlags>(VK_BUFFER_USAGE_TRANSFER_SRC_BIT) |
    static_cast<VkBufferUsageFlags>(VK_BUFFER_USAGE_TRANSFER_DST_BIT);
constexpr auto readback_usage =
    static_cast<VkBufferUsageFlags>(VK_BUFFER_USAGE_TRANSFER_DST_BIT);

// Merges one Tiled block (the read-back cells of the block's emitter rows)
// into the host rows. Each emitter adds only into its own host row, so the
// rows merge in parallel without changing any sum (cells add with wrap: the
// exchange lost column may carry negative, wrapped, Russian-roulette
// adjustments). On large scenes this merge, run on one thread, was nearly
// all of a trace's time while the GPU sat idle.
template <typename Cell>
void merge_block_rows(std::span<const Cell> scratch, std::size_t cols,
                      std::span<const std::uint32_t> block_emitters,
                      std::uint32_t row_offset,
                      std::vector<HostCountRow>& host_rows) {
    const unsigned threads =
        worker_count({}, block_emitters.size(), block_emitters.size() * cols);
    parallel_for_index(block_emitters.size(), threads, [&](std::size_t index) {
        const std::uint32_t face = block_emitters[index];
        host_rows[face].add_dense(scratch.subspan(
            static_cast<std::size_t>(face - row_offset) * cols, cols));
    });
}

// An accumulator cell buffer is bound as one storage buffer, which a kernel
// can address only up to the device's maxStorageBufferRange: beyond it the
// deposits and the readback cover part of the buffer, with no error (a Dense
// 5.3 GiB exchange buffer on a 4 GiB-range RTX 4070 lost 81 % of its
// couplings). Refused here, with the way out.
void check_storage_range(const SceneImpl& scene, VkDeviceSize bytes) {
    const std::uint64_t range = scene.device().max_storage_buffer_range;
    if (range > 0 && bytes > range) {
        throw std::length_error(
            "pycanha::radiative: the accumulator buffer (" +
            std::to_string(bytes) +
            " bytes) exceeds the device's storage "
            "buffer range (" +
            std::to_string(range) +
            " bytes); use AccumLayout::Tiled with fewer tile_rows");
    }
}

}  // namespace

VfAccumImpl::VfAccumImpl(SceneImpl& scene, const AccumConfig& config)
    : _scene(scene), _config(config) {
    const std::uint64_t faces = _scene.num_faces();
    std::uint64_t buffer_rows = faces;
    if (_config.layout == AccumLayout::Tiled) {
        if (_config.tile_rows == 0) {
            throw std::invalid_argument(
                "pycanha::radiative: the Tiled layout needs tile_rows > 0");
        }
        _config.tile_rows = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(_config.tile_rows, faces));
        buffer_rows = _config.tile_rows;
        _host_rows.resize(faces);
    }
    const VkDeviceSize cell_bytes =
        buffer_rows * matrix_columns(faces) * sizeof(std::uint32_t);
    check_storage_range(_scene, cell_bytes);
    _counts = _scene.create_buffer(cell_bytes, accum_usage,
                                   BufferMemory::DeviceLocal);
    _readback = _scene.create_buffer(cell_bytes, readback_usage,
                                     BufferMemory::HostReadback);
    _rays_per_row.assign(faces, 0);
    reset();
}

VfAccumImpl::~VfAccumImpl() {
    _scene.destroy_buffer(_counts);
    _scene.destroy_buffer(_readback);
}

void VfAccumImpl::reset() {
    clear_block_scratch();
    for (auto& row : _host_rows) {
        row.clear();
    }
    std::ranges::fill(_rays_per_row, 0);
    _rays_per_face = 0;
    _total_rays = 0;
}

void VfAccumImpl::clear_block_scratch() { _scene.clear_buffer(_counts); }

void VfAccumImpl::absorb_block(std::span<const std::uint32_t> block_emitters,
                               std::uint32_t row_offset) {
    _scene.read_back(_counts, _readback);
    const std::size_t cols = matrix_columns(_scene.num_faces());
    const std::span<const std::uint32_t> scratch(
        static_cast<const std::uint32_t*>(checked_mapped(_readback)),
        static_cast<std::size_t>(_config.tile_rows) * cols);
    merge_block_rows(scratch, cols, block_emitters, row_offset, _host_rows);
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
        _scene.read_back(_counts, _readback);
        const std::span<const std::uint32_t> cells(
            static_cast<const std::uint32_t*>(checked_mapped(_readback)),
            faces * matrix_columns(faces));
        result =
            assemble_vf(cells, _scene.face_areas(), _rays_per_row, _config);
    } else {
        result = assemble_vf(std::span<const HostCountRow>(_host_rows),
                             _scene.face_areas(), _rays_per_row, _config);
    }
    result.stats.total_rays = _total_rays;
    result.stats.rays_per_face = _rays_per_face;
    // TODO(radiative): fill gpu_time from timestamp queries around each
    // dispatch chunk.
    return result;
}

ExchangeAccumImpl::ExchangeAccumImpl(SceneImpl& scene, Band band,
                                     const AccumConfig& config)
    : _scene(scene), _band(band), _config(config) {
    const std::uint64_t faces = _scene.num_faces();
    std::uint64_t buffer_rows = faces;
    if (_config.layout == AccumLayout::Tiled) {
        if (_config.tile_rows == 0) {
            throw std::invalid_argument(
                "pycanha::radiative: the Tiled layout needs tile_rows > 0");
        }
        _config.tile_rows = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(_config.tile_rows, faces));
        buffer_rows = _config.tile_rows;
        _host_rows.resize(faces);
    }
    const VkDeviceSize cell_bytes =
        buffer_rows * matrix_columns(faces) * sizeof(std::uint64_t);
    check_storage_range(_scene, cell_bytes);
    _cells = _scene.create_buffer(cell_bytes, accum_usage,
                                  BufferMemory::DeviceLocal);
    _readback = _scene.create_buffer(cell_bytes, readback_usage,
                                     BufferMemory::HostReadback);
    _rays_per_row.assign(faces, 0);
    reset();
}

ExchangeAccumImpl::~ExchangeAccumImpl() {
    _scene.destroy_buffer(_cells);
    _scene.destroy_buffer(_readback);
}

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

void ExchangeAccumImpl::clear_block_scratch() { _scene.clear_buffer(_cells); }

void ExchangeAccumImpl::absorb_block(
    std::span<const std::uint32_t> block_emitters, std::uint32_t row_offset) {
    _scene.read_back(_cells, _readback);
    const std::size_t cols = matrix_columns(_scene.num_faces());
    const std::span<const std::uint64_t> scratch(
        static_cast<const std::uint64_t*>(checked_mapped(_readback)),
        static_cast<std::size_t>(_config.tile_rows) * cols);
    merge_block_rows(scratch, cols, block_emitters, row_offset, _host_rows);
}

void ExchangeAccumImpl::absorb_compact(
    std::span<const std::uint32_t> block_emitters,
    std::span<const std::uint32_t> first, std::span<const std::uint32_t> counts,
    std::span<const std::uint64_t> entries) {
    const unsigned threads =
        worker_count({}, block_emitters.size(), entries.size());
    parallel_for_index(block_emitters.size(), threads, [&](std::size_t index) {
        _host_rows[block_emitters[index]].add_sparse(
            entries.subspan(static_cast<std::size_t>(first[index]) * 2,
                            static_cast<std::size_t>(counts[index]) * 2));
    });
}

float ExchangeAccumImpl::prepare_batch(std::uint64_t rays_per_face) {
    if (_fp_scale <= 0.0) {
        _fp_scale = select_fp_scale(rays_per_face);
    }
    if (_rays_per_face + rays_per_face > max_cumulative_rays(_fp_scale)) {
        throw std::invalid_argument(
            "pycanha::radiative: cumulative rays_per_face exceeds the "
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
    _scene.read_back(_cells, _readback);
    const std::size_t faces = _scene.num_faces();
    return std::span<const std::uint64_t>(
        static_cast<const std::uint64_t*>(checked_mapped(_readback)),
        faces * matrix_columns(faces));
}

ExchangeResult ExchangeAccumImpl::build_result() const {
    const std::vector<double> emissivity =
        band_emissivity(_scene.materials(), _band);
    ExchangeResult result =
        assemble_exchange(ExchangeInputs{.cells = cells(),
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

NodeExchangeAccumImpl::NodeExchangeAccumImpl(
    SceneImpl& scene, Band band, std::span<const NodeNum> node_numbers,
    const AccumConfig& config)
    : _scene(scene), _band(band), _config(config) {
    const std::size_t faces = _scene.num_faces();
    if (node_numbers.size() != faces) {
        throw std::invalid_argument(
            "pycanha::radiative: node_numbers size must match the face count");
    }
    const TriangulationConfig& tri = _config.triangulation;
    if (tri.mode == TriangulationMode::ConstrainedLeastSquares) {
        throw std::invalid_argument(
            "pycanha::radiative: NodeExchangeAccumulator takes "
            "TriangulationMode None or RayDensity; ConstrainedLeastSquares "
            "is a projection over the whole face matrix (use "
            "ExchangeAccumulator)");
    }
    if (tri.keep_full_matrix) {
        throw std::invalid_argument(
            "pycanha::radiative: keep_full_matrix is a face matrix; "
            "NodeExchangeAccumulator never holds one (use "
            "ExchangeAccumulator)");
    }
    if (!std::isfinite(tri.exponent) || tri.exponent <= 0.0) {
        throw std::invalid_argument(
            "pycanha::radiative: the triangulation exponent must be finite "
            "and greater than zero");
    }
    _nodes = aggregate_nodes(node_numbers);
    const std::uint64_t nodes = _nodes.size();
    const std::uint64_t columns =
        nodes + 1 + static_cast<std::uint64_t>(num_virtual_columns);
    if (columns > std::numeric_limits<std::uint32_t>::max()) {
        throw std::length_error("pycanha::radiative: too many nodes");
    }
    _face_column.assign(faces, static_cast<std::uint32_t>(nodes));
    std::vector<std::uint64_t> node_faces(nodes, 0);
    for (std::size_t face = 0; face < faces; ++face) {
        if (node_numbers[face] == gmm::NO_NODE) {
            continue;
        }
        const auto slot = static_cast<std::uint32_t>(std::distance(
            _nodes.begin(),
            std::ranges::lower_bound(_nodes, node_numbers[face])));
        _face_column[face] = slot;
        ++node_faces[slot];
    }
    for (const std::uint64_t count : node_faces) {
        _max_node_faces = std::max(_max_node_faces, count);
    }

    std::uint64_t buffer_rows = nodes;
    if (_config.layout == AccumLayout::Tiled) {
        if (_config.tile_rows == 0) {
            throw std::invalid_argument(
                "pycanha::radiative: the Tiled layout needs tile_rows > 0");
        }
        _config.tile_rows = static_cast<std::uint32_t>(std::min<std::uint64_t>(
            _config.tile_rows, std::max<std::uint64_t>(nodes, 1)));
        buffer_rows = _config.tile_rows;
        _host_rows.resize(nodes);
    }
    const VkDeviceSize cell_bytes = std::max<std::uint64_t>(buffer_rows, 1) *
                                    columns * sizeof(std::uint64_t);
    check_storage_range(_scene, cell_bytes);
    _cells = _scene.create_buffer(cell_bytes, accum_usage,
                                  BufferMemory::DeviceLocal);
    // Dense reads its cells back whole; a Tiled block is compacted on the
    // device and only falls back to a whole readback (absorb_block), which
    // allocates the host mirror when it first happens: on the reference
    // scene's 6 053-row blocks it was 2.1 GB of host memory never touched.
    if (_config.layout == AccumLayout::Dense) {
        _readback = _scene.create_buffer(cell_bytes, readback_usage,
                                         BufferMemory::HostReadback);
    }
    _face_columns_buf = _scene.upload_to_new_buffer(
        _face_column.data(), faces * sizeof(std::uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    // Filled by the first batch (fix_weights); zero until then.
    _face_weights_buf = _scene.create_buffer(
        std::max<VkDeviceSize>(faces * 4 * sizeof(float), 16), accum_usage,
        BufferMemory::DeviceLocal);
    _scene.clear_buffer(_face_weights_buf);
    reset();
}

NodeExchangeAccumImpl::~NodeExchangeAccumImpl() {
    _scene.destroy_buffer(_cells);
    _scene.destroy_buffer(_readback);
    _scene.destroy_buffer(_face_columns_buf);
    _scene.destroy_buffer(_face_weights_buf);
}

void NodeExchangeAccumImpl::reset() {
    clear_block_scratch();
    for (auto& row : _host_rows) {
        row.clear();
    }
    _emitters.clear();
    _emitted_area = 0.0;
    _fp_scale = 0.0;
    _rays_per_face = 0;
    _total_rays = 0;
}

void NodeExchangeAccumImpl::clear_block_scratch() {
    _scene.clear_buffer(_cells);
}

std::vector<std::uint32_t> NodeExchangeAccumImpl::prepare_emitters(
    std::vector<std::uint32_t> emitters) {
    if (emitters.empty()) {
        return emitters;  // nothing to trace (rays_per_face == 0)
    }
    const std::uint32_t none = num_nodes();
    std::erase_if(emitters, [this, none](const std::uint32_t face) {
        return _face_column[face] == none;
    });
    std::ranges::sort(emitters, [this](std::uint32_t a, std::uint32_t b) {
        return _face_column[a] != _face_column[b]
                   ? _face_column[a] < _face_column[b]
                   : a < b;
    });
    if (std::ranges::adjacent_find(emitters) != emitters.end()) {
        throw std::invalid_argument(
            "pycanha::radiative: an emitter is listed twice; a "
            "NodeExchangeAccumulator needs every emitter to send the same "
            "rays");
    }
    if (_emitters.empty()) {
        _emitters = emitters;
        fix_weights();
    } else if (emitters != _emitters) {
        throw std::invalid_argument(
            "pycanha::radiative: every batch of a NodeExchangeAccumulator must "
            "trace the emitters of its first (the face pair weights were "
            "fixed by them); reset it or use another accumulator");
    }
    return emitters;
}

void NodeExchangeAccumImpl::fix_weights() {
    const std::size_t faces = _face_column.size();
    const std::size_t nodes = _nodes.size();
    const std::vector<double> emissivity =
        band_emissivity(_scene.materials(), _band);
    const std::span<const double> areas = _scene.face_areas();
    _node_scale.assign(nodes, 0.0);
    _node_emissive_area.assign(nodes, 0.0);
    for (std::size_t face = 0; face < faces; ++face) {
        const std::uint32_t node = _face_column[face];
        if (node < nodes) {
            const double emissive = areas[face] * emissivity[face];
            _node_scale[node] = std::max(_node_scale[node], emissive);
            _node_emissive_area[node] += emissive;
        }
    }
    // (share, u as a float and the rest of the double, unused) per face. u is
    // the face's area when it emits: on a trace where every emitter sends the
    // same rays, A / N and A over the mean rays (both settings of
    // area_weights) only differ by the common N, which the weight's ratio
    // cancels.
    constexpr std::size_t record = 4;
    std::vector<float> weights(record * faces, 0.0F);
    for (std::size_t face = 0; face < faces; ++face) {
        const std::uint32_t node = _face_column[face];
        if (node < nodes && _node_scale[node] > 0.0) {
            weights[record * face] = static_cast<float>(
                areas[face] * emissivity[face] / _node_scale[node]);
        }
    }
    _emitted_area = 0.0;
    for (const std::uint32_t face : _emitters) {
        const auto high = static_cast<float>(areas[face]);
        weights[(record * face) + 1] = high;
        weights[(record * face) + 2] =
            static_cast<float>(areas[face] - static_cast<double>(high));
        _emitted_area += areas[face] * emissivity[face];
    }
    _scene.upload_to_device_buffer(_face_weights_buf, weights.data(),
                                   weights.size() * sizeof(float));
}

float NodeExchangeAccumImpl::prepare_batch(std::uint64_t rays_per_face) {
    // A node row holds the deposits of all its faces' rays.
    if (_fp_scale <= 0.0) {
        _fp_scale = select_fp_scale(rays_per_face * _max_node_faces);
    }
    if ((_rays_per_face + rays_per_face) * _max_node_faces >
        max_cumulative_rays(_fp_scale)) {
        throw std::invalid_argument(
            "pycanha::radiative: cumulative rays_per_face exceeds the "
            "fixed-point accumulation range of the largest node; reset the "
            "accumulator or use fewer rays");
    }
    return static_cast<float>(_fp_scale);
}

void NodeExchangeAccumImpl::absorb_block(std::span<const std::uint32_t> rows,
                                         std::uint32_t row_offset) {
    if (_readback.buffer == VK_NULL_HANDLE) {
        _readback = _scene.create_buffer(_cells.size, readback_usage,
                                         BufferMemory::HostReadback);
    }
    _scene.read_back(_cells, _readback);
    const std::size_t cols =
        columns() + static_cast<std::size_t>(num_virtual_columns);
    const std::span<const std::uint64_t> scratch(
        static_cast<const std::uint64_t*>(checked_mapped(_readback)),
        static_cast<std::size_t>(_config.tile_rows) * cols);
    merge_block_rows(scratch, cols, rows, row_offset, _host_rows);
}

void NodeExchangeAccumImpl::absorb_compact(
    std::span<const std::uint32_t> rows, std::span<const std::uint32_t> first,
    std::span<const std::uint32_t> counts,
    std::span<const std::uint64_t> entries) {
    const unsigned threads = worker_count({}, rows.size(), entries.size());
    parallel_for_index(rows.size(), threads, [&](std::size_t index) {
        _host_rows[rows[index]].add_sparse(
            entries.subspan(static_cast<std::size_t>(first[index]) * 2,
                            static_cast<std::size_t>(counts[index]) * 2));
    });
}

void NodeExchangeAccumImpl::record_batch(
    std::span<const std::uint32_t> emitters, std::uint64_t rays_per_face) {
    _rays_per_face += rays_per_face;
    _total_rays += rays_per_face * emitters.size();
}

NodeExchangeResult NodeExchangeAccumImpl::build_result() const {
    const std::size_t nodes = _nodes.size();
    const std::size_t cols =
        columns() + static_cast<std::size_t>(num_virtual_columns);
    ExchangeCellSource cells = std::span<const HostCountRow>(_host_rows);
    if (_config.layout == AccumLayout::Dense && nodes > 0) {
        _scene.read_back(_cells, _readback);
        cells = std::span<const std::uint64_t>(
            static_cast<const std::uint64_t*>(checked_mapped(_readback)),
            nodes * cols);
    } else if (_config.layout == AccumLayout::Dense) {
        cells = std::span<const std::uint64_t>();
    }
    NodeExchangeResult result = assemble_node_exchange(
        NodeExchangeInputs{.cells = cells,
                           .num_nodes = nodes,
                           .node_scale = _node_scale,
                           .node_emissive_area = _node_emissive_area,
                           .rays_per_face = _rays_per_face,
                           .emitted_area = _emitted_area,
                           .fp_scale = _fp_scale,
                           .band = _band},
        _config);
    result.nodes = _nodes;
    result.stats.total_rays = _total_rays;
    result.stats.rays_per_face = _rays_per_face;
    return result;
}

SolarAccumImpl::SolarAccumImpl(SceneImpl& scene) : _scene(scene) {
    const std::uint64_t faces = _scene.num_faces();
    const VkDeviceSize cell_bytes = faces * sizeof(std::uint64_t);
    _direct = _scene.create_buffer(cell_bytes, accum_usage,
                                   BufferMemory::DeviceLocal);
    _total = _scene.create_buffer(cell_bytes, accum_usage,
                                  BufferMemory::DeviceLocal);
    _direct_readback = _scene.create_buffer(cell_bytes, readback_usage,
                                            BufferMemory::HostReadback);
    _total_readback = _scene.create_buffer(cell_bytes, readback_usage,
                                           BufferMemory::HostReadback);
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
    _area_units = std::max<std::uint64_t>(
        1, static_cast<std::uint64_t>(std::ceil(total_area)));
    reset();
}

SolarAccumImpl::~SolarAccumImpl() {
    _scene.destroy_buffer(_direct);
    _scene.destroy_buffer(_total);
    _scene.destroy_buffer(_direct_readback);
    _scene.destroy_buffer(_total_readback);
}

void SolarAccumImpl::reset() {
    _scene.clear_buffer(_direct);
    _scene.clear_buffer(_total);
    _sun_recorded = false;
    _fp_scale = 0.0;
    _rays_per_face = 0;
    _total_rays = 0;
}

SolarAccumImpl::BatchSetup SolarAccumImpl::prepare_batch(
    const SolarState& sun, std::uint64_t rays_per_face, double ray_area) {
    const double norm = sun.direction.norm();
    if (!(norm > 0.0)) {
        throw std::invalid_argument(
            "pycanha::radiative: the sun direction must be a nonzero vector");
    }
    if (sun.irradiance < 0.0) {
        throw std::invalid_argument(
            "pycanha::radiative: the solar irradiance cannot be negative");
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
    // A sun-plane ray carries its rectangle's area, which can exceed the
    // emitters' total: the deposit units then follow it.
    const auto ray_units = static_cast<std::uint64_t>(std::ceil(ray_area));
    if (_fp_scale <= 0.0) {
        _area_units = std::max(_area_units, ray_units);
        _fp_scale = select_fp_scale(rays_per_face * _area_units);
    } else if (ray_units > _area_units) {
        throw std::invalid_argument(
            "pycanha::radiative: a later solar batch carries more energy per "
            "ray than the accumulator's scale allows; use a fresh accumulator");
    }
    if ((_rays_per_face + rays_per_face) * _area_units >
        max_cumulative_rays(_fp_scale)) {
        throw std::invalid_argument(
            "pycanha::radiative: cumulative rays_per_face exceeds the "
            "fixed-point accumulation range; reset the accumulator or use "
            "fewer rays");
    }
    return BatchSetup{.sun_dir = {static_cast<float>(direction.x()),
                                  static_cast<float>(direction.y()),
                                  static_cast<float>(direction.z())},
                      .fp_scale = static_cast<float>(_fp_scale)};
}

void SolarAccumImpl::record_batch(std::uint64_t rays_per_face,
                                  std::size_t num_emitters) {
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

    _scene.read_back(_direct, _direct_readback);
    _scene.read_back(_total, _total_readback);
    const std::span<const std::uint64_t> direct(
        static_cast<const std::uint64_t*>(checked_mapped(_direct_readback)),
        static_cast<std::size_t>(faces));
    const std::span<const std::uint64_t> total(
        static_cast<const std::uint64_t*>(checked_mapped(_total_readback)),
        static_cast<std::size_t>(faces));

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
        const double direct_watts =
            _sun.irradiance * static_cast<double>(direct[index]) * norm;
        const double total_watts =
            _sun.irradiance * static_cast<double>(total[index]) * norm;
        result.direct(face) = direct_watts;
        result.total(face) = total_watts;
        if (total_watts > 0.0) {
            // Advisory precision estimate: the Bernoulli bound on the flux
            // fraction (exact only without concentration above 1 sun),
            // scaled back to watts.
            const double flux_fraction = total_watts / (_sun.irradiance * area);
            const double entry_stderr =
                _sun.irradiance * area *
                std::sqrt(std::min(flux_fraction, 1.0) *
                          std::max(1.0 - flux_fraction, 0.0) / rays);
            stderr_sum += entry_stderr;
            stderr_max = std::max(stderr_max, entry_stderr);
            ++stderr_entries;
        }
    }
    result.stats.mean_stderr =
        stderr_entries > 0 ? stderr_sum / static_cast<double>(stderr_entries)
                           : 0.0;
    result.stats.max_stderr = stderr_max;
    return result;
}

}  // namespace pycanha::radiative::detail
