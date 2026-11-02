#pragma once

// src-private accumulators: GPU integer cells plus the host bookkeeping
// needed to normalize them into results and compute statistics. Matrix
// accumulators (VF counts, exchange fixed-point energy) come in two
// layouts, one result:
//  - Dense: one num_num_faces x num_faces buffer, cells stay on the GPU.
//  - Tiled: a tile_rows x num_faces scratch buffer reused per row block;
//    cells stream into per-row host maps after every block.
// Cells are integers, so both layouts produce bit-identical results for the
// same seed — a property the tests assert. The solar accumulator is two
// per-face vectors and needs no layout machinery.
//
// Every cell buffer lives in device-local memory: the kernels take one
// 64-bit atomic per deposited ray, and on a discrete GPU a host-visible
// accumulator would put every one of those on the PCIe bus (measured at
// 117x on an RTX 4070). Each accumulator therefore carries a host-visible
// staging buffer of the same size, and the host reads that instead — one
// copy per readback, about 22 GB/s, which is a fraction of a percent of a
// real trace even where it buys nothing.

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "exchange_assemble.hpp"
#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/scene.hpp"
#include "pycanha-core/radiative/settings.hpp"
#include "vf_assemble.hpp"
#include "vk_scene.hpp"

namespace pycanha::radiative::detail {

class VfAccumImpl {
  public:
    VfAccumImpl(SceneImpl& scene, const AccumConfig& config);
    ~VfAccumImpl();
    VfAccumImpl(const VfAccumImpl&) = delete;
    VfAccumImpl& operator=(const VfAccumImpl&) = delete;
    VfAccumImpl(VfAccumImpl&&) = delete;
    VfAccumImpl& operator=(VfAccumImpl&&) = delete;

    void reset();
    [[nodiscard]] VfResult build_result() const;

    // --- called by SceneImpl::accumulate_vf --------------------------------
    [[nodiscard]] AccumLayout layout() const noexcept { return _config.layout; }
    [[nodiscard]] std::uint32_t tile_rows() const noexcept {
        return _config.tile_rows;
    }
    // The GPU cell buffer: whole matrix (Dense) or block scratch (Tiled).
    [[nodiscard]] VkBuffer buffer() const noexcept { return _counts.buffer; }
    // Tiled only: zeroes the block scratch before a new row block.
    void clear_block_scratch();
    // Tiled only: reads the scratch rows of `block_emitters` (block-relative
    // row = face - row_offset) and adds nonzero cells into the host maps.
    void absorb_block(std::span<const std::uint32_t> block_emitters,
                      std::uint32_t row_offset);
    void record_batch(std::span<const std::uint32_t> emitters,
                      std::uint64_t rays_per_face);
    [[nodiscard]] std::uint64_t rays_per_face() const noexcept {
        return _rays_per_face;
    }
    [[nodiscard]] SceneImpl& scene() const noexcept { return _scene; }

  private:
    SceneImpl& _scene;
    AccumConfig _config;
    GpuBuffer _counts;    // device-local u32 cells, row stride faces + 3
    GpuBuffer _readback;  // host-visible mirror of _counts, filled per read
    // Tiled layout: accumulated counts per row (column -> count). Dense
    // keeps everything in the GPU buffer instead.
    std::vector<HostCountRow> _host_rows;
    // Rays emitted per matrix row, accumulated across batches (rows in an
    // emitter subset differ from rows that never emitted).
    std::vector<std::uint64_t> _rays_per_row;
    std::uint64_t _rays_per_face = 0;
    std::uint64_t _total_rays = 0;
};

class ExchangeAccumImpl {
  public:
    ExchangeAccumImpl(SceneImpl& scene, Band band, const AccumConfig& config);
    ~ExchangeAccumImpl();
    ExchangeAccumImpl(const ExchangeAccumImpl&) = delete;
    ExchangeAccumImpl& operator=(const ExchangeAccumImpl&) = delete;
    ExchangeAccumImpl(ExchangeAccumImpl&&) = delete;
    ExchangeAccumImpl& operator=(ExchangeAccumImpl&&) = delete;

    void reset();
    [[nodiscard]] ExchangeResult build_result() const;
    [[nodiscard]] std::uint64_t conservation_error() const;

    // --- called by SceneImpl::accumulate_exchange ---------------------------
    [[nodiscard]] Band band() const noexcept { return _band; }
    [[nodiscard]] AccumLayout layout() const noexcept { return _config.layout; }
    [[nodiscard]] std::uint32_t tile_rows() const noexcept {
        return _config.tile_rows;
    }
    [[nodiscard]] VkBuffer buffer() const noexcept { return _cells.buffer; }
    // Fixes the fixed-point scale on the first batch (a power of two with
    // headroom for follow-up batches) and guards the cumulative ray budget
    // against cell overflow. Returns the scale for the push constants.
    [[nodiscard]] float prepare_batch(std::uint64_t rays_per_face);
    // Tiled only: zeroes the block scratch before a new row block.
    void clear_block_scratch();
    // Tiled only: adds the scratch rows of `block_emitters` into the host
    // maps (block-relative row = face - row_offset).
    void absorb_block(std::span<const std::uint32_t> block_emitters,
                      std::uint32_t row_offset);
    // Tiled only: the same, from the block compacted on the device
    // (SceneImpl::compact_block): emitter i's row is the counts[i] (column,
    // cell) pairs of `entries` starting at pair first[i], columns ascending.
    void absorb_compact(std::span<const std::uint32_t> block_emitters,
                        std::span<const std::uint32_t> first,
                        std::span<const std::uint32_t> counts,
                        std::span<const std::uint64_t> entries);
    void record_batch(std::span<const std::uint32_t> emitters,
                      std::uint64_t rays_per_face);
    [[nodiscard]] SceneImpl& scene() const noexcept { return _scene; }

  private:
    // The accumulated cells in whichever layout this accumulator used, made
    // readable on the host first. The assembly itself is backend-agnostic
    // and lives in exchange_assemble.
    [[nodiscard]] ExchangeCellSource cells() const;

    SceneImpl& _scene;
    Band _band;
    AccumConfig _config;
    GpuBuffer _cells;     // device-local u64 fixed point, stride faces + 3
    GpuBuffer _readback;  // host-visible mirror of _cells, filled per read
    // Tiled layout: host-side accumulation (Dense reads the GPU buffer).
    std::vector<HostCountRow> _host_rows;
    std::vector<std::uint64_t> _rays_per_row;
    // Power of two chosen on the first batch; 0 = not chosen yet. Kept as
    // double for exact integer arithmetic on the host (f32 in the shader).
    double _fp_scale = 0.0;
    std::uint64_t _rays_per_face = 0;
    std::uint64_t _total_rays = 0;
};

// The exchange kernel in node rows (NodeExchangeAccumulator): cells are
// num_nodes x (num_nodes + 1 + 3), the extra real column shared by the faces
// in no node. Dense keeps them on the GPU; Tiled streams blocks of node rows
// into host rows as ExchangeAccumImpl does with face rows.
class NodeExchangeAccumImpl {
  public:
    NodeExchangeAccumImpl(
        SceneImpl& scene, Band band, std::span<const NodeNum> node_numbers,
        const AccumConfig& config,
        std::span<const std::int32_t> absorptance_groups = {});
    ~NodeExchangeAccumImpl();
    NodeExchangeAccumImpl(const NodeExchangeAccumImpl&) = delete;
    NodeExchangeAccumImpl& operator=(const NodeExchangeAccumImpl&) = delete;
    NodeExchangeAccumImpl(NodeExchangeAccumImpl&&) = delete;
    NodeExchangeAccumImpl& operator=(NodeExchangeAccumImpl&&) = delete;

    void reset();
    [[nodiscard]] NodeExchangeResult build_result() const;

    // --- called by SceneImpl::accumulate_exchange ---------------------------
    [[nodiscard]] Band band() const noexcept { return _band; }
    [[nodiscard]] AccumLayout layout() const noexcept { return _config.layout; }
    [[nodiscard]] std::uint32_t tile_rows() const noexcept {
        return _config.tile_rows;
    }
    [[nodiscard]] std::uint32_t num_nodes() const noexcept {
        return static_cast<std::uint32_t>(_nodes.size());
    }
    // Real columns of a row: the nodes and the column of the faces in none.
    [[nodiscard]] std::uint32_t columns() const noexcept {
        return num_nodes() + 1;
    }
    [[nodiscard]] float blend_exponent() const noexcept {
        return static_cast<float>(_config.triangulation.exponent);
    }
    [[nodiscard]] bool forward_only() const noexcept {
        return _config.triangulation.mode == TriangulationMode::None;
    }
    [[nodiscard]] VkBuffer buffer() const noexcept { return _cells.buffer; }
    [[nodiscard]] VkBuffer face_columns_buffer() const noexcept {
        return _face_columns_buf.buffer;
    }
    [[nodiscard]] VkBuffer face_weights_buffer() const noexcept {
        return _face_weights_buf.buffer;
    }
    // Absorptance groups (0 when none): the kernel's per-face group and the
    // derivative cells (count per cell, side by side).
    [[nodiscard]] std::uint32_t derivative_count() const noexcept {
        return _derivative_count;
    }
    [[nodiscard]] VkBuffer groups_buffer() const noexcept {
        return _groups_buf.buffer;
    }
    [[nodiscard]] VkBuffer derivative_buffer() const noexcept {
        return _derivative_cells.buffer;
    }
    // A face's node row (num_nodes for a face in no node).
    [[nodiscard]] std::uint32_t row_of(std::uint32_t face) const noexcept {
        return _face_column[face];
    }
    // The resolved emitters that belong to a node, ordered by node. The
    // first batch fixes them and uploads the face weights (a face's blend u
    // depends on whether it emits); a later batch must name the same set.
    [[nodiscard]] std::vector<std::uint32_t> prepare_emitters(
        std::vector<std::uint32_t> emitters);
    // As ExchangeAccumImpl::prepare_batch, with the budget of a node row: the
    // rays of all its faces.
    [[nodiscard]] float prepare_batch(std::uint64_t rays_per_face);
    void clear_block_scratch();
    // Tiled only: the block's node rows (absolute, ascending) into the host
    // rows, read back whole or compacted on the device.
    void absorb_block(std::span<const std::uint32_t> rows,
                      std::uint32_t row_offset);
    void absorb_compact(std::span<const std::uint32_t> rows,
                        std::span<const std::uint32_t> first,
                        std::span<const std::uint32_t> counts,
                        std::span<const std::uint64_t> entries);
    void record_batch(std::span<const std::uint32_t> emitters,
                      std::uint64_t rays_per_face);

  private:
    // On the first batch: each face's emission share and blend u (the band's
    // emissivity as the materials are then), uploaded for the kernel, and
    // the per-node figures the assembly needs.
    void fix_weights();

    SceneImpl& _scene;
    Band _band;
    AccumConfig _config;
    std::vector<NodeNum> _nodes;              // aggregate_nodes, the row labels
    std::vector<std::uint32_t> _face_column;  // per face: its node's position
    // Per node: the largest A eps of its faces (a face deposits its A eps
    // over this, the assembly multiplies it back) and the summed A eps (what
    // the threshold compares against).
    std::vector<double> _node_scale;
    std::vector<double> _node_emissive_area;
    double _emitted_area = 0.0;         // summed A eps of the emitters
    std::uint64_t _max_node_faces = 1;  // fixed-point budget of a row
    GpuBuffer _cells;             // device-local u64, stride columns() + 3
    GpuBuffer _readback;          // host-visible mirror of _cells
    GpuBuffer _face_columns_buf;  // u32 per face
    GpuBuffer _face_weights_buf;  // float4 per face (share, u, u rest, 0)
    std::vector<HostCountRow> _host_rows;  // Tiled: per node row
    std::vector<std::uint32_t> _emitters;  // fixed by the first batch
    double _fp_scale = 0.0;
    std::uint64_t _rays_per_face = 0;
    std::uint64_t _total_rays = 0;
    // Absorptance groups (Dense only).
    std::uint32_t _derivative_count = 0;
    GpuBuffer _groups_buf;        // u32 per face, ~0u for no group
    GpuBuffer _derivative_cells;  // device-local signed u64, count per cell
    GpuBuffer _derivative_readback;
};

class SolarAccumImpl {
  public:
    explicit SolarAccumImpl(SceneImpl& scene);
    ~SolarAccumImpl();
    SolarAccumImpl(const SolarAccumImpl&) = delete;
    SolarAccumImpl& operator=(const SolarAccumImpl&) = delete;
    SolarAccumImpl(SolarAccumImpl&&) = delete;
    SolarAccumImpl& operator=(SolarAccumImpl&&) = delete;

    void reset();
    [[nodiscard]] SolarResult build_result() const;

    // --- called by SceneImpl::accumulate_solar ------------------------------
    [[nodiscard]] VkBuffer direct_buffer() const noexcept {
        return _direct.buffer;
    }
    [[nodiscard]] VkBuffer total_buffer() const noexcept {
        return _total.buffer;
    }
    // Records the sun on the first batch (later batches must match — summing
    // different suns is meaningless), fixes the fixed-point scale and guards
    // the cumulative ray budget. Returns the normalized sun direction and
    // the scale for the push constants.
    struct BatchSetup {
        std::array<float, 3> sun_dir;
        float fp_scale;
    };
    // ray_area: the largest energy one ray can deposit, in square metres of
    // collector (0 = a face sample's, bounded by the emitter area; the
    // sun-plane source passes its rectangle's area).
    [[nodiscard]] BatchSetup prepare_batch(const SolarState& sun,
                                           std::uint64_t rays_per_face,
                                           double ray_area = 0.0);
    void record_batch(std::uint64_t rays_per_face, std::size_t num_emitters);
    [[nodiscard]] SceneImpl& scene() const noexcept { return _scene; }

  private:
    SceneImpl& _scene;
    GpuBuffer _direct;  // u64 fixed-point absorbed direct energy per face
    GpuBuffer _total;   // u64 fixed-point absorbed total energy per face
    GpuBuffer _direct_readback;  // host-visible mirrors, filled per read
    GpuBuffer _total_readback;
    SolarState _sun{};
    bool _sun_recorded = false;
    double _fp_scale = 0.0;
    // Deposits carry the emitting face's area, so the fixed-point budget
    // scales with the summed face area (rounded up, at least 1).
    std::uint64_t _area_units = 1;
    std::uint64_t _rays_per_face = 0;
    std::uint64_t _total_rays = 0;
};

}  // namespace pycanha::radiative::detail
