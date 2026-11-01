#pragma once

#include <cstdint>

#include "pycanha-core/radiative/settings.hpp"

namespace pycanha::radiative {

class RadiativeScene;

// Sizing mechanism for the accumulator memory policy: the C++ side reports
// exact byte requirements, the caller (the Python layer) decides layout and
// tile_rows against Device::memory_budget() — so runs fail fast with a clear
// message instead of exhausting device memory mid-trace.
struct MemoryEstimate {
    // Full num_faces x (num_faces + virtual columns) accumulator, sized for
    // the u64 exchange cells (vf counting cells take half).
    std::uint64_t gpu_bytes_dense = 0;
    // One row of the tiled block scratch (+ nothing else — the scratch is
    // the only per-row cost).
    std::uint64_t gpu_bytes_per_tile_row = 0;
    // Resident scene cost: geometry, acceleration structures, tables.
    std::uint64_t gpu_bytes_scene = 0;
    // One readback block for the configured layout.
    std::uint64_t host_bytes_block = 0;
};

[[nodiscard]] MemoryEstimate estimate_memory(const RadiativeScene& scene,
                                             const AccumConfig& config = {});

// The same for a NodeExchangeAccumulator over `num_nodes` nodes: its rows are
// nodes and its row stride num_nodes + 1 + 3 (the faces in no node share one
// column), so the dense and per-row figures follow the node count; the
// scene cost is the scene's.
[[nodiscard]] MemoryEstimate estimate_node_memory(
    const RadiativeScene& scene, std::uint64_t num_nodes,
    const AccumConfig& config = {});

}  // namespace pycanha::radiative
