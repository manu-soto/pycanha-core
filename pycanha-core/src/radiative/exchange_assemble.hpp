#pragma once

// Backend-agnostic exchange assembly. Turning accumulated fixed-point
// deposits into an ExchangeResult is pure host arithmetic — no Vulkan, no
// Objective-C — so both backends call the same implementation instead of
// mirroring it.
//
// That matters more than ordinary de-duplication: the Metal backend cannot
// be compiled on a machine without a Mac, so a mistake mirrored into the
// .mm file produces no error, no warning and no failing test until someone
// with Apple hardware next builds the branch. Keeping the logic here means
// the only Metal-side code is a call.
//
// The traversal itself — the cache-blocked tile-pair walk, the sparse merge,
// the weight table, the CSR packing — is shared with the view-factor
// assembly and lives in pair_walk.hpp. What stays here is only what knows
// that a cell is fixed-point deposited energy.

#include <cstddef>
#include <cstdint>
#include <span>
#include <variant>
#include <vector>

#include "pair_walk.hpp"
#include "pycanha-core/radiative/materials.hpp"
#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/settings.hpp"

namespace pycanha::radiative::detail {

// Where the accumulated deposits live, which is the only thing that differs
// between the two layouts:
//  - Dense: the mapped GPU cell block, rows * (rows + num_virtual_columns)
//    u64 cells in row-major order. The caller has already made it readable
//    on the host (Vulkan needs an explicit invalidate; Metal's unified
//    memory does not).
//  - Tiled: one host map per row. The GPU buffer only ever held one block
//    of scratch, which was drained into these maps after every dispatch.
//
// Cells hold energy in fixed point: the integer count of 1/fp_scale units
// deposited. Integer adds are associative, so the two layouts accumulate to
// the same cells and every number derived from them is bit-identical.
using ExchangeCellSource =
    std::variant<std::span<const std::uint64_t>, std::span<const HostCountRow>>;

// The traced band's absorptivity per face, which is what weights
// emission and absorption alike. A face with no material assigned is a
// blackbody (the scene warns about that at build time), so it reads 1.
[[nodiscard]] std::vector<double> band_emissivity(
    const MaterialTable& materials, Band band);

struct ExchangeInputs {
    ExchangeCellSource cells;
    // Per face; `rays_per_row` defines the face count.
    std::span<const double> areas;
    std::span<const double> emissivity;
    std::span<const std::uint64_t> rays_per_row;
    // The power of two the accumulator fixed on its first batch, or 0 when
    // nothing was ever traced (which yields an empty result).
    double fp_scale = 0.0;
    Band band = Band::IR;
};

// Ray counts and trace timings that belong to the accumulator rather than
// the cells (total_rays, rays_per_face, gpu_time) are filled in by the
// caller.
[[nodiscard]] ExchangeResult assemble_exchange(
    const ExchangeInputs& in, const AccumConfig& config,
    const AssemblyTuning& tuning = {});

// Node rows (NodeExchangeAccumulator). The kernel has already summed the
// cells to nodes and scaled every deposit by its emission share and pair
// weight, so what is left is a unit per row, the transpose sum that makes
// the upper triangle, and the buckets. A row's cells (stride num_nodes + 4):
// the node columns, then the faces in no node, space, inactive, lost.
struct NodeExchangeInputs {
    ExchangeCellSource cells;
    std::size_t num_nodes = 0;
    // Per node: the largest A eps of its faces, which a deposit was divided
    // by in the kernel, and the summed A eps (the threshold's divisor).
    std::span<const double> node_scale;
    std::span<const double> node_emissive_area;
    // The rays every emitter sent, and the summed A eps of the emitters
    // (the lost-energy fraction's denominator).
    std::uint64_t rays_per_face = 0;
    double emitted_area = 0.0;
    double fp_scale = 0.0;
    Band band = Band::IR;
};

// `nodes`, total_rays and rays_per_face are filled in by the caller. The
// lost-energy fraction is the lost H over the emitted A eps (the face result
// takes it over the rays); the per-entry standard errors and the reciprocity
// residual are face-pair quantities and stay 0.
[[nodiscard]] NodeExchangeResult assemble_node_exchange(
    const NodeExchangeInputs& in, const AccumConfig& config);

// Largest deviation of a row's total deposits from the energy that row
// emitted, in fixed-point units. Everything wraps mod 2^64 — the identity
// the kernels maintain — so a correct trace returns exactly 0.
[[nodiscard]] std::uint64_t exchange_conservation_error(
    ExchangeCellSource cells, std::span<const std::uint64_t> rays_per_row,
    double fp_scale);

}  // namespace pycanha::radiative::detail
