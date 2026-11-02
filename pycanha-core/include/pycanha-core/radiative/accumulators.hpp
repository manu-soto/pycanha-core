#pragma once

#include <memory>
#include <span>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/settings.hpp"

namespace pycanha::radiative {

class RadiativeScene;

namespace detail {
class VfAccumImpl;
class ExchangeAccumImpl;
class NodeExchangeAccumImpl;
class SolarAccumImpl;
}  // namespace detail

// Owns the GPU-side counting buffer plus the bookkeeping needed to turn raw
// counts into a VfResult. Batch-additive: successive accumulate_vf calls add
// up; reset() clears everything. Cells are integers, so results are
// bit-deterministic for a given seed regardless of dispatch shape.
class VfAccumulator {
  public:
    // Dense allocates the full num_num_faces x num_faces buffer up front (the
    // small-model fast path); Tiled bounds GPU memory to tile_rows x
    // num_faces and streams row blocks into host-side sparse storage. Both
    // produce bit-identical results for the same seed.
    explicit VfAccumulator(const RadiativeScene& scene,
                           AccumConfig config = {});
    ~VfAccumulator();
    VfAccumulator(VfAccumulator&&) noexcept;
    VfAccumulator& operator=(VfAccumulator&&) noexcept;
    VfAccumulator(const VfAccumulator&) = delete;
    VfAccumulator& operator=(const VfAccumulator&) = delete;

    void reset();
    // Readback + normalization + statistics. Callable repeatedly; each call
    // reflects everything accumulated so far.
    [[nodiscard]] VfResult result() const;

    [[nodiscard]] detail::VfAccumImpl& impl() noexcept;

  private:
    std::unique_ptr<detail::VfAccumImpl> _impl;
};

// Exchange-factor accumulator: u64 fixed-point energy cells plus per-row
// space/lost balances. The band is fixed at construction (mixing bands in
// one matrix would be meaningless). Same Dense/Tiled layout semantics and
// bit-determinism guarantees as VfAccumulator.
class ExchangeAccumulator {
  public:
    explicit ExchangeAccumulator(const RadiativeScene& scene, Band band,
                                 AccumConfig config = {});
    ~ExchangeAccumulator();
    ExchangeAccumulator(ExchangeAccumulator&&) noexcept;
    ExchangeAccumulator& operator=(ExchangeAccumulator&&) noexcept;
    ExchangeAccumulator(const ExchangeAccumulator&) = delete;
    ExchangeAccumulator& operator=(const ExchangeAccumulator&) = delete;

    void reset();
    [[nodiscard]] ExchangeResult result() const;
    // Max over rows of |full row sum - rays * scale| in raw fixed-point
    // units (wrapping u64 arithmetic, virtual bucket columns included).
    // Zero by construction — the kernel flushes every ray's remaining
    // balance into a column — so a nonzero value means a broken kernel,
    // not Monte-Carlo noise.
    [[nodiscard]] std::uint64_t conservation_error() const;

    [[nodiscard]] detail::ExchangeAccumImpl& impl() noexcept;

  private:
    std::unique_ptr<detail::ExchangeAccumImpl> _impl;
};

// Exchange factors summed to nodes while they are traced. Where
// ExchangeAccumulator holds face x face cells and aggregate_matrix sums them
// to nodes afterwards, this one holds node x node cells: a face deposits
// into the row and column of its node, scaled by the factors the face
// assembly would apply to it (its emissive area, and the weight that
// combines a face pair's two directions). The result is the aggregated face
// result up to float rounding, while the GPU buffer and the host memory
// follow the node pairs instead of the face pairs, and stop growing with the
// rays.
//
// node_numbers: one per scene face, NO_NODE for a face in no node, as
// aggregate_matrix takes them. Faces in no node are not traced; energy that
// lands on them is reported in the unassigned column.
//
// Requirements, checked: every batch traces the same emitters (rays may
// differ between batches but not between faces, which is what lets the
// face pair weights be known while tracing); TriangulationMode None or
// RayDensity (ConstrainedLeastSquares is a projection over the whole face
// matrix); keep_full_matrix off. A RayDensity trace weighs the two
// directions by the faces' areas, which on such a uniform trace is what
// both settings of TriangulationConfig::area_weights give. Vulkan only.
//
// absorptance_groups (optional): one per scene face, -1 for none or 0 to
// max_absorptance_groups - 1. The result then also holds, per group, the
// derivatives of its factors with respect to the band absorptance of the
// group's faces (NodeExchangeResult::absorptance_derivatives), traced with
// the same rays. Dense layout only.
class NodeExchangeAccumulator {
  public:
    NodeExchangeAccumulator(
        const RadiativeScene& scene, Band band,
        std::span<const NodeNum> node_numbers, AccumConfig config = {},
        std::span<const std::int32_t> absorptance_groups = {});
    ~NodeExchangeAccumulator();
    NodeExchangeAccumulator(NodeExchangeAccumulator&&) noexcept;
    NodeExchangeAccumulator& operator=(NodeExchangeAccumulator&&) noexcept;
    NodeExchangeAccumulator(const NodeExchangeAccumulator&) = delete;
    NodeExchangeAccumulator& operator=(const NodeExchangeAccumulator&) = delete;

    void reset();
    [[nodiscard]] NodeExchangeResult result() const;

    [[nodiscard]] detail::NodeExchangeAccumImpl& impl() noexcept;

  private:
    std::unique_ptr<detail::NodeExchangeAccumImpl> _impl;
};

// Solar-absorption accumulator: per-face direct/total energy vectors
// (the solar kernel is O(Nf); there is no matrix and no layout distinction).
class SolarAccumulator {
  public:
    explicit SolarAccumulator(const RadiativeScene& scene);
    ~SolarAccumulator();
    SolarAccumulator(SolarAccumulator&&) noexcept;
    SolarAccumulator& operator=(SolarAccumulator&&) noexcept;
    SolarAccumulator(const SolarAccumulator&) = delete;
    SolarAccumulator& operator=(const SolarAccumulator&) = delete;

    void reset();
    [[nodiscard]] SolarResult result() const;

    [[nodiscard]] detail::SolarAccumImpl& impl() noexcept;

  private:
    std::unique_ptr<detail::SolarAccumImpl> _impl;
};

}  // namespace pycanha::radiative
