// Public RadiativeScene / VfAccumulator API over the src-private backend
// scene and accumulator layers (Vulkan everywhere except macOS, where it is
// Metal). Both backends define the same detail:: class names, so nothing
// below this comment is backend-specific.

#include "pycanha-core/radiative/scene.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/radiative/accumulators.hpp"
#include "pycanha-core/radiative/device.hpp"
#include "pycanha-core/radiative/materials.hpp"
#include "pycanha-core/radiative/memory.hpp"
#include "pycanha-core/radiative/results.hpp"
#include "pycanha-core/radiative/scene_part.hpp"
#include "pycanha-core/radiative/settings.hpp"
#ifdef __APPLE__
#include "mtl_accum.hpp"
#include "mtl_scene.hpp"
#else
#include "vk_accum.hpp"
#include "vk_scene.hpp"
#endif

namespace pycanha::radiative {

RadiativeScene::RadiativeScene(Device& device, std::vector<ScenePart> parts,
                               MaterialTable materials)
    : _impl(std::make_unique<detail::SceneImpl>(device.impl(), std::move(parts),
                                                std::move(materials))) {}

RadiativeScene::~RadiativeScene() = default;
RadiativeScene::RadiativeScene(RadiativeScene&&) noexcept = default;
RadiativeScene& RadiativeScene::operator=(RadiativeScene&&) noexcept = default;

void RadiativeScene::set_part_transform(
    std::uint32_t part_id, const gmm::CoordinateTransformation& world_tf) {
    _impl->set_part_transform(part_id, world_tf);
}

void RadiativeScene::commit() { _impl->commit(); }

void RadiativeScene::accumulate_vf(VfAccumulator& acc,
                                   const TraceSettings& settings,
                                   std::span<const std::uint32_t> emitters) {
    _impl->accumulate_vf(acc.impl(), settings, emitters);
}

void RadiativeScene::accumulate_exchange(
    ExchangeAccumulator& acc, const TraceSettings& settings,
    std::span<const std::uint32_t> emitters) {
    _impl->accumulate_exchange(acc.impl(), settings, emitters);
}

#ifdef __APPLE__
namespace detail {
// The Metal backend has no node accumulator yet; the class only completes
// the public type.
class NodeExchangeAccumImpl {};
}  // namespace detail

namespace {
[[noreturn]] void throw_no_metal_nodes() {
    throw std::runtime_error(
        "pycanha::radiative: NodeExchangeAccumulator is not available on the "
        "Metal backend yet; use ExchangeAccumulator and aggregate_matrix");
}
}  // namespace

void RadiativeScene::accumulate_exchange(
    NodeExchangeAccumulator& /*acc*/, const TraceSettings& /*settings*/,
    std::span<const std::uint32_t> /*emitters*/) {
    throw_no_metal_nodes();
}
#else
void RadiativeScene::accumulate_exchange(
    NodeExchangeAccumulator& acc, const TraceSettings& settings,
    std::span<const std::uint32_t> emitters) {
    _impl->accumulate_exchange(acc.impl(), settings, emitters);
}
#endif

void RadiativeScene::accumulate_solar(const SolarState& sun,
                                      SolarAccumulator& acc,
                                      const TraceSettings& settings) {
    _impl->accumulate_solar(sun, acc.impl(), settings);
}

void RadiativeScene::update_materials(const MaterialTable& materials) {
    _impl->update_materials(materials);
}

std::uint32_t RadiativeScene::num_faces() const noexcept {
    return _impl->num_faces();
}

const MaterialTable& RadiativeScene::materials() const noexcept {
    return _impl->materials();
}

std::span<const double> RadiativeScene::face_areas() const noexcept {
    return _impl->face_areas();
}

detail::SceneImpl& RadiativeScene::impl() const noexcept { return *_impl; }

VfAccumulator::VfAccumulator(const RadiativeScene& scene, AccumConfig config)
    : _impl(std::make_unique<detail::VfAccumImpl>(scene.impl(), config)) {}

VfAccumulator::~VfAccumulator() = default;
VfAccumulator::VfAccumulator(VfAccumulator&&) noexcept = default;
VfAccumulator& VfAccumulator::operator=(VfAccumulator&&) noexcept = default;

void VfAccumulator::reset() { _impl->reset(); }

VfResult VfAccumulator::result() const { return _impl->build_result(); }

detail::VfAccumImpl& VfAccumulator::impl() noexcept { return *_impl; }

ExchangeAccumulator::ExchangeAccumulator(const RadiativeScene& scene, Band band,
                                         AccumConfig config)
    : _impl(std::make_unique<detail::ExchangeAccumImpl>(scene.impl(), band,
                                                        config)) {}

ExchangeAccumulator::~ExchangeAccumulator() = default;
ExchangeAccumulator::ExchangeAccumulator(ExchangeAccumulator&&) noexcept =
    default;
ExchangeAccumulator& ExchangeAccumulator::operator=(
    ExchangeAccumulator&&) noexcept = default;

void ExchangeAccumulator::reset() { _impl->reset(); }

ExchangeResult ExchangeAccumulator::result() const {
    return _impl->build_result();
}

std::uint64_t ExchangeAccumulator::conservation_error() const {
    return _impl->conservation_error();
}

detail::ExchangeAccumImpl& ExchangeAccumulator::impl() noexcept {
    return *_impl;
}

#ifdef __APPLE__
NodeExchangeAccumulator::NodeExchangeAccumulator(
    const RadiativeScene& /*scene*/, Band /*band*/,
    std::span<const NodeNum> /*node_numbers*/, AccumConfig /*config*/) {
    throw_no_metal_nodes();
}

void NodeExchangeAccumulator::reset() { throw_no_metal_nodes(); }

NodeExchangeResult NodeExchangeAccumulator::result() const {
    throw_no_metal_nodes();
}
#else
NodeExchangeAccumulator::NodeExchangeAccumulator(
    const RadiativeScene& scene, Band band,
    std::span<const NodeNum> node_numbers, AccumConfig config)
    : _impl(std::make_unique<detail::NodeExchangeAccumImpl>(
          scene.impl(), band, node_numbers, config)) {}

void NodeExchangeAccumulator::reset() { _impl->reset(); }

NodeExchangeResult NodeExchangeAccumulator::result() const {
    return _impl->build_result();
}
#endif

NodeExchangeAccumulator::~NodeExchangeAccumulator() = default;
NodeExchangeAccumulator::NodeExchangeAccumulator(
    NodeExchangeAccumulator&&) noexcept = default;
NodeExchangeAccumulator& NodeExchangeAccumulator::operator=(
    NodeExchangeAccumulator&&) noexcept = default;

detail::NodeExchangeAccumImpl& NodeExchangeAccumulator::impl() noexcept {
    return *_impl;
}

SolarAccumulator::SolarAccumulator(const RadiativeScene& scene)
    : _impl(std::make_unique<detail::SolarAccumImpl>(scene.impl())) {}

SolarAccumulator::~SolarAccumulator() = default;
SolarAccumulator::SolarAccumulator(SolarAccumulator&&) noexcept = default;
SolarAccumulator& SolarAccumulator::operator=(SolarAccumulator&&) noexcept =
    default;

void SolarAccumulator::reset() { _impl->reset(); }

SolarResult SolarAccumulator::result() const { return _impl->build_result(); }

detail::SolarAccumImpl& SolarAccumulator::impl() noexcept { return *_impl; }

MemoryEstimate estimate_memory(const RadiativeScene& scene,
                               const AccumConfig& config) {
    const std::uint64_t faces = scene.num_faces();
    const std::uint64_t cols =
        faces + static_cast<std::uint64_t>(num_virtual_columns);
    // Sized for the u64 exchange cells — the worst case (vf uses u32).
    constexpr std::uint64_t cell_bytes = sizeof(std::uint64_t);

    MemoryEstimate out;
    out.gpu_bytes_dense = faces * cols * cell_bytes;
    out.gpu_bytes_per_tile_row = cols * cell_bytes;
    out.gpu_bytes_scene = scene.impl().scene_bytes();
    const std::uint64_t block_rows =
        config.layout == AccumLayout::Tiled
            ? std::min<std::uint64_t>(
                  std::max<std::uint64_t>(config.tile_rows, 1), faces)
            : faces;
    out.host_bytes_block = out.gpu_bytes_per_tile_row * block_rows;
    return out;
}

MemoryEstimate estimate_node_memory(const RadiativeScene& scene,
                                    std::uint64_t num_nodes,
                                    const AccumConfig& config) {
    // The node columns, the one shared by faces in no node, the buckets.
    const std::uint64_t cols =
        num_nodes + 1 + static_cast<std::uint64_t>(num_virtual_columns);
    constexpr std::uint64_t cell_bytes = sizeof(std::uint64_t);

    MemoryEstimate out;
    out.gpu_bytes_dense = num_nodes * cols * cell_bytes;
    out.gpu_bytes_per_tile_row = cols * cell_bytes;
    out.gpu_bytes_scene = scene.impl().scene_bytes();
    const std::uint64_t block_rows =
        config.layout == AccumLayout::Tiled
            ? std::min<std::uint64_t>(
                  std::max<std::uint64_t>(config.tile_rows, 1), num_nodes)
            : num_nodes;
    out.host_bytes_block = out.gpu_bytes_per_tile_row * block_rows;
    return out;
}

}  // namespace pycanha::radiative
