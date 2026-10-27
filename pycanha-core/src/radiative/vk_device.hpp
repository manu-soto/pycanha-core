#pragma once

// src-private: volk / VMA / all Vk* usage stays below src/radiative so the
// public API exposes no Vulkan types (keeps a future non-Vulkan backend a
// pure implementation swap). Public headers must never include this file.

// This header is the module's Vulkan gateway: it exports the Vulkan API to
// every radiative TU. volk MUST precede any Vulkan-including header (it
// defines VK_NO_PROTOTYPES; a prototyped vulkan.h seen first is a hard
// error), so no other file may include vulkan/volk/VMA headers directly.
#include <volk.h>                // IWYU pragma: export
#include <vulkan/vulkan_core.h>  // IWYU pragma: export

// VMA's function-pointer sourcing (VMA_STATIC_VULKAN_FUNCTIONS=0 /
// VMA_DYNAMIC_VULKAN_FUNCTIONS=1, volk provides everything) is configured as
// target-wide compile definitions in CMake so every TU sees one config.
#include <vk_mem_alloc.h>  // IWYU pragma: export

#include <cstdint>
#include <optional>
#include <vector>

#include "pycanha-core/radiative/device.hpp"

namespace pycanha::radiative::detail {

// Process-wide loader + instance bootstrap. Created once on first use and
// kept alive for the process lifetime: volk's global dispatch table is
// loaded from this instance, so destroying/recreating instances would
// invalidate every loaded function pointer. Returns VK_NULL_HANDLE when no
// Vulkan driver is present — the one place that detects "no Vulkan".
[[nodiscard]] VkInstance shared_instance();

// One physical device with its capability verdict. `info.ray_tracing` is
// true only when the device supports everything the kernels need:
// Vulkan >= 1.3; extensions acceleration_structure + ray_query +
// deferred_host_operations; features shaderInt64, bufferDeviceAddress,
// scalarBlockLayout, timelineSemaphore, shaderBufferInt64Atomics (the
// fixed-point accumulator cells), synchronization2, maintenance4,
// accelerationStructure, rayQuery.
struct PhysicalDeviceCheck {
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    DeviceInfo info;
    bool has_memory_budget = false;
    // maxComputeWorkGroupCount[1]; see DeviceImpl::max_dispatch_emitters.
    std::uint32_t max_dispatch_emitters = 0;
    // limits.maxStorageBufferRange; see DeviceImpl::max_storage_buffer_range.
    std::uint64_t max_storage_buffer_range = 0;
    // Preference when no device index is requested, highest wins: discrete >
    // integrated > other hardware > software (a software rasterizer like
    // lavapipe is a valid last resort — same SPIR-V, just slow). Ranking the
    // devices here rather than in the shared selection code keeps every
    // driver query inside the backend.
    int score = 0;
};

// Empty when no driver / no devices. Order matches enumerate_devices().
[[nodiscard]] std::vector<PhysicalDeviceCheck> enumerate_physical_devices();

// Logical device + queues + VMA allocator over one capable physical device.
class DeviceImpl {
  public:
    // `picked` must have info.ray_tracing == true.
    explicit DeviceImpl(const PhysicalDeviceCheck& picked);
    ~DeviceImpl();
    DeviceImpl(const DeviceImpl&) = delete;
    DeviceImpl& operator=(const DeviceImpl&) = delete;
    DeviceImpl(DeviceImpl&&) = delete;
    DeviceImpl& operator=(DeviceImpl&&) = delete;

    [[nodiscard]] std::uint64_t memory_budget() const;

    DeviceInfo info;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    std::uint32_t compute_family = 0;
    VkQueue compute_queue = VK_NULL_HANDLE;
    // Dedicated transfer family (TRANSFER without GRAPHICS|COMPUTE) when the
    // hardware has one; otherwise transfer_queue == compute_queue.
    std::optional<std::uint32_t> transfer_family;
    VkQueue transfer_queue = VK_NULL_HANDLE;
    VmaAllocator allocator = VK_NULL_HANDLE;
    bool has_memory_budget = false;
    // Most emitting faces one kernel launch may cover. The kernels give each
    // emitter one workgroup row in the dispatch's Y dimension, and Vulkan
    // only guarantees maxComputeWorkGroupCount[1] >= 65535 (NVIDIA reports
    // exactly that), so longer emitter lists go out in batches of this size.
    std::uint32_t max_dispatch_emitters = 0;
    // The largest storage buffer a kernel can address (4 GiB on NVIDIA): an
    // accumulator buffer beyond it would be written and read only partly,
    // with no error, so the accumulators refuse it.
    std::uint64_t max_storage_buffer_range = 0;
};

}  // namespace pycanha::radiative::detail
