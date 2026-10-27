#include "vk_device.hpp"

#include <spdlog/spdlog.h>
#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "pycanha-core/radiative/device.hpp"
#include "pycanha-core/utils/logger.hpp"
#include "vk_test_hooks.hpp"

namespace pycanha::radiative::detail {

namespace {

// Workgroup x-size every compute kernel is built with; together with the
// device's max workgroup count it bounds the rays per dispatch.
constexpr std::uint64_t workgroup_size = 64;

constexpr std::array<const char*, 3> required_device_extensions = {
    VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VK_KHR_RAY_QUERY_EXTENSION_NAME,
    VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
};

// The full enum name does not fit the line-length limit anywhere it is
// needed, so alias it once.
constexpr VkStructureType stype_accel_features =
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;

// Zero-initialized feature-struct chain used both to QUERY capabilities and
// (with the required bits set) to ENABLE them at device creation.
struct FeatureChain {
    VkPhysicalDeviceRayQueryFeaturesKHR ray_query{};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accel{};
    VkPhysicalDeviceVulkan13Features vk13{};
    VkPhysicalDeviceVulkan12Features vk12{};
    VkPhysicalDeviceFeatures2 features2{};

    FeatureChain() noexcept {
        ray_query.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
        accel.sType = stype_accel_features;
        accel.pNext = &ray_query;
        vk13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        vk13.pNext = &accel;
        vk12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        vk12.pNext = &vk13;
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = &vk12;
    }

    [[nodiscard]] bool has_all_required() const noexcept {
        return features2.features.shaderInt64 == VK_TRUE &&
               vk12.bufferDeviceAddress == VK_TRUE &&
               vk12.scalarBlockLayout == VK_TRUE &&
               vk12.timelineSemaphore == VK_TRUE &&
               vk12.shaderBufferInt64Atomics == VK_TRUE &&
               vk13.synchronization2 == VK_TRUE &&
               vk13.maintenance4 == VK_TRUE &&
               accel.accelerationStructure == VK_TRUE &&
               ray_query.rayQuery == VK_TRUE;
    }

    void set_all_required() noexcept {
        features2.features.shaderInt64 = VK_TRUE;
        vk12.bufferDeviceAddress = VK_TRUE;
        vk12.scalarBlockLayout = VK_TRUE;
        vk12.timelineSemaphore = VK_TRUE;
        vk12.shaderBufferInt64Atomics = VK_TRUE;
        vk13.synchronization2 = VK_TRUE;
        vk13.maintenance4 = VK_TRUE;
        accel.accelerationStructure = VK_TRUE;
        ray_query.rayQuery = VK_TRUE;
    }
};

struct ExtensionSupport {
    bool all_required = false;
    bool memory_budget = false;
};

[[nodiscard]] ExtensionSupport query_extensions(VkPhysicalDevice device) {
    std::uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> extensions(count);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count,
                                         extensions.data());

    const auto has = [&extensions](const char* name) {
        return std::ranges::any_of(
            extensions, [name](const VkExtensionProperties& ext) {
                return std::strcmp(static_cast<const char*>(ext.extensionName),
                                   name) == 0;
            });
    };

    ExtensionSupport support;
    support.all_required =
        std::ranges::all_of(required_device_extensions,
                            [&has](const char* name) { return has(name); });
    support.memory_budget = has(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    return support;
}

[[nodiscard]] PhysicalDeviceCheck check_device(VkPhysicalDevice device,
                                               std::uint32_t index) {
    PhysicalDeviceCheck check;
    check.physical_device = device;

    VkPhysicalDeviceProperties2 props2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = nullptr,
        .properties = {}};
    vkGetPhysicalDeviceProperties2(device, &props2);
    const VkPhysicalDeviceProperties& props = props2.properties;

    check.info.name = static_cast<const char*>(props.deviceName);
    check.info.index = index;
    check.info.software = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
    check.info.max_dispatch_rays =
        static_cast<std::uint64_t>(props.limits.maxComputeWorkGroupCount[0]) *
        workgroup_size;
    check.max_dispatch_emitters = props.limits.maxComputeWorkGroupCount[1];
    check.max_storage_buffer_range = props.limits.maxStorageBufferRange;
    if (check.info.software) {
        check.score = 0;
    } else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
        check.score = 3;
    } else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
        check.score = 2;
    } else {
        check.score = 1;
    }

    const ExtensionSupport extensions = query_extensions(device);
    check.has_memory_budget = extensions.memory_budget;

    FeatureChain chain;
    vkGetPhysicalDeviceFeatures2(device, &chain.features2);

    check.info.ray_tracing = props.apiVersion >= VK_API_VERSION_1_3 &&
                             extensions.all_required &&
                             chain.has_all_required();
    return check;
}

struct QueueFamilies {
    std::uint32_t compute = 0;
    std::optional<std::uint32_t> dedicated_transfer;
};

[[nodiscard]] QueueFamilies pick_queue_families(VkPhysicalDevice device) {
    std::uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());

    QueueFamilies picked;
    bool compute_found = false;
    for (std::uint32_t family = 0; family < count; ++family) {
        const VkQueueFlags flags = families[family].queueFlags;
        if (!compute_found && (flags & VK_QUEUE_COMPUTE_BIT) != 0U) {
            picked.compute = family;
            compute_found = true;
        }
        constexpr auto graphics_or_compute =
            static_cast<VkQueueFlags>(VK_QUEUE_GRAPHICS_BIT) |
            static_cast<VkQueueFlags>(VK_QUEUE_COMPUTE_BIT);
        if (!picked.dedicated_transfer.has_value() &&
            (flags & VK_QUEUE_TRANSFER_BIT) != 0U &&
            (flags & graphics_or_compute) == 0U) {
            picked.dedicated_transfer = family;
        }
    }
    if (!compute_found) {
        throw std::runtime_error(
            "pycanha::radiative: selected Vulkan device has no compute "
            "queue family");
    }
    return picked;
}

}  // namespace

VkInstance shared_instance() {
    static VkInstance instance = []() -> VkInstance {
        if (volkInitialize() != VK_SUCCESS) {
            // Expected on any machine without a Vulkan driver, and already
            // reported through is_available(); not worth a record per import.
            SPDLOG_LOGGER_DEBUG(pycanha::get_logger(),
                                "radiative: no Vulkan loader/driver found");
            return VK_NULL_HANDLE;
        }

        const VkApplicationInfo app_info{
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pNext = nullptr,
            .pApplicationName = "pycanha-core",
            .applicationVersion = 0,
            .pEngineName = "pycanha::radiative",
            .engineVersion = 0,
            .apiVersion = VK_API_VERSION_1_3};
        const VkInstanceCreateInfo create_info{
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .pApplicationInfo = &app_info,
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = nullptr,
            .enabledExtensionCount = 0,
            .ppEnabledExtensionNames = nullptr};

        VkInstance created = VK_NULL_HANDLE;
        if (vkCreateInstance(&create_info, nullptr, &created) != VK_SUCCESS) {
            SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                               "radiative: Vulkan instance creation failed");
            return VK_NULL_HANDLE;
        }
        volkLoadInstance(created);
        // Intentionally never destroyed: process-lifetime singleton (volk's
        // dispatch table points into it).
        return created;
    }();
    return instance;
}

std::vector<PhysicalDeviceCheck> enumerate_physical_devices() {
    std::vector<PhysicalDeviceCheck> checks;
    VkInstance instance = shared_instance();
    if (instance == VK_NULL_HANDLE) {
        return checks;
    }

    std::uint32_t count = 0;
    if (vkEnumeratePhysicalDevices(instance, &count, nullptr) != VK_SUCCESS) {
        return checks;
    }
    std::vector<VkPhysicalDevice> devices(count);
    if (vkEnumeratePhysicalDevices(instance, &count, devices.data()) !=
        VK_SUCCESS) {
        return checks;
    }

    checks.reserve(count);
    for (std::uint32_t index = 0; index < count; ++index) {
        checks.push_back(check_device(devices[index], index));
    }
    return checks;
}

DeviceImpl::DeviceImpl(const PhysicalDeviceCheck& picked)
    : info(picked.info),
      physical_device(picked.physical_device),
      has_memory_budget(picked.has_memory_budget),
      max_dispatch_emitters(picked.max_dispatch_emitters),
      max_storage_buffer_range(picked.max_storage_buffer_range) {
    const QueueFamilies families = pick_queue_families(physical_device);
    compute_family = families.compute;
    transfer_family = families.dedicated_transfer;

    const float priority = 1.0F;
    std::vector<VkDeviceQueueCreateInfo> queue_infos;
    queue_infos.push_back(VkDeviceQueueCreateInfo{
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .queueFamilyIndex = compute_family,
        .queueCount = 1,
        .pQueuePriorities = &priority});
    if (transfer_family.has_value()) {
        queue_infos.push_back(VkDeviceQueueCreateInfo{
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .queueFamilyIndex = *transfer_family,
            .queueCount = 1,
            .pQueuePriorities = &priority});
    }

    std::vector<const char*> extensions(std::begin(required_device_extensions),
                                        std::end(required_device_extensions));
    if (has_memory_budget) {
        extensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    }

    FeatureChain enabled;
    enabled.set_all_required();

    const VkDeviceCreateInfo create_info{
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &enabled.features2,
        .flags = 0,
        .queueCreateInfoCount = static_cast<std::uint32_t>(queue_infos.size()),
        .pQueueCreateInfos = queue_infos.data(),
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = nullptr,
        .enabledExtensionCount = static_cast<std::uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
        .pEnabledFeatures = nullptr};

    if (vkCreateDevice(physical_device, &create_info, nullptr, &device) !=
        VK_SUCCESS) {
        throw std::runtime_error(
            "pycanha::radiative: Vulkan device creation failed for '" +
            info.name + "'");
    }
    volkLoadDevice(device);

    vkGetDeviceQueue(device, compute_family, 0, &compute_queue);
    if (transfer_family.has_value()) {
        vkGetDeviceQueue(device, *transfer_family, 0, &transfer_queue);
    } else {
        transfer_queue = compute_queue;
    }

    VmaVulkanFunctions vma_functions{};
    vma_functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vma_functions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
    VmaAllocatorCreateInfo allocator_info{};
    allocator_info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    allocator_info.physicalDevice = physical_device;
    allocator_info.device = device;
    allocator_info.instance = shared_instance();
    allocator_info.vulkanApiVersion = VK_API_VERSION_1_3;
    allocator_info.pVulkanFunctions = &vma_functions;
    if (vmaCreateAllocator(&allocator_info, &allocator) != VK_SUCCESS) {
        vkDestroyDevice(device, nullptr);
        device = VK_NULL_HANDLE;
        throw std::runtime_error(
            "pycanha::radiative: VMA allocator creation failed for '" +
            info.name + "'");
    }

    SPDLOG_LOGGER_INFO(pycanha::get_logger(),
                       "radiative: created device '{}' (software={}, "
                       "dedicated transfer queue={})",
                       info.name, info.software, transfer_family.has_value());
}

DeviceImpl::~DeviceImpl() {
    if (device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device);
        if (allocator != VK_NULL_HANDLE) {
            vmaDestroyAllocator(allocator);
        }
        vkDestroyDevice(device, nullptr);
    }
}

std::uint64_t DeviceImpl::memory_budget() const {
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT,
        .pNext = nullptr,
        .heapBudget = {},
        .heapUsage = {}};
    VkPhysicalDeviceMemoryProperties2 memory2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
        .pNext = has_memory_budget ? &budget : nullptr,
        .memoryProperties = {}};
    vkGetPhysicalDeviceMemoryProperties2(physical_device, &memory2);

    const VkPhysicalDeviceMemoryProperties& memory = memory2.memoryProperties;
    const std::span<const VkMemoryHeap> heaps(memory.memoryHeaps);
    std::size_t largest_heap = 0;
    VkDeviceSize largest_size = 0;
    for (std::size_t heap = 0; heap < memory.memoryHeapCount; ++heap) {
        if ((heaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0U &&
            heaps[heap].size > largest_size) {
            largest_heap = heap;
            largest_size = heaps[heap].size;
        }
    }

    if (has_memory_budget) {
        const std::span<const VkDeviceSize> budgets(budget.heapBudget);
        const std::span<const VkDeviceSize> usages(budget.heapUsage);
        const VkDeviceSize available =
            budgets[largest_heap] > usages[largest_heap]
                ? budgets[largest_heap] - usages[largest_heap]
                : 0;
        return static_cast<std::uint64_t>(available);
    }
    // Without the budget extension: assume 80% of the largest DEVICE_LOCAL
    // heap is usable (leaves headroom for the driver and other processes).
    return static_cast<std::uint64_t>(largest_size) * 8 / 10;
}

std::uint32_t max_dispatch_emitters(const Device& device) {
    return device.impl().max_dispatch_emitters;
}

void lower_max_dispatch_emitters(Device& device, std::uint32_t limit) {
    DeviceImpl& impl = device.impl();
    if (limit == 0 || limit > impl.max_dispatch_emitters) {
        throw std::invalid_argument(
            "pycanha::radiative: the emitter limit can only be lowered, and "
            "not to zero");
    }
    impl.max_dispatch_emitters = limit;
}

}  // namespace pycanha::radiative::detail
