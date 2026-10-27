#pragma once

// src-private scene implementation for the Metal backend: geometry/material
// upload, emission tables, acceleration structures, the compute pipelines
// (vf / exchange / solar) and their chunked dispatch. Mirrors vk_scene.hpp
// one-to-one — same class and method names, so the shared scene.cpp compiles
// against either backend. The two are deliberately independent: there is no
// backend abstraction layer.

#import <Metal/Metal.h>

#include <Eigen/Geometry>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "mtl_device.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/radiative/materials.hpp"
#include "pycanha-core/radiative/scene.hpp"
#include "pycanha-core/radiative/scene_part.hpp"
#include "pycanha-core/radiative/settings.hpp"

namespace pycanha::radiative::detail {

class VfAccumImpl;
class ExchangeAccumImpl;
class SolarAccumImpl;

// Minimal RAII-by-owner buffer: created/destroyed by SceneImpl helpers.
// Every buffer is MTLStorageModeShared — on the unified memory of Apple
// Silicon that one storage mode serves both the staging and the device-local
// role, so there is no separate host-visible flag as on the Vulkan side.
struct GpuBuffer {
    id<MTLBuffer> buffer = nil;
    std::size_t size = 0;
};

// Shared-storage buffers always have host-visible contents; a null mapping
// would be a logic error, and checking makes that explicit. Defined out of
// line so this header holds no Objective-C message sends: the repository's
// clang-format and cpplint checks read .hpp files as plain C++, where a
// message send parses as something else entirely.
[[nodiscard]] void* checked_mapped(const GpuBuffer& buffer);

// C++ mirror of the InstanceData struct in kernels/common.slang — must
// match byte-for-byte (80 bytes). tf_rows are rotation rows with the
// translation in .w (world = R * p + t).
struct InstanceDataGpu {
    std::uint64_t vertices_addr;
    std::uint64_t indices_addr;
    std::uint64_t tri_face_addr;
    std::uint32_t part_id;
    std::uint32_t flags;
    std::array<std::array<float, 4>, 3> tf_rows;
};
static_assert(sizeof(InstanceDataGpu) == 80);

// C++ mirror of the Push struct in kernels/common.slang (64 bytes). The sun
// direction is three scalars there precisely so this one mirror is valid for
// both backends.
struct PushConstants {
    std::uint32_t row_offset;
    std::uint32_t num_emitters;
    std::uint32_t rays_this_chunk;
    std::uint32_t ray_index_offset;
    std::uint32_t batch_seed;
    std::uint32_t max_bounces;
    std::uint32_t flags;
    std::uint32_t num_faces;
    float energy_threshold;
    float fp_scale;
    float inv_fp_scale;
    float ray_tmin_scale;
    std::array<float, 3> sun_dir;
    float pad;
};
static_assert(sizeof(PushConstants) == 64);

// Instance transform as Metal wants it: MTLPackedFloat4x3 is COLUMN-major
// 4x3 (columns[3] is the translation), the transpose of the row-major
// tf_rows the shader reads. Exposed for the unit test that checks it against
// gmm::CoordinateTransformation::apply.
[[nodiscard]] MTLPackedFloat4x3 to_mtl_transform(const InstanceDataGpu& inst);

class SceneImpl {
  public:
    SceneImpl(DeviceImpl& device, std::vector<ScenePart> parts,
              MaterialTable materials);
    ~SceneImpl();
    SceneImpl(const SceneImpl&) = delete;
    SceneImpl& operator=(const SceneImpl&) = delete;
    SceneImpl(SceneImpl&&) = delete;
    SceneImpl& operator=(SceneImpl&&) = delete;

    void set_part_transform(std::uint32_t part_id,
                            const gmm::CoordinateTransformation& world_tf);
    void commit();

    void accumulate_vf(VfAccumImpl& acc, const TraceSettings& settings,
                       std::span<const std::uint32_t> emitters);
    void accumulate_exchange(ExchangeAccumImpl& acc,
                             const TraceSettings& settings,
                             std::span<const std::uint32_t> emitters);
    void accumulate_solar(const SolarState& sun, SolarAccumImpl& acc,
                          const TraceSettings& settings);
    void update_materials(const MaterialTable& materials);

    [[nodiscard]] std::uint32_t num_faces() const noexcept {
        return _num_faces;
    }
    [[nodiscard]] const MaterialTable& materials() const noexcept {
        return _materials;
    }
    [[nodiscard]] std::span<const double> face_areas() const noexcept {
        return _face_areas;
    }
    // Active, non-planet faces with geometry: the faces that draw sun samples
    // and the default emitters of the matrix kernels.
    [[nodiscard]] std::span<const std::uint32_t> default_emitters()
        const noexcept {
        return _default_emitters;
    }
    [[nodiscard]] DeviceImpl& device() const noexcept { return _device; }
    // GPU bytes resident after construction (geometry, acceleration
    // structures, tables) — the fixed cost of the scene for the memory
    // estimate; accumulators come on top.
    [[nodiscard]] std::uint64_t scene_bytes() const noexcept {
        return _scene_bytes;
    }

    // Buffer helpers shared with the accumulator implementations. Metal
    // needs no usage flags: a buffer is a buffer, and every one here is
    // shared storage.
    [[nodiscard]] GpuBuffer create_buffer(std::size_t size);
    void destroy_buffer(GpuBuffer& buffer) noexcept;
    // Creates a buffer and copies `bytes` of `data` into it. Empty inputs get
    // a minimal valid buffer (a zero-length MTLBuffer cannot be created).
    [[nodiscard]] GpuBuffer upload_to_new_buffer(const void* data,
                                                 std::size_t bytes);

  private:
    struct PartGpu {
        GpuBuffer vertices;
        GpuBuffer indices;
        GpuBuffer tri_face;
        id<MTLAccelerationStructure> blas = nil;
        std::uint32_t num_triangles = 0;
    };

    // Buffer indices of one kernel's parameters. Metal ignores the
    // [[vk::binding]] attributes and numbers buffers per kernel in
    // declaration order, so these come from the generated bindings headers
    // and differ between kernels; kernels without a parameter carry
    // `no_binding` for it.
    static constexpr std::uint32_t no_binding = 0xFFFFFFFFU;
    struct KernelBindings {
        std::uint32_t acc = no_binding;  // vf/exchange acc, solar direct_acc
        std::uint32_t total_acc = no_binding;
        std::uint32_t face_areas = no_binding;
        std::uint32_t pc = no_binding;
        std::uint32_t tlas = no_binding;
        std::uint32_t instance_data = no_binding;
        std::uint32_t materials = no_binding;
        std::uint32_t face_record = no_binding;
        std::uint32_t emitters = no_binding;
        std::uint32_t emit_tri_offset = no_binding;
        std::uint32_t emit_tri_part = no_binding;
        std::uint32_t emit_tri_prim = no_binding;
        std::uint32_t emit_cum_area = no_binding;
    };

    // Records into a one-shot command buffer, commits and waits.
    template <class Record>
    void submit_once(Record&& record);

    void upload_geometry(const std::vector<ScenePart>& parts);
    void upload_part(const ScenePart& part, std::size_t index,
                     Eigen::AlignedBox3d& world_box);
    void build_face_tables(const std::vector<ScenePart>& parts);
    void build_emission_tables(const std::vector<ScenePart>& parts);
    void build_blas(const std::vector<ScenePart>& parts);
    void build_tlas_first();
    void rebuild_tlas();
    void write_instance_buffers();
    void create_pipelines();
    [[nodiscard]] id<MTLComputePipelineState> build_compute_pipeline(
        std::span<const std::uint8_t> metallib, const char* what) const;
    // Everything about one kernel launch that is not the emitter list or the
    // trace settings, including the accumulator buffers (Metal binds them on
    // the encoder, so there is no persistent descriptor set to pre-write).
    struct KernelDispatch {
        id<MTLComputePipelineState> pipeline = nil;
        const KernelBindings* bindings = nullptr;
        id<MTLBuffer> acc = nil;        // vf/exchange acc, solar direct_acc
        id<MTLBuffer> total_acc = nil;  // solar only
        std::uint32_t row_offset = 0;
        std::uint32_t flags = 0;
        float fp_scale = 1.0F;
        std::array<float, 3> sun_dir = {0.0F, 0.0F, 0.0F};
    };
    // Binds the scene tables, the accumulators and the residency of every
    // resource the kernel reaches through a raw device address.
    void bind_resources(id<MTLComputeCommandEncoder> encoder,
                        const KernelDispatch& kernel,
                        const PushConstants& push) const;
    // Traces settings.rays_per_face rays for `emitters` (all within
    // [row_offset, row_offset + accumulator rows)), split into watchdog-safe
    // chunks.
    void dispatch_rows(const KernelDispatch& kernel,
                       std::span<const std::uint32_t> emitters,
                       const TraceSettings& settings);
    // Shared argument validation of the accumulate_* entry points; returns
    // the effective emitter list (the default list when `emitters` is
    // empty). An empty return means there is nothing to trace.
    [[nodiscard]] std::vector<std::uint32_t> resolve_emitters(
        std::span<const std::uint32_t> emitters,
        const TraceSettings& settings) const;

    DeviceImpl& _device;
    MaterialTable _materials;
    std::uint32_t _num_faces = 0;
    std::vector<double> _face_areas;
    std::vector<std::uint32_t> _default_emitters;
    float _ray_tmin_scale = 1e-4F;
    // Live sum of create_buffer allocations; snapshotted into _scene_bytes
    // at the end of construction (build scratch is already freed by then).
    std::uint64_t _allocated_bytes = 0;
    std::uint64_t _scene_bytes = 0;

    std::vector<PartGpu> _parts;
    std::vector<InstanceDataGpu> _instances_host;  // updated by transforms

    GpuBuffer _instance_ssbo;   // InstanceDataGpu[]
    GpuBuffer _tlas_instances;  // MTLAccelerationStructureInstanceDescriptor[]
    GpuBuffer _tlas_scratch;
    id<MTLAccelerationStructure> _tlas = nil;
    // The instanced structures the TLAS descriptor references, by index; must
    // outlive every build and stay in part order.
    NSArray<id<MTLAccelerationStructure>>* _blas_array = nil;

    GpuBuffer _materials_buf;
    GpuBuffer _face_record_buf;
    GpuBuffer _face_areas_buf;  // f32 pair areas (solar kernel weighting)
    GpuBuffer _emit_tri_offset_buf;
    GpuBuffer _emit_tri_part_buf;
    GpuBuffer _emit_tri_prim_buf;
    GpuBuffer _emit_cum_area_buf;
    GpuBuffer _emitters_buf;  // grown on demand per accumulate call
    // Placeholder behind an accumulator binding a kernel does not use, so
    // every buffer index the pipeline declares holds a valid buffer.
    GpuBuffer _dummy_buf;

    KernelBindings _vf_bindings;
    KernelBindings _exchange_bindings;
    KernelBindings _solar_bindings;
    id<MTLComputePipelineState> _vf_pipeline = nil;
    id<MTLComputePipelineState> _exchange_pipeline = nil;
    id<MTLComputePipelineState> _solar_pipeline = nil;
};

}  // namespace pycanha::radiative::detail
