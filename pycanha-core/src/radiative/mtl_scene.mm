#include "mtl_scene.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <dispatch/dispatch.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "face_record.hpp"
#include "mtl_accum.hpp"
#include "mtl_device.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/mesh/ops/compute_areas.hpp"
#include "pycanha-core/gmm/mesh/trimesh.hpp"
#include "pycanha-core/gmm/scene/coordinate_transformation.hpp"
#include "pycanha-core/radiative/kernels/exchange_bindings.h"
#include "pycanha-core/radiative/kernels/exchange_metallib.h"
#include "pycanha-core/radiative/kernels/solar_bindings.h"
#include "pycanha-core/radiative/kernels/solar_metallib.h"
#include "pycanha-core/radiative/kernels/vf_bindings.h"
#include "pycanha-core/radiative/kernels/vf_metallib.h"
#include "pycanha-core/radiative/materials.hpp"
#include "pycanha-core/radiative/scene.hpp"
#include "pycanha-core/radiative/scene_part.hpp"
#include "pycanha-core/radiative/settings.hpp"
#include "pycanha-core/utils/logger.hpp"

namespace pycanha::radiative::detail {

namespace {

// Keep single GPU submissions well under the GPU watchdog limit by bounding
// the rays per dispatch. TODO(radiative): replace the fixed bound with
// timestamp-based calibration targeting ~0.25 s chunks.
constexpr std::uint64_t max_rays_per_chunk_total = 1U << 22U;

constexpr std::uint32_t workgroup_size_x = 64;

// Host mirrors of the kernel flag bits in common.slang.
constexpr std::uint32_t flag_normal_emission = 1;
constexpr std::uint32_t flag_solar_band = 2;
constexpr std::uint32_t flag_qmc = 4;

[[nodiscard]] std::uint32_t sampler_flags(const TraceSettings& settings) {
    return settings.sampler == Sampler::Sobol ? flag_qmc : 0U;
}

// Tolerated float slack when validating that absorptivity + specular +
// transmission of a band does not exceed one.
constexpr float property_sum_slack = 1e-6F;

[[nodiscard]] std::string error_text(NSError* error) {
    if (error == nil) {
        return "unknown error";
    }
    return std::string([[error localizedDescription] UTF8String]);
}

void write_transform_rows(const gmm::CoordinateTransformation& tf,
                          std::array<std::array<float, 4>, 3>& rows) {
    const Eigen::Matrix3d& rotation = tf.rotation();
    const Vector3D& translation = tf.translation();
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t col = 0; col < 3; ++col) {
            rows.at(row).at(col) = static_cast<float>(
                rotation(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)));
        }
        rows.at(row).at(3) = static_cast<float>(translation(static_cast<Eigen::Index>(row)));
    }
}

// The kernels assume physically-consistent rows (the shaders never
// renormalize), so reject impossible tables up front for both bands.
void validate_material_properties(const MaterialTable& materials) {
    for (Eigen::Index row = 0; row < materials.properties.rows(); ++row) {
        for (int band_offset = 0; band_offset < 6; band_offset += 3) {
            const float a = materials.properties(row, band_offset);
            const float s = materials.properties(row, band_offset + 1);
            const float t = materials.properties(row, band_offset + 2);
            if (a < 0.0F || s < 0.0F || t < 0.0F || a + s + t > 1.0F + property_sum_slack) {
                throw std::invalid_argument(
                    "pycanha::radiative: material row " + std::to_string(row) + " has an invalid " +
                    (band_offset == 0 ? std::string("IR") : std::string("solar")) +
                    " property triplet (each in [0, 1], sum <= 1)");
            }
        }
    }
}

// Packs the 6-DOF property rows into the tightly-packed float layout the
// shaders index by flat offset.
[[nodiscard]] std::vector<float> pack_material_rows(const MaterialTable& materials) {
    std::vector<float> rows(static_cast<std::size_t>(materials.properties.rows()) * 6);
    for (Eigen::Index row = 0; row < materials.properties.rows(); ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * 6;
        for (int dof = 0; dof < 6; ++dof) {
            rows[base + static_cast<std::size_t>(dof)] = materials.properties(row, dof);
        }
    }
    return rows;
}

// Autoreleased; use inside an @autoreleasepool.
[[nodiscard]] MTLInstanceAccelerationStructureDescriptor* make_tlas_descriptor(
    id<MTLBuffer> instances, std::size_t count, NSArray<id<MTLAccelerationStructure>>* structures) {
    MTLInstanceAccelerationStructureDescriptor* descriptor =
        [MTLInstanceAccelerationStructureDescriptor descriptor];
    descriptor.instanceDescriptorBuffer = instances;
    descriptor.instanceDescriptorBufferOffset = 0;
    descriptor.instanceDescriptorStride = sizeof(MTLAccelerationStructureInstanceDescriptor);
    descriptor.instanceCount = count;
    descriptor.instancedAccelerationStructures = structures;
    descriptor.instanceDescriptorType = MTLAccelerationStructureInstanceDescriptorTypeDefault;
    return descriptor;
}

}  // namespace

void* checked_mapped(const GpuBuffer& buffer) {
    void* contents = buffer.buffer == nil ? nullptr : [buffer.buffer contents];
    if (contents == nullptr) {
        throw std::logic_error("pycanha::radiative: buffer is not host-mapped");
    }
    return contents;
}

MTLPackedFloat4x3 to_mtl_transform(const InstanceDataGpu& inst) {
    // MTLPackedFloat4x3 is column-major 4x3 — the transpose of the row-major
    // tf_rows the shader reads, with columns[3] carrying the translation.
    MTLPackedFloat4x3 out{};
    for (std::size_t col = 0; col < 4; ++col) {
        for (std::size_t row = 0; row < 3; ++row) {
            out.columns[col].elements[row] = inst.tf_rows.at(row).at(col);
        }
    }
    return out;
}

GpuBuffer SceneImpl::create_buffer(std::size_t size) {
    // Metal cannot create a zero-length buffer. Every allocation is at least
    // page-aligned, which covers the 16-byte placement the 64-bit fixed-point
    // cells need (a 64-bit atomic at a 4-byte-aligned offset misbehaves on
    // some hardware), so no explicit alignment request is necessary.
    const std::size_t bytes = std::max<std::size_t>(size, 16);
    GpuBuffer out;
    out.buffer = [_device.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (out.buffer == nil) {
        throw std::runtime_error("pycanha::radiative: Metal buffer allocation failed (" +
                                 std::to_string(bytes) + " bytes)");
    }
    out.size = bytes;
    _allocated_bytes += bytes;
    return out;
}

void SceneImpl::destroy_buffer(GpuBuffer& buffer) noexcept {
    if (buffer.buffer != nil) {
        _allocated_bytes -= buffer.size;
        buffer = GpuBuffer{};  // ARC releases the Metal object
    }
}

GpuBuffer SceneImpl::upload_to_new_buffer(const void* data, std::size_t bytes) {
    GpuBuffer buffer = create_buffer(bytes);
    if (bytes > 0) {
        std::memcpy(checked_mapped(buffer), data, bytes);
    }
    return buffer;
}

template <class Record>
void SceneImpl::submit_once(Record&& record) {
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [_device.queue commandBuffer];
        if (cmd == nil) {
            throw std::runtime_error("pycanha::radiative: Metal command buffer allocation failed");
        }
        std::forward<Record>(record)(cmd);
        [cmd commit];
        [cmd waitUntilCompleted];
        if ([cmd error] != nil) {
            throw std::runtime_error("pycanha::radiative: GPU command buffer failed: " +
                                     error_text([cmd error]));
        }
    }
}

SceneImpl::SceneImpl(DeviceImpl& device, std::vector<ScenePart> parts, MaterialTable materials)
    : _device(device), _materials(std::move(materials)) {
    if (parts.empty()) {
        throw std::invalid_argument("pycanha::radiative: a scene needs at least one part");
    }
    _num_faces = static_cast<std::uint32_t>(_materials.face_material.rows());
    if (_num_faces == 0 || (_num_faces % 2) != 0) {
        throw std::invalid_argument("pycanha::radiative: material table has no faces (build it "
                                    "from the same model as the parts)");
    }
    validate_material_properties(_materials);
    for (std::size_t i = 0; i < parts.size(); ++i) {
        if (parts[i].part_id != i) {
            throw std::invalid_argument(
                "pycanha::radiative: part_id must equal the part's position "
                "in the vector");
        }
        if (parts[i].mesh.nt() == 0) {
            throw std::invalid_argument("pycanha::radiative: part " + std::to_string(i) +
                                        " has an empty mesh");
        }
        if (static_cast<std::uint32_t>(parts[i].mesh.nf()) > _num_faces) {
            throw std::invalid_argument("pycanha::radiative: part " + std::to_string(i) +
                                        " has face ids beyond the material table");
        }
    }

    upload_geometry(parts);
    build_emission_tables(parts);
    build_blas(parts);
    write_instance_buffers();
    build_tlas_first();
    create_pipelines();
    _scene_bytes = _allocated_bytes;

    SPDLOG_LOGGER_INFO(pycanha::get_logger(), "radiative: scene built ({} parts, {} faces)",
                       parts.size(), _num_faces);
}

// Every submission in this file waits for completion, so nothing is in
// flight here and ARC can release the buffers, acceleration structures and
// pipelines in member order.
SceneImpl::~SceneImpl() = default;

void SceneImpl::upload_geometry(const std::vector<ScenePart>& parts) {
    _face_areas.assign(_num_faces, 0.0);
    _parts.resize(parts.size());
    _instances_host.resize(parts.size());

    // World bounding box of the non-celestial parts: the self-intersection
    // epsilon must follow the spacecraft scale, not the planet distance.
    Eigen::AlignedBox3d world_box;
    for (std::size_t p = 0; p < parts.size(); ++p) {
        upload_part(parts[p], p, world_box);
    }
    const double characteristic = world_box.isEmpty() ? 1.0 : world_box.diagonal().norm();
    _ray_tmin_scale = static_cast<float>(1e-4 * std::max(characteristic, 1e-6));

    build_face_tables(parts);
}

void SceneImpl::upload_part(const ScenePart& part, std::size_t index,
                            Eigen::AlignedBox3d& world_box) {
    const gmm::TriMeshF& mesh = part.mesh;
    const auto num_points = static_cast<Eigen::Index>(mesh.np());
    const auto num_triangles = static_cast<Eigen::Index>(mesh.nt());
    PartGpu& gpu = _parts[index];
    gpu.num_triangles = static_cast<std::uint32_t>(num_triangles);

    // Eigen matrices are column-major in memory: repack row by row into the
    // tightly-packed layouts the shaders index by pointer.
    std::vector<float> vertices(static_cast<std::size_t>(num_points) * 3);
    for (Eigen::Index v = 0; v < num_points; ++v) {
        const std::size_t base = static_cast<std::size_t>(v) * 3;
        vertices[base + 0] = mesh.vertices(v, 0);
        vertices[base + 1] = mesh.vertices(v, 1);
        vertices[base + 2] = mesh.vertices(v, 2);
    }
    std::vector<std::uint32_t> indices(static_cast<std::size_t>(num_triangles) * 3);
    std::vector<std::uint32_t> tri_face(static_cast<std::size_t>(num_triangles));
    for (Eigen::Index t = 0; t < num_triangles; ++t) {
        const std::size_t base = static_cast<std::size_t>(t) * 3;
        indices[base + 0] = mesh.triangles(t, 0);
        indices[base + 1] = mesh.triangles(t, 1);
        indices[base + 2] = mesh.triangles(t, 2);
        tri_face[static_cast<std::size_t>(t)] = mesh.face_ids(t);
    }

    gpu.vertices = upload_to_new_buffer(vertices.data(), vertices.size() * sizeof(float));
    gpu.indices = upload_to_new_buffer(indices.data(), indices.size() * sizeof(std::uint32_t));
    gpu.tri_face = upload_to_new_buffer(tri_face.data(), tri_face.size() * sizeof(std::uint32_t));

    // Rigid transforms preserve areas: part-local areas are world areas.
    const Eigen::VectorXd part_areas = gmm::mesh::ops::compute_face_areas(mesh);
    for (Eigen::Index face = 0; face < part_areas.rows(); ++face) {
        _face_areas[static_cast<std::size_t>(face)] += part_areas(face);
    }

    InstanceDataGpu& instance = _instances_host[index];
    instance.part_id = part.part_id;
    instance.flags = part.kind == PartKind::CelestialBody ? 1U : 0U;
    write_transform_rows(part.transform, instance.tf_rows);

    if (part.kind != PartKind::CelestialBody) {
        constexpr std::array<Eigen::AlignedBox3d::CornerType, 8> corners = {
            Eigen::AlignedBox3d::BottomLeftFloor, Eigen::AlignedBox3d::BottomRightFloor,
            Eigen::AlignedBox3d::TopLeftFloor,    Eigen::AlignedBox3d::TopRightFloor,
            Eigen::AlignedBox3d::BottomLeftCeil,  Eigen::AlignedBox3d::BottomRightCeil,
            Eigen::AlignedBox3d::TopLeftCeil,     Eigen::AlignedBox3d::TopRightCeil};
        const Eigen::AlignedBox3d local_box = gmm::mesh::ops::bounding_box(mesh);
        for (const auto corner : corners) {
            world_box.extend(part.transform.apply(local_box.corner(corner)));
        }
    }
}

void SceneImpl::build_face_tables(const std::vector<ScenePart>& parts) {
    // Global per-face tables. Missing material (-1) is tolerated (treated
    // as blackbody by the exchange kernels); inactive faces never emit.
    const std::vector<float> material_rows = pack_material_rows(_materials);
    const std::vector<std::uint32_t> face_records =
        detail::pack_face_records(_materials, parts, _num_faces);

    _materials_buf =
        upload_to_new_buffer(material_rows.data(), material_rows.size() * sizeof(float));
    _face_record_buf =
        upload_to_new_buffer(face_records.data(), face_records.size() * sizeof(std::uint32_t));
    std::vector<float> face_areas_f32(_num_faces);
    for (std::uint32_t face = 0; face < _num_faces; ++face) {
        face_areas_f32[face] = static_cast<float>(_face_areas[face]);
    }
    _face_areas_buf =
        upload_to_new_buffer(face_areas_f32.data(), face_areas_f32.size() * sizeof(float));

    // Default emitter list: active, non-planet faces with geometry.
    _default_emitters.clear();
    for (std::uint32_t face = 0; face < _num_faces; ++face) {
        if (detail::face_emits(face_records[face]) && _face_areas[face] > 0.0) {
            _default_emitters.push_back(face);
        }
    }
}

void SceneImpl::build_emission_tables(const std::vector<ScenePart>& parts) {
    struct EmitTriangle {
        std::uint32_t pair_base;
        std::uint32_t part;
        std::uint32_t prim;
        double area;
    };
    std::vector<EmitTriangle> triangles;
    for (std::size_t p = 0; p < parts.size(); ++p) {
        const gmm::TriMeshF& mesh = parts[p].mesh;
        const Eigen::VectorXd areas = gmm::mesh::ops::compute_areas(mesh);
        const auto num_triangles = static_cast<Eigen::Index>(mesh.nt());
        for (Eigen::Index t = 0; t < num_triangles; ++t) {
            triangles.push_back(EmitTriangle{.pair_base = mesh.face_ids(t),
                                             .part = static_cast<std::uint32_t>(p),
                                             .prim = static_cast<std::uint32_t>(t),
                                             .area = areas(t)});
        }
    }
    // Face faces partition across parts, so a stable sort by face keeps the
    // per-part triangle order within each face.
    std::ranges::stable_sort(triangles, {}, &EmitTriangle::pair_base);

    // Per-face offsets: even face = first triangle of the pair, odd face =
    // one past the last (both sides of a pair share the triangle list, so
    // the shader reads [pair_base] and [pair_base + 1]).
    std::vector<std::uint32_t> offsets(_num_faces, 0);
    std::vector<std::uint32_t> part_ids(triangles.size());
    std::vector<std::uint32_t> prim_ids(triangles.size());
    std::vector<float> cum_area(triangles.size());

    std::size_t index = 0;
    for (std::uint32_t pair = 0; pair < _num_faces; pair += 2) {
        offsets[pair] = static_cast<std::uint32_t>(index);
        const std::size_t begin = index;
        double total = 0.0;
        while (index < triangles.size() && triangles[index].pair_base == pair) {
            total += triangles[index].area;
            ++index;
        }
        double running = 0.0;
        for (std::size_t i = begin; i < index; ++i) {
            part_ids[i] = triangles[i].part;
            prim_ids[i] = triangles[i].prim;
            running += triangles[i].area;
            // Normalize to exactly 1.0 at the last triangle so the binary
            // search never falls off the end; zero-area pairs cannot be
            // emitters (filtered by area in the default list).
            cum_area[i] = total > 0.0 ? static_cast<float>(running / total) : 1.0F;
        }
        offsets[pair + 1U] = static_cast<std::uint32_t>(index);
    }

    _emit_tri_offset_buf =
        upload_to_new_buffer(offsets.data(), offsets.size() * sizeof(std::uint32_t));
    _emit_tri_part_buf =
        upload_to_new_buffer(part_ids.data(), part_ids.size() * sizeof(std::uint32_t));
    _emit_tri_prim_buf =
        upload_to_new_buffer(prim_ids.data(), prim_ids.size() * sizeof(std::uint32_t));
    _emit_cum_area_buf = upload_to_new_buffer(cum_area.data(), cum_area.size() * sizeof(float));
}

void SceneImpl::build_blas(const std::vector<ScenePart>& parts) {
    // alloc/init rather than the convenience constructor: this array outlives
    // the per-part pools below, and the library must not depend on the caller
    // having an autorelease pool in place.
    NSMutableArray<id<MTLAccelerationStructure>>* structures =
        [[NSMutableArray alloc] initWithCapacity:parts.size()];
    for (std::size_t p = 0; p < parts.size(); ++p) {
        PartGpu& gpu = _parts[p];
        @autoreleasepool {
            MTLAccelerationStructureTriangleGeometryDescriptor* geometry =
                [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
            geometry.vertexBuffer = gpu.vertices.buffer;
            geometry.vertexBufferOffset = 0;
            geometry.vertexFormat = MTLAttributeFormatFloat3;
            geometry.vertexStride = 3 * sizeof(float);
            geometry.indexBuffer = gpu.indices.buffer;
            geometry.indexBufferOffset = 0;
            geometry.indexType = MTLIndexTypeUInt32;
            geometry.triangleCount = gpu.num_triangles;
            // The kernels trace with force-opaque rays; no any-hit logic.
            geometry.opaque = YES;

            MTLPrimitiveAccelerationStructureDescriptor* descriptor =
                [MTLPrimitiveAccelerationStructureDescriptor descriptor];
            descriptor.geometryDescriptors = @[ geometry ];

            const MTLAccelerationStructureSizes sizes =
                [_device.device accelerationStructureSizesWithDescriptor:descriptor];
            const NSUInteger blas_size = sizes.accelerationStructureSize;
            gpu.blas = [_device.device newAccelerationStructureWithSize:blas_size];
            if (gpu.blas == nil) {
                throw std::runtime_error("pycanha::radiative: BLAS creation failed for part " +
                                         std::to_string(p));
            }
            _allocated_bytes += sizes.accelerationStructureSize;
            GpuBuffer scratch = create_buffer(sizes.buildScratchBufferSize);
            id<MTLAccelerationStructure> blas = gpu.blas;
            id<MTLBuffer> scratch_buffer = scratch.buffer;
            submit_once([blas, descriptor, scratch_buffer](id<MTLCommandBuffer> cmd) {
                id<MTLAccelerationStructureCommandEncoder> encoder =
                    [cmd accelerationStructureCommandEncoder];
                [encoder buildAccelerationStructure:blas
                                         descriptor:descriptor
                                      scratchBuffer:scratch_buffer
                                scratchBufferOffset:0];
                [encoder endEncoding];
            });
            destroy_buffer(scratch);
            [structures addObject:gpu.blas];
        }
    }
    // The TLAS instance descriptors reference these by index, so the array
    // must stay in part order for the lifetime of the scene.
    _blas_array = structures;
}

void SceneImpl::write_instance_buffers() {
    const std::size_t count = _instances_host.size();
    if (_instance_ssbo.buffer == nil) {
        _instance_ssbo = create_buffer(count * sizeof(InstanceDataGpu));
        _tlas_instances = create_buffer(count * sizeof(MTLAccelerationStructureInstanceDescriptor));
    }
    const std::span<MTLAccelerationStructureInstanceDescriptor> tlas_span(
        static_cast<MTLAccelerationStructureInstanceDescriptor*>(checked_mapped(_tlas_instances)),
        count);
    for (std::size_t p = 0; p < count; ++p) {
        InstanceDataGpu& instance = _instances_host[p];
        instance.vertices_addr = [_parts[p].vertices.buffer gpuAddress];
        instance.indices_addr = [_parts[p].indices.buffer gpuAddress];
        instance.tri_face_addr = [_parts[p].tri_face.buffer gpuAddress];

        MTLAccelerationStructureInstanceDescriptor mtl_instance{};
        mtl_instance.transformationMatrix = to_mtl_transform(instance);
        mtl_instance.options = MTLAccelerationStructureInstanceOptionOpaque;
        mtl_instance.mask = 0xFF;
        mtl_instance.intersectionFunctionTableOffset = 0;
        // The kernels read the committed instance id, which is the position
        // of the instance in this buffer — the same value the Vulkan backend
        // gets — so instance order must equal part order, as it does here.
        mtl_instance.accelerationStructureIndex = static_cast<std::uint32_t>(p);
        tlas_span[p] = mtl_instance;
    }
    std::memcpy(checked_mapped(_instance_ssbo), _instances_host.data(),
                count * sizeof(InstanceDataGpu));
}

void SceneImpl::build_tlas_first() {
    @autoreleasepool {
        MTLInstanceAccelerationStructureDescriptor* descriptor =
            make_tlas_descriptor(_tlas_instances.buffer, _parts.size(), _blas_array);
        const MTLAccelerationStructureSizes sizes =
            [_device.device accelerationStructureSizesWithDescriptor:descriptor];
        const NSUInteger tlas_size = sizes.accelerationStructureSize;
        _tlas = [_device.device newAccelerationStructureWithSize:tlas_size];
        if (_tlas == nil) {
            throw std::runtime_error("pycanha::radiative: TLAS creation failed");
        }
        _allocated_bytes += sizes.accelerationStructureSize;
        _tlas_scratch =
            create_buffer(std::max(sizes.buildScratchBufferSize, sizes.refitScratchBufferSize));
    }
    rebuild_tlas();
}

void SceneImpl::rebuild_tlas() {
    // Full rebuild, not refit: N instances is tiny and rebuilds keep
    // traversal quality under large rotations.
    @autoreleasepool {
        MTLInstanceAccelerationStructureDescriptor* descriptor =
            make_tlas_descriptor(_tlas_instances.buffer, _parts.size(), _blas_array);
        id<MTLAccelerationStructure> tlas = _tlas;
        id<MTLBuffer> scratch = _tlas_scratch.buffer;
        submit_once([tlas, descriptor, scratch](id<MTLCommandBuffer> cmd) {
            id<MTLAccelerationStructureCommandEncoder> encoder =
                [cmd accelerationStructureCommandEncoder];
            [encoder buildAccelerationStructure:tlas
                                     descriptor:descriptor
                                  scratchBuffer:scratch
                            scratchBufferOffset:0];
            [encoder endEncoding];
        });
    }
}

void SceneImpl::set_part_transform(std::uint32_t part_id,
                                   const gmm::CoordinateTransformation& world_tf) {
    if (part_id >= _instances_host.size()) {
        throw std::invalid_argument("pycanha::radiative: unknown part_id " +
                                    std::to_string(part_id));
    }
    write_transform_rows(world_tf, _instances_host[part_id].tf_rows);
}

void SceneImpl::commit() {
    write_instance_buffers();
    rebuild_tlas();
}

id<MTLComputePipelineState> SceneImpl::build_compute_pipeline(
    std::span<const std::uint8_t> metallib, const char* what) const {
    id<MTLComputePipelineState> pipeline = nil;
    @autoreleasepool {
        dispatch_data_t data = dispatch_data_create(metallib.data(), metallib.size(), nullptr,
                                                    DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        NSError* error = nil;
        id<MTLLibrary> library = [_device.device newLibraryWithData:data error:&error];
        if (library == nil) {
            throw std::runtime_error(std::string("pycanha::radiative: ") + what +
                                     " failed: " + error_text(error));
        }
        // MSL keeps the Slang entry-point name (only SPIR-V renames it).
        id<MTLFunction> function = [library newFunctionWithName:@"csMain"];
        if (function == nil) {
            throw std::runtime_error(std::string("pycanha::radiative: ") + what +
                                     " failed: no csMain entry point in the "
                                     "compiled Metal library");
        }
        pipeline = [_device.device newComputePipelineStateWithFunction:function error:&error];
        if (pipeline == nil) {
            throw std::runtime_error(std::string("pycanha::radiative: ") + what +
                                     " failed: " + error_text(error));
        }
    }
    return pipeline;
}

void SceneImpl::create_pipelines() {
    // Buffer indices come from the shader reflection at build time: Metal
    // ignores the [[vk::binding]] attributes and numbers each kernel's
    // buffers independently, so these tables are NOT interchangeable.
    _vf_bindings = KernelBindings{.acc = kernels::vf_binding_acc,
                                  .pc = kernels::vf_binding_pc,
                                  .tlas = kernels::vf_binding_tlas,
                                  .instance_data = kernels::vf_binding_instance_data,
                                  .materials = kernels::vf_binding_materials,
                                  .face_record = kernels::vf_binding_face_record,
                                  .emitters = kernels::vf_binding_emitters,
                                  .emit_tri_offset = kernels::vf_binding_emit_tri_offset,
                                  .emit_tri_part = kernels::vf_binding_emit_tri_part,
                                  .emit_tri_prim = kernels::vf_binding_emit_tri_prim,
                                  .emit_cum_area = kernels::vf_binding_emit_cum_area};
    _exchange_bindings =
        KernelBindings{.acc = kernels::exchange_binding_acc,
                       .pc = kernels::exchange_binding_pc,
                       .tlas = kernels::exchange_binding_tlas,
                       .instance_data = kernels::exchange_binding_instance_data,
                       .materials = kernels::exchange_binding_materials,
                       .face_record = kernels::exchange_binding_face_record,
                       .emitters = kernels::exchange_binding_emitters,
                       .emit_tri_offset = kernels::exchange_binding_emit_tri_offset,
                       .emit_tri_part = kernels::exchange_binding_emit_tri_part,
                       .emit_tri_prim = kernels::exchange_binding_emit_tri_prim,
                       .emit_cum_area = kernels::exchange_binding_emit_cum_area};
    _solar_bindings = KernelBindings{.acc = kernels::solar_binding_direct_acc,
                                     .total_acc = kernels::solar_binding_total_acc,
                                     .face_areas = kernels::solar_binding_face_areas,
                                     .pc = kernels::solar_binding_pc,
                                     .tlas = kernels::solar_binding_tlas,
                                     .instance_data = kernels::solar_binding_instance_data,
                                     .materials = kernels::solar_binding_materials,
                                     .face_record = kernels::solar_binding_face_record,
                                     .emitters = kernels::solar_binding_emitters,
                                     .emit_tri_offset = kernels::solar_binding_emit_tri_offset,
                                     .emit_tri_part = kernels::solar_binding_emit_tri_part,
                                     .emit_tri_prim = kernels::solar_binding_emit_tri_prim,
                                     .emit_cum_area = kernels::solar_binding_emit_cum_area};

    _vf_pipeline = build_compute_pipeline(
        std::span<const std::uint8_t>(kernels::vf_metallib, kernels::vf_metallib_size),
        "VF pipeline creation");
    _exchange_pipeline = build_compute_pipeline(
        std::span<const std::uint8_t>(kernels::exchange_metallib, kernels::exchange_metallib_size),
        "exchange pipeline creation");
    _solar_pipeline = build_compute_pipeline(
        std::span<const std::uint8_t>(kernels::solar_metallib, kernels::solar_metallib_size),
        "solar pipeline creation");

    // Keeps every accumulator index a kernel declares pointing at a real
    // buffer even when that accumulator is unused by the current dispatch.
    _dummy_buf = create_buffer(4);
}

void SceneImpl::bind_resources(id<MTLComputeCommandEncoder> encoder, const KernelDispatch& kernel,
                               const PushConstants& push) const {
    const KernelBindings& bindings = *kernel.bindings;
    const auto set_buffer = [encoder](std::uint32_t index, id<MTLBuffer> buffer) {
        if (index != no_binding) {
            [encoder setBuffer:buffer offset:0 atIndex:index];
        }
    };
    set_buffer(bindings.acc, kernel.acc != nil ? kernel.acc : _dummy_buf.buffer);
    set_buffer(bindings.total_acc, kernel.total_acc != nil ? kernel.total_acc : _dummy_buf.buffer);
    set_buffer(bindings.face_areas, _face_areas_buf.buffer);
    set_buffer(bindings.instance_data, _instance_ssbo.buffer);
    set_buffer(bindings.materials, _materials_buf.buffer);
    set_buffer(bindings.face_record, _face_record_buf.buffer);
    set_buffer(bindings.emitters, _emitters_buf.buffer);
    set_buffer(bindings.emit_tri_offset, _emit_tri_offset_buf.buffer);
    set_buffer(bindings.emit_tri_part, _emit_tri_part_buf.buffer);
    set_buffer(bindings.emit_tri_prim, _emit_tri_prim_buf.buffer);
    set_buffer(bindings.emit_cum_area, _emit_cum_area_buf.buffer);
    if (bindings.pc != no_binding) {
        // 64 bytes, once per chunk: well inside the small-argument fast path.
        [encoder setBytes:&push length:sizeof(PushConstants) atIndex:bindings.pc];
    }
    if (bindings.tlas != no_binding) {
        [encoder setAccelerationStructure:_tlas atBufferIndex:bindings.tlas];
    }

    // Residency has no Vulkan counterpart and no diagnostic: a resource the
    // kernel reaches WITHOUT being bound through the encoder is simply not
    // paged in, and the traversal reports misses instead of faulting. Two
    // classes need it here — the geometry buffers the shader dereferences
    // through the raw device addresses in InstanceData, and the primitive
    // structures the TLAS references (binding the TLAS does not cover them).
    for (const PartGpu& part : _parts) {
        [encoder useResource:part.vertices.buffer usage:MTLResourceUsageRead];
        [encoder useResource:part.indices.buffer usage:MTLResourceUsageRead];
        [encoder useResource:part.tri_face.buffer usage:MTLResourceUsageRead];
        [encoder useResource:part.blas usage:MTLResourceUsageRead];
    }
    [encoder useResource:_tlas usage:MTLResourceUsageRead];
}

std::vector<std::uint32_t> SceneImpl::resolve_emitters(std::span<const std::uint32_t> emitters,
                                                       const TraceSettings& settings) const {
    if (settings.rays_per_face == 0) {
        return {};
    }
    std::vector<std::uint32_t> list(emitters.begin(), emitters.end());
    if (list.empty()) {
        list = _default_emitters;
    }
    const std::uint32_t face_count = _num_faces;
    if (std::ranges::any_of(
            list, [face_count](const std::uint32_t face) { return face >= face_count; })) {
        throw std::invalid_argument("pycanha::radiative: emitter face out of range");
    }
    return list;
}

void SceneImpl::accumulate_vf(VfAccumImpl& acc, const TraceSettings& settings,
                              std::span<const std::uint32_t> emitters) {
    const std::vector<std::uint32_t> list = resolve_emitters(emitters, settings);
    if (list.empty()) {
        return;
    }
    // u32 counting cells: the CUMULATIVE per-face ray count must stay below
    // 2^31 or cells could overflow.
    if (acc.rays_per_face() + settings.rays_per_face > (1ULL << 31U)) {
        throw std::invalid_argument("pycanha::radiative: cumulative rays_per_face exceeds the u32 "
                                    "counting range; reset the accumulator or use fewer rays");
    }

    KernelDispatch kernel;
    kernel.pipeline = _vf_pipeline;
    kernel.bindings = &_vf_bindings;
    kernel.acc = acc.buffer();
    kernel.flags = (settings.normal_emission ? flag_normal_emission : 0U) |
                   sampler_flags(settings);

    if (acc.layout() == AccumLayout::Dense) {
        dispatch_rows(kernel, list, settings);
    } else {
        // Row blocks of tile_rows emitters; the scratch buffer is zeroed and
        // absorbed into the host accumulation per block. Chunking/tiling
        // never changes results: the RNG is keyed on (face, ray, seed).
        const std::uint32_t tile_rows = acc.tile_rows();
        for (std::uint32_t row_offset = 0; row_offset < _num_faces; row_offset += tile_rows) {
            std::vector<std::uint32_t> block;
            std::ranges::copy_if(list, std::back_inserter(block),
                                 [row_offset, tile_rows](const std::uint32_t face) {
                                     return face >= row_offset && face < row_offset + tile_rows;
                                 });
            if (block.empty()) {
                continue;
            }
            acc.clear_block_scratch();
            kernel.row_offset = row_offset;
            dispatch_rows(kernel, block, settings);
            acc.absorb_block(block, row_offset);
        }
    }

    acc.record_batch(list, settings.rays_per_face);
}

void SceneImpl::accumulate_exchange(ExchangeAccumImpl& acc, const TraceSettings& settings,
                                    std::span<const std::uint32_t> emitters) {
    const std::vector<std::uint32_t> list = resolve_emitters(emitters, settings);
    if (list.empty()) {
        return;
    }
    const float fp_scale = acc.prepare_batch(settings.rays_per_face);

    KernelDispatch kernel;
    kernel.pipeline = _exchange_pipeline;
    kernel.bindings = &_exchange_bindings;
    kernel.acc = acc.buffer();
    kernel.flags = (acc.band() == Band::Solar ? flag_solar_band : 0U) |
                   sampler_flags(settings);
    if (settings.normal_emission) {
        kernel.flags |= flag_normal_emission;
    }
    kernel.fp_scale = fp_scale;

    if (acc.layout() == AccumLayout::Dense) {
        dispatch_rows(kernel, list, settings);
    } else {
        const std::uint32_t tile_rows = acc.tile_rows();
        for (std::uint32_t row_offset = 0; row_offset < _num_faces; row_offset += tile_rows) {
            std::vector<std::uint32_t> block;
            std::ranges::copy_if(list, std::back_inserter(block),
                                 [row_offset, tile_rows](const std::uint32_t face) {
                                     return face >= row_offset && face < row_offset + tile_rows;
                                 });
            if (block.empty()) {
                continue;
            }
            acc.clear_block_scratch();
            kernel.row_offset = row_offset;
            dispatch_rows(kernel, block, settings);
            acc.absorb_block(block, row_offset);
        }
    }

    acc.record_batch(list, settings.rays_per_face);
}

void SceneImpl::accumulate_solar(const SolarState& sun, SolarAccumImpl& acc,
                                 const TraceSettings& settings) {
    // Every active non-planet face receives its rays_per_face sun samples;
    // there is no emitter-subset variant (the kernel is O(Nf) already).
    const std::vector<std::uint32_t> list = resolve_emitters({}, settings);
    if (list.empty()) {
        return;
    }
    const SolarAccumImpl::BatchSetup setup = acc.prepare_batch(sun, settings.rays_per_face);

    KernelDispatch kernel;
    kernel.pipeline = _solar_pipeline;
    kernel.bindings = &_solar_bindings;
    kernel.acc = acc.direct_buffer();
    kernel.total_acc = acc.total_buffer();
    kernel.flags = flag_solar_band | sampler_flags(settings);
    kernel.fp_scale = setup.fp_scale;
    kernel.sun_dir = setup.sun_dir;
    dispatch_rows(kernel, list, settings);

    acc.record_batch(settings.rays_per_face, list.size());
}

void SceneImpl::update_materials(const MaterialTable& materials) {
    if (materials.face_material.rows() != _materials.face_material.rows() ||
        (materials.face_material.array() != _materials.face_material.array()).any()) {
        throw std::invalid_argument("pycanha::radiative: update_materials must keep the same "
                                    "face_material mapping; changing it needs a scene rebuild");
    }
    if (materials.face_active.rows() != _materials.face_active.rows() ||
        (materials.face_active.array() != _materials.face_active.array()).any()) {
        throw std::invalid_argument("pycanha::radiative: update_materials must keep the same "
                                    "face activity; changing it needs a scene rebuild");
    }
    if (materials.properties.rows() != _materials.properties.rows()) {
        throw std::invalid_argument(
            "pycanha::radiative: update_materials must keep the same number "
            "of material rows (face_material indexes into them)");
    }
    validate_material_properties(materials);

    // Overwrite the mapped property rows in place — geometry, acceleration
    // structures and every other table stay untouched.
    const std::vector<float> material_rows = pack_material_rows(materials);
    if (!material_rows.empty()) {
        std::memcpy(checked_mapped(_materials_buf), material_rows.data(),
                    material_rows.size() * sizeof(float));
    }
    _materials.properties = materials.properties;
}

void SceneImpl::dispatch_rows(const KernelDispatch& kernel, std::span<const std::uint32_t> emitters,
                              const TraceSettings& settings) {
    // (Re)upload the emitter list, growing the buffer when needed.
    const std::size_t needed = emitters.size() * sizeof(std::uint32_t);
    if (_emitters_buf.buffer == nil || _emitters_buf.size < needed) {
        destroy_buffer(_emitters_buf);
        _emitters_buf = create_buffer(needed);
    }
    std::memcpy(checked_mapped(_emitters_buf), emitters.data(), needed);

    PushConstants push{.row_offset = kernel.row_offset,
                       .num_emitters = static_cast<std::uint32_t>(emitters.size()),
                       .rays_this_chunk = 0,   // set per chunk below
                       .ray_index_offset = 0,  // set per chunk below
                       .batch_seed = settings.seed,
                       .max_bounces = settings.max_bounces,
                       .flags = kernel.flags,
                       .num_faces = _num_faces,
                       .energy_threshold = settings.energy_threshold,
                       .fp_scale = kernel.fp_scale,
                       .inv_fp_scale = 1.0F / kernel.fp_scale,
                       .ray_tmin_scale = _ray_tmin_scale,
                       .sun_dir = kernel.sun_dir,
                       .pad = 0.0F};

    const std::uint64_t rays_per_chunk =
        std::max<std::uint64_t>(1, max_rays_per_chunk_total / emitters.size());
    std::uint64_t done = 0;
    while (done < settings.rays_per_face) {
        const std::uint64_t chunk = std::min(rays_per_chunk, settings.rays_per_face - done);
        push.rays_this_chunk = static_cast<std::uint32_t>(chunk);
        push.ray_index_offset = static_cast<std::uint32_t>(done);
        const auto groups_x =
            static_cast<std::uint64_t>((chunk + workgroup_size_x - 1) / workgroup_size_x);

        submit_once([this, &kernel, &push, groups_x](id<MTLCommandBuffer> cmd) {
            id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
            [encoder setComputePipelineState:kernel.pipeline];
            bind_resources(encoder, kernel, push);
            [encoder dispatchThreadgroups:MTLSizeMake(groups_x, push.num_emitters, 1)
                    threadsPerThreadgroup:MTLSizeMake(workgroup_size_x, 1, 1)];
            [encoder endEncoding];
            // Shared storage on unified memory plus the wait in submit_once
            // is all the host needs to see the deposits — no equivalent of
            // the Vulkan host-read barrier is required.
        });
        done += chunk;
    }
}

}  // namespace pycanha::radiative::detail
