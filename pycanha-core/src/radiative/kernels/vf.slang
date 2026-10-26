// VF kernel: geometric view factors, no materials. One trace per ray; score
// count[row][hit_face] += 1 on the first committed hit. Planet and inactive
// faces DO score at their own columns (the kernel is purely geometric;
// spacecraft->planet view factors feed albedo/planet-IR post-processing).
// A miss scores the virtual space column, so every row's counts sum to
// exactly rays_i and VF[i][j] = count[i][j] / rays_i closes to one.

import common;

// Plain u32 counting cells (counts need no fixed-point scale); the host
// guards rays_per_face < 2^31 so cells cannot overflow. Row stride is
// num_faces + 3: the virtual space/inactive/lost columns follow the
// real faces (vf only ever uses space).
[[vk::binding(9, 0)]] RWStructuredBuffer<uint> acc;

[shader("compute")]
[numthreads(64, 1, 1)]
void csMain(uint3 gid: SV_DispatchThreadID) {
    uint e = gid.y;
    if (e >= pc.num_emitters) {
        return;  // the whole workgroup: control flow stays uniform
    }
    // The emitter face comes from the emitters buffer, NEVER from gid
    // arithmetic over faces (emitter subsets have non-contiguous faces);
    // ray_index includes the chunk offset so results are invariant under
    // dispatch chunking.
    uint face = emitters[e];
    uint row = face - pc.row_offset;  // block-relative accumulator row
    uint stride = pc.num_faces + 3u;

    // Lanes past the chunk's rays do not return early, so the miss count
    // below runs in uniform control flow (see fp_add_raw_subgroup).
    bool missed = false;
    if (gid.x < pc.rays_this_chunk) {
        uint ray_index = pc.ray_index_offset + gid.x;
        Rng rng = rng_init(face, ray_index, pc.batch_seed);
        EmittedRay ray = emit_from_face(face, rng);
        Hit hit = trace_ray(ray.origin, ray.direction);
        if (hit.miss) {
            missed = true;
        } else {
            InterlockedAdd(acc[row * stride + hit.face], 1u);
        }
    }
#if defined(PYCANHA_METAL_BACKEND)
    if (missed) {
        InterlockedAdd(acc[row * stride + pc.num_faces], 1u);
    }
#else
    // Every lane of the subgroup traces from this row's emitter (gid.y), so
    // the misses all count into the same space cell: one atomic for the
    // subgroup.
    uint misses = WaveActiveCountBits(missed);
    if (WaveIsFirstLane() && misses != 0u) {
        InterlockedAdd(acc[row * stride + pc.num_faces], misses);
    }
#endif
}
