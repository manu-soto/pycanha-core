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
    uint ray_local = gid.x;
    uint e = gid.y;
    if (ray_local >= pc.rays_this_chunk || e >= pc.num_emitters) {
        return;
    }
    // The emitter face comes from the emitters buffer, NEVER from gid
    // arithmetic over faces (emitter subsets have non-contiguous faces);
    // ray_index includes the chunk offset so results are invariant under
    // dispatch chunking.
    uint face = emitters[e];
    uint ray_index = pc.ray_index_offset + ray_local;
    Rng rng = rng_init(face, ray_index, pc.batch_seed);
    uint row = face - pc.row_offset;  // block-relative accumulator row
    uint stride = pc.num_faces + 3u;

    EmittedRay ray = emit_from_face(face, ray_index, rng);
    Hit hit = trace_ray(ray.origin, ray.direction);
    uint col = hit.miss ? pc.num_faces : hit.face;
    InterlockedAdd(acc[row * stride + col], 1u);
}
