// Exchange kernel: multi-bounce MCRT radiative exchange factors for one
// band (IR or solar, selected by FLAG_SOLAR_BAND — the solar band uses the
// solar property triplet at EVERY bounce). Each ray starts with energy 1;
// absorption is deterministic (deposit E*a, continue with E*(1-a)), the
// specular/transmission/diffuse choice is stochastic. Normalization on the
// CPU: B[i][j] = deposited[i][j] / rays_i.
//
// The accumulator row stride is num_faces + 3: after the real face
// columns come the virtual space / inactive / lost columns, so every
// termination has an explicit, node-mappable destination.
//
// Exact energy accounting: every ray also carries its remaining energy as a
// fixed-point integer, deposits are the DIFFERENCE of consecutive
// fixed-point values and every termination path flushes the remaining
// balance into a column. Per row, the sum over ALL columns equals
// rays * fp_scale exactly in wrapping u64 arithmetic — the CI conservation
// assert. Russian-roulette boosts withdraw their fixed-point gain from the
// lost column (wrapping subtraction) and kills deposit the balance there,
// so the identity survives RR while the estimate stays unbiased (the two
// adjustments cancel in expectation, which is why the lost column is read
// back as a SIGNED total).

import common;

// FpBuffer is a u64 cell on SPIR-V and a lo/hi u32 pair on Metal (same bytes);
// deposits go through fp_add / fp_add_raw, never a raw InterlockedAdd here.
[[vk::binding(9, 0)]] FpBuffer acc;

[shader("compute")]
[numthreads(64, 1, 1)]
void csMain(uint3 gid: SV_DispatchThreadID) {
    uint ray_local = gid.x;
    uint e = gid.y;
    if (ray_local >= pc.rays_this_chunk || e >= pc.num_emitters) {
        return;
    }
    // The emitter face comes from the emitters buffer, NEVER from gid
    // arithmetic over faces; ray_index includes the chunk offset so results
    // are invariant under dispatch chunking.
    uint face = emitters[e];
    uint ray_index = pc.ray_index_offset + ray_local;
    Rng rng = rng_init(face, ray_index, pc.batch_seed);
    uint stride = pc.num_faces + 3u;
    // Block-relative accumulator row base.
    uint row_base = (face - pc.row_offset) * stride;
    uint space_col = pc.num_faces;
    uint inactive_col = pc.num_faces + 1u;
    uint lost_col = pc.num_faces + 2u;

    EmittedRay ray = emit_from_face(face, ray_index, rng);
    float3 origin = ray.origin;
    float3 dir = ray.direction;
    float energy = 1.0;
    // fp_scale is a power of two, so to_fp(1.0) equals it exactly and the
    // host can compute rays * fp_scale without rounding.
    uint64_t fp_remaining = to_fp(1.0);

    for (uint bounce = 0u; bounce < pc.max_bounces; ++bounce) {
        Hit hit = trace_ray(origin, dir);
        if (hit.miss) {
            fp_add_raw(acc, row_base + space_col, fp_remaining);
            return;
        }
        // One load per bounce: the flags and the material index live in the
        // same word, and both of the tests below plus the optical lookup read
        // it from here.
        uint record = face_record[hit.face];
        if ((record & FACE_FLAG_PLANET) != 0u) {
            // The planet absorbs every band; its radiance is CPU
            // post-processing over the VF matrix, not a kernel concern.
            fp_add_raw(acc, row_base + hit.face, fp_remaining);
            return;
        }
        if ((record & FACE_FLAG_ACTIVE) == 0u) {
            // Non-radiative face: absorbed, but scored apart so the
            // consumer can decide what that energy means.
            fp_add_raw(acc, row_base + inactive_col, fp_remaining);
            return;
        }

        float a;
        float s;
        float t;
        if (!band_properties_of(record, a, s, t)) {
            // No material assigned: blackbody, absorbs the remainder.
            fp_add_raw(acc, row_base + hit.face, fp_remaining);
            return;
        }
        // Diffuse reflectance is the clamped remainder; never renormalize
        // upward — the host has already validated a + s + t <= 1.
        float d = max(1.0 - a - s - t, 0.0);

        // Deposit BEFORE the RR test and before event selection (reordering
        // would bias the estimate).
        float energy_after = energy * (1.0 - a);
        uint64_t fp_after = to_fp(energy_after);
        fp_add_raw(acc, row_base + hit.face, fp_remaining - fp_after);
        fp_remaining = fp_after;
        energy = energy_after;
        if (fp_remaining == uint64_t(0)) {
            return;  // nothing representable left to distribute
        }

        // Russian roulette below the threshold — unbiased termination.
        if (energy < pc.energy_threshold) {
            float u_rr = rng_next(rng);
            if (u_rr * pc.energy_threshold < energy) {
                uint64_t fp_boost = to_fp(pc.energy_threshold);
                // Wrapping subtraction: the boost's fixed-point gain is
                // withdrawn from the lost column to keep the row balance
                // exact.
                fp_add_raw(acc, row_base + lost_col, fp_remaining - fp_boost);
                fp_remaining = fp_boost;
                energy = pc.energy_threshold;
            } else {
                fp_add_raw(acc, row_base + lost_col, fp_remaining);
                return;
            }
        }

        // Event selection among the non-absorbed part. Do not normalize by
        // (1 - a): s + t + d may fall below it only through float rounding.
        float u_event = rng_next(rng) * (s + t + d);
        if (u_event < s) {
            dir = normalize(reflect(dir, hit.normal));
            origin = offset_origin(hit.position, hit.normal);
        } else if (u_event < s + t) {
            // Thin-surface transmission: same direction, far side.
            origin = offset_origin(hit.position, -hit.normal);
        } else {
            dir = cosine_hemisphere(hit.normal, rng);
            origin = offset_origin(hit.position, hit.normal);
        }
    }
    // The max_bounces safety bound killed the ray.
    fp_add_raw(acc, row_base + lost_col, fp_remaining);
}
