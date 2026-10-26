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

// What one path leaves in the row's virtual columns (space, inactive, lost),
// in wrapping u64. These cells are shared by every lane of the workgroup, so
// csMain adds them once per subgroup after the path (fp_add_raw_subgroup).
struct Tails {
    uint64_t space;
    uint64_t inactive;
    uint64_t lost;
};

// One path from `face`: deposits on real faces go to the accumulator
// directly, the balances that end in a virtual column go to `tails`.
void trace_path(uint face, uint ray_index, uint row_base, inout Tails tails) {
    Rng rng = rng_init(face, ray_index, pc.batch_seed);

    EmittedRay ray = emit_from_face(face, rng);
    float3 origin = ray.origin;
    float3 dir = ray.direction;
    float energy = 1.0;
    // fp_scale is a power of two, so to_fp(1.0) equals it exactly and the
    // host can compute rays * fp_scale without rounding.
    uint64_t fp_remaining = to_fp(1.0);

    for (uint bounce = 0u; bounce < pc.max_bounces; ++bounce) {
        Hit hit = trace_ray(origin, dir);
        if (hit.miss) {
            tails.space += fp_remaining;
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
            tails.inactive += fp_remaining;
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
                tails.lost += fp_remaining - fp_boost;
                fp_remaining = fp_boost;
                energy = pc.energy_threshold;
            } else {
                tails.lost += fp_remaining;
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
    tails.lost += fp_remaining;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csMain(uint3 gid: SV_DispatchThreadID) {
    uint e = gid.y;
    if (e >= pc.num_emitters) {
        return;  // the whole workgroup: control flow stays uniform
    }
    // The emitter face comes from the emitters buffer, NEVER from gid
    // arithmetic over faces; ray_index includes the chunk offset so results
    // are invariant under dispatch chunking.
    uint face = emitters[e];
    uint stride = pc.num_faces + 3u;
    // Block-relative accumulator row base.
    uint row_base = (face - pc.row_offset) * stride;

    // Lanes past the chunk's rays do not return early: they take part in the
    // subgroup sums below with zeros, so those run in uniform control flow.
    Tails tails = { uint64_t(0), uint64_t(0), uint64_t(0) };
    if (gid.x < pc.rays_this_chunk) {
        trace_path(face, pc.ray_index_offset + gid.x, row_base, tails);
    }
    fp_add_raw_subgroup(acc, row_base + pc.num_faces, tails.space);
    fp_add_raw_subgroup(acc, row_base + pc.num_faces + 1u, tails.inactive);
    fp_add_raw_subgroup(acc, row_base + pc.num_faces + 2u, tails.lost);
}
