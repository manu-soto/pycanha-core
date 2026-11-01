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

// Node rows (NodeExchangeAccumulator; SPIR-V only, Metal keeps face rows).
// The accumulator is node x node: a face's row and column are its node's
// (node_column; the faces in no node share column num_nodes, and
// pc.num_faces carries num_nodes + 1 so that the stride and the virtual
// columns follow). Every deposit is scaled here by what the face assembly
// would apply to it: the emitting face's share of its node's emission (its
// A eps over the largest A eps among the node's faces), and on a real column
// the weight that combines the face pair's two directions, which on a trace
// where every emitter sends the same rays is fixed by the two faces' areas.
// The sum of the scaled deposits over the face pairs of two nodes is then
// the aggregated face result, up to float rounding, without ever holding a
// face x face cell.
static const uint FLAG_NODE_ROWS = 128u;
// Node rows, TriangulationMode::None: a face pair keeps the estimate traced
// from its lower-numbered face, as the face assembly does.
static const uint FLAG_NODE_FORWARD_ONLY = 256u;

#if !defined(PYCANHA_METAL_BACKEND)
[[vk::binding(10, 0)]] StructuredBuffer<uint> node_column;
// x: the face's emission share; y + z: the u its blend weight reads (the
// face's area when it emits, 0 when it does not) as a float and the rest of
// the double below it; w unused.
[[vk::binding(11, 0)]] StructuredBuffer<float4> node_weight;

// The fixed-point amount times a share in [0, 1], in integers: the share as a
// 30-bit fraction q (the float share times 2^30, exact for shares of 2^-7 and
// up), the amount's two 32-bit halves multiplied by it, the low product
// rounded to nearest. A share of 1 returns the amount itself, bit for bit.
// Through floats it did not, on an RTX 4070: float(amount) * share made a
// directly escaping ray's S = 2^47 into S + 2^32 (the space column 2^-15
// high), and the same in 32-bit halves still changed some small balances (the
// lost column's sum came out 12 times its value); every conversion left here
// is of a value below 2^31.
uint64_t scaled(uint64_t amount, float share) {
    uint64_t q = uint64_t(uint(share * 1073741824.0 + 0.5));
    uint64_t high = (amount >> 32) * q;            // below 2^56 for amount < 2^58
    uint64_t low = (amount & 0xFFFFFFFFull) * q;   // below 2^62
    return (high << 2) + ((low + 536870912ull) >> 30);
}

// The same for a balance that may be negative in wrapping u64 (the lost
// column's Russian-roulette withdrawals): its magnitude scaled, the sign kept.
uint64_t scaled_signed(uint64_t amount, float share) {
    if ((amount >> 63) == uint64_t(0)) {
        return scaled(amount, share);
    }
    return uint64_t(0) - scaled(uint64_t(0) - amount, share);
}

// Weight of the deposits `face` makes on `hit_face` in the combined pair
// (pair_walk's WeightTable::forward_weight, in float): its own estimate
// alone when the partner never emits or for a self-hit. u and v come as a
// float and its remainder, so that the difference of two nearly equal areas
// keeps the double's digits: |ratio|^n has an infinite slope at 0, and with
// the areas rounded to floats two equal-looking faces (a cube's cells) came
// out at ratio ~3e-8 and weights 1e-3 off.
float blend_weight(uint face, float2 u, uint hit_face) {
    if (hit_face == face) {
        return 1.0;
    }
    if ((pc.flags & FLAG_NODE_FORWARD_ONLY) != 0u) {
        return face < hit_face ? 1.0 : 0.0;
    }
    float2 v = node_weight[hit_face].yz;
    if (!(v.x > 0.0)) {
        return 1.0;
    }
    float ratio = ((v.x - u.x) + (v.y - u.y)) / (u.x + v.x);
    float magnitude = pow(abs(ratio), pc.blend_exponent);
    return 0.5 * (1.0 + (ratio < 0.0 ? -magnitude : magnitude));
}
#endif

// One deposit of `face`'s path onto `hit_face`: into the face's row at the
// hit face's column, or in node rows into the nodes' cell, scaled.
// `emitter` is node_weight[face] (unused in face rows).
void deposit(uint row_base, uint face, float4 emitter, uint hit_face,
             uint64_t amount) {
#if !defined(PYCANHA_METAL_BACKEND)
    if ((pc.flags & FLAG_NODE_ROWS) != 0u) {
        uint column = node_column[hit_face];
        // The faces in no node (the last real column) are never traced:
        // there is no other direction to combine with, only the share.
        float weight = column + 1u == pc.num_faces
                           ? 1.0
                           : blend_weight(face, emitter.yz, hit_face);
        uint64_t value = scaled(amount, emitter.x * weight);
        if (value != uint64_t(0)) {
            fp_add_raw(acc, row_base + column, value);
        }
        return;
    }
#endif
    fp_add_raw(acc, row_base + hit_face, amount);
}

// A balance that ends in a virtual column, as the row holds it: as traced, or
// in node rows times the emitting face's share alone (a virtual column has no
// partner direction). Scaled where the path produces it, so that csMain
// reaches its subgroup sums exactly as in face rows.
uint64_t tail(uint64_t amount, float4 emitter) {
#if !defined(PYCANHA_METAL_BACKEND)
    if ((pc.flags & FLAG_NODE_ROWS) != 0u) {
        return scaled(amount, emitter.x);
    }
#endif
    return amount;
}

// The same for a wrapping balance that may be negative (a Russian-roulette
// withdrawal from the lost column).
uint64_t tail_signed(uint64_t amount, float4 emitter) {
#if !defined(PYCANHA_METAL_BACKEND)
    if ((pc.flags & FLAG_NODE_ROWS) != 0u) {
        return scaled_signed(amount, emitter.x);
    }
#endif
    return amount;
}

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
void trace_path(uint face, uint ray_index, uint row_base, float4 emitter,
                inout Tails tails) {
    Rng rng = rng_init(face, ray_index, pc.batch_seed);

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
            tails.space += tail(fp_remaining, emitter);
            return;
        }
        // One load per bounce: the flags and the material index live in the
        // same word, and both of the tests below plus the optical lookup read
        // it from here.
        uint record = face_record[hit.face];
        if ((record & FACE_FLAG_PLANET) != 0u) {
            // The planet absorbs every band; its radiance is CPU
            // post-processing over the VF matrix, not a kernel concern.
            deposit(row_base, face, emitter, hit.face, fp_remaining);
            return;
        }
        if ((record & FACE_FLAG_ACTIVE) == 0u) {
            // Non-radiative face: absorbed, but scored apart so the
            // consumer can decide what that energy means.
            tails.inactive += tail(fp_remaining, emitter);
            return;
        }

        float a;
        float s;
        float t;
        if (!band_properties_of(record, a, s, t)) {
            // No material assigned: blackbody, absorbs the remainder.
            deposit(row_base, face, emitter, hit.face, fp_remaining);
            return;
        }
        // Diffuse reflectance is the clamped remainder; never renormalize
        // upward — the host has already validated a + s + t <= 1.
        float d = max(1.0 - a - s - t, 0.0);

        // Deposit BEFORE the RR test and before event selection (reordering
        // would bias the estimate).
        float energy_after = energy * (1.0 - a);
        uint64_t fp_after = to_fp(energy_after);
        deposit(row_base, face, emitter, hit.face, fp_remaining - fp_after);
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
                tails.lost += tail_signed(fp_remaining - fp_boost, emitter);
                fp_remaining = fp_boost;
                energy = pc.energy_threshold;
            } else {
                tails.lost += tail(fp_remaining, emitter);
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
    tails.lost += tail(fp_remaining, emitter);
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
    uint row = face;
    float4 emitter = float4(1.0, 0.0, 0.0, 0.0);
#if !defined(PYCANHA_METAL_BACKEND)
    if ((pc.flags & FLAG_NODE_ROWS) != 0u) {
        row = node_column[face];
        emitter = node_weight[face];
    }
#endif
    // Block-relative accumulator row base.
    uint row_base = (row - pc.row_offset) * stride;

    // Lanes past the chunk's rays do not return early: they take part in the
    // subgroup sums below with zeros, so those run in uniform control flow.
    Tails tails = { uint64_t(0), uint64_t(0), uint64_t(0) };
    if (gid.x < pc.rays_this_chunk) {
        trace_path(face, pc.ray_index_offset + gid.x, row_base, emitter, tails);
    }
    fp_add_raw_subgroup(acc, row_base + pc.num_faces, tails.space);
    fp_add_raw_subgroup(acc, row_base + pc.num_faces + 1u, tails.inactive);
    fp_add_raw_subgroup(acc, row_base + pc.num_faces + 2u, tails.lost);
}
