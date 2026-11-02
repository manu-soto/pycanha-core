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
// `emitter` is node_weight[face] (unused in face rows). Returns the value
// added (in node rows the scaled one), and in `cell` where it went.
uint64_t deposit(uint row_base, uint face, float4 emitter, uint hit_face,
                 uint64_t amount, out uint cell) {
#if !defined(PYCANHA_METAL_BACKEND)
    if ((pc.flags & FLAG_NODE_ROWS) != 0u) {
        uint column = node_column[hit_face];
        // The faces in no node (the last real column) are never traced:
        // there is no other direction to combine with, only the share.
        float weight = column + 1u == pc.num_faces
                           ? 1.0
                           : blend_weight(face, emitter.yz, hit_face);
        uint64_t value = scaled(amount, emitter.x * weight);
        cell = row_base + column;
        if (value != uint64_t(0)) {
            fp_add_raw(acc, cell, value);
        }
        return value;
    }
#endif
    cell = row_base + hit_face;
    fp_add_raw(acc, cell, amount);
    return amount;
}

#if !defined(PYCANHA_METAL_BACKEND)
// Node rows with absorptance groups (NodeExchangeAccumulator's
// absorptance_groups): beside every cell, its derivative with respect to the
// band absorptance a of each of up to four groups of faces, the reflectance
// taking a change of a with its specular share kept (transmittance fixed), so
// that the choice between a specular and a diffuse reflection does not
// depend on a. The derivative of an expected deposit is the expectation of
// the deposit times the derivative of the log of its path's weight (the
// path's "score"): 1/a of the emitting face when it is in the group (its
// emission share is A eps), -1/rho for every reflection off a face of the
// group (rho = 1 - a - t: the energy kept, (1 - a), times the chance of a
// reflection, rho / (1 - a)), 0 for a transmission, and for the deposit on a
// face of the group 1/a of that face. A Russian-roulette survivor keeps its
// score: its chance of surviving is its energy over the threshold, whose
// derivative the score already holds. Only paths the trace takes are scored:
// on a face with a = 0 or with no reflectance the part a path never takes
// there (a deposit, a reflection) is left out of the derivative.
static const uint FLAG_DERIVATIVES = 512u;
// Bits 10-12 of the flags: the number of groups.
static const uint DERIVATIVE_COUNT_SHIFT = 10u;
// No group.
static const uint NO_GROUP = 0xFFFFFFFFu;
// Per face: its group, or NO_GROUP.
[[vk::binding(12, 0)]] StructuredBuffer<uint> absorptance_group;
// Per cell, the groups' derivatives side by side (cell * count + group),
// signed in wrapping u64, in units of 1024 of the deposits' fixed point: a
// score reaches 1/a, so the sums keep 10 bits of headroom over the cells.
[[vk::binding(13, 0)]] FpBuffer dacc;
static const float DERIVATIVE_UNIT = 1.0 / 1024.0;

uint derivative_count() {
    return (pc.flags & FLAG_DERIVATIVES) != 0u
               ? ((pc.flags >> DERIVATIVE_COUNT_SHIFT) & 7u)
               : 0u;
}

// A face's group, or NO_GROUP (also when no group is traced).
uint group_of(uint face) {
    return derivative_count() > 0u ? absorptance_group[face] : NO_GROUP;
}

float component(float4 v, uint i) {
    return i == 0u ? v.x : (i == 1u ? v.y : (i == 2u ? v.z : v.w));
}

float4 with_component(float4 v, uint i, float x) {
    return float4(i == 0u ? x : v.x, i == 1u ? x : v.y, i == 2u ? x : v.z,
                  i == 3u ? x : v.w);
}

// The deposit `value` (as the cell received it, signed in wrapping u64 for
// a Russian-roulette withdrawal) times each group's score, plus 1/a of the
// hit face for its own group (`group`, `inv_a`).
void derivative_add(uint cell, uint64_t value, float4 score, uint group,
                    float inv_a) {
    uint count = derivative_count();
    if (count == 0u || value == uint64_t(0)) {
        return;
    }
    float v = float(int64_t(value)) * DERIVATIVE_UNIT;
    for (uint k = 0u; k < count; ++k) {
        float c = component(score, k) + (k == group ? inv_a : 0.0);
        if (c != 0.0) {
            float x = v * c;
            int64_t d = int64_t(x + (x < 0.0 ? -0.5 : 0.5));
            if (d != int64_t(0)) {
                fp_add_raw(dacc, cell * count + k, uint64_t(d));
            }
        }
    }
}
#else
// Metal keeps face rows and traces no derivatives.
static const uint NO_GROUP = 0xFFFFFFFFu;
uint derivative_count() { return 0u; }
uint group_of(uint face) { return NO_GROUP; }
float component(float4 v, uint i) { return 0.0; }
float4 with_component(float4 v, uint i, float x) { return v; }
void derivative_add(uint cell, uint64_t value, float4 score, uint group,
                    float inv_a) {}
#endif

// A balance that ends in a virtual column, as the row holds it: as traced, or
// in node rows times the emitting face's share alone (a virtual column has no
// partner direction).
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
    uint space_col = pc.num_faces;
    uint inactive_col = pc.num_faces + 1u;
    uint lost_col = pc.num_faces + 2u;

    EmittedRay ray = emit_from_face(face, rng);
    float3 origin = ray.origin;
    float3 dir = ray.direction;
    float energy = 1.0;
    // fp_scale is a power of two, so to_fp(1.0) equals it exactly and the
    // host can compute rays * fp_scale without rounding.
    uint64_t fp_remaining = to_fp(1.0);
    // Absorptance groups: the path's score per group, which starts at 1/a of
    // the emitting face for its own group (see derivative_add).
    float4 score = float4(0.0, 0.0, 0.0, 0.0);
    uint cell = 0u;
    uint64_t value = uint64_t(0);
    uint emitter_group = group_of(face);
    if (emitter_group != NO_GROUP) {
        float ae;
        float se;
        float te;
        band_properties_of(face_record[face], ae, se, te);
        if (ae > 0.0) {
            score = with_component(score, emitter_group, 1.0 / ae);
        }
    }

    for (uint bounce = 0u; bounce < pc.max_bounces; ++bounce) {
        Hit hit = trace_ray(origin, dir);
        if (hit.miss) {
            value = tail(fp_remaining, emitter);
            fp_add_raw(acc, row_base + space_col, value);
            derivative_add(row_base + space_col, value, score, NO_GROUP, 0.0);
            return;
        }
        // One load per bounce: the flags and the material index live in the
        // same word, and both of the tests below plus the optical lookup read
        // it from here.
        uint record = face_record[hit.face];
        if ((record & FACE_FLAG_PLANET) != 0u) {
            // The planet absorbs every band; its radiance is CPU
            // post-processing over the VF matrix, not a kernel concern.
            value = deposit(row_base, face, emitter, hit.face, fp_remaining,
                            cell);
            derivative_add(cell, value, score, NO_GROUP, 0.0);
            return;
        }
        if ((record & FACE_FLAG_ACTIVE) == 0u) {
            // Non-radiative face: absorbed, but scored apart so the
            // consumer can decide what that energy means.
            value = tail(fp_remaining, emitter);
            fp_add_raw(acc, row_base + inactive_col, value);
            derivative_add(row_base + inactive_col, value, score, NO_GROUP,
                           0.0);
            return;
        }

        float a;
        float s;
        float t;
        if (!band_properties_of(record, a, s, t)) {
            // No material assigned: blackbody, absorbs the remainder.
            value = deposit(row_base, face, emitter, hit.face, fp_remaining,
                            cell);
            derivative_add(cell, value, score, NO_GROUP, 0.0);
            return;
        }
        // Diffuse reflectance is the clamped remainder; never renormalize
        // upward — the host has already validated a + s + t <= 1.
        float d = max(1.0 - a - s - t, 0.0);
        uint hit_group = group_of(hit.face);

        // Deposit BEFORE the RR test and before event selection (reordering
        // would bias the estimate).
        float energy_after = energy * (1.0 - a);
        uint64_t fp_after = to_fp(energy_after);
        value = deposit(row_base, face, emitter, hit.face,
                        fp_remaining - fp_after, cell);
        derivative_add(cell, value, score, hit_group, a > 0.0 ? 1.0 / a : 0.0);
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
                value = tail_signed(fp_remaining - fp_boost, emitter);
                fp_add_raw(acc, row_base + lost_col, value);
                // With the kill's deposit below, the lost column's derivative
                // stays zero-mean as the column itself does.
                derivative_add(row_base + lost_col, value, score, NO_GROUP,
                               0.0);
                fp_remaining = fp_boost;
                energy = pc.energy_threshold;
            } else {
                value = tail(fp_remaining, emitter);
                fp_add_raw(acc, row_base + lost_col, value);
                derivative_add(row_base + lost_col, value, score, NO_GROUP,
                               0.0);
                return;
            }
        }

        // Event selection among the non-absorbed part. Do not normalize by
        // (1 - a): s + t + d may fall below it only through float rounding.
        float u_event = rng_next(rng) * (s + t + d);
        bool reflected = true;
        if (u_event < s) {
            dir = normalize(reflect(dir, hit.normal));
            origin = offset_origin(hit.position, hit.normal);
        } else if (u_event < s + t) {
            // Thin-surface transmission: same direction, far side.
            origin = offset_origin(hit.position, -hit.normal);
            reflected = false;
        } else {
            dir = cosine_hemisphere(hit.normal, rng);
            origin = offset_origin(hit.position, hit.normal);
        }
        if (reflected && hit_group != NO_GROUP && s + d > 0.0) {
            score = with_component(
                score, hit_group,
                component(score, hit_group) - 1.0 / (s + d));
        }
    }
    // The max_bounces safety bound killed the ray.
    value = tail(fp_remaining, emitter);
    fp_add_raw(acc, row_base + lost_col, value);
    derivative_add(row_base + lost_col, value, score, NO_GROUP, 0.0);
}
