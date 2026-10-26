// Solar kernel: direct + reflected absorbed solar energy per face.
// The source is a parallel sun along sun_direction() (unit, sun -> scene); the
// host sets FLAG_SOLAR_BAND so every material lookup uses the solar triplet.
// Each thread samples one point on its face, casts a shadow ray toward the
// sun (any hit — including a planet part — means occluded/eclipsed), scores
// the absorbed direct energy, then follows the reflected/transmitted path
// with the shared bounce physics, scoring absorption into total[] at every
// hit.
//
// E0 = cos(theta) * face_area: every face traces the same rays_per_face
// regardless of its size, so a sample of face i represents area_i / rays
// square meters of collector. A ray's deposits are therefore its WATT
// content (per unit irradiance): a cell receives energy from MANY faces'
// rays and reads back directly as absorbed power,
// watts_j = irradiance * cell_j / (fp_scale * rays).
//
// Outputs are per-face vectors (direct absorbed, total absorbed incl.
// reflections); there is no row/matrix concept and no exact-conservation
// bucket accounting — energy that escapes or dies is simply not absorbed.

import common;

// FpBuffer is a u64 cell on SPIR-V and a lo/hi u32 pair on Metal (same bytes);
// deposits go through fp_add, never a raw InterlockedAdd here.
[[vk::binding(9, 0)]] FpBuffer direct_acc;
[[vk::binding(10, 0)]] FpBuffer total_acc;
[[vk::binding(11, 0)]] StructuredBuffer<float> face_areas;

// Post-interaction survival: Russian roulette below the threshold (1 draw
// when triggered). Returns false when the ray dies.
bool roulette(inout float energy, inout Rng rng) {
    if (energy <= 0.0) {
        return false;
    }
    if (energy < pc.energy_threshold) {
        float u_rr = rng_next(rng);
        if (u_rr * pc.energy_threshold >= energy) {
            return false;
        }
        energy = pc.energy_threshold;
    }
    return true;
}

// Event selection among the non-absorbed part (1 draw + 2 when diffuse):
// updates the outgoing direction and offsets the origin off the chosen side.
void scatter(inout float3 origin, inout float3 dir, float3 position, float3 n,
             float s, float t, float d, inout Rng rng) {
    float u_event = rng_next(rng) * (s + t + d);
    if (u_event < s) {
        dir = normalize(reflect(dir, n));
        origin = offset_origin(position, n);
    } else if (u_event < s + t) {
        origin = offset_origin(position, -n);  // thin-surface transmission
    } else {
        dir = cosine_hemisphere(n, rng);
        origin = offset_origin(position, n);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void csMain(uint3 gid: SV_DispatchThreadID) {
    uint ray_local = gid.x;
    uint e = gid.y;
    if (ray_local >= pc.rays_this_chunk || e >= pc.num_emitters) {
        return;
    }
    uint face = emitters[e];
    uint ray_index = pc.ray_index_offset + ray_local;
    Rng rng = rng_init(face, ray_index, pc.batch_seed);
    // The push constant stores the direction as three scalars (identical
    // layout on both backends); rebuild the vector once per thread.
    float3 sun_dir = sun_direction();

    SurfaceSample sample = sample_face_point(face, ray_index, rng);
    float cos_theta = dot(sample.normal, -sun_dir);
    if (cos_theta <= 0.0) {
        return;  // this side faces away from the sun
    }
    float3 origin = offset_origin(sample.position, sample.normal);
    Hit shadow = trace_ray(origin, -sun_dir);
    if (!shadow.miss) {
        return;  // occluded by other geometry (or eclipsed by the planet)
    }

    float a;
    float s;
    float t;
    band_properties(face, a, s, t);
    float d = max(1.0 - a - s - t, 0.0);

    float e0 = cos_theta * face_areas[face];
    fp_add(direct_acc, face, e0 * a);
    fp_add(total_acc, face, e0 * a);

    // Follow the non-absorbed part: the incoming light travels along the sun
    // direction when it interacts with the surface.
    float energy = e0 * (1.0 - a);
    if (!roulette(energy, rng)) {
        return;
    }
    float3 dir = sun_dir;
    scatter(origin, dir, sample.position, sample.normal, s, t, d, rng);

    for (uint bounce = 0u; bounce < pc.max_bounces; ++bounce) {
        Hit hit = trace_ray(origin, dir);
        if (hit.miss) {
            return;  // escaped to space
        }
        // One load per bounce: both tests and the optical lookup share it.
        uint record = face_record[hit.face];
        if ((record & FACE_FLAG_PLANET) != 0u) {
            return;  // the planet absorbs everything, nothing to score
        }
        if ((record & FACE_FLAG_ACTIVE) == 0u) {
            return;
        }
        band_properties_of(record, a, s, t);
        d = max(1.0 - a - s - t, 0.0);

        fp_add(total_acc, hit.face, energy * a);
        energy *= (1.0 - a);
        if (!roulette(energy, rng)) {
            return;
        }
        scatter(origin, dir, hit.position, hit.normal, s, t, d, rng);
    }
}
