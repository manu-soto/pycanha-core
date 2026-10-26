// Shared module of every radiative kernel: RNG, emission sampling, hit
// resolution, fixed-point deposits, shared scene bindings and push constants.
//
// Determinism contract: the per-ray random stream depends only on
// (face, ray_index, batch_seed), and the number of RNG draws per step
// is FROZEN (see the draw counts noted at each sampling function) — never
// insert a draw, only append at the end of a step, or every golden-value
// test and the CPU reference implementation silently break.

module common;

// ---------------------------------------------------------------------------
// Push constants — one struct for all kernels. Dispatches are split into
// ray-index chunks (GPU watchdog limits + progress reporting), which is why
// the chunk fields exist; chunking never changes results because the RNG is
// keyed on the global ray index. The C++ mirror struct on the host must
// match this byte-for-byte (scalar layout, 64 B).
// ---------------------------------------------------------------------------
public struct Push {
    public uint row_offset;        // first emitter row of this block
    public uint num_emitters;      // emitters in this dispatch
    public uint rays_this_chunk;   // rays per face in this chunk
    public uint ray_index_offset;  // first global ray index of this chunk
    public uint batch_seed;        // settings.seed + batch_index
    public uint max_bounces;
    public uint flags;             // bit0: normal-emission mode
    public uint num_faces;    // total faces (accumulator row stride)
    public float energy_threshold;
    public float fp_scale;         // fixed-point scale S of the deposits
    public float inv_fp_scale;     // 1/S
    public float ray_tmin_scale;   // self-intersection epsilon scale
    // Solar kernel only (unit, sun -> scene). Three scalars rather than a
    // float3 because a float3 is 12 B under Vulkan scalar layout but 16 B
    // (16-aligned) in Metal Shading Language, which would make this struct 64 B
    // for SPIR-V and 80 B for Metal and silently break the byte-for-byte
    // contract with the C++ mirror. Scalars are laid out identically by both.
    public float sun_dir_x;
    public float sun_dir_y;
    public float sun_dir_z;
    public float _pad;
};

[[vk::push_constant]]
public Push pc;

public float3 sun_direction() {
    return float3(pc.sun_dir_x, pc.sun_dir_y, pc.sun_dir_z);
}

public static const uint FLAG_NORMAL_EMISSION = 1u;
// Selects the solar property triplet (alpha_sol/spec_sol/tau_sol) instead of
// the IR one. A push-constant flag keeps one exchange pipeline for both
// bands; the branch is dispatch-uniform, so it costs nothing on the GPU.
public static const uint FLAG_SOLAR_BAND = 2u;
// Emission from the scrambled Sobol' sequence instead of the PCG stream
// (TraceSettings::sampler == Sampler::Sobol). Dispatch-uniform.
public static const uint FLAG_QMC = 4u;

// ---------------------------------------------------------------------------
// Shared scene bindings (set 0, bindings 0-9). Kernel-specific accumulator
// buffers start at binding 10 in the kernel files.
// ---------------------------------------------------------------------------

// One TLAS instance per rigid part (80 bytes, scalar layout). The host
// writes the SAME transform here and into the TLAS instance on every
// commit(): the TLAS positions geometry for ray traversal, but emission
// sampling needs the transform explicitly — and it must be read per
// dispatch, never baked into the vertex buffers, because parts move
// between snapshots.
public struct InstanceData {
    public uint64_t vertices_addr;  // devaddr float3[] (12 B stride, scalar)
    public uint64_t indices_addr;   // devaddr uint3[]
    public uint64_t tri_face_addr;  // devaddr uint[] (even side-1 face / tri)
    public uint part_id;
    public uint flags;              // bit0: is_planet_part
    // Part -> world as three explicit row vectors (xyz = rotation row,
    // w = translation component). An array of float4 has an unambiguous
    // memory layout, unlike matrix types whose row/column-major storage
    // depends on compiler settings.
    public float4 tf_rows[3];
};

// world = R * p + t, with R rows / t components taken from tf_rows.
public float3 transform_point(InstanceData inst, float3 p) {
    return float3(dot(inst.tf_rows[0].xyz, p) + inst.tf_rows[0].w,
                  dot(inst.tf_rows[1].xyz, p) + inst.tf_rows[1].w,
                  dot(inst.tf_rows[2].xyz, p) + inst.tf_rows[2].w);
}

// Rotation only (directions/normals; rigid transforms preserve lengths).
public float3 transform_direction(InstanceData inst, float3 d) {
    return float3(dot(inst.tf_rows[0].xyz, d), dot(inst.tf_rows[1].xyz, d),
                  dot(inst.tf_rows[2].xyz, d));
}

[[vk::binding(0, 0)]] public RaytracingAccelerationStructure tlas;
[[vk::binding(1, 0)]] public StructuredBuffer<InstanceData> instance_data;
[[vk::binding(2, 0)]] public StructuredBuffer<float> materials;  // 6*M floats
// One word per face: the activity/planet flags in the low bits and the
// material index above them. Flags and material used to be two buffers
// indexed by the same face id, so a bounce touched two cache lines for what
// is conceptually one per-face record.
[[vk::binding(3, 0)]] public StructuredBuffer<uint> face_record;
[[vk::binding(4, 0)]] public StructuredBuffer<uint> emitters;  // this block
[[vk::binding(5, 0)]] public StructuredBuffer<uint> emit_tri_offset;
[[vk::binding(6, 0)]] public StructuredBuffer<uint> emit_tri_part;
[[vk::binding(7, 0)]] public StructuredBuffer<uint> emit_tri_prim;
[[vk::binding(8, 0)]] public StructuredBuffer<float> emit_cum_area;

public static const uint FACE_FLAG_ACTIVE = 1u;
public static const uint FACE_FLAG_PLANET = 2u;
// The material index sits above the flags, biased by one so that zero means
// "no material assigned" and material row 0 stays usable.
public static const uint FACE_MATERIAL_SHIFT = 2u;
public static const uint INSTANCE_FLAG_PLANET = 1u;

// ---------------------------------------------------------------------------
// RNG — counter-based and grouping-invariant (PCG-XSH-RR style hash +
// stream). Streams depend ONLY on (face, ray_index, batch_seed) —
// never on thread id, dispatch shape, tile size or emitter order — so any
// chunking/tiling of the dispatch reproduces bit-identical results.
// ---------------------------------------------------------------------------
public struct Rng {
    public uint state;
};

public uint pcg_hash(uint x) {
    x = x * 747796405u + 2891336453u;
    uint w = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    return (w >> 22u) ^ w;
}

public Rng rng_init(uint face, uint ray_index, uint batch_seed) {
    Rng r;
    r.state = pcg_hash(face ^ pcg_hash(ray_index ^ pcg_hash(batch_seed)));
    return r;
}

// Uniform in [0, 1) — 24-bit mantissa, never 1.0.
public float rng_next(inout Rng r) {
    r.state = r.state * 747796405u + 2891336453u;
    uint w = ((r.state >> ((r.state >> 28u) + 4u)) ^ r.state) * 277803737u;
    w = (w >> 22u) ^ w;
    return float(w >> 8) * (1.0 / 16777216.0);
}

// ---------------------------------------------------------------------------
// Randomised quasi-Monte Carlo emission (FLAG_QMC). A ray's emission uses four
// numbers: the face point (2) and the direction (2). They come from a 4-D
// Sobol' sequence indexed by the ray index, each dimension Owen-scrambled with
// a seed derived from (face, dimension, batch_seed), so the sequence is
// unbiased, independent between faces and between batches, and — being keyed
// on the global ray index like the PCG stream — invariant under chunking.
// Direction numbers: Joe & Kuo (2008), new-joe-kuo-6.21201, dimensions 1-4
// (checked point by point against scipy.stats.qmc.Sobol). Scrambling: nested
// uniform (Owen) scrambling by the hash-based method of Burley (2020,
// "Practical Hash-based Owen Scrambling", JCGT 9(4)) with the improved hash of
// Vegdahl (2021, "Building a Better LK Hash").
// ---------------------------------------------------------------------------
#if defined(PYCANHA_METAL_BACKEND) || defined(PYCANHA_QMC)
static const uint SOBOL_DIRECTIONS[128] = {
    // dimension 0 (van der Corput)
    0x80000000u, 0x40000000u, 0x20000000u, 0x10000000u, 0x08000000u, 0x04000000u, 0x02000000u, 0x01000000u,
    0x00800000u, 0x00400000u, 0x00200000u, 0x00100000u, 0x00080000u, 0x00040000u, 0x00020000u, 0x00010000u,
    0x00008000u, 0x00004000u, 0x00002000u, 0x00001000u, 0x00000800u, 0x00000400u, 0x00000200u, 0x00000100u,
    0x00000080u, 0x00000040u, 0x00000020u, 0x00000010u, 0x00000008u, 0x00000004u, 0x00000002u, 0x00000001u,
    // dimension 1
    0x80000000u, 0xc0000000u, 0xa0000000u, 0xf0000000u, 0x88000000u, 0xcc000000u, 0xaa000000u, 0xff000000u,
    0x80800000u, 0xc0c00000u, 0xa0a00000u, 0xf0f00000u, 0x88880000u, 0xcccc0000u, 0xaaaa0000u, 0xffff0000u,
    0x80008000u, 0xc000c000u, 0xa000a000u, 0xf000f000u, 0x88008800u, 0xcc00cc00u, 0xaa00aa00u, 0xff00ff00u,
    0x80808080u, 0xc0c0c0c0u, 0xa0a0a0a0u, 0xf0f0f0f0u, 0x88888888u, 0xccccccccu, 0xaaaaaaaau, 0xffffffffu,
    // dimension 2
    0x80000000u, 0xc0000000u, 0x60000000u, 0x90000000u, 0xe8000000u, 0x5c000000u, 0x8e000000u, 0xc5000000u,
    0x68800000u, 0x9cc00000u, 0xee600000u, 0x55900000u, 0x80680000u, 0xc09c0000u, 0x60ee0000u, 0x90550000u,
    0xe8808000u, 0x5cc0c000u, 0x8e606000u, 0xc5909000u, 0x6868e800u, 0x9c9c5c00u, 0xeeee8e00u, 0x5555c500u,
    0x8000e880u, 0xc0005cc0u, 0x60008e60u, 0x9000c590u, 0xe8006868u, 0x5c009c9cu, 0x8e00eeeeu, 0xc5005555u,
    // dimension 3
    0x80000000u, 0xc0000000u, 0x20000000u, 0x50000000u, 0xf8000000u, 0x74000000u, 0xa2000000u, 0x93000000u,
    0xd8800000u, 0x25400000u, 0x59e00000u, 0xe6d00000u, 0x78080000u, 0xb40c0000u, 0x82020000u, 0xc3050000u,
    0x208f8000u, 0x51474000u, 0xfbea2000u, 0x75d93000u, 0xa0858800u, 0x914e5400u, 0xdbe79e00u, 0x25db6d00u,
    0x58800080u, 0xe54000c0u, 0x79e00020u, 0xb6d00050u, 0x800800f8u, 0xc00c0074u, 0x200200a2u, 0x50050093u
};

uint sobol_u32(uint index, uint dim) {
    uint x = 0u;
    uint base = dim * 32u;
    for (uint bit = 0u; index != 0u; index >>= 1u, ++bit) {
        if ((index & 1u) != 0u) {
            x ^= SOBOL_DIRECTIONS[base + bit];
        }
    }
    return x;
}

uint lk_hash(uint x, uint seed) {
    x ^= x * 0x3d20adeau;
    x += seed;
    x *= (seed >> 16u) | 1u;
    x ^= x * 0x05526c56u;
    x ^= x * 0x53a22864u;
    return x;
}

uint owen_scramble(uint x, uint seed) {
    return reversebits(lk_hash(reversebits(x), seed));
}

// Scrambled Sobol' coordinate `dim` (0-3) of ray `ray_index` of `face`, in
// [0, 1) with a 24-bit mantissa like rng_next.
public float qmc_sample(uint face, uint ray_index, uint dim) {
    uint seed = pcg_hash(face ^ pcg_hash(dim + 0x9e3779b9u ^ pcg_hash(pc.batch_seed)));
    uint x = owen_scramble(sobol_u32(ray_index, dim), seed);
    return float(x >> 8) * (1.0 / 16777216.0);
}
#else
// The default Vulkan kernels never call it (qmc_mode() is false there); the
// stub keeps the table and the scrambling out of them whatever the optimizer.
public float qmc_sample(uint face, uint ray_index, uint dim) {
    return 0.0;
}
#endif

// On Vulkan the sampler is chosen at compile time: every kernel is built twice,
// the second time with PYCANHA_QMC defined (<kernel>_qmc_spv), and the host
// picks the build. The default kernels then hold none of the QMC code or its
// table: as a dispatch-uniform push-constant flag they traced ~5 % slower than
// main on an RTX 4070, and so did a specialization constant. Metal keeps the
// flag.
public bool qmc_mode() {
#if defined(PYCANHA_METAL_BACKEND)
    return (pc.flags & FLAG_QMC) != 0u;
#elif defined(PYCANHA_QMC)
    return true;
#else
    return false;
#endif
}

// ---------------------------------------------------------------------------
// Fixed-point deposit: float energy is converted to an integer once and
// accumulated atomically. Integer adds are associative (float atomics are
// not), which is what makes results independent of thread scheduling and
// accumulator layout. No zero-check: a zero add is harmless and branchless is
// faster.
//
// A 64-bit accumulator cell is a single u64 on SPIR-V, but Metal Shading
// Language has NO 64-bit buffer atomics of any kind (not add, min/max,
// load/store or compare-exchange, on any -std), so there the cell is a pair of
// u32 counters holding its low and high halves. Two adjacent u32 at an 8-byte
// stride ARE the little-endian byte image of the u64, so buffer sizes and host
// readback are identical on both backends.
// ---------------------------------------------------------------------------
#if defined(PYCANHA_METAL_BACKEND)
public typealias FpBuffer = RWStructuredBuffer<uint>;
#else
public typealias FpBuffer = RWStructuredBuffer<uint64_t>;
#endif

// Fixed-point representation of a remaining-energy float. Monotone
// non-decreasing in `energy` (f32 multiply, +0.5 and the u64 truncation all
// preserve order), which is what lets deposits be computed as differences of
// consecutive values without ever underflowing.
public uint64_t to_fp(float energy) {
    return uint64_t(energy * pc.fp_scale + 0.5);
}

// Raw integer add for pre-converted fixed-point amounts (deposit
// differences and terminal balance flushes). u64 arithmetic wraps, which the
// exchange kernel exploits for the Russian-roulette boost withdrawal.
//
// The split path is exact, not an approximation: each 32-bit add is itself
// atomic, so whatever the thread order the low halves total sum(V_lo) mod 2^32
// and the number of low-half wraps is floor(sum(V_lo) / 2^32); folding that
// carry into the high half reconstructs sum(V) mod 2^64 exactly. Results stay
// bit-identical to the single-atomic path, including the wrapping subtraction.
public void fp_add_raw(FpBuffer buf, uint idx, uint64_t value) {
#if defined(PYCANHA_METAL_BACKEND)
    uint lo = uint(value & 0xFFFFFFFFull);
    uint hi = uint(value >> 32);
    uint old_lo;
    InterlockedAdd(buf[2u * idx], lo, old_lo);
    uint carry = (old_lo + lo) < old_lo ? 1u : 0u;  // unsigned wrap == carry
    InterlockedAdd(buf[2u * idx + 1u], hi + carry);
#else
    InterlockedAdd(buf[idx], value);
#endif
}

// Single definition of the float deposit, so the two backends' integer paths
// cannot drift apart.
public void fp_add(FpBuffer buf, uint idx, float value) {
    fp_add_raw(buf, idx, to_fp(value));
}

// Band-selected optical triplet of a face: absorptivity, specular
// reflectance, transmittance. The band comes from pc.flags (FLAG_SOLAR_BAND).
// Faces without a material (-1) are blackbodies: a=1, s=t=0, returns false.
public bool band_properties_of(uint record, out float a, out float s,
                               out float t) {
    uint biased_material = record >> FACE_MATERIAL_SHIFT;
    if (biased_material == 0u) {
        a = 1.0;
        s = 0.0;
        t = 0.0;
        return false;
    }
    uint base = (biased_material - 1u) * 6u +
                (((pc.flags & FLAG_SOLAR_BAND) != 0u) ? 3u : 0u);
    a = materials[base];
    s = materials[base + 1u];
    t = materials[base + 2u];
    return true;
}

// Convenience for callers that have not already loaded the record. A hit path
// that needs the flags too should read the record ONCE and use the overload
// above instead of paying for a second load.
public bool band_properties(uint face, out float a, out float s, out float t) {
    return band_properties_of(face_record[face], a, s, t);
}

// ---------------------------------------------------------------------------
// Geometry helpers (published reference implementations — do not re-derive).
// ---------------------------------------------------------------------------

// Branchless orthonormal basis (Duff et al. 2017). Sign subtlety at
// n.z ~ -1 is handled by `s`; do not substitute a cross(n, up) version.
public void onb(float3 n, out float3 t, out float3 b) {
    float s = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float c = n.x * n.y * a;
    t = float3(1.0 + s * n.x * n.x * a, s * c, -s * n.x);
    b = float3(c, s + n.y * n.y * a, -n.y);
}

// Cosine-weighted hemisphere about n (2 draws).
public float3 cosine_hemisphere(float3 n, inout Rng rng) {
    float u1 = rng_next(rng);
    float u2 = rng_next(rng);
    return cosine_hemisphere_from(n, u1, u2);
}

// The same map from two given numbers in [0, 1) (the QMC emission's).
public float3 cosine_hemisphere_from(float3 n, float u1, float u2) {
    float phi = 6.28318530718 * u1;
    float r = sqrt(u2);
    float3 t;
    float3 b;
    onb(n, t, b);
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) +
                     n * sqrt(max(1.0 - u2, 0.0)));
}

// Self-intersection avoidance: rays start at the hit point pushed out along
// the geometric normal by an epsilon that scales with the distance from the
// origin, so it works both at spacecraft scale (meters) and at
// planet-distance scale (1e7 m) where float32 positions are coarse.
// TODO(radiative): if validation scenes show light leaks on thin geometry,
// upgrade to an adaptive integer-offset scheme.
public float3 offset_origin(float3 hit_p, float3 n_geom) {
    float eps = pc.ray_tmin_scale * max(1.0, length(hit_p));
    return hit_p + n_geom * eps;
}

// ---------------------------------------------------------------------------
// Emission sampling. FROZEN draw order: u0 triangle pick, u1/u2 barycentric
// point, u3/u4 direction (the direction draws are skipped in normal-emission
// mode).
// ---------------------------------------------------------------------------
public struct EmittedRay {
    public float3 origin;
    public float3 direction;
};

// A uniformly-sampled surface point with the side-resolved normal. The
// position is NOT offset yet — callers push it out with offset_origin along
// whatever side they trace from.
public struct SurfaceSample {
    public float3 position;
    public float3 normal;
};

// Draws u0 (triangle pick) + u1/u2 (barycentric point) — the first three
// draws of the frozen emission order. In QMC mode (FLAG_QMC) no draw is taken:
// Sobol' dimension 0 picks the triangle and, rescaled to its slice of the
// cumulative area, gives the first barycentric number; dimension 1 the second.
// That keeps the face point a 2-D low-discrepancy sample over the whole face.
public SurfaceSample sample_face_point(uint face, uint ray_index, inout Rng rng) {
    uint pair_base = face & ~1u;  // even side-1 face of the pair
    bool qmc = qmc_mode();

    // 1. Area-weighted triangle pick: binary search u0 in the face's
    //    cumulative-area table (normalized to 1.0 at the last triangle).
    float u0;
    if (qmc) {
        u0 = qmc_sample(face, ray_index, 0u);
    } else {
        u0 = rng_next(rng);
    }
    uint first = emit_tri_offset[pair_base];
    uint lo = first;
    uint hi = emit_tri_offset[pair_base + 1u];  // one past the last triangle
    while (lo + 1u < hi) {
        uint mid = (lo + hi) / 2u;
        if (u0 < emit_cum_area[mid - 1u]) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    uint part = emit_tri_part[lo];
    uint prim = emit_tri_prim[lo];

    InstanceData inst = instance_data[part];
    uint3 *indices = (uint3 *)inst.indices_addr;
    float3 *vertices = (float3 *)inst.vertices_addr;
    uint3 tri = indices[prim];
    float3 v0 = vertices[tri.x];
    float3 v1 = vertices[tri.y];
    float3 v2 = vertices[tri.z];

    // 2. Uniform barycentric point (sqrt trick, 2 draws).
    float u1;
    float u2;
    if (qmc) {
        float c_lo = lo == first ? 0.0 : emit_cum_area[lo - 1u];
        float c_hi = min(emit_cum_area[lo], 1.0);
        u1 = clamp((u0 - c_lo) / max(c_hi - c_lo, 1e-30), 0.0, 0.99999994);
        u2 = qmc_sample(face, ray_index, 1u);
    } else {
        u1 = rng_next(rng);
        u2 = rng_next(rng);
    }
    float s = sqrt(u1);
    float b0 = 1.0 - s;
    float b1 = s * (1.0 - u2);
    float b2 = s * u2;
    float3 p_local = b0 * v0 + b1 * v1 + b2 * v2;
    float3 p_world = transform_point(inst, p_local);

    // 3. Geometric normal, world-rotated. Winding: n points off side 1
    //    (even face); odd faces (side 2) flip.
    float3 n_local = normalize(cross(v1 - v0, v2 - v0));
    float3 n = normalize(transform_direction(inst, n_local));
    if ((face & 1u) != 0u) {
        n = -n;
    }

    SurfaceSample sample;
    sample.position = p_world;
    sample.normal = n;
    return sample;
}

public EmittedRay emit_from_face(uint face, uint ray_index, inout Rng rng) {
    SurfaceSample sample = sample_face_point(face, ray_index, rng);

    // 4. Direction: cosine hemisphere (2 draws, or Sobol' dimensions 2-3 in
    //    QMC mode) or the normal itself.
    float3 dir;
    if ((pc.flags & FLAG_NORMAL_EMISSION) != 0u) {
        dir = sample.normal;
    } else if (qmc_mode()) {
        dir = cosine_hemisphere_from(sample.normal, qmc_sample(face, ray_index, 2u),
                                     qmc_sample(face, ray_index, 3u));
    } else {
        dir = cosine_hemisphere(sample.normal, rng);
    }

    EmittedRay ray;
    ray.origin = offset_origin(sample.position, sample.normal);
    ray.direction = dir;
    return ray;
}

// ---------------------------------------------------------------------------
// Hit resolution — the ONLY place face-side parity is decided (front face
// <=> side 1 <=> even face). Emission uses the same winding convention; a
// sign mismatch between the two produces view-factor matrices that are
// almost right, which is why parity lives in exactly one function.
// ---------------------------------------------------------------------------
public struct Hit {
    public bool miss;
    public uint face;      // global face (side-resolved)
    public float3 normal;  // geometric, flipped so dot(n, ray_dir) < 0
    public float3 position;
};

public Hit trace_ray(float3 origin, float3 direction) {
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.0;
    ray.TMax = 3.4e38;

    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(tlas, RAY_FLAG_FORCE_OPAQUE, 0xFF, ray);
    q.Proceed();

    Hit hit;
    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT) {
        hit.miss = true;
        hit.face = 0u;
        hit.normal = float3(0.0, 0.0, 1.0);
        hit.position = origin;
        return hit;
    }

    uint inst_index = q.CommittedInstanceIndex();
    uint prim = q.CommittedPrimitiveIndex();
    InstanceData inst = instance_data[inst_index];
    uint *tri_face = (uint *)inst.tri_face_addr;
    uint base = tri_face[prim];  // even side-1 face

    uint3 *indices = (uint3 *)inst.indices_addr;
    float3 *vertices = (float3 *)inst.vertices_addr;
    uint3 tri = indices[prim];
    float3 n_local = normalize(
        cross(vertices[tri.y] - vertices[tri.x],
              vertices[tri.z] - vertices[tri.x]));
    float3 n = normalize(transform_direction(inst, n_local));

    // Front-facing is exactly "the winding normal opposes the ray" for an
    // orientation-preserving instance transform, which every part has (rigid,
    // determinant +1, and transform_direction applies rotation only). They
    // could differ only at dot(n, direction) == 0 — a ray exactly in the
    // triangle plane, which cannot produce a committed triangle hit.
    // The two forms are kept apart because slangc's Metal back end has no
    // working CommittedTriangleFrontFace() (it emits an accessor MSL does not
    // declare), while on SPIR-V deriving `front` from `n` forces the vertex
    // fetch before the face is known and the compiler then emits the triangle
    // fetch twice (measured: +23 executable instructions in the vf kernel).
#if defined(PYCANHA_METAL_BACKEND)
    bool front = dot(n, direction) < 0.0;
#else
    bool front = q.CommittedTriangleFrontFace();
#endif

    hit.miss = false;
    hit.face = front ? base : base + 1u;
    hit.position = origin + q.CommittedRayT() * direction;
    // Flip toward the incoming ray (same convention as the parity above:
    // front face <=> side 1 <=> winding normal against the ray).
    hit.normal = dot(n, direction) < 0.0 ? n : -n;
    return hit;
}
