#pragma once

#include <cstdint>

namespace pycanha::radiative {

// How each ray's emission point and direction are drawn (vf, exchange and
// solar kernels; the bounces after the first interaction stay pseudo-random).
enum class Sampler : std::uint8_t {
    // Pseudo-random (PCG) — the default, the stream every golden value pins.
    Random,
    // Randomised quasi-Monte Carlo: the face point and the direction come from
    // a 4-D Sobol' sequence (Joe & Kuo direction numbers) indexed by the ray
    // index, Owen-scrambled per face, dimension and seed (hash-based nested
    // uniform scrambling). Unbiased; still deterministic and invariant under
    // chunking, and batches with different seeds are independent
    // randomisations. Converges faster than Random on exchange integrals.
    Sobol,
};

// Where the solar kernel's rays start (accumulate_solar only).
enum class SolarSource : std::uint8_t {
    // rays_per_face sample points on EVERY emitting face, each with a shadow
    // ray toward the sun — the default, the stream every golden value pins.
    // A face gets the same rays whatever its size or its share of the sun.
    FaceSamples,
    // rays_per_face rays IN TOTAL start on a plane square to the sun that
    // covers the scene's non-celestial parts and travel along the sun
    // direction; each carries the irradiance of its share of the plane's
    // area (its projected area, so the cosine is implicit) to the first face
    // it meets and on through the reflections. Rays go only where the sun
    // reaches: a face receives rays in proportion to its sunlit projected
    // area, and the shadowed or back-facing faces cost nothing.
    SunPlane,
};

// Monte-Carlo trace control: a fixed ray count plus a seed make every run
// deterministic and reproducible. Accuracy-targeted convergence is
// intentionally NOT a C++ concept — the accumulate_* calls are
// batch-additive, so callers re-invoke until the reported statistics
// satisfy them.
struct TraceSettings {
    // Rays per emitting face, per accumulate() call.
    std::uint64_t rays_per_face = 10'000;
    // Deterministic seed; batches use seed + batch_index.
    std::uint32_t seed = 0;
    // MCRT ray-kill energy cutoff (exchange kernels).
    float energy_threshold = 1e-4F;
    // Hard safety bound on the bounce loop. Russian roulette below
    // energy_threshold already ends every path whose surfaces absorb
    // anything, so the bound must not be reached before it does: the energy a
    // ray still carries at the bound is dropped (a closed enclosure of
    // emittance 0.05 keeps 0.95^64 = 3.7 % of it after 64 bounces). Only an
    // enclosure that absorbs nothing needs the bound.
    std::uint32_t max_bounces = 10'000;
    // Emit along the face normal instead of the cosine-weighted hemisphere
    // (the legacy `_Nodes` debugging mode; vf/exchange kernels only).
    bool normal_emission = false;
    // Emission sampling (see Sampler).
    Sampler sampler = Sampler::Random;
    // Solar ray origin (see SolarSource); the other kernels ignore it.
    SolarSource solar_source = SolarSource::FaceSamples;
    // Rays leave a surface by a fixed number of float ULPs of the point
    // (Wächter & Binder 2019; ~15 um at 1 m) instead of an epsilon of 1e-4
    // of the scene's diagonal (0.15-0.2 mm on a 1.5 m spacecraft), through
    // which rays leaked past sub-millimetre features. Off by default: the
    // golden values pin the scaled epsilon.
    bool robust_offset = false;
};

// Accumulator buffer layout. Kernels are identical in both layouts and cells
// accumulate as integers (integer adds are associative, unlike float adds),
// so Dense and Tiled results are bit-identical for the same seed.
enum class AccumLayout : std::uint8_t {
    Dense,  // one Nf x Nf buffer — the small-model fast path
    Tiled,  // tile_rows x Nf blocks streamed + CPU-sparsified — the big path
};

// How the two independent Monte-Carlo estimates of a face pair are combined
// into the single reciprocity-consistent value that gets stored.
enum class TriangulationMode : std::uint8_t {
    // Keep the forward estimate as traced. Nothing is combined, so the
    // stored triangle is still two independent noisy numbers per pair — the
    // mode to reach for when bisecting a suspected assembly bug.
    None,
    // Weight each direction by how densely it was sampled:
    //   X_i = 1/2 (1 + sign(Y) |Y|^n),  Y = (v - u) / (v + u)
    // with u = A_i/N_i and v = A_j/N_j, the two estimator variances up to a
    // factor that cancels. n = 1 makes this exactly inverse-variance
    // (minimum-variance) weighting; the default n = 0.4 is the more
    // aggressive, empirically tuned rule the reference implementation uses.
    //
    // The SAME proxy applies to the exchange kernel even though that kernel
    // deposits energy rather than unit hits: its per-cell variance carries a
    // further factor eps_i eps_j, which appears in both directions and
    // therefore cancels out of the weight along with everything else.
    RayDensity,
    // Impose reciprocity AND row closure together instead of one after the
    // other. RayDensity blends each pair independently, which leaves rows no
    // longer summing to the energy they emitted; renormalising afterwards
    // then partly undoes the reciprocity. This mode solves for the matrix
    // closest to the raw estimate in the inverse-variance metric among those
    // that close every row exactly, which is a sparse symmetric system of
    // one equation per face.
    //
    // It ignores `exponent`: the combination it starts from has to be the
    // unconstrained minimum-variance one (n = 1) or the result is not the
    // least-squares solution of anything.
    ConstrainedLeastSquares,
};

struct TriangulationConfig {
    TriangulationMode mode = TriangulationMode::RayDensity;
    // Must be > 0. 1.0 reduces the weight to inverse-variance weighting,
    // which measured marginally better than 0.4 on the exchange path (where
    // the ray-density proxy is exact rather than empirical) — 0.4 is kept as
    // the single default for both paths so that a default-constructed config
    // never means two different things.
    double exponent = 0.4;
    // Also keep the raw, untriangulated matrix (both triangles) alongside
    // the combined upper triangle. Off at every model size: it doubles the
    // result memory and exists only for someone debugging a model.
    bool keep_full_matrix = false;
};

struct AccumConfig {
    AccumLayout layout = AccumLayout::Dense;
    // Tiled only; 0 = invalid, must be set (the Python policy derives it).
    std::uint32_t tile_rows = 0;
    // Entries with |value| <= threshold are dropped from the result CSR;
    // row sums and every other statistic are computed BEFORE thresholding,
    // so closure/conservation accounting stays exact. 0 keeps any nonzero.
    //
    // Both matrices compare on the INTENSIVE value: max(F_ij, F_ji) is the
    // stored G_ij over the smaller face area, max(B_ij, B_ji) the stored
    // H_ij over the smaller A*eps. A pair is therefore dropped only when
    // both directions are negligible. That comparison happens AFTER the two
    // directions are combined: dropping them independently first would leave
    // a surviving entry combined against a zero, which re-breaks the
    // reciprocity just imposed and biases the survivor low.
    double sparse_threshold = 0.0;
    // Applies to the VF and the exchange accumulator alike.
    TriangulationConfig triangulation;
};

}  // namespace pycanha::radiative
