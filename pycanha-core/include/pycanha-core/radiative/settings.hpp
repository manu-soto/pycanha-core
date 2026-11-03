#pragma once

#include <cstdint>

namespace pycanha::radiative {

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
    // Hard safety bound on the bounce loop.
    std::uint32_t max_bounces = 64;
    // Emit along the face normal instead of the cosine-weighted hemisphere
    // (the legacy `_Nodes` debugging mode; vf/exchange kernels only).
    bool normal_emission = false;
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
    // RayDensity only: what N_i the weight's u = A_i/N_i reads. false: the
    // rays face i actually emitted, the minimum-variance weight for the
    // allocation that was traced. true: the mean over the emitting faces, so
    // the weight follows the areas alone, as a uniform trace sets it.
    //
    // The combined value's expectation is the weighted mean of the two
    // directions' expectations. Where those differ (faces that touch or
    // overlap, a ray origin offset that leaks), actual-count weights move it
    // whenever the rays are allocated other than uniformly (more rays to
    // some faces, fewer to others) and the result then depends on the
    // allocation. With area weights an uneven allocation changes only the
    // noise. On a uniform trace both settings give the same weights, bit for
    // bit. ConstrainedLeastSquares ignores it.
    bool area_weights = false;
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
