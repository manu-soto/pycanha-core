#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>
#include <cstdint>
#include <optional>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/radiative/materials.hpp"

namespace pycanha::radiative {

// Every sparse radiative result is a row-major Eigen sparse matrix: the same
// type the TMM coupling matrices use, and the one nanobind converts straight
// to scipy.sparse.csr_matrix without a hand-written binding.
using SparseMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;
// Column/row-pointer type of the CSR arrays above (Eigen's default `int`).
using SparseIndex = SparseMatrix::StorageIndex;

// Matrix results carry three VIRTUAL bucket columns appended after the
// num_faces real columns, so every parcel of emitted energy has an
// explicit, node-mappable destination:
//  - space:    rays/energy that escaped the scene,
//  - inactive: energy absorbed at inactive (non-radiative) faces
//              (exchange only — the geometric vf kernel scores inactive
//              faces at their own column),
//  - lost:     the purely mathematical residue (Russian-roulette balance,
//              max_bounces cutoff). Zero-mean adjustments make this entry
//              SIGNED and it may come out slightly negative.
// Full rows therefore account for everything: a vf row sums to exactly 1
// and an exchange row conserves energy exactly.
inline constexpr std::int64_t num_virtual_columns = 3;
inline constexpr std::int64_t space_column_offset = 0;
inline constexpr std::int64_t inactive_column_offset = 1;
inline constexpr std::int64_t lost_column_offset = 2;

// Every result carries its own statistics so callers can judge Monte-Carlo
// convergence without re-tracing.
struct TraceStats {
    std::uint64_t total_rays = 0;
    // Cumulative across accumulate() calls.
    std::uint64_t rays_per_face = 0;
    // Per-entry standard-error estimates.
    double mean_stderr = 0.0;
    double max_stderr = 0.0;
    // Matrix results: the largest normalized disagreement of the two
    // directions of a face pair, measured on the RAW estimates before they
    // are combined. Measured after combining it would be identically zero
    // by construction and would stop being the winding/parity check it
    // exists to be.
    double reciprocity_residual = 0.0;
    // Exchange only: energy killed below the threshold.
    double lost_energy_fraction = 0.0;
    std::chrono::nanoseconds gpu_time{0};
};

// Matrix rows are faces (global, both sides); columns are the same
// faces plus the virtual bucket columns above (cols == rows +
// num_virtual_columns). The consumer maps the buckets to real nodes (e.g.
// the space node) or drops them.
//
// The stored value is the SYMMETRIC, EXTENSIVE quantity
//
//     G_ij = A_i F_ij = A_j F_ji     (units m^2)
//
// and only the UPPER TRIANGLE (j >= i) of the real face columns is kept,
// matching the TMM coupling convention. Storing G rather than F is what
// makes that lossless: both view factors come back as F_ij = G_ij/A_i and
// F_ji = G_ij/A_j from the face areas the scene already owns, whereas an
// upper-triangular F would silently discard one direction. The diagonal is
// retained even though a planar face cannot see itself, so nothing is
// dropped without saying so. The three bucket columns are always > i, have
// no transpose partner, and pass through untriangulated.
struct VfResult {
    SparseMatrix vf;
    // Per-row sum over ALL columns of the RAW estimate, before combining
    // and before thresholding (== 1 exactly for rows that emitted). This is
    // closure, and it is deliberately not the row sum of `vf`: after
    // triangulation the stored triangle's row sums mean nothing of the
    // kind.
    Eigen::VectorXd row_sums;
    // The raw, untriangulated G with BOTH triangles, present only when
    // TriangulationConfig::keep_full_matrix asked for it. Entry (i, j)
    // holds A_i F_ij and entry (j, i) holds A_j F_ji, so the two
    // independent estimates of the same coupling can be read off and
    // compared directly.
    std::optional<SparseMatrix> full_vf;
    TraceStats stats;
};

// Same shape and the same convention as VfResult: rows are faces,
// columns are the faces plus the virtual bucket columns, and only the UPPER
// TRIANGLE of the real face columns is kept.
//
// The stored value is the SYMMETRIC, EXTENSIVE quantity
//
//     H_ij = A_i eps_i B_ij = A_j eps_j B_ji     (units m^2)
//
// where B is the Gebhart absorption factor the kernel traces (deposited
// energy per unit emitted) and eps is the traced band's absorptivity. H is
// what actually transports heat and what the node-level GR is built from, so
// storing it is what makes the upper triangle lossless: both factors come
// back as B_ij = H_ij/(A_i eps_i) and B_ji = H_ij/(A_j eps_j).
//
// A face with eps = 0 in the traced band is the one case where that does not
// invert. It needs no special handling because there is nothing to recover:
// such a face absorbs nothing, so B_ji = 0 for every j, and it emits nothing,
// so its row transports no heat whatever the kernel's unit-energy rays did.
// Both directions vanish consistently and H stays reciprocal. `full_factors`
// keeps the raw B for anyone who wants the geometry behind that zero.
struct ExchangeResult {
    Band band = Band::IR;
    SparseMatrix factors;
    // The raw, untriangulated INTENSIVE B with BOTH triangles, present only
    // when TriangulationConfig::keep_full_matrix asked for it. Deliberately
    // not the extensive H that `full_vf` would suggest: for view factors the
    // intensive form is always recoverable because areas are positive, for
    // exchange it is not, so the debugging matrix carries the form that
    // cannot be reconstructed.
    std::optional<SparseMatrix> full_factors;
    TraceStats stats;
};

// NodeExchangeResult carries one more virtual column than the face results:
// after space, inactive and lost (at the offsets above) comes the energy the
// node's faces sent to faces that belong to no node.
inline constexpr std::int64_t num_node_virtual_columns = 4;
inline constexpr std::int64_t unassigned_column_offset = 3;

// The exchange result summed to nodes while it is traced
// (NodeExchangeAccumulator). Rows are nodes, indexed by their position in
// `nodes` (aggregate_nodes of the node numbers the accumulator was given);
// columns are the same nodes plus the num_node_virtual_columns buckets.
//
// The stored value is ExchangeResult's H summed over the face pairs of each
// node pair, H_mn = sum over faces i of m and j of n of H_ij, with the same
// combination of each face pair's two directions: the matrix aggregate_matrix
// makes of the face result, up to float rounding. Only the UPPER TRIANGLE of
// the node columns is kept and the node diagonal is left out (radiation a
// node exchanges with itself moves no heat), its total reported in
// intra_node_total as aggregate_matrix reports it.
//
// The buckets are each node's own, as traced, with no transpose partner: the
// summed space / inactive / lost columns of its faces' rows, and in the
// unassigned column what landed on faces in no node (the forward estimate
// alone, since those faces are not traced).
//
// absorptance_derivatives (when the accumulator was given absorptance
// groups): one matrix per group, laid out as `factors`, holding the
// derivative of every entry with respect to the band absorptance a (the IR
// emissivity in the IR band) of all the faces of that group, the
// reflectance taking a change of a with its specular share kept and the
// transmittance fixed. Traced with the same rays as `factors` (the score of
// each path, see exchange.slang): no second trace, no step. The score only
// sees paths the trace can take: a face whose a is 0 absorbs and emits
// nothing, and one that reflects nothing (a + t = 1, a black body) sends no
// reflected path, so what raising a would add there (its first deposits and
// emission) or lowering it would add (the reflections it would start to make)
// is not traced. Give a group faces with 0 < a < 1 - t.
inline constexpr std::int32_t max_absorptance_groups = 4;

struct NodeExchangeResult {
    Band band = Band::IR;
    std::vector<NodeNum> nodes;
    SparseMatrix factors;
    double intra_node_total = 0.0;
    TraceStats stats;
    std::vector<SparseMatrix> absorptance_derivatives;
};

// Per-face absorbed power; the solar kernel is O(Nf) and needs no
// matrix. Watts (extensive): node mapping is a plain per-node sum, and
// flux is watts / face area when needed.
struct SolarResult {
    Eigen::VectorXd direct;  // W absorbed, direct illumination
    Eigen::VectorXd total;   // W absorbed incl. reflections
    TraceStats stats;
};

}  // namespace pycanha::radiative
