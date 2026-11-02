#include "auto_configure.hpp"

#include <Eigen/Sparse>
#include <algorithm>
#include <cstdint>
#include <format>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/solvers/settings.hpp"

namespace pycanha::solvers::detail {

namespace {

using SparseMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;

AutoField text(std::string_view name, std::string_view value,
               std::string_view reason) {
    return {.name = name,
            .value = std::string(value),
            .reason = std::string(reason)};
}

AutoField flag(std::string_view name, bool value, std::string_view reason) {
    return {.name = name, .value = value, .reason = std::string(reason)};
}

// When dense storage is chosen, measured on radiative enclosures, real
// radiative models and conduction plates of 100 to 5000 nodes: dense
// factorizes faster wherever each node is coupled to hundreds of others, and
// it needs no analysis. A sparse transient does its analysis once for all its
// steps, so its rule asks for a fuller matrix.
struct DenseRule {
    double min_mean_couplings = 0.0;
    double min_fill = 0.0;
    Index max_nodes = 0;
};
constexpr DenseRule steady_dense_rule{
    .min_mean_couplings = 100.0, .min_fill = 0.0, .max_nodes = 5000};
constexpr DenseRule transient_dense_rule{
    .min_mean_couplings = 0.0, .min_fill = 0.2, .max_nodes = 2000};

// Dense storage with the library of the build; it reuses factors where the
// solver type and the library allow it.
AutoCandidate dense(bool mkl_available, bool with_reuse_field,
                    const std::string& reason) {
    AutoCandidate candidate{
        {text("storage", "DENSE", reason),
         text("backend", mkl_available ? "LAPACK" : "EIGEN",
              mkl_available ? "this build has MKL: LAPACK factorizes in "
                              "parallel"
                            : "this build has no MKL"),
         text("factorization", "AUTO",
              "Cholesky where the matrix is symmetric, LU otherwise")}};
    if (mkl_available) {
        candidate.fields.push_back({.name = "threads",
                                    .value = std::int64_t{0},
                                    .reason = "MKL's own thread setting"});
    }
    if (with_reuse_field) {
        candidate.fields.push_back(flag(
            "reuse_factors", mkl_available,
            mkl_available
                ? "the matrix changes little between iterations, so FGMRES on "
                  "the previous factors avoids most factorizations"
                : "Eigen cannot reuse factors"));
    }
    return candidate;
}

constexpr std::string_view pardiso_reason =
    "this build has MKL: PARDISO factorizes in parallel";
constexpr std::string_view auto_reason =
    "Cholesky where the matrix is symmetric, LU otherwise";
constexpr std::string_view threads_reason = "MKL's own thread setting";
constexpr std::string_view safe_reason =
    "the two-level factorization with nested dissection does not livelock";
constexpr std::string_view sparse_reason =
    "the model is too large or too sparsely coupled for dense storage";
constexpr std::string_view fallback_reason =
    "fallback, tried when the preferred configuration does not fit "
    "max_memory";

// PARDISO with the two-level factorization, the safe configuration.
AutoCandidate pardiso_two_level(bool with_reuse_field,
                                std::string_view reuse_reason) {
    AutoCandidate candidate{
        {text("storage", "SPARSE", sparse_reason),
         text("backend", "PARDISO", pardiso_reason),
         text("factorization", "AUTO", auto_reason),
         {.name = "threads",
          .value = std::int64_t{0},
          .reason = std::string(threads_reason)},
         text("pardiso.level", "TWO_LEVEL", safe_reason),
         text("pardiso.ordering", "NESTED_DISSECTION", safe_reason)}};
    if (with_reuse_field) {
        candidate.fields.push_back(
            flag("reuse_factors", /*value=*/false, reuse_reason));
    }
    return candidate;
}

// PARDISO one-level with minimum degree, reusing factors or as a fallback.
struct MinDegreeReasons {
    std::string_view reuse;
    std::string_view level;
    std::string_view ordering;
};

constexpr MinDegreeReasons reuse_reasons{
    .reuse =
        "the matrix changes little between time steps, so iterating on "
        "the previous factors avoids most factorizations; the minimum "
        "degree analysis this needs is done once, and can take long on "
        "models where a few nodes are coupled to many",
    .level = "reuse_factors needs the one-level factorization",
    .ordering =
        "nested dissection can livelock with the one-level "
        "factorization"};

constexpr MinDegreeReasons fallback_reasons{.reuse = fallback_reason,
                                            .level = fallback_reason,
                                            .ordering = fallback_reason};

AutoCandidate pardiso_min_degree(bool with_reuse_field, bool reuse,
                                 const MinDegreeReasons& reasons) {
    AutoCandidate candidate{
        {text("storage", "SPARSE", sparse_reason),
         text("backend", "PARDISO", pardiso_reason),
         text("factorization", "AUTO", auto_reason),
         {.name = "threads",
          .value = std::int64_t{0},
          .reason = std::string(threads_reason)},
         text("pardiso.level", "ONE_LEVEL", reasons.level),
         text("pardiso.ordering", "MIN_DEGREE", reasons.ordering)}};
    if (with_reuse_field) {
        candidate.fields.push_back(flag("reuse_factors", reuse, reasons.reuse));
    }
    return candidate;
}

AutoCandidate eigen(std::string_view ordering, std::string_view reason) {
    return {{text("storage", "SPARSE", sparse_reason),
             text("backend", "EIGEN", "this build has no MKL"),
             text("factorization", "AUTO", auto_reason),
             text("eigen.ordering", ordering, reason)}};
}

// Accelerate cannot reuse factors, so reuse_factors keeps its default.
AutoCandidate accelerate(std::string_view ordering, std::string_view reason) {
    return {{text("storage", "SPARSE", sparse_reason),
             text("backend", "ACCELERATE",
                  "this build has no MKL: Accelerate factorizes in parallel"),
             text("factorization", "AUTO", auto_reason),
             text("accelerate.ordering", ordering, reason)}};
}

// The sparse candidates, preferred first.
std::vector<AutoCandidate> sparse_candidates(SolverType type,
                                             const Libraries& libraries) {
    if (!libraries.mkl && libraries.accelerate) {
        return {accelerate("METIS", "nested dissection, Accelerate's default"),
                accelerate("AMD", fallback_reason)};
    }
    if (!libraries.mkl) {
        return {eigen("COLAMD",
                      "the ordering with the least fill on most "
                      "models"),
                eigen("AMD", fallback_reason)};
    }
    if (type == SolverType::STEADY_STATE) {
        return {pardiso_two_level(/*with_reuse_field=*/true,
                                  "a steady state factorizes again only when "
                                  "the matrix changes"),
                pardiso_min_degree(/*with_reuse_field=*/true, /*reuse=*/false,
                                   fallback_reasons)};
    }
    if (type == SolverType::CRANK_NICOLSON) {
        return {pardiso_min_degree(/*with_reuse_field=*/true, /*reuse=*/true,
                                   reuse_reasons),
                pardiso_two_level(/*with_reuse_field=*/true, fallback_reason)};
    }
    // The sensitivity solve needs the factors of the current matrix, so
    // reuse_factors does not apply.
    return {pardiso_two_level(/*with_reuse_field=*/false, {}),
            pardiso_min_degree(/*with_reuse_field=*/false, /*reuse=*/false,
                               fallback_reasons)};
}

}  // namespace

NetworkShape network_shape(const SparseMatrix& conductive,
                           const SparseMatrix& radiative) {
    const Index nodes = conductive.rows();
    if (nodes == 0) {
        return {};
    }
    // Pairs coupled conductively, radiatively or both, counted once: the
    // two sorted rows are merged.
    double pairs = 0.0;
    for (Index row = 0; row < nodes; ++row) {
        SparseMatrix::InnerIterator gl(conductive, row);
        SparseMatrix::InnerIterator gr(radiative, row);
        while (gl || gr) {
            const Index col =
                !gr || (gl && gl.col() < gr.col()) ? gl.col() : gr.col();
            if (col != row) {
                pairs += 1.0;
            }
            if (gl && gl.col() == col) {
                ++gl;
            }
            if (gr && gr.col() == col) {
                ++gr;
            }
        }
    }
    const auto n = static_cast<double>(nodes);
    return {.nodes = nodes,
            .mean_couplings = 2.0 * pairs / n,
            .fill = (n + (2.0 * pairs)) / (n * n)};
}

std::string dense_reason(SolverType type, const NetworkShape& shape) {
    if (type == SolverType::STEADY_STATE) {
        const DenseRule& rule = steady_dense_rule;
        if (shape.nodes <= rule.max_nodes &&
            shape.mean_couplings >= rule.min_mean_couplings) {
            return std::format(
                "{} nodes, each coupled to {:.0f} others on average: a dense "
                "factorization is faster from {:.0f} couplings per node up "
                "to {} nodes",
                shape.nodes, shape.mean_couplings, rule.min_mean_couplings,
                rule.max_nodes);
        }
        return {};
    }
    const DenseRule& rule = transient_dense_rule;
    if (shape.nodes <= rule.max_nodes && shape.fill >= rule.min_fill) {
        return std::format(
            "{} nodes, {:.0f} % of the matrix filled: a dense factorization "
            "is faster from {:.0f} % filled up to {} nodes",
            shape.nodes, 100.0 * shape.fill, 100.0 * rule.min_fill,
            rule.max_nodes);
    }
    return {};
}

std::vector<AutoCandidate> auto_candidates(SolverType type,
                                           const Libraries& libraries,
                                           const NetworkShape& shape) {
    std::vector<AutoCandidate> candidates;
    if (const std::string reason = dense_reason(type, shape); !reason.empty()) {
        candidates.push_back(dense(
            libraries.mkl, type != SolverType::JACOBIAN_PROPAGATION, reason));
    }
    std::ranges::move(sparse_candidates(type, libraries),
                      std::back_inserter(candidates));
    return candidates;
}

void clear_auto_fields(Settings& settings) {
    settings.storage.reset();
    settings.backend.reset();
    settings.factorization.reset();
    settings.reuse_factors.reset();
    settings.threads.reset();
    settings.pardiso.ordering.reset();
    settings.pardiso.level.reset();
    settings.pardiso.iparm.reset();
    settings.eigen.ordering.reset();
    settings.accelerate.ordering.reset();
}

}  // namespace pycanha::solvers::detail
