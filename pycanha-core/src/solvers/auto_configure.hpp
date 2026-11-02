#pragma once

#include <Eigen/Sparse>
#include <string>
#include <string_view>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/solvers/settings.hpp"

namespace pycanha::solvers::detail {

/// One setting an auto_configure() candidate sets, with the reason.
struct AutoField {
    std::string_view name;
    SettingValue value;
    std::string reason;
};

/// One configuration auto_configure() may choose.
struct AutoCandidate {
    std::vector<AutoField> fields;
};

/// How many diffusive nodes a model has and how they are coupled: what decides
/// between dense and sparse storage.
struct NetworkShape {
    Index nodes = 0;
    /// Other diffusive nodes each node is coupled to (conductively or
    /// radiatively), on average.
    double mean_couplings = 0.0;
    /// Entries of the n x n matrix that can be non-zero (couplings and
    /// diagonal) over n^2.
    double fill = 0.0;
};

/// The shape of the diffusive network from the upper triangles of its
/// coupling matrices.
[[nodiscard]] NetworkShape network_shape(
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& conductive,
    const Eigen::SparseMatrix<double, Eigen::RowMajor>& radiative);

/// Why dense storage suits this solver type and model; empty when it does
/// not. A steady state needs at least 100 couplings per node and at most
/// 5000 nodes; a transient, whose sparse analysis is spread over many steps,
/// a matrix at least 20 % filled and at most 2000 nodes.
[[nodiscard]] std::string dense_reason(SolverType type,
                                       const NetworkShape& shape);

/// The factorization libraries a solver can use.
struct Libraries {
    bool mkl = false;         ///< PARDISO and LAPACK
    bool accelerate = false;  ///< Apple Accelerate's sparse solvers
};

/// The candidates for a solver type and model, preferred first; the later
/// ones are tried only when the earlier ones do not fit max_memory. Sparse
/// storage takes PARDISO with MKL, else Accelerate, else Eigen.
[[nodiscard]] std::vector<AutoCandidate> auto_candidates(
    SolverType type, const Libraries& libraries, const NetworkShape& shape);

/// The settings auto_configure() owns: storage, backend, factorization,
/// reuse_factors, threads, pardiso.ordering, pardiso.level, pardiso.iparm,
/// eigen.ordering and accelerate.ordering. Unsets them.
void clear_auto_fields(Settings& settings);

}  // namespace pycanha::solvers::detail
