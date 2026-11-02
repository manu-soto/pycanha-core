#pragma once

#include "pycanha-core/solvers/settings.hpp"

#if PYCANHA_USE_ACCELERATE
#include <memory>

#include "factorizer.hpp"

namespace pycanha::solvers::detail {

/// Sparse direct factorization of the Apple Accelerate framework.
/**
 * Multi-threaded supernodal factorization: LU (threshold partial pivoting)
 * for a matrix holding both triangles, Cholesky when `symmetric` (upper
 * triangle only, every diagonal entry stored). The ordering is the
 * fill-reducing one: METIS (nested dissection) or AMD.
 *
 * The matrix is passed without copying its values: the row-major (CSR)
 * arrays are read as the transpose of a CSC matrix for LU, and as the lower
 * triangle of a CSC matrix for Cholesky. The index arrays are copied at
 * analyze_pattern(), and refactorizations reuse the symbolic analysis and
 * the factor storage. Requires accelerate_available().
 */
[[nodiscard]] std::unique_ptr<Factorizer> make_accelerate_factorizer(
    AccelerateOrdering ordering, bool symmetric);

}  // namespace pycanha::solvers::detail

#endif  // PYCANHA_USE_ACCELERATE
