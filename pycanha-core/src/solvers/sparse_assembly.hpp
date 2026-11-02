#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <memory>

#include "linearized_network.hpp"

namespace pycanha::solvers::detail {

/// Writes the values of the sparse matrix A + diag(D) of a network.
/**
 * A is the linearized matrix described in LinearizedNetwork. The pattern of
 * the matrix is fixed at construction (sparse_pattern()); assemble() only
 * writes its values, in place.
 */
class SparseAssembly {
  public:
    SparseAssembly() = default;
    virtual ~SparseAssembly() = default;
    SparseAssembly(const SparseAssembly&) = delete;
    SparseAssembly& operator=(const SparseAssembly&) = delete;
    SparseAssembly(SparseAssembly&&) = delete;
    SparseAssembly& operator=(SparseAssembly&&) = delete;

    /// A new solve() starts.
    virtual void begin_solve() noexcept {}

    /// A + diag(diagonal) at the temperatures `t` (t3 = 4 sigma t^3) into
    /// `matrix`, and 0.75 A_rad t into `radiation`. An empty `diagonal`
    /// adds nothing. Returns false only when the values are known to be
    /// those of the previous call.
    [[nodiscard]] virtual bool assemble(SparseMatrix& matrix,
                                        const Eigen::VectorXd& t3,
                                        const Eigen::VectorXd& t,
                                        const Eigen::VectorXd& diagonal,
                                        Eigen::VectorXd& radiation) = 0;

    /// Memory of everything the assembly holds [bytes].
    [[nodiscard]] virtual std::int64_t bytes() const noexcept = 0;
};

/// The pattern of A with zero values: both triangles, or the upper one in
/// the Cholesky form.
[[nodiscard]] SparseMatrix sparse_pattern(const NetworkCouplings& couplings,
                                          const NetworkOptions& options);

/// The assembly the options select: the couplings read at every
/// assemble(), or once per solve() with fixed_couplings.
[[nodiscard]] std::unique_ptr<SparseAssembly> make_sparse_assembly(
    const SparseMatrix& pattern, const NetworkCouplings& couplings,
    const NetworkOptions& options);

}  // namespace pycanha::solvers::detail
