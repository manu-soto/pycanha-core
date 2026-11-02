#pragma once

#include <Eigen/Core>
#include <cstdint>
#include <memory>

#include "linearized_network.hpp"
#include "pycanha-core/globals.hpp"

namespace pycanha::solvers::detail {

using DenseMatrix = Eigen::MatrixXd;

/// n x n doubles [bytes].
[[nodiscard]] inline std::int64_t square_bytes(Index size) noexcept {
    return static_cast<std::int64_t>(size) * size *
           static_cast<std::int64_t>(sizeof(double));
}

/// Writes the dense matrix A + diag(D) of a network.
/**
 * A is the linearized matrix described in LinearizedNetwork, stored as a
 * full n x n array.
 */
class DenseAssembly {
  public:
    DenseAssembly() = default;
    virtual ~DenseAssembly() = default;
    DenseAssembly(const DenseAssembly&) = delete;
    DenseAssembly& operator=(const DenseAssembly&) = delete;
    DenseAssembly(DenseAssembly&&) = delete;
    DenseAssembly& operator=(DenseAssembly&&) = delete;

    /// The storage only a run needs; estimate_memory() never calls it.
    virtual void allocate() = 0;

    /// A new solve() starts.
    virtual void begin_solve() noexcept {}

    /// A + diag(diagonal) at the temperatures `t` (t3 = 4 sigma t^3) into
    /// `matrix`, and 0.75 A_rad t into `radiation`. An empty `diagonal`
    /// adds nothing. With `rewrite` the matrix holds anything (a
    /// factorization took its storage) and is written in full; otherwise it
    /// holds what the previous call wrote. Returns false only when the
    /// values are known to be those of the previous call.
    [[nodiscard]] virtual bool assemble(DenseMatrix& matrix, bool rewrite,
                                        const Eigen::VectorXd& t3,
                                        const Eigen::VectorXd& t,
                                        const Eigen::VectorXd& diagonal,
                                        Eigen::VectorXd& radiation) = 0;

    /// Memory of everything the assembly holds [bytes].
    [[nodiscard]] virtual std::int64_t bytes() const noexcept = 0;
};

/// The assembly the options select: the couplings read at every
/// assemble(), or once per solve() with fixed_couplings.
[[nodiscard]] std::unique_ptr<DenseAssembly> make_dense_assembly(
    const NetworkCouplings& couplings, const NetworkOptions& options);

}  // namespace pycanha::solvers::detail
