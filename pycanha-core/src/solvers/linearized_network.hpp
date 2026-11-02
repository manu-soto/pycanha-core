#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/solvers/settings.hpp"

namespace pycanha::solvers::detail {

using SparseMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;

/// The couplings of the diffusive nodes (not owned, never null), read again
/// at every linearize(): formulas and callbacks may change their values at
/// any time.
struct NetworkCouplings {
    /// Upper triangle [W/K].
    const SparseMatrix* conductive = nullptr;
    /// Diffusive x boundary [W/K].
    const SparseMatrix* conductive_boundary = nullptr;
    /// Upper triangle [m^2].
    const SparseMatrix* radiative = nullptr;
    /// Diffusive x boundary [m^2].
    const SparseMatrix* radiative_boundary = nullptr;
};

/// How a network is built.
struct NetworkOptions {
    /// Cholesky form: the model has no radiative couplings.
    bool symmetric = false;
    /// The model has radiative couplings.
    bool radiation = false;
    /// solve() compares the matrix with the one last factorized and reuses
    /// the factors when they are equal. Not needed with fixed_couplings,
    /// which knows when the matrix changes.
    bool skip_unchanged = false;
    /// The matrix is still needed after solve() (multiply()); otherwise a
    /// dense factorization may overwrite it, and the next linearize()
    /// rebuilds it.
    bool keep_matrix = false;
    /// The couplings and the diagonal passed to linearize() stay as they are
    /// at the first linearize() of each solve(): the user's promise, trusted
    /// and never checked.
    bool fixed_couplings = false;
};

/// What solve() did.
struct SolveOutcome {
    bool succeeded = false;
    /// The factors were computed in this call (not reused).
    bool factorized = false;
};

/// The linearized matrix of the diffusive nodes and its factorization.
/**
 * linearize() sets A = -K, with K the conductance matrix with the radiation
 * linearized at the given temperatures: K_ij = GL_ij + GR_ij 4 sigma Tj^3 off
 * the diagonal, and minus the sum of the couplings of row i, boundary
 * couplings included, on it. The network holds A + D, with D the diagonal
 * the solver passes to linearize() (2C/dt in a transient). A is symmetric
 * positive definite without radiative couplings.
 *
 * With fixed_couplings the matrix is built from constants read once per
 * solve(); otherwise every linearize() reads the couplings again. Either way
 * the network knows when its factors belong to the current matrix.
 */
class LinearizedNetwork {
  public:
    LinearizedNetwork() = default;
    virtual ~LinearizedNetwork() = default;
    LinearizedNetwork(const LinearizedNetwork&) = delete;
    LinearizedNetwork& operator=(const LinearizedNetwork&) = delete;
    LinearizedNetwork(LinearizedNetwork&&) = delete;
    LinearizedNetwork& operator=(LinearizedNetwork&&) = delete;

    /// The analysis of the factorization (sparse only); false on failure.
    [[nodiscard]] virtual bool analyze() = 0;
    /// The storage only a run needs; estimate_memory() never calls it.
    virtual void allocate() = 0;
    /// A new solve() starts: with fixed_couplings the next linearize() reads
    /// the couplings and the diagonal again.
    virtual void begin_solve() = 0;

    /// A + diag(diagonal) at the temperatures `t` (t3 = 4 sigma t^3), and
    /// 0.75 A_rad t in `radiation`: the part of the Newton linearisation
    /// that moves to the right-hand side (zero without radiative couplings).
    /// An empty `diagonal` adds nothing.
    virtual void linearize(const Eigen::VectorXd& t3, const Eigen::VectorXd& t,
                           const Eigen::VectorXd& diagonal,
                           Eigen::VectorXd& radiation) = 0;
    /// y = (A + D) x.
    virtual void multiply(const Eigen::VectorXd& x,
                          Eigen::VectorXd& y) const = 0;
    virtual void multiply(const Eigen::MatrixXd& x,
                          Eigen::MatrixXd& y) const = 0;

    /// Solves (A + D) x = b. Factorizes the matrix, unless the factors
    /// already belong to it or reuse_factors iterates on older ones,
    /// starting from the x given.
    [[nodiscard]] virtual SolveOutcome solve(Eigen::VectorXd& b,
                                             Eigen::VectorXd& x) = 0;
    /// Several right-hand sides with the current factors; false on failure.
    [[nodiscard]] virtual bool solve(Eigen::MatrixXd& b,
                                     Eigen::MatrixXd& x) = 0;
    /// The next solve() factorizes again.
    virtual void discard_factors() noexcept = 0;

    /// Threads and verbosity, which may change between runs.
    virtual void apply_live_settings(const ResolvedSettings& settings) = 0;

    /// The last failure was a singular matrix (a zero or tiny pivot).
    [[nodiscard]] virtual bool zero_pivot() const noexcept = 0;
    [[nodiscard]] virtual std::string error_message() const = 0;
    /// Pivots PARDISO replaced by a small value in the last factorization.
    [[nodiscard]] virtual Index perturbed_pivots() const noexcept { return 0; }
    /// A warning about the factorization chosen; empty when there is none.
    [[nodiscard]] virtual std::optional<std::string> hint(
        const ResolvedSettings& /*settings*/) const {
        return std::nullopt;
    }

    /// Memory of the factors [bytes], and whether it is known rather than
    /// predicted.
    [[nodiscard]] virtual std::int64_t factor_bytes() const noexcept = 0;
    [[nodiscard]] virtual bool factor_bytes_exact() const noexcept = 0;
    /// Memory of the matrix and of everything else the network holds.
    [[nodiscard]] virtual std::int64_t matrix_bytes() const noexcept = 0;
};

/// The network the resolved settings select (storage, backend,
/// factorization, reuse). Throws std::invalid_argument for a configuration
/// the build cannot run.
[[nodiscard]] std::unique_ptr<LinearizedNetwork> make_network(
    const ResolvedSettings& settings, const NetworkCouplings& couplings,
    const NetworkOptions& options);

/// Sparse storage (PARDISO or Eigen's sparse solvers).
[[nodiscard]] std::unique_ptr<LinearizedNetwork> make_sparse_network(
    const ResolvedSettings& settings, const NetworkCouplings& couplings,
    const NetworkOptions& options);

/// Dense storage (LAPACK or Eigen's dense solvers).
[[nodiscard]] std::unique_ptr<LinearizedNetwork> make_dense_network(
    const ResolvedSettings& settings, const NetworkCouplings& couplings,
    const NetworkOptions& options);

}  // namespace pycanha::solvers::detail
