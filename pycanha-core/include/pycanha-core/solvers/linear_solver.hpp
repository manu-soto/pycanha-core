#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <map>
#include <memory>
#include <string>
#include <string_view>

namespace pycanha {

/// Library that factorises the linearised system of a solver.
enum class SolverEngine {
    /// Intel MKL PARDISO: multi-threaded supernodal factorisation. Only in
    /// builds with MKL (pycanha::MKL_ENABLED).
    MKL,
    /// Eigen's sparse solvers: single-threaded, available in every build.
    EIGEN,
    /// Apple Accelerate: multi-threaded supernodal factorisation. Only in
    /// macOS builds with Accelerate, on macOS 15.5 or newer
    /// (accelerate_available()).
    ACCELERATE,
};

/// Factorisation of a direct solver. Every type computes a complete
/// factorisation: the linear system is solved exactly (to rounding) at every
/// pass, so all types give the same temperatures and differ only in time and
/// memory.
///
/// MKL PARDISO types. Models without radiative couplings are factorised with
/// Cholesky when the solver allows it, the others with LU.
///  - TWO_LEVEL: parallel nested dissection ordering (METIS) and PARDISO's
///    two-level factorisation. The default with MKL, and the safe choice.
///  - ONE_LEVEL: same ordering, PARDISO's classic one-level factorisation.
///    It seems slightly faster on most models, but with MKL 2025.3 it can
///    livelock (never finish) with 4 or more threads on some radiative models.
///  - MIN_DEGREE: minimum degree ordering, one-level factorisation. It seems
///    faster on nearly dense models (many radiative couplings per node), and
///    its analysis is slow on models where a few nodes are coupled to many.
///
/// Eigen types:
///  - COLAMD: SparseLU with COLAMD ordering. The default with Eigen.
///  - AMD: SparseLU with approximate minimum degree ordering. It seems faster
///    than COLAMD only on models where a few nodes are coupled to many.
///  - LDLT: SimplicialLDLT with AMD ordering, only for models without
///    radiative couplings. It seems faster than COLAMD on small and medium
///    models, and uses less memory.
///
/// Accelerate types (the ordering; models without radiative couplings are
/// factorised with Cholesky when the solver allows it, the others with LU):
///  - METIS: nested dissection ordering. The default with Accelerate.
///  - AMD: approximate minimum degree ordering (shared with Eigen).
enum class DirectSolverType {
    DEFAULT,  ///< TWO_LEVEL with MKL, COLAMD with Eigen, METIS with Accelerate
    TWO_LEVEL,
    ONE_LEVEL,
    MIN_DEGREE,
    COLAMD,
    AMD,
    LDLT,
    METIS,
};

/// Factorisation of an iterative solver (MKL PARDISO only). After the first
/// factorisation, a changed matrix is first solved with CGS (or CG for
/// Cholesky) preconditioned by the previous factors, and only refactorised
/// when that iteration fails. The iteration stops at a relative residual of
/// 10^-L, with L the tens digit of pardiso_iparm_3 (61: 1e-6), and that
/// residual remains in the converged temperatures. PARDISO only supports the
/// iteration with its one-level factorisation, hence the two types:
///  - MIN_DEGREE: minimum degree ordering. The default. Its analysis is slow
///    on models where a few nodes are coupled to many.
///  - ONE_LEVEL: parallel nested dissection ordering (METIS). It seems faster,
///    but with MKL 2025.3 it can livelock with 4 or more threads on some
///    radiative models.
enum class IterativeSolverType {
    MIN_DEGREE,
    ONE_LEVEL,
};

/// MKL when the library is built with it, then Accelerate when available,
/// Eigen otherwise.
[[nodiscard]] SolverEngine default_solver_engine() noexcept;

[[nodiscard]] std::string_view to_string(SolverEngine engine) noexcept;
[[nodiscard]] std::string_view to_string(DirectSolverType type) noexcept;

/// DEFAULT replaced by the engine's default type.
[[nodiscard]] DirectSolverType resolve_solver_type(
    SolverEngine engine, DirectSolverType type) noexcept;

struct LinearSolverOptions {
    SolverEngine engine = SolverEngine::EIGEN;
    DirectSolverType type = DirectSolverType::DEFAULT;
    /// The matrix is symmetric positive definite and holds only its upper
    /// triangle (PARDISO Cholesky, Eigen LDLT). Otherwise it holds both
    /// triangles of a structurally symmetric pattern.
    bool symmetric_positive_definite = false;
    /// PARDISO iparm[3] (iterative step), 0 for none.
    int iparm_3 = 0;
    /// PARDISO iparm entries applied last (index 3 replaces iparm_3).
    std::map<int, int> iparm_overrides;
    /// Threads for the PARDISO calls, 0 for MKL's setting. Accelerate has no
    /// per-call thread setting and ignores it.
    int threads = 0;
    /// PARDISO statistics printed to standard output.
    bool verbose = false;
};

/// Sparse direct solver of A x = b for a compressed row-major matrix whose
/// pattern is fixed between analyze_pattern() and the next one.
class SparseLinearSolver {
  public:
    using SparseMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;

    SparseLinearSolver() = default;
    virtual ~SparseLinearSolver() = default;
    SparseLinearSolver(const SparseLinearSolver&) = delete;
    SparseLinearSolver& operator=(const SparseLinearSolver&) = delete;
    SparseLinearSolver(SparseLinearSolver&&) = delete;
    SparseLinearSolver& operator=(SparseLinearSolver&&) = delete;

    virtual void analyze_pattern(const SparseMatrix& matrix) = 0;
    virtual void factorize(const SparseMatrix& matrix) = 0;
    /// With the current factors. b is not modified.
    virtual void solve(const SparseMatrix& matrix, Eigen::VectorXd& b,
                       Eigen::VectorXd& x) = 0;
    /// Several right-hand sides (columns) with the current factors.
    virtual void solve(const SparseMatrix& matrix, Eigen::MatrixXd& b,
                       Eigen::MatrixXd& x) = 0;
    /// Factorise and solve. Iterative solvers may iterate on the previous
    /// factors instead (see factors_recomputed()).
    virtual void factorize_and_solve(const SparseMatrix& matrix,
                                     Eigen::VectorXd& b, Eigen::VectorXd& x);

    [[nodiscard]] virtual bool succeeded() const noexcept = 0;
    /// The last failure was a zero pivot (a singular matrix).
    [[nodiscard]] virtual bool zero_pivot() const noexcept = 0;
    [[nodiscard]] virtual std::string error_message() const = 0;
    /// Pivots PARDISO replaced by a small value, 0 for Eigen.
    [[nodiscard]] virtual Eigen::Index perturbed_pivots() const noexcept {
        return 0;
    }
    /// Non-zeros of the factors (PARDISO predicts it at the analysis), -1
    /// when unknown.
    [[nodiscard]] virtual Eigen::Index factor_nonzeros() const noexcept {
        return -1;
    }
    /// False after factorize_and_solve() only when it iterated on the
    /// previous factors, which then still belong to an older matrix.
    [[nodiscard]] virtual bool factors_recomputed() const noexcept {
        return true;
    }
    /// Iterations of the last factorize_and_solve() that iterated on the
    /// previous factors, 0 when it factorised or the solver does not iterate.
    [[nodiscard]] virtual int iterative_steps() const noexcept { return 0; }
    [[nodiscard]] virtual bool iterative() const noexcept { return false; }
};

/// Throws std::invalid_argument when the combination is not available: MKL
/// or Accelerate when not available, a type of another engine, LDLT without
/// symmetric_positive_definite, an iterative step with Eigen, Accelerate or
/// TWO_LEVEL.
[[nodiscard]] std::unique_ptr<SparseLinearSolver> make_linear_solver(
    const LinearSolverOptions& options);

/// Logs a warning suggesting DirectSolverType::MIN_DEGREE when PARDISO
/// predicts a large, near-dense factor for a nested dissection ordering.
void suggest_min_degree_if_dense(const SparseLinearSolver& solver,
                                 const LinearSolverOptions& options,
                                 Eigen::Index size,
                                 std::string_view solver_name);

}  // namespace pycanha
