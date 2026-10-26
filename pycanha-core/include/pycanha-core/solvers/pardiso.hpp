#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "pycanha-core/solvers/linear_solver.hpp"

#if PYCANHA_USE_MKL
#include <mkl_types.h>

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace pycanha {

/// RAII wrapper of one MKL PARDISO handle for a real sparse matrix.
/**
 * The matrix is passed to PARDISO in place, without a copy, as CSR with
 * zero-based indices (Eigen row-major, compressed). Matrix types 1 and 11
 * take both triangles; type 2 takes the upper triangle with every diagonal
 * entry stored. The pattern passed to analyze_pattern() must not change until
 * the next analyze_pattern() or release(); only the values may.
 */
class PardisoSolver : public SparseLinearSolver {
  public:
    enum class MatrixType : MKL_INT {
        STRUCTURALLY_SYMMETRIC = 1,
        SYMMETRIC_POSITIVE_DEFINITE = 2,
        UNSYMMETRIC = 11,
    };

    static constexpr std::size_t iparm_size = 64;

    /// iparm[3] is the CGS/CG control (10 * L + K). K is corrected to CG (2)
    /// for type 2 and to CGS (1) for the unsymmetric types. It is fixed here
    /// because PARDISO must know it at the analysis: with the two-level
    /// factorisation, enabling it only before phase 23 crashes MKL 2025.3.
    /// A non-zero value also makes PARDISO use the one-level factorisation.
    PardisoSolver(MatrixType type, MKL_INT iparm_3);
    ~PardisoSolver() override;

    PardisoSolver(const PardisoSolver&) = delete;
    PardisoSolver& operator=(const PardisoSolver&) = delete;
    PardisoSolver(PardisoSolver&&) = delete;
    PardisoSolver& operator=(PardisoSolver&&) = delete;

    /// Overrides one iparm entry. Only allowed before analyze_pattern(), and
    /// iparm[3] only through the constructor (see there).
    void set_iparm(std::size_t index, MKL_INT value);
    [[nodiscard]] MKL_INT iparm(std::size_t index) const;
    /// Threads for the PARDISO calls of this handle, 0 for MKL's setting.
    void set_threads(int threads) noexcept { _threads = threads; }
    /// PARDISO statistics printed to standard output.
    void set_verbose(bool verbose) noexcept {
        _message_level = verbose ? 1 : 0;
    }

    void analyze_pattern(const SparseMatrix& matrix) override;  // phase 11
    void factorize(const SparseMatrix& matrix) override;        // phase 22
    void factorize_and_solve(const SparseMatrix& matrix, Eigen::VectorXd& b,
                             Eigen::VectorXd& x) override;  // phase 23
    void solve(const SparseMatrix& matrix, Eigen::VectorXd& b,
               Eigen::VectorXd& x) override;  // phase 33
    void solve(const SparseMatrix& matrix, Eigen::MatrixXd& b,
               Eigen::MatrixXd& x) override;  // phase 33, several columns
    /// Phase -1 (frees the factors) if a pattern was analysed. Idempotent.
    void release() noexcept;

    [[nodiscard]] bool succeeded() const noexcept override {
        return _error == 0;
    }
    [[nodiscard]] bool zero_pivot() const noexcept override;
    [[nodiscard]] std::string error_message() const override;
    [[nodiscard]] Eigen::Index perturbed_pivots() const noexcept override;
    [[nodiscard]] Eigen::Index factor_nonzeros() const noexcept override;
    [[nodiscard]] bool factors_recomputed() const noexcept override;
    [[nodiscard]] int iterative_steps() const noexcept override;
    [[nodiscard]] bool iterative() const noexcept override;

    [[nodiscard]] Eigen::ComputationInfo info() const noexcept;
    [[nodiscard]] MKL_INT error() const noexcept { return _error; }
    [[nodiscard]] MatrixType matrix_type() const noexcept { return _type; }
    [[nodiscard]] bool analyzed() const noexcept { return _analyzed; }
    /// iparm[19] after phase 23: > 0 CGS/CG iterations with the previous
    /// factors, < 0 the iteration failed and PARDISO refactorised, 0 no
    /// iteration was attempted (fresh factorisation).
    [[nodiscard]] MKL_INT cgs_iterations() const noexcept;

  private:
    MatrixType _type;
    std::array<void*, iparm_size> _pt{};
    std::array<MKL_INT, iparm_size> _iparm{};
    std::vector<MKL_INT> _perm;
    MKL_INT _size = 0;
    MKL_INT _error = 0;
    MKL_INT _last_phase = 0;
    MKL_INT _message_level = 0;
    int _threads = 0;
    bool _analyzed = false;
    Eigen::Index _analyzed_non_zeros = 0;

    void run_phase(MKL_INT phase, const SparseMatrix& matrix, MKL_INT nrhs,
                   double* b, double* x);
};

}  // namespace pycanha

#endif  // PYCANHA_USE_MKL
