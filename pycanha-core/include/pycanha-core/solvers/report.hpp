#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace pycanha::solvers {

/// Memory of a solver [bytes]: its factors, its own matrices and vectors, and
/// the output it stores in the thermal data.
struct SolverMemory {
    /// The factorization: PARDISO's own figure (analysis and factorization
    /// peak), or Eigen's factor non-zeros.
    std::int64_t factors = 0;
    /// The matrices, index tables and vectors the solver assembles.
    std::int64_t matrix = 0;
    /// The output model of a transient (stored temperatures, loads, coupling
    /// matrices, Jacobian).
    std::int64_t output = 0;
    /// The factor size is known, not predicted: PARDISO two-level and Eigen
    /// Cholesky from their analysis, PARDISO one-level (an upper bound
    /// before) and Eigen LU after their first factorization.
    bool exact = false;

    [[nodiscard]] std::int64_t total() const noexcept {
        return factors + matrix + output;
    }
    /// "1.21 GB (factors 1.02 GB, matrix 121 MB, output 64.0 MB)".
    [[nodiscard]] std::string to_string() const;
};

/// "1.21 GB", "512 kB", "17 B" (powers of 1000).
[[nodiscard]] std::string format_bytes(std::int64_t bytes);

/// Bytes of "16 GB", "512 MB", "1.5 GiB" or "2000000". kB, MB, GB, TB are
/// powers of 1000, KiB, MiB, GiB, TiB powers of 1024; case and the space
/// before the unit do not matter. Throws std::invalid_argument for anything
/// else, and for a size that is not positive.
[[nodiscard]] std::int64_t parse_memory_size(std::string_view text);

/// How the last solve() ended.
enum class SolveStatus : std::uint8_t {
    NOT_RUN,        ///< No solve() since initialize().
    CONVERGED,      ///< Every iteration loop met the tolerance.
    NOT_CONVERGED,  ///< Some loop stopped at max_iterations.
    FAILED,         ///< solve() stopped early: an error or an exception.
};

[[nodiscard]] std::string_view to_string(SolveStatus status) noexcept;

/// A time step (or the steady state) that stopped at max_iterations.
struct NonConvergedStep {
    int step = 0;        ///< Time step, counted from 1; 0 for the steady state.
    double time = 0.0;   ///< Time at the end of the step [s].
    int iterations = 0;  ///< Iterations done.
    double max_dT = 0.0;    ///< Largest change of the last iteration [K].
    std::int64_t node = 0;  ///< Node number with that change.
};

/// A report value, as SolverReport::to_dict() gives it.
using ReportValue =
    std::variant<bool, std::int64_t, double, std::string,
                 std::vector<std::string>, std::vector<std::int64_t>,
                 std::vector<double>>;
using ReportDict = std::map<std::string, ReportValue, std::less<>>;

/// What the last solve() did. initialize() clears it.
/**
 * The counters are aggregates kept during the run (a few additions per
 * iteration and two clock reads around each callback); the text is built
 * only by to_string(). Only the first max_listed_steps non-converged steps
 * are listed, all of them are counted.
 */
struct SolverReport {
    static constexpr std::size_t max_listed_steps = 100;

    std::string solver;  ///< The solver name, "CrankNicolson".
    SolveStatus status = SolveStatus::NOT_RUN;

    /// Time steps done; 0 for the steady state.
    int steps = 0;
    /// Iterations of every loop, added up.
    int iterations = 0;
    /// The longest iteration loop.
    int max_step_iterations = 0;
    /// Loops that stopped at max_iterations.
    int nonconverged_steps = 0;
    /// The first max_listed_steps of them.
    std::vector<NonConvergedStep> nonconverged;
    /// Largest change of the last iteration [K].
    double max_dT = 0.0;
    /// Estimated error the iteration on old factors leaves in the
    /// temperatures [K]: reuse_tolerance times the largest change of the last
    /// iteration of the steady state, or of any time step; 0 without
    /// reuse_factors. An order of magnitude, not a bound: LAPACK's FGMRES,
    /// which stops on the residual without the factors, and long transients,
    /// where it adds up over the steps, can leave a few times more. Rounding
    /// errors, the same as a direct solve's, come on top.
    double reuse_error = 0.0;

    /// Numerical factorizations.
    int factorizations = 0;
    /// Linear solves; those beyond the factorizations used existing or
    /// earlier factors.
    int solves = 0;

    double initialize_time = 0.0;  ///< Wall time of initialize() [s].
    double solve_time = 0.0;       ///< Wall time of solve() [s].
    double callback_time = 0.0;    ///< Time inside the callbacks [s].
    double linear_time = 0.0;      ///< Factorizations and solves [s].

    /// The solver's own memory after the run (not the process peak).
    SolverMemory memory;
    /// The warnings of the last initialize() and solve().
    std::vector<std::string> warnings;

    [[nodiscard]] bool converged() const noexcept {
        return status == SolveStatus::CONVERGED;
    }
    /// Fraction of the solves done without a new factorization.
    [[nodiscard]] double reuse_rate() const noexcept;

    [[nodiscard]] std::string to_string() const;
    /// Flat: the non-converged list as parallel lists ("nonconverged.step",
    /// "nonconverged.time", ...), memory as "memory.factors", ...
    [[nodiscard]] ReportDict to_dict() const;
};

}  // namespace pycanha::solvers
