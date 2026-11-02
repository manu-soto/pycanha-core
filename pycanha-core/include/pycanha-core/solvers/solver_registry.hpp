#pragma once

#include <memory>

namespace pycanha {
class ThermalMathematicalModel;
}  // namespace pycanha

namespace pycanha::solvers {

namespace steady {
class SteadyState;
}  // namespace steady
namespace transient {
class CrankNicolson;
}  // namespace transient
namespace sensitivity {
class JacobianPropagation;
}  // namespace sensitivity

/// Steady-state solvers of a model, created on first use.
class SteadySolvers {
  public:
    explicit SteadySolvers(std::shared_ptr<ThermalMathematicalModel> tmm);
    ~SteadySolvers();
    SteadySolvers(const SteadySolvers&) = delete;
    SteadySolvers& operator=(const SteadySolvers&) = delete;
    SteadySolvers(SteadySolvers&&) noexcept = delete;
    SteadySolvers& operator=(SteadySolvers&&) noexcept = delete;

    [[nodiscard]] steady::SteadyState& steady_state();

  private:
    std::shared_ptr<ThermalMathematicalModel> _tmm;
    std::unique_ptr<steady::SteadyState> _steady_state;
};

/// Transient solvers of a model, created on first use.
class TransientSolvers {
  public:
    explicit TransientSolvers(std::shared_ptr<ThermalMathematicalModel> tmm);
    ~TransientSolvers();
    TransientSolvers(const TransientSolvers&) = delete;
    TransientSolvers& operator=(const TransientSolvers&) = delete;
    TransientSolvers(TransientSolvers&&) noexcept = delete;
    TransientSolvers& operator=(TransientSolvers&&) noexcept = delete;

    [[nodiscard]] transient::CrankNicolson& crank_nicolson();

  private:
    std::shared_ptr<ThermalMathematicalModel> _tmm;
    std::unique_ptr<transient::CrankNicolson> _crank_nicolson;
};

/// Sensitivity solvers of a model, created on first use.
class SensitivitySolvers {
  public:
    explicit SensitivitySolvers(std::shared_ptr<ThermalMathematicalModel> tmm);
    ~SensitivitySolvers();
    SensitivitySolvers(const SensitivitySolvers&) = delete;
    SensitivitySolvers& operator=(const SensitivitySolvers&) = delete;
    SensitivitySolvers(SensitivitySolvers&&) noexcept = delete;
    SensitivitySolvers& operator=(SensitivitySolvers&&) noexcept = delete;

    [[nodiscard]] sensitivity::JacobianPropagation& jacobian_propagation();

  private:
    std::shared_ptr<ThermalMathematicalModel> _tmm;
    std::unique_ptr<sensitivity::JacobianPropagation> _jacobian_propagation;
};

/// One solver of each type per model, grouped by the question it answers:
/// tm.solvers().transient().crank_nicolson().
class SolverRegistry {
  public:
    explicit SolverRegistry(std::shared_ptr<ThermalMathematicalModel> tmm);
    ~SolverRegistry();

    SolverRegistry(const SolverRegistry&) = delete;
    SolverRegistry& operator=(const SolverRegistry&) = delete;
    SolverRegistry(SolverRegistry&&) noexcept = delete;
    SolverRegistry& operator=(SolverRegistry&&) noexcept = delete;

    [[nodiscard]] SteadySolvers& steady() noexcept { return _steady; }
    [[nodiscard]] TransientSolvers& transient() noexcept { return _transient; }
    [[nodiscard]] SensitivitySolvers& sensitivity() noexcept {
        return _sensitivity;
    }

    [[nodiscard]] std::shared_ptr<ThermalMathematicalModel> tmm_ptr()
        const noexcept {
        return _tmm;
    }

  private:
    std::shared_ptr<ThermalMathematicalModel> _tmm;
    SteadySolvers _steady;
    TransientSolvers _transient;
    SensitivitySolvers _sensitivity;
};

}  // namespace pycanha::solvers
