#include "pycanha-core/solvers/solver_registry.hpp"

#include <memory>
#include <stdexcept>
#include <utility>

#include "pycanha-core/solvers/crank_nicolson.hpp"
#include "pycanha-core/solvers/jacobian_propagation.hpp"
#include "pycanha-core/solvers/steady_state.hpp"

namespace pycanha::solvers {

namespace {

std::shared_ptr<ThermalMathematicalModel> require_model(
    std::shared_ptr<ThermalMathematicalModel> tmm) {
    if (tmm == nullptr) {
        throw std::invalid_argument(
            "SolverRegistry requires a ThermalMathematicalModel");
    }
    return tmm;
}

}  // namespace

SteadySolvers::SteadySolvers(std::shared_ptr<ThermalMathematicalModel> tmm)
    : _tmm(std::move(tmm)) {}

SteadySolvers::~SteadySolvers() = default;

steady::SteadyState& SteadySolvers::steady_state() {
    if (_steady_state == nullptr) {
        _steady_state = std::make_unique<steady::SteadyState>(_tmm);
    }
    return *_steady_state;
}

TransientSolvers::TransientSolvers(
    std::shared_ptr<ThermalMathematicalModel> tmm)
    : _tmm(std::move(tmm)) {}

TransientSolvers::~TransientSolvers() = default;

transient::CrankNicolson& TransientSolvers::crank_nicolson() {
    if (_crank_nicolson == nullptr) {
        _crank_nicolson = std::make_unique<transient::CrankNicolson>(_tmm);
    }
    return *_crank_nicolson;
}

SensitivitySolvers::SensitivitySolvers(
    std::shared_ptr<ThermalMathematicalModel> tmm)
    : _tmm(std::move(tmm)) {}

SensitivitySolvers::~SensitivitySolvers() = default;

sensitivity::JacobianPropagation& SensitivitySolvers::jacobian_propagation() {
    if (_jacobian_propagation == nullptr) {
        _jacobian_propagation =
            std::make_unique<sensitivity::JacobianPropagation>(_tmm);
    }
    return *_jacobian_propagation;
}

SolverRegistry::SolverRegistry(std::shared_ptr<ThermalMathematicalModel> tmm)
    : _tmm(require_model(std::move(tmm))),
      _steady(_tmm),
      _transient(_tmm),
      _sensitivity(_tmm) {}

SolverRegistry::~SolverRegistry() = default;

}  // namespace pycanha::solvers
