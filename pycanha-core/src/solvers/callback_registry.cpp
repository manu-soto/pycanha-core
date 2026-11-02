#include "pycanha-core/solvers/callback_registry.hpp"

#include <stdexcept>

#include "pycanha-core/solvers/solver.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/tmm/thermalmodel.hpp"

namespace pycanha {

CallbackContext::CallbackContext(ThermalModel& tm,
                                 solvers::Solver* solver) noexcept
    : _tm(&tm), _solver(solver) {}

ThermalModel& CallbackContext::tm() noexcept { return *_tm; }

const ThermalModel& CallbackContext::tm() const noexcept { return *_tm; }

ThermalMathematicalModel& CallbackContext::tmm() noexcept { return _tm->tmm(); }

const ThermalMathematicalModel& CallbackContext::tmm() const noexcept {
    return _tm->tmm();
}

solvers::Solver& CallbackContext::solver() {
    if (_solver == nullptr) {
        throw std::runtime_error("CallbackContext has no active solver");
    }

    return *_solver;
}

const solvers::Solver& CallbackContext::solver() const {
    if (_solver == nullptr) {
        throw std::runtime_error("CallbackContext has no active solver");
    }

    return *_solver;
}

double CallbackContext::time() const noexcept { return _tm->tmm().time; }

CallbackRegistry::CallbackRegistry(ThermalModel& tm) : _tm(&tm) {
    auto& tmm = tm.tmm();
    tmm.python_callbacks_active = true;
    tmm.python_extern_callback_solver_loop = [this]() { invoke_solver_loop(); };
    tmm.python_extern_callback_transient_time_change = [this]() {
        invoke_time_change();
    };
    tmm.python_extern_callback_transient_after_timestep = [this]() {
        invoke_after_timestep();
    };
}

void CallbackRegistry::invoke_solver_loop() {
    if (!active || !solver_loop) {
        return;
    }

    auto context = make_context();
    solver_loop(context);
}

void CallbackRegistry::invoke_time_change() {
    if (!active || !time_change) {
        return;
    }

    auto context = make_context();
    time_change(context);
}

void CallbackRegistry::invoke_after_timestep() {
    if (!active || !after_timestep) {
        return;
    }

    auto context = make_context();
    after_timestep(context);
}

CallbackContext CallbackRegistry::make_context() const {
    if (_tm == nullptr) {
        throw std::runtime_error(
            "CallbackRegistry is not associated to a model");
    }

    return {*_tm, _tm->tmm().current_callback_solver()};
}

}  // namespace pycanha
