#pragma once

#include <functional>

namespace pycanha::solvers {
class Solver;
}  // namespace pycanha::solvers

namespace pycanha {

class ThermalMathematicalModel;
class ThermalModel;

class CallbackContext {
  public:
    CallbackContext(ThermalModel& tm, solvers::Solver* solver) noexcept;

    [[nodiscard]] ThermalModel& tm() noexcept;
    [[nodiscard]] const ThermalModel& tm() const noexcept;
    [[nodiscard]] ThermalMathematicalModel& tmm() noexcept;
    [[nodiscard]] const ThermalMathematicalModel& tmm() const noexcept;
    [[nodiscard]] solvers::Solver& solver();
    [[nodiscard]] const solvers::Solver& solver() const;
    [[nodiscard]] double time() const noexcept;

  private:
    ThermalModel* _tm{nullptr};
    solvers::Solver* _solver{nullptr};
};

class CallbackRegistry {
  public:
    explicit CallbackRegistry(ThermalModel& tm);

    bool active{true};
    std::function<void(CallbackContext&)> solver_loop;
    std::function<void(CallbackContext&)> time_change;
    std::function<void(CallbackContext&)> after_timestep;

    void invoke_solver_loop();
    void invoke_time_change();
    void invoke_after_timestep();

  private:
    [[nodiscard]] CallbackContext make_context() const;

    ThermalModel* _tm{nullptr};
};

}  // namespace pycanha
