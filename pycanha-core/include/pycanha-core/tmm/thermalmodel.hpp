#pragma once

#include <memory>
#include <string>

#include "pycanha-core/conduction/options.hpp"
#include "pycanha-core/gmm/geometrymodel.hpp"

namespace pycanha::solvers {
class SolverRegistry;
}  // namespace pycanha::solvers

namespace pycanha {

class Formulas;
class Parameters;
class CallbackRegistry;
class ThermalData;
class ThermalMathematicalModel;

class ThermalModel {
  public:
    explicit ThermalModel(const std::string& model_name);
    ThermalModel(std::string model_name,
                 std::shared_ptr<ThermalMathematicalModel> tmm,
                 std::shared_ptr<gmm::GeometryModel> gmm);
    ~ThermalModel() = default;

    ThermalModel(const ThermalModel&) = delete;
    ThermalModel& operator=(const ThermalModel&) = delete;
    ThermalModel(ThermalModel&&) noexcept = delete;
    ThermalModel& operator=(ThermalModel&&) noexcept = delete;

    [[nodiscard]] const std::string& name() const noexcept { return _name; }

    [[nodiscard]] ThermalMathematicalModel& tmm() noexcept;
    [[nodiscard]] const ThermalMathematicalModel& tmm() const noexcept;

    [[nodiscard]] gmm::GeometryModel& gmm() noexcept;
    [[nodiscard]] const gmm::GeometryModel& gmm() const noexcept;

    [[nodiscard]] Parameters& parameters() noexcept;
    [[nodiscard]] const Parameters& parameters() const noexcept;

    [[nodiscard]] Formulas& formulas() noexcept;
    [[nodiscard]] const Formulas& formulas() const noexcept;

    [[nodiscard]] ThermalData& thermal_data() noexcept;
    [[nodiscard]] const ThermalData& thermal_data() const noexcept;

    [[nodiscard]] solvers::SolverRegistry& solvers() noexcept;
    [[nodiscard]] const solvers::SolverRegistry& solvers() const noexcept;

    [[nodiscard]] CallbackRegistry& callbacks() noexcept;
    [[nodiscard]] const CallbackRegistry& callbacks() const noexcept;

    /// Populates the tmm from the gmm: one node per active face that
    /// carries a node number — active meaning it takes part in conduction,
    /// radiation or both — plus the in-plane and through-thickness conductors
    /// the conductively active ones imply. Requires an empty tmm; see
    /// pycanha::conduction::build_tmm_from_gmm.
    [[nodiscard]] conduction::TmmBuildReport build_tmm_from_gmm(
        const conduction::TmmBuildOptions& options = {});

  private:
    std::string _name;
    std::shared_ptr<ThermalMathematicalModel> _tmm;
    std::shared_ptr<gmm::GeometryModel> _gmm;
    std::shared_ptr<Parameters> _parameters;
    std::shared_ptr<Formulas> _formulas;
    std::shared_ptr<ThermalData> _thermal_data;
    std::shared_ptr<solvers::SolverRegistry> _solvers;
    std::shared_ptr<CallbackRegistry> _callbacks;
};

}  // namespace pycanha
