#include "pycanha-core/tmm/thermalmodel.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "pycanha-core/conduction/builder.hpp"
#include "pycanha-core/conduction/options.hpp"
#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/parameters/formulas.hpp"
#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/solvers/callback_registry.hpp"
#include "pycanha-core/solvers/solver_registry.hpp"
#include "pycanha-core/thermaldata/thermaldata.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"

namespace pycanha {

namespace {

[[nodiscard]] std::shared_ptr<ThermalMathematicalModel> require_tmm(
    std::shared_ptr<ThermalMathematicalModel> tmm) {
    if (tmm == nullptr) {
        throw std::invalid_argument(
            "ThermalModel requires a ThermalMathematicalModel");
    }

    return tmm;
}

[[nodiscard]] std::shared_ptr<gmm::GeometryModel> require_gmm(
    std::shared_ptr<gmm::GeometryModel> gmm) {
    if (gmm == nullptr) {
        throw std::invalid_argument("ThermalModel requires a GeometryModel");
    }

    return gmm;
}

}  // namespace

ThermalModel::ThermalModel(const std::string& model_name)
    : _name(model_name),
      _tmm(std::make_shared<ThermalMathematicalModel>(model_name)),
      _gmm(std::make_shared<gmm::GeometryModel>(model_name)),
      _parameters(_tmm->parameters_ptr()),
      _formulas(_tmm->formulas_ptr()),
      _thermal_data(_tmm->thermal_data_ptr()),
      _solvers(std::make_shared<solvers::SolverRegistry>(_tmm)) {
    _tmm->associate_solvers(*_solvers);
    _callbacks = std::make_shared<CallbackRegistry>(*this);
}

ThermalModel::ThermalModel(std::string model_name,
                           std::shared_ptr<ThermalMathematicalModel> tmm,
                           std::shared_ptr<gmm::GeometryModel> gmm)
    : _name(std::move(model_name)),
      _tmm(require_tmm(std::move(tmm))),
      _gmm(require_gmm(std::move(gmm))),
      _parameters(_tmm->parameters_ptr()),
      _formulas(_tmm->formulas_ptr()),
      _thermal_data(_tmm->thermal_data_ptr()),
      _solvers(std::make_shared<solvers::SolverRegistry>(_tmm)) {
    // TODO: If GMM later depends on shared model resources, validate here
    // that the injected GMM and TMM reference the same instances.
    _tmm->associate_solvers(*_solvers);
    _callbacks = std::make_shared<CallbackRegistry>(*this);
}

ThermalMathematicalModel& ThermalModel::tmm() noexcept { return *_tmm; }

const ThermalMathematicalModel& ThermalModel::tmm() const noexcept {
    return *_tmm;
}

gmm::GeometryModel& ThermalModel::gmm() noexcept { return *_gmm; }

const gmm::GeometryModel& ThermalModel::gmm() const noexcept { return *_gmm; }

Parameters& ThermalModel::parameters() noexcept { return *_parameters; }

const Parameters& ThermalModel::parameters() const noexcept {
    return *_parameters;
}

Formulas& ThermalModel::formulas() noexcept { return *_formulas; }

const Formulas& ThermalModel::formulas() const noexcept { return *_formulas; }

ThermalData& ThermalModel::thermal_data() noexcept { return *_thermal_data; }

const ThermalData& ThermalModel::thermal_data() const noexcept {
    return *_thermal_data;
}

solvers::SolverRegistry& ThermalModel::solvers() noexcept { return *_solvers; }

const solvers::SolverRegistry& ThermalModel::solvers() const noexcept {
    return *_solvers;
}

CallbackRegistry& ThermalModel::callbacks() noexcept { return *_callbacks; }

const CallbackRegistry& ThermalModel::callbacks() const noexcept {
    return *_callbacks;
}

conduction::TmmBuildReport ThermalModel::build_tmm_from_gmm(
    const conduction::TmmBuildOptions& options) {
    return conduction::build_tmm_from_gmm(*this, options);
}

}  // namespace pycanha
