#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"

#include <spdlog/spdlog.h>

#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/parameters/entity.hpp"
#include "pycanha-core/parameters/formulas.hpp"
#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/parameters/variable.hpp"
#include "pycanha-core/solvers/solver_registry.hpp"
#include "pycanha-core/thermaldata/lookup_table.hpp"
#include "pycanha-core/thermaldata/thermaldata.hpp"
#include "pycanha-core/tmm/bulk.hpp"
#include "pycanha-core/tmm/conductivecouplings.hpp"
#include "pycanha-core/tmm/coupling.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/radiativecouplings.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"
#include "pycanha-core/utils/logger.hpp"

namespace pycanha {

namespace {

[[nodiscard]] std::shared_ptr<ThermalNetwork> require_network(
    std::shared_ptr<ThermalNetwork> network) {
    if (network == nullptr) {
        throw std::invalid_argument(
            "ThermalMathematicalModel requires a ThermalNetwork");
    }

    return network;
}

[[nodiscard]] std::shared_ptr<Parameters> require_parameters(
    std::shared_ptr<Parameters> parameters) {
    if (parameters == nullptr) {
        throw std::invalid_argument(
            "ThermalMathematicalModel requires Parameters");
    }

    return parameters;
}

[[nodiscard]] std::shared_ptr<Formulas> require_formulas(
    std::shared_ptr<Formulas> formulas) {
    if (formulas == nullptr) {
        throw std::invalid_argument(
            "ThermalMathematicalModel requires Formulas");
    }

    return formulas;
}

[[nodiscard]] std::shared_ptr<ThermalData> require_thermal_data(
    std::shared_ptr<ThermalData> thermal_data) {
    if (thermal_data == nullptr) {
        throw std::invalid_argument(
            "ThermalMathematicalModel requires ThermalData");
    }

    return thermal_data;
}

}  // namespace

ThermalMathematicalModel::ThermalMathematicalModel(std::string model_name)
    : _network(std::make_shared<ThermalNetwork>()),
      _parameters_shptr(std::make_shared<Parameters>()),
      _formulas_shptr(std::make_shared<Formulas>()),
      _thermal_data_shptr(std::make_shared<ThermalData>()),
      name(std::move(model_name)) {
    associate_resources();
    initialize_internal_time_parameter();
    const auto logger = get_logger();

    SPDLOG_LOGGER_TRACE(logger,
                        "ThermalMathematicalModel: default constructor");
}

ThermalMathematicalModel::ThermalMathematicalModel(
    std::string model_name, std::shared_ptr<ThermalNetwork> network,
    std::shared_ptr<Parameters> parameters, std::shared_ptr<Formulas> formulas,
    std::shared_ptr<ThermalData> thermal_data)
    : _network(require_network(std::move(network))),
      _parameters_shptr(require_parameters(std::move(parameters))),
      _formulas_shptr(require_formulas(std::move(formulas))),
      _thermal_data_shptr(require_thermal_data(std::move(thermal_data))),
      name(std::move(model_name)) {
    associate_resources();
    initialize_internal_time_parameter();
    const auto logger = get_logger();

    SPDLOG_LOGGER_TRACE(
        logger, "ThermalMathematicalModel: constructor with custom resources");
}

ThermalMathematicalModel::~ThermalMathematicalModel() {
    const auto logger = get_logger();
    SPDLOG_LOGGER_TRACE(logger, "ThermalMathematicalModel: destructor");
}

ThermalNetwork& ThermalMathematicalModel::network() noexcept {
    return *_network;
}

const ThermalNetwork& ThermalMathematicalModel::network() const noexcept {
    return *_network;
}

std::shared_ptr<ThermalNetwork>
ThermalMathematicalModel::network_ptr() noexcept {
    return _network;
}

std::shared_ptr<const ThermalNetwork> ThermalMathematicalModel::network_ptr()
    const noexcept {
    return _network;
}

Nodes& ThermalMathematicalModel::nodes() noexcept { return _network->nodes(); }

const Nodes& ThermalMathematicalModel::nodes() const noexcept {
    return _network->nodes();
}

std::shared_ptr<Nodes> ThermalMathematicalModel::nodes_ptr() noexcept {
    return _network->nodes_ptr();
}

std::shared_ptr<const Nodes> ThermalMathematicalModel::nodes_ptr()
    const noexcept {
    return _network->nodes_ptr();
}

std::shared_ptr<Parameters>
ThermalMathematicalModel::parameters_ptr() noexcept {
    return _parameters_shptr;
}

std::shared_ptr<const Parameters> ThermalMathematicalModel::parameters_ptr()
    const noexcept {
    return _parameters_shptr;
}

std::shared_ptr<Formulas> ThermalMathematicalModel::formulas_ptr() noexcept {
    return _formulas_shptr;
}

std::shared_ptr<const Formulas> ThermalMathematicalModel::formulas_ptr()
    const noexcept {
    return _formulas_shptr;
}

std::shared_ptr<ThermalData>
ThermalMathematicalModel::thermal_data_ptr() noexcept {
    return _thermal_data_shptr;
}

std::shared_ptr<const ThermalData> ThermalMathematicalModel::thermal_data_ptr()
    const noexcept {
    return _thermal_data_shptr;
}

Parameters& ThermalMathematicalModel::parameters() noexcept {
    return *_parameters_shptr;
}

const Parameters& ThermalMathematicalModel::parameters() const noexcept {
    return *_parameters_shptr;
}

Formulas& ThermalMathematicalModel::formulas() noexcept {
    return *_formulas_shptr;
}

const Formulas& ThermalMathematicalModel::formulas() const noexcept {
    return *_formulas_shptr;
}

ThermalData& ThermalMathematicalModel::thermal_data() noexcept {
    return *_thermal_data_shptr;
}

const ThermalData& ThermalMathematicalModel::thermal_data() const noexcept {
    return *_thermal_data_shptr;
}

StructureVersions ThermalMathematicalModel::structure_versions()
    const noexcept {
    return _network->structure_versions();
}

ConductiveCouplings& ThermalMathematicalModel::conductive_couplings() noexcept {
    return _network->conductive_couplings();
}

const ConductiveCouplings& ThermalMathematicalModel::conductive_couplings()
    const noexcept {
    return _network->conductive_couplings();
}

RadiativeCouplings& ThermalMathematicalModel::radiative_couplings() noexcept {
    return _network->radiative_couplings();
}

const RadiativeCouplings& ThermalMathematicalModel::radiative_couplings()
    const noexcept {
    return _network->radiative_couplings();
}

void ThermalMathematicalModel::add_node(Node node) { _network->add_node(node); }

void ThermalMathematicalModel::add_node(Index node_num) {
    Node node(static_cast<int>(node_num));
    _network->add_node(node);
}

BulkReport ThermalMathematicalModel::add_nodes(const NodeBatch& batch) {
    return _network->add_nodes(batch);
}

BulkReport ThermalMathematicalModel::add_conductive_couplings(
    std::span<const NodeNum> node_nums_1, std::span<const NodeNum> node_nums_2,
    std::span<const double> values, CouplingMerge merge) {
    return conductive_couplings().add_couplings(node_nums_1, node_nums_2,
                                                values, merge);
}

BulkReport ThermalMathematicalModel::add_radiative_couplings(
    std::span<const NodeNum> node_nums_1, std::span<const NodeNum> node_nums_2,
    std::span<const double> values, CouplingMerge merge) {
    return radiative_couplings().add_couplings(node_nums_1, node_nums_2, values,
                                               merge);
}

void ThermalMathematicalModel::add_conductive_coupling(Index node_num_1,
                                                       Index node_num_2,
                                                       double value) {
    conductive_couplings().add_coupling(node_num_1, node_num_2, value);
}

void ThermalMathematicalModel::add_radiative_coupling(Index node_num_1,
                                                      Index node_num_2,
                                                      double value) {
    radiative_couplings().add_coupling(node_num_1, node_num_2, value);
}

void ThermalMathematicalModel::add_conductive_coupling(Coupling coupling) {
    conductive_couplings().add_coupling(coupling);
}

void ThermalMathematicalModel::add_radiative_coupling(Coupling coupling) {
    radiative_couplings().add_coupling(coupling);
}

double ThermalMathematicalModel::flow_conductive(Index node_num_1,
                                                 Index node_num_2) {
    return network().flow_conductive(node_num_1, node_num_2);
}

double ThermalMathematicalModel::flow_conductive(
    const std::vector<Index>& node_nums_1,
    const std::vector<Index>& node_nums_2) {
    return network().flow_conductive(node_nums_1, node_nums_2);
}

double ThermalMathematicalModel::flow_radiative(Index node_num_1,
                                                Index node_num_2) {
    return network().flow_radiative(node_num_1, node_num_2);
}

double ThermalMathematicalModel::flow_radiative(
    const std::vector<Index>& node_nums_1,
    const std::vector<Index>& node_nums_2) {
    return network().flow_radiative(node_nums_1, node_nums_2);
}

void ThermalMathematicalModel::add_time_variable(const std::string& var_name,
                                                 Eigen::VectorXd x_data,
                                                 Eigen::VectorXd y_data,
                                                 InterpolationMethod interp,
                                                 ExtrapolationMethod extrap) {
    if (_temperature_variable_names.contains(var_name)) {
        throw std::invalid_argument("Variable name '" + var_name +
                                    "' is already used by a "
                                    "TemperatureVariable");
    }

    const auto [iterator, inserted] = _time_variables.try_emplace(
        var_name, var_name,
        LookupTable1D(std::move(x_data), std::move(y_data), interp, extrap),
        parameters(), std::addressof(time));
    if (!inserted) {
        throw std::invalid_argument("TimeVariable '" + var_name +
                                    "' already exists");
    }

    (void)iterator;
}

void ThermalMathematicalModel::remove_time_variable(
    const std::string& var_name) {
    _time_variables.erase(var_name);
}

bool ThermalMathematicalModel::has_time_variable(
    const std::string& var_name) const noexcept {
    return _time_variables.contains(var_name);
}

const TimeVariable& ThermalMathematicalModel::get_time_variable(
    const std::string& var_name) const {
    const auto iterator = _time_variables.find(var_name);
    if (iterator == _time_variables.end()) {
        throw std::out_of_range("TimeVariable '" + var_name +
                                "' does not exist");
    }

    return iterator->second;
}

void ThermalMathematicalModel::add_temperature_variable(
    const std::string& var_name, Eigen::VectorXd x_data, Eigen::VectorXd y_data,
    InterpolationMethod interp, ExtrapolationMethod extrap) {
    if (parameters().contains(var_name) || _time_variables.contains(var_name)) {
        throw std::invalid_argument("Variable name '" + var_name +
                                    "' is already in use");
    }

    const auto [iterator, inserted] = _temperature_variables.try_emplace(
        var_name, var_name,
        LookupTable1D(std::move(x_data), std::move(y_data), interp, extrap));
    if (!inserted) {
        throw std::invalid_argument("TemperatureVariable '" + var_name +
                                    "' already exists");
    }

    _temperature_variable_names.insert(iterator->first);
}

void ThermalMathematicalModel::remove_temperature_variable(
    const std::string& var_name) {
    _temperature_variables.erase(var_name);
    _temperature_variable_names.erase(var_name);
}

bool ThermalMathematicalModel::has_temperature_variable(
    const std::string& var_name) const noexcept {
    return _temperature_variables.contains(var_name);
}

const TemperatureVariable& ThermalMathematicalModel::get_temperature_variable(
    const std::string& var_name) const {
    const auto iterator = _temperature_variables.find(var_name);
    if (iterator == _temperature_variables.end()) {
        throw std::out_of_range("TemperatureVariable '" + var_name +
                                "' does not exist");
    }

    return iterator->second;
}

void ThermalMathematicalModel::callback_solver_loop() {
    const auto logger = get_logger();
    SPDLOG_LOGGER_DEBUG(logger,
                        "ThermalMathematicalModel: callback_solver_loop");

    if (!callbacks_active) {
        return;
    }

    if (internal_callbacks_active) {
        internal_callback_solver_loop();
    }
    if (c_callbacks_active) {
        c_extern_callback_solver_loop(this);
    }
    if (python_callbacks_active) {
        python_extern_callback_solver_loop();
    }
}

void ThermalMathematicalModel::callback_transient_time_change() {
    const auto logger = get_logger();
    SPDLOG_LOGGER_DEBUG(
        logger, "ThermalMathematicalModel: callback_transient_time_change");

    if (!callbacks_active) {
        return;
    }

    if (internal_callbacks_active) {
        internal_callback_transient_time_change();
    }
    if (c_callbacks_active) {
        c_extern_callback_transient_time_change(this);
    }
    if (python_callbacks_active) {
        python_extern_callback_transient_time_change();
    }
}

void ThermalMathematicalModel::callback_transient_after_timestep() {
    const auto logger = get_logger();
    SPDLOG_LOGGER_DEBUG(
        logger, "ThermalMathematicalModel: callback_transient_after_timestep");

    if (!callbacks_active) {
        return;
    }

    if (internal_callbacks_active) {
        internal_callback_transient_after_timestep();
    }
    if (c_callbacks_active) {
        c_extern_callback_transient_after_timestep(this);
    }
    if (python_callbacks_active) {
        python_extern_callback_transient_after_timestep();
    }
}

void ThermalMathematicalModel::associate_solvers(
    solvers::SolverRegistry& solvers) noexcept {
    _solvers = std::addressof(solvers);
}

void ThermalMathematicalModel::set_current_callback_solver(
    solvers::Solver* solver) noexcept {
    _current_callback_solver = solver;
}

solvers::Solver* ThermalMathematicalModel::current_callback_solver()
    const noexcept {
    return _current_callback_solver;
}

solvers::SolverRegistry& ThermalMathematicalModel::solvers() {
    if (_solvers == nullptr) {
        throw std::runtime_error(
            "ThermalMathematicalModel is not associated with a SolverRegistry");
    }

    return *_solvers;
}

const solvers::SolverRegistry& ThermalMathematicalModel::solvers() const {
    if (_solvers == nullptr) {
        throw std::runtime_error(
            "ThermalMathematicalModel is not associated with a SolverRegistry");
    }

    return *_solvers;
}

std::optional<Entity> ThermalMathematicalModel::find_entity(
    std::string_view text) const {
    return Entity::from_string(*_network, text);
}

Entity ThermalMathematicalModel::entity(std::string_view text) const {
    const auto resolved = find_entity(text);
    if (!resolved.has_value()) {
        throw std::invalid_argument("Unknown thermal entity '" +
                                    std::string(text) + "'");
    }

    return *resolved;
}

EntitiesHelper& ThermalMathematicalModel::entities() noexcept {
    return _entities;
}

const EntitiesHelper& ThermalMathematicalModel::entities() const noexcept {
    return _entities;
}

void ThermalMathematicalModel::internal_callback_common() {
    if (_time_parameter_ptr != nullptr) {
        *_time_parameter_ptr = time;
    }

    for (auto& [var_name, variable] : _time_variables) {
        (void)var_name;
        variable.update();
    }

    if (parameters().is_structure_locked() &&
        formulas().is_compiled_current()) {
        formulas().apply_compiled_formulas();
    } else {
        formulas().apply_formulas();
    }

    if (python_formulas_active) {
        python_apply_formulas();
    }
}

void ThermalMathematicalModel::internal_callback_solver_loop() {
    internal_callback_common();
}

void ThermalMathematicalModel::internal_callback_transient_time_change() {
    const auto logger = get_logger();
    SPDLOG_LOGGER_DEBUG(logger,
                        "ThermalMathematicalModel: internal transient change");

    internal_callback_common();
}

void ThermalMathematicalModel::internal_callback_transient_after_timestep() {
    const auto logger = get_logger();
    SPDLOG_LOGGER_DEBUG(logger,
                        "ThermalMathematicalModel: internal after timestep");

    internal_callback_common();
}

void ThermalMathematicalModel::associate_resources() {
    _formulas_shptr->associate(_network, _parameters_shptr);
    _formulas_shptr->set_temperature_variable_names(
        std::addressof(_temperature_variable_names));
    _thermal_data_shptr->associate(_network);
}

void ThermalMathematicalModel::initialize_internal_time_parameter() {
    if (!parameters().contains("time")) {
        parameters().add_internal_parameter("time", 0.0);
    }

    if (!parameters().is_internal_parameter("time")) {
        SPDLOG_LOGGER_WARN(get_logger(),
                           "Parameter 'time' already exists and is not an "
                           "internal parameter; runtime time synchronization "
                           "is disabled");
        _time_parameter_ptr = nullptr;
        return;
    }

    const auto time_idx = parameters().get_idx("time");
    if (!time_idx.has_value()) {
        _time_parameter_ptr = nullptr;
        return;
    }

    _time_parameter_ptr = parameters().get_double_ptr(*time_idx);
}

Entity EntitiesHelper::attribute(std::string_view token,
                                 NodeNum node_num) const {
    return _tmm->entity(std::string(token) + std::to_string(node_num));
}

Entity EntitiesHelper::temperature(NodeNum node_num) const {
    return attribute("T", node_num);
}

Entity EntitiesHelper::capacity(NodeNum node_num) const {
    return attribute("C", node_num);
}

Entity EntitiesHelper::solar_heat(NodeNum node_num) const {
    return attribute("QS", node_num);
}

Entity EntitiesHelper::albedo_heat(NodeNum node_num) const {
    return attribute("QA", node_num);
}

Entity EntitiesHelper::earth_ir(NodeNum node_num) const {
    return attribute("QE", node_num);
}

Entity EntitiesHelper::internal_heat(NodeNum node_num) const {
    return attribute("QI", node_num);
}

Entity EntitiesHelper::other_heat(NodeNum node_num) const {
    return attribute("QR", node_num);
}

Entity EntitiesHelper::conductive_coupling(NodeNum node_num_1,
                                           NodeNum node_num_2) const {
    return _tmm->entity("GL(" + std::to_string(node_num_1) + "," +
                        std::to_string(node_num_2) + ")");
}

Entity EntitiesHelper::radiative_coupling(NodeNum node_num_1,
                                          NodeNum node_num_2) const {
    return _tmm->entity("GR(" + std::to_string(node_num_1) + "," +
                        std::to_string(node_num_2) + ")");
}

}  // namespace pycanha
