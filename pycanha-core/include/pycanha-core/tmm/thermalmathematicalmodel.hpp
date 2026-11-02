#pragma once

#include <Eigen/Core>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/parameters/variable.hpp"
#include "pycanha-core/tmm/bulk.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"

namespace pycanha::solvers {
class Solver;
class SolverRegistry;
}  // namespace pycanha::solvers

namespace pycanha {

class Coupling;
class ConductiveCouplings;
class Entity;
class Formulas;
class Node;
class Nodes;
class Parameters;
class RadiativeCouplings;
class ThermalData;
class ThermalNetwork;
class ThermalMathematicalModel;

class EntitiesHelper {
  public:
    explicit EntitiesHelper(ThermalMathematicalModel& tmm) noexcept
        : _tmm(&tmm) {}

    [[nodiscard]] Entity attribute(std::string_view token,
                                   NodeNum node_num) const;
    [[nodiscard]] Entity temperature(NodeNum node_num) const;
    [[nodiscard]] Entity capacity(NodeNum node_num) const;
    [[nodiscard]] Entity solar_heat(NodeNum node_num) const;
    [[nodiscard]] Entity albedo_heat(NodeNum node_num) const;
    [[nodiscard]] Entity earth_ir(NodeNum node_num) const;
    [[nodiscard]] Entity internal_heat(NodeNum node_num) const;
    [[nodiscard]] Entity other_heat(NodeNum node_num) const;
    [[nodiscard]] Entity conductive_coupling(NodeNum node_num_1,
                                             NodeNum node_num_2) const;
    [[nodiscard]] Entity radiative_coupling(NodeNum node_num_1,
                                            NodeNum node_num_2) const;

  private:
    ThermalMathematicalModel* _tmm;
};

class ThermalMathematicalModel {
  public:
    ThermalMathematicalModel(const ThermalMathematicalModel&) = delete;
    ThermalMathematicalModel& operator=(const ThermalMathematicalModel&) =
        delete;
    ThermalMathematicalModel(ThermalMathematicalModel&&) noexcept = delete;
    ThermalMathematicalModel& operator=(ThermalMathematicalModel&&) noexcept =
        delete;

    explicit ThermalMathematicalModel(std::string model_name);
    ThermalMathematicalModel(std::string model_name,
                             std::shared_ptr<ThermalNetwork> network,
                             std::shared_ptr<Parameters> parameters,
                             std::shared_ptr<Formulas> formulas,
                             std::shared_ptr<ThermalData> thermal_data);
    ~ThermalMathematicalModel();

    [[nodiscard]] ThermalNetwork& network() noexcept;
    [[nodiscard]] const ThermalNetwork& network() const noexcept;
    [[nodiscard]] std::shared_ptr<ThermalNetwork> network_ptr() noexcept;
    [[nodiscard]] std::shared_ptr<const ThermalNetwork> network_ptr()
        const noexcept;

    [[nodiscard]] Nodes& nodes() noexcept;
    [[nodiscard]] const Nodes& nodes() const noexcept;
    [[nodiscard]] std::shared_ptr<Nodes> nodes_ptr() noexcept;
    [[nodiscard]] std::shared_ptr<const Nodes> nodes_ptr() const noexcept;

    [[nodiscard]] std::shared_ptr<Parameters> parameters_ptr() noexcept;
    [[nodiscard]] std::shared_ptr<const Parameters> parameters_ptr()
        const noexcept;
    [[nodiscard]] std::shared_ptr<Formulas> formulas_ptr() noexcept;
    [[nodiscard]] std::shared_ptr<const Formulas> formulas_ptr() const noexcept;
    [[nodiscard]] std::shared_ptr<ThermalData> thermal_data_ptr() noexcept;
    [[nodiscard]] std::shared_ptr<const ThermalData> thermal_data_ptr()
        const noexcept;

    [[nodiscard]] Parameters& parameters() noexcept;
    [[nodiscard]] const Parameters& parameters() const noexcept;
    [[nodiscard]] Formulas& formulas() noexcept;
    [[nodiscard]] const Formulas& formulas() const noexcept;
    [[nodiscard]] ThermalData& thermal_data() noexcept;
    [[nodiscard]] const ThermalData& thermal_data() const noexcept;

    [[nodiscard]] ConductiveCouplings& conductive_couplings() noexcept;
    [[nodiscard]] const ConductiveCouplings& conductive_couplings()
        const noexcept;

    [[nodiscard]] RadiativeCouplings& radiative_couplings() noexcept;
    [[nodiscard]] const RadiativeCouplings& radiative_couplings()
        const noexcept;

    /// Structure versions of the nodes and of the conductive and radiative
    /// couplings, see ThermalNetwork::structure_versions. Solvers compare
    /// them with the snapshot taken by initialize().
    [[nodiscard]] StructureVersions structure_versions() const noexcept;

    void add_node(Node node);
    void add_node(Index node_num);
    /// Bulk node insertion, see Nodes::add_nodes.
    BulkReport add_nodes(const NodeBatch& batch);
    /// Bulk coupling insertion by node number, see Couplings::add_couplings.
    BulkReport add_conductive_couplings(
        std::span<const NodeNum> node_nums_1,
        std::span<const NodeNum> node_nums_2, std::span<const double> values,
        CouplingMerge merge = CouplingMerge::OVERWRITE);
    BulkReport add_radiative_couplings(
        std::span<const NodeNum> node_nums_1,
        std::span<const NodeNum> node_nums_2, std::span<const double> values,
        CouplingMerge merge = CouplingMerge::OVERWRITE);
    void add_conductive_coupling(Index node_num_1, Index node_num_2,
                                 double value);
    void add_radiative_coupling(Index node_num_1, Index node_num_2,
                                double value);
    void add_conductive_coupling(Coupling coupling);
    void add_radiative_coupling(Coupling coupling);
    [[nodiscard]] double flow_conductive(Index node_num_1, Index node_num_2);
    [[nodiscard]] double flow_conductive(const std::vector<Index>& node_nums_1,
                                         const std::vector<Index>& node_nums_2);
    [[nodiscard]] double flow_radiative(Index node_num_1, Index node_num_2);
    [[nodiscard]] double flow_radiative(const std::vector<Index>& node_nums_1,
                                        const std::vector<Index>& node_nums_2);
    void add_time_variable(
        const std::string& var_name, Eigen::VectorXd x_data,
        Eigen::VectorXd y_data,
        InterpolationMethod interp = InterpolationMethod::Linear,
        ExtrapolationMethod extrap = ExtrapolationMethod::Constant);
    void remove_time_variable(const std::string& var_name);
    [[nodiscard]] bool has_time_variable(
        const std::string& var_name) const noexcept;
    [[nodiscard]] const TimeVariable& get_time_variable(
        const std::string& var_name) const;
    void add_temperature_variable(
        const std::string& var_name, Eigen::VectorXd x_data,
        Eigen::VectorXd y_data,
        InterpolationMethod interp = InterpolationMethod::Linear,
        ExtrapolationMethod extrap = ExtrapolationMethod::Constant);
    void remove_temperature_variable(const std::string& var_name);
    [[nodiscard]] bool has_temperature_variable(
        const std::string& var_name) const noexcept;
    [[nodiscard]] const TemperatureVariable& get_temperature_variable(
        const std::string& var_name) const;

    bool callbacks_active = true;
    bool internal_callbacks_active = true;
    bool c_callbacks_active = false;
    bool python_callbacks_active = false;
    bool python_formulas_active = true;

    void callback_solver_loop();
    void callback_transient_time_change();
    void callback_transient_after_timestep();

    void associate_solvers(solvers::SolverRegistry& solvers) noexcept;
    [[nodiscard]] solvers::SolverRegistry& solvers();
    [[nodiscard]] const solvers::SolverRegistry& solvers() const;
    void set_current_callback_solver(solvers::Solver* solver) noexcept;
    [[nodiscard]] solvers::Solver* current_callback_solver() const noexcept;
    [[nodiscard]] std::optional<Entity> find_entity(
        std::string_view text) const;
    [[nodiscard]] Entity entity(std::string_view text) const;
    [[nodiscard]] EntitiesHelper& entities() noexcept;
    [[nodiscard]] const EntitiesHelper& entities() const noexcept;

    void internal_callback_solver_loop();
    void internal_callback_transient_time_change();
    void internal_callback_transient_after_timestep();
    std::function<void()> python_apply_formulas = []() {};

    void (*c_extern_callback_solver_loop)(ThermalMathematicalModel*) =
        [](ThermalMathematicalModel*) {};
    void (*c_extern_callback_transient_time_change)(ThermalMathematicalModel*) =
        [](ThermalMathematicalModel*) {};
    void (*c_extern_callback_transient_after_timestep)(
        ThermalMathematicalModel*) = [](ThermalMathematicalModel*) {};

    std::function<void()> python_extern_callback_solver_loop = []() {};
    std::function<void()> python_extern_callback_transient_time_change = []() {
    };
    std::function<void()> python_extern_callback_transient_after_timestep =
        []() {};

  private:
    std::shared_ptr<ThermalNetwork> _network;
    std::shared_ptr<Parameters> _parameters_shptr;
    std::shared_ptr<Formulas> _formulas_shptr;
    std::shared_ptr<ThermalData> _thermal_data_shptr;
    solvers::SolverRegistry* _solvers{nullptr};
    solvers::Solver* _current_callback_solver{nullptr};
    EntitiesHelper _entities{*this};
    double* _time_parameter_ptr{nullptr};
    std::unordered_map<std::string, TimeVariable> _time_variables;
    std::unordered_map<std::string, TemperatureVariable> _temperature_variables;
    std::unordered_set<std::string> _temperature_variable_names;

    void associate_resources();
    void initialize_internal_time_parameter();
    inline void internal_callback_common();

  public:
    std::string name;
    double time{0.0};
};

}  // namespace pycanha
