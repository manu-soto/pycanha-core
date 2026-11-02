#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/parameters/formulas.hpp"
#include "pycanha-core/parameters/parameters.hpp"
#include "pycanha-core/solvers/solver_registry.hpp"
#include "pycanha-core/thermaldata/thermaldata.hpp"
#include "pycanha-core/tmm/conductivecouplings.hpp"
#include "pycanha-core/tmm/coupling.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/radiativecouplings.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

namespace {

struct CallbackCountState {
    int solver_calls = 0;
    int time_calls = 0;
    int after_calls = 0;
};

CallbackCountState& c_callback_counts() {
    static CallbackCountState counts;
    return counts;
}

void reset_c_callback_counts() { c_callback_counts() = {}; }

void count_solver_callback(ThermalMathematicalModel* model) {
    static_cast<void>(model);
    ++c_callback_counts().solver_calls;
}

void count_time_callback(ThermalMathematicalModel* model) {
    static_cast<void>(model);
    ++c_callback_counts().time_calls;
}

void count_after_callback(ThermalMathematicalModel* model) {
    static_cast<void>(model);
    ++c_callback_counts().after_calls;
}

void require_callback_counts(const CallbackCountState& actual_counts,
                             const CallbackCountState& expected_counts) {
    REQUIRE(actual_counts.solver_calls == expected_counts.solver_calls);
    REQUIRE(actual_counts.time_calls == expected_counts.time_calls);
    REQUIRE(actual_counts.after_calls == expected_counts.after_calls);
}

}  // namespace

TEST_CASE("ThermalMathematicalModel composes a ThermalNetwork",
          "[thermalmathematicalmodel]") {
    ThermalMathematicalModel model("test-model");

    auto network = model.network_ptr();
    REQUIRE(static_cast<bool>(network));

    auto nodes_storage = model.nodes_ptr();
    REQUIRE(static_cast<bool>(nodes_storage));

    const bool shares_nodes_storage = nodes_storage == network->nodes_ptr();
    REQUIRE(shares_nodes_storage);

    const bool formulas_associated = model.formulas().network() == network;
    REQUIRE(formulas_associated);

    const bool thermal_data_associated =
        model.thermal_data().network_ptr() == network;
    REQUIRE(thermal_data_associated);

    Node diffusive(1);
    diffusive.set_T(275.0);
    model.add_node(diffusive);

    model.add_node(Index{2});  // Default diffusive node

    Node boundary(3);
    boundary.set_type('B');
    model.add_node(boundary);

    const auto& nodes = model.nodes();
    const bool has_two_diff_nodes = nodes.get_num_diff_nodes() == 2;
    REQUIRE(has_two_diff_nodes);

    const bool has_one_bound_node = nodes.get_num_bound_nodes() == 1;
    REQUIRE(has_one_bound_node);

    model.add_conductive_coupling(Index{1}, Index{2}, 10.0);
    const auto conductive_value =
        model.conductive_couplings().get_coupling_value(1, 2);
    REQUIRE_THAT(conductive_value, Catch::Matchers::WithinAbs(10.0, 1e-12));

    const Coupling coupling(Index{2}, Index{3}, 5.0);
    model.add_radiative_coupling(coupling);
    const auto radiative_value =
        model.radiative_couplings().get_coupling_value(2, 3);
    REQUIRE_THAT(radiative_value, Catch::Matchers::WithinAbs(5.0, 1e-12));
}

TEST_CASE("ThermalMathematicalModel rejects null injected resources",
          "[thermalmathematicalmodel]") {
    auto network = std::make_shared<ThermalNetwork>();
    auto parameters = std::make_shared<Parameters>();
    auto formulas = std::make_shared<Formulas>();
    auto thermal_data = std::make_shared<ThermalData>();

    REQUIRE_THROWS_AS(
        ThermalMathematicalModel("missing-network", nullptr, parameters,
                                 formulas, thermal_data),
        std::invalid_argument);
    REQUIRE_THROWS_AS(ThermalMathematicalModel("missing-parameters", network,
                                               nullptr, formulas, thermal_data),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(
        ThermalMathematicalModel("missing-formulas", network, parameters,
                                 nullptr, thermal_data),
        std::invalid_argument);
    REQUIRE_THROWS_AS(ThermalMathematicalModel("missing-data", network,
                                               parameters, formulas, nullptr),
                      std::invalid_argument);
}

TEST_CASE("ThermalMathematicalModel associates injected resources",
          "[thermalmathematicalmodel]") {
    auto network = std::make_shared<ThermalNetwork>();
    auto parameters = std::make_shared<Parameters>();
    auto formulas = std::make_shared<Formulas>();
    auto thermal_data = std::make_shared<ThermalData>();

    ThermalMathematicalModel model("custom", network, parameters, formulas,
                                   thermal_data);

    REQUIRE(model.network_ptr() == network);
    REQUIRE(model.parameters_ptr() == parameters);
    REQUIRE(model.formulas_ptr() == formulas);
    REQUIRE(model.thermal_data_ptr() == thermal_data);
    REQUIRE(model.formulas().network() == network);
    REQUIRE(model.thermal_data().network_ptr() == network);
    REQUIRE(model.parameters().contains("time"));
}

TEST_CASE("ThermalMathematicalModel keeps the reserved time parameter synced",
          "[thermalmathematicalmodel]") {
    auto network = std::make_shared<ThermalNetwork>();
    auto parameters = std::make_shared<Parameters>();
    auto formulas = std::make_shared<Formulas>();
    auto thermal_data = std::make_shared<ThermalData>();

    parameters->add_parameter("time", 5.0);
    ThermalMathematicalModel model("manual-time", network, parameters, formulas,
                                   thermal_data);

    REQUIRE(model.parameters().is_internal_parameter("time"));
    model.time = 9.0;
    model.callback_solver_loop();

    REQUIRE(std::get<double>(model.parameters().get_parameter("time")) ==
            Catch::Approx(9.0));
}

TEST_CASE("ThermalMathematicalModel variable helpers enforce conflicts",
          "[thermalmathematicalmodel]") {
    ThermalMathematicalModel model("variable-helpers");
    model.parameters().add_parameter("existing", 1.0);

    model.add_time_variable("load", (Eigen::Vector2d{} << 0.0, 1.0).finished(),
                            (Eigen::Vector2d{} << 10.0, 20.0).finished());
    REQUIRE(model.has_time_variable("load"));
    REQUIRE(model.get_time_variable("load").name() == "load");
    REQUIRE_THROWS_AS(model.get_time_variable("missing"), std::out_of_range);

    model.add_temperature_variable(
        "kappa", (Eigen::Vector2d{} << 200.0, 300.0).finished(),
        (Eigen::Vector2d{} << 1.0, 2.0).finished());
    REQUIRE(model.has_temperature_variable("kappa"));
    REQUIRE(model.get_temperature_variable("kappa").name() == "kappa");
    REQUIRE_THROWS_AS(model.get_temperature_variable("missing"),
                      std::out_of_range);

    REQUIRE_THROWS_AS(model.add_time_variable(
                          "kappa", (Eigen::Vector2d{} << 0.0, 1.0).finished(),
                          (Eigen::Vector2d{} << 1.0, 2.0).finished()),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(
        model.add_temperature_variable(
            "existing", (Eigen::Vector2d{} << 0.0, 1.0).finished(),
            (Eigen::Vector2d{} << 1.0, 2.0).finished()),
        std::invalid_argument);
    REQUIRE_THROWS_AS(model.add_temperature_variable(
                          "load", (Eigen::Vector2d{} << 0.0, 1.0).finished(),
                          (Eigen::Vector2d{} << 1.0, 2.0).finished()),
                      std::invalid_argument);

    model.remove_time_variable("load");
    model.remove_temperature_variable("kappa");
    REQUIRE_FALSE(model.has_time_variable("load"));
    REQUIRE_FALSE(model.has_temperature_variable("kappa"));
}

TEST_CASE("ThermalMathematicalModel rejects duplicate variable names",
          "[thermalmathematicalmodel]") {
    ThermalMathematicalModel model("duplicate-variables");

    model.add_time_variable("schedule",
                            (Eigen::Vector2d{} << 0.0, 1.0).finished(),
                            (Eigen::Vector2d{} << 10.0, 20.0).finished());
    REQUIRE_THROWS_AS(
        model.add_time_variable("schedule",
                                (Eigen::Vector2d{} << 0.0, 1.0).finished(),
                                (Eigen::Vector2d{} << 30.0, 40.0).finished()),
        std::invalid_argument);

    model.add_temperature_variable(
        "ambient", (Eigen::Vector2d{} << 250.0, 300.0).finished(),
        (Eigen::Vector2d{} << 1.0, 2.0).finished());
    REQUIRE_THROWS_AS(
        model.add_temperature_variable(
            "ambient", (Eigen::Vector2d{} << 260.0, 310.0).finished(),
            (Eigen::Vector2d{} << 2.0, 3.0).finished()),
        std::invalid_argument);
}

TEST_CASE(
    "ThermalMathematicalModel const accessors preserve the same model state",
    "[thermalmathematicalmodel]") {
    ThermalMathematicalModel model("const-accessors");

    Node node1(1);
    node1.set_T(300.0);
    node1.set_C(10.0);
    node1.set_qs(1.0);
    node1.set_qa(2.0);
    node1.set_qe(3.0);
    node1.set_qi(4.0);
    node1.set_qr(5.0);

    Node node2(2);
    node2.set_type(BOUNDARY_NODE);
    node2.set_T(250.0);

    model.add_node(node1);
    model.add_node(node2);
    model.add_conductive_coupling(1, 2, 5.0);
    model.add_radiative_coupling(1, 2, 0.5);

    const auto& const_model = std::as_const(model);
    REQUIRE(const_model.parameters_ptr() == model.parameters_ptr());
    REQUIRE(const_model.formulas_ptr() == model.formulas_ptr());
    REQUIRE(const_model.thermal_data_ptr() == model.thermal_data_ptr());
    REQUIRE(&const_model.parameters() == &model.parameters());
    REQUIRE(&const_model.formulas() == &model.formulas());
    REQUIRE(&const_model.thermal_data() == &model.thermal_data());
    REQUIRE(&const_model.conductive_couplings() ==
            &model.conductive_couplings());
    REQUIRE(&const_model.radiative_couplings() == &model.radiative_couplings());
    REQUIRE(
        model.flow_radiative(std::vector<Index>{1}, std::vector<Index>{2}) ==
        Catch::Approx(model.flow_radiative(1, 2)));

    const auto& entities = const_model.entities();
    REQUIRE(entities.temperature(1).string_representation() == "T1");
    REQUIRE(entities.capacity(1).string_representation() == "C1");
    REQUIRE(entities.solar_heat(1).string_representation() == "QS1");
    REQUIRE(entities.albedo_heat(1).string_representation() == "QA1");
    REQUIRE(entities.earth_ir(1).string_representation() == "QE1");
    REQUIRE(entities.internal_heat(1).string_representation() == "QI1");
    REQUIRE(entities.other_heat(1).string_representation() == "QR1");
    REQUIRE(entities.conductive_coupling(1, 2).string_representation() ==
            "GL(1,2)");
    REQUIRE(entities.radiative_coupling(1, 2).string_representation() ==
            "GR(1,2)");
}

TEST_CASE("ThermalMathematicalModel callback flags gate every callback family",
          "[thermalmathematicalmodel]") {
    ThermalMathematicalModel model("callback-flags");

    CallbackCountState py_callback_counts;

    reset_c_callback_counts();

    model.internal_callbacks_active = false;
    model.c_callbacks_active = true;
    model.python_callbacks_active = true;
    model.c_extern_callback_solver_loop = count_solver_callback;
    model.python_extern_callback_solver_loop = [&]() {
        ++py_callback_counts.solver_calls;
    };
    model.c_extern_callback_transient_time_change = count_time_callback;
    model.python_extern_callback_transient_time_change = [&]() {
        ++py_callback_counts.time_calls;
    };
    model.c_extern_callback_transient_after_timestep = count_after_callback;
    model.python_extern_callback_transient_after_timestep = [&]() {
        ++py_callback_counts.after_calls;
    };

    model.callback_solver_loop();
    model.callback_transient_time_change();
    model.callback_transient_after_timestep();

    require_callback_counts(
        c_callback_counts(),
        CallbackCountState{
            .solver_calls = 1, .time_calls = 1, .after_calls = 1});
    require_callback_counts(
        py_callback_counts,
        CallbackCountState{
            .solver_calls = 1, .time_calls = 1, .after_calls = 1});

    model.callbacks_active = false;
    model.callback_solver_loop();
    model.callback_transient_time_change();
    model.callback_transient_after_timestep();

    require_callback_counts(
        c_callback_counts(),
        CallbackCountState{
            .solver_calls = 1, .time_calls = 1, .after_calls = 1});
    require_callback_counts(
        py_callback_counts,
        CallbackCountState{
            .solver_calls = 1, .time_calls = 1, .after_calls = 1});
}

TEST_CASE("ThermalMathematicalModel surfaces missing solver associations",
          "[thermalmathematicalmodel]") {
    ThermalMathematicalModel model("association-checks");
    REQUIRE_THROWS_AS(model.solvers(), std::runtime_error);
    REQUIRE_THROWS_AS(std::as_const(model).solvers(), std::runtime_error);
    REQUIRE_FALSE(model.find_entity("missing").has_value());
    REQUIRE_THROWS_AS(model.entity("missing"), std::invalid_argument);

    auto shared_model =
        std::make_shared<ThermalMathematicalModel>("associated-model");
    solvers::SolverRegistry registry(shared_model);
    shared_model->associate_solvers(registry);

    REQUIRE(&shared_model->solvers() == &registry);
    REQUIRE(&std::as_const(*shared_model).solvers() == &registry);
    REQUIRE(shared_model->current_callback_solver() == nullptr);
}
