#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>

#include "pycanha-core/config.hpp"
#include "pycanha-core/parameters/formulas.hpp"
#include "pycanha-core/solvers/jacobian_propagation.hpp"
#include "pycanha-core/solvers/settings.hpp"
#include "pycanha-core/thermaldata/dense_matrix_time_series.hpp"
#include "pycanha-core/thermaldata/dense_time_series.hpp"
#include "pycanha-core/thermaldata/thermaldata.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"

namespace {

constexpr double initial_temperature = 0.0;
constexpr double environment_temperature = 1.0;
constexpr double heat_load = 1.0;
constexpr double end_time = 5.0;
constexpr double time_step = 0.01;
constexpr double output_stride = 0.1;

constexpr double example_conductive_coupling = 1.0;
constexpr double example_capacity = 1.0;
constexpr double comparison_tolerance = 5.0e-6;

constexpr std::array<double, 6> sample_times = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};

constexpr std::array<std::array<double, 2>, sample_times.size()>
    expected_temperature_samples = {{
        {0.0, 0.0},
        {1.0, 1.26424718},
        {2.0, 1.72933389},
        {3.0, 1.90042832},
        {4.0, 1.96336993},
        {5.0, 1.98652466},
    }};

constexpr std::array<std::array<double, 2>, sample_times.size()>
    expected_dk_samples = {{
        {0.0, 0.0},
        {1.0, 0.10364756},
        {2.0, -0.32332125},
        {3.0, -0.65149169},
        {4.0, -0.83516103},
        {5.0, -0.92588396},
    }};

constexpr std::array<std::array<double, 2>, sample_times.size()>
    expected_dc_samples = {{
        {0.0, 0.0},
        {1.0, -0.73577107},
        {2.0, -0.54134564},
        {3.0, -0.29872244},
        {4.0, -0.14652392},
        {5.0, -0.06737837},
    }};

std::shared_ptr<pycanha::ThermalMathematicalModel> make_python_example_model() {
    auto model = std::make_shared<pycanha::ThermalMathematicalModel>(
        "jacobian_python_example");

    pycanha::Node diffusive_node(1);
    pycanha::Node boundary_node(2);

    diffusive_node.set_T(initial_temperature);
    diffusive_node.set_C(example_capacity);
    diffusive_node.set_qi(heat_load);

    boundary_node.set_type(pycanha::BOUNDARY_NODE);
    boundary_node.set_T(environment_temperature);

    model->add_node(diffusive_node);
    model->add_node(boundary_node);
    model->add_conductive_coupling(1, 2, example_conductive_coupling);

    model->parameters().add_parameter("k", example_conductive_coupling);
    model->parameters().add_parameter("C", example_capacity);

    // This is the C++ equivalent of the Python formulas GL(1,2)=k and C1=C.
    // The derivative with respect to the corresponding parameter is 1.
    static_cast<void>(model->formulas().add_parameter_formula("GL(1,2)", "k"));
    static_cast<void>(model->formulas().add_parameter_formula("C1", "C"));
    model->formulas().apply_formulas();
    model->formulas().parameters_with_derivatives().add_parameter("k");
    model->formulas().parameters_with_derivatives().add_parameter("C");
    return model;
}

Eigen::Index find_time_row(const pycanha::DenseTimeSeries& table,
                           double time_value) {
    for (Eigen::Index row = 0; row < table.num_timesteps(); ++row) {
        if (std::abs(table.times()(row) - time_value) < 1.0e-12) {
            return row;
        }
    }

    throw std::runtime_error(
        "Requested sample time not found in solver output");
}

Eigen::Index find_time_row(const pycanha::DenseMatrixTimeSeries& table,
                           double time_value) {
    for (Eigen::Index row = 0; row < table.num_timesteps(); ++row) {
        if (std::abs(table.time_at(row) - time_value) < 1.0e-12) {
            return row;
        }
    }

    throw std::runtime_error(
        "Requested sample time not found in solver output");
}

void require_temperature_sample(
    const pycanha::DenseTimeSeries& temperature_output, Eigen::Index row,
    const std::array<double, 2>& expected_temperature) {
    REQUIRE(
        temperature_output.times()(row) ==
        Catch::Approx(expected_temperature.at(0)).margin(comparison_tolerance));
    REQUIRE(
        temperature_output.values()(row, 0) ==
        Catch::Approx(expected_temperature.at(1)).margin(comparison_tolerance));
}

void require_jacobian_sample(
    const pycanha::DenseMatrixTimeSeries& jacobian_output, Eigen::Index row,
    const std::array<double, 2>& expected_dk,
    const std::array<double, 2>& expected_dc) {
    REQUIRE(jacobian_output.time_at(row) ==
            Catch::Approx(expected_dk.at(0)).margin(comparison_tolerance));
    REQUIRE(jacobian_output.at(row)(0, 0) ==
            Catch::Approx(expected_dk.at(1)).margin(comparison_tolerance));
    REQUIRE(jacobian_output.at(row)(0, 1) ==
            Catch::Approx(expected_dc.at(1)).margin(comparison_tolerance));
}

void require_python_example_samples(
    const pycanha::DenseTimeSeries& temperature_output,
    const pycanha::DenseMatrixTimeSeries& jacobian_output) {
    for (std::size_t sample_index = 0; sample_index < sample_times.size();
         ++sample_index) {
        const auto& expected_temperature =
            expected_temperature_samples.at(sample_index);
        const auto& expected_dk = expected_dk_samples.at(sample_index);
        const auto& expected_dc = expected_dc_samples.at(sample_index);
        const Eigen::Index row =
            find_time_row(temperature_output, sample_times.at(sample_index));
        const Eigen::Index jacobian_row =
            find_time_row(jacobian_output, sample_times.at(sample_index));

        require_temperature_sample(temperature_output, row,
                                   expected_temperature);
        require_jacobian_sample(jacobian_output, jacobian_row, expected_dk,
                                expected_dc);
    }
}

}  // namespace

TEST_CASE("JacobianPropagation reproduces the Python example",
          "[solver][transient][jacobian]") {
    auto model = make_python_example_model();

    pycanha::solvers::sensitivity::JacobianPropagation solver(model);
    pycanha::solvers::Settings settings;
    settings.max_iterations = 50;
    settings.tolerance = 1.0e-9;
    solver.set_settings(settings);
    solver.set_simulation_time(0.0, end_time, time_step, output_stride);
    solver.initialize();
    solver.solve();

    REQUIRE(model->thermal_data().models().has_model("jacobian_propagation"));

    const auto& output_model =
        model->thermal_data().models().get_model("jacobian_propagation");
    const auto& temperature_output = output_model.T();
    const auto& jacobian_output = output_model.jacobian();
    REQUIRE(jacobian_output.rows() == 1);
    REQUIRE(jacobian_output.cols() == 2);
    REQUIRE(jacobian_output.num_timesteps() >= 2);
    REQUIRE(solver.parameter_names().size() == 2);
    REQUIRE(solver.parameter_names().at(0) == "k");
    REQUIRE(solver.parameter_names().at(1) == "C");
    REQUIRE(solver.derivative_parameter_names().size() == 2);
    REQUIRE(solver.derivative_parameter_names().at(0) == "k");
    REQUIRE(solver.derivative_parameter_names().at(1) == "C");

    require_python_example_samples(temperature_output, jacobian_output);
}

TEST_CASE("JacobianPropagation gives the same result with every configuration",
          "[solver][transient][jacobian][dense]") {
    using pycanha::solvers::Backend;
    using pycanha::solvers::Factorization;
    using pycanha::solvers::Storage;
    struct Option {
        Storage storage;
        Backend backend;
        std::optional<Factorization> factorization;
    };
    const auto option = GENERATE(
        Option{Storage::SPARSE, Backend::EIGEN, Factorization::LU},
        Option{Storage::SPARSE, Backend::EIGEN, Factorization::CHOLESKY},
        Option{Storage::SPARSE, Backend::PARDISO, std::nullopt},
        Option{Storage::SPARSE, Backend::PARDISO, Factorization::LU},
        Option{Storage::DENSE, Backend::EIGEN, std::nullopt},
        Option{Storage::DENSE, Backend::EIGEN, Factorization::LU},
        Option{Storage::DENSE, Backend::LAPACK, std::nullopt},
        Option{Storage::DENSE, Backend::LAPACK, Factorization::LU},
        Option{Storage::SPARSE, Backend::ACCELERATE, std::nullopt},
        Option{Storage::SPARSE, Backend::ACCELERATE, Factorization::LU});
    const bool needs_mkl =
        option.backend == Backend::PARDISO || option.backend == Backend::LAPACK;
    if (needs_mkl && !pycanha::MKL_ENABLED) {
        SKIP("MKL not built");
    }
    if (option.backend == Backend::ACCELERATE &&
        !pycanha::solvers::accelerate_available()) {
        SKIP("Accelerate not available");
    }
    const bool fixed = GENERATE(false, true);
    auto model = make_python_example_model();
    pycanha::solvers::sensitivity::JacobianPropagation solver(model);
    pycanha::solvers::Settings settings;
    settings.fixed_couplings = fixed;
    settings.max_iterations = 50;
    settings.tolerance = 1.0e-9;
    settings.storage = option.storage;
    settings.backend = option.backend;
    settings.factorization = option.factorization;
    solver.set_settings(settings);
    solver.set_simulation_time(0.0, end_time, time_step, output_stride);
    solver.initialize();
    solver.solve();
    const auto& output_model =
        model->thermal_data().models().get_model("jacobian_propagation");
    require_python_example_samples(output_model.T(), output_model.jacobian());
}

TEST_CASE("JacobianPropagation needs a derivative parameter",
          "[solver][transient][jacobian]") {
    auto model = make_python_example_model();
    static_cast<void>(
        model->formulas().parameters_with_derivatives().remove_parameter("k"));
    static_cast<void>(
        model->formulas().parameters_with_derivatives().remove_parameter("C"));
    pycanha::solvers::sensitivity::JacobianPropagation solver(model);
    solver.set_simulation_time(0.0, end_time, time_step, output_stride);
    REQUIRE_THROWS_AS(solver.initialize(), std::invalid_argument);
    REQUIRE_FALSE(solver.initialized());
}
