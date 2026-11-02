#include <Eigen/Core>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pycanha-core/config.hpp"
#include "pycanha-core/solvers/crank_nicolson.hpp"
#include "pycanha-core/solvers/jacobian_propagation.hpp"
#include "pycanha-core/solvers/report.hpp"
#include "pycanha-core/solvers/settings.hpp"
#include "pycanha-core/solvers/steady_state.hpp"
#include "pycanha-core/thermaldata/data_model.hpp"
#include "pycanha-core/thermaldata/dense_time_series.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "solver_test_models.hpp"
#include "solvers/auto_configure.hpp"

namespace {

using pycanha::solvers::Backend;
using pycanha::solvers::Factorization;
using pycanha::solvers::Settings;
using pycanha::solvers::SolverMemory;
using pycanha::solvers::SolverType;
using pycanha::solvers::Storage;
using pycanha::solvers::steady::SteadyState;
using pycanha::solvers::transient::CrankNicolson;
namespace models = pycanha::test_models;

constexpr int plate_size = 6;
constexpr std::int64_t double_bytes = sizeof(double);
constexpr std::int64_t int_bytes = sizeof(int);

struct Config {
    std::string name;
    Settings settings;
};

Config config(std::string name, Storage storage, Backend backend,
              std::optional<Factorization> factorization = std::nullopt,
              bool reuse = false) {
    Config made{.name = std::move(name), .settings = {}};
    made.settings.storage = storage;
    made.settings.backend = backend;
    made.settings.factorization = factorization;
    made.settings.reuse_factors = reuse;
    return made;
}

// Every transient configuration of this build. Cholesky only without
// radiation; AUTO picks it for PARDISO and dense storage then.
std::vector<Config> transient_configs(bool radiation) {
    std::vector<Config> configs = {
        config("EIGEN LU", Storage::SPARSE, Backend::EIGEN, Factorization::LU),
        config("DENSE EIGEN", Storage::DENSE, Backend::EIGEN),
        config("DENSE EIGEN LU", Storage::DENSE, Backend::EIGEN,
               Factorization::LU)};
    if (!radiation) {
        configs.push_back(config("EIGEN CHOLESKY", Storage::SPARSE,
                                 Backend::EIGEN, Factorization::CHOLESKY));
    }
    if (pycanha::MKL_ENABLED) {
        configs.push_back(config("PARDISO", Storage::SPARSE, Backend::PARDISO));
        configs.push_back(config("PARDISO LU", Storage::SPARSE,
                                 Backend::PARDISO, Factorization::LU));
        configs.push_back(config("PARDISO reuse", Storage::SPARSE,
                                 Backend::PARDISO, std::nullopt,
                                 /*reuse=*/true));
        configs.push_back(
            config("DENSE LAPACK", Storage::DENSE, Backend::LAPACK));
        configs.push_back(config("DENSE LAPACK LU", Storage::DENSE,
                                 Backend::LAPACK, Factorization::LU));
        configs.push_back(config("DENSE LAPACK reuse", Storage::DENSE,
                                 Backend::LAPACK, std::nullopt,
                                 /*reuse=*/true));
    }
    // Each configuration again with fixed_couplings.
    const std::size_t varying = configs.size();
    for (std::size_t k = 0; k < varying; ++k) {
        Config fixed = configs[k];
        fixed.name += ", fixed couplings";
        fixed.settings.fixed_couplings = true;
        configs.push_back(std::move(fixed));
    }
    return configs;
}

// The temperatures of every output of a 100 s transient on the plate.
Eigen::MatrixXd run_transient(const models::Model& model, Settings settings,
                              double start_temperature) {
    models::reset_temperatures(model, start_temperature);
    CrankNicolson solver(model);
    settings.max_iterations = 50;
    settings.tolerance = 1e-10;
    solver.set_settings(settings);
    solver.set_simulation_time(0.0, 100.0, 10.0, 20.0);
    solver.initialize();
    solver.solve();
    REQUIRE(solver.converged());
    return solver.output_model().T().values();
}

double max_difference(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
    return (a - b).cwiseAbs().maxCoeff();
}

// n nodes that all exchange radiation with each other and with a boundary
// node, like an enclosure.
models::Model make_enclosure(int n) {
    auto model = std::make_shared<pycanha::ThermalMathematicalModel>("box");
    for (int k = 1; k <= n; ++k) {
        pycanha::Node node(k);
        node.set_T(300.0);
        node.set_C(1.0);
        model->add_node(node);
    }
    pycanha::Node sink(models::boundary_node);
    sink.set_type(pycanha::BOUNDARY_NODE);
    sink.set_T(250.0);
    model->add_node(sink);
    for (int k = 1; k <= n; ++k) {
        model->add_conductive_coupling(k, models::boundary_node, 0.1);
        for (int other = k + 1; other <= n; ++other) {
            model->add_radiative_coupling(k, other, 0.01);
        }
    }
    return model;
}

pycanha::solvers::ResolvedSettings auto_configured(
    pycanha::solvers::Solver& solver) {
    static_cast<void>(solver.auto_configure());
    return solver.resolved_settings();
}

// auto_configure() picked dense storage, the build's dense library and, with
// MKL, factor reuse, and explain() says why.
void require_dense_with_reuse(pycanha::solvers::Solver& solver) {
    INFO(solver.name());
    const auto resolved = auto_configured(solver);
    REQUIRE(resolved.storage == Storage::DENSE);
    REQUIRE(resolved.backend ==
            (pycanha::MKL_ENABLED ? Backend::LAPACK : Backend::EIGEN));
    REQUIRE(resolved.reuse_factors == pycanha::MKL_ENABLED);
    REQUIRE(solver.explain().contains("a dense factorization is faster"));
}

SolverMemory steady_estimate(const Config& made) {
    const auto model = models::make_plate(plate_size, /*radiation=*/false);
    SteadyState solver(model);
    solver.set_settings(made.settings);
    return solver.estimate_memory();
}

}  // namespace

TEST_CASE("Every configuration gives the same conduction transient",
          "[solver][transient][dense]") {
    const auto model = models::make_plate(plate_size, /*radiation=*/false);
    const Eigen::MatrixXd reference = run_transient(
        model, transient_configs(/*radiation=*/false).front().settings, 293.15);
    const auto made =
        GENERATE(from_range(transient_configs(/*radiation=*/false)));
    INFO(made.name);
    REQUIRE(max_difference(run_transient(model, made.settings, 293.15),
                           reference) < 1e-8);
}

TEST_CASE("Every configuration gives the same radiative transient",
          "[solver][transient][dense]") {
    const auto model = models::make_plate(plate_size, /*radiation=*/true);
    const Eigen::MatrixXd reference = run_transient(
        model, transient_configs(/*radiation=*/true).front().settings, 250.0);
    const auto made =
        GENERATE(from_range(transient_configs(/*radiation=*/true)));
    INFO(made.name);
    REQUIRE(max_difference(run_transient(model, made.settings, 250.0),
                           reference) < 1e-6);
}

TEST_CASE("A conduction transient uses Cholesky",
          "[solver][transient][dense]") {
    const auto storage = GENERATE(Storage::SPARSE, Storage::DENSE);
    const auto model = models::make_plate(plate_size, /*radiation=*/false);
    CrankNicolson solver(model);
    Settings settings;
    settings.storage = storage;
    solver.set_settings(settings);
    solver.set_simulation_time(0.0, 100.0, 10.0, 20.0);
    // Only Eigen's sparse storage prefers its LU.
    const Factorization expected =
        models::default_sparse_backend() != Backend::EIGEN ||
                storage == Storage::DENSE
            ? Factorization::CHOLESKY
            : Factorization::LU;
    REQUIRE(solver.resolved_settings().factorization == expected);
}

TEST_CASE("Dense memory is n^2 doubles and exact", "[solver][memory][dense]") {
    constexpr auto n = static_cast<std::int64_t>(plate_size) * plate_size;
    constexpr std::int64_t square = n * n * double_bytes;
    // LLT and dpotrf: the factor only.
    const SolverMemory cholesky =
        steady_estimate(config("DENSE EIGEN", Storage::DENSE, Backend::EIGEN));
    REQUIRE(cholesky.exact);
    REQUIRE(cholesky.factors == square);
    // PartialPivLU: the factors and two permutations.
    const SolverMemory lu = steady_estimate(config(
        "DENSE EIGEN LU", Storage::DENSE, Backend::EIGEN, Factorization::LU));
    REQUIRE(lu.factors == square + (2 * n * int_bytes));
    // The matrix and the copy of the last factorized one.
    REQUIRE(lu.matrix >= 2 * square);
}

TEST_CASE("A dense run keeps the estimated memory", "[solver][memory][dense]") {
    const auto backend = GENERATE(Backend::EIGEN, Backend::LAPACK);
    if (backend == Backend::LAPACK && !pycanha::MKL_ENABLED) {
        SKIP("MKL not built");
    }
    const auto model = models::make_plate(plate_size, /*radiation=*/true);
    SteadyState solver(model);
    Settings settings;
    settings.storage = Storage::DENSE;
    settings.backend = backend;
    solver.set_settings(settings);
    const SolverMemory estimate = solver.estimate_memory();
    solver.initialize();
    solver.solve();
    REQUIRE(solver.converged());
    REQUIRE(solver.report().memory.factors == estimate.factors);
    REQUIRE(solver.report().memory.total() == estimate.total());
}

TEST_CASE("explain() names the dense configuration", "[solver][dense]") {
    const auto model = models::make_plate(plate_size, /*radiation=*/true);
    SteadyState solver(model);
    Settings settings;
    settings.storage = Storage::DENSE;
    solver.set_settings(settings);
    const std::string text = solver.explain();
    REQUIRE(text.contains("DENSE"));
    REQUIRE(text.contains("dense storage"));
    REQUIRE_FALSE(text.contains("iparm"));
    REQUIRE_FALSE(text.contains("pardiso.level"));
}

TEST_CASE("auto_configure() owns the storage", "[solver][auto][dense]") {
    const auto model = models::make_plate(plate_size, /*radiation=*/true);
    SteadyState solver(model);
    Settings settings;
    settings.storage = Storage::DENSE;
    solver.set_settings(settings);
    static_cast<void>(solver.auto_configure());
    REQUIRE(solver.resolved_settings().storage == Storage::SPARSE);
}

TEST_CASE("LAPACK gives the same result on any number of threads",
          "[solver][dense]") {
    if (!pycanha::MKL_ENABLED) {
        SKIP("MKL not built");
    }
    const auto model = models::make_plate(plate_size, /*radiation=*/true);
    const auto solve_with = [&model](int threads) {
        models::reset_temperatures(model, 250.0);
        SteadyState solver(model);
        Settings settings;
        settings.storage = Storage::DENSE;
        settings.threads = threads;
        settings.tolerance = 1e-10;
        settings.max_iterations = 50;
        solver.set_settings(settings);
        solver.initialize();
        solver.solve();
        REQUIRE(solver.converged());
        return models::temperatures(model);
    };
    const Eigen::VectorXd one = solve_with(1);
    REQUIRE(models::max_difference(solve_with(2), one) < 1e-9);
}

TEST_CASE("The dense rule of auto_configure()", "[solver][auto][dense]") {
    using pycanha::solvers::detail::dense_reason;
    using pycanha::solvers::detail::NetworkShape;
    const auto steady = SolverType::STEADY_STATE;
    REQUIRE_FALSE(
        dense_reason(steady, {.nodes = 5000, .mean_couplings = 100.0}).empty());
    REQUIRE(dense_reason(steady, {.nodes = 5001, .mean_couplings = 4000.0})
                .empty());
    REQUIRE(
        dense_reason(steady, {.nodes = 400, .mean_couplings = 99.0}).empty());

    const auto transient =
        GENERATE(SolverType::CRANK_NICOLSON, SolverType::JACOBIAN_PROPAGATION);
    REQUIRE_FALSE(
        dense_reason(transient, {.nodes = 2000, .fill = 0.2}).empty());
    REQUIRE(dense_reason(transient, {.nodes = 2001, .fill = 1.0}).empty());
    REQUIRE(dense_reason(transient, {.nodes = 400, .fill = 0.19}).empty());
}

TEST_CASE("The network shape counts each coupled pair once",
          "[solver][auto][dense]") {
    // A 3 x 3 plate: 12 conductive pairs. A radiative coupling on one of
    // them adds nothing; one between two other nodes adds a pair.
    const auto model = models::make_plate(3, /*radiation=*/false);
    model->add_radiative_coupling(1, 2, 0.01);
    model->add_radiative_coupling(1, 9, 0.01);
    auto& network = model->network();
    network.synchronize_structure();
    const auto shape = pycanha::solvers::detail::network_shape(
        network.conductive_matrices().sparse_dd,
        network.radiative_matrices().sparse_dd);
    REQUIRE(shape.nodes == 9);
    REQUIRE(shape.mean_couplings == Catch::Approx(26.0 / 9.0));
    REQUIRE(shape.fill == Catch::Approx(35.0 / 81.0));
}

TEST_CASE("auto_configure() chooses dense storage for an enclosure",
          "[solver][auto][dense]") {
    const auto model = make_enclosure(102);
    SteadyState steady(model);
    require_dense_with_reuse(steady);
    CrankNicolson transient(model);
    transient.set_simulation_time(0.0, 100.0, 10.0, 0.0);
    require_dense_with_reuse(transient);
}

TEST_CASE("auto_configure() keeps sparse storage for a plate",
          "[solver][auto][dense]") {
    const auto model = models::make_plate(plate_size, /*radiation=*/true);
    SteadyState steady(model);
    CrankNicolson transient(model);
    transient.set_simulation_time(0.0, 100.0, 10.0, 0.0);
    REQUIRE(auto_configured(steady).storage == Storage::SPARSE);
    REQUIRE(auto_configured(transient).storage == Storage::SPARSE);
}

TEST_CASE("auto_configure() gives a dense Jacobian no factor reuse",
          "[solver][auto][dense]") {
    const auto model = make_enclosure(30);
    pycanha::solvers::sensitivity::JacobianPropagation solver(model);
    solver.set_simulation_time(0.0, 100.0, 10.0, 0.0);
    const auto resolved = auto_configured(solver);
    REQUIRE(resolved.storage == Storage::DENSE);
    REQUIRE_FALSE(resolved.applies("reuse_factors"));
}
