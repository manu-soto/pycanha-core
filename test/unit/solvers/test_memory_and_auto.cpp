#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

#include "pycanha-core/config.hpp"
#include "pycanha-core/solvers/crank_nicolson.hpp"
#include "pycanha-core/solvers/jacobian_propagation.hpp"
#include "pycanha-core/solvers/report.hpp"
#include "pycanha-core/solvers/settings.hpp"
#include "pycanha-core/solvers/steady_state.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "solver_test_models.hpp"
#include "solvers/auto_configure.hpp"

namespace {

using pycanha::solvers::Backend;
using pycanha::solvers::Factorization;
using pycanha::solvers::Origin;
using pycanha::solvers::PardisoLevel;
using pycanha::solvers::PardisoOrdering;
using pycanha::solvers::Settings;
using pycanha::solvers::SolverMemory;
using pycanha::solvers::steady::SteadyState;
using pycanha::solvers::transient::CrankNicolson;
namespace models = pycanha::test_models;

constexpr int plate_size = 6;
constexpr double time_step = 10.0;
constexpr double end_time = 100.0;
constexpr double output_interval = 20.0;

std::unique_ptr<CrankNicolson> plate_transient(const models::Model& model) {
    auto solver = std::make_unique<CrankNicolson>(model);
    solver->set_simulation_time(0.0, end_time, time_step, output_interval);
    return solver;
}

// A plate plus one node exchanging radiation with every plate node: a few
// nodes coupled to many.
models::Model make_hub_plate(int n) {
    auto model = models::make_plate(n, /*radiation=*/false);
    constexpr int hub = 50000;
    constexpr double exchange = 0.001;
    pycanha::Node node(hub);
    node.set_T(300.0);
    node.set_C(1.0);
    model->add_node(node);
    model->add_conductive_coupling(hub, models::boundary_node, 1.0);
    for (int k = 1; k <= n * n; ++k) {
        model->add_radiative_coupling(hub, k, exchange);
    }
    return model;
}

bool same_temperatures(const models::Model& a, const models::Model& b,
                       double tolerance) {
    return models::max_difference(models::temperatures(a),
                                  models::temperatures(b)) < tolerance;
}

}  // namespace

TEST_CASE("Memory sizes are parsed and printed", "[solver][memory]") {
    using pycanha::solvers::format_bytes;
    using pycanha::solvers::parse_memory_size;
    REQUIRE(parse_memory_size("16 GB") == std::int64_t{16'000'000'000});
    REQUIRE(parse_memory_size(" 1.5gib ") == std::int64_t{1'610'612'736});
    REQUIRE(parse_memory_size("2000000") == std::int64_t{2'000'000});
    REQUIRE((format_bytes(1'210'000'000) == "1.21 GB" &&
             format_bytes(17) == "17 B"));
}

TEST_CASE("Invalid memory sizes are refused", "[solver][memory]") {
    using pycanha::solvers::parse_memory_size;
    REQUIRE_THROWS_AS(parse_memory_size("0 GB"), std::invalid_argument);
    REQUIRE_THROWS_AS(parse_memory_size("16 XB"), std::invalid_argument);
    REQUIRE_THROWS_AS(parse_memory_size("GB"), std::invalid_argument);
}

TEST_CASE("A PARDISO estimate is exact and initialize() reuses it",
          "[solver][memory]") {
    if (!pycanha::MKL_ENABLED) {
        SKIP("MKL not built");
    }
    const auto model = models::make_plate(plate_size, /*radiation=*/true);
    SteadyState solver(model);
    const SolverMemory estimate = solver.estimate_memory();
    REQUIRE((estimate.exact && estimate.factors > 0 && estimate.output == 0));
    REQUIRE_FALSE(solver.initialized());

    solver.initialize();
    solver.solve();
    const auto reference = models::make_plate(plate_size, /*radiation=*/true);
    SteadyState direct(reference);
    direct.initialize();
    direct.solve();
    REQUIRE(same_temperatures(model, reference, 1e-12));
    REQUIRE(solver.report().memory.factors > 0);
}

TEST_CASE("A one-level PARDISO estimate is exact only after factorizing",
          "[solver][memory]") {
    if (!pycanha::MKL_ENABLED) {
        SKIP("MKL not built");
    }
    const auto model = models::make_plate(plate_size, /*radiation=*/true);
    auto solver = plate_transient(model);
    solver->set("reuse_factors", true);
    const SolverMemory estimate = solver->estimate_memory();
    REQUIRE_FALSE(estimate.exact);

    solver->initialize();
    solver->solve();
    REQUIRE(solver->report().memory.exact);
    REQUIRE(solver->report().memory.factors <= estimate.factors);
}

TEST_CASE("An estimate is redone after a model change", "[solver][memory]") {
    const auto model = models::make_plate(plate_size, /*radiation=*/false);
    SteadyState solver(model);
    const SolverMemory before = solver.estimate_memory();
    model->add_conductive_coupling(1, plate_size * plate_size, 1.0);
    const SolverMemory after = solver.estimate_memory();
    REQUIRE(after.matrix > before.matrix);

    solver.initialize();
    solver.solve();
    REQUIRE(solver.converged());
}

TEST_CASE("The Eigen Cholesky estimate is exact", "[solver][memory]") {
    const auto model = models::make_plate(plate_size, /*radiation=*/false);
    SteadyState solver(model);
    Settings settings;
    settings.backend = Backend::EIGEN;
    settings.factorization = Factorization::CHOLESKY;
    solver.set_settings(settings);
    const SolverMemory estimate = solver.estimate_memory();
    REQUIRE(estimate.exact);

    solver.initialize();
    solver.solve();
    REQUIRE(solver.report().memory.factors == estimate.factors);
}

TEST_CASE("The Eigen LU estimate is predicted, then counted",
          "[solver][memory]") {
    const auto model = models::make_plate(plate_size, /*radiation=*/true);
    SteadyState solver(model);
    solver.set("backend", std::string("EIGEN"));
    const SolverMemory estimate = solver.estimate_memory();
    REQUIRE((!estimate.exact && estimate.factors > 0));

    solver.initialize();
    solver.solve();
    REQUIRE(solver.report().memory.exact);
    REQUIRE(solver.report().memory.factors > 0);
}

TEST_CASE("A transient estimate includes the output model",
          "[solver][memory]") {
    const auto model = models::make_plate(plate_size, /*radiation=*/true);
    auto solver = plate_transient(model);
    const SolverMemory estimate = solver->estimate_memory();
    // 10 steps, an output every 2: 6 outputs of the temperatures of every
    // node (the plate, its edge and space) and their time.
    const auto nodes = std::int64_t{(plate_size * plate_size) + 2};
    constexpr std::int64_t outputs = 6;
    REQUIRE(estimate.output ==
            outputs * (nodes + 1) * std::int64_t{sizeof(double)});

    solver->initialize();
    REQUIRE(solver->output_model().T().num_timesteps() == outputs);
}

TEST_CASE("A Jacobian estimate needs a derivative parameter",
          "[solver][memory]") {
    const auto model = models::make_plate(plate_size, /*radiation=*/false);
    pycanha::solvers::sensitivity::JacobianPropagation solver(model);
    solver.set_simulation_time(0.0, end_time, time_step, output_interval);
    REQUIRE_THROWS_AS(solver.estimate_memory(), std::invalid_argument);
}

TEST_CASE("auto_configure() picks the steady-state configuration",
          "[solver][auto]") {
    const auto model = models::make_plate(plate_size, /*radiation=*/true);
    SteadyState solver(model);
    Settings settings;
    settings.tolerance = 1e-5;
    settings.pardiso.iparm = std::map<int, int>{{1, 0}};
    solver.set_settings(settings);
    const Settings chosen = solver.auto_configure();

    REQUIRE(chosen == solver.settings());
    REQUIRE(chosen.tolerance == 1e-5);
    REQUIRE_FALSE(chosen.pardiso.iparm.has_value());
    REQUIRE(chosen.backend == models::default_sparse_backend());
    REQUIRE(solver.origin("backend") == Origin::AUTO_CONFIGURE);
}

TEST_CASE("auto_configure() prefers Accelerate without MKL", "[solver][auto]") {
    // The candidates themselves, so that every platform checks them.
    using pycanha::solvers::SettingValue;
    using pycanha::solvers::SolverType;
    const auto type =
        GENERATE(SolverType::STEADY_STATE, SolverType::CRANK_NICOLSON,
                 SolverType::JACOBIAN_PROPAGATION);
    const auto candidates = pycanha::solvers::detail::auto_candidates(
        type, {.mkl = false, .accelerate = true},
        pycanha::solvers::detail::NetworkShape{
            .nodes = 10000, .mean_couplings = 4.0, .fill = 5e-4});
    REQUIRE(candidates.size() == 2);
    for (const auto& [index, ordering] :
         std::map<std::size_t, std::string>{{0, "METIS"}, {1, "AMD"}}) {
        Settings settings;
        for (const auto& field : candidates.at(index).fields) {
            settings.set(field.name, field.value);
        }
        REQUIRE(settings.backend == Backend::ACCELERATE);
        REQUIRE(settings.get("accelerate.ordering") == SettingValue{ordering});
        REQUIRE_FALSE(settings.reuse_factors.value_or(false));
    }
}

TEST_CASE("An Accelerate estimate comes from its analysis",
          "[solver][memory]") {
    if (!pycanha::solvers::accelerate_available()) {
        SKIP("Accelerate not available");
    }
    // Cholesky without radiation is fixed by the analysis; the LU may
    // pivot, so its figure stays a prediction.
    const bool radiation = GENERATE(false, true);
    const auto model = models::make_plate(plate_size, radiation);
    SteadyState solver(model);
    solver.set("backend", std::string("ACCELERATE"));
    const SolverMemory estimate = solver.estimate_memory();
    REQUIRE(estimate.factors > 0);
    REQUIRE(estimate.exact == !radiation);

    solver.initialize();
    solver.solve();
    REQUIRE(solver.converged());
    REQUIRE(solver.report().memory.factors == estimate.factors);
}

TEST_CASE("auto_configure() reuses factors in a transient", "[solver][auto]") {
    if (!pycanha::MKL_ENABLED) {
        SKIP("MKL not built");
    }
    const auto model = models::make_plate(plate_size, /*radiation=*/true);
    auto solver = plate_transient(model);
    const Settings chosen = solver->auto_configure();
    REQUIRE(chosen.reuse_factors == true);
    REQUIRE((chosen.pardiso.level == PardisoLevel::ONE_LEVEL &&
             chosen.pardiso.ordering == PardisoOrdering::MIN_DEGREE));

    solver->initialize();
    solver->solve();
    const auto reference = models::make_plate(plate_size, /*radiation=*/true);
    auto direct = plate_transient(reference);
    direct->initialize();
    direct->solve();
    REQUIRE(same_temperatures(model, reference, 1e-6));
}

TEST_CASE("auto_configure() reuses factors on a hub too", "[solver][auto]") {
    if (!pycanha::MKL_ENABLED) {
        SKIP("MKL not built");
    }
    constexpr int hub_plate = 20;
    const auto model = make_hub_plate(hub_plate);
    auto solver = plate_transient(model);
    const Settings chosen = solver->auto_configure();
    REQUIRE(chosen.reuse_factors == true);
    REQUIRE(solver->explain().contains("a few nodes are coupled to many"));
    solver->initialize();
    solver->solve();
    REQUIRE(solver->converged());
}

TEST_CASE("auto_configure() takes the smallest when nothing fits",
          "[solver][auto]") {
    const auto model = models::make_plate(plate_size, /*radiation=*/true);
    SteadyState solver(model);
    const Settings chosen = solver.auto_configure("1 kB");
    REQUIRE(chosen.backend.has_value());
    solver.initialize();
    solver.solve();
    REQUIRE(solver.converged());
}

TEST_CASE("auto_configure() refuses an initialized solver", "[solver][auto]") {
    const auto model = models::make_plate(plate_size, /*radiation=*/false);
    SteadyState solver(model);
    solver.initialize();
    REQUIRE_THROWS_AS(solver.auto_configure(), std::runtime_error);
}

TEST_CASE("A changed auto_configure() value becomes the user's",
          "[solver][auto]") {
    const auto model = models::make_plate(plate_size, /*radiation=*/false);
    SteadyState solver(model);
    static_cast<void>(solver.auto_configure());
    solver.set("backend", std::string("EIGEN"));
    // A change only when auto_configure() chose another backend.
    REQUIRE(solver.origin("backend") ==
            (models::default_sparse_backend() == Backend::EIGEN
                 ? Origin::AUTO_CONFIGURE
                 : Origin::USER));
    REQUIRE(solver.origin("factorization") == Origin::AUTO_CONFIGURE);
}
