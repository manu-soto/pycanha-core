// API walkthrough of pycanha::conduction — the compile-time contract of the
// public surface: a geometry model with node numbers, thicknesses and bulk
// materials in, a populated thermal mathematical model out. The physics of the
// individual conductances lives in test/unit/conduction.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <string>
#include <utility>

#include "pycanha-core/conduction/conduction.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/gmm/geometrymodel.hpp"
#include "pycanha-core/gmm/materials/bulk_material.hpp"
#include "pycanha-core/gmm/mesh/thermal_mesh.hpp"
#include "pycanha-core/gmm/primitives/rectangle.hpp"
#include "pycanha-core/gmm/scene/geometry_item.hpp"
#include "pycanha-core/solvers/solver_registry.hpp"
#include "pycanha-core/solvers/steady_state.hpp"
#include "pycanha-core/tmm/conductivecouplings.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/nodes.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/tmm/thermalmodel.hpp"

namespace {

using pycanha::ThermalModel;
using pycanha::conduction::TmmBuildOptions;
using pycanha::conduction::TmmBuildReport;
using pycanha::gmm::ActiveSide;
using pycanha::gmm::BulkMaterial;
using pycanha::gmm::GeometryItem;
using pycanha::gmm::Rectangle;
using pycanha::gmm::ThermalMesh;

// A 1 m x 0.1 m aluminium strip cut into four cells along its length, node
// numbers 1..4 running from one end to the other. Only side 1 conducts, so the
// strip is a single sheet 2 mm thick.
void add_strip(ThermalModel& model) {
    ThermalMesh mesh({0.0, 0.25, 0.5, 0.75, 1.0}, {0.0, 1.0});
    mesh.set_side1_material(
        std::make_shared<BulkMaterial>("aluminium", 2700.0, 167.0, 896.0));
    mesh.set_side1_thick(0.002);
    mesh.set_node1_start(1);
    mesh.set_node1_step(1);
    mesh.set_conductive_active_side(ActiveSide::Side1);

    model.gmm().add(std::make_shared<GeometryItem>(
        "strip", Rectangle({0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 0.1, 0.0}),
        std::move(mesh)));
}

// The conductance the strip's own cells carry:
// k * t * width / spacing = 167 * 0.002 * 0.1 / 0.25.
constexpr double cell_conductance = 167.0 * 0.002 * 0.1 / 0.25;

void require_strip_node(pycanha::Nodes& nodes, pycanha::NodeNum node_num) {
    // Each cell is 0.25 m x 0.1 m: rho * cp * t * A.
    const double expected_capacitance = 2700.0 * 896.0 * 0.002 * 0.25 * 0.1;
    REQUIRE(nodes.is_node(node_num));
    REQUIRE(nodes.get_type(node_num) == 'D');
    REQUIRE(nodes.get_T(node_num) == Catch::Approx(273.15));
    REQUIRE(nodes.get_C(node_num) == Catch::Approx(expected_capacitance));
    REQUIRE(nodes.get_a(node_num) == Catch::Approx(0.025));
}

void require_strip_nodes(ThermalModel& model) {
    pycanha::Nodes& nodes = model.tmm().nodes();
    for (pycanha::NodeNum node_num = 1; node_num <= 4; ++node_num) {
        require_strip_node(nodes, node_num);
    }
    // Cell centres along the strip.
    REQUIRE(nodes.get_fx(1) == Catch::Approx(0.125));
    REQUIRE(nodes.get_fx(4) == Catch::Approx(0.875));
}

void require_strip_conductors(ThermalModel& model) {
    auto& couplings = model.tmm().conductive_couplings();
    REQUIRE(couplings.get_coupling_value(1, 2) ==
            Catch::Approx(cell_conductance));
    REQUIRE(couplings.get_coupling_value(2, 3) ==
            Catch::Approx(cell_conductance));
    REQUIRE(couplings.get_coupling_value(3, 4) ==
            Catch::Approx(cell_conductance));
    // Nothing was invented between the two ends.
    REQUIRE(couplings.get_coupling_value(1, 4) == Catch::Approx(0.0));
}

}  // namespace

TEST_CASE("gmm -> tmm: a strip becomes a chain of nodes and conductors",
          "[api][conduction]") {
    ThermalModel model("strip_model");
    add_strip(model);

    TmmBuildOptions options;
    options.initial_temperature = 273.15;
    const TmmBuildReport report = model.build_tmm_from_gmm(options);

    REQUIRE(report.items_processed == 1U);
    REQUIRE(report.items_skipped == 0U);
    REQUIRE(report.nodes_created == 4U);
    REQUIRE(report.conductors_created == 3U);
    REQUIRE(report.diagnostics.empty());

    // The node area is the triangulated one, set by its own call; the build
    // never triangulates.
    REQUIRE(model.tmm().nodes().get_a(1) == 0.0);
    REQUIRE(pycanha::conduction::assign_node_areas(model).accepted == 4U);

    require_strip_nodes(model);
    require_strip_conductors(model);
}

TEST_CASE("gmm -> tmm: the generated network solves at steady state",
          "[api][conduction]") {
    ThermalModel model("strip_solve");
    add_strip(model);
    REQUIRE(model.build_tmm_from_gmm().nodes_created == 4U);

    // The builder only ever writes diffusive nodes, so the boundary condition
    // is added afterwards: a sink held at 300 K, tied to one end of the strip
    // with the same conductance the strip's own cells have.
    pycanha::Node sink(0);
    sink.set_type(pycanha::BOUNDARY_NODE);
    sink.set_T(300.0);
    model.tmm().add_node(std::move(sink));
    model.tmm().add_conductive_coupling(0, 1, cell_conductance);
    model.tmm().nodes().set_qi(4, 1.0);

    auto& solver = model.solvers().steady().steady_state();
    solver.initialize();
    solver.solve();
    REQUIRE(solver.converged());

    // 1 W crosses every conductor of the chain, so the profile is linear.
    for (pycanha::NodeNum node_num = 1; node_num <= 4; ++node_num) {
        REQUIRE(
            model.tmm().nodes().get_T(node_num) ==
            Catch::Approx(300.0 + (node_num / cell_conductance)).epsilon(1e-9));
    }
    solver.deinitialize();
}
