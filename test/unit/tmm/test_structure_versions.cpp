#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/tmm/bulk.hpp"
#include "pycanha-core/tmm/couplings.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/nodes.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"

namespace {

using pycanha::CouplingMerge;
using pycanha::Couplings;
using pycanha::Node;
using pycanha::NodeNum;
using pycanha::Nodes;

// Diffusive nodes 10, 20 and 40, boundary node 30.
std::shared_ptr<Nodes> make_nodes() {
    auto nodes = std::make_shared<Nodes>();
    for (const int num : {10, 20, 40}) {
        Node node(num);
        nodes->add_node(node);
    }
    Node boundary(30);
    boundary.set_type(pycanha::BOUNDARY_NODE);
    nodes->add_node(boundary);
    return nodes;
}

// Through the non-const accessor, which first sizes the matrices to nodes
// added since the last access, as every coupling call does.
std::uint64_t version_of(Couplings& couplings) {
    return couplings.get_coupling_matrices().structure_version();
}

using AddOne = void (Couplings::*)(pycanha::Index, pycanha::Index, double);

// Inserting a new pair bumps the version once; the same call on the pair
// again only changes (or keeps) its value and leaves the version alone.
bool add_bumps_only_on_insertion(AddOne add) {
    Couplings couplings(make_nodes());
    const auto before = version_of(couplings);
    (couplings.*add)(10, 20, 1.0);
    const auto after_insert = version_of(couplings);
    (couplings.*add)(20, 10, 2.0);
    (couplings.*add)(10, 30, 1.0);
    const auto after_second_insert = version_of(couplings);
    (couplings.*add)(10, 30, 3.0);
    return after_insert == before + 1U &&
           after_second_insert == after_insert + 1U &&
           version_of(couplings) == after_second_insert;
}

}  // namespace

TEST_CASE("Every single coupling add bumps the version only when it inserts",
          "[couplings][structure_version]") {
    const auto add =
        GENERATE(static_cast<AddOne>(&Couplings::add_ovw_coupling),
                 static_cast<AddOne>(&Couplings::add_ovw_coupling_verbose),
                 static_cast<AddOne>(&Couplings::add_sum_coupling),
                 static_cast<AddOne>(&Couplings::add_sum_coupling_verbose),
                 static_cast<AddOne>(&Couplings::add_new_coupling),
                 static_cast<AddOne>(&Couplings::add_coupling));
    REQUIRE(add_bumps_only_on_insertion(add));
}

TEST_CASE("Rejected coupling adds and value changes keep the version",
          "[couplings][structure_version]") {
    Couplings couplings(make_nodes());
    couplings.add_coupling(10, 20, 1.0);
    const auto version = version_of(couplings);

    couplings.add_coupling(10, 99, 1.0);   // unknown node
    couplings.add_coupling(10, 10, 1.0);   // same node twice
    couplings.add_coupling(10, 40, -1.0);  // negative value
    couplings.set_coupling_value(10, 20, 5.0);
    const std::array<NodeNum, 1> first{10};
    const std::array<NodeNum, 1> second{20};
    const std::array<double, 1> value{7.0};
    static_cast<void>(couplings.set_values(first, second, value));
    *couplings.get_coupling_value_ref(10, 20) = 8.0;

    REQUIRE(version_of(couplings) == version);
}

TEST_CASE("Bulk coupling insertion bumps the version once",
          "[couplings][structure_version]") {
    Couplings couplings(make_nodes());
    const std::array<NodeNum, 3> first{10, 10, 20};
    const std::array<NodeNum, 3> second{20, 30, 40};
    const std::array<double, 3> values{1.0, 2.0, 3.0};
    const auto before = version_of(couplings);
    static_cast<void>(couplings.add_couplings(first, second, values));
    REQUIRE(version_of(couplings) == before + 1U);

    // The same pairs again only overwrite or add up values.
    for (const auto merge :
         {CouplingMerge::OVERWRITE, CouplingMerge::SUM, CouplingMerge::NEW}) {
        static_cast<void>(
            couplings.add_couplings(first, second, values, merge));
    }
    REQUIRE(version_of(couplings) == before + 1U);

    // One new pair among existing ones takes the merge path.
    const std::array<NodeNum, 2> mixed_first{10, 20};
    const std::array<NodeNum, 2> mixed_second{20, 30};
    const std::array<double, 2> mixed_values{1.0, 1.0};
    static_cast<void>(
        couplings.add_couplings(mixed_first, mixed_second, mixed_values));
    REQUIRE(version_of(couplings) == before + 2U);
}

TEST_CASE("Bulk calls that store nothing keep the version",
          "[couplings][structure_version]") {
    Couplings couplings(make_nodes());
    const auto before = version_of(couplings);
    const std::array<NodeNum, 1> unknown{99};
    const std::array<NodeNum, 1> known{10};
    const std::array<double, 1> value{1.0};
    static_cast<void>(couplings.add_couplings(unknown, known, value));
    static_cast<void>(couplings.add_couplings(std::span<const NodeNum>{},
                                              std::span<const NodeNum>{},
                                              std::span<const double>{}));
    static_cast<void>(couplings.append_couplings({}));
    REQUIRE(version_of(couplings) == before);
}

TEST_CASE("Appending couplings by internal index bumps the version once",
          "[couplings][structure_version]") {
    Couplings couplings(make_nodes());
    // Internal order: diffusive 10, 20, 40, then boundary 30.
    const std::vector<std::int32_t> idx_1{0, 0, 1};
    const std::vector<std::int32_t> idx_2{1, 3, 2};
    const std::vector<double> values{1.0, 2.0, 3.0};
    const std::array<pycanha::CouplingChunk, 1> chunks{pycanha::CouplingChunk{
        .idx_1 = idx_1, .idx_2 = idx_2, .values = values, .offset = 0}};
    const auto before = version_of(couplings);
    static_cast<void>(couplings.append_couplings(chunks));
    REQUIRE(version_of(couplings) == before + 1U);
    REQUIRE(couplings.coupling_exists(10, 30));
}

TEST_CASE("Node changes that move stored couplings bump the version",
          "[couplings][structure_version]") {
    auto nodes = make_nodes();
    Couplings couplings(nodes);
    couplings.add_coupling(10, 20, 1.0);
    couplings.add_coupling(20, 30, 1.0);

    // Appending at the end of a block moves nothing.
    auto version = version_of(couplings);
    Node appended(50);
    nodes->add_node(appended);
    REQUIRE(version_of(couplings) == version);

    Node inserted(15);
    nodes->add_node(inserted);
    REQUIRE(version_of(couplings) > version);

    version = version_of(couplings);
    nodes->remove_node(15);
    REQUIRE(version_of(couplings) > version);

    version = version_of(couplings);
    REQUIRE(nodes->set_type(40, 'B'));
    REQUIRE(version_of(couplings) > version);
}

TEST_CASE("Replacing couplings by assignment gives a newer version",
          "[couplings][structure_version]") {
    auto nodes = make_nodes();
    Couplings target(nodes);
    Couplings source(nodes);
    source.add_coupling(10, 20, 1.0);
    target.add_coupling(10, 30, 1.0);
    target.add_coupling(20, 30, 1.0);
    const auto target_before = version_of(target);

    target = source;
    REQUIRE(version_of(target) > target_before);
    REQUIRE(version_of(target) > version_of(source));

    Couplings moved_into(nodes);
    moved_into = std::move(target);
    REQUIRE(version_of(moved_into) > target_before);
}

TEST_CASE("The model reports nodes, conductive and radiative versions apart",
          "[tmm][structure_version]") {
    pycanha::ThermalMathematicalModel model("versions");
    for (const int num : {1, 2, 3}) {
        model.add_node(num);
    }
    const auto before = model.structure_versions();

    model.add_conductive_coupling(1, 2, 1.0);
    auto now = model.structure_versions();
    REQUIRE(now.conductive == before.conductive + 1U);
    REQUIRE(now.radiative == before.radiative);

    model.add_radiative_coupling(2, 3, 0.1);
    now = model.structure_versions();
    REQUIRE(now.radiative == before.radiative + 1U);
    REQUIRE(now.nodes == before.nodes);

    model.add_node(4);
    REQUIRE(model.structure_versions().nodes != before.nodes);
    REQUIRE(model.structure_versions() == model.network().structure_versions());
}
