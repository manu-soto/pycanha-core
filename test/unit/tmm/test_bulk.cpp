#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/solvers/model_changed_error.hpp"
#include "pycanha-core/solvers/settings.hpp"
#include "pycanha-core/solvers/steady_state.hpp"
#include "pycanha-core/tmm/bulk.hpp"
#include "pycanha-core/tmm/conductivecouplings.hpp"
#include "pycanha-core/tmm/couplingmatrices.hpp"
#include "pycanha-core/tmm/couplings.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/nodes.hpp"
#include "pycanha-core/tmm/radiativecouplings.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

// Catch2's assertion macros expand to branches, which the complexity check
// counts against every test case.
// NOLINTBEGIN(readability-function-cognitive-complexity)

namespace {

using PairValues = std::map<std::pair<NodeNum, NodeNum>, double>;

// Every stored coupling of any coupling container, keyed by its (smaller,
// larger) node numbers, which is what must survive any reordering of nodes.
template <typename CouplingSet>
[[nodiscard]] PairValues couplings_by_pair(CouplingSet& couplings) {
    PairValues pairs;
    const auto arrays = couplings.to_arrays();
    for (std::size_t entry = 0; entry < arrays.values.size(); ++entry) {
        const NodeNum first = arrays.node_1[entry];
        const NodeNum second = arrays.node_2[entry];
        pairs[{std::min(first, second), std::max(first, second)}] =
            arrays.values[entry];
    }
    return pairs;
}

// The whole observable state of a Nodes, attribute by attribute.
void require_same_nodes(const Nodes& lhs, const Nodes& rhs) {
    REQUIRE(lhs.node_numbers() == rhs.node_numbers());
    REQUIRE(lhs.get_num_diff_nodes() == rhs.get_num_diff_nodes());
    REQUIRE(lhs.T_vector == rhs.T_vector);
    REQUIRE(lhs.C_vector == rhs.C_vector);
    const auto sparse_pairs = {std::pair{&lhs.qs_vector, &rhs.qs_vector},
                               std::pair{&lhs.qa_vector, &rhs.qa_vector},
                               std::pair{&lhs.qe_vector, &rhs.qe_vector},
                               std::pair{&lhs.qi_vector, &rhs.qi_vector},
                               std::pair{&lhs.qr_vector, &rhs.qr_vector},
                               std::pair{&lhs.a_vector, &rhs.a_vector},
                               std::pair{&lhs.fx_vector, &rhs.fx_vector},
                               std::pair{&lhs.fy_vector, &rhs.fy_vector},
                               std::pair{&lhs.fz_vector, &rhs.fz_vector},
                               std::pair{&lhs.eps_vector, &rhs.eps_vector},
                               std::pair{&lhs.aph_vector, &rhs.aph_vector}};
    for (const auto& [left, right] : sparse_pairs) {
        REQUIRE(left->size() == right->size());
        REQUIRE(left->nonZeros() == right->nonZeros());
        for (Index entry = 0; entry < left->nonZeros(); ++entry) {
            REQUIRE(left->data().index(entry) == right->data().index(entry));
            REQUIRE(left->data().value(entry) == right->data().value(entry));
        }
    }
    for (const NodeNum node_num : lhs.node_numbers()) {
        REQUIRE(lhs.get_idx_from_node_num(node_num) ==
                rhs.get_idx_from_node_num(node_num));
    }
}

// A batch's columns, owned, so tests can build them and hand out spans.
struct BatchColumns {
    char type = 'D';
    std::vector<NodeNum> numbers;
    std::vector<double> temperature;
    std::vector<double> capacity;
    std::vector<double> qi;
    std::vector<double> fx;

    [[nodiscard]] NodeBatch batch() const {
        return NodeBatch{.type = type,
                         .numbers = numbers,
                         .temperature = temperature,
                         .capacity = capacity,
                         .qi = qi,
                         .fx = fx};
    }
};

// Values that exercise the storage threshold: some exact zeros, some just
// below ZERO_THR_ATTR (not stored) and some just above it (stored).
[[nodiscard]] double attribute_value(std::mt19937& rng) {
    std::uniform_int_distribution<int> kind(0, 3);
    switch (kind(rng)) {
        case 0:
            return 0.0;
        case 1:
            return 0.5 * ZERO_THR_ATTR;
        case 2:
            return 2.0 * ZERO_THR_ATTR;
        default:
            return std::uniform_real_distribution<double>(-5.0, 5.0)(rng);
    }
}

[[nodiscard]] BatchColumns random_batch(std::mt19937& rng, char type,
                                        std::vector<NodeNum> numbers,
                                        bool with_sparse) {
    std::ranges::shuffle(numbers, rng);
    BatchColumns columns;
    columns.type = type;
    columns.numbers = std::move(numbers);
    for (std::size_t entry = 0; entry < columns.numbers.size(); ++entry) {
        columns.temperature.push_back(
            std::uniform_real_distribution<double>(200.0, 300.0)(rng));
        columns.capacity.push_back(attribute_value(rng));
        if (with_sparse) {
            columns.qi.push_back(attribute_value(rng));
            columns.fx.push_back(attribute_value(rng));
        }
    }
    return columns;
}

// The same batch through the per-element path.
void add_one_by_one(Nodes& nodes, const BatchColumns& columns) {
    for (std::size_t entry = 0; entry < columns.numbers.size(); ++entry) {
        Node node(columns.numbers[entry]);
        node.set_type(columns.type);
        node.set_T(columns.temperature[entry]);
        node.set_C(columns.capacity[entry]);
        if (!columns.qi.empty()) {
            node.set_qi(columns.qi[entry]);
            node.set_fx(columns.fx[entry]);
        }
        nodes.add_node(node);
    }
}

[[nodiscard]] std::vector<NodeNum> number_range(NodeNum first, NodeNum last,
                                                NodeNum step) {
    std::vector<NodeNum> numbers;
    for (NodeNum number = first; number <= last; number += step) {
        numbers.push_back(number);
    }
    return numbers;
}

}  // namespace

// ---------------------------------------------------------------------------
// Type changes
// ---------------------------------------------------------------------------

TEST_CASE("set_types moves nodes across blocks with everything they carry",
          "[nodes][couplings][bulk]") {
    std::seed_seq seed{2026U};
    std::mt19937 rng(seed);
    const BatchColumns diffusive =
        random_batch(rng, 'D', number_range(1, 60, 1), /*with_sparse=*/true);
    const BatchColumns boundary =
        random_batch(rng, 'B', number_range(101, 105, 1), /*with_sparse=*/true);

    // Couplings between random pairs of all the nodes, so that every block
    // holds some before and after the move.
    std::vector<NodeNum> all = diffusive.numbers;
    all.insert(all.end(), boundary.numbers.begin(), boundary.numbers.end());
    std::uniform_int_distribution<std::size_t> pick(0, all.size() - 1U);
    std::vector<std::tuple<NodeNum, NodeNum, double>> pairs;
    while (pairs.size() < 300U) {
        const NodeNum first = all[pick(rng)];
        const NodeNum second = all[pick(rng)];
        if (first != second) {
            pairs.emplace_back(
                first, second,
                std::uniform_real_distribution<double>(0.1, 2.0)(rng));
        }
    }
    const auto couple = [&pairs](ThermalMathematicalModel& model) {
        for (const auto& [first, second, value] : pairs) {
            model.add_conductive_coupling(first, second, value);
            model.add_radiative_coupling(first, second, 0.5 * value);
        }
    };

    ThermalMathematicalModel model("retyped");
    REQUIRE(model.add_nodes(diffusive.batch()).accepted == 60U);
    REQUIRE(model.add_nodes(boundary.batch()).accepted == 5U);
    couple(model);
    const auto version = model.nodes().structure_version();

    const std::vector<NodeNum> to_boundary{60, 1, 30, 7, 7};
    REQUIRE(model.nodes().set_types(to_boundary, 'B').accepted == 5U);
    const std::vector<NodeNum> to_diffusive{103, 999};
    const BulkReport report = model.nodes().set_types(to_diffusive, 'D');
    REQUIRE(report.accepted == 1U);
    REQUIRE(report.rejected == 1U);
    REQUIRE(model.nodes().structure_version() != version);

    // The same nodes and couplings, created with their final types.
    BatchColumns final_diffusive;
    BatchColumns final_boundary;
    final_boundary.type = 'B';
    const std::vector<NodeNum> now_boundary{1, 7, 30, 60};
    for (const BatchColumns* source : {&diffusive, &boundary}) {
        for (std::size_t entry = 0; entry < source->numbers.size(); ++entry) {
            const NodeNum number = source->numbers[entry];
            const bool is_boundary =
                number != 103 &&
                (number > 100 || std::ranges::contains(now_boundary, number));
            BatchColumns& target =
                is_boundary ? final_boundary : final_diffusive;
            target.numbers.push_back(number);
            target.temperature.push_back(source->temperature[entry]);
            target.capacity.push_back(source->capacity[entry]);
            target.qi.push_back(source->qi[entry]);
            target.fx.push_back(source->fx[entry]);
        }
    }
    ThermalMathematicalModel reference("reference");
    static_cast<void>(reference.add_nodes(final_diffusive.batch()));
    static_cast<void>(reference.add_nodes(final_boundary.batch()));
    couple(reference);

    require_same_nodes(model.nodes(), reference.nodes());
    REQUIRE(couplings_by_pair(model.conductive_couplings()) ==
            couplings_by_pair(reference.conductive_couplings()));
    REQUIRE(couplings_by_pair(model.radiative_couplings()) ==
            couplings_by_pair(reference.radiative_couplings()));
}

TEST_CASE("set_type moves one node and keeps its container valid",
          "[nodes][bulk]") {
    ThermalMathematicalModel model("one");
    const std::vector<NodeNum> numbers{1, 2, 3};
    static_cast<void>(model.add_nodes({.type = 'D', .numbers = numbers}));
    model.add_conductive_coupling(1, 2, 2.0);
    model.add_conductive_coupling(2, 3, 3.0);

    REQUIRE(model.nodes().set_type(2, 'B'));
    REQUIRE(model.nodes().get_type(2) == 'B');
    REQUIRE(model.nodes().get_num_diff_nodes() == 2);
    REQUIRE(model.conductive_couplings().get_coupling_value(1, 2) == 2.0);
    REQUIRE(model.conductive_couplings().get_coupling_value(2, 3) == 3.0);

    // Asking for the type a node already has succeeds and changes nothing.
    REQUIRE(model.nodes().set_type(2, 'B'));
    REQUIRE_FALSE(model.nodes().set_type(9, 'B'));
    REQUIRE_FALSE(model.nodes().set_type(1, 'X'));
}

// ---------------------------------------------------------------------------
// Nodes and couplings stay in sync
// ---------------------------------------------------------------------------

TEST_CASE("Nodes::add_node refuses a number it already holds",
          "[nodes][bulk]") {
    Nodes nodes;
    Node first(1);
    Node second(2);
    nodes.add_node(first);
    nodes.add_node(second);
    Node again(1);
    nodes.add_node(again);
    Node again_as_boundary(2);
    again_as_boundary.set_type('B');
    nodes.add_node(again_as_boundary);
    REQUIRE(nodes.num_nodes() == 2);
}

TEST_CASE("couplings follow nodes inserted in the middle of their block",
          "[nodes][couplings][bulk]") {
    SECTION("through Nodes directly") {
        auto nodes = std::make_shared<Nodes>();
        ConductiveCouplings conductive(nodes);
        for (const NodeNum number : {10, 20, 30}) {
            Node node(number);
            nodes->add_node(node);
        }
        Node boundary(100);
        boundary.set_type('B');
        nodes->add_node(boundary);
        conductive.add_coupling(10, 20, 1.0);
        conductive.add_coupling(20, 30, 2.0);
        conductive.add_coupling(30, 100, 3.0);
        const PairValues before = couplings_by_pair(conductive);

        // Diffusive and boundary nodes landing before, between and after the
        // coupled ones.
        for (const NodeNum number : {5, 15, 25}) {
            Node node(number);
            nodes->add_node(node);
        }
        for (const NodeNum number : {50, 150}) {
            Node node(number);
            node.set_type('B');
            nodes->add_node(node);
        }
        REQUIRE(couplings_by_pair(conductive) == before);
        REQUIRE(conductive.get_coupling_value(30, 100) == 3.0);

        // Removal drops the node's couplings and keeps the others in place.
        nodes->remove_node(20);
        REQUIRE(couplings_by_pair(conductive) == PairValues{{{30, 100}, 3.0}});
        nodes->remove_node(50);
        REQUIRE(couplings_by_pair(conductive) == PairValues{{{30, 100}, 3.0}});
    }

    SECTION("through the thermal mathematical model, with a copy") {
        ThermalMathematicalModel tmm("sync");
        for (const NodeNum number : {10, 20, 30}) {
            tmm.add_node(Index{number});
        }
        tmm.add_conductive_coupling(10, 30, 4.0);
        tmm.add_radiative_coupling(20, 30, 0.5);
        ConductiveCouplings copy = tmm.conductive_couplings();

        tmm.add_node(Index{15});
        tmm.add_node(Index{25});
        tmm.nodes().remove_node(20);

        REQUIRE(couplings_by_pair(tmm.conductive_couplings()) ==
                PairValues{{{10, 30}, 4.0}});
        REQUIRE(couplings_by_pair(copy) == PairValues{{{10, 30}, 4.0}});
        REQUIRE(tmm.radiative_couplings().get_coupling_value(15, 25) == 0.0);
        REQUIRE(tmm.radiative_couplings().to_arrays().values.empty());
    }
}

TEST_CASE("removing a node keeps the sparse attributes aligned",
          "[nodes][bulk]") {
    Nodes nodes;
    for (const NodeNum number : {1, 2, 3, 4}) {
        Node node(number);
        node.set_qi(static_cast<double>(number));
        nodes.add_node(node);
    }
    nodes.remove_node(2);
    REQUIRE(nodes.qi_vector.size() == 3);
    REQUIRE(nodes.get_qi(1) == 1.0);
    REQUIRE(nodes.get_qi(3) == 3.0);
    REQUIRE(nodes.get_qi(4) == 4.0);
}

TEST_CASE("a conduction-only model built through Nodes solves",
          "[nodes][couplings][bulk][solver]") {
    auto tmm = std::make_shared<ThermalMathematicalModel>("conduction_only");
    for (const NodeNum number : {1, 2, 3}) {
        Node node(number);
        node.set_T(300.0);
        node.set_C(1.0);
        tmm->nodes().add_node(node);
    }
    Node sink(4);
    sink.set_type('B');
    sink.set_T(200.0);
    tmm->nodes().add_node(sink);
    tmm->conductive_couplings().add_coupling(1, 2, 1.0);
    tmm->conductive_couplings().add_coupling(2, 3, 1.0);
    tmm->conductive_couplings().add_coupling(3, 4, 1.0);
    REQUIRE(tmm->nodes().set_qi(1, 10.0));

    // The radiative matrices were never touched: the solver sizes them.
    solvers::steady::SteadyState solver(tmm);
    solvers::Settings settings;
    settings.max_iterations = 10;
    solver.set_settings(settings);
    solver.initialize();
    solver.solve();
    REQUIRE(solver.converged());
    // 10 W through three unit conductors in series.
    REQUIRE(std::abs(tmm->nodes().get_T(1) - 230.0) < 1e-6);

    // Adding a node after initialize() invalidates the solver.
    tmm->add_node(Index{5});
    tmm->nodes().set_T(1, 0.0);
    REQUIRE_THROWS_AS(solver.solve(), solvers::ModelChangedError);
    REQUIRE(tmm->nodes().get_T(1) == 0.0);
}

// ---------------------------------------------------------------------------
// Bulk node insertion
// ---------------------------------------------------------------------------

TEST_CASE("bulk node insertion equals the per-element path", "[nodes][bulk]") {
    std::seed_seq seed{38U};
    std::mt19937 rng(seed);
    for (const bool with_sparse : {false, true}) {
        Nodes bulk;
        Nodes reference;
        // Existing nodes, then batches landing below, between and above them,
        // shuffled, for both types.
        const std::vector<std::pair<char, std::vector<NodeNum>>> batches{
            {'D', number_range(100, 200, 5)},
            {'B', number_range(1000, 1050, 7)},
            {'D', number_range(1, 300, 3)},
            {'B', number_range(990, 1100, 2)},
            {'D', number_range(301, 400, 1)}};
        for (const auto& [type, numbers] : batches) {
            // Numbers already present (each block's first batch) are dropped
            // from later ones before comparing with the per-element path,
            // which would refuse them one by one.
            std::vector<NodeNum> fresh;
            std::ranges::copy_if(
                numbers, std::back_inserter(fresh),
                [&bulk](NodeNum number) { return !bulk.is_node(number); });
            const BatchColumns columns =
                random_batch(rng, type, fresh, with_sparse);
            const BulkReport report = bulk.add_nodes(columns.batch());
            REQUIRE(report.accepted == fresh.size());
            REQUIRE(report.rejected == 0U);
            add_one_by_one(reference, columns);
            require_same_nodes(bulk, reference);
        }
    }
}

TEST_CASE("bulk node insertion rejects bad entries and keeps the rest",
          "[nodes][bulk]") {
    Nodes nodes;
    const std::vector<NodeNum> existing{10, 20};
    REQUIRE(nodes.add_nodes({.type = 'D', .numbers = existing}).accepted == 2U);
    const std::vector<NodeNum> boundary{30};
    REQUIRE(nodes.add_nodes({.type = 'B', .numbers = boundary}).accepted == 1U);

    SECTION("duplicates inside the batch and against the model") {
        const std::vector<NodeNum> numbers{5, 7, 20, 7, 30, 40};
        const std::vector<double> temperature{1, 2, 3, 4, 5, 6};
        const BulkReport report = nodes.add_nodes(
            {.type = 'D', .numbers = numbers, .temperature = temperature});
        REQUIRE(report.accepted == 2U);  // 5 and 40
        REQUIRE(report.rejected == 4U);  // both 7s, 20, 30
        REQUIRE(nodes.node_numbers() ==
                std::vector<NodeNum>{5, 10, 20, 40, 30});
        REQUIRE(nodes.get_T(5) == 1.0);
        REQUIRE(nodes.get_T(40) == 6.0);
    }

    SECTION("a wrong type or a mismatched span rejects the whole batch") {
        const std::vector<NodeNum> numbers{1, 2};
        const std::vector<double> short_column{1.0};
        REQUIRE(nodes.add_nodes({.type = 'X', .numbers = numbers}).rejected ==
                2U);
        REQUIRE(nodes
                    .add_nodes(
                        {.type = 'D', .numbers = numbers, .qi = short_column})
                    .rejected == 2U);
        REQUIRE(nodes.num_nodes() == 3);
    }
}

TEST_CASE("bulk node insertion into a coupled model keeps every coupling",
          "[nodes][couplings][bulk]") {
    ThermalMathematicalModel tmm("merge");
    const std::vector<NodeNum> diffusive = number_range(10, 100, 10);
    const std::vector<NodeNum> boundary{500, 600};
    tmm.add_nodes({.type = 'D', .numbers = diffusive});
    tmm.add_nodes({.type = 'B', .numbers = boundary});
    for (std::size_t entry = 0; entry + 1 < diffusive.size(); ++entry) {
        tmm.add_conductive_coupling(diffusive[entry], diffusive[entry + 1],
                                    static_cast<double>(entry + 1));
    }
    tmm.add_conductive_coupling(100, 500, 7.5);
    tmm.add_radiative_coupling(10, 600, 0.25);
    tmm.add_radiative_coupling(500, 600, 0.5);
    const PairValues conductive = couplings_by_pair(tmm.conductive_couplings());

    std::vector<NodeNum> interleaved = number_range(5, 105, 10);
    std::ranges::reverse(interleaved);
    const BulkReport report =
        tmm.add_nodes({.type = 'D', .numbers = interleaved});
    REQUIRE(report.accepted == interleaved.size());
    const std::vector<NodeNum> more_boundary{550, 450, 700};
    REQUIRE(tmm.add_nodes({.type = 'B', .numbers = more_boundary}).accepted ==
            3U);

    REQUIRE(couplings_by_pair(tmm.conductive_couplings()) == conductive);
    REQUIRE(tmm.radiative_couplings().get_coupling_value(10, 600) == 0.25);
    REQUIRE(tmm.radiative_couplings().get_coupling_value(500, 600) == 0.5);
}

TEST_CASE("per-element insertion in node order is linear", "[nodes][bulk]") {
    // Quadratic before the append path existed: every insertion rebuilt the
    // node map and copied the sparse attributes. Not a benchmark, a guard.
    constexpr NodeNum num_nodes = 100000;
    const auto start = std::chrono::steady_clock::now();
    ThermalMathematicalModel tmm("append");
    for (NodeNum number = 1; number <= num_nodes; ++number) {
        Node node(number);
        node.set_qi(1.0);
        node.set_fx(static_cast<double>(number));
        tmm.add_node(node);
    }
    for (NodeNum number = 1; number < num_nodes; ++number) {
        tmm.add_conductive_coupling(number, number + 1, 1.0);
    }
    const std::chrono::duration<double> elapsed =
        std::chrono::steady_clock::now() - start;
    REQUIRE(tmm.nodes().num_nodes() == num_nodes);
    REQUIRE(tmm.nodes().get_fx(num_nodes) == static_cast<double>(num_nodes));
    REQUIRE(elapsed.count() < 20.0);
}

// ---------------------------------------------------------------------------
// Bulk coupling insertion
// ---------------------------------------------------------------------------

namespace {

struct CouplingInput {
    std::vector<NodeNum> node_1;
    std::vector<NodeNum> node_2;
    std::vector<double> values;
};

// Diffusive nodes 1..num_diffusive and boundary nodes 1001..1000+num_boundary.
[[nodiscard]] std::shared_ptr<Nodes> make_nodes(NodeNum num_diffusive,
                                                NodeNum num_boundary) {
    auto nodes = std::make_shared<Nodes>();
    const std::vector<NodeNum> diffusive = number_range(1, num_diffusive, 1);
    const std::vector<NodeNum> boundary =
        number_range(1001, 1000 + num_boundary, 1);
    nodes->add_nodes({.type = 'D', .numbers = diffusive});
    nodes->add_nodes({.type = 'B', .numbers = boundary});
    return nodes;
}

[[nodiscard]] CouplingInput random_couplings(std::mt19937& rng,
                                             std::size_t size,
                                             NodeNum num_diffusive,
                                             NodeNum num_boundary) {
    std::uniform_int_distribution<NodeNum> any(1, num_diffusive + num_boundary);
    const auto to_number = [num_diffusive](NodeNum draw) {
        return draw <= num_diffusive ? draw : 1000 + draw - num_diffusive;
    };
    CouplingInput input;
    while (input.values.size() < size) {
        const NodeNum first = to_number(any(rng));
        const NodeNum second = to_number(any(rng));
        if (first == second) {
            continue;
        }
        input.node_1.push_back(first);
        input.node_2.push_back(second);
        input.values.push_back(
            std::uniform_real_distribution<double>(0.1, 2.0)(rng));
    }
    return input;
}

void add_per_element(Couplings& couplings, const CouplingInput& input,
                     CouplingMerge merge) {
    for (std::size_t entry = 0; entry < input.values.size(); ++entry) {
        const Index first = input.node_1[entry];
        const Index second = input.node_2[entry];
        switch (merge) {
            case CouplingMerge::OVERWRITE:
                couplings.add_ovw_coupling(first, second, input.values[entry]);
                break;
            case CouplingMerge::SUM:
                couplings.add_sum_coupling(first, second, input.values[entry]);
                break;
            case CouplingMerge::NEW:
                couplings.add_new_coupling(first, second, input.values[entry]);
                break;
        }
    }
}

}  // namespace

TEST_CASE("bulk coupling insertion equals the per-element calls",
          "[couplings][bulk]") {
    std::seed_seq seed{1938U};
    std::mt19937 rng(seed);
    for (const CouplingMerge merge :
         {CouplingMerge::OVERWRITE, CouplingMerge::SUM, CouplingMerge::NEW}) {
        auto nodes = make_nodes(40, 6);
        Couplings bulk(nodes);
        Couplings reference(nodes);
        // Two calls, so the second one meets existing couplings; random
        // pairs repeat, so both meet duplicates inside the call too.
        for (int call = 0; call < 2; ++call) {
            const CouplingInput input = random_couplings(rng, 300, 40, 6);
            const BulkReport report = bulk.add_couplings(
                input.node_1, input.node_2, input.values, merge);
            REQUIRE(report.accepted == input.values.size());
            REQUIRE(report.rejected == 0U);
            add_per_element(reference, input, merge);
            REQUIRE(couplings_by_pair(bulk) == couplings_by_pair(reference));
        }
    }
}

TEST_CASE("sorted bulk couplings are appended as given", "[couplings][bulk]") {
    auto nodes = make_nodes(1000, 0);
    Couplings couplings(nodes);
    CouplingInput input;
    for (NodeNum node = 1; node < 1000; ++node) {
        input.node_1.push_back(node);
        input.node_2.push_back(node + 1);
        input.values.push_back(static_cast<double>(node));
        if (node + 10 <= 1000) {
            input.node_1.push_back(node + 10);  // either order is fine
            input.node_2.push_back(node);
            input.values.push_back(0.5);
        }
    }
    const BulkReport report =
        couplings.add_couplings(input.node_1, input.node_2, input.values);
    REQUIRE(report.accepted == input.values.size());
    REQUIRE(report.merged == 0U);
    const auto& matrices = couplings.get_coupling_matrices();
    REQUIRE(matrices.sparse_dd.isCompressed());
    REQUIRE(matrices.get_num_total_couplings() == to_idx(input.values.size()));
    REQUIRE(couplings.get_coupling_value(500, 501) == 500.0);
    REQUIRE(couplings.get_coupling_value(510, 500) == 0.5);

    // A second sorted call after the first one is appended as well, and a
    // call that interleaves is merged; both keep what was there.
    const std::vector<NodeNum> tail_1{999};
    const std::vector<NodeNum> tail_2{1000};
    const std::vector<double> tail_value{9.0};
    REQUIRE(couplings.add_couplings(tail_1, tail_2, tail_value).merged == 1U);
    REQUIRE(couplings.get_coupling_value(999, 1000) == 9.0);
    const std::vector<NodeNum> middle_1{3, 1};
    const std::vector<NodeNum> middle_2{7, 3};
    const std::vector<double> middle_values{1.5, 2.5};
    REQUIRE(
        couplings.add_couplings(middle_1, middle_2, middle_values).accepted ==
        2U);
    REQUIRE(couplings.get_coupling_value(3, 7) == 1.5);
    REQUIRE(couplings.get_coupling_value(1, 3) == 2.5);
    REQUIRE(couplings.get_coupling_value(500, 501) == 500.0);
}

TEST_CASE("bulk coupling insertion drops bad entries", "[couplings][bulk]") {
    auto nodes = make_nodes(3, 1);
    Couplings couplings(nodes);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    const std::vector<NodeNum> node_1{1, 1, 2, 2, 3, 99, 1};
    const std::vector<NodeNum> node_2{2, 1, 3, 3, 1001, 1, 3};
    const std::vector<double> values{1.0, 1.0, -1.0, nan, 2.0, 1.0, inf};
    const BulkReport report = couplings.add_couplings(node_1, node_2, values);
    REQUIRE(report.accepted == 2U);
    REQUIRE(report.rejected == 5U);
    REQUIRE(report.first_rejections.size() ==
            BulkReport::max_reported_rejections);
    REQUIRE(couplings_by_pair(couplings) ==
            PairValues{{{1, 2}, 1.0}, {{3, 1001}, 2.0}});

    // 64-bit numbers are read in place; one out of range is rejected, not
    // wrapped onto node 1.
    const std::vector<std::int64_t> wide_1{
        1, std::int64_t{std::numeric_limits<std::uint32_t>::max()} + 2};
    const std::vector<std::int64_t> wide_2{3, 2};
    const std::vector<double> wide_values{4.0, 4.0};
    const BulkReport wide =
        couplings.add_couplings(wide_1, wide_2, wide_values);
    REQUIRE(wide.accepted == 1U);
    REQUIRE(wide.rejected == 1U);
    REQUIRE(couplings.get_coupling_value(1, 3) == 4.0);
    REQUIRE(couplings.get_coupling_value(1, 2) == 1.0);

    const std::vector<double> short_values{1.0};
    REQUIRE(couplings.add_couplings(node_1, node_2, short_values).rejected ==
            node_1.size());
}

TEST_CASE("append_couplings writes ordered chunks by internal index",
          "[couplings][bulk]") {
    auto nodes = make_nodes(6, 2);  // internal 0..5 diffusive, 6..7 boundary
    Couplings couplings(nodes);
    // Two parts of three nodes each, given with part-local indices.
    const std::vector<std::int32_t> part_1{0, 1};
    const std::vector<std::int32_t> part_2{1, 2};
    const std::vector<double> part_values{1.0, 2.0};
    const std::vector<std::int32_t> boundary_1{2};
    const std::vector<std::int32_t> boundary_2{6};
    const std::vector<double> boundary_values{5.0};
    const std::vector<CouplingChunk> chunks{
        {.idx_1 = part_1, .idx_2 = part_2, .values = part_values, .offset = 0},
        {.idx_1 = part_1, .idx_2 = part_2, .values = part_values, .offset = 3},
        {.idx_1 = boundary_1,
         .idx_2 = boundary_2,
         .values = boundary_values,
         .offset = 0}};
    const BulkReport report = couplings.append_couplings(chunks);
    REQUIRE(report.accepted == 5U);
    REQUIRE(report.rejected == 0U);
    REQUIRE(couplings_by_pair(couplings) == PairValues{{{1, 2}, 1.0},
                                                       {{2, 3}, 2.0},
                                                       {{4, 5}, 1.0},
                                                       {{5, 6}, 2.0},
                                                       {{3, 1001}, 5.0}});

    // Out of order against what is stored: the chunk is rejected whole.
    const std::vector<CouplingChunk> late{
        {.idx_1 = part_1, .idx_2 = part_2, .values = part_values, .offset = 0}};
    const BulkReport rejected = couplings.append_couplings(late);
    REQUIRE(rejected.accepted == 0U);
    REQUIRE(rejected.rejected == 2U);
    REQUIRE(couplings.get_coupling_matrices().get_num_total_couplings() == 5);
}

// ---------------------------------------------------------------------------
// Bulk getters and setters
// ---------------------------------------------------------------------------

TEST_CASE("bulk node getters and setters", "[nodes][bulk]") {
    Nodes nodes;
    const std::vector<NodeNum> numbers{1, 2, 3, 4};
    nodes.add_nodes({.type = 'D', .numbers = numbers});

    const std::vector<NodeNum> targets{4, 2, 99, 2};
    const std::vector<double> values{4.0, 2.0, 9.0, 2.5};
    const BulkReport set = nodes.set_values(NodeAttribute::QI, targets, values);
    REQUIRE(set.accepted == 3U);
    REQUIRE(set.rejected == 1U);
    REQUIRE(nodes.get_qi(2) == 2.5);  // the last value wins
    REQUIRE(nodes.get_qi(4) == 4.0);
    REQUIRE(nodes.qi_vector.nonZeros() == 2);

    // Below the storage threshold a value is removed, not stored.
    const std::vector<NodeNum> cleared{4};
    const std::vector<double> tiny{0.5 * ZERO_THR_ATTR};
    REQUIRE(nodes.set_values(NodeAttribute::QI, cleared, tiny).accepted == 1U);
    REQUIRE(nodes.qi_vector.nonZeros() == 1);

    REQUIRE(nodes
                .set_values(NodeAttribute::T, numbers,
                            std::vector<double>{10, 20, 30, 40})
                .accepted == 4U);
    std::vector<double> read(4);
    const std::vector<NodeNum> queried{3, 1, 7, 2};
    const BulkReport get = nodes.get_values(NodeAttribute::T, queried, read);
    REQUIRE(get.rejected == 1U);
    REQUIRE(read[0] == 30.0);
    REQUIRE(read[1] == 10.0);
    REQUIRE(std::isnan(read[2]));
    REQUIRE(nodes.get_values(NodeAttribute::QI) ==
            std::vector<double>{0.0, 2.5, 0.0, 0.0});
}

TEST_CASE("bulk coupling getters, setters and arrays", "[couplings][bulk]") {
    auto nodes = make_nodes(4, 1);
    Couplings couplings(nodes);
    couplings.add_coupling(1, 2, 1.0);
    couplings.add_coupling(3, 1001, 2.0);
    couplings.add_coupling(2, 4, 3.0);

    const std::vector<NodeNum> node_1{2, 1001, 1};
    const std::vector<NodeNum> node_2{1, 3, 4};
    std::vector<double> values(3);
    REQUIRE(couplings.get_values(node_1, node_2, values).rejected == 1U);
    REQUIRE(values[0] == 1.0);
    REQUIRE(values[1] == 2.0);
    REQUIRE(std::isnan(values[2]));

    const std::vector<double> updated{5.0, 6.0, 7.0};
    const BulkReport set = couplings.set_values(node_1, node_2, updated);
    REQUIRE(set.accepted == 2U);
    REQUIRE(set.rejected == 1U);
    REQUIRE_FALSE(couplings.coupling_exists(1, 4));

    const auto arrays = couplings.to_arrays();
    REQUIRE(arrays.node_1 == std::vector<NodeNum>{1, 2, 3});
    REQUIRE(arrays.node_2 == std::vector<NodeNum>{2, 4, 1001});
    REQUIRE(arrays.values == std::vector<double>{5.0, 3.0, 6.0});
}

// NOLINTEND(readability-function-cognitive-complexity)
