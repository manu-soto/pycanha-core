#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/tmm/bulk.hpp"

namespace pycanha {

class Node;
class Nodes;
class ConductiveCouplings;
class RadiativeCouplings;
class CouplingMatrices;
class SteadyStateNonSymmetricSolver;

/// Structure versions of the three parts of a network: the nodes (see
/// Nodes::structure_version) and the conductive and radiative coupling
/// patterns (see CouplingMatrices::structure_version). Two equal snapshots of
/// one network mean the same nodes in the same order and the same stored
/// couplings; values may differ.
struct StructureVersions {
    std::uint64_t nodes = 0;
    std::uint64_t conductive = 0;
    std::uint64_t radiative = 0;

    [[nodiscard]] bool operator==(const StructureVersions&) const = default;
};

class ThermalNetwork {
    friend class Nodes;
    friend class CouplingMatrices;
    friend class SteadyStateNonSymmetricSolver;

  public:
    ThermalNetwork();
    ThermalNetwork(std::shared_ptr<Nodes> nodes,
                   std::shared_ptr<ConductiveCouplings> conductive,
                   std::shared_ptr<RadiativeCouplings> radiative);

    ThermalNetwork(const ThermalNetwork&) = delete;
    ThermalNetwork& operator=(const ThermalNetwork&) = delete;
    ThermalNetwork(ThermalNetwork&&) noexcept = default;
    ThermalNetwork& operator=(ThermalNetwork&&) noexcept = default;
    ~ThermalNetwork() = default;

    void add_node(Node& node);
    /// Bulk node insertion, see Nodes::add_nodes.
    BulkReport add_nodes(const NodeBatch& batch);
    void remove_node(Index node_num);

    /// Sizes both coupling containers to the current nodes (nodes appended
    /// at the end of their block grow them lazily).
    void synchronize_structure();

    /// Snapshot of the structure versions of the nodes and both coupling
    /// containers.
    [[nodiscard]] StructureVersions structure_versions() const noexcept;

    [[nodiscard]] Nodes& nodes() noexcept;
    [[nodiscard]] const Nodes& nodes() const noexcept;

    [[nodiscard]] CouplingMatrices& conductive_matrices() noexcept;
    [[nodiscard]] const CouplingMatrices& conductive_matrices() const noexcept;
    [[nodiscard]] CouplingMatrices& radiative_matrices() noexcept;
    [[nodiscard]] const CouplingMatrices& radiative_matrices() const noexcept;

    [[nodiscard]] ConductiveCouplings& conductive_couplings() noexcept;
    [[nodiscard]] const ConductiveCouplings& conductive_couplings()
        const noexcept;

    [[nodiscard]] RadiativeCouplings& radiative_couplings() noexcept;
    [[nodiscard]] const RadiativeCouplings& radiative_couplings()
        const noexcept;

    [[nodiscard]] double flow_conductive(Index node_num_1, Index node_num_2);
    [[nodiscard]] double flow_conductive(const std::vector<Index>& node_nums_1,
                                         const std::vector<Index>& node_nums_2);

    [[nodiscard]] double flow_radiative(Index node_num_1, Index node_num_2);
    [[nodiscard]] double flow_radiative(const std::vector<Index>& node_nums_1,
                                        const std::vector<Index>& node_nums_2);

    [[nodiscard]] std::shared_ptr<Nodes> nodes_ptr() noexcept;
    [[nodiscard]] std::shared_ptr<const Nodes> nodes_ptr() const noexcept;

  private:
    std::shared_ptr<Nodes> _nodes;
    std::shared_ptr<ConductiveCouplings> _conductive_couplings;
    std::shared_ptr<RadiativeCouplings> _radiative_couplings;
};

}  // namespace pycanha
