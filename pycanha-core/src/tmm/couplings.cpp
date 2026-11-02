#include "pycanha-core/tmm/couplings.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/tmm/bulk.hpp"
#include "pycanha-core/tmm/coupling.hpp"
#include "pycanha-core/tmm/couplingmatrices.hpp"
#include "pycanha-core/tmm/nodes.hpp"
#include "pycanha-core/utils/SparseUtils.hpp"
#include "pycanha-core/utils/logger.hpp"

namespace pycanha {

namespace {

[[nodiscard]] std::optional<NodeNum> to_int_node_number(Index node_num) {
    if (node_num > static_cast<Index>(std::numeric_limits<NodeNum>::max()) ||
        node_num < static_cast<Index>(std::numeric_limits<NodeNum>::min())) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "Couplings: Node number {} exceeds supported "
                           "NodeNum range.",
                           node_num);
        return std::nullopt;
    }
    return static_cast<NodeNum>(node_num);
}

// The structure version of matrices replaced by `replacement`: newer than
// both, so a solver initialised with either one sees the change.
[[nodiscard]] std::uint64_t replaced_structure_version(
    std::uint64_t old_version, const CouplingMatrices& replacement) noexcept {
    return std::max(old_version, replacement.structure_version()) + 1U;
}

template <typename SparseMatrix>
void ensure_sparse_dimensions(SparseMatrix& matrix, Index rows, Index cols) {
    // Nodes appended at the end of their block only add empty rows and
    // columns, which a conservative resize adds in place; the stored entries
    // do not move.
    if (matrix.rows() != rows || matrix.cols() != cols) {
        matrix.conservativeResize(rows, cols);
    }
}

}  // namespace

Couplings::Couplings(std::shared_ptr<Nodes> nodes) noexcept
    : _nodes(std::move(nodes)) {
    if (_nodes != nullptr) {
        _nodes->add_observer(this);
    }
    synchronize_structure();
}

Couplings::~Couplings() {
    if (_nodes != nullptr) {
        _nodes->remove_observer(this);
    }
}

Couplings::Couplings(const Couplings& other)
    : _nodes(other._nodes),
      _matrices(other._matrices),
      _synced_version(other._synced_version) {
    if (_nodes != nullptr) {
        _nodes->add_observer(this);
    }
}

Couplings& Couplings::operator=(const Couplings& other) {
    if (this != &other) {
        if (_nodes != nullptr) {
            _nodes->remove_observer(this);
        }
        _nodes = other._nodes;
        const std::uint64_t version = replaced_structure_version(
            _matrices.structure_version(), other._matrices);
        _matrices = other._matrices;
        _matrices._structure_version = version;
        _synced_version = other._synced_version;
        if (_nodes != nullptr) {
            _nodes->add_observer(this);
        }
    }
    return *this;
}

Couplings::Couplings(Couplings&& other) noexcept
    : _matrices(std::move(other._matrices)),
      _synced_version(other._synced_version) {
    if (other._nodes != nullptr) {
        other._nodes->remove_observer(&other);
    }
    _nodes = std::move(other._nodes);
    if (_nodes != nullptr) {
        _nodes->add_observer(this);
    }
}

Couplings& Couplings::operator=(Couplings&& other) noexcept {
    if (this != &other) {
        if (_nodes != nullptr) {
            _nodes->remove_observer(this);
        }
        if (other._nodes != nullptr) {
            other._nodes->remove_observer(&other);
        }
        _nodes = std::move(other._nodes);
        const std::uint64_t version = replaced_structure_version(
            _matrices.structure_version(), other._matrices);
        _matrices = std::move(other._matrices);
        _matrices._structure_version = version;
        _synced_version = other._synced_version;
        if (_nodes != nullptr) {
            _nodes->add_observer(this);
        }
    }
    return *this;
}

const CouplingMatrices& Couplings::get_coupling_matrices() const noexcept {
    return _matrices;
}

CouplingMatrices& Couplings::get_coupling_matrices() noexcept {
    synchronize_structure();
    return _matrices;
}

double Couplings::get_coupling_value(Index node_num_1, Index node_num_2) {
    const auto indices = get_indices_from_node_numbers(node_num_1, node_num_2);
    if (!indices.has_value()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    return _matrices.get_conductor_value_from_idx(indices->first,
                                                  indices->second);
}

void Couplings::set_coupling_value(Index node_num_1, Index node_num_2,
                                   double value) {
    const auto indices = get_indices_from_node_numbers(node_num_1, node_num_2);
    if (!indices.has_value()) {
        return;
    }

    _matrices.set_conductor_value_from_idx(indices->first, indices->second,
                                           value);
}

void Couplings::add_ovw_coupling(Index node_num_1, Index node_num_2,
                                 double value) {
    const auto indices = get_indices_from_node_numbers(node_num_1, node_num_2);
    if (!indices.has_value()) {
        return;
    }

    _matrices.add_ovw_coupling_from_node_idxs(indices->first, indices->second,
                                              value);
}

void Couplings::add_ovw_coupling(const Coupling& coupling) {
    add_ovw_coupling(coupling.get_node_1(), coupling.get_node_2(),
                     coupling.get_value());
}

void Couplings::add_ovw_coupling_verbose(Index node_num_1, Index node_num_2,
                                         double value) {
    const auto indices = get_indices_from_node_numbers(node_num_1, node_num_2);
    if (!indices.has_value()) {
        return;
    }

    _matrices.add_ovw_coupling_from_node_idxs_verbose(indices->first,
                                                      indices->second, value);
}

void Couplings::add_ovw_coupling_verbose(const Coupling& coupling) {
    add_ovw_coupling_verbose(coupling.get_node_1(), coupling.get_node_2(),
                             coupling.get_value());
}

void Couplings::add_sum_coupling(Index node_num_1, Index node_num_2,
                                 double value) {
    const auto indices = get_indices_from_node_numbers(node_num_1, node_num_2);
    if (!indices.has_value()) {
        return;
    }

    _matrices.add_sum_coupling_from_node_idxs(indices->first, indices->second,
                                              value);
}

void Couplings::add_sum_coupling(const Coupling& coupling) {
    add_sum_coupling(coupling.get_node_1(), coupling.get_node_2(),
                     coupling.get_value());
}

void Couplings::add_sum_coupling_verbose(Index node_num_1, Index node_num_2,
                                         double value) {
    const auto indices = get_indices_from_node_numbers(node_num_1, node_num_2);
    if (!indices.has_value()) {
        return;
    }

    _matrices.add_sum_coupling_from_node_idxs_verbose(indices->first,
                                                      indices->second, value);
}

void Couplings::add_sum_coupling_verbose(const Coupling& coupling) {
    add_sum_coupling_verbose(coupling.get_node_1(), coupling.get_node_2(),
                             coupling.get_value());
}

void Couplings::add_new_coupling(Index node_num_1, Index node_num_2,
                                 double value) {
    const auto indices = get_indices_from_node_numbers(node_num_1, node_num_2);
    if (!indices.has_value()) {
        return;
    }

    _matrices.add_new_coupling_from_node_idxs(indices->first, indices->second,
                                              value);
}

void Couplings::add_new_coupling(const Coupling& coupling) {
    add_new_coupling(coupling.get_node_1(), coupling.get_node_2(),
                     coupling.get_value());
}

void Couplings::add_coupling(Index node_num_1, Index node_num_2, double value) {
    add_new_coupling(node_num_1, node_num_2, value);
}

void Couplings::add_coupling(const Coupling& coupling) {
    add_new_coupling(coupling);
}

double* Couplings::get_coupling_value_ref(Index node_num_1, Index node_num_2) {
    const auto indices = get_indices_from_node_numbers(node_num_1, node_num_2);
    if (!indices.has_value()) {
        return nullptr;
    }

    return _matrices.get_conductor_value_ref_from_idx(indices->first,
                                                      indices->second);
}

IntAddress Couplings::get_coupling_value_address(Index node_num_1,
                                                 Index node_num_2) {
    const auto indices = get_indices_from_node_numbers(node_num_1, node_num_2);
    if (!indices.has_value()) {
        return 0U;
    }

    return _matrices.get_conductor_value_address_from_idx(indices->first,
                                                          indices->second);
}

bool Couplings::coupling_exists(Index node_num_1, Index node_num_2) {
    const auto indices = get_indices_from_node_numbers(node_num_1, node_num_2);
    if (!indices.has_value()) {
        return false;
    }

    return _matrices.coupling_exists_from_idxs(indices->first, indices->second);
}

Coupling Couplings::get_coupling_from_coupling_idx(Index cidx) {
    const auto [idx1, idx2, value] =
        _matrices.get_idxs_and_coupling_value_from_coupling_idx(cidx);
    if (idx1 < 0 || idx2 < 0) {
        return {Index{-1}, Index{-1}, std::numeric_limits<double>::quiet_NaN()};
    }

    Index node_num_1 = idx1;
    Index node_num_2 = idx2;

    if (_nodes != nullptr) {
        const auto resolved_node_num_1 = _nodes->get_node_num_from_idx(idx1);
        const auto resolved_node_num_2 = _nodes->get_node_num_from_idx(idx2);

        if (!resolved_node_num_1.has_value() ||
            !resolved_node_num_2.has_value()) {
            return {Index{-1}, Index{-1},
                    std::numeric_limits<double>::quiet_NaN()};
        }

        node_num_1 = to_idx(*resolved_node_num_1);
        node_num_2 = to_idx(*resolved_node_num_2);
    }

    if (node_num_1 < 0 || node_num_2 < 0) {
        return {Index{-1}, Index{-1}, std::numeric_limits<double>::quiet_NaN()};
    }
    return {node_num_1, node_num_2, value};
}

void Couplings::synchronize_structure() {
    if (_nodes == nullptr || _synced_version == _nodes->structure_version()) {
        return;
    }

    const auto diff_count = to_idx(_nodes->_diff_node_num_vector.size());
    const auto bound_count = to_idx(_nodes->_bound_node_num_vector.size());

    ensure_sparse_dimensions(_matrices.sparse_dd, diff_count, diff_count);
    ensure_sparse_dimensions(_matrices.sparse_db, diff_count, bound_count);
    ensure_sparse_dimensions(_matrices.sparse_bb, bound_count, bound_count);
    _synced_version = _nodes->structure_version();
}

void Couplings::remap_nodes(char type, std::span<const Index> old_to_new) {
    // The remap sizes every block to the current nodes, so appends that were
    // still pending are applied on the way.
    const auto diff_count = to_idx(_nodes->_diff_node_num_vector.size());
    const auto bound_count = to_idx(_nodes->_bound_node_num_vector.size());
    const std::span<const Index> identity;
    if (type == 'D') {
        CouplingMatrices::remap(_matrices.sparse_dd, old_to_new, old_to_new,
                                diff_count, diff_count);
        CouplingMatrices::remap(_matrices.sparse_db, old_to_new, identity,
                                diff_count, bound_count);
        ensure_sparse_dimensions(_matrices.sparse_bb, bound_count, bound_count);
    } else {
        ensure_sparse_dimensions(_matrices.sparse_dd, diff_count, diff_count);
        CouplingMatrices::remap(_matrices.sparse_db, identity, old_to_new,
                                diff_count, bound_count);
        CouplingMatrices::remap(_matrices.sparse_bb, old_to_new, old_to_new,
                                bound_count, bound_count);
    }
    _matrices.note_structure_change();
    _synced_version = _nodes->structure_version();
}

void Couplings::reorder_nodes(std::span<const Index> old_to_new,
                              Index old_diff_count) {
    _matrices.reorder(old_to_new, old_diff_count,
                      to_idx(_nodes->_diff_node_num_vector.size()),
                      to_idx(_nodes->_bound_node_num_vector.size()));
    _synced_version = _nodes->structure_version();
}

// ---------------------------------------------------------------------------
// Bulk insertion
// ---------------------------------------------------------------------------

namespace {

std::string describe_pair(std::int64_t node_num_1, std::int64_t node_num_2) {
    return "(" + std::to_string(node_num_1) + ", " +
           std::to_string(node_num_2) + ")";
}

}  // namespace

template <typename NodeNumber>
BulkReport Couplings::add_couplings_impl(
    std::span<const NodeNumber> node_nums_1,
    std::span<const NodeNumber> node_nums_2, std::span<const double> values,
    CouplingMerge merge) {
    BulkReport report;
    const std::size_t count = values.size();
    if (node_nums_1.size() != count || node_nums_2.size() != count ||
        _nodes == nullptr) {
        report.rejected =
            std::max({node_nums_1.size(), node_nums_2.size(), count});
        report.first_rejections.emplace_back(
            _nodes == nullptr ? "no nodes to couple"
                              : "node and value arrays differ in length");
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "add_couplings: {} couplings rejected: {}",
                           report.rejected, report.first_rejections.front());
        return report;
    }

    synchronize_structure();
    _nodes->ensure_node_map();
    const auto index_of = [this](NodeNumber node_num) -> std::optional<Index> {
        if constexpr (!std::is_same_v<NodeNumber, NodeNum>) {
            if (std::cmp_less(node_num, std::numeric_limits<NodeNum>::min()) ||
                std::cmp_greater(node_num,
                                 std::numeric_limits<NodeNum>::max())) {
                return std::nullopt;
            }
        }
        return _nodes->lookup_node_index(static_cast<NodeNum>(node_num));
    };
    // Entry `entry` in its block, or block -1 when it cannot be stored; the
    // reason is recorded only on the first pass.
    const auto resolve = [&](std::size_t entry, bool first_pass) {
        const auto first = index_of(node_nums_1[entry]);
        const auto second = index_of(node_nums_2[entry]);
        CouplingMatrices::Resolved resolved;
        if (first.has_value() && second.has_value() &&
            std::isfinite(values[entry]) && values[entry] >= 0.0) {
            resolved = _matrices.resolve(*first, *second);
        }
        if (resolved.block_id < 0 && first_pass) {
            report.reject(
                describe_pair(node_nums_1[entry], node_nums_2[entry]) +
                ": unknown node, same node twice, or negative or "
                "non-finite value");
        }
        return resolved;
    };

    CouplingMatrices::BulkWriter writer(_matrices);
    for (std::size_t entry = 0; entry < count; ++entry) {
        const auto resolved = resolve(entry, /*first_pass=*/true);
        if (resolved.block_id >= 0) {
            writer.plan(resolved);
        }
    }
    writer.open();
    for (std::size_t entry = 0; entry < count; ++entry) {
        const auto resolved = resolve(entry, /*first_pass=*/false);
        if (resolved.block_id >= 0) {
            writer.put(resolved, values[entry],
                       static_cast<std::uint32_t>(entry));
        }
    }
    report.accepted = writer.finish(values, merge, report);

    SPDLOG_LOGGER_DEBUG(pycanha::get_logger(),
                        "add_couplings: {} couplings added ({} merged), {} "
                        "rejected",
                        report.accepted, report.merged, report.rejected);
    if (report.rejected > 0) {
        SPDLOG_LOGGER_WARN(
            pycanha::get_logger(),
            "add_couplings: {} of {} couplings rejected, e.g. {}",
            report.rejected, count, report.first_rejections.front());
    }
    if (merge == CouplingMerge::NEW && report.merged > 0) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "add_couplings: {} duplicated couplings dropped, "
                           "the first value of each kept",
                           report.merged);
    }
    return report;
}

BulkReport Couplings::add_couplings(std::span<const NodeNum> node_nums_1,
                                    std::span<const NodeNum> node_nums_2,
                                    std::span<const double> values,
                                    CouplingMerge merge) {
    return add_couplings_impl(node_nums_1, node_nums_2, values, merge);
}

BulkReport Couplings::add_couplings(std::span<const std::int64_t> node_nums_1,
                                    std::span<const std::int64_t> node_nums_2,
                                    std::span<const double> values,
                                    CouplingMerge merge) {
    return add_couplings_impl(node_nums_1, node_nums_2, values, merge);
}

BulkReport Couplings::append_couplings(std::span<const CouplingChunk> chunks) {
    synchronize_structure();
    return _matrices.append_couplings(chunks);
}

BulkReport Couplings::get_values(std::span<const NodeNum> node_nums_1,
                                 std::span<const NodeNum> node_nums_2,
                                 std::span<double> values) {
    BulkReport report;
    if (node_nums_1.size() != values.size() ||
        node_nums_2.size() != values.size() || _nodes == nullptr) {
        report.rejected = values.size();
        report.first_rejections.emplace_back(
            "node and value arrays differ in length");
        SPDLOG_LOGGER_WARN(pycanha::get_logger(), "get_values: {}",
                           report.first_rejections.front());
        return report;
    }
    synchronize_structure();
    _nodes->ensure_node_map();
    for (std::size_t entry = 0; entry < values.size(); ++entry) {
        const auto first = _nodes->lookup_node_index(node_nums_1[entry]);
        const auto second = _nodes->lookup_node_index(node_nums_2[entry]);
        const double* value =
            first.has_value() && second.has_value() && *first != *second
                ? _matrices.get_conductor_value_ref_from_idx(*first, *second)
                : nullptr;
        if (value == nullptr) {
            values[entry] = std::numeric_limits<double>::quiet_NaN();
            report.reject(
                describe_pair(node_nums_1[entry], node_nums_2[entry]) +
                ": no such coupling");
            continue;
        }
        values[entry] = *value;
        ++report.accepted;
    }
    if (report.rejected > 0) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "get_values: {} of {} couplings do not exist, e.g. "
                           "{}",
                           report.rejected, values.size(),
                           report.first_rejections.front());
    }
    return report;
}

BulkReport Couplings::set_values(std::span<const NodeNum> node_nums_1,
                                 std::span<const NodeNum> node_nums_2,
                                 std::span<const double> values) {
    BulkReport report;
    if (node_nums_1.size() != values.size() ||
        node_nums_2.size() != values.size() || _nodes == nullptr) {
        report.rejected = values.size();
        report.first_rejections.emplace_back(
            "node and value arrays differ in length");
        SPDLOG_LOGGER_WARN(pycanha::get_logger(), "set_values: {}",
                           report.first_rejections.front());
        return report;
    }
    synchronize_structure();
    _nodes->ensure_node_map();
    for (std::size_t entry = 0; entry < values.size(); ++entry) {
        const auto first = _nodes->lookup_node_index(node_nums_1[entry]);
        const auto second = _nodes->lookup_node_index(node_nums_2[entry]);
        double* value =
            first.has_value() && second.has_value() && *first != *second
                ? _matrices.get_conductor_value_ref_from_idx(*first, *second)
                : nullptr;
        if (value == nullptr || !std::isfinite(values[entry]) ||
            values[entry] < 0.0) {
            report.reject(
                describe_pair(node_nums_1[entry], node_nums_2[entry]) +
                (value == nullptr ? ": no such coupling"
                                  : ": negative or non-finite value"));
            continue;
        }
        *value = values[entry];
        ++report.accepted;
    }
    if (report.rejected > 0) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "set_values: {} of {} couplings skipped, e.g. {}",
                           report.rejected, values.size(),
                           report.first_rejections.front());
    }
    return report;
}

Couplings::CouplingArrays Couplings::to_arrays() {
    synchronize_structure();
    CouplingArrays arrays;
    if (_nodes == nullptr) {
        return arrays;
    }
    const std::vector<NodeNum> numbers = _nodes->node_numbers();
    const Index num_diff = _matrices.sparse_dd.rows();
    const auto total = to_sizet(_matrices.get_num_total_couplings());
    arrays.node_1.reserve(total);
    arrays.node_2.reserve(total);
    arrays.values.reserve(total);
    const std::array<
        std::pair<const Eigen::SparseMatrix<double, Eigen::RowMajor>*,
                  std::pair<Index, Index>>,
        3>
        blocks{{{&_matrices.sparse_dd, {0, 0}},
                {&_matrices.sparse_db, {0, num_diff}},
                {&_matrices.sparse_bb, {num_diff, num_diff}}}};
    for (const auto& [matrix, offsets] : blocks) {
        for (Index row = 0; row < matrix->outerSize(); ++row) {
            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(
                     *matrix, row);
                 it; ++it) {
                arrays.node_1.push_back(
                    numbers[to_sizet(it.row() + offsets.first)]);
                arrays.node_2.push_back(
                    numbers[to_sizet(it.col() + offsets.second)]);
                arrays.values.push_back(it.value());
            }
        }
    }
    return arrays;
}

std::optional<std::pair<Index, Index>> Couplings::get_indices_from_node_numbers(
    Index node_num_1, Index node_num_2) {
    if (_nodes == nullptr) {
        SPDLOG_LOGGER_WARN(
            pycanha::get_logger(),
            "Couplings::get_indices_from_node_numbers called with "
            "null Nodes pointer.");
        return std::nullopt;
    }

    synchronize_structure();

    const auto node_num_1_int = to_int_node_number(node_num_1);
    const auto node_num_2_int = to_int_node_number(node_num_2);
    if (!node_num_1_int.has_value() || !node_num_2_int.has_value()) {
        return std::nullopt;
    }

    auto idx1 = _nodes->get_idx_from_node_num(*node_num_1_int);
    auto idx2 = _nodes->get_idx_from_node_num(*node_num_2_int);

    if (!idx1.has_value() || !idx2.has_value()) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "Couplings: Invalid node numbers {}, {}", node_num_1,
                           node_num_2);
        return std::nullopt;
    }

    if (*idx1 == *idx2) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "Couplings: Node numbers correspond to the same "
                           "node.");
        return std::nullopt;
    }

    if (*idx1 > *idx2) {
        std::swap(idx1, idx2);
    }

    return std::pair<Index, Index>{*idx1, *idx2};
}

std::tuple<Index, Index, Eigen::SparseMatrix<double, Eigen::RowMajor>*>
Couplings::get_indices_and_sparse_from_node_numbers(Index node_num_1,
                                                    Index node_num_2) {
    const auto indices = get_indices_from_node_numbers(node_num_1, node_num_2);
    if (!indices.has_value()) {
        return {Index{-1}, Index{-1}, nullptr};
    }

    auto sp_idx1 = indices->first;
    auto sp_idx2 = indices->second;
    auto* sparse_ptr =
        static_cast<Eigen::SparseMatrix<double, Eigen::RowMajor>*>(nullptr);

    const auto num_diff_nodes = _matrices.sparse_dd.rows();

    if (sp_idx2 < num_diff_nodes) {
        sparse_ptr = &_matrices.sparse_dd;
    } else if (sp_idx1 < num_diff_nodes) {
        sparse_ptr = &_matrices.sparse_db;
        sp_idx2 -= num_diff_nodes;
    } else {
        sparse_ptr = &_matrices.sparse_bb;
        sp_idx1 -= num_diff_nodes;
        sp_idx2 -= num_diff_nodes;
    }

    return {sp_idx1, sp_idx2, sparse_ptr};
}

bool Couplings::is_coupling_trivial_zero_from_node_numbers(Index node_num_1,
                                                           Index node_num_2) {
    auto [idx1, idx2, sparse_ptr] =
        get_indices_and_sparse_from_node_numbers(node_num_1, node_num_2);
    if (sparse_ptr == nullptr) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "Couplings: Invalid sparse pointer for nodes "
                           "{}, {}",
                           node_num_1, node_num_2);
        return false;
    }

    return sparse_utils::is_trivial_zero(*sparse_ptr, idx1, idx2);
}

}  // namespace pycanha
