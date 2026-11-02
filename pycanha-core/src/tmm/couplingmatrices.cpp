#include "pycanha-core/tmm/couplingmatrices.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "pycanha-core/globals.hpp"
#include "pycanha-core/tmm/bulk.hpp"
#include "pycanha-core/utils/SparseUtils.hpp"
#include "pycanha-core/utils/logger.hpp"

using namespace pycanha;  // NOLINT(build/namespaces)

namespace {
bool are_coupling_values_almost_equal(double val1, double val2) {
    return fabs(val1 - val2) <=
           ((fabs(val1) < fabs(val2) ? fabs(val2) : fabs(val1)) *
            ALMOST_EQUAL_COUPLING_EPSILON);
}
}  // namespace

CouplingMatrices::CouplingMatrices() = default;

inline Index CouplingMatrices::get_num_diff_nodes() const {
    return sparse_dd.rows();
}

inline Index CouplingMatrices::get_num_bound_nodes() const {
    return sparse_db.cols();
}

inline Index CouplingMatrices::get_num_nodes() const {
    return sparse_db.rows() + sparse_db.cols();
}

const Eigen::SparseMatrix<double, Eigen::RowMajor>*
CouplingMatrices::return_sparse_dd() {
    sparse_dd.makeCompressed();
    return &sparse_dd;
}

Eigen::SparseMatrix<double, Eigen::RowMajor> CouplingMatrices::get_sparse_dd()
    const {
    return sparse_dd;
}

void CouplingMatrices::add_ovw_coupling_from_node_idxs(Index idx1, Index idx2,
                                                       double val) {
    _validate_coupling_call_add_generic(
        idx1, idx2, val, &CouplingMatrices::_add_ovw_coupling_sparse);
}
void CouplingMatrices::add_ovw_coupling_from_node_idxs_verbose(Index idx1,
                                                               Index idx2,
                                                               double val) {
    _validate_coupling_call_add_generic(
        idx1, idx2, val, &CouplingMatrices::_add_ovw_coupling_sparse_verbose);
}
void CouplingMatrices::add_sum_coupling_from_node_idxs(Index idx1, Index idx2,
                                                       double val) {
    _validate_coupling_call_add_generic(
        idx1, idx2, val, &CouplingMatrices::_add_sum_coupling_sparse);
}
void CouplingMatrices::add_sum_coupling_from_node_idxs_verbose(Index idx1,
                                                               Index idx2,
                                                               double val) {
    _validate_coupling_call_add_generic(
        idx1, idx2, val, &CouplingMatrices::_add_sum_coupling_sparse_verbose);
}
void CouplingMatrices::add_new_coupling_from_node_idxs(Index idx1, Index idx2,
                                                       double val) {
    _validate_coupling_call_add_generic(
        idx1, idx2, val, &CouplingMatrices::_add_new_coupling_sparse);
}

double CouplingMatrices::get_conductor_value_from_idx(Index idx1, Index idx2) {
    auto [sp_ptr, sp_idx1, sp_idx2] = _get_sp_ptr_and_sp_idx(idx1, idx2);
    if (sp_ptr == nullptr) {
        SPDLOG_LOGGER_ERROR(pycanha::get_logger(), "Invalid indexes.");
        return nan("");
    }
    return sp_ptr->coeff(sp_idx1, sp_idx2);
}

void CouplingMatrices::set_conductor_value_from_idx(Index idx1, Index idx2,
                                                    double val) {
    // This method will change the conductor value only if it was previously
    // created

    auto [sp_ptr, sp_idx1, sp_idx2] = _get_sp_ptr_and_sp_idx(idx1, idx2);

    if (sp_ptr == nullptr) {
        SPDLOG_LOGGER_ERROR(pycanha::get_logger(),
                            "Conductor has not been set.");
        return;
    }

    if (!_validate_conductor_value(val)) {
        // TODO: Error/log handling
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "Conductor should be positive.");
        return;
    }

    // Check existance of the conductor
    if (!sparse_utils::is_trivial_zero(*sp_ptr, sp_idx1, sp_idx2)) {
        sp_ptr->coeffRef(sp_idx1, sp_idx2) = val;
    } else {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "Conductor does not exist. Value has not been set. "
                           "Add it before trying to change the value.");
    }
}

double* CouplingMatrices::get_conductor_value_ref_from_idx(Index idx1,
                                                           Index idx2) {
    // TODO: make this method const, it does not change the model

    // Obtain where the value should be
    auto [sp_ptr, sp_idx1, sp_idx2] = _get_sp_ptr_and_sp_idx(idx1, idx2);

    // Invalid indexes return nullptr
    if (sp_ptr == nullptr) {
        return nullptr;
    }

    // Get the memory address of the value of a conductor.
    if (!sparse_utils::is_trivial_zero(*sp_ptr, sp_idx1, sp_idx2)) {
        return &(sp_ptr->coeffRef(sp_idx1, sp_idx2));
    } else {
        return nullptr;
    }
}

IntAddress CouplingMatrices::get_conductor_value_address_from_idx(Index idx1,
                                                                  Index idx2) {
    // TODO: make this method const, it does not change the model
    auto* ptr = get_conductor_value_ref_from_idx(idx1, idx2);  // double*
    static_assert(sizeof(IntAddress) >= sizeof(ptr),
                  "IntAddress must be at least pointer-sized.");

    const auto addr = std::bit_cast<std::uintptr_t>(ptr);
    return static_cast<IntAddress>(addr);  // IntAddress == uint64_t
}

Eigen::SparseMatrix<double, Eigen::RowMajor>
CouplingMatrices::sparse_dd_copy() {
    sparse_dd.makeCompressed();
    return sparse_dd;
}

Eigen::SparseMatrix<double, Eigen::RowMajor>
CouplingMatrices::sparse_db_copy() {
    sparse_db.makeCompressed();
    return sparse_db;
}

Eigen::SparseMatrix<double, Eigen::RowMajor>
CouplingMatrices::sparse_bb_copy() {
    sparse_bb.makeCompressed();
    return sparse_bb;
}

Index CouplingMatrices::get_num_diff_diff_couplings() const {
    return sparse_dd.nonZeros();
}

Index CouplingMatrices::get_num_diff_bound_couplings() const {
    return sparse_db.nonZeros();
}

Index CouplingMatrices::get_num_bound_bound_couplings() const {
    return sparse_bb.nonZeros();
}

Index CouplingMatrices::get_num_total_couplings() const {
    return get_num_diff_diff_couplings() + get_num_diff_bound_couplings() +
           get_num_bound_bound_couplings();
}

std::tuple<Index, Index, double>
CouplingMatrices::get_idxs_and_coupling_value_from_coupling_idx(
    Index cidx) const {
    if (cidx < 0) {
        SPDLOG_LOGGER_ERROR(pycanha::get_logger(),
                            "Invalid coupling index: {}. Should be positive.",
                            cidx);
        return std::make_tuple(-1, -1, nan(""));
    }

    if (cidx < get_num_diff_diff_couplings()) {
        // The cidx is less than the number of dd couplings, so return value
        // from sparse_dd
        return sparse_utils::get_row_col_value_from_value_idx(sparse_dd, cidx);
    }

    cidx -= get_num_diff_diff_couplings();

    if (cidx < get_num_diff_bound_couplings()) {
        // The cidx is between dd couplings and dd + db couplings, so return
        // value from sparse_
        auto [row, col, val] =
            sparse_utils::get_row_col_value_from_value_idx(sparse_db, cidx);

        // Col need to be increased to the number of diffusive nodes
        return std::make_tuple(row, col + sparse_dd.cols(), val);
    }

    cidx -= get_num_diff_bound_couplings();
    if (cidx < get_num_bound_bound_couplings()) {
        // The cidx is between dd couplings and dd + db couplings, so return
        // value from sparse_
        auto [row, col, val] =
            sparse_utils::get_row_col_value_from_value_idx(sparse_bb, cidx);

        // Col need to be increased to the number of diffusive nodes
        return std::make_tuple(row + sparse_dd.rows(), col + sparse_dd.cols(),
                               val);
    }

    // cidx is out of bounds
    SPDLOG_LOGGER_ERROR(
        pycanha::get_logger(),
        "Invalid coupling index: {}. Index >= total num couplings.",
        cidx + get_num_total_couplings());
    return std::make_tuple(-1, -1, nan(""));
}

bool CouplingMatrices::coupling_exists_from_idxs(Index idx1, Index idx2) {
    // The function get_conductor_value_ref_from_idx return nullptr if
    //  wrong indices or if the coupling doesn't exists. Maybe is not the most
    //  efficcient way of checking this, but it works for now

    const auto* val_ptr = get_conductor_value_ref_from_idx(idx1, idx2);
    return val_ptr != nullptr;
}

void CouplingMatrices::_move_node([[maybe_unused]] Index to_idx,
                                  [[maybe_unused]] Index from_idx) {
    // TODO
}

// Writes to stdout rather than through the logger: the caller asked for the
// dump, so it must appear whatever the log thresholds are set to.
void CouplingMatrices::print_sparse() const {
    std::cout << "     Kdd matrix\n-------------------\n";
    sparse_utils::print_sparse(sparse_dd);

    std::cout << "     Kdb matrix\n-------------------\n";
    sparse_utils::print_sparse(sparse_db);

    std::cout << "     Kbb matrix\n-------------------\n";
    sparse_utils::print_sparse(sparse_bb);
}

bool CouplingMatrices::_add_ovw_coupling_sparse(
    Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse, Index sp_idx1,
    Index sp_idx2, double val) {
    const bool inserted =
        sparse_utils::is_trivial_zero(sparse, sp_idx1, sp_idx2);
    sparse.coeffRef(sp_idx1, sp_idx2) = val;
    return inserted;
}

bool CouplingMatrices::_add_ovw_coupling_sparse_verbose(
    Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse, Index sp_idx1,
    Index sp_idx2, double val) {
    if (sparse_utils::is_trivial_zero(sparse, sp_idx1, sp_idx2)) {
        sparse.coeffRef(sp_idx1, sp_idx2) = val;
        return true;
    }
    double& coupling_val = sparse.coeffRef(sp_idx1, sp_idx2);
    if (!are_coupling_values_almost_equal(coupling_val, val)) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "Duplicated coupling at indexes ({}, {}). "
                           "Overwriting old value: {} with: {}",
                           sp_idx1, sp_idx2, coupling_val, val);
    }
    coupling_val = val;
    return false;
}

bool CouplingMatrices::_add_sum_coupling_sparse(
    Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse, Index sp_idx1,
    Index sp_idx2, double val) {
    const bool inserted =
        sparse_utils::is_trivial_zero(sparse, sp_idx1, sp_idx2);
    sparse.coeffRef(sp_idx1, sp_idx2) += val;
    return inserted;
}

bool CouplingMatrices::_add_sum_coupling_sparse_verbose(
    Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse, Index sp_idx1,
    Index sp_idx2, double val) {
    const bool inserted =
        sparse_utils::is_trivial_zero(sparse, sp_idx1, sp_idx2);
    if (!inserted) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "Duplicated coupling at indexes ({}, {}). "
                           "Adding up old value: {} with: {}",
                           sp_idx1, sp_idx2, sparse.coeff(sp_idx1, sp_idx2),
                           val);
    }
    sparse.coeffRef(sp_idx1, sp_idx2) += val;
    return inserted;
}

bool CouplingMatrices::_add_new_coupling_sparse(
    Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse, Index sp_idx1,
    Index sp_idx2, double val) {
    if (!sparse_utils::is_trivial_zero(sparse, sp_idx1, sp_idx2)) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "Duplicated coupling at indexes ({}, {}). "
                           "Old value: {} left unchanged.",
                           sp_idx1, sp_idx2, sparse.coeff(sp_idx1, sp_idx2));
        return false;
    }
    sparse.coeffRef(sp_idx1, sp_idx2) = val;
    return true;
}

inline std::tuple<Eigen::SparseMatrix<double, Eigen::RowMajor>*, Index, Index>
CouplingMatrices::_get_sp_ptr_and_sp_idx(Index idx1, Index idx2) {
    if (!_validate_idxs(idx1, idx2)) {
        return {nullptr, -1, -1};
    }

    // Initialize to silence cppcoreguidelines-init-variables
    auto* sparse_ptr =
        static_cast<Eigen::SparseMatrix<double, Eigen::RowMajor>*>(nullptr);
    const Index num_diff_nodes = get_num_diff_nodes();  // no narrowing

    if (idx2 < num_diff_nodes) {
        sparse_ptr = &sparse_dd;
    } else if (idx1 < num_diff_nodes) {
        sparse_ptr = &sparse_db;
        idx2 = idx2 - num_diff_nodes;
    } else {
        sparse_ptr = &sparse_bb;
        idx2 = idx2 - num_diff_nodes;
        idx1 = idx1 - num_diff_nodes;
    }

    return {sparse_ptr, idx1, idx2};
}

inline bool CouplingMatrices::_validate_idxs(Index& idx1, Index& idx2) const {
    if (idx1 < 0 || idx2 < 0 || idx2 >= get_num_nodes() ||
        idx1 >= get_num_nodes()) {
        return false;
    }

    if (idx1 > idx2) {
        std::swap(idx1, idx2);
    }

    return true;
}

inline bool CouplingMatrices::_validate_idx(Index idx) const {
    // return !(idx < 0 || idx > get_num_nodes());
    return idx >= 0 && idx <= get_num_nodes();
}

inline bool CouplingMatrices::_validate_conductor_value(double value) {
    return value >= 0.0;
}

void CouplingMatrices::_validate_coupling_call_add_generic(
    Index idx1, Index idx2, double val, AddCouplingGeneric add_coupling_fun) {
    auto [sp_ptr, sp_idx1, sp_idx2] = _get_sp_ptr_and_sp_idx(idx1, idx2);
    if (sp_ptr == nullptr) {
        SPDLOG_LOGGER_ERROR(pycanha::get_logger(), "Invalid indexes.");
        return;
    }
    if (!_validate_conductor_value(val)) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "Coupling should be positive.");
        return;
    }

    // Call the plain function pointer (no narrowing: Index -> Index)
    if (add_coupling_fun(*sp_ptr, sp_idx1, sp_idx2, val)) {
        note_structure_change();
    }
}

// ---------------------------------------------------------------------------
// Bulk insertion
// ---------------------------------------------------------------------------

namespace {

using SpMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;

// The compressed storage of a block as spans, valid until it is resized.
[[nodiscard]] std::span<SpMat::StorageIndex> outer_of(SpMat& matrix) {
    return {matrix.outerIndexPtr(), to_sizet(matrix.outerSize() + 1)};
}
[[nodiscard]] std::span<SpMat::StorageIndex> inner_of(SpMat& matrix) {
    return {matrix.innerIndexPtr(), to_sizet(matrix.data().size())};
}
[[nodiscard]] std::span<double> values_of(SpMat& matrix) {
    return {matrix.valuePtr(), to_sizet(matrix.data().size())};
}

[[nodiscard]] bool is_valid_coupling_value(double value) noexcept {
    return std::isfinite(value) && value >= 0.0;
}

// Compresses a block and grows its storage by `count`; returns the number of
// entries it held.
[[nodiscard]] Index grow_storage(SpMat& matrix, Index count) {
    matrix.makeCompressed();
    const Index stored = matrix.nonZeros();
    matrix.resizeNonZeros(stored + count);
    return stored;
}

void note(BulkReport& report, std::string text) {
    if (report.first_rejections.size() < BulkReport::max_reported_rejections) {
        report.first_rejections.push_back(std::move(text));
    }
}

}  // namespace

CouplingMatrices::Appender::Appender(SpMat& matrix, Index count)
    : _write(grow_storage(matrix, count)),
      _outer(outer_of(matrix)),
      _inner(inner_of(matrix)),
      _values(values_of(matrix)) {}

void CouplingMatrices::Appender::push(StorageIndex row, StorageIndex col,
                                      double value) noexcept {
    if (_fill_row < 0) {
        _fill_row = row;
    }
    // Every row passed over ends where the next entry starts.
    while (_fill_row < row) {
        ++_fill_row;
        _outer[to_sizet(_fill_row)] = static_cast<StorageIndex>(_write);
    }
    _inner[to_sizet(_write)] = col;
    _values[to_sizet(_write)] = value;
    ++_write;
}

void CouplingMatrices::Appender::finish() noexcept {
    if (_fill_row < 0) {
        return;
    }
    const Index rows = to_idx(_outer.size()) - 1;
    while (_fill_row < rows) {
        ++_fill_row;
        _outer[to_sizet(_fill_row)] = static_cast<StorageIndex>(_write);
    }
}

CouplingMatrices::BlockAppenders::BlockAppenders(
    CouplingMatrices& matrices, const std::array<Index, 3>& counts) {
    for (std::size_t block_id = 0; block_id < counts.size(); ++block_id) {
        if (counts.at(block_id) > 0) {
            _appenders.at(block_id).emplace(
                matrices.block(static_cast<int>(block_id)),
                counts.at(block_id));
        }
    }
}

void CouplingMatrices::BlockAppenders::push(const Resolved& resolved,
                                            double value) {
    auto& appender = _appenders.at(to_sizet(resolved.block_id));
    if (appender.has_value()) {
        appender->push(resolved.position.row, resolved.position.col, value);
    }
}

void CouplingMatrices::BlockAppenders::finish() noexcept {
    for (auto& appender : _appenders) {
        if (appender.has_value()) {
            appender->finish();
        }
    }
}

SpMat& CouplingMatrices::block(int block_id) noexcept {
    if (block_id == 0) {
        return sparse_dd;
    }
    return block_id == 1 ? sparse_db : sparse_bb;
}

CouplingMatrices::Resolved CouplingMatrices::resolve(Index first,
                                                     Index second) const {
    const Index num_diff = sparse_dd.rows();
    const Index num_nodes = num_diff + sparse_db.cols();
    if (first < 0 || second < 0 || first >= num_nodes || second >= num_nodes ||
        first == second) {
        return {};
    }
    const Index low = std::min(first, second);
    const Index high = std::max(first, second);
    if (high < num_diff) {
        return {.block_id = 0,
                .position = {.row = static_cast<StorageIndex>(low),
                             .col = static_cast<StorageIndex>(high)}};
    }
    if (low < num_diff) {
        return {
            .block_id = 1,
            .position = {.row = static_cast<StorageIndex>(low),
                         .col = static_cast<StorageIndex>(high - num_diff)}};
    }
    return {.block_id = 2,
            .position = {.row = static_cast<StorageIndex>(low - num_diff),
                         .col = static_cast<StorageIndex>(high - num_diff)}};
}

CouplingMatrices::BlockPosition CouplingMatrices::last_stored(
    const SpMat& matrix) {
    const Index stored = matrix.nonZeros();
    if (stored == 0) {
        return {};
    }
    const std::span<const StorageIndex> outer(matrix.outerIndexPtr(),
                                              to_sizet(matrix.outerSize() + 1));
    // The last row whose start lies before the last entry.
    const auto row_end =
        std::ranges::upper_bound(outer, static_cast<StorageIndex>(stored - 1));
    const auto row =
        static_cast<StorageIndex>(std::distance(outer.begin(), row_end) - 1);
    const std::span<const StorageIndex> inner(matrix.innerIndexPtr(),
                                              to_sizet(stored));
    return {.row = row, .col = inner.back()};
}

bool CouplingMatrices::check_chunk(const CouplingChunk& chunk,
                                   std::size_t chunk_id,
                                   std::array<BlockPosition, 3>& last,
                                   std::array<Index, 3>& counts,
                                   BulkReport& report) const {
    const std::size_t size = chunk.idx_1.size();
    if (chunk.idx_2.size() != size || chunk.values.size() != size) {
        report.rejected += size;
        note(report, "chunk " + std::to_string(chunk_id) +
                         ": index and value arrays differ in length");
        return false;
    }
    std::array<BlockPosition, 3> chunk_last = last;
    std::array<Index, 3> chunk_counts{};
    for (std::size_t entry = 0; entry < size; ++entry) {
        const Resolved resolved =
            resolve(Index{chunk.offset} + chunk.idx_1[entry],
                    Index{chunk.offset} + chunk.idx_2[entry]);
        if (resolved.block_id < 0 ||
            !is_valid_coupling_value(chunk.values[entry]) ||
            !(chunk_last.at(to_sizet(resolved.block_id)) < resolved.position)) {
            report.rejected += size;
            note(report, "chunk " + std::to_string(chunk_id) + ", entry " +
                             std::to_string(entry) +
                             ": bad index or value, or out of order");
            return false;
        }
        chunk_last.at(to_sizet(resolved.block_id)) = resolved.position;
        ++chunk_counts.at(to_sizet(resolved.block_id));
    }
    last = chunk_last;
    for (std::size_t block_id = 0; block_id < counts.size(); ++block_id) {
        counts.at(block_id) += chunk_counts.at(block_id);
    }
    return true;
}

BulkReport CouplingMatrices::append_couplings(
    std::span<const CouplingChunk> chunks) {
    BulkReport report;
    std::array<BlockPosition, 3> last{};
    for (std::size_t block_id = 0; block_id < last.size(); ++block_id) {
        SpMat& matrix = block(static_cast<int>(block_id));
        matrix.makeCompressed();
        last.at(block_id) = last_stored(matrix);
    }

    // Pass 1: every chunk is checked whole, before anything is written.
    std::array<Index, 3> counts{};
    std::vector<bool> accepted(chunks.size(), false);
    for (std::size_t chunk_id = 0; chunk_id < chunks.size(); ++chunk_id) {
        accepted[chunk_id] =
            check_chunk(chunks[chunk_id], chunk_id, last, counts, report);
    }

    // Pass 2: straight into the storage.
    BlockAppenders appenders(*this, counts);
    for (std::size_t chunk_id = 0; chunk_id < chunks.size(); ++chunk_id) {
        if (!accepted[chunk_id]) {
            continue;
        }
        const CouplingChunk& chunk = chunks[chunk_id];
        for (std::size_t entry = 0; entry < chunk.idx_1.size(); ++entry) {
            appenders.push(resolve(Index{chunk.offset} + chunk.idx_1[entry],
                                   Index{chunk.offset} + chunk.idx_2[entry]),
                           chunk.values[entry]);
        }
        report.accepted += chunk.idx_1.size();
    }
    appenders.finish();
    if (report.accepted > 0) {
        note_structure_change();
    }

    if (report.rejected > 0) {
        SPDLOG_LOGGER_WARN(pycanha::get_logger(),
                           "append_couplings: {} couplings rejected, e.g. {}",
                           report.rejected, report.first_rejections.front());
    }
    return report;
}

void CouplingMatrices::BulkWriter::plan(const Resolved& resolved) {
    Block& target = _blocks.at(to_sizet(resolved.block_id));
    if (target.count == 0) {
        target.first = resolved.position;
    } else {
        target.in_order = target.in_order && target.last < resolved.position;
    }
    target.last = resolved.position;
    ++target.count;
}

void CouplingMatrices::BulkWriter::open() {
    for (std::size_t block_id = 0; block_id < _blocks.size(); ++block_id) {
        Block& target = _blocks.at(block_id);
        if (target.count == 0) {
            continue;
        }
        SpMat& matrix = _matrices->block(static_cast<int>(block_id));
        matrix.makeCompressed();
        if (target.in_order && last_stored(matrix) < target.first) {
            target.appender.emplace(matrix, target.count);
        } else {
            target.pending.reserve(to_sizet(target.count));
        }
    }
}

void CouplingMatrices::BulkWriter::put(const Resolved& resolved, double value,
                                       std::uint32_t source) {
    Block& target = _blocks.at(to_sizet(resolved.block_id));
    if (target.appender.has_value()) {
        target.appender->push(resolved.position.row, resolved.position.col,
                              value);
    } else {
        target.pending.push_back({.row = resolved.position.row,
                                  .col = resolved.position.col,
                                  .source = source});
    }
}

std::size_t CouplingMatrices::BulkWriter::finish(std::span<const double> values,
                                                 CouplingMerge merge,
                                                 BulkReport& report) {
    std::size_t taken = 0;
    Index inserted = 0;
    for (std::size_t block_id = 0; block_id < _blocks.size(); ++block_id) {
        Block& target = _blocks.at(block_id);
        if (target.appender.has_value()) {
            target.appender->finish();
            // Appended entries all come after what the block stored.
            inserted += target.count;
        } else {
            inserted +=
                merge_entries(_matrices->block(static_cast<int>(block_id)),
                              target.pending, values, merge, report);
        }
        taken += to_sizet(target.count);
    }
    if (inserted > 0) {
        _matrices->note_structure_change();
    }
    return taken;
}

void CouplingMatrices::sort_entries(std::vector<BlockEntry>& entries,
                                    Index rows) {
    // A stable counting sort on the rows, then each row on its columns.
    // Stable both times, so duplicates keep the call order.
    const auto by_position = [](const BlockEntry& lhs, const BlockEntry& rhs) {
        return lhs.row < rhs.row || (lhs.row == rhs.row && lhs.col < rhs.col);
    };
    if (std::ranges::is_sorted(entries, by_position)) {
        return;
    }
    std::vector<Index> row_start(to_sizet(rows + 1), 0);
    for (const BlockEntry& entry : entries) {
        ++row_start[to_sizet(entry.row + 1)];
    }
    std::partial_sum(row_start.begin(), row_start.end(), row_start.begin());
    std::vector<BlockEntry> sorted(entries.size());
    for (const BlockEntry& entry : entries) {
        sorted[to_sizet(row_start[to_sizet(entry.row)]++)] = entry;
    }
    entries.swap(sorted);
    auto begin = entries.begin();
    while (begin != entries.end()) {
        const auto end = std::ranges::find_if(
            begin, entries.end(), [row = begin->row](const BlockEntry& entry) {
                return entry.row != row;
            });
        std::stable_sort(begin, end, by_position);
        begin = end;
    }
}

std::vector<CouplingMatrices::RowGrowth> CouplingMatrices::count_growth(
    SpMat& matrix, std::span<const BlockEntry> entries, BulkReport& report) {
    const auto outer = outer_of(matrix);
    const auto inner = inner_of(matrix);
    std::vector<RowGrowth> growth;
    std::size_t next = 0;
    while (next < entries.size()) {
        const StorageIndex row = entries[next].row;
        Index stored = outer[to_sizet(row)];
        const Index stored_end = outer[to_sizet(row) + 1];
        Index added = 0;
        while (next < entries.size() && entries[next].row == row) {
            // A group of entries for the same column counts once.
            const StorageIndex col = entries[next].col;
            // Measured within the tail: iterators of two different spans
            // cannot be subtracted, and MSVC's debug STL asserts on it.
            const auto tail = entries.subspan(next);
            const std::size_t group_end =
                next + to_sizet(std::ranges::find_if(
                                    tail,
                                    [row, col](const BlockEntry& entry) {
                                        return entry.row != row ||
                                               entry.col != col;
                                    }) -
                                tail.begin());
            while (stored < stored_end && inner[to_sizet(stored)] < col) {
                ++stored;
            }
            const bool exists =
                stored < stored_end && inner[to_sizet(stored)] == col;
            report.merged += (group_end - next) - (exists ? 0U : 1U);
            added += exists ? 0 : 1;
            next = group_end;
        }
        growth.push_back({.row = row, .added = added});
    }
    return growth;
}

namespace {

// The value a column ends with once the group of entries for it is merged
// into what it stores, if anything.
template <typename Entry>
double merged_value(std::span<const Entry> group,
                    std::span<const double> values, CouplingMerge merge,
                    const double* existing) {
    if (merge == CouplingMerge::OVERWRITE) {
        return values[group.back().source];
    }
    if (merge == CouplingMerge::NEW) {
        return existing != nullptr ? *existing : values[group.front().source];
    }
    // The same additions, in the same order, as calling add_sum_coupling
    // once per entry.
    const double first = existing != nullptr
                             ? *existing + values[group.front().source]
                             : values[group.front().source];
    return std::accumulate(group.begin() + 1, group.end(), first,
                           [values](double sum, const Entry& entry) {
                               return sum + values[entry.source];
                           });
}

}  // namespace

Index CouplingMatrices::merge_row(SpMat& matrix,
                                  std::span<const BlockEntry> row_entries,
                                  std::span<const double> values,
                                  CouplingMerge merge, Index old_begin,
                                  Index old_end, Index write) {
    const auto inner = inner_of(matrix);
    const auto stored_values = values_of(matrix);
    Index read = old_end;
    std::size_t next = row_entries.size();
    while (next > 0 || read > old_begin) {
        const StorageIndex new_col =
            next > 0 ? row_entries[next - 1].col : StorageIndex{-1};
        const StorageIndex old_col =
            read > old_begin ? inner[to_sizet(read - 1)] : StorageIndex{-1};
        --write;
        if (old_col > new_col) {
            inner[to_sizet(write)] = old_col;
            stored_values[to_sizet(write)] = stored_values[to_sizet(read - 1)];
            --read;
            continue;
        }
        // The group of entries for this column, in call order.
        std::size_t group_begin = next;
        while (group_begin > 0 && row_entries[group_begin - 1].col == new_col) {
            --group_begin;
        }
        const auto group = row_entries.subspan(group_begin, next - group_begin);
        const bool exists = old_col == new_col;
        const double value =
            merged_value(group, values, merge,
                         exists ? &stored_values[to_sizet(read - 1)] : nullptr);
        if (exists) {
            --read;
        }
        inner[to_sizet(write)] = new_col;
        stored_values[to_sizet(write)] = value;
        next = group_begin;
    }
    return write;
}

Index CouplingMatrices::merge_entries(SpMat& matrix,
                                      std::vector<BlockEntry>& entries,
                                      std::span<const double> values,
                                      CouplingMerge merge, BulkReport& report) {
    if (entries.empty()) {
        return 0;
    }
    matrix.makeCompressed();
    sort_entries(entries, matrix.outerSize());
    const std::vector<RowGrowth> growth = count_growth(matrix, entries, report);
    const Index inserted = std::accumulate(
        growth.begin(), growth.end(), Index{0},
        [](Index sum, const RowGrowth& row) { return sum + row.added; });
    Index shift = inserted;
    matrix.resizeNonZeros(matrix.nonZeros() + shift);
    const auto outer = outer_of(matrix);
    const auto inner = inner_of(matrix);
    const auto stored_values = values_of(matrix);

    // From the last row up: rows only ever move towards the end, by the
    // growth of the rows before them, so nothing is overwritten before it is
    // read and no second buffer is needed.
    std::size_t pending_rows = growth.size();
    std::size_t entry_end = entries.size();
    for (Index row = matrix.outerSize() - 1;
         row >= 0 && (shift > 0 || pending_rows > 0); --row) {
        const Index old_begin = outer[to_sizet(row)];
        const Index old_end = outer[to_sizet(row) + 1];
        outer[to_sizet(row) + 1] = static_cast<StorageIndex>(old_end + shift);
        if (pending_rows == 0 || growth[pending_rows - 1].row != row) {
            std::copy_backward(inner.begin() + old_begin,
                               inner.begin() + old_end,
                               inner.begin() + old_end + shift);
            std::copy_backward(stored_values.begin() + old_begin,
                               stored_values.begin() + old_end,
                               stored_values.begin() + old_end + shift);
            continue;
        }
        std::size_t entry_begin = entry_end;
        while (entry_begin > 0 && entries[entry_begin - 1].row == row) {
            --entry_begin;
        }
        static_cast<void>(merge_row(
            matrix,
            std::span(entries).subspan(entry_begin, entry_end - entry_begin),
            values, merge, old_begin, old_end, old_end + shift));
        shift -= growth[pending_rows - 1].added;
        --pending_rows;
        entry_end = entry_begin;
    }
    return inserted;
}

void CouplingMatrices::reorder(std::span<const Index> old_to_new,
                               Index old_diff_count, Index diff_count,
                               Index bound_count) {
    // The old blocks are read; fresh blocks of the new sizes are written.
    std::array<SpMat, 3> old_blocks{std::move(sparse_dd), std::move(sparse_db),
                                    std::move(sparse_bb)};
    sparse_dd = SpMat(diff_count, diff_count);
    sparse_db = SpMat(diff_count, bound_count);
    sparse_bb = SpMat(bound_count, bound_count);

    // Where each old block's rows and columns start in the old order.
    const std::array<std::pair<Index, Index>, 3> offsets{
        {{0, 0}, {0, old_diff_count}, {old_diff_count, old_diff_count}}};
    const auto for_each_entry = [&](const auto& visit) {
        for (std::size_t block_id = 0; block_id < old_blocks.size();
             ++block_id) {
            const SpMat& old_block = old_blocks.at(block_id);
            const auto [row_offset, col_offset] = offsets.at(block_id);
            for (Index row = 0; row < old_block.outerSize(); ++row) {
                for (SpMat::InnerIterator entry(old_block, row); entry;
                     ++entry) {
                    visit(
                        resolve(old_to_new[to_sizet(row + row_offset)],
                                old_to_new[to_sizet(entry.col() + col_offset)]),
                        entry.value());
                }
            }
        }
    };

    // The bulk writer appends a block whose entries arrive in storage order
    // and sorts the others; its sort reads the values by position.
    std::vector<double> values;
    values.reserve(to_sizet(old_blocks[0].nonZeros() +
                            old_blocks[1].nonZeros() +
                            old_blocks[2].nonZeros()));
    BulkWriter writer(*this);
    for_each_entry([&](const Resolved& resolved, double value) {
        writer.plan(resolved);
        values.push_back(value);
    });
    writer.open();
    std::uint32_t source = 0;
    for_each_entry([&](const Resolved& resolved, double value) {
        writer.put(resolved, value, source++);
    });
    // Every stored coupling is written again into the fresh blocks, so the
    // writer bumps the structure version once when there is any.
    BulkReport report;
    static_cast<void>(writer.finish(values, CouplingMerge::OVERWRITE, report));
}

void CouplingMatrices::remap(SpMat& matrix, std::span<const Index> row_map,
                             std::span<const Index> col_map, Index rows,
                             Index cols) {
    matrix.makeCompressed();
    const auto map_row = [&row_map](Index row) {
        return row_map.empty() ? row : row_map[to_sizet(row)];
    };
    const auto map_col = [&col_map](Index col) {
        return col_map.empty() ? col : col_map[to_sizet(col)];
    };

    const auto outer = outer_of(matrix);
    const auto inner = inner_of(matrix);
    const auto values = values_of(matrix);
    std::vector<StorageIndex> new_outer(to_sizet(rows + 1), 0);
    Index write = 0;
    // Both maps are monotone, so the rows keep their order and every row
    // keeps its columns sorted: one forward compaction, never a sort.
    for (Index row = 0; row < matrix.outerSize(); ++row) {
        const Index new_row = map_row(row);
        if (new_row < 0) {
            continue;
        }
        StorageIndex kept = 0;
        for (Index read = outer[to_sizet(row)]; read < outer[to_sizet(row) + 1];
             ++read) {
            const Index new_col = map_col(inner[to_sizet(read)]);
            if (new_col < 0) {
                continue;
            }
            inner[to_sizet(write)] = static_cast<StorageIndex>(new_col);
            values[to_sizet(write)] = values[to_sizet(read)];
            ++write;
            ++kept;
        }
        new_outer[to_sizet(new_row + 1)] = kept;
    }
    std::partial_sum(new_outer.begin(), new_outer.end(), new_outer.begin());

    // Hand the compacted storage to a matrix of the new size: the index and
    // value arrays change owner, they are not copied.
    SpMat resized(rows, cols);
    resized.data().swap(matrix.data());
    resized.data().resize(write);
    std::ranges::copy(new_outer, resized.outerIndexPtr());
    matrix.swap(resized);
}
