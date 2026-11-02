
/// C++ Implementation of ThermalConductos (TCs)
/**
 * Contain all the information regarding the thermal conductors in such a way
 * that it can be easily an efficiently readed by the solvers. The information
 * is stored in sparse matrices as defined by Eigen
 * (Eigen::SparseMatrix<double>).
 *
 */

#pragma once

#include <Eigen/Sparse>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <tuple>
#include <vector>

#include "pycanha-core/tmm/bulk.hpp"
#include "pycanha-core/tmm/nodes.hpp"

namespace pycanha {

class CouplingMatrices {
    friend class Nodes;
    friend class Couplings;
    friend class ThermalNetwork;

  public:
    // Interface
    [[nodiscard]] inline Index get_num_diff_nodes() const;
    [[nodiscard]] inline Index get_num_bound_nodes() const;
    [[nodiscard]] inline Index get_num_nodes() const;

    // Numerical values. TODO: Make private
    Eigen::SparseMatrix<double, Eigen::RowMajor> sparse_dd;
    Eigen::SparseMatrix<double, Eigen::RowMajor> sparse_db;
    Eigen::SparseMatrix<double, Eigen::RowMajor> sparse_bb;

    // Constructors
    CouplingMatrices();

    // Add couplings
    void add_ovw_coupling_from_node_idxs(
        Index idx1, Index idx2,
        double val);  ///< Add coupling, overwriting if already exists.
    void add_ovw_coupling_from_node_idxs_verbose(
        Index idx1, Index idx2,
        double val);  ///< Add coupling, overwriting if already exists. Prints
                      ///< message if overwrite.
    void add_sum_coupling_from_node_idxs(
        Index idx1, Index idx2,
        double val);  ///< Add coupling, sum the values if already exists.
    void add_sum_coupling_from_node_idxs_verbose(
        Index idx1, Index idx2,
        double val);  ///< Add coupling, sum the values if already exists.
                      ///< Prints message if sum.
    void add_new_coupling_from_node_idxs(
        Index idx1, Index idx2,
        double val);  ///< Add coupling, only if wasn't there already.

    // Get value
    double get_conductor_value_from_idx(Index idx1, Index idx2);

    // Set value
    void set_conductor_value_from_idx(Index idx1, Index idx2, double val);

    // Get value ref
    double* get_conductor_value_ref_from_idx(Index idx1, Index idx2);

    // Get address of value
    IntAddress get_conductor_value_address_from_idx(Index idx1, Index idx2);

    // Return the sparse matrix representation
    const Eigen::SparseMatrix<double, Eigen::RowMajor>* return_sparse_dd();
    [[nodiscard]] Eigen::SparseMatrix<double, Eigen::RowMajor> get_sparse_dd()
        const;

    // Return a copy of the sparse matrix
    Eigen::SparseMatrix<double, Eigen::RowMajor> sparse_dd_copy();
    Eigen::SparseMatrix<double, Eigen::RowMajor> sparse_db_copy();
    Eigen::SparseMatrix<double, Eigen::RowMajor> sparse_bb_copy();

    /// Changes whenever a coupling is inserted or removed, or the stored
    /// couplings move for a new node order; a bulk call changes it once. A
    /// new value for an existing coupling leaves it unchanged. Solvers compare
    /// it with the value seen by initialize(), because their patterns and index
    /// tables are built from the stored entries.
    [[nodiscard]] std::uint64_t structure_version() const noexcept {
        return _structure_version;
    }

    // Num couplings
    [[nodiscard]] Index get_num_diff_diff_couplings() const;
    [[nodiscard]] Index get_num_diff_bound_couplings() const;
    [[nodiscard]] Index get_num_bound_bound_couplings() const;
    [[nodiscard]] Index get_num_total_couplings() const;

    // For coupling iteration
    [[nodiscard]] std::tuple<Index, Index, double>
    get_idxs_and_coupling_value_from_coupling_idx(Index cidx) const;
    bool coupling_exists_from_idxs(Index idx1, Index idx2);

    void print_sparse() const;

    /// Specific bulk insertion by internal node index.
    /**
     * Writes the chunks straight into the compressed storage, in one pass and
     * without any temporary: no node-number lookup, no sorting, no merging.
     * The caller is in charge of the order. Within each block (dd, db, bb)
     * the entries of all the chunks, taken in turn, must be strictly
     * increasing in (lower index, higher index) and come after every
     * coupling already stored in that block; the row pointers are finalised
     * once, after the last chunk.
     *
     * That order is checked, O(k): a chunk that breaks it, or that holds an
     * index out of range, the same node twice or a negative or non-finite
     * value, is rejected whole and reported, never written. The matrices
     * must already have the size of the nodes (Couplings::append_couplings
     * sees to it).
     */
    BulkReport append_couplings(std::span<const CouplingChunk> chunks);

  private:
    using SparseRowMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;
    using StorageIndex = SparseRowMatrix::StorageIndex;

    std::uint64_t _structure_version = 0;

    void note_structure_change() noexcept { ++_structure_version; }

    // A coupling position in one block, compared in storage order.
    struct BlockPosition {
        StorageIndex row = -1;
        StorageIndex col = -1;

        [[nodiscard]] bool operator<(
            const BlockPosition& other) const noexcept {
            return row < other.row || (row == other.row && col < other.col);
        }
    };

    // A pair of internal node indices placed in its block (0 dd, 1 db, 2 bb);
    // block -1 when the pair cannot be stored.
    struct Resolved {
        int block_id = -1;
        BlockPosition position;
    };

    // One entry of a block, as (row, col) in that block, with the position
    // of its value in the caller's value array.
    struct BlockEntry {
        StorageIndex row;
        StorageIndex col;
        std::uint32_t source;
    };

    // How many entries a merge adds to one row.
    struct RowGrowth {
        StorageIndex row;
        Index added;
    };

    // Writes entries given in (row, col) order after the last stored entry
    // of a compressed block: grows the storage once, fills the row pointers
    // as the rows go by and closes them in finish().
    class Appender {
      public:
        Appender(SparseRowMatrix& matrix, Index count);
        void push(StorageIndex row, StorageIndex col, double value) noexcept;
        void finish() noexcept;

      private:
        Index _write;
        std::span<StorageIndex> _outer;
        std::span<StorageIndex> _inner;
        std::span<double> _values;
        Index _fill_row{-1};
    };

    // One appender per block that receives entries.
    class BlockAppenders {
      public:
        BlockAppenders(CouplingMatrices& matrices,
                       const std::array<Index, 3>& counts);
        void push(const Resolved& resolved, double value);
        void finish() noexcept;

      private:
        std::array<std::optional<Appender>, 3> _appenders;
    };

    // The general bulk path of Couplings::add_couplings, block by block:
    // every valid entry is first planned (to learn whether a block's entries
    // come in storage order after what it holds), then put: appended straight
    // into the block when they do, collected and merged in place otherwise.
    class BulkWriter {
      public:
        explicit BulkWriter(CouplingMatrices& matrices) noexcept
            : _matrices(&matrices) {}
        void plan(const Resolved& resolved);
        void open();
        void put(const Resolved& resolved, double value, std::uint32_t source);
        // Closes the appends and runs the merges; returns the entries taken.
        // Bumps the structure version once when a coupling that was not
        // stored before is stored.
        std::size_t finish(std::span<const double> values, CouplingMerge merge,
                           BulkReport& report);

      private:
        struct Block {
            Index count = 0;
            bool in_order = true;
            BlockPosition first;
            BlockPosition last;
            std::optional<Appender> appender;
            std::vector<BlockEntry> pending;
        };
        CouplingMatrices* _matrices;
        std::array<Block, 3> _blocks;
    };

    // Block 0 is dd, 1 db, 2 bb.
    [[nodiscard]] SparseRowMatrix& block(int block_id) noexcept;

    // Places a pair of internal node indices in its block.
    [[nodiscard]] Resolved resolve(Index first, Index second) const;

    // Last stored entry of a compressed block, or (-1, -1) when it is empty.
    [[nodiscard]] static BlockPosition last_stored(
        const SparseRowMatrix& matrix);

    // Checks one chunk of append_couplings against the order left by the
    // chunks accepted before it, and accounts for it when it passes.
    [[nodiscard]] bool check_chunk(const CouplingChunk& chunk,
                                   std::size_t chunk_id,
                                   std::array<BlockPosition, 3>& last,
                                   std::array<Index, 3>& counts,
                                   BulkReport& report) const;

    // General bulk path: sorts `entries` by (row, col), keeping call order
    // among duplicates, and merges them into the block in place, from the
    // back, after one growth of the storage. Returns how many couplings it
    // inserted (the others were already stored).
    static Index merge_entries(SparseRowMatrix& matrix,
                               std::vector<BlockEntry>& entries,
                               std::span<const double> values,
                               CouplingMerge merge, BulkReport& report);
    static void sort_entries(std::vector<BlockEntry>& entries, Index rows);
    [[nodiscard]] static std::vector<RowGrowth> count_growth(
        SparseRowMatrix& matrix, std::span<const BlockEntry> entries,
        BulkReport& report);
    // Merges one row's sorted entries with its stored ones, writing from
    // `write` down; returns where the row now starts.
    static Index merge_row(SparseRowMatrix& matrix,
                           std::span<const BlockEntry> row_entries,
                           std::span<const double> values, CouplingMerge merge,
                           Index old_begin, Index old_end, Index write);

    // Moves every stored entry to its new (row, col) under two monotone maps
    // (old index -> new index, -1 to drop it; an empty map is the identity)
    // and resizes the block. The values are never copied, only compacted.
    // Rebuilds the three blocks for a new node order that may move nodes
    // across blocks (old_to_new over all internal indices, old_diff_count
    // diffusive nodes before, diff_count and bound_count after).
    void reorder(std::span<const Index> old_to_new, Index old_diff_count,
                 Index diff_count, Index bound_count);

    static void remap(SparseRowMatrix& matrix, std::span<const Index> row_map,
                      std::span<const Index> col_map, Index rows, Index cols);

    static void _move_node(Index to_idx, Index from_idx);

    inline bool _is_thermal_nodes_valid();

    inline std::tuple<Index, Index, bool> _get_internal_node_numbers(Index i,
                                                                     Index j);

    // The single-coupling writers return whether they inserted a coupling
    // that was not stored before.
    using AddCouplingGeneric = bool (*)(
        Eigen::SparseMatrix<double, Eigen::RowMajor>&, Index, Index, double);

    // Static helpers
    static bool _add_ovw_coupling_sparse(
        Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse, Index sp_idx1,
        Index sp_idx2, double val);

    static bool _add_sum_coupling_sparse(
        Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse, Index sp_idx1,
        Index sp_idx2, double val);

    // These can also be static (they don't use `this`)
    static bool _add_ovw_coupling_sparse_verbose(
        Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse, Index sp_idx1,
        Index sp_idx2, double val);

    static bool _add_sum_coupling_sparse_verbose(
        Eigen::SparseMatrix<double, Eigen::RowMajor>& sparse, Index sp_idx1,
        Index sp_idx2, double val);

    static bool _add_new_coupling_sparse(
        Eigen::SparseMatrix<double, Eigen::RowMajor>&, Index, Index, double);

    inline std::tuple<Eigen::SparseMatrix<double, Eigen::RowMajor>*, Index,
                      Index>
    _get_sp_ptr_and_sp_idx(Index idx1, Index idx2);

    // These don’t mutate object state -> const
    inline bool _validate_idxs(Index& idx1, Index& idx2) const;
    inline bool _validate_idx(Index idx) const;

    // Doesn’t depend on object state -> static
    static inline bool _validate_conductor_value(double value);

    void _validate_coupling_call_add_generic(
        Index idx1, Index idx2, double val,
        AddCouplingGeneric add_coupling_fun);

    // TODO
    void _diff_to_bound(Index insert_position, Index int_bound_num);
};

}  // namespace pycanha
