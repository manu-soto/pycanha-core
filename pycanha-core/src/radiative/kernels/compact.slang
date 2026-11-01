// Tiled exchange: one block's accumulator rows compacted on the device, so
// that the host reads back the non-zero cells instead of the whole block.
//
// A Tiled block is tile_rows x (num_faces + 3) cells. Read back whole, a
// 375 000-face scene moved its full Nf x Nf matrix through the PCIe bus and
// the host's caches once per trace (1.1 TB in 1 051 blocks of 1 GiB, ~3.5 min
// on an RTX 4070 whatever the ray count), where the non-zero cells of all its
// rows are a few hundred million. Here each emitter row of the block is
// scanned by one workgroup in column order:
//
//   count pass (flags without FLAG_COMPACT_WRITE):
//       counts[e] = number of non-zero cells in emitter e's row
//   write pass (FLAG_COMPACT_WRITE): the host has replaced counts[e] by the
//       row's first output slot (an exclusive prefix over the block's rows);
//       the k-th non-zero cell of the row, in ascending column, writes
//       entries[2 (first + k)] = column and entries[2 (first + k) + 1] = cell.
//
// Columns stay in ascending order within a row, so the host merges the
// entries exactly as it merged the dense row (HostCountRow::add_sparse), and
// the result is bit-identical to the dense readback. The scan within a row
// goes through workgroup shared memory, not subgroup operations, so it does
// not depend on how a device lays subgroups out in a workgroup.

import common;

[[vk::binding(9, 0)]] RWStructuredBuffer<uint64_t> cells;
[[vk::binding(10, 0)]] RWStructuredBuffer<uint> counts;
[[vk::binding(11, 0)]] RWStructuredBuffer<uint64_t> entries;

static const uint GROUP = 256u;
static const uint FLAG_COMPACT_WRITE = 64u;

groupshared uint scan[GROUP];

[shader("compute")]
[numthreads(256, 1, 1)]
void csMain(uint3 group: SV_GroupID, uint3 local: SV_GroupThreadID) {
    uint e = group.y;
    if (e >= pc.num_emitters) {
        return;  // the whole workgroup: control flow stays uniform
    }
    uint t = local.x;
    uint cols = pc.num_faces + 3u;
    uint row_base = (emitters[e] - pc.row_offset) * cols;
    bool write = (pc.flags & FLAG_COMPACT_WRITE) != 0u;
    uint first = write ? counts[e] : 0u;
    uint total = 0u;
    for (uint c0 = 0u; c0 < cols; c0 += GROUP) {
        uint c = c0 + t;
        uint64_t cell = c < cols ? cells[row_base + c] : uint64_t(0);
        uint flag = cell != uint64_t(0) ? 1u : 0u;
        // Inclusive prefix sum of the flags over the workgroup (Hillis-Steele).
        scan[t] = flag;
        GroupMemoryBarrierWithGroupSync();
        for (uint d = 1u; d < GROUP; d <<= 1u) {
            uint add = t >= d ? scan[t - d] : 0u;
            GroupMemoryBarrierWithGroupSync();
            scan[t] += add;
            GroupMemoryBarrierWithGroupSync();
        }
        if (write && flag != 0u) {
            uint k = first + total + scan[t] - 1u;
            entries[2u * k] = uint64_t(c);
            entries[2u * k + 1u] = cell;
        }
        total += scan[GROUP - 1u];
        GroupMemoryBarrierWithGroupSync();  // scan[] is rewritten next chunk
    }
    if (!write && t == 0u) {
        counts[e] = total;
    }
}
