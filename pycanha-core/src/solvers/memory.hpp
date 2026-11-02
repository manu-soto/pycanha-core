#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <cstdint>
#include <vector>

namespace pycanha::solvers::detail {

/// Bytes of a compressed sparse matrix: values, inner and outer indices.
template <typename Scalar, int Options, typename StorageIndex>
[[nodiscard]] std::int64_t bytes_of(
    const Eigen::SparseMatrix<Scalar, Options, StorageIndex>& matrix) {
    return (static_cast<std::int64_t>(matrix.nonZeros()) *
            static_cast<std::int64_t>(sizeof(Scalar) + sizeof(StorageIndex))) +
           ((static_cast<std::int64_t>(matrix.outerSize()) + 1) *
            static_cast<std::int64_t>(sizeof(StorageIndex)));
}

/// Bytes of a dense vector or matrix.
template <typename Derived>
[[nodiscard]] std::int64_t bytes_of(const Eigen::DenseBase<Derived>& dense) {
    return static_cast<std::int64_t>(dense.size()) *
           static_cast<std::int64_t>(sizeof(typename Derived::Scalar));
}

template <typename T>
[[nodiscard]] std::int64_t bytes_of(const std::vector<T>& values) {
    return static_cast<std::int64_t>(values.size()) *
           static_cast<std::int64_t>(sizeof(T));
}

}  // namespace pycanha::solvers::detail
