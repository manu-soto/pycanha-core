#pragma once

#include <stdexcept>

namespace pycanha::solvers {

/// Thrown by a solver when the model's nodes or coupling patterns changed
/// after initialize(): nodes added, removed, retyped or reordered, or
/// couplings inserted or removed. The solver's sizes, matrix patterns and
/// index tables were built for the old structure, so it refuses to go on; call
/// initialize() again. New values of existing couplings, capacities, loads or
/// temperatures are not structural and never raise it.
class ModelChangedError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

}  // namespace pycanha::solvers
