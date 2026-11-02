# pycanha-core


![C++](https://img.shields.io/badge/C%2B%2B-23-blue)
[![ci](https://github.com/pycanha-project/pycanha-core/actions/workflows/ci.yml/badge.svg)](https://github.com/pycanha-project/pycanha-core/actions/workflows/ci.yml)
[![Code Checks](https://github.com/pycanha-project/pycanha-core/actions/workflows/code-checks.yml/badge.svg)](https://github.com/pycanha-project/pycanha-core/actions/workflows/code-checks.yml)
[![codecov](https://codecov.io/gh/pycanha-project/pycanha-core/graph/badge.svg?token=XZRHKH2G8I)](https://codecov.io/gh/pycanha-project/pycanha-core)
[![docs](https://img.shields.io/badge/doc-GitHub%20Pages-blue)](https://pycanha-project.github.io/pycanha-core/)

`pycanha-core` is the C++ core library behind pycanha. The current codebase is organized around a model-owned API for thermal network construction, parameterized model definition, solver execution, and post-processing of transient results.

## Current implementation status

The public entry point is `pycanha::ThermalModel`. It owns the main subsystems and keeps their identities stable across a workflow:

- `tmm()`: thermal network and thermal mathematical model data.
- `gmm()`: geometry model and geometric helpers.
- `parameters()`: named scalar parameters.
- `formulas()`: symbolic expressions that map parameters to model quantities such as heat loads, conductances, and capacities.
- `thermal_data()`: lookup tables and solver-generated output models.
- `solvers()`: solver registry, grouped as `steady()`, `transient()` and `sensitivity()`.
- `callbacks()`: lifecycle hooks for solver-controlled updates during transient runs.

The currently implemented public modules include:

- Thermal network primitives: nodes, couplings, conductive and radiative coupling containers, and thermal network/model wrappers.
- Parameter and formula infrastructure: entities, variables, formulas, parameter collections, and derivative-aware parameter registration.
- Solver types: `steady::SteadyState`, `transient::CrankNicolson` and `sensitivity::JacobianPropagation`, all configured through one `solvers::Settings`.
- Thermal data containers: lookup tables, dense and sparse time series, data models, and stores.
- Geometry support under `gmm`: primitives, transformations, meshes, triangulation, materials, and geometry model wrappers.
- Utilities: logging, profiling support, sparse matrix helpers, random generators, and package metadata.

## How it works

The intended workflow is:

1. Create a `ThermalModel`.
2. Add nodes and couplings to `tm.tmm()`.
3. Register named parameters in `tm.parameters()`.
4. Bind those parameters to model quantities through `tm.formulas()` and call `apply_formulas()`.
5. Retrieve a solver from `tm.solvers()`, configure it with `set_settings()`, then call `initialize()`, `solve()`, and `deinitialize()`.
6. Read temperatures directly from the model or inspect solver outputs stored in `tm.thermal_data()`.

This is the smallest complete steady-state example exercised by the API tests:

```cpp
#include "pycanha-core/pycanha-core.hpp"

pycanha::ThermalModel tm("overview_model");

pycanha::Node plate(1);
plate.set_T(300.0);
plate.set_C(100.0);

pycanha::Node sink(2);
sink.set_T(250.0);
sink.set_type(pycanha::BOUNDARY_NODE);

tm.tmm().add_node(plate);
tm.tmm().add_node(sink);
tm.tmm().add_conductive_coupling(1, 2, 0.0);

tm.parameters().add_parameter("heater_power", 50.0);
tm.parameters().add_parameter("contact_conductance", 5.0);

tm.formulas().add_formula("QI1", "heater_power");
tm.formulas().add_formula("GL(1,2)", "contact_conductance");
tm.formulas().apply_formulas();

auto& solver = tm.solvers().steady().steady_state();
solver.initialize();
solver.solve();
solver.deinitialize();
```

Transient workflows follow the same ownership model. `CrankNicolson` stores its output in `tm.thermal_data().models()`, and `callbacks()` can inject model updates during the solver loop.

## Build and runtime status

- The package is built as a static library. It is no longer a header-only distribution.
- The build requires C++23.
- Conan and CMake are the supported build entry points.
- The CI matrix currently validates Ubuntu with GCC 14, Windows with MSVC 2022, and macOS with Clang. Debug and Release builds are exercised for each entry.
- `PYCANHA_OPTION_USE_MKL` is enabled by default on supported platforms and uses dynamic MKL linking. When MKL is disabled, the code falls back to Eigen's `SparseLU` path.
- Unit and API tests are built with Catch2 and run through CTest.
- Documentation is generated with Doxygen and published to GitHub Pages on release.

## Additional documentation

- Logging and profiling details: [logging.md](logging.md)
- MKL integration details: [mkl-configuration.md](mkl-configuration.md)
- Build, tool, and workflow pages: [index.md](index.md)