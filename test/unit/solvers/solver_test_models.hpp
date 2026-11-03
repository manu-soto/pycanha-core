#pragma once

// Small models and helpers shared by the steady-state solver tests.

#include <Eigen/Core>
#include <memory>

#include "pycanha-core/config.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/solvers/settings.hpp"
#include "pycanha-core/tmm/conductivecouplings.hpp"
#include "pycanha-core/tmm/node.hpp"
#include "pycanha-core/tmm/nodes.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"

namespace pycanha::test_models {

using Model = std::shared_ptr<ThermalMathematicalModel>;

inline constexpr int boundary_node = 100000;
inline constexpr int space_node = 100001;

// The backend of sparse storage by default in this build and on this
// machine: PARDISO with MKL, otherwise Accelerate where available, otherwise
// Eigen.
inline solvers::Backend default_sparse_backend() {
    if (MKL_ENABLED) {
        return solvers::Backend::PARDISO;
    }
    return solvers::accelerate_available() ? solvers::Backend::ACCELERATE
                                           : solvers::Backend::EIGEN;
}

// Two conductive and five radiative couplings; the expected steady state is
// known (the steady-state reference test).
inline Model make_small_radiative_model() {
    auto model = std::make_shared<ThermalMathematicalModel>("small");
    for (const int num : {10, 15, 20, 25}) {
        Node node(num);
        node.set_T(273.15);
        node.set_C(2.0e5);
        model->add_node(node);
    }
    Node env(99);
    env.set_T(3.15);
    env.set_type(BOUNDARY_NODE);
    model->add_node(env);
    model->nodes().set_qi(15, 500.0);

    model->add_conductive_coupling(10, 15, 0.1);
    model->add_conductive_coupling(20, 25, 0.1);
    model->add_radiative_coupling(10, 99, 1.0);
    model->add_radiative_coupling(20, 99, 1.0);
    model->add_radiative_coupling(15, 25, 0.2);
    model->add_radiative_coupling(15, 99, 0.8);
    model->add_radiative_coupling(25, 99, 0.8);
    return model;
}

// Every plate node radiates to space and exchanges with its diagonal
// neighbour, so the radiative block has off-diagonal entries too.
inline void add_plate_radiation(ThermalMathematicalModel& model, int n) {
    constexpr double to_space = 0.01;
    constexpr double diagonal_exchange = 0.002;
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            const int node = (row * n) + col + 1;
            model.add_radiative_coupling(node, space_node, to_space);
            if (row + 1 < n && col + 1 < n) {
                model.add_radiative_coupling(node, node + n + 1,
                                             diagonal_exchange);
            }
        }
    }
}

// Square plate of n x n diffusive nodes (numbered 1..n*n row by row), each
// edge node conductively coupled to one boundary node. Uniform dissipation.
// With radiation, add_plate_radiation() is applied too.
inline Model make_plate(int n, bool radiation) {
    auto model = std::make_shared<ThermalMathematicalModel>("plate");
    const auto node_num = [n](int row, int col) { return (row * n) + col + 1; };

    for (int k = 1; k <= n * n; ++k) {
        Node node(k);
        node.set_T(radiation ? 250.0 : 293.15);
        node.set_C(1.0);
        model->add_node(node);
    }
    Node edge(boundary_node);
    edge.set_type(BOUNDARY_NODE);
    edge.set_T(radiation ? 200.0 : 293.15);
    model->add_node(edge);
    if (radiation) {
        Node space(space_node);
        space.set_type(BOUNDARY_NODE);
        space.set_T(3.0);
        model->add_node(space);
    }

    constexpr double conductance = 0.5;
    constexpr double dissipation = 2.0;
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            const int node = node_num(row, col);
            model->nodes().set_qi(node, dissipation);
            if (col + 1 < n) {
                model->add_conductive_coupling(node, node_num(row, col + 1),
                                               conductance);
            }
            if (row + 1 < n) {
                model->add_conductive_coupling(node, node_num(row + 1, col),
                                               conductance);
            }
            if (row == 0 || col == 0 || row == n - 1 || col == n - 1) {
                model->add_conductive_coupling(node, boundary_node,
                                               conductance);
            }
        }
    }
    if (radiation) {
        add_plate_radiation(*model, n);
    }
    return model;
}

// Nodes 1 and 2 only see each other. Node 3 is coupled to a sink.
inline Model make_floating_model(bool radiation) {
    auto model = std::make_shared<ThermalMathematicalModel>("floating");
    for (const int num : {1, 2, 3}) {
        Node node(num);
        node.set_T(300.0);
        model->add_node(node);
    }
    Node sink(9);
    sink.set_type(BOUNDARY_NODE);
    sink.set_T(280.0);
    model->add_node(sink);
    model->nodes().set_qi(1, 10.0);
    model->add_conductive_coupling(1, 2, 1.0);
    model->add_conductive_coupling(3, 9, 1.0);
    if (radiation) {
        model->add_radiative_coupling(3, 9, 0.1);
    }
    return model;
}

// One diffusive node conductively coupled to a boundary node 50 K warmer:
// one iteration reaches the boundary temperature exactly, so the next one
// has a zero residual.
inline Model make_settling_model() {
    auto model = std::make_shared<ThermalMathematicalModel>("settling");
    Node node(1);
    node.set_T(250.0);
    node.set_C(1.0);
    model->add_node(node);
    Node sink(9);
    sink.set_type(BOUNDARY_NODE);
    sink.set_T(300.0);
    model->add_node(sink);
    model->add_conductive_coupling(1, 9, 1.0);
    return model;
}

inline Eigen::VectorXd temperatures(const Model& model) {
    const auto& values = model->nodes().T_vector;
    return Eigen::Map<const Eigen::VectorXd>(values.data(),
                                             static_cast<Index>(values.size()));
}

inline void reset_temperatures(const Model& model,
                               double diffusive_temperature) {
    auto& nodes = model->nodes();
    for (Index idx = 0; idx < nodes.get_num_diff_nodes(); ++idx) {
        nodes.T_vector[to_sizet(idx)] = diffusive_temperature;
    }
}

inline double max_difference(const Eigen::VectorXd& a,
                             const Eigen::VectorXd& b) {
    return (a - b).cwiseAbs().maxCoeff();
}

// State shared with the C callbacks below, which cannot capture.
struct CallbackState {
    int calls = 0;
    int coupling_node_1 = 0;
    int coupling_node_2 = 0;
    double coupling_value = 0.0;
};

inline CallbackState& callback_state() {
    static CallbackState state;
    return state;
}

// Doubles one conductive coupling after the first pass only.
inline void double_coupling_once(ThermalMathematicalModel* model) {
    auto& state = callback_state();
    ++state.calls;
    if (state.calls == 1) {
        model->conductive_couplings().set_coupling_value(
            state.coupling_node_1, state.coupling_node_2,
            2.0 * state.coupling_value);
    }
}

// Changes one conductive coupling after every pass.
inline void change_coupling_every_pass(ThermalMathematicalModel* model) {
    auto& state = callback_state();
    ++state.calls;
    model->conductive_couplings().set_coupling_value(
        state.coupling_node_1, state.coupling_node_2,
        state.coupling_value * (1.0 + (0.1 * state.calls)));
}

inline void install_callback(const Model& model,
                             void (*callback)(ThermalMathematicalModel*),
                             int node_1, int node_2) {
    auto& state = callback_state();
    state = CallbackState{};
    state.coupling_node_1 = node_1;
    state.coupling_node_2 = node_2;
    state.coupling_value =
        model->conductive_couplings().get_coupling_value(node_1, node_2);
    model->c_extern_callback_solver_loop = callback;
    model->c_callbacks_active = true;
}

}  // namespace pycanha::test_models
