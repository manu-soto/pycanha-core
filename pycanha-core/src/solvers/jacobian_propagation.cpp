#include "pycanha-core/solvers/jacobian_propagation.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "linearized_network.hpp"
#include "memory.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/parameters/entity.hpp"
#include "pycanha-core/parameters/formula.hpp"
#include "pycanha-core/parameters/formulas.hpp"
#include "pycanha-core/solvers/crank_nicolson.hpp"
#include "pycanha-core/solvers/report.hpp"
#include "pycanha-core/solvers/settings.hpp"
#include "pycanha-core/solvers/solver.hpp"
#include "pycanha-core/thermaldata/data_model.hpp"
#include "pycanha-core/thermaldata/dense_matrix_time_series.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/utils/logger.hpp"
#include "pycanha-core/utils/profiling.hpp"

namespace pycanha::solvers::sensitivity {

namespace {

[[nodiscard]] bool is_heat_flux_entity(EntityType type) {
    return type == EntityType::qs || type == EntityType::qa ||
           type == EntityType::qe || type == EntityType::qi ||
           type == EntityType::qr;
}

[[nodiscard]] Index require_node_index(const Nodes& nodes, NodeNum node_num) {
    const auto resolved = nodes.get_idx_from_node_num(node_num);
    if (!resolved.has_value()) {
        throw std::invalid_argument("Jacobian references a missing node");
    }
    return *resolved;
}

}  // namespace

JacobianPropagation::JacobianPropagation(
    std::shared_ptr<ThermalMathematicalModel> tmm_shptr)
    : CrankNicolson(std::move(tmm_shptr), SolverType::JACOBIAN_PROPAGATION) {}

void JacobianPropagation::prepare(bool log_warnings) {
    collect_parameter_names();
    CrankNicolson::prepare(log_warnings);
}

SolverMemory JacobianPropagation::memory_now() const {
    using detail::bytes_of;
    SolverMemory memory = CrankNicolson::memory_now();
    const auto parameters =
        static_cast<std::int64_t>(_derivative_parameter_names.size());
    // _d_capacity_matrix, _d_heat_flux_matrix, _m_k, _m_q, _m_c, _mt and _mb
    // are nd x parameters, also before initialize() sizes them.
    constexpr std::int64_t dense_matrices = 7;
    const std::int64_t dense_bytes = static_cast<std::int64_t>(nd) *
                                     parameters *
                                     static_cast<std::int64_t>(sizeof(double));
    memory.matrix += dense_matrices * dense_bytes;
    for (const auto* matrices : {&_d_kl_dd_matrices, &_d_kl_db_matrices,
                                 &_d_kr_dd_matrices, &_d_kr_db_matrices}) {
        for (const SpMatRow& matrix : *matrices) {
            memory.matrix += bytes_of(matrix);
        }
    }
    // The JAC attribute holds one nd x parameters sample per output.
    memory.output += static_cast<std::int64_t>(num_outputs) *
                     (dense_bytes + static_cast<std::int64_t>(sizeof(double)));
    return memory;
}

void JacobianPropagation::initialize() {
    // The selection may have changed since an estimate_memory().
    collect_parameter_names();
    CrankNicolson::initialize();

    auto& jacobian =
        output_model().get_matrix_attribute(DataModelAttribute::JAC);
    jacobian =
        DenseMatrixTimeSeries(nd, to_idx(_derivative_parameter_names.size()));
    jacobian.reserve(to_sizet(num_outputs));

    _d_kl_dd_matrices.clear();
    _d_kl_db_matrices.clear();
    _d_kr_dd_matrices.clear();
    _d_kr_db_matrices.clear();

    _d_kl_dd_matrices.reserve(_derivative_parameter_names.size());
    _d_kl_db_matrices.reserve(_derivative_parameter_names.size());
    _d_kr_dd_matrices.reserve(_derivative_parameter_names.size());
    _d_kr_db_matrices.reserve(_derivative_parameter_names.size());

    for (std::size_t parameter_index = 0;
         parameter_index < _derivative_parameter_names.size();
         ++parameter_index) {
        _d_kl_dd_matrices.emplace_back(nd, nd);
        _d_kl_db_matrices.emplace_back(nd, nb);
        _d_kr_dd_matrices.emplace_back(nd, nd);
        _d_kr_db_matrices.emplace_back(nd, nb);
    }

    _d_capacity_matrix =
        DenseJacobian::Zero(nd, to_idx(_derivative_parameter_names.size()));
    _d_heat_flux_matrix =
        DenseJacobian::Zero(nd, to_idx(_derivative_parameter_names.size()));

    _m_k = DenseJacobian::Zero(nd, to_idx(_derivative_parameter_names.size()));
    _m_q = DenseJacobian::Zero(nd, to_idx(_derivative_parameter_names.size()));
    _m_c = DenseJacobian::Zero(nd, to_idx(_derivative_parameter_names.size()));
    _mt = DenseJacobian::Zero(nd, to_idx(_derivative_parameter_names.size()));
    _mb = DenseJacobian::Zero(nd, to_idx(_derivative_parameter_names.size()));

    std::unordered_map<std::string, Index> parameter_lookup;
    parameter_lookup.reserve(_derivative_parameter_names.size());
    for (std::size_t parameter_index = 0;
         parameter_index < _derivative_parameter_names.size();
         ++parameter_index) {
        parameter_lookup.emplace(_derivative_parameter_names[parameter_index],
                                 to_idx(parameter_index));
    }

    for (const auto& formula : tmm.formulas().formulas()) {
        auto* parameter_formula =
            dynamic_cast<ParameterFormula*>(formula.get());
        if (parameter_formula == nullptr) {
            continue;
        }

        parameter_formula->calculate_derivatives();
        const auto* derivatives = formula->get_derivative_values();
        if ((derivatives == nullptr) || derivatives->empty()) {
            continue;
        }

        const auto& dependencies = formula->parameter_dependencies();
        if (dependencies.size() != derivatives->size()) {
            throw std::invalid_argument(
                "Formula derivative metadata size mismatch for " +
                formula->entity().string_representation());
        }

        for (std::size_t dependency_index = 0;
             dependency_index < dependencies.size(); ++dependency_index) {
            const auto search =
                parameter_lookup.find(dependencies[dependency_index]);
            if (search == parameter_lookup.end()) {
                continue;
            }

            fill_matrices(formula->entity(), search->second,
                          derivatives->at(dependency_index));
        }
    }
}

void JacobianPropagation::collect_parameter_names() {
    _parameter_names.clear();
    _derivative_parameter_names.clear();

    std::unordered_set<std::string> seen_parameters;
    for (const auto& formula : tmm.formulas().formulas()) {
        if (dynamic_cast<ParameterFormula*>(formula.get()) == nullptr) {
            continue;
        }

        const auto& dependencies = formula->parameter_dependencies();

        std::ranges::copy_if(
            dependencies, std::back_inserter(_parameter_names),
            [&seen_parameters](const std::string& dependency) {
                return seen_parameters.insert(dependency).second;
            });
    }

    for (const auto& parameter_name :
         tmm.formulas().parameters_with_derivatives().parameter_names()) {
        const auto found = std::ranges::find(_parameter_names, parameter_name);
        if (found == _parameter_names.end()) {
            throw std::invalid_argument(
                "Derivative parameter '" + parameter_name +
                "' is not produced by any ParameterFormula");
        }
        _derivative_parameter_names.push_back(parameter_name);
    }
    if (_derivative_parameter_names.empty()) {
        throw std::invalid_argument(
            "JacobianPropagation requires at least one selected derivative "
            "parameter");
    }
}

void JacobianPropagation::fill_matrices(const Entity& entity,
                                        Index parameter_index,
                                        double derivative_value) {
    const auto type = entity.type();

    if (type == EntityType::gl || type == EntityType::gr) {
        const Index index_1 = require_node_index(tmm.nodes(), entity.node_1());
        const Index index_2 = require_node_index(tmm.nodes(), entity.node_2());

        const bool first_is_domain = index_1 < nd;
        const bool second_is_domain = index_2 < nd;

        if (first_is_domain && second_is_domain) {
            const auto row = std::min(index_1, index_2);
            const auto col = std::max(index_1, index_2);
            auto& matrix = (type == EntityType::gl)
                               ? _d_kl_dd_matrices[to_sizet(parameter_index)]
                               : _d_kr_dd_matrices[to_sizet(parameter_index)];
            matrix.coeffRef(row, col) += derivative_value;
            return;
        }

        if (first_is_domain != second_is_domain) {
            const auto diff_index = first_is_domain ? index_1 : index_2;
            const auto bound_index = first_is_domain ? index_2 : index_1;
            auto& matrix = (type == EntityType::gl)
                               ? _d_kl_db_matrices[to_sizet(parameter_index)]
                               : _d_kr_db_matrices[to_sizet(parameter_index)];
            matrix.coeffRef(diff_index, bound_index - nd) += derivative_value;
        }

        return;
    }

    if (type == EntityType::c) {
        const Index index = require_node_index(tmm.nodes(), entity.node_1());
        if (index < nd) {
            _d_capacity_matrix(index, parameter_index) += derivative_value;
        }
        return;
    }

    if (is_heat_flux_entity(type)) {
        const Index index = require_node_index(tmm.nodes(), entity.node_1());
        if (index < nd) {
            _d_heat_flux_matrix(index, parameter_index) += derivative_value;
        }
        return;
    }

    SPDLOG_LOGGER_WARN(get_logger(), "Unsupported jacobian entity type '{}'",
                       entity.string_representation());
}

void JacobianPropagation::build_mk() {
    for (std::size_t parameter_index = 0;
         parameter_index < _derivative_parameter_names.size();
         ++parameter_index) {
        const auto& d_kl_dd = _d_kl_dd_matrices[parameter_index];
        const auto& d_kl_db = _d_kl_db_matrices[parameter_index];
        const auto& d_kr_dd = _d_kr_dd_matrices[parameter_index];
        const auto& d_kr_db = _d_kr_db_matrices[parameter_index];

        VectorXd conductive_diagonal =
            d_kl_dd.selfadjointView<Eigen::Upper>() * (-_ones_domain);
        conductive_diagonal.noalias() -= d_kl_db * _ones_boundary;

        VectorXd radiative_diagonal =
            d_kr_dd.selfadjointView<Eigen::Upper>() * (-_ones_domain);
        radiative_diagonal.noalias() -= d_kr_db * _ones_boundary;

        _m_k.col(to_idx(parameter_index)) =
            d_kl_dd.selfadjointView<Eigen::Upper>() * Td +
            d_kr_dd.selfadjointView<Eigen::Upper>() * _t4_domain +
            conductive_diagonal.cwiseProduct(Td) +
            radiative_diagonal.cwiseProduct(_t4_domain);
    }
}

void JacobianPropagation::build_mq() {
    for (std::size_t parameter_index = 0;
         parameter_index < _derivative_parameter_names.size();
         ++parameter_index) {
        const auto& d_kl_db = _d_kl_db_matrices[parameter_index];
        const auto& d_kr_db = _d_kr_db_matrices[parameter_index];

        _m_q.col(to_idx(parameter_index)) =
            _d_heat_flux_matrix.col(to_idx(parameter_index));
        _m_q.col(to_idx(parameter_index)).noalias() += d_kl_db * Tb;
        _m_q.col(to_idx(parameter_index)).noalias() += d_kr_db * _t4_boundary;
    }
}

void JacobianPropagation::build_mc() {
    const VectorXd initial_temperature_derivative =
        _capacities_inverse.asDiagonal() * (-_kt_q_n0);
    _m_c = initial_temperature_derivative.asDiagonal() * _d_capacity_matrix;
}

void JacobianPropagation::solve_jacobian_step() {
    // With the factors of the last solve_step(), several right-hand sides.
    if (!_network->solve(_mb, _mt)) {
        throw std::runtime_error(std::string(name()) +
                                 ": jacobian solve failed (" +
                                 _network->error_message() + ")");
    }
}

void JacobianPropagation::solve() {
    // Throws first when the solver is not initialised.
    restart_solve();
    const auto start = Clock::now();
    begin_solve_report();
    refresh_live_settings();
    _network->apply_live_settings(_resolved);
    _network->begin_solve();
    SPDLOG_LOGGER_INFO(get_logger(), "{} solving...", name());

    const FormulaExecutionGuard formula_execution(*this);

    auto& jacobian =
        output_model().get_matrix_attribute(DataModelAttribute::JAC);
    jacobian =
        DenseMatrixTimeSeries(nd, to_idx(_derivative_parameter_names.size()));
    jacobian.reserve(to_sizet(num_outputs));
    _mt.setZero();
    _mb.setZero();

    callback_transient_time_change();
    callback_solver_loop();
    outputs_first_last();

    build_conductance_matrix();
    build_heat_flux();
    store_heat_flux_at_n0();

    build_mc();
    build_mk();
    build_mq();
    save_jacobian_data();

    for (time_iter = 0; time_iter < num_time_steps; ++time_iter) {
        callback_solver_loop();
        build_conductance_matrix();
        build_heat_flux();
        store_heat_flux_at_n0();

        // (K + 2C/dt) Mt, with K = -A: the network holds A + 2C/dt.
        _network->multiply(_mt, _mb);
        _mb = (2.0 * (_capacity_diagonal.asDiagonal() * _mt)) - _mb;
        _mb += _m_c + _m_k + _m_q;

        time += _resolved.time_step;
        tmm.time = time;
        callback_transient_time_change();

        int iterations = 0;
        bool converged = false;
        for (solver_iter = 0; solver_iter < _resolved.max_iterations;
             ++solver_iter) {
            iterations = solver_iter + 1;
            callback_solver_loop();
            build_conductance_matrix();
            build_heat_flux();
            solve_step();

            converged = temperature_convergence_check();

            {
                PYCANHA_PROFILE_SCOPE("Write Td in TMM");
                Td = Td_solver;
            }

            if (converged) {
                break;
            }
        }
        record_step_iterations(time_iter, time, iterations, converged);

        // K Td + Qd: the network holds -K + 2C/dt after solve_step().
        _network->multiply(Td, _kt_q_n0);
        _kt_q_n0 = _capacity_diagonal.cwiseProduct(Td) - _kt_q_n0 + Qd;

        build_mc();
        build_mk();
        build_mq();

        _mb += _m_c + _m_k + _m_q;
        solve_jacobian_step();

        callback_transient_after_timestep();
        outputs();
        if (((time_iter + 1) % wait_n_dtimes) == 0) {
            save_jacobian_data();
        }
    }

    outputs_first_last();
    save_jacobian_data();
    finish_solve_report(start);
}

void JacobianPropagation::save_jacobian_data() {
    auto& output_series =
        output_model().get_matrix_attribute(DataModelAttribute::JAC);
    if ((output_series.num_timesteps() > 0) &&
        (std::abs(output_series.time_at(output_series.num_timesteps() - 1) -
                  time) <= eps_time)) {
        output_series.at(output_series.num_timesteps() - 1) = _mt;
        return;
    }

    output_series.push_back(time, _mt);
}

}  // namespace pycanha::solvers::sensitivity
