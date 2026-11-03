#include "pycanha-core/solvers/crank_nicolson.hpp"

#include <spdlog/spdlog.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "factorizer.hpp"
#include "linearized_network.hpp"
#include "memory.hpp"
#include "pycanha-core/globals.hpp"
#include "pycanha-core/solvers/report.hpp"
#include "pycanha-core/solvers/settings.hpp"
#include "pycanha-core/solvers/solver.hpp"
#include "pycanha-core/thermaldata/data_model.hpp"
#include "pycanha-core/thermaldata/sparse_time_series.hpp"
#include "pycanha-core/thermaldata/thermaldata.hpp"
#include "pycanha-core/tmm/couplingmatrices.hpp"
#include "pycanha-core/tmm/nodes.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/utils/logger.hpp"
#include "pycanha-core/utils/profiling.hpp"

namespace pycanha::solvers::transient {

namespace {

constexpr std::array<DataModelAttribute, 13> dense_attributes = {
    DataModelAttribute::T,   DataModelAttribute::C,  DataModelAttribute::QS,
    DataModelAttribute::QA,  DataModelAttribute::QE, DataModelAttribute::QI,
    DataModelAttribute::QR,  DataModelAttribute::A,  DataModelAttribute::APH,
    DataModelAttribute::EPS, DataModelAttribute::FX, DataModelAttribute::FY,
    DataModelAttribute::FZ,
};

std::vector<Index> build_node_numbers(const Nodes& nodes) {
    std::vector<Index> node_numbers;
    node_numbers.reserve(to_sizet(nodes.get_num_nodes()));

    for (Index index = 0; index < nodes.get_num_nodes(); ++index) {
        const auto node_num = nodes.get_node_num_from_idx(index);
        if (!node_num.has_value()) {
            throw std::runtime_error(
                "Could not resolve node number for solver output mapping");
        }

        node_numbers.push_back(to_idx(*node_num));
    }

    return node_numbers;
}

void set_dense_output(DataModel& model, DataModelAttribute attribute,
                      Index row_index, double time,
                      const Eigen::Ref<const Eigen::VectorXd>& values) {
    model.get_dense_attribute(attribute).set_row(row_index, time, values);
}

Eigen::VectorXd densify_sparse_vector(const Eigen::SparseVector<double>& sparse,
                                      Index size) {
    Eigen::VectorXd dense = Eigen::VectorXd::Zero(size);
    for (Eigen::SparseVector<double>::InnerIterator it(sparse); it; ++it) {
        dense(it.index()) = it.value();
    }

    return dense;
}

// The node attribute stored as a sparse vector, null for T and C.
const Eigen::SparseVector<double>* sparse_attribute(
    const Nodes& nodes, DataModelAttribute attribute) {
    switch (attribute) {
        case DataModelAttribute::QS:
            return &nodes.qs_vector;
        case DataModelAttribute::QA:
            return &nodes.qa_vector;
        case DataModelAttribute::QE:
            return &nodes.qe_vector;
        case DataModelAttribute::QI:
            return &nodes.qi_vector;
        case DataModelAttribute::QR:
            return &nodes.qr_vector;
        case DataModelAttribute::A:
            return &nodes.a_vector;
        case DataModelAttribute::APH:
            return &nodes.aph_vector;
        case DataModelAttribute::EPS:
            return &nodes.eps_vector;
        case DataModelAttribute::FX:
            return &nodes.fx_vector;
        case DataModelAttribute::FY:
            return &nodes.fy_vector;
        case DataModelAttribute::FZ:
            return &nodes.fz_vector;
        default:
            return nullptr;
    }
}

SparseTimeSeries::SparseMatrixType build_full_coupling_matrix(
    const CouplingMatrices& matrices) {
    using SparseMatrix = SparseTimeSeries::SparseMatrixType;
    using StorageIndex = SparseMatrix::StorageIndex;
    using Triplet = Eigen::Triplet<double, StorageIndex>;

    const Index nd = matrices.sparse_dd.rows();
    const Index total_nodes =
        matrices.sparse_db.rows() + matrices.sparse_db.cols();

    std::vector<Triplet> triplets;
    triplets.reserve(to_sizet((matrices.sparse_dd.nonZeros() * 2) +
                              (matrices.sparse_db.nonZeros() * 2) +
                              (matrices.sparse_bb.nonZeros() * 2)));

    for (Index row = 0; row < matrices.sparse_dd.outerSize(); ++row) {
        for (SparseMatrix::InnerIterator it(matrices.sparse_dd, row); it;
             ++it) {
            triplets.emplace_back(static_cast<StorageIndex>(it.row()),
                                  static_cast<StorageIndex>(it.col()),
                                  it.value());
            if (it.row() != it.col()) {
                triplets.emplace_back(static_cast<StorageIndex>(it.col()),
                                      static_cast<StorageIndex>(it.row()),
                                      it.value());
            }
        }
    }

    for (Index row = 0; row < matrices.sparse_db.outerSize(); ++row) {
        for (SparseMatrix::InnerIterator it(matrices.sparse_db, row); it;
             ++it) {
            const Index boundary_col = nd + it.col();
            triplets.emplace_back(static_cast<StorageIndex>(it.row()),
                                  static_cast<StorageIndex>(boundary_col),
                                  it.value());
            triplets.emplace_back(static_cast<StorageIndex>(boundary_col),
                                  static_cast<StorageIndex>(it.row()),
                                  it.value());
        }
    }

    for (Index row = 0; row < matrices.sparse_bb.outerSize(); ++row) {
        for (SparseMatrix::InnerIterator it(matrices.sparse_bb, row); it;
             ++it) {
            const Index mapped_row = nd + it.row();
            const Index mapped_col = nd + it.col();
            triplets.emplace_back(static_cast<StorageIndex>(mapped_row),
                                  static_cast<StorageIndex>(mapped_col),
                                  it.value());
            if (mapped_row != mapped_col) {
                triplets.emplace_back(static_cast<StorageIndex>(mapped_col),
                                      static_cast<StorageIndex>(mapped_row),
                                      it.value());
            }
        }
    }

    SparseMatrix output(total_nodes, total_nodes);
    output.setFromTriplets(triplets.begin(), triplets.end(),
                           [](double lhs, double rhs) { return lhs + rhs; });
    output.makeCompressed();
    return output;
}

// Bytes of one stored sample of build_full_coupling_matrix(), which holds
// both triangles of every block.
std::int64_t full_coupling_bytes(const CouplingMatrices& matrices,
                                 Index total_nodes) {
    constexpr auto entry_bytes =
        static_cast<std::int64_t>(sizeof(double) + sizeof(int));
    const auto stored = static_cast<std::int64_t>(
        matrices.sparse_dd.nonZeros() + matrices.sparse_db.nonZeros() +
        matrices.sparse_bb.nonZeros());
    return (2 * stored * entry_bytes) +
           ((static_cast<std::int64_t>(total_nodes) + 1) *
            static_cast<std::int64_t>(sizeof(int))) +
           static_cast<std::int64_t>(sizeof(double));
}

void append_or_replace_sparse_sample(SparseTimeSeries& series, double time,
                                     SparseTimeSeries::SparseMatrixType matrix,
                                     double eps_time) {
    if ((series.num_timesteps() > 0) &&
        (std::abs(series.time_at(series.num_timesteps() - 1) - time) <=
         eps_time)) {
        series.at(series.num_timesteps() - 1) = std::move(matrix);
        return;
    }

    series.push_back(time, std::move(matrix));
}

}  // namespace

CrankNicolson::CrankNicolson(
    std::shared_ptr<ThermalMathematicalModel> tmm_shptr)
    : CrankNicolson(std::move(tmm_shptr), SolverType::CRANK_NICOLSON) {}

CrankNicolson::CrankNicolson(
    std::shared_ptr<ThermalMathematicalModel> tmm_shptr, SolverType type)
    : Solver(std::move(tmm_shptr), type) {}

CrankNicolson::~CrankNicolson() = default;

void CrankNicolson::set_simulation_time(double start_time, double end_time,
                                        double time_step,
                                        double output_interval) {
    if (end_time < start_time) {
        throw std::invalid_argument("end_time must not be before start_time");
    }
    if (!(time_step > 0.0)) {
        throw std::invalid_argument("time_step must be positive");
    }
    if (output_interval < 0.0) {
        throw std::invalid_argument("output_interval must not be negative");
    }
    Settings replacement = settings();
    replacement.start_time = start_time;
    replacement.end_time = end_time;
    replacement.time_step = time_step;
    replacement.output_interval = output_interval;
    set_settings(replacement);
}

DataModel& CrankNicolson::output_model() {
    return tmm.thermal_data().models().get_model(_resolved.output_name);
}

const DataModel& CrankNicolson::output_model() const {
    return tmm.thermal_data().models().get_model(_resolved.output_name);
}

void CrankNicolson::initialize() {
    const auto start = Clock::now();
    begin_initialize_report();
    if (preparation_matches()) {
        // estimate_memory() built the pattern and ran the analysis.
        initialize_common();
    } else {
        prepare(/*log_warnings=*/true);
    }
    _network->allocate();
    if (const auto hint = _network->hint(_resolved)) {
        warn(*hint);
    }
    create_output_model();
    SPDLOG_LOGGER_INFO(get_logger(), "{} initializing ({})", name(),
                       detail::describe(_resolved, _symmetric));
    _prepared = false;
    _initialized = true;
    end_initialize_report(start);
}

void CrankNicolson::prepare(bool log_warnings) {
    release();
    // Throws before anything is built when the settings contradict.
    initialize_common(log_warnings);
    compute_output_schedule();
    _symmetric = _resolved.factorization == Factorization::CHOLESKY;
    _network =
        detail::make_network(_resolved,
                             {.conductive = &KLdd,
                              .conductive_boundary = &KLdb,
                              .radiative = &KRdd,
                              .radiative_boundary = &KRdb},
                             {.symmetric = _symmetric,
                              .radiation = has_radiation(),
                              .skip_unchanged = false,
                              .keep_matrix = keeps_matrix_after_solve(),
                              .fixed_couplings = _resolved.fixed_couplings});

    _rhs = VectorXd::Zero(nd);
    _heat_flux_n0 = VectorXd::Zero(nd);
    _capacities = VectorXd::Zero(nd);
    _capacities_inverse = VectorXd::Zero(nd);
    _capacity_diagonal = VectorXd::Zero(nd);
    _radiation_linear_term = VectorXd::Zero(nd);
    _kt_q_n0 = VectorXd::Zero(nd);
    _a_times_t = VectorXd::Zero(nd);
    _ones_domain = VectorXd::Ones(nd);
    _ones_boundary = VectorXd::Ones(nb);
    _t3_domain = VectorXd::Zero(nd);
    _t4_domain = VectorXd::Zero(nd);
    _t4_boundary = VectorXd::Zero(nb);

    if (!_network->analyze()) {
        throw std::runtime_error(std::string(name()) + ": analysis failed (" +
                                 _network->error_message() + ")");
    }
    mark_prepared();
}

SolverMemory CrankNicolson::memory_now() const {
    using detail::bytes_of;
    SolverMemory memory;
    if (_network) {
        memory.factors = _network->factor_bytes();
        memory.exact = _network->factor_bytes_exact();
        memory.matrix = _network->matrix_bytes();
    }
    memory.matrix += bytes_of(_rhs) + bytes_of(_heat_flux_n0) +
                     bytes_of(_capacities) + bytes_of(_capacities_inverse) +
                     bytes_of(_capacity_diagonal) + bytes_of(_t3_domain) +
                     bytes_of(_t4_domain) + bytes_of(_t4_boundary) +
                     bytes_of(_radiation_linear_term) + bytes_of(_kt_q_n0) +
                     bytes_of(_a_times_t) + bytes_of(_ones_domain) +
                     bytes_of(_ones_boundary) + common_bytes();
    memory.output = output_bytes();
    return memory;
}

std::int64_t CrankNicolson::output_bytes() const {
    const auto samples = static_cast<std::int64_t>(num_outputs);
    const auto& attributes = _resolved.output_attributes;
    // A dense sample stores one value per node and its time.
    const auto dense_sample = static_cast<std::int64_t>(
        (static_cast<std::size_t>(N) + 1) * sizeof(double));
    const auto dense_stored = static_cast<std::int64_t>(
        std::ranges::count_if(dense_attributes, [&attributes](auto attribute) {
            return attributes.contains(attribute);
        }));
    std::int64_t bytes = samples * dense_stored * dense_sample;
    if (attributes.contains(DataModelAttribute::KL)) {
        bytes += samples * full_coupling_bytes(ltcs, N);
    }
    if (attributes.contains(DataModelAttribute::KR)) {
        bytes += samples * full_coupling_bytes(rtcs, N);
    }
    return bytes;
}

void CrankNicolson::compute_output_schedule() {
    const double total_sim_time = _resolved.end_time - _resolved.start_time;
    num_time_steps = static_cast<int>(
        std::ceil((total_sim_time - eps_time) / _resolved.time_step));
    num_time_steps = std::max(num_time_steps, 1);

    if (_resolved.output_interval <= 0.0) {
        wait_n_dtimes = num_time_steps;
    } else {
        wait_n_dtimes = static_cast<int>(std::ceil(
            (_resolved.output_interval - eps_time) / _resolved.time_step));
        wait_n_dtimes = std::max(wait_n_dtimes, 1);
    }

    num_outputs = ((num_time_steps - 1) / wait_n_dtimes) + 2;
    num_outputs = std::max(num_outputs, 2);
}

void CrankNicolson::create_output_model() {
    const auto& attributes = _resolved.output_attributes;
    DataModel model(build_node_numbers(tmm.nodes()));
    for (const auto attribute : dense_attributes) {
        if (attributes.contains(attribute)) {
            model.get_dense_attribute(attribute).resize(num_outputs, N);
        }
    }
    if (attributes.contains(DataModelAttribute::KL)) {
        model.get_sparse_attribute(DataModelAttribute::KL)
            .reserve(to_sizet(num_outputs));
    }
    if (attributes.contains(DataModelAttribute::KR)) {
        model.get_sparse_attribute(DataModelAttribute::KR)
            .reserve(to_sizet(num_outputs));
    }

    tmm.thermal_data().models().add_model(_resolved.output_name,
                                          std::move(model));
}

void CrankNicolson::save_output_data() {
    auto& model = output_model();
    const auto& attributes = _resolved.output_attributes;

    for (const auto attribute : dense_attributes) {
        if (!attributes.contains(attribute)) {
            continue;
        }
        if (attribute == DataModelAttribute::T) {
            set_dense_output(model, attribute, idata_out, time, T);
        } else if (attribute == DataModelAttribute::C) {
            set_dense_output(model, attribute, idata_out, time, C);
        } else {
            set_dense_output(model, attribute, idata_out, time,
                             densify_sparse_vector(
                                 *sparse_attribute(tmm.nodes(), attribute), N));
        }
    }
    if (attributes.contains(DataModelAttribute::KL)) {
        append_or_replace_sparse_sample(
            model.get_sparse_attribute(DataModelAttribute::KL), time,
            build_full_coupling_matrix(ltcs), eps_time);
    }
    if (attributes.contains(DataModelAttribute::KR)) {
        append_or_replace_sparse_sample(
            model.get_sparse_attribute(DataModelAttribute::KR), time,
            build_full_coupling_matrix(rtcs), eps_time);
    }
}

void CrankNicolson::outputs_first_last() {
    PYCANHA_PROFILE_SCOPE("Outputs");
    if (idata_out == 0) {
        save_output_data();
    } else if (idata_out < num_outputs - 1) {
        idata_out = num_outputs - 1;
        save_output_data();
    }
}

void CrankNicolson::outputs() {
    PYCANHA_PROFILE_SCOPE("Outputs");
    if (((time_iter + 1) % wait_n_dtimes) == 0) {
        ++idata_out;
        save_output_data();
    }
}

void CrankNicolson::restart_solve() {
    if (!_initialized) {
        throw std::invalid_argument(
            std::string(name()) +
            " has not been initialized. Call initialize() before solve().");
    }

    _converged = false;
    time = _resolved.start_time;
    tmm.time = time;

    time_iter = -1;
    idata_out = 0;

    SPDLOG_LOGGER_DEBUG(pycanha::get_logger(), "(Re)starting solve...");
}

void CrankNicolson::solve() {
    // Throws first when the solver is not initialised.
    restart_solve();
    const auto start = Clock::now();
    begin_solve_report();
    refresh_live_settings();
    _network->apply_live_settings(_resolved);
    _network->begin_solve();
    SPDLOG_LOGGER_INFO(get_logger(), "{} solving...", name());

    const FormulaExecutionGuard formula_execution(*this);

    callback_transient_time_change();
    callback_solver_loop();
    outputs_first_last();

    for (time_iter = 0; time_iter < num_time_steps; ++time_iter) {
        callback_solver_loop();
        build_conductance_matrix();
        build_heat_flux();
        store_heat_flux_at_n0();

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

        callback_transient_after_timestep();
        outputs();
    }

    outputs_first_last();
    finish_solve_report(start);
}

void CrankNicolson::deinitialize() { release(); }

void CrankNicolson::release() noexcept {
    _network.reset();
    _symmetric = false;
    _prepared = false;
    _initialized = false;
}

void CrankNicolson::build_capacities() {
    PYCANHA_PROFILE_SCOPE("Build C");
    _capacities = Cd;
    _capacities.array() += _resolved.capacity_delta;
    _capacities_inverse = _capacities.cwiseInverse();
    _capacity_diagonal = (2.0 / _resolved.time_step) * _capacities;
}

// -K + 2C/dt at the current temperatures, couplings and capacities.
void CrankNicolson::build_conductance_matrix() {
    build_capacities();
    PYCANHA_PROFILE_SCOPE("Linearization");
    _t3_domain = (4.0 * STF_BOLTZ) * Td.array().cube();
    _t4_domain = STF_BOLTZ * Td.array().square().square();
    _t4_boundary = STF_BOLTZ * Tb.array().square().square();
    _network->linearize(_t3_domain, Td, _capacity_diagonal,
                        _radiation_linear_term);
}

void CrankNicolson::build_heat_flux() {
    PYCANHA_PROFILE_SCOPE("Build Q");
    Q = QI_sp + QS_sp + QA_sp + QE_sp + QR_sp;
    new (&Qd) WrappVectorXd(Q.data(), nd);
    Qd += _radiation_linear_term + KLdb * Tb + KRdb * _t4_boundary;
}

void CrankNicolson::store_heat_flux_at_n0() {
    PYCANHA_PROFILE_SCOPE("Store Q_n0");
    // K Td + Qd, with K = -A: the network holds A + 2C/dt.
    _network->multiply(Td, _kt_q_n0);
    _kt_q_n0 = Qd - _kt_q_n0 + _capacity_diagonal.cwiseProduct(Td);
    _heat_flux_n0 =
        _kt_q_n0 + (2.0 / _resolved.time_step) * _capacities.cwiseProduct(Td);
}

void CrankNicolson::solve_step() {
    PYCANHA_PROFILE_SCOPE("Solver Step");
    _rhs = Qd + _heat_flux_n0;
    if (_resolved.reuse_factors) {
        // Solved for the change from the last inner iteration: the residual
        // of an iteration on old factors then stays in a change that shrinks
        // as the step converges. It starts from zero, closer to this change
        // than the last one is.
        _network->multiply(Td, _a_times_t);
        _rhs -= _a_times_t;
        Td_solver.setZero();
        if (_rhs.isZero(0.0)) {
            // A zero residual is solved by a zero change, and PARDISO's
            // iteration on old factors fails on a zero right-hand side.
            Td_solver = Td;
            return;
        }
    }

    // The network holds -K + 2C/dt, with a positive diagonal.
    const auto start = Clock::now();
    const detail::SolveOutcome outcome = _network->solve(_rhs, Td_solver);
    if (!outcome.succeeded) {
        throw std::runtime_error(std::string(name()) + ": solve failed (" +
                                 _network->error_message() + ")");
    }
    if (_resolved.reuse_factors) {
        Td_solver += Td;
    }
    record_linear_solve(start, outcome.factorized);
}

}  // namespace pycanha::solvers::transient
