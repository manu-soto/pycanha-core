#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "pycanha-core/config.hpp"
#include "pycanha-core/thermaldata/data_model.hpp"

namespace pycanha {
class ThermalMathematicalModel;
}  // namespace pycanha

namespace pycanha::solvers {

/// The solver type a Settings is meant for.
enum class SolverType : std::uint8_t {
    STEADY_STATE,          ///< Steady state of the network.
    CRANK_NICOLSON,        ///< Crank-Nicolson transient.
    JACOBIAN_PROPAGATION,  ///< Crank-Nicolson transient with the derivatives
                           ///< of the temperatures with respect to parameters.
};

/// How the matrix of an implicit solver is stored and factorised.
enum class Storage : std::uint8_t {
    /// Only the non-zero entries: the choice for large models.
    SPARSE,
    /// Every entry of the n x n matrix [n^2 * 8 bytes each]: faster for small
    /// models whose nodes are coupled to most of the others (radiative
    /// enclosures).
    DENSE,
};

/// Library that factorises the matrix of an implicit solver.
enum class Backend : std::uint8_t {
    /// Intel MKL PARDISO: multi-threaded supernodal factorisation, sparse
    /// storage only. Only in builds with MKL.
    PARDISO,
    /// Eigen's sparse or dense solvers: in every build.
    EIGEN,
    /// Intel MKL LAPACK: multi-threaded dense factorisation, dense storage
    /// only. Only in builds with MKL.
    LAPACK,
    /// Apple Accelerate: multi-threaded supernodal factorisation, sparse
    /// storage only. Only in macOS builds with Accelerate, on macOS 15.5 or
    /// newer (see accelerate_available()).
    ACCELERATE,
};

/// Factorisation of the matrix.
enum class Factorization : std::uint8_t {
    /// Cholesky when the matrix is symmetric (no radiative couplings) and the
    /// backend has a Cholesky factorisation it prefers: PARDISO, ACCELERATE,
    /// LAPACK and dense Eigen. LU otherwise.
    AUTO,
    /// LU, for any matrix.
    LU,
    /// Cholesky: faster and smaller than LU, only for symmetric matrices
    /// (models without radiative couplings).
    CHOLESKY,
};

/// Fill-reducing ordering of PARDISO.
enum class PardisoOrdering : std::uint8_t {
    /// Parallel nested dissection (METIS).
    NESTED_DISSECTION,
    /// Minimum degree. It seems faster on nearly dense models (many radiative
    /// couplings per node); its analysis is slow on models where a few nodes
    /// are coupled to many.
    MIN_DEGREE,
};

/// Numerical factorisation of PARDISO.
enum class PardisoLevel : std::uint8_t {
    /// Two-level factorisation, with nested dissection only. The safe
    /// choice.
    TWO_LEVEL,
    /// Classic one-level factorisation, needed to reuse factors. With nested
    /// dissection it can livelock (never finish) with MKL 2025.3 at 4 or more
    /// threads on some radiative models.
    ONE_LEVEL,
};

/// Ordering of Eigen's sparse LU.
enum class EigenOrdering : std::uint8_t {
    /// COLAMD.
    COLAMD,
    /// Approximate minimum degree. It seems faster than COLAMD only on models
    /// where a few nodes are coupled to many.
    AMD,
};

/// Fill-reducing ordering of Accelerate.
enum class AccelerateOrdering : std::uint8_t {
    /// Nested dissection (METIS).
    METIS,
    /// Approximate minimum degree.
    AMD,
};

/// Whether a setting may change while the solver is initialised.
enum class Mutability : std::uint8_t {
    /// Read at every solve(): it may change at any time.
    LIVE,
    /// Read by initialize(): it is frozen while the solver is initialised.
    INIT,
};

/// Where an effective setting value comes from.
enum class Origin : std::uint8_t {
    DEFAULT,         ///< The solver's default.
    USER,            ///< Set in the Settings.
    AUTO_CONFIGURE,  ///< Set by Solver::auto_configure() and unchanged since.
};

[[nodiscard]] std::string_view to_string(SolverType type) noexcept;
[[nodiscard]] std::string_view to_string(Storage storage) noexcept;
[[nodiscard]] std::string_view to_string(Backend backend) noexcept;
[[nodiscard]] std::string_view to_string(Factorization factorization) noexcept;
[[nodiscard]] std::string_view to_string(PardisoOrdering ordering) noexcept;
[[nodiscard]] std::string_view to_string(PardisoLevel level) noexcept;
[[nodiscard]] std::string_view to_string(EigenOrdering ordering) noexcept;
[[nodiscard]] std::string_view to_string(AccelerateOrdering ordering) noexcept;
[[nodiscard]] std::string_view to_string(Mutability mutability) noexcept;
[[nodiscard]] std::string_view to_string(Origin origin) noexcept;

/// Whether backend = ACCELERATE can be used: the build has Apple Accelerate
/// and the running macOS has its sparse LU (macOS 15.5 or newer).
[[nodiscard]] bool accelerate_available() noexcept;

/// Every temperature, capacity, load and node attribute stored per node.
[[nodiscard]] std::set<DataModelAttribute> all_dense_output_attributes();
/// The dense attributes plus the coupling matrices and the Jacobian.
[[nodiscard]] std::set<DataModelAttribute> all_output_attributes();

/// A setting value as Settings::get and to_dict() give it: enums and output
/// attributes by name, integers as 64-bit.
using SettingValue = std::variant<bool, std::int64_t, double, std::string,
                                  std::map<int, int>, std::vector<std::string>>;
/// Set settings by name ("tolerance", "pardiso.level"), plus "solver_type".
using SettingsDict = std::map<std::string, SettingValue, std::less<>>;

/// PARDISO settings, used only with Backend::PARDISO.
struct PardisoSettings {
    std::optional<PardisoOrdering> ordering;
    std::optional<PardisoLevel> level;
    /// iparm entries (zero-based index, value) applied after every other
    /// setting.
    std::optional<std::map<int, int>> iparm;
    std::optional<bool> verbose;

    [[nodiscard]] bool operator==(const PardisoSettings&) const = default;
};

/// Eigen settings, used only with Backend::EIGEN.
struct EigenSettings {
    std::optional<EigenOrdering> ordering;

    [[nodiscard]] bool operator==(const EigenSettings&) const = default;
};

/// Accelerate settings, used only with Backend::ACCELERATE.
struct AccelerateSettings {
    std::optional<AccelerateOrdering> ordering;

    [[nodiscard]] bool operator==(const AccelerateSettings&) const = default;
};

/// The settings of every solver type.
/**
 * Each field is set (holds a value) or unset (the solver's default, which may
 * depend on the solver type, the build, the other fields and the model). One
 * Settings can be shared by solvers of different types: a set field that does
 * not apply to a solver is ignored there with a warning (see resolve()).
 * setting_fields() lists every field with its documentation.
 */
struct Settings {
    /// The solver type these settings are for; unset for a loose set of
    /// values.
    std::optional<SolverType> solver_type;

    std::optional<double> tolerance;
    std::optional<int> max_iterations;
    std::optional<Storage> storage;
    std::optional<Backend> backend;
    std::optional<Factorization> factorization;
    std::optional<bool> reuse_factors;
    std::optional<double> reuse_tolerance;
    std::optional<bool> fixed_couplings;
    std::optional<int> threads;
    std::optional<double> start_time;
    std::optional<double> end_time;
    std::optional<double> time_step;
    std::optional<double> output_interval;
    std::optional<std::set<DataModelAttribute>> output_attributes;
    std::optional<std::string> output_name;
    std::optional<double> capacity_delta;
    PardisoSettings pardiso;
    EigenSettings eigen;
    AccelerateSettings accelerate;

    [[nodiscard]] bool operator==(const Settings&) const = default;

    /// Access by name. Unknown names, values of the wrong type and values
    /// out of range throw std::invalid_argument; the settings are then left
    /// unchanged.
    [[nodiscard]] bool is_set(std::string_view name) const;
    [[nodiscard]] std::optional<SettingValue> get(std::string_view name) const;
    void set(std::string_view name, const SettingValue& value);
    void unset(std::string_view name);

    /// The set fields only. from_dict() throws std::invalid_argument for an
    /// unknown key or a value of the wrong type or out of range.
    [[nodiscard]] SettingsDict to_dict() const;
    [[nodiscard]] static Settings from_dict(const SettingsDict& dict);

    /// Without a solver type, the set fields. With one, every field that
    /// applies to it with its effective value, then the set fields it
    /// ignores and the contradictions.
    [[nodiscard]] std::string to_string() const;
};

/// Documentation of one setting.
struct SettingField {
    std::string_view name;   ///< "tolerance", "pardiso.level"
    std::string_view group;  ///< "", "pardiso", "eigen" or "accelerate"
    std::string_view doc;
    Mutability mutability = Mutability::INIT;
    std::uint8_t solver_types = 0;  ///< One bit per SolverType.

    [[nodiscard]] constexpr bool applies_to(SolverType type) const noexcept {
        return (solver_types & (1U << static_cast<unsigned>(type))) != 0U;
    }
};

/// Every setting, in the order Settings::to_string() prints them.
[[nodiscard]] std::span<const SettingField> setting_fields() noexcept;
/// Throws std::invalid_argument for an unknown name.
[[nodiscard]] const SettingField& setting_field(std::string_view name);

/// What the defaults and the checks depend on besides the settings.
struct ResolveContext {
    SolverType solver_type = SolverType::STEADY_STATE;
    /// Whether the model has radiative couplings; unset when no model is
    /// known (factorization AUTO then stays AUTO and the Cholesky check is
    /// skipped).
    std::optional<bool> has_radiation;
    /// Whether this build has MKL (PARDISO and LAPACK).
    bool mkl_available = MKL_ENABLED;
    /// Whether ACCELERATE can be used (see accelerate_available()).
    bool accelerate_available = solvers::accelerate_available();
};

/// The context of a solver of `type` for `model`.
[[nodiscard]] ResolveContext resolve_context(
    SolverType type, const ThermalMathematicalModel& model);

/// A set field that does not apply, with the reason.
struct NotApplied {
    std::string name;
    std::string reason;
};

/// Effective PARDISO settings.
struct ResolvedPardiso {
    PardisoOrdering ordering = PardisoOrdering::NESTED_DISSECTION;
    PardisoLevel level = PardisoLevel::TWO_LEVEL;
    std::map<int, int> iparm;
    bool verbose = false;
};

/// Effective Eigen settings.
struct ResolvedEigen {
    EigenOrdering ordering = EigenOrdering::COLAMD;
};

/// Effective Accelerate settings.
struct ResolvedAccelerate {
    AccelerateOrdering ordering = AccelerateOrdering::METIS;
};

/// The effective value of every setting for one solver type.
struct ResolvedSettings {
    SolverType solver_type = SolverType::STEADY_STATE;

    double tolerance = 0.0;
    int max_iterations = 0;
    Storage storage = Storage::SPARSE;
    Backend backend = Backend::EIGEN;
    Factorization factorization = Factorization::AUTO;
    bool reuse_factors = false;
    double reuse_tolerance = 0.0;
    bool fixed_couplings = false;
    int threads = 0;
    double start_time = 0.0;
    double end_time = 0.0;
    double time_step = 0.0;
    double output_interval = 0.0;
    std::set<DataModelAttribute> output_attributes;
    std::string output_name;
    double capacity_delta = 0.0;
    ResolvedPardiso pardiso;
    ResolvedEigen eigen;
    ResolvedAccelerate accelerate;

    /// Set fields ignored for this solver type or backend.
    std::vector<NotApplied> not_applied;
    /// Accepted combinations that deserve a warning.
    std::vector<std::string> warnings;

    /// Whether the field applies to the solver type and the selected
    /// backend, and where its value comes from (DEFAULT when it does not
    /// apply). Throw std::invalid_argument for an unknown name.
    [[nodiscard]] bool applies(std::string_view name) const;
    [[nodiscard]] Origin origin(std::string_view name) const;
    [[nodiscard]] SettingValue value(std::string_view name) const;

    /// One entry per setting_fields() entry.
    struct FieldState {
        bool applies = false;
        Origin origin = Origin::DEFAULT;
    };
    std::vector<FieldState> fields;
};

/// The effective settings of a solver: each unset field, and each set field
/// that does not apply, takes its default.
/**
 * Throws std::invalid_argument listing every contradiction at once: a value
 * out of range, a combination the backend cannot do (reuse_factors with
 * Eigen or ACCELERATE, minimum degree with the two-level factorisation,
 * PARDISO or ACCELERATE with dense storage), or one the build or the model
 * cannot (PARDISO or LAPACK without MKL, ACCELERATE when
 * accelerate_available() is false, Cholesky with radiation). The set
 * fields that do not apply are listed in not_applied, for the solver to warn
 * about.
 */
[[nodiscard]] ResolvedSettings resolve(const Settings& settings,
                                       const ResolveContext& context);

/// The effective value of one applicable setting and why it has it.
struct FieldExplanation {
    std::string name;
    SettingValue value;
    /// The value as Settings::to_string() prints it.
    std::string value_text;
    /// DEFAULT or USER; a solver reports AUTO_CONFIGURE for the values its
    /// auto_configure() set.
    Origin origin = Origin::DEFAULT;
    /// Why a default has its value ("reuse_factors needs the one-level
    /// factorization"), or what a set value resolves to; empty when there is
    /// nothing to add.
    std::string reason;
};

/// What resolve() gives, with a reason for every applicable field. The
/// contradictions are listed instead of thrown.
struct SettingsExplanation {
    /// The effective values (each contradicting field keeps its value).
    ResolvedSettings resolved;
    /// The applicable fields, in setting_fields() order.
    std::vector<FieldExplanation> fields;
    std::vector<std::string> contradictions;
};

[[nodiscard]] SettingsExplanation explain_settings(
    const Settings& settings, const ResolveContext& context);

/// The INIT fields whose effective value differs between `before` and
/// `after` (fields that do not apply under both are skipped).
[[nodiscard]] std::vector<std::string_view> changed_init_fields(
    const Settings& before, const Settings& after,
    const ResolveContext& context);

/// Throws std::runtime_error naming the INIT fields that `after` changes,
/// when the solver is initialised: they are read by initialize() only.
void check_settings_change(const Settings& before, const Settings& after,
                           const ResolveContext& context, bool initialized);

}  // namespace pycanha::solvers
