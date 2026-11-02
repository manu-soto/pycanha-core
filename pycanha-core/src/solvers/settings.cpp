#include "pycanha-core/solvers/settings.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "pycanha-core/config.hpp"
#include "pycanha-core/thermaldata/data_model.hpp"
#include "pycanha-core/tmm/couplingmatrices.hpp"
#include "pycanha-core/tmm/thermalmathematicalmodel.hpp"
#include "pycanha-core/tmm/thermalnetwork.hpp"

namespace pycanha::solvers {

namespace {

// ---------------------------------------------------------------------------
// Enum names, indexed by the enumerator's value
// ---------------------------------------------------------------------------

constexpr std::array<std::string_view, 3> names_of(SolverType /*tag*/) {
    return {"STEADY_STATE", "CRANK_NICOLSON", "JACOBIAN_PROPAGATION"};
}
constexpr std::array<std::string_view, 2> names_of(Storage /*tag*/) {
    return {"SPARSE", "DENSE"};
}
constexpr std::array<std::string_view, 4> names_of(Backend /*tag*/) {
    return {"PARDISO", "EIGEN", "LAPACK", "ACCELERATE"};
}
constexpr std::array<std::string_view, 3> names_of(Factorization /*tag*/) {
    return {"AUTO", "LU", "CHOLESKY"};
}
constexpr std::array<std::string_view, 2> names_of(PardisoOrdering /*tag*/) {
    return {"NESTED_DISSECTION", "MIN_DEGREE"};
}
constexpr std::array<std::string_view, 2> names_of(PardisoLevel /*tag*/) {
    return {"TWO_LEVEL", "ONE_LEVEL"};
}
constexpr std::array<std::string_view, 2> names_of(EigenOrdering /*tag*/) {
    return {"COLAMD", "AMD"};
}
constexpr std::array<std::string_view, 2> names_of(AccelerateOrdering /*tag*/) {
    return {"METIS", "AMD"};
}
constexpr std::array<std::string_view, 2> names_of(Mutability /*tag*/) {
    return {"LIVE", "INIT"};
}
constexpr std::array<std::string_view, 3> names_of(Origin /*tag*/) {
    return {"DEFAULT", "USER", "AUTO_CONFIGURE"};
}
constexpr std::array<std::string_view, 16> names_of(
    DataModelAttribute /*tag*/) {
    return {"T",   "C",   "QS", "QA", "QE", "QI", "QR", "A",
            "APH", "EPS", "FX", "FY", "FZ", "KL", "KR", "JAC"};
}

template <typename Enum>
std::string_view enum_name(Enum value) noexcept {
    const auto names = names_of(Enum{});
    const auto index = static_cast<std::size_t>(value);
    return index < names.size() ? names.at(index) : std::string_view{"?"};
}

template <typename Enum>
std::optional<Enum> parse_enum(std::string_view text) {
    const auto names = names_of(Enum{});
    for (std::size_t index = 0; index < names.size(); ++index) {
        if (names.at(index) == text) {
            return static_cast<Enum>(index);
        }
    }
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// Field values to and from SettingValue
// ---------------------------------------------------------------------------

[[noreturn]] void wrong_type(std::string_view name, std::string_view expected) {
    throw std::invalid_argument(
        std::format("setting '{}' expects {}", name, expected));
}

template <typename T>
struct Codec;

template <>
struct Codec<double> {
    static SettingValue encode(double value) { return value; }
    static double decode(const SettingValue& value, std::string_view name) {
        if (const auto* number = std::get_if<double>(&value)) {
            return *number;
        }
        if (const auto* integer = std::get_if<std::int64_t>(&value)) {
            return static_cast<double>(*integer);
        }
        wrong_type(name, "a number");
    }
};

template <>
struct Codec<int> {
    static SettingValue encode(int value) { return std::int64_t{value}; }
    static int decode(const SettingValue& value, std::string_view name) {
        const auto* integer = std::get_if<std::int64_t>(&value);
        if (integer == nullptr || *integer < std::numeric_limits<int>::min() ||
            *integer > std::numeric_limits<int>::max()) {
            wrong_type(name, "an integer");
        }
        return static_cast<int>(*integer);
    }
};

template <>
struct Codec<bool> {
    static SettingValue encode(bool value) { return value; }
    static bool decode(const SettingValue& value, std::string_view name) {
        if (const auto* flag = std::get_if<bool>(&value)) {
            return *flag;
        }
        wrong_type(name, "true or false");
    }
};

template <>
struct Codec<std::string> {
    static SettingValue encode(const std::string& value) { return value; }
    static std::string decode(const SettingValue& value,
                              std::string_view name) {
        if (const auto* text = std::get_if<std::string>(&value)) {
            return *text;
        }
        wrong_type(name, "a string");
    }
};

template <>
struct Codec<std::map<int, int>> {
    static SettingValue encode(const std::map<int, int>& value) {
        return value;
    }
    static std::map<int, int> decode(const SettingValue& value,
                                     std::string_view name) {
        if (const auto* entries = std::get_if<std::map<int, int>>(&value)) {
            return *entries;
        }
        wrong_type(name, "a map of integers to integers");
    }
};

template <>
struct Codec<std::set<DataModelAttribute>> {
    static SettingValue encode(const std::set<DataModelAttribute>& value) {
        std::vector<std::string> names;
        names.reserve(value.size());
        std::ranges::transform(value, std::back_inserter(names),
                               [](DataModelAttribute attribute) {
                                   return std::string(enum_name(attribute));
                               });
        return names;
    }
    static std::set<DataModelAttribute> decode(const SettingValue& value,
                                               std::string_view name) {
        const auto* names = std::get_if<std::vector<std::string>>(&value);
        if (names == nullptr) {
            wrong_type(name, "a list of output attribute names");
        }
        std::set<DataModelAttribute> attributes;
        for (const std::string& attribute_name : *names) {
            const auto attribute =
                parse_enum<DataModelAttribute>(attribute_name);
            if (!attribute.has_value()) {
                throw std::invalid_argument(
                    std::format("setting '{}': unknown output attribute '{}'",
                                name, attribute_name));
            }
            attributes.insert(*attribute);
        }
        return attributes;
    }
};

template <typename Enum>
    requires std::is_enum_v<Enum>
struct Codec<Enum> {
    static SettingValue encode(Enum value) {
        return std::string(enum_name(value));
    }
    static Enum decode(const SettingValue& value, std::string_view name) {
        const auto* text = std::get_if<std::string>(&value);
        const auto parsed =
            text != nullptr ? parse_enum<Enum>(*text) : std::optional<Enum>{};
        if (!parsed.has_value()) {
            std::string expected = "one of";
            for (const std::string_view option : names_of(Enum{})) {
                expected += std::format(" {}", option);
            }
            wrong_type(name, expected);
        }
        return *parsed;
    }
};

// ---------------------------------------------------------------------------
// Value checks: empty when the value is valid, the requirement otherwise
// ---------------------------------------------------------------------------

template <typename T>
std::string_view any_value(const T& /*value*/) {
    return {};
}
std::string_view positive(const double& value) {
    return value > 0.0 && std::isfinite(value) ? std::string_view{}
                                               : "must be positive";
}
std::string_view non_negative(const double& value) {
    return value >= 0.0 && std::isfinite(value) ? std::string_view{}
                                                : "must not be negative";
}
std::string_view finite(const double& value) {
    return std::isfinite(value) ? std::string_view{} : "must be finite";
}
std::string_view at_least_one(const int& value) {
    return value >= 1 ? std::string_view{} : "must be at least 1";
}
std::string_view non_negative_count(const int& value) {
    return value >= 0 ? std::string_view{} : "must not be negative";
}
std::string_view below_one(const double& value) {
    return value > 0.0 && value < 1.0 ? std::string_view{}
                                      : "must be between 0 and 1";
}
std::string_view not_empty(const std::string& value) {
    return value.empty() ? "must not be empty" : std::string_view{};
}
std::string_view iparm_indices(const std::map<int, int>& value) {
    // PARDISO's iparm has 64 entries.
    constexpr int iparm_size = 64;
    for (const auto& [index, entry] : value) {
        if (index < 0 || index >= iparm_size) {
            return "indices must be between 0 and 63";
        }
    }
    return {};
}

// ---------------------------------------------------------------------------
// Defaults: (raw settings, the fields resolved so far, context) -> value
// ---------------------------------------------------------------------------

constexpr double default_tolerance = 1e-3;
constexpr int default_max_iterations = 5;
constexpr bool default_reuse_factors = false;
constexpr bool default_fixed_couplings = false;
constexpr int default_threads = 0;
constexpr double default_time = 0.0;
constexpr double default_capacity_delta = 1e-7;
constexpr bool default_verbose = false;
constexpr Factorization default_factorization = Factorization::AUTO;
constexpr EigenOrdering default_eigen_ordering = EigenOrdering::COLAMD;
constexpr AccelerateOrdering default_accelerate_ordering =
    AccelerateOrdering::METIS;

template <const auto& Value>
std::remove_cvref_t<decltype(Value)> constant(
    const Settings& /*settings*/, const ResolvedSettings& /*resolved*/,
    const ResolveContext& /*context*/) {
    return Value;
}

// Dense when backend = LAPACK asks for it, sparse otherwise.
Storage default_storage(const Settings& settings,
                        const ResolvedSettings& /*resolved*/,
                        const ResolveContext& /*context*/) {
    return settings.backend == Backend::LAPACK ? Storage::DENSE
                                               : Storage::SPARSE;
}

// The MKL library for the storage when the build has MKL. Without it,
// Accelerate for sparse storage when available, Eigen otherwise.
Backend default_backend(const Settings& /*settings*/,
                        const ResolvedSettings& resolved,
                        const ResolveContext& context) {
    const bool dense = resolved.storage == Storage::DENSE;
    if (context.mkl_available) {
        return dense ? Backend::LAPACK : Backend::PARDISO;
    }
    if (context.accelerate_available && !dense) {
        return Backend::ACCELERATE;
    }
    return Backend::EIGEN;
}

std::set<DataModelAttribute> default_output_attributes(
    const Settings& /*settings*/, const ResolvedSettings& /*resolved*/,
    const ResolveContext& /*context*/) {
    return {DataModelAttribute::T};
}

std::string default_output_name(const Settings& /*settings*/,
                                const ResolvedSettings& /*resolved*/,
                                const ResolveContext& context) {
    std::string name(enum_name(context.solver_type));
    std::ranges::transform(name, name.begin(), [](char letter) {
        return letter >= 'A' && letter <= 'Z'
                   ? static_cast<char>(letter - 'A' + 'a')
                   : letter;
    });
    return name;
}

// The relative residual at which the iteration on old factors stops. It
// stays in each change, and the loop stops on a change below the tolerance,
// so the error it leaves scales with the tolerance whatever this value:
// 1e-2 keeps it near 1 % of the tolerance and is the fastest such value.
double default_reuse_tolerance(const Settings& /*settings*/,
                               const ResolvedSettings& /*resolved*/,
                               const ResolveContext& /*context*/) {
    constexpr double relative_residual = 1e-2;
    return relative_residual;
}

std::map<int, int> default_iparm(const Settings& /*settings*/,
                                 const ResolvedSettings& /*resolved*/,
                                 const ResolveContext& /*context*/) {
    return {};
}

// Whether pardiso.iparm sets iparm[3] to an iteration on the previous
// factors (10 * L + K with K != 0), which works with the one-level
// factorization only.
bool iparm_iterates(const Settings& settings) {
    if (!settings.pardiso.iparm.has_value()) {
        return false;
    }
    constexpr int iterative_step = 3;
    constexpr int decimal = 10;
    const auto found = settings.pardiso.iparm->find(iterative_step);
    return found != settings.pardiso.iparm->end() &&
           found->second % decimal != 0;
}

// The reuse route, an iterating iparm[3] and the minimum degree ordering need
// the one-level factorization.
bool needs_one_level(const Settings& settings,
                     const ResolvedSettings& resolved) {
    return resolved.reuse_factors || iparm_iterates(settings);
}

// Minimum degree whenever the factorization is one-level: nested dissection
// one-level can livelock, so it is never a default.
PardisoOrdering default_pardiso_ordering(const Settings& settings,
                                         const ResolvedSettings& resolved,
                                         const ResolveContext& /*context*/) {
    const bool one_level = settings.pardiso.level == PardisoLevel::ONE_LEVEL ||
                           needs_one_level(settings, resolved);
    return one_level ? PardisoOrdering::MIN_DEGREE
                     : PardisoOrdering::NESTED_DISSECTION;
}

// One-level only when something needs it.
PardisoLevel default_pardiso_level(const Settings& settings,
                                   const ResolvedSettings& resolved,
                                   const ResolveContext& /*context*/) {
    const bool one_level =
        settings.pardiso.ordering == PardisoOrdering::MIN_DEGREE ||
        needs_one_level(settings, resolved);
    return one_level ? PardisoLevel::ONE_LEVEL : PardisoLevel::TWO_LEVEL;
}

// ---------------------------------------------------------------------------
// Applicability beyond the solver type
// ---------------------------------------------------------------------------

bool reuses_factors(const ResolvedSettings& resolved) {
    return resolved.reuse_factors;
}
bool uses_pardiso(const ResolvedSettings& resolved) {
    return resolved.backend == Backend::PARDISO;
}
bool uses_mkl(const ResolvedSettings& resolved) {
    return resolved.backend == Backend::PARDISO ||
           resolved.backend == Backend::LAPACK;
}
bool uses_accelerate(const ResolvedSettings& resolved) {
    return resolved.backend == Backend::ACCELERATE;
}
bool uses_eigen_sparse_lu(const ResolvedSettings& resolved) {
    return resolved.backend == Backend::EIGEN &&
           resolved.storage == Storage::SPARSE &&
           resolved.factorization != Factorization::CHOLESKY;
}

// ---------------------------------------------------------------------------
// Reasons: why a default has its value, or what a set value resolves to;
// empty when there is nothing to add to "default" or "set"
// ---------------------------------------------------------------------------

using Reason = std::string (*)(const Settings&, const ResolvedSettings&,
                               const ResolveContext&);

std::string backend_reason(const Settings& /*settings*/,
                           const ResolvedSettings& resolved,
                           const ResolveContext& context) {
    std::string_view library = "this build has MKL";
    if (!context.mkl_available) {
        library = resolved.backend == Backend::ACCELERATE
                      ? "this build has no MKL; Accelerate is available"
                      : "this build has no MKL";
    }
    return std::format("{}; {} storage", library,
                       resolved.storage == Storage::DENSE ? "dense" : "sparse");
}

std::string factorization_reason(const Settings& settings,
                                 const ResolvedSettings& resolved,
                                 const ResolveContext& context) {
    if (settings.factorization.value_or(Factorization::AUTO) !=
        Factorization::AUTO) {
        return {};
    }
    if (resolved.factorization == Factorization::CHOLESKY) {
        return "AUTO: no radiative couplings, so the matrix is symmetric";
    }
    if (resolved.factorization == Factorization::AUTO) {
        return "AUTO: Cholesky or LU once the model is known";
    }
    if (!context.has_radiation.value_or(false)) {
        return "AUTO: Eigen's sparse LU is preferred to its Cholesky";
    }
    return "AUTO: radiative couplings make the matrix unsymmetric";
}

std::string threads_reason(const Settings& settings,
                           const ResolvedSettings& /*resolved*/,
                           const ResolveContext& /*context*/) {
    return settings.threads.value_or(0) == 0
               ? "0: MKL's own setting (MKL_NUM_THREADS or the physical cores)"
               : std::string{};
}

std::string output_name_reason(const Settings& /*settings*/,
                               const ResolvedSettings& /*resolved*/,
                               const ResolveContext& /*context*/) {
    return "the solver type in lower case";
}

// Why the one-level factorization is needed, empty when nothing needs it.
std::string one_level_cause(const Settings& settings,
                            const ResolvedSettings& resolved) {
    if (resolved.reuse_factors) {
        return "reuse_factors needs the one-level factorization";
    }
    if (iparm_iterates(settings)) {
        return "pardiso.iparm[3] iterates on the previous factors, which "
               "needs the one-level factorization";
    }
    if (settings.pardiso.ordering == PardisoOrdering::MIN_DEGREE) {
        return "pardiso.ordering = MIN_DEGREE needs the one-level "
               "factorization";
    }
    return {};
}

std::string pardiso_level_reason(const Settings& settings,
                                 const ResolvedSettings& resolved,
                                 const ResolveContext& /*context*/) {
    if (settings.pardiso.level.has_value()) {
        return {};
    }
    const std::string cause = one_level_cause(settings, resolved);
    return cause.empty() ? "the factorization that does not livelock" : cause;
}

std::string pardiso_ordering_reason(const Settings& settings,
                                    const ResolvedSettings& resolved,
                                    const ResolveContext& /*context*/) {
    if (settings.pardiso.ordering.has_value()) {
        return {};
    }
    return resolved.pardiso.ordering == PardisoOrdering::MIN_DEGREE
               ? "one-level factorization: nested dissection with it can "
                 "livelock"
               : "nested dissection, the ordering of the two-level "
                 "factorization";
}

// ---------------------------------------------------------------------------
// The field table
// ---------------------------------------------------------------------------

// A member, or a member of a member group, of Settings or ResolvedSettings.
template <auto... Members>
struct Path;
template <auto Member>
struct Path<Member> {
    template <typename Object>
    static constexpr auto& of(Object& object) noexcept {
        return object.*Member;
    }
};
template <auto Group, auto Member>
struct Path<Group, Member> {
    template <typename Object>
    static constexpr auto& of(Object& object) noexcept {
        return (object.*Group).*Member;
    }
};

struct FieldEntry {
    SettingField info;
    bool (*is_set)(const Settings&);
    void (*unset)(Settings&);
    // Unset when the field is unset.
    std::optional<SettingValue> (*get)(const Settings&);
    void (*set)(Settings&, const SettingValue&, std::string_view);
    // Empty when the set value is valid (or the field unset).
    std::string_view (*check)(const Settings&);
    SettingValue (*get_resolved)(const ResolvedSettings&);
    // Copies the set value, if any.
    void (*take)(const Settings&, ResolvedSettings&);
    void (*take_default)(const Settings&, ResolvedSettings&,
                         const ResolveContext&);
    // Whether the field applies, given the fields resolved before it; null
    // when only the solver type decides.
    bool (*applies)(const ResolvedSettings&);
    std::string_view applies_reason;
    // Null when there is nothing to explain beyond "default" or "set".
    Reason reason;
};

template <typename Raw, typename Resolved, auto Default, auto Check>
struct FieldOps {
    using Value = std::remove_cvref_t<decltype(Raw::of(
        std::declval<Settings&>()))>::value_type;

    static bool is_set(const Settings& settings) {
        return Raw::of(settings).has_value();
    }
    static void unset(Settings& settings) { Raw::of(settings).reset(); }
    static std::optional<SettingValue> get(const Settings& settings) {
        const auto& value = Raw::of(settings);
        if (!value.has_value()) {
            return std::nullopt;
        }
        return Codec<Value>::encode(*value);
    }
    static void set(Settings& settings, const SettingValue& value,
                    std::string_view name) {
        Value decoded = Codec<Value>::decode(value, name);
        if (const std::string_view problem = Check(decoded); !problem.empty()) {
            throw std::invalid_argument(
                std::format("setting '{}' {}", name, problem));
        }
        Raw::of(settings) = std::move(decoded);
    }
    static std::string_view check(const Settings& settings) {
        const auto& value = Raw::of(settings);
        return value.has_value() ? Check(*value) : std::string_view{};
    }
    static SettingValue get_resolved(const ResolvedSettings& resolved) {
        return Codec<Value>::encode(Resolved::of(resolved));
    }
    static void take(const Settings& settings, ResolvedSettings& resolved) {
        if (const auto& value = Raw::of(settings); value.has_value()) {
            Resolved::of(resolved) = *value;
        }
    }
    static void take_default(const Settings& settings,
                             ResolvedSettings& resolved,
                             const ResolveContext& context) {
        Resolved::of(resolved) = Default(settings, resolved, context);
    }
};

template <typename Raw, typename Resolved, auto Default, auto Check>
constexpr FieldEntry field(SettingField info,
                           bool (*applies)(const ResolvedSettings&) = nullptr,
                           std::string_view applies_reason = {},
                           Reason reason = nullptr) {
    using Ops = FieldOps<Raw, Resolved, Default, Check>;
    return {.info = info,
            .is_set = &Ops::is_set,
            .unset = &Ops::unset,
            .get = &Ops::get,
            .set = &Ops::set,
            .check = &Ops::check,
            .get_resolved = &Ops::get_resolved,
            .take = &Ops::take,
            .take_default = &Ops::take_default,
            .applies = applies,
            .applies_reason = applies_reason,
            .reason = reason};
}

constexpr unsigned bit(SolverType type) {
    return 1U << static_cast<unsigned>(type);
}
constexpr auto all_types = static_cast<std::uint8_t>(
    bit(SolverType::STEADY_STATE) | bit(SolverType::CRANK_NICOLSON) |
    bit(SolverType::JACOBIAN_PROPAGATION));
constexpr auto transient_types = static_cast<std::uint8_t>(
    bit(SolverType::CRANK_NICOLSON) | bit(SolverType::JACOBIAN_PROPAGATION));
// The sensitivity solve reuses the factors of the last step for the
// derivatives, so they must belong to the current matrix.
constexpr auto reuse_types = static_cast<std::uint8_t>(
    bit(SolverType::STEADY_STATE) | bit(SolverType::CRANK_NICOLSON));

constexpr std::string_view only_pardiso = "only with backend = PARDISO";
constexpr std::string_view only_mkl = "only with backend = PARDISO or LAPACK";

using S = Settings;
using R = ResolvedSettings;

// Resolution walks the table in order: a default or an applicability rule
// may only read fields above it.
constexpr std::size_t field_count = 22;
constexpr std::array<FieldEntry, field_count> field_table{
    field<Path<&S::tolerance>, Path<&R::tolerance>,
          &constant<default_tolerance>, &positive>(
        {.name = "tolerance",
         .group = "",
         .doc = "Convergence tolerance [K]: the iterations of the steady "
                "state, or of a time step, stop when no diffusive temperature "
                "changes by more than this between two iterations.",
         .mutability = Mutability::LIVE,
         .solver_types = all_types}),
    field<Path<&S::max_iterations>, Path<&R::max_iterations>,
          &constant<default_max_iterations>, &at_least_one>(
        {.name = "max_iterations",
         .group = "",
         .doc = "Maximum iterations of the steady state, or of each time "
                "step.",
         .mutability = Mutability::LIVE,
         .solver_types = all_types}),
    field<Path<&S::storage>, Path<&R::storage>, &default_storage,
          &any_value<Storage>>(
        {.name = "storage",
         .group = "",
         .doc = "SPARSE or DENSE storage of the matrix. DENSE needs n^2 * 8 "
                "bytes per matrix and pays off for small models whose nodes "
                "are coupled to most of the others. Default: SPARSE, or DENSE "
                "with backend = LAPACK.",
         .mutability = Mutability::INIT,
         .solver_types = all_types}),
    field<Path<&S::backend>, Path<&R::backend>, &default_backend,
          &any_value<Backend>>(
        {.name = "backend",
         .group = "",
         .doc = "Library that factorizes the matrix: PARDISO (sparse) or "
                "LAPACK (dense) in builds with MKL, ACCELERATE (sparse) in "
                "macOS builds on macOS 15.5 or newer, EIGEN (both) in every "
                "build. Default: the MKL library for the storage in builds "
                "with MKL, otherwise ACCELERATE for sparse storage where "
                "available, EIGEN otherwise.",
         .mutability = Mutability::INIT,
         .solver_types = all_types},
        nullptr, {}, &backend_reason),
    field<Path<&S::factorization>, Path<&R::factorization>,
          &constant<default_factorization>, &any_value<Factorization>>(
        {.name = "factorization",
         .group = "",
         .doc = "LU, CHOLESKY or AUTO. CHOLESKY needs a model without "
                "radiative couplings. AUTO is Cholesky for such a model with "
                "PARDISO, ACCELERATE, LAPACK or dense Eigen, LU otherwise.",
         .mutability = Mutability::INIT,
         .solver_types = all_types},
        nullptr, {}, &factorization_reason),
    field<Path<&S::reuse_factors>, Path<&R::reuse_factors>,
          &constant<default_reuse_factors>, &any_value<bool>>(
        {.name = "reuse_factors",
         .group = "",
         .doc = "Solve a changed matrix by iterating on the factors of an "
                "earlier one, and refactorize only when that fails. Faster "
                "when the matrix changes little between iterations or time "
                "steps. Each iteration then solves for the temperature "
                "change, A dT = b - A T, so the iteration's residual "
                "(reuse_tolerance) stays in a change that shrinks as the "
                "solve converges, not in the temperatures. PARDISO iterates "
                "with CGS (iparm[3], overridden by pardiso.iparm) and needs "
                "the one-level factorization; LAPACK with FGMRES, on 20 "
                "iterations at most. Not with EIGEN or ACCELERATE.",
         .mutability = Mutability::INIT,
         .solver_types = reuse_types}),
    field<Path<&S::reuse_tolerance>, Path<&R::reuse_tolerance>,
          &default_reuse_tolerance, &below_one>(
        {.name = "reuse_tolerance",
         .group = "",
         .doc = "Relative residual at which the iteration on old factors "
                "stops (reuse_factors). It is left in each change, so the "
                "temperatures keep about reuse_tolerance x tolerance of it. "
                "PARDISO stops at the decade at or below it (iparm[3] = "
                "10 L + 1 for 1e-L); LAPACK's FGMRES at it. Default: 1e-2.",
         .mutability = Mutability::INIT,
         .solver_types = reuse_types},
        &reuses_factors, "only with reuse_factors = true"),
    field<Path<&S::fixed_couplings>, Path<&R::fixed_couplings>,
          &constant<default_fixed_couplings>, &any_value<bool>>(
        {.name = "fixed_couplings",
         .group = "",
         .doc = "The user's promise that the conductive and radiative "
                "couplings and the capacities stay as they are when solve() "
                "starts: no formula or callback changes them during the run. "
                "The constant part of the matrix is then built once per "
                "solve(), and the matrix is factorized again only when the "
                "temperatures change it (never without radiative couplings). "
                "Trusted, not checked: a change during the run gives wrong "
                "results. "
                "Default: false, every iteration reads the couplings and the "
                "capacities again. auto_configure() never sets it.",
         .mutability = Mutability::INIT,
         .solver_types = all_types}),
    field<Path<&S::threads>, Path<&R::threads>, &constant<default_threads>,
          &non_negative_count>({.name = "threads",
                                .group = "",
                                .doc = "Threads of the PARDISO and LAPACK "
                                       "calls; 0 keeps MKL's setting "
                                       "(MKL_NUM_THREADS).",
                                .mutability = Mutability::LIVE,
                                .solver_types = all_types},
                               &uses_mkl, only_mkl, &threads_reason),
    field<Path<&S::start_time>, Path<&R::start_time>, &constant<default_time>,
          &finite>({.name = "start_time",
                    .group = "",
                    .doc = "Start of the transient [s].",
                    .mutability = Mutability::INIT,
                    .solver_types = transient_types}),
    field<Path<&S::end_time>, Path<&R::end_time>, &constant<default_time>,
          &finite>({.name = "end_time",
                    .group = "",
                    .doc = "End of the transient [s].",
                    .mutability = Mutability::INIT,
                    .solver_types = transient_types}),
    field<Path<&S::time_step>, Path<&R::time_step>, &constant<default_time>,
          &positive>({.name = "time_step",
                      .group = "",
                      .doc = "Time step [s]. Required.",
                      .mutability = Mutability::INIT,
                      .solver_types = transient_types}),
    field<Path<&S::output_interval>, Path<&R::output_interval>,
          &constant<default_time>, &non_negative>(
        {.name = "output_interval",
         .group = "",
         .doc = "Time between stored outputs [s]; 0 stores only the start "
                "and the end.",
         .mutability = Mutability::INIT,
         .solver_types = transient_types}),
    field<Path<&S::output_attributes>, Path<&R::output_attributes>,
          &default_output_attributes, &any_value<std::set<DataModelAttribute>>>(
        {.name = "output_attributes",
         .group = "",
         .doc = "Node attributes stored at each output. Default: the "
                "temperatures only.",
         .mutability = Mutability::INIT,
         .solver_types = transient_types}),
    field<Path<&S::output_name>, Path<&R::output_name>, &default_output_name,
          &not_empty>({.name = "output_name",
                       .group = "",
                       .doc = "Name of the output model in the thermal data. "
                              "Default: the solver type in lower case.",
                       .mutability = Mutability::INIT,
                       .solver_types = transient_types},
                      nullptr, {}, &output_name_reason),
    field<Path<&S::capacity_delta>, Path<&R::capacity_delta>,
          &constant<default_capacity_delta>, &non_negative>(
        {.name = "capacity_delta",
         .group = "",
         .doc = "Added to every thermal capacity [J/K], so that nodes "
                "without capacity can be integrated.",
         .mutability = Mutability::INIT,
         .solver_types = transient_types}),
    field<Path<&S::pardiso, &PardisoSettings::ordering>,
          Path<&R::pardiso, &ResolvedPardiso::ordering>,
          &default_pardiso_ordering, &any_value<PardisoOrdering>>(
        {.name = "pardiso.ordering",
         .group = "pardiso",
         .doc = "Fill-reducing ordering. Default: NESTED_DISSECTION, or "
                "MIN_DEGREE with the one-level factorization.",
         .mutability = Mutability::INIT,
         .solver_types = all_types},
        &uses_pardiso, only_pardiso, &pardiso_ordering_reason),
    field<Path<&S::pardiso, &PardisoSettings::level>,
          Path<&R::pardiso, &ResolvedPardiso::level>, &default_pardiso_level,
          &any_value<PardisoLevel>>(
        {.name = "pardiso.level",
         .group = "pardiso",
         .doc = "Numerical factorization. Default: TWO_LEVEL, or ONE_LEVEL "
                "when reuse_factors is true or pardiso.ordering is "
                "MIN_DEGREE.",
         .mutability = Mutability::INIT,
         .solver_types = all_types},
        &uses_pardiso, only_pardiso, &pardiso_level_reason),
    field<Path<&S::pardiso, &PardisoSettings::iparm>,
          Path<&R::pardiso, &ResolvedPardiso::iparm>, &default_iparm,
          &iparm_indices>(
        {.name = "pardiso.iparm",
         .group = "pardiso",
         .doc = "PARDISO iparm entries (zero-based index: value) applied "
                "after every other setting, for diagnosis and workarounds.",
         .mutability = Mutability::INIT,
         .solver_types = all_types},
        &uses_pardiso, only_pardiso),
    field<Path<&S::pardiso, &PardisoSettings::verbose>,
          Path<&R::pardiso, &ResolvedPardiso::verbose>,
          &constant<default_verbose>, &any_value<bool>>(
        {.name = "pardiso.verbose",
         .group = "pardiso",
         .doc = "Print PARDISO's statistics to standard output.",
         .mutability = Mutability::LIVE,
         .solver_types = all_types},
        &uses_pardiso, only_pardiso),
    field<Path<&S::eigen, &EigenSettings::ordering>,
          Path<&R::eigen, &ResolvedEigen::ordering>,
          &constant<default_eigen_ordering>, &any_value<EigenOrdering>>(
        {.name = "eigen.ordering",
         .group = "eigen",
         .doc = "Ordering of Eigen's sparse LU. Default: COLAMD.",
         .mutability = Mutability::INIT,
         .solver_types = all_types},
        &uses_eigen_sparse_lu,
        "only with backend = EIGEN, sparse storage and an LU factorization"),
    field<Path<&S::accelerate, &AccelerateSettings::ordering>,
          Path<&R::accelerate, &ResolvedAccelerate::ordering>,
          &constant<default_accelerate_ordering>,
          &any_value<AccelerateOrdering>>(
        {.name = "accelerate.ordering",
         .group = "accelerate",
         .doc = "Fill-reducing ordering of Accelerate's LU and Cholesky. "
                "Default: METIS.",
         .mutability = Mutability::INIT,
         .solver_types = all_types},
        &uses_accelerate, "only with backend = ACCELERATE"),
};

constexpr std::array<SettingField, field_count> make_public_fields() {
    std::array<SettingField, field_count> fields{};
    for (std::size_t index = 0; index < field_count; ++index) {
        fields.at(index) = field_table.at(index).info;
    }
    return fields;
}

constexpr std::array<SettingField, field_count> public_fields =
    make_public_fields();

std::optional<std::size_t> find_field(std::string_view name) noexcept {
    for (std::size_t index = 0; index < field_table.size(); ++index) {
        if (field_table.at(index).info.name == name) {
            return index;
        }
    }
    return std::nullopt;
}

std::size_t field_index(std::string_view name) {
    const auto index = find_field(name);
    if (!index.has_value()) {
        throw std::invalid_argument(std::format("unknown setting '{}'", name));
    }
    return *index;
}

constexpr std::string_view solver_type_key = "solver_type";

// ---------------------------------------------------------------------------
// Resolution
// ---------------------------------------------------------------------------

struct Resolution {
    ResolvedSettings resolved;
    std::vector<std::string> contradictions;
};

// AUTO becomes LU or Cholesky once the backend and the model are known; it
// stays AUTO without a model. Eigen's sparse LU is preferred to its sparse
// Cholesky (LDLT), which is faster only on some conduction models.
void resolve_auto_factorization(ResolvedSettings& resolved,
                                const ResolveContext& context) {
    if (resolved.factorization != Factorization::AUTO) {
        return;
    }
    if (resolved.backend == Backend::EIGEN &&
        resolved.storage == Storage::SPARSE) {
        resolved.factorization = Factorization::LU;
    } else if (context.has_radiation.has_value()) {
        resolved.factorization = *context.has_radiation
                                     ? Factorization::LU
                                     : Factorization::CHOLESKY;
    }
}

void check_pardiso(const Settings& settings, const ResolvedSettings& resolved,
                   const ResolveContext& context,
                   std::vector<std::string>& contradictions) {
    if (!context.mkl_available) {
        contradictions.emplace_back("backend = PARDISO needs a build with MKL");
    }
    if (resolved.storage == Storage::DENSE) {
        contradictions.emplace_back(
            "backend = PARDISO needs storage = SPARSE; use backend = LAPACK "
            "or EIGEN for dense storage");
    }
    if (resolved.reuse_factors &&
        resolved.pardiso.level == PardisoLevel::TWO_LEVEL) {
        contradictions.emplace_back(
            "reuse_factors = true needs pardiso.level = ONE_LEVEL");
    }
    if (iparm_iterates(settings) &&
        resolved.pardiso.level == PardisoLevel::TWO_LEVEL) {
        contradictions.emplace_back(
            "pardiso.iparm[3] iterates on the previous factors and needs "
            "pardiso.level = ONE_LEVEL");
    }
    if (resolved.pardiso.ordering == PardisoOrdering::MIN_DEGREE &&
        resolved.pardiso.level == PardisoLevel::TWO_LEVEL) {
        contradictions.emplace_back(
            "pardiso.ordering = MIN_DEGREE needs pardiso.level = ONE_LEVEL: "
            "the two-level factorization uses nested dissection");
    }
}

void check_accelerate(const ResolvedSettings& resolved,
                      const ResolveContext& context,
                      std::vector<std::string>& contradictions) {
    if (!context.accelerate_available) {
        contradictions.emplace_back(
            "backend = ACCELERATE needs a macOS build with Accelerate, "
            "running on macOS 15.5 or newer");
    }
    if (resolved.storage == Storage::DENSE) {
        contradictions.emplace_back(
            "backend = ACCELERATE needs storage = SPARSE; use backend = "
            "LAPACK or EIGEN for dense storage");
    }
    if (resolved.reuse_factors) {
        contradictions.emplace_back(
            "reuse_factors = true needs backend = PARDISO or LAPACK: "
            "Accelerate cannot reuse factors");
    }
}

void check_transient(const Settings& settings, const ResolvedSettings& resolved,
                     const ResolveContext& context,
                     std::vector<std::string>& contradictions) {
    const std::string_view type = enum_name(context.solver_type);
    if (!settings.time_step.has_value()) {
        contradictions.push_back(std::format("{} needs time_step", type));
    }
    if (resolved.end_time < resolved.start_time) {
        contradictions.emplace_back("end_time is before start_time");
    }
}

void check_combinations(const Settings& settings, const ResolveContext& context,
                        Resolution& out) {
    const ResolvedSettings& resolved = out.resolved;
    auto& contradictions = out.contradictions;
    if (resolved.backend == Backend::PARDISO) {
        check_pardiso(settings, resolved, context, contradictions);
    } else if (resolved.backend == Backend::LAPACK) {
        if (!context.mkl_available) {
            contradictions.emplace_back(
                "backend = LAPACK needs a build with MKL");
        }
        if (resolved.storage == Storage::SPARSE) {
            contradictions.emplace_back(
                "backend = LAPACK needs storage = DENSE");
        }
    } else if (resolved.backend == Backend::ACCELERATE) {
        check_accelerate(resolved, context, contradictions);
    } else if (resolved.reuse_factors) {
        contradictions.emplace_back(
            "reuse_factors = true needs backend = PARDISO or LAPACK: Eigen "
            "cannot reuse factors");
    }
    if (resolved.factorization == Factorization::CHOLESKY &&
        context.has_radiation.value_or(false)) {
        contradictions.emplace_back(
            "factorization = CHOLESKY needs a model without radiative "
            "couplings");
    }
    if (context.solver_type != SolverType::STEADY_STATE) {
        check_transient(settings, resolved, context, contradictions);
    }
    if (resolved.backend == Backend::PARDISO &&
        resolved.pardiso.level == PardisoLevel::ONE_LEVEL &&
        resolved.pardiso.ordering == PardisoOrdering::NESTED_DISSECTION) {
        out.resolved.warnings.emplace_back(
            "pardiso.ordering = NESTED_DISSECTION with pardiso.level = "
            "ONE_LEVEL can livelock with MKL 2025.3 at 4 or more threads on "
            "some radiative models; MIN_DEGREE or TWO_LEVEL avoid it");
    }
}

// Every field takes its value or default; nothing is thrown.
Resolution resolve_unchecked(const Settings& settings,
                             const ResolveContext& context) {
    Resolution out;
    ResolvedSettings& resolved = out.resolved;
    resolved.solver_type = context.solver_type;
    resolved.fields.resize(field_table.size());
    for (std::size_t index = 0; index < field_table.size(); ++index) {
        const FieldEntry& entry = field_table.at(index);
        const bool for_type = entry.info.applies_to(context.solver_type);
        const bool applies =
            for_type && (entry.applies == nullptr || entry.applies(resolved));
        resolved.fields.at(index).applies = applies;
        if (entry.is_set(settings)) {
            if (const auto problem = entry.check(settings); !problem.empty()) {
                out.contradictions.push_back(
                    std::format("{} {}", entry.info.name, problem));
            }
            if (applies) {
                entry.take(settings, resolved);
                resolved.fields.at(index).origin = Origin::USER;
                continue;
            }
            resolved.not_applied.push_back(
                {.name = std::string(entry.info.name),
                 .reason = for_type
                               ? std::string(entry.applies_reason)
                               : std::format("not used by {}",
                                             enum_name(context.solver_type))});
        }
        entry.take_default(settings, resolved, context);
    }
    resolve_auto_factorization(resolved, context);
    check_combinations(settings, context, out);
    return out;
}

// ---------------------------------------------------------------------------
// Printing
// ---------------------------------------------------------------------------

std::string format_value(const SettingValue& value) {
    struct Formatter {
        std::string operator()(bool flag) const {
            return flag ? "true" : "false";
        }
        std::string operator()(std::int64_t integer) const {
            return std::format("{}", integer);
        }
        std::string operator()(double number) const {
            return std::format("{}", number);
        }
        std::string operator()(const std::string& text) const { return text; }
        std::string operator()(const std::map<int, int>& entries) const {
            std::string text = "{";
            for (const auto& [index, entry] : entries) {
                text += std::format("{}{}: {}", text.size() > 1 ? ", " : "",
                                    index, entry);
            }
            return text + "}";
        }
        std::string operator()(const std::vector<std::string>& names) const {
            std::string text = "{";
            for (const std::string& name : names) {
                text += std::format("{}{}", text.size() > 1 ? ", " : "", name);
            }
            return text + "}";
        }
    };
    return std::visit(Formatter{}, value);
}

constexpr int name_width = 19;

std::string print_set_fields(const Settings& settings) {
    std::string text = "Settings (no solver type)";
    bool any = false;
    for (const FieldEntry& entry : field_table) {
        if (const auto value = entry.get(settings); value.has_value()) {
            text += std::format("\n  {:<{}}{}", entry.info.name, name_width,
                                format_value(*value));
            any = true;
        }
    }
    return any ? text : text + ": nothing set";
}

std::string print_for_type(const Settings& settings, SolverType type) {
    const Resolution resolution = resolve_unchecked(
        settings,
        ResolveContext{.solver_type = type,
                       .has_radiation = std::nullopt,
                       .mkl_available = MKL_ENABLED,
                       .accelerate_available = accelerate_available()});
    const ResolvedSettings& resolved = resolution.resolved;
    std::string text = std::format("Settings for {}", enum_name(type));
    for (std::size_t index = 0; index < field_table.size(); ++index) {
        const auto& state = resolved.fields.at(index);
        if (!state.applies) {
            continue;
        }
        const FieldEntry& entry = field_table.at(index);
        text +=
            std::format("\n  {:<{}}{}{}", entry.info.name, name_width,
                        format_value(entry.get_resolved(resolved)),
                        state.origin == Origin::DEFAULT ? "  (default)" : "");
    }
    if (!resolved.not_applied.empty()) {
        text += std::format("\nSet but not applied to {}", enum_name(type));
        for (const NotApplied& ignored : resolved.not_applied) {
            const FieldEntry& entry = field_table.at(field_index(ignored.name));
            text += std::format("\n  {:<{}}{}  {}", ignored.name, name_width,
                                format_value(entry.get(settings).value_or(
                                    SettingValue{std::string{}})),
                                ignored.reason);
        }
    }
    for (const std::string& warning : resolved.warnings) {
        text += std::format("\nWarning: {}", warning);
    }
    for (const std::string& contradiction : resolution.contradictions) {
        text += std::format("\nError: {}", contradiction);
    }
    return text;
}

std::string join(const std::vector<std::string>& parts) {
    std::string text;
    for (const std::string& part : parts) {
        text += text.empty() ? part : "; " + part;
    }
    return text;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public functions
// ---------------------------------------------------------------------------

std::string_view to_string(SolverType type) noexcept { return enum_name(type); }
std::string_view to_string(Storage storage) noexcept {
    return enum_name(storage);
}
std::string_view to_string(Backend backend) noexcept {
    return enum_name(backend);
}
std::string_view to_string(Factorization factorization) noexcept {
    return enum_name(factorization);
}
std::string_view to_string(PardisoOrdering ordering) noexcept {
    return enum_name(ordering);
}
std::string_view to_string(PardisoLevel level) noexcept {
    return enum_name(level);
}
std::string_view to_string(EigenOrdering ordering) noexcept {
    return enum_name(ordering);
}
std::string_view to_string(AccelerateOrdering ordering) noexcept {
    return enum_name(ordering);
}
std::string_view to_string(Mutability mutability) noexcept {
    return enum_name(mutability);
}
std::string_view to_string(Origin origin) noexcept { return enum_name(origin); }

std::set<DataModelAttribute> all_dense_output_attributes() {
    return {DataModelAttribute::T,   DataModelAttribute::C,
            DataModelAttribute::QS,  DataModelAttribute::QA,
            DataModelAttribute::QE,  DataModelAttribute::QI,
            DataModelAttribute::QR,  DataModelAttribute::A,
            DataModelAttribute::APH, DataModelAttribute::EPS,
            DataModelAttribute::FX,  DataModelAttribute::FY,
            DataModelAttribute::FZ};
}

std::set<DataModelAttribute> all_output_attributes() {
    auto attributes = all_dense_output_attributes();
    attributes.insert({DataModelAttribute::KL, DataModelAttribute::KR,
                       DataModelAttribute::JAC});
    return attributes;
}

std::span<const SettingField> setting_fields() noexcept {
    return public_fields;
}

const SettingField& setting_field(std::string_view name) {
    return public_fields.at(field_index(name));
}

bool Settings::is_set(std::string_view name) const {
    if (name == solver_type_key) {
        return solver_type.has_value();
    }
    return field_table.at(field_index(name)).is_set(*this);
}

std::optional<SettingValue> Settings::get(std::string_view name) const {
    if (name == solver_type_key) {
        return solver_type.has_value()
                   ? std::optional<SettingValue>{Codec<SolverType>::encode(
                         *solver_type)}
                   : std::nullopt;
    }
    return field_table.at(field_index(name)).get(*this);
}

void Settings::set(std::string_view name, const SettingValue& value) {
    if (name == solver_type_key) {
        solver_type = Codec<SolverType>::decode(value, name);
        return;
    }
    const FieldEntry& entry = field_table.at(field_index(name));
    entry.set(*this, value, entry.info.name);
}

void Settings::unset(std::string_view name) {
    if (name == solver_type_key) {
        solver_type.reset();
        return;
    }
    field_table.at(field_index(name)).unset(*this);
}

SettingsDict Settings::to_dict() const {
    SettingsDict dict;
    if (solver_type.has_value()) {
        dict.emplace(solver_type_key, Codec<SolverType>::encode(*solver_type));
    }
    for (const FieldEntry& entry : field_table) {
        if (auto value = entry.get(*this); value.has_value()) {
            dict.emplace(entry.info.name, std::move(*value));
        }
    }
    return dict;
}

Settings Settings::from_dict(const SettingsDict& dict) {
    Settings settings;
    for (const auto& [name, value] : dict) {
        settings.set(name, value);
    }
    return settings;
}

std::string Settings::to_string() const {
    return solver_type.has_value() ? print_for_type(*this, *solver_type)
                                   : print_set_fields(*this);
}

ResolveContext resolve_context(SolverType type,
                               const ThermalMathematicalModel& model) {
    const CouplingMatrices& radiative = model.network().radiative_matrices();
    return {.solver_type = type,
            .has_radiation = radiative.get_num_diff_diff_couplings() != 0 ||
                             radiative.get_num_diff_bound_couplings() != 0,
            .mkl_available = MKL_ENABLED,
            .accelerate_available = accelerate_available()};
}

bool ResolvedSettings::applies(std::string_view name) const {
    return fields.at(field_index(name)).applies;
}

Origin ResolvedSettings::origin(std::string_view name) const {
    return fields.at(field_index(name)).origin;
}

SettingValue ResolvedSettings::value(std::string_view name) const {
    return field_table.at(field_index(name)).get_resolved(*this);
}

ResolvedSettings resolve(const Settings& settings,
                         const ResolveContext& context) {
    Resolution resolution = resolve_unchecked(settings, context);
    if (!resolution.contradictions.empty()) {
        throw std::invalid_argument(
            std::format("settings for {}: {}", enum_name(context.solver_type),
                        join(resolution.contradictions)));
    }
    return std::move(resolution.resolved);
}

SettingsExplanation explain_settings(const Settings& settings,
                                     const ResolveContext& context) {
    Resolution resolution = resolve_unchecked(settings, context);
    SettingsExplanation explanation;
    for (std::size_t index = 0; index < field_table.size(); ++index) {
        const auto& state = resolution.resolved.fields.at(index);
        if (!state.applies) {
            continue;
        }
        const FieldEntry& entry = field_table.at(index);
        SettingValue value = entry.get_resolved(resolution.resolved);
        std::string value_text = format_value(value);
        explanation.fields.push_back(
            {.name = std::string(entry.info.name),
             .value = std::move(value),
             .value_text = std::move(value_text),
             .origin = state.origin,
             .reason =
                 entry.reason == nullptr
                     ? std::string{}
                     : entry.reason(settings, resolution.resolved, context)});
    }
    explanation.resolved = std::move(resolution.resolved);
    explanation.contradictions = std::move(resolution.contradictions);
    return explanation;
}

std::vector<std::string_view> changed_init_fields(
    const Settings& before, const Settings& after,
    const ResolveContext& context) {
    const ResolvedSettings old_values =
        resolve_unchecked(before, context).resolved;
    const ResolvedSettings new_values =
        resolve_unchecked(after, context).resolved;
    std::vector<std::string_view> changed;
    for (std::size_t index = 0; index < field_table.size(); ++index) {
        const FieldEntry& entry = field_table.at(index);
        // A field left at its default in both never counts: a default that
        // follows another field keeps the value initialize() saw.
        const bool user_set = entry.is_set(before) || entry.is_set(after);
        if (entry.info.mutability == Mutability::INIT && user_set &&
            old_values.fields.at(index).applies &&
            new_values.fields.at(index).applies &&
            entry.get_resolved(old_values) != entry.get_resolved(new_values)) {
            changed.push_back(entry.info.name);
        }
    }
    return changed;
}

void check_settings_change(const Settings& before, const Settings& after,
                           const ResolveContext& context, bool initialized) {
    if (!initialized) {
        return;
    }
    const auto changed = changed_init_fields(before, after, context);
    if (changed.empty()) {
        return;
    }
    std::string names;
    for (const std::string_view name : changed) {
        names += names.empty() ? std::string(name) : std::format(", {}", name);
    }
    throw std::runtime_error(std::format(
        "{} can only change while the solver is not initialized: call "
        "deinitialize() first",
        names));
}

}  // namespace pycanha::solvers
