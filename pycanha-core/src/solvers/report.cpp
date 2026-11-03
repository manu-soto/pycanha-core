#include "pycanha-core/solvers/report.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace pycanha::solvers {

namespace {

struct Unit {
    std::string_view name;
    double bytes;
};

constexpr double kilo = 1.0e3;
constexpr double kibi = 1024.0;
constexpr int report_label_width = 17;

// Lower-case names; the parser lower-cases its input.
constexpr std::array<Unit, 10> memory_units{{
    {.name = "b", .bytes = 1.0},
    {.name = "kb", .bytes = kilo},
    {.name = "mb", .bytes = kilo * kilo},
    {.name = "gb", .bytes = kilo * kilo * kilo},
    {.name = "tb", .bytes = kilo * kilo * kilo * kilo},
    {.name = "kib", .bytes = kibi},
    {.name = "mib", .bytes = kibi * kibi},
    {.name = "gib", .bytes = kibi * kibi * kibi},
    {.name = "tib", .bytes = kibi * kibi * kibi * kibi},
    {.name = "", .bytes = 1.0},
}};

std::string_view trim(std::string_view text) {
    const auto first = text.find_first_not_of(" \t");
    if (first == std::string_view::npos) {
        return {};
    }
    const auto last = text.find_last_not_of(" \t");
    return text.substr(first, last - first + 1);
}

std::string lower(std::string_view text) {
    std::string result(text);
    std::ranges::transform(result, result.begin(), [](char letter) {
        return static_cast<char>(
            std::tolower(static_cast<unsigned char>(letter)));
    });
    return result;
}

// Bytes per unit, empty for an unknown unit.
std::optional<double> unit_bytes(std::string_view unit) {
    const auto index = static_cast<std::size_t>(std::ranges::distance(
        memory_units.begin(),
        std::ranges::find(memory_units, unit, &Unit::name)));
    if (index == memory_units.size()) {
        return std::nullopt;
    }
    return memory_units.at(index).bytes;
}

std::string format_seconds(double seconds) {
    return std::format("{:.3g} s", seconds);
}

}  // namespace

std::string format_bytes(std::int64_t bytes) {
    constexpr std::array<std::string_view, 5> prefixes = {"B", "kB", "MB", "GB",
                                                          "TB"};
    auto value = static_cast<double>(bytes);
    std::size_t prefix = 0;
    while (std::abs(value) >= kilo && prefix + 1 < prefixes.size()) {
        value /= kilo;
        ++prefix;
    }
    if (prefix == 0) {
        return std::format("{} B", bytes);
    }
    return std::format("{:.3g} {}", value, prefixes.at(prefix));
}

std::int64_t parse_memory_size(std::string_view text) {
    const std::string trimmed(trim(text));
    double number = 0.0;
    const auto [end, error] =
        std::from_chars(trimmed.data(), std::to_address(trimmed.end()), number);
    const auto parsed =
        static_cast<std::size_t>(std::distance(trimmed.data(), end));
    const auto scale =
        unit_bytes(lower(trim(std::string_view(trimmed).substr(parsed))));
    const bool valid = error == std::errc{} && parsed > 0 &&
                       scale.has_value() && number > 0.0 &&
                       std::isfinite(number);
    const double bytes = valid ? number * *scale : 0.0;
    if (!valid || bytes >= static_cast<double>(
                               std::numeric_limits<std::int64_t>::max())) {
        throw std::invalid_argument(std::format(
            "'{}' is not a memory size such as \"16 GB\", \"512 MiB\" or a "
            "positive number of bytes",
            text));
    }
    return static_cast<std::int64_t>(bytes);
}

std::string SolverMemory::to_string() const {
    return std::format("{} (factors {}{}, matrix {}, output {})",
                       format_bytes(total()), format_bytes(factors),
                       exact ? "" : " estimated", format_bytes(matrix),
                       format_bytes(output));
}

std::string_view to_string(SolveStatus status) noexcept {
    constexpr std::array<std::string_view, 4> names = {
        "NOT_RUN", "CONVERGED", "NOT_CONVERGED", "FAILED"};
    const auto index = static_cast<std::size_t>(status);
    return index < names.size() ? names.at(index) : std::string_view{"?"};
}

double SolverReport::reuse_rate() const noexcept {
    if (solves <= 0) {
        return 0.0;
    }
    return static_cast<double>(solves - factorizations) /
           static_cast<double>(solves);
}

std::string SolverReport::to_string() const {
    constexpr double percent = 100.0;
    std::string text = std::format("{} report", solver);
    const auto line = [&text](std::string_view label, std::string_view value) {
        text += std::format("\n  {:<{}}{}", label, report_label_width, value);
    };

    std::string status_text(solvers::to_string(status));
    if (status == SolveStatus::NOT_CONVERGED && steps > 0) {
        status_text +=
            std::format(" ({} of {} time steps)", nonconverged_steps, steps);
    }
    line("status", status_text);
    if (status == SolveStatus::NOT_RUN) {
        return text;
    }
    if (steps > 0) {
        line("time steps", std::format("{}", steps));
        line("iterations", std::format("{} (up to {} per step, mean {:.3g})",
                                       iterations, max_step_iterations,
                                       static_cast<double>(iterations) /
                                           static_cast<double>(steps)));
    } else {
        line("iterations", std::format("{}", iterations));
    }
    line("max. dT", std::format("{:.3g} K (last iteration)", max_dT));
    if (reuse_error > 0.0) {
        line("reuse error",
             std::format("{:.3g} K (estimated: reuse_tolerance x max. dT)",
                         reuse_error));
    }
    line("factorizations",
         std::format("{} for {} solves ({:.3g} % reused)", factorizations,
                     solves, percent * reuse_rate()));
    line("times",
         std::format("initialize {}, solve {} (factorize and solve "
                     "{}, callbacks {})",
                     format_seconds(initialize_time),
                     format_seconds(solve_time), format_seconds(linear_time),
                     format_seconds(callback_time)));
    line("memory", memory.to_string());

    if (nonconverged_steps > 0) {
        text += std::format(
            "\nNon-converged {} ({}{})", steps > 0 ? "time steps" : "solve",
            nonconverged_steps,
            nonconverged.size() < static_cast<std::size_t>(nonconverged_steps)
                ? std::format(", first {} listed", nonconverged.size())
                : "");
        for (const NonConvergedStep& step : nonconverged) {
            text += std::format(
                "\n  step {}  t = {} s  {} iterations  max. dT {:.3g} K at "
                "node {}",
                step.step, step.time, step.iterations, step.max_dT, step.node);
        }
    }
    if (!warnings.empty()) {
        text += "\nWarnings";
        for (const std::string& warning : warnings) {
            text += std::format("\n  {}", warning);
        }
    }
    return text;
}

ReportDict SolverReport::to_dict() const {
    std::vector<std::int64_t> step_numbers;
    std::vector<double> times;
    std::vector<std::int64_t> step_iterations;
    std::vector<double> changes;
    std::vector<std::int64_t> nodes;
    for (const NonConvergedStep& step : nonconverged) {
        step_numbers.push_back(step.step);
        times.push_back(step.time);
        step_iterations.push_back(step.iterations);
        changes.push_back(step.max_dT);
        nodes.push_back(step.node);
    }
    return {
        {"solver", solver},
        {"status", std::string(solvers::to_string(status))},
        {"converged", converged()},
        {"steps", std::int64_t{steps}},
        {"iterations", std::int64_t{iterations}},
        {"max_step_iterations", std::int64_t{max_step_iterations}},
        {"nonconverged_steps", std::int64_t{nonconverged_steps}},
        {"nonconverged.step", step_numbers},
        {"nonconverged.time", times},
        {"nonconverged.iterations", step_iterations},
        {"nonconverged.max_dT", changes},
        {"nonconverged.node", nodes},
        {"max_dT", max_dT},
        {"reuse_error", reuse_error},
        {"factorizations", std::int64_t{factorizations}},
        {"solves", std::int64_t{solves}},
        {"reuse_rate", reuse_rate()},
        {"initialize_time", initialize_time},
        {"solve_time", solve_time},
        {"callback_time", callback_time},
        {"linear_time", linear_time},
        {"memory.total", memory.total()},
        {"memory.factors", memory.factors},
        {"memory.matrix", memory.matrix},
        {"memory.output", memory.output},
        {"memory.exact", memory.exact},
        {"warnings", warnings},
    };
}

}  // namespace pycanha::solvers
