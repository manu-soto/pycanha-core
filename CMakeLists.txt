cmake_minimum_required(VERSION 3.25)

# The following options can be used to customize the build

# - PYCANHA_OPTION_LIBRARY:
# Deactivating this options is useful if you want to build only the documentation.
# In that case, you should desactivate the other options as well (tests, benchmarks, etc.), 
# as they depend on the library.

# - PYCANHA_OPTION_WARNINGS:
# Enable/disable compiler warnings. The warnings are defined as flags for MSVC, GCC and Clang in
# the Warnings.cmake file

# - PYCANHA_OPTION_WARNINGS_AS_ERRORS:
# Treat warnings as error in the compiler, clang-tidy and cppcheck

# - BUILD_TESTING:
# Variable set to ON by CMake when calling include(CTest). However, we control
# the tests from conan, which uses the same variable. If BUILD_TESTING is not set to OFF
# (that is ON or not set), the tests will be built. TODO: Use another variable?

# - PYCANHA_OPTION_COVERAGE:
# Build with Clang source-based coverage instrumentation
# (-fprofile-instr-generate -fcoverage-mapping). The report is produced in CI
# from the .profraw files with llvm-profdata/llvm-cov. Requires Clang.

# - PYCANHA_OPTION_INCLUDE_WHAT_YOU_USE:
# Not working right now.

# - PYCANHA_OPTION_CLANG_TIDY:
# Add a target for running clang-tidy. 
# Not necessary if you use the clang-tidy ingegration in your IDE (eg. VSCode)

# - PYCANHA_OPTION_CPPCHECK:
# Add a target to use cppcheck. 
# Not necessary if you run cppcheck from the command line or through your IDE.

# - PYCANHA_OPTION_SANITIZE_ADDR:
# Enable address sanitizer. Only works on Unix systems.

# - PYCANHA_OPTION_SANITIZE_UNDEF:
# Enable undefined behavior sanitizer. Only works on Unix systems.

# - PYCANHA_OPTION_LTO:
# Enable Link Time Optimization.

# - PYCANHA_OPTION_DOCS:
# Add a 'doc' target to build documentation.


set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Create a compile_commands.json file for clang-tidy and cppcheck 

option(PYCANHA_OPTION_LIBRARY              "Include the Library"                           ON  )
option(PYCANHA_OPTION_USE_MKL              "Enable Intel MKL accelerated solvers."         ON  )
option(PYCANHA_OPTION_USE_ACCELERATE       "Enable Apple Accelerate solvers (macOS only)." ON  )
option(PYCANHA_OPTION_LTO                  "Enable Link Time Optimization."                OFF )
option(PYCANHA_OPTION_DOCS                 "Add a 'doc' target to build documentation."    OFF )
option(PYCANHA_OPTION_WARNINGS             "Add warnings to targets."                      ON  )
option(PYCANHA_OPTION_WARNINGS_AS_ERRORS   "Treat warnings as errors."                     OFF )

option(PYCANHA_OPTION_PROFILING            "Enable profiling logging."                     OFF )
option(PYCANHA_OPTION_ACTIVATE_ALL_LOGS_OVERRIDE
       "Compile all spdlog levels regardless of build type."                               OFF )

option(PYCANHA_OPTION_COVERAGE             "Enable a Code Coverage build."                 OFF )
option(PYCANHA_OPTION_INCLUDE_WHAT_YOU_USE "Add iwyu to targets."                          OFF ) # Broken right now
option(PYCANHA_OPTION_CLANG_TIDY           "Enable to add clang tidy."                     OFF )
option(PYCANHA_OPTION_CPPCHECK             "Add cppcheck."                                 OFF ) 
option(PYCANHA_OPTION_SANITIZE_ADDR        "Enable to sanitize address (Unix only)."       OFF )
option(PYCANHA_OPTION_SANITIZE_UNDEF       "Enable to sanitize undefined (Unix only)."     OFF )
option(PYCANHA_OPTION_RAYTRACING           "Enable the GPU RT radiative engine."           ON  )




project(
    "pycanha-core"
    VERSION ${CONAN_PROJECT_VERSION}
    LANGUAGES CXX)

# The macOS radiative backend is written in Objective-C++ (the Metal API is
# Objective-C). CMake only accepts enable_language() at the top level, not in
# the subdirectory that owns those sources.
if(APPLE AND PYCANHA_OPTION_RAYTRACING)
    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 23)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
    set(CMAKE_OBJCXX_EXTENSIONS OFF)
    # The backend requires GPU family Apple9, whose enumerator arrived in the
    # macOS 14 SDK; an older deployment target would only fail later, as
    # availability warnings on every capability query.
    if(CMAKE_OSX_DEPLOYMENT_TARGET AND
       CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS 14.0)
        message(FATAL_ERROR
            "The Metal radiative backend needs a macOS 14 or newer deployment "
            "target (CMAKE_OSX_DEPLOYMENT_TARGET is "
            "${CMAKE_OSX_DEPLOYMENT_TARGET}); build with "
            "PYCANHA_OPTION_RAYTRACING=False to get the stub instead")
    endif()
endif()



# Set the C++ stanadard globally
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)



# CMake target names
set(LIB_NAME "pycanha-core")
set(TEST_TARGET_NAME "tests")
set(EXAMPLES_TARGET_PREFIX "example")
set(BENCHMARK_TARGET_NAME "benchmarks")


# CMAKE MODULES
message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/)


include(Warnings)
include(Tools)
include(LTO)
include(ConfigSafeGuards)



if(PYCANHA_OPTION_SANITIZE_ADDR OR PYCANHA_OPTION_SANITIZE_UNDEF)
    include(Sanitizer)
    add_sanitizer_flags()
endif()

if(PYCANHA_OPTION_COVERAGE)
    # Clang source-based code coverage. Instruments every target defined in the
    # subdirectories below (library + tests); the CI workflow runs the tests and
    # turns the resulting .profraw into an lcov report with llvm-profdata/llvm-cov.
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR "PYCANHA_OPTION_COVERAGE requires a Clang compiler")
    endif()
    add_compile_options(-fprofile-instr-generate -fcoverage-mapping)
    add_link_options(-fprofile-instr-generate)
endif()

if(PYCANHA_OPTION_LTO)
    find_lto(CXX)
endif()



# LIBRARY
if (PYCANHA_OPTION_LIBRARY)
    add_subdirectory(pycanha-core)
endif()

# DOCS
if(PYCANHA_OPTION_DOCS)
    # Include "Docs.cmake" to add the target "doc"
    include(Docs)
endif()

# TESTS
if (NOT BUILD_TESTING STREQUAL OFF)
    enable_testing() #TODO: Not sure if necessary
    add_subdirectory(test)
endif()

# EXAMPLES
#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples)
