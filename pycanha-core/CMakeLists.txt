# Sources and Headers
#set(LIBRARY_SOURCES 
#        ""
#)

#set(LIBRARY_HEADERS 
#        "${CMAKE_CURRENT_SOURCE_DIR}/my_lib.hpp"
#        "${CMAKE_CURRENT_SOURCE_DIR}/core/sum.hpp"
#        "${CMAKE_CURRENT_SOURCE_DIR}/core/multiply.hpp"
#        "${CMAKE_CURRENT_SOURCE_DIR}/utils/package_info.hpp"
#)

# Configure the header "config.hpp" to pass the library name and version
set(LIB_VERSION ${PROJECT_VERSION})
set(PYCANHA_USE_MKL_INT 0)

#set(LIBRARY_INCLUDES "./" "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/configured_files/include")
#set(LIBRARY_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/include")

# External dependencies
#-----------------------------

# Eigen 5 from Conan currently exports Eigen3 CMake config/target names.
find_package(Eigen3 REQUIRED CONFIG)
find_package(spdlog REQUIRED CONFIG)
find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(symengine REQUIRED CONFIG)

# Raytracing backends. Off => only the stub (is_available() == false).
#  - Vulkan (Windows/Linux): vulkan-headers + volk (runtime loader, no
#    link-time Vulkan dependency) + VMA.
#  - Metal (macOS): the Metal and Foundation system frameworks, no extra
#    packages. device.cpp and scene.cpp are the same sources for both
#    backends — they only ever name the backend-private detail:: types — but
#    on Apple they are compiled as Objective-C++ (see below), since that is
#    what the backend's own headers are written in.
if(PYCANHA_OPTION_RAYTRACING)
    include(Slang)
    pycanha_fetch_slang()
    if(APPLE)
        set(PYCANHA_RADIATIVE_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/src/radiative/device.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/radiative/scene.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/radiative/mtl_accum.mm
            ${CMAKE_CURRENT_SOURCE_DIR}/src/radiative/mtl_device.mm
            ${CMAKE_CURRENT_SOURCE_DIR}/src/radiative/mtl_scene.mm
        )
    else()
        find_package(volk REQUIRED CONFIG)
        find_package(VulkanHeaders REQUIRED CONFIG)
        find_package(VulkanMemoryAllocator REQUIRED CONFIG)
        set(PYCANHA_RADIATIVE_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/src/radiative/device.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/radiative/scene.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/radiative/vk_accum.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/radiative/vk_device.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/radiative/vk_scene.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/radiative/vma_impl.cpp
        )
    endif()
else()
    set(PYCANHA_RADIATIVE_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/radiative/radiative_stub.cpp
    )
endif()

if(PYCANHA_USE_FETCHCONTENT_MANIFOLD)
    include(FetchContent)

    set(MANIFOLD_TEST OFF CACHE BOOL "" FORCE)
    set(MANIFOLD_CROSS_SECTION OFF CACHE BOOL "" FORCE)
    set(MANIFOLD_PAR OFF CACHE BOOL "" FORCE)
    set(MANIFOLD_PYBIND OFF CACHE BOOL "" FORCE)

    # Manifold's own CMakeLists defaults BUILD_SHARED_LIBS to ON, so it would
    # build a shared library (manifold.dll / libmanifold.so / libmanifold.dylib).
    # We want a *static* manifold that gets archived into the package and linked
    # statically into consumers (see the package_info() note in conanfile.py):
    #  - conanfile.py only packages *.a / *.lib, so a shared manifold is not
    #    installed on Linux/macOS and the consumer's find_package() fails with
    #    "Library 'manifold' not found in package".
    #  - a static manifold also removes the need to bundle the manifold runtime
    #    (manifold.dll) alongside the Python bindings.
    # Build manifold with PIC so it can be linked into shared consumers (the
    # nanobind extension); Conan sets CMAKE_POSITION_INDEPENDENT_CODE from fPIC.
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        manifold
        GIT_REPOSITORY https://github.com/elalish/manifold.git
        GIT_TAG v${PYCANHA_OPTION_MANIFOLD_VERSION}
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(manifold)
    set(PYCANHA_MANIFOLD_TARGET manifold)
else()
    find_package(manifold REQUIRED CONFIG)
    set(PYCANHA_MANIFOLD_TARGET manifold::manifold)
endif()

# Normalize MANIFOLD_PAR for clang-tidy/tooling.
# TODO: Check if needed
get_target_property(_pycanha_manifold_compile_options
                    ${PYCANHA_MANIFOLD_TARGET}
                    INTERFACE_COMPILE_OPTIONS)
if(_pycanha_manifold_compile_options)
    list(TRANSFORM _pycanha_manifold_compile_options
         REPLACE "^-DMANIFOLD_PAR=-1$" "-DMANIFOLD_PAR=(-1)")
    set_target_properties(${PYCANHA_MANIFOLD_TARGET}
                          PROPERTIES
                          INTERFACE_COMPILE_OPTIONS
                          "${_pycanha_manifold_compile_options}")
endif()


# MyLib Library
## For a compiled library:
#add_library(${LIB_NAME} STATIC ${LIBRARY_SOURCES} ${LIBRARY_HEADERS})
#target_include_directories(${LIB_NAME} PUBLIC ${LIBRARY_INCLUDES})
#target_link_libraries( ${LIB_NAME} PRIVATE 
#    Eigen3::Eigen
#)

# Change from INTERFACE to STATIC or SHARED, depending on your needs
# Add all source files after the library name
add_library(${LIB_NAME} STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gmm/ids.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gmm/geometrymodel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gmm/mesh_parts.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gmm/cutting/cutter_proxy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gmm/cutting/manifold_cut_backend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gmm/cutting/cut_mesh_ops.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gmm/cutting/proxy_shell.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gmm/mesh/thermal_mesh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gmm/mesh/uv_mesher.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gmm/mesh/uv_mesher_cube.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gmm/mesh/uv_mesher_disc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gmm/mesh/uv_mesher_planar.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gmm/mesh/uv_mesher_revolution.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gmm/mesh/uv_mesher_sphere.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gmm/ops/distance.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gmm/ops/transform.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gmm/primitives/cone.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gmm/primitives/cube.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gmm/primitives/cylinder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gmm/primitives/disc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gmm/primitives/face_pair_geometry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gmm/primitives/paraboloid.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gmm/primitives/quadrilateral.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gmm/primitives/rectangle.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gmm/primitives/sphere.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gmm/primitives/triangle.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gmm/primitives/triangular_prism.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gmm/scene/coordinate_transformation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gmm/scene/geometry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gmm/scene/geometry_item.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gmm/scene/resolve.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gmm/scene/geometry_group.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gmm/scene/geometry_group_cutted.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gmm/materials/bulk_material.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gmm/materials/optical_material.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gmm/materials/color.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/conduction/builder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/conduction/links.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/conduction/network_part.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/conduction/profile.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/io/esatan.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/thermaldata/data_model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/thermaldata/dense_matrix_time_series.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/thermaldata/dense_time_series.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/thermaldata/lookup_table.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/thermaldata/named_constants.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/thermaldata/sparse_time_series.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/parameters/formula.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/parameters/parameters.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/parameters/formulas.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/parameters/variable.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/tmm/coupling.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tmm/couplings.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tmm/couplingmatrices.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tmm/conductivecouplings.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tmm/radiativecouplings.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tmm/thermalnetwork.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tmm/node.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tmm/nodes.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tmm/thermalmathematicalmodel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tmm/thermalmodel.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/logger.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/SparseUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/solvers/callback_registry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/solvers/solver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/solvers/settings.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/solvers/report.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/solvers/auto_configure.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/solvers/solver_registry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/solvers/factorizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/solvers/accelerate.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/solvers/pardiso.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/solvers/sparse_assembly.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/solvers/sparse_network.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/solvers/dense_assembly.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/solvers/dense_network.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/solvers/steady_state.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/solvers/crank_nicolson.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/solvers/jacobian_propagation.cpp

    # Pure-CPU radiative services, built regardless of the raytracing option.
    ${CMAKE_CURRENT_SOURCE_DIR}/src/radiative/aggregate.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/radiative/closure_projection.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/radiative/exchange_assemble.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/radiative/gebhart.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/radiative/pair_walk.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/radiative/vf_assemble.cpp
    ${PYCANHA_RADIATIVE_SOURCES}
)
# The two matrix assemblies interpolate a table and blend two estimates, both
# of the form a + f * (b - a). GCC and Clang default to contracting that into
# an FMA, which rounds once instead of twice — a different result from a
# compiler that does not contract, which would silently cost the reproducibility
# these files are otherwise written to guarantee. MSVC does not contract under
# its default /fp:precise, so no equivalent flag is needed there.
if(NOT MSVC)
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/src/radiative/closure_projection.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/radiative/exchange_assemble.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/radiative/pair_walk.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/radiative/vf_assemble.cpp
        PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()

target_include_directories(${LIB_NAME}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
# CDT is entered via #include <CDT/include/CDT.h> through this parent directory.
# It MUST be a separate call with SYSTEM first (SYSTEM is only recognised as the
# first argument after the target); folding it into the call above left CDT as a
# non-system -I, so its third-party-header warnings tripped our -Werror.
target_include_directories(${LIB_NAME} SYSTEM PRIVATE
    $<BUILD_INTERFACE:${cdt_SOURCE_DIR}>)

target_link_libraries( ${LIB_NAME}
    PUBLIC
    Eigen3::Eigen
    spdlog::spdlog
    symengine
    PRIVATE
    HDF5::HDF5
    ${PYCANHA_MANIFOLD_TARGET}
)

if(PYCANHA_OPTION_RAYTRACING)
    if(APPLE)
        # PRIVATE: public headers expose no Metal types.
        target_link_libraries(${LIB_NAME} PRIVATE
            "-framework Metal"
            "-framework Foundation"
        )
        # device.cpp and scene.cpp are shared with the Vulkan backend, but on
        # Apple the backend headers they include are Objective-C++, so they
        # compile as OBJCXX here. Building every one of these translation
        # units with automatic reference counting is what makes the Metal
        # objects held by the backend classes ordinary C++ members with
        # ordinary lifetimes — and it must be ALL of them, since they share
        # those class definitions.
        set_source_files_properties(
            ${CMAKE_CURRENT_SOURCE_DIR}/src/radiative/device.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/radiative/scene.cpp
            PROPERTIES LANGUAGE OBJCXX
        )
        set_source_files_properties(
            ${CMAKE_CURRENT_SOURCE_DIR}/src/radiative/device.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/radiative/scene.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/src/radiative/mtl_accum.mm
            ${CMAKE_CURRENT_SOURCE_DIR}/src/radiative/mtl_device.mm
            ${CMAKE_CURRENT_SOURCE_DIR}/src/radiative/mtl_scene.mm
            PROPERTIES COMPILE_OPTIONS "-fobjc-arc"
        )
        set(PYCANHA_KERNEL_TARGET metal)
    else()
        # PRIVATE: public headers expose no Vulkan types.
        target_link_libraries(${LIB_NAME} PRIVATE
            volk::volk
            Vulkan::Headers
            GPUOpen::VulkanMemoryAllocator
        )
        # VMA sources its Vulkan entry points from volk (one config for all TUs).
        target_compile_definitions(${LIB_NAME} PRIVATE
            VMA_STATIC_VULKAN_FUNCTIONS=0
            VMA_DYNAMIC_VULKAN_FUNCTIONS=1
        )
        set(PYCANHA_KERNEL_TARGET spirv)
    endif()

    # Slang kernels: compiled to SPIR-V or to a Metal library, checked by the
    # respective shader compiler, and embedded as generated headers on every
    # build (compilation IS the shader syntax check).
    pycanha_add_slang_kernel(vf
        SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/radiative/kernels/vf.slang
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/radiative/kernels/common.slang
        TARGET ${PYCANHA_KERNEL_TARGET}
    )
    pycanha_add_slang_kernel(exchange
        SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/radiative/kernels/exchange.slang
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/radiative/kernels/common.slang
        TARGET ${PYCANHA_KERNEL_TARGET}
    )
    pycanha_add_slang_kernel(solar
        SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/radiative/kernels/solar.slang
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/radiative/kernels/common.slang
        TARGET ${PYCANHA_KERNEL_TARGET}
    )
    add_dependencies(${LIB_NAME} pycanha_kernel_vf pycanha_kernel_exchange
        pycanha_kernel_solar)
    if(PYCANHA_KERNEL_TARGET STREQUAL "spirv")
        # TraceSettings::sampler == Sobol: each kernel compiled again with the
        # QMC emission in (common.slang, PYCANHA_QMC), so that the default
        # kernels carry none of it. Metal selects it with a push-constant flag.
        foreach(_kernel IN ITEMS vf exchange solar)
            pycanha_add_slang_kernel(${_kernel}_qmc
                SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/radiative/kernels/${_kernel}.slang
                DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/radiative/kernels/common.slang
                DEFINES PYCANHA_QMC
                TARGET spirv
            )
            add_dependencies(${LIB_NAME} pycanha_kernel_${_kernel}_qmc)
        endforeach()
    endif()
    target_include_directories(${LIB_NAME} PRIVATE ${PYCANHA_KERNEL_EMBED_DIR})
endif()

# ============================================================================
# Intel MKL Configuration
# ============================================================================
if(PYCANHA_OPTION_USE_MKL)
    # Set MKL configuration before find_package
    # See MKLConfig.cmake for all available options
    set(MKL_INTERFACE "lp64")                          # LP64 = 32-bit integers
    set(MKL_THREADING "intel_thread")                  # Intel OpenMP
    set(MKL_LINK "dynamic")                            # Simplified: only dynamic mode
    # If static mode is needed again:
    #  1) restore PYCANHA_OPTION_MKL_LINK in conan/cmake options,
    #  2) pass it here as MKL_LINK,
    #  3) restore static Windows library names in conanfile.py:_mkl_system_libs().

    # MKL paths are resolved in Conan (pip install + discovery) and exposed via
    # MKL_DIR/MKL_ROOT to CMake. Keep CMake side minimal, similar to Eigen.
    if(NOT DEFINED CACHE{MKL_DIR})
        message(FATAL_ERROR
            "MKL_DIR is not defined. Configure with Conan so pip-installed MKL "
            "is discovered and exported to CMake.")
    endif()

    # NO_DEFAULT_PATH ensures CMake ONLY looks in MKL_DIR — cannot find system MKL
    find_package(MKL CONFIG REQUIRED NO_DEFAULT_PATH PATHS "${MKL_DIR}")

    message(STATUS "Intel MKL found: ${MKL_VERSION}")
    message(STATUS "  Interface: ${MKL_INTERFACE}, Threading: ${MKL_THREADING}, Link: ${MKL_LINK}")
    message(STATUS "  MKL imported targets: ${MKL_IMPORTED_TARGETS}")

    # Link MKL using the official target (includes headers, compile options, and libraries)
    target_link_libraries(${LIB_NAME} PUBLIC MKL::MKL)

    # Configure Eigen to use MKL BLAS/LAPACK
    target_compile_definitions(${LIB_NAME} PUBLIC 
        EIGEN_USE_MKL_ALL      # Enable MKL for all Eigen operations
    )

    # Configure pycanha to use MKL
    target_compile_definitions(${LIB_NAME} PUBLIC PYCANHA_USE_MKL=1)
    set(PYCANHA_USE_MKL_INT 1)
    # MKL_VERSION is set by find_package(MKL) above
else()
    # MKL disabled - use Eigen native solvers
    set(PYCANHA_USE_MKL_INT 0)
    set(MKL_VERSION "N/A")
    target_compile_definitions(${LIB_NAME} PUBLIC PYCANHA_USE_MKL=0)
    message(STATUS "Intel MKL disabled - using Eigen native solvers")
endif()
# ============================================================================

# ============================================================================
# Apple Accelerate Configuration
# ============================================================================
# The sparse solvers of the Accelerate system framework (macOS only): no extra
# dependency. LU needs macOS 15.5 at run time, which is checked when the
# solver is created.
if(APPLE AND PYCANHA_OPTION_USE_ACCELERATE)
    target_link_libraries(${LIB_NAME} PRIVATE "-framework Accelerate")
    target_compile_definitions(${LIB_NAME} PUBLIC PYCANHA_USE_ACCELERATE=1)
    message(STATUS "Apple Accelerate enabled - sparse LU and Cholesky solvers")
else()
    target_compile_definitions(${LIB_NAME} PUBLIC PYCANHA_USE_ACCELERATE=0)
endif()
# ============================================================================

# ============================================================================
# Logging and Profiling Configuration
# ============================================================================
# SPDLOG_ACTIVE_LEVEL controls compile-time log-level stripping.
if(PYCANHA_OPTION_ACTIVATE_ALL_LOGS_OVERRIDE)
    set(_pycanha_spdlog_active_level_define "SPDLOG_LEVEL_TRACE")
else()
    set(_pycanha_spdlog_active_level_define
        "$<IF:$<CONFIG:Debug>,SPDLOG_LEVEL_TRACE,SPDLOG_LEVEL_INFO>")
endif()

target_compile_definitions(${LIB_NAME} PUBLIC
    "SPDLOG_ACTIVE_LEVEL=${_pycanha_spdlog_active_level_define}")
message(STATUS "PYCANHA SPDLOG_ACTIVE_LEVEL=${_pycanha_spdlog_active_level_define}")

# PYCANHA_PROFILING gates the PYCANHA_PROFILE_SCOPE() macro and the
# spdlog profiling logger.
if(PYCANHA_OPTION_PROFILING)
    target_compile_definitions(${LIB_NAME} PUBLIC PYCANHA_PROFILING)
    message(STATUS "Profiling logging enabled (PYCANHA_PROFILING)")
else()
    message(STATUS "Profiling logging disabled")
endif()
# ============================================================================

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/${LIB_NAME}/config.hpp.in" 
    "${CMAKE_CURRENT_SOURCE_DIR}/include/${LIB_NAME}/config.hpp" 
)

# For a header-only library:
#add_library(${LIB_NAME} INTERFACE)
#target_include_directories(${LIB_NAME} 
#    INTERFACE 
#    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#    $<INSTALL_INTERFACE:include>
#)

#target_link_libraries( ${LIB_NAME} INTERFACE 
#    Eigen3::Eigen
#    CDT::CDT
#)

target_set_warnings(
    TARGET
    ${LIB_NAME}
    ENABLE
    ${PYCANHA_OPTION_WARNINGS}
    AS_ERRORS
    ${PYCANHA_OPTION_WARNINGS_AS_ERRORS})


if(${PYCANHA_OPTION_LTO})
    target_enable_lto(${LIB_NAME} optimized)
endif()

# Add clang-tidy to the library
if (PYCANHA_OPTION_CLANG_TIDY)
  add_clang_tidy_to_target(${LIB_NAME})
  # The clang-tidy job runs without building the library objects, so the
  # Slang kernel headers (e.g. vf_spv.h) generated by pycanha_kernel_vf would
  # be missing when clang-tidy parses the radiative sources. Force the kernels
  # to be generated first.
  if (PYCANHA_OPTION_RAYTRACING AND TARGET ${LIB_NAME}_clangtidy)
    add_dependencies(${LIB_NAME}_clangtidy pycanha_kernel_vf
        pycanha_kernel_exchange pycanha_kernel_solar)
  endif()
endif()

