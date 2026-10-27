
set(UNIT_TEST_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/dependencies/test_conan_dependencies.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/conduction/test_builder.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/conduction/test_capacitance.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/conduction/test_links_invariants.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/conduction/test_links_planar.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/conduction/test_links_revolution.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/conduction/test_links_triangle.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/conduction/test_network_part.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/conduction/test_profile.cpp"
#  "${CMAKE_CURRENT_SOURCE_DIR}/unit/gmm/mesh/test_export_meshes.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/gmm/materials/test_materials.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/gmm/mesh/test_mesh_ops.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/gmm/mesh/test_mesh_winding.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/gmm/mesh/test_thermal_mesh.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/gmm/mesh/test_uv_mesher_cube.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/gmm/mesh/test_uv_mesher_disc.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/gmm/mesh/test_uv_mesher_planar.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/gmm/mesh/test_uv_mesher_revolution.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/gmm/mesh/test_uv_mesher_sphere.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/gmm/cutting/test_manifold_dependency.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/gmm/cutting/test_cutter_proxy.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/gmm/cutting/test_manifold_cut_backend.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/gmm/cutting/test_proxy_shell.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/gmm/ops/test_distance.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/gmm/ops/test_transform.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/gmm/primitives/test_cone.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/gmm/primitives/test_cube.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/gmm/primitives/test_cylinder.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/gmm/primitives/test_disc.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/gmm/primitives/test_face_pair_geometry.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/gmm/primitives/test_paraboloid.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/gmm/primitives/test_quadrilateral.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/gmm/primitives/test_rectangle.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/gmm/primitives/test_sphere.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/gmm/primitives/test_triangle.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/gmm/primitives/test_triangular_prism.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/gmm/scene/test_coordinate_transformation.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/gmm/scene/test_face_numbering.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/gmm/scene/test_group.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/gmm/scene/test_resolve_cuts.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/gmm/scene/test_resolve_scale.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/gmm/scene/test_item.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/gmm/test_face_node_mapping.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/gmm/test_geometry_model.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/gmm/test_material_table.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/gmm/test_mesh_parts.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/gmm/test_primitive_variant.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/radiative/test_aggregate.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/radiative/test_cross_backend_bitdiff.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/radiative/test_device.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/radiative/test_exchange_assemble.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/radiative/test_exchange_kernels.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/radiative/test_exchange_specular.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/radiative/test_gebhart.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/radiative/test_planet.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/radiative/test_rng_golden.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/radiative/test_solar_kernel.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/radiative/test_types.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/radiative/test_vf_analytic.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/radiative/test_vf_assemble.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/radiative/test_vf_properties.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/parameters/test_formula.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/parameters/test_variable.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/parameters/test_smart_formula.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/solvers/test_steady_state.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/solvers/test_crank_nicolson.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/solvers/test_crank_nicolson_outputs.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/solvers/test_crank_nicolson_arithmetic.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/solvers/test_jacobian_propagation.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/solvers/test_callback_registry.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/solvers/test_model_changed.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/solvers/test_settings.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/solvers/test_report.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/solvers/test_memory_and_auto.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/solvers/test_dense_storage.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/solvers/test_fixed_couplings.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/tmm/test_bulk.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/tmm/test_coupling.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/tmm/test_couplings.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/tmm/test_node.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/tmm/test_sparse_utils.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/tmm/test_structure_versions.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/tmm/test_nodes.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/tmm/test_thermal_network.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/tmm/test_thermal_mathematical_model.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/tmm/test_esatan_reader.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/tmm/parameters/test_parameters.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/tmm/parameters/test_internal_parameters.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/tmm/parameters/test_formulas.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/tmm/thermaldata/test_interpolation_utils.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/tmm/thermaldata/test_lookup_table.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/tmm/thermaldata/test_dense_time_series.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/tmm/thermaldata/test_dense_matrix_time_series.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/tmm/thermaldata/test_data_model.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/tmm/thermaldata/test_data_table_store.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/tmm/thermaldata/test_sparse_time_series.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/tmm/thermaldata/test_thermaldata.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/unit/utils/test_logger.cpp"
)

set(API_TEST_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/api/test_overview.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/api/test_geometry_model.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/api/test_gmm_to_tmm.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/api/test_thermal_network.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/api/test_parameters.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/api/test_radiative.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/api/test_formulas.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/api/test_solvers.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/api/test_callbacks.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/api/test_thermaldata.cpp"
)

# The Metal backend has one layout conversion with no Vulkan counterpart (the
# instance transform), so its test is Objective-C++ and only exists where that
# backend is built.
if(APPLE AND PYCANHA_OPTION_RAYTRACING)
  list(APPEND UNIT_TEST_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/unit/radiative/test_mtl_transform.mm")
endif()

# Likewise the Vulkan backend has a per-launch emitter limit (the dispatch's
# Y workgroup count) that Metal grids do not, and its test reaches the limit
# through a hook only the Vulkan backend defines.
if(PYCANHA_OPTION_RAYTRACING AND NOT APPLE)
  list(APPEND UNIT_TEST_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/unit/radiative/test_vk_dispatch_batching.cpp")
endif()

set(TEST_SOURCES
  ${UNIT_TEST_SOURCES}
  ${API_TEST_SOURCES}
)

set(TEST_HEADERS "")

#Tests need Catch2
find_package(Catch2 REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(manifold REQUIRED CONFIG)
find_package(SymEngine REQUIRED CONFIG)

# Create the target "tests_lib". This target only includes the library and prints info
# It is useful for cppcheck to avoid warnings with Catch2
add_executable(${TEST_TARGET_NAME}_lib "${CMAKE_CURRENT_SOURCE_DIR}/lib_info_test.cpp")
target_link_libraries(${TEST_TARGET_NAME}_lib PRIVATE ${LIB_NAME})

#Create the target "tests" 
add_executable(${TEST_TARGET_NAME} ${TEST_SOURCES} ${TEST_HEADERS})

target_sources(
  ${TEST_TARGET_NAME}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/lsan_suppressions.cpp)

#Link with the lib and catch2
target_link_libraries(${TEST_TARGET_NAME} PRIVATE ${LIB_NAME})
target_link_libraries(${TEST_TARGET_NAME} PRIVATE Catch2::Catch2WithMain)
target_link_libraries(${TEST_TARGET_NAME} PRIVATE HDF5::HDF5)
target_link_libraries(${TEST_TARGET_NAME} PRIVATE symengine)
target_link_libraries(${TEST_TARGET_NAME} PRIVATE manifold::manifold)
if(UNIX AND NOT APPLE)
  target_link_libraries(${TEST_TARGET_NAME} PRIVATE m)
endif()

# The view-factor assembly and the Metal transform are both tested white-box
# through their src-private headers: the assembly's cache-blocked walk, thread
# split and weight table each have an obvious counterpart they must reproduce
# exactly, and only a direct call can pin them against it.
target_include_directories(${TEST_TARGET_NAME} PRIVATE
  ${PROJECT_SOURCE_DIR}/pycanha-core/src)

if(APPLE AND PYCANHA_OPTION_RAYTRACING)
  # The Metal test needs the frameworks and reference counting the backend
  # itself is built with.
  target_link_libraries(${TEST_TARGET_NAME} PRIVATE
    "-framework Metal" "-framework Foundation")
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/unit/radiative/test_mtl_transform.mm
    PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
endif()

target_set_warnings(
  TARGET
  ${TEST_TARGET_NAME}
  ENABLE
  ${PYCANHA_OPTION_WARNINGS}
  AS_ERRORS
  ${PYCANHA_OPTION_WARNINGS_AS_ERRORS})

# clang 22's -Wc2y-extensions (enabled by -Wpedantic) flags __COUNTER__, which
# Catch2's TEST_CASE macro expands to. This is a test-only issue (the library
# does not use __COUNTER__), so suppress it just for the test target rather than
# in the library's warning set. See https://github.com/catchorg/Catch2/issues/3076
# Carried on an INTERFACE target linked last so the -Wno- lands after the
# library's propagated -Wpedantic on the compile line. Scoped to upstream
# Clang >= 22: Apple Clang and older Clang reject the unknown flag under -Werror.
add_library(catch2_warning_overrides INTERFACE)
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
   AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 22)
  target_compile_options(catch2_warning_overrides INTERFACE -Wno-c2y-extensions)
endif()
target_link_libraries(${TEST_TARGET_NAME} PRIVATE catch2_warning_overrides)

target_set_warnings(
    TARGET
    ${TEST_TARGET_NAME}_lib
    ENABLE
    ${PYCANHA_OPTION_WARNINGS}
    AS_ERRORS
    ${PYCANHA_OPTION_WARNINGS_AS_ERRORS})
  



# Add Clang-Tidy to the tests (only)
if (PYCANHA_OPTION_CLANG_TIDY)
  add_clang_tidy_to_target(${TEST_TARGET_NAME})
endif()

if (PYCANHA_OPTION_INCLUDE_WHAT_YOU_USE)
  add_iwyu_to_target(${TEST_TARGET_NAME})
endif()

# CTEST add a lot of targets for CI/CD?, by setting this property we avoid that.
set_property(GLOBAL PROPERTY CTEST_TARGETS_ADDED 1)

include(CTest) #TODO: Is it really necessary to use CTest?
include(Catch)

# On Windows, copy MKL DLLs next to test exe so they can be found at runtime.
# This must happen BEFORE catch_discover_tests, which runs the exe at build time.
if(WIN32 AND PYCANHA_OPTION_USE_MKL AND MKL_ROOT)
    # Copy all DLLs from MKL bin dir (MKL loads additional DLLs at runtime
    # beyond what's in the link chain, e.g. mkl_vml_def.2.dll)
    file(GLOB _PYCANHA_MKL_DLLS "${MKL_ROOT}/bin/*.dll")
    list(LENGTH _PYCANHA_MKL_DLLS _num_mkl_dlls)
    message(STATUS "MKL DLL copy: MKL_ROOT=${MKL_ROOT}, found ${_num_mkl_dlls} DLLs")
    foreach(_dll IN LISTS _PYCANHA_MKL_DLLS)
        add_custom_command(TARGET ${TEST_TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_dll}" $<TARGET_FILE_DIR:${TEST_TARGET_NAME}>
        )
    endforeach()
    # Also copy DLLs from TARGET_RUNTIME_DLLS (e.g. libiomp5md.dll from OpenMP)
    add_custom_command(TARGET ${TEST_TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:${TEST_TARGET_NAME}>
            $<TARGET_FILE_DIR:${TEST_TARGET_NAME}>
        COMMAND_EXPAND_LISTS
    )
endif()

catch_discover_tests(${TEST_TARGET_NAME} ADD_TAGS_AS_LABELS)

# Code coverage: with PYCANHA_OPTION_COVERAGE the tests are instrumented in the
# top-level CMakeLists (Clang -fprofile-instr-generate -fcoverage-mapping). The
# report is generated in .github/workflows/coverage.yml by running the tests and
# processing the .profraw files with llvm-profdata/llvm-cov; no CMake target is
# needed here.

#CPPCHECK
if (PYCANHA_OPTION_CPPCHECK)
    if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        message(
            "==> Cppcheck, IWYU and Clang-Tidy Targets do not work with MSVC")
        return()
    endif()
    find_program(CPPCHECK cppcheck)

    # Exit with error if warnings found and if PYCANHA_OPTION_WARNINGS_AS_ERRORS is set
    if(PYCANHA_OPTION_WARNINGS_AS_ERRORS)
      set(CPP_CHECK_WARNING_AS_ERRORS --error-exitcode=1)
    else()
      set(CPP_CHECK_WARNING_AS_ERRORS )
    endif()
    
    if(CPPCHECK)
        message("==> Added Cppcheck for Target:${TEST_TARGET_NAME}_lib")
        add_custom_target(
          ${TEST_TARGET_NAME}_lib_cppcheck
            COMMAND
                ${CPPCHECK} 
                --project=${CMAKE_BINARY_DIR}/compile_commands.json
                --enable=all
                # Analyze all branches. Without this, cppcheck 2.15+ emits the
                # 'normalCheckLevelMaxBranches' information message, which trips
                # --error-exitcode and fails the target. Exhaustive does the
                # deeper analysis the message recommends (no suppression needed).
                --check-level=exhaustive
                --suppress=unusedFunction --suppress=unmatchedSuppression
                --suppress=missingIncludeSystem --suppress=toomanyconfigs
                --suppress=constStatement          
                --inline-suppr
                ${CPP_CHECK_WARNING_AS_ERRORS}
            USES_TERMINAL)
    else()
        message("==> CPPCHECK NOT FOUND")
    endif()  
endif()