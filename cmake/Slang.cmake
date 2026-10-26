# Slang toolchain acquisition + kernel build rules. No Conan recipe exists
# for slang, so a pinned prebuilt GitHub release is fetched at configure
# time and verified by SHA256. Shaders are inputs to the BUILD; neither
# .slang sources, .spv binaries nor slangc are ever shipped.

include(FetchContent)

# Fetches the pinned slangc for the host platform and sets PYCANHA_SLANGC.
# PYCANHA_OPTION_SLANG_VERSION comes from conanfile.py (DEPENDENCY_VERSIONS);
# the per-version SHA256 pins live here — extend the table to bump versions.
function(pycanha_fetch_slang)
    if(NOT PYCANHA_OPTION_SLANG_VERSION)
        message(FATAL_ERROR "PYCANHA_OPTION_SLANG_VERSION is not set (configure through Conan)")
    endif()
    set(_version "${PYCANHA_OPTION_SLANG_VERSION}")

    if(NOT _version STREQUAL "2026.12.2")
        message(FATAL_ERROR
            "No SHA256 pins for slang ${_version} in cmake/Slang.cmake; "
            "download the release archives for each platform, compute their "
            "sha256 and extend the table below")
    endif()
    if(WIN32)
        set(_archive "slang-${_version}-windows-x86_64.zip")
        set(_sha256 "e44a29e4ba766e892db19e7f491b0c1fc21f548a0380a1b2931039569bf747e7")
        set(_slangc_name "slangc.exe")
    elseif(UNIX AND NOT APPLE)
        set(_archive "slang-${_version}-linux-x86_64.tar.gz")
        set(_sha256 "5533415953112ddeb0a935755bdd2da5de530e6528a560a32ad809c9d9faf29c")
        set(_slangc_name "slangc")
    elseif(APPLE)
        if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
            message(FATAL_ERROR
                "The Metal kernels require Apple Silicon (found "
                "${CMAKE_SYSTEM_PROCESSOR}); build with "
                "PYCANHA_OPTION_RAYTRACING=False on Intel Macs")
        endif()
        set(_archive "slang-${_version}-macos-aarch64.zip")
        set(_sha256 "d1b6aac3f10b0031ef3ba7b0432ab678180aee745ff49d6d77b74271ced7bbbc")
        set(_slangc_name "slangc")
    else()
        message(FATAL_ERROR "No pinned slang archive for this platform")
    endif()

    FetchContent_Declare(
        slang_toolchain
        URL "https://github.com/shader-slang/slang/releases/download/v${_version}/${_archive}"
        URL_HASH "SHA256=${_sha256}"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(slang_toolchain)

    set(_slangc "${slang_toolchain_SOURCE_DIR}/bin/${_slangc_name}")
    if(NOT EXISTS "${_slangc}")
        message(FATAL_ERROR "slangc not found in the fetched archive: ${_slangc}")
    endif()
    set(PYCANHA_SLANGC "${_slangc}" PARENT_SCOPE)
    message(STATUS "Slang ${_version}: ${_slangc}")

    if(APPLE)
        # On Apple the MSL produced by slangc still has to be compiled by
        # Apple's own shader compiler, which is where a broken kernel actually
        # shows up. Probe it now so a missing toolchain is a clear configure
        # error instead of a confusing failure halfway through the build.
        find_program(PYCANHA_XCRUN xcrun REQUIRED)
        execute_process(COMMAND "${PYCANHA_XCRUN}" metal --version
            RESULT_VARIABLE _metal_rc
            OUTPUT_VARIABLE _metal_out
            ERROR_VARIABLE _metal_err)
        if(NOT _metal_rc EQUAL 0)
            message(FATAL_ERROR
                "The Metal shader compiler is not available (xcrun metal "
                "--version failed).\nSince Xcode 26 it is an unbundled, "
                "per-user component; install it with:\n"
                "    xcodebuild -downloadComponent MetalToolchain\n"
                "${_metal_err}")
        endif()
        string(REGEX REPLACE "\n.*" "" _metal_version "${_metal_out}${_metal_err}")
        message(STATUS "Metal shader compiler: ${_metal_version}")
        set(PYCANHA_XCRUN "${PYCANHA_XCRUN}" PARENT_SCOPE)
    else()
        # spirv-val catches compiler/driver SPIR-V mismatches at build time
        # instead of first-run time. Optional — the Conan spirv-tools
        # tool-require puts it on the build env PATH; skip quietly when absent.
        find_program(PYCANHA_SPIRV_VAL spirv-val)
        if(PYCANHA_SPIRV_VAL)
            message(STATUS "spirv-val: ${PYCANHA_SPIRV_VAL}")
        else()
            message(STATUS "spirv-val not found — SPIR-V validation step skipped")
        endif()
    endif()
endfunction()

# Metal branch of pycanha_add_slang_kernel(); see its comment for the contract.
# Separate function only to keep either pipeline readable.
function(pycanha_add_metal_kernel _name _source _kernel_dir _embed_dir _depends)
    set(_work "${CMAKE_CURRENT_BINARY_DIR}/kernels")
    set(_msl "${_work}/${_name}.metal")
    set(_reflect "${_work}/${_name}.reflect.json")
    set(_air "${_work}/${_name}.air")
    set(_metallib "${_work}/${_name}.metallib")
    set(_lib_header "${_embed_dir}/${_name}_metallib.h")
    set(_bindings_header "${_embed_dir}/${_name}_bindings.h")

    add_custom_command(
        OUTPUT "${_metallib}" "${_lib_header}" "${_bindings_header}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${_work}" "${_embed_dir}"
        # No -profile/-capability (those select SPIR-V atoms) and no
        # -fvk-use-scalar-layout (Vulkan-only). The define switches the shared
        # module to the deposit and front-face paths Metal can express.
        COMMAND "${PYCANHA_SLANGC}" "${_source}"
                -I "${_kernel_dir}"
                -target metal -O2
                -D PYCANHA_METAL_BACKEND=1
                -entry csMain
                -reflection-json "${_reflect}"
                -o "${_msl}"
        # -Wno-unused-variable applies to GENERATED MSL: slangc assigns every
        # InterlockedAdd result and every bool-returning helper to a temporary
        # nothing reads, and that noise buries real diagnostics.
        COMMAND "${PYCANHA_XCRUN}" metal -Wno-unused-variable
                -c "${_msl}" -o "${_air}"
        COMMAND "${PYCANHA_XCRUN}" metallib "${_air}" -o "${_metallib}"
        COMMAND "${CMAKE_COMMAND}"
                -DBINARY_INPUT=${_metallib}
                -DHEADER_OUTPUT=${_lib_header}
                -DVARIABLE_NAME=${_name}_metallib
                -P "${PROJECT_SOURCE_DIR}/cmake/EmbedBinary.cmake"
        COMMAND "${CMAKE_COMMAND}"
                -DREFLECTION_INPUT=${_reflect}
                -DHEADER_OUTPUT=${_bindings_header}
                -DKERNEL_NAME=${_name}
                -P "${PROJECT_SOURCE_DIR}/cmake/MetalBindings.cmake"
        DEPENDS "${_source}" ${_depends}
                "${PROJECT_SOURCE_DIR}/cmake/EmbedBinary.cmake"
                "${PROJECT_SOURCE_DIR}/cmake/MetalBindings.cmake"
        COMMENT "Compiling Slang kernel ${_name} (Metal)"
        VERBATIM)

    add_custom_target(pycanha_kernel_${_name}
        DEPENDS "${_lib_header}" "${_bindings_header}")
endfunction()

# pycanha_add_slang_kernel(<name>
#     SOURCE <kernel.slang>
#     DEPENDS <module.slang> ...      # Slang emits no depfile: list imports
#     [DEFINES <NAME[=value]> ...]    # preprocessor defines (spirv: a variant
#                                     # of a kernel under another <name>)
#     [TARGET <spirv|metal>])         # default spirv
#
# spirv: compiles the kernel to SPIR-V (Vulkan 1.3, scalar block layout,
# explicit ray-query + 64-bit-atomics capabilities), validates the SPIR-V when
# spirv-val is available, and generates
# ${PYCANHA_KERNEL_EMBED_DIR}/pycanha-core/radiative/kernels/<name>_spv.h
# defining pycanha::radiative::kernels::<name>_spv.
#
# metal: compiles the kernel to MSL, then to a .metallib through Apple's
# shader compiler (which is where a broken kernel actually fails — it plays the
# role spirv-val plays on the Vulkan side), and generates two headers:
# <name>_metallib.h (the library bytes) and <name>_bindings.h (the buffer
# indices, read out of the compiler's own reflection dump because Metal ignores
# the [[vk::binding]] attributes and numbers buffers per kernel).
#
# Either way it creates the custom target pycanha_kernel_<name>; the caller
# wires it into the library.
function(pycanha_add_slang_kernel _name)
    cmake_parse_arguments(PARSE_ARGV 1 _kernel "" "SOURCE;TARGET" "DEPENDS;DEFINES")
    if(NOT _kernel_SOURCE)
        message(FATAL_ERROR "pycanha_add_slang_kernel: SOURCE is required")
    endif()
    if(NOT PYCANHA_SLANGC)
        message(FATAL_ERROR "pycanha_add_slang_kernel: call pycanha_fetch_slang() first")
    endif()
    if(NOT _kernel_TARGET)
        set(_kernel_TARGET "spirv")
    endif()

    set(_embed_root "${CMAKE_BINARY_DIR}/kernel_embed")
    set(_embed_dir "${_embed_root}/pycanha-core/radiative/kernels")
    set(_spv "${CMAKE_CURRENT_BINARY_DIR}/kernels/${_name}.spv")
    set(_header "${_embed_dir}/${_name}_spv.h")
    get_filename_component(_kernel_dir "${_kernel_SOURCE}" DIRECTORY)

    if(_kernel_TARGET STREQUAL "metal")
        pycanha_add_metal_kernel("${_name}" "${_kernel_SOURCE}" "${_kernel_dir}"
            "${_embed_dir}" "${_kernel_DEPENDS}")
        set(PYCANHA_KERNEL_EMBED_DIR "${_embed_root}" PARENT_SCOPE)
        return()
    elseif(NOT _kernel_TARGET STREQUAL "spirv")
        message(FATAL_ERROR
            "pycanha_add_slang_kernel: TARGET must be spirv or metal "
            "(got ${_kernel_TARGET})")
    endif()

    set(_defines "")
    foreach(_define IN LISTS _kernel_DEFINES)
        list(APPEND _defines "-D${_define}")
    endforeach()
    set(_commands
        COMMAND "${CMAKE_COMMAND}" -E make_directory
                "${CMAKE_CURRENT_BINARY_DIR}/kernels" "${_embed_dir}"
        COMMAND "${PYCANHA_SLANGC}" "${_kernel_SOURCE}"
                -I "${_kernel_dir}" ${_defines}
                -profile glsl_460
                -capability spvRayQueryKHR -capability spvInt64Atomics
                -target spirv -O2 -fvk-use-scalar-layout
                -entry csMain -o "${_spv}")
    if(PYCANHA_SPIRV_VAL)
        list(APPEND _commands
             COMMAND "${PYCANHA_SPIRV_VAL}" --target-env vulkan1.3 "${_spv}")
    endif()
    list(APPEND _commands
         COMMAND "${CMAKE_COMMAND}"
                 -DSPV_INPUT=${_spv}
                 -DHEADER_OUTPUT=${_header}
                 -DVARIABLE_NAME=${_name}_spv
                 -P "${PROJECT_SOURCE_DIR}/cmake/EmbedSpirv.cmake")

    add_custom_command(
        OUTPUT "${_spv}" "${_header}"
        ${_commands}
        DEPENDS "${_kernel_SOURCE}" ${_kernel_DEPENDS}
                "${PROJECT_SOURCE_DIR}/cmake/EmbedSpirv.cmake"
        COMMENT "Compiling Slang kernel ${_name}"
        VERBATIM)

    add_custom_target(pycanha_kernel_${_name} DEPENDS "${_header}")
    set(PYCANHA_KERNEL_EMBED_DIR "${_embed_root}" PARENT_SCOPE)
endfunction()
