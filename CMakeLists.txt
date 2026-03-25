cmake_minimum_required(VERSION 3.10)
project(NML VERSION 0.5 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ═══════════════════════════════════════════
# Build Options
# ═══════════════════════════════════════════
option(NML_VISION      "Enable NML-V (Vision) extension"      ON)
option(NML_TRANSFORMER "Enable NML-T (Transformer) extension" ON)
option(NML_REDUCTION   "Enable NML-R (Reduction) extension"   ON)
option(NML_SIGNAL      "Enable NML-S (Signal) extension"      ON)
option(NML_ALL_EXT     "Enable all extensions (default)"      ON)
option(NML_CORE_ONLY   "Build core only, no extensions"       OFF)

# ═══════════════════════════════════════════
# Full runtime (v0.6.4 with extensions)
# ═══════════════════════════════════════════
add_executable(nml runtime/nml.c)
if(NOT MSVC)
    target_link_libraries(nml m)
    target_compile_options(nml PRIVATE -O2 -Wall)
else()
    target_compile_options(nml PRIVATE /O2 /W3 /wd4996 /wd4244 /wd4267 /wd4305)
    target_link_options(nml PRIVATE /STACK:8388608)
endif()

if(NML_CORE_ONLY)
    target_compile_definitions(nml PRIVATE NML_NO_DEFAULT_EXTENSIONS)
    message(STATUS "NML: Building CORE ONLY (35 instructions)")
else()
    if(NML_VISION OR NML_ALL_EXT)
        target_compile_definitions(nml PRIVATE NML_EXT_VISION)
        message(STATUS "NML: NML-V (Vision) ENABLED")
    endif()
    if(NML_TRANSFORMER OR NML_ALL_EXT)
        target_compile_definitions(nml PRIVATE NML_EXT_TRANSFORMER)
        message(STATUS "NML: NML-T (Transformer) ENABLED")
    endif()
    if(NML_REDUCTION OR NML_ALL_EXT)
        target_compile_definitions(nml PRIVATE NML_EXT_REDUCTION)
        message(STATUS "NML: NML-R (Reduction) ENABLED")
    endif()
    if(NML_SIGNAL OR NML_ALL_EXT)
        target_compile_definitions(nml PRIVATE NML_EXT_SIGNAL)
        message(STATUS "NML: NML-S (Signal) ENABLED")
    endif()
endif()

# ═══════════════════════════════════════════
# Core-only build (separate target)
# ═══════════════════════════════════════════
add_executable(nml-core runtime/nml.c)
if(NOT MSVC)
    target_link_libraries(nml-core m)
    target_compile_options(nml-core PRIVATE -O2 -Wall)
else()
    target_compile_options(nml-core PRIVATE /O2 /W3 /wd4996 /wd4244 /wd4267 /wd4305)
    target_link_options(nml-core PRIVATE /STACK:8388608)
endif()
target_compile_definitions(nml-core PRIVATE NML_NO_DEFAULT_EXTENSIONS)

# ═══════════════════════════════════════════
# SYCL backend (Intel oneAPI / AdaptiveCpp)
# ═══════════════════════════════════════════
option(NML_SYCL      "Enable Intel oneAPI SYCL backend (requires icpx)" OFF)
option(NML_SYCL_MKL  "Use oneMKL for GEMM/GEMV in SYCL backend"        OFF)

if(NML_SYCL)
    find_program(ICPX_EXECUTABLE NAMES icpx)
    if(NOT ICPX_EXECUTABLE)
        message(WARNING
            "NML_SYCL=ON but icpx not found — nml-sycl target skipped.\n"
            "Install Intel oneAPI Base Toolkit and source setvars.sh, or set "
            "ICPX_EXECUTABLE to your SYCL compiler path.")
    else()
        set(_sycl_defs NML_USE_SYCL)
        set(_sycl_mkl_libs "")
        if(NML_SYCL_MKL)
            list(APPEND _sycl_defs NML_USE_ONEMKL)
            set(_sycl_mkl_libs -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core)
        endif()

        # nml-sycl: nml.c compiled as C99 under icpx (-x c), linked with
        # nml_backend_sycl.cpp compiled as SYCL C++.
        # CMake doesn't natively support mixed C/SYCL in one target at 3.10,
        # so we drive icpx directly via add_custom_target.
        string(JOIN " " _sycl_def_flags ${_sycl_defs})
        string(REPLACE ";" " -D" _sycl_def_flags "-D${_sycl_def_flags}")

        add_custom_target(nml-sycl
            COMMAND ${ICPX_EXECUTABLE} -fsycl -O3 ${_sycl_def_flags}
                    -x c -std=c99
                    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/nml.c
                    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/nml_backend_sycl.cpp
                    -o ${CMAKE_CURRENT_BINARY_DIR}/nml-sycl
                    -lm ${_sycl_mkl_libs}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Building nml-sycl (SYCL GPU backend via ${ICPX_EXECUTABLE})"
        )

        add_custom_target(nmld-sycl
            COMMAND ${ICPX_EXECUTABLE} -fsycl -O3 ${_sycl_def_flags}
                    -x c -std=c99
                    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/nmld.c
                    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/nml_backend_sycl.cpp
                    -o ${CMAKE_CURRENT_BINARY_DIR}/nmld-sycl
                    -lm ${_sycl_mkl_libs}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Building nmld-sycl (NML daemon with SYCL GPU acceleration)"
        )

        message(STATUS "NML: SYCL backend ENABLED  (compiler: ${ICPX_EXECUTABLE})")
        if(NML_SYCL_MKL)
            message(STATUS "NML: oneMKL GEMM/GEMV path ENABLED")
        endif()
    endif()
endif()

# ═══════════════════════════════════════════
# Install
# ═══════════════════════════════════════════
install(TARGETS nml DESTINATION bin)
install(TARGETS nml-core DESTINATION bin)

# ═══════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════
message(STATUS "")
message(STATUS "═══════════════════════════════════════")
message(STATUS "  NML v${PROJECT_VERSION}")
message(STATUS "  Targets: nml (full), nml-core (35 ops)")
message(STATUS "═══════════════════════════════════════")
message(STATUS "")
message(STATUS "Build:")
message(STATUS "  mkdir build && cd build")
message(STATUS "  cmake ..")
message(STATUS "  make")
message(STATUS "")
message(STATUS "Run:")
message(STATUS "  ./nml <program.nml> [data.nml.data]")
message(STATUS "")
message(STATUS "Custom extension builds:")
message(STATUS "  cmake .. -DNML_CORE_ONLY=ON")
message(STATUS "  cmake .. -DNML_VISION=ON -DNML_TRANSFORMER=OFF")
message(STATUS "")
