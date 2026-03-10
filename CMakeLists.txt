cmake_minimum_required(VERSION 3.10)
project(NML VERSION 0.5 LANGUAGES C)

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
target_link_libraries(nml m)
target_compile_options(nml PRIVATE -O2 -Wall)

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
target_link_libraries(nml-core m)
target_compile_options(nml-core PRIVATE -O2 -Wall)
target_compile_definitions(nml-core PRIVATE NML_NO_DEFAULT_EXTENSIONS)

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
