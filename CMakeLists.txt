cmake_minimum_required(VERSION 3.16)
project(matgen VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(MATGEN_NATIVE "Optimize for the build machine's CPU (-march=native); output bytes are unaffected" ON)

# Required for bit-exact output on every compiler and SIMD width:
# no FMA contraction. -fno-math-errno and -fno-trapping-math only drop errno
# and FP exception-flag semantics (values are unchanged) so that sqrt and the
# branch-free selects can vectorize. Never add -ffast-math.
set(MG_DET_FLAGS -ffp-contract=off -fno-math-errno -fno-trapping-math)
set(MG_FAST_FLAGS ${MG_DET_FLAGS})
if(MATGEN_NATIVE)
  list(APPEND MG_FAST_FLAGS -march=native)
endif()

find_package(Threads REQUIRED)

set(MG_CORE_SOURCES src/json.cpp src/spec.cpp src/hw.cpp src/dataset.cpp)

add_library(matgen_core STATIC ${MG_CORE_SOURCES})
target_include_directories(matgen_core PUBLIC include src)
target_compile_options(matgen_core PRIVATE ${MG_FAST_FLAGS} -Wall -Wextra)
target_link_libraries(matgen_core PUBLIC Threads::Threads)
set_target_properties(matgen_core PROPERTIES POSITION_INDEPENDENT_CODE ON OUTPUT_NAME matgen)

add_executable(matgen tools/matgen.cpp)
target_compile_options(matgen PRIVATE ${MG_FAST_FLAGS} -Wall -Wextra)
target_link_libraries(matgen PRIVATE matgen_core)

add_executable(read_example examples/read_example.c)
target_link_libraries(read_example PRIVATE matgen_core)
set_target_properties(read_example PROPERTIES LINKER_LANGUAGE CXX)

# Tests
enable_testing()
add_executable(matgen_tests tests/tests.cpp)
target_compile_options(matgen_tests PRIVATE ${MG_FAST_FLAGS} -Wall -Wextra)
target_link_libraries(matgen_tests PRIVATE matgen_core)
add_test(NAME unit COMMAND matgen_tests --quick)

# Golden hashes: vectorized native build vs. non-vectorized baseline build.
add_executable(golden_native tests/golden.cpp src/json.cpp src/spec.cpp)
target_include_directories(golden_native PRIVATE include src)
target_compile_options(golden_native PRIVATE ${MG_FAST_FLAGS} -O3)

add_executable(golden_baseline tests/golden.cpp src/json.cpp src/spec.cpp)
target_include_directories(golden_baseline PRIVATE include src)
target_compile_options(golden_baseline PRIVATE ${MG_DET_FLAGS} -O1 -fno-tree-vectorize -fno-tree-slp-vectorize)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  target_compile_options(golden_baseline PRIVATE -march=x86-64)
endif()

add_test(NAME golden_vectorized_vs_baseline
         COMMAND ${CMAKE_COMMAND} -DA=$<TARGET_FILE:golden_native> -DB=$<TARGET_FILE:golden_baseline>
                 -P ${CMAKE_SOURCE_DIR}/tests/compare.cmake)
