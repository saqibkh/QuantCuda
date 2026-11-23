# QuantCUDA Risk Engine - build configuration
#
# The CUDA backend is OPTIONAL: when a CUDA toolkit is found, the full GPU
# engine (all kernel versions V0-V7) is built; otherwise the project builds a
# CPU-only binary where every command works on the scalar/OpenMP backends.
cmake_minimum_required(VERSION 3.24)
project(quantcuda-risk-engine VERSION 1.0.0 LANGUAGES CXX)

option(QUANTCUDA_ENABLE_CUDA "Build the CUDA GPU engine when a toolkit is available" ON)
option(QUANTCUDA_BUILD_TESTS "Build unit and statistical tests" ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W4>
                        $<$<COMPILE_LANGUAGE:CXX>:/permissive->)
else()
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>
                        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
                        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)
endif()

# --------------------------------------------------------------------------
# OpenMP (strong multicore CPU baseline)
# --------------------------------------------------------------------------
find_package(OpenMP COMPONENTS CXX)

# --------------------------------------------------------------------------
# CUDA detection
# --------------------------------------------------------------------------
set(QUANTCUDA_HAS_CUDA FALSE)
if(QUANTCUDA_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        # Set the project default before enable_language(); otherwise CMake
        # installs nvcc's single implicit architecture in the cache first and
        # the documented multi-architecture default never takes effect.
        # A caller-provided -DCMAKE_CUDA_ARCHITECTURES value remains authoritative.
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "75;86;89;90" CACHE STRING
                "CUDA architectures")
        endif()
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        set(QUANTCUDA_HAS_CUDA TRUE)
        # RTX 3050 = sm_86. Default covers Turing/Ampere/Ada/Hopper.
        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        message(STATUS "QuantCUDA: CUDA toolkit found (${CMAKE_CUDA_COMPILER_VERSION}), "
                       "architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    else()
        message(STATUS "QuantCUDA: no CUDA toolkit found - building CPU-only binary "
                       "(scalar + OpenMP backends remain fully functional)")
    endif()
endif()

# --------------------------------------------------------------------------
# Core CPU library
# --------------------------------------------------------------------------
add_library(quantcuda_core STATIC
    src/core/types.cpp
    src/core/json_config.cpp
)
target_include_directories(quantcuda_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/third_party
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(quantcuda_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(quantcuda_core PUBLIC QUANTCUDA_HAS_OPENMP=1)
endif()

# --------------------------------------------------------------------------
# GPU engine (or CPU stub when CUDA is unavailable)
# --------------------------------------------------------------------------
if(QUANTCUDA_HAS_CUDA)
    add_library(quantcuda_gpu STATIC
        src/cuda/rng.cu
        src/cuda/european.cu
        src/cuda/asian.cu
        src/cuda/barrier.cu
        src/cuda/greeks.cu
        src/cuda/correlation.cu
        src/cuda/portfolio_risk.cu
        src/cuda/reduction.cu
        src/cuda/cuda_pricer.cu
    )
    target_include_directories(quantcuda_gpu PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/third_party
    )
    target_link_libraries(quantcuda_gpu PUBLIC quantcuda_core CUDA::cudart)
    target_compile_definitions(quantcuda_gpu PUBLIC QUANTCUDA_HAS_CUDA=1)
    if(MSVC)
        # CUDA 13's CCCL/Thrust headers require MSVC's standards-conforming
        # preprocessor. Pass the option through nvcc to its host compiler.
        target_compile_options(quantcuda_gpu PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
    endif()
    set_target_properties(quantcuda_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
else()
    add_library(quantcuda_gpu STATIC src/cuda/cuda_stub.cpp)
    target_include_directories(quantcuda_gpu PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/third_party
    )
    target_link_libraries(quantcuda_gpu PUBLIC quantcuda_core)
endif()

# --------------------------------------------------------------------------
# Main executable
# --------------------------------------------------------------------------
add_executable(quantcuda
    src/main.cpp
    src/cli/cli.cpp
    src/benchmark/benchmark_runner.cpp
)
target_link_libraries(quantcuda PRIVATE quantcuda_core quantcuda_gpu)
if(OpenMP_CXX_FOUND)
    target_link_libraries(quantcuda PRIVATE OpenMP::OpenMP_CXX)
endif()

# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
if(QUANTCUDA_BUILD_TESTS)
    enable_testing()
    add_executable(quantcuda_tests
        tests/test_main.cpp
        tests/test_black_scholes.cpp
        tests/test_payoffs.cpp
        tests/test_monte_carlo.cpp
        tests/test_greeks.cpp
        tests/test_correlation.cpp
        tests/test_risk.cpp
        tests/test_json_config.cpp
        tests/test_edge_cases.cpp
        tests/test_cpu_gpu_agreement.cpp
    )
    target_link_libraries(quantcuda_tests PRIVATE quantcuda_core quantcuda_gpu)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(quantcuda_tests PRIVATE OpenMP::OpenMP_CXX)
    endif()
    target_include_directories(quantcuda_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    add_test(NAME quantcuda_unit_and_statistical COMMAND quantcuda_tests)
endif()

add_executable(drv driver.cpp)
target_link_libraries(drv PRIVATE quantcuda_core quantcuda_gpu)
if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(drv PRIVATE OpenMP::OpenMP_CXX)
endif()
