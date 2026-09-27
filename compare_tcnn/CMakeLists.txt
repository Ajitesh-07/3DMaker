# TinyMLP vs tiny-cuda-nn head-to-head benchmark.
#
#   git clone --recursive https://github.com/NVlabs/tiny-cuda-nn compare_tcnn/tiny-cuda-nn
#   cmake -S compare_tcnn -B compare_tcnn/build -DCMAKE_BUILD_TYPE=Release
#   cmake --build compare_tcnn/build --config Release -j
#   compare_tcnn/build/Release/bench_tcnn.exe > compare_tcnn/results/raw.csv
#   python compare_tcnn/make_report.py
#
# Both libraries are built with the same nvcc, the same arch and --use_fast_math, and linked
# into ONE executable so they are timed with identical harness code on the same stream.
cmake_minimum_required(VERSION 3.24)
project(TinyMLPvsTCNN LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# ---- tiny-cuda-nn (library only) --------------------------------------------------
set(TCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(TCNN_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
set(TCNN_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
add_subdirectory(tiny-cuda-nn)

# ---- TinyMLP (the repo's own module, unchanged) -----------------------------------
add_subdirectory(../modules/TinyMLP ${CMAKE_BINARY_DIR}/TinyMLP)

# ---- benchmark --------------------------------------------------------------------
add_executable(bench_tcnn bench_tcnn.cu)
set_target_properties(bench_tcnn PROPERTIES CUDA_ARCHITECTURES "${TCNN_CUDA_ARCHITECTURES}")
# tiny-cuda-nn exports its includes/definitions/nvcc flags PUBLIC; only TinyMLP's dir is added.
target_include_directories(bench_tcnn PRIVATE ../modules/TinyMLP)
target_compile_options(bench_tcnn PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
if(MSVC)
    target_compile_options(bench_tcnn PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
endif()
target_link_libraries(bench_tcnn PRIVATE tiny-cuda-nn TinyMLP)
