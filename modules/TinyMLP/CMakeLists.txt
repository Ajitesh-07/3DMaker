cmake_minimum_required(VERSION 3.18)   # 3.18 = minimum for CUDA_ARCHITECTURES

# When included via add_subdirectory() the parent project() already sets the
# languages; the guard prevents a redundant (and sometimes conflicting) call.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(TinyMLP VERSION 0.1 LANGUAGES CXX CUDA)
endif()

# Default to the GPU(s) present on the build host (CMake 3.24+ "native"); override with
# -DCMAKE_CUDA_ARCHITECTURES=120 for an RTX 5090 / Blackwell (sm_120) box.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_library(TinyMLP STATIC
    TinyMLP.cu
    TinyMLPHashGrid.cu
    mlpKernel.cu
    networkFusion.cu
    networkFusionBackward.cu
    optimizerKernel.cu
    networkFusionHashtable.cu
    networkFusionBackwardHashtable.cu
)

# set_source_files_properties(networkFusionHashtable.cu 
#     PROPERTIES COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:-maxrregcount=80>"
# )

# SM 8.0 = Ampere, 8.9 = Ada Lovelace (RTX 4060).
# Minimum for WMMA half-precision is SM 7.0 (Volta).
set_target_properties(TinyMLP PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION OFF
)

# ── Compile options ───────────────────────────────────────────────────────
target_compile_options(TinyMLP PRIVATE
    # CUDA flags applied in all configurations
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math          # fused mul-add, approx intrinsics, FTZ
        --extra-device-vectorization
        --restrict               # assume __restrict__ on all pointers
    >
    # Debug: keep line info for Nsight profiling
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:
        -lineinfo
    >
    # Release: full PTX/SASS optimisation + strip debug
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:
        -O3
        --generate-line-info     # lightweight; keep for Nsight in Release
        --ptxas-options=-v,-O3
    >
    # Host (C++) release optimisation
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:
        $<IF:$<CXX_COMPILER_ID:MSVC>,/O2,-O3>
    >
)

target_link_libraries(TinyMLP PUBLIC cublas)

# ── Optional test / benchmark executable ─────────────────────────────────
option(TINYMLP_BUILD_TESTS "Build TinyMLP self-test and benchmark" OFF)
if(TINYMLP_BUILD_TESTS)
    add_executable(test_networkfusion tests_kernels/testNetworkFusion.cu)
        set_target_properties(test_networkfusion PROPERTIES
        CUDA_ARCHITECTURES "86;89"
    )
    target_compile_options(test_networkfusion PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
    )
    target_link_libraries(test_networkfusion PRIVATE TinyMLP)

    add_executable(test_fusionbackward tests_kernels/testFusionBackward.cu)
        set_target_properties(test_networkfusion PROPERTIES
        CUDA_ARCHITECTURES "86;89"
    )
    target_compile_options(test_fusionbackward PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
    )
    target_link_libraries(test_fusionbackward PRIVATE TinyMLP)

    add_executable(test_optimizer tests_kernels/testOptimizer.cu)
        set_target_properties(test_optimizer PROPERTIES
        CUDA_ARCHITECTURES "86;89"
    )
    target_compile_options(test_optimizer PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
    )
    target_link_libraries(test_optimizer PRIVATE TinyMLP)

    add_executable(test_mlp tests_kernels/testMlp.cu)
        set_target_properties(test_mlp PROPERTIES
        CUDA_ARCHITECTURES "86;89"
    )
    target_compile_options(test_mlp PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
    )
    target_link_libraries(test_mlp PRIVATE TinyMLP)

    add_executable(demoMLP demo.cpp)
    set_target_properties(demoMLP PROPERTIES CUDA_ARCHITECTURES "80;89")
    target_compile_options(demoMLP PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3>
    )
    target_include_directories(demoMLP PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(demoMLP PRIVATE TinyMLP cublas)

    add_executable(test_forward tests_kernels/testNetworkFusionHashtable.cu)
        set_target_properties(test_forward PROPERTIES
        CUDA_ARCHITECTURES "80;86;89"
    )
    target_compile_options(test_forward PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
    )
    target_link_libraries(test_forward PRIVATE TinyMLP)

    add_executable(test_backward tests_kernels/testFusionBackwardHashtable.cu)
        set_target_properties(test_forward PROPERTIES
        CUDA_ARCHITECTURES "80;86;89"
    )
    target_compile_options(test_backward PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
    )
    target_link_libraries(test_backward PRIVATE TinyMLP)

    add_executable(test_modes tests_kernels/testModes.cu)
    set_target_properties(test_modes PROPERTIES CUDA_ARCHITECTURES "80;89")
    target_compile_options(test_modes PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
    )
    target_link_libraries(test_modes PRIVATE TinyMLP)

    add_executable(demo demoHashGrid.cpp)
    set_target_properties(demo PROPERTIES CUDA_ARCHITECTURES "80;89")
    target_compile_options(demo PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3>
    )
    target_include_directories(demo PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(demo PRIVATE TinyMLP cublas)

    # ── JIT (NVRTC) inference experiment ─────────────────────────────────
    find_package(CUDAToolkit REQUIRED)
    add_executable(test_jit_inference tests_kernels/testJitInference.cu)
    set_target_properties(test_jit_inference PROPERTIES CUDA_ARCHITECTURES "89")
    target_compile_options(test_jit_inference PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
    )
    # NVRTC needs the toolkit include dir passed as -I at runtime (it doesn't bundle
    # cuda_fp16.h / mma.h). Hand the main include dir to the test as a macro.
    list(GET CUDAToolkit_INCLUDE_DIRS 0 CUDA_MAIN_INCLUDE)
    target_compile_definitions(test_jit_inference PRIVATE CUDA_MAIN_INCLUDE="${CUDA_MAIN_INCLUDE}")
    # NVRTC for runtime compilation + CUDA driver API for cuModule*/cuLaunchKernel
    target_link_libraries(test_jit_inference PRIVATE TinyMLP CUDA::nvrtc CUDA::cuda_driver)

    # ── Hash-grid scatter A/B prototype (per-level vs all-levels) ─────────
    add_executable(test_scatter_ab tests_kernels/testScatterAB.cu)
    set_target_properties(test_scatter_ab PROPERTIES CUDA_ARCHITECTURES "89")
    target_compile_options(test_scatter_ab PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3 --use_fast_math>
    )

    # ── Dim-32 forward specialization benchmark (WMMA baseline + roofline) ─
    add_executable(test_forward_dim32 tests_kernels/testForwardDim32.cu)
    set_target_properties(test_forward_dim32 PROPERTIES CUDA_ARCHITECTURES "89")
    target_compile_options(test_forward_dim32 PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3 --use_fast_math>
    )
    target_link_libraries(test_forward_dim32 PRIVATE TinyMLP)

    add_executable(test_backward_dim32
        tests_kernels/testBackwardDim32.cu
        tests_kernels/networkFusionBackward_lb4.cu)   # launch_bounds(256,4) copy for A/B
    set_target_properties(test_backward_dim32 PROPERTIES CUDA_ARCHITECTURES "89")
    target_compile_options(test_backward_dim32 PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3 --use_fast_math --restrict>
    )
    target_link_libraries(test_backward_dim32 PRIVATE TinyMLP)

    # ── TinyMLP vs PyTorch comparison (emits CSV) ────────────────────────
    add_executable(bench_compare tests_kernels/benchCompare.cu)
    set_target_properties(bench_compare PROPERTIES CUDA_ARCHITECTURES "89")
    target_compile_options(bench_compare PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3 --use_fast_math>
    )
    target_link_libraries(bench_compare PRIVATE TinyMLP)
endif()