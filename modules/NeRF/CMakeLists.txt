cmake_minimum_required(VERSION 3.18)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(NeRF VERSION 0.1 LANGUAGES CXX CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "80;89")
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # When building standalone, pull in TinyMLP as a subdirectory
    if(NOT TARGET TinyMLP)
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../TinyMLP
                         ${CMAKE_CURRENT_BINARY_DIR}/TinyMLP)
    endif()
endif()

add_library(NeRFModule STATIC
    processRays.cu
    otherKernels.cu
    compositing.cu
    otherKernels.cu
    InstantNerf.cu
    DeviceBuffer.h
    Timers.h
    DataLoader.cu
)

# Headers from TinyMLP module
target_include_directories(NeRFModule PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../TinyMLP
)

# Link TinyMLP (provides TinyMLPHashGrid + TinyMLP classes) and curand
target_link_libraries(NeRFModule PUBLIC TinyMLP curand)

# CUDA 13.x CCCL headers require the conforming MSVC preprocessor.
if(MSVC)
    target_compile_options(NeRFModule PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /Zc:preprocessor>
    )
    target_compile_definitions(NeRFModule PRIVATE
        CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
    )
endif()

target_compile_options(NeRFModule PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        --extra-device-vectorization
        --restrict
    >
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:
        -lineinfo
    >
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:
        -O3
        --generate-line-info
        --ptxas-options=-v,-O3
    >
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:
        $<IF:$<CXX_COMPILER_ID:MSVC>,/O2,-O3>
    >
)

# ── Optional test / benchmark executable ─────────────────────────────────
option(NERF_BUILD_TESTS "Build NeRF kernel tests and benchmarks" OFF)
if(NERF_BUILD_TESTS)
    add_executable(test_processrays test_kernels/testProcessRays.cu)
    set_target_properties(test_processrays PROPERTIES
        CUDA_ARCHITECTURES "80;89"
    )
    if(MSVC)
        target_compile_options(test_processrays PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /Zc:preprocessor>
        )
        target_compile_definitions(test_processrays PRIVATE
            CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
        )
    endif()
    target_compile_options(test_processrays PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
    )
    target_link_libraries(test_processrays PRIVATE NeRFModule)

    add_executable(run_benchmark test_kernels/benchmark.cu)
    set_target_properties(test_processrays PROPERTIES
        CUDA_ARCHITECTURES "80;89"
    )
    if(MSVC)
        target_compile_options(run_benchmark PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /Zc:preprocessor>
        )
        target_compile_definitions(run_benchmark PRIVATE
            CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
        )
    endif()
    target_compile_options(run_benchmark PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
    )
    target_link_libraries(run_benchmark PRIVATE NeRFModule)

    add_executable(test_training test_kernels/test_training.cu)
    set_target_properties(test_processrays PROPERTIES
        CUDA_ARCHITECTURES "80;89"
    )
    if(MSVC)
        target_compile_options(test_training PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /Zc:preprocessor>
        )
        target_compile_definitions(test_training PRIVATE
            CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
        )
    endif()
    target_compile_options(test_training PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
    )
    target_link_libraries(test_training PRIVATE NeRFModule)

    add_executable(train test_kernels/train.cu)
    set_target_properties(test_processrays PROPERTIES
        CUDA_ARCHITECTURES "80;89"
    )
    if(MSVC)
        target_compile_options(train PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /Zc:preprocessor>
        )
        target_compile_definitions(train PRIVATE
            CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
        )
    endif()
    target_compile_options(train PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
    )
    target_link_libraries(train PRIVATE NeRFModule)

    add_executable(test_loading test_kernels/test_loading.cu)
    if(MSVC)
        target_compile_options(test_loading PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /Zc:preprocessor>
        )
        target_compile_definitions(test_loading PRIVATE
            CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
        )
    endif()
    target_compile_options(test_loading PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
    )
    target_link_libraries(test_loading PRIVATE NeRFModule)

    add_executable(render test_kernels/render.cu)
    if(MSVC)
        target_compile_options(render PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /Zc:preprocessor>
        )
        target_compile_definitions(render PRIVATE
            CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
        )
    endif()
    target_compile_options(render PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
    )
    target_link_libraries(render PRIVATE NeRFModule)
    add_executable(compare_hit test_kernels/compare_hit.cu)
    if(MSVC)
        target_compile_options(compare_hit PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /Zc:preprocessor>
        )
        target_compile_definitions(compare_hit PRIVATE
            CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
        )
    endif()
    target_link_libraries(compare_hit PRIVATE NeRFModule)

    add_executable(train_hit test_kernels/train_hit.cu)
    if(MSVC)
        target_compile_options(train_hit PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /Zc:preprocessor>
        )
        target_compile_definitions(train_hit PRIVATE
            CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
        )
    endif()
    target_link_libraries(train_hit PRIVATE NeRFModule)
endif()