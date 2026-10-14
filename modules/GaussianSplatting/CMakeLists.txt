cmake_minimum_required(VERSION 3.18)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(GaussianSplatting VERSION 0.1 LANGUAGES CXX CUDA)
    # Default to the GPU(s) present on the build host (CMake 3.24+ "native"); override with
    # -DCMAKE_CUDA_ARCHITECTURES=120 for an RTX 5090 / Blackwell (sm_120) box, or =89 etc.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

add_library(GaussianSplattingModule STATIC
    GsCommon.h
    Camera.h
    GaussianModel.h
    GaussianModel.cu
    Rasterizer.h
    Rasterizer.cu
)

# Header-only infra (DeviceBuffer.h, CUDA_CHECK) is shared with the NeRF module.
# No library dependency on NeRFModule or TinyMLP.
target_include_directories(GaussianSplattingModule PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../NeRF
)

# CUDA 13.x CCCL headers require the conforming MSVC preprocessor.
if(MSVC)
    target_compile_options(GaussianSplattingModule PUBLIC
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /Zc:preprocessor>
    )
    target_compile_definitions(GaussianSplattingModule PUBLIC
        CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
    )
endif()

target_compile_options(GaussianSplattingModule PRIVATE
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

# ── Optional test / benchmark executables ────────────────────────────────
option(GS_BUILD_TESTS "Build Gaussian Splatting kernel tests and benchmarks" OFF)
if(GS_BUILD_TESTS)
    add_executable(test_gs_skeleton test_kernels/test_skeleton.cu)
    target_link_libraries(test_gs_skeleton PRIVATE GaussianSplattingModule)
endif()
