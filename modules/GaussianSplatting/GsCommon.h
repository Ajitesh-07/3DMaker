#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "DeviceBuffer.h"   // modules/NeRF, header-only; also provides CUDA_CHECK

constexpr int GS_TILE_SIZE  = 16;                                   // rasterizer tile edge (px)
constexpr int GS_SH_DEGREE  = 3;                                    // max SH degree while training
constexpr int GS_SH_COEFFS  = (GS_SH_DEGREE + 1) * (GS_SH_DEGREE + 1);  // 16, incl. DC
constexpr int GS_SH_REST    = GS_SH_COEFFS - 1;                     // 15 view-dependent coeffs

// Floats per Gaussian: mean 3 + log-scale 3 + quat 4 + opacity 1 + SH 16*3 = 59 (236 B).
constexpr int GS_FLOATS_PER_GAUSSIAN = 3 + 3 + 4 + 1 + GS_SH_COEFFS * 3;
