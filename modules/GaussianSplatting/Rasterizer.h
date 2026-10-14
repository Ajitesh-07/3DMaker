#pragma once

#include "GsCommon.h"
#include "Camera.h"
#include "GaussianModel.h"

struct RasterSettings {
    float3 bgColor  = make_float3(0.0f, 0.0f, 0.0f);
    int    shDegree = GS_SH_DEGREE;   // active degree (grown during training)
};

// Forward pipeline (research doc §3–§6):
//   1. preprocess   — per Gaussian: frustum cull, project mean, 3D cov -> 2D cov (EWA),
//                     conic + opacity, SH -> RGB, tile bounding box, depth
//   2. duplicate    — one (tile | depth) key per Gaussian-tile overlap (prefix sum on counts)
//   3. sort         — radix sort keys (CUB)
//   4. tile ranges  — [start, end) per tile from the sorted keys
//   5. blend        — per tile, front-to-back alpha compositing; store final T and last
//                     contributor per pixel for the backward pass
// d_outImage: [height * width * 3], planar or interleaved is your call.
void rasterizeForward(GaussianModel& model, const GsCamera& cam, const RasterSettings& settings,
                      float* d_outImage, cudaStream_t stream = 0);

// Backward pipeline (research doc §7): blend backward -> preprocess backward, accumulating
// into model.grads(). Expects the forward pass for the same camera to have just run.
void rasterizeBackward(GaussianModel& model, const GsCamera& cam, const RasterSettings& settings,
                       const float* d_dLossdImage, cudaStream_t stream = 0);
