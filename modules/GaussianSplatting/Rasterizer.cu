#include "Rasterizer.h"

// TODO(M0): forward stages — preprocess, duplicate, sort, tile ranges, blend.
void rasterizeForward(GaussianModel& model, const GsCamera& cam, const RasterSettings& settings,
                      float* d_outImage, cudaStream_t stream) {
    (void)model; (void)cam; (void)settings; (void)d_outImage; (void)stream;
}

// TODO(M0): backward stages — blend backward, preprocess backward.
void rasterizeBackward(GaussianModel& model, const GsCamera& cam, const RasterSettings& settings,
                       const float* d_dLossdImage, cudaStream_t stream) {
    (void)model; (void)cam; (void)settings; (void)d_dLossdImage; (void)stream;
}
