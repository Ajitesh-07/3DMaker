#pragma once

#include <cuda_runtime.h>
#include <cmath>

// Camera as the rasterizer sees it: OpenCV convention (x right, y down, z forward).
// The conversion from the dataset's NeRF/OpenGL c2w happens once, here, so kernels never
// deal with axis flips.
struct GsCamera {
    float  w2c[12];    // row-major 3x4 world -> camera
    float3 position;   // camera centre in world space (for SH view directions)
    float  fx, fy, cx, cy;
    int    width, height;
};

// c2w: row-major 4x4 from transforms_*.json (OpenGL: x right, y up, camera looks down -z).
// Single shared pinhole intrinsics with a centred principal point, matching DataLoader.
inline GsCamera cameraFromNerf(const float c2w[16], float cameraAngleX, int width, int height) {
    GsCamera cam;

    // OpenGL -> OpenCV camera axes: negate the y and z columns of the rotation.
    float R[3][3];
    for (int r = 0; r < 3; ++r) {
        R[r][0] =  c2w[r * 4 + 0];
        R[r][1] = -c2w[r * 4 + 1];
        R[r][2] = -c2w[r * 4 + 2];
    }
    float t[3] = { c2w[3], c2w[7], c2w[11] };

    // w2c = [R^T | -R^T t]
    for (int r = 0; r < 3; ++r) {
        float tr = 0.0f;
        for (int c = 0; c < 3; ++c) {
            cam.w2c[r * 4 + c] = R[c][r];
            tr -= R[c][r] * t[c];
        }
        cam.w2c[r * 4 + 3] = tr;
    }

    cam.position = make_float3(t[0], t[1], t[2]);
    cam.fx = 0.5f * (float)width / tanf(0.5f * cameraAngleX);
    cam.fy = cam.fx;
    cam.cx = 0.5f * (float)width;
    cam.cy = 0.5f * (float)height;
    cam.width  = width;
    cam.height = height;
    return cam;
}
