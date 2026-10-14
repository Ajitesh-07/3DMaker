#include "../GaussianModel.h"
#include "../Camera.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>

static void toCamera(const GsCamera& cam, const float p[3], float out[3]) {
    for (int r = 0; r < 3; ++r)
        out[r] = cam.w2c[r * 4 + 0] * p[0] + cam.w2c[r * 4 + 1] * p[1] + cam.w2c[r * 4 + 2] * p[2] + cam.w2c[r * 4 + 3];
}

static bool near(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) < eps; }

int main() {
    bool ok = true;

    // ── Model allocation + count cap ──────────────────────────────────────
    const uint32_t capacity = 1u << 20;
    GaussianModel model(capacity);
    printf("[model] capacity=%u  %.1f MB allocated (%.0f B/Gaussian)\n", model.capacity(),
           model.bytesAllocated() / (1024.0 * 1024.0), (double)model.bytesAllocated() / capacity);

    model.setCount(1000);
    model.zeroGrads();
    CUDA_CHECK(cudaDeviceSynchronize());

    bool capThrew = false;
    try { model.setCount(capacity + 1); } catch (const std::runtime_error&) { capThrew = true; }
    printf("[model] setCount past capacity throws: %s\n", capThrew ? "yes" : "NO");
    ok &= capThrew;

    // ── Camera convention: NeRF/OpenGL c2w -> OpenCV w2c ──────────────────
    // Camera at (1,2,3), rotated 90 deg about world y: GL forward (-z cam) points along world -x.
    const float c2w[16] = {
         0, 0, 1, 1,
         0, 1, 0, 2,
        -1, 0, 0, 3,
         0, 0, 0, 1,
    };
    GsCamera cam = cameraFromNerf(c2w, 1.0f, 640, 480);

    float centre[3], inFront[3], above[3];
    const float pCentre[3]  = { 1, 2, 3 };
    const float pInFront[3] = { 1 - 2.0f, 2, 3 };   // 2 units along GL forward (world -x)
    const float pAbove[3]   = { 1 - 2.0f, 3, 3 };   // same, 1 unit up (world +y)
    toCamera(cam, pCentre, centre);
    toCamera(cam, pInFront, inFront);
    toCamera(cam, pAbove, above);

    bool centreOk  = near(centre[0], 0) && near(centre[1], 0) && near(centre[2], 0);
    bool forwardOk = near(inFront[0], 0) && near(inFront[1], 0) && near(inFront[2], 2);
    bool upOk      = above[1] < 0.0f;                               // OpenCV y points down
    bool posOk     = near(cam.position.x, 1) && near(cam.position.y, 2) && near(cam.position.z, 3);

    printf("[camera] centre->origin: %s | forward->+z: %s | up->-y: %s | position: %s\n",
           centreOk ? "yes" : "NO", forwardOk ? "yes" : "NO", upOk ? "yes" : "NO", posOk ? "yes" : "NO");
    printf("[camera] fx=%.2f cx=%.1f cy=%.1f\n", cam.fx, cam.cx, cam.cy);
    ok &= centreOk && forwardOk && upOk && posOk;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
