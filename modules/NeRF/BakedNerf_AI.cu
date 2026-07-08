#include "BakedNerf.h"
#undef CUDA_CHECK
#include "../TinyMLP/TinyMLP.h"
#include "../TinyMLP/EmbeddingTable.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <stdexcept>

// defined in otherKernels.cu (plain __global__ -> cross-TU host-stub launch, no RDC needed)
__global__ void compute_ray_aabb_inv_kernel(
    const uint32_t num_rays,
    const float3* __restrict__ rays_o,
    const float3* __restrict__ rays_d,
    const float3 aabb_min,
    const float3 aabb_max,
    const int num_cascades,
    float3* __restrict__ rays_d_inv,
    float* __restrict__ nears,
    float* __restrict__ fars);

BakedNerf::~BakedNerf() {
    delete m_deferredMLP;
}

// The hash-grid MMA inference kernel (networkFusionHashtable.cu loadVec4) reads input
// positions past n up to its row tile — it assumes the caller over-allocates. Every buffer
// handed to teacher.queryDensityLogit/queryRadiance gets this many extra rows.
static constexpr size_t kQueryPad = 8192;

static int popc64(uint64_t v) { int c = 0; while (v) { v &= v - 1; c++; } return c; }

// mip-360 contraction, copied from processRays.cu/otherKernels.cu (those are static there).
// Maps a metric/cascade-space position to the contracted [0,1]^3 the teacher MLP expects.
static __device__ __forceinline__ float3 bake_contract_pos(float3 pos, float3 aabb_min, float3 aabb_max) {
    float3 ext = make_float3(aabb_max.x - aabb_min.x, aabb_max.y - aabb_min.y, aabb_max.z - aabb_min.z);
    float3 rel = make_float3((pos.x - aabb_min.x) / ext.x, (pos.y - aabb_min.y) / ext.y, (pos.z - aabb_min.z) / ext.z);
    float3 n   = make_float3(rel.x * 2.0f - 1.0f, rel.y * 2.0f - 1.0f, rel.z * 2.0f - 1.0f);
    float3 c;
    c.x = fabsf(n.x) <= 1.0f ? n.x : (2.0f - 1.0f / fabsf(n.x)) * copysignf(1.0f, n.x);
    c.y = fabsf(n.y) <= 1.0f ? n.y : (2.0f - 1.0f / fabsf(n.y)) * copysignf(1.0f, n.y);
    c.z = fabsf(n.z) <= 1.0f ? n.z : (2.0f - 1.0f / fabsf(n.z)) * copysignf(1.0f, n.z);
    return make_float3(
        fmaxf(0.0f, fminf((c.x + 2.0f) * 0.25f, 1.0f)),
        fmaxf(0.0f, fminf((c.y + 2.0f) * 0.25f, 1.0f)),
        fmaxf(0.0f, fminf((c.z + 2.0f) * 0.25f, 1.0f)));
}

// same cascade rule as processRays.cu / InstantNerf.cu get_cascade (copied; static there)
static __device__ __forceinline__ int bake_get_cascade(float3 pos, float3 aabb_min, float3 aabb_max, int num_cascades) {
    float rx = fmaxf(pos.x / aabb_max.x, pos.x / aabb_min.x);
    float ry = fmaxf(pos.y / aabb_max.y, pos.y / aabb_min.y);
    float rz = fmaxf(pos.z / aabb_max.z, pos.z / aabb_min.z);
    float max_scale = fmaxf(rx, fmaxf(ry, rz));
    if (max_scale <= 1.0f) return 0;
    int cascade = (int)ceilf(log2f(max_scale));
    return max(0, min(cascade, num_cascades - 1));
}

// SH3 basis, identical coefficients to compute_SH_gather (render/fit must agree)
static __device__ __forceinline__ void bake_sh16(float3 d, float* sh) {
    float x = d.x, y = d.y, z = d.z;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    sh[0]  =  0.28209479f;
    sh[1]  = -0.48860251f * y;
    sh[2]  =  0.48860251f * z;
    sh[3]  = -0.48860251f * x;
    sh[4]  =  1.09254843f * x * y;
    sh[5]  = -1.09254843f * y * z;
    sh[6]  =  0.31539156f * (3.0f * z2 - 1.0f);
    sh[7]  = -1.09254843f * x * z;
    sh[8]  =  0.54627421f * (x2 - y2);
    sh[9]  = -0.59004359f * y * (3.0f * x2 - y2);
    sh[10] =  2.89061144f * x * y * z;
    sh[11] = -0.45704580f * y * (5.0f * z2 - 1.0f);
    sh[12] =  0.37317633f * z * (5.0f * z2 - 3.0f);
    sh[13] = -0.45704580f * x * (5.0f * z2 - 1.0f);
    sh[14] =  1.44530572f * z * (x2 - y2);
    sh[15] = -0.59004359f * x * (x2 - 3.0f * y2);
}

// metric position -> payload row: cascade -> cell -> compacted block -> mask rank.
// -1 when the point misses the sparse structure (unoccupied block or pruned sub-voxel).
static __device__ __forceinline__ int bake_row_lookup(
    float3 pos, float3 aabbMin, float3 aabbMax, uint3 gridRes, int numCascades,
    const int* __restrict__ reverseMap, const uint64_t* __restrict__ masks,
    const uint32_t* __restrict__ payloadOffset)
{
    const int B = SPARSE_B;
    int cascade = bake_get_cascade(pos, aabbMin, aabbMax, numCascades);
    float scale = exp2f((float)cascade);
    float3 cmin = make_float3(aabbMin.x * scale, aabbMin.y * scale, aabbMin.z * scale);
    float3 cext = make_float3((aabbMax.x - aabbMin.x) * scale, (aabbMax.y - aabbMin.y) * scale, (aabbMax.z - aabbMin.z) * scale);

    int fxmax = gridRes.x * B - 1, fymax = gridRes.y * B - 1, fzmax = gridRes.z * B - 1;
    int fx = min(max((int)floorf((pos.x - cmin.x) / cext.x * gridRes.x * B), 0), fxmax);
    int fy = min(max((int)floorf((pos.y - cmin.y) / cext.y * gridRes.y * B), 0), fymax);
    int fz = min(max((int)floorf((pos.z - cmin.z) / cext.z * gridRes.z * B), 0), fzmax);

    int gx = fx / B, sx = fx % B;
    int gy = fy / B, sy = fy % B;
    int gz = fz / B, sz = fz % B;

    int G = gridRes.x * gridRes.y * gridRes.z;
    int cell = cascade * G + gz * (gridRes.x * gridRes.y) + gy * gridRes.x + gx;
    int block = reverseMap[cell];
    if (block < 0) return -1;

    int sub = sz * B * B + sy * B + sx;
    uint64_t m = masks[block];
    if (!((m >> sub) & 1ull)) return -1;
    return (int)(payloadOffset[block] + __popcll(m & ((1ull << sub) - 1ull)));
}

// ------------------------------------------------------------------ bake kernels

// For a chunk of occupied blocks (dense cell ids = cascade*G + local), emit the contracted
// position of each of the B^3 sub-voxel CENTERS. Output index = blockInChunk*B^3 + sub, matching
// the order queryDensityLogit consumes and k_buildSubVoxelMasks reads.
__global__ void k_genSubVoxelPositions(
    const int* __restrict__ cellIds, int numBlocks, int B,
    uint3 gridRes, float3 aabbMin, float3 aabbMax,
    float* __restrict__ outPos)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int per = B * B * B;
    if (tid >= numBlocks * per) return;

    int sub = tid % per;
    int bi  = tid / per;
    int cell = cellIds[bi];

    int G = gridRes.x * gridRes.y * gridRes.z;
    int cascade = cell / G;
    int local   = cell % G;
    int gx = local % gridRes.x;
    int gy = (local / gridRes.x) % gridRes.y;
    int gz = local / (gridRes.x * gridRes.y);

    int sx = sub % B;
    int sy = (sub / B) % B;
    int sz = sub / (B * B);

    float scale = exp2f((float)cascade);
    float3 cmin = make_float3(aabbMin.x * scale, aabbMin.y * scale, aabbMin.z * scale);
    float3 cmax = make_float3(aabbMax.x * scale, aabbMax.y * scale, aabbMax.z * scale);
    float3 csz  = make_float3((cmax.x - cmin.x) / gridRes.x, (cmax.y - cmin.y) / gridRes.y, (cmax.z - cmin.z) / gridRes.z);

    float invB = 1.0f / (float)B;
    float3 pos = make_float3(
        cmin.x + (gx + (sx + 0.5f) * invB) * csz.x,
        cmin.y + (gy + (sy + 0.5f) * invB) * csz.y,
        cmin.z + (gz + (sz + 0.5f) * invB) * csz.z);

    pos = bake_contract_pos(pos, aabbMin, aabbMax);
    outPos[tid * 3 + 0] = pos.x;
    outPos[tid * 3 + 1] = pos.y;
    outPos[tid * 3 + 2] = pos.z;
}

// Threshold each sub-voxel's sigma against the SAME per-cascade opacity bar the occupancy grid
// uses; write the surviving-bit mask per block and keep the census stats (total + fill histogram).
__global__ void k_buildSubVoxelMasks(
    const float* __restrict__ logits, const int* __restrict__ cellIds,
    int numBlocks, int B, uint3 gridRes,
    float densityBias, float minDensityThreshold, float baseVoxel,
    uint64_t* __restrict__ masksOut,
    unsigned long long* __restrict__ survivorTotal,
    unsigned long long* __restrict__ fillHist)   // [B^3 + 1] bins
{
    int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi >= numBlocks) return;

    int per = B * B * B;
    int G = gridRes.x * gridRes.y * gridRes.z;
    int cascade = cellIds[bi] / G;
    float cascadeThresh = (minDensityThreshold / baseVoxel) / exp2f((float)cascade);

    uint64_t mask = 0;
    for (int s = 0; s < per; ++s) {
        float logit = logits[bi * per + s];
        float sigma = expf(fminf(logit - densityBias, 8.0f));
        if (sigma > cascadeThresh) mask |= (1ull << s);
    }
    masksOut[bi] = mask;
    int popc = __popcll(mask);
    atomicAdd(survivorTotal, (unsigned long long)popc);
    atomicAdd(&fillHist[popc], 1ULL);
}

// Contracted centers of the SURVIVING sub-voxels of a compacted block range, written densely
// at (payloadOffset[block] - rowBase) — i.e. chunk-local payload-row order.
__global__ void k_genSurvivorPositions(
    const int* __restrict__ cellIds, const uint64_t* __restrict__ masks,
    const uint32_t* __restrict__ payloadOffset,
    int firstBlock, int numBlocksChunk, int B,
    uint3 gridRes, float3 aabbMin, float3 aabbMax,
    uint32_t rowBase, float* __restrict__ outPos)
{
    int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi >= numBlocksChunk) return;
    int gb = firstBlock + bi;

    int cell = cellIds[gb];
    uint64_t m = masks[gb];
    uint32_t out = payloadOffset[gb] - rowBase;

    int G = gridRes.x * gridRes.y * gridRes.z;
    int cascade = cell / G;
    int local   = cell % G;
    int gx = local % gridRes.x;
    int gy = (local / gridRes.x) % gridRes.y;
    int gz = local / (gridRes.x * gridRes.y);

    float scale = exp2f((float)cascade);
    float3 cmin = make_float3(aabbMin.x * scale, aabbMin.y * scale, aabbMin.z * scale);
    float3 cmax = make_float3(aabbMax.x * scale, aabbMax.y * scale, aabbMax.z * scale);
    float3 csz  = make_float3((cmax.x - cmin.x) / gridRes.x, (cmax.y - cmin.y) / gridRes.y, (cmax.z - cmin.z) / gridRes.z);

    float invB = 1.0f / (float)B;
    for (int s = 0; s < B * B * B; ++s) {
        if (!((m >> s) & 1ull)) continue;
        int sx = s % B, sy = (s / B) % B, sz = s / (B * B);
        float3 pos = make_float3(
            cmin.x + (gx + (sx + 0.5f) * invB) * csz.x,
            cmin.y + (gy + (sy + 0.5f) * invB) * csz.y,
            cmin.z + (gz + (sz + 0.5f) * invB) * csz.z);
        pos = bake_contract_pos(pos, aabbMin, aabbMax);
        outPos[(size_t)out * 3 + 0] = pos.x;
        outPos[(size_t)out * 3 + 1] = pos.y;
        outPos[(size_t)out * 3 + 2] = pos.z;
        out++;
    }
}

__global__ void k_padPositions(float* __restrict__ pos, int from, int to) {
    int i = from + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= to) return;
    pos[i * 3 + 0] = 0.5f; pos[i * 3 + 1] = 0.5f; pos[i * 3 + 2] = 0.5f;
}

__global__ void k_writeSigma(half* __restrict__ sigmaOut, const float* __restrict__ logits,
                             int n, float densityBias) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    sigmaOut[i] = __float2half(expf(fminf(logits[i] - densityBias, 8.0f)));
}

__global__ void k_initDiffuse(float* __restrict__ masters, uint32_t rowBase,
                              const float* __restrict__ rgb, int n, float invD, int F) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    size_t r = (size_t)(rowBase + i) * F;
    masters[r + 0] += rgb[i * 3 + 0] * invD;
    masters[r + 1] += rgb[i * 3 + 1] * invD;
    masters[r + 2] += rgb[i * 3 + 2] * invD;
}

// ------------------------------------------------------------------ fit kernels

static __device__ __forceinline__ uint32_t bake_hash(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}
static __device__ __forceinline__ float bake_rand01(uint32_t& s) {
    s = bake_hash(s);
    return (s >> 8) * (1.0f / 16777216.0f);
}

// Random orbit rays: camera on a shell around the scene center, aimed at a jittered point
// in the central content box. Realistic-ish (pos, dir) coverage without a dataset.
__global__ void k_genOrbitRays(int numRays, uint32_t seed, float3 aabbMin, float3 aabbMax,
                               float3* __restrict__ o, float3* __restrict__ d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numRays) return;
    uint32_t s = bake_hash(seed * 0x9E3779B9u + (uint32_t)i + 1u);

    float z   = 2.0f * bake_rand01(s) - 1.0f;
    float phi = 6.2831853f * bake_rand01(s);
    float rxy = sqrtf(fmaxf(0.0f, 1.0f - z * z));
    float3 dir0 = make_float3(rxy * cosf(phi), rxy * sinf(phi), z);

    float3 c = make_float3(0.5f * (aabbMin.x + aabbMax.x), 0.5f * (aabbMin.y + aabbMax.y),
                           0.5f * (aabbMin.z + aabbMax.z));
    float half_ext = 0.5f * (aabbMax.x - aabbMin.x);
    float radius = half_ext * (1.2f + 1.3f * bake_rand01(s));

    float3 org = make_float3(c.x + dir0.x * radius, c.y + dir0.y * radius, c.z + dir0.z * radius);
    float3 tgt = make_float3(c.x + (bake_rand01(s) - 0.5f) * half_ext,
                             c.y + (bake_rand01(s) - 0.5f) * half_ext,
                             c.z + (bake_rand01(s) - 0.5f) * half_ext);

    float3 dd = make_float3(tgt.x - org.x, tgt.y - org.y, tgt.z - org.z);
    float len = sqrtf(dd.x * dd.x + dd.y * dd.y + dd.z * dd.z);
    dd = make_float3(dd.x / len, dd.y / len, dd.z / len);

    o[i] = org; d[i] = dd;
}

// Per marched sample: metric position -> payload row + frozen sigma + contracted position
// (for the teacher query). Pad region [totalHits, paddedHits) gets row -1 / sigma 0.
__global__ void k_gatherSamples(
    int totalHits, int paddedHits,
    const float3* __restrict__ rays_o, const float3* __restrict__ rays_d,
    const uint32_t* __restrict__ rayIdx, const float* __restrict__ tHits,
    const int* __restrict__ reverseMap, const uint64_t* __restrict__ masks,
    const uint32_t* __restrict__ payloadOffset, const half* __restrict__ voxelSigma,
    uint3 gridRes, int numCascades, float3 aabbMin, float3 aabbMax,
    int* __restrict__ rowOut, float* __restrict__ sigmaOut, float* __restrict__ posOut)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= paddedHits) return;
    if (i >= totalHits) {
        rowOut[i] = -1; sigmaOut[i] = 0.0f;
        posOut[(size_t)i * 3 + 0] = 0.5f; posOut[(size_t)i * 3 + 1] = 0.5f; posOut[(size_t)i * 3 + 2] = 0.5f;
        return;
    }
    uint32_t r = rayIdx[i];
    float3 o = rays_o[r], d = rays_d[r];
    float t = tHits[i];
    float3 p = make_float3(o.x + t * d.x, o.y + t * d.y, o.z + t * d.z);

    int row = bake_row_lookup(p, aabbMin, aabbMax, gridRes, numCascades, reverseMap, masks, payloadOffset);
    rowOut[i] = row;
    sigmaOut[i] = (row >= 0) ? __half2float(voxelSigma[row]) : 0.0f;

    float3 cp = bake_contract_pos(p, aabbMin, aabbMax);
    posOut[(size_t)i * 3 + 0] = cp.x;
    posOut[(size_t)i * 3 + 1] = cp.y;
    posOut[(size_t)i * 3 + 2] = cp.z;
}

// Volume-rendering weights w_i = T_i * alpha_i along each ray from the frozen baked sigma.
__global__ void k_computeWeights(
    int numRays, const uint32_t* __restrict__ offsets, const uint32_t* __restrict__ steps,
    const float* __restrict__ tHits, const float* __restrict__ sigma,
    float dt0, float* __restrict__ w)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= numRays) return;
    uint32_t off = offsets[r], n = steps[r];

    float T = 1.0f, prevDt = dt0;
    for (uint32_t j = 0; j < n; ++j) {
        float dt = (j + 1 < n) ? fmaxf(tHits[off + j + 1] - tHits[off + j], 0.0f) : prevDt;
        prevDt = dt;
        float a  = 1.0f - expf(-sigma[off + j] * dt);
        w[off + j] = T * a;
        T *= (1.0f - a);
        if (T < 1e-4f) {
            for (uint32_t k = j + 1; k < n; ++k) w[off + k] = 0.0f;
            break;
        }
    }
}

// Deferred MLP input per sample: [diffuse(3), feat(K), SH3(dir)(16)], unpadded half.
__global__ void k_buildDeferredInput(
    int totalHits, int paddedHits, const int* __restrict__ row,
    const float* __restrict__ masters, int K,
    const uint32_t* __restrict__ rayIdx, const float3* __restrict__ rays_d,
    half* __restrict__ mlpIn)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= paddedHits) return;
    int inDim = 3 + K + 16;
    half* out = mlpIn + (size_t)i * inDim;
    if (i >= totalHits) {
        for (int c = 0; c < inDim; ++c) out[c] = __float2half(0.0f);
        return;
    }
    int rr = row[i];
    int F = 3 + K;
    for (int c = 0; c < 3 + K; ++c)
        out[c] = __float2half(rr >= 0 ? masters[(size_t)rr * F + c] : 0.0f);

    float sh[16];
    bake_sh16(rays_d[rayIdx[i]], sh);
    for (int c = 0; c < 16; ++c) out[3 + K + c] = __float2half(sh[c]);
}

// Per-point distillation loss: Cs = sigmoid(g(.)) (already applied by the MLP forward);
// delta = w * 2 * (Cs - Ct). Backward wants the grad w.r.t. the PRE-sigmoid logit, so the
// sigmoid derivative Cs*(1-Cs) is folded in manually — same as compute_color_grad.
__global__ void k_lossGrad(
    int totalHits, int paddedHits,
    const float* __restrict__ gOut, const float* __restrict__ teacherRGB,
    const float* __restrict__ w, float lossScale,
    half* __restrict__ dLdG, float* __restrict__ lossAccum)   // [0]=sum w*|e|^2, [1]=sum w
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= paddedHits) return;
    float wi = 0.0f, e2 = 0.0f;
    if (i >= totalHits) {
        dLdG[(size_t)i * 3 + 0] = __float2half(0.0f);
        dLdG[(size_t)i * 3 + 1] = __float2half(0.0f);
        dLdG[(size_t)i * 3 + 2] = __float2half(0.0f);
    } else {
        wi = w[i];
        for (int c = 0; c < 3; ++c) {
            float cs = gOut[(size_t)i * 3 + c];
            float e  = cs - teacherRGB[(size_t)i * 3 + c];
            float cg = wi * 2.0f * e * cs * (1.0f - cs) * lossScale;
            cg = fmaxf(-65504.0f, fminf(65504.0f, cg));
            if (cg != cg) cg = 0.0f;
            dLdG[(size_t)i * 3 + c] = __float2half(cg);
            e2 += e * e;
        }
    }
    // warp-reduce before the two hot atomics
    float le = wi * e2, lw = wi;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        le += __shfl_xor_sync(0xFFFFFFFF, le, o);
        lw += __shfl_xor_sync(0xFFFFFFFF, lw, o);
    }
    if ((threadIdx.x & 31) == 0 && lw > 0.0f) {
        atomicAdd(&lossAccum[0], le);
        atomicAdd(&lossAccum[1], lw);
    }
}

// One (row, grad7) pair per sample for the EmbeddingTable. With the sigmoid head there is
// no additive diffuse skip: all 3+K channels are MLP inputs, so their gradients are just the
// first F columns of dx (the MLP's PADDED input gradient, stride inPad, still x loss_scale).
__global__ void k_emitTableGrads(
    int totalHits, int paddedHits, const int* __restrict__ row,
    const half* __restrict__ dx, int inPad, int K,
    float invLossScale, float* __restrict__ pairGrads)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= paddedHits) return;
    int F = 3 + K;
    float* g = pairGrads + (size_t)i * F;
    if (i >= totalHits || row[i] < 0) {
        for (int c = 0; c < F; ++c) g[c] = 0.0f;
        return;
    }
    const half* dxi = dx + (size_t)i * inPad;
    for (int c = 0; c < F; ++c)
        g[c] = __half2float(dxi[c]) * invLossScale;
}

// Quantize the fp32 masters into the shipped payload (diffuse uint8 x3, feat half xK).
__global__ void k_writeBackPayload(
    const float* __restrict__ masters, uint32_t numVoxels, int K,
    uint8_t* __restrict__ diffuse, half* __restrict__ feat)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numVoxels) return;
    int F = 3 + K;
    for (int c = 0; c < 3; ++c) {
        float v = fminf(fmaxf(masters[(size_t)i * F + c], 0.0f), 1.0f);
        diffuse[(size_t)i * 3 + c] = (uint8_t)(v * 255.0f + 0.5f);
    }
    for (int k = 0; k < K; ++k)
        feat[(size_t)i * K + k] = __float2half(masters[(size_t)i * F + 3 + k]);
}

// ------------------------------------------------------------------ render kernels
// Per-pixel deferred render (docs/baking_phase2_joint_training.md "Render"):
// accumulate C_diff = sum w*diffuse and F = sum w*feat along the ray, ONE g call per
// pixel on [C_diff, F, SH3(viewdir)], final = clamp(C_diff + g + T*bg, 0, 1).

// Per hit: payload row -> sigma + the 7 quantized appearance channels (u8 diffuse, fp16 feat).
__global__ void k_renderGatherHit(
    int totalHits,
    const float3* __restrict__ rays_o, const float3* __restrict__ rays_d,
    const uint32_t* __restrict__ rayIdx, const float* __restrict__ tHits,
    const int* __restrict__ reverseMap, const uint64_t* __restrict__ masks,
    const uint32_t* __restrict__ payloadOffset, const half* __restrict__ voxelSigma,
    const uint8_t* __restrict__ voxelDiffuse, const half* __restrict__ voxelFeat, int K,
    uint3 gridRes, int numCascades, float3 aabbMin, float3 aabbMax,
    float* __restrict__ sigmaOut, float* __restrict__ rgb7Out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= totalHits) return;
    int F = 3 + K;
    uint32_t r = rayIdx[i];
    float3 o = rays_o[r], d = rays_d[r];
    float t = tHits[i];
    float3 p = make_float3(o.x + t * d.x, o.y + t * d.y, o.z + t * d.z);

    int row = bake_row_lookup(p, aabbMin, aabbMax, gridRes, numCascades, reverseMap, masks, payloadOffset);
    float* out = rgb7Out + (size_t)i * F;
    if (row < 0) {
        sigmaOut[i] = 0.0f;
        for (int c = 0; c < F; ++c) out[c] = 0.0f;
        return;
    }
    sigmaOut[i] = __half2float(voxelSigma[row]);
    for (int c = 0; c < 3; ++c) out[c] = voxelDiffuse[(size_t)row * 3 + c] * (1.0f / 255.0f);
    for (int k = 0; k < K; ++k) out[3 + k] = __half2float(voxelFeat[(size_t)row * K + k]);
}

__global__ void k_compositeDeferred(
    int numRays, const uint32_t* __restrict__ offsets, const uint32_t* __restrict__ steps,
    const float* __restrict__ tHits, const float* __restrict__ sigma,
    const float* __restrict__ rgb7, int K, float dt0,
    float* __restrict__ acc7, float* __restrict__ Tout)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= numRays) return;
    int F = 3 + K;
    uint32_t off = offsets[r], n = steps[r];

    float acc[3 + 16];   // K <= 16
    for (int c = 0; c < F; ++c) acc[c] = 0.0f;
    float T = 1.0f, prevDt = dt0;
    for (uint32_t j = 0; j < n; ++j) {
        float dt = (j + 1 < n) ? fmaxf(tHits[off + j + 1] - tHits[off + j], 0.0f) : prevDt;
        prevDt = dt;
        float a = 1.0f - expf(-sigma[off + j] * dt);
        float w = T * a;
        const float* v = rgb7 + (size_t)(off + j) * F;
        for (int c = 0; c < F; ++c) acc[c] += w * v[c];
        T *= (1.0f - a);
        if (T < 1e-4f) break;
    }
    for (int c = 0; c < F; ++c) acc7[(size_t)r * F + c] = acc[c];
    Tout[r] = T;
}

// The MLP was fit on RAW per-sample [diffuse, feat]; accumulated channels are opacity-scaled
// (sum w = 1-T), so normalize by the weight sum to hand it weighted-AVERAGE features.
__global__ void k_buildPixelInput(
    int numRays, int paddedRays, const float* __restrict__ acc7, int K,
    const float* __restrict__ T,
    const float3* __restrict__ rays_d, half* __restrict__ mlpIn)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= paddedRays) return;
    int inDim = 3 + K + 16;
    half* out = mlpIn + (size_t)r * inDim;
    if (r >= numRays) {
        for (int c = 0; c < inDim; ++c) out[c] = __float2half(0.0f);
        return;
    }
    int F = 3 + K;
    float invW = 1.0f / fmaxf(1.0f - T[r], 1e-4f);
    for (int c = 0; c < F; ++c) out[c] = __float2half(acc7[(size_t)r * F + c] * invW);
    float sh[16];
    bake_sh16(rays_d[r], sh);
    for (int c = 0; c < 16; ++c) out[F + c] = __float2half(sh[c]);
}

// Cs = sigmoid MLP output is the surface color; blend with background by transmittance.
__global__ void k_finalizePixel(
    int numRays, const float* __restrict__ gOut, const float* __restrict__ T,
    float3 bg, float* __restrict__ rgbOut)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= numRays) return;
    float bgc[3] = { bg.x, bg.y, bg.z };
    float Tr = T[r];
    for (int c = 0; c < 3; ++c) {
        float v = (1.0f - Tr) * gOut[(size_t)r * 3 + c] + Tr * bgc[c];
        rgbOut[(size_t)r * 3 + c] = fminf(fmaxf(v, 0.0f), 1.0f);
    }
}

// For composited-color diagnostics: pixel = accumulated (already w-weighted) color + T*bg.
__global__ void k_finalizeComposited(
    int numRays, const float* __restrict__ acc3, const float* __restrict__ T,
    float3 bg, float* __restrict__ rgbOut)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= numRays) return;
    float bgc[3] = { bg.x, bg.y, bg.z };
    for (int c = 0; c < 3; ++c) {
        float v = acc3[(size_t)r * 3 + c] + T[r] * bgc[c];
        rgbOut[(size_t)r * 3 + c] = fminf(fmaxf(v, 0.0f), 1.0f);
    }
}

// ------------------------------------------------------------------ host

void BakedNerf::init(const BakeOptions& opts) {
    m_opts = opts;

    MLPOption deferredOpts;
    deferredOpts.activationType = ACT_RELU;
    deferredOpts.outputActivation = OUT_ACT_SIGMOID;  // Cs = sigmoid(g(.)), teacher-style head;
                                                      // backward gets c*(1-c) folded in manually
    deferredOpts.inputDim = deferredInDim();          // [diffuse(3), feat(K), SH3(dir)(16)] = 23
    deferredOpts.hiddenDim = m_opts.deferredHidden;
    deferredOpts.outputDim = 3;
    deferredOpts.numLayers = m_opts.deferredLayers;

    // Fit-sized batch: hit cap per step (see fitDeferred).
    m_deferredMLP = new TinyMLP(deferredOpts, 1 << 20, 1 << 20);
}

// Build the two-tier structure from the teacher: Tier-1 occupancy bitgrid + census,
// Tier-2 sub-voxel masks -> compacted payload rows with frozen fp16 sigma.
// fineTuneSteps > 0 additionally runs the joint appearance fit (fitDeferred).
void BakedNerf::distil(InstantNerf& teacher) {
    m_teacherOpts = teacher.options();

    const uint3 gr = m_teacherOpts.gridResolution;
    if (m_opts.voxelGridResolution.x != gr.x * SPARSE_B ||
        m_opts.voxelGridResolution.y != gr.y * SPARSE_B ||
        m_opts.voxelGridResolution.z != gr.z * SPARSE_B) {
        throw std::runtime_error(
            "BakedNerf::distil: voxelGridResolution must equal gridResolution * SPARSE_B");
    }

    m_bakeThreshold = (m_opts.sigmaThreshold >= 0.0f) ? m_opts.sigmaThreshold
                                                      : m_teacherOpts.minDensityThreshold;
    m_occupancyGrid = DeviceBuffer<uint8_t>(teacher.occupancyBytes());
    teacher.buildOccupancyBitgrid(m_occupancyGrid.data(), m_bakeThreshold);

    uint8_t* cpuGrid = new uint8_t[teacher.occupancyBytes()];
    m_occupancyGrid.copyHost(cpuGrid, teacher.occupancyBytes());

    int levels = m_teacherOpts.levelsMipmap;
    uint3 res = m_teacherOpts.gridResolution;
    int cascadeOffset = 0;
    int baseElems = (int)(res.x * res.y * res.z) / 8;
    for (int i = 0; i < levels; i++) {
        cascadeOffset += (int)(res.x * res.y * res.z);
        res = make_uint3(res.x / 2, res.y / 2, res.z / 2);
    }

    cascadeOffset /= 8;

    int cascades = m_teacherOpts.numCascades;
    int G = (int)(m_teacherOpts.gridResolution.x * m_teacherOpts.gridResolution.y * m_teacherOpts.gridResolution.z);
    long long perCascadeVoxels = (long long)baseElems * 8;   // 128^3 base cells per cascade
    int numFilledVoxels = 0;
    int offset = 0;

    // Collect occupied base-level cells (dense ids = cascade*G + local) while we count them.
    std::vector<int> occupiedCells;
    occupiedCells.reserve(1 << 21);

    printf("threshold %.4f :\n", m_bakeThreshold);
    for (int i = 0; i < cascades; i++) {
        int cascadeFilled = 0;
        for (int j = 0; j < baseElems; j++) {
            uint8_t byte = cpuGrid[offset + j];
            for (int b = 0; b < 8; ++b) {
                if ((byte >> b) & 1) {
                    cascadeFilled++;
                    occupiedCells.push_back(i * G + j * 8 + b);   // local = j*8 + b
                }
            }
        }
        printf("  cascade %d : filled %8d / %lld (%6.3f%%)\n",
               i, cascadeFilled, perCascadeVoxels,
               100.0 * cascadeFilled / (double)perCascadeVoxels);
        numFilledVoxels += cascadeFilled;
        offset += cascadeOffset;        // full-pyramid bytes per cascade = stride between cascades
    }

    long long totalBaseVoxels = perCascadeVoxels * cascades;   // 128^3 * cascades base cells
    printf("  TOTAL     : filled %8d / %lld (%6.3f%%)\n",
           numFilledVoxels, totalBaseVoxels,
           100.0 * numFilledVoxels / (double)totalBaseVoxels);

    delete[] cpuGrid;

    // ---- Tier-2: sub-voxel masks at the per-cascade bar (census + structure in one sweep) ----
    const int B = SPARSE_B;
    const int per = B * B * B;                       // 64
    const long long numBlocks = (long long)occupiedCells.size();
    m_numBlocks = (uint32_t)numBlocks;
    m_numVoxels = 0;
    if (numBlocks == 0) { printf("[sub-voxel] no occupied blocks\n"); return; }

    DeviceBuffer<int> d_cellIds((size_t)numBlocks);
    cudaMemcpy(d_cellIds.data(), occupiedCells.data(), (size_t)numBlocks * sizeof(int), cudaMemcpyHostToDevice);

    const int chunkBlocks = 1 << 15;                 // 32768 blocks -> 2,097,152 sub-voxels / chunk
    DeviceBuffer<float> d_pos(((size_t)chunkBlocks * per + kQueryPad) * 3);
    DeviceBuffer<float> d_logit((size_t)chunkBlocks * per + kQueryPad);
    DeviceBuffer<uint64_t> d_allMasks((size_t)numBlocks);
    DeviceBuffer<unsigned long long> d_survivors(1);
    DeviceBuffer<unsigned long long> d_hist((size_t)per + 1);
    d_survivors.fill(0);
    d_hist.fill(0);

    const float baseVoxel = (m_teacherOpts.aabbMax.x - m_teacherOpts.aabbMin.x) / (float)m_teacherOpts.gridResolution.x;
    constexpr int BS = 256;

    for (long long c0 = 0; c0 < numBlocks; c0 += chunkBlocks) {
        int blocksThis = (int)std::min((long long)chunkBlocks, numBlocks - c0);
        int nSub = blocksThis * per;

        int gsPos = (nSub + BS - 1) / BS;
        k_genSubVoxelPositions<<<gsPos, BS>>>(
            d_cellIds.data() + c0, blocksThis, B,
            m_teacherOpts.gridResolution, m_teacherOpts.aabbMin, m_teacherOpts.aabbMax,
            d_pos.data());

        teacher.queryDensityLogit(d_pos.data(), nSub, d_logit.data());

        int gsThr = (blocksThis + BS - 1) / BS;
        k_buildSubVoxelMasks<<<gsThr, BS>>>(
            d_logit.data(), d_cellIds.data() + c0, blocksThis, B,
            m_teacherOpts.gridResolution, m_teacherOpts.densityBias, m_bakeThreshold, baseVoxel,
            d_allMasks.data() + c0,
            d_survivors.data(), d_hist.data());
    }
    cudaDeviceSynchronize();

    unsigned long long survivors = 0;
    std::vector<unsigned long long> hist((size_t)per + 1, 0);
    d_survivors.copyHost(&survivors, 1);
    d_hist.copyHost(hist.data(), (size_t)per + 1);
    m_numVoxels = (uint32_t)survivors;

    const long long candidateSubVoxels = numBlocks * per;
    auto bucket = [&](int lo, int hi){ unsigned long long s = 0; for (int k = lo; k <= hi; ++k) s += hist[k]; return s; };

    printf("\n[sub-voxel sigma survival @ B=%d, threshold %.4f]\n", B, m_bakeThreshold);
    printf("  occupied blocks       : %lld\n", numBlocks);
    printf("  candidate sub-voxels  : %lld (blocks x %d)\n", candidateSubVoxels, per);
    printf("  surviving sub-voxels  : %llu  (%.3f%% of candidates, mean %.2f/%d per block)\n",
           survivors, 100.0 * survivors / (double)candidateSubVoxels,
           (double)survivors / (double)numBlocks, per);
    printf("  per-block fill (blocks with N occupied sub-voxels):\n");
    printf("    N=0     : %10llu  (blocks whose 64 sub-centers all missed the bar)\n", hist[0]);
    printf("    N=1-8   : %10llu\n", bucket(1, 8));
    printf("    N=9-16  : %10llu\n", bucket(9, 16));
    printf("    N=17-32 : %10llu\n", bucket(17, 32));
    printf("    N=33-48 : %10llu\n", bucket(33, 48));
    printf("    N=49-63 : %10llu\n", bucket(49, 63));
    printf("    N=64    : %10llu  (fully dense blocks)\n", hist[64]);
    printf("  payload @13B          : %.1f MB   | fit masters+S @32B (<=): %.1f MB\n",
           survivors * 13.0 / (1024.0 * 1024.0), survivors * 32.0 / (1024.0 * 1024.0));

    // ---- compact non-empty blocks -> payload offsets + reverse map (Tier-2 structure) ----
    std::vector<uint64_t> hMasks((size_t)numBlocks);
    d_allMasks.copyHost(hMasks.data(), (size_t)numBlocks);

    std::vector<int>      keptCells;  keptCells.reserve((size_t)numBlocks);
    std::vector<uint64_t> keptMasks;  keptMasks.reserve((size_t)numBlocks);
    std::vector<uint32_t> hOffsets;   hOffsets.reserve((size_t)numBlocks + 1);
    uint32_t acc = 0;
    for (long long b = 0; b < numBlocks; ++b) {
        if (!hMasks[b]) continue;
        keptCells.push_back(occupiedCells[(size_t)b]);
        keptMasks.push_back(hMasks[b]);
        hOffsets.push_back(acc);
        acc += (uint32_t)popc64(hMasks[b]);
    }
    hOffsets.push_back(acc);
    m_numBlocks = (uint32_t)keptCells.size();
    if (acc != (uint32_t)survivors)
        fprintf(stderr, "[bake] WARNING: mask popcount sum %u != census survivors %llu\n", acc, survivors);

    m_voxelMask          = DeviceBuffer<uint64_t>(m_numBlocks);
    m_blockCellId        = DeviceBuffer<int>(m_numBlocks);
    m_blockPayloadOffset = DeviceBuffer<uint32_t>((size_t)m_numBlocks + 1);
    cudaMemcpy(m_voxelMask.data(), keptMasks.data(), (size_t)m_numBlocks * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(m_blockCellId.data(), keptCells.data(), (size_t)m_numBlocks * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(m_blockPayloadOffset.data(), hOffsets.data(), ((size_t)m_numBlocks + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice);

    std::vector<int> rev((size_t)cascades * G, -1);
    for (uint32_t b = 0; b < m_numBlocks; ++b) rev[(size_t)keptCells[b]] = (int)b;
    m_reverseMap = DeviceBuffer<int>((size_t)cascades * G);
    cudaMemcpy(m_reverseMap.data(), rev.data(), (size_t)cascades * G * sizeof(int), cudaMemcpyHostToDevice);

    // ---- frozen sigma payload: query the teacher once more, survivors only ----
    m_voxelSigma = DeviceBuffer<half>(m_numVoxels);
    {
        uint32_t survCap = 0;
        for (uint32_t c0 = 0; c0 < m_numBlocks; c0 += chunkBlocks) {
            uint32_t cEnd = std::min(c0 + (uint32_t)chunkBlocks, m_numBlocks);
            survCap = std::max(survCap, hOffsets[cEnd] - hOffsets[c0]);
        }
        DeviceBuffer<float> d_spos(((size_t)survCap + kQueryPad) * 3);
        DeviceBuffer<float> d_slogit((size_t)survCap + kQueryPad);

        for (uint32_t c0 = 0; c0 < m_numBlocks; c0 += chunkBlocks) {
            uint32_t cEnd = std::min(c0 + (uint32_t)chunkBlocks, m_numBlocks);
            uint32_t rowBase = hOffsets[c0];
            int nSurv = (int)(hOffsets[cEnd] - rowBase);
            if (nSurv == 0) continue;
            int padded = (nSurv + 15) & ~15;

            int gsB = ((int)(cEnd - c0) + BS - 1) / BS;
            k_genSurvivorPositions<<<gsB, BS>>>(
                m_blockCellId.data(), m_voxelMask.data(), m_blockPayloadOffset.data(),
                (int)c0, (int)(cEnd - c0), B,
                m_teacherOpts.gridResolution, m_teacherOpts.aabbMin, m_teacherOpts.aabbMax,
                rowBase, d_spos.data());
            if (padded > nSurv) {
                int gsP = ((padded - nSurv) + BS - 1) / BS;
                k_padPositions<<<gsP, BS>>>(d_spos.data(), nSurv, padded);
            }
            teacher.queryDensityLogit(d_spos.data(), padded, d_slogit.data());
            int gsW = (nSurv + BS - 1) / BS;
            k_writeSigma<<<gsW, BS>>>(m_voxelSigma.data() + rowBase, d_slogit.data(),
                                      nSurv, m_teacherOpts.densityBias);
        }
        cudaDeviceSynchronize();
    }
    printf("  structure: %u blocks kept, %u payload rows, sigma payload %.1f MB\n",
           m_numBlocks, m_numVoxels, m_numVoxels * 2.0 / (1024.0 * 1024.0));

    if (m_opts.fineTuneSteps > 0) fitDeferred(teacher);
}

// Phase-2 joint appearance distillation (docs/baking_phase2_joint_training.md):
// per-point teacher supervision on marched rays; trains per-voxel diffuse+feat rows
// (EmbeddingTable, row-wise Adagrad) jointly with the shared deferred MLP g (Adam).
void BakedNerf::fitDeferred(InstantNerf& teacher) {
    const NerfOptions& t = m_teacherOpts;
    const int   B      = SPARSE_B;
    const int   K      = m_opts.viewFeatures;
    const int   F      = 3 + K;
    const int   inDim  = deferredInDim();
    int inPad = 8; while (inPad < inDim) inPad <<= 1;    // TinyMLP pads input grads to pow2
    const uint32_t N   = m_numVoxels;
    const int   R      = m_opts.fitRaysPerStep;
    const int   hitCap = (1 << 20) - 16;
    const float LOSS_SCALE = 128.0f;
    constexpr int BS = 256;
    const int G = (int)(t.gridResolution.x * t.gridResolution.y * t.gridResolution.z);
    (void)G;

    if (N == 0) { printf("[fit] no payload rows, skipping\n"); return; }

    size_t freeB, totalB;
    cudaMemGetInfo(&freeB, &totalB);
    printf("\n[fit] joint distillation: %u rows x %d ch + deferred MLP | %d steps, %d rays/step | free VRAM %.2f GB\n",
           N, F, m_opts.fineTuneSteps, R, freeB / (1024.0 * 1024.0 * 1024.0));

    // ---- trainable rows over the payload (fp32 masters + 4 B/row Adagrad state)
    std::vector<float> lrs(F, m_opts.learningRate);
    EmbeddingTableOption topt;
    topt.rows = N; topt.num_features = F; topt.lr = lrs.data();
    EmbeddingTable table(topt, 1 << 20);

    // ---- Phase-1 diffuse init: teacher radiance averaged over numDiffuseDirs directions
    std::vector<uint32_t> hOffsets((size_t)m_numBlocks + 1);
    m_blockPayloadOffset.copyHost(hOffsets.data(), (size_t)m_numBlocks + 1);

    const int D = std::max(1, m_opts.numDiffuseDirs);
    const int chunkBlocks = 1 << 15;
    {
        uint32_t survCap = 0;
        for (uint32_t c0 = 0; c0 < m_numBlocks; c0 += chunkBlocks) {
            uint32_t cEnd = std::min(c0 + (uint32_t)chunkBlocks, m_numBlocks);
            survCap = std::max(survCap, hOffsets[cEnd] - hOffsets[c0]);
        }
        DeviceBuffer<float> d_spos(((size_t)survCap + kQueryPad) * 3);
        DeviceBuffer<float> d_srgb(((size_t)survCap + kQueryPad) * 3);

        std::vector<float3> dirs(D);
        for (int i = 0; i < D; i++) {           // fibonacci sphere
            float z = 1.0f - 2.0f * (i + 0.5f) / D;
            float r = sqrtf(std::max(0.0f, 1.0f - z * z));
            float phi = 2.39996323f * i;
            dirs[i] = make_float3(r * cosf(phi), r * sinf(phi), z);
        }

        for (uint32_t c0 = 0; c0 < m_numBlocks; c0 += chunkBlocks) {
            uint32_t cEnd = std::min(c0 + (uint32_t)chunkBlocks, m_numBlocks);
            uint32_t rowBase = hOffsets[c0];
            int nSurv = (int)(hOffsets[cEnd] - rowBase);
            if (nSurv == 0) continue;
            int padded = (nSurv + 15) & ~15;

            int gsB = ((int)(cEnd - c0) + BS - 1) / BS;
            k_genSurvivorPositions<<<gsB, BS>>>(
                m_blockCellId.data(), m_voxelMask.data(), m_blockPayloadOffset.data(),
                (int)c0, (int)(cEnd - c0), B,
                t.gridResolution, t.aabbMin, t.aabbMax, rowBase, d_spos.data());
            if (padded > nSurv) {
                int gsP = ((padded - nSurv) + BS - 1) / BS;
                k_padPositions<<<gsP, BS>>>(d_spos.data(), nSurv, padded);
            }
            for (int di = 0; di < D; ++di) {
                teacher.queryRadiance(d_spos.data(), padded, dirs[di], d_srgb.data());
                int gsI = (nSurv + BS - 1) / BS;
                k_initDiffuse<<<gsI, BS>>>(table.masterWeights(), rowBase, d_srgb.data(),
                                           nSurv, 1.0f / D, F);
            }
        }
        cudaDeviceSynchronize();
        printf("[fit] diffuse initialized from %d teacher directions\n", D);
    }

    // ---- per-step buffers (batch-scaled; freed with scope when the fit ends)
    const int hitBuf = 1 << 20;
    DeviceBuffer<float3>   d_o(R), d_d(R), d_dinv(R);
    DeviceBuffer<float>    d_near(R), d_far(R);
    DeviceBuffer<uint32_t> d_steps(R), d_offs(R), d_bsums(((size_t)R + 1023) / 1024), d_active(1);
    DeviceBuffer<float>    d_marchPos((size_t)hitBuf * 4);      // marcher writes float4/hit
    DeviceBuffer<uint32_t> d_rayIdx(hitBuf);
    DeviceBuffer<float>    d_t(hitBuf);
    DeviceBuffer<int>      d_row(hitBuf);
    DeviceBuffer<float>    d_sig(hitBuf), d_w(hitBuf);
    DeviceBuffer<float>    d_cpos(((size_t)hitBuf + kQueryPad) * 3), d_trgb(((size_t)hitBuf + kQueryPad) * 3);
    DeviceBuffer<float>    d_gout((size_t)hitBuf * 3);
    DeviceBuffer<half>     d_in((size_t)hitBuf * inDim), d_dldg((size_t)hitBuf * 3), d_dx((size_t)hitBuf * inPad);
    DeviceBuffer<float>    d_pair((size_t)hitBuf * F);
    DeviceBuffer<float>    d_loss(2);

    const float dt0 = (t.aabbMax.x - t.aabbMin.x) / (float)t.gridResolution.x
                      / (float)std::max(1, t.samplesPerVoxel);

    // ---- joint fit loop
    for (int step = 0; step < m_opts.fineTuneSteps; ++step) {
        int gsR = (R + BS - 1) / BS;
        k_genOrbitRays<<<gsR, BS>>>(R, (uint32_t)(step + 1), t.aabbMin, t.aabbMax,
                                    d_o.data(), d_d.data());
        compute_ray_aabb_inv_kernel<<<gsR, BS>>>(
            (uint32_t)R, d_o.data(), d_d.data(), t.aabbMin, t.aabbMax, t.numCascades,
            d_dinv.data(), d_near.data(), d_far.data());

        uint32_t totalHits = 0;
        int activeRays = processRaysHitLinear(
            (uint32_t)R, d_o.data(), d_d.data(), d_dinv.data(), d_near.data(), d_far.data(),
            m_occupancyGrid.data(), t.gridResolution, t.aabbMin, t.aabbMax,
            t.numCascades, t.levelsMipmap, t.samplesPerVoxel,
            hitCap, &totalHits, d_active.data(),
            d_steps.data(), d_offs.data(), d_marchPos.data(), d_rayIdx.data(), d_t.data(),
            d_bsums.data(), 0);
        if (activeRays <= 0 || totalHits == 0) continue;
        int padHits = (int)((totalHits + 15) & ~15);

        // marcher only wrote [0, totalHits); pads must be valid ray ids for the SH gather
        if (padHits > (int)totalHits)
            cudaMemsetAsync(d_rayIdx.data() + totalHits, 0, ((size_t)padHits - totalHits) * sizeof(uint32_t));
        cudaMemsetAsync(d_w.data(), 0, (size_t)padHits * sizeof(float));

        int gsH = (padHits + BS - 1) / BS;
        k_gatherSamples<<<gsH, BS>>>(
            (int)totalHits, padHits, d_o.data(), d_d.data(), d_rayIdx.data(), d_t.data(),
            m_reverseMap.data(), m_voxelMask.data(), m_blockPayloadOffset.data(),
            m_voxelSigma.data(), t.gridResolution, t.numCascades, t.aabbMin, t.aabbMax,
            d_row.data(), d_sig.data(), d_cpos.data());

        int gsA = (activeRays + BS - 1) / BS;
        k_computeWeights<<<gsA, BS>>>(activeRays, d_offs.data(), d_steps.data(),
                                      d_t.data(), d_sig.data(), dt0, d_w.data());

        teacher.queryRadiance(d_cpos.data(), padHits, d_d.data(), d_rayIdx.data(), d_trgb.data());

        k_buildDeferredInput<<<gsH, BS>>>(
            (int)totalHits, padHits, d_row.data(), table.masterWeights(), K,
            d_rayIdx.data(), d_d.data(), d_in.data());

        m_deferredMLP->zero_grad();
        m_deferredMLP->forward(d_in.data(), d_gout.data(), padHits);

        cudaMemsetAsync(d_loss.data(), 0, 2 * sizeof(float));
        k_lossGrad<<<gsH, BS>>>(
            (int)totalHits, padHits,
            d_gout.data(), d_trgb.data(), d_w.data(), LOSS_SCALE,
            d_dldg.data(), d_loss.data());

        m_deferredMLP->backward(d_dldg.data(), d_dx.data(), padHits);

        k_emitTableGrads<<<gsH, BS>>>(
            (int)totalHits, padHits, d_row.data(), d_dx.data(), inPad, K,
            1.0f / LOSS_SCALE, d_pair.data());

        table.tableStep(d_row.data(), d_pair.data(), padHits);
        m_deferredMLP->step(m_opts.mlpLearningRate, 0.9f, 0.999f, 1e-8f, LOSS_SCALE);

        if (step == 0 || (step + 1) % 50 == 0 || step + 1 == m_opts.fineTuneSteps) {
            float h[2];
            cudaMemcpy(h, d_loss.data(), 2 * sizeof(float), cudaMemcpyDeviceToHost);
            float mse = (h[1] > 0.0f) ? h[0] / (3.0f * h[1]) : 0.0f;
            printf("[fit] step %5d  hits %8u  rays %5d  weighted mse %.5e  psnr(pt) %.2f dB\n",
                   step + 1, totalHits, activeRays, mse, mse > 0 ? -10.0f * log10f(mse) : 99.0f);
        }
    }

    // ---- write-back: quantize masters into the shipped payload (u8 diffuse, fp16 feat)
    m_voxelDiffuse = DeviceBuffer<uint8_t>((size_t)N * 3);
    m_voxelFeat    = DeviceBuffer<half>((size_t)N * K);
    int gsN = (int)((N + BS - 1) / BS);
    k_writeBackPayload<<<gsN, BS>>>(table.masterWeights(), N, K,
                                    m_voxelDiffuse.data(), m_voxelFeat.data());
    cudaDeviceSynchronize();
    m_deferredTrained = true;
    printf("[fit] done: payload written (%u rows, %.1f MB diffuse+feat)\n",
           N, N * (3.0 + 2.0 * K) / (1024.0 * 1024.0));
}

static constexpr int kRenderChunkRays = 4096;
static constexpr int kRenderHitCap    = (1 << 20) - 16;

void BakedNerf::allocRenderScratch() {
    if (m_scratchReady) return;
    const int K = m_opts.viewFeatures, F = 3 + K;
    m_scratchRays = kRenderChunkRays;

    d_rays_d_inv        = DeviceBuffer<float3>(kRenderChunkRays);
    d_nears             = DeviceBuffer<float>(kRenderChunkRays);
    d_fars              = DeviceBuffer<float>(kRenderChunkRays);
    d_num_steps         = DeviceBuffer<uint32_t>(kRenderChunkRays);
    d_ray_offsets       = DeviceBuffer<uint32_t>(kRenderChunkRays);
    d_block_sums        = DeviceBuffer<uint32_t>((kRenderChunkRays + 1023) / 1024);
    d_active_rays_count = DeviceBuffer<uint32_t>(1);

    d_positions   = DeviceBuffer<float>((size_t)kRenderHitCap * 4);   // marcher writes float4/hit
    d_t_sorted    = DeviceBuffer<float>(kRenderHitCap);
    d_ray_indices = DeviceBuffer<uint32_t>(kRenderHitCap);
    d_sigma       = DeviceBuffer<float>(kRenderHitCap);
    d_feat        = DeviceBuffer<float>((size_t)kRenderHitCap * F);   // per-hit [diffuse3, featK]

    d_acc_feat    = DeviceBuffer<float>((size_t)kRenderChunkRays * F);
    d_T_pix       = DeviceBuffer<float>(kRenderChunkRays);
    d_deferred_in = DeviceBuffer<half>((size_t)kRenderChunkRays * deferredInDim());
    d_specular    = DeviceBuffer<float>((size_t)kRenderChunkRays * 3);
    m_scratchReady = true;
}

// Per-pixel deferred render of the shipped payload (u8 diffuse + fp16 feat + fp16 sigma).
// Standalone: no teacher needed. Rays are processed in chunks; within a chunk the marcher
// sub-batches rays so total hits stay under kRenderHitCap.
void BakedNerf::renderImage(
    const float3* d_rays_o, const float3* d_rays_d, uint32_t numRays,
    float* d_rgb_out, cudaStream_t stream)
{
    if (!m_deferredTrained) {
        fprintf(stderr, "[BakedNerf] renderImage before fitDeferred — no appearance payload\n");
        return;
    }
    if (!d_rgb_out || numRays == 0) return;
    allocRenderScratch();

    const NerfOptions& t = m_teacherOpts;
    const int K = m_opts.viewFeatures;
    constexpr int BS = 256;
    const float dt0 = (t.aabbMax.x - t.aabbMin.x) / (float)t.gridResolution.x
                      / (float)std::max(1, t.samplesPerVoxel);

    uint32_t raysDone = 0;
    while (raysDone < numRays) {
        int chunk = (int)std::min((uint32_t)kRenderChunkRays, numRays - raysDone);
        const float3* co = d_rays_o + raysDone;
        const float3* cd = d_rays_d + raysDone;

        int gsC = (chunk + BS - 1) / BS;
        compute_ray_aabb_inv_kernel<<<gsC, BS, 0, stream>>>(
            (uint32_t)chunk, co, cd, t.aabbMin, t.aabbMax, t.numCascades,
            d_rays_d_inv.data(), d_nears.data(), d_fars.data());

        int innerDone = 0;
        while (innerDone < chunk) {
            int rem = chunk - innerDone;
            uint32_t totalHits = 0;
            int k = processRaysHitLinear(
                (uint32_t)rem, co + innerDone, cd + innerDone,
                d_rays_d_inv.data() + innerDone, d_nears.data() + innerDone, d_fars.data() + innerDone,
                m_occupancyGrid.data(), t.gridResolution, t.aabbMin, t.aabbMax,
                t.numCascades, t.levelsMipmap, t.samplesPerVoxel,
                kRenderHitCap, &totalHits, d_active_rays_count.data(),
                d_num_steps.data(), d_ray_offsets.data(), d_positions.data(),
                d_ray_indices.data(), d_t_sorted.data(), d_block_sums.data(), stream);
            if (k <= 0) {                       // defensive; cannot happen with MAX_HITS < cap
                fprintf(stderr, "[BakedNerf] renderImage: marcher made no progress, bg-filling\n");
                k = rem; totalHits = 0;
                cudaMemsetAsync(d_num_steps.data(), 0, (size_t)rem * sizeof(uint32_t), stream);
            }

            if (totalHits > 0) {
                int gsH = ((int)totalHits + BS - 1) / BS;
                k_renderGatherHit<<<gsH, BS, 0, stream>>>(
                    (int)totalHits, co + innerDone, cd + innerDone,
                    d_ray_indices.data(), d_t_sorted.data(),
                    m_reverseMap.data(), m_voxelMask.data(), m_blockPayloadOffset.data(),
                    m_voxelSigma.data(), m_voxelDiffuse.data(), m_voxelFeat.data(), K,
                    t.gridResolution, t.numCascades, t.aabbMin, t.aabbMax,
                    d_sigma.data(), d_feat.data());
            }

            int gsK = (k + BS - 1) / BS;
            k_compositeDeferred<<<gsK, BS, 0, stream>>>(
                k, d_ray_offsets.data(), d_num_steps.data(), d_t_sorted.data(),
                d_sigma.data(), d_feat.data(), K, dt0,
                d_acc_feat.data(), d_T_pix.data());

            int paddedK = (k + 15) & ~15;
            int gsP = (paddedK + BS - 1) / BS;
            k_buildPixelInput<<<gsP, BS, 0, stream>>>(
                k, paddedK, d_acc_feat.data(), K, d_T_pix.data(),
                cd + innerDone, d_deferred_in.data());

            m_deferredMLP->inference(d_deferred_in.data(), d_specular.data(), paddedK, stream);

            k_finalizePixel<<<gsK, BS, 0, stream>>>(
                k, d_specular.data(), d_T_pix.data(),
                t.bgColor, d_rgb_out + (size_t)(raysDone + innerDone) * 3);

            innerDone += k;
        }
        raysDone += (uint32_t)chunk;
    }
}

// Diagnostic render: teacher radiance at every marched sample x baked sigma weights.
// Needs only the structure + sigma payload (works with fineTuneSteps = 0).
void BakedNerf::renderImageTeacherColor(
    InstantNerf& teacher, const float3* d_rays_o, const float3* d_rays_d,
    uint32_t numRays, float* d_rgb_out)
{
    if (m_voxelSigma.size() == 0) {
        fprintf(stderr, "[BakedNerf] renderImageTeacherColor before distil\n");
        return;
    }
    if (!d_rgb_out || numRays == 0) return;
    allocRenderScratch();

    const NerfOptions& t = m_teacherOpts;
    constexpr int BS = 256;
    const float dt0 = (t.aabbMax.x - t.aabbMin.x) / (float)t.gridResolution.x
                      / (float)std::max(1, t.samplesPerVoxel);

    DeviceBuffer<int>   d_row(kRenderHitCap);
    DeviceBuffer<float> d_cpos(((size_t)kRenderHitCap + kQueryPad) * 3);
    DeviceBuffer<float> d_trgb(((size_t)kRenderHitCap + kQueryPad) * 3);

    uint32_t raysDone = 0;
    while (raysDone < numRays) {
        int chunk = (int)std::min((uint32_t)kRenderChunkRays, numRays - raysDone);
        const float3* co = d_rays_o + raysDone;
        const float3* cd = d_rays_d + raysDone;

        int gsC = (chunk + BS - 1) / BS;
        compute_ray_aabb_inv_kernel<<<gsC, BS>>>(
            (uint32_t)chunk, co, cd, t.aabbMin, t.aabbMax, t.numCascades,
            d_rays_d_inv.data(), d_nears.data(), d_fars.data());

        int innerDone = 0;
        while (innerDone < chunk) {
            int rem = chunk - innerDone;
            uint32_t totalHits = 0;
            int k = processRaysHitLinear(
                (uint32_t)rem, co + innerDone, cd + innerDone,
                d_rays_d_inv.data() + innerDone, d_nears.data() + innerDone, d_fars.data() + innerDone,
                m_occupancyGrid.data(), t.gridResolution, t.aabbMin, t.aabbMax,
                t.numCascades, t.levelsMipmap, t.samplesPerVoxel,
                kRenderHitCap, &totalHits, d_active_rays_count.data(),
                d_num_steps.data(), d_ray_offsets.data(), d_positions.data(),
                d_ray_indices.data(), d_t_sorted.data(), d_block_sums.data(), 0);
            if (k <= 0) {
                k = rem; totalHits = 0;
                cudaMemsetAsync(d_num_steps.data(), 0, (size_t)rem * sizeof(uint32_t));
            }

            if (totalHits > 0) {
                int padHits = (int)((totalHits + 15) & ~15);
                if (padHits > (int)totalHits)
                    cudaMemsetAsync(d_ray_indices.data() + totalHits, 0,
                                    ((size_t)padHits - totalHits) * sizeof(uint32_t));
                int gsH = (padHits + BS - 1) / BS;
                k_gatherSamples<<<gsH, BS>>>(
                    (int)totalHits, padHits, co + innerDone, cd + innerDone,
                    d_ray_indices.data(), d_t_sorted.data(),
                    m_reverseMap.data(), m_voxelMask.data(), m_blockPayloadOffset.data(),
                    m_voxelSigma.data(), t.gridResolution, t.numCascades, t.aabbMin, t.aabbMax,
                    d_row.data(), d_sigma.data(), d_cpos.data());
                teacher.queryRadiance(d_cpos.data(), padHits, cd + innerDone,
                                      d_ray_indices.data(), d_trgb.data());
            }

            int gsK = (k + BS - 1) / BS;
            k_compositeDeferred<<<gsK, BS>>>(
                k, d_ray_offsets.data(), d_num_steps.data(), d_t_sorted.data(),
                d_sigma.data(), d_trgb.data(), 0 /*K: color only*/, dt0,
                d_acc_feat.data(), d_T_pix.data());
            k_finalizeComposited<<<gsK, BS>>>(
                k, d_acc_feat.data(), d_T_pix.data(), t.bgColor,
                d_rgb_out + (size_t)(raysDone + innerDone) * 3);

            innerDone += k;
        }
        raysDone += (uint32_t)chunk;
    }
    cudaDeviceSynchronize();
}
