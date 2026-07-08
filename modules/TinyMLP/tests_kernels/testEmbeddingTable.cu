// EmbeddingTable validation + benchmark (spec: docs/embedding_table_design.md §14).
//   1. step-exactness : tableStep vs a CPU replica of reduce-duplicates -> row-wise
//                       Adagrad (duplicate rows, -1 padding, per-feature lr, partial numPairs)
//   2. convergence    : K=1 nearest fit of a random target table under 81x imbalanced
//                       sampling -- the row-shared normalizer must absorb the imbalance
//   3. benchmark      : ms/step at bake scale (40M rows, F=7, 2M pairs) for uniform /
//                       clustered / skewed row distributions
#include "../EmbeddingTable.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

// grad of 0.5*(w - t)^2 per hit; duplicates of a row sum inside tableStep
__global__ void computeFitGrads(
    const int*   __restrict__ d_rowid,
    const float* __restrict__ d_weights,
    const float* __restrict__ d_target,
    float*       __restrict__ d_grads,
    int n, int F)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int r = d_rowid[i];
    for (int f = 0; f < F; f++)
        d_grads[i * F + f] = d_weights[r * F + f] - d_target[r * F + f];
}

// ------------------------------------------------------------------ test 1
// Exact per-step semantics vs a CPU replica.
static bool testStepExactness() {
    const uint32_t R = 1000;
    const int F = 7, P = 4096;
    std::vector<float> lr(F);
    for (int f = 0; f < F; f++) lr[f] = 0.01f * (f + 1);   // per-feature lr exercised

    EmbeddingTableOption opt;
    opt.rows = R; opt.num_features = F; opt.lr = lr.data();
    EmbeddingTable table(opt, P);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> rowDist(0, R - 1);
    std::uniform_real_distribution<float> gDist(-1.f, 1.f);

    std::vector<float> W(R * F, 0.f), S(R, 0.f);   // CPU replica state
    int* d_rowid;  float* d_grads;
    CUDA_CHECK(cudaMalloc(&d_rowid, sizeof(int) * P));
    CUDA_CHECK(cudaMalloc(&d_grads, sizeof(float) * P * F));

    const int stepPairs[3] = { P, P / 2, P };   // middle step: numPairs < batchSize
    for (int step = 0; step < 3; step++) {
        int n = stepPairs[step];
        std::vector<int> rowid(n);
        std::vector<float> grads((size_t)n * F);
        for (int i = 0; i < n; i++)
            rowid[i] = (rng() % 10 == 0) ? -1 : rowDist(rng);   // ~10% padding sentinels
        for (auto& g : grads) g = gDist(rng);

        CUDA_CHECK(cudaMemcpy(d_rowid, rowid.data(), sizeof(int) * n, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_grads, grads.data(), sizeof(float) * n * F, cudaMemcpyHostToDevice));
        table.tableStep(d_rowid, d_grads, n);

        // CPU: reduce duplicate rows, then row-wise Adagrad on touched rows only
        std::vector<float> G(R * F, 0.f);
        std::vector<char> touched(R, 0);
        for (int i = 0; i < n; i++) {
            if (rowid[i] < 0) continue;
            touched[rowid[i]] = 1;
            for (int f = 0; f < F; f++) G[rowid[i] * F + f] += grads[(size_t)i * F + f];
        }
        for (uint32_t r = 0; r < R; r++) {
            if (!touched[r]) continue;
            float sq = 0.f;
            for (int f = 0; f < F; f++) sq += G[r * F + f] * G[r * F + f];
            S[r] += sq / F;
            float norm = 1.f / (sqrtf(S[r]) + opt.eps);
            for (int f = 0; f < F; f++) W[r * F + f] -= lr[f] * G[r * F + f] * norm;
        }
    }

    std::vector<float> Wgpu(R * F);
    CUDA_CHECK(cudaMemcpy(Wgpu.data(), table.masterWeights(), sizeof(float) * R * F,
                          cudaMemcpyDeviceToHost));
    float maxDiff = 0.f;
    for (size_t i = 0; i < Wgpu.size(); i++)
        maxDiff = std::max(maxDiff, std::fabs(Wgpu[i] - W[i]) / (1.f + std::fabs(W[i])));

    CUDA_CHECK(cudaFree(d_rowid));
    CUDA_CHECK(cudaFree(d_grads));

    bool pass = maxDiff < 1e-5f;
    printf("[1] step-exactness vs CPU (3 steps, dup rows, -1 padding, partial batch)\n"
           "    max rel diff %.3e  -> %s\n", maxDiff, pass ? "PASS" : "FAIL");
    return pass;
}

// ------------------------------------------------------------------ test 2
// K=1 nearest fit under 81x sampling imbalance (90% of pairs in 10% of rows).
static bool testConvergenceImbalanced() {
    const uint32_t R = 65536;
    const int F = 3, P = 131072, STEPS = 300;
    std::vector<float> lr(F, 0.25f);

    EmbeddingTableOption opt;
    opt.rows = R; opt.num_features = F; opt.lr = lr.data();
    EmbeddingTable table(opt, P);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> tDist(0.25f, 0.75f);
    std::vector<float> target(R * F);
    for (auto& t : target) t = tDist(rng);

    float* d_target;  int* d_rowid;  float* d_grads;
    CUDA_CHECK(cudaMalloc(&d_target, sizeof(float) * R * F));
    CUDA_CHECK(cudaMalloc(&d_rowid,  sizeof(int) * P));
    CUDA_CHECK(cudaMalloc(&d_grads,  sizeof(float) * P * F));
    CUDA_CHECK(cudaMemcpy(d_target, target.data(), sizeof(float) * R * F,
                          cudaMemcpyHostToDevice));

    std::uniform_int_distribution<int> hot(0, R / 10 - 1), cold(R / 10, R - 1);
    std::uniform_real_distribution<float> u01(0.f, 1.f);
    std::vector<int> rowid(P);

    float mse = 1e30f;
    std::vector<float> Wgpu(R * F);
    constexpr int BS = 256;
    for (int step = 0; step < STEPS; step++) {
        for (int i = 0; i < P; i++)
            rowid[i] = (u01(rng) < 0.9f) ? hot(rng) : cold(rng);
        CUDA_CHECK(cudaMemcpy(d_rowid, rowid.data(), sizeof(int) * P, cudaMemcpyHostToDevice));

        computeFitGrads<<<(P + BS - 1) / BS, BS>>>(d_rowid, table.masterWeights(),
                                                   d_target, d_grads, P, F);
        table.tableStep(d_rowid, d_grads, P);

        if ((step + 1) % 50 == 0) {
            CUDA_CHECK(cudaMemcpy(Wgpu.data(), table.masterWeights(), sizeof(float) * R * F,
                                  cudaMemcpyDeviceToHost));
            double acc = 0.0;
            for (size_t i = 0; i < Wgpu.size(); i++) {
                double d = (double)Wgpu[i] - target[i];
                acc += d * d;
            }
            mse = (float)(acc / Wgpu.size());
            printf("    step %4d  mse %.3e\n", step + 1, mse);
        }
    }

    CUDA_CHECK(cudaFree(d_target));
    CUDA_CHECK(cudaFree(d_rowid));
    CUDA_CHECK(cudaFree(d_grads));

    bool pass = mse < 1e-6f;
    printf("[2] K=1 fit, 81x imbalanced sampling, %d steps -> mse %.3e  %s\n",
           STEPS, mse, pass ? "PASS" : "FAIL");
    return pass;
}

// ------------------------------------------------------------------ bench
static size_t hostUnique(std::vector<int> v) {
    std::sort(v.begin(), v.end());
    return std::unique(v.begin(), v.end()) - v.begin();
}

static void benchScenario(EmbeddingTable& table, const char* name,
                          const std::vector<int>& rowid, const float* d_grads,
                          int* d_rowid, int P) {
    CUDA_CHECK(cudaMemcpy(d_rowid, rowid.data(), sizeof(int) * P, cudaMemcpyHostToDevice));

    for (int i = 0; i < 5; i++) table.tableStep(d_rowid, d_grads, P);   // warmup

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    const int ITERS = 50;
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < ITERS; i++) table.tableStep(d_rowid, d_grads, P);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    ms /= ITERS;

    printf("    %-10s: uniques %7zuk   %.3f ms/step   %.0f Mpairs/s\n",
           name, hostUnique(rowid) / 1000, ms, P / (ms * 1e3));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
}

static void benchmark() {
    const uint32_t R = 40000000;   // bicycle-scale sub-voxel count
    const int F = 7, P = 2000000;  // 256k samples x 8 trilinear taps
    std::vector<float> lr(F, 0.01f);

    EmbeddingTableOption opt;
    opt.rows = R; opt.num_features = F; opt.lr = lr.data();

    size_t freeBefore, freeAfter, total;
    CUDA_CHECK(cudaMemGetInfo(&freeBefore, &total));
    EmbeddingTable table(opt, P);
    CUDA_CHECK(cudaMemGetInfo(&freeAfter, &total));

    printf("[3] benchmark: rows %u  F %d  pairs/step %d   table+scratch %.2f GB\n",
           R, F, P, (freeBefore - freeAfter) / (1024.0 * 1024.0 * 1024.0));

    int* d_rowid;  float* d_grads;
    CUDA_CHECK(cudaMalloc(&d_rowid, sizeof(int) * P));
    CUDA_CHECK(cudaMalloc(&d_grads, sizeof(float) * (size_t)P * F));

    std::mt19937 rng(7);
    std::vector<float> grads((size_t)P * F);
    std::uniform_real_distribution<float> gDist(-1.f, 1.f);
    for (auto& g : grads) g = gDist(rng);
    CUDA_CHECK(cudaMemcpy(d_grads, grads.data(), sizeof(float) * (size_t)P * F,
                          cudaMemcpyHostToDevice));

    std::uniform_int_distribution<int> rowDist(0, R - 1);
    std::vector<int> rowid(P);

    // uniform: pairs spread over the whole table (~95% unique -> runs of ~1)
    for (auto& r : rowid) r = rowDist(rng);
    benchScenario(table, "uniform", rowid, d_grads, d_rowid, P);

    // clustered: pairs drawn from a 250k-row working set (runs of ~8, like
    // neighbouring samples sharing trilinear corners)
    std::vector<int> subset(250000);
    for (auto& s : subset) s = rowDist(rng);
    std::uniform_int_distribution<int> subDist(0, (int)subset.size() - 1);
    for (auto& r : rowid) r = subset[subDist(rng)];
    benchScenario(table, "clustered", rowid, d_grads, d_rowid, P);

    // skewed: 20% of pairs on one hot row (400k-long run -> warp straggler stress)
    std::uniform_real_distribution<float> u01(0.f, 1.f);
    for (auto& r : rowid) r = (u01(rng) < 0.2f) ? 42 : rowDist(rng);
    benchScenario(table, "skewed", rowid, d_grads, d_rowid, P);

    CUDA_CHECK(cudaFree(d_rowid));
    CUDA_CHECK(cudaFree(d_grads));
}

int main() {
    int failed = 0;
    if (!testStepExactness())        failed++;
    if (!testConvergenceImbalanced()) failed++;
    benchmark();
    printf("\n%s (%d failed)\n", failed == 0 ? "ALL PASS" : "FAILURES", failed);
    return failed;
}
