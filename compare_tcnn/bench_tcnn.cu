// ============================================================================
// TinyMLP vs tiny-cuda-nn — head-to-head in ONE process, same harness, same stream.
//
//   bench_tcnn.exe [--speed speed.csv] [--conv conv.csv] [--ablate ablate.csv] [--quick]
//
// SPEED  (speed.csv: framework,model,op,hidden,layers,batch,ms)
//   framework = tinymlp | tcnn (AOT FullyFusedMLP) | tcnn_jit (runtime-fused, tcnn >= 2.0)
//   model     = mlp        in=H, out=H, ReLU, no output act   (square, like benchCompare.cu)
//               hashgrid   HashGrid(16 lvl, F=2, 2^19, base 16, 1.38) -> MLP -> 16 outputs
//               nerf_color in=32, out=3, H=32, sigmoid        (3DMaker's colour head)
//   op        = inference | backward | train (fwd + L2 loss + bwd + Adam)
//   layers    = number of weight matrices (TinyMLP numLayers); tcnn n_hidden_layers = layers-1
//   ms        = median of 3 timed runs, each the mean over N launches (CUDA events)
//
// CONV   (conv.csv: framework,step,test_mse,train_ms)
//   Same hash-grid model (H=64, 2 layers) fitting a 16-channel multi-frequency sinusoid field,
//   identical data stream for both, Adam lr 1e-2 / 0.9 / 0.99 / eps 1e-15, no weight decay.
//   Checks that TinyMLP learns as well per step, not just faster.
//
// ABLATE (ablate.csv: target,framework,step,test_mse,train_ms)
//   Why does TinyMLP reach lower error? Same CONV setup with TinyMLP biases frozen at zero, then
//   additionally tcnn-style init, on the +0.5-offset target and on a zero-mean target.
//
// Parity notes: both fp16 compute with fp32 master weights + Adam; tcnn FullyFusedMLP has no
// biases (TinyMLP does); each library uses its own default loss scale (tcnn 128, TinyMLP 65536).
// Each library is driven its fastest intended way: TinyMLP hash grid takes 4-float positions
// (no pad kernel) and its train step passes outputs=nullptr (no copy-out, like tcnn's
// training_step). The convergence/ablation runs keep 3-float positions (quality only).
// ============================================================================
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/trainer.h>

#include "TinyMLP.h"
// Bench-only access to TinyMLPHashGrid internals so the bias ablation can run Adam on weights +
// hash table while skipping biases. The library itself is not modified.
#define private public
#include "TinyMLPHashGrid.h"
#undef private

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <vector>

using tcnn::json;
using tcnn::GPUMatrix;
using tcnn::GPUMatrixDynamic;
using tcnn::MatrixLayout;
using P = tcnn::network_precision_t;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

// ---------------------------------------------------------------- data kernels
__device__ __forceinline__ uint32_t pcg(uint32_t v) { uint32_t s = v * 747796405u + 2891336453u; uint32_t w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }
__device__ __forceinline__ float pcgf(uint32_t i, uint32_t sd) { return __int_as_float((pcg(sd ^ pcg(i)) >> 9) | 0x3f800000u) - 1.0f; }
__global__ void fillF01(float* d, size_t n, uint32_t sd) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) d[i] = pcgf((uint32_t)i, sd); }
__global__ void fillH01(half* d, size_t n, uint32_t sd) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) d[i] = __float2half(pcgf((uint32_t)i, sd)); }
__global__ void fillF(float* d, size_t n, uint32_t sd) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) d[i] = pcgf((uint32_t)i, sd); }
__global__ void f2h(const float* s, half* d, size_t n) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) d[i] = __float2half(s[i]); }
static unsigned nb(size_t n) { return (unsigned)((n + 255) / 256); }

// 16-channel target field: sinusoids of rising frequency along hashed directions.
__global__ void targetField(const float* pos, float* tgt, int B, float offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;
    float x = pos[i * 3 + 0], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
    for (int c = 0; c < 16; ++c) {
        float a = pcgf(c, 11u) * 2.f - 1.f, b = pcgf(c, 22u) * 2.f - 1.f, g = pcgf(c, 33u) * 2.f - 1.f;
        float inv = rsqrtf(a * a + b * b + g * g + 1e-6f);
        float freq = 1.0f + 2.0f * c;                        // 1 .. 31 cycles across the unit cube
        float ph = pcgf(c, 44u) * 6.2831853f;
        tgt[i * 16 + c] = offset + 0.4f * __sinf(6.2831853f * freq * (a * x + b * y + g * z) * inv + ph);
    }
}
__global__ void mseKernel(const float* pred, int predStride, const float* tgt, int B, float* acc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float s = 0.f;
    if (i < B) for (int c = 0; c < 16; ++c) { float d = pred[(size_t)i * predStride + c] - tgt[(size_t)i * 16 + c]; s += d * d; }
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(acc, s);
}

// ---------------------------------------------------------------- timing
static cudaStream_t g_s;
static cudaEvent_t g_e0, g_e1;
static bool g_quick = false;

template <class F> static float timeMs(F f, int batch) {
    int warm = 8, iters = batch >= (1 << 20) ? 15 : (batch >= (1 << 18) ? 30 : 60);
    if (g_quick) { warm = 3; iters = 5; }
    for (int i = 0; i < warm; ++i) f();
    CK(cudaStreamSynchronize(g_s));
    std::vector<float> runs;
    for (int r = 0; r < 3; ++r) {
        CK(cudaEventRecord(g_e0, g_s));
        for (int i = 0; i < iters; ++i) f();
        CK(cudaEventRecord(g_e1, g_s));
        CK(cudaEventSynchronize(g_e1));
        float ms = 0; CK(cudaEventElapsedTime(&ms, g_e0, g_e1));
        runs.push_back(ms / iters);
    }
    std::sort(runs.begin(), runs.end());
    return runs[1];
}

static FILE* g_speed = nullptr;
static void emit(const char* fw, const char* model, const char* op, int H, int L, int B, float ms) {
    fprintf(g_speed, "%s,%s,%s,%d,%d,%d,%.5f\n", fw, model, op, H, L, B, ms);
    fflush(g_speed);
    fprintf(stderr, "  %-9s %-10s %-9s H=%-3d L=%d B=%-7d %9.3f ms\n", fw, model, op, H, L, B, ms);
}

// ---------------------------------------------------------------- model descriptions
struct Case {
    const char* model;  // mlp | hashgrid | nerf_color
    int in, out, H, L;
    bool sigmoid, hash;
};

static json tcnnNet(const Case& c) {
    return {{"otype", "FullyFusedMLP"}, {"activation", "ReLU"},
            {"output_activation", c.sigmoid ? "Sigmoid" : "None"},
            {"n_neurons", c.H}, {"n_hidden_layers", c.L - 1},
            {"n_input_dims", c.hash ? 32 : c.in}, {"n_output_dims", c.out}};
}
static json tcnnEnc() {
    return {{"otype", "HashGrid"}, {"n_levels", 16}, {"n_features_per_level", 2},
            {"log2_hashmap_size", 19}, {"base_resolution", 16}, {"per_level_scale", 1.38}};
}
static json tcnnAdam(float lr, float b2, float eps) {
    return {{"otype", "Adam"}, {"learning_rate", lr}, {"beta1", 0.9}, {"beta2", b2},
            {"epsilon", eps}, {"l2_reg", 0.0}};
}

// ---------------------------------------------------------------- TinyMLP side
static void benchTinyMLP(const Case& c, int B, void* d_in, half* d_tgt_h, float* d_out) {
    if (!c.hash) {
        MLPOption o{}; o.inputDim = c.in; o.hiddenDim = c.H; o.outputDim = c.out; o.numLayers = c.L;
        o.activationType = ACT_RELU; o.outputActivation = c.sigmoid ? OUT_ACT_SIGMOID : OUT_ACT_NONE;
        const half* in = (const half*)d_in;
        { TinyMLP net(o, B, B, 42, false);
          emit("tinymlp", c.model, "inference", c.H, c.L, B, timeMs([&] { net.inference(in, d_out, B, g_s); }, B)); }
        { TinyMLP net(o, B, 0, 42, true);
          net.forward(in, d_out, B, g_s); net.calculate_loss_and_grad(d_tgt_h, B, 65536.f, false, g_s);
          emit("tinymlp", c.model, "backward", c.H, c.L, B, timeMs([&] { net.backward(B, g_s); }, B));
          emit("tinymlp", c.model, "train", c.H, c.L, B, timeMs([&] {
              net.zero_grad(g_s); net.forward(in, nullptr, B, g_s);   // outputs not needed to train
              net.calculate_loss_and_grad(d_tgt_h, B, 65536.f, false, g_s);
              net.backward(B, g_s); net.step(1e-3f, 0.9f, 0.999f, 1e-8f, 65536.f, g_s); }, B)); }
    } else {
        // vectorDim = 4: positions as (x,y,z,pad), exactly how 3DMaker feeds the density head,
        // so TinyMLP skips its 3->4 pad kernel. tcnn reads the same buffer as 3-float positions.
        MLPGridOptions o{}; o.vectorDim = 4; o.hiddenDim = c.H; o.outputDim = c.out; o.numLayers = c.L;
        o.activationType = ACT_RELU; o.tableSize = 1 << 19; o.numLevels = 16; o.b = 1.38f; o.lowestSize = 16; o.featuresLevel = 2;
        const float* in = (const float*)d_in;
        { TinyMLPHashGrid net(o, B, B, 42, false);
          emit("tinymlp", c.model, "inference", c.H, c.L, B, timeMs([&] { net.inference(in, d_out, B, g_s); }, B)); }
        { TinyMLPHashGrid net(o, B, 0, 42, true);
          net.forward(in, d_out, B, g_s); net.calculate_loss_and_grad(d_tgt_h, B, 65536.f, false, g_s);
          emit("tinymlp", c.model, "backward", c.H, c.L, B, timeMs([&] { net.backward(B, g_s); }, B));
          emit("tinymlp", c.model, "train", c.H, c.L, B, timeMs([&] {
              net.zero_grad(g_s); net.forward(in, nullptr, B, g_s);   // outputs not needed to train
              net.calculate_loss_and_grad(d_tgt_h, B, 65536.f, false, g_s);
              net.backward(B, g_s); net.step(1e-3f, 0.9f, 0.999f, 1e-8f, 65536.f, g_s); }, B)); }
    }
    CK(cudaStreamSynchronize(g_s));
}

// ---------------------------------------------------------------- tiny-cuda-nn side
template <typename TIn>
static void benchTcnnImpl(const Case& c, int B, TIn* d_in, float* d_tgt_f, bool jit,
                          std::shared_ptr<tcnn::DifferentiableObject<TIn, P, P>> model) {
    const char* fw = jit ? "tcnn_jit" : "tcnn";
    std::shared_ptr<tcnn::Loss<P>> loss{tcnn::create_loss<P>({{"otype", "L2"}})};
    std::shared_ptr<tcnn::Optimizer<P>> opt{tcnn::create_optimizer<P>(tcnnAdam(1e-3f, 0.999f, 1e-8f))};
    auto trainer = std::make_shared<tcnn::Trainer<TIn, P, P>>(model, opt, loss);

    // JIT prefers feature-major (RM) data; AOT prefers sample-major (CM). Contents are i.i.d.
    // uniform, so reinterpreting the same buffer under either layout is statistically identical.
    MatrixLayout lay = jit ? tcnn::RM : tcnn::CM;
    GPUMatrixDynamic<TIn> in{d_in, (uint32_t)(c.hash ? 3 : c.in), (uint32_t)B, lay};
    GPUMatrix<float> tgt{d_tgt_f, (uint32_t)c.out, (uint32_t)B};
    // float inference() writes the unpadded width (3 for the colour head), like TinyMLP does
    GPUMatrixDynamic<float> out{model->output_width(), (uint32_t)B, g_s, lay};

    if (jit) {
        try { model->set_jit_fusion(true); }
        catch (const std::exception& e) { fprintf(stderr, "  [tcnn_jit] unsupported: %s\n", e.what()); return; }
    }

    float inf = timeMs([&] { model->inference(g_s, in, out); }, B);
    float trn = timeMs([&] { trainer->training_step(g_s, in, tgt); }, B);
    if (jit && !model->jit_fusion()) {   // tcnn silently falls back to AOT when RTC fails
        fprintf(stderr, "  [tcnn_jit] JIT compilation failed / disabled -> not reported\n");
        return;
    }
    emit(fw, c.model, "inference", c.H, c.L, B, inf);
    if (!jit) {  // JIT only fuses the whole training step; there is no standalone JIT backward
        auto ctx = trainer->forward(g_s, tcnn::default_loss_scale<P>(), in, tgt);
        emit(fw, c.model, "backward", c.H, c.L, B, timeMs([&] { trainer->backward(g_s, *ctx, in); }, B));
    }
    emit(fw, c.model, "train", c.H, c.L, B, trn);
    CK(cudaStreamSynchronize(g_s));
}

static void benchTcnn(const Case& c, int B, void* d_in, float* d_tgt_f, bool jit) {
    try {
        if (c.hash) {
            auto m = std::make_shared<tcnn::NetworkWithInputEncoding<P>>(3u, (uint32_t)c.out, tcnnEnc(), tcnnNet(c));
            benchTcnnImpl<float>(c, B, (float*)d_in, d_tgt_f, jit, m);
        } else {
            std::shared_ptr<tcnn::Network<P>> m{tcnn::create_network<P>(tcnnNet(c))};
            benchTcnnImpl<P>(c, B, (P*)d_in, d_tgt_f, jit, m);
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "  [%s] %s H=%d L=%d B=%d failed: %s\n", jit ? "tcnn_jit" : "tcnn", c.model, c.H, c.L, B, e.what());
    }
    tcnn::free_all_gpu_memory_arenas();
    CK(cudaDeviceSynchronize());
}

static void runCase(const Case& c, int B) {
    size_t nIn = (size_t)B * (c.hash ? 4 : c.in), nOut = (size_t)B * std::max(c.out, 16);
    void* d_in; float *d_tgt_f, *d_out; half* d_tgt_h;
    CK(cudaMalloc(&d_in, nIn * sizeof(float)));
    CK(cudaMalloc(&d_tgt_f, nOut * sizeof(float)));
    CK(cudaMalloc(&d_tgt_h, nOut * sizeof(half)));
    CK(cudaMalloc(&d_out, nOut * sizeof(float)));
    if (c.hash) fillF01<<<nb(nIn), 256>>>((float*)d_in, nIn, 1u);
    else        fillH01<<<nb(nIn), 256>>>((half*)d_in, nIn, 1u);
    fillF<<<nb(nOut), 256>>>(d_tgt_f, nOut, 2u);
    f2h<<<nb(nOut), 256>>>(d_tgt_f, d_tgt_h, nOut);
    CK(cudaDeviceSynchronize());

    benchTinyMLP(c, B, d_in, d_tgt_h, d_out);
    benchTcnn(c, B, d_in, d_tgt_f, false);
    benchTcnn(c, B, d_in, d_tgt_f, true);

    cudaFree(d_in); cudaFree(d_tgt_f); cudaFree(d_tgt_h); cudaFree(d_out);
}

// ---------------------------------------------------------------- convergence
// Adam on MLP weights + hash table only: biases keep their zero init, i.e. a bias-free MLP
// like tcnn's FullyFusedMLP. Mirrors TinyMLPHashGrid::step() minus launchAdamBiasOptim.
static void stepNoBias(TinyMLPHashGrid& n, float lr, float b1, float b2, float eps, float ls, cudaStream_t s) {
    if (n.current_step < 50000) n.current_step++;
    float bc1 = 1.0f - powf(b1, (float)n.current_step), bc2 = 1.0f - powf(b2, (float)n.current_step);
    launchAdamWeightsOptim(&n.mlp_opt, n.d_master_weights, n.d_fwd_weights, n.d_w_grad, n.d_w_m, n.d_w_v,
                           lr, b1, b2, eps, bc1, bc2, 1.0f / ls, s);
    launchAdamHashGridOptim(&n.hw_opt, n.d_master_hashtable, n.d_fwd_hashtable, n.d_hashtable_grads,
                            n.d_hash_m, n.d_hash_v, lr, b1, b2, eps, bc1, bc2, 1.0f / ls, s);
}

// tcnn's initialisation: hash table U(+-1e-4), MLP Xavier-uniform, zero biases.
static void tcnnStyleInit(TinyMLPHashGrid& n, int in, int H, int out, int L) {
    std::mt19937 g(7);
    std::vector<float> hash((size_t)16 * (1 << 19) * 2);
    std::uniform_real_distribution<float> hd(-1e-4f, 1e-4f);
    for (auto& v : hash) v = hd(g);
    n.loadHashgrid(hash.data());
    std::vector<float> w, b;
    for (int l = 0; l < L; ++l) {
        int fi = l == 0 ? in : H, fo = l == L - 1 ? out : H;
        float bound = sqrtf(6.f / (fi + fo));
        std::uniform_real_distribution<float> wd(-bound, bound);
        for (int k = 0; k < fi * fo; ++k) w.push_back(wd(g));
        b.insert(b.end(), fo, 0.0f);
    }
    n.loadWeights(w.data(), b.data());
}

// Trains each variant on an identical batch stream and logs held-out MSE.
//   variants: tinymlp | tinymlp_nobias | tinymlp_nobias_tcnninit | tcnn | tcnn_jit
//   rowPrefix is prepended to every CSV row (e.g. "offset," for the ablation file).
static void convRun(FILE* f, const char* rowPrefix, float offset, const std::vector<std::string>& variants) {
    const int B = 1 << 18, BT = 1 << 18, STEPS = g_quick ? 200 : 3000, H = 64, L = 2;
    const float LR = 1e-2f, B2 = 0.99f, EPS = 1e-15f;
    const std::vector<int> evalAt = {0, 25, 50, 100, 200, 400, 700, 1000, 1500, 2000, 3000};

    float *d_pos, *d_tgt, *d_tpos, *d_ttgt, *d_pred, *d_acc; half* d_tgt_h;
    CK(cudaMalloc(&d_pos, (size_t)B * 3 * 4));  CK(cudaMalloc(&d_tgt, (size_t)B * 16 * 4));
    CK(cudaMalloc(&d_tgt_h, (size_t)B * 16 * 2));
    CK(cudaMalloc(&d_tpos, (size_t)BT * 3 * 4)); CK(cudaMalloc(&d_ttgt, (size_t)BT * 16 * 4));
    CK(cudaMalloc(&d_pred, (size_t)BT * 16 * 4)); CK(cudaMalloc(&d_acc, 4));
    fillF01<<<nb((size_t)BT * 3), 256>>>(d_tpos, (size_t)BT * 3, 999u);
    targetField<<<nb(BT), 256>>>(d_tpos, d_ttgt, BT, offset);

    auto batch = [&](int step) {  // identical stream of training batches for every variant
        fillF01<<<nb((size_t)B * 3), 256, 0, g_s>>>(d_pos, (size_t)B * 3, 1000u + step);
        targetField<<<nb(B), 256, 0, g_s>>>(d_pos, d_tgt, B, offset);
        f2h<<<nb((size_t)B * 16), 256, 0, g_s>>>(d_tgt, d_tgt_h, (size_t)B * 16);
    };
    auto mse = [&]() {
        CK(cudaMemsetAsync(d_acc, 0, 4, g_s));
        mseKernel<<<nb(BT), 256, 0, g_s>>>(d_pred, 16, d_ttgt, BT, d_acc);
        float h = 0; CK(cudaMemcpyAsync(&h, d_acc, 4, cudaMemcpyDeviceToHost, g_s));
        CK(cudaStreamSynchronize(g_s));
        return h / (BT * 16.0f);
    };
    auto run = [&](const char* fw, auto step, auto predict) {
        float trainMs = 0;
        size_t next = 0;
        for (int s = 0; s <= STEPS; ++s) {
            if (next < evalAt.size() && evalAt[next] == s) {
                predict(); float m = mse();
                fprintf(f, "%s%s,%d,%.6g,%.3f\n", rowPrefix, fw, s, m, trainMs); fflush(f);
                fprintf(stderr, "  conv %s%-24s step %5d  test MSE %.5f  (train %.0f ms)\n", rowPrefix, fw, s, m, trainMs);
                ++next;
            }
            if (s == STEPS) break;
            batch(s);
            CK(cudaEventRecord(g_e0, g_s)); step(); CK(cudaEventRecord(g_e1, g_s));
            CK(cudaEventSynchronize(g_e1)); float ms; CK(cudaEventElapsedTime(&ms, g_e0, g_e1)); trainMs += ms;
        }
    };

    for (const std::string& v : variants) {
        if (v.rfind("tinymlp", 0) == 0) {
            bool noBias = v.find("nobias") != std::string::npos;
            MLPGridOptions o{}; o.vectorDim = 3; o.hiddenDim = H; o.outputDim = 16; o.numLayers = L;
            o.activationType = ACT_RELU; o.tableSize = 1 << 19; o.numLevels = 16; o.b = 1.38f; o.lowestSize = 16; o.featuresLevel = 2;
            TinyMLPHashGrid net(o, B, BT, 42, true);
            if (v.find("tcnninit") != std::string::npos) tcnnStyleInit(net, 32, H, 16, L);
            run(v.c_str(),
                [&] { net.zero_grad(g_s); net.forward(d_pos, nullptr, B, g_s);
                      net.calculate_loss_and_grad(d_tgt_h, B, 65536.f, false, g_s);
                      net.backward(B, g_s);
                      if (noBias) stepNoBias(net, LR, 0.9f, B2, EPS, 65536.f, g_s);
                      else        net.step(LR, 0.9f, B2, EPS, 65536.f, g_s); },
                [&] { net.inference(d_tpos, d_pred, BT, g_s); });
        } else {
            bool jit = v == "tcnn_jit";
            Case c{"hashgrid", 3, 16, H, L, false, true};
            auto model = std::make_shared<tcnn::NetworkWithInputEncoding<P>>(3u, 16u, tcnnEnc(), tcnnNet(c));
            std::shared_ptr<tcnn::Loss<P>> loss{tcnn::create_loss<P>({{"otype", "L2"}})};
            std::shared_ptr<tcnn::Optimizer<P>> opt{tcnn::create_optimizer<P>(tcnnAdam(LR, B2, EPS))};
            auto trainer = std::make_shared<tcnn::Trainer<float, P, P>>(model, opt, loss);
            if (jit) model->set_jit_fusion(true);
            GPUMatrix<float> in{d_pos, 3u, (uint32_t)B}, tgt{d_tgt, 16u, (uint32_t)B};
            GPUMatrix<float> tin{d_tpos, 3u, (uint32_t)BT}, tout{d_pred, 16u, (uint32_t)BT};
            run(v.c_str(),
                [&] { trainer->training_step(g_s, in, tgt); },
                [&] { model->inference(g_s, tin, tout); });
            if (jit && !model->jit_fusion()) fprintf(stderr, "  conv tcnn_jit fell back to AOT (JIT unavailable)\n");
            tcnn::free_all_gpu_memory_arenas();
        }
    }
    cudaFree(d_pos); cudaFree(d_tgt); cudaFree(d_tgt_h); cudaFree(d_tpos); cudaFree(d_ttgt); cudaFree(d_pred); cudaFree(d_acc);
}

static void convergence(const char* path) {
    FILE* f = fopen(path, "w");
    fprintf(f, "framework,step,test_mse,train_ms\n");
    convRun(f, "", 0.5f, {"tinymlp", "tcnn", "tcnn_jit"});
    fclose(f);
}

// Is TinyMLP's lower final error due to its biases? Remove them (and then tcnn's init too),
// on the original +0.5-offset target and on a zero-mean target where a bias gives no free DC term.
static void ablation(const char* path) {
    FILE* f = fopen(path, "w");
    fprintf(f, "target,framework,step,test_mse,train_ms\n");
    const std::vector<std::string> v = {"tinymlp", "tinymlp_nobias", "tinymlp_nobias_tcnninit", "tcnn"};
    convRun(f, "offset,", 0.5f, v);
    convRun(f, "zeromean,", 0.0f, v);
    fclose(f);
}

// ---------------------------------------------------------------- main
int main(int argc, char** argv) {
    const char* speedPath = nullptr; const char* convPath = nullptr; const char* ablatePath = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--speed") && i + 1 < argc) speedPath = argv[++i];
        else if (!strcmp(argv[i], "--conv") && i + 1 < argc) convPath = argv[++i];
        else if (!strcmp(argv[i], "--ablate") && i + 1 < argc) ablatePath = argv[++i];
        else if (!strcmp(argv[i], "--quick")) g_quick = true;
    }
    if (!speedPath && !convPath && !ablatePath) { speedPath = "speed.csv"; convPath = "conv.csv"; }

    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    fprintf(stderr, "GPU: %s (sm_%d%d)\n", p.name, p.major, p.minor);
    CK(cudaStreamCreate(&g_s)); CK(cudaEventCreate(&g_e0)); CK(cudaEventCreate(&g_e1));

    if (speedPath) {
        g_speed = fopen(speedPath, "w");
        fprintf(g_speed, "framework,model,op,hidden,layers,batch,ms\n");
        std::vector<int> batches = g_quick ? std::vector<int>{1 << 18} : std::vector<int>{1 << 16, 1 << 18, 1 << 20};
        std::vector<int> hiddens = g_quick ? std::vector<int>{64} : std::vector<int>{32, 64, 128};
        std::vector<int> layers  = g_quick ? std::vector<int>{2} : std::vector<int>{2, 4};
        for (int H : hiddens) for (int L : layers) for (int B : batches) {
            runCase({"mlp", H, H, H, L, false, false}, B);
            runCase({"hashgrid", 3, 16, H, L, false, true}, B);
        }
        for (int B : batches) runCase({"nerf_color", 32, 3, 32, 3, true, false}, B);
        fclose(g_speed);
    }
    if (convPath) convergence(convPath);
    if (ablatePath) ablation(ablatePath);
    fprintf(stderr, "done\n");
    return 0;
}
