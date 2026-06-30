// ============================================================================
// Dim-32 forward specialization benchmark.
//
// Goal: establish whether the general WMMA fused forward (networkFusionMMA_2d,
// dispatched for maxDim=32 as block(32,2,4), WARP_FACTOR=4) is COMPUTE-bound
// (tensor cores already near roofline -> no point specializing) or OVERHEAD-bound
// (inter-layer shmem round-trip + __syncthreads + epilogue scatter dominate the
// tiny 32x32 MMA -> a hand-specialized kernel can win).
//
// RTX 4060 Laptop (sm_89): ~118 TFLOPS fp16 tensor, ~30 TFLOPS fp16 CUDA-core.
// If the WMMA forward achieves << 30 TFLOPS here, it is overhead-bound and a
// sync-free / register-resident specialization has real headroom.
// ============================================================================
#include "../TinyMLP.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cmath>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    printf("CUDA error %s:%d : %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

// ---- device-side PCG init (no host RNG -> no downclock) ----
__device__ __forceinline__ uint32_t pcg_hash(uint32_t v){
    uint32_t s=v*747796405u+2891336453u; uint32_t w=((s>>((s>>28u)+4u))^s)*277803737u; return (w>>22u)^w; }
__device__ __forceinline__ float pcg_uniform(uint32_t i,uint32_t seed){
    uint32_t r=pcg_hash(seed^pcg_hash(i)); return __int_as_float((r>>9)|0x3f800000u)-1.0f; }      // [0,1)
__global__ void pcg_fill_h(half* d,int n,uint32_t seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) d[i]=__float2half(pcg_uniform(i,seed)*2.0f-1.0f); }   // [-1,1)
__global__ void pcg_fill_f(float* d,int n,uint32_t seed){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) d[i]=pcg_uniform(i,seed)*2.0f-1.0f; }                  // [-1,1)

// ============================================================================
// Specialized dim-32 forward: thread-per-sample, sync-free, register-resident.
//   - each thread owns ONE sample; its 32-wide activation lives in registers
//     across all layers (no inter-layer shmem round-trip, no __syncthreads)
//   - all layers' weights preloaded to shmem once; weight reads are warp
//     broadcasts (every lane hits the same address -> conflict-free)
//   - CUDA-core half2 mul, float accumulate (matches WMMA fp32 accumulate)
// Assumes inputDim == hiddenDim == outputDim == 32 (the proof-of-concept case).
// ============================================================================
template<int NL>
__global__ void __launch_bounds__(256) forwardDim32_spec(
    const half*  __restrict__ in,
    const half*  const* __restrict__ W,    // [layer] -> 32*32 row-major [N][K]
    const float* const* __restrict__ Bs,   // [layer] -> 32
    float* __restrict__ out,
    int batchSize)
{
    const int D = 32;
    __shared__ half  Wsh[NL*D*D];
    __shared__ float Bsh[NL*D];

    for (int i = threadIdx.x; i < NL*D*D; i += blockDim.x) Wsh[i] = W[i/(D*D)][i%(D*D)];
    for (int i = threadIdx.x; i < NL*D;   i += blockDim.x) Bsh[i] = Bs[i/D][i%D];
    __syncthreads();

    int sample = blockIdx.x*blockDim.x + threadIdx.x;
    if (sample >= batchSize) return;

    const int H = D/2;                                   // 16 half2 lanes
    half2 act2[H];
    const half2* in2 = (const half2*)(in + (size_t)sample*D);
    #pragma unroll
    for (int k=0;k<H;k++) act2[k] = in2[k];

    #pragma unroll
    for (int l=0;l<NL;l++){
        const half2* Wl2 = (const half2*)(Wsh + l*D*D);
        const float* Bl  = Bsh + l*D;
        half nxt[D];
        #pragma unroll
        for (int j=0;j<D;j++){
            // two independent half2 accumulators -> break the 16-deep dep chain (ILP)
            half2 a0 = __halves2half2((half)0,(half)0), a1 = a0;
            #pragma unroll
            for (int k=0;k<H;k+=2){
                a0 = __hfma2(act2[k],   Wl2[j*H+k],   a0);   // shmem broadcast
                a1 = __hfma2(act2[k+1], Wl2[j*H+k+1], a1);
            }
            half2 acc2 = __hadd2(a0, a1);
            float acc = __low2float(acc2) + __high2float(acc2) + Bl[j];
            if (l < NL-1) nxt[j] = __float2half(fmaxf(acc, 0.0f));        // ReLU
            else          out[(size_t)sample*D + j] = acc;               // last layer (OUT_ACT_NONE)
        }
        if (l < NL-1) {
            #pragma unroll
            for (int k=0;k<H;k++) act2[k] = __halves2half2(nxt[2*k], nxt[2*k+1]);
        }
    }
}

template<int NL>
static void launch_spec(const half* in, half** W, float** Bs, float* out, int B, cudaStream_t s){
    forwardDim32_spec<NL><<<(B+255)/256, 256, 0, s>>>(in, W, Bs, out, B);
}

int main(){
    const int inputDim=32, hiddenDim=32, outputDim=32, numLayers=3;
    MLPOption opt = {inputDim, hiddenDim, outputDim, numLayers, ACT_RELU};
    cudaStream_t s = 0;

    std::vector<int> batches = {1<<16, 1<<18, 1<<20, 1<<21};
    int maxB = batches.back();

    half*  d_inputs;  float* d_outputs;
    CUDA_CHECK(cudaMalloc(&d_inputs,  (size_t)maxB*inputDim *sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_outputs, (size_t)maxB*outputDim*sizeof(float)));
    pcg_fill_h<<<((size_t)maxB*inputDim+255)/256,256>>>(d_inputs, maxB*inputDim, 1u);

    std::vector<half*> hw(numLayers); std::vector<float*> hb(numLayers);
    for(int l=0;l<numLayers;l++){
        int K=(l==0)?inputDim:hiddenDim, N=(l==numLayers-1)?outputDim:hiddenDim;
        CUDA_CHECK(cudaMalloc(&hw[l], (size_t)N*K*sizeof(half)));
        CUDA_CHECK(cudaMalloc(&hb[l], (size_t)N  *sizeof(float)));
        pcg_fill_h<<<(N*K+255)/256,256>>>(hw[l], N*K, 100u+l);
        pcg_fill_f<<<(N  +255)/256,256>>>(hb[l], N,   200u+l);
    }
    half** dw; float** db;
    CUDA_CHECK(cudaMalloc(&dw, numLayers*sizeof(half*)));
    CUDA_CHECK(cudaMalloc(&db, numLayers*sizeof(float*)));
    CUDA_CHECK(cudaMemcpy(dw, hw.data(), numLayers*sizeof(half*),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, hb.data(), numLayers*sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("\nDim-32 fused forward baseline  (in=%d hidden=%d out=%d layers=%d)\n",
           inputDim, hiddenDim, outputDim, numLayers);
    printf("Peak fp16: ~118 TFLOPS tensor / ~30 TFLOPS CUDA-core (RTX 4060 Laptop)\n\n");
    float* d_out_spec;
    CUDA_CHECK(cudaMalloc(&d_out_spec, (size_t)maxB*outputDim*sizeof(float)));

    auto timeKernel = [&](auto fn)->float{
        fn(); CUDA_CHECK(cudaStreamSynchronize(s));
        cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        const int RUNS=50; float tot=0;
        for(int r=0;r<RUNS;r++){ cudaEventRecord(e0,s); fn(); cudaEventRecord(e1,s); cudaEventSynchronize(e1);
                                 float ms=0; cudaEventElapsedTime(&ms,e0,e1); tot+=ms; }
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        return tot/RUNS;
    };

    printf("%-12s %-12s %-12s %-10s %-10s %-9s\n", "Batch", "WMMA(ms)", "Spec(ms)", "WMMA-TF", "Spec-TF", "Speedup");
    printf("----------------------------------------------------------------------\n");
    for(int B : batches){
        double flops=0;
        for(int l=0;l<numLayers;l++){ int K=(l==0)?inputDim:hiddenDim, N=(l==numLayers-1)?outputDim:hiddenDim; flops += 2.0*B*K*N; }

        float wmma_ms = timeKernel([&]{ launchNetworkFusionKernel(&opt, d_inputs, dw, db, d_outputs,  B, s); });
        float spec_ms = timeKernel([&]{ launch_spec<3>(d_inputs, dw, db, d_out_spec, B, s); });
        double wtf=(flops/(wmma_ms*1e-3))/1e12, stf=(flops/(spec_ms*1e-3))/1e12;
        printf("%-12d %-12.4f %-12.4f %-10.2f %-10.2f %-9.3f\n", B, wmma_ms, spec_ms, wtf, stf, wmma_ms/spec_ms);
    }

    // correctness: WMMA vs spec at the largest batch
    {
        int B = batches.back();
        launchNetworkFusionKernel(&opt, d_inputs, dw, db, d_outputs,  B, s);
        launch_spec<3>(d_inputs, dw, db, d_out_spec, B, s);
        CUDA_CHECK(cudaStreamSynchronize(s));
        std::vector<float> a((size_t)B*outputDim), b((size_t)B*outputDim);
        CUDA_CHECK(cudaMemcpy(a.data(), d_outputs,  a.size()*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b.data(), d_out_spec, b.size()*sizeof(float), cudaMemcpyDeviceToHost));
        float mx=0; double mse=0;
        for(size_t i=0;i<a.size();i++){ float d=fabsf(a[i]-b[i]); if(d>mx)mx=d; mse+=(double)d*d; }
        mse/=a.size();
        printf("\nCorrectness (WMMA vs spec, B=%d): MaxDiff=%.5f  MSE=%.3e  -> %s\n",
               B, mx, mse, (mx<0.1f ? "[MATCH]" : "[DIFFERS]"));
    }
    printf("\n");
    return 0;
}
