// ============================================================================
// Dim-32 PLAIN MLP backward roofline. Is networkFusionMMA_Backward (dispatched
// for maxDim=32) compute-bound or memory-bound, and how close to optimal?
//
// Backward traffic (per step): reads d_loss (B*out) + all saved activations
// (numLayers * B * dim, read ~2x: once for dW=dZ^T@A, once for the ReLU mask)
// + weights(tiny); writes d_dx_out (B*in) + dW/db (tiny, atomic).
// Compute: per layer dW (2*B*N*K) + dA propagate (2*B*N*K) ~= 2x forward.
// RTX 4060 Laptop: ~272 GB/s DRAM, ~118 TFLOPS fp16 tensor.
// ============================================================================
#include "../TinyMLP.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    printf("CUDA error %s:%d : %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

__device__ __forceinline__ uint32_t pcg_hash(uint32_t v){
    uint32_t s=v*747796405u+2891336453u; uint32_t w=((s>>((s>>28u)+4u))^s)*277803737u; return (w>>22u)^w; }
__device__ __forceinline__ float pcg_uniform(uint32_t i,uint32_t seed){
    uint32_t r=pcg_hash(seed^pcg_hash(i)); return __int_as_float((r>>9)|0x3f800000u)-1.0f; }     // [0,1)
__global__ void pcg_fill_h_pos(half* d,int n,uint32_t seed){    // [0,1) -> post-ReLU-like activations
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) d[i]=__float2half(pcg_uniform(i,seed)); }
__global__ void pcg_fill_h_sym(half* d,int n,uint32_t seed){    // [-1,1) -> loss / weights
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) d[i]=__float2half(pcg_uniform(i,seed)*2.0f-1.0f); }

// Defined in networkFusionBackward_lb4.cu: identical kernel, __launch_bounds__(256,4).
void launchNetworkFusionBackwardKernel_lb4(
    MLPOption* opt, half* d_loss_output, half** d_weights_array, float** d_biases_array,
    half** d_activations, float** d_grad_weights, float** d_grad_biases,
    half* d_dx_out, int batchSize, cudaStream_t stream);
// Same kernel, FREE_ACT=1: activations clamped to L2-resident window => "free" activation reads.
// (lb4 / freeact) is the CEILING of what activation-recompute could ever buy.
void launchNetworkFusionBackwardKernel_lb4_freeact(
    MLPOption* opt, half* d_loss_output, half** d_weights_array, float** d_biases_array,
    half** d_activations, float** d_grad_weights, float** d_grad_biases,
    half* d_dx_out, int batchSize, cudaStream_t stream);

int main(){
    const int inputDim=32, hiddenDim=32, outputDim=32, numLayers=3;
    MLPOption opt = {inputDim, hiddenDim, outputDim, numLayers, ACT_RELU};
    cudaStream_t s = 0;

    std::vector<int> batches = {1<<16, 1<<18, 1<<20, 1<<21};
    int maxB = batches.back();

    half*  d_loss; half* d_dx;
    CUDA_CHECK(cudaMalloc(&d_loss, (size_t)maxB*outputDim*sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_dx,   (size_t)maxB*inputDim *sizeof(half)));
    pcg_fill_h_sym<<<((size_t)maxB*outputDim+255)/256,256>>>(d_loss, maxB*outputDim, 7u);

    auto dimOfLayerInput = [&](int l){ return (l==0)?inputDim:hiddenDim; };

    std::vector<half*>  hw(numLayers), ha(numLayers);
    std::vector<float*> hgw(numLayers), hgb(numLayers);
    for(int l=0;l<numLayers;l++){
        int K = dimOfLayerInput(l);
        int N = (l==numLayers-1)?outputDim:hiddenDim;
        CUDA_CHECK(cudaMalloc(&hw[l],  (size_t)N*K*sizeof(half)));
        CUDA_CHECK(cudaMalloc(&ha[l],  (size_t)maxB*K*sizeof(half)));     // activation = input to layer l
        CUDA_CHECK(cudaMalloc(&hgw[l], (size_t)N*K*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&hgb[l], (size_t)N  *sizeof(float)));
        pcg_fill_h_sym<<<(N*K+255)/256,256>>>(hw[l], N*K, 100u+l);
        pcg_fill_h_pos<<<((size_t)maxB*K+255)/256,256>>>(ha[l], maxB*K, 300u+l);
        CUDA_CHECK(cudaMemset(hgw[l], 0, (size_t)N*K*sizeof(float)));
        CUDA_CHECK(cudaMemset(hgb[l], 0, (size_t)N  *sizeof(float)));
    }
    half** d_wt; half** d_act; float** d_gw; float** d_gb;
    CUDA_CHECK(cudaMalloc(&d_wt,  numLayers*sizeof(half*)));
    CUDA_CHECK(cudaMalloc(&d_act, numLayers*sizeof(half*)));
    CUDA_CHECK(cudaMalloc(&d_gw,  numLayers*sizeof(float*)));
    CUDA_CHECK(cudaMalloc(&d_gb,  numLayers*sizeof(float*)));
    CUDA_CHECK(cudaMemcpy(d_wt,  hw.data(),  numLayers*sizeof(half*),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_act, ha.data(),  numLayers*sizeof(half*),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gw,  hgw.data(), numLayers*sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gb,  hgb.data(), numLayers*sizeof(float*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("\nDim-32 plain MLP backward roofline  (in=%d hidden=%d out=%d layers=%d)\n",
           inputDim, hiddenDim, outputDim, numLayers);
    printf("Peak: ~272 GB/s DRAM, ~118 TFLOPS fp16 tensor (RTX 4060 Laptop)\n\n");
    auto timeFn = [&](auto launch, int B)->float{
        launch(B); CUDA_CHECK(cudaStreamSynchronize(s));
        cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        const int RUNS=50; float tot=0;
        for(int r=0;r<RUNS;r++){ cudaEventRecord(e0,s); launch(B);
            cudaEventRecord(e1,s); cudaEventSynchronize(e1); float ms=0; cudaEventElapsedTime(&ms,e0,e1); tot+=ms; }
        cudaEventDestroy(e0); cudaEventDestroy(e1); return tot/RUNS;
    };
    auto lb4  = [&](int B){ launchNetworkFusionBackwardKernel_lb4        (&opt, d_loss, d_wt, nullptr, d_act, d_gw, d_gb, d_dx, B, s); };
    auto free = [&](int B){ launchNetworkFusionBackwardKernel_lb4_freeact(&opt, d_loss, d_wt, nullptr, d_act, d_gw, d_gb, d_dx, B, s); };

    // CEILING experiment: lb4 = real activations from DRAM; freeact = same kernel but activation
    // reads clamped to an L2-resident window (~free). lb4/freeact upper-bounds recompute's payoff.
    printf("%-10s %-13s %-13s %-12s\n", "Batch", "lb4(ms)", "freeAct(ms)", "ceiling x");
    printf("------------------------------------------------------\n");
    for(int B : batches){
        float ms4   = timeFn(lb4,  B);
        float msF   = timeFn(free, B);
        printf("%-10d %-13.4f %-13.4f %-12.3f\n", B, ms4, msF, ms4/msF);   // ms4/msF = max speedup recompute could give
    }
    printf("\n(ceiling x = lb4 / freeAct = upper bound on any activation-recompute speedup)\n\n");
    return 0;
}
