// ============================================================================
// TinyMLP vs PyTorch comparison benchmark (TinyMLP side).
// Emits CSV: framework,model,op,hidden,layers,batch,ms
//   model = mlp | hashgrid
//   op    = inference | backward | train   (train = fwd + loss + bwd + Adam step)
// PyTorch counterpart: tests/bench_compare.py (same sweep). Combined by tests/make_report.py.
// ============================================================================
#include "../TinyMLP.h"
#include "../TinyMLPHashGrid.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdint>
#include <vector>

#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); } }while(0)

__device__ __forceinline__ uint32_t pcg(uint32_t v){ uint32_t s=v*747796405u+2891336453u; uint32_t w=((s>>((s>>28u)+4u))^s)*277803737u; return (w>>22u)^w; }
__device__ __forceinline__ float pcgf(uint32_t i,uint32_t sd){ return __int_as_float((pcg(sd^pcg(i))>>9)|0x3f800000u)-1.0f; }     // [0,1)
__global__ void fillH01(half* d,int n,uint32_t sd){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) d[i]=__float2half(pcgf(i,sd)); }
__global__ void fillHsym(half* d,int n,uint32_t sd){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) d[i]=__float2half(pcgf(i,sd)*2.f-1.f); }
__global__ void fillF01(float* d,int n,uint32_t sd){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) d[i]=pcgf(i,sd); }

static cudaEvent_t e0,e1;
template<class F> static float timeMs(F f,int warmup,int iters){
    for(int i=0;i<warmup;i++) f();
    CK(cudaDeviceSynchronize());
    cudaEventRecord(e0); for(int i=0;i<iters;i++) f(); cudaEventRecord(e1);
    cudaEventSynchronize(e1); float ms=0; cudaEventElapsedTime(&ms,e0,e1); return ms/iters;
}

static const int WARMUP=8, ITERS=40;

static void benchMLP(int H,int L,int B){
    MLPOption opt{}; opt.inputDim=H; opt.hiddenDim=H; opt.outputDim=H; opt.numLayers=L;
    opt.activationType=ACT_RELU; opt.outputActivation=OUT_ACT_NONE;

    half* d_in;  CK(cudaMalloc(&d_in,(size_t)B*H*sizeof(half)));
    float* d_out;CK(cudaMalloc(&d_out,(size_t)B*H*sizeof(float)));
    half* d_tgt; CK(cudaMalloc(&d_tgt,(size_t)B*H*sizeof(half)));
    fillH01<<<((size_t)B*H+255)/256,256>>>(d_in,B*H,1u);
    fillHsym<<<((size_t)B*H+255)/256,256>>>(d_tgt,B*H,2u);
    CK(cudaDeviceSynchronize());

    // inference (eval mode)
    { TinyMLP net(opt,B,B,42,false);
      float ms=timeMs([&]{ net.inference(d_in,d_out,B); },WARMUP,ITERS);
      printf("tinymlp,mlp,inference,%d,%d,%d,%.5f\n",H,L,B,ms); }
    // backward + train (training mode)
    { TinyMLP net(opt,B,0,42,true);
      net.forward(d_in,d_out,B); CK(cudaDeviceSynchronize());
      float msb=timeMs([&]{ net.backward(B); },WARMUP,ITERS);
      printf("tinymlp,mlp,backward,%d,%d,%d,%.5f\n",H,L,B,msb);
      float mst=timeMs([&]{ net.zero_grad(); net.forward(d_in,d_out,B);
                            net.calculate_loss_and_grad(d_tgt,B); net.backward(B); net.step(); },WARMUP,ITERS);
      printf("tinymlp,mlp,train,%d,%d,%d,%.5f\n",H,L,B,mst); }
    fflush(stdout);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_tgt);
}

static void benchHash(int H,int L,int B){
    MLPGridOptions opt{}; opt.vectorDim=3; opt.hiddenDim=H; opt.outputDim=16; opt.numLayers=L;
    opt.activationType=ACT_RELU; opt.tableSize=1<<19; opt.numLevels=16; opt.b=1.38f; opt.lowestSize=16; opt.featuresLevel=2;

    float* d_pos;CK(cudaMalloc(&d_pos,(size_t)B*3*sizeof(float)));
    float* d_out;CK(cudaMalloc(&d_out,(size_t)B*16*sizeof(float)));
    half* d_tgt; CK(cudaMalloc(&d_tgt,(size_t)B*16*sizeof(half)));
    fillF01<<<((size_t)B*3+255)/256,256>>>(d_pos,B*3,3u);
    fillHsym<<<((size_t)B*16+255)/256,256>>>(d_tgt,B*16,4u);
    CK(cudaDeviceSynchronize());

    { TinyMLPHashGrid net(opt,B,B,42,false);
      float ms=timeMs([&]{ net.inference(d_pos,d_out,B); },WARMUP,ITERS);
      printf("tinymlp,hashgrid,inference,%d,%d,%d,%.5f\n",H,L,B,ms); }
    { TinyMLPHashGrid net(opt,B,0,42,true);
      net.forward(d_pos,d_out,B); CK(cudaDeviceSynchronize());
      float msb=timeMs([&]{ net.backward(B); },WARMUP,ITERS);
      printf("tinymlp,hashgrid,backward,%d,%d,%d,%.5f\n",H,L,B,msb);
      float mst=timeMs([&]{ net.zero_grad(); net.forward(d_pos,d_out,B);
                            net.calculate_loss_and_grad(d_tgt,B); net.backward(B); net.step(); },WARMUP,ITERS);
      printf("tinymlp,hashgrid,train,%d,%d,%d,%.5f\n",H,L,B,mst); }
    fflush(stdout);
    cudaFree(d_pos); cudaFree(d_out); cudaFree(d_tgt);
}

int main(){
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    printf("framework,model,op,hidden,layers,batch,ms\n");
    std::vector<int> hiddens={32,64,128};
    std::vector<int> layersL={2,4};
    std::vector<int> batches={65536,262144,1048576};
    for(int H:hiddens) for(int L:layersL) for(int B:batches){
        benchMLP(H,L,B);
        benchHash(H,L,B);
    }
    return 0;
}
