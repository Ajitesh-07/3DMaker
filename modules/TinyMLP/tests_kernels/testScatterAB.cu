// ============================================================================
// Hash-grid backward SCATTER: A/B prototype
// ----------------------------------------------------------------------------
// Isolates just the gradient scatter (the ~90% / 40-50%-of-train cost) as a
// standalone kernel, and compares the access-pattern structures:
//
//   A) all-levels-per-thread + warp dedup     -> TinyMLP's CURRENT inline pattern
//   B) per-level grid (blockIdx.y = level) + warp dedup   -> structure change only
//   C) per-level grid + dedup ONLY for dense levels       -> full tcnn-style proposal
//
// All three read dL/d(encoding) [B x 32 half] from global (a fair comparison;
// the real kernel reads it from shmem) and scatter into grid_grad [64 MB fp32].
// Correctness is checked vs a CPU reference at a small batch; timing at large
// batches. A vs B isolates the locality effect; B vs C isolates the dedup policy.
// ============================================================================
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <random>
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cmath>

#define CUDA_CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

// ---- config (matches the NeRF density grid) --------------------------------
static const int   NUM_LEVELS = 16;
static const int   FEAT       = 2;             // features per level
static const int   IN_DIM     = NUM_LEVELS*FEAT; // 32 = encoding width
static const int   TABLE      = 1 << 19;       // 2^19 buckets / level
static const int   LOWEST     = 16;
#define GROW_B 1.38f                            // float const isn't usable in device code; use a macro

__device__ __forceinline__ void redAddF32(float* addr, float val) {
    #ifndef __INTELLISENSE__
    asm volatile("red.global.add.f32 [%0], %1;" :: "l"(addr), "f"(val) : "memory");
    #endif
}
__device__ __forceinline__ uint32_t hash_coords(int cx,int cy,int cz,int T){
    return ( ((uint32_t)cx*1u) ^ ((uint32_t)cy*2654435761u) ^ ((uint32_t)cz*805459861u) ) & (uint32_t)(T-1);
}
__device__ __forceinline__ uint32_t dense_index(int cx,int cy,int cz,int N){
    return cx + cy*(N+1) + cz*(N+1)*(N+1);
}

// Scatter one (sample,level): 8 trilinear corners. Matches networkFusionBackwardHashtable.cu.
__device__ __forceinline__ void scatter_one(
    float px, float py, float pz, float gx, float gy,
    int l, int denseLevelStart, float* grads, bool useDedup)
{
    float N_l_float = LOWEST * __powf(GROW_B, (float)l);
    int   N_l = (int)N_l_float;
    float x_l = px*N_l_float, y_l = py*N_l_float, z_l = pz*N_l_float;
    int x0=(int)floorf(x_l), y0=(int)floorf(y_l), z0=(int)floorf(z_l);
    x0=max(0,min(x0,N_l-1)); y0=max(0,min(y0,N_l-1)); z0=max(0,min(z0,N_l-1));
    int x1=min(x0+1,N_l), y1=min(y0+1,N_l), z1=min(z0+1,N_l);
    float wx1=1.0f-x_l+floorf(x_l), wy1=1.0f-y_l+floorf(y_l), wz1=1.0f-z_l+floorf(z_l);

    float w[8]   = { wx1*wy1*wz1, (1-wx1)*wy1*wz1, wx1*(1-wy1)*wz1, (1-wx1)*(1-wy1)*wz1,
                     wx1*wy1*(1-wz1), (1-wx1)*wy1*(1-wz1), wx1*(1-wy1)*(1-wz1), (1-wx1)*(1-wy1)*(1-wz1) };
    int cx[8]={x0,x1,x0,x1,x0,x1,x0,x1};
    int cy[8]={y0,y0,y1,y1,y0,y0,y1,y1};
    int cz[8]={z0,z0,z0,z0,z1,z1,z1,z1};
    bool isDense = l < denseLevelStart;
    uint32_t laneId = threadIdx.x & 31u;

    #pragma unroll
    for (int c=0;c<8;c++){
        uint32_t ti = isDense ? dense_index(cx[c],cy[c],cz[c],N_l) : hash_coords(cx[c],cy[c],cz[c],TABLE);
        uint32_t idx = (uint32_t)l*TABLE*FEAT + ti*FEAT;
        float vx = gx*w[c], vy = gy*w[c];
        if (useDedup) {
            unsigned active = __activemask();
            unsigned grp = __match_any_sync(active, idx);
            int leader = __ffs(grp)-1;
            float sx=0.f, sy=0.f; unsigned tmp=grp;
            while(tmp){ int sl=__ffs(tmp)-1; sx+=__shfl_sync(grp,vx,sl); sy+=__shfl_sync(grp,vy,sl); tmp&=tmp-1; }
            if ((int)laneId==leader){ redAddF32(&grads[idx], sx); redAddF32(&grads[idx+1], sy); }
        } else {
            redAddF32(&grads[idx], vx);
            redAddF32(&grads[idx+1], vy);
        }
    }
}

// (A) all-levels-per-thread: thread = sample, loop over all 16 levels (CURRENT pattern)
template<bool DEDUP_ALL>
__global__ void scatter_allLevels(int B, const float* __restrict__ pos,
                                  const half* __restrict__ dx, float* grads, int denseLevelStart){
    int s = blockIdx.x*blockDim.x + threadIdx.x;
    if (s >= B) return;
    float px=pos[s*4+0], py=pos[s*4+1], pz=pos[s*4+2];
    #pragma unroll 1
    for (int l=0;l<NUM_LEVELS;l++){
        half2 g = *reinterpret_cast<const half2*>(&dx[s*IN_DIM + l*FEAT]);
        float2 gf = __half22float2(g);
        bool dd = DEDUP_ALL ? true : (l < denseLevelStart);
        scatter_one(px,py,pz, gf.x,gf.y, l, denseLevelStart, grads, dd);
    }
}

// (B/C) per-level grid: blockIdx.y = level, thread = sample for that single level (tcnn pattern)
template<bool DEDUP_ALL>
__global__ void scatter_perLevel(int B, const float* __restrict__ pos,
                                 const half* __restrict__ dx, float* grads, int denseLevelStart){
    int l = blockIdx.y;
    int s = blockIdx.x*blockDim.x + threadIdx.x;
    if (s >= B) return;
    float px=pos[s*4+0], py=pos[s*4+1], pz=pos[s*4+2];
    half2 g = *reinterpret_cast<const half2*>(&dx[s*IN_DIM + l*FEAT]);
    float2 gf = __half22float2(g);
    bool dd = DEDUP_ALL ? true : (l < denseLevelStart);
    scatter_one(px,py,pz, gf.x,gf.y, l, denseLevelStart, grads, dd);
}

// ---- CPU reference ----------------------------------------------------------
static uint32_t hash_cpu(int cx,int cy,int cz,int T){
    return ( ((uint32_t)cx*1u) ^ ((uint32_t)cy*2654435761u) ^ ((uint32_t)cz*805459861u) ) & (uint32_t)(T-1);
}
static uint32_t dense_cpu(int cx,int cy,int cz,int N){ return cx + cy*(N+1) + cz*(N+1)*(N+1); }

static void scatterCPU(int B, const std::vector<float>& pos, const std::vector<float>& dx,
                       std::vector<double>& grads, int denseLevelStart){
    for (int s=0;s<B;s++){
        float px=pos[s*4+0], py=pos[s*4+1], pz=pos[s*4+2];
        for (int l=0;l<NUM_LEVELS;l++){
            float N_l_float = LOWEST*powf(GROW_B,(float)l);
            int N_l=(int)N_l_float;
            float x_l=px*N_l_float,y_l=py*N_l_float,z_l=pz*N_l_float;
            int x0=(int)floorf(x_l),y0=(int)floorf(y_l),z0=(int)floorf(z_l);
            x0=std::max(0,std::min(x0,N_l-1)); y0=std::max(0,std::min(y0,N_l-1)); z0=std::max(0,std::min(z0,N_l-1));
            int x1=std::min(x0+1,N_l),y1=std::min(y0+1,N_l),z1=std::min(z0+1,N_l);
            float wx1=1.0f-x_l+floorf(x_l),wy1=1.0f-y_l+floorf(y_l),wz1=1.0f-z_l+floorf(z_l);
            float w[8]={wx1*wy1*wz1,(1-wx1)*wy1*wz1,wx1*(1-wy1)*wz1,(1-wx1)*(1-wy1)*wz1,
                        wx1*wy1*(1-wz1),(1-wx1)*wy1*(1-wz1),wx1*(1-wy1)*(1-wz1),(1-wx1)*(1-wy1)*(1-wz1)};
            int cx[8]={x0,x1,x0,x1,x0,x1,x0,x1},cy[8]={y0,y0,y1,y1,y0,y0,y1,y1},cz[8]={z0,z0,z0,z0,z1,z1,z1,z1};
            bool isDense=l<denseLevelStart;
            float gx=dx[s*IN_DIM+l*FEAT], gy=dx[s*IN_DIM+l*FEAT+1];
            for(int c=0;c<8;c++){
                uint32_t ti=isDense?dense_cpu(cx[c],cy[c],cz[c],N_l):hash_cpu(cx[c],cy[c],cz[c],TABLE);
                size_t idx=(size_t)l*TABLE*FEAT + (size_t)ti*FEAT;
                grads[idx]   += (double)(gx*w[c]);
                grads[idx+1] += (double)(gy*w[c]);
            }
        }
    }
}

int main(){
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    printf("Device: %s,  L2 = %d MB\n\n", prop.name, prop.l2CacheSize/(1024*1024));

    // denseLevelStart (matches NeRF: floor(log_b((cbrt(T)-1)/LOWEST))+1, clamped)
    float inner=(cbrtf((float)TABLE)-1.0f)/LOWEST;
    int denseLevelStart = (int)floorf(log2f(inner)/log2f(GROW_B))+1;
    denseLevelStart = std::max(0,std::min(denseLevelStart,NUM_LEVELS));
    printf("denseLevelStart = %d  (levels 0..%d dense, %d..15 hashed)\n",
           denseLevelStart, denseLevelStart-1, denseLevelStart);

    const size_t GRAD_N = (size_t)NUM_LEVELS*TABLE*FEAT;
    printf("grid_grad table = %.0f MB (one level = %.1f MB)\n\n",
           GRAD_N*sizeof(float)/1e6, (double)TABLE*FEAT*sizeof(float)/1e6);

    float* d_grads; CUDA_CHECK(cudaMalloc(&d_grads, GRAD_N*sizeof(float)));

    std::mt19937 gen(7);
    std::uniform_real_distribution<float> posd(0.f,1.f), gd(-1.f,1.f);

    auto setup = [&](int B, float** d_pos, half** d_dx, std::vector<float>& h_pos, std::vector<float>& h_dxf){
        h_pos.resize((size_t)B*4); h_dxf.resize((size_t)B*IN_DIM);
        std::vector<half> h_dx((size_t)B*IN_DIM);
        for(int s=0;s<B;s++){ h_pos[s*4+0]=posd(gen); h_pos[s*4+1]=posd(gen); h_pos[s*4+2]=posd(gen); h_pos[s*4+3]=0; }
        for(size_t i=0;i<h_dxf.size();i++){ float v=gd(gen); h_dxf[i]=v; h_dx[i]=__float2half(v); }
        CUDA_CHECK(cudaMalloc(d_pos,(size_t)B*4*sizeof(float)));
        CUDA_CHECK(cudaMalloc(d_dx,(size_t)B*IN_DIM*sizeof(half)));
        CUDA_CHECK(cudaMemcpy(*d_pos,h_pos.data(),(size_t)B*4*sizeof(float),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(*d_dx,h_dx.data(),(size_t)B*IN_DIM*sizeof(half),cudaMemcpyHostToDevice));
    };

    // ---------------- Correctness (small batch vs CPU) ----------------
    {
        int B=8192;
        float* d_pos; half* d_dx; std::vector<float> h_pos,h_dxf;
        setup(B,&d_pos,&d_dx,h_pos,h_dxf);
        std::vector<double> ref(GRAD_N,0.0); scatterCPU(B,h_pos,h_dxf,ref,denseLevelStart);

        std::vector<float> out(GRAD_N);
        auto check=[&](const char* name, auto launch){
            CUDA_CHECK(cudaMemset(d_grads,0,GRAD_N*sizeof(float)));
            launch(); CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(out.data(),d_grads,GRAD_N*sizeof(float),cudaMemcpyDeviceToHost));
            double maxd=0,sse=0; size_t nz=0;
            for(size_t i=0;i<GRAD_N;i++){ double d=fabs((double)out[i]-ref[i]); if(d>maxd)maxd=d; sse+=d*d; if(ref[i]!=0)nz++; }
            printf("  %-26s maxAbsDiff=%.3e  rmse=%.3e  (%zu nonzero)  %s\n",
                   name,maxd,sqrt(sse/GRAD_N),nz, maxd<1e-1?"[OK]":"[CHECK]");
        };
        int bs=256, gx=(B+bs-1)/bs;
        printf("=== Correctness vs CPU (B=%d) ===\n",B);
        check("A all-levels+dedup",  [&](){ scatter_allLevels<true><<<gx,bs>>>(B,d_pos,d_dx,d_grads,denseLevelStart); });
        check("B per-level+dedup",   [&](){ dim3 g(gx,NUM_LEVELS); scatter_perLevel<true><<<g,bs>>>(B,d_pos,d_dx,d_grads,denseLevelStart); });
        check("C per-level+denseDedup",[&](){ dim3 g(gx,NUM_LEVELS); scatter_perLevel<false><<<g,bs>>>(B,d_pos,d_dx,d_grads,denseLevelStart); });
        cudaFree(d_pos); cudaFree(d_dx);
    }

    // ---------------- Timing (large batches) ----------------
    printf("\n=== Timing (ms, avg of 30 runs) ===\n");
    printf("%-10s %-14s %-14s %-14s %-10s %-10s\n","Batch","A all-lvl","B per-lvl","C per-lvl/dD","B/A","C/A");
    int bs=256;
    cudaEvent_t e0,e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
    for (int B : {262144, 524288, 1048576}) {
        float* d_pos; half* d_dx; std::vector<float> h_pos,h_dxf;
        setup(B,&d_pos,&d_dx,h_pos,h_dxf);
        int gx=(B+bs-1)/bs; dim3 gpl(gx,NUM_LEVELS);
        const int WARM=5, RUNS=30;
        auto timeit=[&](auto launch)->float{
            CUDA_CHECK(cudaMemset(d_grads,0,GRAD_N*sizeof(float)));
            for(int i=0;i<WARM;i++) launch();
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaEventRecord(e0));
            for(int i=0;i<RUNS;i++) launch();
            CUDA_CHECK(cudaEventRecord(e1)); CUDA_CHECK(cudaEventSynchronize(e1));
            float ms; CUDA_CHECK(cudaEventElapsedTime(&ms,e0,e1)); return ms/RUNS;
        };
        float a=timeit([&](){ scatter_allLevels<true><<<gx,bs>>>(B,d_pos,d_dx,d_grads,denseLevelStart); });
        float b=timeit([&](){ scatter_perLevel<true><<<gpl,bs>>>(B,d_pos,d_dx,d_grads,denseLevelStart); });
        float c=timeit([&](){ scatter_perLevel<false><<<gpl,bs>>>(B,d_pos,d_dx,d_grads,denseLevelStart); });
        printf("%-10d %-14.4f %-14.4f %-14.4f %-10.3f %-10.3f\n",B,a,b,c,b/a,c/a);
        cudaFree(d_pos); cudaFree(d_dx);
    }
    cudaFree(d_grads);
    return 0;
}
