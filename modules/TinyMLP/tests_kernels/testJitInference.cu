// ============================================================================
// JIT (NVRTC) inference experiment
// ----------------------------------------------------------------------------
// Question: does runtime-compiling (NVRTC) the fused inference kernel with the
// network dims/layers baked in as COMPILE-TIME CONSTANTS beat the shipping AOT
// path (launchNetworkFusionKernel, which is template-specialized on the block
// geometry but passes inputDim/hiddenDim/outputDim as RUNTIME args)?
//
// Both run the *same* kernel body. The JIT version differs only in that the
// dims and layer count are constexpr (so div/mod by dims strength-reduce, bounds
// checks can fold, registers tune to the exact shape). We verify the outputs
// match, then time JIT-kernel vs AOT-kernel (and report NVRTC compile time).
// ============================================================================
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda.h>        // driver API (cuModule*, cuLaunchKernel)
#include <nvrtc.h>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include "../TinyMLP.h"

#define CUDA_CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)
#define CU_CHECK(call) do { CUresult r=(call); if(r!=CUDA_SUCCESS){ const char* s; cuGetErrorString(r,&s); \
    fprintf(stderr,"CU driver error %s:%d: %s\n",__FILE__,__LINE__,s); exit(1);} } while(0)
#define NVRTC_CHECK(call) do { nvrtcResult r=(call); if(r!=NVRTC_SUCCESS){ \
    fprintf(stderr,"NVRTC error %s:%d: %s\n",__FILE__,__LINE__,nvrtcGetErrorString(r)); exit(1);} } while(0)

// ---------------------------------------------------------------------------
// The JIT kernel body: a faithful copy of networkFusionMMA_2d (networkFusion.cu),
// turned into an extern "C" kernel with the template params + dims supplied as
// JIT_* macros (prepended at compile time). cp.async smem-address conversion is
// done with explicit PTX so it compiles cleanly under NVRTC.
// ---------------------------------------------------------------------------
static const char* KERNEL_SRC = R"NVRTC(
#include <cuda_fp16.h>
#include <mma.h>
typedef unsigned int uint32_t;   // NVRTC has no <cstdint>; identical-typedef redefinition is legal

__device__ __forceinline__ void cp_async_128(void* smem, const void* global, bool valid) {
    uint32_t smem_int;
    asm volatile("{ .reg .u64 sp; cvta.to.shared.u64 sp, %1; cvt.u32.u64 %0, sp; }"
                 : "=r"(smem_int) : "l"(smem));
    int src_size = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(smem_int), "l"(global), "r"(src_size) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait()   { asm volatile("cp.async.wait_group 0;\n" ::); }

extern "C" __global__ void __launch_bounds__(256, 3) jit_infer(
    half* d_inputs, half** d_weights_array, float** d_biases_array,
    float* d_outputs, int batchSize
) {
    const int TILE_COUNT_Y = JIT_TCY;
    const int TILE_COUNT_X = JIT_TCX;
    const int WARP_FACTOR  = JIT_WF;
    const int NUM_LAYERS   = JIT_NUM_LAYERS;
    const int OUT_ACT      = JIT_OUT_ACT;
    const int inputDim     = JIT_IN;
    const int hiddenDim    = JIT_HID;
    const int outputDim    = JIT_OUT;

    uint32_t idx   = threadIdx.z * (blockDim.x * blockDim.y) + threadIdx.y * blockDim.x + threadIdx.x;
    uint32_t warpM = blockIdx.x * TILE_COUNT_Y + threadIdx.z;
    uint32_t warpN = threadIdx.y;
    uint32_t laneId = threadIdx.x;

    const int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16, PAD = 8;

    __shared__ half shmem_A[WMMA_M * TILE_COUNT_Y][TILE_COUNT_X*WMMA_K + PAD];
    __shared__ half shmem_B[2][TILE_COUNT_X*WMMA_N][WMMA_K + PAD];

    #pragma unroll
    for (int i = 0; i < WARP_FACTOR; i++) {
        uint32_t chunk_row = (idx + i*256) / (TILE_COUNT_X * WMMA_K / 8);
        uint32_t chunk_col = (idx + i*256) % (TILE_COUNT_X * WMMA_K / 8) * 8;
        uint32_t global_row = (blockIdx.x * TILE_COUNT_Y) * WMMA_M + chunk_row;
        uint32_t global_col = chunk_col;
        bool a_valid = (global_row < batchSize) && (global_col < inputDim);
        cp_async_128(&shmem_A[chunk_row][chunk_col], &d_inputs[global_row * inputDim + global_col], a_valid);
    }
    cp_async_commit();
    cp_async_wait();
    __syncthreads();

    #pragma unroll
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        int currentK = (layer == 0) ? inputDim : hiddenDim;
        int currentN = (layer == NUM_LAYERS - 1) ? outputDim : hiddenDim;
        bool isLastLayer = layer == NUM_LAYERS - 1;

        uint32_t chunk_row_b = idx / (WMMA_K / 8);
        uint32_t chunk_col_b = idx % (WMMA_K / 8) * 8;

        if (chunk_row_b < TILE_COUNT_X * WMMA_N) {
            bool b_valid = (chunk_row_b < currentN) && (chunk_col_b < currentK);
            cp_async_128(&shmem_B[0][chunk_row_b][chunk_col_b],
                         &d_weights_array[layer][chunk_row_b * currentK + chunk_col_b], b_valid);
        }
        cp_async_commit();
        cp_async_wait();
        __syncthreads();

        int buf = 0;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_acc[WARP_FACTOR];
        #pragma unroll
        for(int i = 0; i < WARP_FACTOR; i++) nvcuda::wmma::fill_fragment(frag_acc[i], 0.0f);

        for (int k = 0; k < currentK; k += WMMA_K) {
            int next_buf = 1 - buf;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> frag_a;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> frag_b;

            if (k + WMMA_K < currentK) {
                if (chunk_row_b < TILE_COUNT_X * WMMA_N) {
                    bool isValid = (chunk_row_b < currentN) && (chunk_col_b + k + WMMA_K) < currentK;
                    cp_async_128(&shmem_B[next_buf][chunk_row_b][chunk_col_b],
                                 &d_weights_array[layer][chunk_row_b * currentK + (chunk_col_b + k + WMMA_K)], isValid);
                }
                cp_async_commit();
            }

            nvcuda::wmma::load_matrix_sync(frag_b, &shmem_B[buf][warpN * WMMA_K][0], WMMA_K + PAD);

            #pragma unroll
            for (int i = 0; i < WARP_FACTOR; i++) {
                nvcuda::wmma::load_matrix_sync(frag_a, &shmem_A[(threadIdx.z + i*blockDim.z)*WMMA_M][k], TILE_COUNT_X*WMMA_K + PAD);
                nvcuda::wmma::mma_sync(frag_acc[i], frag_a, frag_b, frag_acc[i]);
            }

            if (k + WMMA_K < currentK) { cp_async_wait(); __syncthreads(); }
            buf = next_buf;
        }

        int base_col = warpN * WMMA_N;
        #pragma unroll
        for(int j = 0; j < WARP_FACTOR; j++) {
            #pragma unroll
            for (int i = 0; i < frag_acc[j].num_elements; i++) {
                uint32_t local_row = (laneId / 4) + ((i / 2) % 2) * 8;
                uint32_t local_col = (laneId % 4) * 2 + (i % 2) + (i / 4) * 8;
                uint32_t global_r = (warpM + j*blockDim.z) * WMMA_M + local_row;
                uint32_t global_c = base_col + local_col;
                float val = frag_acc[j].x[i];
                if (global_c < currentN) val += d_biases_array[layer][global_c];
                if(!isLastLayer) { val = fmaxf(val, 0.0f); }
                else if constexpr (OUT_ACT == 1) { val = 1.0f / (1.0f + expf(-val)); }
                if (global_r < batchSize && global_c < currentN) {
                    if (isLastLayer) d_outputs[global_r * outputDim + global_c] = val;
                    else shmem_A[(threadIdx.z + j*blockDim.z)*WMMA_M + local_row][base_col + local_col] = __float2half(val);
                }
            }
        }
        __syncthreads();
    }
}
)NVRTC";

struct Geom { int tcy, tcx, wf; };
static Geom geometry(int maxDim) {
    const int WF = 4;
    int tcx = maxDim / 16; if (tcx == 0) tcx = 1;
    int tcy = 8 / tcx;            // warps in Y
    return { tcy * WF, tcx, WF }; // template TILE_COUNT_Y = tcy*WF
}

int main() {
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n\n", prop.name);
    CU_CHECK(cuInit(0));

    struct Cfg { int dim; int layers; };
    std::vector<Cfg> cfgs = { {32,2}, {32,3}, {64,2}, {64,3} };
    const int batch = 1 << 20;

    printf("%-6s %-7s %-12s %-12s %-9s %-12s %-10s\n",
           "Dim","Layers","AOT(ms)","JIT(ms)","speedup","JITcompile","maxAbsDiff");
    printf("--------------------------------------------------------------------------\n");

    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-0.3f, 0.3f);

    for (auto cfg : cfgs) {
        int D = cfg.dim, L = cfg.layers;
        MLPOption opt; opt.inputDim=D; opt.hiddenDim=D; opt.outputDim=D;
        opt.numLayers=L; opt.activationType=ACT_RELU; opt.outputActivation=OUT_ACT_NONE;

        // ---- weights / biases / inputs ----
        std::vector<half*>  h_w(L); std::vector<float*> h_b(L);
        for (int l=0;l<L;l++){
            std::vector<half> w(D*D); for(auto&v:w) v=__float2half(dist(gen));
            std::vector<float> b(D, 0.f);
            CUDA_CHECK(cudaMalloc(&h_w[l], D*D*sizeof(half)));
            CUDA_CHECK(cudaMalloc(&h_b[l], D*sizeof(float)));
            CUDA_CHECK(cudaMemcpy(h_w[l], w.data(), D*D*sizeof(half), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(h_b[l], b.data(), D*sizeof(float), cudaMemcpyHostToDevice));
        }
        half** d_w; float** d_b;
        CUDA_CHECK(cudaMalloc(&d_w, L*sizeof(half*)));
        CUDA_CHECK(cudaMalloc(&d_b, L*sizeof(float*)));
        CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), L*sizeof(half*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), L*sizeof(float*), cudaMemcpyHostToDevice));

        half* d_in; float* d_out_aot; float* d_out_jit;
        std::vector<half> in(batch*D); for(auto&v:in) v=__float2half(dist(gen));
        CUDA_CHECK(cudaMalloc(&d_in, (size_t)batch*D*sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_out_aot, (size_t)batch*D*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out_jit, (size_t)batch*D*sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_in, in.data(), (size_t)batch*D*sizeof(half), cudaMemcpyHostToDevice));

        cudaStream_t stream=0;
        cudaEvent_t e0,e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
        const int RUNS=50, WARM=10;

        // ---- AOT ----
        for(int i=0;i<WARM;i++) launchNetworkFusionKernel(&opt,d_in,d_w,d_b,d_out_aot,batch,stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaEventRecord(e0,stream));
        for(int i=0;i<RUNS;i++) launchNetworkFusionKernel(&opt,d_in,d_w,d_b,d_out_aot,batch,stream);
        CUDA_CHECK(cudaEventRecord(e1,stream)); CUDA_CHECK(cudaEventSynchronize(e1));
        float aot_ms=0; CUDA_CHECK(cudaEventElapsedTime(&aot_ms,e0,e1)); aot_ms/=RUNS;

        // ---- JIT compile (specialized) ----
        Geom g = geometry(D);
        std::string defs =
            "#define JIT_TCY " + std::to_string(g.tcy) + "\n"
            "#define JIT_TCX " + std::to_string(g.tcx) + "\n"
            "#define JIT_WF "  + std::to_string(g.wf)  + "\n"
            "#define JIT_NUM_LAYERS " + std::to_string(L) + "\n"
            "#define JIT_OUT_ACT 0\n"
            "#define JIT_IN "  + std::to_string(D) + "\n"
            "#define JIT_HID " + std::to_string(D) + "\n"
            "#define JIT_OUT " + std::to_string(D) + "\n";
        std::string src = defs + KERNEL_SRC;

        cudaEvent_t c0,c1; CUDA_CHECK(cudaEventCreate(&c0)); CUDA_CHECK(cudaEventCreate(&c1));
        // (compile time measured on host wall clock instead of events)
        auto t_compile0 = std::chrono::high_resolution_clock::now();

        nvrtcProgram prog;
        NVRTC_CHECK(nvrtcCreateProgram(&prog, src.c_str(), "jit_infer.cu", 0, nullptr, nullptr));
        std::string incOpt = std::string("--include-path=") + CUDA_MAIN_INCLUDE;
        const char* opts[] = { "--gpu-architecture=compute_89", "--std=c++17",
                               "--use_fast_math", incOpt.c_str() };
        nvrtcResult cr = nvrtcCompileProgram(prog, 4, opts);
        if (cr != NVRTC_SUCCESS) {
            size_t logSize; nvrtcGetProgramLogSize(prog,&logSize);
            std::string log(logSize,'\0'); nvrtcGetProgramLog(prog,&log[0]);
            fprintf(stderr,"NVRTC compile FAILED:\n%s\n", log.c_str());
            return 1;
        }
        size_t ptxSize; NVRTC_CHECK(nvrtcGetPTXSize(prog,&ptxSize));
        std::string ptx(ptxSize,'\0'); NVRTC_CHECK(nvrtcGetPTX(prog,&ptx[0]));
        NVRTC_CHECK(nvrtcDestroyProgram(&prog));

        CUmodule mod; CUfunction fn;
        CU_CHECK(cuModuleLoadDataEx(&mod, ptx.c_str(), 0, nullptr, nullptr));
        CU_CHECK(cuModuleGetFunction(&fn, mod, "jit_infer"));
        auto t_compile1 = std::chrono::high_resolution_clock::now();
        float compile_ms = std::chrono::duration<double,std::milli>(t_compile1 - t_compile0).count();

        // ---- JIT launch (same geometry as AOT) ----
        int totalRows = (g.tcy/g.wf) * g.wf * 16;  // tileCountY * WF * 16
        // tileCountY (warps in Y) = g.tcy / g.wf ; block(32, tcx, tileCountY)
        int tileCountY = g.tcy / g.wf;
        int numBlocks = (batch + totalRows - 1) / totalRows;
        void* args[] = { &d_in, &d_w, &d_b, &d_out_jit, (void*)&batch };
        auto launchJit = [&](){
            CU_CHECK(cuLaunchKernel(fn, numBlocks,1,1, 32, g.tcx, tileCountY, 0, (CUstream)stream, args, nullptr));
        };
        for(int i=0;i<WARM;i++) launchJit();
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaEventRecord(e0,stream));
        for(int i=0;i<RUNS;i++) launchJit();
        CUDA_CHECK(cudaEventRecord(e1,stream)); CUDA_CHECK(cudaEventSynchronize(e1));
        float jit_ms=0; CUDA_CHECK(cudaEventElapsedTime(&jit_ms,e0,e1)); jit_ms/=RUNS;

        // ---- correctness ----
        std::vector<float> oa((size_t)batch*D), oj((size_t)batch*D);
        CUDA_CHECK(cudaMemcpy(oa.data(), d_out_aot, (size_t)batch*D*sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(oj.data(), d_out_jit, (size_t)batch*D*sizeof(float), cudaMemcpyDeviceToHost));
        float maxdiff=0; for(size_t i=0;i<oa.size();i++){ float d=fabsf(oa[i]-oj[i]); if(d>maxdiff) maxdiff=d; }

        printf("%-6d %-7d %-12.4f %-12.4f %-9.3f %-12.2f %-10.2e\n",
               D, L, aot_ms, jit_ms, aot_ms/jit_ms, compile_ms, maxdiff);

        // cleanup
        cuModuleUnload(mod);
        for(int l=0;l<L;l++){ cudaFree(h_w[l]); cudaFree(h_b[l]); }
        cudaFree(d_w); cudaFree(d_b); cudaFree(d_in); cudaFree(d_out_aot); cudaFree(d_out_jit);
        cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(c0); cudaEventDestroy(c1);
    }
    return 0;
}
