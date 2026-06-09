#include "TinyMLPHashGrid.h"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <cstdint>
#include <string>

__global__ void hg_padHalfToHalf(const half* __restrict__ src,half* __restrict__ dst,int batch,int undim,int padim){
    int r=blockIdx.x*blockDim.x+threadIdx.x, c=blockIdx.y*blockDim.y+threadIdx.y;
    if(r<batch&&c<padim) dst[r*padim+c]=(c<undim)?src[r*undim+c]:__float2half(0.0f);
}

__global__ void hg_floatToHalf(const float* __restrict__ src,half* __restrict__ dst,int batch,int dim){
    int r=blockIdx.x*blockDim.x+threadIdx.x, c=blockIdx.y*blockDim.y+threadIdx.y;
    if(r<batch&&c<dim) dst[r*dim+c]=__float2half(src[r*dim+c]);
}

__global__ void hg_unpadFloat(const float* __restrict__ src,float* __restrict__ dst,int batch,int padim,int undim){
    int r=blockIdx.x*blockDim.x+threadIdx.x, c=blockIdx.y*blockDim.y+threadIdx.y;
    if(r<batch&&c<undim) dst[r*undim+c]=src[r*padim+c];
}

__global__ void hg_padInputFloat(const float* __restrict__ src,float* __restrict__ dst,int batch,int undim,int padim){
    int r=blockIdx.x*blockDim.x+threadIdx.x, c=blockIdx.y*blockDim.y+threadIdx.y;
    if(r<batch&&c<padim) dst[r*padim+c]=(c<undim)?src[r*undim+c]:0.0f;
}

__global__ void hg_f2h(const float* __restrict__ f,half* __restrict__ h,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) h[i]=__float2half(f[i]);
}

TinyMLPHashGrid::TinyMLPHashGrid(const MLPGridOptions& options, int maxBatchSize, int inferBatchSize, unsigned int seed, bool isTraining)
    : user_opt(options), m_maxBatchSize(maxBatchSize), m_inferBatchSize(inferBatchSize <= 0 ? maxBatchSize : inferBatchSize), current_step(0), m_isTraining(isTraining) {
    if(options.vectorDim != 3 && options.vectorDim != 4) throw std::runtime_error("TinyMLPHashGrid: vectorDim must be 3 or 4");
    if(options.featuresLevel != 2) throw std::runtime_error("TinyMLPHashGrid: featuresLevel must be 2");

    hw_opt = options;
    hw_opt.hiddenDim = nextPowerOf2(options.hiddenDim);
    hw_opt.outputDim = nextPowerOf2(options.outputDim);

    mlp_opt.inputDim  = hw_opt.numLevels * hw_opt.featuresLevel;
    mlp_opt.hiddenDim = hw_opt.hiddenDim;
    mlp_opt.outputDim = hw_opt.outputDim;
    mlp_opt.numLayers = hw_opt.numLayers;
    mlp_opt.activationType = hw_opt.activationType;

    m_totalHashElements = hw_opt.numLevels * hw_opt.tableSize * hw_opt.featuresLevel;

    allocate_memory();

    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> hash_dist(-0.1f, 0.1f);

    std::vector<float> h_hash(m_totalHashElements);
    for(auto& v : h_hash) v = hash_dist(gen);
    CUDA_CHECK(cudaMemcpy(d_master_hashtable, h_hash.data(), m_totalHashElements*sizeof(float), cudaMemcpyHostToDevice));
    int hblocks=(m_totalHashElements+255)/256;
    hg_f2h<<<hblocks,256>>>(d_master_hashtable, d_fwd_hashtable, m_totalHashElements);

    std::vector<float*> h_mw(mlp_opt.numLayers), h_mb(mlp_opt.numLayers);
    std::vector<half*>  h_fw(mlp_opt.numLayers);
    cudaMemcpy(h_mw.data(), d_master_weights, mlp_opt.numLayers*sizeof(float*), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mb.data(), d_master_biases,  mlp_opt.numLayers*sizeof(float*), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fw.data(), d_fwd_weights,    mlp_opt.numLayers*sizeof(half*),  cudaMemcpyDeviceToHost);

    std::uniform_real_distribution<float> w_dist;
    for(int l=0;l<mlp_opt.numLayers;l++){
        int in_d=(l==0)?mlp_opt.inputDim:mlp_opt.hiddenDim;
        int out_d=(l==mlp_opt.numLayers-1)?mlp_opt.outputDim:mlp_opt.hiddenDim;
        float bound=1.0f/std::sqrt((float)in_d);
        w_dist=std::uniform_real_distribution<float>(-bound,bound);
        int we=in_d*out_d;
        std::vector<float> hw(we), hb(out_d,0.0f);
        for(auto& v:hw) v=w_dist(gen);
        CUDA_CHECK(cudaMemcpy(h_mw[l], hw.data(), we*sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(h_mb[l], hb.data(), out_d*sizeof(float), cudaMemcpyHostToDevice));
        hg_f2h<<<(we+255)/256,256>>>(h_mw[l], h_fw[l], we);
    }
    cudaDeviceSynchronize();
}

TinyMLPHashGrid::~TinyMLPHashGrid(){
    cudaFree(d_padded_inputs);
    cudaFree(d_padded_targets);
    cudaFree(d_padded_outputs_float);
    cudaFree(d_dLoss_internal);
    cudaFree(d_total_loss);
    cudaFree(d_dx_out);
    cudaFree(d_master_hashtable);
    cudaFree(d_fwd_hashtable);
    cudaFree(d_hashtable_grads);
    cudaFree(d_hash_m);
    cudaFree(d_hash_v);
    free_pointer_array(d_master_weights, mlp_opt.numLayers);
    free_pointer_array(d_master_biases,  mlp_opt.numLayers);
    free_pointer_array(d_fwd_weights,    mlp_opt.numLayers);
    free_pointer_array(d_w_grad, mlp_opt.numLayers);
    free_pointer_array(d_b_grad, mlp_opt.numLayers);
    free_pointer_array(d_w_m,   mlp_opt.numLayers);
    free_pointer_array(d_w_v,   mlp_opt.numLayers);
    free_pointer_array(d_b_m,   mlp_opt.numLayers);
    free_pointer_array(d_b_v,   mlp_opt.numLayers);
    free_pointer_array(d_activations, mlp_opt.numLayers, false);
}

void TinyMLPHashGrid::allocate_memory(){
    int inDim   = mlp_opt.inputDim;
    int outDim  = mlp_opt.outputDim;
    int hidDim  = mlp_opt.hiddenDim;
    int nLayers = mlp_opt.numLayers;

    int max_pad = std::max(m_maxBatchSize, m_inferBatchSize);
    bool needs_in_pad = (user_opt.vectorDim != 4);
    bool needs_out_pad = (user_opt.outputDim != hw_opt.outputDim);

    if (m_isTraining || needs_in_pad) {
        if (!d_padded_inputs) CUDA_CHECK(cudaMalloc(&d_padded_inputs, (size_t)max_pad*4*sizeof(float)));
    }
    if (m_isTraining) {
        if (!d_padded_targets) CUDA_CHECK(cudaMalloc(&d_padded_targets, (size_t)m_maxBatchSize*outDim*sizeof(half)));
    }
    if (m_isTraining || needs_out_pad) {
        if (!d_padded_outputs_float) CUDA_CHECK(cudaMalloc(&d_padded_outputs_float, (size_t)max_pad*outDim*sizeof(float)));
    }
    if (m_isTraining) {
        if (!d_dLoss_internal) CUDA_CHECK(cudaMalloc(&d_dLoss_internal, (size_t)m_maxBatchSize*outDim*sizeof(half)));
        if (!d_total_loss) CUDA_CHECK(cudaMalloc(&d_total_loss, sizeof(float)));
        if (!d_dx_out) CUDA_CHECK(cudaMalloc(&d_dx_out, (size_t)m_maxBatchSize*inDim*sizeof(half)));
    }

    if (!d_master_hashtable) CUDA_CHECK(cudaMalloc(&d_master_hashtable, (size_t)m_totalHashElements*sizeof(float)));
    if (!d_fwd_hashtable) CUDA_CHECK(cudaMalloc(&d_fwd_hashtable,    (size_t)m_totalHashElements*sizeof(half)));

    if (m_isTraining) {
        if (!d_hashtable_grads) {
            CUDA_CHECK(cudaMalloc(&d_hashtable_grads,  (size_t)m_totalHashElements*sizeof(float)));
            CUDA_CHECK(cudaMemset(d_hashtable_grads, 0, (size_t)m_totalHashElements*sizeof(float)));
        }
        if (!d_hash_m) {
            CUDA_CHECK(cudaMalloc(&d_hash_m, (size_t)m_totalHashElements*sizeof(float)));
            CUDA_CHECK(cudaMemset(d_hash_m, 0, (size_t)m_totalHashElements*sizeof(float)));
        }
        if (!d_hash_v) {
            CUDA_CHECK(cudaMalloc(&d_hash_v, (size_t)m_totalHashElements*sizeof(float)));
            CUDA_CHECK(cudaMemset(d_hash_v, 0, (size_t)m_totalHashElements*sizeof(float)));
        }
    }

    auto makeZeroF = [&](bool isWeight, float**& ptr){
        if (ptr) return;
        std::vector<float*> h(nLayers);
        for(int l=0;l<nLayers;l++){
            int in_d=(l==0)?inDim:hidDim;
            int out_d=(l==nLayers-1)?outDim:hidDim;
            int n=isWeight?(in_d*out_d):out_d;
            CUDA_CHECK(cudaMalloc(&h[l], n*sizeof(float)));
            CUDA_CHECK(cudaMemset(h[l], 0, n*sizeof(float)));
        }
        float** d; CUDA_CHECK(cudaMalloc(&d, nLayers*sizeof(float*)));
        CUDA_CHECK(cudaMemcpy(d, h.data(), nLayers*sizeof(float*), cudaMemcpyHostToDevice));
        ptr = d;
    };

    if (!d_master_weights) {
        CUDA_CHECK(cudaMalloc(&d_master_weights, nLayers*sizeof(float*)));
        CUDA_CHECK(cudaMalloc(&d_master_biases,  nLayers*sizeof(float*)));
        CUDA_CHECK(cudaMalloc(&d_fwd_weights,    nLayers*sizeof(half*)));

        std::vector<float*> hw(nLayers), hb(nLayers);
        std::vector<half*>  hfw(nLayers);
        for(int l=0;l<nLayers;l++){
            int in_d=(l==0)?inDim:hidDim;
            int out_d=(l==nLayers-1)?outDim:hidDim;
            CUDA_CHECK(cudaMalloc(&hw[l],  in_d*out_d*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&hb[l],  out_d*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&hfw[l], in_d*out_d*sizeof(half)));
        }
        CUDA_CHECK(cudaMemcpy(d_master_weights, hw.data(),  nLayers*sizeof(float*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_master_biases,  hb.data(),  nLayers*sizeof(float*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_fwd_weights,    hfw.data(), nLayers*sizeof(half*),  cudaMemcpyHostToDevice));
    }

    if (m_isTraining) {
        makeZeroF(true, d_w_grad);  makeZeroF(false, d_b_grad);
        makeZeroF(true, d_w_m);     makeZeroF(false, d_b_m);
        makeZeroF(true, d_w_v);     makeZeroF(false, d_b_v);

        if (!d_activations) {
            std::vector<half*> ha(nLayers);
            CUDA_CHECK(cudaMalloc(&ha[0], (size_t)m_maxBatchSize*inDim*sizeof(half)));
            for(int l=1;l<nLayers;l++){
                CUDA_CHECK(cudaMalloc(&ha[l], (size_t)m_maxBatchSize*hidDim*sizeof(half)));
            }
            CUDA_CHECK(cudaMalloc(&d_activations, nLayers*sizeof(half*)));
            CUDA_CHECK(cudaMemcpy(d_activations, ha.data(), nLayers*sizeof(half*), cudaMemcpyHostToDevice));
        }
    }
}

void TinyMLPHashGrid::switchToInferenceMode() {
    if (!m_isTraining) return;

    if (user_opt.vectorDim == 4 && d_padded_inputs) {
        cudaFree(d_padded_inputs); d_padded_inputs = nullptr;
    }
    if (d_padded_targets) { cudaFree(d_padded_targets); d_padded_targets = nullptr; }
    if (user_opt.outputDim == hw_opt.outputDim && d_padded_outputs_float) {
        cudaFree(d_padded_outputs_float); d_padded_outputs_float = nullptr;
    }
    if (d_dLoss_internal) { cudaFree(d_dLoss_internal); d_dLoss_internal = nullptr; }
    if (d_total_loss) { cudaFree(d_total_loss); d_total_loss = nullptr; }
    if (d_dx_out) { cudaFree(d_dx_out); d_dx_out = nullptr; }

    if (d_hashtable_grads) { cudaFree(d_hashtable_grads); d_hashtable_grads = nullptr; }
    if (d_hash_m) { cudaFree(d_hash_m); d_hash_m = nullptr; }
    if (d_hash_v) { cudaFree(d_hash_v); d_hash_v = nullptr; }

    free_pointer_array(d_w_grad, mlp_opt.numLayers); d_w_grad = nullptr;
    free_pointer_array(d_b_grad, mlp_opt.numLayers); d_b_grad = nullptr;
    free_pointer_array(d_w_m, mlp_opt.numLayers); d_w_m = nullptr;
    free_pointer_array(d_w_v, mlp_opt.numLayers); d_w_v = nullptr;
    free_pointer_array(d_b_m, mlp_opt.numLayers); d_b_m = nullptr;
    free_pointer_array(d_b_v, mlp_opt.numLayers); d_b_v = nullptr;

    free_pointer_array(d_activations, mlp_opt.numLayers, false); d_activations = nullptr;

    m_isTraining = false;
}

void TinyMLPHashGrid::switchToTrainingMode() {
    if (m_isTraining) return;
    m_isTraining = true;
    allocate_memory();
}

void TinyMLPHashGrid::zero_grad(cudaStream_t stream){
    if (!m_isTraining) throw std::runtime_error("zero_grad called while in inference mode!");
    launchZeroGradients(&mlp_opt, d_w_grad, d_b_grad, stream);
    // Hash table gradients are zeroed inside the optimizer kernel after reading (fused zero_grad)
}

void TinyMLPHashGrid::forward(const float* d_unpadded_inputs, float* d_outputs, int batchSize, cudaStream_t stream){
    if (!m_isTraining) throw std::runtime_error("forward called while in inference mode!");
    if(batchSize>m_maxBatchSize) throw std::runtime_error("batchSize exceeds maxBatchSize");

    const float* input_ptr = d_unpadded_inputs;
    if (user_opt.vectorDim == 3) {
        dim3 blk(16,16);
        dim3 grid_in((batchSize+15)/16, (4+15)/16);
        hg_padInputFloat<<<grid_in,blk,0,stream>>>(d_unpadded_inputs, d_padded_inputs, batchSize, 3, 4);
        input_ptr = d_padded_inputs;
    }
    
    d_current_inputs_backward = input_ptr;

    launchNetworkFusionHashTableGradKernel(&hw_opt, d_fwd_hashtable, const_cast<float*>(input_ptr),
        d_activations, d_fwd_weights, d_master_biases, d_padded_outputs_float, batchSize, stream);

    if (d_outputs != nullptr) {
        if (user_opt.outputDim == hw_opt.outputDim) {
            CUDA_CHECK(cudaMemcpyAsync(d_outputs, d_padded_outputs_float, batchSize * hw_opt.outputDim * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        } else {
            dim3 blk(16,16);
            dim3 grid_out((batchSize+15)/16,(user_opt.outputDim+15)/16);
            hg_unpadFloat<<<grid_out,blk,0,stream>>>(
                d_padded_outputs_float, d_outputs, batchSize, hw_opt.outputDim, user_opt.outputDim);
        }
    }
}

void TinyMLPHashGrid::inference(const float* d_unpadded_inputs, float* d_outputs, int batchSize, cudaStream_t stream){
    if(batchSize>m_inferBatchSize) {
        for (int offset = 0; offset < batchSize; offset += m_inferBatchSize) {
            int current_batch = std::min(m_inferBatchSize, batchSize - offset);
            const float* current_in = d_unpadded_inputs + offset * user_opt.vectorDim;
            float* current_out = (d_outputs != nullptr) ? (d_outputs + offset * user_opt.outputDim) : nullptr;
            inference(current_in, current_out, current_batch, stream);
        }
        return;
    }

    const float* input_ptr = d_unpadded_inputs;
    if (user_opt.vectorDim == 3) {
        dim3 blk(16,16);
        dim3 grid_in((batchSize+15)/16,(4+15)/16);
        hg_padInputFloat<<<grid_in,blk,0,stream>>>(d_unpadded_inputs, d_padded_inputs, batchSize, 3, 4);
        input_ptr = d_padded_inputs;
    }

    if (user_opt.outputDim == hw_opt.outputDim) {
        launchNetworkFusionHashTableKernel(&hw_opt, d_fwd_hashtable, const_cast<float*>(input_ptr),
            d_fwd_weights, d_master_biases, d_outputs, batchSize, stream);
    } else {
        launchNetworkFusionHashTableKernel(&hw_opt, d_fwd_hashtable, const_cast<float*>(input_ptr),
            d_fwd_weights, d_master_biases, d_padded_outputs_float, batchSize, stream);
        
        if (d_outputs != nullptr) {
            dim3 blk(16,16);
            dim3 grid_out((batchSize+15)/16,(user_opt.outputDim+15)/16);
            hg_unpadFloat<<<grid_out,blk,0,stream>>>(
                d_padded_outputs_float, d_outputs, batchSize, hw_opt.outputDim, user_opt.outputDim);
        }
    }
}

float TinyMLPHashGrid::calculate_loss_and_grad(const half* d_unpadded_targets, int batchSize, float loss_scale, bool fetch_loss, cudaStream_t stream){
    if (!m_isTraining) throw std::runtime_error("calculate_loss_and_grad called while in inference mode!");
    CUDA_CHECK(cudaMemsetAsync(d_total_loss, 0, sizeof(float), stream));

    dim3 blk(16,16);
    dim3 grid((batchSize+15)/16,(hw_opt.outputDim+15)/16);

    const half* final_targets = d_unpadded_targets;
    if(user_opt.outputDim!=hw_opt.outputDim){
        hg_padHalfToHalf<<<grid,blk,0,stream>>>(
            d_unpadded_targets, d_padded_targets, batchSize, user_opt.outputDim, hw_opt.outputDim);
        final_targets = d_padded_targets;
    }

    launchMSELossGrad(d_padded_outputs_float, final_targets, d_dLoss_internal, d_total_loss,
        batchSize, hw_opt.outputDim, user_opt.outputDim, loss_scale, stream);

    if(fetch_loss){
        float h_loss;
        CUDA_CHECK(cudaMemcpyAsync(&h_loss, d_total_loss, sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
        return h_loss/(batchSize*user_opt.outputDim);
    }
    return 0.0f;
}

void TinyMLPHashGrid::backward(int batchSize, cudaStream_t stream){
    if (!m_isTraining) throw std::runtime_error("backward called while in inference mode!");
    launchNetworkFusionHashTableBackwardKernel(
        &hw_opt, d_dLoss_internal, d_fwd_weights, d_master_biases,
        d_activations, const_cast<float*>(d_current_inputs_backward), d_w_grad, d_b_grad,
        d_hashtable_grads, d_dx_out, batchSize, stream);
}

void TinyMLPHashGrid::backward(const half* custom_loss_grad, int batchSize, cudaStream_t stream){
    if (!m_isTraining) throw std::runtime_error("backward called while in inference mode!");
    if(user_opt.outputDim!=hw_opt.outputDim){
        dim3 blk(16,16);
        dim3 grid((batchSize+15)/16,(hw_opt.outputDim+15)/16);
        hg_padHalfToHalf<<<grid,blk,0,stream>>>(custom_loss_grad, d_dLoss_internal, batchSize, user_opt.outputDim, hw_opt.outputDim);
    } else {
        CUDA_CHECK(cudaMemcpyAsync(d_dLoss_internal, custom_loss_grad, batchSize * hw_opt.outputDim * sizeof(half), cudaMemcpyDeviceToDevice, stream));
    }
    launchNetworkFusionHashTableBackwardKernel(
        &hw_opt, d_dLoss_internal, d_fwd_weights, d_master_biases,
        d_activations, const_cast<float*>(d_current_inputs_backward), d_w_grad, d_b_grad,
        d_hashtable_grads, d_dx_out, batchSize, stream);
}

void TinyMLPHashGrid::reset_step(){ current_step=0; }

void TinyMLPHashGrid::step(float lr, float beta1, float beta2, float epsilon, float loss_scale, cudaStream_t stream){
    if (!m_isTraining) throw std::runtime_error("step called while in inference mode!");
    if(current_step<50000) current_step++;
    float bc1=1.0f-std::pow(beta1,(float)current_step);
    float bc2=1.0f-std::pow(beta2,(float)current_step);
    float ils=1.0f/loss_scale;

    launchAdamWeightsOptim(&mlp_opt, d_master_weights, d_fwd_weights, d_w_grad, d_w_m, d_w_v,
        lr,beta1,beta2,epsilon,bc1,bc2,ils,stream);
    launchAdamBiasOptim(&mlp_opt, d_master_biases, d_b_grad, d_b_m, d_b_v,
        lr,beta1,beta2,epsilon,bc1,bc2,ils,stream);
    launchAdamHashGridOptim(&hw_opt, d_master_hashtable, d_fwd_hashtable, d_hashtable_grads,
        d_hash_m, d_hash_v, lr,beta1,beta2,epsilon,bc1,bc2,ils,stream);
}

void TinyMLPHashGrid::loadWeights(const std::string& filename){
    std::ifstream in(filename, std::ios::binary);
    if(!in.is_open()) throw std::runtime_error("Cannot open: "+filename);
    
    MLPGridOptions fopt;
    in.read(reinterpret_cast<char*>(&fopt),sizeof(MLPGridOptions));
    
    if (fopt.numLevels != hw_opt.numLevels || fopt.tableSize != hw_opt.tableSize || fopt.featuresLevel != hw_opt.featuresLevel ||
        fopt.hiddenDim != hw_opt.hiddenDim || fopt.outputDim != hw_opt.outputDim || fopt.numLayers != hw_opt.numLayers) {
        throw std::runtime_error("Weight file dimension mismatch");
    }

    std::vector<float> h_hash(m_totalHashElements);
    in.read(reinterpret_cast<char*>(h_hash.data()), m_totalHashElements * sizeof(float));
    CUDA_CHECK(cudaMemcpy(d_master_hashtable, h_hash.data(), m_totalHashElements * sizeof(float), cudaMemcpyHostToDevice));
    
    int hblocks = (m_totalHashElements + 255) / 256;
    hg_f2h<<<hblocks, 256>>>(d_master_hashtable, d_fwd_hashtable, m_totalHashElements);

    std::vector<float*> hmw(mlp_opt.numLayers),hmb(mlp_opt.numLayers);
    std::vector<half*>  hfw(mlp_opt.numLayers);
    cudaMemcpy(hmw.data(),d_master_weights,mlp_opt.numLayers*sizeof(float*),cudaMemcpyDeviceToHost);
    cudaMemcpy(hmb.data(),d_master_biases, mlp_opt.numLayers*sizeof(float*),cudaMemcpyDeviceToHost);
    cudaMemcpy(hfw.data(),d_fwd_weights,   mlp_opt.numLayers*sizeof(half*), cudaMemcpyDeviceToHost);
    for(int l=0;l<mlp_opt.numLayers;l++){
        int in_d=(l==0)?mlp_opt.inputDim:mlp_opt.hiddenDim;
        int out_d=(l==mlp_opt.numLayers-1)?mlp_opt.outputDim:mlp_opt.hiddenDim;
        int we=in_d*out_d;
        std::vector<float> w(we),b(out_d);
        in.read(reinterpret_cast<char*>(w.data()),we*sizeof(float));
        in.read(reinterpret_cast<char*>(b.data()),out_d*sizeof(float));
        CUDA_CHECK(cudaMemcpy(hmw[l],w.data(),we*sizeof(float),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(hmb[l],b.data(),out_d*sizeof(float),cudaMemcpyHostToDevice));
        hg_f2h<<<(we+255)/256,256>>>(hmw[l],hfw[l],we);
    }
    cudaDeviceSynchronize();
}

void TinyMLPHashGrid::saveWeights(const std::string& filename){
    std::ofstream out(filename,std::ios::binary);
    if(!out.is_open()) throw std::runtime_error("Cannot open: "+filename);
    
    out.write(reinterpret_cast<const char*>(&hw_opt),sizeof(MLPGridOptions));

    std::vector<float> h_hash(m_totalHashElements);
    cudaMemcpy(h_hash.data(), d_master_hashtable, m_totalHashElements * sizeof(float), cudaMemcpyDeviceToHost);
    out.write(reinterpret_cast<const char*>(h_hash.data()), m_totalHashElements * sizeof(float));

    std::vector<float*> hmw(mlp_opt.numLayers),hmb(mlp_opt.numLayers);
    cudaMemcpy(hmw.data(),d_master_weights,mlp_opt.numLayers*sizeof(float*),cudaMemcpyDeviceToHost);
    cudaMemcpy(hmb.data(),d_master_biases, mlp_opt.numLayers*sizeof(float*),cudaMemcpyDeviceToHost);
    for(int l=0;l<mlp_opt.numLayers;l++){
        int in_d=(l==0)?mlp_opt.inputDim:mlp_opt.hiddenDim;
        int out_d=(l==mlp_opt.numLayers-1)?mlp_opt.outputDim:mlp_opt.hiddenDim;
        int we=in_d*out_d;
        std::vector<float> w(we),b(out_d);
        cudaMemcpy(w.data(),hmw[l],we*sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(b.data(),hmb[l],out_d*sizeof(float),cudaMemcpyDeviceToHost);
        out.write(reinterpret_cast<const char*>(w.data()),we*sizeof(float));
        out.write(reinterpret_cast<const char*>(b.data()),out_d*sizeof(float));
    }
}

void TinyMLPHashGrid::loadWeights(const float* hw, const float* hb){
    int wo=0,bo=0;
    std::vector<float*> hmw(mlp_opt.numLayers),hmb(mlp_opt.numLayers);
    std::vector<half*>  hfw(mlp_opt.numLayers);
    cudaMemcpy(hmw.data(),d_master_weights,mlp_opt.numLayers*sizeof(float*),cudaMemcpyDeviceToHost);
    cudaMemcpy(hmb.data(),d_master_biases, mlp_opt.numLayers*sizeof(float*),cudaMemcpyDeviceToHost);
    cudaMemcpy(hfw.data(),d_fwd_weights,   mlp_opt.numLayers*sizeof(half*), cudaMemcpyDeviceToHost);
    for(int l=0;l<mlp_opt.numLayers;l++){
        int in_d=(l==0)?mlp_opt.inputDim:mlp_opt.hiddenDim;
        int out_d=(l==mlp_opt.numLayers-1)?mlp_opt.outputDim:mlp_opt.hiddenDim;
        int we=in_d*out_d;
        CUDA_CHECK(cudaMemcpy(hmw[l],hw+wo,we*sizeof(float),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(hmb[l],hb+bo,out_d*sizeof(float),cudaMemcpyHostToDevice));
        hg_f2h<<<(we+255)/256,256>>>(hmw[l],hfw[l],we);
        wo+=we; bo+=out_d;
    }
    cudaDeviceSynchronize();
}

void TinyMLPHashGrid::saveWeights(float* hw, float* hb){
    int wo=0,bo=0;
    std::vector<float*> hmw(mlp_opt.numLayers),hmb(mlp_opt.numLayers);
    cudaMemcpy(hmw.data(),d_master_weights,mlp_opt.numLayers*sizeof(float*),cudaMemcpyDeviceToHost);
    cudaMemcpy(hmb.data(),d_master_biases, mlp_opt.numLayers*sizeof(float*),cudaMemcpyDeviceToHost);
    for(int l=0;l<mlp_opt.numLayers;l++){
        int in_d=(l==0)?mlp_opt.inputDim:mlp_opt.hiddenDim;
        int out_d=(l==mlp_opt.numLayers-1)?mlp_opt.outputDim:mlp_opt.hiddenDim;
        int we=in_d*out_d;
        CUDA_CHECK(cudaMemcpy(hw+wo,hmw[l],we*sizeof(float),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hb+bo,hmb[l],out_d*sizeof(float),cudaMemcpyDeviceToHost));
        wo+=we; bo+=out_d;
    }
}


void TinyMLPHashGrid::loadHashgrid(const float* host_hashgrid){
    CUDA_CHECK(cudaMemcpy(d_master_hashtable, host_hashgrid, m_totalHashElements * sizeof(float), cudaMemcpyHostToDevice));
    int hblocks = (m_totalHashElements + 255) / 256;
    hg_f2h<<<hblocks, 256>>>(d_master_hashtable, d_fwd_hashtable, m_totalHashElements);
    cudaDeviceSynchronize();
}

void TinyMLPHashGrid::saveHashgrid(float* host_hashgrid){
    CUDA_CHECK(cudaMemcpy(host_hashgrid, d_master_hashtable, m_totalHashElements * sizeof(float), cudaMemcpyDeviceToHost));
}

void TinyMLPHashGrid::free_pointer_array(float** d_arr,int count){
    if(!d_arr) return;
    std::vector<float*> h(count);
    cudaMemcpy(h.data(),d_arr,count*sizeof(float*),cudaMemcpyDeviceToHost);
    for(int i=0;i<count;i++) cudaFree(h[i]);
    cudaFree(d_arr);
}

void TinyMLPHashGrid::free_pointer_array(half** d_arr,int count,bool skip_first){
    if(!d_arr) return;
    std::vector<half*> h(count);
    cudaMemcpy(h.data(),d_arr,count*sizeof(half*),cudaMemcpyDeviceToHost);
    for(int i=(skip_first?1:0);i<count;i++) cudaFree(h[i]);
    cudaFree(d_arr);
}
