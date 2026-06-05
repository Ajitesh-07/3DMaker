#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "TinyMLP.h"
#include "TinyMLPHashGrid.h"

__global__ void fusedAdamWeightsOptim(
    float** __restrict__ master_weights,
    half** __restrict__ fwd_weights,
    const float** __restrict__ gradients, 
    float** __restrict__ m,
    float** __restrict__ v,
    int inputDim,
    int hiddenDim,
    int outputDim,
    int numLayers,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    float bias_correction1,
    float bias_correction2,
    float inv_loss_scale
) {
    int layer = blockIdx.y;
    int in_d = (layer == 0) ? inputDim : hiddenDim;
    int out_d = (layer == numLayers - 1) ? outputDim : hiddenDim;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int numElements = in_d * out_d;

    if (idx < numElements) {
        float g = gradients[layer][idx] * inv_loss_scale;
        float m_val = m[layer][idx];
        float v_val = v[layer][idx];
        float w_val = master_weights[layer][idx]; 

        m_val = beta1 * m_val + (1.0f - beta1) * g;
        v_val = beta2 * v_val + (1.0f - beta2) * g * g;

        float m_hat = m_val / bias_correction1;
        float v_hat = v_val / bias_correction2;

        w_val = w_val - (lr * m_hat / (sqrtf(v_hat) + epsilon));

        m[layer][idx] = m_val;
        v[layer][idx] = v_val;
        master_weights[layer][idx] = w_val;
        fwd_weights[layer][idx] = __float2half(w_val);
    }
}

__global__ void fusedAdamBiasOptim(
    float** __restrict__ biases,
    const float** __restrict__ gradients, 
    float** __restrict__ m,
    float** __restrict__ v,
    int hiddenDim,
    int outputDim,
    int numLayers,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    float bias_correction1,
    float bias_correction2,
    float inv_loss_scale
) {
    // Y-dimension maps the layer
    int layer = blockIdx.y;
    int out_d = (layer == numLayers - 1) ? outputDim : hiddenDim;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < out_d) {
        // 1. Read everything and unscale gradient
        float g = gradients[layer][idx] * inv_loss_scale;
        float m_val = m[layer][idx];
        float v_val = v[layer][idx];
        float b_val = biases[layer][idx]; 

        // 2. Update biased moments
        m_val = beta1 * m_val + (1.0f - beta1) * g;
        v_val = beta2 * v_val + (1.0f - beta2) * g * g;

        // 3. Apply bias correction
        float m_hat = m_val / bias_correction1;
        float v_hat = v_val / bias_correction2;

        // 4. Update bias in pure FP32
        b_val = b_val - (lr * m_hat / (sqrtf(v_hat) + epsilon));

        // 5. Write back to global memory
        m[layer][idx] = m_val;
        v[layer][idx] = v_val;
        biases[layer][idx] = b_val; 
    }
}


__global__ void fusedAdamHashGridOptim(
    float* __restrict__ master_hash,
    half*  __restrict__ fwd_hash,
    float* __restrict__ gradients,      // non-const: we zero after reading
    float* __restrict__ m,
    float* __restrict__ v,
    int numLevels,
    int tableSize,
    int FEATURES_PER_LEVEL,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    float bias_correction1,
    float bias_correction2,
    float inv_loss_scale
) {
    int level = blockIdx.y;
    if (level >= numLevels) return;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int levelElements = tableSize * FEATURES_PER_LEVEL;
    if (idx >= levelElements) return;

    int flat = level * levelElements + idx;

    // Read gradient and immediately zero it (fused zero_grad)
    float g_raw = gradients[flat];
    if (g_raw == 0.0f) return;   // Nothing to do — skip m/v/master reads entirely
    gradients[flat] = 0.0f;      // Zero for next step (eliminates separate memset)

    float g = g_raw * inv_loss_scale;

    float m_val = beta1 * m[flat] + (1.0f - beta1) * g;
    float v_val = beta2 * v[flat] + (1.0f - beta2) * g * g;

    float m_hat = m_val / bias_correction1;
    float v_hat = v_val / bias_correction2;

    float w_val = master_hash[flat] - lr * m_hat / (sqrtf(v_hat) + epsilon);

    m[flat]           = m_val;
    v[flat]           = v_val;
    master_hash[flat] = w_val;
    fwd_hash[flat]    = __float2half(w_val);
}

template <int OUT_ACT = 0>
__global__ void fusedMSELossGrad_Kernel(
    const float* __restrict__ predictions, 
    const half* __restrict__ targets,     
    half* __restrict__ dLoss_out,         
    float* __restrict__ total_loss,       
    int batchSize,
    int padded_dim,
    int valid_dim,
    float grad_scale      
) {
    // We process 2 elements per thread using half2
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    int num_elements = batchSize * padded_dim;
    int stride = blockDim.x * gridDim.x * 2;

    float local_loss_sum = 0.0f;

    for (int i = idx; i < num_elements; i += stride) {
        // Calculate which specific feature columns we are looking at (0 to 7)
        int col_idx_1 = i % padded_dim;
        int col_idx_2 = col_idx_1 + 1;

        float2 p_f2 = *(reinterpret_cast<const float2*>(&predictions[i]));
        half2 t2 = *(reinterpret_cast<const half2*>(&targets[i]));

        float2 t_f2 = __half22float2(t2);

        // GRADIENT MASKING LOGIC
        float diff_x = (col_idx_1 < valid_dim) ? (p_f2.x - t_f2.x) : 0.0f;
        float diff_y = (col_idx_2 < valid_dim) ? (p_f2.y - t_f2.y) : 0.0f;

        local_loss_sum += (diff_x * diff_x) + (diff_y * diff_y);

        float grad_x = diff_x * grad_scale;
        float grad_y = diff_y * grad_scale;

        // Apply sigmoid derivative: dL/dz = dL/dy * y * (1 - y)
        if constexpr (OUT_ACT == 1) {
            grad_x *= p_f2.x * (1.0f - p_f2.x);
            grad_y *= p_f2.y * (1.0f - p_f2.y);
        }
        
        half2 grad_h2 = __floats2half2_rn(grad_x, grad_y);
        *(reinterpret_cast<half2*>(&dLoss_out[i])) = grad_h2;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_loss_sum += __shfl_down_sync(0xFFFFFFFF, local_loss_sum, offset);
    }

    if (threadIdx.x % 32 == 0) {
        atomicAdd(total_loss, local_loss_sum);
    }
}

__global__ void zeroGradientsKernel(float** w_grads, float** b_grads, int inputDim, int hiddenDim, int outputDim, int numLayers) {
    int layer = blockIdx.y;
    int in_d = (layer == 0) ? inputDim : hiddenDim;
    int out_d = (layer == numLayers - 1) ? outputDim : hiddenDim;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < in_d * out_d) {
        w_grads[layer][idx] = 0.0f;
    }
    if (idx < out_d) {
        b_grads[layer][idx] = 0.0f;
    }
}

void launchAdamWeightsOptim(
    MLPOption*   opt,
    float**      master_weights,
    half**       fwd_weights,
    float**      gradients,
    float**      m,
    float**      v,
    float        lr,
    float        beta1,
    float        beta2,
    float        epsilon,
    float        bias_correction1,
    float        bias_correction2,
    float        inv_loss_scale,
    cudaStream_t stream
) {

    // Find max weight element count across layers for uniform grid
    int maxElements = 0;
    for (int l = 0; l < opt->numLayers; l++) {
        int in_d  = (l == 0) ? opt->inputDim : opt->hiddenDim;
        int out_d = (l == opt->numLayers - 1) ? opt->outputDim : opt->hiddenDim;
        int n = in_d * out_d;
        if (n > maxElements) maxElements = n;
    }

    int threadsPerBlock = 256;
    int blocksX = (maxElements + threadsPerBlock - 1) / threadsPerBlock;
    dim3 grid(blocksX, opt->numLayers, 1);
    dim3 block(threadsPerBlock, 1, 1);

    fusedAdamWeightsOptim<<<grid, block, 0, stream>>>(
        master_weights, fwd_weights, (const float**)gradients, m, v,
        opt->inputDim, opt->hiddenDim, opt->outputDim, opt->numLayers,
        lr, beta1, beta2, epsilon, bias_correction1, bias_correction2,
        inv_loss_scale
    );
}

void launchAdamBiasOptim(
    MLPOption*   opt,
    float**      biases,
    float**      gradients,
    float**      m,
    float**      v,
    float        lr,
    float        beta1,
    float        beta2,
    float        epsilon,
    float        bias_correction1,
    float        bias_correction2,
    float        inv_loss_scale,
    cudaStream_t stream
) {
    int maxDim = (opt->hiddenDim > opt->outputDim) ? opt->hiddenDim : opt->outputDim;

    int threadsPerBlock = 256;
    int blocksX = (maxDim + threadsPerBlock - 1) / threadsPerBlock;
    dim3 grid(blocksX, opt->numLayers, 1);
    dim3 block(threadsPerBlock, 1, 1);

    fusedAdamBiasOptim<<<grid, block, 0, stream>>>(
        biases, (const float**)gradients, m, v,
        opt->hiddenDim, opt->outputDim, opt->numLayers,
        lr, beta1, beta2, epsilon, bias_correction1, bias_correction2,
        inv_loss_scale
    );
}

void launchAdamHashGridOptim(
    MLPGridOptions* opt,
    float*          master_hash,
    half*           fwd_hash,
    float*          gradients,          // non-const: fused zero_grad
    float*          m,
    float*          v,
    float           lr,
    float           beta1,
    float           beta2,
    float           epsilon,
    float           bias_correction1,
    float           bias_correction2,
    float           inv_loss_scale,
    cudaStream_t    stream
) {
    int levelElements    = opt->tableSize * opt->featuresLevel;
    int threadsPerBlock  = 256;
    int blocksX          = (levelElements + threadsPerBlock - 1) / threadsPerBlock;

    dim3 grid(blocksX, opt->numLevels, 1);
    dim3 block(threadsPerBlock, 1, 1);

    fusedAdamHashGridOptim<<<grid, block, 0, stream>>>(
        master_hash, fwd_hash, gradients, m, v,
        opt->numLevels,
        opt->tableSize,
        opt->featuresLevel,
        lr, beta1, beta2, epsilon,
        bias_correction1, bias_correction2,
        inv_loss_scale
    );
}

void launchZeroGradients(
    MLPOption*   opt,
    float**      w_grads,
    float**      b_grads,
    cudaStream_t stream
) {
    int maxElements = 0;
    for (int l = 0; l < opt->numLayers; l++) {
        int in_d  = (l == 0) ? opt->inputDim : opt->hiddenDim;
        int out_d = (l == opt->numLayers - 1) ? opt->outputDim : opt->hiddenDim;
        int n = in_d * out_d;
        if (n > maxElements) maxElements = n;
    }

    int threadsPerBlock = 256;
    int blocksX = (maxElements + threadsPerBlock - 1) / threadsPerBlock;
    dim3 grid(blocksX, opt->numLayers, 1);
    dim3 block(threadsPerBlock, 1, 1);

    zeroGradientsKernel<<<grid, block, 0, stream>>>(
        w_grads, b_grads,
        opt->inputDim, opt->hiddenDim, opt->outputDim, opt->numLayers
    );
}

void launchMSELossGrad(
    const float* predictions,
    const half*  targets,
    half*        dLoss_out,
    float*       total_loss,
    int          batchSize,
    int          paddedDim,
    int          validDim,
    float        loss_scale,
    cudaStream_t stream,
    int          outputActivation
) {
    int num_elements = batchSize * paddedDim;
    // Each thread processes 2 elements (half2)
    int threadsPerBlock = 256;
    int numThreads = (num_elements + 1) / 2;
    int blocks = (numThreads + threadsPerBlock - 1) / threadsPerBlock;

    // Best Practice: Mean over all elements
    float grad_scale = (2.0f / (float)(batchSize * validDim)) * loss_scale;

    if (outputActivation == OUT_ACT_SIGMOID) {
        fusedMSELossGrad_Kernel<1><<<blocks, threadsPerBlock, 0, stream>>>(
            predictions, targets, dLoss_out, total_loss,
            batchSize, paddedDim, validDim, grad_scale
        );
    } else {
        fusedMSELossGrad_Kernel<0><<<blocks, threadsPerBlock, 0, stream>>>(
            predictions, targets, dLoss_out, total_loss,
            batchSize, paddedDim, validDim, grad_scale
        );
    }
}
