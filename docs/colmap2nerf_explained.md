Looked at render_rays_distortion_kernel. The structure is right and the prefix-sum idea is exactly the O(N) trick from the paper — but there are a few correctness issues, one of them significant. Let me walk through them.

What the loss/gradient actually is

mip-NeRF 360 distortion loss (per ray):

$$\mathcal{L}{dist} = \underbrace{\sum{i,j} w_i w_j,|m_i - m_j|}{\text{cross}} ;+; \underbrace{\tfrac{1}{3}\sum_i w_i^2, d_i}{\text{self}}, \qquad m_i = t_i + \tfrac{d_i}{2}$$

You only want $\partial \mathcal{L}/\partial w_k$:

$$\frac{\partial \mathcal{L}}{\partial w_k} = \lambda\left[,2\sum_j w_j,|m_k - m_j| ;+; \tfrac{2}{3} w_k d_k,\right]$$

Issue 1 — you only compute the below-k half of the cross term (the main bug)

Your line:
dw_out[offset+i] = 2*lambda_dist*(m*dw_sum - dw_sum2) + c*weight*delta_t;
At step k, dw_sum = Σ_{j<k} w_j and dw_sum2 = Σ_{j<k} w_j m_j (prefix, before the update). So m*dw_sum - dw_sum2 = Σ_{j<k} w_j(m_k - m_j) — that's only the j<k part. The full gradient needs all j:

$$2\sum_j w_j|m_k-m_j| = 2\Big[\underbrace{\textstyle\sum_{j<k} w_j(m_k-m_j)}{\text{you have this}} + \underbrace{\textstyle\sum_{j>k} w_j(m_j-m_k)}{\text{missing}}\Big]$$

As written, every sample only "feels" the samples in front of it — the gradient is asymmetric and biased. You can get the missing half with the per-ray totals $W=\sum w_j$, $WM=\sum w_j m_j$ (sorted $m$):

$$\sum_j w_j|m_k-m_j| = 2\big(m_k W_{<k} - WM_{<k}\big) + \big(WM - m_k W\big)$$

That needs the totals first, so it's two sweeps (the second has no MLP, just re-reads sigma — cheap).

Issue 2 — lambda_dist isn't applied to the self term

2*lambda_dist*(...) scales only the cross term; c*weight*delta_t is added raw. Both terms must be ×lambda_dist or their balance is wrong (with a small λ the self term dominates by orders of magnitude). (c = 2.0f/3 is numerically fine, just not scaled.)

Issue 3 — m uses raw t, not normalized s

The paper's loss is defined on normalized distance $s\in[0,1]$ (the $[t_n,t_f]$ / contraction mapping). With raw t, the cross and self terms aren't commensurate and λ's meaning changes per ray. At minimum normalize by the ray's $(t_f-t_n)$; otherwise far samples get over-penalized. Not catastrophic, but it's "not the paper's loss."

Issue 4 — the kernel drops the photometric outputs

It computes r_c,g_c,b_c,depth but never writes final_rgb/final_depth/phi_out, has no bg composite, no early-stop — and launchVolumeRendering still calls only render_rays_kernel. If the intent is fusion, do the color render and the distortion totals in the same forward pass (reuse the weights you already compute), then one extra weight-only sweep for dw_out.

Corrected fused kernel

  Read 1 file

Here's the corrected version. It does color + distortion totals in one forward sweep, stashes each weight into dw_out as scratch to avoid a second expf, then a cheap second sweep finalizes the gradient with both halves:

__global__ void render_rays_distortion_kernel(
    const uint32_t num_rays,
    const uint32_t* __restrict__ ray_offsets,
    const uint32_t* __restrict__ num_steps,
    const float* __restrict__ t_sorted,
    const float* __restrict__ density_sigma,
    const float* __restrict__ rgb_output,
    const float* __restrict__ rgb_true,
    float* __restrict__ final_rgb,
    float* __restrict__ final_depth,
    float* __restrict__ phi_out,
    float* __restrict__ dw_out,
    float lambda_dist,
    float3 bg_color
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= num_rays) return;

    uint32_t offset = ray_offsets[r];
    uint32_t count  = num_steps[r];

    // ---- Pass 1: color render + accumulate per-ray distortion totals ----
    float T = 1.0f;j
    float r_c = 0.0f, g_c = 0.0f, b_c = 0.0f, depth = 0.0f;
    float W = 0.0f, WM = 0.0f;                 // total weight, total weight*midpoint
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t idx = offset + i;
        float t = t_sorted[idx];
        float delta_t = (i < count - 1) ? (t_sorted[idx + 1] - t) : 1e-3f;
        if (delta_t < 0.0f) delta_t = 0.0f;
        float m = t + 0.5f * delta_t;

        float sigma  = density_sigma[idx];
        float alpha  = 1.0f - expf(-sigma * delta_t);
        float weight = alpha * T;

        r_c += weight * rgb_output[idx * 3 + 0];
        g_c += weight * rgb_output[idx * 3 + 1];
        b_c += weight * rgb_output[idx * 3 + 2];
        depth += weight * t;

        W  += weight;
        WM += weight * m;
        dw_out[idx] = weight;                  // stash weight; pass 2 reads it back
        T *= (1.0f - alpha);
    }

    // composite background + write photometric outputs (same as render_rays_kernel)
    r_c += T * bg_color.x;  g_c += T * bg_color.y;  b_c += T * bg_color.z;
    final_rgb[r*3+0] = r_c;  final_rgb[r*3+1] = g_c;  final_rgb[r*3+2] = b_c;
    final_depth[r] = depth;
    if (rgb_true != nullptr && phi_out != nullptr) {
        phi_out[r*3+0] = 2.0f * (r_c - rgb_true[r*3+0]);
        phi_out[r*3+1] = 2.0f * (g_c - rgb_true[r*3+1]);
        phi_out[r*3+2] = 2.0f * (b_c - rgb_true[r*3+2]);
    }

    // ---- Pass 2: dL/dw_k = lambda*( 2*sum_j w_j|m_k-m_j| + (2/3) w_k d_k ) ----
    //   sum_j w_j|m_k-m_j| = 2(m_k W_<k - WM_<k) + (WM - m_k W)   (m sorted)
    const float c = 2.0f / 3.0f;
    float Wpre = 0.0f, WMpre = 0.0f;           // prefix sums for j<k
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t idx = offset + i;
        float t = t_sorted[idx];
        float delta_t = (i < count - 1) ? (t_sorted[idx + 1] - t) : 1e-3f;
        if (delta_t < 0.0f) delta_t = 0.0f;
        float m = t + 0.5f * delta_t;

        float weight  = dw_out[idx];           // stashed in pass 1
        float abs_sum = 2.0f * (m * Wpre - WMpre) + (WM - m * W);
        dw_out[idx]   = lambda_dist * (2.0f * abs_sum + c * weight * delta_t);

        Wpre  += weight;
        WMpre += weight * m;
    }
}

I verified abs_sum on a 2-sample ray by hand: for both k=0 and k=1 it reduces to w_other·|m₁−m₀|, which is correct.

One thing this doesn't finish: dw_out is dL/dw, not dL/dσ

dw_out[k] is the gradient w.r.t. the weight w_k = α_k·T_k. Since w_k depends on σ_k and every earlier σ_j (through T), it still has to be backpropagated into the density gradient — the same weight→σ chain your color path already does from phi_out. Until that's wired in (likely folded into the existing density-gradient kernel around otherKernels.cu:398), dw_out won't actually affect training. That's the next piece, and it's the non-trivial one — happy to derive/implement it.

Want me to:
1. apply this corrected kernel to compositing.cu, and
2. work out the dL/dw → dL/dσ backward so it actually trains?

(Also note: the distortion m should ideally be normalized to s∈[0,1] — issue 3 — but that can come after the gradient path is correct.)