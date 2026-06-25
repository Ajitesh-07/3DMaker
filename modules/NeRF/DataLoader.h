#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>

class DataLoader {
public:
    // derive_bounds: when true, run the ray-convergence center + recenter/scale + per-axis AABB
    // derivation. When false, process the dataset exactly as authored (no recenter/scale, cube +/-1.5).
    // dense_fraction in [0,1]: fraction of each training ray chunk drawn as depth-prior ("dense")
    // rays instead of uniform-random pixels (0 = depth supervision off). See fetchRayChunk.
    DataLoader(const std::string& dataset_path, uint32_t ray_chunk_size, bool is_training = true,
               bool derive_bounds = true, float dense_fraction = 0.0f);
    // Reuse a scene transform derived by another loader (e.g. test loader matching the
    // training loader) instead of deriving its own — keeps both in the same normalized frame.
    DataLoader(const std::string& dataset_path, uint32_t ray_chunk_size, bool is_training,
               float3 ext_center, float ext_scale, float3 ext_aabb_min, float3 ext_aabb_max);
    ~DataLoader();

    const float3* getChunkRaysO() const { return d_chunk_rays_o; }
    const float3* getChunkRaysD() const { return d_chunk_rays_d; }
    const float* getChunkRgbTrue() const { return d_chunk_rgb_true; }
    // Per-ray depth prior for the current chunk: D (along-ray, normalized units) and its
    // Gaussian sigma. For non-depth (RGB) rays, dense = -1 (sentinel "no prior") and sigma = 0.
    const float* getChunkRaysDense() const { return d_chunk_rays_dense; }
    const float* getChunkRaysSigma() const { return d_chunk_rays_sigma; }

    void  setDenseRaysFraction(float f) { m_dense_fraction = f; }
    float getDenseRaysFraction() const { return m_dense_fraction; }
    // # of depth priors loaded (0 if none / not a training loader with a depth_file).
    uint32_t getNumDepthRecords() const { return m_num_records; }

    void loadDataToGPU();
    // holdout_every (>=2): exclude every Nth image (img % N == 0) from the randomized training
    // draw, so those images stay a clean held-out test set. 0 = no exclusion (default).
    void fetchRayChunk(int offset, int size, uint32_t seed, float3 bg_color = make_float3(1.0f, 1.0f, 1.0f), cudaStream_t stream = 0, bool forceSequential = false, int holdout_every = 0);
    void freeVRAM();

    uint32_t getTotalRays() const { return total_rays; }
    int getWidth() const { return width; }
    int getHeight() const { return height; }

    void getSceneOrientation(float3& center, float3& up) const {
        if (frames.empty()) {
            center = make_float3(0, 0, 0);
            up = make_float3(0, 0, 1);
            return;
        }
        float sum_px = 0, sum_py = 0, sum_pz = 0;
        float sum_ux = 0, sum_uy = 0, sum_uz = 0;
        for (const auto& f : frames) {
            sum_px += f.transform_matrix[3];
            sum_py += f.transform_matrix[7];
            sum_pz += f.transform_matrix[11];
            sum_ux += f.transform_matrix[1];
            sum_uy += f.transform_matrix[5];
            sum_uz += f.transform_matrix[9];
        }
        center = make_float3(sum_px / frames.size(), sum_py / frames.size(), sum_pz / frames.size());
        float len = sqrtf(sum_ux*sum_ux + sum_uy*sum_uy + sum_uz*sum_uz);
        if (len > 0) {
            up = make_float3(sum_ux/len, sum_uy/len, sum_uz/len);
        } else {
            up = make_float3(0, 0, 1);
        }
    }
    float getCameraAngleX() const { return camera_angle_x; }
    int getNumCascades() const { return num_cascades; }

    // Per-axis grid bounds derived from the camera poses (see computeSceneBounds).
    // Defaults to the legacy +/-1.5 cube until derivation runs / when poses are too sparse.
    void getGridBounds(float3& aabb_min, float3& aabb_max) const {
        aabb_min = m_aabb_min;
        aabb_max = m_aabb_max;
    }

    // The recenter+scale baked into the frame transforms (share with a matching test loader).
    void getSceneTransform(float3& center, float& scale) const {
        center = m_scene_center;
        scale  = m_scene_scale;
    }

    // Number of registered images in the dataset.
    int getNumImages() const { return (int)frames.size(); }

    // Copy frame i's camera-to-world (row-major 4x4, normalized frame) into out[16].
    void getFramePose(int i, float out[16]) const {
        if (i < 0 || i >= (int)frames.size()) { for (int k = 0; k < 16; ++k) out[k] = 0.0f; return; }
        for (int k = 0; k < 16; ++k) out[k] = frames[i].transform_matrix[k];
    }

private:
    std::string dataset_path;
    bool m_is_training;

    int width;
    int height;
    
    float camera_angle_x;
    int num_cascades = 1;   // from transforms.json; defaults to 1 (bounded) when absent
    bool m_num_cascades_from_json = false; // respect an explicit json value over the pose estimate
    float focal_length;
    uint32_t  m_ray_chunk_size;

    // Scene-bounds derivation (pose-only path). Recenter+scale is baked into the frame
    // transforms; the AABB is exposed via getGridBounds().
    float3 m_aabb_min = make_float3(-1.5f, -1.5f, -1.5f);
    float3 m_aabb_max = make_float3( 1.5f,  1.5f,  1.5f);
    float3 m_scene_center = make_float3(0.0f, 0.0f, 0.0f);
    float  m_scene_scale = 1.0f;

    // When false, computeSceneBounds() skips derivation entirely (dataset used as-is).
    bool   m_derive_bounds = true;
    // When true, computeSceneBounds() applies a supplied transform instead of deriving one.
    bool   m_use_external_transform = false;
    float3 m_ext_center = make_float3(0.0f, 0.0f, 0.0f);
    float  m_ext_scale  = 1.0f;

    struct Frame {
        std::string file_path;
        float transform_matrix[16];
    };

    std::vector<Frame> frames;
    
    // Pinned host memory and mapped device pointer
    uint8_t* h_images_rgba = nullptr;
    uint8_t* d_images_rgba = nullptr;
    
    // GPU Buffers
    float* d_transforms = nullptr;
    float3* d_chunk_rays_o = nullptr;
    float3* d_chunk_rays_d = nullptr;
    float* d_chunk_rgb_true = nullptr;
    float* d_chunk_rays_dense = nullptr;   // per-ray depth prior D (-1 = no prior)
    float* d_chunk_rays_sigma = nullptr;   // per-ray Gaussian sigma (0 = no prior)

    // Sparse depth priors loaded from depth_file (DS-NeRF-style). One record per kept
    // COLMAP keypoint; rays are generated from these for the "dense" fraction of a chunk.
    // See scripts/colmap_depth.py (writer) and docs/depth_supervision.md.
    std::string m_depth_file;              // from transforms.json "depth_file" (empty = none)
    float    m_dense_fraction = 0.0f;      // fraction of a chunk drawn as depth rays
    uint32_t m_num_records = 0;
    int      m_depth_width = 0;            // native (u,v) resolution of the priors
    int      m_depth_height = 0;
    int*   d_rec_frame = nullptr;          // per-record frame index (offset-table expanded)
    float* d_rec_u = nullptr;              // per-record pixel u, v (native depth resolution)
    float* d_rec_v = nullptr;
    float* d_rec_D = nullptr;
    float* d_rec_sigma = nullptr;

    uint32_t total_rays;


    void parseTransformsJson();
    void loadImages();
    void loadDepthBin();         // parse depth_file -> device record arrays (training only)
    void computeSceneBounds();   // ray-convergence center + regime-gated per-axis AABB
};
