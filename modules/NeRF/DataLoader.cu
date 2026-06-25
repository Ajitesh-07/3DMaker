#include "DataLoader.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "../../third_party/stb_image.h"
#include "../../third_party/json.hpp"

#include <cub/cub.cuh>
#include <curand.h>

using json = nlohmann::json;
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


DataLoader::DataLoader(const std::string& dataset_path, uint32_t ray_chunk_size, bool is_training,
                       bool derive_bounds, float dense_fraction)
        : dataset_path(dataset_path), m_is_training(is_training) {
    int width = 0;
    int height = 0;
    m_ray_chunk_size = ray_chunk_size;
    m_derive_bounds = derive_bounds;
    m_dense_fraction = dense_fraction;

    parseTransformsJson();
    loadImages();          // sets width/height (needed for the vertical FOV / aspect)
    computeSceneBounds();  // recenter+scale the poses in place, derive the per-axis AABB
    loadDataToGPU();       // uploads the (now normalized) transforms
}

DataLoader::DataLoader(const std::string& dataset_path, uint32_t ray_chunk_size, bool is_training,
                       float3 ext_center, float ext_scale, float3 ext_aabb_min, float3 ext_aabb_max)
        : dataset_path(dataset_path), m_is_training(is_training) {
    m_ray_chunk_size = ray_chunk_size;
    m_use_external_transform = true;
    m_ext_center = ext_center;
    m_ext_scale  = ext_scale;
    m_aabb_min   = ext_aabb_min;
    m_aabb_max   = ext_aabb_max;

    parseTransformsJson();
    loadImages();
    computeSceneBounds();  // applies the supplied transform (external branch)
    loadDataToGPU();
}

DataLoader::~DataLoader() {
    freeVRAM();
    if (h_images_rgba) cudaFreeHost(h_images_rgba);
}

void DataLoader::freeVRAM() {
    if (d_transforms) cudaFree(d_transforms);
    if (d_chunk_rays_o) cudaFree(d_chunk_rays_o);
    if (d_chunk_rays_d) cudaFree(d_chunk_rays_d);
    if (d_chunk_rgb_true) cudaFree(d_chunk_rgb_true);
    if (d_chunk_rays_dense) cudaFree(d_chunk_rays_dense);
    if (d_chunk_rays_sigma) cudaFree(d_chunk_rays_sigma);
    if (d_rec_frame) cudaFree(d_rec_frame);
    if (d_rec_u) cudaFree(d_rec_u);
    if (d_rec_v) cudaFree(d_rec_v);
    if (d_rec_D) cudaFree(d_rec_D);
    if (d_rec_sigma) cudaFree(d_rec_sigma);
}


void DataLoader::parseTransformsJson() {
    std::string json_file = dataset_path + (m_is_training ? "/transforms_train.json" : "/transforms_test.json");
    std::ifstream file(json_file);

    if (!file.is_open()) {
        std::cerr << "Failed to open " << json_file << std::endl;
        exit(1);
    }

    json j;
    file >> j;

    camera_angle_x = j["camera_angle_x"];

    // Optional: number of spatial-occupancy cascades estimated by colmap2nerf for this scene.
    // Absent in legacy/synthetic datasets -> stays 1 (bounded, single occupancy grid).
    if (j.contains("num_cascades")) {
        num_cascades = j["num_cascades"];
        if (num_cascades < 1) num_cascades = 1;
        m_num_cascades_from_json = true;
    }

    // Optional sparse depth-prior sidecar (binary). Present only in transforms_train.json.
    if (j.contains("depth_file")) {
        m_depth_file = j["depth_file"].get<std::string>();
    }

    for (auto& frame : j["frames"]) {
        Frame f;
        std::string path = frame["file_path"];
        if (path.find(".png") == std::string::npos && 
            path.find(".jpg") == std::string::npos && 
            path.find(".jpeg") == std::string::npos) {
            path += ".png";
        }

        f.file_path = dataset_path + "/" + path;

                int idx = 0;
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                f.transform_matrix[idx++] = frame["transform_matrix"][row][col];
            }
        }
        frames.push_back(f);
    }
}

// ---------------------------------------------------------------------------
// Scene-bounds helpers (host-side, operate on the N poses)
// ---------------------------------------------------------------------------
static inline float vlen3(float x, float y, float z) { return sqrtf(x*x + y*y + z*z); }

// Percentile of an already-mutable vector (sorts it). q in [0,1].
static float pctl(std::vector<float>& v, float q) {
    if (v.empty()) return 0.0f;
    std::sort(v.begin(), v.end());
    int n = (int)v.size();
    int idx = (int)(q * (n - 1) + 0.5f);
    if (idx < 0) idx = 0;
    if (idx >= n) idx = n - 1;
    return v[idx];
}

static double det3(const double M[3][3]) {
    return M[0][0]*(M[1][1]*M[2][2] - M[1][2]*M[2][1])
         - M[0][1]*(M[1][0]*M[2][2] - M[1][2]*M[2][0])
         + M[0][2]*(M[1][0]*M[2][1] - M[1][1]*M[2][0]);
}

// Solve M x = b for a 3x3 system via the adjugate. Returns false if near-singular.
static bool solve3x3(const double M[3][3], const double b[3], double x[3]) {
    double d = det3(M);
    if (fabs(d) < 1e-12) return false;
    double inv[3][3];
    inv[0][0] =  (M[1][1]*M[2][2] - M[1][2]*M[2][1]) / d;
    inv[0][1] = -(M[0][1]*M[2][2] - M[0][2]*M[2][1]) / d;
    inv[0][2] =  (M[0][1]*M[1][2] - M[0][2]*M[1][1]) / d;
    inv[1][0] = -(M[1][0]*M[2][2] - M[1][2]*M[2][0]) / d;
    inv[1][1] =  (M[0][0]*M[2][2] - M[0][2]*M[2][0]) / d;
    inv[1][2] = -(M[0][0]*M[1][2] - M[0][2]*M[1][0]) / d;
    inv[2][0] =  (M[1][0]*M[2][1] - M[1][1]*M[2][0]) / d;
    inv[2][1] = -(M[0][0]*M[2][1] - M[0][1]*M[2][0]) / d;
    inv[2][2] =  (M[0][0]*M[1][1] - M[0][1]*M[1][0]) / d;
    for (int r = 0; r < 3; ++r)
        x[r] = inv[r][0]*b[0] + inv[r][1]*b[1] + inv[r][2]*b[2];
    return true;
}

// Derive scene center + isotropic scale + per-axis AABB from the camera poses, then
// bake the recenter+scale into the frame transforms. Regime-gated per docs/dataloader_scene_bounds.md:
//   - orbit (inward-converging)  -> default +/-1.5 cube, object scaled to ~unit radius
//   - nadir/forward/panorama     -> per-axis box from a frustum march
// COLMAP point-cloud refinement (doc S16) is intentionally on hold; this is the pose-only path.
void DataLoader::computeSceneBounds() {
    const int N = (int)frames.size();
    std::cout << "\n[SceneBounds] " << N << " poses." << std::endl;

    // Reuse a transform derived elsewhere (test loader matching training loader). The AABB was
    // already set from the external values in the constructor — do NOT reset it below.
    if (m_use_external_transform) {
        m_scene_center = m_ext_center;
        m_scene_scale  = m_ext_scale;   // m_aabb_min/max were set from the external values
        for (int i = 0; i < N; ++i) {
            float* m = frames[i].transform_matrix;
            m[3]  = (m[3]  - m_scene_center.x) * m_scene_scale;
            m[7]  = (m[7]  - m_scene_center.y) * m_scene_scale;
            m[11] = (m[11] - m_scene_center.z) * m_scene_scale;
        }
        std::cout << "[SceneBounds] Applied EXTERNAL transform: center=(" << m_scene_center.x << ","
                  << m_scene_center.y << "," << m_scene_center.z << ") scale=" << m_scene_scale << std::endl;
        std::cout << "[SceneBounds] EXTERNAL AABB min=(" << m_aabb_min.x << "," << m_aabb_min.y << "," << m_aabb_min.z
                  << ") max=(" << m_aabb_max.x << "," << m_aabb_max.y << "," << m_aabb_max.z << ")\n" << std::endl;
        return;
    }

    // Safe defaults for the derivation path (also the fallback for sparse/degenerate inputs).
    m_aabb_min   = make_float3(-1.5f, -1.5f, -1.5f);
    m_aabb_max   = make_float3( 1.5f,  1.5f,  1.5f);
    m_scene_center = make_float3(0.0f, 0.0f, 0.0f);
    m_scene_scale  = 1.0f;

    if (!m_derive_bounds) {
        std::cout << "[SceneBounds] Derivation DISABLED -- dataset used as-is "
                     "(cube [-1.5,1.5]^3, identity transform, num_cascades unchanged).\n" << std::endl;
        return;
    }

    std::cout << "[SceneBounds] Deriving bounds (pose-only path)..." << std::endl;

    if (N < 4) {
        std::cout << "[SceneBounds] Too few poses; keeping default cube [-1.5,1.5]^3." << std::endl;
        return;
    }

    // --- 1. Extract per-camera basis -------------------------------------------------
    std::vector<float3> C(N), F(N), Rt(N), Ut(N);
    float3 meanF = make_float3(0, 0, 0), meanC = make_float3(0, 0, 0);
    for (int i = 0; i < N; ++i) {
        const float* m = frames[i].transform_matrix;       // row-major c2w
        C[i] = make_float3(m[3], m[7], m[11]);              // camera center  = col3
        float3 f = make_float3(-m[2], -m[6], -m[10]);       // forward (view) = -col2
        float fl = vlen3(f.x, f.y, f.z);
        if (fl > 1e-8f) { f.x /= fl; f.y /= fl; f.z /= fl; }
        F[i]  = f;
        Rt[i] = make_float3(m[0], m[4], m[8]);              // right = col0
        Ut[i] = make_float3(m[1], m[5], m[9]);              // up    = col1
        meanF.x += f.x; meanF.y += f.y; meanF.z += f.z;
        meanC.x += C[i].x; meanC.y += C[i].y; meanC.z += C[i].z;
    }
    meanF.x /= N; meanF.y /= N; meanF.z /= N;
    meanC.x /= N; meanC.y /= N; meanC.z /= N;

    // --- 2. Ray-convergence center: (sum A_i) p = sum A_i C_i, A_i = I - f f^T --------
    double M[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
    double bvec[3] = {0,0,0};
    for (int i = 0; i < N; ++i) {
        float3 f = F[i];
        double A[3][3] = {
            {1.0 - f.x*f.x,      -f.x*f.y,      -f.x*f.z},
            {     -f.y*f.x, 1.0 - f.y*f.y,      -f.y*f.z},
            {     -f.z*f.x,      -f.z*f.y, 1.0 - f.z*f.z}
        };
        double Cx = C[i].x, Cy = C[i].y, Cz = C[i].z;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) M[r][c] += A[r][c];
            bvec[r] += A[r][0]*Cx + A[r][1]*Cy + A[r][2]*Cz;
        }
    }
    // Degeneracy: A_i has eigenvalues {0 (along f), 1, 1}. With varied forwards, (M/N) ~ (2/3)I
    // (det ~ 0.30); with parallel forwards (forward-facing/nadir) one eigenvalue -> 0, det -> 0.
    double Mn[3][3];
    for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) Mn[r][c] = M[r][c] / N;
    double detN = det3(Mn);
    bool degenerate = fabs(detN) < 0.02;

    double psol[3];
    bool solved = solve3x3(M, bvec, psol);
    float3 pstar = (solved && !degenerate)
        ? make_float3((float)psol[0], (float)psol[1], (float)psol[2])
        : meanC;  // fall back to camera centroid when rays don't converge

    // --- 3. Regime signals -----------------------------------------------------------
    float inwardness = 0.0f;
    for (int i = 0; i < N; ++i) {
        float gx = pstar.x - C[i].x, gy = pstar.y - C[i].y, gz = pstar.z - C[i].z;
        float gl = vlen3(gx, gy, gz);
        if (gl > 1e-8f) { gx /= gl; gy /= gl; gz /= gl; }
        inwardness += F[i].x*gx + F[i].y*gy + F[i].z*gz;
    }
    inwardness /= N;
    float spread = 1.0f - vlen3(meanF.x, meanF.y, meanF.z); // 0 = parallel forwards, ~1 = spread

    const float ORBIT_INWARD_THRESH = 0.7f;
    bool orbit = (!degenerate) && (inwardness > ORBIT_INWARD_THRESH);

    std::cout << "[SceneBounds] det(M/N)=" << detN << (degenerate ? " (DEGENERATE)" : "")
              << "  inwardness=" << inwardness << "  spread=" << spread << std::endl;
    std::cout << "[SceneBounds] convergence center p*=(" << pstar.x << "," << pstar.y << "," << pstar.z << ")"
              << (solved && !degenerate ? "" : " [fallback: camera centroid]") << std::endl;
    std::cout << "[SceneBounds] regime = " << (orbit ? "ORBIT (cube)" : "NADIR/FORWARD/PANORAMA (per-axis)") << std::endl;

    if (orbit) {
        // --- 4a. Orbit: object -> ~unit radius inside the default +/-1.5 cube ---------
        std::vector<float> d(N);
        for (int i = 0; i < N; ++i)
            d[i] = vlen3(C[i].x - pstar.x, C[i].y - pstar.y, C[i].z - pstar.z);
        std::vector<float> d_a = d, d_b = d;
        float med = pctl(d_a, 0.50f);
        float q10 = pctl(d_b, 0.10f);   // shell bound: object can't extend past nearest camera

        const float F_FILL = 0.5f;      // subject fills ~half the frame
        float r_obj = med * tanf(0.5f * camera_angle_x) * F_FILL;
        if (r_obj > q10) r_obj = q10;
        if (r_obj < 1e-3f) r_obj = 1e-3f;

        m_scene_center = pstar;
        m_scene_scale  = 1.0f / r_obj;  // object -> radius ~1, leaving 1.0->1.5 box headroom
        // AABB stays the default cube.
        std::cout << "[SceneBounds] ORBIT: median cam dist=" << med
                  << "  r_obj=" << r_obj << "  scale=" << m_scene_scale
                  << " -> AABB = default cube [-1.5,1.5]^3" << std::endl;
    } else {
        // --- 4b. Diverging/parallel: frustum march -> per-axis observed box -----------
        std::vector<float> camd(N);
        for (int i = 0; i < N; ++i)
            camd[i] = vlen3(C[i].x - meanC.x, C[i].y - meanC.y, C[i].z - meanC.z);
        float footprint = pctl(camd, 0.95f);
        const float MARCH_K = 1.5f;
        float D = MARCH_K * footprint;
        if (D < 1e-4f) D = 1.0f;

        float tan_hx = tanf(0.5f * camera_angle_x);
        float tan_hy = (width > 0) ? tan_hx * ((float)height / (float)width) : tan_hx;
        const int MARCH_SAMPLES = 8;
        const float corners[5][2] = {{0,0},{1,1},{1,-1},{-1,1},{-1,-1}};

        std::vector<float> xs, ys, zs;
        xs.reserve(N * 5 * MARCH_SAMPLES + N);
        ys.reserve(xs.capacity());
        zs.reserve(xs.capacity());
        for (int i = 0; i < N; ++i) {
            xs.push_back(C[i].x); ys.push_back(C[i].y); zs.push_back(C[i].z); // camera itself
            for (int ci = 0; ci < 5; ++ci) {
                float sx = corners[ci][0], sy = corners[ci][1];
                float dx = F[i].x + tan_hx*sx*Rt[i].x + tan_hy*sy*Ut[i].x;
                float dy = F[i].y + tan_hx*sx*Rt[i].y + tan_hy*sy*Ut[i].y;
                float dz = F[i].z + tan_hx*sx*Rt[i].z + tan_hy*sy*Ut[i].z;
                float dl = vlen3(dx, dy, dz);
                if (dl > 1e-8f) { dx /= dl; dy /= dl; dz /= dl; }
                for (int s = 0; s < MARCH_SAMPLES; ++s) {
                    float t = D * ((float)s + 0.5f) / MARCH_SAMPLES;
                    xs.push_back(C[i].x + t*dx);
                    ys.push_back(C[i].y + t*dy);
                    zs.push_back(C[i].z + t*dz);
                }
            }
        }
        float loX = pctl(xs, 0.02f), hiX = pctl(xs, 0.98f);
        float loY = pctl(ys, 0.02f), hiY = pctl(ys, 0.98f);
        float loZ = pctl(zs, 0.02f), hiZ = pctl(zs, 0.98f);

        float cX = 0.5f*(loX+hiX), cY = 0.5f*(loY+hiY), cZ = 0.5f*(loZ+hiZ);
        float hX = 0.5f*(hiX-loX), hY = 0.5f*(hiY-loY), hZ = 0.5f*(hiZ-loZ);
        float maxh = std::max(hX, std::max(hY, hZ));
        if (maxh < 1e-5f) maxh = 1.0f;

        const float NADIR_FILL = 1.5f, MARGIN = 1.1f, H_MIN = 0.1f, BASE = 1.5f;
        m_scene_center = make_float3(cX, cY, cZ);
        m_scene_scale  = NADIR_FILL / maxh;     // widest axis -> 1.5

        float HX = std::min(BASE, std::max(H_MIN, hX * m_scene_scale * MARGIN));
        float HY = std::min(BASE, std::max(H_MIN, hY * m_scene_scale * MARGIN));
        float HZ = std::min(BASE, std::max(H_MIN, hZ * m_scene_scale * MARGIN));
        m_aabb_min = make_float3(-HX, -HY, -HZ);
        m_aabb_max = make_float3( HX,  HY,  HZ);

        std::cout << "[SceneBounds] PER-AXIS: footprint(Q95)=" << footprint << "  march D=" << D
                  << "  scale=" << m_scene_scale << std::endl;
        std::cout << "[SceneBounds] half-extents (normalized): X=" << HX << " Y=" << HY << " Z=" << HZ << std::endl;
    }

    // --- 5. Bake recenter + isotropic scale into the frame transforms ----------------
    for (int i = 0; i < N; ++i) {
        float* m = frames[i].transform_matrix;
        m[3]  = (m[3]  - m_scene_center.x) * m_scene_scale;
        m[7]  = (m[7]  - m_scene_center.y) * m_scene_scale;
        m[11] = (m[11] - m_scene_center.z) * m_scene_scale;
    }

    // --- 6. Grid scale (num_cascades) estimate ---------------------------------------
    if (m_num_cascades_from_json) {
        std::cout << "[SceneBounds] num_cascades = " << num_cascades
                  << " (from transforms.json, respected)." << std::endl;
    } else {
        int est;
        if (degenerate && fabs(inwardness) < 0.3f) est = 4;   // forward-facing / parallel
        else if (inwardness < -0.3f)               est = 3;   // outward panorama
        else                                       est = 1;   // bounded (orbit / nadir slab)
        if (est < 1) est = 1; if (est > 6) est = 6;
        num_cascades = est;
        std::cout << "[SceneBounds] num_cascades estimate = " << num_cascades
                  << " (pose-only heuristic -- COLMAP points needed for an accurate count)." << std::endl;
    }

    std::cout << "[SceneBounds] FINAL center=(" << m_scene_center.x << "," << m_scene_center.y << ","
              << m_scene_center.z << ")  scale=" << m_scene_scale << std::endl;
    std::cout << "[SceneBounds] FINAL AABB min=(" << m_aabb_min.x << "," << m_aabb_min.y << "," << m_aabb_min.z
              << ")  max=(" << m_aabb_max.x << "," << m_aabb_max.y << "," << m_aabb_max.z << ")\n" << std::endl;
}

void DataLoader::loadImages() {
    if (frames.empty()) return;
    std::cout << "Loading " << frames.size() << " images..." << std::endl;

    int w, h, channels;
    unsigned char* img0 = stbi_load(frames[0].file_path.c_str(), &w, &h, &channels, 4);
    if (!img0) {
        std::cerr << "Failed to load image: " << frames[0].file_path << std::endl;
        exit(1);
    }

    width = w;
    height = h;
    focal_length = 0.5f * width / tanf(0.5f * camera_angle_x);

    size_t pixels_per_image = width * height;
    size_t bytes_per_image = pixels_per_image * 4;

    CUDA_CHECK(cudaHostAlloc((void**)&h_images_rgba, frames.size() * bytes_per_image, cudaHostAllocMapped));

    std::memcpy(h_images_rgba, img0, bytes_per_image);
    stbi_image_free(img0);

    for (int i = 1; i < (int)frames.size(); ++i) {
        int thread_w, thread_h, thread_channels;
        
        unsigned char* img = stbi_load(frames[i].file_path.c_str(), &thread_w, &thread_h, &thread_channels, 4);
        
        if (!img) {
            #pragma omp critical
            {
                std::cerr << "Failed to load image: " << frames[i].file_path << std::endl;
            }
            continue;
        }

        size_t offset = i * bytes_per_image;
        std::memcpy(h_images_rgba + offset, img, bytes_per_image);

        stbi_image_free(img);
    }

    total_rays = width * height * frames.size();
    std::cout << "Loaded " << total_rays << " total rays compactly via Pinned Memory. Focal length: " << focal_length << std::endl;
}

void DataLoader::loadDataToGPU() {
    std::cout << "Mapping Pinned Memory to GPU..." << std::endl;
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_images_rgba, (void*)h_images_rgba, 0));

    std::vector<float> h_transforms(frames.size() * 16);
    for (size_t i = 0; i < frames.size(); ++i) {
        for (int j = 0; j < 16; ++j) {
            h_transforms[i * 16 + j] = frames[i].transform_matrix[j];
        }
    }
    CUDA_CHECK(cudaMalloc(&d_transforms, frames.size() * 16 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_transforms, h_transforms.data(), frames.size() * 16 * sizeof(float), cudaMemcpyHostToDevice));

    if (!m_is_training) {
        m_ray_chunk_size = width * height; // Ensure buffer is large enough for a full image
    }

    CUDA_CHECK(cudaMalloc(&d_chunk_rays_o, m_ray_chunk_size * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_chunk_rays_d, m_ray_chunk_size * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_chunk_rgb_true, m_ray_chunk_size * 3 * sizeof(float)));
    // Per-ray depth-prior outputs (always allocated; kernel writes -1/0 when no prior).
    CUDA_CHECK(cudaMalloc(&d_chunk_rays_dense, m_ray_chunk_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_chunk_rays_sigma, m_ray_chunk_size * sizeof(float)));

    if (m_is_training && !m_depth_file.empty()) {
        loadDepthBin();
    }
}

// Parse the binary depth-prior sidecar (scripts/colmap_depth.py write_depth_bin) into device
// record arrays. Layout: header 'DPT1'|version|num_frames|width|height (20B), then
// (num_frames+1) u32 record-offsets, then records {u:u16, v:u16, D:f32, sigma:f32} (12B each)
// sorted per frame. We expand the offset table into a per-record frame index so a sampled
// record yields (frame, u, v, D, sigma) in O(1).
void DataLoader::loadDepthBin() {
    std::string bin_path = dataset_path + "/" + m_depth_file;
    std::ifstream f(bin_path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "[Depth] Could not open " << bin_path << "; depth supervision disabled." << std::endl;
        return;
    }

    char magic[4];
    uint32_t version = 0, num_frames_bin = 0, w = 0, h = 0;
    f.read(magic, 4);
    f.read(reinterpret_cast<char*>(&version), 4);
    f.read(reinterpret_cast<char*>(&num_frames_bin), 4);
    f.read(reinterpret_cast<char*>(&w), 4);
    f.read(reinterpret_cast<char*>(&h), 4);
    if (!f || std::string(magic, 4) != "DPT1") {
        std::cerr << "[Depth] Bad/short header in " << bin_path << " (expected magic DPT1); skipping." << std::endl;
        return;
    }
    m_depth_width  = (int)w;
    m_depth_height = (int)h;

    std::vector<uint32_t> offsets(num_frames_bin + 1);
    f.read(reinterpret_cast<char*>(offsets.data()), (std::streamsize)(num_frames_bin + 1) * sizeof(uint32_t));
    uint32_t total = offsets.back();
    if (!f || total == 0) {
        std::cerr << "[Depth] " << bin_path << " has 0 records; depth supervision disabled." << std::endl;
        return;
    }
    if (num_frames_bin != frames.size()) {
        std::cerr << "[Depth] WARNING: depth_file has " << num_frames_bin << " frames but transforms has "
                  << frames.size() << "; record frame ids assume the bin's order." << std::endl;
    }

    std::vector<uint8_t> raw((size_t)total * 12);
    f.read(reinterpret_cast<char*>(raw.data()), (std::streamsize)total * 12);
    if (!f) {
        std::cerr << "[Depth] Truncated record block in " << bin_path << "; skipping." << std::endl;
        return;
    }
    f.close();

    std::vector<int>   h_frame(total);
    std::vector<float> h_u(total), h_v(total), h_D(total), h_sigma(total);
    for (uint32_t fr = 0; fr < num_frames_bin; ++fr) {
        for (uint32_t r = offsets[fr]; r < offsets[fr + 1]; ++r) h_frame[r] = (int)fr;
    }
    for (uint32_t r = 0; r < total; ++r) {
        const uint8_t* p = raw.data() + (size_t)r * 12;  // little-endian, matches x86 host
        uint16_t u, v; float D, sg;
        std::memcpy(&u,  p + 0, 2);
        std::memcpy(&v,  p + 2, 2);
        std::memcpy(&D,  p + 4, 4);
        std::memcpy(&sg, p + 8, 4);
        h_u[r] = (float)u; h_v[r] = (float)v; h_D[r] = D; h_sigma[r] = sg;
    }

    m_num_records = total;
    CUDA_CHECK(cudaMalloc(&d_rec_frame, total * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rec_u,     total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rec_v,     total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rec_D,     total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rec_sigma, total * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_rec_frame, h_frame.data(), total * sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rec_u,     h_u.data(),     total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rec_v,     h_v.data(),     total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rec_D,     h_D.data(),     total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rec_sigma, h_sigma.data(), total * sizeof(float), cudaMemcpyHostToDevice));

    std::cout << "[Depth] Loaded " << total << " priors from " << m_depth_file
              << " (" << m_depth_width << "x" << m_depth_height << "), "
              << num_frames_bin << " frames, dense_fraction=" << m_dense_fraction << "." << std::endl;
}

__device__ uint32_t pcg_hash(uint32_t input) {
    uint32_t state = input * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__global__ void fetchRayChunkStreamingKernel(
    int offset, int size, int total_rays, uint32_t seed, bool randomize,
    int width, int height, float focal_length,
    const float* d_transforms,
    const uint8_t* d_images_rgba,
    float3* d_chunk_rays_o, float3* d_chunk_rays_d, float* d_chunk_rgb,
    float* d_chunk_dense, float* d_chunk_sigma,
    const int* rec_frame, const float* rec_u, const float* rec_v,
    const float* rec_D, const float* rec_sigma, uint32_t num_records,
    float dense_fraction, float uv_scale_x, float uv_scale_y,
    float3 bg_color, int holdout_every
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    uint32_t global_idx;
    if (randomize) {
        uint32_t rand_val = pcg_hash(offset + idx + seed);
        if (holdout_every >= 2) {
            int pixels    = width * height;
            int num_imgs  = total_rays / pixels;
            int num_test  = (num_imgs + holdout_every - 1) / holdout_every;
            int num_train = num_imgs - num_test;
            if (num_train < 1) {
                global_idx = rand_val % total_rays; 
            } else {
                int train_slot = (int)(rand_val % (uint32_t)num_train);
                int block   = train_slot / (holdout_every - 1); // (N-1) train images per N-image block
                int within  = train_slot % (holdout_every - 1);
                int img_idx = block * holdout_every + within + 1; // +1 skips the held-out block start (b*N)
                uint32_t r2 = pcg_hash(rand_val);              // decorrelate pixel choice from image choice
                int pixel_idx = (int)(r2 % (uint32_t)pixels);
                global_idx = (uint32_t)img_idx * (uint32_t)pixels + (uint32_t)pixel_idx;
            }
        } else {
            global_idx = rand_val % total_rays;
        }
    } else {
        global_idx = (offset + idx) % total_rays;
    }

    int pixels = width * height;
    int img_idx = global_idx / pixels;
    int pixel_idx = global_idx % pixels;
    float px = (float)(pixel_idx % width);
    float py = (float)(pixel_idx / width);

    // Dense (depth-prior) ray override: with probability dense_fraction, replace this
    // uniform-random RGB ray with a ray generated AT a COLMAP keypoint, so the depth loss
    // actually fires (priors are far too sparse to ever hit by random pixel sampling).
    // Decided per slot via an independent hash -> re-randomized every step and evenly spread
    // across the 256Ki hit-batches (hence every per-batch SGD step). RGB rays keep dense=-1.
    float out_D = -1.0f, out_sigma = -1.0f;
    if (randomize && dense_fraction > 0.0f && num_records > 0) {
        uint32_t hsel = pcg_hash((uint32_t)(idx + 0x9E3779B9u) ^ (seed * 2654435761u));
        if (hsel * (1.0f / 4294967296.0f) < dense_fraction) {
            uint32_t r = pcg_hash(hsel) % num_records;
            int f = rec_frame[r];
            // Skip priors on held-out images during validation (would leak test geometry).
            if (!(holdout_every >= 2 && (f % holdout_every == 0))) {
                img_idx = f;
                px = rec_u[r] * uv_scale_x;
                py = rec_v[r] * uv_scale_y;
                out_D = rec_D[r];
                out_sigma = rec_sigma[r];
                int px_i = min(max((int)(px + 0.5f), 0), width - 1);
                int py_i = min(max((int)(py + 0.5f), 0), height - 1);
                global_idx = (uint32_t)img_idx * (uint32_t)pixels + (uint32_t)(py_i * width + px_i);
            }
        }
    }

    float dir_x = (px - width * 0.5f) / focal_length;
    float dir_y = -(py - height * 0.5f) / focal_length;
    float dir_z = -1.0f;

    float inv_norm = rsqrtf(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z);
    dir_x *= inv_norm;
    dir_y *= inv_norm;
    dir_z *= inv_norm;

    const float* c2w = &d_transforms[img_idx * 16];

    float world_dir_x = c2w[0]*dir_x + c2w[1]*dir_y + c2w[2]*dir_z;
    float world_dir_y = c2w[4]*dir_x + c2w[5]*dir_y + c2w[6]*dir_z;
    float world_dir_z = c2w[8]*dir_x + c2w[9]*dir_y + c2w[10]*dir_z;

    float world_o_x = c2w[3];
    float world_o_y = c2w[7];
    float world_o_z = c2w[11];

    d_chunk_rays_o[idx] = make_float3(world_o_x, world_o_y, world_o_z);
    d_chunk_rays_d[idx] = make_float3(world_dir_x, world_dir_y, world_dir_z);

    uint8_t r = d_images_rgba[global_idx * 4 + 0];
    uint8_t g = d_images_rgba[global_idx * 4 + 1];
    uint8_t b = d_images_rgba[global_idx * 4 + 2];
    uint8_t a = d_images_rgba[global_idx * 4 + 3];

    float fr = r / 255.0f;
    float fg = g / 255.0f;
    float fb = b / 255.0f;
    float fa = a / 255.0f;

    fr = fr * fa + bg_color.x * (1.0f - fa);
    fg = fg * fa + bg_color.y * (1.0f - fa);
    fb = fb * fa + bg_color.z * (1.0f - fa);

    d_chunk_rgb[idx * 3 + 0] = fr;
    d_chunk_rgb[idx * 3 + 1] = fg;
    d_chunk_rgb[idx * 3 + 2] = fb;

    d_chunk_dense[idx] = out_D;
    d_chunk_sigma[idx] = out_sigma;
}

void DataLoader::fetchRayChunk(int offset, int size, uint32_t seed, float3 bg_color, cudaStream_t stream, bool forceSequential, int holdout_every) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    bool randomize = m_is_training && !forceSequential;
    float uv_scale_x = (m_depth_width  > 0) ? (float)width  / (float)m_depth_width  : 1.0f;
    float uv_scale_y = (m_depth_height > 0) ? (float)height / (float)m_depth_height : 1.0f;
    fetchRayChunkStreamingKernel<<<gridSize, blockSize, 0, stream>>>(
        offset, size, total_rays, seed, randomize,
        width, height, focal_length,
        d_transforms, d_images_rgba,
        d_chunk_rays_o, d_chunk_rays_d, d_chunk_rgb_true,
        d_chunk_rays_dense, d_chunk_rays_sigma,
        d_rec_frame, d_rec_u, d_rec_v, d_rec_D, d_rec_sigma, m_num_records,
        m_dense_fraction, uv_scale_x, uv_scale_y,
        bg_color, holdout_every
    );
}

