#include "NerfTrainer.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <climits>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static const float kPI = 3.14159265359f;

static __global__ void mse_kernel(const float* pred, const float* target, float* loss_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float r = pred[idx * 3 + 0] - target[idx * 3 + 0];
        float g = pred[idx * 3 + 1] - target[idx * 3 + 1];
        float b = pred[idx * 3 + 2] - target[idx * 3 + 2];
        atomicAdd(loss_out, (r * r + g * g + b * b) / 3.0f);
    }
}

static __global__ void generate_custom_rays_kernel(
    int width, int height, float focal,
    float c00, float c01, float c02, float c03,
    float c10, float c11, float c12, float c13,
    float c20, float c21, float c22, float c23,
    float3* d_rays_o, float3* d_rays_d
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;
    int px = idx % width;
    int py = idx / width;

    float dx = (px - width  * 0.5f) / focal;
    float dy = -(py - height * 0.5f) / focal;
    float dz = -1.0f;
    float inv = rsqrtf(dx * dx + dy * dy + dz * dz);
    dx *= inv; dy *= inv; dz *= inv;

    d_rays_o[idx] = make_float3(c03, c13, c23);
    d_rays_d[idx] = make_float3(c00 * dx + c01 * dy + c02 * dz,
                                c10 * dx + c11 * dy + c12 * dz,
                                c20 * dx + c21 * dy + c22 * dz);
}

void INerfTrainer::init(const NerfConfig& config) {
    m_config = config;

    NerfOptions& opts = m_masterConfig;
    opts.gridResolution  = make_uint3(128, 128, 128);
    opts.aabbMin         = make_float3(-1.5f, -1.5f, -1.5f);
    opts.aabbMax         = make_float3( 1.5f,  1.5f,  1.5f);
    opts.levelsMipmap    = 4;
    opts.numCascades     = (config.numCascades > 0) ? config.numCascades : 1;
    opts.samplesPerVoxel = config.sampleK;

    opts.densityHiddenDim = 64;
    opts.densityNumLayers = 2;
    opts.colorHiddenDim   = 32;
    opts.colorNumLayers   = 3;

    opts.hashTableSize    = 1 << 19;
    opts.numLevels        = 16;
    opts.growthFactor     = 1.3819f;
    opts.baseResolution   = 16;
    opts.featuresPerLevel = 2;

    opts.renderBatchSize = config.renderBatchSize;
    opts.batchSize       = config.batchSize;
    opts.rayChunkSize    = config.batchSize / 4;

    opts.lambdaDist   = config.lambdaDist;
    opts.learningRate = config.maxLR;
    opts.beta1 = 0.9f; opts.beta2 = 0.999f; opts.epsilon = 1e-8f;
    opts.lossScale = 128.0f;

    opts.bgColor             = make_float3(1.0f, 1.0f, 1.0f);
    opts.minDensityThreshold = config.minDensityThreshold;
    opts.decayValue          = 0.95f;
    opts.densityBias         = 0.0f;
    opts.legacyRenderFlag    = false;
    opts.isProfiling         = false;

    m_holdout_every = config.shouldValidate ? 8 : 0;

    CUDA_CHECK(cudaMalloc(&m_d_loss, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m_d_chunkRgb, opts.rayChunkSize * 3 * sizeof(float)));

}

INerfTrainer::~INerfTrainer() {
    delete m_nerf;
    delete m_dataloader;
    if (m_d_loss)     cudaFree(m_d_loss);
    if (m_d_chunkRgb) cudaFree(m_d_chunkRgb);
}

void INerfTrainer::buildModel() {
    if (m_nerf) return;
    m_nerf = new InstantNerf();
    m_nerf->init(m_masterConfig, TRAINING);
}

void INerfTrainer::loadDataset(const std::string& path) {
    delete m_dataloader;
    m_dataloader = new DataLoader(path, m_masterConfig.rayChunkSize, true, false);
    m_total_rays = m_dataloader->getTotalRays();

    if (m_config.numCascades <= 0)
        m_masterConfig.numCascades = std::max(1, m_dataloader->getNumCascades());

    buildModel();

    float3 camCentroid, up, loaderCenter; float loaderScale;
    m_dataloader->getSceneOrientation(camCentroid, up); 
    m_dataloader->getSceneTransform(loaderCenter, loaderScale);
    m_sceneCenter  = { 0.0f, 0.0f, 0.0f };
    m_sceneUp      = { up.x, up.y, up.z };
    m_sceneScale   = loaderScale;
    m_cameraAngleX = m_dataloader->getCameraAngleX();

    m_globalStep = 0;
    m_rayCursor  = 0;
    m_lossHistory.clear();
}

float INerfTrainer::currentLR() const {
    int anneal = (m_config.annealSteps > 0) ? m_config.annealSteps : std::max(1, m_targetSteps);
    float progress = fminf((float)m_globalStep / (float)anneal, 1.0f);
    return m_config.minLR + 0.5f * (m_config.maxLR - m_config.minLR) * (1.0f + cosf(kPI * progress));
}

float INerfTrainer::trainOneChunk(bool computeLoss) {
    const uint32_t rayChunk = m_masterConfig.rayChunkSize;

    if (m_rayCursor >= m_total_rays) m_rayCursor = 0;
    if (m_rayCursor == 0)            m_epochSeed = m_seed_dist(m_rng);

    int chunkSize = (int)std::min(rayChunk, m_total_rays - m_rayCursor);

    m_nerf->setLearningRate(currentLR());
    m_nerf->setMemoryMode(TRAINING);

    float3 bg = make_float3(m_dist(m_rng), m_dist(m_rng), m_dist(m_rng)); // random-bg augmentation
    m_nerf->setBgColor(bg);

    m_dataloader->fetchRayChunk((int)m_rayCursor, chunkSize, m_epochSeed, bg, 0, false, (int)m_holdout_every);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (computeLoss) CUDA_CHECK(cudaMemset(m_d_loss, 0, sizeof(float)));

    std::vector<uint32_t> hitCounts(chunkSize, 0);
    m_nerf->trainWithRaysHit(
        m_dataloader->getChunkRaysO(),
        m_dataloader->getChunkRaysD(),
        m_dataloader->getChunkRgbTrue(),
        (uint32_t)chunkSize,
        m_globalStep,
        hitCounts.data(),
        computeLoss ? m_d_chunkRgb : nullptr,
        0
    );

    m_rayCursor += chunkSize;
    m_raysSeen  += chunkSize;

    if (!computeLoss) return -1.0f;
    int threads = 256, blocks = (chunkSize + threads - 1) / threads;
    mse_kernel<<<blocks, threads>>>(m_d_chunkRgb, m_dataloader->getChunkRgbTrue(), m_d_loss, chunkSize);
    float h_loss = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_loss, m_d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    return h_loss / std::max(1, chunkSize);
}

void INerfTrainer::beginTraining(int totalSteps) {
    m_targetSteps   = std::max(1, totalSteps);
    m_globalStep    = 0;
    m_rayCursor     = 0;
    m_raysSeen      = 0.0;
    m_emaMsPerStep  = 0.0;
    m_lossHistory.clear();
}

TrainStats INerfTrainer::trainSteps(int nSteps) {
    TrainStats s;
    if (!m_nerf || !m_dataloader || m_total_rays == 0) return s;
    if (m_targetSteps == 0) beginTraining(20000);

    int target    = std::min(m_globalStep + std::max(1, nSteps), m_targetSteps);
    int startStep = m_globalStep;
    auto t0 = std::chrono::high_resolution_clock::now();

    float mse   = -1.0f;
    bool  first = true;
    while (m_globalStep < target) {
        float chunkMse = trainOneChunk(first);
        if (chunkMse > 0.0f) mse = chunkMse;
        first = false;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    int stepsDone = m_globalStep - startStep;
    if (stepsDone > 0) {
        double msPerStep = std::chrono::duration<double, std::milli>(t1 - t0).count() / stepsDone;
        m_emaMsPerStep = (m_emaMsPerStep > 0.0) ? (0.70 * m_emaMsPerStep + 0.30 * msPerStep) : msPerStep;
    }

    if (mse > 0.0f) m_lossHistory.push_back(mse);
    int remaining = std::max(0, m_targetSteps - m_globalStep);
    s.step        = m_globalStep;
    s.targetSteps = m_targetSteps;
    s.loss        = (mse > 0.0f) ? mse : (m_lossHistory.empty() ? 0.0f : m_lossHistory.back());
    s.psnr        = (s.loss > 0.0f) ? -10.0f * log10f(fmaxf(s.loss, 1e-8f)) : 0.0f;
    s.lr          = currentLR();
    s.epochs      = epochs();
    s.msPerStep   = (float)m_emaMsPerStep;
    s.etaSeconds  = (float)(m_emaMsPerStep * remaining / 1000.0);
    return s;
}

static std::string formatETA(float seconds) {
    if (!(seconds >= 0.0f)) return "--:--";
    int t = (int)(seconds + 0.5f);
    int h = t / 3600; t %= 3600;
    int m = t / 60;   t %= 60;
    char buf[32];
    if (h > 0) snprintf(buf, sizeof(buf), "%d:%02d:%02d", h, m, t);
    else       snprintf(buf, sizeof(buf), "%02d:%02d", m, t);
    return buf;
}

static void renderTrainingBar(const TrainStats& s) {
    const int kWidth = 28;
    float frac = (s.targetSteps > 0) ? (float)s.step / (float)s.targetSteps : 0.0f;
    frac = fminf(1.0f, fmaxf(0.0f, frac));
    int filled = (int)(frac * kWidth + 0.5f);
    std::string bar(filled, '#');
    bar.append(kWidth - filled, ' ');
    printf("\r[%s] %d/%d steps | epoch %.2f | %.2f ms/step | ETA %s | PSNR %.1f   ",
           bar.c_str(), s.step, s.targetSteps, s.epochs, s.msPerStep,
           formatETA(s.etaSeconds).c_str(), s.psnr);
    fflush(stdout);
}

void INerfTrainer::train(int maxSteps, int maxEpochs) {
    (void)maxEpochs;
    if (!m_nerf || !m_dataloader || m_total_rays == 0) {
        std::cerr << "[INerfTrainer] train() called before loadDataset()." << std::endl;
        return;
    }
    beginTraining(maxSteps);
    int tick = std::max(1, maxSteps / 100);
    while (!isTrainingDone()) {
        TrainStats s = trainSteps(tick);
        renderTrainingBar(s);
    }
    printf("\n");
}

static const char kSceneMagic[4] = {'S', 'C', 'N', '1'};
static const std::streamoff kSceneFooterBytes = 4 + 8 * (std::streamoff)sizeof(float);

void INerfTrainer::save(const std::string& path) {
    if (!m_nerf) return;
    m_nerf->save(path);

    std::ofstream out(path, std::ios::binary | std::ios::app);
    if (!out) return;
    float buf[8] = { m_sceneCenter.x, m_sceneCenter.y, m_sceneCenter.z,
                     m_sceneUp.x,     m_sceneUp.y,     m_sceneUp.z,
                     m_sceneScale,    m_cameraAngleX };
    out.write(kSceneMagic, 4);
    out.write(reinterpret_cast<const char*>(buf), sizeof(buf));
}

void INerfTrainer::load(const std::string& path) {
    buildModel();
    m_nerf->load(path);

    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) return;
    std::streamoff size = in.tellg();
    if (size < kSceneFooterBytes) return;
    in.seekg(size - kSceneFooterBytes, std::ios::beg);
    char magic[4];
    in.read(magic, 4);
    if (std::string(magic, 4) != std::string(kSceneMagic, 4)) return;
    float buf[8];
    in.read(reinterpret_cast<char*>(buf), sizeof(buf));
    m_sceneCenter  = { buf[0], buf[1], buf[2] };
    m_sceneUp      = { buf[3], buf[4], buf[5] };
    m_sceneScale   = buf[6];
    m_cameraAngleX = buf[7];
}

Image INerfTrainer::renderC2W(const float c2w[16], float focal, int width, int height) {
    Image img;
    img.width = width; img.height = height; img.rgb.assign((size_t)width * height * 3, 0.0f);
    if (!m_nerf || width <= 0 || height <= 0) return img;
    int numRays = width * height;

    float3 *d_o = nullptr, *d_d = nullptr; float* d_rgb = nullptr;
    CUDA_CHECK(cudaMalloc(&d_o,   numRays * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_d,   numRays * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_rgb, numRays * 3 * sizeof(float)));

    int threads = 256, blocks = (numRays + threads - 1) / threads;
    generate_custom_rays_kernel<<<blocks, threads>>>(
        width, height, focal,
        c2w[0], c2w[1], c2w[2],  c2w[3],
        c2w[4], c2w[5], c2w[6],  c2w[7],
        c2w[8], c2w[9], c2w[10], c2w[11],
        d_o, d_d);

    m_nerf->setMemoryMode(INFERENCE);
    m_nerf->renderImageHit(d_o, d_d, (uint32_t)numRays, d_rgb, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(img.rgb.data(), d_rgb, numRays * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_o); cudaFree(d_d); cudaFree(d_rgb);
    return img;
}

Image INerfTrainer::renderOrbit(float azimuthDeg, float elevationDeg, float radius,
                                float fovDeg, int width, int height) {
    if (!m_nerf) return Image{};

    float3 center = make_float3(m_sceneCenter.x, m_sceneCenter.y, m_sceneCenter.z);
    float3 up     = make_float3(m_sceneUp.x, m_sceneUp.y, m_sceneUp.z);
    float ul = sqrtf(up.x*up.x + up.y*up.y + up.z*up.z); if (ul > 0) { up.x/=ul; up.y/=ul; up.z/=ul; }

    float3 ref = (fabsf(up.y) < 0.9f) ? make_float3(0,1,0) : make_float3(1,0,0);
    float3 a = make_float3(up.y*ref.z - up.z*ref.y, up.z*ref.x - up.x*ref.z, up.x*ref.y - up.y*ref.x);
    float al = sqrtf(a.x*a.x + a.y*a.y + a.z*a.z); a.x/=al; a.y/=al; a.z/=al;
    float3 b = make_float3(up.y*a.z - up.z*a.y, up.z*a.x - up.x*a.z, up.x*a.y - up.y*a.x);

    float az = azimuthDeg * kPI / 180.0f, el = elevationDeg * kPI / 180.0f;
    float3 planar = make_float3(cosf(az)*a.x + sinf(az)*b.x,
                                cosf(az)*a.y + sinf(az)*b.y,
                                cosf(az)*a.z + sinf(az)*b.z);
    float3 dir = make_float3(cosf(el)*planar.x + sinf(el)*up.x,
                             cosf(el)*planar.y + sinf(el)*up.y,
                             cosf(el)*planar.z + sinf(el)*up.z);
    float3 eye = make_float3(center.x + radius*dir.x, center.y + radius*dir.y, center.z + radius*dir.z);

    float3 f = make_float3(center.x-eye.x, center.y-eye.y, center.z-eye.z);
    float fl = sqrtf(f.x*f.x + f.y*f.y + f.z*f.z); f.x/=fl; f.y/=fl; f.z/=fl;
    float3 r = make_float3(f.y*up.z - f.z*up.y, f.z*up.x - f.x*up.z, f.x*up.y - f.y*up.x);
    float rl = sqrtf(r.x*r.x + r.y*r.y + r.z*r.z); r.x/=rl; r.y/=rl; r.z/=rl;
    float3 u = make_float3(r.y*f.z - r.z*f.y, r.z*f.x - r.x*f.z, r.x*f.y - r.y*f.x);

    float c2w[16] = {
        r.x, u.x, -f.x, eye.x,
        r.y, u.y, -f.y, eye.y,
        r.z, u.z, -f.z, eye.z,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    float focal = 0.5f * width / tanf(0.5f * fovDeg * kPI / 180.0f);
    return renderC2W(c2w, focal, width, height);
}

Image INerfTrainer::renderTrainView(int index, int width, int height) {
    if (!m_nerf || !m_dataloader) return Image{};
    if (width  <= 0) width  = m_dataloader->getWidth();
    if (height <= 0) height = m_dataloader->getHeight();
    float c2w[16];
    m_dataloader->getFramePose(index, c2w);
    float focal = 0.5f * width / tanf(0.5f * m_dataloader->getCameraAngleX());
    return renderC2W(c2w, focal, width, height);
}

float INerfTrainer::validate() {
    if (!m_nerf || !m_dataloader) return 0.0f;
    const int HOLDOUT = 8;
    int pixels  = m_dataloader->getWidth() * m_dataloader->getHeight();
    if (pixels <= 0) return 0.0f;
    int numImgs = m_dataloader->getNumImages();
    int tile    = (int)m_masterConfig.rayChunkSize;

    m_nerf->setMemoryMode(INFERENCE);
    float3 valBg = make_float3(1.0f, 1.0f, 1.0f);
    CUDA_CHECK(cudaMemset(m_d_loss, 0, sizeof(float)));
    long long totalPixels = 0;

    for (int i = 0; i < numImgs; ++i) {
        if (i % HOLDOUT != 0) continue;
        for (int off = 0; off < pixels; off += tile) {
            int count = std::min(tile, pixels - off);
            m_dataloader->fetchRayChunk(i * pixels + off, count, 0, valBg, 0, true, 0);
            CUDA_CHECK(cudaDeviceSynchronize());
            m_nerf->renderImageHit(m_dataloader->getChunkRaysO(), m_dataloader->getChunkRaysD(),
                                   (uint32_t)count, m_d_chunkRgb, 0);
            CUDA_CHECK(cudaDeviceSynchronize());
            int threads = 256, blocks = (count + threads - 1) / threads;
            mse_kernel<<<blocks, threads>>>(m_d_chunkRgb, m_dataloader->getChunkRgbTrue(), m_d_loss, count);
            totalPixels += count;
        }
    }
    float h_loss = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_loss, m_d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    float mse = h_loss / (float)std::max(1LL, totalPixels);
    return -10.0f * log10f(fmaxf(mse, 1e-8f));
}

int INerfTrainer::numTrainViews() const {
    if (!m_dataloader) return 0;
    int px = m_dataloader->getWidth() * m_dataloader->getHeight();
    return px > 0 ? (int)(m_total_rays / px) : 0;
}
