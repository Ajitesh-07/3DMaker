#pragma once
#include <vector>
#include <random>
#include <cstdint>
#include <string>
#include "InstantNerf.h"  
#include "DataLoader.h"

struct Vec3 { float x, y, z; };

struct Image {
    int width = 0, height = 0;
    std::vector<float> rgb;
};

struct TrainStats {
    int   step        = 0;
    int   targetSteps = 0;
    float loss        = 0.0f;
    float psnr        = 0.0f;
    float lr          = 0.0f;
    float epochs      = 0.0f;
    float msPerStep   = 0.0f;
    float etaSeconds  = 0.0f;
};

struct NerfConfig {
    float lambdaDist          = 0.1f;
    int   numCascades         = 0;
    float minDensityThreshold = 0.01f;
    int   sampleK             = 1;
    int   batchSize           = 256 * 1024;
    int   renderBatchSize     = 256 * 1024;
    bool  shouldValidate      = false;

    float maxLR      = 1e-2f;
    float minLR      = 1e-3f;
    int   annealSteps = 0;
};

class INerfTrainer {
public:
    INerfTrainer() = default;
    ~INerfTrainer();

    INerfTrainer(const INerfTrainer&) = delete;
    INerfTrainer& operator=(const INerfTrainer&) = delete;

    void init(const NerfConfig& config);
    void loadDataset(const std::string& path);

    void train(int maxSteps, int maxEpochs = 0);

    void beginTraining(int totalSteps);
    TrainStats trainSteps(int nSteps);
    bool isTrainingDone() const { return m_targetSteps > 0 && m_globalStep >= m_targetSteps; }
    const std::vector<float>& lossHistory() const { return m_lossHistory; }
    int   step() const { return m_globalStep; }
    int   targetSteps() const { return m_targetSteps; }
    float epochs() const { return (m_total_rays > 0) ? (float)(m_raysSeen / (double)m_total_rays) : 0.0f; }
    float msPerStep() const { return (float)m_emaMsPerStep; }

    float validate();

    void save(const std::string& path);
    void load(const std::string& path);

    Vec3  sceneCenter() const { return m_sceneCenter; }
    Vec3  sceneUp() const { return m_sceneUp; }

    Image renderOrbit(float azimuthDeg, float elevationDeg, float radius,
                      float fovDeg = 50.0f, int width = 800, int height = 800);
    Image renderTrainView(int index, int width = 0, int height = 0);
    int   numTrainViews() const;

private:
    void  buildModel();                          
    float trainOneChunk(bool computeLoss);       
    float currentLR() const;                     
    Image renderC2W(const float c2w[16], float focal, int width, int height);  

    NerfConfig   m_config;
    NerfOptions  m_masterConfig;
    InstantNerf* m_nerf       = nullptr;
    DataLoader*  m_dataloader = nullptr;

    uint32_t m_holdout_every = 0;
    uint32_t m_total_rays    = 0;

    Vec3  m_sceneCenter  = {0.0f, 0.0f, 0.0f};
    Vec3  m_sceneUp      = {0.0f, 1.0f, 0.0f};
    float m_sceneScale   = 1.0f;
    float m_cameraAngleX = 0.0f;

    int      m_globalStep  = 0;
    int      m_targetSteps = 0;
    uint32_t m_rayCursor   = 0;
    uint32_t m_epochSeed   = 0;
    std::vector<float> m_lossHistory;

    double   m_raysSeen     = 0.0;  
    double   m_emaMsPerStep = 0.0; 

    float* m_d_loss     = nullptr;
    float* m_d_chunkRgb = nullptr;

    std::mt19937 m_rng{42};
    std::uniform_real_distribution<float>   m_dist{0.0f, 1.0f};
    std::uniform_int_distribution<uint32_t> m_seed_dist{0u, 0xFFFFFFFFu};
};
