#pragma once

#include <string>

// End-to-end training pipeline (parity with train_hit). numCascades=0 uses the dataset's estimate.
void run_training_pipeline(
    const std::string& dataset_path,
    int totalEpochs = 100,
    int totalSteps = 50000,
    int numCascades = 0,
    float lambdaDist = 0.1f,
    int samplesPerVoxel = 1,
    bool runValidation = false,
    bool renderVideo = false,
    float maxLR = 1e-2f,
    float minLR = 1e-4f
);
