#pragma once

#include <string>

void run_training_pipeline(const std::string& dataset_path, int totalEpochs, int totalSteps = 0, int numCascades = 0);
