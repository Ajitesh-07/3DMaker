#include <iostream>
#include <vector>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../TinyMLP.h"
#include "../TinyMLPHashGrid.h"

// Macro to expect exception
#define EXPECT_THROW(stmt) \
    do { \
        bool threw = false; \
        try { stmt; } catch (const std::exception& e) { threw = true; } \
        if (!threw) { std::cerr << "Expected exception not thrown at line " << __LINE__ << std::endl; exit(1); } \
    } while(0)

#define EXPECT_NO_THROW(stmt) \
    do { \
        try { stmt; } catch (const std::exception& e) { \
            std::cerr << "Unexpected exception thrown at line " << __LINE__ << ": " << e.what() << std::endl; exit(1); \
        } \
    } while(0)

void printVRAM(const std::string& tag) {
    size_t free_byte;
    size_t total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);
    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double used_db = total_db - free_db;
    std::cout << "[VRAM] " << tag << " -> Used VRAM: " << used_db / (1024.0 * 1024.0) << " MB\n";
}

void testTinyMLPModes() {
    std::cout << "\n--- Testing TinyMLP Modes ---\n";
    MLPOption opt;
    opt.inputDim = 16;
    opt.hiddenDim = 32;
    opt.outputDim = 16;
    opt.numLayers = 3;
    opt.activationType = 0;

    int batchSize = 1024;

    // Start in inference mode
    TinyMLP mlp(opt, batchSize, batchSize, 42, false);

    half* d_inputs;
    float* d_outputs;
    cudaMalloc(&d_inputs, batchSize * opt.inputDim * sizeof(half));
    cudaMalloc(&d_outputs, batchSize * opt.outputDim * sizeof(float));

    // Inference should not throw
    EXPECT_NO_THROW(mlp.inference(d_inputs, d_outputs, batchSize));

    // Training methods should throw
    EXPECT_THROW(mlp.forward(d_inputs, d_outputs, batchSize));
    EXPECT_THROW(mlp.zero_grad());
    EXPECT_THROW(mlp.backward(batchSize));
    EXPECT_THROW(mlp.step());

    // Switch to training mode
    mlp.switchToTrainingMode();

    // Now forward should not throw
    EXPECT_NO_THROW(mlp.forward(d_inputs, d_outputs, batchSize));
    EXPECT_NO_THROW(mlp.zero_grad());

    // Switch back to inference
    mlp.switchToInferenceMode();
    EXPECT_THROW(mlp.forward(d_inputs, d_outputs, batchSize));
    EXPECT_NO_THROW(mlp.inference(d_inputs, d_outputs, batchSize));

    cudaFree(d_inputs);
    cudaFree(d_outputs);
    std::cout << "TinyMLP Modes Test Passed!\n";
}

void testTinyMLPHashGridModes() {
    std::cout << "\n--- Testing TinyMLPHashGrid VRAM & Modes ---\n";
    MLPGridOptions opt;
    opt.vectorDim = 4;   // No input padding needed!
    opt.hiddenDim = 64;
    opt.outputDim = 16;  // No output padding needed!
    opt.numLayers = 2;
    opt.activationType = 0;
    opt.tableSize = 1024;
    opt.numLevels = 4;
    opt.b = 1.5f;
    opt.lowestSize = 16;
    opt.featuresLevel = 2;

    int batchSize = 1024 * 512; // Large batch size to easily spot VRAM difference (512K)

    cudaDeviceSynchronize();
    printVRAM("Baseline (Before Allocation)");

    {
        // Start purely in inference mode
        TinyMLPHashGrid grid_infer(opt, batchSize, batchSize, 42, false);
        cudaDeviceSynchronize();
        printVRAM("Grid Instantiated in Inference Mode (0 VRAM for Batch Buffers expected!)");
    }

    cudaDeviceSynchronize();
    printVRAM("After Deleting Inference Grid");

    float* d_inputs;
    float* d_outputs;
    cudaMalloc(&d_inputs, batchSize * opt.vectorDim * sizeof(float));
    cudaMalloc(&d_outputs, batchSize * opt.outputDim * sizeof(float));

    {
        // Start in training mode
        TinyMLPHashGrid grid_train(opt, batchSize, batchSize, 42, true);
        cudaDeviceSynchronize();
        printVRAM("Grid Instantiated in Training Mode");

        // Forward should work
        EXPECT_NO_THROW(grid_train.forward(d_inputs, d_outputs, batchSize));

        // Switch to inference
        grid_train.switchToInferenceMode();
        cudaDeviceSynchronize();
        printVRAM("Grid Switched to Inference Mode");

        // Inference should work, forward should throw
        EXPECT_NO_THROW(grid_train.inference(d_inputs, d_outputs, batchSize));
        EXPECT_THROW(grid_train.forward(d_inputs, d_outputs, batchSize));
        EXPECT_THROW(grid_train.zero_grad());
    }

    cudaFree(d_inputs);
    cudaFree(d_outputs);
    std::cout << "TinyMLPHashGrid Modes Test Passed!\n";
}

int main() {
    try {
        testTinyMLPModes();
        testTinyMLPHashGridModes();
        std::cout << "\nAll mode tests passed successfully!\n";
    } catch (const std::exception& e) {
        std::cerr << "Test failed with error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
