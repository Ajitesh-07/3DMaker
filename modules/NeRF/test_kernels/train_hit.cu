// Thin CLI wrapper over the INerfTrainer facade. The whole training/validation/render pipeline
// now lives in NerfTrainer.{h,cu}; this file just parses args, drives the facade, and writes PNGs.
#include "../NerfTrainer.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <filesystem>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../../third_party/stb_image_write.h"

// Save a facade Image (float RGB in [0,1]) as a PNG.
static void saveImagePNG(const Image& img, const std::string& path) {
    if (img.width <= 0 || img.height <= 0) return;
    std::vector<uint8_t> bytes((size_t)img.width * img.height * 3);
    for (size_t i = 0; i < bytes.size(); ++i) {
        float v = std::min(1.0f, std::max(0.0f, img.rgb[i]));
        bytes[i] = (uint8_t)(v * 255.0f + 0.5f);
    }
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    stbi_write_png(path.c_str(), img.width, img.height, 3, bytes.data(), img.width * 3);
}

int main(int argc, char** argv) {
    for (int a = 1; a < argc; ++a) {
        std::string arg = argv[a];
        if (arg == "--help" || arg == "-h") {
            std::cout <<
                "train_hit -- thin CLI over the INerfTrainer facade\n\n"
                "Usage:\n"
                "  train_hit [dataset] [lambdaDist] [K] [maxSteps] [numCascades] [profiling] [validate] [video] [maxLR] [minLR] [minDensity]\n\n"
                "Positional args (all optional, parsed left-to-right):\n"
                "  dataset      dataset dir with transforms_train.json + images   (default ../../data/nerf_synthetic/lego)\n"
                "  lambdaDist   distortion-loss weight (pair ~0.1/K)               (default 0.1)\n"
                "  K            samplesPerVoxel: sub-voxel samples / occupied voxel (default 1)\n"
                "  maxSteps     training-step cap                                  (default 15000)\n"
                "  numCascades  occupancy cascades; 0 = take the dataset estimate  (default 0 = auto)\n"
                "  profiling    (ignored; not exposed by the facade yet)           (default 0)\n"
                "  validate     1 = hold out every 8th image, report test PSNR     (default 0)\n"
                "  video        1 = render a 360 orbit mp4 after training          (default 0)\n"
                "  maxLR        anneal-and-hold schedule start LR                  (default 0.01)\n"
                "  minLR        anneal-and-hold schedule floor LR                  (default 0.001)\n"
                "  minDensity   per-cascade opacity cull bar; raise to prune fog   (default 0.01)\n\n"
                "Examples:\n"
                "  train_hit ../../data/video_chair 0.05 2 20000 0 0 1 0          # K=2 + paired lambda, validate\n"
                "  train_hit ../../data/video 0.1 4 20000 1 0 1 0 1e-3 1e-4 0.03  # prune floaters\n"
                "  train_hit --help\n";
            return 0;
        }
    }

    // ----- Parse positional args into a NerfConfig (defaults are the parity config) -----
    std::string dataset = (argc > 1) ? argv[1] : "../../data/nerf_synthetic/lego";
    NerfConfig cfg;                       // lambdaDist=0.1, sampleK=1, numCascades=0(auto),
    int  maxSteps    = 15000;             // maxLR=0.01, minLR=0.001, minDensity=0.01
    bool renderVideo = false;

    if (argc >  2) cfg.lambdaDist          = (float)atof(argv[2]);
    if (argc >  3) cfg.sampleK             = atoi(argv[3]);
    if (argc >  4) maxSteps                = atoi(argv[4]);
    if (argc >  5) cfg.numCascades         = atoi(argv[5]);   // 0 = auto from dataset
    // argv[6] (profiling) is parsed historically but the facade doesn't expose it -> ignored.
    if (argc >  7) cfg.shouldValidate      = (atoi(argv[7]) != 0);
    if (argc >  8) renderVideo             = (atoi(argv[8]) != 0);
    if (argc >  9) cfg.maxLR               = (float)atof(argv[9]);
    if (argc > 10) cfg.minLR               = (float)atof(argv[10]);
    if (argc > 11) cfg.minDensityThreshold = (float)atof(argv[11]);

    std::cout << "[train_hit] dataset=" << dataset
              << " lambda=" << cfg.lambdaDist << " K=" << cfg.sampleK
              << " maxSteps=" << maxSteps << " cascades=" << cfg.numCascades
              << " validate=" << cfg.shouldValidate << " video=" << renderVideo
              << " maxLR=" << cfg.maxLR << " minLR=" << cfg.minLR
              << " minDensity=" << cfg.minDensityThreshold << std::endl;

    // ----- Drive the facade -----
    INerfTrainer trainer;
    trainer.init(cfg);
    trainer.loadDataset(dataset);
    trainer.train(maxSteps);   // blocking train with the live progress bar (rendered by the facade)

    if (cfg.shouldValidate) {
        float psnr = trainer.validate();
        std::cout << "\n========================================\n"
                  << "  HELD-OUT VALIDATION PSNR: " << std::fixed << std::setprecision(2)
                  << psnr << " dB\n"
                  << "========================================" << std::endl;
    }

    // Save a few reconstructed training views to eyeball quality.
    int views = std::min(3, trainer.numTrainViews());
    for (int i = 0; i < views; ++i) {
        Image img = trainer.renderTrainView(i, 0, 0);   // native resolution
        std::string fn = "../benchmarks/frames/view_" + std::to_string(i) + ".png";
        saveImagePNG(img, fn);
        std::cout << "Saved " << fn << std::endl;
    }

    if (renderVideo) {
        std::cout << "\nRendering 360 orbit (120 frames)..." << std::endl;
        for (int frame = 0; frame < 120; ++frame) {
            float az = frame * (360.0f / 120.0f);
            Image img = trainer.renderOrbit(az, /*elevationDeg=*/20.0f, /*radius=*/3.0f,
                                            /*fovDeg=*/50.0f, 800, 800);
            char fn[256];
            snprintf(fn, sizeof(fn), "../benchmarks/frames/video_frame_%03d.png", frame);
            saveImagePNG(img, fn);
            if (frame % 10 == 0) std::cout << "  frame " << frame << "/120" << std::endl;
        }
        std::system("ffmpeg -y -framerate 30 -i ../benchmarks/frames/video_frame_%03d.png "
                    "-c:v libx264 -pix_fmt yuv420p ../benchmarks/nerf_360.mp4");
    }

    trainer.save("../benchmarks/saved/model.inerf");
    std::cout << "Saved ../benchmarks/saved/model.inerf\nTraining Complete." << std::endl;
    return 0;
}
