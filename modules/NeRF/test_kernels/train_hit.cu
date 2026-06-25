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
#include <unordered_map>
#include <unordered_set>

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

static const char* HELP =
    "train_hit -- thin CLI over the INerfTrainer facade\n\n"
    "Usage:\n"
    "  train_hit <dataset> [--flag value ...]\n\n"
    "dataset is positional; everything else is a named flag (override only what you\n"
    "need, any order). Booleans (--validate/--video) work bare or with 0/1. Defaults\n"
    "come from NerfConfig:\n"
    "  --lambda-dist V     distortion-loss weight (pair ~0.1/K)         (0.1)\n"
    "  --k N               samplesPerVoxel: sub-voxel samples / voxel   (1)\n"
    "  --max-steps N       training-step cap                           (15000)\n"
    "  --cascades N        occupancy cascades; 0 = dataset estimate     (0 = auto)\n"
    "  --validate          hold out every 8th image, report test PSNR   (off)\n"
    "  --video             render a 360 orbit mp4 after training        (off)\n"
    "  --max-lr V          anneal-and-hold schedule start LR            (0.01)\n"
    "  --min-lr V          anneal-and-hold schedule floor LR            (0.001)\n"
    "  --min-density V     per-cascade opacity cull bar; raise to prune (0.01)\n"
    "  --lambda-depth V    DS-NeRF depth-supervision weight; 0 = off     (0.01)\n"
    "  --depth-fraction V  frac of each chunk drawn as depth rays [0..1] (0.125)\n"
    "  dataset             dir with transforms_train.json + images       (../../data/nerf_synthetic/lego)\n\n"
    "Examples:\n"
    "  train_hit ../../data/video --k 2 --validate --lambda-depth 0.02\n"
    "  train_hit ../../data/video --k 4 --cascades 1 --validate --min-density 0.03\n"
    "  train_hit --help\n";

int main(int argc, char** argv) {
    // ----- Named-flag CLI. dataset positional; `--key value` for the rest. Defaults
    //       live in NerfConfig, so each flag falls back to cfg's own field value. -----
    NerfConfig cfg;
    int  maxSteps    = 15000;
    bool renderVideo = false;
    std::string dataset = "../../data/nerf_synthetic/lego";

    static const std::unordered_set<std::string> KNOWN = {
        "lambda-dist", "k", "max-steps", "cascades", "validate", "video",
        "max-lr", "min-lr", "min-density", "lambda-depth", "depth-fraction"
    };
    static const std::unordered_set<std::string> BOOLS = { "validate", "video" };

    std::unordered_map<std::string, std::string> flags;
    for (int a = 1; a < argc; ++a) {
        std::string arg = argv[a];
        if (arg == "--help" || arg == "-h") { std::cout << HELP; return 0; }

        if (arg.rfind("--", 0) == 0) {
            std::string key = arg.substr(2);
            if (!KNOWN.count(key)) {
                std::cerr << "[train_hit] unknown flag '" << arg << "' (try --help)\n";
                return 1;
            }
            if (BOOLS.count(key)) {
                // bare --flag = true; only swallow the next token if it's an explicit 0/1
                std::string nxt = (a + 1 < argc) ? argv[a + 1] : "";
                flags[key] = (nxt == "0" || nxt == "1") ? (++a, nxt) : "1";
            } else if (a + 1 < argc && std::string(argv[a + 1]).rfind("--", 0) != 0) {
                flags[key] = argv[++a];
            } else {
                std::cerr << "[train_hit] flag '" << arg << "' needs a value (try --help)\n";
                return 1;
            }
        } else {
            dataset = arg;   // the one positional
        }
    }

    auto getf = [&](const char* k, float d){ auto it = flags.find(k); return it != flags.end() ? (float)atof(it->second.c_str()) : d; };
    auto geti = [&](const char* k, int   d){ auto it = flags.find(k); return it != flags.end() ? atoi(it->second.c_str())       : d; };
    auto getb = [&](const char* k, bool  d){ auto it = flags.find(k); return it != flags.end() ? (atoi(it->second.c_str()) != 0) : d; };

    cfg.lambdaDist          = getf("lambda-dist",    cfg.lambdaDist);
    cfg.sampleK             = geti("k",              cfg.sampleK);
    maxSteps                = geti("max-steps",      maxSteps);
    cfg.numCascades         = geti("cascades",       cfg.numCascades);
    cfg.shouldValidate      = getb("validate",       cfg.shouldValidate);
    renderVideo             = getb("video",          renderVideo);
    cfg.maxLR               = getf("max-lr",         cfg.maxLR);
    cfg.minLR               = getf("min-lr",         cfg.minLR);
    cfg.minDensityThreshold = getf("min-density",    cfg.minDensityThreshold);
    cfg.lambdaDepth         = getf("lambda-depth",   cfg.lambdaDepth);
    cfg.depthFraction       = getf("depth-fraction", cfg.depthFraction);

    std::cout << "[train_hit] dataset=" << dataset
              << " lambdaDist=" << cfg.lambdaDist << " K=" << cfg.sampleK
              << " maxSteps=" << maxSteps << " cascades=" << cfg.numCascades
              << " validate=" << cfg.shouldValidate << " video=" << renderVideo
              << " maxLR=" << cfg.maxLR << " minLR=" << cfg.minLR
              << " minDensity=" << cfg.minDensityThreshold
              << " lambdaDepth=" << cfg.lambdaDepth
              << " depthFraction=" << cfg.depthFraction << std::endl;

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
