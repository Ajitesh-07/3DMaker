#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <algorithm>

#include <NeRF/NerfTrainer.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../third_party/stb_image_write.h"

namespace fs = std::filesystem;

static void saveImagePNG(const Image& img, const fs::path& path) {
    if (img.width <= 0 || img.height <= 0) return;
    std::vector<uint8_t> bytes((size_t)img.width * img.height * 3);
    for (size_t i = 0; i < bytes.size(); ++i) {
        float v = std::min(1.0f, std::max(0.0f, img.rgb[i]));
        bytes[i] = (uint8_t)(v * 255.0f + 0.5f);
    }
    fs::create_directories(path.parent_path());
    stbi_write_png(path.string().c_str(), img.width, img.height, 3, bytes.data(), img.width * 3);
}

int main(int argc, char** argv) {
    std::vector<std::string> args(argv, argv + argc);

    if (args.size() < 4 || args[1] != "train" || args[2] != "--data") {
        std::cout << "Usage: 3DMaker.exe train --data <path_to_dataset> [options]\n" << std::endl;
        std::cout << "Runs the end-to-end COLMAP (if needed) + NeRF training pipeline.\n" << std::endl;
        std::cout << "Options (all optional):" << std::endl;
        std::cout << "  --steps N      training-step cap                    (default 50000)" << std::endl;
        std::cout << "  --epochs N     outer-loop epoch cap (step-bounded)  (default 100)" << std::endl;
        std::cout << "  --cascades N   occupancy cascades, 0 = dataset est. (default 0)" << std::endl;
        std::cout << "  --lambda F     distortion-loss weight              (default 0.1)" << std::endl;
        std::cout << "  --K N          sub-voxel samples per occupied voxel (default 1)" << std::endl;
        std::cout << "  --max-lr F     cosine LR schedule start             (default 0.01)" << std::endl;
        std::cout << "  --min-lr F     cosine LR schedule floor             (default 0.0001)" << std::endl;
        std::cout << "  --validate     hold out every 8th image, report test PSNR" << std::endl;
        std::cout << "  --video        render the 360 orbit mp4 after training" << std::endl;
        std::cout << "\nExample:" << std::endl;
        std::cout << "  3DMaker.exe train --data captures/chair --steps 50000 --cascades 4 --lambda 0.15 --K 2 --validate" << std::endl;
        return 1;
    }

    int   p_steps = 50000, p_epochs = 100, p_cascades = 0, p_K = 1;
    float p_lambda = 0.1f, p_maxLR = 1e-2f, p_minLR = 1e-4f;
    bool  p_validate = false, p_video = false;
    for (size_t i = 4; i < args.size(); ++i) {
        const std::string& a = args[i];
        if      (a == "--steps"    && i + 1 < args.size()) p_steps    = std::stoi(args[++i]);
        else if (a == "--epochs"   && i + 1 < args.size()) p_epochs   = std::stoi(args[++i]);
        else if (a == "--cascades" && i + 1 < args.size()) p_cascades = std::stoi(args[++i]);
        else if (a == "--lambda"   && i + 1 < args.size()) p_lambda   = std::stof(args[++i]);
        else if (a == "--K"        && i + 1 < args.size()) p_K        = std::stoi(args[++i]);
        else if (a == "--max-lr"   && i + 1 < args.size()) p_maxLR    = std::stof(args[++i]);
        else if (a == "--min-lr"   && i + 1 < args.size()) p_minLR    = std::stof(args[++i]);
        else if (a == "--validate") p_validate = true;
        else if (a == "--video")    p_video    = true;
        else std::cerr << "Warning: ignoring unknown argument '" << a << "'" << std::endl;
    }

    std::string dataset_path = args[3];
    fs::path dataset_dir(dataset_path);
    std::string python_arg = dataset_path;

    if (fs::is_regular_file(dataset_dir)) {
        dataset_dir = dataset_dir.parent_path() / dataset_dir.stem();
        dataset_path = dataset_dir.string();
    }

    fs::path transforms_train = dataset_dir / "transforms_train.json";

    if (!fs::exists(transforms_train)) {
        std::cout << "transforms_train.json not found. Launching COLMAP pipeline..." << std::endl;

        fs::path exe_path = fs::absolute(fs::path(argv[0])).parent_path();
        fs::path script_path = exe_path / "scripts" / "colmap2nerf.py";

        if (!fs::exists(script_path)) {
            script_path = exe_path.parent_path() / "scripts" / "colmap2nerf.py";
        }
        if (!fs::exists(script_path)) {
            script_path = exe_path.parent_path().parent_path() / "scripts" / "colmap2nerf.py";
        }
        if (!fs::exists(script_path)) {
            script_path = fs::current_path() / "scripts" / "colmap2nerf.py";
        }

        if (!fs::exists(script_path)) {
            std::cerr << "Error: Could not find scripts/colmap2nerf.py" << std::endl;
            return 1;
        }

        std::string cmd = "python \"" + script_path.string() + "\" \"" + python_arg + "\"";
        std::cout << "Executing: " << cmd << std::endl;

        int ret = std::system(cmd.c_str());
        if (ret != 0) {
            std::cerr << "Error: COLMAP pipeline failed." << std::endl;
            return 1;
        }
    } else {
        std::cout << "Found existing transforms_train.json. Skipping COLMAP pipeline." << std::endl;
    }

    std::cout << "\n============================================\n";
    std::cout << "       Starting 3DMaker NeRF Training       \n";
    std::cout << "============================================\n\n";

    NerfConfig cfg;
    cfg.lambdaDist     = p_lambda;
    cfg.sampleK        = p_K;
    cfg.numCascades    = p_cascades;    
    cfg.maxLR          = p_maxLR;
    cfg.minLR          = p_minLR;
    cfg.shouldValidate = p_validate;

    std::cout << "[3DMaker] dataset=" << dataset_path
              << " lambda=" << cfg.lambdaDist << " K=" << cfg.sampleK
              << " steps=" << p_steps << " cascades=" << cfg.numCascades
              << " validate=" << cfg.shouldValidate << " video=" << p_video
              << " maxLR=" << cfg.maxLR << " minLR=" << cfg.minLR << std::endl;

    INerfTrainer trainer;
    trainer.init(cfg);
    trainer.loadDataset(dataset_path);
    trainer.train(p_steps, p_epochs);

    if (p_validate) {
        float psnr = trainer.validate();
        std::cout << "\n========================================\n"
                  << "  HELD-OUT VALIDATION PSNR: " << psnr << " dB\n"
                  << "========================================" << std::endl;
    }

    int views = std::min(3, trainer.numTrainViews());
    for (int i = 0; i < views; ++i) {
        Image img = trainer.renderTrainView(i, 0, 0);   // native resolution
        fs::path fn = dataset_dir / "frames" / ("view_" + std::to_string(i) + ".png");
        saveImagePNG(img, fn);
        std::cout << "Saved " << fn.string() << std::endl;
    }

    if (p_video) {
        std::cout << "\nRendering 360 orbit (120 frames)..." << std::endl;
        for (int frame = 0; frame < 120; ++frame) {
            float az = frame * (360.0f / 120.0f);
            Image img = trainer.renderOrbit(az, 20.0f, 3.0f, 50.0f, 800, 800);
            char name[64];
            snprintf(name, sizeof(name), "orbit_%03d.png", frame);
            saveImagePNG(img, dataset_dir / "frames" / name);
            if (frame % 10 == 0) std::cout << "  frame " << frame << "/120" << std::endl;
        }
        fs::path pattern = dataset_dir / "frames" / "orbit_%03d.png";
        fs::path mp4     = dataset_dir / "nerf_360.mp4";
        std::string cmd = "ffmpeg -y -framerate 30 -i \"" + pattern.string() +
                          "\" -c:v libx264 -pix_fmt yuv420p \"" + mp4.string() + "\"";
        std::system(cmd.c_str());
        std::cout << "Saved " << mp4.string() << std::endl;
    }

    fs::path model_path = dataset_dir / "model.inerf";
    trainer.save(model_path.string());
    std::cout << "\nSaved trained model: " << model_path.string() << std::endl;
    std::cout << "View it with:  3DViewer.exe \"" << transforms_train.string()
              << "\" \"" << model_path.string() << "\"" << std::endl;
    std::cout << "Training Complete." << std::endl;
    return 0;
}
