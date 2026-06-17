#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <cstdlib>

#include <NeRF/Pipeline.h>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    std::vector<std::string> args(argv, argv + argc);

    if (args.size() < 4 || args[1] != "train" || args[2] != "--data") {
        std::cout << "Usage: 3DMaker.exe train --data <path_to_dataset> [options]\n" << std::endl;
        std::cout << "Runs the end-to-end COLMAP (if needed) + NeRF training pipeline.\n" << std::endl;
        std::cout << "Options (all optional):" << std::endl;
        std::cout << "  --steps N      training-step cap                    (default 50000)" << std::endl;
        std::cout << "  --epochs N     outer-loop epoch cap                 (default 100)" << std::endl;
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

    // Optional named training flags (parsed after `train --data <path>`).
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

    run_training_pipeline(dataset_path, p_epochs, p_steps, p_cascades,
                          p_lambda, p_K, p_validate, p_video, p_maxLR, p_minLR);

    return 0;
}
