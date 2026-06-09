#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <cstdlib>

// Clean library import as requested!
#include <NeRF/Pipeline.h>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    std::vector<std::string> args(argv, argv + argc);

    if (args.size() < 3 || args[1] != "train" || args[2] != "--data") {
        std::cout << "Usage: 3DMaker.exe train --data <path_to_dataset>\n" << std::endl;
        std::cout << "Available commands:" << std::endl;
        std::cout << "  train    Run the end-to-end COLMAP & NeRF training pipeline" << std::endl;
        return 1;
    }

    std::string dataset_path = args[3];
    fs::path dataset_dir(dataset_path);
    std::string python_arg = dataset_path;
    
    // If it's a video file, the python script will output to a folder of the same name.
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

    // Call the encapsulated CUDA training pipeline from our library!
    run_training_pipeline(dataset_path, 10, 5000);

    return 0;
}
