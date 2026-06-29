#include "../BakedNerf.h"
#include <vector>
#include <string>
#include <cstdlib>

int main(int argc, char** argv) {
    // Args: any token containing ".inerf" is a model path; numeric tokens are sigma thresholds.
    std::vector<std::string> models;
    std::vector<float> sigmas;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a.find(".inerf") != std::string::npos) models.push_back(a);
        else sigmas.push_back((float)atof(a.c_str()));
    }
    if (models.empty()) models.push_back("../benchmarks/saved/model.inerf");
    if (sigmas.empty()) sigmas = {0.005f, 0.01f, 0.02f, 0.03f, 0.05f, 0.1f};

    for (const std::string& modelPath : models) {
        printf("\n================ model: %s ================\n", modelPath.c_str());
        InstantNerf nerf;
        nerf.load(modelPath);

        const NerfOptions& topts = nerf.options();
        printf("gridRes %dx%dx%d  cascades %d  mips %d  trained minDensityThreshold %.4f\n",
               topts.gridResolution.x, topts.gridResolution.y, topts.gridResolution.z,
               topts.numCascades, topts.levelsMipmap, topts.minDensityThreshold);

        for (float s : sigmas) {
            BakeOptions opts;
            uint3 gr = topts.gridResolution;
            opts.voxelGridResolution = make_uint3(gr.x * SPARSE_B, gr.y * SPARSE_B, gr.z * SPARSE_B);
            opts.sigmaThreshold = s;
            opts.queryBatch     = 1 << 17;

            BakedNerf bnerf;
            bnerf.init(opts);
            bnerf.distil(nerf);
        }
    }
    return 0;
}
