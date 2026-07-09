// Part-1 distillation test over saved teachers.
// For each *.inerf in ../benchmarks/saved (x each sigma):
//   1. run BakedNerf::distil() STANDALONE (no prior diagonstic — proves distil computes its
//      own bake threshold and depends on nothing external),
//   2. poll device free-memory on a background thread to capture the true PEAK VRAM during it,
//   3. run diagonstic() afterwards as an INDEPENDENT ORACLE and cross-check that distil's
//      deduped vertex count (numVoxels()) equals diagonstic's bitset count (uniqueSurvivors).
//      A match validates buildGrid's per-vertex ownership/dedup logic (two unrelated methods agree).
// Usage: test_baking [sigma ...] [--dirs N] [--only substr] [--no-distil]
#include "../BakedNerf.h"
#include <vector>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <filesystem>
#include <thread>
#include <atomic>
#include <chrono>
#include <cuda_runtime.h>

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);   // unbuffered so we see progress even on a crash
    std::vector<float> sigmas;
    int  dirs = -1;                 // override BakeOptions.bakeDiffuseN when > 0
    std::string only;               // substring filter on model path
    bool doDistil = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--dirs" && i + 1 < argc) dirs = atoi(argv[++i]);
        else if (a == "--only" && i + 1 < argc) only = argv[++i];
        else if (a == "--no-distil")            doDistil = false;
        else sigmas.push_back((float)atof(a.c_str()));
    }
    if (sigmas.empty()) sigmas = { 0.03f };

    std::vector<std::string> models;
    const char* dir = "../benchmarks/saved";
    if (std::filesystem::is_directory(dir)) {
        for (const auto& e : std::filesystem::directory_iterator(dir))
            if (e.path().extension() == ".inerf") models.push_back(e.path().string());
        std::sort(models.begin(), models.end());
    }
    if (!only.empty())
        models.erase(std::remove_if(models.begin(), models.end(),
            [&](const std::string& m){ return m.find(only) == std::string::npos; }), models.end());
    if (models.empty()) { fprintf(stderr, "no .inerf models found in %s (filter '%s')\n", dir, only.c_str()); return 1; }

    size_t f0 = 0, totalVram = 0; cudaMemGetInfo(&f0, &totalVram);
    printf("GPU total VRAM: %.2f GB   free at start: %.2f GB\n", totalVram / 1e9, f0 / 1e9);

    struct Row { std::string model; float sigma; uint32_t blocks; uint32_t distilU; long long oracleU;
                 double peakGB; double distilGB; bool ok; bool match; };
    std::vector<Row> rows;

    for (const std::string& modelPath : models) {
        printf("\n================ model: %s ================\n", modelPath.c_str());
        InstantNerf nerf;
        nerf.load(modelPath);
        const NerfOptions& topts = nerf.options();

        for (float s : sigmas) {
            BakeOptions opts;
            uint3 gr = topts.gridResolution;
            opts.voxelGridResolution = make_uint3(gr.x * SPARSE_B, gr.y * SPARSE_B, gr.z * SPARSE_B);
            opts.sigmaThreshold = s;
            opts.fineTuneSteps  = 0;
            if (dirs > 0) opts.bakeDiffuseN = dirs;

            BakedNerf bnerf;
            bnerf.init(opts);

            uint32_t distilU = 0;
            double peakGB = 0, distilGB = 0;
            bool ok = true;

            if (doDistil) {
                cudaDeviceSynchronize();
                size_t baseFree = 0, tot = 0; cudaMemGetInfo(&baseFree, &tot);

                // Background poller: record the minimum free-memory seen while distil runs.
                std::atomic<size_t> minFree{baseFree};
                std::atomic<bool>   running{true};
                std::thread poller([&]{
                    while (running.load(std::memory_order_relaxed)) {
                        size_t fr = 0, tt = 0;
                        if (cudaMemGetInfo(&fr, &tt) == cudaSuccess) {
                            size_t cur = minFree.load(std::memory_order_relaxed);
                            while (fr < cur && !minFree.compare_exchange_weak(cur, fr)) {}
                        }
                        std::this_thread::sleep_for(std::chrono::microseconds(300));
                    }
                });

                auto t0 = std::chrono::steady_clock::now();
                try {
                    bnerf.bakeGeometry(nerf, 0);
                    cudaError_t e = cudaDeviceSynchronize();
                    if (e != cudaSuccess) { ok = false; printf("  [distil] CUDA error: %s\n", cudaGetErrorString(e)); }
                } catch (const std::exception& ex) {
                    ok = false; printf("  [distil] threw: %s\n", ex.what());
                }
                auto t1 = std::chrono::steady_clock::now();

                running.store(false); poller.join();
                size_t mf = minFree.load();
                peakGB   = (tot - mf) / 1e9;
                distilGB = (baseFree > mf ? baseFree - mf : 0) / 1e9;
                distilU  = ok ? bnerf.numVoxels() : 0;           // capture BEFORE diagonstic overwrites it
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                printf("  [distil] ok=%d  %.0f ms  peak VRAM=%.2f GB   distil footprint=+%.2f GB   deduped voxels=%u\n",
                       (int)ok, ms, peakGB, distilGB, distilU);
            }

            // Independent oracle: diagonstic's bitset gives the true unique-vertex counts.
            // distil now PRUNES to sigma-surviving vertices -> compare to uniqueSurvivors.
            long long oracleSurv = -1, oracleAll = -1;
            try {
                bnerf.diagonstic(nerf);
                oracleSurv = bnerf.diagnostics().uniqueSurvivors;
                oracleAll  = bnerf.diagnostics().uniqueAll;
            } catch (const std::exception& ex) {
                printf("  [oracle] diagonstic threw: %s\n", ex.what());
            }

            bool match = doDistil && ok && ((long long)distilU == oracleSurv);
            if (doDistil)
                printf("  [verify] distil stored=%u   oracle sigma-surviving=%lld   MATCH=%s   (all-unique=%lld, pruned %.2fx)\n",
                       distilU, oracleSurv, match ? "YES" : "NO  <-- mismatch!",
                       oracleAll, (double)oracleAll / (double)(oracleSurv > 0 ? oracleSurv : 1));

            rows.push_back({ std::filesystem::path(modelPath).filename().string(),
                             s, bnerf.numBlocks(), distilU, oracleSurv, peakGB, distilGB, ok, match });
        }
    }

    printf("\n================ part-1 distil summary ================\n");
    printf("%-32s %7s %10s %13s %13s %8s %9s %6s %6s\n",
           "model", "sigma", "blocks", "distilUniq", "oracleUniq", "peakGB", "footprint", "run", "match");
    for (auto& r : rows)
        printf("%-32s %7.4f %10u %13u %13lld %8.2f %8.2fG %6s %6s\n",
               r.model.c_str(), r.sigma, r.blocks, r.distilU, r.oracleU,
               r.peakGB, r.distilGB, r.ok ? "OK" : "FAIL", r.match ? "YES" : "no");
    return 0;
}
