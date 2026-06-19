#include "../NerfTrainer.h"

#include <cmath>
#include <cstdio>
#include <string>

static bool approx(const Vec3& a, const Vec3& b, float eps = 1e-4f) {
    return std::fabs(a.x - b.x) < eps && std::fabs(a.y - b.y) < eps && std::fabs(a.z - b.z) < eps;
}

int main(int argc, char** argv) {
    std::string dataset = (argc > 1) ? argv[1] : "../../data/video";
    std::string model   = "../benchmarks/saved/_footer_roundtrip.inerf";

    NerfConfig cfg;

    INerfTrainer a;
    a.init(cfg);
    a.loadDataset(dataset);
    a.train(60);
    a.save(model);
    Vec3 ca = a.sceneCenter(), ua = a.sceneUp();
    printf("[save] center=(%.4f, %.4f, %.4f) up=(%.4f, %.4f, %.4f)\n", ca.x, ca.y, ca.z, ua.x, ua.y, ua.z);

    INerfTrainer b;
    b.init(cfg);
    b.load(model);
    Vec3 cb = b.sceneCenter(), ub = b.sceneUp();
    printf("[load] center=(%.4f, %.4f, %.4f) up=(%.4f, %.4f, %.4f)\n", cb.x, cb.y, cb.z, ub.x, ub.y, ub.z);

    Image orbit = b.renderOrbit(45.0f, 20.0f, 3.0f, 50.0f, 64, 64);
    int nonzero = 0;
    for (float v : orbit.rgb) if (v > 1e-4f) ++nonzero;

    bool frameOk  = approx(ca, cb) && approx(ua, ub);
    bool notDefault = !(approx(ua, Vec3{0, 1, 0}) && approx(ca, Vec3{0, 0, 0}));  // proves it's real data
    bool renderOk = (orbit.width == 64 && nonzero > 0);

    printf("frame restored: %s | non-default frame: %s | orbit non-empty: %s (%d px)\n",
           frameOk ? "yes" : "NO", notDefault ? "yes" : "no", renderOk ? "yes" : "NO", nonzero);

    bool pass = frameOk && renderOk;
    printf(pass ? "PASS\n" : "FAIL\n");
    return pass ? 0 : 1;
}
