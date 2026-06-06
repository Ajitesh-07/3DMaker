#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include <limits>
#include <algorithm>

struct MetricTracker {
    double totalMs = 0.0;
    uint64_t count = 0;
    float minMs = std::numeric_limits<float>::max();
    float maxMs = -std::numeric_limits<float>::max();
    double mean = 0.0;
    double m2 = 0.0;

    void update(float ms) {
        totalMs += ms;
        count++;
        if (ms < minMs) minMs = ms;
        if (ms > maxMs) maxMs = ms;
        
        double delta = ms - mean;
        mean += delta / count;
        double delta2 = ms - mean;
        m2 += delta * delta2;
    }

    float getAverage() const {
        return count == 0 ? 0.0f : static_cast<float>(totalMs / count);
    }

    float getMin() const {
        return count == 0 ? 0.0f : minMs;
    }

    float getMax() const {
        return count == 0 ? 0.0f : maxMs;
    }

    float getStdDev() const {
        return count == 0 ? 0.0f : static_cast<float>(std::sqrt(m2 / count));
    }

    void reset() {
        totalMs = 0.0;
        count = 0;
        minMs = std::numeric_limits<float>::max();
        maxMs = -std::numeric_limits<float>::max();
        mean = 0.0;
        m2 = 0.0;
    }
};

struct PendingTimer {
    cudaEvent_t start;
    cudaEvent_t stop;
    MetricTracker* tracker;
};

struct MetricGroup {
    std::vector<const MetricTracker*> trackers;

    void add(const MetricTracker& tracker) {
        trackers.push_back(&tracker);
    }

    void add(const MetricGroup& otherGroup) {
        for (const auto* t : otherGroup.trackers) {
            trackers.push_back(t);
        }
    }

    float getAverage() const {
        int minCount = INT_MAX;
        for (const auto* t : trackers) {
            if (t->count > 0 && minCount > t->count) minCount = t->count;
        }

        return getTotalMs() / minCount;
    }

    double getTotalMs() const {
        double total = 0.0;
        for (const auto* t : trackers) {
            total += t->totalMs;
        }
        return total;
    }

    int getCount() const {
        int minCount = INT_MAX;
        for (const auto* t : trackers) {
            if (t->count > 0 && minCount > t->count) minCount = t->count;
        }

        return minCount;
    }
};
