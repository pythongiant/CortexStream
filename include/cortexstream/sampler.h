#ifndef CORTEXSTREAM_SAMPLER_H
#define CORTEXSTREAM_SAMPLER_H

#include "model.h"
#include <vector>
#include <cstdint>

namespace cortexstream {

// Sampler abstraction layer
// Can be used in ModelBackend or InferenceEngine

class Sampler {
public:
    explicit Sampler(uint32_t seed = 0);
    ~Sampler();

    // Sampling strategies
    int greedy(const Tensor& logits);
    
    int topK(const Tensor& logits, int k, float temperature = 1.0f);
    
    int topP(const Tensor& logits, float p, float temperature = 1.0f);
    
    int topKP(const Tensor& logits, int k, float p, float temperature = 1.0f);
    
    // Utility
    void setSeed(uint32_t seed);

private:
    uint32_t seed;
    
    // Helper functions
    std::vector<std::pair<float, int>> topKIndices(const Tensor& logits, int k);
    std::vector<float> softmax(const std::vector<float>& logits, float temperature);
    int sampleFromDistribution(const std::vector<float>& probs);
};

}  // namespace cortexstream

#endif  // CORTEXSTREAM_SAMPLER_H
