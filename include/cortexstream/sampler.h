#ifndef CORTEXSTREAM_SAMPLER_H
#define CORTEXSTREAM_SAMPLER_H

#include "model.h"
#include <vector>
#include <cstdint>
#include <random>
#include <optional>

namespace cortexstream {

// Sampling metadata (optional diagnostics)
struct SamplingMetadata {
    float chosenProb;           // Probability of selected token
    float entropy;              // Shannon entropy of distribution
    std::vector<int> topTokens; // Top-K candidates
    std::vector<float> topProbs; // Their probabilities
    int numFiltered;            // Tokens filtered out
};

// Complete sampling configuration (mirrors TensorRT-LLM)
struct SamplingParams {
    // Core strategies
    float temperature = 1.0f;
    int topK = 1;              // 1 = greedy (argmax)
    float topP = 1.0f;         // 1.0 = disabled
    bool doSample = false;     // Override to greedy regardless

    // Optional penalties
    bool repetitionPenaltyEnabled = false;
    float repetitionPenalty = 1.1f;

    // Determinism
    int seed = -1;             // -1 = random seed from device

    // Diagnostics
    bool returnLogprobs = false;
    bool returnMetadata = false;

    // Validation
    bool validate() const;
};

// Sampler: converts logits → token indices
class Sampler {
public:
    Sampler();
    ~Sampler();

    // Configuration
    void setParams(const SamplingParams& params);
    const SamplingParams& getParams() const;
    void setSeed(int seed);

    // Core API: sample single token
    int sampleToken(const Tensor& logits,
                    const std::vector<int>& generatedHistory = {});

    // Optional: get metadata for debugging
    std::optional<SamplingMetadata> getLastMetadata() const;

    // Batch API (future upgrade)
    std::vector<int> sampleBatch(
        const Tensor& batchedLogits,
        const std::vector<std::vector<int>>& histories = {});

private:
    SamplingParams params;
    std::mt19937 rng;
    std::optional<SamplingMetadata> lastMetadata;

    // RNG initialization with seed support
    void initRNG();

    // Device-aware tensor operations (MLX compatible)
    // These work with both CPU and GPU tensors via MLX
    Tensor applyTemperature(const Tensor& logits);
    Tensor applyRepetitionPenalty(const Tensor& logits,
                                  const std::vector<int>& history);
    Tensor softmaxNormalize(const Tensor& logits);

    // Core sampling strategies
    int greedySelect(const Tensor& logits);
    int topKSample(const Tensor& logits);
    int topPSample(const Tensor& logits);
    int topKPSample(const Tensor& logits);

    // Utilities
    std::vector<std::pair<float, int>> getTopK(
        const std::vector<float>& logits, int k);
    
    std::vector<std::pair<float, int>> getNucleus(
        const std::vector<float>& probs, float p);
    
    int categoricalSample(const std::vector<float>& probs);
    
    float computeEntropy(const std::vector<float>& probs);

    // Numerical safety
    static constexpr float MIN_LOGIT = -1e9f;
    static constexpr float MAX_LOGIT = 1e9f;
    
    void safeSoftmax(std::vector<float>& logits, float temperature);
};

}  // namespace cortexstream

#endif  // CORTEXSTREAM_SAMPLER_H
