#include "cortexstream/sampler.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <iostream>
#include <limits>

namespace cortexstream {

// Validation helper
bool SamplingParams::validate() const {
    if (temperature < 0.0f) return false;
    if (topK < 1 && topK != 0) return false;
    if (topP < 0.0f || topP > 1.0f) return false;
    if (repetitionPenalty < 1.0f) return false;
    return true;
}

Sampler::Sampler() {
    initRNG();
}

Sampler::~Sampler() = default;

void Sampler::setParams(const SamplingParams& newParams) {
    if (!newParams.validate()) {
        throw std::invalid_argument("Invalid sampling parameters");
    }
    params = newParams;
    initRNG();
}

const SamplingParams& Sampler::getParams() const {
    return params;
}

void Sampler::setSeed(int seed) {
    params.seed = seed;
    initRNG();
}

std::optional<SamplingMetadata> Sampler::getLastMetadata() const {
    return lastMetadata;
}

int Sampler::sampleToken(const Tensor& logits,
                        const std::vector<int>& generatedHistory) {
    if (logits.shape.empty() || logits.data.empty()) {
        throw std::invalid_argument("Invalid logits tensor");
    }

    // Make working copy
    Tensor workingLogits = logits;

    // Step 1: Apply repetition penalty if enabled
    if (params.repetitionPenaltyEnabled && !generatedHistory.empty()) {
        workingLogits = applyRepetitionPenalty(workingLogits, generatedHistory);
    }

    // Step 2: Greedy override
    if (params.doSample || (params.topK == 1 && params.topP >= 1.0f)) {
        return greedySelect(workingLogits);
    }

    // Step 3: Apply temperature
    if (params.temperature != 1.0f) {
        workingLogits = applyTemperature(workingLogits);
    }

    // Step 4: Route to sampling strategy
    int selectedToken = -1;

    if (params.topK > 1 && params.topP < 1.0f) {
        // Top-K + Top-P combined
        selectedToken = topKPSample(workingLogits);
    } else if (params.topK > 1) {
        // Top-K only
        selectedToken = topKSample(workingLogits);
    } else if (params.topP < 1.0f) {
        // Top-P (Nucleus) only
        selectedToken = topPSample(workingLogits);
    } else {
        // Fallback to greedy
        selectedToken = greedySelect(workingLogits);
    }

    return selectedToken;
}

std::vector<int> Sampler::sampleBatch(
    const Tensor& batchedLogits,
    const std::vector<std::vector<int>>& histories) {
    
    // MVP: Simple sequential sampling per sequence
    // Future: Vectorized batch operations on GPU
    
    std::vector<int> tokens;
    
    int batchSize = batchedLogits.shape[0];
    int vocabSize = batchedLogits.shape[1];
    
    for (int i = 0; i < batchSize; ++i) {
        // Extract logits for this sequence
        Tensor seqLogits;
        seqLogits.shape = {1, static_cast<int64_t>(vocabSize)};
        seqLogits.data.resize(vocabSize);
        
        std::copy(
            batchedLogits.data.begin() + i * vocabSize,
            batchedLogits.data.begin() + (i + 1) * vocabSize,
            seqLogits.data.begin()
        );

        // Sample using history if provided
        auto history = !histories.empty() ? histories[i] : std::vector<int>{};
        int token = sampleToken(seqLogits, history);
        tokens.push_back(token);
    }

    return tokens;
}

void Sampler::initRNG() {
    if (params.seed >= 0) {
        rng.seed(params.seed);
    } else {
        std::random_device rd;
        rng.seed(rd());
    }
}

Tensor Sampler::applyTemperature(const Tensor& logits) {
    if (params.temperature <= 0.0f) {
        throw std::invalid_argument("Temperature must be positive");
    }

    Tensor result = logits;
    for (auto& val : result.data) {
        val = val / params.temperature;
    }
    return result;
}

Tensor Sampler::applyRepetitionPenalty(const Tensor& logits,
                                       const std::vector<int>& history) {
    if (params.repetitionPenalty <= 1.0f) {
        return logits;  // No penalty
    }

    Tensor result = logits;
    
    // Count token frequencies in history
    std::vector<int> frequencies(result.data.size(), 0);
    for (int token : history) {
        if (token >= 0 && token < static_cast<int>(result.data.size())) {
            frequencies[token]++;
        }
    }

    // Apply penalty: reduce logits of repeated tokens
    for (size_t i = 0; i < result.data.size(); ++i) {
        if (frequencies[i] > 0) {
            // Penalty reduces probability
            if (result.data[i] > 0) {
                result.data[i] = result.data[i] / params.repetitionPenalty;
            } else {
                result.data[i] = result.data[i] * params.repetitionPenalty;
            }
        }
    }

    return result;
}

Tensor Sampler::softmaxNormalize(const Tensor& logits) {
    Tensor result = logits;
    
    // Numerical stability: subtract max
    float maxLogit = *std::max_element(result.data.begin(), result.data.end());
    
    // Exp and sum
    float sum = 0.0f;
    for (auto& val : result.data) {
        val = std::exp(std::clamp(val - maxLogit, MIN_LOGIT, MAX_LOGIT));
        sum += val;
    }

    // Normalize
    if (sum > 0.0f) {
        for (auto& val : result.data) {
            val = val / sum;
        }
    }

    return result;
}

int Sampler::greedySelect(const Tensor& logits) {
    if (logits.data.empty()) {
        return 0;
    }

    int maxIdx = 0;
    float maxVal = logits.data[0];
    
    for (size_t i = 1; i < logits.data.size(); ++i) {
        if (logits.data[i] > maxVal) {
            maxVal = logits.data[i];
            maxIdx = i;
        }
    }

    return maxIdx;
}

int Sampler::topKSample(const Tensor& logits) {
    auto topKPairs = getTopK(logits.data, params.topK);
    
    if (topKPairs.empty()) {
        return 0;
    }

    // Convert to probabilities
    std::vector<float> probs;
    probs.reserve(topKPairs.size());
    
    float maxVal = topKPairs[0].first;
    float sumExp = 0.0f;
    
    for (const auto& [logit, idx] : topKPairs) {
        float p = std::exp(std::clamp(logit - maxVal, MIN_LOGIT, MAX_LOGIT));
        probs.push_back(p);
        sumExp += p;
    }

    // Normalize
    if (sumExp > 0.0f) {
        for (auto& p : probs) {
            p /= sumExp;
        }
    }

    // Sample
    int sampledIdx = categoricalSample(probs);
    return topKPairs[sampledIdx].second;
}

int Sampler::topPSample(const Tensor& logits) {
    // Convert to probabilities
    Tensor probs = softmaxNormalize(logits);
    
    auto topIndices = getNucleus(probs.data, params.topP);
    
    if (topIndices.empty()) {
        return 0;
    }

    // Renormalize nucleus probabilities
    std::vector<float> nucProbs;
    float sum = 0.0f;
    
    for (const auto& [prob, idx] : topIndices) {
        nucProbs.push_back(prob);
        sum += prob;
    }

    if (sum > 0.0f) {
        for (auto& p : nucProbs) {
            p /= sum;
        }
    }

    // Sample
    int sampledIdx = categoricalSample(nucProbs);
    return topIndices[sampledIdx].second;
}

int Sampler::topKPSample(const Tensor& logits) {
    // Apply both constraints: top-K AND top-P
    
    // Step 1: Get top-K
    auto topK = getTopK(logits.data, params.topK);
    
    if (topK.empty()) {
        return 0;
    }

    // Step 2: Convert to probabilities
    float maxVal = topK[0].first;
    std::vector<float> probs;
    float sumExp = 0.0f;
    
    for (const auto& [logit, idx] : topK) {
        float p = std::exp(std::clamp(logit - maxVal, MIN_LOGIT, MAX_LOGIT));
        probs.push_back(p);
        sumExp += p;
    }

    if (sumExp > 0.0f) {
        for (auto& p : probs) {
            p /= sumExp;
        }
    }

    // Step 3: Apply nucleus filter
    std::vector<float> filtered;
    float cumProb = 0.0f;
    
    for (float p : probs) {
        cumProb += p;
        if (cumProb <= params.topP) {
            filtered.push_back(p);
        }
    }

    if (filtered.empty()) {
        filtered = probs;  // Fallback
    }

    // Renormalize
    float newSum = std::accumulate(filtered.begin(), filtered.end(), 0.0f);
    if (newSum > 0.0f) {
        for (auto& p : filtered) {
            p /= newSum;
        }
    }

    // Sample
    int sampledIdx = categoricalSample(filtered);
    
    // Map back to original token indices
    int count = 0;
    for (size_t i = 0; i < probs.size(); ++i) {
        float cumProb2 = 0.0f;
        for (float p : filtered) {
            cumProb2 += p;
            if (cumProb2 >= params.topP) break;
        }
        if (count == sampledIdx) {
            return topK[i].second;
        }
        count++;
    }

    return topK.back().second;
}

std::vector<std::pair<float, int>> Sampler::getTopK(
    const std::vector<float>& logits, int k) {
    
    if (logits.empty()) {
        return {};
    }

    int actualK = std::min(k, static_cast<int>(logits.size()));
    
    // Create pairs
    std::vector<std::pair<float, int>> pairs;
    pairs.reserve(logits.size());
    
    for (size_t i = 0; i < logits.size(); ++i) {
        pairs.emplace_back(logits[i], i);
    }

    // Partial sort to get top-K
    std::nth_element(
        pairs.begin(),
        pairs.begin() + actualK,
        pairs.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; }
    );

    // Extract top-K
    std::vector<std::pair<float, int>> result(
        pairs.begin(),
        pairs.begin() + actualK
    );

    // Sort descending
    std::sort(
        result.begin(),
        result.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; }
    );

    return result;
}

std::vector<std::pair<float, int>> Sampler::getNucleus(
    const std::vector<float>& probs, float p) {
    
    if (probs.empty() || p >= 1.0f) {
        // Return all
        std::vector<std::pair<float, int>> result;
        for (size_t i = 0; i < probs.size(); ++i) {
            result.emplace_back(probs[i], i);
        }
        return result;
    }

    // Create pairs and sort
    std::vector<std::pair<float, int>> pairs;
    pairs.reserve(probs.size());
    
    for (size_t i = 0; i < probs.size(); ++i) {
        pairs.emplace_back(probs[i], i);
    }

    std::sort(
        pairs.begin(),
        pairs.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; }
    );

    // Find nucleus (cumulative probability >= p)
    std::vector<std::pair<float, int>> nucleus;
    float cumProb = 0.0f;
    
    for (const auto& [prob, idx] : pairs) {
        cumProb += prob;
        nucleus.emplace_back(prob, idx);
        
        if (cumProb >= p) {
            break;
        }
    }

    return nucleus;
}

int Sampler::categoricalSample(const std::vector<float>& probs) {
    if (probs.empty()) {
        return 0;
    }

    // Validate probabilities
    float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
    if (sum <= 0.0f || !std::isfinite(sum)) {
        // Fallback: return highest
        return std::max_element(probs.begin(), probs.end()) - probs.begin();
    }

    // Sample using inverse transform
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float rand = dist(rng);
    
    float cumProb = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        cumProb += probs[i];
        if (rand < cumProb) {
            return i;
        }
    }

    return probs.size() - 1;
}

float Sampler::computeEntropy(const std::vector<float>& probs) {
    float entropy = 0.0f;
    const float epsilon = 1e-10f;
    
    for (float p : probs) {
        if (p > epsilon) {
            entropy -= p * std::log(p);
        }
    }

    return entropy;
}

void Sampler::safeSoftmax(std::vector<float>& logits, float temperature) {
    if (logits.empty()) return;

    // Numerical stability: subtract max
    float maxLogit = *std::max_element(logits.begin(), logits.end());
    
    // Apply temperature
    float sum = 0.0f;
    for (auto& val : logits) {
        float scaled = (val - maxLogit) / temperature;
        val = std::exp(std::clamp(scaled, MIN_LOGIT, MAX_LOGIT));
        sum += val;
    }

    // Normalize
    if (sum > 0.0f) {
        for (auto& val : logits) {
            val = val / sum;
        }
    } else {
        // Fallback: uniform
        float uniform = 1.0f / logits.size();
        for (auto& val : logits) {
            val = uniform;
        }
    }
}

}  // namespace cortexstream


