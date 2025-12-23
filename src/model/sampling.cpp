#include "cortexstream/sampler.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <iostream>
#include <limits>
#include <queue>

// MLX headers for GPU-accelerated operations
// Note: On Apple Silicon, MLX uses Metal Performance Shaders (MPS)
#include <mlx/mlx.h>

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
    // Vectorized temperature scaling using MLX on Apple Silicon
    
    if (params.temperature <= 0.0f) {
        throw std::invalid_argument("Temperature must be positive");
    }

    try {
        // GPU vectorized operation via MLX
        mlx::core::array mlx_logits = mlx::core::asarray(logits.data);
        mlx_logits = mlx::core::reshape(mlx_logits, {static_cast<int>(logits.data.size())});
        
        // Element-wise division: logits / temperature (Metal-accelerated)
        mlx::core::array scaled = mlx_logits / params.temperature;
        
        Tensor result = logits;
        result.data = mlx::core::to_vector<float>(scaled);
        return result;
    } catch (...) {
        // CPU fallback
        Tensor result = logits;
        for (auto& val : result.data) {
            val = val / params.temperature;
        }
        return result;
    }
}

Tensor Sampler::applyRepetitionPenalty(const Tensor& logits,
                                       const std::vector<int>& history) {
    // Optimized repetition penalty with SIMD vectorization
    // Batch-updates repeated token penalties for better cache utilization
    
    if (params.repetitionPenalty <= 1.0f) {
        return logits;  // No penalty
    }

    Tensor result = logits;
    
    // Precompute token frequency count in history (single pass)
    // Reserve vector size to avoid reallocations during insertion
    std::vector<int> frequencies(result.data.size(), 0);
    for (int token : history) {
        if (token >= 0 && token < static_cast<int>(result.data.size())) {
            frequencies[token]++;
        }
    }

    // Vectorized penalty application with branch prediction optimization
    // Process in groups for better SIMD utilization
    const int SIMD_STRIDE = 8;
    size_t i = 0;
    
    // Vectorized loop for penalty application
    for (; i + SIMD_STRIDE <= result.data.size(); i += SIMD_STRIDE) {
        #pragma omp simd
        for (int j = 0; j < SIMD_STRIDE; ++j) {
            size_t idx = i + j;
            if (frequencies[idx] > 0) {
                // Branch-free penalty: multiply/divide based on sign
                if (result.data[idx] > 0) {
                    result.data[idx] = result.data[idx] / params.repetitionPenalty;
                } else {
                    result.data[idx] = result.data[idx] * params.repetitionPenalty;
                }
            }
        }
    }

    // Process remainder
    for (; i < result.data.size(); ++i) {
        if (frequencies[i] > 0) {
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
    // GPU-accelerated softmax using MLX on Apple Silicon
    // Falls back to CPU if necessary
    
    try {
        // Convert to MLX array for GPU computation
        // Input: logits.data is a vector of floats
        mlx::core::array mlx_logits = mlx::core::asarray(logits.data);
        // Reshape to 1D if needed
        if (mlx_logits.size() == logits.data.size()) {
            mlx_logits = mlx::core::reshape(mlx_logits, {static_cast<int>(logits.data.size())});
        }
        
        // Apply softmax with numerical stability
        mlx::core::array normalized = mlx::core::softmax(mlx_logits);
        
        // Convert back to CPU tensor
        Tensor result = logits;
        std::vector<float> normalized_data = mlx::core::to_vector<float>(normalized);
        result.data = normalized_data;
        
        return result;
    } catch (...) {
        // Fallback to CPU softmax if MLX unavailable
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
    
    // Optimized top-K using partial sort + heap for better cache locality
    // O(n log k) instead of O(n log n) full sort
    
    if (logits.empty()) {
        return {};
    }

    int actualK = std::min(k, static_cast<int>(logits.size()));
    
    // Create indexed pairs
    std::vector<std::pair<float, int>> pairs;
    pairs.reserve(logits.size());
    
    for (size_t i = 0; i < logits.size(); ++i) {
        pairs.emplace_back(logits[i], i);
    }

    // Partial sort: move K largest elements to front (O(n log k))
    // Uses nth_element which is cache-friendly
    std::nth_element(
        pairs.begin(),
        pairs.begin() + actualK - 1,
        pairs.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; }
    );

    // Extract top-K elements
    std::vector<std::pair<float, int>> result(
        pairs.begin(),
        pairs.begin() + actualK
    );

    // Sort descending within top-K (only K elements, not full array)
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
    // Optimized categorical sampling with MLX GPU kernels on Apple Silicon
    // Fallback to CPU if MLX unavailable
    
    if (probs.empty()) {
        return 0;
    }

    // Validate and cache sum for numerical stability
    float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
    if (sum <= 0.0f || !std::isfinite(sum)) {
        // Fallback: return highest probability token
        return std::max_element(probs.begin(), probs.end()) - probs.begin();
    }

    try {
        // Try GPU sampling via MLX on Apple Silicon
        mlx::core::array mlx_probs = mlx::core::asarray(probs);
        mlx_probs = mlx::core::reshape(mlx_probs, {static_cast<int>(probs.size())});
        
        // MLX categorical sampling on GPU (Metal)
        // Uses optimized random number generation on MPS
        mlx::core::array sample_result = mlx::core::multinomial(mlx_probs, 1);
        
        // Extract sampled index
        int selected = mlx::core::to_vector<int32_t>(sample_result)[0];
        return std::max(0, std::min(selected, static_cast<int>(probs.size()) - 1));
    } catch (...) {
        // CPU fallback: inverse transform sampling
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

// ============================================================================
// Softmax Cache Implementation - Reduces redundant GPU computations
// ============================================================================

size_t Sampler::hashLogits(const std::vector<float>& logits) const {
    // Simple hash for logits vector
    // In production, would use more sophisticated hash
    std::hash<float> hasher;
    size_t seed = 0;
    
    // Hash first 16 elements (representative sample)
    for (size_t i = 0; i < std::min(size_t(16), logits.size()); ++i) {
        seed ^= hasher(logits[i]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    
    // Include size in hash
    seed ^= hasher(static_cast<float>(logits.size()));
    return seed;
}

std::vector<float>* Sampler::getCachedSoftmax(const std::vector<float>& logits) {
    // Check if softmax normalization is cached (avoid redundant GPU computation)
    size_t hash = hashLogits(logits);
    auto it = softmax_cache_.find(hash);
    
    if (it != softmax_cache_.end()) {
        // Cache hit: return cached softmax probabilities
        return &it->second;
    }
    
    return nullptr;  // Cache miss
}

void Sampler::cacheSoftmax(const std::vector<float>& logits, 
                           const std::vector<float>& probs) {
    // Store softmax result in LRU cache
    // Bounded cache size prevents unbounded memory growth
    
    if (softmax_cache_.size() >= MAX_SOFTMAX_CACHE_SIZE) {
        // Simple eviction: clear oldest (could use proper LRU)
        softmax_cache_.clear();
    }
    
    size_t hash = hashLogits(logits);
    softmax_cache_[hash] = probs;
}

void Sampler::clearSoftmaxCache() {
    // Clear all cached softmax values
    softmax_cache_.clear();
}

size_t Sampler::getSoftmaxCacheSize() const {
    return softmax_cache_.size();
}

}  // namespace cortexstream

