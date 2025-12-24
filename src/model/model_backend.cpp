#include "cortexstream/model.h"
#include <algorithm>
#include <stdexcept>
#include <random>
#include <cmath>
#include <numeric>

namespace cortexstream {

ModelBackend::ModelBackend(Device device, DType dtype)
    : device(device), dtype(dtype) {}

ModelBackend::~ModelBackend() = default;

bool ModelBackend::loadModel(const std::string& path) {
    modelPath = path;
    // Placeholder metadata for demo builds
    hiddenSize = hiddenSize == 0 ? 4096 : hiddenSize;
    numLayers = numLayers == 0 ? 32 : numLayers;
    vocabSize = vocabSize == 0 ? 32000 : vocabSize;
    loaded = true;
    initializeBuffers();
    return true;
}

bool ModelBackend::isLoaded() const {
    return loaded;
}

Tensor ModelBackend::prefill(const Batch& batch, const std::vector<int>& /*tokenIds*/) {
    if (!loaded) throw std::runtime_error("Model not loaded");
    Tensor logits;
    logits.shape = {static_cast<int64_t>(batch.batchSize), static_cast<int64_t>(vocabSize)};
    logits.data.assign(static_cast<size_t>(batch.batchSize * vocabSize), 0.0f);
    logits.dtype = dtype;
    return logits;
}

Tensor ModelBackend::decode(const Batch& batch, const std::vector<int>& /*tokenIds*/) {
    // Reuse same dummy output shape as prefill
    return prefill(batch, {});
}

int ModelBackend::sampleToken(const Tensor& logits, const SamplingParams& /*params*/) {
    if (logits.data.empty()) return 0;
    auto it = std::max_element(logits.data.begin(), logits.data.end());
    return static_cast<int>(std::distance(logits.data.begin(), it));
}

size_t ModelBackend::getHiddenSize() const { return hiddenSize; }
size_t ModelBackend::getNumLayers() const { return numLayers; }
size_t ModelBackend::getVocabSize() const { return vocabSize; }
Device ModelBackend::getDevice() const { return device; }
DType ModelBackend::getDType() const { return dtype; }

void ModelBackend::warmup() {
    warmed = true;
}

bool ModelBackend::preloadGraph() {
    return true;
}

void ModelBackend::enableMetalOptimizations() { metal_optimized_ = true; }
void ModelBackend::disableMetalOptimizations() { metal_optimized_ = false; }
bool ModelBackend::isMetalOptimized() const { return metal_optimized_; }

void ModelBackend::initializeBuffers() {
    tempBuffer.shape = {static_cast<int64_t>(hiddenSize)};
    tempBuffer.data.resize(hiddenSize, 0.0f);
}

Tensor ModelBackend::forwardImpl(const Batch& batch,
                                 const std::vector<int>& tokenIds,
                                 bool isPrefill) {
    return isPrefill ? prefill(batch, tokenIds) : decode(batch, tokenIds);
}

Tensor ModelBackend::forwardMLX(const Batch& batch,
                                const std::vector<int>& tokenIds,
                                bool isPrefill) {
    // Stub mirrors CPU path for compatibility
    return forwardImpl(batch, tokenIds, isPrefill);
}

mlx::core::array ModelBackend::toMLXArray(const std::vector<float>& /*data*/,
                                          const std::vector<int64_t>& /*shape*/) {
    // Minimal placeholder array (single zero) for link compatibility
    return mlx::core::array({0.0f});
}

std::vector<float> ModelBackend::fromMLXArray(const mlx::core::array& /*arr*/) {
    return {};
}

int ModelBackend::sampleGreedy(const Tensor& logits) {
    if (logits.data.empty()) {
        return 0;
    }
    auto it = std::max_element(logits.data.begin(), logits.data.end());
    return static_cast<int>(std::distance(logits.data.begin(), it));
}

int ModelBackend::sampleTopK(const Tensor& logits, int k, float temperature) {
    if (logits.data.empty() || k <= 0) {
        return sampleGreedy(logits);
    }

    // Create indexed pairs of (logit, index)
    std::vector<std::pair<float, int>> pairs;
    pairs.reserve(logits.data.size());
    for (size_t i = 0; i < logits.data.size(); ++i) {
        pairs.emplace_back(logits.data[i], static_cast<int>(i));
    }

    // Partial sort to get top-K elements
    int actualK = std::min(k, static_cast<int>(pairs.size()));
    std::nth_element(
        pairs.begin(),
        pairs.begin() + actualK - 1,
        pairs.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; }
    );

    // Apply temperature and convert to probabilities
    std::vector<float> probs(actualK);
    float maxLogit = pairs[0].first;
    for (int i = 1; i < actualK; ++i) {
        maxLogit = std::max(maxLogit, pairs[i].first);
    }

    float sum = 0.0f;
    for (int i = 0; i < actualK; ++i) {
        float scaled = (pairs[i].first - maxLogit) / std::max(temperature, 1e-6f);
        probs[i] = std::exp(std::clamp(scaled, -88.0f, 88.0f));
        sum += probs[i];
    }

    // Normalize
    if (sum > 0.0f) {
        for (auto& p : probs) {
            p /= sum;
        }
    }

    // Sample from categorical distribution
    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float rand = dist(rng);

    float cumProb = 0.0f;
    for (int i = 0; i < actualK; ++i) {
        cumProb += probs[i];
        if (rand < cumProb) {
            return pairs[i].second;
        }
    }

    return pairs[actualK - 1].second;
}

int ModelBackend::sampleTopP(const Tensor& logits, float p, float temperature) {
    if (logits.data.empty() || p <= 0.0f) {
        return sampleGreedy(logits);
    }

    // Apply temperature and softmax
    std::vector<std::pair<float, int>> pairs;
    pairs.reserve(logits.data.size());

    float maxLogit = *std::max_element(logits.data.begin(), logits.data.end());

    for (size_t i = 0; i < logits.data.size(); ++i) {
        float scaled = (logits.data[i] - maxLogit) / std::max(temperature, 1e-6f);
        float prob = std::exp(std::clamp(scaled, -88.0f, 88.0f));
        pairs.emplace_back(prob, static_cast<int>(i));
    }

    // Sort by probability descending
    std::sort(pairs.begin(), pairs.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // Normalize probabilities
    float sum = 0.0f;
    for (const auto& pair : pairs) {
        sum += pair.first;
    }
    if (sum > 0.0f) {
        for (auto& pair : pairs) {
            pair.first /= sum;
        }
    }

    // Find nucleus (tokens with cumulative probability >= p)
    std::vector<std::pair<float, int>> nucleus;
    float cumProb = 0.0f;
    for (const auto& pair : pairs) {
        cumProb += pair.first;
        nucleus.push_back(pair);
        if (cumProb >= p) {
            break;
        }
    }

    // Renormalize nucleus probabilities
    float nucleusSum = 0.0f;
    for (const auto& pair : nucleus) {
        nucleusSum += pair.first;
    }
    if (nucleusSum > 0.0f) {
        for (auto& pair : nucleus) {
            pair.first /= nucleusSum;
        }
    }

    // Sample from nucleus
    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float rand = dist(rng);

    cumProb = 0.0f;
    for (const auto& pair : nucleus) {
        cumProb += pair.first;
        if (rand < cumProb) {
            return pair.second;
        }
    }

    return nucleus.back().second;
}

}  // namespace cortexstream


