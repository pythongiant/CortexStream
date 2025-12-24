#include "cortexstream/model.h"
#include <algorithm>
#include <stdexcept>

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

}  // namespace cortexstream


