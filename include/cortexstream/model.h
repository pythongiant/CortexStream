#ifndef CORTEXSTREAM_MODEL_H
#define CORTEXSTREAM_MODEL_H

#include "request.h"
#include "scheduler.h"
#include <vector>
#include <string>
#include <memory>
#include <cstdint>

namespace cortexstream {

enum class Device {
    MPS,    // Metal Performance Shaders (Apple Silicon)
    CPU
};

enum class DType {
    FP32,
    FP16,
    INT8
};

// Minimal tensor abstraction (in practice, would use MLX tensor)
struct Tensor {
    std::vector<float> data;
    std::vector<int64_t> shape;
    DType dtype = DType::FP16;
    
    int64_t numElements() const {
        int64_t n = 1;
        for (auto s : shape) n *= s;
        return n;
    }
};

class ModelBackend {
public:
    explicit ModelBackend(Device device = Device::MPS, DType dtype = DType::FP16);
    ~ModelBackend();

    // Model lifecycle
    bool loadModel(const std::string& modelPath);
    bool isLoaded() const;
    
    // Forward passes
    Tensor prefill(const Batch& batch, const std::vector<int>& tokenIds);
    Tensor decode(const Batch& batch, const std::vector<int>& tokenIds);
    
    // Sampling
    int sampleToken(const Tensor& logits, const SamplingParams& params);
    
    // Model metadata
    size_t getHiddenSize() const;
    size_t getNumLayers() const;
    size_t getVocabSize() const;
    Device getDevice() const;
    DType getDType() const;
    
    // Performance optimization
    void warmup();
    bool preloadGraph();

private:
    Device device;
    DType dtype;
    
    // MLX model (in real implementation, would be mlx::core::array)
    bool loaded = false;
    std::string modelPath;
    
    // Model architecture info
    size_t hiddenSize = 0;
    size_t numLayers = 0;
    size_t vocabSize = 0;
    
    // Performance buffers
    Tensor tempBuffer;
    bool warmed = false;
    
    // Internal methods
    void initializeBuffers();
    Tensor forwardImpl(const Batch& batch, 
                      const std::vector<int>& tokenIds, 
                      bool isPrefill);
    int sampleGreedy(const Tensor& logits);
    int sampleTopK(const Tensor& logits, int k, float temperature);
    int sampleTopP(const Tensor& logits, float p, float temperature);
};

}  // namespace cortexstream

#endif  // CORTEXSTREAM_MODEL_H
