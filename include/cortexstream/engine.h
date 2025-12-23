#ifndef CORTEXSTREAM_ENGINE_H
#define CORTEXSTREAM_ENGINE_H

#include "model.h"
#include "scheduler.h"
#include "kv_cache.h"
#include "request.h"
#include <memory>
#include <atomic>
#include <chrono>

namespace cortexstream {

// Forward declaration
class StreamManager;

struct EngineStats {
    size_t tokensProcessed = 0;
    size_t requestsCompleted = 0;
    size_t requestsFailed = 0;
    float avgBatchSize = 0.0f;
    std::chrono::milliseconds totalLatency{0};
};

class InferenceEngine {
public:
    InferenceEngine(std::shared_ptr<ModelBackend> backend,
                    std::shared_ptr<Scheduler> scheduler,
                    std::shared_ptr<KVCache> cache);
    ~InferenceEngine();

    // Lifecycle
    bool initialize();
    void run();
    void shutdown();
    
    // Control
    bool isRunning() const;
    void pause();
    void resume();
    
    // Statistics
    const EngineStats& getStats() const;
    int getActiveRequests() const;

private:
    std::shared_ptr<ModelBackend> backend;
    std::shared_ptr<Scheduler> scheduler;
    std::shared_ptr<KVCache> cache;
    
    std::atomic<bool> running{false};
    std::atomic<bool> paused{false};
    
    EngineStats stats;
    
    // Main loop
    void mainLoop();
    
    // Processing stages
    void processPrefill(const Batch& prefillBatch);
    void processDecode(const Batch& decodeBatch);
    
    // Token emission and streaming
    void emitTokens(const Batch& batch, const Tensor& logits);
    int sampleAndApply(const Tensor& logits, 
                      std::shared_ptr<Request> request);
    
    // Cleanup and resource management
    void cleanup();
    void cleanupRequest(const std::string& requestId);
    void validateMemoryState();
    
    // Failure handling
    void handleBackendFailure(const std::string& reason);
    void handleOOM();
    void handleStuckRequest(const std::string& requestId);
};

}  // namespace cortexstream

#endif  // CORTEXSTREAM_ENGINE_H
