#ifndef CORTEXSTREAM_ENGINE_H
#define CORTEXSTREAM_ENGINE_H

#include "model.h"
#include "scheduler.h"
#include "kv_cache.h"
#include "request.h"
#include <memory>
#include <atomic>
#include <chrono>

// ============================================================================
// OPTIMIZATION GUIDE - InferenceEngine for Apple Silicon
// ============================================================================
//
// Parallel Processing Optimizations:
// 1. emitTokens(): Parallel token extraction & sampling per-request (OpenMP)
//    - Each request sampled in separate thread (no inter-request dependencies)
//    - SIMD-aligned memory copies within parallel context
//    - Critical sections for atomic updates only
//
// 2. processDecode(): Parallel last-token extraction (OpenMP dynamic schedule)
//    - Extracts final tokens from all decode-stage requests in parallel
//    - Reduces latency for large batches
//
// 3. processPrefill(): Parallel prompt token collection
//    - Extracts prompt tokens with ordered synchronization
//    - Maintains batch contiguity while using parallelism
//
// Metal/GPU Acceleration:
// 1. prefill() and decode() calls use MLX backend (automatically Metal on Apple Silicon)
// 2. Logits returned as GPU tensors when possible (reduced GPU->CPU transfers)
// 3. Warmup() compiles Metal computation graphs for lower latency
//
// Memory Management:
// 1. KV cache uses buddy allocator: O(log n) allocation vs O(n) linear scan
// 2. Automatic coalescing reduces fragmentation on long-running workloads
// 3. Block-based pooling avoids malloc/free overhead
//
// Scheduling Optimizations:
// 1. buildPrefillBatch(): Sorts by prompt length (shorter first)
//    - Shorter sequences complete faster, freeing GPU memory
// 2. buildDecodeBatch(): Sorts by generation progress (newer first)
//    - Minimizes time-to-first-token (TTFT) for recent requests
// 3. Both use dynamic priority scheduling for better throughput
//
// ============================================================================

namespace cortexstream {

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
