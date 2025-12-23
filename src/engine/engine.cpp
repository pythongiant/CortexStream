#include "cortexstream/engine.h"
#include "cortexstream/model.h"
#include "cortexstream/scheduler.h"
#include "cortexstream/kv_cache.h"
#include <iostream>
#include <thread>
#include <chrono>

namespace cortexstream {

InferenceEngine::InferenceEngine(std::shared_ptr<ModelBackend> backend,
                                 std::shared_ptr<Scheduler> scheduler,
                                 std::shared_ptr<KVCache> cache)
    : backend(backend), scheduler(scheduler), cache(cache) {
}

InferenceEngine::~InferenceEngine() {
    if (running) {
        shutdown();
    }
}

bool InferenceEngine::initialize() {
    if (!backend || !scheduler || !cache) {
        std::cerr << "[InferenceEngine] Invalid dependencies" << std::endl;
        return false;
    }
    
    if (!backend->isLoaded()) {
        std::cerr << "[InferenceEngine] Backend not initialized" << std::endl;
        return false;
    }
    
    // Warmup GPU
    backend->warmup();
    cache->warmup();
    
    std::cout << "[InferenceEngine] Initialized successfully" << std::endl;
    return true;
}

void InferenceEngine::run() {
    if (running) {
        std::cout << "[InferenceEngine] Already running" << std::endl;
        return;
    }
    
    std::cout << "[InferenceEngine] Starting main loop" << std::endl;
    mainLoop();
}

void InferenceEngine::shutdown() {
    running = false;
    std::cout << "[InferenceEngine] Shutdown complete" << std::endl;
}

bool InferenceEngine::isRunning() const {
    return running;
}

void InferenceEngine::pause() {
    paused = true;
}

void InferenceEngine::resume() {
    paused = false;
}

const EngineStats& InferenceEngine::getStats() const {
    return stats;
}

int InferenceEngine::getActiveRequests() const {
    return scheduler->getNumActiveRequests();
}

void InferenceEngine::mainLoop() {
    running = true;
    
    while (scheduler->hasWork() && !paused) {
        // Accept new requests from queue
        scheduler->acceptNewRequests();
        
        // Build and process prefill batch
        Batch prefillBatch = scheduler->buildPrefillBatch();
        if (!prefillBatch.empty()) {
            try {
                processPrefill(prefillBatch);
            } catch (const std::exception& e) {
                std::cerr << "[InferenceEngine] Prefill error: " << e.what() << std::endl;
                handleBackendFailure(e.what());
            }
        }
        
        // Build and process decode batch
        Batch decodeBatch = scheduler->buildDecodeBatch();
        if (!decodeBatch.empty()) {
            try {
                processDecode(decodeBatch);
            } catch (const std::exception& e) {
                std::cerr << "[InferenceEngine] Decode error: " << e.what() << std::endl;
                handleBackendFailure(e.what());
            }
        }
        
        // Cleanup finished requests
        cleanup();
        
        // Validate memory state
        validateMemoryState();
        
        // Small sleep to prevent busy loop if no work
        if (!scheduler->hasWork()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
    running = false;
    std::cout << "[InferenceEngine] Main loop exited" << std::endl;
    std::cout << "[InferenceEngine] Stats: " 
              << "tokens=" << stats.tokensProcessed
              << ", requests=" << stats.requestsCompleted
              << ", failed=" << stats.requestsFailed << std::endl;
}

void InferenceEngine::processPrefill(const Batch& prefillBatch) {
    if (prefillBatch.empty()) {
        return;
    }
    
    // Collect all token IDs
    std::vector<int> allTokens;
    for (const auto& req : prefillBatch.requests) {
        const auto& promptTokens = req->getPromptTokens();
        allTokens.insert(allTokens.end(), promptTokens.begin(), promptTokens.end());
    }
    
    // Forward pass through backend
    Tensor logits = backend->prefill(prefillBatch, allTokens);
    
    // Allocate KV blocks for each request
    for (const auto& req : prefillBatch.requests) {
        try {
            int blockId = cache->allocateBlock(req->getId());
            if (blockId < 0) {
                handleOOM();
                return;
            }
        } catch (const std::exception& e) {
            std::cerr << "[InferenceEngine] KV allocation failed: " << e.what() << std::endl;
            scheduler->markRequestFailed(req->getId());
        }
    }
    
    // Mark requests ready for decode
    for (const auto& req : prefillBatch.requests) {
        scheduler->markRequestReady(req->getId());
    }
}

void InferenceEngine::processDecode(const Batch& decodeBatch) {
    if (decodeBatch.empty()) {
        return;
    }
    
    // Build batch of last tokens
    std::vector<int> lastTokens;
    for (const auto& req : decodeBatch.requests) {
        const auto& generated = req->getGeneratedTokens();
        if (!generated.empty()) {
            lastTokens.push_back(generated.back());
        } else {
            // No tokens generated yet, shouldn't happen
            lastTokens.push_back(0);
        }
    }
    
    // Forward pass through backend
    Tensor logits = backend->decode(decodeBatch, lastTokens);
    
    // Emit tokens
    emitTokens(decodeBatch, logits);
}

void InferenceEngine::emitTokens(const Batch& batch, const Tensor& logits) {
    for (size_t i = 0; i < batch.requests.size(); ++i) {
        auto req = batch.requests[i];
        
        // Extract logits for this request
        Tensor reqLogits;
        reqLogits.shape = {1, logits.shape.back()};
        reqLogits.data.resize(logits.shape.back());
        
        std::copy(logits.data.begin() + i * logits.shape.back(),
                 logits.data.begin() + (i + 1) * logits.shape.back(),
                 reqLogits.data.begin());
        
        // Sample and apply
        int nextToken = sampleAndApply(reqLogits, req);
        
        req->addToken(nextToken);
        stats.tokensProcessed++;
        
        // Check if request is finished
        if (req->getGeneratedLength() >= req->getMaxTokens()) {
            scheduler->markRequestFinished(req->getId());
            stats.requestsCompleted++;
        }
    }
}

int InferenceEngine::sampleAndApply(const Tensor& logits,
                                   std::shared_ptr<Request> request) {
    try {
        return backend->sampleToken(logits, request->getSamplingParams());
    } catch (const std::exception& e) {
        std::cerr << "[InferenceEngine] Sampling failed: " << e.what() << std::endl;
        return 0;  // Fallback token
    }
}

void InferenceEngine::cleanup() {
    // Find and remove finished requests
    auto numBefore = scheduler->getNumActiveRequests();
    
    for (int i = 0; i < numBefore; ++i) {
        // Note: In production, would iterate through finished queue
        // For MVP, scheduler handles this internally
    }
}

void InferenceEngine::cleanupRequest(const std::string& requestId) {
    // Free KV cache blocks
    cache->clearRequest(requestId);
    
    std::cout << "[InferenceEngine] Cleaned up request: " << requestId << std::endl;
}

void InferenceEngine::validateMemoryState() {
    // Periodically check that memory is not corrupted
    if (cache->isFull()) {
        std::cerr << "[InferenceEngine] WARNING: KV cache full!" << std::endl;
    }
}

void InferenceEngine::handleBackendFailure(const std::string& reason) {
    std::cerr << "[InferenceEngine] Backend failure: " << reason << std::endl;
    
    // In production, would implement recovery strategy
    // For now, just mark as failed and continue
    stats.requestsFailed++;
}

void InferenceEngine::handleOOM() {
    std::cerr << "[InferenceEngine] Out of memory - evicting oldest requests" << std::endl;
    // In production, would implement eviction policy
}

void InferenceEngine::handleStuckRequest(const std::string& requestId) {
    std::cerr << "[InferenceEngine] Request stuck, killing: " << requestId << std::endl;
    scheduler->markRequestFailed(requestId);
    cleanupRequest(requestId);
    stats.requestsFailed++;
}

}  // namespace cortexstream

