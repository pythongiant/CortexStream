// Simple inference example demonstrating CortexStream architecture
#include "cortexstream/engine.h"
#include "cortexstream/model.h"
#include "cortexstream/scheduler.h"
#include "cortexstream/kv_cache.h"
#include "cortexstream/request.h"
#include <iostream>
#include <vector>
#include <memory>
#include <thread>

using namespace cortexstream;

int main() {
    std::cout << "=== CortexStream Simple Inference Example ===" << std::endl;
    
    // 1. Initialize components
    std::cout << "\n[Setup] Initializing components..." << std::endl;
    
    // Create ModelBackend (MLX-backed)
    auto backend = std::make_shared<ModelBackend>(Device::MPS, DType::FP16);
    if (!backend->loadModel("path/to/model.mlx")) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }
    
    // Create scheduler
    auto scheduler = std::make_shared<Scheduler>(32);  // max batch size
    
    // Create KV cache
    // Size: 8GB, hidden: 4096, layers: 32
    auto cache = std::make_shared<KVCache>(
        8UL * 1024 * 1024 * 1024,  // 8GB cache
        backend->getHiddenSize(),
        backend->getNumLayers()
    );
    
    // Create inference engine
    auto engine = std::make_shared<InferenceEngine>(backend, scheduler, cache);
    if (!engine->initialize()) {
        std::cerr << "Failed to initialize engine" << std::endl;
        return 1;
    }
    
    // 2. Submit some requests
    std::cout << "\n[Requests] Submitting inference requests..." << std::endl;
    
    std::vector<std::shared_ptr<Request>> requests;
    
    // Request 1: "What is the capital of France?"
    {
        std::vector<int> promptTokens = {101, 2054, 2003, 1996, 3007, 1997, 2605};
        auto req = std::make_shared<Request>("req_001", promptTokens, 128);
        
        SamplingParams params;
        params.temperature = 0.7f;
        params.topK = 40;
        params.topP = 0.9f;
        req->setSamplingParams(params);
        
        scheduler->submitRequest(req);
        requests.push_back(req);
        std::cout << "  [req_001] Submitted with " << promptTokens.size() << " prompt tokens" << std::endl;
    }
    
    // Request 2: "Explain machine learning in simple terms"
    {
        std::vector<int> promptTokens = {102, 3407, 3231, 2628, 3567, 2031};
        auto req = std::make_shared<Request>("req_002", promptTokens, 256);
        
        SamplingParams params;
        params.temperature = 0.9f;
        params.topP = 0.95f;
        req->setSamplingParams(params);
        
        scheduler->submitRequest(req);
        requests.push_back(req);
        std::cout << "  [req_002] Submitted with " << promptTokens.size() << " prompt tokens" << std::endl;
    }
    
    // 3. Run inference engine
    std::cout << "\n[Inference] Starting inference engine..." << std::endl;
    
    // Run engine in a separate thread (for demo purposes)
    std::thread engineThread([&engine]() {
        engine->run();
    });
    
    // Monitor progress
    std::cout << "\n[Monitor] Waiting for completions..." << std::endl;
    
    bool allFinished = false;
    int checkCount = 0;
    while (!allFinished && checkCount < 100) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        allFinished = true;
        for (const auto& req : requests) {
            if (!req->isFinished()) {
                allFinished = false;
                break;
            }
        }
        
        if (checkCount % 10 == 0) {
            std::cout << "  Active requests: " << engine->getActiveRequests() << std::endl;
        }
        checkCount++;
    }
    
    // 4. Results
    std::cout << "\n[Results] Inference Complete" << std::endl;
    for (const auto& req : requests) {
        std::cout << "\n  Request: " << req->getId() << std::endl;
        std::cout << "  Prompt tokens: " << req->getPromptLength() << std::endl;
        std::cout << "  Generated tokens: " << req->getGeneratedLength() << std::endl;
        std::cout << "  State: ";
        
        switch (req->getState()) {
            case RequestState::Finished:
                std::cout << "FINISHED";
                break;
            case RequestState::Failed:
                std::cout << "FAILED";
                break;
            case RequestState::Decoding:
                std::cout << "DECODING";
                break;
            case RequestState::Prefilling:
                std::cout << "PREFILLING";
                break;
            case RequestState::Pending:
                std::cout << "PENDING";
                break;
        }
        std::cout << std::endl;
    }
    
    // 5. Statistics
    std::cout << "\n[Stats]" << std::endl;
    const auto& stats = engine->getStats();
    std::cout << "  Total tokens processed: " << stats.tokensProcessed << std::endl;
    std::cout << "  Requests completed: " << stats.requestsCompleted << std::endl;
    std::cout << "  Requests failed: " << stats.requestsFailed << std::endl;
    
    // Cleanup
    engine->shutdown();
    engineThread.join();
    
    std::cout << "\n=== Example Complete ===" << std::endl;
    return 0;
}

