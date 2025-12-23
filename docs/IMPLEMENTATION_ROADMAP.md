# Implementation Roadmap: Request/Response Integration

## Current Status ✅

- ✅ Request class (180 lines) - Complete implementation
- ✅ Response class (200 lines) - Complete implementation  
- ✅ SamplingParams validation - Complete
- ✅ Streaming callback system - Complete
- ✅ Error handling framework - Complete
- ✅ Nanosecond-precision timing - Complete
- ✅ Comprehensive documentation (3000+ lines)

---

## Next Phase: Engine Integration

### Phase 1: Scheduler Integration (Week 1)

#### 1.1 Extend Scheduler Interface

**File**: `include/cortexstream/scheduler.h`

```cpp
namespace cortexstream {

class Scheduler {
public:
    /**
     * Schedule a batch of requests for execution.
     * Uses Request metadata for intelligent batching.
     */
    Batch scheduleBatch(const std::vector<Request*>& pendingRequests);
    
    /**
     * Build prefill batch (full prompt processing).
     * Respects Request hints and prioritization.
     */
    Batch buildPrefillBatch(
        const std::vector<Request*>& candidates,
        int maxBatchTokens = 2048);
    
    /**
     * Build decode batch (single-token generation).
     * Interleaves multiple sequences.
     */
    Batch buildDecodeBatch(
        const std::vector<Request*>& inFlight,
        int maxSequences = 32);
    
    /**
     * Get scheduling hints from Request.
     */
    static int getPriority(const Request& req);
    static int getMaxTokens(const Request& req);
    static bool canSkipDecode(const Request& req);
    
private:
    // Priority-based queue
    std::priority_queue<Request*> priorityQueue_;
    
    // Metrics
    struct SchedulingMetrics {
        int batchesPrefilled = 0;
        int batchesDecoded = 0;
        int totalTokensProcessed = 0;
    } metrics_;
};

}  // namespace cortexstream
```

#### 1.2 Implement Scheduler Methods

**File**: `src/scheduler/scheduler.cpp`

```cpp
#include "cortexstream/scheduler.h"

namespace cortexstream {

Batch Scheduler::scheduleBatch(
    const std::vector<Request*>& pendingRequests) {
    
    // Sort by priority
    auto sorted = pendingRequests;
    std::sort(sorted.begin(), sorted.end(),
             [](const Request* a, const Request* b) {
                 return a->getPriority() > b->getPriority();
             });
    
    // Separate by phase
    std::vector<Request*> prefillCandidates;
    std::vector<Request*> decodeCandidates;
    
    for (auto* req : sorted) {
        if (req->getGeneratedTokenCount() == 0) {
            // Not started yet - prefill
            prefillCandidates.push_back(req);
        } else if (!req->isFinished()) {
            // In progress - decode
            decodeCandidates.push_back(req);
        }
    }
    
    // Build batches
    auto prefillBatch = buildPrefillBatch(prefillCandidates);
    auto decodeBatch = buildDecodeBatch(decodeCandidates);
    
    // Merge batches (prefill first)
    Batch combined = prefillBatch;
    combined.merge(decodeBatch);
    
    // Update metrics
    metrics_.batchesPrefilled++;
    metrics_.totalTokensProcessed += combined.totalTokenCount();
    
    return combined;
}

Batch Scheduler::buildPrefillBatch(
    const std::vector<Request*>& candidates,
    int maxBatchTokens) {
    
    Batch batch;
    int totalTokens = 0;
    
    for (auto* req : candidates) {
        int reqLen = req->getInputTokenCount();
        
        // Check capacity
        if (totalTokens + reqLen > maxBatchTokens) {
            break;  // Batch is full
        }
        
        // Add to batch
        batch.addRequest(req, Phase::Prefill);
        totalTokens += reqLen;
    }
    
    return batch;
}

Batch Scheduler::buildDecodeBatch(
    const std::vector<Request*>& inFlight,
    int maxSequences) {
    
    Batch batch;
    
    for (auto* req : inFlight) {
        if (batch.getSequenceCount() >= maxSequences) {
            break;  // Batch is full
        }
        
        // Add to batch
        batch.addRequest(req, Phase::Decode);
    }
    
    return batch;
}

}  // namespace cortexstream
```

---

### Phase 2: Engine Integration (Week 2)

#### 2.1 Extend Engine Interface

**File**: `include/cortexstream/engine.h`

```cpp
namespace cortexstream {

class Engine {
public:
    /**
     * Synchronous generation (blocking).
     */
    Response generate(Request& req);
    
    /**
     * Asynchronous generation (returns future).
     */
    std::future<Response> generateAsync(Request& req);
    
    /**
     * Prefill phase only (for length prediction).
     */
    void prefill(Request& req, Response& resp);
    
    /**
     * Decode phase only (continues from cached prefill).
     */
    void decode(Request& req, Response& resp);

private:
    std::shared_ptr<Scheduler> scheduler_;
    std::shared_ptr<KVCache> kvCache_;
    std::shared_ptr<ModelBackend> model_;
    
    // Prefill implementation
    void executePrefill(Request& req, Response& resp);
    
    // Decode implementation
    void executeDecode(Request& req, Response& resp);
    
    // Helper: check stop conditions
    bool shouldStop(const Request& req, const Response& resp);
};

}  // namespace cortexstream
```

#### 2.2 Implement Prefill Phase

**File**: `src/engine/engine_prefill.cpp`

```cpp
#include "cortexstream/engine.h"

namespace cortexstream {

void Engine::executePrefill(Request& req, Response& resp) {
    // Record input count
    resp.setInputTokenCount(req.getInputTokenCount());
    
    // 1. Check if we can reuse cache
    bool reuseCache = req.shouldReuseCache();
    CacheAllocation alloc;
    
    if (reuseCache) {
        // Try to find existing cached state
        auto existing = kvCache_->getCachedState(req.getId());
        if (existing) {
            // Reuse: just extend for new tokens
            alloc = kvCache_->extendCacheAllocation(existing, req.getMaxTokens());
        } else {
            // Not found: allocate new
            alloc = kvCache_->allocateBlocks(
                calculateNeededBlocks(req.getInputTokenCount()),
                req.getId()
            );
        }
    } else {
        // Allocate fresh cache
        alloc = kvCache_->allocateBlocks(
            calculateNeededBlocks(req.getInputTokenCount()),
            req.getId()
        );
    }
    
    // 2. Forward pass: encode entire prompt
    const auto& inputTokens = req.getInputTokens();
    const auto& samplingParams = req.getSamplingParams();
    
    auto logits = model_->forward(inputTokens, kvCache_->getKV(req.getId()));
    
    // 3. Store embeddings for decode phase
    kvCache_->storeEmbeddings(logits, alloc);
    
    // 4. If prefillOnly mode, sample first token and return
    if (req.allowsPrefillOnly()) {
        int firstToken = model_->sampleToken(
            logits.back(),
            samplingParams
        );
        req.addGeneratedToken(firstToken);
        resp.addToken(firstToken);
        
        // Notify callback
        if (req.isStreamingEnabled()) {
            std::string text = tokenizer_->decode(firstToken);
            resp.appendText(text);
            req.notifyToken(firstToken, false);
        }
    }
}

}  // namespace cortexstream
```

#### 2.3 Implement Decode Phase

**File**: `src/engine/engine_decode.cpp`

```cpp
#include "cortexstream/engine.h"

namespace cortexstream {

void Engine::executeDecode(Request& req, Response& resp) {
    const auto& samplingParams = req.getSamplingParams();
    const auto& stopTokens = req.getStopTokens();
    const auto& stopString = req.getStopString();
    
    int outputCount = 0;
    const int maxTokens = req.getMaxTokens();
    
    while (outputCount < maxTokens) {
        // 1. Check cancellation (thread-safe)
        if (req.isCancelled()) {
            resp.setStoppedByUser();
            break;
        }
        
        // 2. Get previous token
        int prevToken = req.getGeneratedTokenCount() > 0
                       ? req.getGeneratedTokens().back()
                       : EOS_TOKEN_ID;
        
        // 3. Single-token forward pass (KV cache reused!)
        auto logits = model_->forward(
            {prevToken},
            kvCache_->getKV(req.getId()),
            outputCount  // position in sequence
        );
        
        // 4. Sample next token
        int nextToken = model_->sampleToken(logits[0], samplingParams);
        
        // 5. Accumulate in request and response
        req.addGeneratedToken(nextToken);
        resp.addToken(nextToken);
        
        // 6. Decode to text
        std::string textPiece = tokenizer_->decode(nextToken);
        resp.appendText(textPiece);
        
        // 7. Stream immediately if enabled (low latency!)
        if (req.isStreamingEnabled()) {
            req.notifyToken(nextToken, false);
        }
        
        // 8. Check stop conditions
        if (nextToken == EOS_TOKEN_ID) {
            resp.setStoppedByEOS();
            break;
        }
        
        if (!stopTokens.empty() &&
            std::find(stopTokens.begin(), stopTokens.end(), nextToken)
            != stopTokens.end()) {
            resp.setStoppedByStopToken();
            break;
        }
        
        if (!stopString.empty() &&
            resp.getText().find(stopString) != std::string::npos) {
            resp.setStoppedByStopString();
            break;
        }
        
        if (outputCount + 1 >= maxTokens) {
            resp.setStoppedByMaxTokens();
            break;
        }
        
        outputCount++;
    }
    
    // 9. Mark response as finished (records end time)
    resp.finish();
    
    // 10. Notify final token
    if (req.isStreamingEnabled() && outputCount > 0) {
        int lastToken = req.getGeneratedTokens().back();
        req.notifyToken(lastToken, true);
    }
    
    // 11. Cleanup cache if requested
    if (req.shouldFreeCacheOnFinish()) {
        kvCache_->freeCacheBlocks(req.getId());
    }
}

}  // namespace cortexstream
```

#### 2.4 Implement Main Generate Methods

**File**: `src/engine/engine.cpp`

```cpp
#include "cortexstream/engine.h"

namespace cortexstream {

Response Engine::generate(Request& req) {
    Response resp(req.getId());
    
    try {
        // 1. Prefill (encode prompt)
        executePrefill(req, resp);
        
        // 2. Decode (generate tokens)
        if (!req.allowsPrefillOnly() && !req.isCancelled()) {
            executeDecode(req, resp);
        }
        
        // Mark request as finished
        // (Response already marked in executeDecode)
        
    } catch (const std::exception& e) {
        // Capture error in response
        resp.setError(std::string("Generation error: ") + e.what());
        
        // Mark request as failed
        req.setError(e.what());
    }
    
    return resp;
}

std::future<Response> Engine::generateAsync(Request& req) {
    return std::async(std::launch::async, [this, &req]() {
        return generate(req);
    });
}

}  // namespace cortexstream
```

---

### Phase 3: Streaming Integration (Week 3)

#### 3.1 Add Streaming Server

**File**: `include/cortexstream/streaming_server.h`

```cpp
namespace cortexstream {

class StreamingServer {
public:
    /**
     * Stream chunks to client as they arrive.
     * Called by Engine for each token.
     */
    void streamChunk(const ResponseChunk& chunk);
    
    /**
     * Register callback for chunk delivery.
     */
    using ChunkCallback = std::function<void(const ResponseChunk&)>;
    void setChunkCallback(ChunkCallback callback);
    
    /**
     * Get full response once streaming is complete.
     */
    Response waitForCompletion(const std::string& requestId);

private:
    std::map<std::string, std::vector<ResponseChunk>> chunks_;
    std::map<std::string, std::promise<Response>> completions_;
    ChunkCallback callback_;
    std::mutex mutex_;
};

}  // namespace cortexstream
```

#### 3.2 Integrate Streaming

Update `engine_decode.cpp`:

```cpp
// In executeDecode, after token accumulation:

if (req.isStreamingEnabled()) {
    ResponseChunk chunk(
        req.getId(),
        nextToken,
        textPiece,
        false  // not finished yet
    );
    
    // Stream to client immediately (callback)
    req.notifyToken(nextToken, false);
    
    // Also notify streaming server
    if (streamingServer_) {
        streamingServer_->streamChunk(chunk);
    }
}

// At the end, before freeing cache:

if (req.isStreamingEnabled()) {
    ResponseChunk finalChunk(
        req.getId(),
        -1,  // no token
        "",
        true  // finished
    );
    
    if (streamingServer_) {
        streamingServer_->streamChunk(finalChunk);
    }
}
```

---

### Phase 4: Testing (Week 4)

#### 4.1 Unit Tests

**File**: `tests/test_request_response.cpp`

```cpp
#include <gtest/gtest.h>
#include "cortexstream/request.h"
#include "cortexstream/response.h"

namespace cortexstream {

class RequestTest : public ::testing::Test {};

TEST_F(RequestTest, ConstructionAndImmutability) {
    Request req("id-1", "prompt", {1, 2, 3}, 256);
    
    EXPECT_EQ(req.getId(), "id-1");
    EXPECT_EQ(req.getPrompt(), "prompt");
    EXPECT_EQ(req.getInputTokenCount(), 3);
    EXPECT_EQ(req.getMaxTokens(), 256);
}

TEST_F(RequestTest, SamplingParamsValidation) {
    SamplingParams valid;
    valid.temperature = 0.7f;
    valid.topK = 40;
    valid.topP = 0.9f;
    
    EXPECT_TRUE(valid.isValid());
    
    SamplingParams invalid;
    invalid.temperature = -1.0f;
    EXPECT_FALSE(invalid.isValid());
}

TEST_F(RequestTest, Cancellation) {
    Request req("id-1", "prompt", {1, 2, 3}, 256);
    
    EXPECT_FALSE(req.isCancelled());
    
    req.cancel();
    
    EXPECT_TRUE(req.isCancelled());
}

class ResponseTest : public ::testing::Test {};

TEST_F(ResponseTest, TokenAccumulation) {
    Response resp("id-1");
    
    resp.addToken(100);
    resp.addToken(200);
    resp.addToken(300);
    
    EXPECT_EQ(resp.getOutputTokenCount(), 3);
    EXPECT_EQ(resp.getTokens()[0], 100);
}

TEST_F(ResponseTest, CompletionReasons) {
    Response resp("id-1");
    
    EXPECT_FALSE(resp.hasStoppedByEOS());
    
    resp.setStoppedByEOS();
    
    EXPECT_TRUE(resp.hasStoppedByEOS());
    EXPECT_EQ(resp.getCompletionReason(), "end_of_sequence");
}

TEST_F(ResponseTest, Timing) {
    Response resp("id-1");
    uint64_t startNs = resp.getStartTimeNs();
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    resp.finish();
    
    uint64_t latencyNs = resp.getLatencyNs();
    EXPECT_GT(latencyNs, 50'000'000);  // At least 50ms
}

}  // namespace cortexstream
```

#### 4.2 Integration Tests

**File**: `tests/test_engine_integration.cpp`

```cpp
#include <gtest/gtest.h>
#include "cortexstream/engine.h"

namespace cortexstream {

class EngineIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize mock model
        // Initialize cache
        // Initialize engine
    }
};

TEST_F(EngineIntegrationTest, BasicGeneration) {
    Request req("test-1", "Hello", {1, 2, 3}, 10);
    
    auto resp = engine_.generate(req);
    
    EXPECT_FALSE(resp.hasError());
    EXPECT_GT(resp.getOutputTokenCount(), 0);
    EXPECT_GT(resp.getLatencyMs(), 0.0);
}

TEST_F(EngineIntegrationTest, Streaming) {
    Request req("test-2", "Test", {1, 2}, 5);
    req.setStreaming(true);
    
    std::vector<int> streamedTokens;
    req.setTokenCallback([&](int token, bool done) {
        streamedTokens.push_back(token);
    });
    
    auto resp = engine_.generate(req);
    
    EXPECT_EQ(resp.getOutputTokenCount(), streamedTokens.size());
}

TEST_F(EngineIntegrationTest, Cancellation) {
    Request req("test-3", "Long text", {1, 2, 3}, 1000);
    
    auto future = engine_.generateAsync(req);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    req.cancel();
    
    auto resp = future.get();
    
    EXPECT_TRUE(resp.hasStoppedByUser());
}

}  // namespace cortexstream
```

---

## Implementation Timeline

### Week 1: Scheduler Integration
- [ ] Extend Scheduler interface
- [ ] Implement priority-based batching
- [ ] Implement prefill/decode batch building
- [ ] Unit tests for Scheduler

### Week 2: Engine Core
- [ ] Implement prefill phase
- [ ] Implement decode phase
- [ ] Connect to KV Cache
- [ ] Connect to Model Backend
- [ ] Integration tests

### Week 3: Streaming
- [ ] Implement StreamingServer
- [ ] Wire callbacks through generation
- [ ] Low-latency token delivery
- [ ] Client-side streaming tests

### Week 4: Polish & Testing
- [ ] Performance benchmarking
- [ ] Error path testing
- [ ] Thread safety validation
- [ ] Load testing with multiple concurrent requests

---

## Success Criteria

- [ ] Request/Response fully integrated with Engine
- [ ] All methods tested (unit + integration)
- [ ] Streaming works with <5ms latency per token
- [ ] Scheduler can handle 32+ concurrent sequences
- [ ] Memory-efficient with O(1) cache allocation
- [ ] Thread-safe cancellation working correctly
- [ ] Error paths properly handled
- [ ] Performance metrics accurate (nanosecond precision)
- [ ] Documentation complete and up-to-date
- [ ] CI/CD tests passing

---

## Known Limitations & Future Work

### Current Limitations
1. Single engine instance (no distributed inference)
2. No tensor parallelism
3. No pipeline parallelism
4. No speculative decoding
5. Batch size limited by GPU memory

### Future Enhancements
1. Multi-GPU support via Scheduler
2. Request prefixing (shared prompt processing)
3. Batched sampling for higher throughput
4. Adaptive batching based on request latency SLAs
5. Token-level caching for prefix reuse

---

## Resources

- **Request/Response Contract**: [REQUEST_RESPONSE_CONTRACT.md](REQUEST_RESPONSE_CONTRACT.md)
- **Architecture**: [ARCHITECTURE_INTEGRATION.md](ARCHITECTURE_INTEGRATION.md)
- **Quick Reference**: [REQUEST_RESPONSE_QUICK_REF.md](REQUEST_RESPONSE_QUICK_REF.md)
- **KV Cache Design**: [KV_CACHE_DESIGN.md](KV_CACHE_DESIGN.md)

