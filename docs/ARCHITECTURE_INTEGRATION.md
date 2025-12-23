# CortexStream Architecture: Integration Guide

## System Overview

CortexStream follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────┐
│  Client API (Request/Response)                          │
│  - User-facing contracts                                │
│  - Streaming support                                    │
│  - Error handling                                       │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│  Inference Engine                                       │
│  - Prefill phase (encode full prompt)                   │
│  - Decode phase (generate token-by-token)               │
│  - Scheduling decisions                                 │
│  - Cache management                                     │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│  KV Cache Layer (KVBlockAllocator + KVCache)           │
│  - Zero-copy tensor views                               │
│  - O(1) block allocation                                │
│  - Memory-efficient storage                             │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│  Model Backend (MLX/MPS)                                │
│  - Forward pass execution                               │
│  - Token sampling                                       │
│  - Logit computation                                    │
└─────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Request/Response Layer

#### Request
**Responsibility**: Encapsulate client intent with scheduling hints

```cpp
// Client creates request
Request req("request-1", "What is AI?", inputTokens, maxTokens=256);
req.setSamplingParams(params);
req.setStreaming(true);
req.setTokenCallback(onToken);
req.setPriority(5);
req.setReuseCache(true);

// Engine never modifies input fields
// Engine modifies: generatedTokens_, finished_, failed_, errorMessage_
// Engine reads: scheduling hints, sampling params, stop conditions
```

#### Response
**Responsibility**: Accumulate generation results with metadata

```cpp
// Engine creates response
Response resp(req.getId());

// During generation
for (int i = 0; i < maxTokens; i++) {
    int token = model.sampleNextToken(...);
    resp.addToken(token);
    req.notifyToken(token, false);  // Callback to client
    
    if (token == stopToken) {
        resp.setStoppedByStopToken();
        break;
    }
}

// On completion
resp.finish();  // Records end time

// Client accesses result
std::cout << resp.getText() << std::endl;
std::cout << resp.getLatencyMs() << " ms" << std::endl;
```

---

### 2. Inference Engine

#### Prefill Phase
Processes entire input prompt in one forward pass:

```cpp
void Engine::prefill(Request& req, Response& resp, KVCache& cache) {
    // Read from request
    const auto& tokens = req.getInputTokens();
    const auto& samplingParams = req.getSamplingParams();
    
    // Allocate cache for this sequence
    auto cacheAlloc = cache.allocateBlocks(
        numBlocks=calculateNeeded(tokens.size()),
        sequenceId=req.getId()
    );
    
    // Forward pass
    auto logits = model.forward(tokens, cache);
    
    // Cache embeddings for decode phase
    cache.storeEmbeddings(logits, cacheAlloc);
    
    // Don't sample yet - wait for decode
}
```

#### Decode Phase
Generates tokens one at a time, with KV cache hits:

```cpp
void Engine::decode(Request& req, Response& resp, KVCache& cache) {
    int outputCount = 0;
    
    while (outputCount < req.getMaxTokens()) {
        // Check cancellation (thread-safe)
        if (req.isCancelled()) {
            resp.setStoppedByUser();
            break;
        }
        
        // Get previous token (or EOS at start)
        int prevToken = req.getGeneratedTokenCount() > 0 
                       ? req.getGeneratedTokens().back()
                       : EOS_TOKEN;
        
        // Single-token forward pass (KV cache reused!)
        auto logits = model.forward({prevToken}, cache);
        
        // Sample from logits
        int nextToken = sampler.sample(
            logits,
            req.getSamplingParams()
        );
        
        // Accumulate result
        resp.addToken(nextToken);
        req.addGeneratedToken(nextToken);
        resp.appendText(tokenizer.decode(nextToken));
        
        // Stream to client if enabled
        if (req.isStreamingEnabled()) {
            req.notifyToken(nextToken, false);
        }
        
        // Check stop conditions
        if (nextToken == EOS_TOKEN) {
            resp.setStoppedByEOS();
            break;
        }
        if (matchesStopString(resp.getText(), req.getStopString())) {
            resp.setStoppedByStopString();
            break;
        }
        if (std::find(req.getStopTokens().begin(),
                      req.getStopTokens().end(),
                      nextToken) != req.getStopTokens().end()) {
            resp.setStoppedByStopToken();
            break;
        }
        
        outputCount++;
    }
    
    // Mark finished
    resp.finish();
    
    // Cleanup if requested
    if (req.shouldFreeCacheOnFinish()) {
        cache.freeCacheBlocks(req.getId());
    }
}
```

---

### 3. KV Cache Integration

#### Cache-Aware Scheduling
```cpp
// Scheduler reads Request hints
bool shouldReuse = req.shouldReuseCache();
bool shouldFree = req.shouldFreeCacheOnFinish();

if (shouldReuse) {
    // Try to find cached state from previous request
    // Saves recomputation of shared prefix
    auto prevState = cache.getState(req.getId());
    if (prevState) {
        cache.attachToExisting(req.getId(), prevState);
    }
}
```

#### Cache Lifecycle
```cpp
// Allocate at prefill
auto alloc = cache.allocateBlocks(numBlocks, req.getId());

// Use during decode (zero-copy access)
auto kv = cache.getKV(req.getId());
auto logits = model.forward(token, kv);

// Free after generation (if requested)
if (req.shouldFreeCacheOnFinish()) {
    cache.freeCacheBlocks(req.getId());
}
```

#### Memory Efficiency
- Tokens for position $t$ reuse KV from positions $0...t-1$
- No recomputation for seen tokens
- Block-level allocation prevents fragmentation
- O(1) allocation/deallocation

---

### 4. Scheduler Integration

#### Batching Decisions
```cpp
class Scheduler {
public:
    Batch buildPrefillBatch(const std::vector<Request*>& pendingReqs) {
        Batch batch;
        int totalTokens = 0;
        
        for (auto* req : pendingReqs) {
            int reqLen = req->getInputTokenCount();
            
            // Don't exceed max batch tokens
            if (totalTokens + reqLen > MAX_PREFILL_TOKENS) break;
            
            // Honor prefillOnly hints
            if (req->allowsPrefillOnly()) {
                batch.add(req);  // Do prefill only
            }
            
            totalTokens += reqLen;
        }
        
        return batch;
    }
    
    Batch buildDecodeBatch(const std::vector<Request*>& inFlightReqs) {
        // Sort by priority for fairness
        auto sorted = inFlightReqs;
        std::sort(sorted.begin(), sorted.end(),
                 [](const auto* a, const auto* b) {
                     return a->getPriority() > b->getPriority();
                 });
        
        Batch batch;
        for (auto* req : sorted) {
            if (!req->isFinished()) {
                batch.add(req);
            }
        }
        
        return batch;
    }
};
```

#### Priority-Based Scheduling
```cpp
// High-priority request
req.setPriority(10);  // SLA: 100ms max latency

// Low-priority background task
req.setPriority(1);   // Best effort
```

---

## Workflow: End-to-End Example

### Client Submission
```cpp
// 1. Client creates request
Request req("user-req-42", 
           "Explain quantum computing in 100 words",
           tokenizer.encode("Explain quantum computing in 100 words"),
           maxTokens=100);

// 2. Configure generation
SamplingParams params;
params.temperature = 0.7f;
params.topP = 0.9f;
req.setSamplingParams(params);

// 3. Enable streaming
req.setStreaming(true);
req.setTokenCallback([](int token, bool done) {
    std::cout << "[" << token << "]";
    if (done) std::cout << " [COMPLETE]" << std::endl;
});

// 4. Submit to engine
auto future = engine.generateAsync(req);
```

### Scheduler Processing
```cpp
// Scheduler receives request
// Priority: 0 (default, best effort)
// Input length: 12 tokens
// Output budget: 100 tokens
// Streaming: enabled
// Cache: no reuse

// Scheduler decides:
// - Add to prefill batch (all 12 tokens at once)
// - Decode later (1 token at a time, low latency)
```

### Prefill Execution
```cpp
// Engine::prefill()
// 1. Allocate cache blocks for sequence
//    Calculated: 100 tokens = ~5 blocks
CacheAllocation alloc = cache.allocateBlocks(5, "user-req-42");

// 2. Forward pass: 12 input tokens → embeddings
auto embeddings = model.forward(tokenIds, cache);

// 3. Store in cache for decode phase
cache.store(embeddings, alloc);

// 4. Response gets input count
resp.setInputTokenCount(12);

// Request ready for decode
```

### Decode Execution
```cpp
// Engine::decode()
for (int step = 0; step < 100; step++) {
    // 1. Single-token forward (cached KV reused!)
    int prevToken = req.getGeneratedTokenCount() > 0
                   ? req.getGeneratedTokens().back()
                   : EOS_TOKEN;
    
    auto logits = model.forward({prevToken}, cache);
    
    // 2. Sample from logits
    int nextToken = sampler.sample(logits, params);
    
    // 3. Accumulate
    resp.addToken(nextToken);
    req.addGeneratedToken(nextToken);
    auto text = tokenizer.decode(nextToken);
    resp.appendText(text);
    
    // 4. Stream immediately (low latency!)
    req.notifyToken(nextToken, false);  // Callback executes
    
    // 5. Check stop conditions
    if (nextToken == EOS_TOKEN) {
        resp.setStoppedByEOS();
        break;
    }
    
    // 6. Check cancellation (thread-safe)
    if (req.isCancelled()) {
        resp.setStoppedByUser();
        break;
    }
}

// Completion
resp.finish();  // Records end time
cache.freeCacheBlocks("user-req-42");  // Return memory
```

### Client Receives Result
```cpp
// Wait for completion
auto resp = future.get();

// Access result
std::cout << "Output: " << resp.getText() << std::endl;
std::cout << "Tokens: " << resp.getOutputTokenCount() << std::endl;
std::cout << "Latency: " << resp.getLatencyMs() << " ms" << std::endl;
std::cout << "Throughput: " << resp.getTokensPerSecond() << " tok/s" << std::endl;
std::cout << "Reason: " << resp.getCompletionReason() << std::endl;

// Expected output:
// Output: Quantum computing leverages quantum mechanics...
// Tokens: 87
// Latency: 1234.5 ms
// Throughput: 70.4 tok/s
// Reason: end_of_sequence
```

---

## Data Flow Diagram

```
┌──────────────┐
│   Request    │  - Input tokens
│              │  - Sampling params
│              │  - Scheduling hints
│              │  - Streaming callback
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────┐
│         Scheduler                    │
│  - Priority-based batching           │
│  - Prefill/decode phase decisions    │
└──────┬───────────────────────────────┘
       │
       ├─────────────────┬──────────────────┐
       ▼                 ▼                  ▼
    PREFILL            DECODE             ERROR
       │                 │                  │
       ├──────────────┬──┤                  │
       ▼              ▼  ▼                  ▼
   ┌────────────────────────────────────────────┐
   │      KV Cache Layer                        │
   │  - Block allocation                        │
   │  - Zero-copy tensor views                  │
   │  - Embedding storage                       │
   └────────────────────────────────────────────┘
       │
       ▼
   ┌────────────────────────────────────────────┐
   │      Model Forward Pass                    │
   │  - MLX tensors                             │
   │  - MPS acceleration                        │
   │  - Logits computation                      │
   └────────────────────────────────────────────┘
       │
       ▼
   ┌────────────────────────────────────────────┐
   │      Sampling                              │
   │  - Temperature scaling                     │
   │  - Top-K filtering                         │
   │  - Top-P nucleus sampling                  │
   └────────────────────────────────────────────┘
       │
       ▼
   ┌────────────────────────────────────────────┐
   │      Token Accumulation                    │
   │  - Add to response                         │
   │  - Add to request                          │
   │  - Decode to text                          │
   └────────────────────────────────────────────┘
       │
       ├─ Streaming enabled? ──→ req.notifyToken() ──→ Client Callback
       │                                            (immediate)
       │
       ▼
   ┌────────────────────────────────────────────┐
   │      Stop Condition Check                  │
   │  - EOS token?                              │
   │  - Max tokens?                             │
   │  - Stop string?                            │
   │  - User cancelled?                         │
   └────────────────────────────────────────────┘
       │
       ├─ Stop? ──→ resp.finish() ──→ Record end time
       │
       ├─ Cache free? ──→ cache.freeCacheBlocks()
       │
       ▼
   ┌────────────────────────────────────────────┐
   │      Response (Complete)                   │
   │  - Generated text                          │
   │  - Token sequence                          │
   │  - Completion reason                       │
   │  - Latency metrics                         │
   │  - Error (if any)                          │
   └────────────────────────────────────────────┘
       │
       ▼
   Client receives response
```

---

## Concurrency Model

### Thread Safety Guarantees

| Component | Thread Safety | Notes |
|-----------|--------------|-------|
| Request::cancelled_ | atomic<bool> | acquire/release semantics |
| Request::generatedTokens_ | NOT safe | Engine thread only |
| Response::tokens_ | NOT safe | Engine thread only |
| Callbacks | Safe | Called from engine thread |
| KV Cache | Depends | Single engine thread typical |

### Recommended Threading

```
Client Thread              Engine Thread
    │                          │
    ├─ Create Request ─────────┤
    │                          │
    ├─ Submit (async) ─────────┤
    │                          │
    │                          ├─ Prefill
    │                          │
    ├─ req.cancel() ───────────┤ (atomic)
    │                          │
    │                          ├─ Check isCancelled()
    │                          │
    │                          ├─ Decode
    │                          │
    │                          ├─ req.notifyToken() 
    │                          │    (callback)
    │                          │
    │◄──────────────────────────┤
    │                          │
    └─ Get Response ───────────┤
```

### Synchronization Points

1. **Request Submission**: Channel between client and engine
2. **Token Streaming**: Callback from engine to client
3. **Cancellation**: Atomic flag for thread-safe interruption
4. **Final Response**: Blocking wait for completion

---

## Performance Characteristics

### Latency
- Prefill: O(n) where n = prompt length (few ms)
- Decode: O(m) where m = max tokens (100-1000ms typical)
- Per-token latency: 10-15ms (depends on model size)
- Streaming startup: <5ms (quick first token)

### Throughput
- Batch size: 1-64 sequences (depends on memory)
- Tokens/sec: 50-100 typical (varies with model)
- Can batch prefill + decode phases

### Memory
- Per-sequence KV cache: ~2KB per token (depends on model dim)
- Request object: ~400 bytes
- Response object: ~500 bytes
- Cached embeddings: efficient block allocation

### Scalability
- Multiple engines: can run in parallel processes
- Single engine: handles one batch at a time
- Scheduler: optimizes throughput via smart batching
- KV cache: O(1) block management

---

## Configuration Options

### Request Tuning
```cpp
req.setPriority(10);              // High priority SLA
req.setStreaming(true);           // Enable low-latency streaming
req.setReuseCache(true);          // Reuse from previous request
req.setFreeCacheOnFinish(true);   // Memory-constrained
req.setAllowPrefillOnly(false);   // Must do full generation
```

### Engine Tuning
```cpp
Engine engine;
engine.setMaxBatchTokens(2048);   // Prefill capacity
engine.setMaxSequences(32);       // Concurrent sequences
engine.setDecodeBatchSize(16);    // Decode phase capacity
engine.enableStreaming(true);     // Activate streaming path
```

### Cache Tuning
```cpp
KVBlockAllocator alloc;
alloc.setBlockSize(256);          // Tokens per block
alloc.setMaxBlocks(1024);         // Total memory budget
alloc.setFragmentationThreshold(0.2);  // Trigger defrag
```

---

## Next Steps

1. **Implement Engine**: Integrate prefill/decode phases
2. **Implement Scheduler**: Batching and priority logic
3. **Wire Streaming**: Connect callbacks through generation
4. **Performance Tuning**: Measure and optimize latency/throughput
5. **Error Handling**: Comprehensive error paths
6. **Testing**: Unit + integration + performance benchmarks

