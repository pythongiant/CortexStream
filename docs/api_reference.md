# CortexStream API Reference

## Table of Contents
1. [Request](#request)
2. [KVCache](#kvcache)
3. [Scheduler](#scheduler)
4. [ModelBackend](#modelbackend)
5. [InferenceEngine](#inferenceengine)
6. [Sampler](#sampler)

---

## Request

Represents a single inference request with associated metadata and state.

### Constructor
```cpp
Request(const std::string& id,
        const std::vector<int>& promptTokens,
        int maxTokens = 512);
```

### State Queries

#### `const std::string& getId() const`
Returns the unique request identifier.

#### `RequestState getState() const`
Returns current request state:
- `RequestState::Pending` - Waiting to be scheduled
- `RequestState::Prefilling` - Processing prompt
- `RequestState::Decoding` - Generating tokens
- `RequestState::Finished` - Completed successfully
- `RequestState::Failed` - Error occurred

#### `bool isFinished() const`
Returns true if state is `Finished` or `Failed`.

#### `bool isFailed() const`
Returns true if state is `Failed`.

#### `int getPromptLength() const`
Returns size of initial prompt.

#### `int getGeneratedLength() const`
Returns number of generated tokens so far.

#### `const std::vector<int>& getPromptTokens() const`
Returns immutable prompt tokens.

#### `const std::vector<int>& getGeneratedTokens() const`
Returns immutable generated tokens.

#### `int getMaxTokens() const`
Returns maximum tokens to generate.

#### `const SamplingParams& getSamplingParams() const`
Returns sampling configuration.

#### `std::chrono::system_clock::time_point getCreationTime() const`
Returns request creation timestamp.

### State Management

#### `void setState(RequestState state)`
Updates request state.

#### `void addToken(int token)`
Appends generated token and calls callback if registered.

#### `void setSamplingParams(const SamplingParams& params)`
Configures sampling strategy:
```cpp
struct SamplingParams {
    float temperature = 1.0f;  // 0 = greedy, higher = more random
    int topK = 40;             // Top-K sampling
    float topP = 0.9f;         // Nucleus sampling
    bool greedy = false;       // Override all: greedy only
    uint32_t seed = 0;         // Random seed
};
```

#### `void setTokenCallback(TokenCallback callback)`
Registers callback for token streaming:
```cpp
using TokenCallback = std::function<void(int token, bool finished)>;
```

---

## KVCache

Block-based key-value cache for storing transformer activations.

### Constructor
```cpp
KVCache(size_t cacheSize,      // Total bytes (e.g., 8GB)
        size_t hiddenSize,      // Model hidden dimension
        size_t numLayers,       // Transformer layers
        size_t blockSize = 16);  // Tokens per block
```

### Block Management

#### `int allocateBlock(const std::string& requestId)`
Allocates KV cache block for request.
- **Returns**: Block ID (≥0) on success, -1 on OOM
- **Note**: Automatically associates block with request

#### `void freeBlock(int blockId)`
Deallocates KV cache block.

#### `KVBlock* getBlock(int blockId)`
Returns mutable pointer to block data.
- **Returns**: nullptr if blockId invalid

#### `const KVBlock* getBlock(int blockId) const`
Returns const pointer to block data.

### Request Management

#### `std::vector<int> getBlocksForRequest(const std::string& requestId)`
Returns all blocks allocated to request.

#### `void associateBlockWithRequest(int blockId, const std::string& requestId)`
Links block to request tracking.

#### `void clearRequest(const std::string& requestId)`
Deallocates all blocks for request and removes association.

### Statistics

#### `size_t getTotalAllocated() const`
Returns bytes currently allocated.

#### `size_t getTotalFree() const`
Returns bytes available for allocation.

#### `int getNumAllocatedBlocks() const`
Returns number of currently allocated blocks.

#### `bool isFull() const`
Returns true if no free blocks available.

### Performance

#### `void warmup()`
Prefetches cache memory to GPU. Call once during initialization.

---

## Scheduler

Request batching and scheduling orchestrator.

### Constructor
```cpp
Scheduler(int maxBatchSize = 32);  // Max requests per batch
```

### Request Submission

#### `bool submitRequest(std::shared_ptr<Request> request)`
Adds request to pending queue.
- **Returns**: true on success, false if invalid
- **Thread-safe**: Yes (mutex protected)

### State Queries

#### `bool hasWork() const`
Returns true if pending or active requests exist.

#### `bool hasPendingRequests() const`
Returns true if requests awaiting prefill exist.

#### `bool hasActiveRequests() const`
Returns true if decoding requests exist.

#### `int getNumActiveRequests() const`
Returns count of currently decoding requests.

#### `int getMaxBatchSize() const`
Returns configured maximum batch size.

### Batch Building

#### `void acceptNewRequests()`
Moves pending requests to active list (up to maxBatchSize).
- **Call**: Once per inference loop iteration

#### `Batch buildPrefillBatch()`
Returns batch of requests in Prefilling state.
```cpp
struct Batch {
    std::vector<std::shared_ptr<Request>> requests;
    std::vector<int> sequenceLengths;
    int batchSize;
    bool isPrefill;
    bool empty() const;
    void clear();
};
```

#### `Batch buildDecodeBatch()`
Returns batch of requests in Decoding state.
- **Note**: Max size limited to maxBatchSize

### Request Management

#### `void markRequestReady(const std::string& requestId)`
Transitions request from Prefilling → Decoding.

#### `void markRequestFinished(const std::string& requestId)`
Transitions request to Finished, removes from active list.

#### `void markRequestFailed(const std::string& requestId)`
Transitions request to Failed, removes from active list.

#### `std::shared_ptr<Request> getRequest(const std::string& requestId)`
Looks up request by ID.
- **Returns**: Request pointer or nullptr if not found

---

## ModelBackend

GPU execution layer for model inference.

### Constructor
```cpp
ModelBackend(Device device = Device::MPS,
             DType dtype = DType::FP16);

enum class Device { MPS, CPU };
enum class DType { FP32, FP16, INT8 };
```

### Model Lifecycle

#### `bool loadModel(const std::string& modelPath)`
Loads MLX model from disk.
- **Returns**: true on success, false on error
- **Precondition**: modelPath must be valid MLX format
- **Side Effect**: Initializes hiddenSize, numLayers, vocabSize

#### `bool isLoaded() const`
Returns true if model successfully loaded.

### Forward Passes

#### `Tensor prefill(const Batch& batch, const std::vector<int>& tokenIds)`
Processes prompt tokens, outputs logits for all positions.
- **Input**: Batch of requests, concatenated token IDs
- **Output**: Logits tensor [batchSize, vocabSize]
- **Side Effect**: Computes KV cache (stored externally)
- **Note**: Call before decode for each request

#### `Tensor decode(const Batch& batch, const std::vector<int>& tokenIds)`
Processes last token only, uses KV cache from prefill.
- **Input**: Batch of requests, last token per request
- **Output**: Logits tensor [batchSize, vocabSize]
- **Requirement**: KV cache must exist from prefill phase

### Sampling

#### `int sampleToken(const Tensor& logits, const SamplingParams& params)`
Samples next token from logits according to params.
- **Input**: Logits [1, vocabSize], sampling strategy
- **Output**: Token ID (0 to vocabSize-1)
- **Strategy**:
  - If `greedy=true`: Argmax (deterministic)
  - Else if `topK>0`: Top-K sampling
  - Else if `topP>0`: Nucleus sampling
  - Else: Argmax

### Model Metadata

#### `size_t getHiddenSize() const`
Returns model hidden dimension (e.g., 4096).

#### `size_t getNumLayers() const`
Returns number of transformer layers (e.g., 32).

#### `size_t getVocabSize() const`
Returns vocabulary size (e.g., 128000).

#### `Device getDevice() const`
Returns configured device (MPS or CPU).

#### `DType getDType() const`
Returns data type (FP32, FP16, INT8).

### Performance

#### `void warmup()`
Executes dummy forward pass to warm GPU graph cache.
- **Side Effect**: Sets internal warmed flag
- **Time**: ~100ms on GPU, negligible on CPU
- **Recommendation**: Call once during engine initialization

#### `bool preloadGraph()`
Traces and caches computation graph for faster replay.
- **Returns**: true on success
- **Note**: MLX specific optimization

---

## InferenceEngine

Orchestrator running continuous batching inference loop.

### Constructor
```cpp
InferenceEngine(std::shared_ptr<ModelBackend> backend,
                std::shared_ptr<Scheduler> scheduler,
                std::shared_ptr<KVCache> cache);
```

### Lifecycle

#### `bool initialize()`
Sets up engine and validates dependencies.
- **Returns**: true if ready
- **Call**: Before first `run()`
- **Side Effect**: Calls warmup on backend and cache

#### `void run()`
Main inference loop (blocks until completion).
```cpp
while (scheduler->hasWork()) {
    scheduler->acceptNewRequests();
    
    Batch prefill = scheduler->buildPrefillBatch();
    if (!prefill.empty())
        backend->prefill(prefill, tokens);
    
    Batch decode = scheduler->buildDecodeBatch();
    if (!decode.empty()) {
        auto logits = backend->decode(decode, tokens);
        emitTokens(decode, logits);
    }
    
    cleanup();
}
```
- **Note**: Blocking - run in separate thread
- **Error Handling**: Continues on failure, logs errors

#### `void shutdown()`
Gracefully stops engine.
- **Idempotent**: Safe to call multiple times

### Control

#### `bool isRunning() const`
Returns true if main loop is active.

#### `void pause()`
Pauses inference (main loop sleeps).

#### `void resume()`
Resumes from pause.

### Monitoring

#### `const EngineStats& getStats() const`
Returns accumulated statistics:
```cpp
struct EngineStats {
    size_t tokensProcessed;    // Total tokens generated
    size_t requestsCompleted;  // Finished requests
    size_t requestsFailed;     // Failed requests
    float avgBatchSize;        // Average batch size
    std::chrono::milliseconds totalLatency;
};
```

#### `int getActiveRequests() const`
Returns number of currently decoding requests.

---

## Sampler

Token sampling strategies.

### Constructor
```cpp
Sampler(uint32_t seed = 0);  // Random seed
```

### Sampling Methods

#### `int greedy(const Tensor& logits)`
Argmax sampling - deterministic, always picks highest logit.
- **Time**: O(vocabSize)

#### `int topK(const Tensor& logits, int k, float temperature = 1.0f)`
Sample from K highest logits with temperature scaling.
- **K** ranges: 1-100 typical
- **Temperature**: 1.0 = unscaled, <1 = sharper, >1 = flatter

#### `int topP(const Tensor& logits, float p, float temperature = 1.0f)`
Nucleus sampling - sample from smallest set with cumulative probability ≥ P.
- **P** ranges: 0.7-0.99 typical
- **Temperature**: Same as topK

#### `int topKP(const Tensor& logits, int k, float p, float temperature = 1.0f)`
Combined top-K and top-P constraints.
- Effective only when both conditions are met

### Configuration

#### `void setSeed(uint32_t seed)`
Sets random seed for reproducibility.

---

## Common Patterns

### Basic Inference Loop
```cpp
// Setup
auto backend = std::make_shared<ModelBackend>();
backend->loadModel("model.mlx");

auto scheduler = std::make_shared<Scheduler>(32);
auto cache = std::make_shared<KVCache>(8GB, 4096, 32);
auto engine = std::make_shared<InferenceEngine>(
    backend, scheduler, cache
);
engine->initialize();

// Submit request
auto req = std::make_shared<Request>("user_001", prompt, 256);
scheduler->submitRequest(req);

// Run (in thread)
std::thread t([&] { engine->run(); });

// Wait
while (!req->isFinished()) {
    std::this_thread::sleep_for(10ms);
}

// Get result
auto tokens = req->getGeneratedTokens();
```

### Custom Sampling
```cpp
SamplingParams params;
params.temperature = 0.8f;
params.topK = 50;
params.topP = 0.95f;
params.seed = 12345;

req->setSamplingParams(params);
```

### Token Streaming
```cpp
req->setTokenCallback([](int token, bool finished) {
    if (!finished) {
        std::cout << tokenToString(token);
        std::cout.flush();
    } else {
        std::cout << "[END]" << std::endl;
    }
});
```

### Error Handling
```cpp
if (!backend->loadModel(path)) {
    std::cerr << "Model load failed" << std::endl;
    return 1;
}

if (!engine->initialize()) {
    std::cerr << "Engine init failed" << std::endl;
    return 1;
}

// Check for failures
while (!req->isFinished()) {
    std::this_thread::sleep_for(10ms);
}

if (req->isFailed()) {
    std::cerr << "Request failed" << std::endl;
} else {
    std::cout << "Success: " << req->getGeneratedLength() << " tokens" << std::endl;
}
```

---

## Thread Safety Summary

| Component | Thread-Safe | Notes |
|-----------|------------|-------|
| Request | Mostly | State updates via setState() |
| KVCache | No | Use from single engine thread |
| Scheduler | Yes | Mutex-protected queue |
| ModelBackend | No | Stateless, call from single thread |
| InferenceEngine | Mostly | run() is blocking, use pause/resume |

---

## Performance Tips

1. **Batch Size**: Larger batches (32) better GPU utilization
2. **FP16**: Faster and uses less memory than FP32
3. **Warmup**: Always call before inference
4. **Block Size**: 16 tokens good default
5. **Temperature**: Higher = more diversity, lower = more deterministic
6. **Top-K**: 40-50 typical, higher = more options
7. **Top-P**: 0.9-0.95 typical, <0.9 = stricter

---

## Error Codes & Status

All methods return status via:
- Return value (true/false or pointer/nullptr)
- Exceptions (caught by engine)
- Request state (for failures)

No silent failures - always check return values or request state.

