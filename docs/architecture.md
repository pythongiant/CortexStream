# CortexStream Architecture

## Overview

CortexStream is a high-performance inference engine for large language models optimized for Apple Silicon (MLX backend). It's designed with a **brain-inspired architecture** where different components handle distinct responsibilities:

```
    InferenceEngine (Central Nervous System)
           ↓
    ┌──────┼──────────┐
    ↓      ↓          ↓
Scheduler KVCache   ModelBackend
  (Decision)  (Memory) (Motor Cortex)
```

## Core Components

### 1. **ModelBackend** - Hardware Execution Layer

**Role**: The "motor cortex" - executes forward passes on the GPU.

**Responsibilities**:
- Load and manage MLX models
- Enforce device placement (MPS for Apple Silicon)
- Implement `prefill()` and `decode()` operations
- Provide sampling strategies (greedy, top-k, top-p)
- Manage temporary buffers to avoid malloc churn

**Key Methods**:
```cpp
Tensor prefill(const Batch& batch, const std::vector<int>& tokenIds);
Tensor decode(const Batch& batch, const std::vector<int>& tokenIds);
int sampleToken(const Tensor& logits, const SamplingParams& params);
```

**Performance Guarantees**:
- Zero host↔device sync in hot paths
- Reused tensor buffers (no allocation)
- GPU graph caching
- Metal/MPS optimized

**Device Handling**:
- Targets Apple Silicon with MPS (Metal Performance Shaders)
- Falls back to CPU if needed
- FP16 preferred for speed and memory

---

### 2. **KVCache** - Memory Management

**Role**: The "hippocampus" - stores cached key-value states.

**Responsibilities**:
- Allocate fixed blocks for each request
- Track which blocks belong to which request
- Implement efficient block reuse
- Prevent memory fragmentation

**Key Methods**:
```cpp
int allocateBlock(const std::string& requestId);
void freeBlock(int blockId);
std::vector<int> getBlocksForRequest(const std::string& requestId);
void clearRequest(const std::string& requestId);
```

**Block Management**:
- Fixed block size (e.g., 16 tokens per block)
- Preallocated unified buffer
- O(1) block lookup and allocation

---

### 3. **Scheduler** - Decision Making

**Role**: The "prefrontal cortex" - decides request ordering and batching.

**Responsibilities**:
- Accept new requests from queue
- Build prefill batches (unprocessed prompts)
- Build decode batches (active token generation)
- Track request state (Pending → Prefilling → Decoding → Finished)
- Ensure fairness (no request starvation)

**Key Methods**:
```cpp
bool submitRequest(std::shared_ptr<Request> request);
bool hasWork() const;
Batch buildPrefillBatch();
Batch buildDecodeBatch();
void markRequestReady(const std::string& requestId);
```

**State Machine**:
```
Pending → Prefilling → Decoding → Finished
           │            ↓
           └→ Failed ←──┘
```

---

### 4. **InferenceEngine** - Orchestrator

**Role**: The "central nervous system" - coordinates everything.

**Responsibilities**:
- Run continuous batching loop
- Call Scheduler to get batches
- Call ModelBackend for forward passes
- Manage KVCache lifecycle
- Stream tokens to clients
- Handle failures gracefully

**Main Loop**:
```cpp
void mainLoop() {
    while (scheduler->hasWork()) {
        // 1. Accept new requests
        scheduler->acceptNewRequests();
        
        // 2. Process prefill batch
        Batch prefill = scheduler->buildPrefillBatch();
        if (!prefill.empty())
            backend->prefill(prefill, tokens);
        
        // 3. Process decode batch  
        Batch decode = scheduler->buildDecodeBatch();
        if (!decode.empty()) {
            auto logits = backend->decode(decode, tokens);
            emitTokens(decode, logits);
        }
        
        // 4. Cleanup finished requests
        cleanup();
    }
}
```

---

### 5. **Request** - Client Interface

Represents a single inference request.

**Fields**:
- `id`: Unique identifier
- `promptTokens`: Initial prompt (fixed)
- `generatedTokens`: Generated so far (growing)
- `samplingParams`: Temperature, top-k, top-p
- `state`: Current state (Pending, Prefilling, Decoding, etc.)

**State Transitions**:
```
Pending → Prefilling → Decoding → Finished
                ↘         ↓
                  → Failed
```

---

## Data Flow

### Prefill Phase
1. Client submits request with prompt tokens
2. Scheduler groups multiple requests into prefill batch
3. ModelBackend processes entire prompt → KV cache
4. Request transitions from Prefilling → Decoding

### Decode Phase
1. Scheduler groups active requests into decode batch
2. ModelBackend processes last token only, uses KV cache
3. Sample next token from logits
4. Append token to request
5. If max_tokens reached → Finished; else continue decoding

### Cleanup Phase
1. Remove finished requests from scheduler
2. Free KV cache blocks
3. Return blocks to allocator

---

## Performance Characteristics

### Prefill
- **Throughput-bound**: Process entire prompt in one batch
- **Memory-bound**: Store KV cache for all positions
- **Latency**: O(prompt_length) = single forward pass

### Decode  
- **Compute-bound**: Single token at a time, but for many requests
- **Memory-bound**: KV cache access
- **Latency**: O(1) per token (amortized)
- **Batch throughput**: Process 32 requests/batch

### Overall
- **Throughput**: Constrained by GPU utilization
- **Latency**: Prefill latency + (generated_length × decode_latency)
- **Memory**: Prefill size + (batch_size × max_generated_length × cache_per_token)

---

## Failure Handling

### Backend Failure
- Catch exception in InferenceEngine
- Mark request as Failed
- Log error and continue with other requests

### Out of Memory (OOM)
- Detect in cache allocator
- Kill oldest low-priority request
- Free KV blocks
- Retry current request

### Stuck Request
- Timeout after max_iterations
- Kill request gracefully
- Free resources
- Increment failed counter

### Stale Cache
- Validate block associations periodically
- Ensure no orphaned blocks
- Defragment if needed

---

## Threading Model

### MVP (Current)
- Single main inference thread
- Scheduler accessed via mutex
- RequestQueue may run on networking thread
- Safe enqueue of new requests

### Future
- Multiple inference threads (sharded batches)
- Parallel prefill + decode
- NUMA affinity
- Pipeline parallelism

---

## Tensor Format

```cpp
struct Tensor {
    std::vector<float> data;        // Flat data
    std::vector<int64_t> shape;     // Dimensions
    DType dtype;                    // FP32, FP16, INT8
};
```

### Shapes
- **Logits**: `[batchSize, vocabSize]`
- **KV cache block**: `[blockSize, hiddenSize, numLayers]` × 2 (K and V)
- **Hidden states**: `[seqLen, batchSize, hiddenSize]`

---

## Configuration

### ModelBackend
```cpp
ModelBackend backend(Device::MPS, DType::FP16);
backend.loadModel("path/to/model.mlx");
backend.warmup();  // GPU graph warmup
```

### KVCache
```cpp
KVCache cache(
    8UL * 1024 * 1024 * 1024,  // 8GB total
    4096,                        // hiddenSize
    32,                          // numLayers
    16                           // blockSize
);
```

### Scheduler
```cpp
Scheduler scheduler(32);  // max batch size
```

### InferenceEngine
```cpp
auto engine = std::make_shared<InferenceEngine>(
    backend, scheduler, cache
);
engine->initialize();
std::thread(/* engine->run() */).detach();
```

---

## Example: End-to-End Inference

```cpp
// 1. Setup
auto backend = std::make_shared<ModelBackend>();
backend->loadModel("model.mlx");

auto scheduler = std::make_shared<Scheduler>(32);
auto cache = std::make_shared<KVCache>(8GB, 4096, 32);
auto engine = std::make_shared<InferenceEngine>(
    backend, scheduler, cache
);
engine->initialize();

// 2. Submit request
std::vector<int> prompt = {101, 2054, 2003, ...};
auto req = std::make_shared<Request>("req_1", prompt, 256);
scheduler->submitRequest(req);

// 3. Run inference (threaded)
std::thread t([&engine] { engine->run(); });

// 4. Wait for completion
while (!req->isFinished()) {
    std::this_thread::sleep_for(10ms);
}

// 5. Retrieve result
auto output = req->getGeneratedTokens();
t.join();
```

---

## Key Design Decisions

1. **Stateless ModelBackend**: Simplifies correctness and error recovery
2. **Prefill + Decode split**: Leverages different computational patterns
3. **Block-based KV cache**: O(1) allocation, easy eviction
4. **Continuous batching**: High throughput for latency-sensitive workloads
5. **Callback model**: Streaming tokens without blocking main loop

---

## Future Optimizations

- [ ] Speculative decoding
- [ ] Multi-GPU batching
- [ ] Request priorities
- [ ] Dynamic batch sizing
- [ ] KV cache compression
- [ ] Flash-attention integration
- [ ] Paged attention
- [ ] Tensor parallelism

