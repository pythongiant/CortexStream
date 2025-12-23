# CortexStream Configuration Examples

## ModelBackend Configuration

### Apple Silicon (MPS) - Default
```cpp
auto backend = std::make_shared<ModelBackend>(Device::MPS, DType::FP16);
backend->loadModel("llama2-7b.mlx");
backend->warmup();  // Warm GPU graph
```

### CPU Fallback
```cpp
auto backend = std::make_shared<ModelBackend>(Device::CPU, DType::FP32);
backend->loadModel("llama2-7b.mlx");
```

### Full Precision (Slower, More Accurate)
```cpp
auto backend = std::make_shared<ModelBackend>(Device::MPS, DType::FP32);
```

---

## KVCache Configuration

### 8GB Cache (Typical)
```cpp
auto cache = std::make_shared<KVCache>(
    8UL * 1024 * 1024 * 1024,  // 8GB
    4096,                        // hiddenSize
    32,                          // numLayers
    16                           // blockSize (tokens)
);
```

### Large Cache (16GB)
```cpp
auto cache = std::make_shared<KVCache>(
    16UL * 1024 * 1024 * 1024,
    4096, 32, 16
);
```

### Small Cache (4GB, Mobile)
```cpp
auto cache = std::make_shared<KVCache>(
    4UL * 1024 * 1024 * 1024,
    2048,  // Smaller model
    16,    // Fewer layers
    8      // Smaller blocks
);
```

### Very Large Model (Llama 70B)
```cpp
auto cache = std::make_shared<KVCache>(
    64UL * 1024 * 1024 * 1024,  // 64GB
    8192,                         // hiddenSize
    80,                           // numLayers
    32                            // blockSize
);
```

---

## Scheduler Configuration

### Standard (32 concurrent)
```cpp
auto scheduler = std::make_shared<Scheduler>(32);
```

### Conservative (8 requests)
```cpp
auto scheduler = std::make_shared<Scheduler>(8);
```

### Aggressive (128 requests)
```cpp
auto scheduler = std::make_shared<Scheduler>(128);
```

### Dynamic (Future)
```cpp
auto scheduler = std::make_shared<DynamicScheduler>(
    minBatch=4,
    maxBatch=128,
    targetLatency=100ms
);
```

---

## Sampling Presets

### Code Generation
```cpp
SamplingParams params;
params.temperature = 0.2f;
params.topK = 5;
params.topP = 0.95f;
params.repetitionPenaltyEnabled = true;
params.repetitionPenalty = 1.15f;
params.seed = -1;  // Random

sampler.setParams(params);
```

### Creative Writing
```cpp
SamplingParams params;
params.temperature = 0.9f;
params.topK = 50;
params.topP = 0.95f;
params.repetitionPenaltyEnabled = false;
params.seed = -1;

sampler.setParams(params);
```

### Chat / Dialogue
```cpp
SamplingParams params;
params.temperature = 0.7f;
params.topK = 40;
params.topP = 0.9f;
params.repetitionPenaltyEnabled = true;
params.repetitionPenalty = 1.1f;
params.seed = -1;

sampler.setParams(params);
```

### Translation
```cpp
SamplingParams params;
params.temperature = 0.5f;
params.topK = 20;
params.topP = 0.9f;
params.repetitionPenaltyEnabled = true;
params.repetitionPenalty = 1.2f;
params.seed = -1;

sampler.setParams(params);
```

### Deterministic / Reproducible
```cpp
SamplingParams params;
params.temperature = 0.3f;
params.topK = 1;   // Greedy
params.topP = 1.0f;
params.doSample = true;  // Force greedy
params.seed = 42;  // Fixed seed

sampler.setParams(params);
```

### Q&A / Knowledge Retrieval
```cpp
SamplingParams params;
params.temperature = 0.1f;
params.topK = 3;
params.topP = 0.8f;
params.repetitionPenaltyEnabled = true;
params.repetitionPenalty = 1.3f;
params.seed = -1;

sampler.setParams(params);
```

### Maximum Diversity
```cpp
SamplingParams params;
params.temperature = 1.3f;
params.topK = 100;
params.topP = 1.0f;
params.repetitionPenaltyEnabled = false;
params.seed = -1;

sampler.setParams(params);
```

---

## Complete System Configuration

### Production Setup
```cpp
// Model
auto backend = std::make_shared<ModelBackend>(Device::MPS, DType::FP16);
if (!backend->loadModel("models/llama2-7b.mlx")) {
    throw std::runtime_error("Model load failed");
}

// Memory
auto cache = std::make_shared<KVCache>(
    8UL * 1024 * 1024 * 1024,
    4096, 32, 16
);

// Scheduling
auto scheduler = std::make_shared<Scheduler>(32);

// Engine
auto engine = std::make_shared<InferenceEngine>(backend, scheduler, cache);
if (!engine->initialize()) {
    throw std::runtime_error("Engine init failed");
}

// Run
std::thread engineThread([&] { engine->run(); });

// Setup is complete - ready for requests
```

### Development/Testing Setup
```cpp
auto backend = std::make_shared<ModelBackend>(Device::CPU, DType::FP32);
backend->loadModel("models/tiny-test.mlx");

auto cache = std::make_shared<KVCache>(
    512UL * 1024 * 1024,  // 512MB
    256, 4, 8
);

auto scheduler = std::make_shared<Scheduler>(4);

auto engine = std::make_shared<InferenceEngine>(backend, scheduler, cache);
engine->initialize();

// Small setup for fast iteration
```

### Mobile Setup
```cpp
auto backend = std::make_shared<ModelBackend>(Device::MPS, DType::FP16);
backend->loadModel("models/llama2-3b-mobile.mlx");

auto cache = std::make_shared<KVCache>(
    2UL * 1024 * 1024 * 1024,  // 2GB
    2048, 16, 8
);

auto scheduler = std::make_shared<Scheduler>(8);

auto engine = std::make_shared<InferenceEngine>(backend, scheduler, cache);
engine->initialize();

// Low memory footprint
```

### High-Throughput Setup
```cpp
auto backend = std::make_shared<ModelBackend>(Device::MPS, DType::FP16);
backend->loadModel("models/llama2-7b.mlx");

auto cache = std::make_shared<KVCache>(
    32UL * 1024 * 1024 * 1024,  // 32GB
    4096, 32, 32
);

auto scheduler = std::make_shared<Scheduler>(128);

auto engine = std::make_shared<InferenceEngine>(backend, scheduler, cache);
engine->initialize();

// Maximize throughput
```

---

## Request Configuration Examples

### Simple Chat
```cpp
auto req = std::make_shared<Request>(
    "chat_001",
    tokenize("What is machine learning?"),
    256  // max tokens
);

SamplingParams params;
params.temperature = 0.7f;
params.topK = 40;
params.topP = 0.9f;
req->setSamplingParams(params);

scheduler->submitRequest(req);
```

### Code Generation (Long Context)
```cpp
auto req = std::make_shared<Request>(
    "code_001",
    tokenize("def fibonacci(n):"),
    512  // longer generation
);

SamplingParams params;
params.temperature = 0.2f;
params.topK = 5;
params.topP = 0.95f;
params.repetitionPenaltyEnabled = true;
params.repetitionPenalty = 1.15f;
req->setSamplingParams(params);

// Callback for real-time token streaming
req->setTokenCallback([](int token, bool finished) {
    if (!finished) {
        std::cout << tokenToString(token);
        std::cout.flush();
    }
});

scheduler->submitRequest(req);
```

### Batch of Requests
```cpp
std::vector<std::string> prompts = {
    "Hello, how are you?",
    "What is the capital of France?",
    "Write a poem about nature.",
    "Explain quantum computing."
};

for (size_t i = 0; i < prompts.size(); ++i) {
    auto req = std::make_shared<Request>(
        "batch_" + std::to_string(i),
        tokenize(prompts[i]),
        128
    );
    
    SamplingParams params;
    params.temperature = 0.7f;
    params.topK = 40;
    params.topP = 0.9f;
    req->setSamplingParams(params);
    
    scheduler->submitRequest(req);
}

// Engine will batch process them
```

### Deterministic Reproduction
```cpp
auto req = std::make_shared<Request>(
    "reproducible_001",
    tokenize("Translate to French: Hello world"),
    64
);

SamplingParams params;
params.temperature = 0.0f;
params.topK = 1;
params.doSample = true;  // Greedy
params.seed = 12345;     // FIXED seed
req->setSamplingParams(params);

scheduler->submitRequest(req);

// Same seed + same input = always same output
```

---

## Performance Tuning

### Maximize Throughput
```cpp
// Larger batches
auto scheduler = std::make_shared<Scheduler>(128);

// More memory for cache
auto cache = std::make_shared<KVCache>(
    32UL * 1024 * 1024 * 1024,  // 32GB
    4096, 32, 32
);

// Faster sampling
SamplingParams params;
params.topK = 1;  // Greedy
params.temperature = 1.0f;
params.doSample = true;
```

### Minimize Latency
```cpp
// Smaller batches
auto scheduler = std::make_shared<Scheduler>(8);

// Reduced context
auto cache = std::make_shared<KVCache>(
    4UL * 1024 * 1024 * 1024,
    2048, 16, 8
);

// Fast sampling
SamplingParams params;
params.topK = 1;
```

### Balance Quality & Speed
```cpp
auto scheduler = std::make_shared<Scheduler>(32);

auto cache = std::make_shared<KVCache>(
    8UL * 1024 * 1024 * 1024,
    4096, 32, 16
);

SamplingParams params;
params.temperature = 0.7f;
params.topK = 40;
params.topP = 0.9f;
```

---

## Debugging Configuration

### Verbose Logging
```cpp
// Enable detailed logs
SamplingParams params;
params.returnLogprobs = true;
params.returnMetadata = true;
```

### Deterministic Debug
```cpp
// All operations reproducible
SamplingParams params;
params.seed = 0;  // Fixed
params.temperature = 0.0f;
params.topK = 1;
params.doSample = true;

// All requests sequential
auto scheduler = std::make_shared<Scheduler>(1);
```

### Memory Profiling
```cpp
// Check cache stats
size_t used = cache->getTotalAllocated();
size_t free = cache->getTotalFree();
int blocks = cache->getNumAllocatedBlocks();

std::cout << "Cache: " << used / 1e9 << "GB used, " 
          << free / 1e9 << "GB free, "
          << blocks << " blocks" << std::endl;
```

---

## Summary: Quick Start Configurations

| Use Case | Backend | Batch | Cache | Temp | TopK | TopP |
|----------|---------|-------|-------|------|------|------|
| Chat | MPS/FP16 | 32 | 8GB | 0.7 | 40 | 0.9 |
| Code | MPS/FP16 | 16 | 8GB | 0.2 | 5 | 0.95 |
| Diverse | MPS/FP16 | 32 | 8GB | 1.2 | 100 | 1.0 |
| Deterministic | MPS/FP16 | 32 | 8GB | 0.0 | 1 | 1.0 |
| Mobile | MPS/FP16 | 4 | 2GB | 0.7 | 20 | 0.9 |
| Throughput | MPS/FP16 | 128 | 32GB | 0.0 | 1 | 1.0 |
| Testing | CPU/FP32 | 1 | 512MB | 0.7 | 40 | 0.9 |
