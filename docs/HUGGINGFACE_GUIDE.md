# HuggingFace Model Serving with CortexStream

Complete guide to serving HuggingFace models on Apple Silicon using CortexStream with MLX GPU acceleration.

## Quick Start

### 1. Load a Model

```cpp
#include "cortexstream/model.h"

auto backend = std::make_shared<ModelBackend>(Device::MPS, DType::FP16);

// Load any HuggingFace model - automatic download & conversion
bool success = backend->loadHuggingFaceModel(
    "mistralai/Mistral-7B",  // Model ID
    "./models"                // Cache directory
);
```

### 2. Run Inference

```cpp
auto engine = std::make_shared<InferenceEngine>(backend, scheduler, cache);
engine->initialize();
engine->run();
```

## Supported Models

All HuggingFace models are technically supported. Here are optimized presets for Apple Silicon:

### Recommended Models by Device

#### M1/M1 Pro (8GB unified memory)
- **microsoft/phi-2** (2.7B) - ⭐ Best choice
  - 5GB model size
  - Fast inference, good quality
  - Example: `./huggingface_inference "microsoft/phi-2"`

- **mistralai/Mistral-7B** (7B) - Works, limited batch
  - 14GB model size (requires external cache)
  - Larger context window (32k tokens)
  - Example: `./huggingface_inference "mistralai/Mistral-7B"`

#### M2/M2 Pro (16GB unified memory)
- **mistralai/Mistral-7B** - ⭐ Best choice
- **meta-llama/Llama-2-7b** (7B)
- **HuggingFaceH4/zephyr-7b-beta** (7B)

#### M3 Max (48GB+ unified memory)
- **mistralai/Mixtral-8x7B** (47B)
- **meta-llama/Llama-2-13b** (13B)
- **meta-llama/Llama-2-70b** (70B) - with quantization

## How It Works

### Pipeline

```
HuggingFace Model ID
    ↓
[1] Download from huggingface.co
    ↓
[2] Convert to MLX format (optimize for Metal)
    ↓
[3] Cache locally
    ↓
[4] Load into MLX GPU runtime
    ↓
[5] Serve via CortexStream
```

### First Run (One-time setup)

```
$ ./huggingface_inference "mistralai/Mistral-7B"
=== CortexStream HuggingFace Model Inference ===

[Model] Loading HuggingFace model: mistralai/Mistral-7B
[HF] Downloading model from huggingface.co/mistralai/Mistral-7B
[HF] ⬇ Downloading config.json... ✓
[HF] ⬇ Downloading model.safetensors... ✓  (~14GB, 5-10 min)
[HF] ⬇ Downloading tokenizer.json... ✓
[MLX] Converting HuggingFace weights to MLX format...
[MLX] ✓ Extracted model architecture:
      Hidden size: 4096
      Num layers: 32
      Vocab size: 32000
[Step 3] Loading MLX model into GPU...
[MLX] Loading weights from ./models/mistralai/Mistral-7B-mlx
✅ Model loaded successfully
   Model ID: mistralai/Mistral-7B
   Cache: ./models/mistralai/Mistral-7B-mlx
   Device: Metal (MPS)
   Dtype: FP16

[Setup] Initializing inference pipeline...
✅ Pipeline ready

[Inference] Processing requests...
GPU acceleration: Metal (MPS) on Apple Silicon
```

### Subsequent Runs (Cached - Fast!)

```
$ ./huggingface_inference "mistralai/Mistral-7B"
[Model] Loading HuggingFace model: mistralai/Mistral-7B
[HF] ✓ config.json (cached)
[HF] ✓ model.safetensors (cached)
[HF] ✓ tokenizer.json (cached)
[MLX] Using cached MLX model
✅ Model loaded successfully
   Model ID: mistralai/Mistral-7B
   [... loads in <1 second]
```

## Configuration

### Batch Size

Tune for your device:

```cpp
// M1/M1 Pro: smaller batches
auto scheduler = std::make_shared<Scheduler>(8);

// M2/M2 Pro: medium batches  
auto scheduler = std::make_shared<Scheduler>(32);

// M3 Max: larger batches
auto scheduler = std::make_shared<Scheduler>(64);
```

### Cache Size

```cpp
// M1: 8GB
size_t cacheSize = 8UL * 1024 * 1024 * 1024;

// M2: 16GB
size_t cacheSize = 16UL * 1024 * 1024 * 1024;

// M3 Max: 32GB
size_t cacheSize = 32UL * 1024 * 1024 * 1024;

auto cache = std::make_shared<KVCache>(
    cacheSize,
    backend->getHiddenSize(),
    backend->getNumLayers()
);
```

### Sampling Parameters

```cpp
SamplingParams params;

// Quality vs Speed tradeoff
params.temperature = 0.7f;  // 0.0 = deterministic, 1.0 = random
params.topP = 0.9f;         // nucleus sampling (0.9 = 90% probability mass)
params.topK = 40;           // restrict to top K tokens
params.doSample = true;     // enable sampling

// Prevent repetition
params.repetitionPenaltyEnabled = true;
params.repetitionPenalty = 1.1f;

request->setSamplingParams(params);
```

## Performance Tuning

### 1. Data Type Selection

```cpp
// FP16 (default) - 2x faster
auto backend = std::make_shared<ModelBackend>(Device::MPS, DType::FP16);

// FP32 - higher precision, slower
auto backend = std::make_shared<ModelBackend>(Device::MPS, DType::FP32);

// INT8 - quantized, fastest but lower quality
auto backend = std::make_shared<ModelBackend>(Device::MPS, DType::INT8);
```

### 2. Metal Optimization Flags

```cpp
backend->enableMetalOptimizations();   // Turn on all Metal optimizations
// or
backend->disableMetalOptimizations();  // Use CPU fallback (debug only)
```

### 3. Pre-compilation

```cpp
backend->warmup();       // Pre-compile Metal kernels for first token
backend->preloadGraph(); // Pre-compile computation graphs
```

## Troubleshooting

### Out of Memory

```
❌ Error: Allocation failed (KV cache full)
Solution: Reduce batch size or cache size
```

```cpp
auto scheduler = std::make_shared<Scheduler>(16);  // Reduce from 32
```

### Model Download Fails

```
❌ Error: Failed to download config.json
Solutions:
1. Check internet connection
2. Model might be private - need HuggingFace token
3. Disk space: 7B model needs ~15GB
```

Set HuggingFace token:
```bash
export HF_TOKEN="hf_xxxx"  # Your HF token
./huggingface_inference
```

### Slow Inference

```cpp
// Enable Metal optimizations
backend->enableMetalOptimizations();

// Use FP16 instead of FP32
auto backend = std::make_shared<ModelBackend>(Device::MPS, DType::FP16);

// Increase batch size for better GPU utilization
auto scheduler = std::make_shared<Scheduler>(64);
```

## Model Storage

Models are cached in:
```
./models/
├── mistralai/Mistral-7B/
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer.json
└── mistralai/Mistral-7B-mlx/
    └── weights.safetensors
```

Clear cache to free disk space:
```bash
rm -rf ./models
```

Re-download on next run (first run overhead applies).

## Cost Analysis

For **meta-llama/Llama-2-7b** on M2 Pro:

| Metric | Value |
|--------|-------|
| Download (first run) | 5-10 min |
| MLX Conversion | 5-10 min |
| Prefill (1000 tokens) | 2-3 sec |
| Decode (per token) | 30-50 ms |
| Throughput (batched) | 10-20 tokens/sec |
| Memory (7B model) | ~14GB disk, 4GB VRAM |
| Cost (amortized) | ≈$0 (local, no API calls) |

## API Reference

### loadHuggingFaceModel()

```cpp
bool loadHuggingFaceModel(
    const std::string& modelId,                    // "org/model"
    const std::string& cacheDir = "./models"      // Cache location
);
```

**Parameters:**
- `modelId`: HuggingFace model ID (format: "org/model-name")
- `cacheDir`: Directory for cached models (optional, default: "./models")

**Returns:** 
- `true` if model loaded successfully
- `false` if download or conversion failed

**Example:**
```cpp
backend->loadHuggingFaceModel("mistralai/Mistral-7B", "./my_models");
```

## Examples

### Simple Inference

See [examples/huggingface_inference.cpp](../examples/huggingface_inference.cpp)

```bash
./huggingface_inference "mistralai/Mistral-7B"
```

### Batch Processing

```cpp
std::vector<std::string> prompts = {
    "What is AI?",
    "Explain machine learning.",
    "Write Python code to sort a list."
};

for (const auto& prompt : prompts) {
    auto req = std::make_shared<Request>(
        "req_" + std::to_string(i++),
        prompt,
        256  // max tokens
    );
    scheduler->submitRequest(req);
}

engine->run();
```

### Streaming Responses

```cpp
// CortexStream supports token-by-token streaming
// Tokens are emitted as they're generated (not waiting for full completion)
// Ideal for interactive applications
```

## Limitations

- **Requires internet on first run** (for download)
- **Disk space:** ~2x model size (original + MLX converted)
- **Large models:** 70B+ models need M3 Max or external cache
- **Fine-tuned models:** Custom models from HuggingFace supported if compatible with MLX

## See Also

- [MLX Documentation](https://ml-explore.github.io/mlx/build/html/index.html)
- [HuggingFace Hub](https://huggingface.co)
- [CortexStream Architecture](../docs/architecture.md)
