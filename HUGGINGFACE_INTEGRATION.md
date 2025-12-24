# HuggingFace Integration Status

## Current Implementation Status

CortexStream has **partial HuggingFace support** through tokenizer integration and example code. Full model loading and auto-download are **not yet implemented**.

## What Is Currently Implemented

### 1. **HuggingFace Tokenizer Support**
`src/model/tokenizer.cpp` - Working implementation for:
- Loading tokenizers from `tokenizer.json` files
- Encoding text to tokens
- Decoding tokens back to text
- Factory function `createTokenizer()` for easy instantiation

### 2. **Example Application**
`examples/huggingface_inference.cpp` - Shows:
- How to load a model via `loadModel()` (expects pre-converted weights)
- Tokenizer loading from cache directory
- Pipeline setup with Scheduler, KVCache, InferenceEngine
- Batch inference with sampling parameters

### 3. **Core Infrastructure**
All components work with any model once loaded:
- `ModelBackend` - Loads MLX-format models
- `Scheduler` - Batches requests efficiently
- `KVCache` - GPU-accelerated cache with buddy allocator
- `InferenceEngine` - Orchestrates prefill/decode pipeline
- `Sampler` - GPU-accelerated sampling (Top-K, Top-P, temperature)

## What Is NOT Yet Implemented

| Feature | Status | Notes |
|---------|--------|-------|
| `loadHuggingFaceModel()` method | Not implemented | Requires manual model conversion |
| Auto-download from huggingface.co | Not implemented | Download models manually |
| Auto-convert to MLX format | Not implemented | Use external conversion tools |
| `isHuggingFaceModel()` helper | Not implemented | - |
| `docs/HUGGINGFACE_GUIDE.md` | Not created | - |
| `src/model/huggingface_loader.cpp` | Not created | - |

## Current Usage (Manual Setup Required)

### Step 1: Download Model Manually

```bash
# Using huggingface-cli
pip install huggingface_hub
huggingface-cli download mistralai/Mistral-7B --local-dir ./models/Mistral-7B

# Or download tokenizer.json directly from HuggingFace website
```

### Step 2: Convert to MLX Format (External Tool)

```bash
# Use mlx-lm or similar tool
pip install mlx-lm
python -m mlx_lm.convert --hf-path ./models/Mistral-7B --mlx-path ./models/Mistral-7B-mlx
```

### Step 3: Load in CortexStream

```cpp
#include "cortexstream/model.h"

auto backend = std::make_shared<ModelBackend>(Device::MPS, DType::FP16);

// Load pre-converted MLX model
backend->loadModel("./models/Mistral-7B-mlx");

// Use tokenizer separately
auto tokenizer = createTokenizer("./models/Mistral-7B/tokenizer.json");
```

## Architecture

```
Manual Download (HuggingFace)
    |
    v
External Conversion (mlx-lm)
    |
    v
ModelBackend::loadModel()  <-- Current entry point
    |
    +-> Scheduler (batches requests)
    |
    +-> KVCache (GPU arena allocation)
    |
    +-> InferenceEngine (orchestrates)
        |
        +-> Prefill (encode prompt)
        +-> Decode (generate tokens)
        +-> Sampling (GPU-accelerated)
```

## Files

```
Implemented:
  - src/model/tokenizer.cpp (137 lines - HuggingFace tokenizer support)
  - examples/huggingface_inference.cpp (313 lines - example usage)

Not Yet Implemented:
  - src/model/huggingface_loader.cpp (planned)
  - docs/HUGGINGFACE_GUIDE.md (planned)
  - HuggingFace methods in model.h (planned)
```

## Roadmap

To complete HuggingFace integration:

1. **Model Loader** (`src/model/huggingface_loader.cpp`)
   - Detect HF model ID format
   - Download from huggingface.co via API
   - Integrate MLX conversion

2. **ModelBackend Enhancement**
   - Add `loadHuggingFaceModel(modelId, cacheDir)` method
   - Add `isHuggingFaceModel()` helper
   - Progress tracking during download/conversion

3. **Documentation**
   - Create `docs/HUGGINGFACE_GUIDE.md`
   - Add quick start examples
   - Model compatibility list

## Running the Example

```bash
# Build
cmake -B build && cd build && make -j$(nproc)

# Run (expects pre-converted model in ./models/)
./examples/huggingface_inference "./models/Mistral-7B-mlx" "./models"
```

## Model Requirements

Models must be:
1. Downloaded manually from HuggingFace
2. Converted to MLX format externally
3. Placed in an accessible directory

The `loadModel()` function expects a path to MLX-format weights.
