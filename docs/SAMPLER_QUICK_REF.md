# Sampler Quick Reference

## Setup

```cpp
#include "cortexstream/sampler.h"

// Create sampler
Sampler sampler;

// Configure
SamplingParams params;
params.temperature = 0.8f;
params.topK = 40;
params.topP = 0.9f;
params.seed = 12345;  // For determinism

sampler.setParams(params);
```

## Common Patterns

### Greedy (Deterministic)
```cpp
SamplingParams params;
params.doSample = true;  // Force greedy
sampler.setParams(params);

int token = sampler.sampleToken(logits);
// Always returns argmax, no randomness
```

### Top-K (Creative)
```cpp
params.temperature = 1.0f;
params.topK = 40;      // Sample from 40 best
params.topP = 1.0f;    // Disable nucleus
params.doSample = false;

sampler.setParams(params);
int token = sampler.sampleToken(logits);
```

### Nucleus/Top-P (Balanced)
```cpp
params.temperature = 0.9f;
params.topK = 1;       // Disable top-K
params.topP = 0.95f;   // Cover 95% probability
params.doSample = false;

sampler.setParams(params);
int token = sampler.sampleToken(logits);
```

### Conservative (Safe)
```cpp
params.temperature = 0.3f;   // Sharp
params.topK = 10;            // Constrained
params.topP = 0.8f;          // Nucleus
params.doSample = false;

sampler.setParams(params);
int token = sampler.sampleToken(logits);
```

### Diverse (Creative)
```cpp
params.temperature = 1.2f;   // Flat
params.topK = 100;           // Loose
params.topP = 1.0f;          // Disabled
params.doSample = false;

sampler.setParams(params);
int token = sampler.sampleToken(logits);
```

## With Repetition Penalty

```cpp
params.repetitionPenaltyEnabled = true;
params.repetitionPenalty = 1.2f;  // Penalize repeats

sampler.setParams(params);

std::vector<int> history = {100, 101, 102, 100};  // Token 100 repeated
int token = sampler.sampleToken(logits, history);
// Less likely to sample 100 again
```

## Batch Sampling

```cpp
// Logits: [batchSize, vocabSize]
Tensor batchLogits = backend->decode(batch, tokens);

// Optional: histories per request
std::vector<std::vector<int>> histories;
for (auto& req : batch.requests) {
    histories.push_back(req->getGeneratedTokens());
}

// Sample all
std::vector<int> tokens = sampler.sampleBatch(batchLogits, histories);
```

## Deterministic Runs

```cpp
// Always get same tokens
sampler.setSeed(42);
int t1 = sampler.sampleToken(logits);

sampler.setSeed(42);
int t2 = sampler.sampleToken(logits);

assert(t1 == t2);  // ✅ Guaranteed
```

## Parameter Presets

### For Code Generation
```cpp
params.temperature = 0.2f;   // Sharp, deterministic
params.topK = 5;
params.topP = 0.95f;
params.repetitionPenaltyEnabled = true;
params.repetitionPenalty = 1.15f;
```

### For Creative Writing
```cpp
params.temperature = 0.9f;   // Balanced
params.topK = 50;
params.topP = 0.95f;
params.repetitionPenaltyEnabled = false;
```

### For Chat
```cpp
params.temperature = 0.7f;   // Slightly creative
params.topK = 40;
params.topP = 0.9f;
params.repetitionPenaltyEnabled = true;
params.repetitionPenalty = 1.1f;
```

### For Translation
```cpp
params.temperature = 0.5f;   // Conservative
params.topK = 20;
params.topP = 0.9f;
params.repetitionPenaltyEnabled = true;
params.repetitionPenalty = 1.2f;
```

## Validation

```cpp
SamplingParams params;
// ... set values ...

if (!params.validate()) {
    std::cerr << "Invalid sampling params" << std::endl;
    return;
}

sampler.setParams(params);  // Safe to use
```

## Troubleshooting

### Too repetitive
```cpp
// Increase repetition penalty
params.repetitionPenaltyEnabled = true;
params.repetitionPenalty = 1.3f;  // Was 1.1f
```

### Too random
```cpp
// Lower temperature, constrain top-K
params.temperature = 0.5f;      // Was 1.0f
params.topK = 20;               // Was 40
```

### Token distribution looks off
```cpp
// Check parameters validate
if (!params.validate()) {
    std::cerr << "Bad params" << std::endl;
}

// Check logits are reasonable
float minLogit = *min_element(logits.data.begin(), logits.data.end());
float maxLogit = *max_element(logits.data.begin(), logits.data.end());
std::cout << "Logit range: [" << minLogit << ", " << maxLogit << "]" << std::endl;
```

## In the Pipeline

```cpp
// 1. Get logits from backend
Tensor logits = backend->decode(batch, tokens);

// 2. Sample with sampler
int nextToken = sampler.sampleToken(logits, request->getGeneratedTokens());

// 3. Add to request
request->addToken(nextToken);

// 4. Continue until done
if (request->getGeneratedLength() >= request->getMaxTokens()) {
    scheduler->markRequestFinished(request->getId());
}
```

## Advanced: Custom Strategies

For future use:

```cpp
// Phase 2: Custom sampling hooks
class CustomSampler : public Sampler {
    int sampleToken(const Tensor& logits) override {
        // Your logic here
        return super::sampleToken(logits);
    }
};
```
