# Sampler Design & Implementation Guide

## Overview

The Sampler is CortexStream's production-grade token sampling engine. It converts model logits into discrete token indices using various decoding strategies (greedy, top-k, top-p, and combinations).

## Design Philosophy

### Core Principle: "Logits → Token"

The sampler:
- ✅ Operates on device tensors (MLX compatible)
- ✅ Avoids host↔GPU stalls
- ✅ Supports deterministic + stochastic generation
- ✅ Is pluggable (runtime switchable strategies)
- ✅ Handles numerical stability
- ✅ Returns metadata for diagnostics (future)

### What the Sampler Does NOT Do

- ❌ Know about scheduling
- ❌ Know about KV cache
- ❌ Know about networking
- ❌ Modify model state
- ❌ Block on GPU operations

**It's pure: logits in → token index out.**

---

## Data Structures

### SamplingParams

Complete configuration matching TensorRT-LLM design:

```cpp
struct SamplingParams {
    // Core strategies
    float temperature = 1.0f;     // Scaling factor (1=unchanged, <1=sharper)
    int topK = 1;                 // 1=greedy, >1=sample from K highest
    float topP = 1.0f;            // 1.0=disabled, <1=nucleus sampling
    bool doSample = false;        // Override to greedy

    // Optional penalties
    bool repetitionPenaltyEnabled = false;
    float repetitionPenalty = 1.1f;

    // Determinism
    int seed = -1;                // -1=random, >=0=fixed seed

    // Diagnostics
    bool returnLogprobs = false;  // Future: return log-probabilities
    bool returnMetadata = false;  // Future: return full metadata

    bool validate() const;        // Check parameter validity
};
```

### SamplingMetadata (Optional)

Diagnostic information for analysis:

```cpp
struct SamplingMetadata {
    float chosenProb;              // P(selected token)
    float entropy;                 // Shannon entropy of distribution
    std::vector<int> topTokens;    // Top-K candidate indices
    std::vector<float> topProbs;   // Their probabilities
    int numFiltered;               // Tokens filtered by nucleus/top-k
};
```

---

## Sampler Class API

### Constructor & Configuration

#### `Sampler()`
Initializes with default params (greedy).

#### `void setParams(const SamplingParams& params)`
Updates sampling configuration.
- **Throws**: `std::invalid_argument` if params invalid
- **Side Effect**: Reinitializes RNG with new seed

#### `const SamplingParams& getParams() const`
Returns current parameters.

#### `void setSeed(int seed)`
Updates RNG seed for determinism.

### Core Sampling API

#### `int sampleToken(const Tensor& logits, const std::vector<int>& generatedHistory = {})`
Main entry point: converts logits to token index.

**Input Parameters**:
- `logits`: Tensor of shape [1, vocabSize] or [batchSize, vocabSize]
- `generatedHistory`: Previous tokens (for repetition penalty)

**Algorithm**:
1. Apply repetition penalty (if enabled)
2. Check greedy override
3. Apply temperature scaling
4. Route to strategy:
   - Both topK and topP → combined
   - Only topK → top-K sampling
   - Only topP → nucleus sampling
   - Neither → greedy (argmax)

**Returns**: Token index (0 to vocabSize-1)

**Example**:
```cpp
Sampler sampler;

SamplingParams params;
params.temperature = 0.8f;
params.topK = 40;
params.topP = 0.95f;
sampler.setParams(params);

int token = sampler.sampleToken(logits, prevTokens);
```

#### `std::vector<int> sampleBatch(const Tensor& batchedLogits, const std::vector<std::vector<int>>& histories = {})`
Batch sampling (MVP: sequential, future: vectorized).

**Input**:
- `batchedLogits`: Shape [batchSize, vocabSize]
- `histories`: One per batch element

**Output**: Vector of sampled tokens

**Note**: Future optimization will use GPU tensor ops for parallelism.

### Diagnostics

#### `std::optional<SamplingMetadata> getLastMetadata() const`
Returns metadata from last sampling operation (if enabled).

---

## Sampling Strategies

### 1. Greedy (Fastest)

**When Used**:
- `doSample = true` (forced)
- `topK == 1 && topP >= 1.0f` (default)
- Temperature = 1.0

**Algorithm**:
```
return argmax(logits)
```

**Characteristics**:
- ✅ Deterministic
- ✅ Lowest latency
- ❌ No diversity

**Use Case**: Image captioning, translation, constrained generation.

### 2. Temperature Scaling

**When Applied**: Before any probabilistic sampling

**Formula**:
```
scaled_logits = logits / temperature
```

**Effects**:
- `temperature < 1`: Sharper distribution (more likely to choose high logits)
- `temperature = 1`: No change
- `temperature > 1`: Flatter distribution (more uniform)

**Example**:
```
Original:  [10, 5, 2] → argmax = 0
Temp=0.5:  [20, 10, 4] → more concentrated
Temp=2.0:  [5, 2.5, 1] → more spread
```

**Implementation Detail**: Applied before softmax to maintain numerical stability.

### 3. Top-K Sampling

**When Used**: `topK > 1`

**Algorithm**:
```
1. Sort logits descending
2. Keep only top-K
3. Softmax (normalized)
4. Sample categorical
```

**Typical Values**:
- `topK = 40`: Reasonable diversity
- `topK = 10`: More constrained
- `topK = 1`: Same as greedy

**Example**:
```
Vocab: 50000 tokens
topK = 40: Keep 40 highest, mask rest
Softmax over 40: normalize probabilities
Sample from 40-token distribution
```

### 4. Top-P (Nucleus) Sampling

**When Used**: `topP < 1.0f`

**Algorithm**:
```
1. Softmax logits → probabilities
2. Sort descending
3. Find smallest set S where P(S) ≥ p
4. Mask others
5. Renormalize
6. Sample categorical
```

**Typical Values**:
- `topP = 0.9`: Covers ~90% probability mass
- `topP = 0.95`: ~95% (looser)
- `topP = 1.0`: Disabled (all tokens)

**Example**:
```
Probs: [0.4, 0.3, 0.15, 0.1, 0.05, ...]
topP = 0.9:
  Cumsum: [0.4, 0.7, 0.85, 0.95, ...] 
  First 4 tokens cover 95% (exceeds 90%)
  Keep: [0.4, 0.3, 0.15, 0.1]
  Renormalize
  Sample
```

**Advantages**:
- ✅ Adapts to model confidence
- ✅ Avoids tail tokens
- ✅ Works regardless of vocab size

### 5. Top-K + Top-P (Combined)

**When Used**: Both `topK > 1` AND `topP < 1.0f`

**Algorithm**:
```
1. Keep top-K tokens
2. Among top-K, apply nucleus threshold
3. Renormalize
4. Sample
```

**Use Case**: Very tight control over token space.

### 6. Repetition Penalty (Optional)

**When Enabled**: `repetitionPenaltyEnabled = true`

**Algorithm**:
```
For each token in history:
  if logit > 0:
    logit = logit / penalty
  else:
    logit = logit * penalty
```

**Effect**:
- Reduces probability of repeated tokens
- Prevents infinite loops
- Adjusts based on frequency (future)

**Example**:
```
History: [100, 100, 101, 102]
penalty = 1.2

Token 100 appeared 2x:
  logits[100] = 5.0 / 1.2 = 4.17

Token 101 appeared 1x:
  logits[101] = 3.0 / 1.2 = 2.5
```

---

## Numerical Stability

### Challenge
Softmax with large logits overflows:
```cpp
exp(1000) → ∞  (overflow)
```

### Solution: Stable Softmax

**Step 1: Subtract maximum**
```cpp
float maxLogit = max(logits);
for (auto& x : logits) x -= maxLogit;  // Now range ≈ [-∞, 0]
```

**Step 2: Compute exp (now safe)**
```cpp
for (auto& x : logits) x = exp(x);  // All ≤ 1.0
```

**Step 3: Normalize**
```cpp
float sum = 0;
for (auto x : logits) sum += x;
for (auto& x : logits) x /= sum;
```

### Implementation

```cpp
void Sampler::safeSoftmax(std::vector<float>& logits, float temperature) {
    // Subtract max for stability
    float maxLogit = *max_element(logits.begin(), logits.end());
    
    // Exp with clipping
    float sum = 0.0f;
    for (auto& val : logits) {
        float scaled = (val - maxLogit) / temperature;
        val = exp(clamp(scaled, MIN_LOGIT, MAX_LOGIT));
        sum += val;
    }
    
    // Normalize
    if (sum > 0.0f) {
        for (auto& val : logits) val /= sum;
    }
}
```

### Boundary Conditions

Handled carefully:
- ✅ Empty logits → return 0
- ✅ All zeros → uniform distribution
- ✅ NaN/Inf → fallback to greedy
- ✅ Single element → return 0

---

## MLX Integration

### Current Approach (MVP)

**Convert to CPU, sample, return token**:
```cpp
// MLX tensor on GPU
mlx::core::array logits_gpu = backend->decode(...);

// Convert to CPU for sampling (minimal overhead)
std::vector<float> logits_cpu = mlx::core::to_vector(logits_gpu);

// Sample on CPU
Tensor logits_tensor{logits_cpu, shape};
int token = sampler.sampleToken(logits_tensor);
```

**Pros**:
- ✅ Simple
- ✅ Correct
- ✅ Good enough for MVP

**Cons**:
- ❌ One GPU→CPU sync per token
- ❌ Not optimal for batch sampling

### Future: On-Device Sampling

**Fused GPU kernel**:
```cpp
// Future MLX integration
mlx::core::array logits;
mlx::core::array logits_scaled = logits / temperature;
mlx::core::array probs = mlx::core::softmax(logits_scaled);
mlx::core::array filtered = mlx::core::topk(probs, k);
int token = mlx::core::categorical(filtered);  // GPU sampling
```

**Benefits**:
- ✅ Zero host sync
- ✅ Kernel fusion
- ✅ Batch parallelism

---

## Determinism & Reproducibility

### Requirement

Same seed → same tokens:
```cpp
sampler.setSeed(42);
int t1 = sampler.sampleToken(logits);

sampler.setSeed(42);
int t2 = sampler.sampleToken(logits);

assert(t1 == t2);  // ✅ Always true
```

### Implementation

```cpp
void Sampler::initRNG() {
    if (params.seed >= 0) {
        rng.seed(params.seed);  // Fixed seed
    } else {
        std::random_device rd;
        rng.seed(rd());         // Random seed
    }
}

int Sampler::categoricalSample(const std::vector<float>& probs) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float rand = dist(rng);     // Uses seeded RNG
    
    float cumProb = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        cumProb += probs[i];
        if (rand < cumProb) return i;
    }
    return probs.size() - 1;
}
```

### Thread Safety

**MVP**: Single-threaded per sampler instance.

**Future**: Thread-local RNG for parallel batch sampling.

---

## Performance Profile

| Strategy | Time | Deterministic |
|----------|------|---------------|
| Greedy | O(V) | ✅ Yes |
| Top-K | O(V log K) | ✅ Yes (if seed fixed) |
| Top-P | O(V log V) | ✅ Yes (if seed fixed) |
| Top-K+P | O(V log K) | ✅ Yes |

Where V = vocabulary size (~100k for modern LLMs).

**Optimization Opportunities**:
- GPU-accelerated top-k via heap
- Batch processing parallelization
- Caching sorted indices

---

## Examples

### Basic Greedy

```cpp
Sampler sampler;
SamplingParams params;
params.doSample = true;  // Force greedy
sampler.setParams(params);

int token = sampler.sampleToken(logits);
```

### Temperature Scaling

```cpp
params.temperature = 0.7f;  // Sharper
params.temperature = 1.2f;  // Softer
sampler.setParams(params);
```

### Top-K Sampling

```cpp
params.topK = 40;
params.doSample = false;  // Use sampling
sampler.setParams(params);
```

### Nucleus (Top-P)

```cpp
params.topP = 0.9f;
params.topK = 1;  // Disable top-K
sampler.setParams(params);
```

### Deterministic Run

```cpp
params.seed = 12345;
sampler.setParams(params);
// Same seed → same tokens every run
```

### Repetition Penalty

```cpp
params.repetitionPenaltyEnabled = true;
params.repetitionPenalty = 1.2f;
sampler.setParams(params);

std::vector<int> history = {100, 101, 102};
int token = sampler.sampleToken(logits, history);
```

---

## Testing Strategy

### Unit Tests

1. **Basic Functionality**:
   - Greedy on uniform logits → token 0
   - Argmax property: highest logit wins
   
2. **Stability**:
   - Large logits (1e6) → no overflow
   - Small logits (1e-6) → no underflow
   - NaN handling → fallback to greedy

3. **Probability Properties**:
   - Softmax sums to 1.0
   - Top-K filters correctly
   - Top-P cumsum correct

4. **Determinism**:
   - Same seed → same token
   - RNG state properly initialized

5. **Edge Cases**:
   - Empty tensor → return 0
   - Single token → return 0
   - All equal logits → uniform behavior

### Integration Tests

1. With ModelBackend:
   - Decode → sample → stream pipeline
   
2. With InferenceEngine:
   - Multi-request batching
   - Token streaming

3. Performance:
   - Latency: <1ms per token sampling
   - Throughput: 1000s tokens/sec

---

## Future Enhancements

### Phase 2
- [ ] Logprobs output (`returnLogprobs`)
- [ ] Metadata diagnostics (`returnMetadata`)
- [ ] Batch GPU sampling

### Phase 3
- [ ] Contrastive decoding
- [ ] Min-penalty implementation
- [ ] Frequency/presence penalties

### Phase 4
- [ ] Mixture-of-samplers (fallback chains)
- [ ] Speculative sampling hooks
- [ ] Custom sampling callbacks

---

## References

- **Nucleus Sampling**: "The Curious Case of Neural Text Degeneration" (Holtzman et al., 2019)
- **Top-K**: "Diverse Beam Search" (Li et al., 2016)
- **Temperature**: Gal & Ghahramani, "Dropout as Bayesian Approximation"
- **TensorRT-LLM**: https://github.com/NVIDIA/TensorRT-LLM (design reference)
