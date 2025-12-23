# Request/Response Contract Design

## Overview

The Request/Response contract defines the user-facing API for CortexStream's inference engine. It's designed for:
- **Streaming** inference with incremental token delivery
- **Scheduling** with priority and prefill hints
- **KV cache** integration and control
- **Error handling** without exceptions
- **Performance** measurement with nanosecond precision

This design is adapted from TensorRT-LLM semantics but optimized for C++ and MLX backends.

---

## Request: Client → Engine

### Purpose
Encapsulate everything an inference client needs to specify for a generation task.

### Design Principles
1. **Immutable after submission** — allows safe sharing between Scheduler and Engine
2. **Lightweight** — minimal memory footprint for batching
3. **Scheduler-friendly** — carries metadata for smart scheduling decisions
4. **Cache-aware** — controls KV cache allocation and reuse
5. **Thread-safe** — except for cancellation (atomic<bool>)

### Core Construction
```cpp
Request req("req-123",                    // Unique ID
           "What is AI?",                 // Text prompt
           {1523, 373, 2154},             // Pre-tokenized input
           256);                          // Max output tokens
```

### Input Immutability (Read-Only)

| Method | Type | Purpose |
|--------|------|---------|
| `getId()` | string | Unique request identifier |
| `getPrompt()` | string | Original prompt text |
| `getInputTokens()` | vector<int> | Pre-tokenized prompt |
| `getInputTokenCount()` | int | Prompt length |
| `getMaxTokens()` | int | Requested output tokens |
| `getMinTokens()` | int | Minimum output tokens |

These fields are set at construction and never change. They represent the client's input specification.

### Configuration (Mutable)

#### Sampling Parameters
```cpp
SamplingParams params;
params.temperature = 0.7f;      // [0.0, 2.0] - lower = more deterministic
params.topK = 40;               // [1, 100] - keep top-K tokens
params.topP = 0.9f;             // (0.0, 1.0] - nucleus sampling
params.seed = 42;               // 0 = random
params.repetitionPenalty = 1.2f;// [0.0, 2.0] - discourage repetition

// Validation
if (params.isValid()) {
    req.setSamplingParams(params);
}
```

**Validation rules:**
- Temperature: must be in [0.0, 2.0]
- topK: must be in [1, 100]
- topP: must be in (0.0, 1.0]
- repetitionPenalty: must be in [0.0, 2.0]

Throws `std::invalid_argument` if invalid.

#### Stop Conditions
```cpp
// Stop on specific tokens
req.setStopTokens({2, 0});      // E.g., EOS=2, PAD=0

// Stop on substring (e.g., after first newline)
req.setStopString("\n");
```

### Scheduling Controls

Used by the Scheduler to make smarter batching decisions:

```cpp
// Priority for SLA compliance
req.setPriority(5);             // Higher = process faster

// Hint: can process with prefill only (no decode)
req.setAllowPrefillOnly(true);  // For length predictions

// Verbosity
req.setStreaming(true);         // Enable token-by-token streaming
```

### KV Cache Controls

```cpp
// Reuse cache from previous request (e.g., continuation)
req.setReuseCache(true);        // Avoids prefill recomputation

// Free cache immediately after completion
req.setFreeCacheOnFinish(true); // Memory-constrained systems
```

### Streaming

```cpp
// Define token callback
auto callback = [](int token, bool finished) {
    std::cout << "Token: " << token << ", done: " << finished << std::endl;
};
req.setTokenCallback(callback);

// Enable streaming mode
req.setStreaming(true);

// Engine calls: req.notifyToken(token, finished)
// Client callback executes immediately
```

### Metadata & Timing

```cpp
// When request arrived (nanoseconds since epoch)
uint64_t arrivalNs = req.getArrivalTimestampNs();

// Convert to wall-clock time
auto wallTime = req.getArrivalTime();  // std::chrono::system_clock::time_point

// For latency measurement
// Used by Engine to compute end-to-end latency
```

### Runtime State (Engine-Facing)

```cpp
// Accumulate generated tokens during generation
req.addGeneratedToken(token_id);            // Called by Engine
const auto& generated = req.getGeneratedTokens();
int count = req.getGeneratedTokenCount();   // Current output length

// Query completion state (engine status)
bool done = req.isFinished();               // All tokens generated
bool err = req.isFailed();                  // Generation error
const auto& msg = req.getErrorMessage();    // Error details
```

### Cancellation (Thread-Safe)

```cpp
// Client cancels mid-generation
req.cancel();                           // Thread-safe atomic store

// Engine checks periodically
if (req.isCancelled()) {
    // Early exit, cleanup, notify client
}
```

Uses `std::atomic<bool>` with `acquire`/`release` semantics for cross-thread safety.

---

## Response: Engine → Client

### Purpose
Accumulate and expose the complete result of an inference session, supporting both streaming updates and final summary.

### Construction
```cpp
Response resp("req-123");   // Link back to Request
```

### Generated Output

#### Text Accumulation
```cpp
resp.setText("Once upon a time");     // Set or replace
resp.appendText(" in a land far away");

std::string full = resp.getText();     // Full accumulated text
```

#### Token List
```cpp
resp.addToken(1234);                   // Called by Engine for each token
resp.setTokens({1234, 5678, 9012});    // Or set all at once

const auto& tokens = resp.getTokens();
int outputCount = resp.getOutputTokenCount();  // Length of generation
```

### Debug Information (Optional)

```cpp
// Log probability of each token
std::vector<float> logprobs = {-2.3, -1.5, -3.1};
resp.setLogprobs(logprobs);

// Top-K alternatives for each position
std::vector<std::pair<int, float>> topkAtPos0 = {
    {1234, -2.3},   // Best choice
    {5678, -4.1},   // 2nd best
};
std::vector<std::pair<int, float>> topkAtPos1 = {
    {9012, -1.5},
    {3456, -3.2},
};
resp.addTopKForToken(topkAtPos0);
resp.addTopKForToken(topkAtPos1);

const auto& topk = resp.getTopKLogprobs();  // Access all
```

### Completion Status

```cpp
// Mark generation as complete
resp.finish();              // Sets finished=true, records end time

bool done = resp.isFinished();

// Check multiple completion reasons
if (resp.hasStoppedByEOS()) {
    // Model generated EOS token
} else if (resp.hasStoppedByMaxTokens()) {
    // Hit maximum length limit
} else if (resp.hasStoppedByStopString()) {
    // Generated stop substring
} else if (resp.hasStoppedByUser()) {
    // Client cancelled
} else if (resp.hasError()) {
    // Generation failed
}

// Get human-readable reason
std::string reason = resp.getCompletionReason();
// Returns: "end_of_sequence", "max_tokens", "stop_string", 
//          "stop_token", "user_cancelled", "error", "unknown"
```

### Error Handling

```cpp
// Engine encounters an error
resp.setError("CUDA out of memory");

// Check error state
if (resp.hasError()) {
    std::cerr << "Error: " << resp.getErrorMessage() << std::endl;
}

// Completion reason reflects error
assert(resp.getCompletionReason() == "error");
```

Non-throwing error design enables:
- Partial results even on error
- Graceful degradation
- Client control over error recovery

### Token Counts

```cpp
// Input length (passed from Request)
resp.setInputTokenCount(req.getInputTokenCount());
int inputLen = resp.getInputTokenCount();

// Output length (computed automatically)
int outputLen = resp.getOutputTokenCount();  // tokens_.size()
```

### Timing & Latency

All timestamps are in **nanoseconds since epoch** for precision measurement:

```cpp
// Start time (set at construction)
uint64_t startNs = resp.getStartTimeNs();

// End time (set on finish())
uint64_t endNs = resp.getEndTimeNs();

// Elapsed time
uint64_t latencyNs = resp.getLatencyNs();

// Convert to human-friendly units
double latencyMs = resp.getLatencyMs();     // milliseconds
double latencySec = resp.getLatencySec();   // seconds

// Performance metrics
double tps = resp.getTokensPerSecond();     // Throughput
double avgLatency = resp.getAverageTokenLatencyMs();  // Per-token latency
```

### Streaming Integration

For streaming responses, use `ResponseChunk`:

```cpp
struct ResponseChunk {
    std::string requestId;   // req-123
    int token;              // Token ID
    std::string textPiece;  // Decoded text
    bool finished;          // Is sequence complete
};

// Engine produces
ResponseChunk chunk("req-123", 1234, "Once", false);

// Stream to client immediately (low latency)
// Accumulate in Response for final result
```

### Summary

```cpp
// Human-readable summary
std::string summary = resp.toString();
// Response(requestId=req-123, tokens=42, finished=true, 
//          latencyMs=523.5, reason=end_of_sequence)
```

---

## Workflow: Request → Engine → Response

### Synchronous Generation

```cpp
Request req("id", "prompt", {tokens}, 256);
req.setSamplingParams(params);
req.setMaxTokens(256);

// Submit to engine
auto resp = engine.generate(req);  // Blocks until done

// Query result
std::cout << "Output: " << resp.getText() << std::endl;
std::cout << "Latency: " << resp.getLatencyMs() << "ms" << std::endl;
```

### Streaming Generation

```cpp
Request req("id", "prompt", {tokens}, 256);
req.setStreaming(true);

// Define callback for each token
req.setTokenCallback([](int token, bool finished) {
    std::cout << token << " ";
    if (finished) std::cout << "[DONE]" << std::endl;
});

// Submit
auto resp = engine.generate(req);  // Returns quickly

// Access final result
std::cout << "Full text: " << resp.getText() << std::endl;
std::cout << "Tokens: " << resp.getOutputTokenCount() << std::endl;
```

### Async Generation

```cpp
Request req("id", "prompt", {tokens}, 256);
req.setStreaming(true);
req.setTokenCallback(onToken);

// Submit asynchronously
auto future = engine.generateAsync(req);

// Check progress while doing other work
while (!req.isFinished()) {
    std::cout << "Generated so far: " << req.getGeneratedTokenCount() << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

// Get final response
auto resp = future.get();
std::cout << "Done! Latency: " << resp.getLatencyMs() << "ms" << std::endl;
```

### Early Cancellation

```cpp
Request req("id", "prompt", {tokens}, 256);
auto future = engine.generateAsync(req);

std::this_thread::sleep_for(std::chrono::milliseconds(500));
req.cancel();  // Stop mid-generation

auto resp = future.get();
std::cout << "Stopped by: " << resp.getCompletionReason() << std::endl;
assert(resp.hasStoppedByUser());
```

---

## Integration Points

### Scheduler
```cpp
// Scheduler reads these
int priority = req.getPriority();
int maxTokens = req.getMaxTokens();
bool canSkipDecode = req.allowsPrefillOnly();
bool streaming = req.isStreamingEnabled();

// Decides:
// - When to schedule request
// - How to batch with others
// - When to prefill, when to decode
```

### Engine
```cpp
// Engine reads
const auto& tokens = req.getInputTokens();
const auto& samplingParams = req.getSamplingParams();
const auto& stopTokens = req.getStopTokens();

// Engine writes
resp.addToken(token);
req.addGeneratedToken(token);
req.notifyToken(token, finished);  // Callback

// Engine checks
if (req.isCancelled()) { break; }
if (generating_tokens >= req.getMaxTokens()) { break; }

// Engine finalizes
resp.setStoppedByEOS();
resp.finish();
```

### KV Cache
```cpp
// Request controls cache
bool reuse = req.shouldReuseCache();
bool free = req.shouldFreeCacheOnFinish();

// Response provides stats
uint64_t prefillLatency = ...;  // From timestamps
uint64_t decodeLatency = ...;
```

### Model Backend
```cpp
// Backend reads sampling params
float temp = req.getSamplingParams().temperature;
int topK = req.getSamplingParams().topK;

// Backend produces
resp.addToken(nextToken);
resp.appendText(decodedText);
```

---

## Thread Safety

### Thread-Safe for:
- **Cancellation**: `std::atomic<bool> cancelled_` with acquire/release semantics
- **Status checks**: `isFinished()`, `isFailed()`, `isCancelled()` are thread-safe reads
- **Callback invocation**: Engine thread calls `notifyToken()` safely

### NOT Thread-Safe for:
- Simultaneous `addToken()` calls (use mutex in Engine if needed)
- Simultaneous configuration changes (set all before submission)

### Best Practice:
```cpp
// Setup phase (single-threaded)
req.setSamplingParams(params);
req.setStopTokens(stops);
req.setStreaming(true);
req.setTokenCallback(cb);

// Submit (thread-safe from here)
auto future = engine.generateAsync(req);

// Can safely call from any thread
if (someCondition) req.cancel();

// Poll from any thread
int progress = req.getGeneratedTokenCount();
```

---

## Error Handling Strategy

### No Exceptions During Generation

Generation errors are captured in Response, not thrown:

```cpp
// Good: Capture error, preserve partial results
try {
    auto resp = engine.generate(req);
    if (resp.hasError()) {
        // Handle gracefully
        std::cerr << resp.getErrorMessage() << std::endl;
    }
} catch (const std::exception& e) {
    // Only for setup errors, not generation failures
}
```

### Validation Errors

Only SamplingParams validation throws:

```cpp
SamplingParams params;
params.temperature = -1.0f;  // Invalid
req.setSamplingParams(params);  // Throws std::invalid_argument
```

### Generation Errors

Captured in Response:

```cpp
auto resp = engine.generate(req);

if (resp.hasError()) {
    const auto& msg = resp.getErrorMessage();
    std::cout << "Generation failed: " << msg << std::endl;
    // Client decides recovery strategy
}
```

---

## Performance Considerations

### Memory
- Request: ~200 bytes base + variable (tokens, callbacks)
- Response: ~300 bytes base + token/logprob storage
- Both are lightweight for batching thousands of requests

### Latency
- Nanosecond timestamps enable sub-millisecond measurements
- Atomic cancellation is ultra-fast (single CAS)
- Streaming callbacks have minimal overhead

### Throughput
- Per-token latency tracking enables SLA monitoring
- Scheduler can use `getGeneratedTokenCount()` for progress
- Streaming allows client to start processing before generation finishes

---

## Next Steps

1. **Scheduler Integration**: Use Request metadata for smart batching
2. **Engine Integration**: Wire streaming callbacks through generation loop
3. **Error Handling**: Ensure all error paths populate Response correctly
4. **Testing**: Unit tests for Request/Response, integration tests with Engine
5. **Monitoring**: Use timestamp precision for performance analysis

