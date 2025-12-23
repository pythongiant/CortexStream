# Request/Response API Quick Reference

## Quick Examples

### Basic Generation
```cpp
#include "cortexstream/request.h"
#include "cortexstream/response.h"

// Create request
Request req("req-1", "What is AI?", inputTokens, 256);

// Generate
auto resp = engine.generate(req);

// Get result
std::cout << resp.getText() << std::endl;
```

### Streaming
```cpp
Request req("req-1", "Summarize...", tokens, 100);
req.setStreaming(true);
req.setTokenCallback([](int token, bool done) {
    std::cout << token;
    if (done) std::cout << " [done]" << std::endl;
});

auto resp = engine.generate(req);
```

### Custom Sampling
```cpp
SamplingParams params;
params.temperature = 0.7f;    // [0.0-2.0]
params.topK = 40;             // [1-100]
params.topP = 0.9f;           // (0.0-1.0]
params.repetitionPenalty = 1.2f;

if (params.isValid()) {
    req.setSamplingParams(params);
} else {
    std::cerr << "Invalid sampling params" << std::endl;
}
```

### Stop Conditions
```cpp
req.setStopTokens({2});           // Stop on EOS
req.setStopString("\n");          // Stop on newline
```

### Async with Cancellation
```cpp
Request req("req-1", "...", tokens, 1000);
auto future = engine.generateAsync(req);

// Meanwhile...
std::this_thread::sleep_for(std::chrono::seconds(1));
req.cancel();  // Thread-safe!

auto resp = future.get();
std::cout << resp.getCompletionReason() << std::endl;  // "user_cancelled"
```

### Performance Metrics
```cpp
auto resp = engine.generate(req);

double latencyMs = resp.getLatencyMs();
double tps = resp.getTokensPerSecond();
double avgMs = resp.getAverageTokenLatencyMs();

std::cout << "Latency: " << latencyMs << "ms" << std::endl;
std::cout << "Throughput: " << tps << " tok/s" << std::endl;
std::cout << "Avg token latency: " << avgMs << "ms" << std::endl;
```

### Cache Control
```cpp
// Reuse cache from previous generation
req.setReuseCache(true);

// Free cache immediately after
req.setFreeCacheOnFinish(true);

auto resp = engine.generate(req);
```

### Scheduling Hints
```cpp
// High priority - SLA critical
req.setPriority(10);

// Can skip decode phase if only prefilling
req.setAllowPrefillOnly(true);

// Only do prefill, no generation
engine.prefill(req);
```

---

## Request API Reference

### Construction
```cpp
Request(const std::string& id,
        const std::string& prompt,
        const std::vector<int>& inputTokens,
        int maxTokens)
```

### Input (Immutable)
| Method | Returns |
|--------|---------|
| `getId()` | `const std::string&` |
| `getPrompt()` | `const std::string&` |
| `getInputTokens()` | `const std::vector<int>&` |
| `getInputTokenCount()` | `int` |
| `getMaxTokens()` | `int` |
| `getMinTokens()` | `int` |

### Configuration
| Method | Param Type | Returns |
|--------|-----------|---------|
| `getSamplingParams()` | — | `const SamplingParams&` |
| `setSamplingParams(params)` | `const SamplingParams&` | — |
| `getStopTokens()` | — | `const std::vector<int>&` |
| `setStopTokens(tokens)` | `const std::vector<int>&` | — |
| `getStopString()` | — | `const std::string&` |
| `setStopString(str)` | `const std::string&` | — |

### Scheduling
| Method | Param Type | Returns |
|--------|-----------|---------|
| `getPriority()` | — | `int` |
| `setPriority(p)` | `int` | — |
| `allowsPrefillOnly()` | — | `bool` |
| `setAllowPrefillOnly(allow)` | `bool` | — |

### Cache Control
| Method | Param Type | Returns |
|--------|-----------|---------|
| `shouldReuseCache()` | — | `bool` |
| `setReuseCache(reuse)` | `bool` | — |
| `shouldFreeCacheOnFinish()` | — | `bool` |
| `setFreeCacheOnFinish(free)` | `bool` | — |

### Streaming
| Method | Param Type | Returns |
|--------|-----------|---------|
| `isStreamingEnabled()` | — | `bool` |
| `setStreaming(stream)` | `bool` | — |
| `setTokenCallback(cb)` | `TokenCallback` | — |
| `notifyToken(token, finished)` | `int, bool` | — |

### Metadata
| Method | Param Type | Returns |
|--------|-----------|---------|
| `getArrivalTimestampNs()` | — | `uint64_t` |
| `getArrivalTime()` | — | `std::chrono::system_clock::time_point` |

### Runtime (Engine-Facing)
| Method | Param Type | Returns |
|--------|-----------|---------|
| `getGeneratedTokens()` | — | `std::vector<int>&` (mutable) |
| `getGeneratedTokens() const` | — | `const std::vector<int>&` |
| `addGeneratedToken(token)` | `int` | — |
| `getGeneratedTokenCount()` | — | `int` |
| `isFinished()` | — | `bool` |
| `isFailed()` | — | `bool` |

### Cancellation (Thread-Safe)
| Method | Param Type | Returns |
|--------|-----------|---------|
| `isCancelled()` | — | `bool` |
| `cancel()` | — | — |

### Error Handling
| Method | Param Type | Returns |
|--------|-----------|---------|
| `getErrorMessage()` | — | `const std::string&` |
| `setError(message)` | `const std::string&` | — |

---

## Response API Reference

### Construction
```cpp
Response(const std::string& requestId)
```

### Identity
| Method | Returns |
|--------|---------|
| `getRequestId()` | `const std::string&` |

### Generated Output
| Method | Param Type | Returns |
|--------|-----------|---------|
| `getText()` | — | `const std::string&` |
| `setText(text)` | `const std::string&` | — |
| `appendText(text)` | `const std::string&` | — |
| `getTokens()` | — | `const std::vector<int>&` |
| `setTokens(tokens)` | `const std::vector<int>&` | — |
| `addToken(token)` | `int` | — |

### Debug Info
| Method | Param Type | Returns |
|--------|-----------|---------|
| `getLogprobs()` | — | `const std::vector<float>&` |
| `setLogprobs(lp)` | `const std::vector<float>&` | — |
| `getTopKLogprobs()` | — | `const std::vector<std::vector<std::pair<int, float>>>&` |
| `setTopKLogprobs(topk)` | `const std::vector<...>&` | — |
| `addTopKForToken(topk)` | `const std::vector<...>&` | — |

### Completion Status
| Method | Param Type | Returns |
|--------|-----------|---------|
| `isFinished()` | — | `bool` |
| `finish()` | — | — |

### Completion Reasons
| Method | Param Type | Returns |
|--------|-----------|---------|
| `hasStoppedByEOS()` | — | `bool` |
| `setStoppedByEOS()` | — | — |
| `hasStoppedByMaxTokens()` | — | `bool` |
| `setStoppedByMaxTokens()` | — | — |
| `hasStoppedByStopString()` | — | `bool` |
| `setStoppedByStopString()` | — | — |
| `hasStoppedByStopToken()` | — | `bool` |
| `setStoppedByStopToken()` | — | — |
| `hasStoppedByUser()` | — | `bool` |
| `setStoppedByUser()` | — | — |
| `getCompletionReason()` | — | `std::string` |

### Error State
| Method | Param Type | Returns |
|--------|-----------|---------|
| `hasError()` | — | `bool` |
| `getErrorMessage()` | — | `const std::string&` |
| `setError(message)` | `const std::string&` | — |

### Counts
| Method | Param Type | Returns |
|--------|-----------|---------|
| `getInputTokenCount()` | — | `int` |
| `setInputTokenCount(count)` | `int` | — |
| `getOutputTokenCount()` | — | `int` |

### Timing (Nanoseconds)
| Method | Param Type | Returns |
|--------|-----------|---------|
| `getStartTimeNs()` | — | `uint64_t` |
| `getEndTimeNs()` | — | `uint64_t` |
| `getLatencyNs()` | — | `uint64_t` |

### Statistics
| Method | Returns | Notes |
|--------|---------|-------|
| `getLatencyMs()` | `double` | Milliseconds |
| `getLatencySec()` | `double` | Seconds |
| `getTokensPerSecond()` | `double` | Throughput |
| `getAverageTokenLatencyMs()` | `double` | Per-token latency |

### Utility
| Method | Returns |
|--------|---------|
| `toString()` | `std::string` |

---

## SamplingParams Reference

```cpp
struct SamplingParams {
    float temperature = 1.0f;          // [0.0, 2.0]
    int topK = 40;                     // [1, 100]
    float topP = 0.9f;                 // (0.0, 1.0]
    bool greedy = false;               // Use argmax
    uint32_t seed = 0;                 // 0 = random
    float repetitionPenalty = 1.0f;    // [0.0, 2.0]
    
    bool isValid() const;
};
```

### Validation Rules
- `temperature`: Must be in [0.0, 2.0]
  - 0.0: Always pick highest probability
  - 1.0: Standard softmax
  - 2.0: Maximize entropy (very diverse)

- `topK`: Must be in [1, 100]
  - Filter to top-K tokens before sampling
  - Lower K = more conservative

- `topP`: Must be in (0.0, 1.0]
  - Nucleus sampling: accumulate until prob > P
  - Lower P = more conservative

- `repetitionPenalty`: Must be in [0.0, 2.0]
  - < 1.0: Encourage repetition
  - 1.0: No penalty
  - > 1.0: Discourage repetition

- `seed`: 0 = non-deterministic, >0 = deterministic

- `greedy`: If true, always pick argmax

---

## ResponseChunk Reference

```cpp
struct ResponseChunk {
    std::string requestId;     // Which request
    int token;                 // Token ID
    std::string textPiece;     // Decoded text
    bool finished;             // Sequence complete
};

// Constructor
ResponseChunk(const std::string& reqId, int tok,
              const std::string& text, bool done = false)
```

---

## Common Patterns

### Pattern 1: Batch Processing
```cpp
std::vector<Request> reqs;
for (const auto& prompt : prompts) {
    reqs.emplace_back(/*...*/);
}

// Process all
std::vector<Response> resps;
for (auto& req : reqs) {
    resps.push_back(engine.generate(req));
}

// Collect results
for (const auto& resp : resps) {
    std::cout << resp.getText() << std::endl;
}
```

### Pattern 2: Streaming with Timeout
```cpp
Request req(/*...*/);
req.setStreaming(true);

auto future = engine.generateAsync(req);
auto status = future.wait_for(std::chrono::seconds(5));

if (status == std::future_status::timeout) {
    req.cancel();  // Interrupt
}

auto resp = future.get();
std::cout << "Partial: " << resp.getText() << std::endl;
```

### Pattern 3: Conditional Caching
```cpp
Request req1("req-1", "First question", tokens1, 100);
auto resp1 = engine.generate(req1);

// Follow-up: reuse cache
Request req2("req-2", "Follow-up", tokens2, 100);
req2.setReuseCache(true);
auto resp2 = engine.generate(req2);

// Save time on common prefix
```

### Pattern 4: Priority Queue
```cpp
std::priority_queue<Request*,
                    std::vector<Request*>,
                    std::function<bool(Request*, Request*)>>
queue([](Request* a, Request* b) {
    return a->getPriority() < b->getPriority();
});

// Higher priority requests get processed first
```

### Pattern 5: Error Recovery
```cpp
Request req(/*...*/);

try {
    auto resp = engine.generate(req);
    
    if (resp.hasError()) {
        // Partial result available
        std::cerr << resp.getErrorMessage() << std::endl;
        
        // Decide: retry, use partial, or fail
        if (resp.getOutputTokenCount() > 10) {
            std::cout << "Partial: " << resp.getText() << std::endl;
        } else {
            // Too little, retry
        }
    } else {
        std::cout << "Success: " << resp.getText() << std::endl;
    }
} catch (const std::invalid_argument& e) {
    std::cerr << "Invalid request: " << e.what() << std::endl;
}
```

---

## Frequently Asked Questions

### Q: Can I modify Request after submission?
**A:** No (except cancellation). Set everything before `engine.generate()`.

### Q: How do I track generation progress?
**A:** Use streaming callbacks with `req.setTokenCallback()` or poll `req.getGeneratedTokenCount()`.

### Q: What's the difference between `finish()` and `setFinished()`?
**A:** Use `finish()` (sets timestamp). `setFinished()` is for manual control.

### Q: Can multiple threads cancel the same request?
**A:** Yes, `cancel()` is thread-safe via `std::atomic<bool>`.

### Q: How accurate are timing measurements?
**A:** Nanosecond precision, though actual resolution depends on OS timer.

### Q: What happens if sampling params are invalid?
**A:** `setSamplingParams()` throws `std::invalid_argument` before setting.

### Q: Can Response be modified by multiple threads?
**A:** No. Engine owns Response during generation. Read-only access from client.

### Q: What if cache reuse fails?
**A:** Engine falls back to full prefill. No error, just slower.

