#ifndef CORTEXSTREAM_REQUEST_H
#define CORTEXSTREAM_REQUEST_H

#include <string>
#include <memory>
#include <vector>
#include <functional>
#include <chrono>
#include <atomic>
#include <cstdint>

namespace cortexstream {

// ============================================================================
// SamplingParams: Control token sampling behavior
// ============================================================================

struct SamplingParams {
    float temperature = 1.0f;          // Softmax temperature
    int topK = 40;                     // Top-K filtering
    float topP = 0.9f;                 // Nucleus (top-P) sampling
    bool greedy = false;               // Argmax instead of sampling
    uint32_t seed = 0;                 // For determinism (0 = random)
    float repetitionPenalty = 1.0f;    // Penalize repeated tokens
    
    bool isValid() const;
};

// ============================================================================
// Request: Client-facing input contract
// ============================================================================

/**
 * Encapsulates everything needed for an inference session.
 * 
 * Design principles:
 * - Immutable after submission (except cancellation flag)
 * - Lightweight and scheduler-friendly
 * - KV-cache aware
 * - Streaming aware
 * - Thread-safe for sharing
 */
class Request {
public:
    /**
     * Create a new inference request.
     * 
     * @param id              Unique request identifier
     * @param prompt          Input prompt text
     * @param inputTokens     Pre-tokenized input
     * @param maxTokens       Maximum output tokens (default 256)
     */
    explicit Request(const std::string& id,
                     const std::string& prompt,
                     const std::vector<int>& inputTokens,
                     int maxTokens = 256);
    ~Request();

    // ---- Immutable Input ----
    
    const std::string& getId() const;
    const std::string& getPrompt() const;
    const std::vector<int>& getInputTokens() const;
    int getInputTokenCount() const;
    int getMaxTokens() const;
    int getMinTokens() const;
    
    // ---- Configuration ----
    
    const SamplingParams& getSamplingParams() const;
    void setSamplingParams(const SamplingParams& params);
    
    const std::vector<int>& getStopTokens() const;
    void setStopTokens(const std::vector<int>& tokens);
    
    const std::string& getStopString() const;
    void setStopString(const std::string& stopStr);
    
    // ---- Scheduling Controls ----
    
    int getPriority() const;
    void setPriority(int p);
    
    bool allowsPrefillOnly() const;
    void setAllowPrefillOnly(bool allow);
    
    // ---- KV Cache Controls ----
    
    bool shouldReuseCache() const;
    void setReuseCache(bool reuse);
    
    bool shouldFreeCacheOnFinish() const;
    void setFreeCacheOnFinish(bool free);
    
    // ---- Streaming ----
    
    bool isStreamingEnabled() const;
    void setStreaming(bool stream);
    
    // ---- Metadata ----
    
    uint64_t getArrivalTimestampNs() const;
    std::chrono::system_clock::time_point getArrivalTime() const;
    
    // ---- Runtime State (Mutable) ----
    
    bool isCancelled() const;
    void cancel();
    
    // ---- Execution State (Engine-Facing) ----
    
    std::vector<int>& getGeneratedTokens();
    const std::vector<int>& getGeneratedTokens() const;
    void addGeneratedToken(int token);
    int getGeneratedTokenCount() const;
    
    bool isFinished() const;
    bool isFailed() const;
    
    // ---- Error Handling ----
    
    const std::string& getErrorMessage() const;
    void setError(const std::string& message);
    
    // ---- Callbacks ----
    
    using TokenCallback = std::function<void(int token, bool finished)>;
    void setTokenCallback(TokenCallback callback);
    void notifyToken(int token, bool finished);

private:
    // Immutable after submission
    std::string id_;
    std::string prompt_;
    std::vector<int> inputTokens_;
    int maxTokens_;
    int minTokens_ = 0;
    
    std::vector<int> stopTokens_;
    std::string stopString_;
    
    SamplingParams samplingParams_;
    int priority_ = 0;
    bool allowPrefillOnly_ = false;
    bool reuseCache_ = false;
    bool freeCacheOnFinish_ = true;
    bool streaming_ = true;
    
    uint64_t arrivalTimestampNs_;
    
    // Mutable runtime state
    std::atomic<bool> cancelled_{false};
    
    // Engine-facing state
    std::vector<int> generatedTokens_;
    bool finished_ = false;
    bool failed_ = false;
    std::string errorMessage_;
    
    TokenCallback tokenCallback_;
};

}  // namespace cortexstream

#endif  // CORTEXSTREAM_REQUEST_H
