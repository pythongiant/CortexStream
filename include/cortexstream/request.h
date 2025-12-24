#ifndef CORTEXSTREAM_REQUEST_H
#define CORTEXSTREAM_REQUEST_H

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace cortexstream {

// Request lifecycle states used by the scheduler/engine
enum class RequestState {
    Pending,
    Prefilling,
    Decoding,
    Finished,
    Failed
};

// Sampling configuration (shared with sampler/model backend)
struct SamplingParams {
    // Core strategies
    float temperature = 1.0f;
    int topK = 1;
    float topP = 1.0f;
    bool doSample = false;

    // Penalties
    bool repetitionPenaltyEnabled = false;
    float repetitionPenalty = 1.1f;

    // Determinism
    int seed = -1;  // -1 = random

    // Diagnostics
    bool returnLogprobs = false;
    bool returnMetadata = false;

    bool validate() const;
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
    // Create a request with pre-tokenized prompt
    explicit Request(const std::string& id,
                     const std::vector<int>& promptTokens,
                     int maxTokens = 256,
                     const std::string& promptText = "");
    // Legacy convenience: accepts raw prompt text (simple byte-level tokenization)
    explicit Request(const std::string& id,
                     const std::string& promptText,
                     int maxTokens = 256);
    ~Request();

    // ---- Immutable Input ----
    const std::string& getId() const;
    const std::string& getPrompt() const;              // legacy alias
    const std::string& getPromptText() const;
    const std::vector<int>& getInputTokens() const;    // legacy alias
    const std::vector<int>& getPromptTokens() const;
    int getInputTokenCount() const;                    // legacy alias
    int getPromptLength() const;
    int getMaxTokens() const;
    
    // ---- Configuration ----
    
    const SamplingParams& getSamplingParams() const;
    void setSamplingParams(const SamplingParams& params);
    
    const std::vector<int>& getStopTokens() const;
    void setStopTokens(const std::vector<int>& tokens);
    
    const std::string& getStopString() const;
    void setStopString(const std::string& stopStr);
    
    // ---- Scheduling Controls ----
    
    RequestState getState() const;
    void setState(RequestState state);
    
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
    int getGeneratedTokenCount() const;                // legacy alias
    int getGeneratedLength() const;
    
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
    std::string id_;
    std::string promptText_;
    std::vector<int> promptTokens_;
    int maxTokens_;
    
    std::vector<int> stopTokens_;
    std::string stopString_;

    SamplingParams samplingParams_;
    bool streaming_ = true;
    
    uint64_t arrivalTimestampNs_;
    
    // Mutable runtime state
    std::atomic<bool> cancelled_{false};

    // Engine-facing state
    RequestState state_ = RequestState::Pending;
    std::vector<int> generatedTokens_;
    bool finished_ = false;
    bool failed_ = false;
    std::string errorMessage_;
    
    TokenCallback tokenCallback_;
};

}  // namespace cortexstream

#endif  // CORTEXSTREAM_REQUEST_H
