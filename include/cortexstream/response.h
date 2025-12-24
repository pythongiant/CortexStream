#ifndef CORTEXSTREAM_RESPONSE_H
#define CORTEXSTREAM_RESPONSE_H

#include <string>
#include <vector>
#include <cstdint>
#include <utility>

namespace cortexstream {

// ============================================================================
// ResponseChunk: Streaming response unit
// ============================================================================

/**
 * Represents a single streaming update from the engine.
 * 
 * Sent to client after each token is generated.
 * Enables low-latency perception of generation progress.
 */
struct ResponseChunk {
    std::string requestId;      // Which request this belongs to
    int token;                  // Token ID generated
    std::string textPiece;      // Decoded text for this token
    bool finished;              // Whether sequence is complete
    
    ResponseChunk() = default;
    ResponseChunk(const std::string& reqId, int tok, const std::string& textPiece);
};

// ============================================================================
// Response: Server-facing output contract
// ============================================================================

/**
 * Complete result of an inference session.
 * 
 * Populated incrementally during generation.
 * Supports streaming updates and final summary.
 * 
 * Completion can be due to:
 * - EOS token (stoppedByEOS)
 * - Max tokens reached (stoppedByLimit)
 * - Stop string found (stoppedByString)
 * - User cancellation (stoppedByUser)
 * - Error (errored)
 */
class Response {
public:
    /**
     * Create a new response for a request.
     * 
     * @param requestId Request identifier to link response
     */
    explicit Response(const std::string& requestId);
    ~Response() = default;

    // ---- Identity ----
    
    const std::string& getRequestId() const;

    // ---- Generated Output ----
    
    const std::string& getText() const;
    void appendText(const std::string& text);
    void setText(const std::string& text);
    
    const std::vector<int>& getTokens() const;
    void addToken(int token);
    void setTokens(const std::vector<int>& tokens);
    
    // ---- Optional Debug Info ----
    
    const std::vector<float>& getLogprobs() const;
    void setLogprobs(const std::vector<float>& logprobs);
    
    const std::vector<std::vector<std::pair<int, float>>>& getTopKLogprobs() const;
    void setTopKLogprobs(
        const std::vector<std::vector<std::pair<int, float>>>& topk);
    void addTopKForToken(
        const std::vector<std::pair<int, float>>& topk);

    // ---- Completion Reason ----
    
    bool isFinished() const;
    void finish();
    
    bool hasStoppedByEOS() const;
    void setStoppedByEOS();
    
    bool hasStoppedByMaxTokens() const;
    void setStoppedByMaxTokens();
    
    bool hasStoppedByStopString() const;
    void setStoppedByStopString();
    
    bool hasStoppedByStopToken() const;
    void setStoppedByStopToken();
    
    bool hasStoppedByUser() const;
    void setStoppedByUser();
    
    // ---- Error State ----
    
    bool hasError() const;
    const std::string& getErrorMessage() const;
    void setError(const std::string& message);

    // ---- Statistics ----
    
    int getInputTokenCount() const;
    void setInputTokenCount(int count);
    
    int getOutputTokenCount() const;
    
    uint64_t getStartTimeNs() const;
    uint64_t getEndTimeNs() const;
    uint64_t getLatencyNs() const;
    
    double getLatencyMs() const;
    double getLatencySec() const;
    double getTokensPerSecond() const;
    double getAverageTokenLatencyMs() const;

    // ---- Utility ----
    
    std::string getCompletionReason() const;
    std::string toString() const;

private:
    // Identity
    std::string requestId_;
    
    // Generated output
    std::string text_;
    std::vector<int> tokens_;
    
    // Optional debug
    std::vector<float> logprobs_;
    std::vector<std::vector<std::pair<int, float>>> topkLogprobs_;
    
    // Completion reason (mutually exclusive)
    bool finished_ = false;
    bool stoppedByEOS_ = false;
    bool stoppedByMaxTokens_ = false;
    bool stoppedByStopString_ = false;
    bool stoppedByStopToken_ = false;
    bool stoppedByUser_ = false;
    
    // Error
    bool errored_ = false;
    std::string errorMessage_;
    
    // Statistics
    int inputTokenCount_ = 0;
    uint64_t startTimeNs_ = 0;
    uint64_t endTimeNs_ = 0;
};

}  // namespace cortexstream

#endif  // CORTEXSTREAM_RESPONSE_H
