#include "cortexstream/request.h"
#include <chrono>
#include <stdexcept>

namespace cortexstream {

// ============================================================================
// SamplingParams Implementation
// ============================================================================

bool SamplingParams::isValid() const {
    if (temperature < 0.0f || temperature > 2.0f) return false;
    if (topK < 1 || topK > 100) return false;
    if (topP <= 0.0f || topP > 1.0f) return false;
    if (repetitionPenalty < 0.0f || repetitionPenalty > 2.0f) return false;
    return true;
}

// ============================================================================
// Request Implementation
// ============================================================================

Request::Request(const std::string& id,
                 const std::string& prompt,
                 const std::vector<int>& inputTokens,
                 int maxTokens)
    : id_(id),
      prompt_(prompt),
      inputTokens_(inputTokens),
      maxTokens_(maxTokens) {
    
    // Timestamp arrival (nanoseconds since epoch)
    auto now = std::chrono::high_resolution_clock::now();
    arrivalTimestampNs_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()
    ).count();
}

Request::~Request() = default;

// ---- Immutable Input ----

const std::string& Request::getId() const {
    return id_;
}

const std::string& Request::getPrompt() const {
    return prompt_;
}

const std::vector<int>& Request::getInputTokens() const {
    return inputTokens_;
}

int Request::getInputTokenCount() const {
    return inputTokens_.size();
}

int Request::getMaxTokens() const {
    return maxTokens_;
}

int Request::getMinTokens() const {
    return minTokens_;
}

// ---- Configuration ----

const SamplingParams& Request::getSamplingParams() const {
    return samplingParams_;
}

void Request::setSamplingParams(const SamplingParams& params) {
    if (!params.isValid()) {
        throw std::invalid_argument("Invalid sampling parameters");
    }
    samplingParams_ = params;
}

const std::vector<int>& Request::getStopTokens() const {
    return stopTokens_;
}

void Request::setStopTokens(const std::vector<int>& tokens) {
    stopTokens_ = tokens;
}

const std::string& Request::getStopString() const {
    return stopString_;
}

void Request::setStopString(const std::string& stopStr) {
    stopString_ = stopStr;
}

// ---- Scheduling Controls ----

int Request::getPriority() const {
    return priority_;
}

void Request::setPriority(int p) {
    priority_ = p;
}

bool Request::allowsPrefillOnly() const {
    return allowPrefillOnly_;
}

void Request::setAllowPrefillOnly(bool allow) {
    allowPrefillOnly_ = allow;
}

// ---- KV Cache Controls ----

bool Request::shouldReuseCache() const {
    return reuseCache_;
}

void Request::setReuseCache(bool reuse) {
    reuseCache_ = reuse;
}

bool Request::shouldFreeCacheOnFinish() const {
    return freeCacheOnFinish_;
}

void Request::setFreeCacheOnFinish(bool free) {
    freeCacheOnFinish_ = free;
}

// ---- Streaming ----

bool Request::isStreamingEnabled() const {
    return streaming_;
}

void Request::setStreaming(bool stream) {
    streaming_ = stream;
}

// ---- Metadata ----

uint64_t Request::getArrivalTimestampNs() const {
    return arrivalTimestampNs_;
}

std::chrono::system_clock::time_point Request::getArrivalTime() const {
    auto duration = std::chrono::nanoseconds(arrivalTimestampNs_);
    return std::chrono::system_clock::time_point(duration);
}

// ---- Runtime State (Mutable) ----

bool Request::isCancelled() const {
    return cancelled_.load(std::memory_order_acquire);
}

void Request::cancel() {
    cancelled_.store(true, std::memory_order_release);
}

// ---- Execution State (Engine-Facing) ----

std::vector<int>& Request::getGeneratedTokens() {
    return generatedTokens_;
}

const std::vector<int>& Request::getGeneratedTokens() const {
    return generatedTokens_;
}

void Request::addGeneratedToken(int token) {
    generatedTokens_.push_back(token);
}

int Request::getGeneratedTokenCount() const {
    return generatedTokens_.size();
}

bool Request::isFinished() const {
    return finished_;
}

bool Request::isFailed() const {
    return failed_;
}

// ---- Error Handling ----

const std::string& Request::getErrorMessage() const {
    return errorMessage_;
}

void Request::setError(const std::string& message) {
    failed_ = true;
    errorMessage_ = message;
}

// ---- Callbacks ----

void Request::setTokenCallback(TokenCallback callback) {
    tokenCallback_ = callback;
}

void Request::notifyToken(int token, bool finished) {
    if (tokenCallback_) {
        tokenCallback_(token, finished);
    }
}

}  // namespace cortexstream

