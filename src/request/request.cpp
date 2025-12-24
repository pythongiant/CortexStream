#include "cortexstream/request.h"
#include <chrono>
#include <stdexcept>

namespace cortexstream {

Request::Request(const std::string& id,
                 const std::vector<int>& promptTokens,
                 int maxTokens,
                 const std::string& promptText)
    : id_(id),
      promptText_(promptText),
      promptTokens_(promptTokens),
      maxTokens_(maxTokens) {
    
    // Timestamp arrival (nanoseconds since epoch)
    auto now = std::chrono::high_resolution_clock::now();
    arrivalTimestampNs_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()
    ).count();
}

Request::Request(const std::string& id,
                 const std::string& promptText,
                 int maxTokens)
    : Request(id, 
              std::vector<int>(promptText.begin(), promptText.end()),
              maxTokens,
              promptText) {}

Request::~Request() = default;

// ---- Immutable Input ----

const std::string& Request::getId() const {
    return id_;
}

const std::string& Request::getPrompt() const {
    return promptText_;
}

const std::string& Request::getPromptText() const {
    return promptText_;
}

const std::vector<int>& Request::getInputTokens() const {
    return promptTokens_;
}

const std::vector<int>& Request::getPromptTokens() const {
    return promptTokens_;
}

int Request::getInputTokenCount() const {
    return static_cast<int>(promptTokens_.size());
}

int Request::getPromptLength() const {
    return static_cast<int>(promptTokens_.size());
}

int Request::getMaxTokens() const {
    return maxTokens_;
}

// ---- Configuration ----

const SamplingParams& Request::getSamplingParams() const {
    return samplingParams_;
}

void Request::setSamplingParams(const SamplingParams& params) {
    if (!params.validate()) {
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

RequestState Request::getState() const {
    return state_;
}

void Request::setState(RequestState state) {
    state_ = state;
    if (state == RequestState::Finished) {
        finished_ = true;
    } else if (state == RequestState::Failed) {
        failed_ = true;
    }
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
    return std::chrono::time_point_cast<std::chrono::system_clock::duration>(
        std::chrono::system_clock::time_point{} + duration);
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

int Request::getGeneratedLength() const {
    return static_cast<int>(generatedTokens_.size());
}

int Request::getGeneratedTokenCount() const {
    return getGeneratedLength();
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
    state_ = RequestState::Failed;
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

