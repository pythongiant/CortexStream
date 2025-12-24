#include "cortexstream/response.h"
#include <chrono>

namespace cortexstream {

// ============================================================================
// ResponseChunk Implementation
// ============================================================================

ResponseChunk::ResponseChunk(const std::string& reqId, int tok, const std::string& piece)
    : requestId(reqId),
      token(tok),
      textPiece(piece),
      finished(false) {
}

// ============================================================================
// Response Implementation
// ============================================================================

Response::Response(const std::string& reqId)
    : requestId_(reqId) {
    
    // Record start time in nanoseconds
    auto now = std::chrono::high_resolution_clock::now();
    startTimeNs_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()
    ).count();
}

// ---- Identity ----

const std::string& Response::getRequestId() const {
    return requestId_;
}

// ---- Generated Output ----

const std::string& Response::getText() const {
    return text_;
}

void Response::appendText(const std::string& text) {
    text_ += text;
}

void Response::setText(const std::string& text) {
    text_ = text;
}

const std::vector<int>& Response::getTokens() const {
    return tokens_;
}

void Response::addToken(int token) {
    tokens_.push_back(token);
}

void Response::setTokens(const std::vector<int>& tokens) {
    tokens_ = tokens;
}

// ---- Debug Information ----

const std::vector<float>& Response::getLogprobs() const {
    return logprobs_;
}

void Response::setLogprobs(const std::vector<float>& logprobs) {
    logprobs_ = logprobs;
}

const std::vector<std::vector<std::pair<int, float>>>& Response::getTopKLogprobs() const {
    return topkLogprobs_;
}

void Response::setTopKLogprobs(
    const std::vector<std::vector<std::pair<int, float>>>& topk) {
    topkLogprobs_ = topk;
}

void Response::addTopKForToken(
    const std::vector<std::pair<int, float>>& topk) {
    topkLogprobs_.push_back(topk);
}

// ---- Completion Status ----

bool Response::isFinished() const {
    return finished_;
}

void Response::finish() {
    finished_ = true;
    
    // Record end time when finished
    auto now = std::chrono::high_resolution_clock::now();
    endTimeNs_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()
    ).count();
}

// ---- Completion Reason ----

bool Response::hasStoppedByEOS() const {
    return stoppedByEOS_;
}

void Response::setStoppedByEOS() {
    stoppedByEOS_ = true;
}

bool Response::hasStoppedByMaxTokens() const {
    return stoppedByMaxTokens_;
}

void Response::setStoppedByMaxTokens() {
    stoppedByMaxTokens_ = true;
}

bool Response::hasStoppedByStopString() const {
    return stoppedByStopString_;
}

void Response::setStoppedByStopString() {
    stoppedByStopString_ = true;
}

bool Response::hasStoppedByStopToken() const {
    return stoppedByStopToken_;
}

void Response::setStoppedByStopToken() {
    stoppedByStopToken_ = true;
}

bool Response::hasStoppedByUser() const {
    return stoppedByUser_;
}

void Response::setStoppedByUser() {
    stoppedByUser_ = true;
}

std::string Response::getCompletionReason() const {
    if (errored_) return "error";
    if (stoppedByEOS_) return "end_of_sequence";
    if (stoppedByMaxTokens_) return "max_tokens";
    if (stoppedByStopString_) return "stop_string";
    if (stoppedByStopToken_) return "stop_token";
    if (stoppedByUser_) return "user_cancelled";
    return "unknown";
}

// ---- Error State ----

bool Response::hasError() const {
    return errored_;
}

const std::string& Response::getErrorMessage() const {
    return errorMessage_;
}

void Response::setError(const std::string& message) {
    errored_ = true;
    errorMessage_ = message;
}

// ---- Counts ----

int Response::getInputTokenCount() const {
    return inputTokenCount_;
}

void Response::setInputTokenCount(int count) {
    inputTokenCount_ = count;
}

int Response::getOutputTokenCount() const {
    return tokens_.size();
}

// ---- Timing (Nanoseconds) ----

uint64_t Response::getStartTimeNs() const {
    return startTimeNs_;
}

uint64_t Response::getEndTimeNs() const {
    return endTimeNs_;
}

uint64_t Response::getLatencyNs() const {
    if (endTimeNs_ == 0) {
        auto now = std::chrono::high_resolution_clock::now();
        uint64_t currentNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()
        ).count();
        return currentNs - startTimeNs_;
    }
    return endTimeNs_ - startTimeNs_;
}

// ---- Statistics ----

double Response::getLatencyMs() const {
    return static_cast<double>(getLatencyNs()) / 1'000'000.0;
}

double Response::getLatencySec() const {
    return static_cast<double>(getLatencyNs()) / 1'000'000'000.0;
}

double Response::getTokensPerSecond() const {
    double latencySec = getLatencySec();
    if (latencySec <= 0.0) return 0.0;
    return static_cast<double>(getOutputTokenCount()) / latencySec;
}

double Response::getAverageTokenLatencyMs() const {
    int outputCount = getOutputTokenCount();
    if (outputCount == 0) return 0.0;
    return getLatencyMs() / static_cast<double>(outputCount);
}

// ---- Utility ----

std::string Response::toString() const {
    std::string result = "Response(";
    result += "requestId=" + requestId_;
    result += ", tokens=" + std::to_string(getOutputTokenCount());
    result += ", finished=" + std::string(finished_ ? "true" : "false");
    result += ", latencyMs=" + std::to_string(getLatencyMs());
    result += ", reason=" + getCompletionReason();
    result += ")";
    return result;
}

}  // namespace cortexstream
