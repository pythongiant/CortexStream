#include "cortexstream/request.h"
#include <algorithm>

namespace cortexstream {

Request::Request(const std::string& id,
                 const std::vector<int>& promptTokens,
                 int maxTokens)
    : id(id), promptTokens(promptTokens), maxTokens(maxTokens) {
    state = RequestState::Pending;
    creationTime = std::chrono::system_clock::now();
}

Request::~Request() = default;

const std::string& Request::getId() const {
    return id;
}

const std::vector<int>& Request::getPromptTokens() const {
    return promptTokens;
}

const std::vector<int>& Request::getGeneratedTokens() const {
    return generatedTokens;
}

RequestState Request::getState() const {
    return state;
}

int Request::getMaxTokens() const {
    return maxTokens;
}

const SamplingParams& Request::getSamplingParams() const {
    return samplingParams;
}

std::chrono::system_clock::time_point Request::getCreationTime() const {
    return creationTime;
}

void Request::setState(RequestState newState) {
    state = newState;
}

void Request::addToken(int token) {
    if (generatedTokens.size() < static_cast<size_t>(maxTokens)) {
        generatedTokens.push_back(token);
        if (tokenCallback) {
            tokenCallback(token, false);
        }
    }
}

void Request::setSamplingParams(const SamplingParams& params) {
    samplingParams = params;
}

void Request::setTokenCallback(TokenCallback callback) {
    tokenCallback = callback;
}

bool Request::isFinished() const {
    return state == RequestState::Finished;
}

bool Request::isFailed() const {
    return state == RequestState::Failed;
}

int Request::getPromptLength() const {
    return promptTokens.size();
}

int Request::getGeneratedLength() const {
    return generatedTokens.size();
}

}  // namespace cortexstream

