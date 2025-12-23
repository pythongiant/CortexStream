#include "cortexstream/scheduler.h"
#include <algorithm>

namespace cortexstream {

Scheduler::Scheduler(int maxBatchSize)
    : maxBatchSize(maxBatchSize) {
}

Scheduler::~Scheduler() = default;

bool Scheduler::submitRequest(std::shared_ptr<Request> request) {
    if (!request) return false;
    
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        pendingQueue.push(request);
    }
    return true;
}

bool Scheduler::hasWork() const {
    std::lock_guard<std::mutex> lock(queueMutex);
    return !pendingQueue.empty() || !activeRequests.empty();
}

bool Scheduler::hasPendingRequests() const {
    std::lock_guard<std::mutex> lock(queueMutex);
    return !pendingQueue.empty();
}

bool Scheduler::hasActiveRequests() const {
    std::lock_guard<std::mutex> lock(queueMutex);
    return !activeRequests.empty();
}

int Scheduler::getNumActiveRequests() const {
    std::lock_guard<std::mutex> lock(queueMutex);
    return activeRequests.size();
}

void Scheduler::acceptNewRequests() {
    std::lock_guard<std::mutex> lock(queueMutex);
    
    while (!pendingQueue.empty() && 
           activeRequests.size() < static_cast<size_t>(maxBatchSize)) {
        auto req = pendingQueue.front();
        pendingQueue.pop();
        req->setState(RequestState::Prefilling);
        activeRequests.push_back(req);
    }
}

Batch Scheduler::buildPrefillBatch() {
    std::lock_guard<std::mutex> lock(queueMutex);
    
    Batch batch;
    batch.isPrefill = true;
    batch.batchSize = 0;
    
    for (auto& req : activeRequests) {
        if (req->getState() == RequestState::Prefilling) {
            batch.requests.push_back(req);
            batch.sequenceLengths.push_back(req->getPromptLength());
            batch.batchSize++;
            
            if (batch.batchSize >= maxBatchSize) {
                break;
            }
        }
    }
    
    return batch;
}

Batch Scheduler::buildDecodeBatch() {
    std::lock_guard<std::mutex> lock(queueMutex);
    
    Batch batch;
    batch.isPrefill = false;
    batch.batchSize = 0;
    
    for (auto& req : activeRequests) {
        if (req->getState() == RequestState::Decoding) {
            batch.requests.push_back(req);
            batch.sequenceLengths.push_back(req->getGeneratedLength() + 1);
            batch.batchSize++;
            
            if (batch.batchSize >= maxBatchSize) {
                break;
            }
        }
    }
    
    return batch;
}

void Scheduler::markRequestReady(const std::string& requestId) {
    std::lock_guard<std::mutex> lock(queueMutex);
    
    for (auto& req : activeRequests) {
        if (req->getId() == requestId) {
            if (req->getState() == RequestState::Prefilling) {
                req->setState(RequestState::Decoding);
            }
            return;
        }
    }
}

void Scheduler::markRequestFinished(const std::string& requestId) {
    std::lock_guard<std::mutex> lock(queueMutex);
    
    auto it = std::find_if(activeRequests.begin(), activeRequests.end(),
        [&](const auto& req) { return req->getId() == requestId; });
    
    if (it != activeRequests.end()) {
        (*it)->setState(RequestState::Finished);
        finishedRequests.push_back(*it);
        activeRequests.erase(it);
    }
}

void Scheduler::markRequestFailed(const std::string& requestId) {
    std::lock_guard<std::mutex> lock(queueMutex);
    
    auto it = std::find_if(activeRequests.begin(), activeRequests.end(),
        [&](const auto& req) { return req->getId() == requestId; });
    
    if (it != activeRequests.end()) {
        (*it)->setState(RequestState::Failed);
        activeRequests.erase(it);
    }
}

std::shared_ptr<Request> Scheduler::getRequest(const std::string& requestId) {
    std::lock_guard<std::mutex> lock(queueMutex);
    
    for (auto& req : activeRequests) {
        if (req->getId() == requestId) {
            return req;
        }
    }
    
    for (auto& req : finishedRequests) {
        if (req->getId() == requestId) {
            return req;
        }
    }
    
    return nullptr;
}

int Scheduler::getMaxBatchSize() const {
    return maxBatchSize;
}

void Scheduler::removeFinished() {
    std::lock_guard<std::mutex> lock(queueMutex);
    finishedRequests.clear();
}

}  // namespace cortexstream

