#ifndef CORTEXSTREAM_SCHEDULER_H
#define CORTEXSTREAM_SCHEDULER_H

#include "request.h"
#include <queue>
#include <vector>
#include <memory>
#include <string>
#include <mutex>

namespace cortexstream {

struct Batch {
    std::vector<std::shared_ptr<Request>> requests;
    std::vector<int> sequenceLengths;
    int batchSize;
    bool isPrefill;
    
    bool empty() const { return requests.empty(); }
    void clear() {
        requests.clear();
        sequenceLengths.clear();
        batchSize = 0;
    }
};

class Scheduler {
public:
    explicit Scheduler(int maxBatchSize = 32);
    ~Scheduler();

    // Request submission
    bool submitRequest(std::shared_ptr<Request> request);
    
    // State queries
    bool hasWork() const;
    bool hasPendingRequests() const;
    bool hasActiveRequests() const;
    int getNumActiveRequests() const;
    
    // Batch building
    void acceptNewRequests();
    Batch buildPrefillBatch();
    Batch buildDecodeBatch();
    
    // Request management
    void markRequestReady(const std::string& requestId);
    void markRequestFinished(const std::string& requestId);
    void markRequestFailed(const std::string& requestId);
    
    // Lookup
    std::shared_ptr<Request> getRequest(const std::string& requestId);
    
    // Statistics
    int getMaxBatchSize() const;

private:
    int maxBatchSize;
    
    std::queue<std::shared_ptr<Request>> pendingQueue;
    std::vector<std::shared_ptr<Request>> activeRequests;
    std::vector<std::shared_ptr<Request>> finishedRequests;
    
    mutable std::mutex queueMutex;
    
    void removeFinished();
};

}  // namespace cortexstream

#endif  // CORTEXSTREAM_SCHEDULER_H
