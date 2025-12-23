#ifndef CORTEXSTREAM_KV_CACHE_H
#define CORTEXSTREAM_KV_CACHE_H

#include <vector>
#include <cstddef>
#include <memory>
#include <map>

namespace cortexstream {

// Forward declarations
class Request;

struct KVBlock {
    int blockId;
    int numTokens;
    int maxTokens;
    bool isFull;
    float* kData;  // K cache
    float* vData;  // V cache
};

class KVCache {
public:
    explicit KVCache(size_t cacheSize,
                     size_t hiddenSize,
                     size_t numLayers,
                     size_t blockSize = 16);
    ~KVCache();

    // Block management
    int allocateBlock(const std::string& requestId);
    void freeBlock(int blockId);
    
    // Cache access
    KVBlock* getBlock(int blockId);
    const KVBlock* getBlock(int blockId) const;
    
    // Request management
    std::vector<int> getBlocksForRequest(const std::string& requestId);
    void associateBlockWithRequest(int blockId, const std::string& requestId);
    void clearRequest(const std::string& requestId);
    
    // Statistics
    size_t getTotalAllocated() const;
    size_t getTotalFree() const;
    int getNumAllocatedBlocks() const;
    bool isFull() const;
    
    // Device operations
    void warmup();

private:
    size_t cacheSize;
    size_t hiddenSize;
    size_t numLayers;
    size_t blockSize;
    
    std::vector<KVBlock> blocks;
    std::vector<bool> blockAllocated;
    std::map<std::string, std::vector<int>> requestToBlocks;
    
    float* cacheBuffer;
    size_t nextFreeIndex = 0;
    
    int allocateBlockInternal();
    void defragment();
};

}  // namespace cortexstream

#endif  // CORTEXSTREAM_KV_CACHE_H
