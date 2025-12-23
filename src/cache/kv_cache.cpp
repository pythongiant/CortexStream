#include "cortexstream/kv_cache.h"
#include "cortexstream/request.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace cortexstream {

KVCache::KVCache(size_t cacheSize,
                 size_t hiddenSize,
                 size_t numLayers,
                 size_t blockSize)
    : cacheSize(cacheSize), 
      hiddenSize(hiddenSize),
      numLayers(numLayers),
      blockSize(blockSize) {
    
    // Allocate unified buffer
    cacheBuffer = new float[cacheSize];
    
    // Initialize blocks
    size_t numBlocks = cacheSize / (2 * blockSize * hiddenSize * numLayers);
    blocks.reserve(numBlocks);
    blockAllocated.resize(numBlocks, false);
    
    for (size_t i = 0; i < numBlocks; ++i) {
        KVBlock block;
        block.blockId = i;
        block.numTokens = 0;
        block.maxTokens = blockSize;
        block.isFull = false;
        
        size_t blockByteOffset = i * 2 * blockSize * hiddenSize * numLayers * sizeof(float);
        block.kData = cacheBuffer + blockByteOffset / sizeof(float);
        block.vData = block.kData + blockSize * hiddenSize * numLayers;
        
        blocks.push_back(block);
    }
}

KVCache::~KVCache() {
    delete[] cacheBuffer;
}

int KVCache::allocateBlock(const std::string& requestId) {
    int blockId = allocateBlockInternal();
    if (blockId >= 0) {
        associateBlockWithRequest(blockId, requestId);
    }
    return blockId;
}

void KVCache::freeBlock(int blockId) {
    if (blockId >= 0 && blockId < static_cast<int>(blocks.size())) {
        blockAllocated[blockId] = false;
        blocks[blockId].numTokens = 0;
        blocks[blockId].isFull = false;
    }
}

KVBlock* KVCache::getBlock(int blockId) {
    if (blockId >= 0 && blockId < static_cast<int>(blocks.size())) {
        return &blocks[blockId];
    }
    return nullptr;
}

const KVBlock* KVCache::getBlock(int blockId) const {
    if (blockId >= 0 && blockId < static_cast<int>(blocks.size())) {
        return &blocks[blockId];
    }
    return nullptr;
}

std::vector<int> KVCache::getBlocksForRequest(const std::string& requestId) {
    auto it = requestToBlocks.find(requestId);
    if (it != requestToBlocks.end()) {
        return it->second;
    }
    return {};
}

void KVCache::associateBlockWithRequest(int blockId, const std::string& requestId) {
    requestToBlocks[requestId].push_back(blockId);
}

void KVCache::clearRequest(const std::string& requestId) {
    auto it = requestToBlocks.find(requestId);
    if (it != requestToBlocks.end()) {
        for (int blockId : it->second) {
            freeBlock(blockId);
        }
        requestToBlocks.erase(it);
    }
}

size_t KVCache::getTotalAllocated() const {
    int allocated = 0;
    for (bool b : blockAllocated) {
        if (b) allocated++;
    }
    return allocated * 2 * blockSize * hiddenSize * numLayers * sizeof(float);
}

size_t KVCache::getTotalFree() const {
    return cacheSize - getTotalAllocated();
}

int KVCache::getNumAllocatedBlocks() const {
    int count = 0;
    for (bool b : blockAllocated) {
        if (b) count++;
    }
    return count;
}

bool KVCache::isFull() const {
    return getTotalFree() == 0;
}

void KVCache::warmup() {
    // Touch memory to ensure pages are allocated
    for (size_t i = 0; i < cacheSize; i += 4096) {
        cacheBuffer[i / sizeof(float)] = 0.0f;
    }
}

int KVCache::allocateBlockInternal() {
    for (size_t i = 0; i < blockAllocated.size(); ++i) {
        if (!blockAllocated[i]) {
            blockAllocated[i] = true;
            return i;
        }
    }
    return -1;  // No free blocks
}

void KVCache::defragment() {
    // Simple defragmentation: collect free blocks
    // In production, might use more sophisticated strategies
}

}  // namespace cortexstream

