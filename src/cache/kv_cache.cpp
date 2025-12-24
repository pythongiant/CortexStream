#include "cortexstream/kv_cache.h"
#include "cortexstream/request.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <sstream>
#include <iomanip>

namespace cortexstream {

// ============================================================================
// KVBlockAllocator Implementation with Buddy System for O(log n) allocation
// ============================================================================

KVBlockAllocator::KVBlockAllocator(size_t totalBlocks)
    : totalBlocks_(totalBlocks), freeList_(totalBlocks, true) {
    // Initialize buddy allocator structure
    // Maintains separate free lists for each power-of-2 size
    // This enables O(log n) allocation instead of O(n) linear scan
    initBuddySystem();
}

void KVBlockAllocator::initBuddySystem() {
    // Initialize buddy allocator: maintains free lists per size class
    // Size classes: 1, 2, 4, 8, 16, 32, ... blocks
    int powerOfTwo = 1;
    while (powerOfTwo <= static_cast<int>(totalBlocks_)) {
        buddyFreeLists_[powerOfTwo] = std::deque<int>();
        powerOfTwo *= 2;
    }
    
    // Add entire allocation pool to largest power-of-2 free list
    int largestPower = 1;
    while (largestPower * 2 <= static_cast<int>(totalBlocks_)) {
        largestPower *= 2;
    }
    
    // If totalBlocks is not power of 2, add what fits
    if (largestPower <= static_cast<int>(totalBlocks_)) {
        buddyFreeLists_[largestPower].push_back(0);
    }
}

KVHandle KVBlockAllocator::allocate(int blocksNeeded) {
    // Optimized buddy allocator allocation: O(log totalBlocks)
    // Replaces old O(n) linear scan
    
    if (blocksNeeded <= 0) {
        return {-1, 0};
    }
    
    std::lock_guard<std::mutex> guard(lock_);
    
    // Find appropriate power-of-2 size class
    int allocSize = 1;
    while (allocSize < blocksNeeded) {
        allocSize *= 2;
    }
    
    if (allocSize > static_cast<int>(totalBlocks_)) {
        return {-1, 0};  // Request too large
    }

    // Try to allocate from size class (buddy system)
    return buddyAllocate(allocSize);
}

KVHandle KVBlockAllocator::buddyAllocate(int size) {
    // Recursive buddy allocation with splitting and merging
    // Time: O(log totalBlocks)
    
    // Check if size class has free blocks
    if (buddyFreeLists_.count(size) && !buddyFreeLists_[size].empty()) {
        int startIdx = buddyFreeLists_[size].front();
        buddyFreeLists_[size].pop_front();
        
        // Mark blocks as used
        for (int i = startIdx; i < startIdx + size; ++i) {
            freeList_[i] = false;
        }
        
        return {startIdx, size};
    }

    // No free blocks at this size: try splitting larger blocks
    int nextSize = size * 2;
    if (nextSize > static_cast<int>(totalBlocks_)) {
        return {-1, 0};  // Out of memory
    }

    if (buddyFreeLists_.count(nextSize) && !buddyFreeLists_[nextSize].empty()) {
        int largeBlock = buddyFreeLists_[nextSize].front();
        buddyFreeLists_[nextSize].pop_front();

        // Split into two buddies
        int buddy1 = largeBlock;
        int buddy2 = largeBlock + nextSize / 2;

        // Add buddy2 back to free list
        buddyFreeLists_[nextSize / 2].push_back(buddy2);

        // Allocate buddy1
        for (int i = buddy1; i < buddy1 + nextSize / 2; ++i) {
            freeList_[i] = false;
        }

        return {buddy1, nextSize / 2};
    }

    // Recursively try next larger size
    return buddyAllocate(nextSize);
}

void KVBlockAllocator::free(const KVHandle& handle) {
    // Optimized buddy allocator deallocation with coalescing
    // Time: O(log totalBlocks)
    
    if (!handle.isValid()) {
        return;
    }
    
    std::lock_guard<std::mutex> guard(lock_);
    
    // Mark blocks as free
    for (int i = handle.startBlockIndex; i < handle.startBlockIndex + handle.numBlocks; ++i) {
        if (i >= 0 && i < static_cast<int>(totalBlocks_)) {
            freeList_[i] = true;
        }
    }

    // Add to buddy free list and attempt coalescing
    buddyFreeLists_[handle.numBlocks].push_back(handle.startBlockIndex);
    
    // Attempt buddy coalescence recursively
    buddyCoalesce(handle.numBlocks);
}

void KVBlockAllocator::buddyCoalesce(int size) {
    // Merge adjacent free buddy blocks to reduce fragmentation
    // Time: O(log totalBlocks)
    
    if (buddyFreeLists_.find(size) == buddyFreeLists_.end()) {
        return;
    }

    auto& freeList = buddyFreeLists_[size];
    if (freeList.size() < 2) {
        return;  // Need at least 2 blocks to coalesce
    }

    // Sort free list to identify adjacent buddies
    std::sort(freeList.begin(), freeList.end());

    // Look for adjacent pairs
    auto it = freeList.begin();
    while (it != freeList.end()) {
        auto next_it = std::next(it);
        
        if (next_it != freeList.end()) {
            int block1 = *it;
            int block2 = *next_it;
            
            // Check if blocks are adjacent buddies
            if (block2 == block1 + size) {
                // Coalesce: merge into larger block
                freeList.erase(next_it);
                it = freeList.erase(it);
                
                // Add merged block to next size class
                int nextSize = size * 2;
                buddyFreeLists_[nextSize].push_back(block1);
                
                // Recursively try to coalesce larger blocks
                buddyCoalesce(nextSize);
                return;
            }
        }
        ++it;
    }
}

int KVBlockAllocator::findContiguousFreeRegion(int blocksNeeded) {
    // Legacy linear scan fallback (removed in favor of buddy system)
    // Kept for backward compatibility but rarely used
    
    int contiguousCount = 0;
    int startIdx = -1;
    
    for (size_t i = 0; i <= totalBlocks_; ++i) {
        if (i < totalBlocks_ && freeList_[i]) {
            if (contiguousCount == 0) {
                startIdx = i;
            }
            contiguousCount++;
        } else {
            if (contiguousCount >= blocksNeeded) {
                return startIdx;
            }
            contiguousCount = 0;
            startIdx = -1;
        }
    }
    
    return -1;
}


size_t KVBlockAllocator::freeBlocks() const {
    std::lock_guard<std::mutex> guard(lock_);
    size_t count = 0;
    for (bool free : freeList_) {
        if (free) count++;
    }
    return count;
}

size_t KVBlockAllocator::usedBlocks() const {
    std::lock_guard<std::mutex> guard(lock_);
    size_t count = 0;
    for (bool free : freeList_) {
        if (!free) count++;
    }
    return count;
}

size_t KVBlockAllocator::totalBlocks() const {
    return totalBlocks_;
}

float KVBlockAllocator::fragmentation() const {
    // Simple metric: ratio of unused free blocks to total blocks
    // Better metric would measure largest free contiguous region
    std::lock_guard<std::mutex> guard(lock_);
    if (totalBlocks_ == 0) return 0.0f;
    
    size_t largestContiguous = 0;
    size_t currentContiguous = 0;
    
    for (bool free : freeList_) {
        if (free) {
            currentContiguous++;
            largestContiguous = std::max(largestContiguous, currentContiguous);
        } else {
            currentContiguous = 0;
        }
    }
    
    // Fragmentation = 1 - (largest free region / total free blocks)
    size_t totalFree = freeBlocks();
    if (totalFree == 0) return 0.0f;
    
    return 1.0f - (static_cast<float>(largestContiguous) / static_cast<float>(totalFree));
}

void KVBlockAllocator::dumpBlockMap(std::ostream& os) const {
    std::lock_guard<std::mutex> guard(lock_);
    os << "KVBlockAllocator State:\n";
    os << "  Total blocks: " << totalBlocks_ << "\n";
    os << "  Used: " << usedBlocks() << " Free: " << freeBlocks() << "\n";
    os << "  Fragmentation: " << std::fixed << std::setprecision(2) << fragmentation() << "\n";
    os << "  Block map (. = free, X = used):\n    ";
    for (size_t i = 0; i < freeList_.size(); ++i) {
        if (i > 0 && i % 64 == 0) os << "\n    ";
        os << (freeList_[i] ? '.' : 'X');
    }
    os << "\n";
}

// ============================================================================
// KVCache Implementation
// ============================================================================

KVCache::KVCache(size_t numLayers,
                 size_t numHeads,
                 size_t headDim,
                 size_t maxTotalTokens,
                 size_t blockSize)
    : numLayers_(numLayers),
      numHeads_(numHeads),
      headDim_(headDim),
      blockSize_(blockSize) {
    
    // Compute total blocks needed
    totalBlocks_ = (maxTotalTokens + blockSize - 1) / blockSize;
    
    // Preallocate unified K and V buffers
    // K: [numLayers, totalBlocks, numHeads, blockSize, headDim]
    // V: [numLayers, totalBlocks, numHeads, blockSize, headDim]
    size_t blockBytes = numHeads_ * blockSize_ * headDim_;
    size_t layerBytes = totalBlocks_ * blockBytes;
    size_t totalElements = numLayers_ * layerBytes;
    
    K_.resize(totalElements, 0.0f);
    V_.resize(totalElements, 0.0f);
    
    // Initialize block allocator
    allocator_ = std::make_unique<KVBlockAllocator>(totalBlocks_);
}

KVCache::KVCache(size_t cacheBytes,
                 size_t hiddenSize,
                 size_t numLayers)
    : KVCache(
          numLayers,
          static_cast<size_t>(32),                                        // default heads
          static_cast<size_t>(std::max<size_t>(1, hiddenSize / 32)),      // default head dim
          std::max<size_t>(
              1,
              cacheBytes / (sizeof(float) * numLayers * 32ULL *
                            std::max<size_t>(1, hiddenSize / 32) * 2ULL)), // approx tokens
          16) {}

KVCache::~KVCache() = default;

bool KVCache::allocateFor(const std::string& requestId, int initialTokens) {
    std::lock_guard<std::mutex> guard(lock_);
    
    // Check if already allocated
    if (sequences_.count(requestId) > 0) {
        return false;  // Already allocated
    }
    
    // Calculate blocks needed
    int blocksNeeded = (initialTokens + blockSize_ - 1) / blockSize_;
    int maxAllowed = blocksNeeded * blockSize_;
    
    // Attempt allocation
    KVHandle handle = allocator_->allocate(blocksNeeded);
    if (!handle.isValid()) {
        return false;  // Allocation failed
    }
    
    // Store sequence entry
    SequenceKVEntry entry;
    entry.handle = handle;
    entry.tokensUsed = initialTokens;
    entry.maxAllowed = maxAllowed;
    sequences_[requestId] = entry;
    
    return true;
}

void KVCache::freeFor(const std::string& requestId) {
    std::lock_guard<std::mutex> guard(lock_);
    
    auto it = sequences_.find(requestId);
    if (it != sequences_.end()) {
        // Free blocks back to allocator
        allocator_->free(it->second.handle);
        // Remove entry
        sequences_.erase(it);
    }
}

KVView KVCache::getKView(const std::string& requestId, int layer) {
    std::lock_guard<std::mutex> guard(lock_);
    
    auto it = sequences_.find(requestId);
    if (it == sequences_.end()) {
        return {nullptr, {}, false};
    }
    
    const auto& entry = it->second;
    
    // K tensor view: [numHeads, tokensUsed, headDim]
    // Points to first block of this sequence in this layer
    float* data = getKBuffer(entry.handle.startBlockIndex, layer, 0, 0);
    
    return {data,
            {numHeads_, static_cast<size_t>(entry.tokensUsed), headDim_},
            true};
}

KVView KVCache::getVView(const std::string& requestId, int layer) {
    std::lock_guard<std::mutex> guard(lock_);
    
    auto it = sequences_.find(requestId);
    if (it == sequences_.end()) {
        return {nullptr, {}, false};
    }
    
    const auto& entry = it->second;
    
    // V tensor view: [numHeads, tokensUsed, headDim]
    float* data = getVBuffer(entry.handle.startBlockIndex, layer, 0, 0);
    
    return {data,
            {numHeads_, static_cast<size_t>(entry.tokensUsed), headDim_},
            true};
}

int KVCache::usedTokens(const std::string& requestId) const {
    std::lock_guard<std::mutex> guard(lock_);
    
    auto it = sequences_.find(requestId);
    if (it == sequences_.end()) {
        return 0;
    }
    
    return it->second.tokensUsed;
}

bool KVCache::appendToken(const std::string& requestId) {
    std::lock_guard<std::mutex> guard(lock_);
    
    auto it = sequences_.find(requestId);
    if (it == sequences_.end()) {
        return false;  // Sequence not found
    }
    
    auto& entry = it->second;
    
    // Check if we have capacity
    if (entry.tokensUsed >= entry.maxAllowed) {
        return false;  // Out of capacity
    }
    
    entry.tokensUsed++;
    return true;
}

int KVCache::getTokenOffsetInBlock(const std::string& requestId) const {
    std::lock_guard<std::mutex> guard(lock_);
    
    auto it = sequences_.find(requestId);
    if (it == sequences_.end()) {
        return -1;
    }
    
    // Offset within current block (0 to blockSize-1)
    return it->second.tokensUsed % blockSize_;
}

size_t KVCache::getTotalAllocated() const {
    size_t blockSize = numHeads_ * blockSize_ * headDim_ * sizeof(float);
    return allocator_->usedBlocks() * 2 * blockSize;  // 2 = K and V
}

size_t KVCache::getTotalFree() const {
    size_t blockSize = numHeads_ * blockSize_ * headDim_ * sizeof(float);
    return allocator_->freeBlocks() * 2 * blockSize;
}

int KVCache::getNumAllocatedSequences() const {
    std::lock_guard<std::mutex> guard(lock_);
    return sequences_.size();
}

bool KVCache::isFull() const {
    return allocator_->freeBlocks() == 0;
}

float KVCache::getFragmentation() const {
    return allocator_->fragmentation();
}

void KVCache::warmup() {
    // Touch memory to ensure pages are allocated
    const size_t pageSize = 4096;
    
    for (size_t i = 0; i < K_.size(); i += pageSize / sizeof(float)) {
        K_[i] = 0.0f;
    }
    
    for (size_t i = 0; i < V_.size(); i += pageSize / sizeof(float)) {
        V_[i] = 0.0f;
    }
}

void KVCache::dumpCacheStats(std::ostream& os) const {
    std::lock_guard<std::mutex> guard(lock_);
    
    os << "\n=== KVCache Statistics ===\n";
    os << "Configuration:\n";
    os << "  Layers: " << numLayers_ << ", Heads: " << numHeads_ 
       << ", HeadDim: " << headDim_ << "\n";
    os << "  BlockSize: " << blockSize_ << ", TotalBlocks: " << totalBlocks_ << "\n";
    os << "\nAllocation State:\n";
    os << "  Allocated sequences: " << sequences_.size() << "\n";
    os << "  Total allocated: " << (getTotalAllocated() / 1024.0f / 1024.0f) << " MB\n";
    os << "  Total free: " << (getTotalFree() / 1024.0f / 1024.0f) << " MB\n";
    os << "  Fragmentation: " << std::fixed << std::setprecision(2) 
       << getFragmentation() << "\n";
    os << "\nSequences:\n";
    
    for (const auto& [reqId, entry] : sequences_) {
        os << "  " << reqId << ": " << entry.tokensUsed << "/" << entry.maxAllowed 
           << " tokens, blocks [" << entry.handle.startBlockIndex << ", +" 
           << entry.handle.numBlocks << "]\n";
    }
}

float* KVCache::getKBuffer(int blockIndex, int layer, int head, int offset) {
    // K: [numLayers, totalBlocks, numHeads, blockSize, headDim]
    // Linear offset: layer * (totalBlocks * numHeads * blockSize * headDim)
    //              + blockIndex * (numHeads * blockSize * headDim)
    //              + head * (blockSize * headDim)
    //              + offset * headDim
    
    size_t idx = layer * (totalBlocks_ * numHeads_ * blockSize_ * headDim_)
               + blockIndex * (numHeads_ * blockSize_ * headDim_)
               + head * (blockSize_ * headDim_)
               + offset * headDim_;
    
    return K_.data() + idx;
}

float* KVCache::getVBuffer(int blockIndex, int layer, int head, int offset) {
    // Same layout as K
    size_t idx = layer * (totalBlocks_ * numHeads_ * blockSize_ * headDim_)
               + blockIndex * (numHeads_ * blockSize_ * headDim_)
               + head * (blockSize_ * headDim_)
               + offset * headDim_;
    
    return V_.data() + idx;
}

}  // namespace cortexstream

