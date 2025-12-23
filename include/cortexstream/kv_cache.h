#ifndef CORTEXSTREAM_KV_CACHE_H
#define CORTEXSTREAM_KV_CACHE_H

#include <vector>
#include <cstddef>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <stdexcept>
#include <ostream>

namespace cortexstream {

// Forward declarations
class Request;

// ============================================================================
// KVBlockAllocator: Low-level block bookkeeping
// ============================================================================

/**
 * KVHandle represents a contiguous block allocation.
 * For MVP: contiguous blocks
 * Future: can be extended to support scattered paging
 */
struct KVHandle {
    int startBlockIndex;    // First block in allocation
    int numBlocks;          // Number of blocks allocated
    
    bool isValid() const {
        return startBlockIndex >= 0 && numBlocks > 0;
    }
};

/**
 * KVBlockAllocator
 * 
 * Low-level memory allocator that manages a preallocated pool of fixed-size blocks.
 * Does not understand sequences or transformers—only hands out block handles.
 * 
 * Design properties:
 *   - O(1) allocation in MVP (linear scan for contiguous free region)
 *   - Zero fragmentation guarantee (contiguous allocation)
 *   - Fail-fast: allocate returns invalid handle on failure
 *   - Thread-safe with mutex (can upgrade to lock-free later)
 */
class KVBlockAllocator {
public:
    explicit KVBlockAllocator(size_t totalBlocks);
    ~KVBlockAllocator() = default;

    /**
     * Allocate contiguous blocks.
     * Returns valid handle on success, invalid handle (startBlockIndex < 0) on failure.
     * Time: O(totalBlocks) in MVP, O(log totalBlocks) future with buddy allocator.
     */
    KVHandle allocate(int blocksNeeded);

    /**
     * Release allocated blocks back to free pool.
     * Time: O(numBlocks) to mark free.
     */
    void free(const KVHandle& handle);

    // Statistics (for monitoring and debugging)
    size_t freeBlocks() const;
    size_t usedBlocks() const;
    size_t totalBlocks() const;
    float fragmentation() const;  // 0.0 = perfect, 1.0 = maximum

    // Debug introspection
    void dumpBlockMap(std::ostream& os) const;

private:
    size_t totalBlocks_;
    std::vector<bool> freeList_;     // true = free, false = used
    mutable std::mutex lock_;        // Protect concurrent access

    // MVP: linear scan for contiguous region
    // Returns startBlockIndex or -1 if allocation fails
    int findContiguousFreeRegion(int blocksNeeded);
};

// ============================================================================
// KVCache: Logical KV memory system
// ============================================================================

/**
 * Simple Tensor abstraction for slicing.
 * In production, this would be a proper MLX tensor.
 */
struct Tensor {
    float* data;
    std::vector<size_t> shape;  // [layers, heads, blockSize, headDim] etc.
    bool valid;
};

/**
 * Per-sequence KV allocation metadata.
 */
struct SequenceKVEntry {
    KVHandle handle;        // Block allocation handle
    int tokensUsed = 0;     // Current write position
    int maxAllowed = 0;     // Max tokens this sequence can hold
};

/**
 * KVCache
 * 
 * Logical KV memory system. Owns the GPU KV tensor arena and manages
 * sequence-to-blocks mapping. Provides tensor views for prefill/decode reads/writes.
 * 
 * Memory Layout (per sequence):
 *   K: [numLayers, totalBlocks, numHeads, blockSize, headDim]
 *   V: [numLayers, totalBlocks, numHeads, blockSize, headDim]
 * 
 * Indexing logic:
 *   blockIndex = (handle.startBlockIndex + offset)
 *   tokenOffsetInBlock = tokensUsed % blockSize
 * 
 * Zero-Copy Design:
 *   - Tensor slices reference arena memory directly
 *   - No copying on allocation or token append
 *   - Block layout ensures coalesced GPU access (MLX/MPS friendly)
 */
class KVCache {
public:
    /**
     * Initialize KVCache.
     * 
     * @param numLayers       Number of transformer layers
     * @param numHeads        Number of attention heads per layer
     * @param headDim         Dimension per head (e.g., 64)
     * @param maxTotalTokens  Total tokens across all sequences
     * @param blockSize       Tokens per block (e.g., 16)
     */
    explicit KVCache(size_t numLayers,
                     size_t numHeads,
                     size_t headDim,
                     size_t maxTotalTokens,
                     size_t blockSize = 16);
    ~KVCache();

    // ---- Sequence Allocation ----

    /**
     * Allocate KV blocks for a new sequence.
     * Called at prefill start.
     * 
     * @param requestId       Unique sequence identifier
     * @param initialTokens   Initial token count (from prompt)
     * @return true on success, false if insufficient memory
     */
    bool allocateFor(const std::string& requestId, int initialTokens);

    /**
     * Free all KV blocks for a sequence.
     * Called when sequence is complete.
     */
    void freeFor(const std::string& requestId);

    // ---- Tensor Access (Zero-Copy Views) ----

    /**
     * Get K tensor view for a sequence.
     * References arena memory directly—no copy.
     * View shape: [numLayers, numHeads, tokensUsed, headDim]
     */
    Tensor getKView(const std::string& requestId, int layer);

    /**
     * Get V tensor view for a sequence.
     * References arena memory directly—no copy.
     * View shape: [numLayers, numHeads, tokensUsed, headDim]
     */
    Tensor getVView(const std::string& requestId, int layer);

    // ---- Token Management ----

    /**
     * Current token count for a sequence.
     */
    int usedTokens(const std::string& requestId) const;

    /**
     * Append one token to sequence.
     * Updates write position. Returns success.
     */
    bool appendToken(const std::string& requestId);

    /**
     * Get write position for current token in block.
     * Used by backend to know where to write KV.
     */
    int getTokenOffsetInBlock(const std::string& requestId) const;

    // ---- Statistics & Monitoring ----

    size_t getTotalAllocated() const;
    size_t getTotalFree() const;
    int getNumAllocatedSequences() const;
    bool isFull() const;
    float getFragmentation() const;

    // ---- Warmup ----
    void warmup();

    // ---- Debugging ----
    void dumpCacheStats(std::ostream& os) const;

private:
    // Configuration
    size_t numLayers_;
    size_t numHeads_;
    size_t headDim_;
    size_t blockSize_;
    size_t totalBlocks_;

    // Global arena (shared by all sequences)
    // K: [numLayers, totalBlocks, numHeads, blockSize, headDim]
    // V: [numLayers, totalBlocks, numHeads, blockSize, headDim]
    std::vector<float> K_;
    std::vector<float> V_;

    // Block allocator
    std::unique_ptr<KVBlockAllocator> allocator_;

    // Sequence tracking
    std::unordered_map<std::string, SequenceKVEntry> sequences_;
    mutable std::mutex lock_;

    // Helpers
    float* getKBuffer(int blockIndex, int layer, int head, int offset);
    float* getVBuffer(int blockIndex, int layer, int head, int offset);
};

}  // namespace cortexstream

#endif  // CORTEXSTREAM_KV_CACHE_H