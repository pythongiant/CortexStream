#ifndef CORTEXSTREAM_TOKENIZER_H
#define CORTEXSTREAM_TOKENIZER_H

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace cortexstream {

class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    // Encode text to token IDs
    virtual std::vector<int32_t> encode(const std::string& text) = 0;
    
    // Decode token IDs to text
    virtual std::string decode(const std::vector<int32_t>& token_ids) = 0;
    
    // Get the model's special tokens
    virtual int32_t getEosTokenId() const = 0;
    virtual int32_t getBosTokenId() const = 0;
    virtual int32_t getPadTokenId() const = 0;
    
    // Get the tokenizer's vocabulary size
    virtual size_t getVocabSize() const = 0;
    
    // Check if tokenizer is loaded
    virtual bool isLoaded() const = 0;
};

// Factory function to create appropriate tokenizer
std::unique_ptr<Tokenizer> createTokenizer(
    const std::string& model_path_or_id,
    const std::string& cache_dir = "");

} // namespace cortexstream

#endif // CORTEXSTREAM_TOKENIZER_H
