#include "cortexstream/tokenizer.h"
#include <tokenizers_cpp.h>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cstdlib>
#include <iostream>

namespace cortexstream {

class HuggingFaceTokenizer : public Tokenizer {
public:
    HuggingFaceTokenizer(const std::string& model_path, const std::string& cache_dir) {
        std::string resolved_path = resolve_model_path(model_path, cache_dir);
        if (resolved_path.empty()) {
            throw std::runtime_error("Could not resolve tokenizer path for: " + model_path);
        }

        // Read the tokenizer.json file
        std::ifstream ifs(resolved_path);
        if (!ifs.is_open()) {
            throw std::runtime_error("Failed to open tokenizer file: " + resolved_path);
        }

        std::stringstream buffer;
        buffer << ifs.rdbuf();
        std::string json_blob = buffer.str();

        // Initialize tokenizer from JSON blob
        tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(json_blob);
        if (!tokenizer_) {
            throw std::runtime_error("Failed to parse tokenizer from: " + resolved_path);
        }
        loaded_ = true;
    }

    std::vector<int32_t> encode(const std::string& text) override {
        if (!loaded_ || !tokenizer_) return {};

        std::vector<int> ids = tokenizer_->Encode(text);

        // Convert to int32_t
        std::vector<int32_t> result;
        result.reserve(ids.size());
        for (int id : ids) {
            result.push_back(static_cast<int32_t>(id));
        }

        return result;
    }

    std::string decode(const std::vector<int32_t>& token_ids) override {
        if (!loaded_ || !tokenizer_ || token_ids.empty()) return "";

        // Convert to int vector
        std::vector<int> ids;
        ids.reserve(token_ids.size());
        for (int32_t id : token_ids) {
            ids.push_back(static_cast<int>(id));
        }

        return tokenizer_->Decode(ids);
    }

    int32_t getEosTokenId() const override {
        // Common EOS token IDs - tokenizers-cpp doesn't expose this directly
        // These are typical values for Llama/Mistral models
        return 2;  // </s>
    }

    int32_t getBosTokenId() const override {
        return 1;  // <s>
    }

    int32_t getPadTokenId() const override {
        return 0;  // <pad>
    }

    size_t getVocabSize() const override {
        if (!loaded_ || !tokenizer_) return 0;
        return tokenizer_->GetVocabSize();
    }

    bool isLoaded() const override { return loaded_; }

private:
    std::string resolve_model_path(const std::string& model_path, const std::string& cache_dir) {
        // If it's a direct path to tokenizer.json, use it
        if (std::filesystem::exists(model_path)) {
            // Check if it's a file or directory
            if (std::filesystem::is_regular_file(model_path)) {
                return model_path;
            }
            // If it's a directory, look for tokenizer.json inside
            std::filesystem::path dir_path(model_path);
            std::filesystem::path tokenizer_file = dir_path / "tokenizer.json";
            if (std::filesystem::exists(tokenizer_file)) {
                return tokenizer_file.string();
            }
        }

        // Try to find it in the cache directory
        if (!cache_dir.empty()) {
            std::string cache_path = get_cache_path(model_path, cache_dir);
            if (std::filesystem::exists(cache_path)) {
                return cache_path;
            }
        }

        // Try default cache location
        std::string default_cache = get_cache_path(model_path, "");
        if (std::filesystem::exists(default_cache)) {
            return default_cache;
        }

        return "";
    }

    std::string get_cache_path(const std::string& model_id, const std::string& cache_dir) {
        std::string base_dir;
        if (cache_dir.empty()) {
            const char* home = std::getenv("HOME");
            base_dir = home ? std::string(home) + "/.cache/cortexstream" : ".cache/cortexstream";
        } else {
            base_dir = cache_dir;
        }

        // Create a filesystem-safe name from the model ID
        std::string safe_name = model_id;
        std::replace(safe_name.begin(), safe_name.end(), '/', '_');

        return base_dir + "/" + safe_name + "/tokenizer.json";
    }

    std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
    bool loaded_ = false;
};

// Factory function
std::unique_ptr<Tokenizer> createTokenizer(
    const std::string& model_path_or_id,
    const std::string& cache_dir) {
    try {
        return std::make_unique<HuggingFaceTokenizer>(model_path_or_id, cache_dir);
    } catch (const std::exception& e) {
        std::cerr << "Error creating tokenizer: " << e.what() << std::endl;
        return nullptr;
    }
}

} // namespace cortexstream
