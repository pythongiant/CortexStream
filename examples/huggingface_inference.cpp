// HuggingFace Model Integration Example
// Demonstrates loading popular HuggingFace models and serving them via CortexStream
//
// Supported models:
// - Meta Llama 2: "meta-llama/Llama-2-7b", "meta-llama/Llama-2-13b"
// - Mistral AI: "mistralai/Mistral-7B", "mistralai/Mistral-7B-Instruct"
// - Microsoft Phi: "microsoft/phi-2"
// - OpenChat: "openchat/openchat-3.5"
// - Zephyr: "HuggingFaceH4/zephyr-7b-beta"

#include "cortexstream/engine.h"
#include "cortexstream/model.h"
#include "cortexstream/scheduler.h"
#include "cortexstream/kv_cache.h"
#include "cortexstream/request.h"
#include <iostream>
#include <vector>
#include <memory>
#include <string>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>

#if defined(CORTEXSTREAM_WITH_TOKENIZERS_CPP)
#include <tokenizers_cpp.h>
#endif

using namespace cortexstream;

static bool readFileToString(const std::filesystem::path& path, std::string* out) {
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs) {
        return false;
    }
    std::ostringstream ss;
    ss << ifs.rdbuf();
    *out = ss.str();
    return true;
}

static std::optional<std::filesystem::path> findTokenizerFile(
    const std::filesystem::path& cacheDir,
    const std::string& modelId) {

    std::error_code ec;
    std::vector<std::filesystem::path> roots;
    roots.push_back(cacheDir / modelId);
    roots.push_back(cacheDir);

    for (const auto& root : roots) {
        if (!std::filesystem::exists(root, ec)) {
            continue;
        }

        if (std::filesystem::is_regular_file(root, ec)) {
            auto filename = root.filename().string();
            if (filename == "tokenizer.json" || filename == "tokenizer.model") {
                return root;
            }
        }

        if (!std::filesystem::is_directory(root, ec)) {
            continue;
        }

        for (auto it = std::filesystem::recursive_directory_iterator(root, ec);
             it != std::filesystem::recursive_directory_iterator();
             it.increment(ec)) {

            if (ec) {
                break;
            }

            const auto& p = it->path();
            auto filename = p.filename().string();
            if (filename == "tokenizer.json" || filename == "tokenizer.model") {
                return p;
            }
        }
    }

    return std::nullopt;
}

int main(int argc, char* argv[]) {
    std::cout << "=== CortexStream HuggingFace Model Inference ===" << std::endl;
    
    // Get model ID from command line or use default
    std::string modelId = "mistralai/Mistral-7B";
    if (argc > 1) {
        modelId = argv[1];
    }

    std::string cacheDir = "./models";
    if (argc > 2) {
        cacheDir = argv[2];
    }
    
    std::cout << "\n[Model] Loading HuggingFace model: " << modelId << std::endl;
    
    // 1. Create ModelBackend with HuggingFace support
    // On first run: Downloads weights from huggingface.co and converts to MLX
    // On subsequent runs: Uses cached MLX weights (much faster)
    auto backend = std::make_shared<ModelBackend>(Device::MPS, DType::FP16);
    
    // Load model (expects pre-converted MLX file or accessible path)
    if (!backend->loadModel(modelId)) {
        std::cerr << "❌ Failed to load model: " << modelId << std::endl;
        std::cerr << "   Make sure you have:" << std::endl;
        std::cerr << "   1. Internet connection (for downloading)" << std::endl;
        std::cerr << "   2. Enough disk space (~15GB for 7B model)" << std::endl;
        std::cerr << "   3. HuggingFace token (for gated models)" << std::endl;
        return 1;
    }
    
    std::cout << "✅ Model loaded successfully" << std::endl;
    std::cout << "   Architecture: " << backend->getNumLayers() << " layers, "
              << backend->getHiddenSize() << " hidden size, "
              << backend->getVocabSize() << " vocab size" << std::endl;

#if defined(CORTEXSTREAM_WITH_TOKENIZERS_CPP)
    std::unique_ptr<tokenizers::Tokenizer> tokenizer;
    {
        std::cout << "\n[Tokenizer] Loading tokenizer (cacheDir=" << cacheDir << ")..." << std::endl;
        auto tokPathOpt = findTokenizerFile(std::filesystem::path(cacheDir), modelId);
        if (!tokPathOpt) {
            std::cout << "⚠️  Tokenizer not found under cache directory; responses will be shown as token IDs." << std::endl;
            std::cout << "   Tip: pass cache dir as second argument: ./huggingface_inference \"" << modelId
                      << "\" \"./models\"" << std::endl;
        } else {
            const auto& tokPath = *tokPathOpt;
            std::string blob;
            if (!readFileToString(tokPath, &blob)) {
                std::cout << "⚠️  Failed to read tokenizer file: " << tokPath.string() << std::endl;
            } else {
                if (tokPath.filename() == "tokenizer.json") {
                    tokenizer = tokenizers::Tokenizer::FromBlobJSON(blob);
                } else if (tokPath.filename() == "tokenizer.model") {
                    tokenizer = tokenizers::Tokenizer::FromBlobSentencePiece(blob);
                }

                if (!tokenizer) {
                    std::cout << "⚠️  Tokenizer load failed (unsupported tokenizer format)." << std::endl;
                } else {
                    std::cout << "✅ Tokenizer loaded: " << tokPath.string() << std::endl;
                }
            }
        }
    }
#endif
    
    // 2. Initialize components
    std::cout << "\n[Setup] Initializing inference pipeline..." << std::endl;
    
    // Create scheduler (batch size 32 for 7B model on Apple Silicon)
    // Adjust down for 13B+ or up for 3B models
    auto scheduler = std::make_shared<Scheduler>(32);
    
    // Create KV cache (sized for max tokens and layer count)
    // For Llama/Mistral: 4096 hidden, 32 layers
    // Adjust cache size based on your device:
    // - M1: ~8GB
    // - M2: ~16GB
    // - M3 Max: ~32GB
    size_t cacheSize = 8UL * 1024 * 1024 * 1024;  // 8GB default
    auto cache = std::make_shared<KVCache>(
        cacheSize,
        backend->getHiddenSize(),
        backend->getNumLayers()
    );
    
    // Create inference engine
    auto engine = std::make_shared<InferenceEngine>(backend, scheduler, cache);
    if (!engine->initialize()) {
        std::cerr << "❌ Failed to initialize inference engine" << std::endl;
        return 1;
    }
    
    std::cout << "✅ Pipeline ready" << std::endl;
    
    // 3. Submit inference requests
    std::cout << "\n[Inference] Processing requests..." << std::endl;
    
    std::vector<std::shared_ptr<Request>> requests;
    
    // Example prompts for different use cases
    std::vector<std::string> prompts = {
        "What is machine learning?",
        "Explain quantum computing in simple terms.",
        "Write a Python function to sort a list."
    };
    
    // Submit requests
    for (size_t i = 0; i < prompts.size(); ++i) {
        auto req = std::make_shared<Request>(
            "req_" + std::to_string(i),
            prompts[i],
            256  // max tokens to generate
        );
        
        // Configure sampling for better quality
        SamplingParams samplingParams;
        samplingParams.temperature = 0.7f;  // Balanced creativity
        samplingParams.topP = 0.9f;         // Nucleus sampling
        samplingParams.topK = 40;            // Restrict to top 40 tokens
        req->setSamplingParams(samplingParams);
        
        scheduler->submitRequest(req);
        requests.push_back(req);
        
        std::cout << "  Request " << i << ": " << prompts[i] << std::endl;
    }
    
    // 4. Process requests
    std::cout << "\n[Processing] Running inference..." << std::endl;
    std::cout << "GPU acceleration: Metal (MPS) on Apple Silicon" << std::endl;
    std::cout << "Batch processing: Up to 32 sequences in parallel" << std::endl;
    
    // Run inference (this would normally be asynchronous)
    engine->run();
    
    // 5. Collect results
    std::cout << "\n[Results] Generated completions:" << std::endl;
    
    for (size_t i = 0; i < requests.size(); ++i) {
        const auto& req = requests[i];
        std::cout << "\n--- Request " << i << " ---" << std::endl;
        std::cout << "Prompt: " << prompts[i] << std::endl;
        
        std::cout << "Tokens generated: " << req->getGeneratedLength() << std::endl;
        if (req->getState() == RequestState::Finished) {
            std::cout << "Status: ✅ Completed" << std::endl;
        } else if (req->getState() == RequestState::Failed) {
            std::cout << "Status: ❌ Failed" << std::endl;
            std::cout << "Error: " << req->getErrorMessage() << std::endl;
        } else {
            std::cout << "Status: ⏳ In progress" << std::endl;
        }

        if (req->getState() == RequestState::Finished) {
            const auto& gen = req->getGeneratedTokens();

#if defined(CORTEXSTREAM_WITH_TOKENIZERS_CPP)
            if (tokenizer) {
                std::vector<int32_t> ids;
                ids.reserve(gen.size());
                for (int t : gen) {
                    ids.push_back(static_cast<int32_t>(t));
                }
                std::string decoded = tokenizer->Decode(ids);
                std::cout << "\nResponse:\n" << decoded << std::endl;
            } else
#endif
            {
                std::cout << "\nResponse (token IDs; build with -DWITH_TOKENIZERS_CPP=ON to decode):\n";
                std::cout << "[";
                size_t limit = std::min<size_t>(gen.size(), 64);
                for (size_t j = 0; j < limit; ++j) {
                    if (j) std::cout << ' ';
                    std::cout << gen[j];
                }
                if (gen.size() > limit) {
                    std::cout << " ...";
                }
                std::cout << "]" << std::endl;
            }
        }
    }
    
    // 6. Print statistics
    const auto& stats = engine->getStats();
    std::cout << "\n[Statistics]" << std::endl;
    std::cout << "Total tokens processed: " << stats.tokensProcessed << std::endl;
    std::cout << "Requests completed: " << stats.requestsCompleted << std::endl;
    std::cout << "Failed requests: " << stats.requestsFailed << std::endl;
    std::cout << "Average batch size: " << stats.avgBatchSize << std::endl;
    
    // 7. Cleanup
    engine->shutdown();
    
    std::cout << "\n✅ Inference completed" << std::endl;
    
    return 0;
}

// USAGE EXAMPLES:
//
// Load Mistral-7B (default):
//   ./huggingface_inference
//
// Load Llama 2 7B:
//   ./huggingface_inference "meta-llama/Llama-2-7b"
//
// Load Phi-2 (smaller, faster):
//   ./huggingface_inference "microsoft/phi-2"
//
// Load with custom cache directory:
//   ./huggingface_inference "mistralai/Mistral-7B" /path/to/cache
//
// FIRST RUN NOTES:
// - Model download: 5-20 minutes (depends on internet speed)
// - MLX conversion: 5-10 minutes (quantization + optimization)
// - Subsequent runs: Model loads from cache in <1 second
//
// MODEL SIZE GUIDE:
// - 3B models: ~6GB disk, ~2GB VRAM - good for M1
// - 7B models: ~14GB disk, ~4GB VRAM - good for M1/M2
// - 13B models: ~26GB disk, ~8GB VRAM - need M2 Pro or M3 Max
// - 70B models: Not recommended without 64GB+ unified memory
