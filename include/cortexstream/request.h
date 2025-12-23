#ifndef CORTEXSTREAM_REQUEST_H
#define CORTEXSTREAM_REQUEST_H

#include <string>
#include <memory>
#include <vector>
#include <functional>
#include <chrono>

namespace cortexstream {

enum class RequestState {
    Pending,      // Waiting to be scheduled
    Prefilling,   // Prompt processing
    Decoding,     // Token generation
    Finished,     // Completed
    Failed        // Error
};

struct SamplingParams {
    float temperature = 1.0f;
    int topK = 40;
    float topP = 0.9f;
    bool greedy = false;
    uint32_t seed = 0;
};

// Token completion callback
using TokenCallback = std::function<void(int token, bool finished)>;

class Request {
public:
    explicit Request(const std::string& id,
                     const std::vector<int>& promptTokens,
                     int maxTokens = 512);
    ~Request();

    // Getters
    const std::string& getId() const;
    const std::vector<int>& getPromptTokens() const;
    const std::vector<int>& getGeneratedTokens() const;
    RequestState getState() const;
    int getMaxTokens() const;
    const SamplingParams& getSamplingParams() const;
    std::chrono::system_clock::time_point getCreationTime() const;

    // State management
    void setState(RequestState state);
    void addToken(int token);
    void setSamplingParams(const SamplingParams& params);
    void setTokenCallback(TokenCallback callback);
    
    // Helper
    bool isFinished() const;
    bool isFailed() const;
    int getPromptLength() const;
    int getGeneratedLength() const;

private:
    std::string id;
    std::vector<int> promptTokens;
    std::vector<int> generatedTokens;
    RequestState state = RequestState::Pending;
    int maxTokens;
    SamplingParams samplingParams;
    TokenCallback tokenCallback;
    std::chrono::system_clock::time_point creationTime;
};

}  // namespace cortexstream

#endif  // CORTEXSTREAM_REQUEST_H
