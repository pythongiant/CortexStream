#include "cortexstream/sampler.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <numeric>

namespace cortexstream {

Sampler::Sampler(uint32_t seed) : seed(seed) {
}

Sampler::~Sampler() = default;

int Sampler::greedy(const Tensor& logits) {
    if (logits.shape.empty() || logits.data.empty()) {
        return 0;
    }
    
    int maxIdx = 0;
    float maxVal = logits.data[0];
    
    for (size_t i = 1; i < logits.data.size(); ++i) {
        if (logits.data[i] > maxVal) {
            maxVal = logits.data[i];
            maxIdx = i;
        }
    }
    
    return maxIdx;
}

int Sampler::topK(const Tensor& logits, int k, float temperature) {
    if (logits.shape.empty() || logits.data.empty()) {
        return 0;
    }
    
    auto topKIdx = topKIndices(logits, k);
    
    // Apply temperature and softmax
    std::vector<float> scores;
    scores.reserve(topKIdx.size());
    
    float maxScore = topKIdx[0].first;
    for (const auto& [logit, idx] : topKIdx) {
        scores.push_back(logit / temperature);
    }
    
    auto probs = softmax(scores, temperature);
    return sampleFromDistribution(probs);
}

int Sampler::topP(const Tensor& logits, float p, float temperature) {
    if (logits.shape.empty() || logits.data.empty()) {
        return 0;
    }
    
    // Sort all logits
    std::vector<std::pair<float, int>> scored;
    scored.reserve(logits.data.size());
    
    for (size_t i = 0; i < logits.data.size(); ++i) {
        scored.emplace_back(logits.data[i], i);
    }
    
    std::sort(scored.begin(), scored.end(),
             [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Find nucleus tokens (top-p)
    std::vector<float> scores;
    std::vector<int> indices;
    
    float cumProb = 0.0f;
    float maxScore = scored[0].first;
    float denom = 0.0f;
    
    for (const auto& [logit, idx] : scored) {
        float score = (logit - maxScore) / temperature;
        float expScore = std::exp(score);
        denom += expScore;
        
        scores.push_back(expScore);
        indices.push_back(idx);
        
        cumProb += expScore;
        
        // Stop when cumulative probability exceeds p
        if (cumProb > p * (cumProb + std::exp((scored.back().first - maxScore) / temperature))) {
            break;
        }
    }
    
    // Normalize probabilities
    for (auto& s : scores) {
        s /= denom;
    }
    
    return sampleFromDistribution(scores);
}

int Sampler::topKP(const Tensor& logits, int k, float p, float temperature) {
    if (logits.shape.empty() || logits.data.empty()) {
        return 0;
    }
    
    auto topKIdx = topKIndices(logits, k);
    
    // Apply top-p within top-k
    std::vector<float> scores;
    float maxScore = topKIdx[0].first;
    
    for (const auto& [logit, idx] : topKIdx) {
        float score = (logit - maxScore) / temperature;
        scores.push_back(std::exp(score));
    }
    
    // Normalize
    float sum = 0.0f;
    for (auto s : scores) sum += s;
    for (auto& s : scores) s /= sum;
    
    // Apply top-p threshold
    std::vector<std::pair<float, int>> probs;
    for (size_t i = 0; i < scores.size(); ++i) {
        probs.emplace_back(scores[i], i);
    }
    
    std::sort(probs.begin(), probs.end(),
             [](const auto& a, const auto& b) { return a.first > b.first; });
    
    float cumProb = 0.0f;
    std::vector<float> finalProbs;
    for (const auto& [prob, i] : probs) {
        cumProb += prob;
        finalProbs.push_back(prob);
        if (cumProb > p) break;
    }
    
    // Renormalize
    float finalSum = 0.0f;
    for (auto p : finalProbs) finalSum += p;
    for (auto& p : finalProbs) p /= finalSum;
    
    return sampleFromDistribution(finalProbs);
}

void Sampler::setSeed(uint32_t newSeed) {
    seed = newSeed;
}

std::vector<std::pair<float, int>> Sampler::topKIndices(const Tensor& logits, int k) {
    std::vector<std::pair<float, int>> scored;
    scored.reserve(logits.data.size());
    
    for (size_t i = 0; i < logits.data.size(); ++i) {
        scored.emplace_back(logits.data[i], i);
    }
    
    k = std::min(k, static_cast<int>(scored.size()));
    
    std::nth_element(scored.begin(), scored.begin() + k, scored.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });
    
    std::vector<std::pair<float, int>> result(scored.begin(), scored.begin() + k);
    std::sort(result.begin(), result.end(),
             [](const auto& a, const auto& b) { return a.first > b.first; });
    
    return result;
}

std::vector<float> Sampler::softmax(const std::vector<float>& logits, float temperature) {
    if (logits.empty()) {
        return {};
    }
    
    std::vector<float> result;
    result.reserve(logits.size());
    
    float maxLogit = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;
    
    for (float logit : logits) {
        float exp_val = std::exp((logit - maxLogit) / temperature);
        result.push_back(exp_val);
        sum += exp_val;
    }
    
    for (auto& val : result) {
        val /= sum;
    }
    
    return result;
}

int Sampler::sampleFromDistribution(const std::vector<float>& probs) {
    if (probs.empty()) {
        return 0;
    }
    
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    float r = dis(gen);
    float cumProb = 0.0f;
    
    for (size_t i = 0; i < probs.size(); ++i) {
        cumProb += probs[i];
        if (r < cumProb) {
            return i;
        }
    }
    
    return probs.size() - 1;
}

}  // namespace cortexstream

