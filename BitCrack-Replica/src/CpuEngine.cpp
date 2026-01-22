#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <iomanip>
#include <chrono>
#include <cstdint>
#include "CudaEngine.cuh"
#include "Logger.hpp"
#include "PatternManager.hpp"

namespace Replica {

void runCpuSearch(int numThreads, const std::string& mask, unsigned int* targetHash) {
    auto maskParts = PatternManager::parseMask(mask);
    std::atomic<uint64_t> totalKeysChecked(0);
    std::atomic<bool> found(false);
    
    Logger::info("Starting Pattern Search: " + mask + " using " + std::to_string(numThreads) + " threads.");

    std::vector<std::jthread> threads;
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([&, i]() {
            unsigned int privKey[8];
            while (!found) {
                // Generate a random key based on the mask
                PatternManager::fillRandom(maskParts, privKey);
                
                // Simulation of ECC check
                if (privKey[0] == targetHash[0]) { 
                    found = true;
                    Logger::found("Match found by CPU Thread " + std::to_string(i));
                }
                
                totalKeysChecked.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    auto start = std::chrono::high_resolution_clock::now();
    while (!found) {
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
        if (elapsed > 0) {
            float speed = (float)totalKeysChecked / elapsed / 1000000.0f;
            std::cout << "\r" << "CPU Searching [Pattern]... | Speed: " << std::fixed << std::setprecision(2) << speed << " MKey/s | Total: " << totalKeysChecked.load() << std::flush;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    std::cout << std::endl;
}

} // namespace Replica
