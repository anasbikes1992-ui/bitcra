#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <iomanip>
#include <cstdint>
#include <cuda_runtime.h>
#include "Logger.hpp"
#include "../cuda/CudaEngine.cuh"
#include "PatternManager.hpp"

using namespace Replica;

void print_splash() {
    std::cout << "\033[1;36m";
    std::cout << R"(
    ██████╗ ██╗████████╗ ██████╗██████╗  █████╗  ██████╗██╗  ██╗
    ██╔══██╗██║╚══██╔══╝██╔════╝██╔══██╗██╔══██╗██╔════╝██║ ██╔╝
    ██████╔╝██║   ██║   ██║     ██████╔╝███████║██║     █████╔╝ 
    ██╔══██╗██║   ██║   ██║     ██╔══██╗██╔══██║██║     ██╔═██╗ 
    ██████╔╝██║   ██║   ╚██████╗██║  ██║██║  ██║╚██████╗██║  ██╗
    ╚═════╝ ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
                    REPLICA - Pattern Mode                      
    )" << std::endl;
    std::cout << "\033[0m" << std::endl;
}

int main(int argc, char* argv[]) {
    print_splash();

    if (argc < 3) {
        Logger::warn("Usage: Replica.exe <mode: --cpu or --cuda> <target_address> <--pattern mask>");
        Logger::info("Example: Replica.exe --cpu 1B1fz6... --pattern DE????AD");
        Logger::info("The '?' characters in the mask will be filled with random hex values.");
        return 1;
    }

    std::string mode = argv[1];
    std::string target = argv[2];
    std::string mask = "????????????????????????????????????????????????????????????????"; // Default random

    for (int i = 3; i < argc; ++i) {
        if (std::string(argv[i]) == "--pattern" && i + 1 < argc) {
            mask = argv[i + 1];
        }
    }

    // Mock target hash (160-bit)
    unsigned int h_target[5] = {0x12345678, 0x9ABCDEF0, 0x01234567, 0x89ABCDEF, 0xFEDCBA98};

    if (mode == "--cuda") {
        Logger::info("Mode: CUDA Pattern Search selected.");
        
        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount == 0) {
            Logger::error("No CUDA-capable devices found!");
            return 1;
        }

        const int numThreads = 256;
        const int numBlocks = 64;
        const int totalPoints = numThreads * numBlocks;

        // Parse mask for CUDA
        CudaMask deviceMask = {0};
        auto parts = PatternManager::parseMask(mask);
        unsigned char fixedBytes[32] = {0};
        unsigned char maskBytes[32] = {0};
        
        for (size_t i = 0; i < 32; ++i) {
            if (i < parts.size()) {
                if (parts[i].isWildcard) {
                    maskBytes[i] = 0xFF;
                } else {
                    fixedBytes[i] = parts[i].value;
                }
            } else {
                maskBytes[i] = 0xFF; // Padding is wildcard
            }
        }

        // Pack into uint32 arrays (Big Endian)
        for (int i = 0; i < 8; ++i) {
            deviceMask.fixed[i] = (fixedBytes[i*4] << 24) | (fixedBytes[i*4+1] << 16) | (fixedBytes[i*4+2] << 8) | fixedBytes[i*4+3];
            deviceMask.mask[i] = (maskBytes[i*4] << 24) | (maskBytes[i*4+1] << 16) | (maskBytes[i*4+2] << 8) | maskBytes[i*4+3];
        }

        Point* d_points;
        SearchResult* d_result;
        unsigned int* d_targetHash;

        cudaMalloc(&d_points, totalPoints * sizeof(Point));
        cudaMalloc(&d_result, sizeof(SearchResult));
        cudaMalloc(&d_targetHash, 5 * sizeof(unsigned int));

        SearchResult h_result = {false};
        cudaMemcpy(d_result, &h_result, sizeof(SearchResult), cudaMemcpyHostToDevice);
        cudaMemcpy(d_targetHash, h_target, 5 * sizeof(unsigned int), cudaMemcpyHostToDevice);

        auto startTime = std::chrono::high_resolution_clock::now();
        auto lastTime = startTime;
        uint64_t lastKeys = 0;
        uint64_t totalKeys = 0;

        while (!h_result.found) {
             // seed offset based on totalKeys
            launchCudaSearch(numBlocks, numThreads, deviceMask, d_targetHash, d_result, totalKeys);
            cudaMemcpy(&h_result, d_result, sizeof(SearchResult), cudaMemcpyDeviceToHost);
            totalKeys += totalPoints;
            
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastTime).count();
            if (elapsed > 1000) {
                double speed = (double)(totalKeys - lastKeys) / (elapsed / 1000.0) / 1000000.0;
                std::cout << "\r" << "CUDA Searching [Pattern]... | Keys: " << totalKeys << " | Speed: " << std::fixed << std::setprecision(2) << speed << " M/s" << std::flush;
                lastTime = now;
                lastKeys = totalKeys;
            }
            
            if (totalKeys > 10000000000ULL) { // Increased limit
                 Logger::info("\nReached 10B keys (Limit).");
                 break;
            }
        }
        
        std::cout << "\n";
        
        cudaFree(d_points);
        cudaFree(d_result);
        cudaFree(d_targetHash);
        
    } else if (mode == "--cpu") {
        int threads = std::thread::hardware_concurrency();
        runCpuSearch(threads, mask, h_target);
    } else {
        Logger::error("Invalid mode. Use --cpu or --cuda.");
        return 1;
    }

    Logger::success("Operation finished.");
    return 0;
}
