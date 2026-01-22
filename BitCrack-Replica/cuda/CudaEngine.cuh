#ifndef _CUDA_ENGINE_CUH
#define _CUDA_ENGINE_CUH

#include <cuda_runtime.h>
#include <cstdint>

#ifdef __CUDACC__
    #define DEVICE_CODE __device__
    #define HOST_DEVICE __host__ __device__
    #define INLINE_CODE __forceinline__
    #define GLOBAL_CODE __global__
#else
    #define DEVICE_CODE
    #define HOST_DEVICE
    #define INLINE_CODE inline
    #define GLOBAL_CODE
#endif

namespace Replica {

struct Point {
    unsigned int x[8];
    unsigned int y[8];
};

struct SearchResult {
    bool found;
    unsigned int privKey[8];
    unsigned int x[8];
    unsigned int y[8];
};

struct CudaMask {
    unsigned int fixed[8];
    unsigned int mask[8];
};

// ECC operations
HOST_DEVICE void addPoints(const Point& p1, const Point& p2, Point& out);
HOST_DEVICE void doublePoint(const Point& p, Point& out);

// CUDA specific search kernel
#ifdef __CUDACC__
GLOBAL_CODE void searchKernel(CudaMask mask, unsigned int* targetHash, SearchResult* result, uint64_t seed);
#endif

#include <string>

// Host wrappers for orchestration
void launchCudaSearch(int blocks, int threads, CudaMask mask, unsigned int* d_targetHash, SearchResult* d_result, uint64_t seed);
void runCpuSearch(int numThreads, const std::string& mask, unsigned int* targetHash);

} // namespace Replica

#endif
