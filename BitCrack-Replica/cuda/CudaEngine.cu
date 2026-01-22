#include "CudaEngine.cuh"
#include "secp256k1.cuh"
#include "sha256.cuh"
#include "ripemd160.cuh"

namespace Replica {

// PTX Optimized Math Wrappers mapping to secp256k1.cuh functions
// Note: secp256k1.cuh defines addModP, etc as __device__ static. 
// We can use them directly inside our device functions.

// Implementation of Point Arithmetic (Affine Coordinates)
// Uses invModP which is expensive, but simple to implement for random point generation.
// For faster implementations, Projective coordinates should be used.

__device__ void copyPoint(const Point& src, Point& dst) {
    for(int i=0; i<8; i++) {
        dst.x[i] = src.x[i];
        dst.y[i] = src.y[i];
    }
}

__device__ void setInfinity(Point& p) {
    for(int i=0; i<8; i++) {
        p.x[i] = 0xFFFFFFFF;
        p.y[i] = 0xFFFFFFFF;
    }
}

__device__ bool isInf(const Point& p) {
    return p.x[0] == 0xFFFFFFFF && p.x[1] == 0xFFFFFFFF && p.x[2] == 0xFFFFFFFF && p.x[3] == 0xFFFFFFFF &&
           p.x[4] == 0xFFFFFFFF && p.x[5] == 0xFFFFFFFF && p.x[6] == 0xFFFFFFFF && p.x[7] == 0xFFFFFFFF;
}


__device__ void doublePoint(const Point& p, Point& out) {
    if (isInf(p)) { setInfinity(out); return; }

    unsigned int lambda[8];
    unsigned int tx[8], ty[8];
    unsigned int t1[8], t2[8];

    // lambda = (3x^2) * (2y)^-1
    squareModP(p.x, t1); // x^2
    mulModP(t1, _LAMBDA, t2); // 3x^2? No _LAMBDA is likely unrelated constant. 
    // We compute 3*x^2 manually
    addModP(t1, t1, t2); // 2x^2
    addModP(t2, t1, t1); // 3x^2

    addModP(p.y, p.y, t2); // 2y
    invModP(t2); // (2y)^-1
    mulModP(t1, t2, lambda);

    // x3 = lambda^2 - 2x
    squareModP(lambda, t1); // lambda^2
    addModP(p.x, p.x, t2); // 2x
    subModP(t1, t2, out.x);

    // y3 = lambda(x - x3) - y
    subModP(p.x, out.x, t1);
    mulModP(lambda, t1, t2);
    subModP(t2, p.y, out.y);
}

__device__ void addPoints(const Point& p1, const Point& p2, Point& out) {
    if (isInf(p1)) { copyPoint(p2, out); return; }
    if (isInf(p2)) { copyPoint(p1, out); return; }

    if (equal(p1.x, p2.x)) {
        if (equal(p1.y, p2.y)) {
             doublePoint(p1, out);
             return;
        } else {
             setInfinity(out);
             return;
        }
    }

    unsigned int lambda[8];
    unsigned int t1[8], t2[8];

    // lambda = (y2 - y1) * (x2 - x1)^-1
    subModP(p2.y, p1.y, t1);
    subModP(p2.x, p1.x, t2);
    invModP(t2);
    mulModP(t1, t2, lambda);

    // x3 = lambda^2 - x1 - x2
    squareModP(lambda, t1);
    subModP(t1, p1.x, t2);
    subModP(t2, p2.x, out.x);

    // y3 = lambda(x1 - x3) - y1
    subModP(p1.x, out.x, t1);
    mulModP(lambda, t1, t2);
    subModP(t2, p1.y, out.y);
}

__device__ void mulPoint(const unsigned int scalar[8], Point& out) {
    Point R;
    setInfinity(R);
    Point G;
    // _GX, _GY are constants in secp256k1.cuh
    copyBigInt(_GX, G.x);
    copyBigInt(_GY, G.y);

    Point tempG;
    copyPoint(G, tempG);

    // Double and Add
    // Iterate bits from 0 to 255.
    // Scalar is little endian in algorithm? Usually definitions are big endian arrays.
    // _GX is Big Endian. scalar is Big Endian array.
    // We should process from High Bit to Low bit, or Low Bit to High Bit?
    // Standard Double-and-Add LSB to MSB:
    // for i = 0 to 255:
    //   if bit set: R = R + tempG
    //   tempG = 2 * tempG
    
    // Arrays are u32[8], index 0 is MSB? 
    // Usually BitCrack uses index 0 as MSB (Big Endian). 0xFF... at index 0.
    // So bit 0 is in scalar[7] & 1.
    
    for (int i = 0; i < 256; i++) {
        // Get bit at position i (0 = LSB)
        int wordIdx = 7 - (i / 32);
        int bitIdx = i % 32;
        bool bit = (scalar[wordIdx] >> bitIdx) & 1;

        if (bit) {
            Point nextR;
            addPoints(R, tempG, nextR);
            copyPoint(nextR, R);
        }

        Point nextTemp;
        doublePoint(tempG, nextTemp);
        copyPoint(nextTemp, tempG);
    }
    copyPoint(R, out);
}

// Simple Xorshift RNG
__device__ unsigned int xorshift32(unsigned int& state) {
    unsigned int x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;
    return x;
}

// Hashing Helpers
__device__ void hashPublicKey(const unsigned int* x, const unsigned int* y, unsigned int* digestOut) {
    unsigned int hash[8];
    // sha256PublicKey implementation in secp256k1.cuh? No, it's in sha256.cuh likely, but might not be exposed as function taking raw pointers nicely.
    // CudaKeySearchDevice calls sha256PublicKey(x, y, hash);
    // Let's assume sha256.cuh has it.
    
    // Actually sha256.cuh usually defines global functions or device functions.
    // I need to check if I can call them. 
    // Reference CudaKeySearchDevice.cu creates its own wrapper `hashPublicKey`.
    // I will copy that wrapper.
    
    sha256PublicKey(x, y, hash);

    // Swap to little-endian for RIPEMD
    for(int i = 0; i < 8; i++) {
        hash[i] = endian(hash[i]);
    }

    ripemd160sha256NoFinal(hash, digestOut);
}

__global__ void searchKernel(CudaMask mask, unsigned int* targetHash, SearchResult* result, uint64_t seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int rngState = (unsigned int)(seed + tid) ^ 0x9e3779b9; // simple seed mix

    // 1. Generate Private Key
    unsigned int privKey[8];
    for (int i = 0; i < 8; i++) {
        unsigned int r = xorshift32(rngState);
        // mask.fixed has value, mask.mask has 0xFF if wildcard.
        // If wildcard (mask=0xFF), use r. If fixed (mask=0x00), use fixed.
        // Wait, main.cpp: if wildcard mask=FF, else 00.
        // result = (fixed & const_mask) | (r & wildcard_mask)
        // My Logic: FIXED | (R & MASK)
        // If mask is 0, (R & 0) -> 0. Fixed | 0 -> Fixed.
        // If mask is FF, (R & FF) -> R. Fixed | R -> R (assuming fixed is 0 where mask is FF).
        // main.cpp sets fixed to 0 where wildcard is true. So Fixed | R works.
        
        privKey[i] = mask.fixed[i] | (r & mask.mask[i]);
    }

    // 2. Compute Public Key
    Point P;
    mulPoint(privKey, P);

    // 3. Hash
    unsigned int digest[5];
    // Compressed or Uncompressed?
    // BitCrack checks both usually.
    // Let's check uncompressed for simplicity or both.
    // I'll check uncompressed first.
    
    hashPublicKey(P.x, P.y, digest);
    
    // 4. Compare
    bool match = true;
    for (int i=0; i<5; i++) {
        if (digest[i] != targetHash[i]) {
            match = false;
            break;
        }
    }
    
    if (match) {
        result->found = true;
        for(int i=0; i<8; i++) result->privKey[i] = privKey[i];
        for(int i=0; i<8; i++) result->x[i] = P.x[i];
        for(int i=0; i<8; i++) result->y[i] = P.y[i];
    }
}

void launchCudaSearch(int blocks, int threads, CudaMask mask, unsigned int* d_targetHash, SearchResult* d_result, uint64_t seed) {
    searchKernel<<<blocks, threads>>>(mask, d_targetHash, d_result, seed);
    cudaDeviceSynchronize();
}

} // namespace Replica
