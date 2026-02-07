#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

static inline void ck(cudaError_t e, const char* msg){
    if(e != cudaSuccess){
        std::fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(e));
        std::exit(1);
    }
}

__host__ __device__ static inline uint32_t hash_u32(uint32_t x){
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}
__host__ __device__ static inline float u32_to_float_signed(uint32_t x){
    uint32_t m = x & 0x00FFFFFFU;
    float f = (float)m / (float)0x00800000U;
    return f - 1.0f;
}
__device__ static inline float gen_input_dev(int b, int d){
    uint32_t idx = (uint32_t)(b * 1315423911u) ^ (uint32_t)(d * 2654435761u) ^ 0xA5A5A5A5u;
    return 0.5f * u32_to_float_signed(hash_u32(idx));
}
__device__ static inline float relu(float x){ return x > 0.f ? x : 0.f; }

// Layout (transposed for memory coalescing):
// W1T: [D][H] contiguous in H
// W2T: [H][C] contiguous in C

// Tile sizes
#ifndef BM
#define BM 16
#endif
#ifndef BN
#define BN 16
#endif
#ifndef BK
#define BK 32
#endif

// hidden[b,h] = relu( sum_d x[b,d]*W1T[d,h] + b1[h] )
__global__ void hidden_tiled(float* __restrict__ hidden,
                             const float* __restrict__ W1T,
                             const float* __restrict__ b1,
                             int B, int D, int H, int b_base)
{
    __shared__ float Xs[BM][BK];
    __shared__ float Ws[BK][BN];

    int b = blockIdx.y * BM + threadIdx.y;  // row in batch
    int h = blockIdx.x * BN + threadIdx.x;  // col in hidden

    float acc = 0.f;

    for(int k0=0; k0<D; k0+=BK){
        // load X tile (generate on the fly)
        int d = k0 + threadIdx.x; // use x-thread to load columns
        if(threadIdx.y < BM && threadIdx.x < BK){
            if(b < B && d < D) Xs[threadIdx.y][threadIdx.x] = gen_input_dev(b_base + b, d);
            else Xs[threadIdx.y][threadIdx.x] = 0.f;
        }

        // load W tile
        int dk = k0 + threadIdx.y; // use y-thread to load rows
        if(threadIdx.y < BK && threadIdx.x < BN){
            if(dk < D && h < H) Ws[threadIdx.y][threadIdx.x] = W1T[(size_t)dk * H + h];
            else Ws[threadIdx.y][threadIdx.x] = 0.f;
        }

        __syncthreads();

        if(b < B && h < H){
            #pragma unroll
            for(int k=0; k<BK; k++){
                acc = fmaf(Xs[threadIdx.y][k], Ws[k][threadIdx.x], acc);
            }
        }

        __syncthreads();
    }

    if(b < B && h < H){
        acc += b1[h];
        hidden[(size_t)b * H + h] = relu(acc);
    }
}

// out[b,c] = sum_h hidden[b,h]*W2T[h,c] + b2[c] ; and accumulate checksum
__global__ void out_tiled_checksum(const float* __restrict__ hidden,
                                   const float* __restrict__ W2T,
                                   const float* __restrict__ b2,
                                   int B, int H, int C,
                                   double* __restrict__ checksum)
{
    __shared__ float Hs[BM][BK];
    __shared__ float Ws[BK][BN];

    int b = blockIdx.y * BM + threadIdx.y;
    int c = blockIdx.x * BN + threadIdx.x;

    float acc = 0.f;

    for(int k0=0; k0<H; k0+=BK){
        // load hidden tile
        int h = k0 + threadIdx.x;
        if(threadIdx.y < BM && threadIdx.x < BK){
            if(b < B && h < H) Hs[threadIdx.y][threadIdx.x] = hidden[(size_t)b * H + h];
            else Hs[threadIdx.y][threadIdx.x] = 0.f;
        }

        // load W2 tile
        int hk = k0 + threadIdx.y;
        if(threadIdx.y < BK && threadIdx.x < BN){
            if(hk < H && c < C) Ws[threadIdx.y][threadIdx.x] = W2T[(size_t)hk * C + c];
            else Ws[threadIdx.y][threadIdx.x] = 0.f;
        }

        __syncthreads();

        if(b < B && c < C){
            #pragma unroll
            for(int k=0; k<BK; k++){
                acc = fmaf(Hs[threadIdx.y][k], Ws[k][threadIdx.x], acc);
            }
        }

        __syncthreads();
    }

    if(b < B && c < C){
        acc += b2[c];
        // atomic add into double checksum
        atomicAdd(checksum, (double)acc);
    }
}

int main(int argc, char** argv){
    if(argc < 5){
        std::fprintf(stderr, "Usage: ./mlp_cuda B D H C\n");
        return 1;
    }
    int B = std::atoi(argv[1]);
    int D = std::atoi(argv[2]);
    int H = std::atoi(argv[3]);
    int C = std::atoi(argv[4]);
    if(B<=0||D<=0||H<=0||C<=0){
        std::fprintf(stderr, "All params must be > 0\n");
        return 1;
    }

    // Host: build transposed weights for coalesced reads
    auto host_hash_u32 = [](uint32_t x){
        x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x;
    };
    auto host_u32_to_float_signed = [](uint32_t x){
        uint32_t m = x & 0x00FFFFFFU;
        float f = (float)m / (float)0x00800000U;
        return f - 1.0f;
    };
    auto gen_w1 = [&](int h2, int d2){
        uint32_t idx = (uint32_t)(h2 * 2246822519u) ^ (uint32_t)(d2 * 3266489917u) ^ 0x12345678u;
        return 0.1f * host_u32_to_float_signed(host_hash_u32(idx));
    };
    auto gen_b1 = [&](int h2){
        uint32_t idx = (uint32_t)(h2 * 374761393u) ^ 0x0BADF00Du;
        return 0.01f * host_u32_to_float_signed(host_hash_u32(idx));
    };
    auto gen_w2 = [&](int c2, int h2){
        uint32_t idx = (uint32_t)(c2 * 1103515245u) ^ (uint32_t)(h2 * 12345u) ^ 0x87654321u;
        return 0.1f * host_u32_to_float_signed(host_hash_u32(idx));
    };
    auto gen_b2 = [&](int c2){
        uint32_t idx = (uint32_t)(c2 * 2654435761u) ^ 0xCAFEBABEu;
        return 0.01f * host_u32_to_float_signed(host_hash_u32(idx));
    };

    std::vector<float> hW1T((size_t)D * H);
    std::vector<float> hb1((size_t)H);
    for(int h2=0; h2<H; h2++){
        hb1[h2] = gen_b1(h2);
        for(int d2=0; d2<D; d2++){
            hW1T[(size_t)d2 * H + h2] = gen_w1(h2,d2);
        }
    }

    std::vector<float> hW2T((size_t)H * C);
    std::vector<float> hb2((size_t)C);
    for(int c2=0; c2<C; c2++){
        hb2[c2] = gen_b2(c2);
        for(int h2=0; h2<H; h2++){
            hW2T[(size_t)h2 * C + c2] = gen_w2(c2,h2);
        }
    }

    float *dW1T=nullptr, *db1=nullptr, *dW2T=nullptr, *db2=nullptr;
    float *dHidden=nullptr;
    double *dChecksum=nullptr;

    ck(cudaMalloc(&dW1T, (size_t)D * H * sizeof(float)), "malloc dW1T");
    ck(cudaMalloc(&db1,  (size_t)H * sizeof(float)),     "malloc db1");
    ck(cudaMalloc(&dW2T, (size_t)H * C * sizeof(float)), "malloc dW2T");
    ck(cudaMalloc(&db2,  (size_t)C * sizeof(float)),     "malloc db2");
    ck(cudaMalloc(&dChecksum, sizeof(double)),           "malloc dChecksum");

    ck(cudaMemcpy(dW1T, hW1T.data(), (size_t)D * H * sizeof(float), cudaMemcpyHostToDevice), "cpy W1T");
    ck(cudaMemcpy(db1,  hb1.data(),  (size_t)H * sizeof(float),     cudaMemcpyHostToDevice), "cpy b1");
    ck(cudaMemcpy(dW2T, hW2T.data(), (size_t)H * C * sizeof(float), cudaMemcpyHostToDevice), "cpy W2T");
    ck(cudaMemcpy(db2,  hb2.data(),  (size_t)C * sizeof(float),     cudaMemcpyHostToDevice), "cpy b2");

    // Chunk B to control memory for hidden
    int chunkB = 4096;
    if(B < chunkB) chunkB = B;
    ck(cudaMalloc(&dHidden, (size_t)chunkB * H * sizeof(float)), "malloc dHidden");

    ck(cudaDeviceSynchronize(), "sync pre");
    auto t0 = std::chrono::high_resolution_clock::now();

    double checksum_host = 0.0;

    for(int base=0; base<B; base+=chunkB){
        int Bc = std::min(chunkB, B-base);
        ck(cudaMemset(dChecksum, 0, sizeof(double)), "memset checksum");

        dim3 block(BN, BM);
        dim3 gridHidden((H + BN - 1)/BN, (Bc + BM - 1)/BM);
        hidden_tiled<<<gridHidden, block>>>(dHidden, dW1T, db1, Bc, D, H, base);
        ck(cudaGetLastError(), "hidden_tiled");

        dim3 gridOut((C + BN - 1)/BN, (Bc + BM - 1)/BM);
        out_tiled_checksum<<<gridOut, block>>>(dHidden, dW2T, db2, Bc, H, C, dChecksum);
        ck(cudaGetLastError(), "out_tiled_checksum");

        ck(cudaMemcpy(&checksum_host, dChecksum, sizeof(double), cudaMemcpyDeviceToHost), "cpy checksum");
        // accumulate across chunks
        // (reuse checksum_host var; add into running total)
        static double running = 0.0;
        running += checksum_host;
        checksum_host = running;
    }

    ck(cudaDeviceSynchronize(), "sync post");
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout.setf(std::ios::fixed);
    std::cout << "runtime_ms " << std::setprecision(3) << ms << "\n";
    std::cout << "checksum "   << std::setprecision(6) << checksum_host << "\n";

    cudaFree(dHidden);
    cudaFree(dW1T); cudaFree(db1); cudaFree(dW2T); cudaFree(db2);
    cudaFree(dChecksum);
    return 0;
}
