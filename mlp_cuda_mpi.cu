#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>

#include <mpi.h>
#include <cuda_runtime.h>

static inline void ck(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(e));
        std::exit(1);
    }
}

__host__ __device__ static inline uint32_t hash_u32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}
__host__ __device__ static inline float u32_to_float_signed(uint32_t x) {
    uint32_t m = x & 0x00FFFFFFU;
    float f = (float)m / (float)0x00800000U;
    return f - 1.0f;
}

static inline float gen_input(int b, int d) {
    uint32_t idx = (uint32_t)(b * 1315423911u) ^ (uint32_t)(d * 2654435761u) ^ 0xA5A5A5A5u;
    return 0.5f * u32_to_float_signed(hash_u32(idx));
}
static inline float gen_w1(int h, int d) {
    uint32_t idx = (uint32_t)(h * 2246822519u) ^ (uint32_t)(d * 3266489917u) ^ 0x12345678u;
    return 0.1f * u32_to_float_signed(hash_u32(idx));
}
static inline float gen_b1(int h) {
    uint32_t idx = (uint32_t)(h * 374761393u) ^ 0x0BADF00Du;
    return 0.01f * u32_to_float_signed(hash_u32(idx));
}
static inline float gen_w2(int c, int h) {
    uint32_t idx = (uint32_t)(c * 1103515245u) ^ (uint32_t)(h * 12345u) ^ 0x87654321u;
    return 0.1f * u32_to_float_signed(hash_u32(idx));
}
static inline float gen_b2(int c) {
    uint32_t idx = (uint32_t)(c * 2654435761u) ^ 0xCAFEBABEu;
    return 0.01f * u32_to_float_signed(hash_u32(idx));
}

__device__ static inline float relu(float x) { return x > 0.f ? x : 0.f; }

__global__ void hidden_kernel(const float* __restrict__ X, const float* __restrict__ W1,
                              const float* __restrict__ b1, float* __restrict__ Hout,
                              int Bc, int D, int H) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (b >= Bc || h >= H) return;

    const float* x = X + (size_t)b * D;
    const float* w = W1 + (size_t)h * D;
    float acc = b1[h];
    for (int d = 0; d < D; d++) acc += w[d] * x[d];
    Hout[(size_t)b * H + h] = relu(acc);
}

__global__ void out_kernel(const float* __restrict__ Hact, const float* __restrict__ W2,
                           const float* __restrict__ b2, float* __restrict__ Out,
                           int Bc, int H, int C) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (b >= Bc || c >= C) return;

    const float* hvec = Hact + (size_t)b * H;
    const float* w = W2 + (size_t)c * H;
    float acc = b2[c];
    for (int h = 0; h < H; h++) acc += w[h] * hvec[h];
    Out[(size_t)b * C + c] = acc;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank=0, world=1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    if (argc < 5) {
        if (rank == 0) std::fprintf(stderr, "Usage: mpirun -np P ./mlp_cuda_mpi B D H C\n");
        MPI_Finalize();
        return 1;
    }
    int B = std::atoi(argv[1]);
    int D = std::atoi(argv[2]);
    int H = std::atoi(argv[3]);
    int C = std::atoi(argv[4]);
    if (B <= 0 || D <= 0 || H <= 0 || C <= 0) {
        if (rank == 0) std::fprintf(stderr, "All params must be > 0\n");
        MPI_Finalize();
        return 1;
    }

    // Simple GPU assignment: rank -> device (mod deviceCount)
    int devCount = 0;
    cudaGetDeviceCount(&devCount);
    if (devCount > 0) ck(cudaSetDevice(rank % devCount), "set device");

    // Weights on host (same for all ranks)
    std::vector<float> hW1((size_t)H * D), hb1((size_t)H);
    std::vector<float> hW2((size_t)C * H), hb2((size_t)C);

    for (int h = 0; h < H; h++) {
        hb1[h] = gen_b1(h);
        for (int d = 0; d < D; d++)
            hW1[(size_t)h * D + d] = gen_w1(h, d);
    }
    for (int c = 0; c < C; c++) {
        hb2[c] = gen_b2(c);
        for (int h = 0; h < H; h++)
            hW2[(size_t)c * H + h] = gen_w2(c, h);
    }

    // Device weights (exclude from timing)
    float *dW1=nullptr, *db1=nullptr, *dW2=nullptr, *db2=nullptr;
    ck(cudaMalloc(&dW1, (size_t)H * D * sizeof(float)), "malloc dW1");
    ck(cudaMalloc(&db1, (size_t)H * sizeof(float)), "malloc db1");
    ck(cudaMalloc(&dW2, (size_t)C * H * sizeof(float)), "malloc dW2");
    ck(cudaMalloc(&db2, (size_t)C * sizeof(float)), "malloc db2");

    ck(cudaMemcpy(dW1, hW1.data(), (size_t)H * D * sizeof(float), cudaMemcpyHostToDevice), "cpy W1");
    ck(cudaMemcpy(db1, hb1.data(), (size_t)H * sizeof(float), cudaMemcpyHostToDevice), "cpy b1");
    ck(cudaMemcpy(dW2, hW2.data(), (size_t)C * H * sizeof(float), cudaMemcpyHostToDevice), "cpy W2");
    ck(cudaMemcpy(db2, hb2.data(), (size_t)C * sizeof(float), cudaMemcpyHostToDevice), "cpy b2");

    // Split batch among ranks
    int base = (int)((long long)B * rank / world);
    int end  = (int)((long long)B * (rank + 1) / world);
    int Bloc = end - base;

    // Chunking inside rank
    int chunkB = 256;
    if (Bloc < chunkB) chunkB = Bloc;

    std::vector<float> hX((size_t)chunkB * D);
    std::vector<float> hOut((size_t)chunkB * C);

    float *dX=nullptr, *dHidden=nullptr, *dOut=nullptr;
    ck(cudaMalloc(&dX,      (size_t)chunkB * D * sizeof(float)), "malloc dX");
    ck(cudaMalloc(&dHidden, (size_t)chunkB * H * sizeof(float)), "malloc dHidden");
    ck(cudaMalloc(&dOut,    (size_t)chunkB * C * sizeof(float)), "malloc dOut");

    ck(cudaDeviceSynchronize(), "sync before timing");
    MPI_Barrier(MPI_COMM_WORLD); // align start (optional)
    double t0 = MPI_Wtime();

    double local_checksum = 0.0;

    for (int gb = base; gb < end; gb += chunkB) {
        int Bc = std::min(chunkB, end - gb);

        for (int b = 0; b < Bc; b++) {
            int global_b = gb + b;
            for (int d = 0; d < D; d++)
                hX[(size_t)b * D + d] = gen_input(global_b, d);
        }

        ck(cudaMemcpy(dX, hX.data(), (size_t)Bc * D * sizeof(float), cudaMemcpyHostToDevice), "cpy X");

        dim3 block1(16, 8);
        dim3 grid1((H + block1.x - 1) / block1.x, (Bc + block1.y - 1) / block1.y);
        hidden_kernel<<<grid1, block1>>>(dX, dW1, db1, dHidden, Bc, D, H);
        ck(cudaGetLastError(), "hidden kernel");

        dim3 block2(16, 8);
        dim3 grid2((C + block2.x - 1) / block2.x, (Bc + block2.y - 1) / block2.y);
        out_kernel<<<grid2, block2>>>(dHidden, dW2, db2, dOut, Bc, H, C);
        ck(cudaGetLastError(), "out kernel");

        ck(cudaMemcpy(hOut.data(), dOut, (size_t)Bc * C * sizeof(float), cudaMemcpyDeviceToHost), "cpy Out");

        for (int i = 0; i < Bc * C; i++) local_checksum += (double)hOut[(size_t)i];
    }

    ck(cudaDeviceSynchronize(), "sync after compute");
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    double global_checksum = 0.0;
    MPI_Reduce(&local_checksum, &global_checksum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double ms = (t1 - t0) * 1000.0;
        std::cout.setf(std::ios::fixed);
        std::cout << "runtime_ms " << std::setprecision(3) << ms << "\n";
        std::cout << "checksum "   << std::setprecision(6) << global_checksum << "\n";
    }

    cudaFree(dX); cudaFree(dHidden); cudaFree(dOut);
    cudaFree(dW1); cudaFree(db1); cudaFree(dW2); cudaFree(db2);

    MPI_Finalize();
    return 0;
}
