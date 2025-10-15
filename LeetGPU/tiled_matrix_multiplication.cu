#include <cuda_runtime.h>

#ifndef TILE
#define TILE 32  // 32x32 threads per block, 2 tiles in shared = ~8KB
#endif

// A: MxN, B: NxK, C: MxK  (row-major)
__global__ void matmul_tiled_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                          float* __restrict__ C,
                                    int M, int N, int K)
{
    // Output tile this block is responsible for
    int row = blockIdx.y * TILE + threadIdx.y;  // [block row] + [thread row]
    int col = blockIdx.x * TILE + threadIdx.x;  // [block col] + [thread col]

    // Shared tiles
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    float acc = 0.0f;

    // Number of tiles to cover the N dimension
    int numTiles = (N + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; ++t) {
        // Global indices to load for this tile
        int aRow = row;                 // same as output row
        int aCol = t * TILE + threadIdx.x;

        int bRow = t * TILE + threadIdx.y;
        int bCol = col;                 // same as output col

        // Load A tile element (with bounds checks)
        As[threadIdx.y][threadIdx.x] =
            (aRow < M && aCol < N) ? A[aRow * N + aCol] : 0.0f;

        // Load B tile element (with bounds checks)
        Bs[threadIdx.y][threadIdx.x] =
            (bRow < N && bCol < K) ? B[bRow * K + bCol] : 0.0f;

        __syncthreads();

        // Multiply-accumulate the tile
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result (with bounds check)
    if (row < M && col < K) {
        C[row * K + col] = acc;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K)
{
    // Use 32x32 blocks; grid covers the MxK output
    dim3 threadsPerBlock(TILE, TILE);
    dim3 blocksPerGrid( (K + TILE - 1) / TILE,
                        (M + TILE - 1) / TILE );

    matmul_tiled_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
