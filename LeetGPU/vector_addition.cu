#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {

    // each thread computes one element of the result
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // cover all elements, even if N >> blockdim.x * griddim.x
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i< N ; i += stride) {
        C[i] = A[i] + B[i];
    }

}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
