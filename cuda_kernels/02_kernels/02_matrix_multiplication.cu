/*
    Matrix Multiplication using CUDA

    This program multiplies two square matrices A and B of size N x N
    and stores the result in matrix C. The multiplication is performed
    on the GPU using CUDA.

    1. Allocate memory for matrices A, B, and C on both host and device.
    2. Initialize matrices A and B with random values on the host.
    3. Copy matrices A and B from host to device.
    4. Define a CUDA kernel to perform matrix multiplication.
    5. Launch the kernel with an appropriate grid and block size.
    6. Copy the result matrix C from device to host.
    7. Print the input matrices and the result matrix.

    Note:
Each thread will compute one element of the result matrix C.
    C[i][j] = sum(A[i][k] * B[k][j])

    
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 256  // Number of rows in matrix A and C
#define K 512 // Number of columns in matrix A and rows in matrix B
#define N 256  // Number of columns in matrix B and C
#define BLOCK_SIZE 32 // Block size (number of threads per block in each dimension)




// Matrix Multiplication on CPU

void matmul_cpu(float *A, float *B, float *C, int m, int k, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i * n + j] = 0;
            for (int p = 0; p < k; ++p) {
                C[i * n + j] += A[i * k + p] * B[p * n + j];
            }
        }
    }
}

// CUDA kernel for Matrix Multiplication

__global__ void  matmul_gpu(float *A, float *B, float *C, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index of C
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index

    if (row < m && col < n) {
        float value = 0;
        for (int p = 0; p < k; ++p) {
            value += A[row * k + p] * B[p * n + col];
        }
        C[row * n + col] = value;
    }
}

// Now initialize it with random values

void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat[i * cols + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
}


// function to measure time 

double getTime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}


// now the main function
int main() {
    float *h_A, *h_B, *h_C_cpu, *h_C_gpu;
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // allocate memory on host
    h_A = (float *)malloc(size_A);
    h_B = (float *)malloc(size_B);
    h_C_cpu = (float *)malloc(size_C);
    h_C_gpu = (float *)malloc(size_C);

    // initialize matrices A and B
    srand(time(NULL));
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    // allocate memory on device
    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_B, size_B);
    cudaMalloc((void **)&d_C, size_C);

    // copy matrices A and B from host to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // define block and grid sizes
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // warm up
    printf("Warming up GPU...\n");
    for (int i = 0; i < 10; ++i) {
        matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
        matmul_gpu<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);
        cudaDeviceSynchronize();
    }

    // benchmark CPU
    printf("Benchmarking CPU...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 100; ++i) {
        double start = getTime();
        matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);
        double end = getTime();
        cpu_total_time += end - start;
    }
    printf("Average CPU time: %.6f seconds\n", cpu_total_time / 100);

    // benchmark GPU
    printf("Benchmarking GPU...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 100; ++i) {
        double start = getTime();
        matmul_gpu<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);
        cudaDeviceSynchronize();
        double end = getTime();
        gpu_total_time += end - start;
    }
    printf("Average GPU time: %.6f seconds\n", gpu_total_time / 100);
    printf("Speedup: %.2f\n", (cpu_total_time / 100) / (gpu_total_time / 100));

    // free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);

    return 0;
}