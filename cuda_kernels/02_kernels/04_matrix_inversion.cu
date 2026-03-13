/*
    Matrix Transpose using CUDA

    This program transposes a matrix A of size ROWS x COLS
    and stores the result in matrix AT of size COLS x ROWS.

    Transpose means:
        AT[j][i] = A[i][j]

    Since matrices are stored in row-major order:
        A[i][j]   -> A[i * COLS + j]
        AT[j][i]  -> AT[j * ROWS + i]

    Steps:
    1. Allocate memory for input and output matrices on host and device.
    2. Initialize the input matrix on the host.
    3. Copy the input matrix from host to device.
    4. Launch a CUDA kernel where each thread handles one element.
    5. Copy the transposed matrix back to the host.
    6. Verify the GPU result against a CPU implementation.
    7. Benchmark CPU and GPU execution times.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define ROWS 7000
#define COLS 6000
#define BLOCK_X 16
#define BLOCK_Y 16

// CPU implementation of matrix transpose
void transpose_cpu(const float *input, float *output, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

// CUDA kernel for matrix transpose
__global__ void transpose_gpu(const float *input, float *output, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        int input_idx = row * cols + col;
        int output_idx = col * rows + row;
        output[output_idx] = input[input_idx];
    }
}

// Initialize matrix with random float values
void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

// Verify CPU and GPU results
int verify_matrices(const float *cpu, const float *gpu, int rows, int cols) {
    int total = rows * cols;
    for (int i = 0; i < total; ++i) {
        if (cpu[i] != gpu[i]) {
            printf("Mismatch at index %d: CPU = %f, GPU = %f\n", i, cpu[i], gpu[i]);
            return 0;
        }
    }
    return 1;
}

// Print a small matrix
void print_matrix(const float *mat, int rows, int cols, int max_r, int max_c) {
    for (int i = 0; i < rows && i < max_r; ++i) {
        for (int j = 0; j < cols && j < max_c; ++j) {
            printf("%8.4f ", mat[i * cols + j]);
        }
        printf("\n");
    }
}

// Timing helper
double getTime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main() {
    float *h_input, *h_output_cpu, *h_output_gpu;
    float *d_input, *d_output;

    size_t size_input = ROWS * COLS * sizeof(float);
    size_t size_output = COLS * ROWS * sizeof(float);

    // Allocate host memory
    h_input = (float *)malloc(size_input);
    h_output_cpu = (float *)malloc(size_output);
    h_output_gpu = (float *)malloc(size_output);

    // Initialize input matrix
    srand(time(NULL));
    init_matrix(h_input, ROWS, COLS);

    // Allocate device memory
    cudaMalloc((void **)&d_input, size_input);
    cudaMalloc((void **)&d_output, size_output);

    // Copy input matrix from host to device
    cudaMemcpy(d_input, h_input, size_input, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(BLOCK_X, BLOCK_Y);
    dim3 gridSize((COLS + BLOCK_X - 1) / BLOCK_X,
                  (ROWS + BLOCK_Y - 1) / BLOCK_Y);

    // Warm-up
    printf("Warming up CPU and GPU...\n");
    for (int i = 0; i < 5; ++i) {
        transpose_cpu(h_input, h_output_cpu, ROWS, COLS);
        transpose_gpu<<<gridSize, blockSize>>>(d_input, d_output, ROWS, COLS);
        cudaDeviceSynchronize();
    }

    // Benchmark CPU
    printf("Benchmarking CPU...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 10; ++i) {
        double start = getTime();
        transpose_cpu(h_input, h_output_cpu, ROWS, COLS);
        double end = getTime();
        cpu_total_time += end - start;
    }
    printf("Average CPU time: %.6f seconds\n", cpu_total_time / 10.0);

    // Benchmark GPU
    printf("Benchmarking GPU...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 10; ++i) {
        double start = getTime();
        transpose_gpu<<<gridSize, blockSize>>>(d_input, d_output, ROWS, COLS);
        cudaDeviceSynchronize();
        double end = getTime();
        gpu_total_time += end - start;
    }
    printf("Average GPU time: %.6f seconds\n", gpu_total_time / 10.0);
    printf("Speedup: %.2f\n", (cpu_total_time / 10.0) / (gpu_total_time / 10.0));

    // Copy GPU result back to host
    cudaMemcpy(h_output_gpu, d_output, size_output, cudaMemcpyDeviceToHost);

    // Verify correctness
    if (verify_matrices(h_output_cpu, h_output_gpu, COLS, ROWS)) {
        printf("Verification passed: CPU and GPU results match.\n");
    } else {
        printf("Verification failed.\n");
    }

    // Print a small sample
    printf("\nInput matrix sample:\n");
    print_matrix(h_input, ROWS, COLS, 5, 5);

    printf("\nTransposed matrix sample:\n");
    print_matrix(h_output_gpu, COLS, ROWS, 5, 5);

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output_cpu);
    free(h_output_gpu);

    return 0;
}