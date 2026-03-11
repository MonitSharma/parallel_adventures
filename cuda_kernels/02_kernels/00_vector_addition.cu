/* 
We will:

    1. Create two arrays A and B
    2. Add them element-wise into array C using GPU threads
    3. Print the result
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>

#define N 10000000 // Number of elements in the arrays are 10 million
#define BLOCK_SIZE 256 // Number of threads in each block



// Vector addition on CPU

void vectorAddCPU(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}



// Vector addition kernel on GPU

__global__ void addVectors(float *a, float *b, float *c, int n ) {
    // Each thread computes one element of the result
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}


// INitialize vecotrs with random values

void initVectors(float *vec, int n) {
    for (int i = 0; i < n; ++i) {
        vec[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}



// function to measure the time

double getTime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}


// now the main function

int main() {
    // Allocate host memory
    float *h_a = (float *)malloc(N * sizeof(float));
    float *h_b = (float *)malloc(N * sizeof(float));
    float *h_c_cpu = (float *)malloc(N * sizeof(float));
    float *h_c_gpu = (float *)malloc(N * sizeof(float));

    float *d_a, *d_b, *d_c; // Device vectors
    // Initialize input vectors
    srand(time(NULL));
    initVectors(h_a, N);
    initVectors(h_b, N);

    // Allocate device memory
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    // copy input vectors from host to device
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);


    // define grid and block dimensions
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE; // Why this formula?
                                                        // It ensures that we have enough blocks to cover all elements

    printf("performing warm up runs...\n");
    // warm up runs
    for (int i = 0; i < 10; ++i) {
        vectorAddCPU(h_a, h_b, h_c_cpu, N);
        addVectors<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
    }

    // benchmark CPU
    printf("benchmarking CPU...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; ++i) {
        double start = getTime();
        vectorAddCPU(h_a, h_b, h_c_cpu, N);
        double end = getTime();
        cpu_total_time += end - start;
    }

    double cpu_avg_time = cpu_total_time / 20.0;

    // benchmark GPU
    printf("benchmarking GPU...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; ++i) {
        double start = getTime();
        addVectors<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        double end = getTime();
        gpu_total_time += end - start;
    }

    double gpu_avg_time = gpu_total_time / 20.0;

    // Print the results
    printf("CPU average time: %f seconds\n", cpu_avg_time);
    printf("GPU average time: %f seconds\n", gpu_avg_time);

    printf("Speedup: %f\n", cpu_avg_time / gpu_avg_time);


    // verify the result
    cudaMemcpy(h_c_gpu, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5) {
            correct = false;
            printf("Mismatch at index %d: CPU %f, GPU %f\n", i, h_c_cpu[i], h_c_gpu[i]);
            break;
        }
    }

    printf("Result verification: %s\n", correct ? "SUCCESS" : "FAILURE");

    // Cleanup
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

// nvcc -o vector_addition 00_vector_addition.cu
// ./vector_addition

// or do cudarun 00_vector_addition.cu