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
#include <iostream>

#define N 10000000 // Number of elements in the arrays are 10 million
#define BLOCK_SIZE_1D 1024 // Number of threads in each block
#define BLOCK_SIZE_3D_X 16 // Number of threads in each block in X direction
#define BLOCK_SIZE_3D_Y 8 // Number of threads in each block in Y direction
#define BLOCK_SIZE_3D_Z 8  // Number of threads in each block in Z direction
// since 16*8*8 = 1024 threads per block


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

// vector addition kernel for 3D grid and 3D blocks

__global__ void addVectors3D(float *a, float *b, float *c, int nx, int ny, int nz) {
    // Compute the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    // there are three addition operations, three multiplication operations

    if (idx < nx && idy < ny && idz < nz) {
        int index = idx + idy * nx + idz * nx * ny; // row-major order
        if (index < nx * ny * nz) {
            c[index] = a[index] + b[index];
        }
    }
}




// IInitialize vectors with random values

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
    float *h_c_gpu_1d = (float *)malloc(N * sizeof(float));
    float *h_c_gpu_3d = (float *)malloc(N * sizeof(float));

    float *d_a, *d_b, *d_c_1d, *d_c_3d; // Device vectors
    // Initialize input vectors
    srand(time(NULL));
    initVectors(h_a, N);
    initVectors(h_b, N);

    // Allocate device memory
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c_1d, N * sizeof(float));
    cudaMalloc(&d_c_3d, N * sizeof(float));

    // copy input vectors from host to device
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);


    // define grid and block dimensions
    int num_blocks_1d = (N + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D; // Why this formula?
                                                        // It ensures that we have enough blocks to cover all elements


    // define the grid and block dimesnions for 3 d
    int nx = 100, ny = 100, nz = 1000; // 100*100*1000 = 10 million elements
    dim3 block_size_3d(BLOCK_SIZE_3D_X, BLOCK_SIZE_3D_Y, BLOCK_SIZE_3D_Z);
    dim3 num_blocks_3d( (nx + BLOCK_SIZE_3D_X - 1) / BLOCK_SIZE_3D_X,
                        (ny + BLOCK_SIZE_3D_Y - 1) / BLOCK_SIZE_3D_Y,
                        (nz + BLOCK_SIZE_3D_Z - 1) / BLOCK_SIZE_3D_Z );
    

    // warm up runs


    printf("performing warm up runs...\n");
    
    for (int i = 0; i < 3; ++i) {
        vectorAddCPU(h_a, h_b, h_c_cpu, N);
        addVectors<<<num_blocks_1d, BLOCK_SIZE_1D>>>(d_a, d_b, d_c_1d, N);
        addVectors3D<<<num_blocks_3d, block_size_3d>>>(d_a, d_b, d_c_3d, nx, ny, nz);
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

    // benchmark GPU 1D kernel
    printf("benchmarking GPU 1D kernel...\n");
    double gpu_1d_total_time = 0.0;
    for (int i = 0; i < 100; ++i) {
        cudaMemset(d_c_1d, 0, N * sizeof(float)); // reset output array
        double start = getTime();
        addVectors<<<num_blocks_1d, BLOCK_SIZE_1D>>>(d_a, d_b, d_c_1d, N);
        cudaDeviceSynchronize();
        double end = getTime();
        gpu_1d_total_time += end - start;
    }

    double gpu_1d_avg_time = gpu_1d_total_time / 100.0;

    // Print the results
    printf("CPU average time: %f seconds\n", cpu_avg_time);
    printf("GPU 1D average time: %f seconds\n", gpu_1d_avg_time);

    printf("Speedup: %f\n", cpu_avg_time / gpu_1d_avg_time);


    // verify the result
    cudaMemcpy(h_c_gpu_1d, d_c_1d, N * sizeof(float), cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_c_cpu[i] - h_c_gpu_1d[i]) > 1e-5) {
            correct = false;
            printf("Mismatch at index %d: CPU %f, GPU %f\n", i, h_c_cpu[i], h_c_gpu_1d[i]);
            break;
        }
    }

    printf("Result verification for GPU 1D: %s\n", correct ? "SUCCESS" : "FAILURE");


    // benchmark GPU 3D kernel
    printf("benchmarking GPU 3D kernel...\n");
    double gpu_3d_total_time = 0.0;
    for (int i = 0; i < 100; ++i) {
        cudaMemset(d_c_3d, 0, N * sizeof(float)); // reset output array
        double start = getTime();
        addVectors3D<<<num_blocks_3d, block_size_3d>>>(d_a, d_b, d_c_3d, nx, ny, nz);
        cudaDeviceSynchronize();
        double end = getTime();
        gpu_3d_total_time += end - start;
    }

    double gpu_3d_avg_time = gpu_3d_total_time / 100.0;


    // verify the result
    cudaMemcpy(h_c_gpu_3d, d_c_3d, N * sizeof(float), cudaMemcpyDeviceToHost);
    correct = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_c_cpu[i] - h_c_gpu_3d[i]) > 1e-5) {
            correct = false;
            printf("Mismatch at index %d: CPU %f, GPU %f\n", i, h_c_cpu[i], h_c_gpu_3d[i]);
            break;
        }
    }

    // Print the results
    printf("Result verification for GPU 3D: %s\n", correct ? "SUCCESS" : "FAILURE");
    printf("GPU 3D average time: %f seconds\n", gpu_3d_avg_time);
    printf("Speedup (3D kernel): %f\n", cpu_avg_time / gpu_3d_avg_time);


    printf("CPU average time: %f milliseconds\n", cpu_avg_time * 1000);
    printf("GPU 1D average time: %f milliseconds\n", gpu_1d_avg_time * 1000);
    printf("GPU 3D average time: %f milliseconds\n", gpu_3d_avg_time * 1000);
    printf("Speedup (CPU vs GPU 1D): %fx\n", cpu_avg_time / gpu_1d_avg_time);
    printf("Speedup (CPU vs GPU 3D): %fx\n", cpu_avg_time / gpu_3d_avg_time);
    printf("Speedup (GPU 1D vs GPU 3D): %fx\n", gpu_1d_avg_time / gpu_3d_avg_time);

    // Cleanup
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu_1d);
    free(h_c_gpu_3d);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c_1d);
    cudaFree(d_c_3d);

    return 0;
}