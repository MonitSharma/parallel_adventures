/*
We'll multiply two square matrices:
    Matrix A : size NxN
    Matrix B : size NxN
    Result C : size NxN

Each thread will compute one element of the result matrix C.
    C[i][j] = sum(A[i][k] * B[k][j])

    
*/

#include <stdio.h>
#include <stdlib.h>

#define N 16  // Size of the matrices (N x N)

__global__ void matMul(int *A, int *B, int *C, int width) {
    // Each thread computes one element of the result matrix C
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index of the element
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index of the element

    if (row < width && col < width) {
        int sum = 0;
        for (int k = 0; k < width; ++k) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

void printMatrix( int *mat, int width) {
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            printf("%4d ", mat[i * width + j]);
        }
        printf("\n");
    }
}

int main() {
    int size = N * N * sizeof(int);

    // allocate memory for matrices A, B, C on host
    int *h_A = (int *)malloc(size);
    int *h_B = (int *)malloc(size);
    int *h_C = (int *)malloc(size);

    // Initialize input matrixes
    for (int i = 0; i < N*N; ++i) {
        h_A[i] = rand() % 10; // Random values between 0 and 9
        h_B[i] = rand() % 10; // Random values between 0 and 9
    }

    // Allocate memory for matrices A, B, C on device
    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define block size and grid size
    dim3 blockSize(16, 16); // 16x16 threads per block
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y); // Calculate grid size

    // Launch the kernel
    matMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize(); // Wait for the kernel to finish


    // Copy result matrix C from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print the input matrices and the result
    printf("Matrix A:\n");
    printMatrix(h_A, N);
    printf("\nMatrix B:\n");
    printMatrix(h_B, N);
    printf("\nResult Matrix C:\n");
    printMatrix(h_C, N);
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}