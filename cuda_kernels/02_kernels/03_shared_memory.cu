#include <stdio.h>
#include <stdlib.h>

#define N 1024  // Size of the matrices (N x N)
#define TILE_WIDTH 32 // Tile width for shared memory

__global__ void matMulTiled(float *A, float *B, float *C, int width) {
    
    // Shared memory for tiles
    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y; // Row index of the element
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x; // Column index of the element


    // boundary check
    if (row >= width || col >= width) return;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (width + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        // Load tiles into shared memory with boundary checks

        int a_col = t * TILE_WIDTH + threadIdx.x; // Column index in A tile
        int b_row = t * TILE_WIDTH + threadIdx.y; // Row index in B tile

        tile_A[threadIdx.y][threadIdx.x] = (a_col < width) ? A[row * width + a_col] : 0.0f;
        tile_B[threadIdx.y][threadIdx.x] = (b_row < width) ? B[b_row * width + col] : 0.0f;
        

        __syncthreads(); // Ensure all threads have loaded their tiles

        // multiply the tiles
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }

        __syncthreads(); // Ensure all threads have completed their computation
    }

    // write the result to global memory
    C[row * width + col] = sum;
}

void fillMatrix(float *matrix,int n, float val) {
    for (int i = 0; i < n * n; ++i) matrix[i] = val;
}

int main() {
    int size = N * N * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    fillMatrix(h_A, N, 1.0f); // Fill A with 1.0
    fillMatrix(h_B, N, 2.0f); // Fill B with 2.0

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    matMulTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize(); // Wait for the kernel to finish

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);


    printf("Result of matrix multiplication C[0][0]: %f\n", h_C[0]); // Should be 2.0 * N

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);  

    return 0;
}
