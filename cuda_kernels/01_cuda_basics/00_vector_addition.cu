/* 
We will:

    1. Create two arrays A and B
    2. Add them element-wise into array C using GPU threads
    3. Print the result
*/

#include <stdio.h>


__global__ void addVectors(int *a, int *b, int *c, int n ) {
    // Each thread computes one element of the result
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}


int main() {
    const int N = 100;

    int size = N * sizeof(int);

    // Host arryays (CPU memory)
    int h_a[N], h_b[N], h_c[N];

    // Initialize input arrays
    for (int i =0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }


    // Device arrays (GPU memory)
    int *d_a, *d_b , *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);


    // copy data to GPU
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // define block and grid sizes
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // launch kernel
    addVectors<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);


    // wait for GPU to finish
    cudaDeviceSynchronize();


    // copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    //print result
    for (int i = 0; i < N; ++i) {
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);  

    return 0;
}