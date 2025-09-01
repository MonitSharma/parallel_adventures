#include <stdio.h>
#define N 8

__global__ void bitreverse(unsigned int *data) {
    int idx = threadIdx.x;
    unsigned int val = data[idx];
    data[idx] = ((val & 0xf0) >> 4) | ((val & 0x0f) << 4);
}

int main() {
    unsigned int h_data[N];
    for (int i = 0; i < N; i++) h_data[i] = i;
    unsigned int *d_data;
    cudaMalloc(&d_data, N * sizeof(unsigned int));
    cudaMemcpy(d_data, h_data, N * sizeof(unsigned int), cudaMemcpyHostToDevice);

    bitreverse<<<1, N>>>(d_data);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_data, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) printf("%u -> %u\n", i, h_data[i]);

    cudaFree(d_data);
    return 0;
}
