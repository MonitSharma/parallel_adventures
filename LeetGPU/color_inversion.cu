#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;

    if (idx < total_pixels) {
        int base = 4 * idx;  // start of this pixel's RGBA values

        image[base + 0] = 255 - image[base + 0];  // R
        image[base + 1] = 255 - image[base + 1];  // G
        image[base + 2] = 255 - image[base + 2];  // B
        // image[base + 3] is A, leave it unchanged
    }
}

// image is a device pointer (memory already on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int total_pixels = width * height;
    int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}