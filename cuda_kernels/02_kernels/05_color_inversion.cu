/* 
LeetGPU - Color Inversion Kernel
Full code

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define WIDTH 4096
#define HEIGHT 2048
#define BLOCK_SIZE 256
#define CHANNELS 4 // RGBA (as discussed in the problem statement)

// Lets first implement it in CPU

void invert_cpu(unsigned char *image, unsigned char *output, int width, int height) {
    int total_pixels = width * height;

    // touch base
    for (int i =0; i < total_pixels; ++i) {
        int base = 4 * i; // this is the start of the RGBA values for this pixel

        // now the output base thing
        output[base + 0] = 255 - image[base + 0]; // R
        output[base + 1] = 255 - image[base + 1]; // G
        output[base + 2] = 255 - image[base + 2]; // B
        output[base + 3] = image[base + 3]; // A, leave it unchanged
    }
}


// Here comes the GPU kernel. 

__global__ void invert_gpu(unsigned char *image, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;

    if (idx < total_pixels) {
        int base = 4 * idx; // this is the start of the RGBA values for this pixel

        // Invert the colors
        image[base + 0] = 255 - image[base + 0]; // R
        image[base + 1] = 255 - image[base + 1]; // G
        image[base + 2] = 255 - image[base + 2]; // B
        // A remains unchanged
    }
}

// Initialize the image with random data for testing

void init_image(unsigned char *image, int width, int height) {
    int total_pixels = width * height;
    for (int i = 0; i < total_pixels; ++i) {
        int base = 4 * i;
        image[base + 0] = rand() % 256; // R
        image[base + 1] = rand() % 256; // G
        image[base + 2] = rand() % 256; // B
        image[base + 3] = 255; // A, fully opaque
    }
}


// Verify CPU and GPUresults

int verify_image(unsigned char *cpu_image, unsigned char *gpu_image, int width, int height) {
    int total_values = width * height * 4; // total RGBA values bytes

    for (int i = 0; i < total_values; ++i) {
        if (cpu_image[i] != gpu_image[i]) {
            printf("Mismatch at index %d: CPU = %d, GPU = %d\n", i, cpu_image[i], gpu_image[i]);
            return 0; // mismatch found
        }
    }
    return 1; // no mismatches found
}


// print the irst few pixels for debugging

void print_pixels(unsigned char *image, int num_pixels) {
    for (int i = 0; i < num_pixels; ++i) {
        int base = 4 * i;
        printf("Pixel %d: R=%d, G=%d, B=%d, A=%d\n", i, image[base + 0], image[base + 1], image[base + 2], image[base + 3]);
    }
}


// and now the timing function

double getTime() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}


int main() {
    unsigned char *h_image;
    unsigned char *h_output_cpu;
    unsigned char *h_output_gpu;
    unsigned char *d_image;

    int total_pixels = WIDTH * HEIGHT;
    size_t image_size = total_pixels * CHANNELS * sizeof(unsigned char);

    // Allocate host memory
    h_image = (unsigned char *)malloc(image_size);
    h_output_cpu = (unsigned char *)malloc(image_size);
    h_output_gpu = (unsigned char *)malloc(image_size);

    // input image
    srand(time(NULL));
    init_image(h_image, WIDTH, HEIGHT);

    // allocate device memory
    cudaMalloc((void **)&d_image, image_size);

    // copy image to device
    cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice);


    // launch kernelint
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

    // Some Warming up runs
    printf("Warming up CPU and GPU...\n");
    for (int i = 0; i < 10; ++i) {
        invert_cpu(h_image, h_output_cpu , WIDTH, HEIGHT);

        cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice);
        invert_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_image, WIDTH, HEIGHT);
        cudaDeviceSynchronize();
    }

    // Time the CPU version
    printf("Benchmarking CPU...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 100; ++i) {
        double start = getTime();
        invert_cpu(h_image, h_output_cpu, WIDTH, HEIGHT);
        double end = getTime();
        cpu_total_time += end - start;
    }
    printf("Average CPU time: %.6f seconds\n", cpu_total_time / 100.0);

    // Benchmark GPU kernel only
    printf("Benchmarking GPU kernel...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 100; ++i) {
        cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice);

        double start = getTime();
        invert_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_image, WIDTH, HEIGHT);
        cudaDeviceSynchronize();
        double end = getTime();

        gpu_total_time += end - start;
    }
    printf("Average GPU kernel time: %.6f seconds\n", gpu_total_time / 100.0);
    printf("Speedup: %.2f\n", (cpu_total_time / 100.0) / (gpu_total_time / 100.0));

    // Run once more to fetch GPU result for verification
    cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice);
    invert_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_image, WIDTH, HEIGHT);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output_gpu, d_image, image_size, cudaMemcpyDeviceToHost);

    // Verify correctness
    if (verify_image(h_output_cpu, h_output_gpu, WIDTH, HEIGHT)) {
        printf("Verification passed: CPU and GPU results match.\n");
    } else {
        printf("Verification failed.\n");
    }

    // Print a few example pixels
    printf("\nFirst 5 input pixels:\n");
    print_pixels(h_image, 5);

    printf("\nFirst 5 inverted pixels (GPU):\n");
    print_pixels(h_output_gpu, 5);

    // Free memory
    cudaFree(d_image);
    free(h_image);
    free(h_output_cpu);
    free(h_output_gpu);

    return 0;
}
