#include <cuda_runtime.h>

// The __global__ keyword tells the compiler this is a GPU kernel
__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    
    // 1. Calculate the global column (x) and row (y) for this specific thread
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // 2. Ensure the thread is within the bounds of the input matrix
    if (r < rows && c < cols) {
        
        // 3. Calculate 1D memory indices based on row-major layout
        // Input: We are at row 'r', and the row length is 'cols'
        int input_index = r * cols + c;
        
        // Output: We move to row 'c' and col 'r'. The output row length is 'rows'
        int output_index = c * rows + r;

        // 4. Perform the transpose! Read from input, write to output.
        output[output_index] = input[input_index];
    }
}

// Host code (runs on the CPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    // Define a 16x16 tile of threads (256 threads per block)
    dim3 threadsPerBlock(16, 16);
    
    // Calculate how many blocks we need (using ceiling division) to cover the matrix
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel asynchronously on the GPU
    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    
    // Wait for the GPU to finish before returning
    cudaDeviceSynchronize();
}
