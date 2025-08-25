#include <stdio.h>

__global__ void detailed_whoami() {
    // Grid and Block dimensions
    int grid_x = gridDim.x;
    int grid_y = gridDim.y;
    int grid_z = gridDim.z;

    int block_x = blockDim.x;
    int block_y = blockDim.y;
    int block_z = blockDim.z;

    // My block's coordinates in the grid
    int my_block_x = blockIdx.x;
    int my_block_y = blockIdx.y;
    int my_block_z = blockIdx.z;

    // My thread's coordinates inside the block
    int my_thread_x = threadIdx.x;
    int my_thread_y = threadIdx.y;
    int my_thread_z = threadIdx.z;

    // Step 1: Compute a unique block ID
    int block_id = my_block_x +
                   my_block_y * grid_x +
                   my_block_z * grid_x * grid_y;

    // Step 2: Compute how many threads are before my block
    int threads_per_block = block_x * block_y * block_z;
    int block_offset = block_id * threads_per_block;

    // Step 3: Compute my thread offset inside the block
    int thread_offset = my_thread_x +
                        my_thread_y * block_x +
                        my_thread_z * block_x * block_y;

    // Step 4: Global unique thread ID
    int global_thread_id = block_offset + thread_offset;

    // Print detailed information
    printf(
        "\n=== Thread Info ===\n"
        "Grid Dim      = (%d, %d, %d)\n"
        "Block Dim     = (%d, %d, %d)\n"
        "Block Idx     = (%d, %d, %d)\n"
        "Thread Idx    = (%d, %d, %d)\n"
        "Block ID      = %d\n"
        "Threads/Block = %d\n"
        "Block Offset  = %d (threads before my block)\n"
        "Thread Offset = %d (my position in this block)\n"
        "Global Thread ID = %d\n",
        grid_x, grid_y, grid_z,
        block_x, block_y, block_z,
        my_block_x, my_block_y, my_block_z,
        my_thread_x, my_thread_y, my_thread_z,
        block_id,
        threads_per_block,
        block_offset,
        thread_offset,
        global_thread_id
    );
}

int main() {
    dim3 gridDim(2, 2, 1);     // Total: 2x2x1 = 4 blocks
    dim3 blockDim(2, 2, 2);    // Total: 2x2x2 = 8 threads per block

    detailed_whoami<<<gridDim, blockDim>>>();

    cudaDeviceSynchronize();  // Wait for GPU to finish
    return 0;
}
