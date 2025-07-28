#include <stdio.h>

__global__ void whoami(void) {
    // what apartment block am I in?
    int block_id = blockIdx.x +   // which apartment unit on the floor
                    blockIdx.y * gridDim.x +   // which floor in the building
                    blockIdx.z * gridDim.x * gridDim.y; // which building in the city


    // how many people live in the apartments before mine?

    int block_offset =  block_id * blockDim.x * blockDim.y * blockDim.z;
    // so, if you are in apartement 3, and each apartment has 64 people, then 3x64 = 192 people live in the apartments before yours

    
    // what's my number in the apartment?
    int thread_offset = threadIdx.x +             // bed number in the row
                        threadIdx.y * blockDim.x +  // which row of beds
                        threadIdx.z * blockDim.x * blockDim.y; // which bunk layer


    int id = block_offset + thread_offset;


    //printf("Hello from block %d, thread %d, id %d\n", block_id, threadIdx.x, id);
    printf("%04d | Block (%d, %d, %d) | Thread (%d, %d, %d) = %3d\n",
           id, blockIdx.x, blockIdx.y, blockIdx.z,
           threadIdx.x, threadIdx.y, threadIdx.z, thread_offset);
}



int main(int argc, char **argv) {

    // how big is the city?
    const int b_x = 2, b_y = 3, b_z = 4;
    const int t_x = 4, t_y = 4, t_z = 4;

    // there are 24 apratments, each apartment has 64 people

    int blocks_per_grid = b_x * b_y * b_z;
    int threads_per_block = t_x * t_y * t_z;

    printf("Launching %d blocks with %d threads each\n",
           blocks_per_grid, threads_per_block);

    printf("%d total threads in the grid\n",
           blocks_per_grid * threads_per_block);

    dim3 blocksPerGrid(b_x, b_y, b_z);
    dim3 threadsPerBlock(t_x, t_y, t_z);

    whoami<<<blocksPerGrid, threadsPerBlock>>>();

    cudaDeviceSynchronize();

    return 0;
}