The  `whoami` CUDA kernel may look like just a fun way to print IDs, but it's actually very important for understanding how parallel GPU programming works.


CUDA Programming is all about assigning work to many threads, but to do that every thread needs to know:
1. Where it lives in the grid
2. Where it lives in its block
3. What its global ID is

This code teaches you exactly hot to compute that.


![alt text](block_address.png)


Most real-world CUDA programs look like this:

```c
int idx = blockIdx.x * blockDim.x + threadIdx.x;
array[idx] += 1
```

whethere you're adding vectors, multiplying matirces, simulating particles etc.

Each thread must know: *What part of the data am I responsible for?*

Treat this code as the "Helloe World" of CUDA programming.


---

In CUDA, threads within a block are automatically numbered `0,1,2...` via the variable `threadIdx.x`. Threads are in blocks, and blocks are given the index `blockIdx.x` and each block has a size `blockDim.x`. 

So : 
-  `threadIdx.x` - is the id of the individual thread
-  `blockIdx.x` - is the id of the individual block, and one block has many threads
- `blockDim.x` - is the number of threads in each block.



So threads before you is 

        blockIdx.x * blockDim.x

and if we add `threadIdx.x` to it, it gives a unique global index


When we launch a kernel from the host, we choose two parameters:

- `blockSize` = number of threads per block 
- `numBlocks` = number of blocks in your grid.


Now, suppose we have `N` total elements, and we pick `blockSize=256`, i.e there are 256 threads in a block, how can we calculate `numBlocks` i.e the number of blocks required so that the number of threads cover all N elements.

```c
int numBlocks = (N + blockSize - 1) / blockSize;
```

since this is the `ceil` function, for eg `ceil(10/3) = ceil(3.33) = 4` and `(10 + 3-1)/3 = 12/3 = 4`

So the code looks like:

```c
__global__ void vectorAdd(float *A, float *B, float *C, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N>) {
        C[i] = A[i] + B[i];
    }
}