### **Vector Addition**


This code is a classic CUDA workflow, the host code runs on CPU, allocates and copies data, while the device code (kernel) runs massively in parallel on the GPU.



```cpp
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 10000000   // 10 million elements
#define BLOCK_SIZE 256
```


`cuda_runtime.h` gives us the CUDA runtime APIs like `cudaMalloc`, `cudaMemcpy` etc, `N` is the problem size, and `BLOCK_SIZE` is the threads per block

Let's say we have:

```bash
A = [1, 2, 3, 4]
B = [5, 6, 7, 8]
```

and we want to create a third vector:

```bash
C = [A[0]+B[0], A[1]+B[1], A[2]+B[2], A[3]+B[3]]
  = [6, 8, 10, 12]
```

So, for every elelment `i` we do :

```bash
C[i] = A[i] + B[i]
```


----

**How the CPU does it**

A CPU has only a few cores (like 4, 8 ,16), and its optimized for doing one thing at a time, very fast. So the CPU will:

1. Start a loop from `i=0` to `N-1`
2. Add `A[i] + B[i]`
3. Store it in `C[i]`
4. Repeat

In code:

```cpp
for (int i = 0; i < N ; ++i) {
    C[i] = A[i] + B[i];
}
```

This means that the CPU does one or a few addition per clock cycle in a sequential manner. Even if it uses [vectorization](#add_about_it_later) (SIMD) or [multithreading](#this_also), it's still fundamentally limited by the number of CPU cores.


**How the GPU does it**

The GPU is designed for massive parallelism. While our CPU might have only 8 cores, a GPU can have thousands of cores, all working together. So intsead of one loop doing millions of additions one by one, the GPU says:

*Let's give each element to a different worker (thread) and add them all at the same time*

This is why GPUs shine at operations that can be done independently, like vector addition.




### **How GPU Threads are organized**

A GPU organizes its workers like:

```bash
Grid → many blocks
Block → many threads
```

Each thread handles one element in our array, each block is a group of threads, and each grid is a whole set of blocks. (see the [01_cuda_basics](#01_cuda_basics) for more details). So CUDA provides special variables:

1. `thradIdx.x` : Thread's index inside its block
2. `blockIdx.x` : Block's index inside the grid
3. `blockDim.x` : How many threads are in one block

So, to find the *global* element index, we do:

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

So thread 0 adds `A[0] + B[0]` and so on.



--- 

**Kernel- the code which runs on the GPU**

This is the GPU version of the CPU loop:

```cpp
__global__ void addVectors(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) { // to not go out of bounds
        c[idx] = a[idx] + b[idx];
        }
}
```

Notice that there are no `for` loops here, the `__global__` thing tells CUDA that this function runs on GPU. Each thread executes this function independently, and handles a different element of the vectors. Also, the line `if(idx<n)` ensures that we don't read/write outside the array. So, if we launch 10 million threads, all 10 million additions can happen simultaneously.

---

### **Result**

```bash
CPU average time: 0.009426 seconds
GPU average time: 0.000391 seconds
Speedup: 24.084838
Result verification: SUCCESS
```



---

### **3D Kernel**

In the `01_vector_addition_3d.cu` file, we reshaped the `N= 10000000` as $n_x \times n_y \times n_Z = 100 \times 100 \times 1000$

So, the launch shape is :

```cpp
#define BLOCK_SIZE_3D_X 16
#define BLOCK_SIZE_3D_Y 8
#define BLOCK_SIZE_3D_Z 8
// NOTE: 16 * 8 * 8 = 1024 (the comment “16*16*8=2048” is a typo)

dim3 block_size_3d(16, 8, 8); // 1024 threads per block
dim3 num_blocks_3d(
    (nx + 16 - 1) / 16,
    (ny +  8 - 1) /  8,
    (nz +  8 - 1) /  8
);
vector_add_gpu_3d<<<num_blocks_3d, block_size_3d>>>(d_a, d_b, d_c_3d, nx, ny, nz);
```


There are still 1024 threads per block, but now arranges in a 3D block.

The kernel does the 3D to 1D flattening using:

```cpp

__global__ void addVectors3D(float *a, float *b, float *c, int nx, int ny, int nz) {
    // Compute the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    // there are three addition operations, three multiplication operations

    if (idx < nx && idy < ny && idz < nz) {
        int index = idx + idy * nx + idz * nx * ny; // row-major order
        if (index < nx * ny * nz) {
            c[index] = a[index] + b[index];
        }
    }
}
```

So, we compute a 3D coordinated `idx, idy, idz` per thread, then convert it into a linear index `index = idx + idy * nx + idz *nx *ny`.

This shows how to map multi dim problems onto a linear memory.


---

**Performance Intiution**

Both kernels are memory bandwith bound, the GPU mostly waits on DRAM load/stores. The 1D version has the least index math and the simplest control flow, that makes it slightly faster. 
The 3D version does extra integer multiplication and additions to comput the `index`. The compiler is good at optimizaing, but there's still more arithmetic. 



### **When to prefer 1D vs 3D**

<li> 1D: flat arrays, simple element-wise ops, reductions, scans → easiest and often fastest.

<li> 3D: natural 3D domains (images, volumes, PDE grids) where indexing and block tiling along x/y/z give locality and make later optimizations (shared memory tiles, halos) straightforward.

---

**Results**

```bash
benchmarking GPU 1D kernel...
CPU average time: 0.009317 seconds
GPU 1D average time: 0.000447 seconds
Speedup: 20.857456
Result verification for GPU 1D: SUCCESS
benchmarking GPU 3D kernel...
Result verification for GPU 3D: SUCCESS
GPU 3D average time: 0.000474 seconds
Speedup (3D kernel): 19.673312
CPU average time: 9.316754 milliseconds
GPU 1D average time: 0.446687 milliseconds
GPU 3D average time: 0.473573 milliseconds
Speedup (CPU vs GPU 1D): 20.857456x
Speedup (CPU vs GPU 3D): 19.673312x
Speedup (GPU 1D vs GPU 3D): 0.943227x
```