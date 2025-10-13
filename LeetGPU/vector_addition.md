## **Vector Addition**

We want to add two vector arrays, element by element: 

$$ C[i] = A[i] + B[i] $$

for every element `i` in the vectors.

When we write CUDA code, we are writing for two processors:

1. CPU (host)
2. GPU (device)

So, the program has two parts: 
A **kernel** which runs on the GPU and a **launcher** that tells the GPU to run that kernel.


In the `kernel` function, we have


```cpp
__global__ void vector_add(const float* A, const float* B, float* C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}
```

`__global___` is the function that runs on GPU and can be called from CPU, there are 4 parameters, which are the pointers to the input vector `A`, `B`, and `C` on GPU and `N` is the number of elements.

When we run the a kernel, CUDA automatically starts many small threads, individual workers. Each thread can process one (or more) element(s) of the vecotr. Each thread has a unique ID, which we can find using built-in variables:

1. `threadIdx.x` : index of thread within its block.
2. `blockIdx.x`  : which block does it belong to 
3. `blockDim.x`  : how many threads are there in the block

Now, to get a global index, i.e. a unique index for the entire grid, we did:

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

This is how the thread knows which element of `A` and `B` it should process. Then each thread just adds:

```cpp
C[i] = A[i] + B[i];
```

--- 
For the edge case, the GPU's can't always launch as many threads as the elements, like in the question $N\le 10^8$. So as a safety measure, we added a **grid-stride loop**:

```cpp
for (int i = idx; i < N ; i += blockDim.x * gridDim.x)
```

by doing this, each thread starts its index at `idx`, then it jumps ahead by the total number of threads and keep going until it reaches the end, this way, even if we have fewer threads than elements, all elements get processed.

