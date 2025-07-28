# 🚀 CUDA Basics Explained

CUDA (Compute Unified Device Architecture) is NVIDIA’s parallel computing platform that allows you to run C/C++ code directly on the **GPU (Graphics Processing Unit)** instead of the CPU.

This unlocks **massive parallelism** — the ability to execute thousands of operations simultaneously, making it ideal for deep learning, scientific computing, image processing, and high-performance computing (HPC).

---

## 🖥️ CPU (Host) vs GPU (Device)

| Component | Role |
|---------|------|
| **Host (CPU)** | Runs the main program, manages system resources, and uses system RAM. |
| **Device (GPU)** | Specialized for parallel computation; has its own cores and VRAM (video memory). |

Data must be explicitly transferred between the CPU and GPU — they do **not** share memory automatically.

---

## 🔁 The CUDA Runtime Flow

Every CUDA program follows this general workflow:

1. **Allocate** memory on both CPU and GPU.
2. **Copy** input data from CPU → GPU.
3. **Launch** a kernel (parallel function) on the GPU.
4. **Compute** in parallel across thousands of threads.
5. **Copy** results back from GPU → CPU.
6. **Free** GPU memory.

---

## 📛 Naming Convention

To keep track of where data lives, a common naming scheme is used:

| Variable | Meaning |
|--------|--------|
| `h_A` | Data `A` on the **Host** (CPU) |
| `d_A` | Data `A` on the **Device** (GPU) |

Example:

    float *h_input;  // Host array
    float *d_input;  // Device array

---

## 🔧 Special CUDA Function Types

| Keyword | Where It Runs | Callable From | Use Case |
|--------|---------------|----------------|---------|
| `__global__` | GPU | CPU (launch) | Kernel function — entry point for GPU execution |
| `__device__` | GPU | `__global__` or `__device__` | Helper function used inside GPU code |
| `__host__` | CPU | CPU | Regular function (default if omitted) |

> ✅ A function can be both `__host__ __device__` to run on both CPU and GPU.

---

## 🧠 Memory Management

You must manually manage GPU memory using CUDA runtime APIs.

### Allocate Memory on GPU

    cudaMalloc(&d_a, N * sizeof(float));

### Copy Data Between CPU and GPU

    // Host to Device
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);

    // Device to Host
    cudaMemcpy(h_result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost);

### Free GPU Memory

    cudaFree(d_a);

> ⚠️ Always check for errors: `cudaGetLastError()` and `cudaDeviceSynchronize()`.

---

## ⚙️ Compiling with `nvcc`

Use NVIDIA’s CUDA compiler:

    nvcc -arch=sm_86 kernel.cu -o kernel

How it works:
- **Host code** → compiled to x86 instructions.
- **Device code (kernels)** → compiled to **PTX** (Parallel Thread Execution), an intermediate GPU assembly.
- PTX is **JIT-compiled** at runtime to match your GPU architecture.

> 🔍 Use `-arch=sm_86` for RTX 3070 (Compute Capability 8.6).

---

## 🏗️ CUDA Execution Hierarchy

CUDA organizes parallel work in a three-level hierarchy:

    Grid
    └── Block(s)
        └── Thread(s)

Think of it like:
- **Grid** = City
- **Block** = Building
- **Thread** = Worker inside the building

### Key Terms

| Term | Meaning |
|------|--------|
| `gridDim` | Number of blocks in the grid (`gridDim.x`, `.y`, `.z`) |
| `blockIdx` | Index of the current block (`blockIdx.x`, etc.) |
| `blockDim` | Number of threads per block (`blockDim.x`, etc.) |
| `threadIdx` | Index of the current thread within its block |

---

## 🧵 Threads

- A **thread** is the smallest execution unit.
- All threads run the **same kernel code**, but operate on **different data**.
- Example: Adding two arrays — each thread adds one element.

---

## 🔀 Warps

- A **warp** is a group of **32 threads** that execute **in lockstep**.
- Managed automatically by the GPU.
- Understanding warps helps avoid performance issues like **warp divergence** (when some threads take a branch and others don’t).

---

## 🧱 Blocks

- Threads in the same block can:
  - Share fast **shared memory**.
  - Synchronize using `__syncthreads()`.
- Blocks are **independent** — no direct communication between them.
- Max threads per block: **1024** (common sizes: 256, 512, 1024).

---

## 🗺️ Grids

- A **grid** is a collection of blocks.
- Used to scale computations across large datasets.
- Example: Processing a 10,000-element array with 256 threads per block → need `ceil(10000 / 256) = 40` blocks.

---

## 🧮 Example: Add Two Arrays

    __global__ void addArrays(int *a, int *b, int *c, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            c[idx] = a[idx] + b[idx];
        }
    }

### Launch the Kernel

    dim3 blockSize(256);                    // 256 threads per block
    dim3 gridSize((size + 255) / 256);     // Enough blocks to cover all elements
    addArrays<<<gridSize, blockSize>>>(d_a, d_b, d_c, size);
    cudaDeviceSynchronize();  // Wait for GPU to finish

---

## 🧠 Thread Indexing

To compute a unique global thread index:

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

This gives each thread a unique ID so it knows which data element to process.

---

## 🔍 Memory Types (Speed & Scope)

| Memory Type | Access By | Speed | Scope | Use Case |
|------------|----------|-------|-------|---------|
| **Registers** | One thread | Fastest | Private | Local variables |
| **Shared Memory** | All threads in a block | Very fast | Block | Shared cache, communication |
| **Global Memory** | All threads | Slower | Entire device | Input/output arrays |
| **Constant Memory** | All threads | Fast (read-only) | Entire device | Constants, config |
| **Local Memory** | One thread | Slow (in global) | Private | Spilled variables (avoid!) |

> 💡 Use `__shared__` memory to speed up repeated access within a block.

---

## 🧰 Helper Syntax and Types

### `dim3` – Define 1D, 2D, or 3D Dimensions

    dim3 blockSize(16, 16);     // 256 threads per block (16x16)
    dim3 gridSize(10, 10);      // 100 blocks total

### Kernel Launch Syntax: `<<< >>>`

    kernel<<<gridSize, blockSize>>>(args...);

This launches the kernel with the specified grid and block configuration.

---

## 🧪 Synchronization

Wait for the GPU to finish:

    cudaDeviceSynchronize();

Useful after kernel launch to ensure results are ready before copying back or printing.

> 🔍 For async operations, use CUDA streams.

---

## 🧠 Why the Grid-Block-Thread Hierarchy?

Why not just launch millions of threads?

Because:
- GPUs schedule work in **warps of 32 threads**.
- **Blocks** allow threads to cooperate via shared memory and synchronization.
- **Grids** let us scale beyond hardware limits (e.g., more data than fits in one block).
- Each block has limited shared memory and thread count.

This hierarchy balances **flexibility**, **performance**, and **hardware constraints**.

---

## 🔄 Full End-to-End Example (Pseudocode)

    // 1. Host data
    float *h_a, *h_b, *h_result;
    h_a = (float*)malloc(N * sizeof(float));
    h_b = (float*)malloc(N * sizeof(float));
    h_result = (float*)malloc(N * sizeof(float));

    // 2. Device data
    float *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_result, N * sizeof(float));

    // 3. Copy to GPU
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // 4. Launch kernel
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    addArrays<<<gridSize, blockSize>>>(d_a, d_b, d_result, N);

    // 5. Wait for completion
    cudaDeviceSynchronize();

    // 6. Copy result back
    cudaMemcpy(h_result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost);

    // 7. Free memory
    free(h_a); free(h_b); free(h_result);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_result);