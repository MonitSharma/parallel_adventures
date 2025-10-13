
# CUDA Kernels and GPU Information

This directory contains CUDA kernel examples, device queries, and technical documentation for NVIDIA GPUs. It is intended for developers, researchers, and students working with GPU-accelerated computing and CUDA programming.

## Overview

The files and resources here provide:
- Example CUDA kernels for learning and benchmarking
- Device query outputs and detailed GPU specifications
- Explanations of CUDA memory hierarchy, thread/block/grid structure, and advanced features

The primary GPU used for these examples is the **NVIDIA GeForce RTX 3070**.

---

## Example: CUDA Device Query Output

```
CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "NVIDIA GeForce RTX 3070"
  CUDA Driver Version / Runtime Version          12.7 / 12.5
  CUDA Capability Major/Minor version number:    8.6
  Total amount of global memory:                 8192 MBytes (8589410304 bytes)
  (046) Multiprocessors, (128) CUDA Cores/MP:    5888 CUDA Cores
  GPU Max Clock rate:                            1725 MHz (1.73 GHz)
  Memory Clock rate:                             7001 Mhz
  Memory Bus Width:                              256-bit
  L2 Cache Size:                                 4194304 bytes
  ...
```



# NVIDIA GeForce RTX 3070 - GPU Specifications

Below is a detailed breakdown of the capabilities and specifications of the **NVIDIA GeForce RTX 3070**, based on `nvidia-smi` and `deviceQuery` output. This information is useful for CUDA developers, researchers, and performance engineers.

---


## General Information

| Property | Value |
|--------|-------|
| **Device Name** | NVIDIA GeForce RTX 3070 |
| **CUDA Driver Version** | 12.7 |
| **CUDA Runtime Version** | 12.5 |
| **CUDA Capability** | 8.6 |
| **Compute Mode** | Default (Multiple host threads can use the device) |
| **PCI Bus ID** | 1 |


> The driver and runtime versions are compatible.  
> CUDA Capability 8.6 indicates support for modern features like Tensor Cores, concurrent execution, and enhanced memory operations.

---


## Memory & Bandwidth

| Property | Value |
|--------|-------|
| **Global Memory (VRAM)** | 8192 MB (8 GB) |
| **Memory Clock Rate** | 7001 MHz |
| **Memory Bus Width** | 256-bit |
| **L2 Cache Size** | 4 MB |

- **Global Memory**: Where your data (arrays, tensors, etc.) resides during GPU computation.
- **256-bit Bus + High Clock Rate**: Enables high memory bandwidth (~448 GB/s theoretical).
- **L2 Cache**: Helps reduce access latency to frequently used data.

---


## Processing Units

| Property | Value |
|--------|-------|
| **Multiprocessors (SMs)** | 46 |
| **CUDA Cores per SM** | 128 |
| **Total CUDA Cores** | 5888 |

The RTX 3070 has **46 streaming multiprocessors**, each containing **128 CUDA cores**, totaling **5888 cores** — ideal for highly parallel workloads like deep learning, simulations, and image processing.

---


## Thread & Memory Hierarchy


### Shared Memory & Registers

| Resource | Size |
|--------|------|
| **Shared Memory per Block** | 48 KB |
| **Registers per Block** | 65,536 |
| **Constant Memory** | 64 KB |

- **Shared Memory**: Fast on-chip memory shared by threads in a block. Critical for optimizing kernel performance.
- **Registers**: Fastest memory; used per-thread for variables.
- **Constant Memory**: Optimized for read-only data accessed by all threads (e.g., coefficients, lookup tables).

---


## Thread Execution Model

| Parameter | Value |
|--------|-------|
| **Warp Size** | 32 threads |
| **Max Threads per Block** | 1024 |
| **Max Threads per SM** | 1536 |
| **Max Block Dimensions** | (1024, 1024, 64) |
| **Max Grid Dimensions** | (2,147,483,647, 65535, 65535) |

- **Warp**: The fundamental unit of execution. All 32 threads in a warp execute the same instruction at the same time (**SIMT**).
- **Thread Blocks**: Up to 1024 threads can be grouped into a block.
- **Grids**: Can scale to billions of blocks, enabling massive parallelism.

---


## Advanced Features

| Feature | Supported |
|-------|-----------|
| **Unified Addressing** | ✅ Yes |
| **Managed Memory (CUDA Unified Memory)** | ✅ Yes |
| **Concurrent Copy and Kernel Execution** | ✅ Yes |
| **Compute Preemption** | ✅ Yes |
| **Cooperative Kernel Launch** | ✅ Yes |
| **Kernel Execution Timeout** | ✅ Yes (TDR enabled) |
| **ECC Memory** | ❌ Disabled (typical for consumer GPUs) |


### Key Benefits
- **Unified/Managed Memory**: Simplifies memory management — use `cudaMallocManaged()` to let CUDA handle CPU/GPU data movement.
- **Concurrent Execution**: Overlap memory transfers with computation for better performance.
- **Cooperative Kernels**: Enable synchronization between thread blocks (advanced use cases).

---


## Texture & Surface Support

| Property | Value |
|--------|-------|
| **Max 1D Texture Size** | 131,072 elements |
| **Max 2D Texture Size** | 131,072 × 65,536 |
| **Max 3D Texture Size** | 16,384 × 16,384 × 16,384 |

Useful for image processing, rendering, and scientific visualization.

---


## Summary (Quick Reference)

| Feature | Value |
|--------|-------|
| **VRAM** | 8 GB GDDR6 |
| **CUDA Cores** | 5888 |
| **SM Count** | 46 |
| **Core Clock** | 1725 MHz (Max) |
| **Warp Size** | 32 |
| **Threads per Block** | Up to 1024 |
| **Shared Memory** | 48 KB / block |
| **Unified Memory** | Supported |
| **Compute Capability** | 8.6 |

---


## Tips for Developers


- Use `cudaMallocManaged()` and `__device__`/`__managed__` variables for easier memory handling.
- Maximize occupancy by tuning block size and shared memory usage.
- Use `nvprof` or `Nsight Compute` to profile kernel performance.
- Enable `-arch=sm_86` when compiling for optimal performance:
  ```bash
  nvcc -arch=sm_86 kernel.cu -o kernel
  ```

---

## Directory Contents

- Example CUDA kernels: see `01_cuda_basics/`
- Device and memory hierarchy explanations: see `cuda_details.md`
- Block addressing and shared memory: see `01_cuda_basics/block_address.png`

For more details on CUDA programming concepts, refer to the main repository README and the official [CUDA documentation](https://docs.nvidia.com/cuda/).