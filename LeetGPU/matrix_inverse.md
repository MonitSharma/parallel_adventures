## **Matrix Transpose**

If you're learning GPU programming, writing a matrix transpose kerne is a rite of passage. On a CPU, it's just two nested `for` loops. But on a GPU?
You are unleashing thousands of threads simultaneously. To orchestrate this chaos, you need a rock-solid mental model of how data is stored and how threads are mapped to the data.

Let's see:

### **The Goal: Flipping the Grid**

In Linear Algebra, the transpose of a matrix, denoted as $A^T$ flips the matrix over its main diagnol. What was once a row becomes a column, ad vice versa.

Imagine a simple $3\times2$ matrix:

```
[ 1  2 ]
[ 3  4 ]
[ 5  6 ]
```

We transpose it, it becomes a $2\times3$ matrix:

```
[ 1  3  5 ]
[ 2  4  6 ]
```

An element located at row `r` and column `c` in the input matrix, will be moved to row `c` and column `r` in the output matrix.

<li> Input Dimensions : rows x columns
<li> Output Dimensions : columns x rows


### **The Illusion of 2D: Row-Major Memory Layout**

Here is the biggest hurdle for GPU beginners: GPUs do not have 2D memory. Hardware memory (RAM/VRAM) is fundamentally a 1D strip, like a massive tape measure. So how do we store a 2D grid on a 1D tape? 

In [C/C++](https://github.com/MonitSharma/parallel_adventures/tree/main/C), and [CUDA](https://github.com/MonitSharma/parallel_adventures/tree/main/cuda_kernels), we use a **Row-Major** order. This means we lay the matrix out row by row, end to end.

Our $3\times2$ input matrix from above looks like this in the physical GPU memory:

```
Memory Address:  0   1   2   3   4   5
Data:          [ 1 | 2 | 3 | 4 | 5 | 6 ]
                 ---Row 0--- ---Row 1--- ---Row 2---
```

Because memroy is 1D, we have to do the math ourselves to translate a 2D `(r,c)` coordinate inot a 1D memory index. The formula is:
`1D_Index = (row_index * matrix_width) + col_index`

Let's apply this to both our matrices.

**Finding the Input Index:**

<li> The input matrix width is columns
<li> To read the element at (r,c), the formula is input_index = r * columns + c

**Finding the Output Index**

<li> The Element moves to row c and column r
<li> The output matirx has a entirely different width, its width is rows (since the dimensions flipped)
<li> To write the element the formula is : output_index = c * rows + r


### **Mapping Threads to the Matrix**

See the code:

```bash
dim3 threadsPerBlock(16, 16);
dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
```

CUDA executes threads in groups called **Blocks**, and blocks are grouped into a **Grid**.
Here, we are defining a 2D block of $16 \times 16 = 256$ threads. We are taking these $16 \times 16$ "tiles" and paving them over our entire matrix.

Inside the GPU kernel, every single thread runs the exact same code. To make them do different work, each thread uses built-in hardware variables to figure out "who" it is and what `(r, c)` coordinate it is standing on.

1. `threadIdx.x` & `y`: The thread's position inside its specific block (0 to 15).

2. `blockIdx.x` & `y`: Which block the thread belongs to.

3. `blockDim.x` & `y`: The size of the block (16 in our case).

By convention, we map the x-axis to columns and the y-axis to rows.
To find the global, absolute coordinate of a thread across the entire matrix, we calculate:

`int c = blockIdx.x * blockDim.x + threadIdx.x;` (Global Column)


`int r = blockIdx.y * blockDim.y + threadIdx.y;` (Global Row)


### **The Oversized Blanket: Boundary Checks**

What happens if our matrix is $100 \times 100$? Our blocks are $16 \times 16$. If we use $6$ blocks horizontally (6 x 16 = 96), we won't cover the whole matrix, We are forced to use $7$ blocks (7 x 16 = 112)

This is what the complex `(columns + threadsPerBlock.x - 1) / threadsPerBlock.x` math in the template do, its a clever trick for integer ceiling division to ensure we launch enough blocks to cover the matrix. 

However, this creates a new problem: our threads are like an oversized blanket hanging off the edge of the bed. We have 112 threads horizontally, but only 100 columns of data!

If threads 100 through 111 try to read memory, they will access out-of-bounds garbage data, or worse, trigger a segmentation fault that crashes the GPU. We must protect our memory reads/writes with a guardrail:
`if (r < rows && c < cols) { ... }`


### **Assembling the Kernel**

Let's write the code:

1. Calculate our thread's 2D coordinate
2. Check if we are inside the matrix boundaries.

3. Calculate the 1D input and output indices using row-major math.

4. Move the data.


```c
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
```


### **NOTE**

This implementation is perfectly correct and fulfills all the constraints of the problem. It calculates the right math and moves the right data.

However, if you run this on a massive $7000 \times 6000$ matrix, you might notice it isn't as fast as you'd expect a GPU to be. Why? Because of Memory Coalescing.

While our threads read from the input matrix nicely (threads 0, 1, 2 read memory addresses 0, 1, 2), they write to the output matrix in giant jumps (c * rows + r). GPUs hate scattered writes; they want memory accessed in neat, contiguous chunks.

To solve this, professional CUDA developers use a technique involving Shared Memory to buffer the data on-chip, transpose it locally, and then write it back to main memory neatly. But that is a topic for later or [see here](https://github.com/MonitSharma/parallel_adventures/blob/main/cuda_kernels/02_kernels/03_shared_memory.cu)