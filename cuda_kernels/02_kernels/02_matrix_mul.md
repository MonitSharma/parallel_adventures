### **Matrix Multiplication**

**What is matrix Multiplication?**

A matrix is just a 2D grid of numbers, let's say we have:

```bash
Matrix A(N x N) 

Matrix B(N x N)
```

and we want to produce a result:

```bash
Matrix C = A x B
```

each element `C` is the dot product of a row from `A` and a column from `B`.

Mathematically:

```bash
C[i][j] = A[i][0] x B[0][j] + A[i][1] x B[1][j] + .... + A[i][N-1] x B[N-1][j]
```

so to fill one cell of `C`, we multiply and add across an entire row and column.


---

**How the CPU does it**

The CPU usually runs three nested loops:

```cpp
for (int i =0; i < N; i++) //rows of A
    for (int j = 0; j < N; j++) // columns of B
        {
                C[i][j] = 0;
                for (int k = 0; k < N; k++)// across row/col)
                    C[i][j] += A[i][k] * B[k][j];
        }  
```

that's one worker doing all the work in sequence, each iteration calculates one number in `C`, so if `N=16` that's $16 \times 16 \times 16 = 4096$ multiply-add operations.


**How the GPU does it**

A GPU can run thousands of threads at once, so instead of one cpu core calculating all of `C`, we assign each element `C[i][j]` to one GPU thread. That means hundreds of thousands of `C[i][j]` values are computed in parallel.


Although we talk about 2D matrices, they live as flat arrays:

$A$ is $M\times K$

$B$ is $K\times N $

Result $C = A\times B $ is $M \times N$


So, we define the constants:

```cpp
#define M 256
#define K 512
#define N 256
```




<li> A[i,l] is at A[i*K + l]
<li> B[l,j] is at B[l*N + j]
<li> C[i,j] is at C[i*N + j]


### **Inside the GPU kernel**

```cpp
__global__ void matmul_gpu(float *A, float *B, float *C, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int l = 0; l < k; l++) {
            sum += A[row * k + l] * B[l * n + col];
        }
        C[row *n + col] = sum;
    }
}
```

breaking it down, finding the element this thread should compute

```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

Every thread has a unique pair `(row, col)` that corresponds to one element in `C`. So thread `(3,5)` is responsivle for computing `C[3][5]`



then **Doing the dot product**

```cpp
float sum = 0.0f;
for (int l = 0; l <k ; l++) {
    sum += A[row * k + l] * B[l * n + col];
}
C[row * n + col] = sum;
```

This is where each thread computes its one result. It loops over all `l` values, multiplies the right element from `A` and `B` and sums them up. It's the same math as the CPU version, just done in parallel across thousands of threads.


Even though the matrices are 2D, they're stored linearly in memory:

```bash
A[row * K + l] → element from row “row” and column “l” in A
B[l * N + col] → element from row “l” and column “col” in B
C[row * N + col] → element from row “row” and column “col” in C
```
Each thread reads, one row of A (fixed `row`) amd one column of B (fixed `col`)




---

**Result**

```bash
Warming up GPU...
Benchmarking CPU...
Average CPU time: 0.071151 seconds
Benchmarking GPU...
Average GPU time: 0.000685 seconds
Speedup: 103.83
```