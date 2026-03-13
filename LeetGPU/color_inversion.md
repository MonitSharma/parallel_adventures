## **Color Inversion**

You will learn an important GPU habit here : **thinking carefully about how data is laid out in memory**


### **Understanding the Color Inversion Problem**

This problem sounds like image processing, and technically it is, but under the hood, it is a really straightforward *array transformation problem*.

You are given an imahe stored as a 1D array of bytes, and your job is to modify each pixel so that the colors are inverted.

But to write it on a GPU, you need to understand three things clearly:

1. how the image is stored
2. what "invert color" means mathematically.
3. what each GPU thread should be responsible for,

Let;s see one layer at a time.


#### **RGBA**

Each pixel in the image is stored using four values: 
<li> R = Red
<li> G = Green
<li> B = Blue
<li> A = Alpha

Each of these is an `unsigned char`, which means it uses 8 bits and its value is between `0` and `255` ($2^8 -1$), and since its unsigned, it cannot store negative numbers, all `8` bits are used to represent the magnitude of the number.



---
```c
#include <stdio.h>

int main() {
    unsigned char byte = 200; 
    // signed char byte = 200; // This would overflow/wrap to -56 on most systems!

    printf("Value: %d\n", byte); // Prints: 200
    printf("Size: %zu bytes\n", sizeof(byte)); // Prints: 1

    return 0;
}
```
---

So, one pixel looks like this in the memory:

```c
[R,G,B,A]
[255,0,128,255]
```

You must be wondering, what is alpha here? Alpha represents **opacity**. `255` means fully opaque and `0` means fully transparent. Since this task is *color inversion*, we will leave the alpha unchanged.

So, if a pixel is:

```c
[10,20,30,255]
```

after inversion it becomes:

```c
[245,235,225,255]
```

#### **How the Image is stored?**

Even though we think of an image as a 2D grid of pixels, in memory it is stored as one long 1D array. The image is stored row by wor, left to right, top to bottom.

This means if the image has `width` and `height`, then:

1. total number of pixels = `width * height`
2. each pixel uses 4 elements
3. total number of array elements = `widht * height * 4`

So, the array looks like:

```bash
[R0, G0, B0, A0, R1, G1, B1, A1, R2, G2, B2, A2, ...]
```

Each group of 4 belongs to one pixel.

#### **What is the Computational Problem?**

More natural way of thinking this problem is:

*There are `width * height` pixels. Each pixel occupies 4 consecutive bytes. For each pixel, invert the first 3 bytes and keep the 4th byte unchanged.*

A very sensible design is we are not really processing bytes one by one, we are processing pixels. The distinction matters. The array is flat, but logically the data is grouped as :

```bash
[R0, G0, B0, A0, R1, G1, B1, A1, R2, G2, B2, A2, ...]
```

So, every pixel occupies 4 consecutive entries, so the real work unit is a one pixel, not one byte. So one GPU thread handles one pixel, this is the whole design idea.


#### **How Many Pixels are there?**
If the image has `width` and `height`, then the total number of pixels is: 

`total_pixels = width x height`

Each pixel has 4 values, but the number of actual image elements is based on pixels first.

#### **What does one thread represent?**

Each thread gets a global index:

```c
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

This `idx` is the pixel number that thread is responsivle for.

#### **Where does that pixel start in memory?**

Since each pixel uses `4` consecutive bytes, pixel `idx` starts at:

`base = 4 x idx`

So, the channels are: `image[base + 0] -> red` and  `image[base + 1] -> green` and so on. So the code uses:

```c
int base = 4 * idx;
```

#### **What does the inversion means?**

As discussed above, its subtraction from 255, and alpha remaining unchanged. To do that, we should have something like:

```c
image[base + 0] = 255 - image[base + 0];
image[base + 1] = 255 - image[base + 1];
image[base + 2] = 255 - image[base + 2];
```
and the alpha, which is `image[base + 3]` remains unchanged.

#### **Why do we need a bounds check?**

Suppose there are 1000 pixels, and you use 256 threads per block. Then

$ ⌈1000/256⌉=4 blocks $

That launches: $ 4 \times 256 = 1024$ threads, but only the first 1000 threads corresponds to real pixels, the last 24 threads are extra, they must do nothing. So we do

```c
if (idx < total_pixels)
```

and grid formulae is

```c
int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;
```

its a standard trick for ceiling division.

For example, if there are 1000 pixels and 256 threads per block:

3 blocks = 768 threads, not enough

4 blocks = 1024 threads, enough

So we round up.


Here's the full code:

```c
#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;

    if (idx < total_pixels) {
        int base = 4 * idx;  // start of this pixel's RGBA values

        image[base + 0] = 255 - image[base + 0];  // R
        image[base + 1] = 255 - image[base + 1];  // G
        image[base + 2] = 255 - image[base + 2];  // B
        // image[base + 3] is A, leave it unchanged
    }
}

// image is a device pointer (memory already on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int total_pixels = width * height;
    int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}
```