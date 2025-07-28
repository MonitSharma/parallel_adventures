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



