### Why Shared Memory?

In the basic matrix multiplication, each thread reads many elements from global memory , which is slow. Shared memory is fast, shared among threads inside a block, great for reusing data within the block.

This technique is called tiling, we break the big matrices into tiles, and threads cooperate to load and compute them efficiently.

### The Idea

For matrix C = A $\times $ B, 
- Each thread computes one element `C[row][col]`
- Instead of reading full rows/cols repeatedly from global memory each block:
    - Loads a small tile of A and B into shared memory
    - Computes part of C from that tile
    - Repeats this in steps (sliding tiles across)


