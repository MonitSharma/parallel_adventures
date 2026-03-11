#include <iostream>
#include <math.h>


// function to add the elements of two arrays

void add(int n, float *x, float *y)  // keep in mind here, that x and y are pointers to arrays of floats
{
  for (int i = 0; i < n; i++)  // the loop runs from 0 to n-1
    y[i] = x[i] + y[i];        // each iteration reads x[i] and y[i], adds them and stores the result in y[i]
}

int main(void)
{
  int N = 1<<20; // means left-shift 1 by 20 bits, which is 1x2^20 = 1048576

  float *x = new float[N];   // allocate arryas on the heap, not on the stack
  float *y = new float[N];   // we do this because two arrays will be around 8 MB in size, linux has a stack limit of 8 MB per thread, while windows has 1 MB
  // so you might see 'Segmentation fault (stack overflow)` errors if you allocate large arrays on the stack, heap can easily handle this

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // add vectors
  add(N, x, y);

  // verify that the result is correct
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  delete[] x;
  delete[] y;

  return 0;
}

// g++ -o vector_add vector_add.cpp
// ./vector_add
// much better to delete is
// g++ -o vector_add vector_add.cpp && ./vector_add && rm vector_add

// or use bashrc
// cpprun vector_add.cpp