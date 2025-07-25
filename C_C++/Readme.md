# 📘 C/C++ Debugging & Compiler Exploration

This repository contains annotated examples and notes for key topics in C and C++ development, covering pointers, memory layout, macros, type casting, compilers, Makefiles, and both CPU & GPU debugging.



## 📂 Contents

- [Pointers and Memory Layout](#pointers-and-memory-layout)
- [Void and NULL Pointers](#void-and-null-pointers)
- [Arrays, Matrices, and Pointer Arithmetic](#arrays-matrices-and-pointer-arithmetic)
- [size_t and Type Information](#sizet-and-type-information)
- [Structs and Memory](#structs-and-memory)
- [Type Casting in C and C++](#type-casting-in-c-and-c)
- [Macros, Globals, and Conditional Compilation](#macros-globals-and-conditional-compilation)
- [Makefiles and CMake](#makefiles-and-cmake)
- [GDB: GNU Debugger](#gdb-gnu-debugger)
- [CUDA-GDB for GPU Debugging](#cuda-gdb-for-gpu-debugging)



## 🧠 Pointers and Memory Layout

We explored:
- Single, double, and triple pointers (`int*`, `int**`, `int***`)
- How memory addresses are incremented for arrays
- How pointer dereferencing works

Use cases:
- Dynamically managing data structures
- Navigating multidimensional arrays


## ❓ Void and NULL Pointers

**Void pointers**:
- Can point to any data type
- Need to be cast before dereferencing

**NULL pointers**:
- Used to indicate a pointer is not currently pointing to a valid memory location



## 🧮 Arrays, Matrices, and Pointer Arithmetic

We covered:
- Iterating arrays using pointer incrementation
- Matrix representation as an array of pointers
- Contiguous vs non-contiguous memory layouts



## 🧾 `size_t` and Type Information

- `size_t` is an unsigned integer type used to express object sizes (in bytes)
- We computed array length using:

        size_t size = sizeof(arr) / sizeof(arr[0]);

- `%zu` is the correct format specifier for printing `size_t`



## 📦 Structs and Memory

Example:

        typedef struct {
        	float x;
        	float y;
        } Point;

- Explored how struct memory is allocated
- `sizeof(Point)` results in 8 bytes due to float members


## 🔁 Type Casting in C and C++

### C-style:

        float f = 69.69;
        int i = (int)f;   // 69
        char c = (char)i; // 'E'

### C++ style:

| Cast               | Description                                      |
|--------------------|--------------------------------------------------|
| `static_cast`      | Checked at compile-time, safe for basic types    |
| `dynamic_cast`     | Used in inheritance trees, does runtime check    |
| `const_cast`       | Adds/removes const (⚠️ risky)                     |
| `reinterpret_cast` | Dangerous bit-level conversion (⚠️ unsafe)       |



## 🧾 Macros, Globals, and Conditional Compilation

We explored:
- `#define`, `#if`, `#elif`, `#else`, `#endif`
- `#ifndef` and `.PHONY` for Makefile logic
- Macro-based logic for adjusting constants like `radius`

Example:

        #define PI 3.14159
        #define AREA(r) (PI * r * r)

        #ifndef radius
        #define radius 7
        #endif

        #if radius > 10
        #define radius 10
        #elif radius < 5
        #define radius 5
        #else
        #define radius 7
        #endif



## 🧰 Makefiles and CMake

### Basic Makefile Structure

        target: prerequisites
        	command
        	another command

### What is `.PHONY`?

Use `.PHONY` to tell make not to treat a file named `clean` as a target:

        .PHONY: clean
        clean:
        	rm -rf build/*

### `:=` vs `=`

        A = $(B)
        B = hello
        C := $(B)
        B = world

        all:
        	@echo A is $(A)  # A is world
        	@echo C is $(C)  # C is hello

- `=` is lazy (re-evaluates on each use)
- `:=` is immediate (evaluated at definition)



## 🐞 GDB: GNU Debugger

### Compile with debug symbols:

        gcc -g -O0 -o bug_example bug_example.c

### Common GDB Commands

- `run`, `r`: Start program
- `break <func>`, `b <line>`: Set breakpoint
- `next`, `n`: Step over
- `step`, `s`: Step into
- `print`, `p`: Print variable
- `continue`, `c`: Resume execution
- `display`: Auto print on every step
- `quit`: Exit

### Example Session

        (gdb) break factorial
        (gdb) run
        (gdb) next
        (gdb) print n
        (gdb) step
        (gdb) continue



## 🚀 CUDA-GDB for GPU Debugging

### Compile with debug info:

        nvcc -g -G -o kernel kernel.cu

### Launch with:

        cuda-gdb ./kernel

### Example Commands

- `break kernel_func`
- `run`
- `info cuda threads`
- `cuda kernel breakpoints`
- `print threadIdx.x`



## 🎓 Further Learning

- Read: https://www.freecodecamp.org/news/what-is-a-compiler-in-c/
- Watch: https://www.youtube.com/watch?v=86FAWCzIe_4&list=WL&index=8
- `man gdb`, `man gcc`, `man make`

---
