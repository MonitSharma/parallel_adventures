## GDB - The GNU Debugger for C/C++

When your C/C++ program crashes, prints the wrong output, or hangs, you need a way to :

- Stop the program at any point
- Inspect variables and memory
- Step through code line-by-line
- Watch how your logic behaves

That's where a debugger like `gdb` comes in.


### What is GDB?

GDB stands for GNU Debugger. It allows you to:
- Run your program interactively
- Pause at specific lines (breakpoints)
- Inspect the state of your variables, memory and call stack.
- Step through the program manually.


### Installing and Compiling

To install on Linux
        sudo apt install gdb

on Macos

        brew install gdb


Use WSL for windows.


For compiling:

        g++ -g -o myapp main.cpp

The `-g` flag tells the compiler to include source-level debugging information.

🔍 Basic GDB Workflow

Let’s say you have a compiled program `./myapp`.

Start GDB:

    gdb ./myapp

Once inside GDB, you'll see a `(gdb)` prompt. Now you can interact with the program.

---

🧭 Common GDB Commands (with Purpose)

    Command				Description
    run / r				Starts the program
    break / b			Set a breakpoint on a line or function
    next / n			Executes next line (skips over function calls)
    step / s			Executes next line and steps into function calls
    continue / c		Continue program until next breakpoint
    list / l			Shows source code near current position
    print / p			Print value of a variable (p x)
    disable				Disables a breakpoint (without removing it)
    enable				Enables a previously disabled breakpoint
    clear				Removes all breakpoints
    quit / q			Exit GDB

---

🧠 Function-Level Control: next vs step

    Command		Behavior
    next		Executes next line without entering functions
    step		Executes next line and enters any called function

Example:

    int main() {
        foo();  // <- line with function call
    }

- `next` will skip over `foo()`
- `step` will take you inside `foo()` line by line

---

🔬 Assembly-Level: nexti and stepi

    Command		Level			Purpose
    nexti		Instruction		Executes next machine instruction (no step-in)
    stepi		Instruction		Executes next instruction and steps into functions

Useful for low-level debugging (e.g., compiler bugs, memory corruption, optimization issues).

---

🧪 Practical Example Session

    $ gdb ./myapp
    (gdb) break main
    (gdb) run
    (gdb) next
    (gdb) print x
    (gdb) step
    (gdb) continue
    (gdb) quit

---

💡 Tips

- Use `break <function>` to stop at the beginning of a function.
- Use `break <filename>:<line>` to break at a specific line in a file.
- Use `backtrace` to print the current call stack if a crash occurs.
- GDB can analyze segfaults: just run the program and wait for it to crash.

---

🚀 Bonus: CUDA Debugging

If you're debugging CUDA (GPU) programs:

- Use `cuda-gdb` instead of `gdb`
- It supports debugging across both CPU and GPU code

---

✅ Summary

    Tool			Purpose
    gdb				Debug C/C++ programs on CPU
    cuda-gdb		Debug GPU kernels (NVIDIA/CUDA)
    -g				Include debug symbols during compile
    make debug		Often used to build with -g