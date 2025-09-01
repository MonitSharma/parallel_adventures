# **Macros and Conditional Compilation in C**

A guide to the C preprocessor for creating constants, function-like macros, and code that adapts at compile-time.

---

## 📝 **Macros: The Power of Text Substitution**

A macro in C is a rule for text replacement managed by the **preprocessor**, which runs *before* the actual compiler. It's defined using the `#define` directive.

> **Key Idea:** Macros are not variables. They have no type and are not stored in memory. The preprocessor literally swaps the macro's name with its content in your code.

There are two main types of macros:

### 1. Object-Like Macros
These are used to define constants. The preprocessor replaces every occurrence of the macro's name with its value.

    #define PI 3.14159
    #define GREETING "Hello, World!"

    double circumference = 2 * PI * radius;
    puts(GREETING);

### 2. Function-Like Macros
These accept arguments and behave like small code templates.

    #define AREA(r) (PI * (r) * (r))

    // During preprocessing, this line...
    double circle_area = AREA(7);

    // ...is transformed into this before compilation:
    double circle_area = (3.14159 * (7) * (7));

> **⚠️ Important Tip:** Always wrap macro arguments and the entire macro body in parentheses to avoid unexpected order-of-operations errors. For example, `AREA(x+1)` would break without the parentheses around `r`.

---

## 🚦 **Conditional Compilation**

Conditional compilation directives allow you to include or exclude blocks of code based on conditions evaluated by the preprocessor. This is like an `if-else` statement for the compiler.

It's commonly used to create different builds, such as a "debug" version with extra logging or a "release" version that's optimized.

### ✅ Example: Creating a Debug Build

    // Define a macro to control the build mode. Set to 1 for Debug, 0 for Release.
    #define DEBUG_MODE 1

    #include <stdio.h>

    int main() {
        int data = 100;

        #if DEBUG_MODE == 1
            // This block is only included in the code if DEBUG_MODE is 1
            printf("Debug: Data value is %d\n", data);
        #elif DEBUG_MODE == 2
            printf("Verbose Debug: Data at address %p\n", &data);
        #else
            // This is the default block for Release mode (when DEBUG_MODE is 0 or undefined)
            printf("Running in Release Mode.\n");
        #endif

        // This checks if a macro has been defined at all
        #ifdef GREETING
            printf("Greeting macro is defined.\n");
        #endif

        // This checks if a macro is NOT defined
        #ifndef USERNAME
            #define USERNAME "Default"
            printf("Username was not defined, set to %s\n", USERNAME);
        #endif

        return 0;
    }

> **Key Rule:** The expressions in `#if` and `#elif` must be constant integer values. You cannot use variables from your program, as these directives are evaluated long before your program runs.

---

### **`#define` vs. Global Variables: A Key Distinction**

The user's text correctly notes that a `#define` is not a true global variable. Here's a summary of the differences:

| Feature           | `#define` Macro                                   | True Global Variable (`const int`)            |
| ----------------- | ------------------------------------------------- | --------------------------------------------- |
| **Type Safety** | None. It's just text replacement.                 | Strong. The compiler knows it's an `int`.     |
| **Memory** | Not stored in memory.                             | Stored in memory, has a fixed address.        |
| **Scope** | File-scoped from the point of definition.         | Obeys normal C scope rules, can be `extern`.  |
| **Debugging** | Cannot be inspected in a debugger.                | Can be watched and inspected.                 |