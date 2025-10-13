## Makefiles, CMake, and Build Logic in C/C++

In C/C++ projects, especially multi-file ones, manual compilation becomes inefficient and error-prone. To manage dependencies and automate builds, we use:
- **Makefiles** - Rule based automation
- **CMake** - Cross-platform build system generator
Together, these tools make software building repeatable, portable and efficient.

### Makefile Basics

A Makefile is a file named `Makefile` that contains rules for how to build a project.

**General Syntax:**

        target: prerequisites
            <TAB> shell-command

- `target`: typically a file to generate
- `prerequisite` : files that target depends on
- `command` : executed only if target is missing or older than its prerequisites.


**Example**
        app: main.o util.o
            g++ main.o util.o -o app


### Compilation Workflow via Make

For a C++ project:

1. `.cpp` files are compiled to `.o` (object files)
2. `.o` files are linked into an executable


### CMake and `CMakeLists.txt`

CMake is a meta build tool, it doesn't build software itself, but generates platform-specific build files (like Makefiles or Visual Studio projects)

**Purpose**

- Platform-independent configuration (`CMakeLists.txt`)
- Clean abstraction across compilers and OSes
- Supports out-of-source builds, testing, packaging, etc.


### .PHONY target

If a file/directory exists with the same name as a make target, the rule may be skipped. 

So, declare target as `.PHONY`

        .PHONY: clean

        clean:
            rm -rf build/*

This ensures `make clean` always runs, even if a file named `clean` exists.



