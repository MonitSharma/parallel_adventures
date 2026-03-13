# Parallel Adventures


![](images/parallel_adventure.png)

This repository provides a comprehensive exploration of parallel programming concepts and tools, with a focus on C/C++, CUDA, JAX, and PyTorch. It is designed for developers, researchers, and students interested in understanding and experimenting with parallelism on both CPU and GPU architectures.

## Repository Structure

- [`C/`](C/)  
  Contains annotated examples and notes on C and C++ development, including pointers, memory layout, macros, type casting, compilers, Makefiles, and debugging (CPU and GPU).

- [`cuda_kernels/`](cuda_kernels/)  
  Includes CUDA kernel examples, device queries, and documentation related to GPU programming and NVIDIA hardware capabilities.

- [`LeetGPU/`](LeetGPU/)  
  Provides additional CUDA examples and documentation for vector operations and related GPU computations.

- [`JAX/`](JAX/)  
  Features JAX-based notebooks and resources for parallel programming in Python, including installation instructions and introductory tutorials.

- [`PyTorch/`](PyTorch/)  
  Contains PyTorch notebooks and scripts, primarily based on the "Learn PyTorch for Deep Learning: Zero to Mastery" curriculum, covering fundamentals, workflows, and advanced topics.

- [`requirements.txt`](requirements.txt)  
  Lists Python dependencies required for running the JAX and PyTorch notebooks.

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MonitSharma/parallel_adventures.git
   cd parallel_adventures
   ```

2. **Set up the Python environment:**
   It is recommended to use a virtual environment.
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Explore the content:**
   - Review the `C/` and `cuda_kernels/` folders for C/C++ and CUDA code samples.
   - Open the JAX and PyTorch notebooks in your preferred Jupyter environment.

## Notebooks

- JAX and PyTorch notebooks are located in their respective folders.  
- For JAX, follow the installation instructions in `JAX/README.md`.  
- For PyTorch, refer to the notebook list in `PyTorch/README.md`.

## License

This repository is intended for educational and research purposes. Please review individual files and subdirectories for any specific licensing information.
