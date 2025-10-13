
# JAX: High-Performance Numerical Computing

This directory contains resources, tutorials, and notebooks for learning and experimenting with JAX, a high-performance numerical computing library from Google. JAX enables accelerated computation using the power of GPUs and TPUs, and provides a NumPy-like API with automatic differentiation, vectorization, and just-in-time compilation.

## What is JAX?

JAX is a Python library designed for high-performance machine learning research and scientific computing. It combines familiar NumPy syntax with powerful features such as:

- **Automatic differentiation** (autograd)
- **Just-in-time (JIT) compilation** to XLA for speed
- **Vectorization** via `vmap`
- **Seamless GPU/TPU acceleration**

JAX is widely used in research and production for deep learning, optimization, and scientific simulations.

## Installation

It is recommended to use a virtual environment for Python projects.

Install JAX with CUDA support (for NVIDIA GPUs):

```bash
pip install --upgrade "jax[cuda12]"
```

If you do not require GPU support, you can install the CPU-only version:

```bash
pip install --upgrade jax
```

> **Note:** Ensure you have the appropriate CUDA and cuDNN versions installed for GPU support. Refer to the [official JAX installation guide](https://github.com/google/jax#installation) for details.

## Verifying Installation

To verify your JAX installation and check available devices:

```python
import jax
print(jax.devices())
```

To list available Jupyter kernels:

```bash
jupyter kernelspec list
```

## Directory Contents

- `intro_to_parallel_programming.ipynb` – Introduction to parallel programming concepts using JAX
- `quick_start.ipynb` – Quick start guide for JAX basics
- `tutorial_1.ipynb`, `tutorial_async.ipynb`, `tutorial_n.ipynb` – Additional tutorials and advanced topics

## Example: JAX vs NumPy

```python
import jax.numpy as jnp
import numpy as np

# NumPy array
a = np.array([1.0, 2.0, 3.0])
# JAX array
b = jnp.array([1.0, 2.0, 3.0])

print(np.sin(a))      # NumPy computation
print(jnp.sin(b))    # JAX computation (runs on CPU/GPU/TPU)
```

## Resources

- [JAX GitHub Repository](https://github.com/google/jax)
- [JAX Documentation](https://jax.readthedocs.io/en/latest/)
- [JAX Tutorials](https://jax.readthedocs.io/en/latest/notebooks/index.html)
- [JAX Installation Guide](https://github.com/google/jax#installation)

## License

This directory is intended for educational and research purposes. Please refer to the main repository license for details.