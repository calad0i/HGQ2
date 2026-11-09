# Project Status

This page provides information about the current development status of HGQ2 features and components.

## Stable Features

The following features are well-tested and ready for production use (with hls4ml integration):

```{note}
Only `Vitis` and `Vivado` (deprecated) backends are fully supported for `hls4ml` integration. Intra-layer heterogeneous activation will **not** work with other backends. However, heterogeneous weights quantization **may** work with other backends, but we have not tested it.
```

### Core Functionality
- ✅ **Quantization-aware training** with fixed-point quantization
- ✅ **Multi-backend support** (TensorFlow, JAX, PyTorch)
- ✅ **XLA compilation** for accelerated training, JAX and TensorFlow only (for PyTorch, see `torch.compile` below)
- ✅ **alkaid/hls4ml integration** for FPGA deployment

### Quantization Types
- ✅ **kif** (Keep-Integer-Fraction) fixed-point quantization
- ✅ **kbi** (Keep-Bit-Integer) fixed-point quantization

### Layers
- ✅ **Core layers**: Q(Einsum)Dense, QConv1D, QConv2D, QConv3D
- ✅ **Non-linear Activation layers**: QUnaryFunctionLUT
- ✅ **Normalization**: QBatchNormalization
- ✅ **Basic operation layers**: QAdd, QMultiply, QSubtract, QMaximum, QMinimum
- ✅ **Attention**: QMultiHeadAttention. Enable flash attention with `softmax='1pass'` or `'2pass'`.
- ✅ **Lookup Table Based Layers**: Q(Einsum)DenseT, QConv*DT, QMultiHeadAttentionT.

### Beta Features

These features are functional but may undergo changes or improvements:

- 🟡 **torch.compile**: Bunch of patches applied against torch and keras, upstreaming planned but will take time. See [Triton Routes and torch.compile](triton.md).
- 🟡 **Triton routes**: Opt-in Triton kernels for `QDense`, `QEinsumDense`, and `QMultiHeadAttention` with `softmax='1pass'` or `'2pass'` (FA impl), on the `jax` and `torch` backends with GPUs. Unsupported configurations fall back to XLA ops. See [Triton Routes and torch.compile](triton.md).

### Alpha Features
- 🟠 **Minifloat quantization**: Training support is technically, did not find any real use case so far and stucking in limbo.

```{warning}
`QConv*D` layer has greater numerical instability on the `torch` backend compared to `jax` and `tf`. Expect to slight difference between the keras model and the `hls4ml` model. Though, the difference should not affect the performance of the model significantly.
```

## Version Compatibility

- HGQ2 requires Keras v3
- Alkaid is recommended for hardware implementation. `alkaid>=1.0` is required (the tracing mechanism is not compatible with `alkaid<1.0`).
- hls4ml from the latest master supports **most** operations in `io_parallel`.
- `torch.compile` support requires `torch>=2.12`. Triton routes on `jax` require `jax-triton>=0.4`
- The Triton routes and `torch.compile` support are tested with `torch` 2.15 and `triton` 3.9, and with `jax` 0.12 and `jax-triton` 0.4.1, all on a Linux host. Non-linux host may work or may not work.

## Reporting Issues

If you encounter any issues or bugs, please report them on the GitHub repository issue tracker with:

1. A minimal reproducible example
2. Your environment details (OS, backend, Python version, etc.)
3. Expected vs. actual behavior
