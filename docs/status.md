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
- ✅ **XLA compilation** for accelerated training, Jax and TensorFlow only (PyTorch dynamo is under investigation)
- ✅ **hls4ml integration** for FPGA deployment

### Quantization Types
- ✅ **kif** (Keep-Integer-Fraction) fixed-point quantization
- ✅ **kbi** (Keep-Bit-Integer) fixed-point quantization

### Layers
- ✅ **Core layers**: QDense
- ✅ **Non-linear Activation layers**: QUnaryFunctionLUT
- ✅ **Convolutional layers**: QConv1D, QConv2D, QConv3D (non-dilated only)
- ✅ **QDepthwiseConv1D/2D, QSeparableConv1D/2D**: Aliases of QConv. Depthwise is a grouped QConv (groups = in_channels, filters = in_channels * depth_multiplier); separable is a QDepthwiseConv followed by a 1x1 QConv.
- ✅ **Normalization**: QBatchNormalization
- ✅ **Basic operation layers**: QAdd, QMultiply, QSubtract, QMaximum, QMinimum

### Utilities
- ✅ **trace_minmax**: Calibration for bit-exact inference
- ✅ **EBOP tracking**: Resource estimation during training
- ✅ **BetaScheduler**: Dynamic adjustment of regularization strength

## Alpha/Beta Features

These features are functional but may undergo changes or improvements:

### Beta Features
- 🟡 **QEinsumDense, Qinsum**: Only `io_parallel` supported in hls4ml. Mostly stable but implementation at `hls4ml` is not fully optimal. `parallelization_factor` is not implemented.
- 🟡 **QEinsumDenseBatchnorm, QBatchNormDense**: Fused batchnorm layers. We observe some instability for them during training. Once trained, conversion to firmware is regarded as stable.
- 🟡 **QSoftmax**: Bit-exact quantized softmax implementation for `hls4ml` convertion. May be unstable during training. Once trained, conversion to firmware is regarded as stable.
- 🟡 **QDot**: The `hls4ml` implementation supports only `i,i->` pattern dot product, and only works in `io_parallel`. Use `QEinsum` instead if more general dot product is needed.

### Alpha Features
- 🟠 **Minifloat quantization**: Training support is available, but hardware synthesis is not yet supported. Not widely tested.
- 🟠 **QMultiHeadAttention**: Only `io_parallel` supported in `hls4ml`. Implementation is likely not optimal, but can generate synthesizable firmware.
- 🟠 **QSum, QMeanPow2**: No `hls4ml` support at the moment.
- 🟠 **QConv3D**: No `hls4ml` support.

```{warning}
`QConv*D` layer has greater numerical instability on the `torch` backend compared to `jax` and `tf`. Expect to slight difference between the keras model and the `hls4ml` model. Though, the difference should not affect the performance of the model significantly.
```

## Version Compatibility

- HGQ2 requires Keras v3
- For hls4ml integration, the fork at https://github.com/calad0i/hls4ml/tree/da4ml-v2 is required

## Reporting Issues

If you encounter any issues or bugs, please report them on the GitHub repository issue tracker with:

1. A minimal reproducible example
2. Your environment details (OS, backend, Python version, etc.)
3. Expected vs. actual behavior
