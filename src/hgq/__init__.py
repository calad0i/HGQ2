"""High Granularity Quantization 2
---------------------------------------------------

The HGQ2 library provides a set of tools to quantize neural networks meant to be deployed on edge devices, mainly FPGAs with Keras. The library is designed to be scalable, allowing for the construction of fully-quantized models suitable for deployment with minimal effort.

Provides
--------
- Scalable quantization-aware training
- Drop-in replacement quantized keras layers
- Support for various quantization schemes
- Trainable weights and quantization bitwidths
- TensorFlow/JAX/PyTorch backend support with Keras v3

Library Structure
-----------------
The library is organized in a keras-like structure, with the following modules:

- `config`: Configuration settings for the layers and quantizers
- `layers`: Quantized keras layers
- `quantizer`: Quantizer wrappers and internal quantizers
- `utils`: Utility functions and classes, and some useful sugars
- `constraints`: Custom constraints for quantization-aware training
- `regularizers`: Custom regularizers for quantization-aware training

"""

from . import config, layers, quantizer, utils
from ._version import __version__  # noqa: F401

__all__ = ['config', 'layers', 'quantizer', 'utils']
