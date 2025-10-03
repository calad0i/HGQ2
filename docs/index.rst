.. High Granularity Quantization documentation master file, created by
   sphinx-quickstart on Sat Nov 25 16:28:11 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===============================
High Granularity Quantization 2
===============================

.. image:: https://img.shields.io/badge/repo-github-gx
   :target: https://github.com/calad0i/hgq2
   :alt: repo

.. image:: https://img.shields.io/pypi/v/hgq2?color=green
   :target: https://pypi.org/project/HGQ2/
   :alt: PyPI

.. image:: https://img.shields.io/badge/License-LGPLv3-blue.svg
   :target: https://www.gnu.org/licenses/lgpl-3.0.en.html
   :alt: LGPLv3

.. image:: https://github.com/calad0i/HGQ2/actions/workflows/sphinx-build.yml/badge.svg
   :target: https://calad0i.github.io/HGQ2/
   :alt: Documentation

HGQ2 (High Granularity Quantization 2) is a quantization-aware training framework built on Keras v3, targeting real-time deep learning applications on edge devices like FPGAs. It provides a comprehensive set of tools for creating and training quantized neural networks with minimal effort.

HGQ2 implements an gradient-based automatic bitwidth optimization and quantization-aware training algorithm. By laveraging gradients, it allows for bitwidth optimization at arbitrary granularity, up to per-weight and per-activation level.

.. rst-class:: light
.. image:: _static/overview.svg
   :alt: HGQ-overview
   :width: 600

Key Features
-----------


- **High Granularity**: HGQ supports per-weight and per-activation bitwidth optimization, or any other lower granularity.
- **Automatic Quantization**: Bit-widths are optimized via gradients, no need to manually tune them in general.
- **What you see is what you get**: One get exactly what you get from ``Keras`` models from ``RTL`` models.
  - still subject to machine float precision limitation.
- **Accurate Resource Estimation**: ``EBOPs`` estimated by HGQ gives a good indication of the actual resource usage on FPGA, either upper limit of ``LUT`` (``da4ml``) or ``LUT + 55 * DSP`` (``hls4ml``).

In addition, this framework improves upon the old HGQ implementation in the following aspects:

- **Scalability**: HGQ2 supports ``TensorFlow``, ``JAX``, and ``PyTorch``. As XLA compilation in ``JAX`` and ``TensorFlow`` can significantly speed up the training process. Training speed on HGQ2 can be 1.2-5 times faster than the previous implementation.
- **Quantizers**:
  - *Fixed-point*: While the last implementation only optimizes the number of floating bits with one way of parameterizing the fixed-point numbers, HGQ2 supports multiple ways of parametrizing them, and allows of optimizing any part of them via gradients.
  - *Minifloat*: Training with minifloat quantization is supported, also with surrogate gradients support (alpha quality).
- **More Layers**: More layers are supported now, including the powerful ``EinsumDense(BatchNorm)`` layer and the ``MultiHeadAttention`` layer with bit-accurate softmax and scaled dot-product attention.


.. code-block:: python
   :caption: Simple example

   import keras
   from hgq.layers import QDense, QConv2D
   from hgq.config import LayerConfigScope, QuantizerConfigScope

   # Setup quantization configuration
   # These values are the defaults, just for demonstration purposes here
   with (
      # Configuration scope for setting the default quantization type and overflow mode
      # The second configuration scope overrides the first one for the 'datalane' place
      QuantizerConfigScope(place='all', default_q_type='kbi', overflow_mode='SAT_SYM'),
      # Configuration scope for enabling EBOPs and setting the beta0 value
      QuantizerConfigScope(place='datalane', default_q_type='kif', overflow_mode='WRAP'),
      LayerConfigScope(enable_ebops=True, beta0=1e-5),
   ):
      model = keras.Sequential([
         QConv2D(32, (3, 3), activation='relu'),
         keras.layers.MaxPooling2D((2, 2)),
         keras.layers.Flatten(),
         QDense(10)
      ])


Index
=================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   status.md
   install.md
   getting_started.md
   qkeras.md
   faq.md

.. toctree::
   :maxdepth: 3
   :caption: API Reference:

   autodoc/hgq
   autodoc/hgq.config
   autodoc/hgq.constraints
   autodoc/hgq.layers
   autodoc/hgq.quantizer
   autodoc/hgq.regularizers
   autodoc/hgq.utils
   autodoc/qkeras

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
