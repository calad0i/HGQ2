# Getting Started with HGQ2

## Overview

HGQ2 (High Granularity Quantization 2) is a quantization-aware training framework built on Keras v3, targeting real-time deep learning applications on edge devices like FPGAs. It provides a comprehensive set of tools for creating and training quantized neural networks with minimal effort with streamlined integration with hls4ml.

A complete example of using HGQ2 can bbe found at [example/small_jet_tagger.ipynb](https://github.com/calad0i/HGQ2/blob/master/example/small_jet_tagger.ipynb) for a small jet tagging model. This example demonstrates how to create a quantized model, train it, and convert it for hardware deployment using hls4ml.

## Key Features

- **Multi-backend support**: Works with TensorFlow, JAX, and PyTorch through Keras v3
- **Flexible quantization**: Supports different quantization schemes including fixed-point and minifloat
- **Hardware synthesis**: Direct integration with hls4ml for FPGA deployment
- **Trainable quantization parameters**: Optimize bitwidths through gradient-based methods
- **Effective Bit-Operations (EBOP)**: Resource estimation for hardware deployment

## Basic Usage

### 1. Creating a Quantized Model

Here's a simple example of creating a quantized model for MNIST classification:

```python
import keras
import numpy as np
from hgq.layers import QConv2D, QDense
from hgq.config import LayerConfigScope, QuantizerConfigScope

# First, set up quantization configuration
# For weights, use SAT_SYM overflow mode
with QuantizerConfigScope(q_type='kbi', place='weight', overflow_mode='SAT_SYM', round_mode='RND'):
    # For activations, use different config
    with QuantizerConfigScope(q_type='kif', place='datalane', overflow_mode='WRAP', round_mode='RND'):
        with LayerConfigScope(enable_ebops=True, beta0=1e-5):
            # Create model with quantized layers
            model = keras.Sequential([
                keras.layers.Reshape((28, 28, 1)),
                keras.layers.MaxPooling2D((2, 2)),
                QConv2D(16, (3, 3), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),
                QConv2D(32, (3, 3), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Flatten(),
                QDense(10)
            ])

# Compile model as usual
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
```

### 2. Training and Quantization

Training a quantized model is similar to training a standard Keras model, but you can use dynamic beta scheduling to efficiently explore the resource-performance tradeoff:

```python
# Prepare data (example for MNIST)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

from hgq.utils.sugar import FreeEBOPs
ebops = FreeEBOPs()

# Train model with beta scheduler
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=15,
    validation_data=(x_test, y_test),
    callbacks=[ebops] # It is recommended to use the FreeEBOPs callback to monitor EBOPs during training
    verbose=2
)
```

### 3. Converting to Hardware with hls4ml

After training, convert your model for hardware deployment:

```python
from hgq.utils import trace_minmax
from hls4ml.converters import convert_from_keras_model

# Trace the required number of integer bits for activations
# This step is only necessary when using WRAP overflow mode (recommended) for data.

trace_minmax(model, x_test, verbose=True)

# Convert to hls4ml model
hls_model = convert_from_keras_model(
    model,
    output_dir='hls_project',
    backend='Vitis',
    io_type='io_parallel'  # or 'io_stream' for streaming interface
)

# Write out and compile the C++ model for simulation
hls_model.write()
hls_model._compile()

# Test for mismatches
# HGQ2 and hls4ml should produce the same output, up to machine precision
# Notice that due to the quantization, internal mismatches may be amplified, but the vast majority of the output should match
keras_pred = model.predict(x_test)
hls_pred = hls_model.predict(x_test)
print(f"{np.sum(keras_pred != hls_pred)} / {np.prod(keras_pred.shape)} value mismatches")
```

```{note}
For certain models (without resource multiplexing, general non-linear function (e.g., sigmoid or tanh), or variable-to-variable multiplicatin), you may need to use the `da4ml` backend instead of `hls4ml` for HDL or HLS code generation. Consult the [da4ml documentation](https://github.com/calad0i/da4ml) for more details.
```

## Advanced Configuration

### Quantization Types

HGQ2 supports different quantization methods:

- **kif**: Fixed-point quantizer with integer and fractional bits
- **kbi**: Fixed-point quantizer with bit and integer parameters

In general, it is recommended to use the `kif` quantizer for data and `kbi` for weights.
HGQ2 also supports minifloat quantizers, which can be enabled by setting `q_type='float'`.
However, as minifloats are not supported by hls4ml, they are only useful for development at the moment.

```python
# Configure for specific quantization types
with QuantizerConfigScope(q_type='kif', overflow_mode='SAT_SYM', round_mode='RND'):
    # Model creation
```

One can also override the default quantizer type for a place (e.g., weights, table, bias, datalane) by setting the `place` parameter with `default_q_type` argument:

```python
with QuantizerConfigScope(default_q_type=..., place=...):
    # Model creation
```

Quantizer configuration scopes can be nested, and the innermost scope takes precedence. Each scope specifies the quantization type and place it is applied to. When `default_q_type` is set, it also sets the default quantization type for the specified place (Does not affect the scope of quantizers others parameters applies to, limited by `q_type`).

One may also set individual quantizer configurations by passing a `QuantizerConfig` object to the `*q_conf` arguments of the layers. The most common ones are `iq_conf` for input quantizer, and `kq_conf`, `bq_conf` for kernel and bias quantizers, respectively. For example. `QuantizerConfig` objects will take arguments from the current scope by default, the parameters passed to the `QuantizerConfig` object will take precedence.


### Heterogeneous vs. Homogeneous Quantization

Quantization granularity may be controlled with the `heterogeneous_axis` and `homogeneous_axis` parameters:

```python
# Per-channel quantization for weights
with QuantizerConfigScope(heterogeneous_axis=None, homogeneous_axis=()):
    # Model creation

# Per-batch quantization for activations
with QuantizerConfigScope(place='datalane', heterogeneous_axis=None, homogeneous_axis=(0,)):
    # Model creation
```

Only one of `heterogeneous_axis` and `homogeneous_axis` are mutually exclusive, and setting both will raise an error. The tuples passed to these parameters specify the axes along which quantization is (not) applied heterogeneously. For a complete example, please refer to the [example notebooks](https://github.com/calad0i/HGQ2/blob/master/example) located in the repository.

For more advanced configuration options, the user may override `bw_mapper` object in the quantizer config/scope. The `bw_mapper` object is of type `hgq.quantizer.internal.base.BitwidthMapperBase`, which is responsible for mapping between the quantization bitwidths and the data. Please refer to the `hgq.quantizer.internal.base.DefaultBitwidthMapper` class for the default implementation.
