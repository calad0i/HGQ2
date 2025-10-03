# Installation Guide

## Requirements

`python>=3.10` is required to run HGQ2. We recommend using `python>=3.11` for better performance and compatibility.

HGQ2 is built on Keras v3, and essentially requires the same dependencies as Keras.
Keras v3 supports multiple backends, including TensorFlow, JAX, and PyTorch.
The following are the recommended versions for each backend:

- `tensorflow>=2.16`
- `jax>=0.4.28`
- `torch>=2.5.0`

You'll need to have at least one of these backends installed in your environment to train models with HGQ2.

## Installation

```bash
# For development (editable installation)
glt clone https://github.com/calad0i/HGQ2
cd HGQ2
pip install -e .

# For regular installation
pip install HGQ2
```

Consider install also `da4ml` or/and `hls4ml` for model conversion and synthesis:

```bash
pip install da4ml>=0.3
pip install hls4ml>=1.2.0
```

## Troubleshooting

### Common Issues

1. **Incompatible Keras Version**: HGQ2 requires Keras v3. If you see errors related to missing attributes or incompatible methods, check your Keras version:
   ```python
   import keras
   print(keras.__version__)  # Should be 3.x
   ```

2. **hls4ml Conversion Issues**: HGQ2 is supported in hls4ml since v1.2.0. Ensure you have the correct version installed:
   ```bash
   pip install hls4ml>=1.2.0
   ```
   or directly from the repository for the latest features:
   ```bash
   pip install 'git+https://github.com/fastmachinelearning/hls4ml.git'
   ```

### Getting Help

If you encounter issues not covered here:

1. Check the FAQ section in the documentation
2. Open an issue on the GitHub repository with details about your environment and the encountered error - or -
3. Ask for help on the [FastML Slack workspace](https://fastml.slack.com/) in the `#hgq2` channel

## Next Steps
Check out the [Getting Started Guide](getting_started.md) to learn how to use HGQ2 for quantization-aware training.
