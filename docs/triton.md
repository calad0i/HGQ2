# Triton Routes and torch.compile

HGQ2 has opt-in Triton kernels for the dense and attention layers on the `jax` and `torch` backends. Works with XLA in jax and `torch.compile` in torch.

## Triton Routes

### Requirements

The `jax` or `torch`, with a hardware triton works
- `jax`:
  - `jax>=0.12` and `jax-triton>=0.4`.
  - CUDA is supported, ROCm is theoretically supported. Intel XPU and Google TPU are not supported.
- `torch`:
  - `torch>=2.12` (triton is integrated).
  - CUDA is supported, ROCm and Intel XPU are theoretically supported.

See [Installation](install.md) for the optional-dependency extras.

### Enabling


```python
import hgq.triton_ops

hgq.triton_ops.enable()  # qdense=True, attention=True
hgq.triton_ops.enable(qdense=True, attention=False)

hgq.triton_ops.enabled('qdense')  # True
hgq.triton_ops.enabled('attention')  # False

hgq.triton_ops.enable(precision='fp32')  # fp32 dots in training too
hgq.triton_ops.enable(force_fp32_inference=False)  # tf32 dots in inference and tracing too
```

`precision` (`'tf32'` by default, or `'fp32'`) sets the multiplies of the QDense route's Triton dots, accumulation is always fp32. Inference and tracing calls use fp32 dots when `force_fp32_inference` (the default), else `precision`.

Setting `HGQ_TRITON_QDENSE=1` or `HGQ_TRITON_ATTENTION=1` in the environment also switches the corresponding route on at import.


The switches and precision settings are read when a layer call runs or is traced. A function that is already jitted (JAX) or compiled (`torch.compile`) keeps the setting it was traced with, so call `enable` before the first `fit`, `evaluate` or `predict` kind of calls.

### Coverage

- `QDense`, and `QEinsumDense` with any two-operand equation (run as a batched GEMM, as `torch.einsum` and `jnp.einsum` do) except a label repeated within an operand or summed within one operand only, both need `iq` enabled.
- `QMultiHeadAttention` with `softmax='1pass'` or `'2pass'` and `fuse='none'`. Since this is a flash-attention 2 like implementation, the score matrices never materialize and `return_attention_scores=True` returns `None`.

Both routes cover training (`training=True`), inference (`training=False` or `None`) and tracing (`training='tracing'` in `trace_minmax`) calls.

The main conditions, checked per call:

### Fallback

With a route switched on, a call it does not cover runs on keras ops, as with the route off, and warns once per reason. Unsupported configs, like `QMultiHeadAttention` with `softmax='comb'` or `fuse='qkv'` or `'kv'`, runs on keras ops without a warning. Errors raised while building or running the kernels will crash.

### Numerics

On float32 values in the normal range, the forward pass with fp32 dots (inference and tracing with `force_fp32_inference`, or `precision='fp32'`) is identical to the keras-ops path when the quantized operands and their partial sums are exact in fp32 (narrow bit widths), otherwise equal up to floating-point rounding. Gradients agree with the keras-ops path up to floating-point rounding. With tf32 dots (training, by default), each operand of the QDense dots is rounded to tf32 (10 mantissa bits): the forward pass is exact only when the quantized operands fit in tf32, and the gradients carry tf32 rounding. Subnormal values are not covered.

## torch.compile

On the `torch` backend, `model.compile(jit_compile=...)` also accepts:

- a `torch.compile` mode: `'default'`, `'reduce-overhead'`, `'max-autotune'` or `'max-autotune-no-cudagraphs'`, with the `inductor` backend;
- a `torch.compile` backend name: `'inductor'`, `'aot_eager'`, `'eager'`, ... (any name in `torch._dynamo.list_backends(exclude_tags=())`).

HGQ2 then installs its Keras workarounds for `torch.compile` and compiles the train, test and predict steps with `torch.compile`. This requires `torch>=2.12`. Other strings raise a `ValueError`.

`jit_compile=True` keeps Keras's own `torch.compile` call, without these workarounds; use a mode or a backend name instead. `jit_compile='auto'` keeps the Keras default, which runs eagerly on `torch`.

```python
import hgq.triton_ops

hgq.triton_ops.enable()
model.compile(optimizer='adam', loss='mse', jit_compile='default')
model.fit(x_train, y_train, batch_size=256, epochs=10)
```

On `torch`, the Triton routes are registered as torch custom ops (`torch.library.triton_op`) and run inside the compiled step.

## Tested Versions

- `torch` 2.15 with `triton` 3.9
- `jax` 0.12 with `jax-triton` 0.4.1 and `triton` 3.9
- an NVIDIA RTX 4090 (CUDA); ROCm and XPU are untested

These are the versions the routes and `torch.compile` support were tested with, not minimum versions. The code requires `torch>=2.12` for `torch.compile` and `jax-triton>=0.4` for the JAX route.
