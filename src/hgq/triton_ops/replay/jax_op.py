"""hgq2's integer-bit replay in JAX: the Triton kernel replay.kernels.replay_i launched with jax_triton.triton_call, as
attn.jax_op launches the attention kernels. Not differentiable (the state is nondifferentiable data): the caller stops the
operands' gradients and assigns the result.

  replay_integer_bits(i, maxima, f, decay, bounds=None, repeats=1, tracing=False) -> new i
with replay.torch_op's operands, layouts and result. Launch geometry: replay.spec.
"""

import jax
import jax.numpy as jnp

from .spec import WARPS, replay_dims


def replay_integer_bits(i, maxima, f, decay, bounds: tuple[float, float] | None = None, repeats: int = 1, tracing: bool = False):
    """The integer bits a trainable-WRAP KIF quantizer's calls leave (see replay.torch_op's docstring)."""
    import jax_triton

    from . import kernels as K

    shape = tuple(i.shape)
    if tuple(f.shape) != shape:
        raise ValueError(f'f {tuple(f.shape)} does not match i {shape}')
    m = maxima.reshape(*maxima.shape[:2], -1)
    n, dims = replay_dims(shape, m.shape, repeats, tracing)
    i, f, decay = (jnp.asarray(x, jnp.float32).reshape(-1) for x in (i, f, decay))
    if decay.shape != (1,):
        raise ValueError(f'decay {decay.shape} is not one element')
    lo, hi = (0.0, 0.0) if bounds is None else (float(bounds[0]), float(bounds[1]))
    out = jax_triton.triton_call(
        m, i, f, decay,
        lo=lo, hi=hi, kernel=K.replay_i, out_type=jax.ShapeDtypeStruct((n,), jnp.float32), grid=(1,), num_warps=WARPS,
        num_stages=1, CONSTRAINT='none' if bounds is None else 'minmax', **dims,
    )  # fmt: skip
    return out.reshape(shape)
