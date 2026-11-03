"""hgq2's integer-bit replay in JAX: the Triton kernel replay.kernels.replay_i launched with jax_triton.triton_call, as
attn.jax_op launches the attention kernels. Not differentiable (the state is nondifferentiable data): the caller stops the
operands' gradients and assigns the result.

  replay_integer_bits(i, maxima, f, decay, bounds=None, repeats=1) -> new i
with replay.torch_op's operands, layouts and result. Launch geometry: replay.spec.
"""

import warnings

import jax
import jax.numpy as jnp

from .spec import WARPS, replay_dims


def replay_integer_bits(i, maxima, f, decay, bounds: tuple[float, float] | None = None, repeats: int = 1):
    """The integer bits a trainable-WRAP KIF quantizer's calls leave (see replay.torch_op's docstring)."""
    import jax_triton

    from . import kernels as K

    shape = tuple(i.shape)
    if tuple(f.shape) != shape:
        raise ValueError(f'f {tuple(f.shape)} does not match i {shape}')
    f32 = jnp.float32
    m = [jnp.asarray(x, f32).reshape(*x.shape[:4], -1) for x in maxima]
    grid, n, dims = replay_dims(shape, m[0].shape, len(m), repeats)
    i, f, decay = (jnp.asarray(x, f32).reshape(-1) for x in (i, f, decay))
    if decay.shape != (1,):
        raise ValueError(f'decay {decay.shape} is not one element')
    lo, hi = (0.0, 0.0) if bounds is None else (float(bounds[0]), float(bounds[1]))
    outs = (jax.ShapeDtypeStruct((n,), f32), jax.ShapeDtypeStruct((grid[0], n), f32), jax.ShapeDtypeStruct((1,), jnp.int32))
    with warnings.catch_warnings():  # zeroed_outputs (the ticket counter) is deprecated in favor of Ref operands
        warnings.simplefilter('ignore', DeprecationWarning)
        out = jax_triton.triton_call(
            m[0], m[-1], i, f, decay,
            lo=lo, hi=hi, kernel=K.replay_i, out_type=outs, grid=grid, zeroed_outputs=(2,), num_warps=WARPS, num_stages=1,
            CONSTRAINT='none' if bounds is None else 'minmax', **dims,
        )[0]  # fmt: skip
    return out.reshape(shape)
