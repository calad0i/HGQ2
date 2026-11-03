"""hgq2's integer-bit replay in torch: the Triton kernel replay.kernels.replay_i as a torch.library.triton_op
(hgqtri::replay_i, no autograd: the state is nondifferentiable data), so dynamo / inductor trace it. Uses only torch
tensors (no Keras): the caller assigns the result.

  replay_integer_bits(i, maxima, f, decay, bounds=None, repeats=1, tracing=False) -> new i
``maxima``: (calls, H[, L]) int32, the attention kernel's per-call abs-maxima of the quantizer's calls in call order
(float32 bit patterns); each call applies ``repeats`` times in a row. ``i`` the quantizer's raw _i, ``f`` its rounded f,
of the bit-width shape (1, H or 1, 1, L or 1); ``decay`` its i_decay_speed (0-d or one element); ``bounds`` None for no
i constraint or the (min, max) of a hgq.constraints.MinMax; ``tracing`` for the tracing transition (each call reads the
rounded i; pass decay 0). Returns the integer bits after the calls, float32 in i's
shape. Launch geometry: replay.spec.
"""

import torch
from torch.library import triton_op, wrap_triton

from . import kernels as K
from .spec import WARPS, replay_dims

F32 = torch.float32
Tensor = torch.Tensor


@triton_op('hgqtri::replay_i', mutates_args=())
def replay_i(
    m: Tensor, i: Tensor, f: Tensor, decay: Tensor, i_shape: list[int], lo: float, hi: float, constraint: str, repeats: int,
    tracing: bool,
) -> Tensor:  # fmt: skip
    """The replay kernel on contiguous operands: m (calls, H, L) int32, i, f (n,) float32, decay (1,)."""
    n, dims = replay_dims(i_shape, tuple(m.shape), repeats, tracing)
    if tuple(i.shape) != (n,) or tuple(f.shape) != (n,) or decay.numel() != 1:
        raise ValueError(f'i {tuple(i.shape)}, f {tuple(f.shape)}, decay {tuple(decay.shape)} for {n} bit widths')
    out = torch.empty_like(i)
    wrap_triton(K.replay_i)[(1,)](m, i, f, decay, lo, hi, out, CONSTRAINT=constraint, num_warps=WARPS, num_stages=1, **dims)
    return out


def replay_integer_bits(i, maxima, f, decay, bounds: tuple[float, float] | None = None, repeats: int = 1, tracing: bool = False):
    """The integer bits a trainable-WRAP KIF quantizer's calls leave (see the module docstring)."""
    shape = tuple(i.shape)
    if tuple(f.shape) != shape:
        raise ValueError(f'f {tuple(f.shape)} does not match i {shape}')
    m = maxima.reshape(*maxima.shape[:2], -1).contiguous()
    i, f, decay = (x.to(F32).reshape(-1).contiguous() for x in (i, f, decay))
    lo, hi = (0.0, 0.0) if bounds is None else (float(bounds[0]), float(bounds[1]))
    constraint = 'none' if bounds is None else 'minmax'
    return replay_i(m, i, f, decay, list(shape), lo, hi, constraint, repeats, tracing).reshape(shape)
