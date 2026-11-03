"""hgq2's integer-bit replay in torch: the Triton kernel replay.kernels.replay_i as a torch.library.triton_op
(hgqtri::replay_i, no autograd: the state is nondifferentiable data), so dynamo / inductor trace it. Uses only torch
tensors (no Keras): the caller assigns the result.

  replay_integer_bits(i, maxima, f, decay, bounds=None, repeats=1) -> new i
``maxima``: one or two arrays (R, B, H, NB[, L]), the attention kernel's per-round partial abs-maxima of the quantizer's
round-r call (two: the round's first and second call); each call applies ``repeats`` times in a row. ``i`` the
quantizer's raw _i, ``f`` its rounded f, of the bit-width shape (1, H or 1, 1, L or 1); ``decay`` its i_decay_speed (0-d
or one element); ``bounds`` None for no i constraint or the (min, max) of a hgq.constraints.MinMax. Returns the integer
bits after the calls, float32 in i's shape. Launch geometry: replay.spec.
"""

import torch
from torch.library import triton_op, wrap_triton

from . import kernels as K
from .spec import WARPS, replay_dims

F32 = torch.float32
Tensor = torch.Tensor


@triton_op('hgqtri::replay_i', mutates_args=())
def replay_i(
    m0: Tensor, m1: Tensor, i: Tensor, f: Tensor, decay: Tensor, i_shape: list[int], lo: float, hi: float, constraint: str,
    sources: int, repeats: int,
) -> Tensor:  # fmt: skip
    """The replay kernel on float32 contiguous operands: m0, m1 (R, B, H, NB, L), i, f (n,), decay (1,)."""
    grid, n, dims = replay_dims(i_shape, tuple(m0.shape), sources, repeats)
    if tuple(i.shape) != (n,) or tuple(f.shape) != (n,) or decay.numel() != 1:
        raise ValueError(f'i {tuple(i.shape)}, f {tuple(f.shape)}, decay {tuple(decay.shape)} for {n} bit widths')
    out = torch.empty_like(i)
    scratch = torch.empty((grid[0], n), dtype=F32, device=i.device)
    count = torch.zeros((1,), dtype=torch.int32, device=i.device)
    wrap_triton(K.replay_i)[grid](
        m0, m1, i, f, decay, lo, hi, out, scratch, count, CONSTRAINT=constraint, num_warps=WARPS, num_stages=1, **dims
    )
    return out


def replay_integer_bits(i, maxima, f, decay, bounds: tuple[float, float] | None = None, repeats: int = 1):
    """The integer bits a trainable-WRAP KIF quantizer's calls leave (see the module docstring)."""
    shape = tuple(i.shape)
    if tuple(f.shape) != shape:
        raise ValueError(f'f {tuple(f.shape)} does not match i {shape}')
    m = [x.to(F32).reshape(*x.shape[:4], -1).contiguous() for x in maxima]
    i, f, decay = (x.to(F32).reshape(-1).contiguous() for x in (i, f, decay))
    lo, hi = (0.0, 0.0) if bounds is None else (float(bounds[0]), float(bounds[1]))
    constraint = 'none' if bounds is None else 'minmax'
    return replay_i(m[0], m[-1], i, f, decay, list(shape), lo, hi, constraint, len(m), repeats).reshape(shape)
