"""hgq2's quantized attention (softmax '1pass' or '2pass') in torch: the Triton kernels attn.kernels.attn_fwd and
attn_bwd, the same source attn.jax_op launches, as torch.library.triton_op custom ops with register_autograd, so dynamo /
inductor trace them (fullgraph).

The functions mirror attn.jax_op's, with the same operands, layouts, outputs and reductions (see its docstring):
  attention_forward(query, key, value, valid, first, n, qp, akif, settings, block=None) -> (o, l, context, held,
    served, rounded, negative)
  attention_backward(query, key, value, valid, first, n, qp, akif, go, gl, settings, block=None) -> per-program partials
  attention_cotangents(akif, d_query, d_key, d_value, df, dq) -> query, key, value, qp, akif cotangents
  attention(query, key, value, qp, akif, valid, first, n, settings, block=None): the forward, differentiable in query,
    key, value, qp and akif through the backward kernel; context, held, served, rounded, negative are not
    differentiable, valid, first, n get no gradient.
The ops (hgqtri::attn_fwd, hgqtri::attn_bwd) take the settings unpacked (AttentionSettings.args, the query block) and
allocate their outputs (the forward's maxima buffers zeroed) and the backward's scratch; their bodies are their fake
(meta) implementations too. Float operands are cast to float32 and first, n to int32 outside the ops (as attn.jax_op
does), so the casts are differentiable.
"""

import torch
from torch.library import triton_op, wrap_triton

from . import kernels as K
from .spec import AttentionSettings, kernel_dims, maxima_shapes, maxnreg, query_block, warps

F32 = torch.float32
Tensor = torch.Tensor


def _geometry(query, key, value, n, block):
    batch, queries, heads, depth = query.shape
    rounds, lanes = key.shape[1], value.shape[-1]
    if queries % block:
        raise ValueError(f'query block {block} does not divide {queries} queries')
    nb = queries // block
    if tuple(n.shape[:2]) != (batch, nb):
        raise ValueError(f'n has shape {tuple(n.shape)}, expected ({batch}, {nb}, ...) for query block {block}')
    return batch, queries, heads, depth, rounds, lanes, nb


def _launch(kernel, grid, args, dims, settings):
    """``args``: the contiguous operands, then the outputs; ``dims``: attn.spec.kernel_dims, compile-time constants."""
    modes = {k: v.lower() if isinstance(v, str) else v for k, v in settings.consts().items()}  # qdense.torch_op._modes_lower
    bwd = kernel is K.attn_bwd
    cap = maxnreg(dims, bwd, settings.two_pass)
    wrap_triton(kernel)[grid](*args, num_warps=warps(dims, bwd), num_stages=1, maxnreg=cap, **dims, **modes)


@triton_op('hgqtri::attn_fwd', mutates_args=())
def attn_fwd(
    query: Tensor, key: Tensor, value: Tensor, valid: Tensor, first: Tensor, n: Tensor, qp: Tensor, akif: Tensor,
    rnd_s: str, ovf_s: str, null_s: bool, rnd_ei: str, ovf_ei: str, null_ei: bool, rnd_eo: str, ovf_eo: str,
    null_eo: bool, rnd_a: str, rnd_l: str, scaler: float, q_s: str, two_pass: bool, rnd_ii: str, ovf_ii: str,
    null_ii: bool, q_ii: str, rnd_io: str, ovf_io: str, null_io: bool, q_p: str, rnd_p: str, ovf_p: str, null_p: bool,
    epsilon: float, q_a: str, ovf_a: str, null_a: bool, q_l: str, ovf_l: str, null_l: bool, training: bool,
    tracing: bool, block: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:  # fmt: skip
    """The forward kernel: (o, l, context, held, served, rounded, negative) of float32 contiguous operands, the maxima
    and flags int32 (inference: (1,), zero; the flags also outside tracing)."""
    settings = AttentionSettings(
        rnd_s, ovf_s, null_s, rnd_ei, ovf_ei, null_ei, rnd_eo, ovf_eo, null_eo, rnd_a, rnd_l, scaler, q_s,
        two_pass, rnd_ii, ovf_ii, null_ii, q_ii, rnd_io, ovf_io, null_io, q_p, rnd_p, ovf_p, null_p, epsilon, q_a, ovf_a,
        null_a, q_l, ovf_l, null_l, training, tracing,
    )  # fmt: skip
    query, key, value, valid, first, n, qp, akif = (x.contiguous() for x in (query, key, value, valid, first, n, qp, akif))
    batch, queries, heads, depth, rounds, lanes, nb = _geometry(query, key, value, n, block)
    f32 = dict(dtype=F32, device=query.device)
    o = torch.empty((batch, queries, heads, lanes), **f32)
    l = torch.empty((batch, queries, heads), **f32)
    shapes = maxima_shapes(query.shape, key.shape, value.shape, settings)
    maxima = [torch.zeros(s, dtype=torch.int32, device=query.device) for s in shapes]
    sc = torch.full((1,), settings.scaler, **f32)
    args = (query, key, value, valid, first, n, qp, akif, sc, *maxima, o, l)
    dims = kernel_dims(query.shape, key.shape, value.shape, n.shape, qp.shape, block)
    _launch(K.attn_fwd, (batch, heads, nb), args, dims, settings)
    return o, l, *maxima


@triton_op('hgqtri::attn_bwd', mutates_args=())
def attn_bwd(
    query: Tensor, key: Tensor, value: Tensor, valid: Tensor, first: Tensor, n: Tensor, qp: Tensor, akif: Tensor,
    go: Tensor, gl: Tensor,
    rnd_s: str, ovf_s: str, null_s: bool, rnd_ei: str, ovf_ei: str, null_ei: bool, rnd_eo: str, ovf_eo: str,
    null_eo: bool, rnd_a: str, rnd_l: str, scaler: float, q_s: str, two_pass: bool, rnd_ii: str, ovf_ii: str,
    null_ii: bool, q_ii: str, rnd_io: str, ovf_io: str, null_io: bool, q_p: str, rnd_p: str, ovf_p: str, null_p: bool,
    epsilon: float, q_a: str, ovf_a: str, null_a: bool, q_l: str, ovf_l: str, null_l: bool, training: bool,
    tracing: bool, block: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:  # fmt: skip
    """The backward kernel: the per-program partials (d_query, d_key and d_value shares, df shares, dq shares)."""
    settings = AttentionSettings(
        rnd_s, ovf_s, null_s, rnd_ei, ovf_ei, null_ei, rnd_eo, ovf_eo, null_eo, rnd_a, rnd_l, scaler, q_s,
        two_pass, rnd_ii, ovf_ii, null_ii, q_ii, rnd_io, ovf_io, null_io, q_p, rnd_p, ovf_p, null_p, epsilon, q_a, ovf_a,
        null_a, q_l, ovf_l, null_l, training, tracing,
    )  # fmt: skip
    ins = (query, key, value, valid, first, n, qp, akif, go, gl)
    query, key, value, valid, first, n, qp, akif, go, gl = (x.contiguous() for x in ins)
    batch, queries, heads, depth, rounds, lanes, nb = _geometry(query, key, value, n, block)
    width = qp.shape[1]
    if width < 12:
        raise ValueError(f'qp has {width} columns, expected at least 12')
    f32 = dict(dtype=F32, device=query.device)
    d_query = torch.empty((batch, queries, heads, depth), **f32)  # stored with query's (contiguous) strides
    d_key = torch.empty((nb, batch, rounds, heads, depth), **f32)
    d_value = torch.empty((nb, batch, rounds, heads, lanes), **f32)
    df = torch.empty((batch, heads, nb, lanes), **f32)
    dq = torch.empty((batch, heads, nb, width), **f32)
    scratch = [torch.empty((rounds, batch, heads, queries), **f32) for _ in range(3)]  # later (then ds), dm, max
    sc = torch.full((1,), settings.scaler, **f32)
    args = (query, key, value, valid, first, n, qp, akif, sc, go, gl, d_query, d_key, d_value, df, dq, *scratch)
    dims = kernel_dims(query.shape, key.shape, value.shape, n.shape, qp.shape, block)
    _launch(K.attn_bwd, (batch, heads, nb), args, dims, settings)
    return d_query, d_key, d_value, df, dq


def _statics(settings: AttentionSettings, block: int):
    return *settings.args(), block


def _cast(query, key, value, valid, first, n, qp, akif):
    query, key, value, valid, qp, akif = (x.to(F32) for x in (query, key, value, valid, qp, akif))
    return query, key, value, valid, first.to(torch.int32), n.to(torch.int32), qp, akif


def attention_forward(query, key, value, valid, first, n, qp, akif, settings: AttentionSettings, block: int | None = None):
    """The forward kernel on the GPU (attn.jax_op.attention_forward's operands and outputs). Not differentiable."""
    ops = _cast(query, key, value, valid, first, n, qp, akif)
    return attn_fwd(*ops, *_statics(settings, query_block(query.shape[1], block)))


def attention_backward(
    query, key, value, valid, first, n, qp, akif, go, gl, settings: AttentionSettings, block: int | None = None
):
    """The backward kernel on the GPU (attn.jax_op.attention_backward's operands and per-program partials)."""
    ops = _cast(query, key, value, valid, first, n, qp, akif)
    return attn_bwd(*ops, go.to(F32), gl.to(F32), *_statics(settings, query_block(query.shape[1], block)))


def attention_cotangents(akif, d_query, d_key, d_value, df, dq):
    """`attention_backward`'s partials reduced as attn.jax_op.attention_cotangents reduces them."""
    zero = torch.zeros_like(akif[0])
    d_akif = torch.stack([zero, zero, df.sum((0, 2))])
    return d_query, d_key.sum(0), d_value.sum(0), dq.sum((0, 2)), d_akif


def _setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs[:8])
    ctx.statics = inputs[8:]
    ctx.mark_non_differentiable(*output[2:])


def _backward(ctx, go, gl, *_):
    """The maxima's cotangents are ignored (not differentiable); valid, first, n and the settings get none."""
    if not ctx.statics[-3]:
        raise RuntimeError('the inference attention kernel (settings.training False) has no backward')
    query, key, value, valid, first, n, qp, akif = ctx.saved_tensors
    o_shape = (*query.shape[:3], value.shape[-1])
    go = torch.zeros(o_shape, dtype=F32, device=query.device) if go is None else go
    gl = torch.zeros(o_shape[:3], dtype=F32, device=query.device) if gl is None else gl
    partials = attn_bwd(query, key, value, valid, first, n, qp, akif, go, gl, *ctx.statics)
    d_query, d_key, d_value, d_qp, d_akif = attention_cotangents(akif, *partials)
    return d_query, d_key, d_value, None, None, None, d_qp, d_akif, *[None] * len(ctx.statics)


attn_fwd.register_autograd(_backward, setup_context=_setup)


def attention(query, key, value, qp, akif, valid, first, n, settings: AttentionSettings, block: int | None = None):
    """`attention_forward` (o, l, context, held, served, rounded, negative) differentiable in query, key, value, qp and
    akif through the backward kernel, in attn.jax_op.attention's argument order."""
    return attention_forward(query, key, value, valid, first, n, qp, akif, settings, block)
