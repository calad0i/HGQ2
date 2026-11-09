"""hgq2's quantized attention (softmax '1pass' or '2pass') in JAX: the Triton kernels attn.kernels.attn_fwd and attn_bwd
launched with jax_triton.triton_call, and the custom_vjp joining them.

attention_forward takes:
  query (B, T, H, DK), key (B, S, H, DK), value (B, S, H, DV)
  valid (S, B, T) float, nonzero where query t keeps key j
  first (B, T) int32, each query's first kept key
  n (B, T / TB, *) int32, column 0: keys to visit per query block (one past the last kept by any of its queries)
  qp (H, *) per-head (k, i, f) of the score, exp-table input, exp-table output and weight quantizers (columns 0-11);
    2-pass, then of the inverse table's input and output quantizers and the probability quantizer (columns 12-20)
  akif (3, H, DV) the context quantizer's (k, i, f) per value lane
and the softmax's static settings (attn.spec.AttentionSettings), and returns
  o (B, T, H, DV), l (B, T, H): the context and weight accumulators (1-pass: normalization o * inv_table(l) is outside;
    2-pass: o is the normalized context)
  context (2 S, H, DV), held (S, H): per call maxima over the batch and queries of the context quantizer's calls in
  call order (round j: |Qa(r o)| then |Qa(p v)|; 2-pass |Qa(o)| then |Qa(P v)|) and per round of |Ql(r l)|, which the
  integer-bit replay consumes; served (S, H) 2-pass, per round maxima of |P|, the probability quantizer's output
  (1-pass: (1,), unused); rounded (2, H): the maximum of the score quantizer's |output| over the batch, queries and
  keys, and 2-pass of the inverse table's |input quantizer output|. Each int32, the float32 bit patterns, from buffers
  zeroed before the kernel atomically maxes into them (zero Refs); then negative (5, H, DV) int32, 1 where any output
  of the context quantizer (per lane), and at lane 0 of the weight, score, probability and inverse-table input
  quantizers, was negative over the calls (the tracing call's k update; (1,), zero, outside tracing). An inference
  call's settings (training False): each (1,), zero.

attention_backward runs the backward kernel on the same operands and the cotangents go, gl of o, l, and returns its
per-program partials; attention_cotangents reduces them to the operands' shapes. `attention` is the custom_vjp of the
forward in the argument order (query, key, value, qp, akif, valid, first, n), then the static settings and query block;
the maxima outputs are not differentiable (their cotangents are ignored) and valid, first, n get none.
"""

from functools import partial

import jax
import jax.numpy as jnp

from .spec import AttentionSettings, kernel_dims, maxima_shapes, maxnreg, query_block, warps


def attention_forward(query, key, value, valid, first, n, qp, akif, settings: AttentionSettings, block: int | None = None):
    """The forward kernel on the GPU; see the module docstring for operands and outputs. ``block``: queries per program
    (default `QUERY_BLOCK`, at most T; T must be a multiple)."""
    import jax_triton

    from . import kernels as K

    batch, queries, heads, depth = query.shape
    lanes = value.shape[-1]
    block = query_block(queries, block)
    if queries % block:
        raise ValueError(f'query block {block} does not divide {queries} queries')
    nb = queries // block
    if n.shape[:2] != (batch, nb):
        raise ValueError(f'n has shape {n.shape}, expected ({batch}, {nb}, ...) for query block {block}')
    f32 = jnp.float32
    query, key, value = (jnp.asarray(x, f32) for x in (query, key, value))
    valid, qp, akif = jnp.asarray(valid, f32), jnp.asarray(qp, f32), jnp.asarray(akif, f32)
    first, n = jnp.asarray(first, jnp.int32), jnp.asarray(n, jnp.int32)
    scaler = jnp.full((1,), settings.scaler, f32)
    maxima = [jax.new_ref(jnp.zeros(s, jnp.int32)) for s in maxima_shapes(query.shape, key.shape, value.shape, settings)]
    outs = (jax.ShapeDtypeStruct((batch, queries, heads, lanes), f32), jax.ShapeDtypeStruct((batch, queries, heads), f32))
    dims = kernel_dims(query.shape, key.shape, value.shape, n.shape, qp.shape, block)
    o, l = jax_triton.triton_call(
        query,
        key,
        value,
        valid,
        first,
        n,
        qp,
        akif,
        scaler,
        *maxima,
        kernel=K.attn_fwd,
        out_type=outs,
        grid=(batch, heads, nb),
        num_warps=warps(dims, False),
        num_stages=1,
        backend_options=dict(maxnreg=maxnreg(dims, False, settings.two_pass)),
        **dims,
        **settings.consts(),
    )
    return o, l, *(jax.freeze(m) for m in maxima)


def attention_backward(
    query, key, value, valid, first, n, qp, akif, go, gl, settings: AttentionSettings, block: int | None = None
):
    """The backward kernel on the GPU: go (B, T, H, DV) and gl (B, T, H) are the cotangents of o and l. Returns the
    per-program partials: the query cotangent (B, T, H, DK), each query block's shares of the key and value cotangents
    (NB, B, S, H, DK) and (NB, B, S, H, DV), of the context quantizer's f cotangent per lane (B, H, NB, DV) and of qp's
    cotangent (B, H, NB, qp columns)."""
    import jax_triton

    from . import kernels as K

    batch, queries, heads, depth = query.shape
    rounds, lanes = key.shape[1], value.shape[-1]
    block = query_block(queries, block)
    if queries % block:
        raise ValueError(f'query block {block} does not divide {queries} queries')
    nb = queries // block
    if n.shape[:2] != (batch, nb):
        raise ValueError(f'n has shape {n.shape}, expected ({batch}, {nb}, ...) for query block {block}')
    if qp.shape[1] < 12:
        raise ValueError(f'qp has {qp.shape[1]} columns, expected at least 12')
    f32 = jnp.float32
    query, key, value = (jnp.asarray(x, f32) for x in (query, key, value))
    valid, qp, akif = jnp.asarray(valid, f32), jnp.asarray(qp, f32), jnp.asarray(akif, f32)
    go, gl = jnp.asarray(go, f32), jnp.asarray(gl, f32)
    first, n = jnp.asarray(first, jnp.int32), jnp.asarray(n, jnp.int32)
    scaler = jnp.full((1,), settings.scaler, f32)
    width = qp.shape[1]
    out_shapes = (
        (batch, queries, heads, depth),
        (nb, batch, rounds, heads, depth),
        (nb, batch, rounds, heads, lanes),
        (batch, heads, nb, lanes),
        (batch, heads, nb, width),
        *[(rounds, batch, heads, queries)] * 3,  # per-program scratch: later (then ds), dm, max
    )
    outs = tuple(jax.ShapeDtypeStruct(s, f32) for s in out_shapes)
    dims = kernel_dims(query.shape, key.shape, value.shape, n.shape, qp.shape, block)
    return jax_triton.triton_call(
        query,
        key,
        value,
        valid,
        first,
        n,
        qp,
        akif,
        scaler,
        go,
        gl,
        kernel=K.attn_bwd,
        out_type=outs,
        grid=(batch, heads, nb),
        num_warps=warps(dims, True),
        num_stages=1,
        backend_options=dict(maxnreg=maxnreg(dims, True, settings.two_pass)),
        **dims,
        **settings.consts(),
    )[:5]


def attention_cotangents(akif, d_query, d_key, d_value, df, dq):
    """`attention_backward`'s partials reduced to the query, key, value, qp (H, qp columns) and akif (3, H, DV)
    cotangents (the context quantizer's k and i get none)."""
    d_akif = jnp.stack([jnp.zeros_like(akif[0]), jnp.zeros_like(akif[1]), jnp.sum(df, axis=(0, 2))])
    return d_query, jnp.sum(d_key, axis=0), jnp.sum(d_value, axis=0), jnp.sum(dq, axis=(0, 2)), d_akif


@partial(jax.custom_vjp, nondiff_argnums=(8, 9))
def _attention(query, key, value, qp, akif, valid, first, n, settings, block):
    return attention_forward(query, key, value, valid, first, n, qp, akif, settings, block)


def _attention_fwd(query, key, value, qp, akif, valid, first, n, settings, block):
    out = attention_forward(query, key, value, valid, first, n, qp, akif, settings, block)
    return out, (query, key, value, qp, akif, valid, first, n)


def _attention_bwd(settings, block, saved, cotangents):
    query, key, value, qp, akif, valid, first, n = saved
    go, gl = cotangents[:2]  # the maxima's cotangents are ignored: not differentiable
    partials = attention_backward(query, key, value, valid, first, n, qp, akif, go, gl, settings, block)
    return (*attention_cotangents(akif, *partials), None, None, None)


_attention.defvjp(_attention_fwd, _attention_bwd)


def attention(query, key, value, qp, akif, valid, first, n, settings: AttentionSettings, block: int | None = None):
    """`attention_forward` (o, l, context, held, served, rounded, negative) differentiable in query, key, value, qp and
    akif through the backward kernel (see the module docstring)."""
    return _attention(query, key, value, qp, akif, valid, first, n, settings, block)
