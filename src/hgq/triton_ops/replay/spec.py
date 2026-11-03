"""Launch geometry of the integer-bit replay kernel (replay.kernels.replay_i), shared by the JAX (replay.jax_op) and torch
(replay.torch_op) bindings. Pure Python: no torch / JAX / triton import."""

WARPS = 4


def replay_dims(i_shape, m_shape, repeats: int, tracing: bool) -> tuple[int, dict]:
    """(n, constexprs) for bit widths of shape ``i_shape`` (1, H or 1, 1, L or 1) and maxima of shape ``m_shape``
    (calls, H, L), each call ``repeats`` times, in a training or ``tracing`` call. n: the bit-width element count."""
    calls, heads, lanes = m_shape
    if len(i_shape) != 4 or i_shape[0] != 1 or i_shape[2] != 1 or i_shape[1] not in (1, heads) or i_shape[3] not in (1, lanes):
        raise ValueError(f'bit widths {tuple(i_shape)} are not (1, {heads} or 1, 1, {lanes} or 1)')
    dims = dict(
        CALLS=calls, H=heads, L=lanes, HP=1 << (heads - 1).bit_length(), PER_HEAD=i_shape[1] == heads,
        PER_LANE=i_shape[3] == lanes, REPEATS=repeats, TRACING=tracing,
    )  # fmt: skip
    return i_shape[1] * i_shape[3], dims
