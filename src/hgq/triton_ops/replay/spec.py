"""Launch geometry of the integer-bit replay kernel (replay.kernels.replay_i), shared by the JAX (replay.jax_op) and torch
(replay.torch_op) bindings. Pure Python: no torch / JAX / triton import."""

WARPS = 4
TILE = 4096  # partial maxima per program load


def replay_dims(i_shape, m_shape, sources: int, repeats: int) -> tuple[tuple[int, int], int, dict]:
    """(grid, n, constexprs) for bit widths of shape ``i_shape`` (1, H or 1, 1, L or 1), ``sources`` (1 or 2) maxima
    arrays of shape ``m_shape`` (R, B, H, NB, L), each call ``repeats`` times. n: the bit-width element count."""
    rounds, batch, heads, nb, lanes = m_shape
    if len(i_shape) != 4 or i_shape[0] != 1 or i_shape[2] != 1 or i_shape[1] not in (1, heads) or i_shape[3] not in (1, lanes):
        raise ValueError(f'bit widths {tuple(i_shape)} are not (1, {heads} or 1, 1, {lanes} or 1)')
    per_head, per_lane = i_shape[1] == heads, i_shape[3] == lanes
    n = i_shape[1] * i_shape[3]
    dims = dict(
        B=batch, H=heads, NB=nb, L=lanes, PER_HEAD=per_head, PER_LANE=per_lane, CALLS_PER_ROUND=sources, REPEATS=repeats,
        BB=max(1, TILE // (nb * lanes)), NP=max(16, 1 << (n - 1).bit_length()),
    )  # fmt: skip
    return (rounds * sources, i_shape[1]), n, dims
