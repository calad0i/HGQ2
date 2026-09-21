from math import prod

import numpy as np
from alkaid.converter.builtin.keras.layers._base import ReplayOperationBase, to_np_arr
from alkaid.trace import FVArray
from alkaid.trace.ops import _quantize, quantize
from keras import ops
from keras.src.utils.tracking import DotNotTrackScope

from hgq.layers.table import QConvT1D, QConvT2D, QConvTBase, QDenseT, QEinsumDenseT
from hgq.quantizer import Quantizer
from hgq.quantizer.internal import FixedPointQuantizerBase

from ._base import QLayerMixin

try:
    from alkaid.opsched.frontend import SymbolicTensor, apply_in_patches, token_apply
except ImportError:
    raise RuntimeError('alkaid>=0.9.0beta1 is required for this version of hgq2. Please upgrade alkaid or install hgq2<0.3.')


def _spread(quantizer: Quantizer, grid: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The quantizer's (k, i, f), each spread over one presum ``grid`` the way the layer states it."""

    internal: FixedPointQuantizerBase = quantizer.quantizer
    return tuple(to_np_arr(internal.bw_mapper.bw_to_x(v, (1,) + grid)).astype(np.int32).reshape(grid) for v in internal.kif)  # type: ignore


def _entries(op: QEinsumDenseT, low: np.ndarray, high: np.ndarray, step: np.ndarray) -> np.ndarray:
    """Every presum position's table: the sub-network replayed over the lattice that position landed on, right padded
    to the deepest of them, with the layer's batch norm and table quantizer folded into the entries."""

    grid = low.shape
    entries = np.arange(int(np.rint((high - low) / step).max()) + 1).reshape((-1,) + (1,) * len(grid))
    # The sub-network is replayed as it stands, on one grid carrying every position's own lattice at once:
    # entry t of a position is its low plus t of its steps, and the deepest lattice is how far the grid runs.
    read = op.module((low + entries * step).reshape((-1, *grid[1:])).astype(op.dtype)[..., None])
    # The layer folds its own batch norm over the presum grid, which the entries ride as the batch axis.
    content = to_np_arr(op._apply_batch_norm(ops.expand_dims(read, 1))).reshape(len(entries), -1).T.reshape((*grid, len(entries)))

    toq: FixedPointQuantizerBase = op.toq.quantizer
    round_mode = toq.round_mode[2:] if toq.round_mode.startswith('S_') else toq.round_mode
    stated = (side.reshape((*grid, 1)) for side in _spread(op.toq, grid))
    return _quantize(content, *stated, toq.overflow_mode, round_mode)


def _lookup_and_sum(block, op: QEinsumDenseT, lanes: tuple[int, ...], columns: tuple[int, ...]):
    """Land every value of one block on the layer's own iq lattice, read the table its own presum position owns and
    sum the contracted axes away. The presum grid is ``lanes + columns``: one value a lane, spread over the columns
    it feeds, and every position of it answers on the lattice it is handed."""

    grid = (*lanes, *columns)
    values = np.broadcast_to(np.reshape(np.atleast_1d(block), (*lanes, *(1,) * len(columns))), grid)
    iq: FixedPointQuantizerBase = op.iq.quantizer
    k, i, f = _spread(op.iq, grid)
    landed = quantize(values, k, i, f, overflow_mode=iq.overflow_mode, round_mode=iq.round_mode)
    low, high, step = landed.lhs
    content = _entries(op, low, high, step)
    depths = np.rint((high - low) / step).astype(np.intp) + 1
    reads = np.empty(grid, dtype=object)
    for place in np.ndindex(*grid):
        reads[place] = landed[place].lookup(content[place][: depths[place]])
    # The latency-ordered heap is the fold a summed table read has always taken.
    return np.sum(FVArray(reads, landed.solver_options, hwconf=landed.hwconf), axis=op._contract_axes)


class _QEinsumDenseTable(QLayerMixin, ReplayOperationBase):
    handles = (QEinsumDenseT,)
    __input_quantizer_handled__ = True

    def __init__(self, op: QEinsumDenseT):
        super().__init__(op)  # type: ignore[call-arg]
        #: The einsum layer whose tables the replay reads; a dense or conv table carries its weights on an ephemeral one.
        self.table: QEinsumDenseT = op

    def call(self, inputs: FVArray) -> FVArray:
        op = self.table
        assert op.enable_iq, (
            f"{type(op).__name__} '{op.name}': a tabulated contraction needs a declared iq lattice; build it with enable_iq=True"
        )
        shape = tuple(inputs.shape)
        _, presum = op._broadcast_shapes(shape)
        traced = isinstance(inputs, SymbolicTensor)
        rank = len(shape) - len(inputs.token_shape if traced else shape[1:])  # type: ignore
        # The tables differ along the axes the kernel indexes and along the axes a quantizer states its own precision
        # at, so the firings from the first of those on are read in one call and the ones before it stream.
        stated = [to_np_arr(side).shape for quantizer in (op.iq, op.toq) for side in quantizer.quantizer.kif]
        differ = {*op._kernel_axes, *op._contract_axes}
        differ.update(len(presum) - n for axes in stated for n, size in enumerate(reversed(axes), 1) if size > 1)
        split = min((axis for axis in differ if 0 < axis < rank), default=rank)
        # A firing the contraction sums away has to ride in the token; one the tables merely differ along is a tile.
        merged = min((axis for axis in op._contract_axes if split <= axis < rank), default=rank)
        presum = (1,) * split + presum[split:]

        def cell(block):
            """One firing's block -- its tile positions among its lanes -- read through the tables it lands on."""

            return _lookup_and_sum(block, op, presum[: len(shape)], presum[len(shape) :])

        if not traced:
            return np.transpose(cell(inputs), op._output_transpose)  # type: ignore
        block = inputs if merged == rank else np.reshape(inputs, (*shape[:merged], prod(shape[merged:])))
        emits = tuple(size for axis, size in enumerate(presum) if axis >= merged and axis not in op._contract_axes)
        read = token_apply(block, cell, emits=emits, fused_tile=presum[:merged])
        return np.transpose(read, op._output_transpose)  # type: ignore


class _QDenseTable(_QEinsumDenseTable):
    handles = (QDenseT,)

    def __init__(self, op: QDenseT):
        """A dense table is the einsum table of ``...i,ij->...j``, and its trained parts ride an ephemeral one."""

        super().__init__(op)  # type: ignore[call-arg]
        table = QEinsumDenseT(
            '...i,ij->...j',
            (op.n_out,),
            n_hl=0,
            batch_norm=op.enable_bn,
            enable_iq=op.enable_iq,
            enable_oq=False,
            enable_ebops=False,
            dtype=op.dtype,
            **{f'bn_{word}': value for word, value in op.bn_args.items()},
        )
        # The sub-network was built on the layer's own input plus the column and table axes.
        table.build(op.module.input_shape[:-2])  # type: ignore[reportOptionalSubscript]
        with DotNotTrackScope():  # a built layer refuses new state, and every part grafted here is the dense's own
            table.module, table._toq = op.module, op._toq
            if op.enable_iq:
                table._iq = op._iq
            if op.enable_bn:
                bn = op.bn_module
                table.bn_gamma, table.bn_beta = bn.gamma, bn.beta
                table.moving_mean, table.moving_variance = bn.moving_mean, bn.moving_variance
        self.table = table


class _QConvTable(_QDenseTable):
    handles = (QConvT2D, QConvT1D, QConvTBase)

    def call(self, inputs: FVArray) -> FVArray:
        op: QConvTBase = self.op  # type: ignore
        spatial = op.output_shape[1:-1]
        read = super().call
        return apply_in_patches(
            inputs,
            lambda window: read(np.reshape(window, (1, *spatial, op.n_in))),  # type: ignore
            size=op.kernel_size,
            strides=op.strides,
            dilation=op.dilation_rate,
            padding=op.padding,
            data_format=op.data_format,
            fused_tile=spatial,
            emits=(op.n_out,),
        )
