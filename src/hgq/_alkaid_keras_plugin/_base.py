"""Building blocks for the Alkaid-keras second-level plugin.

``QLayerMixin`` is placed to the LEFT of a pure-Keras handler class (e.g.
``alkaid.converter.builtin.keras.layers.dense.ReplayDense``). Its
responsibilities are:

* Remap ``_load_weight('kernel'|'bias')`` to HGQ's pre-quantized
  ``qkernel``/``qbias`` so the base handler's math transparently reads the
  quantized weights.
* Strip the leading ``Q`` in ``_dispatch_key()`` so Alkaid's name-dispatching
  handlers route correctly for HGQ Q-variants.
* Apply HGQ input/output quantizers (``layer.iq`` / ``layer.oq``) around the
  pure handler's ``call()``.

The ``__input_quantizer_handled__`` / ``__output_quantizer_handled__`` flags
default to False and can be overridden on a subclass when the handler itself
manages that quantization (e.g. ``_QDenseTable``, ``_QMHA``).
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from alkaid.converter.builtin.keras.layers._base import ReplayOperationBase, to_np_arr  # noqa: F401
from alkaid.trace import FVArray
from alkaid.trace.ops import quantize

import hgq
from hgq.layers.core.base import MultipleQuantizers, Quantizer
from hgq.quantizer.internal import FixedPointQuantizerBase


def mirror_quantizer(q: Quantizer, v: FVArray) -> FVArray:
    if q.scaler is not None:
        v = v * (1.0 / q.scaler)
    qi: FixedPointQuantizerBase = q.quantizer
    kk, ki, kf = qi.kif
    shape = (1,) + v.shape
    kk = qi.bw_mapper.bw_to_x(kk, shape)
    ki = qi.bw_mapper.bw_to_x(ki, shape)
    kf = qi.bw_mapper.bw_to_x(kf, shape)
    k, i, f = (to_np_arr(x).astype(np.int8)[0] for x in (kk, ki, kf))
    rq = quantize(v, k, i, f, overflow_mode=qi.overflow_mode, round_mode=qi.round_mode)
    if q.affine:
        rq = rq * q.affine[0] + q.affine[1]
    return rq


class QLayerMixin:
    __qweight_remap__ = {'kernel': 'qkernel', 'bias': 'qbias'}
    __input_quantizer_handled__ = False
    __output_quantizer_handled__ = False

    def _load_weight(self, name: str) -> np.ndarray:
        mapped = self.__qweight_remap__.get(name, name)
        w = getattr(self.op, mapped, None)
        if w is None and mapped != name:
            w = getattr(self.op, name, None)
        if w is None:
            return np.array(0.0)
        return to_np_arr(w)

    def _dispatch_key(self) -> str:
        n = type(self.op).__name__
        return n[1:] if n.startswith('Q') else n

    def __call__(self, *args: Any, **kwargs: Any):
        layer = self.op
        if not isinstance(layer, hgq.layers.QLayerBase):
            return super().__call__(*args, **kwargs)

        if not self.__input_quantizer_handled__ and getattr(layer, 'enable_iq', False):
            iq = layer.iq
            first, *rest = args
            if isinstance(first, Sequence) and not isinstance(first, FVArray):
                assert isinstance(iq, MultipleQuantizers)
                first = tuple(mirror_quantizer(q, v) for q, v in zip(iq.quantizers, first))
            else:
                assert isinstance(iq, Quantizer)
                first = mirror_quantizer(iq, first)
            args = (first, *rest)

        trace = super().__call__(*args, **kwargs)

        if not self.__output_quantizer_handled__ and getattr(layer, 'enable_oq', False):
            oq = layer.oq
            final = trace['final']
            if isinstance(oq, MultipleQuantizers):
                final = tuple(mirror_quantizer(q, v) for q, v in zip(oq.quantizers, final))
            else:
                assert len(final) == 1
                final = (mirror_quantizer(oq, final[0]),)
            trace = {**trace, 'final': final}
        return trace
