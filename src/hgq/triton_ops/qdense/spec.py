"""Static description of a fused quantized dense (the GEMM's N and K, quantizer forms, bit-width layouts, tiles), shared
by the torch and JAX bindings; the row count M is a launch argument (it sets the grids and the partial buffers only), so
one Spec serves every batch size. Pure Python: no torch / JAX / triton import.

Tiles and launch parameters come from a measured table (env HGQ_QDENSE_TUNE: a JSON file keyed by kernel, problem and
GPU), else from DEFAULT below, whose tiles are a placeholder for untuned shapes (the Triton matmul tutorial's FP32 tile).
Its num_stages are measured: one stage beat three in every kernel of the layer (RTX 4090, M=163840, K=128,
N=128 / 256). The stages of a Spec are a request: before launching, the bindings lower them per kernel until that
specialization fits the device's shared memory (qdense.kernels.fit_stages).
"""

import dataclasses
import json
import os
from functools import lru_cache

PRECISIONS = ('ieee', 'tf32x3', 'tf32')
# An operand's quantizer form: ('quantize' | 'round' | 'none', round_mode, overflow_mode, null_xgrad_on_zero_width)
NONE = ('none', 'TRN', 'WRAP', False)

# Untuned tiles, measured stages (see module docstring); 'S' (at most this many dx programs along M) and 'SPLIT' (at most
# this many dw partials) None = derived.
DEFAULT = {
    'fwd': dict(BM=64, BN=64, BK=32, GROUP_M=8, num_warps=4, num_stages=1),
    'dx': dict(BM=64, BN=32, BK=64, S=None, num_warps=4, num_stages=1),
    'dw': dict(BM=32, BN=64, BK=64, SPLIT=None, num_warps=4, num_stages=1),
    'vjp2d': dict(BR=32, BC=128, num_warps=4, num_stages=1),
}


# dx row tiles (BM) tried, largest first, when the tuned one does not divide the row period of the x bit widths
DX_ROW_TILES = (128, 64, 32, 16)


def cdiv(a, b):
    return -(-a // b)


@dataclasses.dataclass(frozen=True)
class Spec:
    N: int
    K: int
    qx: tuple  # x quantizer form ('none': x is used as given)
    rx: int  # x bit widths: periodic (rx, cx) layout
    cx: int
    qw: tuple  # weight quantizer form when w is quantized on load; 'none': w is pre-quantized (keras ops)
    rw: int
    cw: int
    bias: bool
    amax: bool  # trainable-WRAP abs-max of q(x) over the bit-width layout
    prec: str = 'ieee'
    store_q: bool = False  # checks: also return q(x) as computed inside the forward GEMM
    negative: bool = False  # tracing (with amax): also whether any q(x) < 0 over the bit-width layout
    fwd: tuple = ()  # tiles, as sorted (key, value) pairs (hashable)
    dx: tuple = ()
    dw: tuple = ()
    vjp2d: tuple = ()

    def cfg(self, kernel):
        return dict(getattr(self, kernel))

    def with_cfg(self, kernel, **params):
        """This Spec with ``params`` of ``kernel``'s launch parameters replaced."""
        return dataclasses.replace(self, **{kernel: tuple(sorted({**self.cfg(kernel), **params}.items()))})

    def dumps(self):
        return json.dumps(dataclasses.asdict(self))

    @staticmethod
    @lru_cache(maxsize=None)
    def loads(s):
        d = json.loads(s)
        for k in ('fwd', 'dx', 'dw', 'vjp2d'):
            d[k] = tuple(tuple(kv) for kv in d[k])
        d['qx'], d['qw'] = tuple(d['qx']), tuple(d['qw'])
        return Spec(**d)

    # derived layout facts
    @property
    def xr(self):
        return self.rx > 1

    @property
    def xc(self):
        return self.cx > 1

    @property
    def amax_mode(self):  # 0 none, 1 per-row-block partials, 2 atomic max (bit widths vary along rows)
        return 0 if not self.amax else (2 if self.xr else 1)

    # launch geometry on M rows
    def fwd_grid(self, M):
        c = self.cfg('fwd')
        return (cdiv(M, c['BM']) * cdiv(self.N, c['BN']),)

    @property
    def dx_fits(self):  # the dx row tile divides the row period (make_spec fits one unless backward is off)
        return not self.xr or self.rx % self.cfg('dx')['BM'] == 0

    def amax_buffer(self, M):  # (shape, dtype): one plane, with negative two (abs-max, any-negative flags)
        planes = 1 + self.negative
        if self.amax_mode == 1:
            return (planes, cdiv(M, self.cfg('fwd')['BM']), self.K), 'float32'
        if self.amax_mode == 2:
            return (planes, self.rx, self.cx), 'int32'
        return (1,), 'float32'

    @property
    def amax_shape(self):  # the bindings' amax output: (rx, cx); with negative (2, rx, cx), then the 0 / 1 flags
        return (2, self.rx, self.cx) if self.negative else (self.rx, self.cx)

    def dx_programs(self, M):  # S: one per row block, at most the Spec's S
        c = self.cfg('dx')
        return cdiv(M, c['BM']) if c['S'] is None else min(cdiv(M, c['BM']), c['S'])

    def dx_grid(self, M):
        return (self.dx_programs(M), cdiv(self.K, self.cfg('dx')['BK']))

    def dx_partials(self, M):
        if self.qx[0] == 'none':
            return (1,)
        if self.xr:
            return (self.dx_programs(M) // (self.rx // self.cfg('dx')['BM']), self.rx, self.K)
        return (self.dx_programs(M), self.K)

    def dw_split(self, M):  # SPLIT: the Spec's, at most the row blocks
        return min(cdiv(M, self.cfg('dw')['BM']), self.cfg('dw')['SPLIT'])

    def dw_grid(self, M):
        c = self.cfg('dw')
        return (cdiv(self.K, c['BK']), cdiv(self.N, c['BN']), self.dw_split(M))

    def vjp2d_grid(self):
        c = self.cfg('vjp2d')
        return (cdiv(self.K, c['BR']), cdiv(self.N, c['BC']))

    # scalar / constexpr kernel arguments (after the pointer operands), the same for every binding
    def fwd_consts(self, M):
        c = self.cfg('fwd')
        return dict(
            M=M,
            N=self.N,
            K=self.K,
            RX=self.rx,
            CX=self.cx,
            RW=self.rw,
            CW=self.cw,
            QX=self.qx[0],
            RNDX=self.qx[1],
            OVFX=self.qx[2],
            NULLX=self.qx[3],
            XR=self.xr,
            XC=self.xc,
            QW=self.qw[0],
            RNDW=self.qw[1],
            OVFW=self.qw[2],
            NULLW=self.qw[3],
            WR=self.rw > 1,
            WC=self.cw > 1,
            HAS_BIAS=self.bias,
            AMAX=self.amax_mode,
            NEG=self.negative,
            STORE_Q=self.store_q,
            PREC=self.prec,
            BM=c['BM'],
            BN=c['BN'],
            BK=c['BK'],
            GROUP_M=c['GROUP_M'],
        )

    def dx_consts(self, M):
        c = self.cfg('dx')
        return dict(
            M=M,
            N=self.N,
            K=self.K,
            RX=self.rx,
            CX=self.cx,
            RW=self.rw,
            CW=self.cw,
            S=self.dx_programs(M),
            QX=self.qx[0],
            RNDX=self.qx[1],
            OVFX=self.qx[2],
            NULLX=self.qx[3],
            XR=self.xr,
            XC=self.xc,
            QW=self.qw[0],
            RNDW=self.qw[1],
            OVFW=self.qw[2],
            NULLW=self.qw[3],
            WR=self.rw > 1,
            WC=self.cw > 1,
            PREC=self.prec,
            BM=c['BM'],
            BN=c['BN'],
            BK=c['BK'],
        )

    def dw_consts(self, M):
        c = self.cfg('dw')
        return dict(
            M=M,
            N=self.N,
            K=self.K,
            RX=self.rx,
            CX=self.cx,
            SPLIT=self.dw_split(M),
            QX=self.qx[0],
            RNDX=self.qx[1],
            OVFX=self.qx[2],
            NULLX=self.qx[3],
            XR=self.xr,
            XC=self.xc,
            HAS_BIAS=self.bias,
            PREC=self.prec,
            BM=c['BM'],
            BN=c['BN'],
            BK=c['BK'],
        )

    def vjp2d_consts(self):
        c = self.cfg('vjp2d')
        q, rnd, ovf, null = self.qw
        return dict(
            R=self.K,
            C=self.N,
            RB=self.rw,
            CB=self.cw,
            Q=q,
            RND=rnd,
            OVF=ovf,
            NULL=null,
            ROWS=self.rw > 1,
            COLS=self.cw > 1,
            BR=c['BR'],
            BC=c['BC'],
        )


@lru_cache(maxsize=None)
def _table():
    path = os.environ.get('HGQ_QDENSE_TUNE', '')
    if not path or not os.path.exists(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def tune_key(kernel, N, K, prec, xr):
    """Tiles depend on the (K, N) problem, the precision and the row layout; M only sets the grid. The table file is
    per GPU type (written by the tune pass on that GPU)."""
    return f'{kernel}|N{N}|K{K}|{prec}|xr{int(xr)}'


def dx_tile_fits(rx) -> bool:
    """Whether some dx row tile divides the row period ``rx`` of the x bit widths (always, when they do not vary along
    rows): dx keeps per-period partials in registers."""
    return rx == 1 or any(rx % b == 0 for b in DX_ROW_TILES)


def make_spec(
    N,
    K,
    qx,
    rx,
    cx,
    qw=NONE,
    rw=1,
    cw=1,
    bias=True,
    amax=False,
    prec='ieee',
    store_q=False,
    n_sm=108,
    overrides=None,
    negative=False,
    backward=True,
):
    """A Spec with tiles from the tune table (else DEFAULT) and the derived bounds S / SPLIT. ``overrides``:
    {kernel: {param: value}} (tuning sweeps). ValueError when no dx row tile up to the tuned BM divides the row period
    (`dx_tile_fits` false) and ``backward`` (False: an inference call's Spec, whose backward then raises), or when
    ``overrides`` give a dx S that is not a multiple of rx / BM."""
    if prec not in PRECISIONS:
        raise ValueError(prec)
    xr = rx > 1
    cfgs = {}
    for kernel in ('fwd', 'dx', 'dw', 'vjp2d'):
        c = dict(DEFAULT[kernel])
        c.update(_table().get(tune_key('fwdplain' if kernel == 'fwd' and qx[0] == 'none' else kernel, N, K, prec, xr), {}))
        c.update((overrides or {}).get(kernel, {}))
        cfgs[kernel] = c
    dx, dw = cfgs['dx'], cfgs['dw']
    fits = [b for b in (dx['BM'], *DX_ROW_TILES) if b <= dx['BM'] and rx % b == 0]
    if xr and not fits and backward:
        raise ValueError(f'no dx row tile up to BM={dx["BM"]} divides the row period {rx}')
    if xr and fits:
        # dx keeps per-period partials in registers: BM must divide the row period, S a multiple of rx / BM (the row
        # blocks are too: rx divides M)
        dx['BM'] = fits[0]
        per = rx // dx['BM']
        if dx['S'] is None:  # enough programs to fill the GPU (2 per SM), as few partial slabs as that allows
            dx['S'] = cdiv(cdiv(2 * n_sm, cdiv(K, dx['BK'])), per) * per
        if dx['S'] % per:
            raise ValueError(f'dx S={dx["S"]} is not a multiple of {per} (rx / BM)')
    if dw['SPLIT'] is None:  # 2 programs per SM over the (K, N) tiles
        dw['SPLIT'] = cdiv(2 * n_sm, cdiv(K, dw['BK']) * cdiv(N, dw['BN']))
    frozen = {k: tuple(sorted(v.items())) for k, v in cfgs.items()}
    args = (N, K, tuple(qx), rx, cx, tuple(qw), rw, cw, bool(bias), bool(amax), prec, bool(store_q), bool(negative))
    return Spec(*args, **frozen)
