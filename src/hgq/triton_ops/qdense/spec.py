"""Static description of one fused quantized-dense call (shapes, quantizer forms, bit-width layouts, tiles), shared by
the torch and JAX bindings. Pure Python: no torch / JAX / triton import.

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

# Untuned tiles, measured stages (see module docstring); 'S' (dx programs along M) and 'SPLIT' (dw partials) None =
# derived.
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
    M: int
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

    def fwd_grid(self):
        c = self.cfg('fwd')
        return (cdiv(self.M, c['BM']) * cdiv(self.N, c['BN']),)

    def amax_buffer(self):  # (shape, dtype)
        if self.amax_mode == 1:
            return (cdiv(self.M, self.cfg('fwd')['BM']), self.K), 'float32'
        if self.amax_mode == 2:
            return (self.rx, self.cx), 'int32'
        return (1,), 'float32'

    def dx_grid(self):
        c = self.cfg('dx')
        return (c['S'], cdiv(self.K, c['BK']))

    def dx_partials(self):
        c = self.cfg('dx')
        if self.qx[0] == 'none':
            return (1,)
        if self.xr:
            return (c['S'] // (self.rx // c['BM']), self.rx, self.K)
        return (c['S'], self.K)

    def dw_grid(self):
        c = self.cfg('dw')
        return (cdiv(self.K, c['BK']), cdiv(self.N, c['BN']), c['SPLIT'])

    def vjp2d_grid(self):
        c = self.cfg('vjp2d')
        return (cdiv(self.K, c['BR']), cdiv(self.N, c['BC']))

    # scalar / constexpr kernel arguments (after the pointer operands), the same for every binding
    def fwd_consts(self):
        c = self.cfg('fwd')
        return dict(
            M=self.M,
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
            STORE_Q=self.store_q,
            PREC=self.prec,
            BM=c['BM'],
            BN=c['BN'],
            BK=c['BK'],
            GROUP_M=c['GROUP_M'],
        )

    def dx_consts(self):
        c = self.cfg('dx')
        return dict(
            M=self.M,
            N=self.N,
            K=self.K,
            RX=self.rx,
            CX=self.cx,
            RW=self.rw,
            CW=self.cw,
            S=c['S'],
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

    def dw_consts(self):
        c = self.cfg('dw')
        return dict(
            M=self.M,
            N=self.N,
            K=self.K,
            RX=self.rx,
            CX=self.cx,
            SPLIT=c['SPLIT'],
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
    M,
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
):
    """A Spec with tiles from the tune table (else DEFAULT) and the derived S / SPLIT. ``overrides``:
    {kernel: {param: value}} (tuning sweeps). ValueError when no dx row tile up to the tuned BM divides the row period
    (`dx_tile_fits` false), or when ``overrides`` give a dx S that is not a multiple of rx / BM up to the row blocks."""
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
    n_m = cdiv(M, dx['BM'])
    if xr:
        # dx keeps per-period partials in registers: BM must divide the row period, S a multiple of rx / BM
        if rx % dx['BM']:
            fits = [b for b in DX_ROW_TILES if b <= dx['BM'] and rx % b == 0]
            if not fits:
                raise ValueError(f'no dx row tile up to BM={dx["BM"]} divides the row period {rx}')
            dx['BM'] = fits[0]
            n_m = cdiv(M, dx['BM'])
        per = rx // dx['BM']
        if dx['S'] is None:  # enough programs to fill the GPU (2 per SM), as few partial slabs as that allows
            dx['S'] = min(n_m, cdiv(cdiv(2 * n_sm, cdiv(K, dx['BK'])), per) * per)
        if dx['S'] % per or dx['S'] > n_m:
            raise ValueError(f'dx S={dx["S"]} is not a multiple of {per} (rx / BM) up to {n_m} (row blocks)')
    elif dx['S'] is None:
        dx['S'] = n_m
    if dw['SPLIT'] is None:  # 2 programs per SM over the (K, N) tiles, at most one row block each
        tiles = cdiv(K, dw['BK']) * cdiv(N, dw['BN'])
        dw['SPLIT'] = max(1, min(cdiv(M, dw['BM']), cdiv(2 * n_sm, tiles)))
    frozen = {k: tuple(sorted(v.items())) for k, v in cfgs.items()}
    return Spec(M, N, K, tuple(qx), rx, cx, tuple(qw), rw, cw, bool(bias), bool(amax), prec, bool(store_q), **frozen)
