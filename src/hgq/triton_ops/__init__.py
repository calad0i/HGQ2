"""Triton kernels for hgq2: the QDense / QEinsumDense training route and the QMultiHeadAttention one."""

import functools
import os
from importlib import import_module

import keras

_on = {'qdense': False, 'attention': False}


def enable(qdense: bool = True, attention: bool = True):
    """Switch the Triton training routes on or off. Switching one on imports it and raises if cannot run on selected backend."""
    backend = keras.backend.backend()
    for kind, on, package in (('qdense', qdense, 'qdense'), ('attention', attention, 'attn')):
        if on:
            if backend not in ('jax', 'torch'):
                raise RuntimeError(f'the Triton {kind} route needs the jax or torch backend, not {backend}')
            try:
                import_module(f'.{package}.layer', __name__).check()
            except ImportError as e:
                needs = 'triton, quantizers.triton' + (' and jax-triton' if backend == 'jax' else '')
                raise ImportError(f'the Triton {kind} route is unavailable on {backend} (needs {needs}): {e}') from e
            if backend == 'torch':
                _patch_inductor()
            else:
                _patch_jax_triton()
        _on[kind] = bool(on)


@functools.cache
def _patch_inductor():
    """torch.compile rebuilds a user Triton kernel's source from its jit closure, importing only symbols of `triton*`
    modules; also import the Triton builtins the closure names from other modules (quantizers.triton's
    `_fixed_round_value`, ...). The imports go last: the source is spliced after a decorator, and globals resolve at
    compile time."""
    import dis

    from torch._inductor.codegen import wrapper
    from triton import JITFunction
    from triton.language.core import TRITON_BUILTIN

    source = getattr(wrapper, 'user_defined_triton_kernel_transitive_closure_source_code', None)
    if source is None:
        return

    def imports(kernel, seen):
        for name in dict.fromkeys(i.argval for i in dis.Bytecode(kernel.fn) if i.opname == 'LOAD_GLOBAL'):
            sym = kernel.fn.__globals__.get(name)
            if isinstance(sym, JITFunction) and sym not in seen:
                seen.add(sym)
                yield from imports(sym, seen)
            elif getattr(sym, TRITON_BUILTIN, False) and not sym.__module__.startswith('triton'):
                yield f'from {sym.__module__} import {sym.__name__} as {name}'

    @functools.wraps(source)
    def closure_source(kernel, *args, **kwargs):
        return '\n'.join([source(kernel, *args, **kwargs), *dict.fromkeys(imports(kernel, {kernel}))]) + '\n'

    wrapper.user_defined_triton_kernel_transitive_closure_source_code = closure_source


@functools.cache
def _patch_jax_triton():
    """Replace jax-triton's dead-code-elimination rule for triton_call.

    jax-triton 0.4.x's rule takes (used_outs, eqn) but JAX 0.12 passes (used_outs, live_ins, eqn), and no jax-triton
    version drops an unused call, so a rematerialized backward reruns the forward kernel. The new rule drops a call
    whose outputs are all unused, unless it has effects or has_side_effect."""
    import inspect

    import jax
    import jax_triton
    from jax._src.interpreters import partial_eval as pe
    from jax_triton import triton_lib

    def binds(n):
        try:
            signature.bind(*[None] * n)
        except TypeError:
            return False
        return True

    if len(inspect.signature(pe._default_dce_rule).parameters) != 3:
        raise RuntimeError(f'JAX {jax.__version__} does not call DCE rules as (used_outs, live_ins, eqn)')
    primitive = triton_lib.triton_kernel_call_p
    rule = pe.dce_rules.get(primitive, pe._default_dce_rule)
    signature = inspect.signature(rule)
    full = binds(3)  # JAX's form, or jax-triton main's (used_outs, *args)
    if not full and not binds(2):
        raise RuntimeError(
            f'jax-triton {jax_triton.__version__} registers a DCE rule {signature} accepting neither'
            ' (used_outs, live_ins, eqn) nor (used_outs, eqn)'
        )

    def dce_rule(used_outs, live_ins, eqn):
        # has_side_effect adds no jaxpr effect, so has_effects alone would drop such a call
        if not any(used_outs) and not eqn.params.get('has_side_effect') and not pe.has_effects(eqn, live_ins):
            return [False] * len(eqn.invars), None
        return rule(used_outs, live_ins, eqn) if full else rule(used_outs, eqn)

    pe.dce_rules[primitive] = dce_rule


def enabled(kind: str) -> bool:
    return _on[kind]


enable(qdense=os.environ.get('HGQ_TRITON_QDENSE') == '1', attention=os.environ.get('HGQ_TRITON_ATTENTION') == '1')
