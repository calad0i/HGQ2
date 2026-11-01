import inspect
import re
from collections.abc import Sequence
from string import ascii_letters
from typing import Any

import numpy as np

numbers = int | float | np.integer | np.floating


def gather_vars_to_kwargs(skip_pattern=None) -> dict[str, Any]:
    """Gather all local variables in the calling function and return them as a dictionary. If a variable is named `kwargs`, it will be updated with the rest of the variables. This function is useful for initializing classes with a large number of arguments.

    Parameters
    ----------
    skip_pattern : str, optional
        A regular expression pattern to match variables to skip. If a variable matches the pattern, it will not be included in the returned dictionary. The default is None.

    Returns
    -------
    dict[str, Any]
        A dictionary containing all local variables in the calling function, except for those that start and end with double underscores.
    """
    vars = dict(inspect.getouterframes(inspect.currentframe(), 2)[1][0].f_locals)
    kwarg = vars.pop('kwargs', {})
    if skip_pattern:
        m = re.compile(skip_pattern)
        for k in list(vars.keys()):
            if m.match(k):
                del vars[k]
    kwarg.update(vars)
    for k in list(kwarg.keys()):
        if k.startswith('__') and k.endswith('__'):
            del kwarg[k]
    return kwarg


def expand_ellipsis(equation: str, ndims: Sequence[int]) -> str:
    lhs, out = (s.strip() for s in equation.split('->', 1))
    specs = [s.strip() for s in lhs.split(',')]
    assert len(ndims) == len(specs), f'Expected {len(specs)} inputs, but got {len(ndims)}.'
    n_elided = None
    for spec, ndim in zip(specs, ndims):
        if '...' not in spec:
            assert len(spec) == ndim, f'Einsum operand {spec} does not match input rank {ndim}.'
            continue
        n = ndim - len(spec) + 3
        assert n >= 0, 'Invalid ellipsis notation: expanded to negative dimensions.'
        assert n_elided in (None, n), 'Inconsistent ellipsis expansion dimensions across inputs.'
        n_elided = n
    if n_elided is not None:
        rep = ''.join(c for c in ascii_letters if c not in equation)[:n_elided]
        specs, out = [s.replace('...', rep) for s in specs], out.replace('...', rep)
    return ','.join(specs) + '->' + out
