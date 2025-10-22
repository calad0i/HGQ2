import inspect
import re
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
