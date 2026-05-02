"""Compatibility shim for chumpy on Python 3.11 + numpy 2.x.

chumpy 0.70 hits two upstream incompatibilities on this stack:

1. inspect.getargspec was removed in Python 3.11. Replace with getfullargspec.
2. chumpy imports `numpy.bool, numpy.int, numpy.float, numpy.complex,
   numpy.object, numpy.unicode, numpy.str` at module load time. Numpy 1.20
   deprecated these and numpy 2.0 removed them entirely.

We back-fill the names on the numpy module so chumpy's top-level imports
succeed. The aliases route to the Python builtins (or str for unicode),
matching the historical numpy semantics.

Why we need chumpy at all: FLAME 2020 generic_model.pkl is a pickled dict
that contains chumpy.Ch instances. Unpickling fails without chumpy.

Import the shim first thing in any module that triggers FLAME loading:

    from ofiq_syngen.three_d.lift import _chumpy_compat  # noqa: F401
"""

from __future__ import annotations

import builtins
import inspect

import numpy as np

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]

_NUMPY_ALIASES = {
    "bool": builtins.bool,
    "int": builtins.int,
    "float": builtins.float,
    "complex": builtins.complex,
    "object": builtins.object,
    "str": builtins.str,
    "unicode": builtins.str,
}

for _name, _value in _NUMPY_ALIASES.items():
    if not hasattr(np, _name):
        setattr(np, _name, _value)
