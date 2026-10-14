"""
tarang._compat

Keep pre-v0.1.0 names working for one release while pointing users to the new ones.

* ``renamed_params(old="new", ...)``: decorator for functions and methods.
  Accepts the old keyword, warns, and passes it on under the new name. Passing
  both old and new raises TypeError. Positional calls are unaffected: renamed
  parameters keep their positions.
* ``RenamedAttributes``: mixin for classes whose attributes were renamed.
  Reading or writing an old attribute name warns and forwards to the new one.
* ``deprecated_module_names(module_globals, {"oldName": "NewName"})``: returns a
  module ``__getattr__`` (PEP 562), so ``from pkg import oldName`` still works
  and warns.

All warnings are ``DeprecationWarning`` (Python hides these by default outside
``__main__``; pytest and Jupyter show them). The old names will be removed in
v0.2.0.

Exposes: renamed_params, RenamedAttributes, deprecated_module_names.

The pre-v0.1.0 package names (``gratingCoupler`` etc.) and the top-level package
layout are not aliased: ``python -m tarang.migrate`` rewrites old imports instead.
"""

from __future__ import annotations

import functools
import warnings

REMOVAL = "v0.2.0"


def _warn(old: str, new: str, kind: str, stacklevel: int = 3) -> None:
    warnings.warn(
        f"{kind} '{old}' is deprecated and will be removed in TARANG {REMOVAL}; "
        f"use '{new}' instead.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


# --------------------------------------------------------------------------- #
# Keyword arguments
# --------------------------------------------------------------------------- #
def renamed_params(**old_to_new: str):
    """Decorator: accept old keyword names as deprecated aliases of new ones."""

    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for old, new in old_to_new.items():
                if old in kwargs:
                    if new in kwargs:
                        raise TypeError(
                            f"{func.__qualname__}() got both '{old}' (deprecated) and "
                            f"'{new}'; pass only '{new}'."
                        )
                    _warn(old, new, "Keyword argument")
                    kwargs[new] = kwargs.pop(old)
            return func(*args, **kwargs)

        wrapper.__renamed_params__ = dict(old_to_new)
        return wrapper

    return decorate


# --------------------------------------------------------------------------- #
# Attributes
# --------------------------------------------------------------------------- #
class RenamedAttributes:
    """Mixin: map deprecated attribute names (``_renamed_attrs``) to new ones."""

    _renamed_attrs: dict = {}

    def __getattr__(self, name):            # only called when normal lookup fails
        new = type(self)._renamed_attrs.get(name)
        if new is None:
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        _warn(name, new, "Attribute")
        return getattr(self, new)

    def __setattr__(self, name, value):
        new = type(self)._renamed_attrs.get(name)
        if new is not None:
            _warn(name, new, "Attribute")
            name = new
        object.__setattr__(self, name, value)


# --------------------------------------------------------------------------- #
# Module-level names (classes, functions)
# --------------------------------------------------------------------------- #
def deprecated_module_names(module_globals: dict, old_to_new: dict):
    """Return a PEP 562 module ``__getattr__`` that serves old names with a warning."""
    module_name = module_globals.get("__name__", "?")

    def __getattr__(name):
        new = old_to_new.get(name)
        if new is None:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        _warn(f"{module_name}.{name}", f"{module_name}.{new}", "Name")
        return module_globals[new]

    return __getattr__
