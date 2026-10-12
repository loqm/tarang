"""Deprecated alias of the ``edge_coupler`` package (renamed in TARANG v0.1.0).

``import edgeCoupler`` and ``import edgeCoupler.<module>`` still work and give the same objects
as ``edge_coupler``, with a DeprecationWarning. This alias will be removed in v0.2.0;
please import ``edge_coupler`` instead.
"""

from tarang._compat import alias_package

alias_package("edgeCoupler", "edge_coupler")

import edge_coupler as _new  # noqa: E402

from edge_coupler import *  # noqa: E402,F401,F403

__all__ = list(getattr(_new, "__all__", []))


def __getattr__(name):
    return getattr(_new, name)    # includes the new package's deprecated class names
