"""Deprecated alias of the ``ring_resonator`` package (renamed in TARANG v0.1.0).

``import ringResonator`` and ``import ringResonator.<module>`` still work and give the same objects
as ``ring_resonator``, with a DeprecationWarning. This alias will be removed in v0.2.0;
please import ``ring_resonator`` instead.
"""

from tarang._compat import alias_package

alias_package("ringResonator", "ring_resonator")

import ring_resonator as _new  # noqa: E402

from ring_resonator import *  # noqa: E402,F401,F403

__all__ = list(getattr(_new, "__all__", []))


def __getattr__(name):
    return getattr(_new, name)    # includes the new package's deprecated class names
