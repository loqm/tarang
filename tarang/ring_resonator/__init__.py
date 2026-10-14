"""Public ring_resonator imports.

Exposes: HalfRingResonator, RingResonator, PartialRingResonator, ResonatorArc.
"""

from .ring_resonator import HalfRingResonator
from .ring_resonator import RingResonator
from .ring_resonator import PartialRingResonator

from .resonator_arc import ResonatorArc

__all__ = ["HalfRingResonator", "RingResonator", "PartialRingResonator", "ResonatorArc"]

from tarang._compat import deprecated_module_names as _dmn

# Pre-v0.1.0 class names keep working (with a DeprecationWarning) until v0.2.0.
__getattr__ = _dmn(globals(), {'halfringResonator': 'HalfRingResonator', 'ringResonator': 'RingResonator', 'partialRingResonator': 'PartialRingResonator', 'resonatorArc': 'ResonatorArc'})
