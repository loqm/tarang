"""Public edge_coupler imports.

Exposes: EdgeCoupler, parametric_optimization_sweep.
"""

from .edge_coupler import EdgeCoupler
from .optimization_sweep import parametric_optimization_sweep

__all__ = ["EdgeCoupler", "parametric_optimization_sweep"]

from tarang._compat import deprecated_module_names as _dmn

# Pre-v0.1.0 class names keep working (with a DeprecationWarning) until v0.2.0.
__getattr__ = _dmn(globals(), {'edgeCoupler': 'EdgeCoupler'})
