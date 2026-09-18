"""Public utilities imports.

Exposes: sim_plotter, load_multinest_samples(), plot_corner(), and lazily
loaded nested_sampling_post_process.
"""

from . import sim_plotter
from .corner_plot import load_multinest_samples, plot_corner


def __getattr__(name):
    """Load optional nested-sampling post-processing only when requested.
    Ordinary simulation plotting does not require MultiNest or GetDist.
    """
    if name == "nested_sampling_post_process":
        from importlib import import_module

        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "sim_plotter",
    "nested_sampling_post_process",
    "load_multinest_samples",
    "plot_corner",
]
