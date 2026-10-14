"""Robust dispersion mapping and design selection for ridge waveguides.

Builds on TARANG's geometry (``tarang.geometries.ridge_waveguide``) and media system, and the
local Tidy3D mode solver (no FlexCredits).  Run the command-line tools from the TARANG
repository root, e.g.::

    python -m tarang.waveguide.dispersion_mapping.sweep --heights 0.60:0.90:0.05 --widths 0.6:1.6:0.1
    python -m tarang.waveguide.dispersion_mapping.maps --data results/air_sweep_50steps.jsonl

Python API::

    from tarang.waveguide.dispersion_mapping import solve_point
    r = solve_point(0.70, 1.00, steps=70)                     # air-clad Si3N4 on SiO2
    r = solve_point(0.74, 1.425, cladding="sio2")    # oxide-clad
    r = solve_point(0.70, 1.00, n_shift=+0.02)                # index-uncertainty study

Modules: core (physics), sweep (grid/finalist runner), maps (figures), convergence
(mesh study), broadband (D(lambda), D_int), report (finalists vs baseline).
"""
from .core import solve_point, build_sim, medium, stack_label, dispersion_from_neff

__all__ = ["solve_point", "build_sim", "medium", "stack_label", "dispersion_from_neff"]
