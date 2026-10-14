"""
tarang.geometries._modes

Shared helper for choosing ModeSpec.target_neff.

The Tidy3D mode solver returns the ``num_modes`` modes whose effective index is
closest to ``target_neff``. A guided mode always satisfies
n_cladding < n_eff < n_core, and the fundamental mode has the largest n_eff, so
targeting the core index selects the fundamental mode for any core material.
A fixed target (the former hard-coded 2.0) only works for SiN: for a silicon
core (n = 3.48) it can select a higher-order mode near n_eff = 2 instead.

Exposes: core_target_neff().
"""

from __future__ import annotations

import numpy as np
import tidy3d as td


def core_target_neff(core_medium: td.AbstractMedium, freq0: float) -> float:
    """Real refractive index of the core at ``freq0`` (Hz), for ModeSpec.target_neff."""
    n = float(np.real(np.sqrt(complex(core_medium.eps_model(freq0)))))
    if not np.isfinite(n) or n <= 1.0:
        raise ValueError(
            f"Core medium gives a non-physical index n = {n} at "
            f"{td.C_0 / freq0:.4f} um; cannot choose a mode-solver target."
        )
    return n
