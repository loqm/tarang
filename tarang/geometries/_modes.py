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


#: cores up to this index keep the index 2 every mesh rule used before v0.2.0 (Si3N4 1.98-2.03,
#: Ta2O5 2.06-2.08: within a few per cent), so their simulations and cloud fingerprints are
#: unchanged; above it the real index sets the cell size.
MESH_INDEX_KEEP = 2.1


def mesh_index(core_medium: td.AbstractMedium, freq0: float) -> float:
    """Index that sets a 'cells per wavelength in the core' mesh: 2.0 for cores up to
    n = 2.1 (as before v0.2.0, when the rule read an attribute fitted media do not have and
    always fell back to 2.0), the core's own index above it (GaP, AlGaAs, Si), so a
    high-index core gets the cells per wavelength the setting asks for."""
    n = core_target_neff(core_medium, freq0)
    return n if n > MESH_INDEX_KEEP else 2.0
