"""Smoke tests for waveguide.dispersion_mapping (coarse mesh, ~1 min, no FlexCredits).

Run from the repository root:  python -m pytest tests/test_dispersion_mapping.py -q
Coarse mesh (30 steps/lambda) for speed; assertions are mesh- and version-independent.
"""
import pytest

from waveguide.dispersion_mapping import solve_point, medium
from waveguide.dispersion_mapping.core import n_of


def test_media_registry_has_malitson():
    assert n_of(medium("sio2"), 1.55) == pytest.approx(1.444, abs=1e-3)


@pytest.mark.slow
def test_reference_point_800x800():
    # Coarse mesh for speed: absolute n_eff at 30 steps/lambda varies by ~0.005 with
    # Tidy3D version (grid placement), so check physics, not 4-decimal values.
    r = solve_point(0.80, 0.80, steps=30)
    n_te, n_tm = r["neff_TE"][0], r["neff_TM"][0]
    assert 1.65 < n_te < 1.68                  # converged value is 1.6605 (70 steps)
    assert n_tm - n_te > 0.008                 # near-square core: TM00 above TE00
    assert r["guided_TE"] == 1
    assert r["D_TE0"] > 0                      # anomalous
    assert r["fit_resid_TE0"] < 1e-4           # smooth n_eff(lambda) fit


@pytest.mark.slow
def test_index_shift_raises_dispersion():
    base = solve_point(0.70, 1.00, steps=30)["D_TE0"]
    up = solve_point(0.70, 1.00, steps=30, n_shift=0.02)["D_TE0"]
    assert up > base
