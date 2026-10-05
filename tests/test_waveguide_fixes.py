"""Regression tests for the RidgeWaveguide / waveguide-sweep fixes (local mode solver,
no FlexCredits, ~2 min).  Run from the repository root:

    python -m pytest tests/test_waveguide_fixes.py -q
"""
import numpy as np
import pytest
import tidy3d as td

from media.medium_helper import get_medium
from geometries import ridge_waveguide as rw
from waveguide import dispersion_in_geometry


def n_at(med, wl_um=1.55):
    return float(np.real(np.sqrt(med.eps_model(td.C_0 / wl_um))))


@pytest.fixture(scope="module")
def media():
    rng = (1.1, 2.4)
    return {"sin": get_medium("si3n4", wl_range=rng), "sio2": get_medium("sio2", wl_range=rng)}


def make_wg(media, width, height, clad=False, steps=30):
    return rw.get_passive_simulation(
        width=width, height=height, length=2.0, core_medium=media["sin"],
        sub_medium=media["sio2"], wavelength_um=1.55, substrate_height=4.0,
        top_cladding_medium=media["sio2"] if clad else None,
        top_cladding_height=2.0 if clad else 0.0, fine_min_steps_per_wvl=steps)


def test_fix5_sio2_is_fused_silica():
    """'sio2' is standard fused silica (Malitson, 1.444); old data kept as sio2_nyakuchena."""
    assert n_at(get_medium("sio2", wl_range=(1.1, 2.4))) == pytest.approx(1.444, abs=1e-3)
    assert n_at(get_medium("sio2_nyakuchena", wl_range=(1.1, 1.65))) == pytest.approx(1.420, abs=2e-3)


def test_fix1_fix3_get_dispersion_air(media):
    """TE00 tracked per wavelength; single-mode counted per polarization (air-clad 800x800).

    800x800 nm is near-square, so TM00 has the HIGHER index: the old code's
    "first TE-like mode index" logic is exactly where a TE/TM mix-up can occur.
    TE00 and TM00 are taken from get_modes() on the same waveguide (same mesh), so the
    check does not depend on Tidy3D version or mesh density.
    """
    wg = make_wg(media, 0.80, 0.80)
    modes = wg.get_modes(num_modes=8)["modes"]
    n_te00 = max(m["neff"] for m in modes if m["polarization"] == "TE")
    n_tm00 = max(m["neff"] for m in modes if m["polarization"] == "TM")
    assert n_tm00 - n_te00 > 0.008                 # TM00 above TE00 for this core

    r = wg.get_dispersion()
    for key in ("group_index", "guided_te_count", "guided_tm_count", "n_cutoff"):
        assert key in r
    assert r["dispersion_ps_nm_km"] > 0            # anomalous
    assert r["guided_te_count"] == 1 and r["is_singlemode"]
    # Returned mode is TE00 (agrees to ~1e-3 with the same-mesh TE00), not TM00
    assert abs(r["neff_fundamental"] - n_te00) < 2e-3
    assert abs(r["neff_fundamental"] - n_tm00) > 5e-3
    assert r["n_cutoff"] == pytest.approx(1.444, abs=1e-3)


def test_fix3_top_cladding_in_mode_count(media):
    """With SiO2 on top the same core supports extra TE modes (counted against max(n_sub, n_top))."""
    r = make_wg(media, 0.80, 0.80, clad=True).get_dispersion()
    assert r["guided_te_count"] > 1 and not r["is_singlemode"]


def test_fix2_fix4_dispersion_in_geometry_options(media, tmp_path):
    """Sweep accepts a top cladding and mesh setting, and skips failures instead of crashing."""
    res = dispersion_in_geometry(
        sweep_heights=[0.80], sweep_widths=[0.80], core_medium=media["sin"],
        sub_medium=media["sio2"], output_dir_name=tmp_path, wavelength_um=1.55,
        substrate_height=4.0, display_results=False,
        top_cladding_medium=None, fine_min_steps_per_wvl=30)
    assert res, "sweep returned no results"
