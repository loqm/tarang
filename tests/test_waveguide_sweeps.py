"""Waveguide mode and confinement sweeps on a coarse mesh (Phase 4). Local mode
solver, no FlexCredits, a few seconds.

    python -m pytest tests/test_waveguide_sweeps.py -q
"""
import tidy3d as td

SIN, SIO2 = td.Medium(permittivity=1.996 ** 2), td.Medium(permittivity=1.444 ** 2)
COMMON = dict(sweep_heights=[0.7], sweep_widths=[0.8, 1.0], core_medium=SIN, box_medium=SIO2,
              wavelength_um=1.55, box_height=3.0, display_results=False, fine_steps_per_wvl=20)


def test_confinement_rises_with_width(tmp_path, no_show):
    from tarang.waveguide import confinement_in_geometry
    res = confinement_in_geometry(output_dir_name=tmp_path, **COMMON)
    widths, conf_pct, _ = res[0.7]
    assert list(widths) == [0.8, 1.0]
    assert all(0.0 < c < 100.0 for c in conf_pct)
    assert conf_pct[1] > conf_pct[0]          # a wider core confines more power
    assert list(tmp_path.glob("*.png"))


def test_mode_sweep_writes_its_figure(tmp_path, no_show):
    from tarang.waveguide import modes_in_geometry
    modes_in_geometry(output_dir_name=tmp_path, num_modes=4, **COMMON)
    assert list(tmp_path.glob("*.png"))
