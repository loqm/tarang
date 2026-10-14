"""Regression tests for the v0.1.0 correctness fixes (audit C2-C5, R1, Q1) and for
the shared cache on every device wrapper. Local only: no FlexCredits.

    python -m pytest tests/test_phase3a_fixes.py -q
"""
import json
import logging
import re
from pathlib import Path

import numpy as np
import pytest
import tidy3d as td

import tarang.utilities.cloud_run as cr

REPO = Path(__file__).resolve().parents[1]

# The short partial ring used here ends inside the domain on purpose; TARANG warns about that.
pytestmark = pytest.mark.filterwarnings("ignore:The supplied partial-ring geometry")
WL = 1.55
F0 = td.C_0 / WL


def _medium(n):
    return td.Medium(permittivity=n ** 2)


SIN, SI, SIO2, AIR = _medium(1.996), _medium(3.476), _medium(1.444), _medium(1.0)


def _mode_specs(sim):
    specs = [m.mode_spec for m in sim.monitors if isinstance(m, td.ModeMonitor)]
    specs += [s.mode_spec for s in sim.sources if isinstance(s, td.ModeSource)]
    return specs


# ---- builders for every cloud-backed wrapper ------------------------------- #
def build_edge(core=SIN, sub=SIO2, out=True):
    from tarang.geometries import edge_coupler as g
    return g.get_active_simulation(
        tip_width=0.2, taper_length=10.0, wg_width=0.8, wg_height=0.8, core_medium=core,
        box_medium=sub, wavelength_um=WL, top_cladding_medium=AIR,
        fine_steps_per_wvl=8, coarse_steps_per_wvl=6,
        in_coupling=not out, out_coupling=out)


def build_ring(core=SIN, sub=SIO2):
    from tarang.geometries import partial_ring_resonator as g
    return g.get_active_simulation(
        radius=20.0, gap=0.3, wg_width=0.8, wg_height=0.8, core_medium=core,
        box_medium=sub, wavelength_um=WL, span_angle_rad=np.pi / 6,
        fine_steps_per_wvl=8, coarse_steps_per_wvl=6)


def build_arc(core=SIN, sub=SIO2):
    from tarang.geometries import resonator_arc as g
    return g.get_active_simulation(
        wavelength_um=WL, radius=20.0, wg_width=0.8, wg_height=0.8, core_medium=core,
        box_medium=sub, top_cladding_medium=AIR, bend_angle_deg=30.0,
        fine_steps_per_wvl=8, coarse_steps_per_wvl=6)


def build_gc(core=SIN, sub=SIO2):
    from tarang.grating_coupler.grating_coupler import GratingConfig, GratingCoupler
    cfg = GratingConfig(wavelength_um=WL, theta_deg=-12.0, mfd=10.4, source_x=3.84,
                        source_gap=2.0, core_height=0.8, top_cladding_height=2.0, box_height=4.0,
                        handle_height=8.0, etch_depth=0.5, period=0.82, ff=0.51,
                        n_teeth=22, w_grating=10.4, pol_angle=np.pi / 2, buffer=3.0)
    return GratingCoupler.from_config(cfg, core, sub, None, AIR,
                                      fine_steps_per_wvl=8, coarse_steps_per_wvl=6)


BUILDERS = {"edge": build_edge, "ring": build_ring, "arc": build_arc, "gc": build_gc}


@pytest.mark.parametrize("name", list(BUILDERS))
def test_every_wrapper_cache_key_tracks_box_medium(name):
    a = BUILDERS[name]()
    b = BUILDERS[name](sub=_medium(1.420))
    assert a.get_dir_name() != b.get_dir_name()
    assert len(a.get_dir_name()) < 90


@pytest.mark.parametrize("name", list(BUILDERS))
def test_every_wrapper_runs_through_cost_guard(name, tmp_path, fake_web):
    fw = fake_web
    fw.cost = 1.0
    dev = BUILDERS[name]()
    dev.max_cost = 0.5
    with pytest.raises(cr.CostLimitExceededError):
        dev._run_sim(tmp_path)
    assert "start" not in fw.names()


# ---- C2: mode-solver target from the core index --------------------------- #
@pytest.mark.parametrize("name", ["edge", "gc", "ring"])
@pytest.mark.parametrize("core,n", [(SIN, 1.996), (SI, 3.476)])
def test_c2_target_neff_is_core_index(name, core, n):
    sim = BUILDERS[name](core=core).sim
    specs = _mode_specs(sim)
    assert specs, "expected at least one mode monitor/source"
    for spec in specs:
        assert spec.target_neff == pytest.approx(n, abs=1e-6)


def test_c2_edge_in_coupling_monitor_also_fixed():
    for spec in _mode_specs(build_edge(core=SI, out=False).sim):
        assert spec.target_neff == pytest.approx(3.476, abs=1e-6)


def test_c2_no_hard_coded_targets_left():
    for p in (REPO / "geometries").glob("*.py"):
        assert "target_neff=2.0" not in p.read_text(encoding="utf-8"), p.name


# ---- C3: partial wavelength coverage warns -------------------------------- #
def test_c3_get_medium_warns_on_extrapolation(tmp_path, monkeypatch, caplog):
    import tarang.media.medium_helper as mh
    monkeypatch.setattr(mh, "FITTED_DIR", tmp_path)
    with caplog.at_level(logging.WARNING, logger="tarang"):
        mh.get_medium("sio2_nyakuchena", wl_range=(1.20, 2.00), tolerance=1e-3, max_num_poles=3)  # data stops at 1.65 um
    assert "extrapolation" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="tarang"):
        mh.get_medium("sio2_nyakuchena", wl_range=(1.20, 1.60), tolerance=1e-3, max_num_poles=3)
    assert "extrapolation" not in caplog.text


# ---- C5: stacked CSV handled without touching the source ------------------ #
STACKED = "wl,n\n1.2,2.00\n1.4,1.99\n1.6,1.98\n1.8,1.97\nwl,k\n1.2,0\n1.4,0\n1.6,0\n1.8,0\n"


def test_c5_read_stacked_in_memory(tmp_path):
    from tarang.media import format_csv
    src = tmp_path / "stacked.csv"
    src.write_text(STACKED)
    df = format_csv.read_stacked_csv(src)
    assert list(df.columns) == ["wl", "n", "k"] and len(df) == 4
    assert src.read_text() == STACKED


def test_c5_fix_writes_copy_and_never_overwrites(tmp_path):
    from tarang.media import format_csv
    src = tmp_path / "stacked.csv"
    src.write_text(STACKED)
    assert format_csv.fix_stacked_csv(src)
    out = tmp_path / "stacked_merged.csv"
    assert out.exists() and src.read_text() == STACKED
    assert format_csv.fix_stacked_csv(src)                      # identical: no-op
    out.write_text("something else")
    with pytest.raises(FileExistsError):
        format_csv.fix_stacked_csv(src)
    with pytest.raises(ValueError):
        format_csv.fix_stacked_csv(src, output_path=src)


def test_c5_merged_copy_has_unix_newlines_on_any_platform(tmp_path, monkeypatch):
    """Windows regression: pandas used os.linesep, and text-mode writing then
    produced \\r\\r\\n (blank rows) and broke the identical-content check."""
    import os
    from tarang.media import format_csv
    monkeypatch.setattr(os, "linesep", "\r\n")          # emulate Windows
    src = tmp_path / "stacked.csv"
    src.write_text(STACKED)
    assert format_csv.fix_stacked_csv(src)
    raw = (tmp_path / "stacked_merged.csv").read_bytes()
    assert b"\r" not in raw and raw.count(b"\n") == 5      # header + 4 rows
    assert format_csv.fix_stacked_csv(src)                  # second call is a no-op


def test_c5_get_medium_reads_stacked_file_without_modifying_it(tmp_path, monkeypatch):
    import tarang.media.medium_helper as mh
    src = tmp_path / "stacked.csv"
    src.write_text(STACKED)
    monkeypatch.setattr(mh, "FITTED_DIR", tmp_path / "fits")
    monkeypatch.setattr(mh, "_material_path", lambda key: src)
    med = mh.get_medium("whatever", wl_range=(1.2, 1.8), tolerance=1e-3, max_num_poles=3)
    n = float(np.real(np.sqrt(med.eps_model(td.C_0 / 1.4))))
    assert n == pytest.approx(1.99, abs=5e-3)
    assert src.read_text() == STACKED


# ---- C4: explicit guided-mode flag ------------------------------------------ #
def test_c4_get_modes_flags_unguided(caplog):
    from tarang.geometries import ridge_waveguide as rw
    from tarang.waveguide.waveguide import WaveguideModeError
    # 150 nm x 100 nm SiN on SiO2 at 1.55 um: far below cutoff.
    wg = rw.get_passive_simulation(
        width=0.15, height=0.10, length=2.0, core_medium=SIN, box_medium=SIO2,
        wavelength_um=WL, box_height=2.0, fine_steps_per_wvl=20)
    with caplog.at_level(logging.WARNING, logger="tarang"):
        r = wg.get_modes(num_modes=2)
    assert r["has_guided_mode"] is False
    assert not any(m["is_guided"] for m in r["modes"])
    assert "No guided modes" in caplog.text
    with pytest.raises(WaveguideModeError):
        wg.get_modes(num_modes=2, require_guided=True)


# ---- R1: Tidy3D 2.12 logging --------------------------------------------- #
def test_r1_log_level_helper():
    from tarang import set_tidy3d_log_level
    set_tidy3d_log_level("warning")
    assert td.config.logging.level == "WARNING"
    set_tidy3d_log_level("ERROR")
    assert td.config.logging.level == "ERROR"


def test_r1_notebooks_use_no_removed_config():
    for nb in REPO.glob("*.ipynb"):
        for cell in json.loads(nb.read_text(encoding="utf-8"))["cells"]:
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            code = "\n".join(l.split("#", 1)[0] for l in src.splitlines())
            assert not re.search(r"config\.logging_level\s*=", code), nb.name
            assert not re.search(r"config\.show_progress_bars\s*=", code), nb.name
