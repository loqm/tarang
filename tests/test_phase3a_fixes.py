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


@pytest.mark.slow
def test_get_medium_accepts_your_own_csv_file(tmp_path, monkeypatch):
    """Measured data (e.g. ellipsometry of your film) can be used by path, without
    editing media.json; it is fitted and cached like the built-in materials."""
    import shutil
    import tarang.media.medium_helper as mh
    monkeypatch.setattr(mh, "FITTED_DIR", tmp_path / "fits")
    own = tmp_path / "my_film.csv"
    shutil.copy(mh._material_path("si3n4"), own)
    m_own = mh.get_medium(str(own), wl_range=(1.45, 1.65))
    m_key = mh.get_medium("si3n4", wl_range=(1.45, 1.65))
    f = 2.99792458e14 / 1.55
    assert abs(float(np.real(m_own.nk_model(f)[0])) - float(np.real(m_key.nk_model(f)[0]))) < 1e-9
    assert (tmp_path / "fits" / "my_film_index.json").exists()
    with pytest.raises(FileNotFoundError):
        mh.get_medium(str(tmp_path / "missing.csv"), wl_range=(1.45, 1.65))


@pytest.mark.slow
def test_fit_follows_sparse_data_between_the_points(tmp_path, monkeypatch):
    """The Si3N4 table has only three points between 0.68 and 0.88 um. A 10-pole fit there
    can pass through them and still swing to n = 3.8 and 1.5 in between (found in Phase 10,
    at 780 nm). Whatever route the fit takes, the result must follow the data between the
    points; good fits, such as the 1550 nm one, are kept unchanged.

    Whether the first 10-pole fit swings depends on floating-point details of the machine
    (it did here, not on GitHub's Ubuntu / Python 3.12 runner), so this test checks the
    result only; test_swinging_fit_is_refitted checks the refit itself."""
    import json as _json
    import pandas as pd
    import tarang.media.medium_helper as mh
    monkeypatch.setattr(mh, "FITTED_DIR", tmp_path / "fits")
    data = pd.read_csv(mh._material_path("si3n4"))
    wl, n = data["wl"].to_numpy(float), data["n"].to_numpy(float)

    m780 = mh.get_medium("si3n4", wl_range=(0.679, 0.881))
    probe = np.linspace(0.679, 0.881, 401)
    n_fit = np.real(m780.nk_model(td.C_0 / probe)[0])
    assert np.max(np.abs(n_fit - np.interp(probe, wl, n))) < 2e-3
    assert mh.fit_excursion(m780, wl, n, 0.679, 0.881) <= mh.MAX_EXCURSION

    mh.get_medium("si3n4", wl_range=(1.35, 1.75))
    entries = _json.loads((tmp_path / "fits" / "Luke_Si3N4_index.json").read_text())["fits"]
    assert len(entries) == 2                                       # 780 nm, then 1550 nm
    assert "smoothness_fallback" not in entries[1]                 # 1550 nm: first fit kept
    assert entries[1]["wl_range"] == [1.35, 1.75]


def _excursion_after_first(monkeypatch, mh, first_value, later=None):
    """Make fit_excursion report `first_value` for the first fit, then the real value (or
    `later`), so the refit path runs the same way on every machine."""
    real = mh.fit_excursion
    calls = []

    def fake(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return first_value
        return real(*args, **kwargs) if later is None else later
    monkeypatch.setattr(mh, "fit_excursion", fake)
    return calls, real


def test_swinging_fit_is_refitted(tmp_path, monkeypatch, caplog):
    """A fit that strays between the data points is refitted with fewer poles, flagged in
    the manifest, reported once, and reused from the cache afterwards."""
    import json as _json
    import pandas as pd
    import tarang.media.medium_helper as mh
    monkeypatch.setattr(mh, "FITTED_DIR", tmp_path / "fits")
    calls, real = _excursion_after_first(monkeypatch, mh, first_value=0.1)

    with caplog.at_level(logging.WARNING, logger="tarang"):
        m = mh.get_medium("si3n4", wl_range=(1.45, 1.65))
    assert any("strays" in r.getMessage() for r in caplog.records)
    assert len(calls) == 2                           # the swinging fit, then one refit
    data = pd.read_csv(mh._material_path("si3n4"))
    wl, n = data["wl"].to_numpy(float), data["n"].to_numpy(float)
    assert real(m, wl, n, 1.45, 1.65) <= mh.MAX_EXCURSION

    entries = _json.loads((tmp_path / "fits" / "Luke_Si3N4_index.json").read_text())["fits"]
    assert len(entries) == 1
    assert entries[0]["smoothness_fallback"] is True
    assert entries[0]["wl_range"] == [1.45, 1.65]   # fewer poles were enough

    m_again = mh.get_medium("si3n4", wl_range=(1.45, 1.65))   # cache hit, no new fit
    entries = _json.loads((tmp_path / "fits" / "Luke_Si3N4_index.json").read_text())["fits"]
    assert len(entries) == 1
    f = td.C_0 / 1.55
    assert float(np.real(m_again.nk_model(f)[0])) == float(np.real(m.nk_model(f)[0]))


def test_fit_that_never_follows_the_data_is_refused(tmp_path, monkeypatch):
    """If no pole count or range brings the fit within 0.05 of the data, get_medium raises
    instead of returning a wrong material, after trying fewer poles and wider ranges."""
    import tarang.media.medium_helper as mh
    monkeypatch.setattr(mh, "FITTED_DIR", tmp_path / "fits")
    tried = []

    class QuickFitter:                               # stands in for the real fitter
        def __init__(self, wvl_range, **kwargs):
            self.rng = tuple(wvl_range)

        def fit(self, tolerance_rms, max_num_poles):
            tried.append((self.rng, max_num_poles))
            return td.PoleResidue(eps_inf=4.0, poles=()), 1e-6
    monkeypatch.setattr(mh, "Fitter", QuickFitter)
    monkeypatch.setattr(mh, "fit_excursion", lambda *a, **k: 0.1)

    with pytest.raises(RuntimeError, match="No pole fit follows the data"):
        mh.get_medium("si3n4", wl_range=(1.45, 1.65))
    assert not (tmp_path / "fits" / "Luke_Si3N4_index.json").exists()   # nothing cached
    assert tried[0] == ((1.45, 1.65), 10)
    assert [p for _, p in tried[1:6]] == [6, 4, 3, 2, 1]                # fewer poles
    widths = [r[1] - r[0] for r, _ in tried]
    assert max(widths) > 0.2 + 1e-9                                       # then wider ranges
