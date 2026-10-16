"""python -m tarang.design: the guided design tool (Phase 10). ~20 s, no FlexCredits.

    python -m pytest tests/test_design.py -q

The local stages call the dispersion-mapping tools, which have their own tests; here the
settings file, the stage bookkeeping, the physics helpers, the cost confirmation of the
cloud stages (with the fake cloud) and the layout and report are checked on a design
whose earlier results are filled in by hand.
"""
import io
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tarang.design import physics
from tarang.design.config import DesignConfig
from tarang.design.stages import Context, _rank_targets, settings_key
from tarang.design.ui import Prompter, parse_list


def _ctx(tmp_path, **cfg_changes):
    cfg = DesignConfig()
    cfg.run.band_um = 0.2
    for dotted, value in cfg_changes.items():
        sec, key = dotted.split("__")
        setattr(getattr(cfg, sec), key, value)
        cfg.given.add(f"{sec}.{key}")
    ui = Prompter(interactive=False, out=io.StringIO())
    ctx = Context(cfg, ui, tmp_path / "run")
    ctx.state["stages"] = {
        "cross_section": {"results": dict(height_um=0.7, width_um=1.0, steps=70, D=379.4,
                                          n_group=2.2233, n_eff=1.7105, n_cut=1.444,
                                          guided_TE=1, guided_TM=1, aeff_um2=0.75,
                                          D_min_tol=336.0, D_max_tol=400.0,
                                          tolerance_box_target=True, mesh_change=0.01,
                                          map_file="m.jsonl", map_figure="m.png",
                                          report="r.md")},
        "ring": {"results": dict(radius_um=107.5, fsr_ghz=199.6, n_group=2.2233, loss_db_cm=0.4,
                                 bend_loss_db_cm=0.0, scattering_loss_db_cm=0.3,
                                 absorption_loss_db_cm=0.1, q_intrinsic=9.785e5,
                                 D2_2pi_MHz=18.9, D3_2pi_kHz=19.8, zdw_nm="none in band",
                                 profile_figure="p.png")},
    }
    return ctx


# ---- settings file ------------------------------------------------------------------ #
def test_settings_file_round_trip_and_unknown_keys(tmp_path):
    cfg = DesignConfig()
    cfg.run.wavelength_um = 0.78
    cfg.stack.core_height_um = 0.3
    cfg.cross_section.widths_um = [0.4, 0.5]
    cfg.layout.ring_layer = [2, 0]
    path = cfg.save(tmp_path / "design.toml")
    back = DesignConfig.load(path)
    assert back.run.wavelength_um == 0.78 and back.stack.core_height_um == 0.3
    assert back.cross_section.widths_um == [0.4, 0.5] and back.layout.ring_layer == [2, 0]
    assert back.cross_section.height_um is None              # unset values stay unset
    assert "stack.core_height_um" in back.given and "cross_section.height_um" not in back.given
    path.write_text(path.read_text() + "\n[ring]\nradiuss_um = 3\n")
    with pytest.raises(Exception):
        DesignConfig.load(path)


def test_list_and_range_answers():
    assert parse_list("0.6, 0.8,1.0") == [0.6, 0.8, 1.0]
    assert parse_list("0.60:0.90:0.10") == [0.6, 0.7, 0.8, 0.9]
    assert parse_list("[1, 0]") == [1, 0]


def test_non_interactive_answers_use_defaults_and_are_recorded(tmp_path):
    cfg = DesignConfig()
    ui = Prompter(interactive=False, out=io.StringIO())
    assert ui.value(cfg, "ring", "radius_um", default=107.5) == 107.5
    assert cfg.ring.radius_um == 107.5 and "ring.radius_um" in cfg.given


def test_interactive_answer_is_checked_and_stored(tmp_path):
    answers = iter(["-3", "0.78"])
    cfg = DesignConfig()
    ui = Prompter(interactive=True, out=io.StringIO(), inp=lambda q: next(answers))
    wl = ui.value(cfg, "run", "wavelength_um",
                  check=lambda v: "" if v > 0 else "must be positive")
    assert wl == 0.78 and cfg.run.wavelength_um == 0.78


def test_a_settings_change_reruns_that_stage_and_later_ones():
    a, b = DesignConfig(), DesignConfig()
    b.ring.fsr_ghz = 100.0
    assert settings_key(a, "cross_section") == settings_key(b, "cross_section")
    assert settings_key(a, "ring") != settings_key(b, "ring")
    assert settings_key(a, "grating") != settings_key(b, "grating")


# ---- physics --------------------------------------------------------------------------- #
def test_physics_helpers():
    R = physics.radius_for_fsr(200.0, 2.2233)
    assert abs(physics.fsr_ghz(R, 2.2233) - 200.0) < 1e-9
    assert abs(R - 107.30) < 0.01
    # Si 220 nm slab in SiO2 at 1550 nm: TE0 effective index 2.848 (textbook value ~2.85)
    assert abs(physics.slab_te_index(3.476, 1.444, 1.444, 0.22, 1.55) - 2.848) < 2e-3
    # 0.4 dB/cm, n_g 2.2233 at 1550 nm -> Q_i = 2 pi n_g / (lambda alpha) = 9.79e5
    assert abs(physics.q_intrinsic(0.4, 2.2233, 1.55) / 9.785e5 - 1) < 1e-3
    assert physics.loaded_q(1e6, 1.0) == 5e5
    with pytest.raises(ValueError):
        physics.slab_te_index(1.4, 1.444, 1.0, 0.7, 1.55)


def test_over_coupled_gap_is_narrower_and_k1_is_critical():
    theta0, gamma, cross = 1.2, 3.0, 0.01                # a ring_gap_sweep-like fit
    theta_t = math.asin(math.sqrt(cross))
    crit = {"fit_params": {"theta_0": theta0, "gamma": gamma}, "target_cross_power": cross,
            "critical_gap_um": (math.log(theta0) - math.log(theta_t)) / gamma}
    assert abs(physics.gap_for_over_coupling(crit, 1.0) - crit["critical_gap_um"]) < 1e-12
    assert physics.gap_for_over_coupling(crit, 2.0) < crit["critical_gap_um"]


def test_candidates_are_ranked_by_robustness():
    def rec(H, W, D, te):
        return dict(H=H, W=W, D_TE0=D, guided_TE=te, guided_TM=1, ng_TE0=2.2, n_cut=1.444,
                    neff_TE=[1.7, 1.43])
    rows = [rec(0.7, 0.9, 100, 1), rec(0.7, 1.0, 200, 1), rec(0.7, 1.1, 250, 1),
            rec(0.8, 0.9, -50, 1), rec(0.8, 1.0, 300, 1), rec(0.8, 1.1, 300, 2)]
    ranked = _rank_targets(rows, [0.7, 0.8], [0.9, 1.0, 1.1])
    assert {(r["H_um"], r["W_um"]) for r in ranked} == {(0.7, 0.9), (0.7, 1.0), (0.7, 1.1), (0.8, 1.0)}
    assert (ranked[0]["H_um"], ranked[0]["W_um"]) == (0.7, 1.0)   # 3 target neighbours


# ---- cloud stages: priced and confirmed before anything runs -------------------------- #
def _fake_gap_fit(gaps, out, R, qi, volume, **kw):
    theta0, gamma, cross = 1.2, 3.0, 0.01
    theta_t = math.asin(math.sqrt(cross))
    theta = [theta0 * math.exp(-gamma * g) for g in gaps]
    coupled = [math.sin(t) ** 2 for t in theta]
    return {"critical_gap_um": (math.log(theta0) - math.log(theta_t)) / gamma,
            "critical_gap_extrapolated": False, "target_cross_power": cross,
            "fit_params": {"theta_0": theta0, "gamma": gamma},
            "sorted_gaps_um": sorted(gaps), "theta_raw_rad": [theta[gaps.index(g)] for g in sorted(gaps)],
            "fit_mask": [True] * len(gaps), "through_power": [1 - c for c in coupled],
            "through_power_source": [1 - c - 0.002 for c in coupled],
            "reference_through_power": None, "fit_gap_range_um": [min(gaps), max(gaps)]}


def test_gap_stage_prices_every_gap_then_runs(tmp_path, fake_web, monkeypatch):
    import tarang.ring_resonator.ring_gap_sweep as rgs
    from tarang.design import stages
    called = []
    monkeypatch.setattr(rgs, "ring_gap_sweep", lambda *a, **k: called.append(1) or _fake_gap_fit(*a, **k))
    ctx = _ctx(tmp_path, gap__gaps_um=[0.4, 0.6], run__max_total_cost=5.0)
    res = stages.stage_gap(ctx)
    assert fake_web.names().count("upload") == 2 and fake_web.names().count("delete") == 2
    assert "start" not in fake_web.names()                   # priced only; the fit is faked
    assert called and res["over_coupled_gap_um"] < res["critical_gap_um"]
    assert abs(ctx.spent - 0.5) < 1e-9


def test_gap_stage_over_budget_is_skipped_without_running(tmp_path, fake_web, monkeypatch):
    import tarang.ring_resonator.ring_gap_sweep as rgs
    from tarang.design import stages
    monkeypatch.setattr(rgs, "ring_gap_sweep", lambda *a, **k: pytest.fail("must not run"))
    ctx = _ctx(tmp_path, gap__gaps_um=[0.4, 0.6], run__max_total_cost=0.3)
    assert stages.stage_gap(ctx) == {"skipped": True}
    assert "start" not in fake_web.names() and ctx.spent == 0


def _fake_spectra(monkeypatch):
    """Grating spectra without a cloud: a Gaussian peak whose wavelength follows the
    first-order phase matching of each design (period, fill factor), and whose height
    is best at fill factor 0.7 and fibre position 5.2 um."""
    from tarang.grating_coupler.grating_coupler import GratingCoupler
    calls = []

    def spectrum(self, output_dir, verbose=False):
        self._run_sim(output_dir)                     # through the (fake) cloud cache
        c = self.config
        calls.append((c.period, c.ff, c.source_x))
        n_avg = c.ff * 1.85 + (1 - c.ff) * 1.0
        peak_wl = c.period * (n_avg - math.sin(math.radians(abs(c.theta_deg))))
        height = 0.4 * math.exp(-((c.ff - 0.7) / 0.15) ** 2) * math.exp(-((c.source_x - 5.2) / 3) ** 2)
        wl = np.linspace(c.wavelength_um - c.bandwidth_um, c.wavelength_um + c.bandwidth_um, 101)
        return wl, height * np.exp(-((wl - peak_wl) / 0.03) ** 2)
    monkeypatch.setattr(GratingCoupler, "get_spectrum", spectrum)
    return calls


def test_grating_stage_maps_retunes_then_places_the_fibre(tmp_path, fake_web, monkeypatch):
    from tarang.design import stages
    fake_web.cost = 0.01
    _fake_spectra(monkeypatch)
    ctx = _ctx(tmp_path, grating__fill_factors=[0.5, 0.7, 0.9], grating__fiber_angle_deg=12.0,
               run__max_total_cost=5.0)
    res = stages.stage_grating(ctx)
    assert res["etch"] == "full" and res["etch_depth_um"] == 0.7      # full etch = film
    assert res["fiber_tilt"] == "towards"
    assert abs(res["peak_wavelength_um"] - 1.55) < 0.004              # peak moved onto 1550 nm
    assert abs(res["fill_factor"] - 0.7) < 1e-9 and res["source_x_um"] == 5.2
    assert res["efficiency_2d"] > 0.35
    starts = fake_web.names().count("batch_start")
    assert starts == 3                                                 # steps A, B and C
    rows = (ctx.out / "tables" / "grating_sweep.csv").read_text().splitlines()
    assert len(rows) == 1 + res["runs"]
    for name in ("5a_grating_geometry", "5b_grating_map", "5c_grating_sensitivity",
                 "5d_grating_spectrum"):
        assert (ctx.out / "figures" / f"{name}.png").exists(), name


def test_grating_stage_flags_a_best_design_at_the_edge(tmp_path, fake_web, monkeypatch):
    from tarang.design import stages
    fake_web.cost = 0.01
    _fake_spectra(monkeypatch)
    ctx = _ctx(tmp_path, grating__fill_factors=[0.4, 0.5], grating__retune=False,
               run__max_total_cost=5.0)
    res = stages.stage_grating(ctx)
    assert any("edge" in n for n in res["notes"])


def test_grating_stage_skips_features_below_the_lithography_limit(tmp_path, fake_web, monkeypatch):
    from tarang.design import stages
    fake_web.cost = 0.01
    calls = _fake_spectra(monkeypatch)
    ctx = _ctx(tmp_path, grating__fill_factors=[0.5, 0.95], grating__min_feature_um=0.1,
               grating__retune=False, run__max_total_cost=5.0)
    stages.stage_grating(ctx)
    assert all(min(ff, 1 - ff) * p >= 0.1 - 1e-9 for p, ff, _ in calls)


# ---- layout and report ------------------------------------------------------------------ #
def _with_cloud_results(ctx):
    ctx.state["stages"]["gap"] = {"results": dict(critical_gap_um=0.55, over_coupling=1.5,
                                                  over_coupled_gap_um=0.48, extrapolated=False,
                                                  q_loaded_critical=4.9e5, q_loaded_over=3.9e5,
                                                  folder="gap")}
    ctx.state["stages"]["grating"] = {"results": dict(
        etch="full", etch_depth_um=0.7, fiber_angle_deg=10.0, fiber_tilt="towards", mfd_um=10.4,
        n_teeth=20, width_um=12.0, period_um=1.24, fill_factor=0.5, source_x_um=5.2,
        efficiency_2d=0.31, loss_db_2d=5.1, peak_efficiency_2d=0.33, peak_wavelength_um=1.553,
        bandwidth_1db_nm=38.0, bandwidth_3db_nm=70.0, efficiency_3d=None,
        period_estimate_um=1.24, n_slab=1.85, table="t.csv")}
    return ctx


def test_layout_reproduces_the_design_dimensions(tmp_path):
    import gdstk
    from tarang.design import layout
    ctx = _with_cloud_results(_ctx(tmp_path))
    res = layout.export(ctx)
    names = {Path(f).name for f in res["files"]}
    assert {"waveguide.gds", "ring_waveguide.gds", "ring_waveguide_gc.gds", "design.gds",
            "parameters.json", "parameters.m", "parameters.csv"} <= names
    lib = gdstk.read_gds(str(ctx.out / "layout" / "ring_waveguide.gds"))
    cell = next(c for c in lib.cells if c.name == "ring_waveguide_gap550nm")
    ring = [q for q in cell.polygons if q.bounding_box()[0][1] > 0.5]   # GDS splits big polygons
    (x0, y0), (x1, y1) = gdstk.Polygon([pt for q in ring for pt in q.points]).bounding_box()
    assert abs((x1 - x0) / 2 - (107.5 + 0.5)) < 1e-3          # outer radius R + W/2
    assert abs(y0 - (0.5 + 0.55)) < 1e-3                      # gap above the bus edge
    gc = next(c for c in gdstk.read_gds(str(ctx.out / "layout" / "ring_waveguide_gc.gds")).cells
              if c.name == "ring_waveguide_gc_gap550nm")
    teeth = [p for p in gc.polygons if abs(p.area() - 1.24 * 0.5 * 12.0) < 1e-6]
    assert len(teeth) == 2 * 20                               # both couplers
    params = json.loads((ctx.out / "layout" / "parameters.json").read_text())
    assert params["grating_tooth_length_um"] == 0.62 and params["gap_critical_um"] == 0.55
    assert "p.ring_radius_um = 107.5;" in (ctx.out / "layout" / "parameters.m").read_text()


def test_trench_tone_draws_the_etched_regions(tmp_path):
    import gdstk
    from tarang.design import layout
    ctx = _with_cloud_results(_ctx(tmp_path, layout__tone="trench", layout__trench_width_um=2.0))
    layout.export(ctx)
    cell = next(c for c in gdstk.read_gds(str(ctx.out / "layout" / "waveguide.gds")).cells)
    (x0, y0), (x1, y1) = gdstk.Polygon([pt for p in cell.polygons for pt in p.points]).bounding_box()
    assert abs(y1 - (0.5 + 2.0)) < 1e-6                      # trench reaches 2 um beyond the core
    # (L + 2t)(W + 2t) - L W: the trench surrounds the 400 um x 1 um bus, ends included
    assert abs(sum(p.area() for p in cell.polygons) - (404.0 * 5.0 - 400.0)) < 1e-3


def test_report_lists_values_to_check_elsewhere(tmp_path):
    from tarang.design import report
    ctx = _with_cloud_results(_ctx(tmp_path))
    res = report.write(ctx)
    text = (ctx.out / "report.md").read_text(encoding="utf-8")
    for heading in ("## 2. Cross-section", "## 3. Ring", "## 4. Ring–bus gap", "## 5. Grating coupler"):
        assert heading in text
    checks = (ctx.out / "tables" / "check_values.csv").read_text(encoding="utf-8")
    assert "critical gap" in checks and "n_g TE00" in checks and "Lumerical" not in checks
    assert json.loads((ctx.out / "design_summary.json").read_text())["ring"]["radius_um"] == 107.5
    assert res["report"].endswith("report.md")


def test_command_line_lists_the_stages():
    out = subprocess.run([sys.executable, "-m", "tarang.design", "--list"],
                         capture_output=True, text=True, check=True).stdout
    assert out.splitlines()[0].startswith("1. materials") and "grating" in out
