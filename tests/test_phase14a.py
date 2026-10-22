"""Phase 14A: design goal (comb ring or test waveguides), recommendations, cross-section
choice, reference waveguides, gap variants, cutback spirals, coupler choice on the mask,
reflection of the kept apodised grating."""
from __future__ import annotations

import csv
import io
import json
import math

import numpy as np
import pytest

from test_phase12b import _ctx


# ---- goal and stages ----------------------------------------------------------------- #
def test_each_goal_runs_its_own_stages_and_old_keys_stay(tmp_path):
    from tarang.design.config import DesignConfig
    from tarang.design.stages import STAGE_NAMES, applies, settings_key
    cfg = DesignConfig()
    assert applies("ring", cfg) and applies("gap", cfg) and not applies("waveguides", cfg)
    others = [s for s in STAGE_NAMES if s != "waveguides"]
    keys = {s: settings_key(cfg, s) for s in others}
    cfg.waveguides.lengths_cm = [0, 3]                 # settings of the other goal: no effect
    cfg.waveguides.spiral_gap_um = 6.0
    assert {s: settings_key(cfg, s) for s in others} == keys
    cfg.run.goal = "loss_test"
    assert not applies("ring", cfg) and not applies("gap", cfg) and applies("waveguides", cfg)
    assert applies("grating", cfg) and applies("layout", cfg) and applies("device", cfg)
    assert settings_key(cfg, "layout") != keys["layout"]
    assert STAGE_NAMES.index("waveguides") == STAGE_NAMES.index("ring") + 1


def test_unknown_goal_is_refused(tmp_path):
    from tarang.design import run
    from tarang.design.config import DesignConfig
    from tarang.design.ui import Prompter
    cfg = DesignConfig()
    cfg.run.goal = "laser"
    cfg.given.add("run.goal")
    with pytest.raises(ValueError):
        run(cfg, tmp_path / "r", interactive=False, only=["report"],
            prompter=Prompter(False, out=io.StringIO()))


def test_checks_ignore_results_of_the_other_goal(tmp_path):
    from tarang.design import checks
    ctx = _ctx(tmp_path, run__goal="loss_test")
    ctx.state["stages"]["gap"] = {"results": dict(critical_gap_um=0.47)}
    assert checks._done(ctx, "gap") is None and checks._done(ctx, "ring") is None
    assert checks._done(ctx, "grating") is not None
    rows = checks.loss_budget(ctx, ctx.results("grating"))
    assert rows[-1]["item"] == "fibre to fibre, off resonance"


def test_run_stops_when_a_choice_is_awaited(tmp_path, monkeypatch):
    import tarang.design as design
    from tarang.design import run, stages
    from tarang.design.ui import Prompter
    new = list(stages.STAGES)
    i = stages.STAGE_NAMES.index("cross_section")
    name, title, kind, _ = new[i]
    new[i] = (name, title, kind, lambda ctx: dict(
        skipped=True, awaiting_choice=True, candidates=[dict(H_um=0.7, W_um=1.0)]))
    ran = []
    j = stages.STAGE_NAMES.index("ring")
    new[j] = ("ring", "Ring", "local", lambda ctx: ran.append(1) or {})
    monkeypatch.setattr(design, "STAGES", new)
    out = io.StringIO()
    ctx = run(_ctx(tmp_path).cfg, tmp_path / "r", interactive=False, only=["cross_section", "ring"],
              prompter=Prompter(False, out=out))
    assert ctx.state["stages"]["cross_section"]["results"]["awaiting_choice"]
    assert not ran and "Stopped for your choice" in out.getvalue()


def test_single_mode_ranking_for_test_waveguides():
    from tarang.design.stages import _rank_targets
    rows = [dict(H=0.7, W=w, D_TE0=d, ng_TE0=2.2, guided_TE=1, guided_TM=1, neff_TE=[1.7],
                 neff_TM=[1.68]) for w, d in ((0.8, -50.0), (0.9, -10.0), (1.0, 20.0))]
    assert [r["W_um"] for r in _rank_targets(rows, [0.7], [0.8, 0.9, 1.0])] == [1.0]
    assert len(_rank_targets(rows, [0.7], [0.8, 0.9, 1.0], anomalous=False)) == 3


# ---- test waveguides: first principles -------------------------------------------------- #
def test_cutback_precision_is_the_least_squares_slope_error():
    from tarang.design import waveguides as wg
    L = [0, 1, 2, 4]
    assert wg.slope_sigma(L, 0.3) == pytest.approx(0.3 / math.sqrt(8.75))
    rec, sig, k = wg.recommend_lengths(L, 0.3, 0.4, 0.1)
    assert rec == [0, 3, 5.5, 10.5] and sig <= 0.04 + 1e-9 and k > 1    # rounded up
    same, _, k1 = wg.recommend_lengths([0, 5, 10, 20], 0.3, 0.4, 0.1)
    assert same == [0, 5, 10, 20] and k1 == 1.0                      # never scaled down
    rows = [dict(radius_um=r, bend_loss_db_cm=b) for r, b in
            ((20, 5.0), (40, 0.03), (60, 0.001), (100, 1e-5))]
    assert wg.choose_radius(rows, 0.4) == 40
    rows[2]["bend_loss_db_cm"] = 0.5                                 # a glitch at 60 µm
    assert wg.choose_radius(rows, 0.4) == 100


def test_waveguides_stage_recommends_radius_and_lengths(tmp_path, monkeypatch):
    from tarang.design import waveguides as wg
    ctx = _ctx(tmp_path, run__goal="loss_test")
    monkeypatch.setattr(wg, "bend_loss_db_cm", lambda ctx, H, W, R: 50.0 * math.exp(-R / 8.0))
    res = wg.stage_waveguides(ctx)
    assert res["alpha_db_cm"] == pytest.approx(0.4)
    expect = min(r for r in wg.default_radii(1.55) if 50 * math.exp(-r / 8) <= 0.04)
    assert res["bend_radius_recommended_um"] == expect == res["bend_radius_um"]
    assert res["lengths_recommended_cm"] == [0, 3, 5.5, 10.5]
    assert res["spirals"] and all(s["inner_radius_um"] >= 2 * expect - 1e-6 for s in res["spirals"])
    assert (ctx.out / "figures" / "3w_bend_loss.png").exists()
    assert (ctx.out / "tables" / "bend_loss.csv").exists()
    assert res["recommend"]["recommended"]["bend_radius_um"] == expect


# ---- spirals ------------------------------------------------------------------------- #
@pytest.mark.parametrize("length_um", [5e3, 2e4, 4e4])
def test_spiral_has_the_length_radius_and_no_crossings(length_um):
    from tarang.design.layout import spiral_path
    R, pitch = 60.0, 5.0
    pts, info = spiral_path(length_um, 2 * R, pitch)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    assert seg.sum() == pytest.approx(length_um, rel=1e-6)
    assert info["min_bend_radius_um"] >= R - 1e-6
    d = np.diff(pts, axis=0)
    assert d[0][0] > 0 and abs(d[0][1]) < 1e-9 and d[-1][0] > 0 and abs(d[-1][1]) < 1e-9
    # curvature radius of every bit of the path >= R (three-point circle, 2 µm apart)
    s = np.concatenate([[0], np.cumsum(seg)])
    u = np.interp(np.arange(0, s[-1], 2.0), s, pts[:, 0]), np.interp(np.arange(0, s[-1], 2.0), s, pts[:, 1])
    q = np.stack(u, axis=1)
    a, b, c = q[:-2], q[1:-1], q[2:]
    area2 = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    sides = (np.hypot(*(b - a).T) * np.hypot(*(c - b).T) * np.hypot(*(c - a).T))
    rc = sides / np.maximum(2 * area2, 1e-12)
    assert rc.min() >= R * 0.98
    # no two parts of the path closer than the pitch (except neighbours along the path)
    sub = q[::3]
    D = np.hypot(sub[:, None, 0] - sub[None, :, 0], sub[:, None, 1] - sub[None, :, 1])
    idx = np.arange(len(sub))
    far = np.abs(idx[:, None] - idx[None, :]) > 2 * pitch / 6.0 + 30
    assert D[far].min() >= pitch * 0.95


# ---- couplers on the mask --------------------------------------------------------------- #
def _with_couplers(ctx, apodised=True, focus=True):
    st = ctx.state["stages"]
    st["grating"]["results"].update(fiber_tilt="towards", etch_depth_um=0.7)
    if apodised:
        st["apodise"] = {"results": dict(verified=True, better_than_uniform=False, efficiency_2d=0.299,
                                         source_x_um=6.36, teeth=[[i * 1.05, 0.8] for i in range(18)])}
    if focus:
        ctx.cfg.focus.run = True
        st["focus"] = {"results": dict(curves=[[1, 2, 1.6]], teeth_kind="uniform", teeth=[[0, 0.8]],
                                       distance_um=8.0, tip_width_um=2.0, tip_taper_um=4.0,
                                       half_angle_rad=0.7, y_max_um=8.0, s=0.2,
                                       eta_focus=0.997, tip_taper_T=0.996, eta_lateral=0.993)}
    return ctx


@pytest.mark.parametrize("setting,kinds,main", [
    ("auto", [("gc", "uniform"), ("fgc", "focusing")], "focusing"),
    ("uniform", [("gc", "uniform")], "uniform"),
    ("apodised", [("gc", "apodised")], "apodised"),
    ("focusing", [("fgc", "focusing")], "focusing"),
    ("all", [("gc", "uniform"), ("agc", "apodised"), ("fgc", "focusing")], "focusing")])
def test_coupler_setting_decides_what_is_drawn(tmp_path, setting, kinds, main):
    from tarang.design.layout import coupler_plan
    ctx = _with_couplers(_ctx(tmp_path, layout__coupler=setting))
    plan = coupler_plan(ctx)
    assert plan["kinds"] == kinds and plan["main"] == main and not plan["notes"]


def test_missing_coupler_kinds_fall_back_with_a_note(tmp_path):
    from tarang.design.layout import coupler_plan
    ctx = _with_couplers(_ctx(tmp_path, layout__coupler="apodised"), apodised=False, focus=False)
    plan = coupler_plan(ctx)
    assert plan["kinds"] == [("gc", "uniform")] and "apodised" in plan["notes"][0]
    ctx2 = _with_couplers(_ctx(tmp_path / "b", layout__coupler="focusing"), apodised=False, focus=False)
    assert "focusing" in coupler_plan(ctx2)["notes"][0]
    ctx3 = _ctx(tmp_path / "c", layout__coupler="sideways")
    assert "auto" in coupler_plan(ctx3)["notes"][0]


def test_ring_mask_has_reference_waveguides_and_gap_variants(tmp_path):
    gdstk = pytest.importorskip("gdstk")
    from tarang.design import layout
    ctx = _with_couplers(_ctx(tmp_path, layout__coupler="all", layout__gap_offsets_um=[-0.05, 0.05]))
    ctx.state["stages"]["gap"] = {"results": dict(critical_gap_um=0.47, over_coupled_gap_um=0.431,
                                                  over_coupling=1.5)}
    p = layout.device_parameters(ctx)
    assert p["gap_variants_um"] == [0.381, 0.42, 0.431, 0.47, 0.481, 0.52]
    files = {f.name for f in layout.build_gds(p, tmp_path)}
    for suf in ("gc", "agc", "fgc"):
        assert f"ring_waveguide_{suf}.gds" in files and f"waveguide_{suf}.gds" in files
    lib = gdstk.read_gds(str(tmp_path / "ring_waveguide_agc.gds"))
    assert len(lib.cells) == 6
    agc = lib.cells[0].get_polygons()
    ugc = gdstk.read_gds(str(tmp_path / "ring_waveguide_gc.gds")).cells[0].get_polygons()
    assert len(ugc) - len(agc) == 2 * (30 - 18)                    # 18 apodised, 30 uniform teeth
    top = gdstk.read_gds(str(tmp_path / "design.gds")).top_level()
    assert len(top) == 1 and len(top[0].references) == 1 + 6 + 3 * 6 + 3
    ctx.cfg.layout.reference = False
    files2 = {f.name for f in layout.build_gds(layout.device_parameters(ctx), tmp_path / "x")}
    assert "waveguide_gc.gds" not in files2


def test_loss_test_mask_has_spirals_with_couplers(tmp_path, monkeypatch):
    gdstk = pytest.importorskip("gdstk")
    from tarang.design import layout, waveguides as wg
    ctx = _ctx(tmp_path, run__goal="loss_test", waveguides__lengths_cm=[0, 0.5, 1])
    ctx.state["stages"]["grating"]["results"].update(fiber_tilt="towards", etch_depth_um=0.7)
    ctx.state["stages"]["gap"] = {"results": dict(critical_gap_um=0.47)}     # ignored
    monkeypatch.setattr(wg, "bend_loss_db_cm", lambda ctx, H, W, R: 50.0 * math.exp(-R / 8.0))
    ctx.state["stages"]["waveguides"] = {"results": wg.stage_waveguides(ctx)}
    p = layout.device_parameters(ctx)
    assert p["ring_radius_um"] is None and p["gap_variants_um"] == []
    out = layout.export(ctx)
    names = {f.split("/")[-1].split("\\")[-1] for f in out["files"]}
    assert {"waveguide_gc.gds", "spiral_cutback_gc.gds", "spirals.csv"} <= names
    assert not any(n.startswith("ring_") for n in names)
    lib = gdstk.read_gds(str(ctx.out / "layout" / "spiral_cutback_gc.gds"))
    assert sorted(c.name for c in lib.cells) == ["spiral_0p5cm_gc", "spiral_1cm_gc"]
    rows = list(csv.DictReader(open(ctx.out / "layout" / "spirals.csv", encoding="utf-8")))
    assert [round(float(r["length_um"])) for r in rows] == [5000, 10000]
    assert (ctx.out / "figures" / "6_layout.png").exists()


def test_parameters_files_hold_the_new_lists(tmp_path):
    from tarang.design import layout
    ctx = _with_couplers(_ctx(tmp_path, layout__coupler="all"))
    files = layout._write_parameters(layout.device_parameters(ctx), tmp_path)
    m = (tmp_path / "parameters.m").read_text()
    assert "p.couplers = 'gc:uniform agc:apodised fgc:focusing';" in m
    assert "p.cutback_lengths_cm = [];" in m
    assert json.loads((tmp_path / "parameters.json").read_text())["main_coupler"] == "focusing"
    assert len(files) == 4                                           # + grating_teeth.csv


# ---- reflection of the grating used ------------------------------------------------------ #
def test_circuit_uses_the_apodised_gratings_own_reflection(tmp_path):
    from tarang.design import circuit
    ctx = _with_couplers(_ctx(tmp_path))
    ctx.state["stages"]["checks"] = {"results": dict(grating=dict(power_budget=dict(
        wavelength_um=[1.5, 1.6], reflected=[0.12, 0.12])))}
    b, src = circuit.coupler_reflection(ctx)
    assert b["reflected"] == [0.12, 0.12] and "uniform" in src
    ctx.state["stages"]["focus"]["results"]["teeth_kind"] = "apodised"
    b, src = circuit.coupler_reflection(ctx)
    assert "not simulated" in src
    ctx.state["stages"]["apodise"]["results"]["power_budget"] = dict(wavelength_um=[1.5, 1.6],
                                                                    reflected=[0.02, 0.02])
    b, src = circuit.coupler_reflection(ctx)
    assert b["reflected"] == [0.02, 0.02] and "apodised" in src


def test_reverse_run_of_explicit_teeth_covers_the_whole_grating(tmp_path):
    from tarang.design.checks import _reverse_sim
    from tarang.design.stages import grating_device
    ctx = _ctx(tmp_path)
    teeth = [(i * 1.05, 0.8) for i in range(18)]
    dev = grating_device(ctx, 1.1001, 0.75, 6.36, H=0.7, depth=0.7, teeth=teeth)
    rev = _reverse_sim(dev, ctx)
    past = next(m for m in rev.monitors if m.name == "past")
    assert past.center[0] >= teeth[-1][0] + teeth[-1][1] + 1.0 - 1e-9 or \
        past.center[0] == pytest.approx(rev.bounds[1][0] - 0.5)


def test_apodise_stage_measures_the_kept_gratings_reflection(tmp_path, fake_web, monkeypatch):
    from test_phase12b import _apod_ctx, _line_field
    from tarang.design import apodise as ap, checks
    from tarang.grating_coupler.grating_coupler import GratingCoupler

    def fake_line(data, f0):
        teeth = [s for s in data.simulation.structures if s.name and s.name.startswith("Tooth_")]
        b = teeth[0].geometry.bounds
        P = teeth[1].geometry.bounds[0][0] - b[0][0]
        ff = (b[1][0] - b[0][0]) / P
        return _line_field(0.02 + 0.6 * (0.9 - ff), 1.55 / P + math.sin(math.radians(12)), P)
    monkeypatch.setattr(ap, "read_line", fake_line)
    monkeypatch.setattr(GratingCoupler, "get_spectrum", lambda self, folder: (
        np.linspace(1.45, 1.65, 101),
        (0.40 if self.config.teeth else 0.313) * np.exp(-((np.linspace(1.45, 1.65, 101) - 1.55) / 0.03) ** 2)))
    seen = {}

    def budget(data, rev, spectra, wl0):
        seen["teeth"] = any((s.name or "").startswith("Tooth_") for s in rev.structures)
        seen["source"] = type(rev.sources[0]).__name__
        return dict(wavelength_um=[1.5, 1.6], reflected=[0.03, 0.03],
                    at_target=dict(reflection_db=-15.2, up=0.5, down=0.3))
    monkeypatch.setattr(checks, "_power_budget", budget)
    ctx = _apod_ctx(tmp_path)
    res = ap.stage_apodise(ctx)
    assert res["power_budget"]["reflected"] == [0.03, 0.03]
    assert seen == {"teeth": True, "source": "ModeSource"}
    assert "-15.2 dB" in ctx.ui.out.getvalue()
    assert res["recommend"]["recommended"]["recipe"] == "measured scattering"
    ctx2 = _apod_ctx(tmp_path / "off")
    ctx2.cfg.apodise.reflection_check = False
    assert "power_budget" not in ap.stage_apodise(ctx2)


# ---- expected cutback and the report ------------------------------------------------------ #
def _loss_ctx(tmp_path, monkeypatch):
    from tarang.design import waveguides as wg
    ctx = _ctx(tmp_path, run__goal="loss_test")
    ctx.state["stages"]["grating"]["results"].update(fiber_tilt="towards", etch_depth_um=0.7)
    t = ctx.out / "tables"
    t.mkdir(parents=True, exist_ok=True)
    with open(t / "grating_spectrum_final.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["wavelength_um", "efficiency", "efficiency_db"])
        for x in np.linspace(1.45, 1.65, 41):
            w.writerow([x, 0.313 * math.exp(-((x - 1.55) / 0.06) ** 2), 0])
    ctx.state["stages"]["checks"] = {"results": dict(grating=dict(power_budget=dict(
        wavelength_um=[1.45, 1.65], reflected=[0.12, 0.12])))}
    monkeypatch.setattr(wg, "bend_loss_db_cm", lambda ctx, H, W, R: 50.0 * math.exp(-R / 8.0))
    ctx.state["stages"]["waveguides"] = {"results": wg.stage_waveguides(ctx)}
    return ctx


def test_expected_cutback_follows_the_loss_and_the_couplers(tmp_path, monkeypatch):
    from tarang.design import circuit
    ctx = _loss_ctx(tmp_path, monkeypatch)
    r = circuit.stage_device(ctx)
    rows = r["rows"]
    assert [x["structure"] for x in rows] == ["reference", "spiral 1 cm", "spiral 2 cm", "spiral 4 cm"]
    eta = 0.313 * 10 ** (-0.0)                   # no taper result in this context
    extra = 2 * ctx.cfg.grating.taper_length_um * 1e-4
    for x in rows:
        assert x["transmission_db"] == pytest.approx(20 * math.log10(eta) - 0.4 * x["length_cm"], abs=2e-3)
    assert rows[1]["length_cm"] == pytest.approx(1 + extra)
    slope = np.polyfit([x["length_cm"] for x in rows], [x["transmission_db"] for x in rows], 1)[0]
    assert slope == pytest.approx(-0.4, rel=1e-6)
    p = 0.12 * 10 ** (-0.4 * rows[0]["length_cm"] / 10)
    assert rows[0]["ripple_pp_db"] == pytest.approx(20 * math.log10((1 + p) / (1 - p)), abs=2e-3)
    assert rows[1]["ripple_period_pm"] < rows[0]["ripple_period_pm"]
    assert (ctx.out / "figures" / "11a_cutback.png").exists()
    assert (ctx.out / "tables" / "device_cutback.csv").exists()


def test_report_has_the_test_waveguide_sections(tmp_path, monkeypatch):
    from tarang.design import circuit, report
    ctx = _loss_ctx(tmp_path, monkeypatch)
    dv = circuit.stage_device(ctx)
    wg = ctx.results("waveguides")
    text = "\n".join(report._waveguides_section(ctx, wg) + report._device_section(ctx, dv))
    assert "## 3w. Test waveguides" in text and "## 7e. Expected cutback" in text
    assert "3w_bend_loss" in text and "11a_cutback" in text


@pytest.mark.slow
def test_bend_loss_follows_the_waveguide_mode_not_a_substrate_mode(tmp_path):
    """Real local solves: the tracked TE00 loss falls steeply with radius and reaches the
    solver's floor by 20 µm; the highest-index solution at 15 µm (a substrate mode) would
    have given thousands of dB/cm."""
    from tarang.design import waveguides as wg
    ctx = _ctx(tmp_path)
    b5, b10, b20 = (wg.bend_loss_db_cm(ctx, 0.7, 1.0, R) for R in (5.0, 10.0, 20.0))
    assert 5 < b5 < 500 and 0.01 < b10 < b5 / 10
    assert b20 < wg.FLOOR_DB_CM
