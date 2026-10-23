"""Phase 12C: focusing grating, efficiency landscapes, layout and budget with the focusing
grating, benchmark-pack sections."""
from __future__ import annotations

import csv
import io
import math

import numpy as np
import pytest

from test_phase12b import _ctx

N_SLAB, N_SIDE, WL = 1.8511, 1.444, 1.55


def _focus_ctx(tmp_path, **kw):
    ctx = _ctx(tmp_path, focus__run=True, **kw)
    ctx.state["stages"]["grating"]["results"].update(fiber_tilt="towards")
    ctx.state["stages"]["taper"] = {"results": dict(eta_y=0.99, taper_loss_db=0.048,
                                                    taper_length_um=290.0, grating_width_um=14.5,
                                                    n_slab=N_SLAB, length_source="3D EME")}
    return ctx


# ---- in-plane focusing model ------------------------------------------------------- #
def test_focusing_peaks_where_the_light_cone_matches_the_tip():
    from tarang.design import focus
    r = [4, 6, 8, 10, 12, 15, 20, 30]
    narrow = [focus.focus_efficiency(N_SLAB, N_SIDE, 1.0, WL, rc, 5.2) for rc in r]
    wide = [focus.focus_efficiency(N_SLAB, N_SIDE, 3.0, WL, rc, 5.2) for rc in r]
    assert all(0 < v <= 1 for v in narrow + wide)
    assert r[int(np.argmax(narrow))] in (8, 10, 12) and max(narrow) > 0.97
    assert r[int(np.argmax(wide))] > r[int(np.argmax(narrow))]   # wider tip: longer focus
    assert max(wide) > 0.97 and narrow[-1] < 0.7                 # too far: spot too large


def test_tip_taper_is_free_for_the_waveguide_itself_and_short_otherwise():
    from tarang.design import focus
    assert focus.tip_taper(N_SLAB, N_SIDE, 1.0, 1.0, WL, 0.05) == (0.0, 1.0)
    L, T = focus.tip_taper(N_SLAB, N_SIDE, 2.0, 1.0, WL, 0.05)
    assert 2 <= L <= 10 and T > 10 ** (-0.005)


# ---- geometry ------------------------------------------------------------------------ #
def test_teeth_are_phase_matched_ellipses_around_the_tip():
    from tarang.design import focus
    P, ff, s = 1.1, 0.75, math.sin(math.radians(12))
    teeth = [(i * P, ff * P) for i in range(30)]
    curves = focus.teeth_curves(teeth, 8.0, WL, s)
    for i, (Ps, Pe, nb) in enumerate(curves):
        assert nb == pytest.approx(WL / P + s)
        assert focus.ellipse_r(0.0, Ps, nb, s) == pytest.approx(8.0 + i * P)        # on the axis
        assert focus.ellipse_r(0.0, Pe, nb, s) == pytest.approx(8.0 + i * P + ff * P)
        phi = np.linspace(-0.6, 0.6, 7)
        r = focus.ellipse_r(phi, Ps, nb, s)
        assert np.allclose(nb * r - s * r * np.cos(phi), Ps)                         # phase law
    g = focus.polygons(teeth, 8.0, 2.0, WL, s, math.radians(40), 8.3)
    assert len(g["teeth"]) == 30
    pts = np.array([p for t in g["teeth"] for p in t])
    assert np.abs(pts[:, 1]).max() <= 8.3 + 1e-6
    ang = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    assert np.abs(ang).max() <= 40 + 1e-6
    assert g["slab"][0] == (0.0, -1.0) and g["slab"][-1] == (0.0, 1.0)


# ---- the stage ----------------------------------------------------------------------- #
def test_focusing_is_opt_in_and_free(tmp_path, fake_web):
    from tarang.design import focus
    ctx = _ctx(tmp_path)
    n = fake_web.names().count("batch_start")
    assert focus.stage_focus(ctx) == {"disabled": True}
    assert fake_web.names().count("batch_start") == n
    assert "not requested" in ctx.ui.out.getvalue()


def test_focusing_stage_picks_an_allowed_design_and_writes_its_figures(tmp_path, fake_web):
    from tarang.design import focus
    ctx = _focus_ctx(tmp_path)
    n = fake_web.names().count("batch_start")
    res = focus.stage_focus(ctx)
    assert fake_web.names().count("batch_start") == n                 # model only, no cloud
    assert res["teeth_kind"] == "uniform" and len(res["curves"]) == 30
    assert math.degrees(5.2 / res["r_c_um"]) <= 20 + 1e-9
    assert res["eta_focus"] > 0.95
    assert res["efficiency_3d_estimate"] == pytest.approx(0.313 * res["eta_focus"] * res["tip_taper_T"])
    assert res["focusing_length_um"] < 0.3 * res["straight_length_um"]
    for f in ("10a_focusing_landscape.png", "10b_focusing_geometry.png"):
        assert (ctx.out / "figures" / f).exists()
    rows = list(csv.DictReader(open(ctx.out / "tables" / "focus_landscape.csv", encoding="utf-8")))
    assert len(rows) == 5 * 41 and {"allowed", "eta_focus", "eta_3d_estimate"} <= set(rows[0])


def test_apodised_teeth_are_curved_when_they_are_better(tmp_path, fake_web):
    from tarang.design import focus
    ctx = _focus_ctx(tmp_path)
    teeth = [[round(i * 1.08, 4), round((0.9 - 0.004 * i) * 1.08, 4)] for i in range(22)]
    ctx.state["stages"]["apodise"] = {"results": dict(verified=True, better_than_uniform=True,
                                                      teeth=teeth, efficiency_2d=0.34,
                                                      source_x_um=9.0)}
    res = focus.stage_focus(ctx)
    assert res["teeth_kind"] == "apodised" and res["efficiency_2d"] == 0.34
    assert len(res["curves"]) == 22 and res["source_x_um"] == 9.0
    ctx.cfg.focus.teeth = "uniform"
    assert focus.stage_focus(ctx)["teeth_kind"] == "uniform"


def test_focusing_3d_check_is_a_valid_priced_simulation(tmp_path, fake_web, monkeypatch):
    import tidy3d as td
    from tarang.design import focus
    from tarang.grating_coupler.grating_coupler import GratingCoupler
    ctx = _focus_ctx(tmp_path, focus__check_3d=True, run__max_total_cost=10.0)
    monkeypatch.setattr(GratingCoupler, "get_spectrum", lambda self, folder: (
        np.linspace(1.45, 1.65, 21), 0.29 * np.exp(-((np.linspace(1.45, 1.65, 21) - 1.552) / 0.04) ** 2)))
    res = focus.stage_focus(ctx)
    dev = focus.focusing_sim(ctx, res)
    assert isinstance(dev.sim, td.Simulation) and dev._readable_name().startswith("gc3D_focus30_")
    dev.sim.validate_pre_upload()
    names = {s.name for s in dev.sim.structures}
    assert {"Waveguide", "Focusing_slab", "Tooth_0", "Tooth_29", "BOX"} <= names
    assert dev.sim.num_cells < 6e7                                    # kept affordable
    assert res["efficiency_3d"] == pytest.approx(0.29 * math.exp(-(0.002 / 0.04) ** 2), rel=1e-6)


# ---- layout, budget, report ---------------------------------------------------------- #
def test_layout_and_budget_use_the_focusing_grating(tmp_path, fake_web):
    pytest.importorskip("gdstk")
    import gdstk
    from tarang.design import checks, focus, layout, report
    ctx = _focus_ctx(tmp_path)
    res = focus.stage_focus(ctx)
    ctx.state["stages"]["focus"] = {"results": res}
    p = layout.device_parameters(ctx)
    assert p["grating_focusing"] and p["focus_distance_um"] == res["distance_um"]
    files = layout._write_parameters(p, tmp_path) + layout.build_gds(p, tmp_path)
    lib = gdstk.read_gds(str(next(f for f in files if f.name == "waveguide_fgc.gds")))
    n_poly = len(lib.top_level()[0].get_polygons())
    n_tip = 1 if res["tip_taper_um"] > 0 else 0
    assert n_poly == 1 + 2 * (n_tip + 1 + 30)                        # bus, 2 x (tip, slab, teeth)
    # the straight coupler is drawn as well: bus, 2 x (taper + 30 teeth)
    lib = gdstk.read_gds(str(next(f for f in files if f.name == "waveguide_gc.gds")))
    assert len(lib.top_level()[0].get_polygons()) == 1 + 2 * (1 + 30)
    top = gdstk.read_gds(str(next(f for f in files if f.name == "design.gds"))).top_level()
    assert {"waveguide_gc", "waveguide_fgc"} <= {c.name for c in top[0].dependencies(True)}
    assert "p.focus_tip_width_um" in (tmp_path / "parameters.m").read_text()
    items = {r["item"]: r["loss_db"] for r in checks.loss_budget(ctx, ctx.results("grating"))}
    assert any("focusing into" in k for k in items) and not any("lateral overlap" in k for k in items)
    assert not any(k.startswith("tapers, 2 x 290") for k in items)
    text = "\n".join(report._focus_section(ctx, res))
    assert "## 7d. Focusing grating" in text and "10a_focusing_landscape" in text
    rows = {r["quantity"] for r in report._check_values_12(dict(focus=res, taper=ctx.results("taper")))}
    assert "focusing grating: 3D estimate" in rows and "grating width" in rows
    from tarang.design import benchmark
    st = {k: v["results"] for k, v in ctx.state["stages"].items() if k in ("taper", "focus")}
    readme = "\n".join(benchmark._readme(ctx, st, [], [], [], (None, None)))
    assert "## 4b. Grating width and taper" in readme and "## 4d. Focusing grating" in readme
    assert "a focusing grating (section 4d)" in readme
    # turned off again: back to the straight grating and its taper
    ctx.cfg.focus.use_in_layout = False
    p2 = layout.device_parameters(ctx)
    assert not p2["grating_focusing"] and p2["focus_available"]       # still drawn as *_fgc


# ---- landscapes ---------------------------------------------------------------------- #
def test_uniform_landscape_takes_the_best_fibre_position_per_design(tmp_path):
    from tarang.design import landscapes
    rows = [dict(dim="2D", period_um=1.1, fill_factor=0.75, efficiency_at_target=e,
                 peak_efficiency=e) for e in (0.2, 0.31, 0.25)]
    rows += [dict(dim="2D", period_um=p, fill_factor=f, efficiency_at_target=0.1, peak_efficiency=0.12)
             for p in (1.05, 1.15) for f in (0.7, 0.8)]
    rows.append(dict(dim="3D", period_um=1.1, fill_factor=0.75, efficiency_at_target=0.9,
                     peak_efficiency=0.9))                           # 3D runs are not mixed in
    pts = landscapes.uniform_points(rows)
    assert len(pts) == 5
    assert max(p["efficiency_at_target"] for p in pts) == 0.31
    out = landscapes.draw_uniform(tmp_path / "5g.png", rows, dict(fill_factor=0.75, period_um=1.1), 1.55)
    assert out and out.exists()
    assert landscapes.draw_uniform(tmp_path / "x.png", rows[:2], None, 1.55) is None
    ap = [dict(scale=s_, source_x_um=x, efficiency_at_target=0.3, peak_efficiency=0.32)
          for s_, x in ((0.98, 8.0), (1.0, 8.2), (1.02, 8.4), (1.0, 7.2), (1.0, 9.2))]
    assert landscapes.draw_apodised(tmp_path / "9c.png", ap, (8.2, 1.0), 1.55).exists()


def test_old_runs_get_the_uniform_landscape_from_their_sweep_table(tmp_path):
    from tarang.design import checks
    ctx = _ctx(tmp_path)
    ctx.state["figure_version"] = 2                                 # made before Phase 12C
    t = ctx.out / "tables"
    t.mkdir(parents=True, exist_ok=True)
    with open(t / "grating_sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["step", "dim", "period_um", "fill_factor", "source_x_um", "efficiency_at_target",
                    "peak_efficiency", "peak_wavelength_um", "peak_inside"])
        for p in (1.05, 1.1, 1.15):
            for ff in (0.7, 0.75, 0.8):
                w.writerow(["A", "2D", p, ff, 5.2, 0.2, 0.25, 1.55, True])
    done = checks.redraw_saved_figures(ctx)
    assert "5g_grating_landscape" in done
    assert (ctx.out / "figures" / "5g_grating_landscape.png").exists()


def test_focusing_stage_runs_through_the_design_session(tmp_path, fake_web):
    from tarang.design import run
    from tarang.design.ui import Prompter
    ctx = _focus_ctx(tmp_path)
    ctx.save_state()
    out = io.StringIO()
    done = run(ctx.cfg, ctx.out, interactive=False, only=["focus"], prompter=Prompter(False, out=out))
    assert "[10/14] Focusing grating" in out.getvalue()
    assert done.state["stages"]["focus"]["results"]["curves"]
