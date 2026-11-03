"""v0.1.5: a price-only run keeps a stage's earlier results and does not run the stages
after an unfinished one; centring the grating's peak may not cost efficiency at the design
wavelength; the grating mesh checks can use their own boundary (absorber); the sidewall
coupling's decay rate comes from gaps spread over 0.2 um."""
from __future__ import annotations

import io
import math
from types import SimpleNamespace

import numpy as np
import pytest


# ------------------------------------------------------------- price only
@pytest.fixture
def two_stages(monkeypatch):
    """grating (prices one 3 FC step) and layout (writes outputs from the grating)."""
    import tarang.design as design
    log = []

    def fake_grating(ctx):
        ran = ctx.ui.proceed("A. period x fill factor map: 11 2D grating simulations", 3.0, 30.0)
        log.append(("grating", ran))
        return {"period_um": 0.5729} if ran else {"period_um": None}

    def fake_layout(ctx):
        log.append(("layout", ctx.results("grating").get("period_um")))
        return {"files": ["design.gds"]}
    fakes = {"grating": fake_grating, "layout": fake_layout}
    stages = [(n, t, k, fakes.get(n, f)) for n, t, k, f in design.STAGES]
    monkeypatch.setattr(design, "STAGES", stages)
    monkeypatch.setattr(design, "credit_limits", lambda ctx: None)
    return log


def _run(tmp_path, **ui_kw):
    from tarang.design import run
    from tarang.design.config import DesignConfig
    from tarang.design.ui import Prompter
    out = io.StringIO()
    ctx = run(DesignConfig(), tmp_path / "run", only=["grating", "layout"],
              prompter=Prompter(interactive=False, out=out, **ui_kw))
    return ctx, out.getvalue()


def test_price_only_keeps_earlier_results_and_stops_later_stages(tmp_path, two_stages):
    ctx, _ = _run(tmp_path)                                         # a finished design
    assert two_stages == [("grating", True), ("layout", 0.5729)]
    two_stages.clear()
    ctx.cfg.grating.n_teeth = 31                                    # new setting: price it
    import json
    from tarang.design import run
    from tarang.design.ui import Prompter
    out = io.StringIO()
    ctx = run(ctx.cfg, tmp_path / "run", only=["grating", "layout"],
              prompter=Prompter(interactive=False, out=out, price_only=True))
    text = out.getvalue()
    assert two_stages == [("grating", False)]                       # layout not run
    st = json.loads((tmp_path / "run" / "state.json").read_text(encoding="utf-8"))["stages"]
    assert st["grating"]["results"] == {"period_um": 0.5729}       # earlier results kept
    assert st["grating"]["key"] is None and st["grating"]["not_run"]
    assert "priced once grating has run" in text
    assert "earlier results are kept" in text


def test_price_only_without_earlier_results_still_records_the_stage(tmp_path, two_stages):
    ctx, text = _run(tmp_path, price_only=True)
    assert two_stages == [("grating", False)]
    assert ctx.state["stages"]["grating"]["key"] is None
    assert "priced once grating has run" in text


# ------------------------------------------------------------- centring the peak
def test_centring_is_refused_when_it_costs_efficiency_at_the_wavelength():
    from tarang.design import stages
    ui = SimpleNamespace(said=[], say=lambda s: ui.said.append(s))
    design = dict(dim="2D", period_um=0.5729, fill_factor=0.7, source_x_um=2.5,
                  peak_wavelength_um=0.7843, efficiency_at_target=0.338, peak_inside=True)
    results = [design]

    def run_set(points, label, tag):                      # the peak does not follow the period
        rows = [dict(dim="2D", period_um=p, fill_factor=f, source_x_um=x,
                     peak_wavelength_um=0.7817, efficiency_at_target=0.228, peak_inside=True)
                for p, f, x in points]
        results.extend(rows)
        return rows
    got = stages._centre_peak(design, results, run_set, 0.78, ui)
    assert got is design
    assert any("design with the higher efficiency" in s for s in ui.said)


def test_centring_is_kept_when_it_costs_little():
    from tarang.design import stages
    ui = SimpleNamespace(say=lambda s: None)
    design = dict(dim="2D", period_um=1.0928, fill_factor=0.75, source_x_um=5.2,
                  peak_wavelength_um=1.5422, efficiency_at_target=0.309, peak_inside=True)
    results = [design]

    def run_set(points, label, tag):
        rows = [dict(dim="2D", period_um=p, fill_factor=f, source_x_um=x,
                     peak_wavelength_um=1.5503, efficiency_at_target=0.311, peak_inside=True)
                for p, f, x in points]
        results.extend(rows)
        return rows
    got = stages._centre_peak(design, results, run_set, 1.55, ui)
    assert got is not design and got["peak_wavelength_um"] == 1.5503


# ------------------------------------------------------------- mesh checks' boundary
def _ctx(tmp_path, ladder_boundary="", boundary="pml", mesh_scale=2.0):
    return SimpleNamespace(out=tmp_path, cfg=SimpleNamespace(
        run=SimpleNamespace(wavelength_um=0.78),
        checks=SimpleNamespace(ladder_boundary=ladder_boundary, grating_mesh_ladder=[2.5, 3.0]),
        grating=SimpleNamespace(mesh_scale=mesh_scale, boundary=boundary)))


def test_ladder_boundary_defaults_to_the_gratings_own(tmp_path):
    from tarang.design.checks import ladder_boundary
    assert ladder_boundary(_ctx(tmp_path)) == "pml"
    assert ladder_boundary(_ctx(tmp_path, boundary="stable_pml")) == "stable_pml"
    assert ladder_boundary(_ctx(tmp_path, ladder_boundary="Absorber ")) == "absorber"


def test_a_ladder_with_its_own_boundary_runs_its_own_design_mesh(tmp_path, monkeypatch):
    from tarang.design import checks
    made = []

    def fake_device(ctx, P, ff, sx, *, H, depth, mesh=1.0, boundary=None, **kw):
        made.append((round(mesh, 4), boundary))
        return SimpleNamespace(mesh=mesh, boundary=boundary)
    monkeypatch.setattr(checks, "grating_device", fake_device)
    gc = dict(period_um=0.5729, fill_factor=0.7, source_x_um=2.5, etch_depth_um=0.35)
    ctx = _ctx(tmp_path, ladder_boundary="absorber", boundary="stable_pml")
    ctx.results = lambda name: {"height_um": 0.35}
    out = checks.mesh_ladder_variants(ctx, gc, {"final": None, "mesh x1.5": None})
    assert sorted(out) == [2.0, 2.5, 3.0]                  # the design mesh joins the ladder
    assert out[3.0] is None                                # = the x1.5 check (45 cells)
    assert made == [(1.0, "absorber"), (1.25, "absorber")]
    # the rows compare the absorber runs only, never the design run with its own boundary
    wl = np.linspace(0.73, 0.83, 41)

    class Spec:
        def __init__(self, e):
            self.e = e

        def get_spectrum(self, folder):
            return wl, self.e * np.exp(-((wl - 0.78) / 0.03) ** 2)
    rows = checks.mesh_ladder_rows(ctx, gc, {2.0: Spec(0.33), 2.5: Spec(0.34)},
                                   {"final": (wl, 0.20 * np.ones_like(wl)),
                                    "mesh x1.5": (wl, 0.35 * np.exp(-((wl - 0.78) / 0.03) ** 2))})
    assert [r["mesh_scale"] for r in rows] == [2.0, 2.5, 3.0]
    assert {r["boundary"] for r in rows} == {"absorber"}
    assert rows[0]["efficiency_at_target"] == pytest.approx(0.33, abs=1e-3)


def test_new_settings_keep_older_runs_done():
    from tarang.design.config import DesignConfig
    from tarang.design.stages import settings_key
    cfg = DesignConfig()
    before = settings_key(cfg, "checks")
    del cfg.checks.__dict__["ladder_boundary"]               # a design.toml from v0.1.4
    assert settings_key(cfg, "checks") == before
    cfg = DesignConfig()
    cfg.checks.ladder_boundary = "absorber"
    assert settings_key(cfg, "checks") != before


# ------------------------------------------------------------- sidewall coupling gaps
def test_sidewall_coupling_uses_gaps_spread_over_0p2_um(tmp_path, monkeypatch):
    from tarang.design import localmodes, walls
    seen = []

    def fake(ctx, H, W, R, gaps, **kw):
        seen.append(list(gaps))
        return dict(theta0=1.0, gamma_per_um=5.5, rows=[dict(dn=1e-3) for _ in gaps])
    monkeypatch.setattr(localmodes, "coupling_from_supermodes", fake)
    cfg = SimpleNamespace(checks=SimpleNamespace(sidewall_reference="top"),
                          cross_section=SimpleNamespace(decision_steps=70))
    state = {"stages": {"gap": {"results": {"fit": {"theta_0": 1.0, "gamma": 5.5},
                                            "critical_gap_um": 0.4696,
                                            "over_coupled_gap_um": 0.4314,
                                            "target_cross_power": 0.01}},
                        "ring": {"results": {"radius_um": 107.5}}}}
    ctx = SimpleNamespace(cfg=cfg, state=state, ui=SimpleNamespace(say=lambda s: None),
                          results=lambda n: {"height_um": 0.7, "width_um": 1.0},
                          path=lambda *p: tmp_path.joinpath(*p))
    (tmp_path / "results").mkdir()
    walls.coupling_sweep(ctx, [90.0])
    assert seen == [[0.3696, 0.4696, 0.5696]]
