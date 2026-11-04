"""v0.1.3: sloped (non-vertical) etched sidewalls in the mode solver and the grating, the
sidewall sweep of the checks stage, a mesh setting and convergence ladder for the grating,
the 3D focusing grating's source kept inside the domain for thin films, and grating
variants (tooth bias, period) on the mask."""
from __future__ import annotations

import io
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _ctx(tmp_path, **cfg_changes):
    from tarang.design.config import DesignConfig
    from tarang.design.stages import Context
    from tarang.design.ui import Prompter
    cfg = DesignConfig()
    cfg.run.band_um = 0.2
    for dotted, value in cfg_changes.items():
        sec, key = dotted.split("__")
        setattr(getattr(cfg, sec), key, value)
        cfg.given.add(f"{sec}.{key}")
    ctx = Context(cfg, Prompter(interactive=False, out=io.StringIO()), tmp_path / "run")
    ctx.state["stages"] = {"cross_section": {"results": dict(height_um=0.7, width_um=1.0, steps=70)}}
    return ctx


# ---- geometry ----------------------------------------------------------------------------- #
def test_sidewall_widths_follow_the_reference_plane():
    from tarang.geometries import sidewall as sw
    s = 0.35 / math.tan(math.radians(85))
    assert sw.wall_shift(0.35, 85) == pytest.approx(s) and s == pytest.approx(0.0306, abs=1e-4)
    assert sw.core_widths(0.5, 0.35, 85, "top") == pytest.approx((0.5 + 2 * s, 0.5))
    assert sw.core_widths(0.5, 0.35, 85, "bottom") == pytest.approx((0.5, 0.5 - 2 * s))
    assert sw.core_widths(0.5, 0.35, 85, "middle") == pytest.approx((0.5 + s, 0.5 - s))
    wb, wt = sw.core_widths(0.5, 0.35, 93, "top")              # re-entrant: narrower at the bottom
    assert wb < wt == 0.5
    assert sw.core_widths(0.5, 0.35, 90) == (0.5, 0.5)
    assert sw.bottom_gap(0.238, 0.35, 85) == pytest.approx(0.238 - 2 * s)
    with pytest.raises(ValueError):
        sw.core_widths(0.05, 1.0, 80, "bottom")                 # the walls meet below the top
    with pytest.raises(ValueError):
        sw.wall_shift(1.0, 30)
    with pytest.raises(ValueError):
        sw.check_reference("side")


def test_first_tooth_keeps_its_joint_with_the_waveguide():
    from tarang.geometries import sidewall as sw
    v = sw.tooth_vertices(0.0, 0.4, 0.0, 0.35, 93, "top", vertical_left=True)
    assert v[0][0] == v[3][0] == pytest.approx(0.0)             # no notch at x = 0
    v = sw.tooth_vertices(0.0, 0.4, 0.0, 0.35, 85, "top", vertical_left=True)
    assert v[0][0] == v[3][0] < 0                               # overlaps the waveguide instead
    v = sw.tooth_vertices(0.56, 0.4, 0.0, 0.35, 85, "top")
    assert v[3][0] == pytest.approx(0.56) and v[2][0] == pytest.approx(0.96)   # top = mask
    assert v[0][0] < 0.56 and v[1][0] > 0.96                    # wider foot


def test_vertical_cross_section_is_unchanged_and_sloped_one_is_a_trapezoid():
    import tidy3d as td
    from tarang.waveguide.dispersion_mapping import core as dm
    a = dm.build_sim(0.35, 0.5, wl0=0.78, steps=30, snap=True)
    b = dm.build_sim(0.35, 0.5, wl0=0.78, steps=30, snap=True, sidewall_deg=90.0)
    assert a == b and dm.find_core(a, 0.35, 0.5) >= 0
    c = dm.build_sim(0.35, 0.5, wl0=0.78, steps=30, snap=True, sidewall_deg=85.0)
    core = next(s for s in c.structures if s.name == "core_sloped")
    assert isinstance(core.geometry, td.PolySlab) and core.geometry.axis == 0
    (_, y0, z0), (_, y1, z1) = core.geometry.bounds
    s = 0.35 / math.tan(math.radians(85))
    assert y1 - y0 == pytest.approx(0.5 + 2 * s) and (z0, z1) == pytest.approx((0.0, 0.35))
    assert len(c.grid_spec.snapping_points) == 2   # v0.1.4: each wall at mid-height + faces
    assert not any(isinstance(s.geometry, td.Box) and abs(s.geometry.size[1] - 0.5) < 1e-9
                   for s in c.structures)


@pytest.mark.real_modes
def test_supermodes_build_two_trapezoids(tmp_path, monkeypatch):
    """The coupler simulation holds two sloped cores, centre-to-centre W + gap apart."""
    import tidy3d as td
    from tarang.design import localmodes
    ctx = _ctx(tmp_path)
    seen = {}

    def fake_solver(ctx, H, W, wl, **kw):
        from tarang.waveguide.dispersion_mapping import core as dm
        sim = dm.build_sim(H, W, wl0=wl, steps=20, sidewall_deg=kw.get("sidewall_deg", 90.0),
                           sidewall_ref=kw.get("sidewall_ref", "top"))
        seen["sim"] = kw["extra"](sim)
        raise RuntimeError("stop")
    monkeypatch.setattr(localmodes, "_solver", fake_solver)
    with pytest.raises(RuntimeError, match="stop"):
        localmodes.coupler_supermodes(ctx, 0.7, 1.0, 0.4, sidewall_deg=86.0)
    cores = [s for s in seen["sim"].structures if s.name and s.name.startswith("core_")]
    assert [type(s.geometry) for s in cores] == [td.PolySlab, td.PolySlab]
    centres = sorted((s.geometry.bounds[0][1] + s.geometry.bounds[1][1]) / 2 for s in cores)
    assert centres[1] - centres[0] == pytest.approx(1.4)
    assert not any(s.name == "core_sloped" for s in seen["sim"].structures)


def test_grating_teeth_slope_only_when_asked(tmp_path):
    import tidy3d as td
    from tarang.design.stages import grating_device
    ctx = _ctx(tmp_path)
    d0 = grating_device(ctx, 1.1, 0.75, 7.8, H=0.7, depth=0.7)
    d1 = grating_device(ctx, 1.1, 0.75, 7.8, H=0.7, depth=0.7, sidewall_deg=90.0)
    assert d0.sim == d1.sim and d0._readable_name() == d1._readable_name()
    assert "wall" not in d0._readable_name()
    d2 = grating_device(ctx, 1.1, 0.75, 7.8, H=0.7, depth=0.7, sidewall_deg=86.0)
    assert d2._readable_name().endswith("_wall86")
    teeth = [s for s in d2.sim.structures if s.name.startswith("Tooth_")]
    assert teeth and all(isinstance(t.geometry, td.PolySlab) and t.geometry.axis == 1 for t in teeth)
    assert teeth[0].geometry.bounds[0][0] < 0.0               # wider foot: overlaps the waveguide
    d3 = grating_device(ctx, 1.1, 0.75, 7.8, H=0.7, depth=0.7, sidewall_deg=93.0)
    t0 = next(s for s in d3.sim.structures if s.name == "Tooth_0")
    assert t0.geometry.bounds[0][0] == pytest.approx(0.0)       # undercut: no notch at the joint


def test_grating_mesh_scale_multiplies_every_grating_run(tmp_path):
    from tarang.design.stages import grating_device
    ctx = _ctx(tmp_path)
    base = grating_device(ctx, 1.1, 0.75, 7.8, H=0.7, depth=0.7)
    x15 = grating_device(ctx, 1.1, 0.75, 7.8, H=0.7, depth=0.7, mesh=1.5)
    ctx.cfg.grating.mesh_scale = 2.0
    two = grating_device(ctx, 1.1, 0.75, 7.8, H=0.7, depth=0.7)
    three = grating_device(ctx, 1.1, 0.75, 7.8, H=0.7, depth=0.7, mesh=1.5)
    assert base.fine_steps_per_wvl == 15 and x15.fine_steps_per_wvl == 22   # as before v0.1.3
    assert two.fine_steps_per_wvl == 30 and two.coarse_steps_per_wvl == 20
    assert three.fine_steps_per_wvl == 45
    from tarang.design.checks import _fine_steps
    assert [_fine_steps(s) for s in (1.0, 1.5, 2.0, 3.0)] == [15, 22, 30, 45]


def test_new_settings_at_their_defaults_keep_every_stage_key(tmp_path):
    """Runs made before v0.1.3 are not repeated: the new keys at their defaults are left out
    of the settings keys (config.LEGACY)."""
    from tarang.design.config import LEGACY, DesignConfig
    from tarang.design.stages import STAGE_NAMES, settings_key
    cfg = DesignConfig()
    for sec, keys in (("checks", ["grating_mesh_ladder", "sidewall_sweep_deg",
                                  "sidewall_grating_deg", "sidewall_reference"]),
                      ("grating", ["mesh_scale"]),
                      ("layout", ["grating_tooth_bias_um", "grating_period_offsets_um"])):
        for k in keys:
            assert LEGACY[sec][k] == getattr(getattr(cfg, sec), k)
    keys = {s: settings_key(cfg, s) for s in STAGE_NAMES}
    for sec, k in (("checks", "sidewall_reference"), ("grating", "mesh_scale")):
        old = dict(getattr(cfg, sec).__dict__)
        getattr(cfg, sec).__dict__.pop(k)                       # as if the key did not exist
        assert {s: settings_key(cfg, s) for s in STAGE_NAMES} == keys
        getattr(cfg, sec).__dict__.update(old)
    cfg.grating.mesh_scale = 2.0
    assert settings_key(cfg, "grating") != keys["grating"]
    assert settings_key(cfg, "gap") == keys["gap"]


def test_mesh_ladder_reuses_runs_that_already_exist(tmp_path):
    from tarang.design import checks
    ctx = _ctx(tmp_path, checks__grating_mesh_ladder=[1.0, 1.5, 2.0, 3.0])
    gc = dict(period_um=1.1, fill_factor=0.75, source_x_um=7.8, etch_depth_um=0.7)
    lad = checks.mesh_ladder_variants(ctx, gc, {"final": None, "mesh x1.5": None})
    assert lad[1.0] is None and lad[1.5] is None                # the final run and the x1.5 check
    assert lad[2.0].fine_steps_per_wvl == 30 and lad[3.0].fine_steps_per_wvl == 45


# ---- sidewall sweep numbers ----------------------------------------------------------------- #
def test_d2_estimate_matches_the_textbook_formula():
    from tarang.design import walls
    # 1550 nm design: FSR 199.8 GHz, n_g 2.07, D = 374 ps/(nm km) -> beta2 = -D lambda^2 / (2 pi c)
    beta2 = -374 * 1550.0 ** 2 / (2 * math.pi * 299792.458)       # ps^2/km
    d2 = walls.d2_estimate_mhz(199.8, 2.07, beta2)
    assert d2 == pytest.approx(17.3, abs=0.2)
    assert walls.angles([93, 85, 87]) == [85.0, 87.0, 90.0, 93.0]
    assert walls.angles([]) == []
    with pytest.raises(ValueError):
        walls.angles([30])


def test_coupling_sweep_anchors_on_the_fdtd_fit(tmp_path, monkeypatch):
    """At 90° the sweep returns exactly the FDTD fit (K = 1 at the critical gap); a steeper
    coupled-mode coupling at another angle scales K by its ratio to 90°."""
    from tarang.design import localmodes, walls
    ctx = _ctx(tmp_path)
    rho = 0.01
    gamma, gc = 10.0, 0.45
    theta0 = math.asin(math.sqrt(rho)) * math.exp(gamma * gc)
    ctx.state["stages"]["ring"] = {"results": dict(radius_um=100.0)}
    ctx.state["stages"]["gap"] = {"results": dict(
        critical_gap_um=gc, over_coupled_gap_um=0.43, target_cross_power=rho,
        fit=dict(theta_0=theta0, gamma=gamma))}
    calls = []

    def fake(ctx, H, W, R, gaps, steps=None, sidewall_deg=90.0, sidewall_ref="top", snap=False,
             snap_sides=1):
        assert snap and snap_sides == 2                          # two-sided snapping (v0.1.5)
        calls.append((sidewall_deg, tuple(gaps)))
        boost = 1.2 if sidewall_deg < 90 else 1.0                # more coupling below 90°
        return dict(theta0=0.5 * boost, gamma_per_um=9.0, rows=[dict(dn=1e-3)] * len(gaps))
    monkeypatch.setattr(localmodes, "coupling_from_supermodes", fake)
    res = walls.coupling_sweep(ctx, [86])
    assert {c[0] for c in calls} == {86.0, 90.0} and calls[0][1] == (0.35, 0.45, 0.55)  # v0.1.5
    r90 = next(r for r in res["rows"] if r["sidewall_deg"] == 90.0)
    r86 = next(r for r in res["rows"] if r["sidewall_deg"] == 86.0)
    assert r90["K_critical_gap"] == pytest.approx(1.0, abs=1e-3)
    assert r90["critical_gap_um"] == pytest.approx(gc, abs=1e-4)
    k86 = math.sin(math.asin(math.sqrt(rho)) * 1.2) ** 2 / rho
    assert r86["K_critical_gap"] == pytest.approx(k86, abs=2e-3)
    assert r86["critical_gap_um"] == pytest.approx(gc + math.log(1.2) / gamma, abs=1e-4)
    calls.clear()                                               # saved: not solved again
    again = walls.coupling_sweep(ctx, [86])
    assert calls == [] and again["rows"] == res["rows"]


# ---- focusing grating -------------------------------------------------------------------- #
def test_focusing_source_is_kept_inside_without_moving_earlier_domains():
    from tarang.design.focus import fit_source
    # 1550 nm: H = 0.7, box top H + 3.5 = 4.2, source plane at 4.0: unchanged
    assert fit_source((5.0, 0.0, 4.0), 10.4, 0.7 + 3.5, -12.0, 30.0, 12.0) == (0.7 + 3.5, -12.0, 30.0, 12.0)
    # 780 nm: H = 0.35, box top 3.85 below the source plane at 4.0: the top moves up
    z, x0, x1, y = fit_source((2.0, 0.0, 4.0), 5.0, 0.35 + 3.5, -12.0, 30.0, 12.0)
    assert z == pytest.approx(4.5) and (x0, x1, y) == (-12.0, 30.0, 12.0)


# ---- mask variants ---------------------------------------------------------------------- #
def _p(**kw):
    p = dict(wavelength_um=0.78, grating_period_um=0.56, grating_fill_factor=0.7,
             grating_tooth_bias_um=[-0.02, 0.01], grating_period_offsets_um=[0.01])
    p.update(kw)
    return p


def test_grating_variant_list_gives_teeth_trenches_and_period_shift():
    from tarang.design import layout
    v = {x["tag"]: x for x in layout.grating_variant_list(_p())}
    assert set(v) == {"toothm20nm", "toothp10nm", "periodp10nm"}
    assert v["toothm20nm"]["tooth_um"] == pytest.approx(0.372)
    assert v["toothm20nm"]["trench_um"] == pytest.approx(0.188)
    assert v["periodp10nm"]["period_um"] == pytest.approx(0.57)
    assert v["periodp10nm"]["expected_peak_shift_nm"] == pytest.approx(780 * 0.01 / 0.56, abs=0.05)
    with pytest.raises(ValueError):
        layout.grating_variant_list(_p(grating_tooth_bias_um=[0.2]))  # no trench left
    assert layout.grating_variant_list(_p(grating_period_um=None)) == []


def test_mask_draws_the_grating_variants(tmp_path):
    gdstk = pytest.importorskip("gdstk")
    from tarang.design import layout
    p = dict(name="t", goal="comb_ring", wavelength_um=0.78, waveguide_width_um=0.5,
             ring_radius_um=None, bus_length_um=100.0, couplers=[["gc", "uniform"]],
             reference_waveguide=True, grating_period_um=0.56, grating_fill_factor=0.7,
             grating_teeth=10, uniform_teeth=10, grating_etch="full", grating_width_um=10.0,
             taper_length_um=50.0, grating_tooth_bias_um=[-0.02, 0.02],
             grating_period_offsets_um=[0.01], tone="structure", trench_width_um=3.0,
             layers={"waveguide": [1, 0], "ring": [1, 0], "taper": [1, 0], "grating": [1, 0],
                     "chip": [99, 0], "lane": [98, 0]})
    files = {f.name: f for f in layout.build_gds(p, tmp_path)}
    assert "grating_variants_gc.gds" in files
    lib = gdstk.read_gds(str(files["grating_variants_gc.gds"]))
    cells = {c.name: c for c in lib.cells}
    assert set(cells) == {"waveguide_gc_toothm20nm", "waveguide_gc_toothp20nm",
                          "waveguide_gc_periodp10nm"}

    def teeth(cell):
        return sorted((round(b[0][0], 4), round(b[1][0] - b[0][0], 4))
                      for b in (q.bounding_box() for q in cell.polygons)
                      if abs(b[1][1] - b[0][1] - 10.0) < 1e-6 and b[0][0] > 99.99)
    t = teeth(cells["waveguide_gc_toothp20nm"])
    assert len(t) == 10
    assert t[1][1] == pytest.approx(0.392 + 0.02, abs=1e-4)          # tooth 2: both edges move
    assert t[0][1] == pytest.approx(0.392 + 0.01, abs=1e-4)          # tooth 1: joined to the taper
    assert t[2][0] - t[1][0] == pytest.approx(0.56, abs=1e-4)        # period unchanged
    t = teeth(cells["waveguide_gc_periodp10nm"])
    assert t[2][0] - t[1][0] == pytest.approx(0.57, abs=1e-4)
    assert t[1][1] == pytest.approx(0.7 * 0.57, abs=1e-4)
    top = next(c for c in gdstk.read_gds(str(files["design.gds"])).cells if c.name == "t_TOP")
    assert len(top.references) >= 5                                  # variants placed on the chip


def test_example_designs_sweep_the_sidewall_and_keep_the_mask_variants():
    import tomllib
    for name in ("design_sin1550.toml", "design_sin780.toml"):
        d = tomllib.loads((ROOT / "examples" / name).read_text(encoding="utf-8"))
        assert d["checks"]["sidewall_sweep_deg"][0] == 85 and d["checks"]["sidewall_sweep_deg"][-1] == 93
        assert d["checks"]["sidewall_grating_deg"] and d["layout"]["grating_tooth_bias_um"]
