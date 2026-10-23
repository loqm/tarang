"""Phase 14B: edge coupling (wide facet, lensed fibre): coupling choice, facet overlap,
taper shapes, the 3D facet run and its analysis, edge couplers on the mask to the chip edge,
the chip as a Fabry-Perot cavity in the device model, the report; and the fixes from the
Phase 14A runs (layout figure cell, loss budget with the 3D focusing grating, report filters)."""
from __future__ import annotations

import csv
import io
import json
import math
import types

import numpy as np
import pytest

from test_phase12b import _ctx

C_UM = 2.99792458e14


# ---- coupling choice and stages ------------------------------------------------------------ #
def test_coupling_choice_picks_the_stages_and_keeps_old_keys():
    from tarang.design.config import DesignConfig
    from tarang.design.stages import STAGE_NAMES, applies, settings_key
    cfg = DesignConfig()
    assert cfg.run.coupling == "grating"
    assert all(applies(s, cfg) for s in ("grating", "taper", "apodise", "focus"))
    assert not applies("edge", cfg)
    keys = {s: settings_key(cfg, s) for s in STAGE_NAMES if s != "edge"}
    cfg.edge.chip_length_um = 3000.0                 # edge settings do not touch grating runs
    cfg.edge.fiber_mfd_um = 2.5
    assert {s: settings_key(cfg, s) for s in keys} == keys
    cfg.run.coupling = "edge"
    assert applies("edge", cfg) and not any(applies(s, cfg) for s in ("grating", "taper", "apodise",
                                                                         "focus"))
    assert applies("checks", cfg) and applies("layout", cfg) and applies("device", cfg)
    assert settings_key(cfg, "layout") != keys["layout"]
    assert STAGE_NAMES.index("edge") == STAGE_NAMES.index("layout") - 1


def test_unknown_coupling_is_refused(tmp_path):
    from tarang.design import run
    from tarang.design.config import DesignConfig
    from tarang.design.ui import Prompter
    cfg = DesignConfig()
    cfg.run.coupling = "prism"
    cfg.given.update({"run.goal", "run.coupling"})
    with pytest.raises(ValueError):
        run(cfg, tmp_path / "r", interactive=False, only=["report"],
            prompter=Prompter(False, out=io.StringIO()))


def test_edge_settings_round_trip_through_the_toml(tmp_path):
    from tarang.design.config import DesignConfig
    cfg = DesignConfig()
    cfg.run.coupling = "edge"
    cfg.edge.profiles = ["parabolic", "linear"]
    cfg.edge.facet_width_um = 4.5
    p = cfg.save(tmp_path / "d.toml")
    back = DesignConfig.load(p)
    assert back.run.coupling == "edge" and back.edge.profiles == ["parabolic", "linear"]
    assert back.edge.facet_width_um == 4.5 and back.edge.chip_layer == [99, 0]


# ---- the fibre spot and the overlaps ----------------------------------------------------------- #
def test_gaussian_is_normalised_and_overlaps_match_the_closed_forms():
    from tarang.design import edge
    y = np.linspace(-9, 9, 721)
    z = np.linspace(-9, 9, 721)
    dA = edge._weights(y, z)
    for dist in (0.0, 3.0, -2.0):
        u = edge.gaussian(y, z, 0.3, -0.2, 1.75, 1.55, dist=dist)
        assert np.sum(np.abs(u) ** 2 * dA) == pytest.approx(1.0, abs=1e-6)
    u = edge.gaussian(y, z, 0, 0, 1.75, 1.55)
    v = edge.gaussian(y, z, 0, 0, 2.0, 1.55)
    assert edge.scalar_overlap(u, 0 * u, y, z, v) == pytest.approx((2 * 1.75 * 2 / (1.75 ** 2 + 4)) ** 2)
    d = 0.8
    w = edge.gaussian(y, z, d, 0, 1.75, 1.55)
    assert edge.scalar_overlap(u, 0 * u, y, z, w) == pytest.approx(math.exp(-d ** 2 / 1.75 ** 2))
    B = 2.0                                           # a field that is the fibre mode itself
    a2, P = edge.vector_overlap(B * u, 0 * u, 0 * u, B * u / edge.ETA0, y, z, u)
    assert a2 / P == pytest.approx(1.0, rel=1e-9)
    assert P == pytest.approx(B ** 2 / (2 * edge.ETA0), rel=1e-6)
    assert edge.fresnel(1.75) == pytest.approx((0.75 / 2.75) ** 2)


def test_facet_width_is_chosen_for_the_average_over_the_spot_sizes():
    from tarang.design import edge
    rows = [dict(width_um=w, **{f"eta_mfd{m:g}": 0.45 - 0.01 * (w - c) ** 2
                               for m, c in ((3.0, 4.0), (3.5, 4.75), (4.0, 5.5))})
            for w in np.arange(1.0, 7.01, 0.5)]
    assert edge.choose_width(rows, [3.0, 3.5, 4.0]) == 4.75
    assert edge.default_widths(1.0, 3.5) == [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0,
                                             6.5, 7.0]


# ---- taper shapes ------------------------------------------------------------------------------ #
@pytest.mark.parametrize("kind", ["linear", "parabolic", "power"])
def test_taper_shapes_and_their_inverse(kind):
    from tarang.design import taper
    s = np.linspace(0, 1, 11)
    w = taper.profile_width(s, 1.0, 4.75, kind, 0.6)
    assert w[0] == pytest.approx(1.0) and w[-1] == pytest.approx(4.75) and np.all(np.diff(w) > 0)
    assert np.allclose(taper.profile_fraction(w, 1.0, 4.75, kind, 0.6), s)
    xb = taper.cell_boundaries(4.75, 1.0, 40, 30.0, kind, 0.6)
    assert xb[0] == 0 and xb[-1] == 30 and np.all(np.diff(xb) > 0)
    pts = taper.taper_outline(4.75, 1.0, 30.0, kind, 0.6)
    ys = [y for x, y in pts]
    assert max(ys) == pytest.approx(4.75 / 2) and min(ys) == pytest.approx(-4.75 / 2)


def test_linear_cells_are_unchanged():
    from tarang.design import taper
    i = np.arange(31) / 30
    wb = 14.5 * (1.0 / 14.5) ** i
    ref = (14.5 - wb) / 13.5 * 290.0
    assert np.allclose(taper.cell_boundaries(14.5, 1.0, 30, 290.0), ref)


def test_smoothed_2d_taper_model_falls_with_length():
    from tarang.design import taper
    lengths = [10, 30, 100, 200, 300, 400]
    sm = taper.eim_taper(1.8511, 1.444, 4.75, 1.0, 1.55, lengths, n_cells=300, smooth=True)["loss_db"]
    assert sm[0] > sm[1] > sm[2]
    assert max(sm[3:]) < 0.005                         # no staircase rise at long lengths
    st = taper.eim_taper(1.8511, 1.444, 4.75, 1.0, 1.55, lengths, n_cells=300)["loss_db"]
    assert st[-1] > 5 * max(abs(sm[-1]), 1e-3)          # the unsmoothed grid steps rise


# ---- the 3D facet run ------------------------------------------------------------------------ #
def _edge_ctx(tmp_path, **kw):
    ctx = _ctx(tmp_path, run__coupling="edge", **kw)
    return ctx


def test_facet_sim_runs_the_waveguide_through_the_boundary(tmp_path):
    pytest.importorskip("tidy3d")
    from tarang.design import edge
    ctx = _edge_ctx(tmp_path)
    sim = edge.facet_sim(ctx, 4.75, 1.8463, 15)
    (x0, y0, z0), (x1, y1, z1) = sim.bounds
    wg = next(s for s in sim.structures if s.name == "Facet waveguide")
    (gx0, gy0, _), (gx1, gy1, _) = wg.geometry.bounds
    assert gx0 < x0 and gx1 == 0.0 and gy1 - gy0 == pytest.approx(4.75)
    names = {m.name for m in sim.monitors}
    assert {"port", "back", "out", "fibre plane", "side view"} <= names
    plane = next(m for m in sim.monitors if m.name == "fibre plane")
    assert plane.center[0] == pytest.approx(edge.FIBRE_PLANE_UM) and len(plane.freqs) == 11
    port = next(m for m in sim.monitors if m.name == "port")
    assert len(port.freqs) == 101 and sim.sources[0].center[0] < port.center[0] < 0
    assert sim.symmetry == (0, -1, 0)
    assert sim.num_cells < 5e6


def _fake_facet_data(eta, R, z0=0.35, w0=1.75, g=0.0):
    import xarray as xr
    from tarang.design import edge
    f = np.linspace(C_UM / 1.65, C_UM / 1.45, 201)
    ff = np.linspace(C_UM / 1.65, C_UM / 1.45, 11)
    amps = np.zeros((2, 1, len(f)), complex)
    amps[0, 0, :] = 1.0
    amps[1, 0, :] = math.sqrt(R)
    port = types.SimpleNamespace(amps=xr.DataArray(amps, dims=("direction", "mode_index", "f"),
                                                   coords=dict(direction=["+", "-"], mode_index=[0], f=f)))

    def flux(v):
        return types.SimpleNamespace(flux=xr.DataArray(np.full(len(f), v), dims=("f",), coords=dict(f=f)))
    y = np.linspace(-4.9, 4.9, 197)
    z = np.linspace(-2.5, 3.2, 115)
    B = math.sqrt(2 * edge.ETA0 * eta)
    E = np.zeros((1, len(y), len(z), len(ff)), complex)
    for k, fk in enumerate(ff):
        E[0, :, :, k] = B * edge.gaussian(y, z, 0.0, z0, w0, C_UM / fk, dist=edge.FIBRE_PLANE_UM - g)
    co = dict(x=[edge.FIBRE_PLANE_UM], y=y, z=z, f=ff)

    def da(a):
        return xr.DataArray(a, dims=("x", "y", "z", "f"), coords=co)
    plane = types.SimpleNamespace(Ey=da(E), Ez=da(0 * E), Hy=da(0 * E), Hz=da(E / edge.ETA0))
    side = types.SimpleNamespace(Ey=xr.DataArray(np.ones((5, 1, 4, 1)), dims=("x", "y", "z", "f"),
                                                 coords=dict(x=np.arange(5.0), y=[0.0], z=np.arange(4.0),
                                                             f=[C_UM / 1.55])))
    return {"port": port, "back": flux(-R - 0.01), "out": flux(1 - R - 0.01), "fibre plane": plane,
            "side view": side}


def test_facet_analysis_finds_the_spot_and_its_tolerances(tmp_path):
    pytest.importorskip("xarray")
    from tarang.design import edge
    ctx = _edge_ctx(tmp_path)
    r = edge.analyse(_fake_facet_data(0.40, 0.09), ctx, 3.5, 0.5, 1.0, 1.55)
    assert r["eta"] == pytest.approx(0.40, rel=5e-3)
    assert r["fibre_height_um"] == pytest.approx(0.35, abs=0.011) and r["working_distance_um"] == 0.0
    assert r["phase_sign"] == 1.0
    assert r["reflection_mode"] == pytest.approx(0.09) and r["transmitted"] == pytest.approx(0.90)
    spot = {s["mfd_um"]: s["eta"] for s in r["spot"]}
    k = (2 * 1.5 * 1.75 / (1.5 ** 2 + 1.75 ** 2)) ** 2
    assert spot[3.0] == pytest.approx(0.40 * k, rel=1e-2)
    for t in r["tolerance"]:                           # 1 um offset: exp(-d^2 / w0^2)
        assert t["change_db"] == pytest.approx(10 * math.log10(math.exp(-1 / 1.75 ** 2)), abs=0.05)
    assert r["gap_curve"][0] == max(r["gap_curve"])
    # a spot with its waist 2 um in front of the facet is found there (curvature sign right).
    # The time convention is read from the field leaving a real facet, which always diverges;
    # this synthetic field converges, so the convention is given here
    r2 = edge.analyse(_fake_facet_data(0.40, 0.09, g=2.0), ctx, 3.5, 0.5, 1.0, 1.55, sign=1.0)
    assert r2["working_distance_um"] == pytest.approx(2.0, abs=0.11) and r2["eta"] == pytest.approx(0.4, rel=5e-3)


def test_time_convention_is_read_from_a_diverging_field():
    from tarang.design import edge
    y = np.linspace(-3, 3, 121)
    z = np.linspace(-3, 3, 121)
    for sgn in (1.0, -1.0):          # a 0.4 um waist 0.5 um behind the plane: strongly diverging
        u = edge.gaussian(y, z, 0, 0, 0.4, 1.55, dist=0.5, sign=sgn)
        assert edge.phase_sign(u, z, 60) == sgn


# ---- the edge stage, with the solves replaced ------------------------------------------------- #
def _fake_mode(width, H=0.7, wl=1.55):
    y = np.linspace(-width / 2 - 3, width / 2 + 3, 241)
    z = np.linspace(-3, 3.7, 171)
    Y, Z = np.meshgrid(y, z, indexing="ij")
    Ey = np.exp(-(Y / (0.45 * width + 0.3)) ** 2 - ((Z - 0.33) / 0.42) ** 2)
    n = 1.70 + 0.15 * (1 - math.exp(-(width - 1.0)))
    return dict(n_eff=n, k_eff=0.0, confinement=0.9, y=y, z=z, Ex=0 * Ey, Ey=Ey.astype(complex),
                Ez=0 * Ey, Hy=0 * Ey, Hz=Ey * n / 376.73, width_um=width, wavelength_um=wl, height_um=H)


def _patched_edge(monkeypatch):
    from tarang.design import edge
    monkeypatch.setattr(edge, "facet_mode", lambda ctx, w, wl=None, steps=None, with_handle=True:
                        _fake_mode(w, wl=wl or 1.55))
    return edge


def test_edge_stage_without_cloud_runs(tmp_path, monkeypatch):
    edge = _patched_edge(monkeypatch)
    ctx = _edge_ctx(tmp_path, edge__run_eme=False, edge__check_3d=False,
                    edge__facet_widths_um=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    res = edge.stage_edge(ctx)
    assert res["facet_width_um"] == res["facet_width_recommended_um"]
    assert 1.0 < res["facet_width_um"] <= 6.0
    assert res["profile"] in ("linear", "parabolic", "power") and res["taper_length_um"] > 0
    assert res["taper_loss_db"] < ctx.cfg.edge.max_loss_db
    assert res["eta_coupler"] == pytest.approx(res["eta_free"] * 10 ** (-res["taper_loss_db"] / 10))
    assert res["recommend"]["recommended"]["facet_width_um"] == res["facet_width_recommended_um"]
    for f in ("12a_edge_facet.png", "12b_edge_taper.png"):
        assert (ctx.out / "figures" / f).exists()
    assert (ctx.out / "tables" / "edge_spectrum_free.csv").exists()
    cache = json.loads((ctx.out / "results" / "edge_facet_map.json").read_text())
    assert len(cache) == 6
    ctx.ui.price_only = True                       # only priced: the stage counts as not done
    ctx.cfg.edge.check_3d = True
    monkeypatch.setattr("tarang.design.checks._run_batch", lambda *a, **k: None)
    priced = edge.stage_edge(ctx)
    assert priced["skipped"]
    from tarang.design import layout
    ctx.state["stages"]["edge"] = {"results": priced}       # a price-only run draws no couplers
    p = layout.device_parameters(ctx)
    assert p["couplers"] == [] and p["edge_facet_width_um"] is None
    calls = []
    monkeypatch.setattr(edge, "facet_mode", lambda *a, **k: calls.append(1) or _fake_mode(a[1]))
    edge.facet_map(ctx, [1.0, 2.0], [3.0, 3.5, 4.0])     # from the cache: no solve
    assert not calls


def test_edge_stage_with_its_cloud_runs_writes_every_table(tmp_path, monkeypatch):
    """The whole stage with the EME and the 3D facet run answered from fake saved runs
    (B3b of the 14B guide crashed writing edge_spot_and_offsets.csv)."""
    pytest.importorskip("xarray")
    import tidy3d as td
    from tarang.design import checks, taper
    from tarang.utilities import cloud_run
    edge = _patched_edge(monkeypatch)
    ctx = _edge_ctx(tmp_path, edge__facet_widths_um=[2.0, 4.0, 6.0])
    monkeypatch.setattr(checks, "_run_batch", lambda *a, **k: True)
    monkeypatch.setattr(cloud_run, "find_cached", lambda sim, folder, name: tmp_path / "fake.hdf5")
    monkeypatch.setattr(cloud_run, "_data_class", lambda sim: types.SimpleNamespace(
        from_file=lambda p: "eme"))

    def fake_eme(data, L0, scales):
        L = [L0 * s for s in scales]
        loss = [2.0 * math.exp(-x / 6.0) + 1e-3 for x in L]
        return dict(length_um=L, transmission=[10 ** (-v / 10) for v in loss], loss_db=loss)
    monkeypatch.setattr(taper, "taper_transmission", fake_eme)
    monkeypatch.setattr(td.SimulationData, "from_file", classmethod(
        lambda cls, p: _fake_facet_data(0.395, 0.148)))
    res = edge.stage_edge(ctx)
    assert set(res["eme"]) == {"linear", "parabolic", "power"}
    assert res["taper_loss_source"] == "3D EME"
    f3 = res["fdtd"]
    assert f3["eta"] == pytest.approx(0.395, rel=5e-3) and f3["reflection_mode"] == pytest.approx(0.148)
    rows = list(csv.DictReader(open(ctx.out / "tables" / "edge_spot_and_offsets.csv", encoding="utf-8")))
    assert [r["kind"] for r in rows] == ["spot"] * 3 + ["offset"] * 4
    assert rows[0]["mfd_um"] and rows[3]["direction"] == "y" and rows[0]["direction"] == ""
    for t in ("edge_reflection_3d.csv", "edge_spectrum_3d.csv", "edge_taper_eme.csv", "edge_facet_map.csv"):
        assert (ctx.out / "tables" / t).exists()
    assert (ctx.out / "figures" / "12c_edge_3d.png").exists()
    assert res["eta_coupler"] == pytest.approx(f3["eta"] * 10 ** (-res["taper_loss_db"] / 10))
    from tarang.design import circuit
    ctx.state["stages"]["edge"] = {"results": res}
    cp = circuit.coupler_model(ctx)
    assert "3D FDTD" in cp["label"]
    b, src = circuit.coupler_reflection(ctx)
    assert max(b["reflected"]) == pytest.approx(0.148) and "3D FDTD" in src


def test_csv_tables_take_rows_of_different_kinds(tmp_path):
    from tarang.design.checks import _write_csv
    p = _write_csv(tmp_path / "t.csv", [dict(kind="a", x=1), dict(kind="b", y=2)])
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    assert rows == [dict(kind="a", x="1", y=""), dict(kind="b", x="", y="2")]


def _edge_results(**kw):
    r = dict(facet_width_um=4.75, facet_width_recommended_um=4.75, fiber_mfd_um=3.5, mfd_tolerance_um=0.5,
             n_eff_facet=1.846, eta_overlap=0.457, fresnel_T=0.912, eta_free=0.417,
             fibre_height_free_um=0.31, reflection_free=0.0884, profiles=["linear", "parabolic"],
             power_exponent=0.6, model_lengths_um=dict(linear=45.0, parabolic=40.0), eme={},
             profile="parabolic", profile_recommended="parabolic", taper_length_um=40.0,
             taper_length_recommended_um=40.0, taper_loss_db=0.01, taper_loss_source="2D model",
             fdtd=None, eta_coupler=0.416,
             spectrum_free=[dict(wavelength_um=w, efficiency=0.417, reflection=0.0884)
                            for w in (1.45, 1.55, 1.65)])
    r.update(kw)
    return r


def test_edge_coupler_model_and_budget(tmp_path):
    from tarang.design import checks, circuit, layout
    ctx = _edge_ctx(tmp_path)
    ctx.state["stages"]["edge"] = {"results": _edge_results()}
    cp = circuit.coupler_model(ctx)
    assert cp["eta"][1] == pytest.approx(0.417 * 10 ** (-0.001))
    assert cp["length_um"] == pytest.approx(40.0 + ctx.cfg.edge.facet_straight_um)
    assert "butt-coupling" in cp["label"]
    b, src = circuit.coupler_reflection(ctx)
    assert b["reflected"][1] == pytest.approx(0.0884) and "Fresnel" in src
    assert layout.bus_length_um(ctx) == pytest.approx(5000 - 2 * 60.0)
    rows = checks.loss_budget(ctx, None)
    assert rows[0]["item"].startswith("facet, input")
    assert rows[-1]["loss_db"] == pytest.approx(2 * 3.80 + 0.02 + 0.2, abs=0.02)


def test_edge_mask_runs_to_the_chip_edge(tmp_path):
    gdstk = pytest.importorskip("gdstk")
    from tarang.design import layout
    ctx = _edge_ctx(tmp_path, layout__gap_offsets_um=[-0.05, 0.05])
    ctx.state["stages"]["gap"] = {"results": dict(critical_gap_um=0.47, over_coupled_gap_um=0.431,
                                                  over_coupling=1.5)}
    ctx.state["stages"]["edge"] = {"results": _edge_results()}
    p = layout.device_parameters(ctx)
    assert p["couplers"] == [["ec", "edge"]] and p["main_coupler"] == "edge"
    assert p["bus_length_um"] == pytest.approx(4880.0) and p["grating_period_um"] is None
    files = {f.name for f in layout.build_gds(p, tmp_path)}
    assert {"ring_waveguide_ec.gds", "waveguide_ec.gds", "chip.json", "design.gds"} <= files
    cell = gdstk.read_gds(str(tmp_path / "waveguide_ec.gds")).cells[0]
    (x0, y0), (x1, y1) = cell.bounding_box()
    assert x1 == pytest.approx(2500 + 10.0) and x0 == pytest.approx(-2510.0)
    assert y1 == pytest.approx(4.75 / 2)
    chip = json.loads((tmp_path / "chip.json").read_text())
    assert chip["fits"] and len(chip["devices"]) == 6 + 1
    ys = chip["facet_y_um"]
    assert min(abs(a - b) for a, b in zip(ys, ys[1:])) >= 127.0 - 1e-6
    tops = gdstk.read_gds(str(tmp_path / "design.gds")).top_level()
    assert len(tops) == 1
    top = tops[0]
    layers = {(q.layer, q.datatype) for q in top.polygons}
    assert {(99, 0), (98, 0)} <= layers
    ctx.cfg.edge.chip_width_um = 300.0
    layout.build_gds(layout.device_parameters(ctx), tmp_path / "small")
    assert not json.loads((tmp_path / "small" / "chip.json").read_text())["fits"]


def test_edge_spirals_have_the_facet_to_facet_length(tmp_path, monkeypatch):
    pytest.importorskip("gdstk")
    from tarang.design import layout, waveguides as wg
    ctx = _edge_ctx(tmp_path, run__goal="loss_test", waveguides__lengths_cm=[0, 1, 2])
    ctx.state["stages"]["edge"] = {"results": _edge_results()}
    monkeypatch.setattr(wg, "bend_loss_db_cm", lambda ctx, H, W, R: 50.0 * math.exp(-R / 8.0))
    ctx.state["stages"]["waveguides"] = {"results": wg.stage_waveguides(ctx)}
    out = layout.export(ctx)
    rows = list(csv.DictReader(open(ctx.out / "layout" / "spirals.csv", encoding="utf-8")))
    assert [round(float(r["length_um"])) for r in rows] == [10000, 20000]
    assert (ctx.out / "figures" / "6_layout.png").exists()
    ctx.cfg.waveguides.lengths_cm = [0, 0.4]                # shorter than the chip
    ctx.state["stages"]["waveguides"]["results"]["lengths_cm"] = [0, 0.4]
    out = layout.export(ctx)
    assert "shorter than the chip" in out["error"]


def test_edge_cutback_and_ring_device(tmp_path, monkeypatch):
    from tarang.design import circuit, waveguides as wg
    ctx = _edge_ctx(tmp_path, run__goal="loss_test")
    ctx.state["stages"]["edge"] = {"results": _edge_results()}
    monkeypatch.setattr(wg, "bend_loss_db_cm", lambda ctx, H, W, R: 50.0 * math.exp(-R / 8.0))
    ctx.state["stages"]["waveguides"] = {"results": wg.stage_waveguides(ctx)}
    r = circuit.stage_device(ctx)
    L = [x["length_cm"] for x in r["rows"]]
    assert L == pytest.approx([0.5, 1.0, 2.0, 4.0])
    p = 0.0884 * 10 ** (-0.4 * 0.5 / 10)
    assert r["rows"][0]["ripple_pp_db"] == pytest.approx(20 * math.log10((1 + p) / (1 - p)), abs=2e-3)
    assert "Regener" in ctx.ui.out.getvalue()


def test_report_has_the_edge_section_and_no_grating_parts(tmp_path, monkeypatch):
    from tarang.design import report
    ctx = _edge_ctx(tmp_path)
    ctx.state["stages"]["edge"] = {"results": _edge_results()}
    text = "\n".join(report._edge_section(ctx, ctx.results("edge")))
    assert "## 5. Edge coupler" in text and "12a_edge_facet" not in text      # no figure written
    assert "parabolic" in text and "Fabry-Perot" in text
    lim = report.known_limits(ctx.cfg)
    assert any(x.startswith("Edge coupler") for x in lim) and not any(x.startswith("Grating") for x in lim)
    chk = report._checks_section(ctx, dict(grating=dict(variants=[dict(case="final")])))
    assert not any("Grating tolerances" in x for x in chk)


# ---- fixes from the Phase 14A runs ----------------------------------------------------------- #
def test_budget_uses_the_3d_focusing_result_without_double_counting(tmp_path):
    from test_phase14a import _with_couplers
    from tarang.design import checks
    ctx = _with_couplers(_ctx(tmp_path))
    ctx.state["stages"]["focus"]["results"].update(efficiency_3d=0.28)
    rows = checks.loss_budget(ctx, ctx.results("grating"))
    items = [r["item"] for r in rows]
    assert items[0].endswith("(3D FDTD)") and not any("tip taper" in i for i in items)
    assert not any("in-plane" in i for i in items)
    db = -10 * math.log10(0.28)
    assert rows[-1]["loss_db"] == pytest.approx(2 * db + 0.4 * (400 + 8) * 1e-4, abs=0.01)


def test_focusing_3d_spectrum_is_not_multiplied_by_the_tip_taper(tmp_path):
    from test_phase14a import _with_couplers
    from tarang.design import circuit
    ctx = _with_couplers(_ctx(tmp_path))
    t = ctx.out / "tables"
    t.mkdir(parents=True, exist_ok=True)
    (t / "focusing_spectrum_3d.csv").write_text("wavelength_um,efficiency,efficiency_db\n1.5,0.2,0\n"
                                                "1.55,0.28,0\n1.6,0.2,0\n")
    cp = circuit.coupler_model(ctx)
    assert cp["eta"][1] == pytest.approx(0.28) and "3D FDTD" in cp["label"]


def test_coupler_spectrum_is_interpolated_in_log():
    from tarang.design.circuit import interp_eta
    e = interp_eta(1.5, [1.4, 1.6], [1e-1, 1e-5])
    assert e == pytest.approx(1e-3)


def test_layout_figure_shows_the_critical_ring(tmp_path):
    pytest.importorskip("gdstk")
    from test_phase14a import _with_couplers
    from tarang.design import layout, visuals
    import matplotlib
    matplotlib.use("Agg")
    ctx = _with_couplers(_ctx(tmp_path, layout__gap_offsets_um=[-0.05, 0.05]))
    ctx.state["stages"]["gap"] = {"results": dict(critical_gap_um=0.47, over_coupled_gap_um=0.431,
                                                  over_coupling=1.5)}
    p = layout.device_parameters(ctx)
    layout.build_gds(p, tmp_path)
    seen = {}
    orig = visuals._save

    def save(fig, path):
        seen["title"] = fig.axes[0].get_title()
        return orig(fig, path)
    visuals._save = save
    try:
        visuals.layout_views(tmp_path / "ring_waveguide_fgc.gds", p, tmp_path / "f.png")
    finally:
        visuals._save = orig
    assert seen["title"].startswith("ring_waveguide_fgc_gap470nm")


def test_loss_test_report_has_no_ring_checks(tmp_path):
    from tarang.design import report
    ctx = _ctx(tmp_path, run__goal="loss_test")
    chk = dict(gap_tolerance=[dict(design="critical", case="as designed", gap_um=0.47, K=1.0,
                                   Q_loaded=4.9e5, dip_depth_db=40.0)],
               mode_families=[dict(family="TE00")], fsr_tolerance=dict(fsr_min_ghz=198.0,
                                                                    fsr_max_ghz=201.0, n_g_min=2.2,
                                                                    n_g_max=2.23),
               loss_budget=[dict(item="fibre to fibre, off resonance", loss_db=11.0)])
    text = "\n".join(report._checks_section(ctx, chk))
    assert "Gap tolerance" not in text and "Mode families" not in text and "FSR tolerance" not in text
    assert "loss budget" in text
    assert not any(x.startswith("Ring") or x.startswith("Gap") for x in report.known_limits(ctx.cfg))


def test_bend_loss_below_the_floor_is_printed_as_a_bound(tmp_path, monkeypatch):
    from tarang.design import waveguides as wg
    ctx = _ctx(tmp_path, run__goal="loss_test")
    monkeypatch.setattr(wg, "bend_loss_db_cm", lambda ctx, H, W, R: 50.0 * math.exp(-R / 2.0))
    wg.stage_waveguides(ctx)
    out = ctx.ui.out.getvalue()
    assert "< 0.001" in out
    rows = list(csv.DictReader(open(ctx.out / "tables" / "bend_loss.csv", encoding="utf-8")))
    assert float(rows[-1]["bend_loss_db_cm"]) < 1e-3                # the csv keeps the number


@pytest.mark.slow
def test_facet_mode_with_the_si_handle_is_the_core_mode(tmp_path):
    pytest.importorskip("tidy3d")
    from tarang.design import edge
    ctx = _edge_ctx(tmp_path)
    m = edge.facet_mode(ctx, 3.5)
    assert 1.83 < m["n_eff"] < 1.85 and m["confinement"] > 0.8
    i = int(np.argmin(np.abs(m["y"])))
    j = int(np.argmin(np.abs(m["z"] - 0.35)))
    assert abs(m["Hz"][i, j] / m["Ey"][i, j]) * edge.ETA0 == pytest.approx(1.87, abs=0.05)
    ov, z0 = edge.best_height(m["Ey"], m["Ez"], m["y"], m["z"], 1.75, 1.55, -1.0, 1.8)
    assert 0.40 < ov < 0.46 and 0.2 < z0 < 0.45
