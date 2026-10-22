"""Phase 12B: taper (grating width and length), arc and grating reflections, figures,
run pictures, apodised grating."""
from __future__ import annotations

import io
import math

import numpy as np
import pytest

import tarang.utilities.cloud_run as cr


def _ctx(tmp_path, **changes):
    from tarang.design.config import DesignConfig
    from tarang.design.stages import Context
    from tarang.design.ui import Prompter
    cfg = DesignConfig()
    cfg.run.band_um = 0.2
    for dotted, value in changes.items():
        sec, key = dotted.split("__")
        setattr(getattr(cfg, sec), key, value)
        cfg.given.add(f"{sec}.{key}")
    ctx = Context(cfg, Prompter(interactive=False, out=io.StringIO()), tmp_path / "run")
    ctx.state["stages"] = {
        "cross_section": {"results": dict(height_um=0.7, width_um=1.0, steps=70, n_eff=1.7066,
                                          n_group=2.2227, aeff_um2=0.746)},
        "ring": {"results": dict(radius_um=107.5, n_group=2.2227, loss_db_cm=0.4,
                                 scattering_loss_db_cm=0.3, absorption_loss_db_cm=0.1,
                                 q_intrinsic=9.78e5, fsr_ghz=199.7)},
        "grating": {"results": dict(period_um=1.1001, fill_factor=0.75, source_x_um=7.8,
                                    n_teeth=30, etch="full", etch_depth_um=0.7, efficiency_2d=0.313,
                                    fiber_angle_deg=12.0, width_um=12.0)},
    }
    return ctx


# ---- slab modes and the lateral overlap ---------------------------------------------- #
def test_asymmetric_slab_reduces_to_the_symmetric_one():
    from tarang.design import taper
    wl, n1, n2, d = 1.55, 2.0, 1.444, 0.7
    n = taper.slab_index_te(n1, n2, n2, d, wl)
    k0 = 2 * math.pi / wl
    k, g = k0 * math.sqrt(n1 ** 2 - n ** 2), k0 * math.sqrt(n ** 2 - n2 ** 2)
    assert math.tan(k * d / 2) == pytest.approx(g / k, rel=1e-6)       # symmetric TE0 condition
    n_air = taper.slab_index_te(n1, n2, 1.0, d, wl)
    assert 1.444 < n_air < n < n1                                      # air on top lowers n_eff


def test_lateral_overlap_peaks_near_2p7_waists_and_never_exceeds_one():
    from tarang.design import taper
    w0 = 5.2
    widths = np.linspace(6, 22, 33)
    eta = [taper.lateral_overlap(1.85, 1.444, w, 2 * w0, 1.55) for w in widths]
    best = widths[int(np.argmax(eta))]
    assert 2.4 * w0 < best < 3.0 * w0
    assert max(eta) < 1 and max(eta) > 0.97
    assert eta[0] < max(eta) and eta[-1] < max(eta)                    # too narrow, too wide
    # the profile: cosine inside, continuous at the wall, decaying outside
    y = np.linspace(-20, 20, 4001)
    n, psi = taper.lateral_mode(1.85, 1.444, 12.0, 1.55, y)
    assert 1.444 < n < 1.85 and abs(psi[2000] - 1) < 1e-12
    i_wall = int(np.argmin(np.abs(y - 6.0)))
    assert abs(psi[i_wall] - psi[i_wall + 1]) < 0.01 and abs(psi[-1]) < 1e-6


def test_shortest_taper_length_interpolates_between_the_swept_lengths():
    from tarang.design import taper
    curve = dict(length_um=[10, 20, 40, 80], loss_db=[1.0, 0.3, 0.08, 0.02])
    L = taper.shortest_length(curve, 0.05)
    t = math.log(0.08 / 0.05) / math.log(0.08 / 0.02)              # log(loss) is interpolated
    assert 40 < L < 80 and L == pytest.approx(40 + 40 * t)
    assert taper.shortest_length(curve, 0.001) is None
    assert taper.shortest_length(curve, 5.0) == 10


# ---- the taper stage --------------------------------------------------------------- #
@pytest.fixture
def fast_taper_model(monkeypatch):
    """A coarser converged 2D taper model (the stage runs it on every call)."""
    from tarang.design import taper
    monkeypatch.setattr(taper, "MODEL_CELLS", 240)


def test_taper_stage_picks_width_and_length_and_the_layout_uses_them(tmp_path, fake_web,
                                                                      fast_taper_model):
    from tarang.design import checks, layout, taper
    ctx = _ctx(tmp_path, run__max_total_cost=5.0)
    ctx.stage = "taper"
    res = taper.stage_taper(ctx)
    assert 13.0 <= res["grating_width_um"] <= 15.0 and res["eta_y"] > res["eta_y_before"]
    # fake EME: T = 1 - exp(-L / 40 um); 0.05 dB -> L = 40 ln(1/(1 - 10^-0.005)), rounded up to 10
    expect = -40 * math.log(1 - 10 ** (-0.005))
    assert res["eme_converged"] and res["length_source"] == "3D EME"
    assert res["taper_length_um"] == pytest.approx(math.ceil(expect / 10) * 10)
    assert res["taper_loss_db"] <= 0.05 + 1e-9
    assert res["model_length_um"] > 0 and res["staircase_error_db"] < 0.1
    assert (ctx.out / "figures" / "8a_grating_width.png").exists()
    assert (ctx.out / "figures" / "8b_taper_length.png").exists()
    assert (ctx.out / "tables" / "taper_length.csv").exists()
    ctx.state["stages"]["taper"] = {"results": res}
    p = layout.device_parameters(ctx)
    assert p["grating_width_um"] == res["grating_width_um"]
    assert p["taper_length_um"] == res["taper_length_um"]
    lb = checks.loss_budget(ctx, ctx.results("grating"))
    items = " ".join(r["item"] for r in lb)
    assert "lateral overlap" in items and "EME" in items
    assert lb[-1]["loss_db"] == pytest.approx(sum(r["loss_db"] for r in lb[:-1]), abs=0.01)
    # the EME run is cached: a second stage run loads it without uploading
    n = fake_web.names().count("batch_start")
    taper.stage_taper(ctx)
    assert fake_web.names().count("batch_start") == n


def test_taper_off_in_layout_keeps_the_previous_values(tmp_path, fake_web, fast_taper_model):
    from tarang.design import layout, taper
    ctx = _ctx(tmp_path, taper__use_in_layout=False, taper__run_eme=False)
    n = fake_web.names().count("batch_start")
    res = taper.stage_taper(ctx)
    assert fake_web.names().count("batch_start") == n                  # nothing on the cloud
    assert res["length_source"] == "2D model" and res["taper_length_um"] == res["model_length_um"]
    ctx.state["stages"]["taper"] = {"results": res}
    p = layout.device_parameters(ctx)
    assert p["grating_width_um"] == 12.0 and p["taper_length_um"] == ctx.cfg.grating.taper_length_um


def test_eme_simulation_is_valid_and_cached_like_any_other(tmp_path, fake_web):
    import tidy3d as td
    from tarang.design import taper
    ctx = _ctx(tmp_path)
    sim = taper.eme_taper(ctx, 14.0, 1.0, 200.0, [0.1, 0.5, 1.0], n_cells=10, n_modes=4)
    assert isinstance(sim, td.EMESimulation) and sim.symmetry == (0, -1, 0)
    run = cr.run_cached(sim, tmp_path / "c", "taper_test")
    assert isinstance(run.data, td.EMESimulationData)
    again = cr.run_cached(sim, tmp_path / "c", "taper_test")
    assert again.from_cache and isinstance(again.data, td.EMESimulationData)
    curve = taper.taper_transmission(again.data, 200.0, [0.1, 0.5, 1.0])
    assert curve["length_um"] == [20.0, 100.0, 200.0]
    assert curve["loss_db"][0] > curve["loss_db"][-1] > 0


def test_a_new_stage_after_checks_does_not_repeat_the_checks(tmp_path):
    from tarang.design.stages import STAGE_NAMES, settings_key
    from tarang.design.config import DesignConfig
    a, b = DesignConfig(), DesignConfig()
    b.taper.max_loss_db = 0.2
    assert STAGE_NAMES.index("taper") > STAGE_NAMES.index("checks")
    assert settings_key(a, "checks") == settings_key(b, "checks")
    assert settings_key(a, "layout") != settings_key(b, "layout")


# ---- apodisation ------------------------------------------------------------------- #
def _line_field(alpha, n_bloch, period, wl=1.55, L=33.0, ripple=0.3):
    """Field on the line above a uniform grating: decaying radiated wave with k_x =
    k0 n_B - 2 pi / period, plus the periodic near-field ripple."""
    x = np.arange(-1.0, L + 1.0, 0.02)
    kx = 2 * math.pi / wl * n_bloch - 2 * math.pi / period
    env = np.where(x > 0, np.exp(-alpha * x), 1.0)
    e = env * np.exp(1j * kx * x) * (1 + ripple * np.cos(2 * math.pi * x / period))
    return x, {"Ey": e, "Ex": 0.05 * e}


def test_scattering_strength_and_bloch_index_are_read_from_the_field_above():
    from tarang.design import apodise as ap
    for alpha, nb in ((0.05, 1.62), (0.2, 1.58)):
        x, comps = _line_field(alpha, nb, 1.1)
        r = ap.alpha_and_index(x, comps, 1.1, 1.55)
        assert r["alpha_per_um"] == pytest.approx(alpha, rel=0.03)
        assert r["n_bloch"] == pytest.approx(nb, abs=2e-3)
        # the opposite time convention (conjugate field) is undone by kx_sign
        conj = {k: np.conj(v) for k, v in comps.items()}
        r2 = ap.alpha_and_index(x, conj, 1.1, 1.55, kx_sign=-1.0)
        assert r2["n_bloch"] == pytest.approx(nb, abs=2e-3)


def test_uniform_gratings_reach_80_percent_and_shaping_reaches_more():
    from tarang.design import apodise as ap
    w_x = 5.2 / math.cos(math.radians(12))
    best = ap.best_profile(0.01, 0.5, w_x)
    assert best["uniform_best"]["eta_model"] == pytest.approx(0.80, abs=0.01)    # textbook value
    assert best["uniform_best"]["alpha"] * w_x == pytest.approx(0.7, abs=0.1)
    assert best["eta_model"] > 0.95
    weak_only = ap.best_profile(0.2, 0.5, w_x)                    # every tooth scatters strongly
    assert weak_only["eta_model"] < 0.82


def test_teeth_scatter_weakly_first_and_strongly_later():
    from tarang.design import apodise as ap
    ffs = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    alpha = lambda f: 0.02 + 0.5 * (0.9 - f)                       # noqa: E731  weaker at large ff
    period = lambda f: 1.10 + 0.2 * (0.75 - f)                     # noqa: E731
    w_x = 5.32
    teeth, ffl = ap.build_teeth(alpha, period, ffs, 8.0, w_x, 0.97, alpha(0.9), alpha(0.5))
    starts = [a for a, _ in teeth]
    assert all(b > a for a, b in zip(starts, starts[1:]))
    # weakest teeth first; the strongest scattering just past the beam centre
    assert ffl[0] == max(ffl) and min(ffl) < 0.65
    strongest = teeth[int(np.argmin(ffl))][0]
    assert 8.0 - w_x < strongest < 8.0 + 2 * w_x
    assert starts[-1] <= 8.0 + 2.5 * w_x + 1.2                     # stops when the beam has ended
    for (a, ln), f in zip(teeth, ffl):
        assert ln == pytest.approx(f * period(f), abs=1e-3)
    m = ap.teeth_model(teeth, lambda i: alpha(ffl[i]), 8.0, w_x)
    assert m["eta_model"] > 0.9 and m["remaining"] < 0.06


def _apod_ctx(tmp_path):
    ctx = _ctx(tmp_path, run__max_total_cost=10.0, apodise__run=True)
    ctx.state["stages"]["grating"]["results"].update(fiber_tilt="towards")
    return ctx


def test_apodise_stage_designs_checks_and_feeds_the_layout(tmp_path, fake_web, monkeypatch):
    from tarang.design import apodise as ap, layout
    from tarang.grating_coupler.grating_coupler import GratingCoupler

    def fake_line(data, f0):
        sim = data.simulation
        teeth = [s for s in sim.structures if s.name and s.name.startswith("Tooth_")]
        b = teeth[0].geometry.bounds
        P = teeth[1].geometry.bounds[0][0] - b[0][0]
        ff = (b[1][0] - b[0][0]) / P
        alpha = 0.02 + 0.6 * (0.9 - ff)
        nb = 1.55 / P + math.sin(math.radians(12))               # phase-matched at this period
        return _line_field(alpha, nb, P)
    monkeypatch.setattr(ap, "read_line", fake_line)

    def spectrum(self, folder):
        wl = np.linspace(1.45, 1.65, 101)
        top = 0.40 if self.config.teeth else 0.313
        return wl, top * np.exp(-((wl - 1.5505) / 0.03) ** 2)
    monkeypatch.setattr(GratingCoupler, "get_spectrum", spectrum)
    ctx = _apod_ctx(tmp_path)
    res = ap.stage_apodise(ctx)
    assert len(res["scattering"]) >= 5
    for r in res["scattering"]:                    # phase-matched at the period it was run at
        assert r["period_matched_um"] == pytest.approx(r["period_run_um"], rel=1e-3)
    assert res["model_apodised"] > res["model_uniform"] + 0.02
    assert res["verified"] and res["better_than_uniform"]
    assert res["efficiency_2d"] == pytest.approx(0.40 * math.exp(-((1.55 - 1.5505) / 0.03) ** 2), rel=1e-3)
    assert (ctx.out / "figures" / "9a_apodise_design.png").exists()
    assert (ctx.out / "figures" / "9b_apodised_spectrum.png").exists()
    assert (ctx.out / "figures" / "9c_apodised_landscape.png").exists()     # Phase 12C
    ctx.state["stages"]["apodise"] = {"results": res}
    p = layout.device_parameters(ctx)
    assert p["grating_apodised"] and p["grating_teeth"] == len(res["teeth"])
    assert p["fiber_position_um"] == res["source_x_um"]
    files = layout._write_parameters(p, tmp_path)
    assert any(f.name == "grating_teeth.csv" for f in files)
    assert "p.grating_teeth_list = [" in (tmp_path / "parameters.m").read_text()
    pytest.importorskip("gdstk")
    gds = layout.build_gds(p, tmp_path)
    import gdstk
    lib = gdstk.read_gds(str(next(f for f in gds if f.name == "waveguide_gc.gds")))
    n_poly = len(lib.top_level()[0].get_polygons())
    assert n_poly == 1 + 2 * (1 + len(res["teeth"]))          # bus, 2 x (taper + teeth)


def test_apodisation_is_not_run_when_teeth_cannot_scatter_weakly(tmp_path, fake_web, monkeypatch):
    from tarang.design import apodise as ap
    monkeypatch.setattr(ap, "read_line", lambda data, f0: _line_field(0.3, 1.6, 1.1))
    ctx = _apod_ctx(tmp_path)
    n = fake_web.names().count("batch_start")
    res = ap.stage_apodise(ctx)
    assert res["verified"] is False
    assert fake_web.names().count("batch_start") == n + 1           # only the scattering runs


def test_apodised_teeth_build_a_valid_simulation_and_uniform_names_are_unchanged(tmp_path):
    from tarang.design.stages import grating_device
    ctx = _ctx(tmp_path)
    uni = grating_device(ctx, 1.1001, 0.75, 7.8, H=0.7, depth=0.7)
    assert uni._readable_name() == "gc2D_p1.1001_ff0.7500_e0.7000_n30_sx7.8000_1.550um"
    teeth = [(i * 1.1, 0.8 + 0.005 * i) for i in range(20)]
    apo = grating_device(ctx, 1.1001, 0.75, 7.8, H=0.7, depth=0.7, teeth=teeth)
    assert apo._readable_name().startswith("gc2D_apod20_")
    tooth = [s for s in apo.sim.structures if s.name == "Tooth_19"][0]
    assert tooth.geometry.size[0] == pytest.approx(0.8 + 0.095)
    assert tooth.geometry.center[0] == pytest.approx(19 * 1.1 + (0.895) / 2)


# ---- reflections ------------------------------------------------------------------- #
def _amps(fwd, bwd, f0=1.934e14):
    import types
    import xarray as xr
    a = xr.DataArray(np.array([[[fwd]], [[bwd]]], complex),
                     coords={"direction": ["+", "-"], "mode_index": [0], "f": [f0]},
                     dims=("direction", "mode_index", "f"))
    return types.SimpleNamespace(amps=a)


def test_arc_end_and_bus_reflections_from_mode_amplitudes():
    from tarang.design import checks
    data = {"Ring_Out_Right": _amps(0.1, 0.01), "Through_In": _amps(1.0, 0.003)}
    r = checks.reflections(data, 1.934e14)
    assert r["arc_end_reflection"] == pytest.approx(0.01)        # |0.01|^2 / |0.1|^2
    assert r["bus_reflection"] == pytest.approx(9e-6)


def test_the_reverse_grating_run_is_a_closed_box(tmp_path):
    import tidy3d as td
    from tarang.design import checks
    from tarang.design.stages import grating_device
    ctx = _ctx(tmp_path)
    dev = grating_device(ctx, 1.1001, 0.75, 7.8, H=0.7, depth=0.7)
    rev = checks._reverse_sim(dev, ctx)
    mons = {m.name: m for m in rev.monitors}
    assert {"up", "down", "past", "back"} <= set(mons)
    src = rev.sources[0]
    back, up, down, past = mons["back"], mons["up"], mons["down"], mons["past"]
    assert back.center[0] < src.center[0]                          # behind the source
    for plane in (up, down):                                       # top and bottom span the box
        assert plane.bounds[0][0] == pytest.approx(back.center[0])
        assert plane.bounds[1][0] == pytest.approx(past.center[0])
    assert isinstance(src, td.ModeSource) and src.direction == "+"


def test_power_budget_includes_what_is_scattered_backwards():
    import types
    import xarray as xr
    from tarang.design import checks
    wl = np.array([1.50, 1.55, 1.60])
    f = 2.99792458e14 / wl
    flux = lambda v: types.SimpleNamespace(flux=xr.DataArray(np.array(v), coords={"f": f}, dims="f"))  # noqa: E731
    amps = xr.DataArray(np.sqrt([[0.01, 0.12, 0.2]]).reshape(1, 1, 3),
                        coords={"direction": ["-"], "mode_index": [0], "f": f},
                        dims=("direction", "mode_index", "f"))
    data = {"up": flux([0.5, 0.49, 0.4]), "down": flux([-0.3, -0.32, -0.3]),
            "past": flux([0.05, 0.03, 0.05]), "back": flux([-0.12, -0.15, -0.24]),
            "wg": types.SimpleNamespace(amps=amps)}
    rev = types.SimpleNamespace(monitors=[types.SimpleNamespace(name=n)
                                          for n in ("wg", "up", "down", "past", "back")])
    t = checks._power_budget(data, rev, {"final": (wl, np.array([0.3, 0.313, 0.2]))}, 1.55)["at_target"]
    assert t["scattered_back"] == pytest.approx(0.15 - 0.12)
    assert t["accounted"] == pytest.approx(0.49 + 0.32 + 0.03 + 0.15)


# ---- figures, pictures, labels ------------------------------------------------------ #
def test_k_equals_one_is_found_between_samples():
    from tarang.design import checks
    wl = np.array([1.50, 1.525, 1.55, 1.575])
    K = np.array([0.6, 0.8, 1.04, 1.3])
    (x,) = checks.k_crossings(wl, K)
    lk = np.log(K)
    assert x == pytest.approx(1.525 + 0.025 * (-lk[1]) / (lk[2] - lk[1]))
    assert checks.k_crossings(wl, K * 3) == []


def _write_band_table(ctx):
    import csv
    p = ctx.out / "tables"
    p.mkdir(parents=True, exist_ok=True)
    with open(p / "coupling_vs_wavelength.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["design", "gap_um", "wavelength_um", "coupled_power", "K"])
        for d, g, s in (("critical", 0.47, 1.0), ("over-coupled K = 1.5", 0.431, 1.5)):
            for wl in np.linspace(1.435, 1.685, 11):
                k = s * math.exp(4.5 * (wl - 1.55) / 0.25 * 1.2)
                w.writerow([d, g, wl, k * 0.0062, k])


def test_old_runs_get_their_figures_redrawn_once_from_saved_tables(tmp_path):
    from tarang.design import checks
    ctx = _ctx(tmp_path)
    ctx.state["figure_version"] = 1                                # made before Phase 12B
    _write_band_table(ctx)
    done = checks.redraw_saved_figures(ctx)
    assert "7a_coupling_vs_wavelength" in done and "3b_bent_mode" in done
    assert (ctx.out / "figures" / "7a_coupling_vs_wavelength.png").exists()
    assert ctx.state["figure_version"] == checks.FIGURE_VERSION
    assert checks.redraw_saved_figures(ctx) == []                  # only once
    from tarang.design.config import DesignConfig
    from tarang.design.stages import Context
    from tarang.design.ui import Prompter
    fresh = Context(DesignConfig(), Prompter(False, out=io.StringIO()), tmp_path / "new")
    assert fresh.state["figure_version"] == checks.FIGURE_VERSION  # new runs never redraw


def test_every_cloud_run_gets_a_geometry_picture_named_like_its_folder(tmp_path):
    from tarang.design.stages import grating_device
    ctx = _ctx(tmp_path)
    d = grating_device(ctx, 1.1001, 0.75, 7.8, H=0.7, depth=0.7)
    jobs = {"final": (d.sim, d._readable_name())}
    assert ctx.sim_pictures(jobs) == 1
    folder, _ = cr.cache_paths(ctx.out / "figures" / "sims", d._readable_name(), d.sim)
    assert folder.with_suffix(".png").exists()
    assert ctx.sim_pictures(jobs) == 0                             # kept, not redrawn
    ctx.cfg.run.sim_pictures = False
    assert ctx.sim_pictures({"x": (d.sim, "other")}) == 0


def test_batch_links_name_the_case_they_belong_to(tmp_path, fake_web):
    import tidy3d as td
    from tarang.design.stages import grating_device
    ctx = _ctx(tmp_path)
    d = grating_device(ctx, 1.1001, 0.75, 7.8, H=0.7, depth=0.7)
    d2 = grating_device(ctx, 1.1001, 0.75, 7.8, H=0.7, depth=0.7, angle=11.0)
    assert d._readable_name() == d2._readable_name()                # same name, two cases
    with ctx.cloud_progress("grating checks", 2):
        cr.run_batch_cached({"angle 11°": (d2.sim, d2._readable_name()),
                             "final": (d.sim, d._readable_name())}, tmp_path / "c")
    out = ctx.ui.out.getvalue()
    assert "angle 11° (gc2D_p1.1001" in out and "final (gc2D_p1.1001" in out
    assert isinstance(d.sim, td.Simulation)


def test_runs_without_links_are_explained_and_get_pictures(tmp_path, fake_web, capsys,
                                                           monkeypatch):
    from tarang.design import credits
    from tarang.design.stages import grating_device
    monkeypatch.setattr(cr, "account_status", lambda timeout=15.0: None)
    ctx = _ctx(tmp_path)
    d = grating_device(ctx, 1.1001, 0.75, 7.8, H=0.7, depth=0.7)
    run = cr.run_cached(d.sim, ctx.out / "cloud" / "grating", d._readable_name())
    (run.path.parent / cr.TASK_FILE).unlink()                    # made with another account
    ctx.save_state()
    credits.main(ctx.cfg, ctx.out, show=False, runs=True)
    text = capsys.readouterr().out
    assert "not in the task list of this account" in text and "this API key cannot see" in text
    assert (ctx.out / "figures" / "sims" / f"{run.path.parent.name}.png").exists()


def test_apodise_stage_reads_periods_right_in_either_time_convention(tmp_path, fake_web, monkeypatch):
    from tarang.design import apodise as ap

    def conj_line(data, f0):
        teeth = [s for s in data.simulation.structures if s.name and s.name.startswith("Tooth_")]
        b = teeth[0].geometry.bounds
        P = teeth[1].geometry.bounds[0][0] - b[0][0]
        x, comps = _line_field(0.3, 1.55 / P + math.sin(math.radians(12)), P)
        return x, {k: np.conj(v) for k, v in comps.items()}
    monkeypatch.setattr(ap, "read_line", conj_line)
    res = ap.stage_apodise(_apod_ctx(tmp_path))
    for r in res["scattering"]:
        assert r["period_matched_um"] == pytest.approx(r["period_run_um"], rel=1e-3)


def test_new_stages_run_through_the_design_session(tmp_path, fake_web, monkeypatch, fast_taper_model):
    from tarang.design import apodise as ap, run
    from tarang.design.ui import Prompter
    monkeypatch.setattr(ap, "read_line", lambda data, f0: _line_field(0.3, 1.6, 1.1))
    ctx = _ctx(tmp_path, run__max_total_cost=10.0)
    ctx.save_state()
    out = io.StringIO()
    # first design run: the apodised grating is off by default; only the uniform one is used
    n = fake_web.names().count("batch_start")
    done = run(ctx.cfg, ctx.out, interactive=False, only=["taper", "apodise"],
               prompter=Prompter(False, out=out))
    st = done.state["stages"]
    assert st["taper"]["results"]["taper_length_um"] > 0
    assert st["apodise"]["results"] == {"disabled": True}
    assert fake_web.names().count("batch_start") == n + 1             # the taper EME only
    assert "[8/13] Taper" in out.getvalue() and "[9/13] Apodised grating" in out.getvalue()
    assert "not requested" in out.getvalue()
    # later: turned on, it runs in the same session; the taper is not repeated
    done.cfg.apodise.run = True
    out2 = io.StringIO()
    again = run(done.cfg, ctx.out, interactive=False, prompter=Prompter(False, out=out2),
                only=["taper", "apodise"])
    assert "Taper: grating width for the fibre and taper length (EME): done earlier" in out2.getvalue()
    assert again.state["stages"]["apodise"]["results"]["verified"] is False   # strong teeth here


@pytest.mark.parametrize("module", ["full_ring_resonator", "half_ring_resonator",
                                    "partial_ring_resonator"])
def test_every_ring_builder_can_have_a_flat_box_and_keeps_its_old_geometry(module):
    import importlib
    import tidy3d as td
    mod = importlib.import_module(f"tarang.geometries.{module}")
    kw = dict(radius=10.0, gap=0.4, wg_width=1.0, wg_height=0.7,
              core_medium=td.Medium(permittivity=4.0), box_medium=td.Medium(permittivity=2.08),
              wavelength_um=1.55, fine_steps_per_wvl=8, coarse_steps_per_wvl=6)
    if module == "partial_ring_resonator":
        kw["span_angle_rad"] = 0.8
    default = mod.get_passive_simulation(**kw)
    legacy = mod.get_passive_simulation(**kw, inner_box_recess_um=0.1)
    flat = mod.get_passive_simulation(**kw, inner_box_recess_um=0.0)
    assert cr.sim_fingerprint(default.sim) == cr.sim_fingerprint(legacy.sim)
    hole = next(s for s in flat.sim.structures if s.name == "Ring_Inner_Hole").geometry
    zmin = hole.bounds[0][2]
    assert zmin == pytest.approx(0.0, abs=1e-9)                      # no pocket in the BOX
    old = next(s for s in legacy.sim.structures if s.name == "Ring_Inner_Hole").geometry
    assert old.bounds[0][2] == pytest.approx(-0.1)


# ---- Phase 12B v2: converged taper, robust apodisation, EME pictures ----------------- #
def test_graded_cells_change_the_width_by_the_same_ratio():
    from tarang.design import taper
    x = taper.cell_boundaries(14.5, 1.0, 40, 300.0)
    assert x[0] == 0 and x[-1] == 300.0 and np.all(np.diff(x) > 0)
    w = 14.5 + (1.0 - 14.5) * x / 300.0
    assert np.allclose(w[1:] / w[:-1], (1 / 14.5) ** (1 / 40))
    assert np.diff(x)[0] > 5 * np.diff(x)[-1]                  # short cells at the narrow end


@pytest.mark.slow
def test_2d_taper_model_is_converged_and_falls_with_length():
    from tarang.design import taper
    L = [10, 30, 60, 100, 200, 300]
    fine = taper.eim_taper(1.8511, 1.444, 14.5, 1.0, 1.55, L, n_cells=600)
    finer = taper.eim_taper(1.8511, 1.444, 14.5, 1.0, 1.55, L, n_cells=1200)
    assert np.allclose(fine["loss_db"], finer["loss_db"], atol=0.01)
    assert taper.rise_db(fine) < 0.005                         # falls with length
    assert fine["loss_db"][0] > 3 and fine["loss_db"][-1] < 0.06
    # a coarse staircase is what made the 3D EME jump with length
    coarse = taper.eim_taper(1.8511, 1.444, 14.5, 1.0, 1.55, L, n_cells=20, n_modes=8)
    assert np.max(np.abs(np.array(coarse["loss_db"]) - fine["loss_db"])) > 0.05


def test_rise_finds_a_loss_that_increases_with_length():
    from tarang.design import taper
    assert taper.rise_db(dict(length_um=[10, 20, 30], loss_db=[3.0, 1.0, 0.5])) == 0
    measured = [7.919, 6.498, 5.85, 6.429, 2.41, 1.964, 0.715, 0.895, 0.201, 0.871, 0.461,
                1.081, 0.418, 0.309]                           # the 30-cell EME of Phase 12B v1
    L = [5, 10, 15, 20, 30, 40, 60, 80, 100, 150, 200, 300, 400, 500]
    assert taper.rise_db(dict(length_um=L, loss_db=measured)) == pytest.approx(0.88, abs=0.01)


def test_unconverged_eme_falls_back_to_the_2d_model(tmp_path, fake_web, fast_taper_model, monkeypatch):
    from tarang.design import checks, taper
    real = taper.taper_transmission

    def jumpy(data, L0, scales):
        c = real(data, L0, scales)
        loss = [0.2 + (0.8 if i % 2 else 0.0) for i in range(len(c["loss_db"]))]
        return dict(length_um=c["length_um"], transmission=[10 ** (-v / 10) for v in loss], loss_db=loss)
    monkeypatch.setattr(taper, "taper_transmission", jumpy)
    ctx = _ctx(tmp_path, run__max_total_cost=5.0)
    res = taper.stage_taper(ctx)
    assert res["eme_converged"] is False and res["eme_rise_db"] == pytest.approx(0.8)
    assert res["length_source"] == "2D model" and res["taper_length_um"] == res["model_length_um"]
    ctx.state["stages"]["taper"] = {"results": res}
    items = " ".join(r["item"] for r in checks.loss_budget(ctx, ctx.results("grating")))
    assert "(2D model)" in items and "not simulated" not in items


def _measured_scattering():
    """The fill-factor table of the 1550 nm design (full etch, air top), from its ten
    uniform-grating reverse runs analysed with the forward + backward wave fit."""
    tab = [(0.5, 1e-6, 1.4078, 1.2923, 0.681), (0.55, 0.0776, 1.4636, 1.2487, 0.487),
           (0.6, 0.0513, 1.4750, 1.2079, 0.240), (0.65, 0.0638, 1.5062, 1.1697, 0.127),
           (0.7, 0.0625, 1.5583, 1.1338, 0.059), (0.75, 0.0478, 1.6185, 1.1001, 0.030),
           (0.8, 0.0279, 1.6766, 1.0683, 0.022), (0.85, 0.0132, 1.7289, 1.0383, 0.027),
           (0.9, 0.0057, 1.7787, 1.0100, 0.050), (0.908, 0.0044, 1.7859, 1.0056, 0.058)]
    return [dict(fill_factor=f, alpha_per_um=a, n_bloch=n, period_run_um=p, window_um=26 * p,
                 residual=r) for f, a, n, p, r in tab]


def test_only_one_monotonic_branch_of_fill_factors_is_used(tmp_path):
    from tarang.design import apodise as ap, report
    rows = ap.validate(_measured_scattering(), 1.55, 1.0, 1.0, 1.8511)
    why = {r["fill_factor"]: r["why"] for r in rows}
    assert "2nd order" in why[0.5] and "2nd order" in why[0.6] and "not two waves" in why[0.55]
    assert all(not why[f] for f in (0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.908))
    branch = ap.weak_branch(rows)
    assert [r["fill_factor"] for r in branch] == [0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.908]
    alpha = dict((r["fill_factor"], r["alpha_per_um"]) for r in branch)
    ffs = [r["fill_factor"] for r in branch]
    teeth, ffl = ap.build_teeth(lambda f: float(np.interp(f, ffs, [alpha[v] for v in ffs])),
                                lambda f: 1.1, ffs, 9.0, 5.32, 0.97, 0.0044, 0.0638)
    assert min(ffl) >= 0.65 and ffl[0] == pytest.approx(0.908)     # weakest teeth first
    # the report explains a stage that found too few usable points (no crash)
    few = ap.validate(_measured_scattering()[:4], 1.55, 1.0, 1.0, 1.8511)
    text = "\n".join(report._apodise_section(_ctx(tmp_path), dict(scattering=few, skipped=True,
                                                                     verified=False)))
    assert "Fewer than three usable" in text and "FF 0.5 (decay too weak" in text
    with pytest.raises(ValueError):                                  # both branches mixed
        ap.build_teeth(lambda f: [0.02, 0.09, 0.03][[0.5, 0.65, 0.85].index(f)] if f in (0.5, 0.65, 0.85)
                       else 0.05, lambda f: 1.1, [0.5, 0.65, 0.85], 9.0, 5.32, 0.97, 0.02, 0.09)


def test_the_wave_reflected_at_the_grating_end_does_not_spoil_the_fit():
    from tarang.design import apodise as ap
    L, P, alpha, nb = 33.0, 1.1, 0.01, 1.73                       # a weak grating
    x, comps = _line_field(alpha, nb, P, L=L)
    kx = 2 * math.pi / 1.55 * nb - 2 * math.pi / P
    back = 0.25 * np.exp(-alpha * (L - x)) * np.exp(-1j * kx * x + 0.7j)
    beat = {k: v + back * (1.0 if k == "Ey" else 0.05) for k, v in comps.items()}
    e2 = np.abs(beat["Ey"]) ** 2
    assert e2[(x > 20) & (x < 30)].min() < 0.7 * e2[(x > 20) & (x < 30)].max()   # deep fringes
    r = ap.alpha_and_index(x, beat, P, 1.55, length=L)
    assert r["alpha_per_um"] == pytest.approx(alpha, rel=0.05)
    assert r["n_bloch"] == pytest.approx(nb, abs=2e-3)
    assert r["residual"] < 0.01 and r["backward_ratio"] == pytest.approx(0.25, rel=0.05)


def test_fit_quality_is_reported_for_a_clean_and_a_messy_field():
    from tarang.design import apodise as ap
    x, comps = _line_field(0.05, 1.62, 1.1)
    clean = ap.alpha_and_index(x, comps, 1.1, 1.55)
    assert clean["residual"] < 0.01
    k0 = 2 * math.pi / 1.55
    other = {k: v + 0.8 * np.abs(v) * np.exp(-1j * 0.95 * k0 * x) for k, v in comps.items()}
    messy = ap.alpha_and_index(x, other, 1.1, 1.55)
    assert messy["residual"] > 0.2                                   # another wave above
    assert ap.second_order_up(1.3653, 1.2923, 1.55, 1.0) and not ap.second_order_up(1.6187, 1.1001, 1.55, 1.0)


def test_apodised_peak_is_moved_onto_the_design_wavelength(tmp_path, fake_web, monkeypatch):
    from tarang.design import apodise as ap
    from tarang.grating_coupler.grating_coupler import GratingCoupler

    def fake_line(data, f0):
        teeth = [s for s in data.simulation.structures if s.name and s.name.startswith("Tooth_")]
        b = teeth[0].geometry.bounds
        P = teeth[1].geometry.bounds[0][0] - b[0][0]
        ff = (b[1][0] - b[0][0]) / P
        return _line_field(0.02 + 0.6 * (0.9 - ff), 1.55 / P + math.sin(math.radians(12)), P)
    monkeypatch.setattr(ap, "read_line", fake_line)
    designed = {}
    real_build = ap.build_teeth

    def build(*a, **k):
        teeth, ffl = real_build(*a, **k)
        designed["teeth"] = teeth
        return teeth, ffl
    monkeypatch.setattr(ap, "build_teeth", build)

    def spectrum(self, folder):
        wl = np.linspace(1.45, 1.65, 801)
        if not self.config.teeth:
            return wl, 0.313 * np.exp(-((wl - 1.55) / 0.03) ** 2)
        scale = self.config.teeth[-1][1] / designed["teeth"][-1][1]
        peak = 1.567 + 0.97 * (scale - 1.0)                 # 17 nm long at the designed periods
        return wl, 0.40 * np.exp(-((wl - peak) / 0.03) ** 2)
    monkeypatch.setattr(GratingCoupler, "get_spectrum", spectrum)
    res = ap.stage_apodise(_apod_ctx(tmp_path))
    assert res["verified"]
    assert res["scale_centred"] == pytest.approx(1 - 0.017 / 0.97, abs=2e-3)
    assert len(res["check"]) == 8                                   # 5 + 3 with the peak centred
    assert abs(res["peak_wavelength_nm"] - 1550) < 1.5
    assert res["scale"] == pytest.approx(res["scale_centred"])


def test_run_pictures_load_eme_simulations(tmp_path):
    import tidy3d as td
    from tarang.design import credits, taper
    ctx = _ctx(tmp_path)
    sim = taper.eme_taper(ctx, 14.5, 1.0, 100.0, [0.5, 1.0], n_cells=12, n_modes=4)
    folder = tmp_path / "cloud" / "taper" / "taper_test_0123456789"
    folder.mkdir(parents=True)
    sim.to_file(str(folder / "simulation.json"))
    assert isinstance(credits.load_simulation(folder / "simulation.json"), td.EMESimulation)
    n = credits.draw_run_pictures(tmp_path, [dict(folder=str(folder), name="taper_test")])
    assert n == 1 and (tmp_path / "figures" / "sims" / "taper_test_0123456789.png").exists()


def test_the_narrowest_trench_limits_the_teeth_and_is_not_paid_for(tmp_path, fake_web, monkeypatch):
    from tarang.design import apodise as ap
    from tarang.grating_coupler.grating_coupler import GratingCoupler

    def fake_line(data, f0):
        teeth = [s for s in data.simulation.structures if s.name and s.name.startswith("Tooth_")]
        b = teeth[0].geometry.bounds
        P = teeth[1].geometry.bounds[0][0] - b[0][0]
        ff = (b[1][0] - b[0][0]) / P
        return _line_field(0.02 + 0.6 * (0.9 - ff), 1.55 / P + math.sin(math.radians(12)), P)
    monkeypatch.setattr(ap, "read_line", fake_line)
    monkeypatch.setattr(GratingCoupler, "get_spectrum", lambda self, folder: (
        np.linspace(1.45, 1.65, 101), 0.35 * np.exp(-((np.linspace(1.45, 1.65, 101) - 1.55) / 0.03) ** 2)))
    ctx = _ctx(tmp_path, run__max_total_cost=10.0, apodise__min_trench_um=0.2, apodise__run=True)
    ctx.state["stages"]["grating"]["results"].update(fiber_tilt="towards")
    res = ap.stage_apodise(ctx)
    runs = [r["fill_factor"] for r in res["scattering"]]
    assert max(runs) <= 0.82                                         # 0.85-0.908 never run
    assert all(r["trench_um"] >= 0.2 - 1e-6 for r in res["scattering"])
    assert res["min_trench_um"] == 0.2 and res["narrowest_trench_um"] >= 0.2 * 0.98
    assert "Narrowest trench allowed: 200 nm" in ctx.ui.out.getvalue()
