"""Phase 11: fibre orientation, grating spectra, ring-gap coupling measures, progress and
account checks, assumptions with sources and the benchmark pack. No FlexCredits, ~30 s.

    python -m pytest tests/test_phase11.py -q
"""
import csv
import datetime as dt
import io
import math
import types
from pathlib import Path

import numpy as np
import pytest
import tidy3d as td

from tarang.grating_coupler import (GratingConfig, GratingCoupler, phase_matched_period,
                                    spectrum_summary)

SIN = td.Medium(permittivity=1.996 ** 2)
SIO2 = td.Medium(permittivity=1.444 ** 2)


def _gc(**kw):
    base = dict(wavelength_um=1.55, theta_deg=12.0, mfd=10.4, source_x=6.0, source_gap=2.0,
                core_height=0.7, top_cladding_height=2.0, box_height=4.0, handle_height=3.0,
                etch_depth=0.7, period=1.2, ff=0.7, n_teeth=20, w_grating=12.0,
                pol_angle=np.pi / 2, buffer=3.0)
    base.update(kw)
    return GratingCoupler.from_config(GratingConfig(**base), core_medium=SIN, box_medium=SIO2)


# ---- fibre orientation ------------------------------------------------------------------ #
@pytest.mark.parametrize("tilt, sign", [("towards", -1), ("away", +1)])
def test_fiber_tilt_sets_the_beam_direction_and_where_it_lands(tilt, sign):
    gc = _gc(fiber_tilt=tilt)
    src = gc.sim.sources[0]
    d = np.array(src._dir_vector)
    assert np.sign(d[0]) == sign and d[2] < 0              # down; towards = towards -x (waveguide)
    assert abs(math.degrees(math.atan2(abs(d[0]), -d[2])) - 12.0) < 1e-9
    t = (src.center[2] - 0.7) / -d[2]
    assert abs(src.center[0] + t * d[0] - 6.0) < 1e-9       # beam centre meets the core at source_x
    assert abs(src._pol_vector[1]) > 0.999                   # TE: E along the grating lines


def test_towards_is_the_default_and_the_sign_of_theta_is_ignored():
    from tarang.utilities.cloud_run import sim_fingerprint
    assert GratingConfig(**{**dict(wavelength_um=1.55, theta_deg=12.0, mfd=10.4, source_x=6.0,
                                   source_gap=2.0, core_height=0.7, top_cladding_height=2.0,
                                   box_height=4.0, handle_height=3.0, etch_depth=0.7, period=1.2,
                                   ff=0.7, n_teeth=20, w_grating=12.0, pol_angle=1.57,
                                   buffer=3.0)}).fiber_tilt == "towards"
    assert sim_fingerprint(_gc(theta_deg=-12.0).sim) == sim_fingerprint(_gc(theta_deg=12.0).sim)
    with pytest.raises(ValueError):
        _gc(fiber_tilt="sideways")


def test_the_mode_monitor_records_a_band_around_the_wavelength():
    f = np.asarray(_gc().sim.monitors[0].freqs)
    assert abs(td.C_0 / f.max() - 1.45) < 1e-9 and abs(td.C_0 / f.min() - 1.65) < 1e-9
    with pytest.raises(ValueError):
        _gc(bandwidth_um=0.0)


def test_phase_matching_formula_for_both_orientations():
    s = math.sin(math.radians(12))
    assert abs(phase_matched_period(1.4256, 1.55, 12, "towards") - 1.55 / (1.4256 - s)) < 1e-12
    assert abs(phase_matched_period(1.4256, 1.55, -12, "away") - 1.55 / (1.4256 + s)) < 1e-12
    # the pre-v0.1.0 example (away, 0.8201 um) is the same grating as 1.052 um towards
    n = 1.55 / 0.8201 - s
    assert abs(phase_matched_period(n, 1.55, 12, "towards") - 1.0516) < 1e-3


# ---- spectra ---------------------------------------------------------------------------- #
def test_spectrum_summary_peak_value_and_bandwidths():
    wl = np.linspace(1.45, 1.65, 401)
    sigma = 0.02
    eta = 0.4 * np.exp(-((wl - 1.56) ** 2) / (2 * sigma ** 2))
    s = spectrum_summary(wl, eta, 1.55)
    assert abs(s["peak_wavelength_um"] - 1.56) < 1e-3 and abs(s["peak_efficiency"] - 0.4) < 1e-6
    assert abs(s["efficiency_at_target"] - 0.4 * math.exp(-0.125)) < 1e-4
    for db in (1, 3):                                       # Gaussian: 2 sigma sqrt(2 ln 10^(db/10))
        expect = 2 * sigma * math.sqrt(2 * math.log(10 ** (db / 10))) * 1e3
        assert abs(s[f"bandwidth_{db}db_nm"] - expect) < 0.5 and s[f"bandwidth_{db}db_closed"]
    edge = spectrum_summary(wl, np.exp(-((wl - 1.70) ** 2) / 0.002), 1.55)
    assert not edge["peak_inside"] and not edge["bandwidth_1db_closed"]


# ---- ring-bus coupling: a floor in 1 - T must not move the gap -------------------------- #
class _FakeRing:
    """Stands in for a partial-ring wrapper: coupled power sin^2(theta0 exp(-gamma g)), and
    a through power that also loses a gap-independent 0.2 % (as seen in real runs)."""
    theta0, gamma, floor = 0.96, 5.3, 0.002
    _cache_label = "fakeRing"

    def __init__(self, radius, gap, wavelength_um, span_angle_rad, **kw):
        self.radius, self.gap, self.wavelength_um, self.span_angle_rad = radius, gap, wavelength_um, span_angle_rad
        box = lambda name, y: td.Structure(geometry=td.Box(center=(0, y, 0), size=(4, 0.5, 0.5)),  # noqa: E731
                                           medium=SIN, name=name)
        mon = lambda name: td.ModeMonitor(center=(1, 0, 0), size=(0, 2, 2), freqs=[2e14],  # noqa: E731
                                          mode_spec=td.ModeSpec(), name=name)
        self.sim = td.Simulation(
            size=(4, 4, 2), run_time=1e-13, grid_spec=td.GridSpec.uniform(dl=0.1),
            structures=[box("Ring_Outer", 1), box("Ring_Inner_Hole", 1.5), box("Bus_Waveguide", 0)],
            sources=[td.ModeSource(center=(-1, 0, 0), size=(0, 2, 2), direction="+",
                                   source_time=td.GaussianPulse(freq0=2e14, fwidth=2e13))],
            monitors=[mon("Through_Out"), mon("Ring_Out_Right")])

    def get_mode_fluxes(self, output_dir=None, verbose=False, force_rerun=False):
        ring = any(s.name == "Ring_Outer" for s in self.sim.structures)
        c = math.sin(self.theta0 * math.exp(-self.gamma * self.gap)) ** 2 if ring else 0.0
        return {"Through_Out": {"flux_forward": 1 - self.floor - c, "flux_backward": 0.0},
                "Ring_Out_Right": {"flux_forward": c, "flux_backward": 0.0}}


def _sweep(tmp_path, reference):
    from tarang.ring_resonator.ring_gap_sweep import ring_gap_sweep
    return ring_gap_sweep([0.4, 0.5, 0.6, 0.7, 0.8, 0.9], tmp_path / reference, 107.5, 9.78e5,
                          470.0, SIN, SIO2, 1.0, 0.7, 1.55, 45.0, n_group=2.2227,
                          partial_builder=_FakeRing, reference=reference, save_plots=False,
                          save_field_maps=False)


def test_ring_port_and_bus_only_remove_the_floor_that_biases_the_gap(tmp_path):
    exact = {}
    for ref in ("source", "ring_port", "bus_only"):
        exact[ref] = _sweep(tmp_path, ref)
    target = exact["ring_port"]["target_cross_power"]
    theta_t = math.asin(math.sqrt(target))
    true_gap = math.log(_FakeRing.theta0 / theta_t) / _FakeRing.gamma
    assert abs(exact["ring_port"]["critical_gap_um"] - true_gap) < 2e-3
    assert abs(exact["bus_only"]["critical_gap_um"] - true_gap) < 2e-3
    assert exact["source"]["critical_gap_um"] > true_gap + 0.03     # the floor pushes it wide
    assert exact["bus_only"]["reference_through_power"] == pytest.approx(1 - _FakeRing.floor)
    assert exact["ring_port"]["reference"] == "ring_port"


# ---- progress lines and account check --------------------------------------------------- #
def test_progress_listener_sees_runs_and_cache_hits(tmp_path, fake_web):
    from tarang.utilities.cloud_run import progress_listener, run_cached
    events = []
    sim = _gc().sim
    with progress_listener(lambda e, info: events.append(e)):
        run_cached(sim, tmp_path, "gc")
        run_cached(sim, tmp_path, "gc")
    assert events == ["estimated", "running", "done", "cached"]


def test_expired_credits_stop_a_cloud_stage_before_anything_is_uploaded(tmp_path, fake_web,
                                                                       monkeypatch):
    import tarang.utilities.cloud_run as cr
    from tarang.design.config import DesignConfig
    from tarang.design.stages import Context
    from tarang.design.ui import Prompter
    past = dt.datetime(2026, 10, 12, 23, 59, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(cr.web, "account", lambda verbose=True: types.SimpleNamespace(
        credit=92.2, credit_expiration=past), raising=False)
    assert cr.account_status()["expired"]
    out = io.StringIO()
    ctx = Context(DesignConfig(), Prompter(interactive=False, out=out), tmp_path)
    assert ctx.account_ok(1.0) is False and "expired on 2026-10-12" in out.getvalue()
    future = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=30)
    monkeypatch.setattr(cr.web, "account", lambda verbose=True: types.SimpleNamespace(
        credit=0.5, credit_expiration=future), raising=False)
    assert ctx.account_ok(1.0) is False and ctx.account_ok(0.2) is True


# ---- settings, assumptions, ranking ------------------------------------------------------ #
def test_settings_file_with_a_byte_order_mark_loads(tmp_path):
    from tarang.design.config import DesignConfig
    path = tmp_path / "design.toml"
    path.write_bytes("﻿[grating]\nfiber_angle_deg = 12.0\n".encode("utf-8"))
    assert DesignConfig.load(path).grating.fiber_angle_deg == 12.0


def test_loss_presets_come_with_sources_and_your_values_are_marked():
    from tarang.design import assumptions
    from tarang.design.config import DesignConfig
    cfg = DesignConfig()
    rows = {r["quantity"]: r for r in assumptions.table(cfg)}
    assert rows["scattering loss"]["value"] == 0.3 and "Sci. Rep." in rows["scattering loss"]["source"]
    assert "Ikeda" in rows["Kerr index n2 of the core"]["source"]
    for p in assumptions.LOSS_PRESETS.values():
        assert p["source"] and p["scattering_db_cm"] > 0 and p["absorption_db_cm"] > 0
    cfg.ring.scattering_loss_db_cm = 0.8
    assert assumptions.loss_source(cfg, "scattering") == "your value (design.toml)"
    with pytest.raises(ValueError):
        assumptions.preset("unknown")


def test_candidates_tie_break_on_the_te_tm_split():
    from tarang.design.stages import _rank_targets
    def rec(W, split):
        return dict(H=0.7, W=W, D_TE0=300, guided_TE=1, guided_TM=1, ng_TE0=2.2, n_cut=1.444,
                    neff_TE=[1.70, 1.43], neff_TM=[1.70 - split, 1.42])
    rows = [rec(0.9, 0.01), rec(1.0, 0.03), rec(1.1, 0.05)]
    ranked = _rank_targets(rows, [0.7], [0.9, 1.0, 1.1])
    assert ranked[0]["W_um"] == 1.0                    # most target neighbours (2)
    assert [r["W_um"] for r in ranked[1:]] == [1.1, 0.9]   # then the larger TE/TM split


def test_nested_sampler_passes_the_fibre_orientation(tmp_path):
    from tarang.grating_coupler.nested_sampler import GratingCouplerBayesianOptimizer
    opt = GratingCouplerBayesianOptimizer(fixed_params=dict(theta_deg=12.0), materials={},
                                          output_dir=tmp_path, show_corner_plots=False)
    assert opt._orientation() == {"fiber_tilt": "towards"}
    opt.fixed_params.update(fiber_tilt="away", bandwidth_um=0.05)
    assert opt._orientation() == {"fiber_tilt": "away", "bandwidth_um": 0.05}


# ---- benchmark pack ----------------------------------------------------------------------- #
def test_report_writes_a_complete_benchmark_pack(tmp_path):
    import zipfile
    from test_design import _ctx, _with_cloud_results
    from tarang.design import report
    ctx = _with_cloud_results(_ctx(tmp_path))
    ctx.cfg.save(ctx.out / "design.toml")
    res = report.write(ctx)
    pack = Path(res["folder"])
    for name in ("README.md", "assumptions.csv", "results.csv", "design.toml"):
        assert (pack / name).exists(), name
    readme = (pack / "README.md").read_text(encoding="utf-8")
    for heading in ("## 0. Stack and materials", "## Assumed values", "## 1. Waveguide",
                    "## 2. Ring", "## 3. Ring–bus gap", "## 4. Grating coupler"):
        assert heading in readme, heading
    assert "towards the waveguide" in readme and "Lumerical" not in readme
    assert "threshold" in (pack / "results.csv").read_text(encoding="utf-8")
    assert zipfile.ZipFile(res["zip"]).namelist()
    text = (ctx.out / "report.md").read_text(encoding="utf-8")
    assert "## Assumed values" in text and "## Known limits" in text


# ---- fixes found in the first 1550 nm run with Phase 11 ----------------------------------- #
def test_describe_sim_reads_monitor_bands_stored_as_arrays(tmp_path):
    """Tidy3D stores a monitor's frequencies as a numpy array; the benchmark README must
    describe a saved grating simulation (it crashed on `array or []`)."""
    from tarang.design.benchmark import describe_sim
    path = tmp_path / "simulation.json"
    _gc().sim.to_file(str(path))
    sim = td.Simulation.from_file(str(path))
    assert any(isinstance(getattr(m, "freqs", None), np.ndarray) and len(m.freqs) > 1
               for m in sim.monitors)
    text = "\n".join(describe_sim(sim, 1.55))
    assert "wavelengths 1.4500-1.6500 µm" in text


def test_grating_stage_through_report_and_benchmark_pack(tmp_path, fake_web, monkeypatch):
    """The whole path that failed: a grating stage with a saved final simulation, then
    the report and the benchmark pack built from it."""
    import tarang.ring_resonator.ring_gap_sweep as rgs
    from test_design import _ctx, _fake_gap_fit, _fake_spectra
    from tarang.design import report, stages
    fake_web.cost = 0.01
    _fake_spectra(monkeypatch)
    monkeypatch.setattr(rgs, "ring_gap_sweep", _fake_gap_fit)
    ctx = _ctx(tmp_path, gap__gaps_um=[0.4, 0.5, 0.6, 0.7], grating__fill_factors=[0.5, 0.7, 0.9],
               run__max_total_cost=5.0)
    ctx.state["stages"]["gap"] = {"results": stages.stage_gap(ctx)}
    # a real saved partial-ring simulation, as the cloud cache leaves it (mode source, no
    # Gaussian beam: the case that failed in the second 1550 nm run)
    from tarang.geometries.partial_ring_resonator import get_active_simulation as partial
    ring_dir = ctx.out / "cloud" / "gap_sweep" / "partialRingResonator_R107.50um_gap600nm_1.550um"
    ring_dir.mkdir(parents=True)
    with pytest.warns(UserWarning, match="ends inside the simulation domain"):
        ring = partial(radius=107.5, gap=0.6, wg_width=1.0, wg_height=0.7, core_medium=SIN,
                       box_medium=SIO2, wavelength_um=1.55, span_angle_rad=math.radians(45),
                       fine_steps_per_wvl=8, coarse_steps_per_wvl=6)
    ring.sim.to_file(str(ring_dir / "simulation.json"))
    for name, func in (("grating", stages.stage_grating), ("layout", stages.stage_layout)):
        ctx.state["stages"][name] = {"results": func(ctx)}
    res = ctx.state["stages"]["grating"]["results"]
    assert Path(res["final_sim"]).exists()
    ctx.cfg.save(ctx.out / "design.toml")
    out = report.write(ctx)
    readme = (Path(out["folder"]) / "README.md").read_text(encoding="utf-8")
    assert "### Final grating-coupler simulation (2D)" in readme
    assert "### Ring–bus gap simulation (gap 0.6 µm shown" in readme
    assert (Path(out["folder"]) / "simulations" / "grating_final.json").exists()


def test_period_is_varied_again_when_the_best_fill_factor_moves(tmp_path, fake_web, monkeypatch):
    """Step B picks a fill factor other than its centre one: step B2 varies the period
    around it, so the period sensitivity belongs to the final fill factor."""
    from test_design import _ctx, _fake_spectra
    from tarang.design import stages
    fake_web.cost = 0.01
    calls = _fake_spectra(monkeypatch)              # best at fill factor 0.7
    ctx = _ctx(tmp_path, grating__fill_factors=[0.5, 0.65, 0.9], run__max_total_cost=5.0)
    res = stages.stage_grating(ctx)
    assert fake_web.names().count("batch_start") == 4          # A, B, B2 and C
    assert abs(res["fill_factor"] - 0.7) < 1e-9
    assert len({p for p, ff, _ in calls if abs(ff - 0.7) < 1e-9}) >= 5
    assert not any("varied at fill factor" in n for n in res["notes"])
    rows = (ctx.out / "tables" / "grating_sweep.csv").read_text().splitlines()
    assert any(r.startswith("B2,") for r in rows)


def test_sensitivity_panels_name_what_their_runs_held_fixed(tmp_path, monkeypatch):
    from tarang.design import visuals
    titles = []
    import matplotlib.axes
    orig = matplotlib.axes.Axes.set_title
    monkeypatch.setattr(matplotlib.axes.Axes, "set_title",
                        lambda self, t, *a, **k: titles.append(t) or orig(self, t, *a, **k))
    row = lambda p, f, x, e: dict(period_um=p, fill_factor=f, source_x_um=x,  # noqa: E731
                                  efficiency_at_target=e)
    ff_rows = [row(1.19, 0.6, 5.2, 0.2), row(1.10, 0.75, 5.2, 0.3)]
    period_rows = [row(1.12, 0.7, 5.2, 0.25), row(1.14, 0.7, 5.2, 0.28)]
    align_rows = [row(1.10, 0.75, 5.2, 0.30), row(1.10, 0.75, 7.8, 0.31)]
    best = row(1.10, 0.75, 7.8, 0.31)
    visuals.grating_sensitivity(ff_rows, period_rows, align_rows, best,
                                tmp_path / "s.png", 1.55)
    assert "FF = 0.70" in titles[1] and "P = 1.1000 µm, FF = 0.75" in titles[2]
    assert (tmp_path / "s.png").exists()


def test_a_run_loaded_twice_is_counted_once(tmp_path):
    from test_design import _ctx
    from tarang.utilities.cloud_run import _emit
    ctx = _ctx(tmp_path)
    with ctx.cloud_progress("gap sweep", 2):
        for fp in ("aaa", "aaa", "bbb", "bbb"):          # data, then the field map
            _emit("cached", name=f"run_{fp}", fingerprint=fp)
    text = ctx.ui.out.getvalue()
    assert "[2/2]" in text and "[3/2]" not in text


# ---- fixes found in the second 1550 nm run ------------------------------------------------ #
def test_describe_sim_without_a_gaussian_beam():
    """A ring simulation (mode source only) is described too; numpy was imported only in
    the Gaussian-beam branch, which left it unbound for every other simulation."""
    from tarang.design.benchmark import describe_sim
    sim = td.Simulation(
        size=(4, 3, 3), run_time=1e-13, grid_spec=td.GridSpec.auto(wavelength=1.55),
        structures=[td.Structure(geometry=td.Box(size=(td.inf, 1, 0.7)), medium=SIN)],
        sources=[td.ModeSource(center=(-1.5, 0, 0), size=(0, 2, 2), direction="+",
                               source_time=td.GaussianPulse(freq0=td.C_0 / 1.55,
                                                            fwidth=td.C_0 / 1.55 / 10),
                               mode_spec=td.ModeSpec(), mode_index=0)],
        monitors=[td.ModeMonitor(center=(1.5, 0, 0), size=(0, 2, 2), freqs=[td.C_0 / 1.55],
                                 mode_spec=td.ModeSpec(), name="through")])
    text = "\n".join(describe_sim(sim, 1.55))
    assert "through: ModeMonitor" in text and "at 1.5500 µm" in text


def test_spectrum_summary_finds_the_peak_between_samples_and_the_band_edges():
    wl = np.linspace(1.45, 1.65, 101)                        # 2 nm samples, as simulated
    eta = 0.3 * np.exp(-((wl - 1.5503) / 0.03) ** 2)
    s = spectrum_summary(wl, eta, 1.55)
    assert abs(s["peak_wavelength_um"] - 1.5503) < 0.0002   # well below the 2 nm spacing
    half = 0.03 * math.sqrt(math.log(10 ** 0.1))             # 1 dB half-width
    assert abs(s["band_1db_lo_um"] - (1.5503 - half)) < 3e-4
    assert abs(s["band_1db_hi_um"] - (1.5503 + half)) < 3e-4


def test_centring_puts_the_peak_on_the_design_wavelength(tmp_path, fake_web, monkeypatch):
    """The best efficiency at 1550 nm can sit on the flank of a spectrum peaking elsewhere
    (as in the second 1550 nm run: peak 1541.6 nm). Step D moves the period until the
    peak is at 1550 nm; with centre_peak = False the old choice is kept."""
    from test_design import _ctx
    from tarang.design import stages
    from tarang.grating_coupler.grating_coupler import GratingCoupler
    fake_web.cost = 0.01

    def spectrum(self, output_dir, verbose=False):
        # peak wavelength nonlinear in the period (so one secant step is not exact), a
        # peak height falling with the period, and a skewed shape
        self._run_sim(output_dir)
        c = self.config
        n_avg = c.ff * 1.85 + (1 - c.ff)
        x = c.period - 1.10
        peak_wl = c.period * (n_avg - math.sin(math.radians(12))) + 0.8 * x * x
        height = 0.33 - 3.0 * x
        wl = np.linspace(c.wavelength_um - c.bandwidth_um, c.wavelength_um + c.bandwidth_um, 101)
        width = np.where(wl < peak_wl, 0.035, 0.02)
        return wl, height * np.exp(-((wl - peak_wl) / width) ** 2)
    monkeypatch.setattr(GratingCoupler, "get_spectrum", spectrum)

    ctx = _ctx(tmp_path, grating__fill_factors=[0.75], grating__detuning=[0.97, 1.0, 1.03],
               run__max_total_cost=5.0)
    res = stages.stage_grating(ctx)
    assert abs(res["peak_wavelength_um"] - 1.55) <= 0.001
    lo, hi = res["band_1db_um"]
    assert lo < 1.55 < hi
    rows = (ctx.out / "tables" / "grating_sweep.csv").read_text().splitlines()
    assert any(r.startswith("D,") for r in rows)

    ctx2 = _ctx(tmp_path / "b", grating__fill_factors=[0.75], grating__detuning=[0.97, 1.0, 1.03],
                grating__centre_peak=False, run__max_total_cost=5.0)
    res2 = stages.stage_grating(ctx2)
    rows2 = list(csv.DictReader(open(ctx2.out / "tables" / "grating_sweep.csv")))
    assert not any(r["step"] == "D" for r in rows2)
    assert abs(res2["efficiency_2d"] - max(float(r["efficiency_at_target"]) for r in rows2)) < 1e-9


def test_a_simulation_that_cannot_be_described_does_not_stop_the_pack(tmp_path):
    from tarang.design.benchmark import _describe_file
    bad = tmp_path / "simulation.json"
    bad.write_text("{not json", encoding="utf-8")
    lines = _describe_file(bad, 1.55)
    assert len(lines) == 1 and "could not describe simulation.json" in lines[0]
