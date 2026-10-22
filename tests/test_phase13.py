"""Phase 13: credit limits at the start of a run, price-only and spend cap, apodisation
preference and linear ramp, the whole-device circuit model, the GUI's data layer and app."""
from __future__ import annotations

import csv
import io
import json
import math
import sys
import time
import types

import numpy as np
import pytest

from test_phase12b import _ctx


# ---- credit limits ------------------------------------------------------------------- #
def test_limits_are_capped_by_the_account_and_each_other():
    from tarang.design import limit_checks
    per_run, total = limit_checks(spent=4.0, credit=10.0, total_now=12.0)
    assert total(14.0) == "" and "at most 14.00" in total(14.5)
    assert "negative" in total(-1)
    assert per_run(2.0) == "" and "total limit" in per_run(13.0)
    assert "account balance" in limit_checks(0.0, 1.0, 5.0)[0](2.0)
    assert limit_checks(0.0, None, 5.0)[1](1e6) == ""          # balance unknown: not capped


def test_price_only_shows_estimates_and_runs_nothing_that_costs():
    from tarang.design.ui import Prompter
    ui = Prompter(False, out=io.StringIO(), price_only=True)
    assert ui.proceed("gap runs", 0.8, 10.0) is False
    assert ui.proceed("cached runs", 0.0, 10.0) is True            # free: results are read
    assert ui.priced == [("gap runs", 0.8), ("cached runs", 0.0)]
    assert "Price only: not run" in ui.out.getvalue()


def test_spend_cap_counts_down_and_skips_what_does_not_fit():
    from tarang.design.ui import Prompter
    ui = Prompter(False, out=io.StringIO(), spend_cap=1.0)
    assert ui.proceed("a", 0.6, 10.0) and ui.spend_cap == pytest.approx(0.4)
    assert ui.proceed("b", 0.5, 10.0) is False
    assert "--spend-cap" in ui.out.getvalue()
    assert ui.proceed("c", 0.4, 10.0) and ui.spend_cap == pytest.approx(0.0)


def test_command_line_has_price_only_and_spend_cap():
    from tarang.design.__main__ import main
    with pytest.raises(SystemExit):
        main(["--help"])


# ---- benchmark: EME simulations are described too ------------------------------------- #
def test_benchmark_describes_an_eme_simulation():
    from tarang.design import benchmark
    sim = types.SimpleNamespace(
        center=(0, 0, 0), size=(100, 10, 4), axis=0,
        boundary_spec=types.SimpleNamespace(**{a: types.SimpleNamespace(
            minus=types.SimpleNamespace(), plus=types.SimpleNamespace()) for a in "xyz"}),
        grid_spec=types.SimpleNamespace(grid_x=types.SimpleNamespace(min_steps_per_wvl=20),
                                        wavelength=1.55, override_structures=[]),
        eme_grid=types.SimpleNamespace(boundaries=list(range(121))), sweep_spec=object(),
        structures=[], sources=[], monitors=[])
    text = "\n".join(benchmark.describe_sim(sim, 1.55))
    assert "Eigenmode expansion along x: 120 cells" in text


# ---- apodisation: which design is kept -------------------------------------------------- #
def _cand(recipe, eff, bw):
    return dict(recipe=recipe, eff=eff, bw=bw)


def test_efficiency_preference_keeps_the_most_efficient():
    from tarang.design.apodise import choose
    cands = [_cand("measured", 0.30, 53), _cand("ramp", 0.33, 40)]
    c, better = choose(cands, 0.313, 35, "efficiency")
    assert c["recipe"] == "ramp" and better
    c, better = choose([_cand("measured", 0.299, 53)], 0.313, 35, "efficiency")
    assert not better                                             # Phase 12C's 1550 nm case


def test_bandwidth_preference_takes_the_wider_band_within_half_a_db():
    from tarang.design.apodise import choose
    c, better = choose([_cand("measured", 0.299, 53)], 0.313, 35, "bandwidth")
    assert better and c["recipe"] == "measured"                    # 0.2 dB less, 18 nm wider
    c, better = choose([_cand("measured", 0.25, 70)], 0.313, 35, "bandwidth")
    assert not better                                              # ~1 dB less: not accepted
    c, better = choose([_cand("a", 0.30, 45), _cand("b", 0.31, 60)], 0.313, 35, "bandwidth")
    assert c["recipe"] == "b" and better


def test_linear_ramp_falls_from_the_weakest_tooth_and_is_phase_matched():
    from tarang.design.apodise import ramp_teeth

    def period(f):
        return 1.0 + 0.2 * f
    teeth = ramp_teeth(0.8, 0.65, 0.01, 30, period, scale=1.02)
    assert teeth[0] == (0.0, round(0.8 * period(0.8) * 1.02, 4))
    starts = [a for a, _ in teeth]
    periods = np.diff(starts)
    ffs = [b / p for (_, b), p in zip(teeth, periods)]
    assert all(f2 <= f1 + 1e-3 for f1, f2 in zip(ffs, ffs[1:]))    # weak -> strong
    assert 0.65 - 1e-3 <= min(ffs) and max(ffs) <= 0.8 + 1e-3
    for f, p in zip(ffs, periods):                                  # each tooth phase-matched
        assert p == pytest.approx(period(f) * 1.02, abs=2e-3)
    tail = ramp_teeth(0.8, 0.65, 1.0, 10, period)                    # steep: clipped at f_min
    assert tail[-1][1] == pytest.approx(0.65 * period(0.65), abs=1e-4)


# ---- circuit model ---------------------------------------------------------------------- #
def test_star_product_gives_the_fabry_perot_formula():
    from tarang.design.circuit import chain
    f = np.linspace(193.0e12, 193.2e12, 40001)
    one = np.ones_like(f)
    R, eta = 0.04, 0.5
    T = np.abs(chain(f, eta * one, eta * one, R * one, 2.0 * one, 1e-3, 0.0, one)) ** 2
    assert T.max() == pytest.approx(eta ** 2 / (1 - R) ** 2, rel=1e-4)
    assert T.min() == pytest.approx(eta ** 2 / (1 + R) ** 2, rel=1e-4)
    T0 = np.abs(chain(f, eta * one, eta * one, 0 * one, 2.0 * one, 1e-3, 0.0, one)) ** 2
    assert np.allclose(T0, eta ** 2)


def test_ring_numbers_match_the_coupled_mode_picture():
    from tarang.design import physics
    from tarang.design.circuit import all_pass, resonance_numbers
    ng, R, wl, loss = 2.2233, 107.5, 1.55, 0.4
    L = 2 * math.pi * R * 1e-6
    a = 10 ** (-loss * L * 100 / 20)
    D1 = 2 * math.pi * 299792458.0 / (ng * L)
    om = 2 * math.pi * 299792458.0 / (wl * 1e-6)
    crit = resonance_numbers(D1, a, a, om)
    assert crit["Q_i"] == pytest.approx(physics.q_intrinsic(loss, ng, wl), rel=2e-3)
    assert crit["K"] == pytest.approx(1.0) and crit["Q_L"] == pytest.approx(crit["Q_i"] / 2, rel=1e-3)
    assert abs(all_pass(0.0, a, a)) ** 2 < 1e-20                    # critical: full dip
    k2 = 1.5 * (1 - a ** 2)                                         # K = 1.5
    over = resonance_numbers(D1, math.sqrt(1 - k2), a, om)
    assert over["K"] == pytest.approx(1.5, rel=5e-3)
    assert over["T_min"] == pytest.approx((0.5 / 2.5) ** 2, rel=1e-2)   # ((1-K)/(1+K))^2


def _device_ctx(tmp_path, band_curve=False, reflections=True):
    ctx = _ctx(tmp_path)
    loss, R = 0.4, 107.5
    rho = 1 - 10 ** (-loss * 2 * math.pi * R * 1e-4 / 10)
    ctx.state["stages"]["gap"] = {"results": dict(critical_gap_um=0.55, over_coupling=1.5,
                                                  over_coupled_gap_um=0.48, target_cross_power=rho)}
    wl = np.linspace(1.35, 1.75, 41)
    n = 1.7066 - 0.39 * (wl - 1.55) - 0.05 * (wl - 1.55) ** 2
    (ctx.out / "results").mkdir(parents=True, exist_ok=True)
    (ctx.out / "results" / "broadband.jsonl").write_text(json.dumps(dict(
        H=0.7, W=1.0, wl=wl.tolist(), neff_TE0=n.tolist(), n_cut=[1.444] * 41)) + "\n")
    (ctx.out / "tables").mkdir(parents=True, exist_ok=True)
    with open(ctx.out / "tables" / "grating_spectrum_final.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["wavelength_um", "efficiency", "efficiency_db"])
        for x in np.linspace(1.5, 1.6, 51):
            w.writerow([x, 0.313 * math.exp(-((x - 1.55) / 0.06) ** 2), 0])
    checks = {}
    if reflections:
        checks["grating"] = dict(power_budget=dict(wavelength_um=[1.5, 1.6], reflected=[0.01, 0.01]))
    if band_curve:
        ww = [1.45, 1.55, 1.65]
        checks["band"] = dict(curves={"critical": dict(gap_um=0.55, wavelength_um=ww,
                                                       coupled=[rho * 0.7, rho, rho * 1.4])})
    ctx.state["stages"]["checks"] = {"results": checks}
    ctx.state["stages"]["taper"] = {"results": dict(eta_y=0.99, taper_loss_db=0.048, taper_length_um=290.0)}
    return ctx


def test_device_stage_writes_every_resonance_the_spectrum_and_figures(tmp_path):
    from tarang.design import circuit
    ctx = _device_ctx(tmp_path)
    r = circuit.stage_device(ctx)
    assert r["resonances"] > 50 and 1500 < r["band_nm"][0] < 1510 and 1590 < r["band_nm"][1] < 1600
    assert set(r["regimes"]) == {"critical", "over-coupled K = 1.5"}
    crit, over = r["regimes"]["critical"], r["regimes"]["over-coupled K = 1.5"]
    assert crit["dip_db_pump"] > 40 and 12 < over["dip_db_pump"] < 16
    assert over["Q_L_pump"] < crit["Q_L_pump"] < crit["Q_i_pump"]
    # off resonance: grating twice, lateral overlap and taper twice, 980 um of 0.4 dB/cm... 0.3+0.1
    eta = 0.313 * 0.99 * 10 ** (-0.0048)
    expect = 20 * math.log10(eta) - 0.4 * (400 + 2 * 290) * 1e-4
    assert r["off_resonance_db_at_pump"] == pytest.approx(expect, abs=0.01)
    Rp = 0.01 * 10 ** (-0.4 * 980e-4 / 10)
    assert r["ripple_db_at_pump"] == pytest.approx(20 * math.log10((1 + Rp) / (1 - Rp)), rel=1e-3)
    for t in r["tables"]:
        assert (ctx.out / t).exists()
    rows = list(csv.DictReader(open(ctx.out / "tables" / "device_resonances.csv", encoding="utf-8")))
    assert len(rows) == 2 * r["resonances"] and {"Q_L", "Dint_2pi_MHz", "dip_db"} <= set(rows[0])
    assert (ctx.out / "figures" / "11a_device_spectrum.png").exists()
    assert (ctx.out / "figures" / "11b_device_resonances.png").exists()
    assert any("constant over the band" in n for n in r["notes"])


def test_device_uses_the_coupling_against_wavelength_when_measured(tmp_path):
    from tarang.design import circuit
    ctx = _device_ctx(tmp_path, band_curve=True, reflections=False)
    md = circuit.model(ctx)
    rows = md["regimes"]["critical"]["rows"]
    K = [x["K"] for x in rows]
    assert min(K) < 0.95 and max(K) > 1.05                   # follows kappa^2(lambda)
    assert not md["use_refl"] and any("no grating reflection" in n for n in md["notes"])
    r = circuit.stage_device(ctx)
    assert r["ripple_db_at_pump"] is None


def test_device_resonance_shape_matches_its_q(tmp_path):
    from tarang.design import circuit
    md = circuit.model(_device_ctx(tmp_path, reflections=False))
    row = md["regimes"]["over-coupled K = 1.5"]["rows"][len(md["om"]) // 2]
    f0, lw = row["frequency_THz"] * 1e12, row["linewidth_MHz"] * 1e6
    f = f0 + np.linspace(-40, 40, 80001) * lw              # wide: T.max is the baseline
    T = circuit.transmission(md, "over-coupled K = 1.5", f)
    T = T / T.max()
    half = (1 + T.min()) / 2
    inside = f[T < half]
    assert inside.max() - inside.min() == pytest.approx(lw, rel=0.02)
    assert -10 * math.log10(T.min()) == pytest.approx(row["dip_db"], abs=0.1)


def test_device_is_a_stage_between_layout_and_report(tmp_path):
    from tarang.design.config import DesignConfig
    from tarang.design.stages import DEPENDS, STAGE_NAMES
    assert STAGE_NAMES[-3:] == ["layout", "device", "report"] and "device" in DEPENDS
    cfg = DesignConfig()
    cfg.device.zoom_fsr = 2.0
    back = DesignConfig.load(cfg.save(tmp_path / "d.toml"))
    assert back.device.zoom_fsr == 2.0 and back.device.reflections


def test_device_section_in_report_and_check_values(tmp_path):
    from tarang.design import circuit, report
    ctx = _device_ctx(tmp_path)
    r = circuit.stage_device(ctx)
    text = "\n".join(report._device_section(ctx, r))
    assert "## 7e. Whole device" in text and "11a_device_spectrum" in text
    q = {x["quantity"] for x in report._check_values_12(dict(device=r))}
    assert "fibre-to-fibre transmission off resonance" in q
    assert "loaded Q at the pump (critical)" in q


def test_device_off_needs_nothing(tmp_path):
    from tarang.design import circuit
    ctx = _ctx(tmp_path, device__run=False)
    assert circuit.stage_device(ctx) == {"skipped": True}
    ctx2 = _ctx(tmp_path / "b")                                 # no broadband solve saved
    assert circuit.stage_device(ctx2) == {"skipped": True}


# ---- GUI data layer --------------------------------------------------------------------- #
def _saved_design(tmp_path):
    from tarang.design import circuit
    ctx = _device_ctx(tmp_path)
    ctx.state["stages"]["device"] = {"key": "x", "results": circuit.stage_device(ctx)}
    ctx.state["spent"] = 3.0
    ctx.save_state()
    return ctx.out


def test_gui_finds_designs_and_reads_their_results(tmp_path):
    from tarang.gui import data
    run = _saved_design(tmp_path)
    assert data.find_runs(tmp_path) == [run]
    rows = data.stage_rows(run)
    assert [r["stage"] for r in rows][-3:] == ["layout", "device", "report"]
    assert {r["status"] for r in rows} <= {"not run", "done", "skipped / off", "settings changed: will rerun"}
    q = {r["quantity"] for r in data.summary_rows(run)}
    assert {"ring radius", "critical gap", "fibre to fibre, off resonance"} <= q
    groups = dict(data.figures(run))
    assert [f.name for f in groups["Whole device"]] == ["11a_device_spectrum.png", "11b_device_resonances.png"]
    assert any(t.name == "device_resonances.csv" for t in data.tables(run))
    assert any(f["file"] == "state.json" for f in data.files(run))
    assert len(data.read_csv(run / "tables" / "device_spectrum.csv", limit=5)) == 5


def test_gui_limits_follow_the_same_checks(tmp_path):
    from tarang.gui import data
    run = _saved_design(tmp_path)
    assert "at most 8.00" in data.set_limits(run, total=9.0, credit=5.0)       # 3 spent + 5
    assert data.set_limits(run, total=8.0, per_run=1.5, credit=5.0) == ""
    c = data.credits(run)
    assert c["limit"] == 8.0 and c["per_run"] == 1.5 and c["left"] == pytest.approx(5.0)
    assert "total limit" in data.set_limits(run, per_run=9.0)


def test_gui_command_lines(tmp_path):
    from tarang.gui import data
    a = data.command(tmp_path, ["device", "report"], "price")
    assert a[:3] == [sys.executable, "-m", "tarang.design"] and "--yes" in a and a[-1] == "--price-only"
    b = data.command(tmp_path, ["gap"], "run", redo=["gap"], spend_cap=1.25)
    assert b[b.index("--only") + 1] == "gap" and b[b.index("--redo") + 1] == "gap"
    assert b[b.index("--spend-cap") + 1] == "1.25"


def test_gui_new_design_from_an_example(tmp_path):
    from pathlib import Path
    from tarang.design.config import DesignConfig
    from tarang.gui import data
    ex = Path(__file__).resolve().parents[1] / "examples" / "design_sin780.toml"
    p = data.new_design(tmp_path / "d780", "d780", 0.78, ex)
    cfg = DesignConfig.load(p)
    assert cfg.run.name == "d780" and cfg.run.wavelength_um == 0.78 and cfg.apodise.min_trench_um == 0.2
    with pytest.raises(FileExistsError):
        data.new_design(tmp_path / "d780", "d780", 0.78)


def test_gui_starts_and_stops_a_background_run(tmp_path):
    from tarang.gui import data
    run = tmp_path / "run"
    run.mkdir()
    info = data.start(run, [sys.executable, "-u", "-c",
                            "import time\nprint('working', flush=True)\ntime.sleep(60)"])
    assert data.job(run)["running"]
    with pytest.raises(RuntimeError):
        data.start(run, [sys.executable, "-c", "pass"])
    t0 = time.time()
    while "working" not in data.log_tail(run) and time.time() - t0 < 20:
        time.sleep(0.2)
    assert "working" in data.log_tail(run)
    assert data.stop(run)
    t0 = time.time()
    while data.job(run)["running"] and time.time() - t0 < 10:
        time.sleep(0.2)
    if data.job(run)["running"]:                       # Ctrl+Break not delivered: force
        assert data.stop(run, force=True)
    t0 = time.time()
    while data.job(run)["running"] and time.time() - t0 < 20:
        time.sleep(0.2)
    assert not data.job(run)["running"]
    assert not data.pid_alive(info["pid"]) or sys.platform == "win32"


def test_gui_app_renders_a_design(tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    import tarang.gui.app as app
    _saved_design(tmp_path)
    at = AppTest.from_file(app.__file__, default_timeout=120)
    at.session_state["root"] = str(tmp_path)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert [t.label for t in at.tabs] == ["Summary", "Whole device", "Figures", "Tables", "Report",
                                          "Files", "Settings", "Log"]
    labels = [b.label for b in at.button]
    assert {"Estimate cost", "Run", "Stop", "Save limits", "Save settings"} <= set(labels)
    assert at.title[0].value.startswith("design")


def test_gui_launcher_explains_a_missing_streamlit(monkeypatch, capsys):
    import importlib.util
    from tarang.gui import __main__ as launcher
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert launcher.main(["--root", "x"]) == 2
    assert "pip install -e .[gui]" in capsys.readouterr().out


def test_apodise_stage_tries_the_ramp_and_keeps_the_preferred_design(tmp_path, fake_web, monkeypatch):
    from test_phase12b import _apod_ctx, _line_field
    from tarang.design import apodise as ap
    from tarang.grating_coupler.grating_coupler import GratingCoupler

    def fake_line(data, f0):
        teeth = [s for s in data.simulation.structures if s.name and s.name.startswith("Tooth_")]
        b = teeth[0].geometry.bounds
        P = teeth[1].geometry.bounds[0][0] - b[0][0]
        ff = (b[1][0] - b[0][0]) / P
        return _line_field(0.02 + 0.6 * (0.9 - ff), 1.55 / P + math.sin(math.radians(12)), P)
    monkeypatch.setattr(ap, "read_line", fake_line)
    ramps = set()
    real = ap.ramp_teeth

    def ramp(*a, **k):
        t = real(*a, **k)
        ramps.add(tuple(map(tuple, t)))
        return t
    monkeypatch.setattr(ap, "ramp_teeth", ramp)

    def spectrum(self, folder):
        wl = np.linspace(1.45, 1.65, 401)
        t = self.config.teeth
        if not t:
            return wl, 0.313 * np.exp(-((wl - 1.55) / 0.03) ** 2)              # uniform
        if tuple(map(tuple, t)) in ramps:
            return wl, 0.30 * np.exp(-((wl - 1.55) / 0.06) ** 2)              # wide band
        return wl, 0.32 * np.exp(-((wl - 1.55) / 0.03) ** 2)                  # measured recipe
    monkeypatch.setattr(GratingCoupler, "get_spectrum", spectrum)
    ctx = _apod_ctx(tmp_path)
    ctx.cfg.apodise.ramp_check = True
    res = ap.stage_apodise(ctx)
    assert res["ramp"]["rows"] and len(res["ramp"]["rows"]) == 15
    assert res["recipe"] == "measured scattering" and res["better_than_uniform"]
    assert (ctx.out / "tables" / "apodised_spectrum.csv").exists()
    assert (ctx.out / "tables" / "apodised_ramp.csv").exists()
    assert (ctx.out / "figures" / "9d_apodised_ramp_landscape.png").exists()
    ctx2 = _apod_ctx(tmp_path / "bw")
    ctx2.cfg.apodise.ramp_check = True
    ctx2.cfg.apodise.prefer = "bandwidth"
    res2 = ap.stage_apodise(ctx2)
    assert res2["recipe"].startswith("linear ramp") and res2["better_than_uniform"]
    assert res2["bandwidth_1db_nm"] > 2 * res2["uniform_bandwidth_1db_nm"] - 5
