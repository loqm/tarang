"""Phase 12A: Tidy3D task records and links, re-attaching after an interruption, the
simpler design session, credits commands, figures and the physics checks."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import tidy3d as td

import tarang.utilities.cloud_run as cr

SIN = td.Medium(permittivity=1.996 ** 2)


def _sim(n=1.0):
    return td.Simulation(
        size=(2, 2, 2), run_time=1e-13, grid_spec=td.GridSpec.uniform(dl=0.1),
        structures=[td.Structure(geometry=td.Box(size=(0.5, 0.5, 0.5)),
                                 medium=td.Medium(permittivity=n))],
        boundary_spec=td.BoundarySpec.all_sides(td.PML()))


class _Interrupt(BaseException):
    """Stands in for Ctrl+C (KeyboardInterrupt) during web.monitor."""


# ---- task records and links ------------------------------------------------------------- #
def test_a_run_records_its_task_and_web_link(tmp_path, fake_web):
    run = cr.run_cached(_sim(), tmp_path, "demo")
    rec = cr.read_task_record(run.path.parent)
    assert rec["task_id"].startswith("tid") and rec["status"] == "downloaded"
    assert rec["url"].startswith("https://") and rec["task_id"] in rec["url"]
    assert rec["estimated_cost"] == fake_web.cost and rec["submitted"]
    rows = cr.collect_runs(tmp_path)
    assert len(rows) == 1 and rows[0]["url"] == rec["url"]


def test_an_interrupted_run_is_reattached_not_paid_again(tmp_path, fake_web, monkeypatch):
    real_monitor = fake_web.monitor

    def broken(tid, **kw):
        raise _Interrupt()
    monkeypatch.setattr(fake_web, "monitor", broken)
    with pytest.raises(_Interrupt):
        cr.run_cached(_sim(), tmp_path, "demo")
    assert fake_web.names().count("upload") == 1 and fake_web.names().count("start") == 1
    monkeypatch.setattr(fake_web, "monitor", real_monitor)
    events = []
    with cr.progress_listener(lambda e, i: events.append((e, i))):
        run = cr.run_cached(_sim(), tmp_path, "demo")
    assert fake_web.names().count("upload") == 1           # not submitted again
    assert fake_web.names().count("start") == 1
    assert run.path.exists() and events[0][0] == "reattached"
    assert events[0][1]["url"].endswith(events[0][1]["task_id"])


def test_a_failed_task_is_submitted_again(tmp_path, fake_web, monkeypatch):
    def broken(tid, **kw):
        fake_web.status[tid] = "error"
        raise _Interrupt()
    monkeypatch.setattr(fake_web, "monitor", broken)
    with pytest.raises(_Interrupt):
        cr.run_cached(_sim(), tmp_path, "demo")
    monkeypatch.undo()
    monkeypatch.setattr(cr, "web", fake_web)
    cr.run_cached(_sim(), tmp_path, "demo")
    assert fake_web.names().count("upload") == 2


def test_batch_runs_record_each_task_and_reattach(tmp_path, fake_web, monkeypatch):
    jobs = {"a": (_sim(1.0), "batchA"), "b": (_sim(2.0), "batchB")}
    Batch = fake_web.Batch

    class Interrupted(Batch):
        def monitor(self):
            raise _Interrupt()
    monkeypatch.setattr(fake_web, "Batch", Interrupted)
    with pytest.raises(_Interrupt):
        cr.run_batch_cached(jobs, tmp_path)
    recs = [cr.read_task_record(d) for d in tmp_path.iterdir() if d.is_dir()
            and (d / cr.TASK_FILE).exists()]
    assert len(recs) == 2 and all(r["status"] == "running" and r["url"] for r in recs)
    monkeypatch.setattr(fake_web, "Batch", Batch)
    out = cr.run_batch_cached(jobs, tmp_path)
    assert set(out) == {"a", "b"} and all(p.exists() for p in out.values())
    assert fake_web.names().count("batch_start") == 1         # only the first, interrupted one
    assert all(cr.read_task_record(p.parent)["status"] == "downloaded" for p in out.values())


def test_links_of_earlier_runs_are_filled_in_from_the_task_list(tmp_path, fake_web):
    run = cr.run_cached(_sim(), tmp_path, "demo")
    (run.path.parent / cr.TASK_FILE).unlink()                 # results from before records
    fake_web.tasks = [dict(task_id="old123", task_name=f"x_{run.path.parent.name.rsplit('_', 1)[-1]}",
                           status="success", created_at="2026-10-14T10:00:00")]
    assert cr.backfill_task_records(tmp_path) == 1
    rec = cr.read_task_record(run.path.parent)
    assert rec["task_id"] == "old123" and rec["backfilled"] and "old123" in rec["url"]
    assert cr.backfill_task_records(tmp_path) == 0            # nothing missing any more


# ---- physics: coupled-mode theory reproduces the 3D FDTD critical gap ------------------- #
@pytest.mark.slow
@pytest.mark.real_modes
def test_supermodes_reproduce_the_fdtd_critical_gap(tmp_path):
    """Even/odd TE supermodes of the 0.7 x 1.0 um SiN cores at 0.4-0.6 um gaps give, with
    theta = pi dn / lambda * sqrt(2 pi R / gamma), the critical gap of the R = 107.5 um ring
    within 10 nm of the 3D FDTD result (469.6 nm) and its decay rate within 5 %."""
    import io
    import math
    from tarang.design import localmodes
    from tarang.design.config import DesignConfig
    from tarang.design.stages import Context
    from tarang.design.ui import Prompter
    cfg = DesignConfig()
    cfg.run.band_um = 0.2
    ctx = Context(cfg, Prompter(False, out=io.StringIO()), tmp_path / "run")
    r = localmodes.coupling_from_supermodes(ctx, 0.7, 1.0, 107.5, [0.4, 0.5, 0.6])
    target = 0.0062021505727820015                      # round-trip loss at 0.4 dB/cm
    g_c = math.log(r["theta0"] / math.asin(math.sqrt(target))) / r["gamma_per_um"]
    assert abs(g_c - 0.4696) < 0.010
    assert abs(r["gamma_per_um"] - 5.324) / 5.324 < 0.05


@pytest.mark.slow
@pytest.mark.real_modes
def test_supermodes_are_true_even_and_odd_modes_at_a_wide_gap(tmp_path):
    """At a 0.9 um gap the coupling is weak and two nearly degenerate modes can come out
    of a solver localised in one core each (seen on Windows CI). The imposed mirror
    symmetry must give an even and an odd mode, straddling the single-core n_eff, with a
    split smaller than at 0.6 um."""
    import io
    from tarang.design import localmodes
    from tarang.design.config import DesignConfig
    from tarang.design.stages import Context
    from tarang.design.ui import Prompter
    cfg = DesignConfig()
    cfg.run.band_um = 0.2
    ctx = Context(cfg, Prompter(False, out=io.StringIO()), tmp_path / "run")
    wide = localmodes.coupler_supermodes(ctx, 0.7, 1.0, 0.9)
    near = localmodes.coupler_supermodes(ctx, 0.7, 1.0, 0.6)
    for name, sign in (("even", 1), ("odd", -1)):
        ey = wide[name]["fields"]["Ey"]
        parity = np.sum(ey * ey[::-1, :]) / np.sum(np.abs(ey) ** 2)
        assert abs(parity - sign) < 0.02                   # mirror image: +1 even, -1 odd
    single = localmodes.te00(localmodes.cross_section_modes(ctx, 0.7, 1.0, num_modes=2))
    assert wide["odd"]["neff"] < single["neff"] < wide["even"]["neff"]
    assert 0 < wide["dn"] < near["dn"] / 3


@pytest.mark.slow
@pytest.mark.real_modes
def test_mode_profiles_tell_te_from_tm(tmp_path):
    import io
    from tarang.design import localmodes
    from tarang.design.config import DesignConfig
    from tarang.design.stages import Context
    from tarang.design.ui import Prompter
    cfg = DesignConfig()
    cfg.run.band_um = 0.2
    ctx = Context(cfg, Prompter(False, out=io.StringIO()), tmp_path / "run")
    ms = localmodes.cross_section_modes(ctx, 0.7, 1.0, num_modes=4, group_index=True)
    te, tm = localmodes.te00(ms), localmodes.tm00(ms)
    assert te["te"] > 0.9 and tm["te"] < 0.1 and te["neff"] > tm["neff"]
    assert abs(te["ng"] - 2.223) < 0.01                       # TARANG's n_g at 1550 nm
    bent = localmodes.te00(localmodes.cross_section_modes(ctx, 0.7, 1.0, bend_radius=107.5))
    assert 0 < bent["yc"] - te["yc"] < 0.05                  # pushed outwards, a little


# ---- shared context ------------------------------------------------------------------ #
def _ctx(tmp_path, **changes):
    import io
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
        "cross_section": {"results": dict(height_um=0.7, width_um=1.0, steps=70, n_eff=1.7105,
                                          n_group=2.2233, aeff_um2=0.75)},
        "ring": {"results": dict(radius_um=107.5, n_group=2.2233, loss_db_cm=0.4,
                                 scattering_loss_db_cm=0.3, absorption_loss_db_cm=0.1,
                                 q_intrinsic=9.785e5, fsr_ghz=199.6)},
        "gap": {"results": dict(critical_gap_um=0.4696, over_coupled_gap_um=0.4312,
                                over_coupling=1.5, target_cross_power=0.0062021505727820015,
                                fit=dict(theta_0=0.1142 * math.exp(5.324 * 0.4),
                                         gamma=5.324))},
    }
    return ctx



def _ask(answers):
    it = iter(answers)
    return lambda q: next(it)


# ---- one table per stage ---------------------------------------------------------------- #
def test_a_stage_form_accepts_with_enter_and_edits_by_number(tmp_path):
    import io
    from tarang.design.config import DesignConfig
    from tarang.design.ui import Prompter
    cfg = DesignConfig()
    out = io.StringIO()
    ui = Prompter(True, out=out, inp=_ask(["2", "0.45, 0.5", ""]))
    items = [dict(section="gap", key="band_points"),
             dict(section="gap", key="gaps_um", default=[0.4, 0.5, 0.6])]
    ui.form(cfg, "Gap settings", items)
    assert cfg.gap.gaps_um == [0.45, 0.5] and cfg.gap.band_points == 11
    assert {"gap.gaps_um", "gap.band_points"} <= cfg.given
    text = out.getvalue()
    assert "Gap settings" in text and "Ring-bus gaps to simulate" in text
    assert "0.45, 0.5 um" in text
    # a second time the saved values are shown compactly and nothing is asked
    ui2 = Prompter(True, out=io.StringIO(), inp=_ask([]))
    ui2.form(cfg, "Gap settings", items)
    assert cfg.gap.gaps_um == [0.45, 0.5]


def test_q_stops_a_form_and_the_question_mark_explains(tmp_path):
    import io
    from tarang.design.config import DesignConfig
    from tarang.design.ui import Prompter
    cfg = DesignConfig()
    out = io.StringIO()
    ui = Prompter(True, out=out, inp=_ask(["?", "q"]))
    from tarang.design.ui import Stop
    with pytest.raises(Stop):
        ui.form(cfg, "Checks", [dict(section="checks", key="litho_bias_um")])
    assert "checks.litho_bias_um" in out.getvalue()


# ---- credits -------------------------------------------------------------------------- #
def test_credit_commands_set_limit_reset_and_list_spending(tmp_path, capsys, monkeypatch):
    from tarang.design import credits
    monkeypatch.setattr(cr, "account_status", lambda timeout=15.0: dict(
        credit=8.54, expires=None, expired=False))
    ctx = _ctx(tmp_path)
    ctx.state["spent"] = 2.0                              # spent before the per-stage log
    ctx.stage = "gap"
    ctx.add_spent(1.25, "3 gaps, 3D")
    ctx.save_state()
    s = credits.summary(ctx.cfg, ctx.out)
    assert s["spent"] == pytest.approx(3.25)
    assert s["rows"][0]["date"] == "(before the log)" and s["rows"][0]["estimate_FC"] == 2.0
    assert s["rows"][1]["stage"] == "gap" and s["rows"][1]["estimate_FC"] == 1.25
    credits.main(ctx.cfg, ctx.out, budget=30, reset=True, show=True)
    text = capsys.readouterr().out
    assert "set to 30 FC" in text and "was 3.250 FC" in text and "8.54 FC" in text
    from tarang.design.config import DesignConfig
    assert DesignConfig.load(ctx.out / "design.toml").run.max_total_cost == 30
    assert credits.summary(ctx.cfg, ctx.out)["spent"] == 0
    with pytest.raises(SystemExit):
        credits.main(ctx.cfg, ctx.out, budget=-1, show=False)


# ---- settings added later do not repeat finished stages ------------------------------- #
def test_settings_added_later_keep_old_results_and_fingerprints(tmp_path):
    from tarang.design.config import LEGACY
    from tarang.design.stages import settings_key
    ctx = _ctx(tmp_path)
    before = settings_key(ctx.cfg, "gap")
    notes = ctx.apply_legacy()
    for key, old in LEGACY["gap"].items():
        assert getattr(ctx.cfg.gap, key) == old and any(f"gap.{key}" in n for n in notes)
    after = settings_key(ctx.cfg, "gap")
    assert after != before                              # new defaults would rerun the gap
    ctx.cfg.gap.band_points = 11                          # a real change is a new key
    assert settings_key(ctx.cfg, "gap") not in (after,)
    fresh = _ctx(tmp_path / "b")
    del fresh.state["stages"]["gap"]                      # a new design: no legacy values
    assert fresh.apply_legacy() == [] and fresh.cfg.gap.band_points == 11


# ---- checks: tolerances, mode families, loss budget ----------------------------------- #
def test_gap_tolerance_moves_the_coupling_ratio_the_right_way(tmp_path):
    from tarang.design import checks
    ctx = _ctx(tmp_path, cross_section__tolerance_width_um=0.02)
    rows = checks.gap_tolerance(ctx, ctx.results("gap"))
    crit = [r for r in rows if r["design"] == "critical"]
    wider, nominal, narrower = crit
    assert nominal["K"] == pytest.approx(1.0, abs=0.01) and nominal["dip_depth_db"] == ">40"
    assert wider["K"] > 1 > narrower["K"]                 # wider cores: smaller gap, more coupling
    assert wider["gap_um"] == pytest.approx(0.4496)
    ql = 9.785e5 / (1 + nominal["K"])
    assert nominal["Q_loaded"] == pytest.approx(ql, rel=1e-3)
    over = [r for r in rows if r["design"].startswith("over")]
    assert len(over) == 3 and over[1]["K"] == pytest.approx(1.5, rel=0.05)


def test_mode_family_crossings_follow_from_the_fsr_difference(tmp_path):
    from tarang.design import checks, physics
    ctx = _ctx(tmp_path)
    rows = checks.mode_families(ctx)
    te, tm = rows[0], rows[1]
    assert te["family"] == "TE00" and tm["family"] == "TM00"
    f0, f1 = physics.fsr_ghz(107.5, 2.2227), physics.fsr_ghz(107.5, 2.2252)
    assert tm["modes_between_crossings"] == round(f0 / abs(f0 - f1))


def test_loss_budget_adds_up(tmp_path):
    from tarang.design import checks
    ctx = _ctx(tmp_path)
    rows = checks.loss_budget(ctx, dict(efficiency_2d=0.313))
    gc = -10 * math.log10(0.313)
    assert rows[0]["loss_db"] == pytest.approx(gc, abs=0.01)
    assert rows[-1]["loss_db"] == pytest.approx(sum(r["loss_db"] for r in rows[:-1]), abs=0.02)
    assert rows[-1]["loss_db"] == pytest.approx(2 * gc, abs=0.1)


def test_power_budget_of_the_reverse_run_gives_directionality_and_overlap():
    """From the reverse run: directionality = up / (up + down); the fibre overlap is the
    forward efficiency divided by the upward fraction; reflection in dB."""
    import types
    import xarray as xr
    from tarang.design import checks
    wl = np.array([1.50, 1.55, 1.60])
    f = 2.99792458e14 / wl
    flux = lambda v: types.SimpleNamespace(flux=xr.DataArray(np.array(v), coords={"f": f}, dims="f"))
    amps = xr.DataArray(np.sqrt([[0.01, 0.02, 0.01]]).reshape(1, 1, 3),
                        coords={"direction": ["-"], "mode_index": [0], "f": f},
                        dims=("direction", "mode_index", "f"))
    data = {"up": flux([0.5, 0.6, 0.5]), "down": flux([-0.3, -0.25, -0.3]),
            "past": flux([0.05, 0.04, 0.05]), "wg": types.SimpleNamespace(amps=amps)}
    rev = types.SimpleNamespace(monitors=[types.SimpleNamespace(name=n)
                                          for n in ("wg", "up", "down", "past")])
    spectra = {"final": (wl, np.array([0.28, 0.313, 0.27]))}
    t = checks._power_budget(data, rev, spectra, 1.55)["at_target"]
    assert t["directionality"] == pytest.approx(0.6 / 0.85)
    assert t["overlap"] == pytest.approx(0.313 / 0.6)
    assert t["reflection_db"] == pytest.approx(10 * math.log10(0.02))
    assert t["up"] + t["down"] + t["past"] + t["reflected"] == pytest.approx(0.91)


# ---- LLE export ---------------------------------------------------------------------- #
def _broadband(ctx):
    wl = np.linspace(1.3, 1.8, 51)
    n = 1.7105 - 0.205 * (wl - 1.55) - 0.03 * (wl - 1.55) ** 2
    rec = dict(H=0.7, W=1.0, steps=70, bend_radius=None, wl=list(wl), neff_TE0=list(n),
               n_cut=[1.444] * len(wl), pump=1.55)
    f = ctx.out / "results" / "broadband.jsonl"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(rec) + "\n", encoding="utf-8")


def test_lle_inputs_are_written_in_pylle_format(tmp_path):
    from tarang.design import lle
    from tarang.waveguide.dispersion_mapping.broadband import analyse
    ctx = _ctx(tmp_path)
    _broadband(ctx)
    r = lle.export(ctx)
    assert r and len(r["files"]) == 3
    data = json.loads((ctx.out / "lle" / "lle_inputs.json").read_text(encoding="utf-8"))
    gamma = 2 * math.pi * 2.4e-19 / (1.55e-6 * 0.75e-12)
    assert data["gamma_per_W_m"] == pytest.approx(gamma * ctx.cfg.assumptions.n2_m2_w / 2.4e-19)
    crit = data["coupling"]["critical"]
    assert crit["Q_c"] == pytest.approx(9.785e5) and crit["Q_L"] == pytest.approx(9.785e5 / 2)
    assert data["pyLLE"]["res"]["dispfile"] == "pyLLE_dispersion.csv"
    lines = (ctx.out / "lle" / "pyLLE_dispersion.csv").read_text().splitlines()
    m, f = lines[0].split(",")
    assert int(m) > 0 and 1e14 < float(f) < 3e14 and not lines[0][0].isalpha()   # no header
    ms = [int(x.split(",")[0]) for x in lines]
    fs = [float(x.split(",")[1]) for x in lines]
    assert ms == sorted(ms) and all(b > a for a, b in zip(fs, fs[1:]))
    # resonance condition m = n_eff(f) 2 pi R f / c at the pump mode
    A = analyse(json.loads((ctx.out / "results" / "broadband.jsonl").read_text()), 107.5)
    for key in ("m0", "om_mu", "om_pump", "D1", "mus", "Dint", "lam_mu"):
        assert key in A
    fsr = data["dispersion"]["FSR_Hz"]
    n_g = 1.7105 + 1.55 * 0.205                     # n - lambda dn/dlambda of the data above
    assert fsr == pytest.approx(299792458 / (2 * math.pi * 107.5e-6 * n_g), rel=1e-3)


def test_lle_export_waits_for_the_ring_stage(tmp_path):
    from tarang.design import lle
    ctx = _ctx(tmp_path)
    del ctx.state["stages"]["ring"]
    assert lle.export(ctx) is None


# ---- figures -------------------------------------------------------------------------- #
@pytest.mark.real_figures
def test_figures_are_written_in_every_format_at_the_chosen_resolution(tmp_path, monkeypatch):
    from tarang.design import visuals
    monkeypatch.setitem(visuals.SETTINGS, "formats", ["png"])
    visuals.configure(["png", "pdf", "svg"], 150)
    plt = visuals._plt()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    png = visuals._save(fig, tmp_path / "x.png")
    assert png.suffix == ".png"
    for ext in ("png", "pdf", "svg"):
        assert (tmp_path / f"x.{ext}").exists()
    from PIL import Image
    assert round(Image.open(png).info["dpi"][0]) == 150
    with pytest.raises(ValueError):
        visuals.configure(["jpg"])


def test_a_figure_open_elsewhere_is_saved_beside_it(tmp_path, monkeypatch):
    import os
    from tarang.design import visuals
    plt = visuals._plt()
    fig, ax = plt.subplots()
    target = tmp_path / "f.png"
    target.write_bytes(b"old")
    real = os.replace

    def locked(src, dst):
        if str(dst) == str(target):
            raise PermissionError("in use")
        return real(src, dst)
    monkeypatch.setattr(visuals.os, "replace", locked)
    out = visuals._write(fig, target, dpi=50)
    assert out.name == "f_new.png" and out.exists() and target.read_bytes() == b"old"
    assert any("f_new.png" in n for n in visuals._LAST_NOTES)
    assert not (tmp_path / "f.tmp.png").exists()


def test_an_unreachable_account_does_not_hang(monkeypatch):
    import time
    monkeypatch.setattr(cr.web, "account", lambda **kw: time.sleep(5), raising=False)
    t = time.time()
    assert cr.account_status(timeout=0.2) is None and time.time() - t < 2


def test_budget_with_only_the_folder_keeps_the_saved_settings(tmp_path, monkeypatch, capsys):
    from tarang.design.__main__ import main
    from tarang.design.config import DesignConfig
    monkeypatch.setattr(cr, "account_status", lambda timeout=15.0: None)
    cfg = DesignConfig()
    cfg.run.wavelength_um = 0.78
    cfg.given.add("run.wavelength_um")
    run = tmp_path / "r"
    run.mkdir()
    cfg.save(run / "design.toml")
    assert main(["--out", str(run), "--budget", "20"]) == 0
    back = DesignConfig.load(run / "design.toml")
    assert back.run.wavelength_um == 0.78 and back.run.max_total_cost == 20
    assert main(["--budget", "5"]) == 2                 # which design? refused, nothing written
