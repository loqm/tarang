"""Phase 14E: choosing the cross-section from every candidate of the map after comparing the
ticked ones side by side (tolerance box, bend, mode profiles, D over the band, D_int), the
finalist report with its bend, and the dispersion-engineering figures and tables kept with
the others (figures/2_*, figures/3_*, tables/), the raw solver data staying in results/."""
from __future__ import annotations

import csv
import io
import json

import pytest

from test_design import _ctx
from test_phase13 import _saved_design


def _png(path):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 3), "white").save(path)


def _csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ---- settings ---------------------------------------------------------------------------- #
def test_compare_setting_round_trips_and_does_not_change_the_stage_keys(tmp_path):
    from tarang.design.config import DesignConfig
    from tarang.design.stages import STAGE_NAMES, settings_key
    a = DesignConfig()
    keys = {s: settings_key(a, s) for s in STAGE_NAMES}
    a.cross_section.compare = ["0.7x1", "0.7x0.9"]
    assert {s: settings_key(a, s) for s in STAGE_NAMES} == keys       # comparing changes no result
    p = a.save(tmp_path / "d.toml")
    assert DesignConfig.load(p).cross_section.compare == ["0.7x1", "0.7x0.9"]


def test_compare_radius_follows_the_goal():
    from tarang.design.config import DesignConfig
    from tarang.design.stages import compare_radius
    c = DesignConfig()
    assert compare_radius(c, 2.2233) == pytest.approx(107.5)               # 200 GHz ring
    c.ring.radius_um = 120.0
    assert compare_radius(c, 2.2233) == 120.0
    c.run.goal = "loss_test"
    assert compare_radius(c, 2.2) == 20.0                                  # 20 µm at 1550 nm
    c.waveguides.bend_radius_um = 15.0
    assert compare_radius(c, 2.2) == 15.0


# ---- the stage's comparison (tools answered by fakes) --------------------------------------- #
def _fake_tools(ctx, monkeypatch, calls):
    """Each dispersion tool writes what the real one writes, where the stage asks."""
    from tarang.design import stages

    def tool(name, *args):
        a = [str(x) for x in args]
        calls.append((name, a))
        opt = lambda k: a[a.index(k) + 1] if k in a else None          # noqa: E731
        if name == "report":
            out = ctx.out / opt("--out")
            pts = opt("--finalists").split(",")
            _png(out / "finalists_comparison.png")
            _png(out / "bend_dispersion.png")
            rows = [dict(name=f"{float(p.split('x')[0]) * 1e3:.0f}x{float(p.split('x')[1]) * 1e3:.0f}",
                         D=300 + i, ng=2.22, dn_TE_TM=0.03, TE1_margin=0.011, D_min_tol=280 + i,
                         D_max_tol=320 + i, all_neighbours_target="True") for i, p in enumerate(pts)]
            _csv(out / "finalists.csv", rows)
            _csv(out / "finalists_bend.csv", [dict(name=r["name"], R_um=107.5, D_bend=r["D"],
                                                   bend_loss_dB_cm=1e-5) for r in rows])
            (out / "report.md").write_text("# finalists\n", encoding="utf-8")
        elif name == "broadband":
            out = ctx.out / opt("--plots-dir")
            _png(out / "compare_D_lambda.png")
            _png(out / "compare_Dint.png")
            rows = [dict(geometry=f"{float(p.split('x')[0]) * 1e3:.0f}×{float(p.split('x')[1]) * 1e3:.0f} nm",
                         D2_2pi_MHz=15.6, ZDWs_nm="none in band") for p in opt("--points").split(",")]
            _csv(out / "profiles_summary.csv", rows)
    monkeypatch.setattr(ctx, "tool", tool)
    monkeypatch.setattr(stages, "fig_mode_profiles", lambda c, h, w, path=None: path and (_png(path) or path))


def test_candidates_are_compared_side_by_side(tmp_path, monkeypatch):
    from tarang.design import stages
    ctx = _ctx(tmp_path)
    calls = []
    _fake_tools(ctx, monkeypatch, calls)
    out = stages.compare_candidates(ctx, [(0.7, 1.0), (0.7, 0.9)], 0.05, 0.02, 2.2233)
    names = [c[0] for c in calls]
    assert names == ["sweep", "sweep", "report", "broadband"]
    straight, bend, report, bb = (c[1] for c in calls)
    assert straight[straight.index("--points") + 1] == "0.7x1,0.7x0.9" and "--tol" in straight
    assert bend[bend.index("--bend-radius") + 1] == "107.5"
    assert report[report.index("--out") + 1] == "figures/2_candidates" and "--bends" in report
    assert bb[bb.index("--plots-dir") + 1] == "figures/2_candidates/profiles"
    assert bb[bb.index("--data") + 1].startswith("results/")                 # raw data stays there
    tables = ctx.out / "tables"
    assert {"candidates_compared.csv", "candidates_bend.csv", "candidates_profiles.csv"} <= \
        {p.name for p in tables.glob("*.csv")}
    assert not (ctx.out / "figures" / "2_candidates" / "finalists.csv").exists()   # moved, not copied
    assert [(r["H_um"], r["W_um"]) for r in out["rows"]] == [(0.7, 1.0), (0.7, 0.9)]
    first = out["rows"][0]
    assert first["D"] == 300 and first["D_min_tol"] == 280 and first["tolerance_box_target"]
    assert first["bend_loss_db_cm"] == pytest.approx(1e-5) and first["D2_2pi_MHz"] == 15.6
    assert (ctx.out / "figures" / "2_candidates" / "modes_H700_W1000.png").exists()


def test_chosen_cross_section_gets_its_bend_and_its_tables(tmp_path, monkeypatch):
    """The finalist report of the chosen cross-section had empty bend panels; its figures
    and tables now go to figures/2_finalists, figures/2_mesh_convergence and tables/."""
    from tarang.design import stages, visuals
    ctx = _ctx(tmp_path)
    calls = []
    _fake_tools(ctx, monkeypatch, calls)
    fin = ctx.path("results", "finalist_40steps.jsonl")
    rec = dict(H=0.7, W=1.0, steps=40, bend_radius=None, D_TE0=362.0, ng_TE0=2.2227,
               guided_TE=1, guided_TM=1, neff_TE=[1.7067], neff_TM=[1.68], n_cut=1.444)
    fin.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    conv = ctx.path("results", "convergence.jsonl")
    conv.write_text("\n".join(json.dumps(dict(H=0.7, W=1.0, steps=s, D_TE0=d)) for s, d in
                              ((40, 360.0), (60, 362.0))) + "\n", encoding="utf-8")
    monkeypatch.setattr(visuals, "safe", lambda *a, **k: None)
    ctx.cfg.cross_section.decision_steps = 40
    r = stages._cross_section_decision(ctx, 0.7, 1.0, [], 0.05, 0.02, steps=40)
    names = [c[0] for c in calls]
    assert names[:3] == ["sweep", "sweep", "report"] and names[-1] == "convergence"
    bend = calls[1][1]
    assert bend[bend.index("--bend-radius") + 1] == "107.5"
    rep = calls[2][1]
    assert "--bends" in rep and rep[rep.index("--out") + 1] == "figures/2_finalists"
    conv_call = calls[-1][1]
    assert conv_call[conv_call.index("--plots-dir") + 1] == "figures/2_mesh_convergence"
    assert (ctx.out / "tables" / "finalists.csv").exists() and (ctx.out / "tables" / "finalists_bend.csv").exists()
    assert r["report"].replace("\\", "/").endswith("figures/2_finalists/report.md")
    assert r["D"] == 362.0


def test_map_tools_write_their_plots_where_they_are_asked(tmp_path):
    """--plots-dir of maps, broadband and convergence (the default stays next to the data)."""
    import subprocess
    import sys
    data = tmp_path / "results" / "map.jsonl"
    data.parent.mkdir(parents=True)
    base = dict(wl0=1.55, steps=40, core="si3n4", substrate="sio2", cladding=None, box=4.0, n_shift=0.0,
                n_cut=1.44402, margin=0.005, si_handle=False, pml=0, bend_radius=None, loss_TE0_dB_cm=0.0,
                Aeff_TE0_um2=0.75, Gamma_TE0=0.8, beta2_TE0=-480.0, ng_TE0=2.22, D_TM0=200.0,
                beta2_TM0=-250.0, ng_TM0=2.2, fit_resid_TE0=1e-6, fit_resid_TM0=1e-6,
                neff_TE=[1.71, 1.43], neff_TM=[1.68, 1.42], hybrid_TE=[False, False],
                hybrid_TM=[False, False], guided_TE=1, guided_TM=1, guided_TE_margin=1,
                guided_TM_margin=1, guided_counts_valid=True, te0_gap_to_next=0.03,
                modes=[dict(neff=1.71, te_frac=0.98, pol="TE", guided=True, guided_margin=True),
                       dict(neff=1.68, te_frac=0.02, pol="TM", guided=True, guided_margin=True)])
    rows = [dict(base, H=h, W=w, D_TE0=300.0 + 100 * w) for h in (0.68, 0.7, 0.72) for w in (0.9, 1.0, 1.1)]
    data.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    import os
    from pathlib import Path
    root = str(Path(__file__).resolve().parents[1])                 # this checkout's tarang
    env = dict(os.environ, PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", ""), MPLBACKEND="Agg")
    r = subprocess.run([sys.executable, "-m", "tarang.waveguide.dispersion_mapping.maps", "--data",
                        str(data), "--overwrite", "--plots-dir", str(tmp_path / "figures" / "2_dispersion_map")],
                       capture_output=True, text=True, cwd=tmp_path, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / "figures" / "2_dispersion_map" / "design_map.png").exists()
    assert (tmp_path / "figures" / "2_dispersion_map" / "maps" / "dashboard.png").exists()
    assert not (tmp_path / "results" / "plots_map").exists()


# ---- the GUI --------------------------------------------------------------------------- #
def test_suite_reads_the_new_folders_and_the_old_ones(tmp_path):
    from tarang.gui import data, wizard as wz
    run = _saved_design(tmp_path)
    old = run / "results" / "plots_map_50steps" / "design_map.png"
    _png(old)
    assert data.latest_map(run) == old.parent                                # an older run
    for rel in ("2_dispersion_map/design_map.png", "2_dispersion_map/maps/dashboard.png",
                "2_candidates/finalists_comparison.png", "2_candidates/profiles/compare_D_lambda.png",
                "2_candidates/modes_H700_W1000.png", "2_finalists/finalists_comparison.png",
                "2_mesh_convergence/conv_summary.png", "3_dispersion_profile/profile_H700_W1000.png"):
        _png(run / "figures" / rel)
    assert data.latest_map(run) == run / "figures" / "2_dispersion_map"     # the new one first
    groups = dict(data.dispersion_suite(run))
    assert [p.name for p in groups["Candidates compared"]] == [
        "finalists_comparison.png", "compare_D_lambda.png", "modes_H700_W1000.png"]
    assert groups["Finalists compared"][0].parent.name == "2_finalists"
    assert groups["Ring: dispersion profile"][0].parent.name == "3_dispersion_profile"
    assert "Candidates compared" in [t for t, _ in wz.suite(run, wz.step("cross_section"))]
    assert [p.name for p in wz.comparison_figures(run)] == [
        "finalists_comparison.png", "compare_D_lambda.png", "modes_H700_W1000.png"]


def test_compare_choice_is_saved_for_the_next_run(tmp_path):
    from tarang.gui import data, wizard as wz
    run = _saved_design(tmp_path)
    assert wz.hw_key(dict(H_um=0.7, W_um=1.0)) == "0.7x1" and wz.hw_key([0.68, 1.1]) == "0.68x1.1"
    assert wz.compare_candidates(run, ["0.7x1", "0.7x0.9"]) == ""
    cfg = data.load_config(run)
    assert cfg.cross_section.compare == ["0.7x1", "0.7x0.9"]
    assert cfg.cross_section.height_um is None and cfg.cross_section.width_um is None
    assert wz.compare_list(run) == ["0.7x1", "0.7x0.9"]


def _waiting(run, n=10, comparison=None):
    p = run / "state.json"
    s = json.loads(p.read_text(encoding="utf-8"))
    cands = [dict(H_um=0.7, W_um=round(0.8 + 0.1 * k, 2), D_TE0=300.0 - k, ng_TE0=2.22, guided_TE=1,
                  guided_TM=1, TE_TM_split=0.03, target_neighbours="8/8") for k in range(n)]
    s["stages"]["cross_section"] = {"key": "x", "results": dict(
        skipped=True, awaiting_choice=True, candidates=cands, comparison=comparison)}
    p.write_text(json.dumps(s), encoding="utf-8")
    return cands


def test_choice_lists_every_candidate_and_compares_the_ticked_ones(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    import tarang.gui.app as app
    from tarang.gui import data, wizard as wz
    run = _saved_design(tmp_path)
    _waiting(run, n=10)
    started = []
    monkeypatch.setattr(data, "start", lambda r, argv: started.append(argv) or {"pid": 0, "argv": argv})
    at = AppTest.from_file(app.__file__, default_timeout=120)
    at.session_state["root"] = str(tmp_path)
    at.session_state[f"step:{run}"] = "cross_section"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("all 10 candidates" in m.value for m in at.markdown)
    assert len(at.selectbox(key="xs_pick").options) == 10                     # not only the top 8
    btn = at.button(key="xs_compare")
    assert btn.label == "Compare the 4 ticked candidates (free)"            # the top 4 pre-ticked
    btn.click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert data.load_config(run).cross_section.compare == ["0.7x0.8", "0.7x0.9", "0.7x1", "0.7x1.1"]
    argv = started[-1]
    assert argv[argv.index("--only") + 1] == "cross_section"
    assert data.load_config(run).cross_section.stop_after_map is True        # it stops again for the choice


def test_choice_shows_the_comparison(tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    import tarang.gui.app as app
    run = _saved_design(tmp_path)
    comp = dict(points=[[0.7, 1.0]], radius_um=107.5, steps=70, rows=[dict(
        H_um=0.7, W_um=1.0, D=362.0, D_min_tol=318.0, D_max_tol=382.0, n_g=2.2227, TE_TM_split=0.029,
        TE1_margin=0.011, tolerance_box_target=True, bend_R_um=107.5, bend_loss_db_cm=1e-5,
        D_bend=361.0, D2_2pi_MHz=15.6, ZDWs_nm="none in band")])
    _waiting(run, n=3, comparison=comp)
    _png(run / "figures" / "2_candidates" / "finalists_comparison.png")
    at = AppTest.from_file(app.__file__, default_timeout=120)
    at.session_state["root"] = str(tmp_path)
    at.session_state[f"step:{run}"] = "cross_section"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(m.value.startswith("**Compared** at 70 steps/λ; bend R = 107.5") for m in at.markdown)
    assert "(compared)" in at.selectbox(key="xs_pick").format_func(2)        # 0.7 x 1.0 is the third
    assert not any(e.label.startswith("Candidates compared") for e in at.expander)   # not twice
