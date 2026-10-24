"""Phase 14D: the release. Settings that cannot be right (nm typed for µm) are refused,
empty settings show the value they stand for, the loss presets name their papers, the
Tidy3D jobs of a design are listed with their links, the GUI has an Exit button, and the
guided screens are walked headlessly through both paths (A: ring + gratings, B: test
waveguides + edge couplers)."""
from __future__ import annotations

import json

import pytest

from test_phase13 import _saved_design


def _set(run, **kw):
    from tarang.gui import data
    cfg = data.load_config(run)
    for k, v in kw.items():
        sec, key = k.split("__")
        setattr(getattr(cfg, sec), key, v)
    cfg.save(run / "design.toml")
    return cfg


# ---- plausible values --------------------------------------------------------------------- #
def test_a_film_in_nanometres_is_caught_with_the_value_meant():
    from tarang.design.config import DesignConfig, implausible, range_problem
    assert range_problem("stack.core_height_um", 0.7) == ""
    assert range_problem("stack.core_height_um", None) == ""
    msg = range_problem("stack.core_height_um", 700)
    assert "700 µm" in msg and "did you mean 0.7 µm (700 nm)" in msg
    assert "did you mean" not in range_problem("stack.core_height_um", 70000000)
    assert range_problem("cross_section.heights_um", [0.68, 0.7, 0.72]) == ""
    assert "699.98" in range_problem("cross_section.heights_um", [699.98, 700])
    c = DesignConfig()
    assert implausible(c) == {}
    c.stack.core_height_um, c.edge.fiber_mfd_um = 700.0, 3500.0
    assert set(implausible(c)) == {"stack.core_height_um", "edge.fiber_mfd_um"}


def test_a_design_run_stops_before_any_stage_on_such_a_value(tmp_path, capsys):
    from tarang.design.__main__ import main
    from tarang.design.config import DesignConfig
    c = DesignConfig()
    c.run.goal, c.run.coupling, c.stack.core_height_um = "loss_test", "edge", 700.0
    p = c.save(tmp_path / "design.toml")
    assert main(["--config", str(p), "--yes", "--only", "materials"]) == 2
    out = capsys.readouterr().out
    assert "stack.core_height_um" in out and "did you mean 0.7 µm" in out
    assert not (tmp_path / "figures" / "1_materials.png").exists()


def test_the_gui_refuses_it_and_takes_the_right_one(tmp_path):
    from tarang.gui import data, wizard as wz
    run = _saved_design(tmp_path)
    assert "did you mean 0.7" in wz.save_fields(run, {"stack.core_height_um": 700.0})
    assert data.load_config(run).stack.core_height_um != 700.0           # nothing was changed
    assert wz.save_fields(run, {"stack.core_height_um": 0.7}) == ""
    assert data.load_config(run).stack.core_height_um == 0.7


# ---- what empty settings stand for -------------------------------------------------------- #
def test_empty_settings_show_their_value(tmp_path):
    from tarang.design.config import DesignConfig
    from tarang.gui import wizard as wz
    c = DesignConfig()
    assert wz.default_text(c, "run.band_um").startswith("0.202 ")             # 0.13 x 1.55
    c.stack.core_height_um = 0.7
    assert wz.default_text(c, "cross_section.heights_um") == "0.68, 0.7, 0.72"
    assert wz.default_text(c, "cross_section.widths_um").startswith("0.6, 0.7")
    assert wz.default_text(c, "ring.scattering_loss_db_cm") == "0.3 (preset typical_ebl)"
    c.ring.loss_preset = "state_of_the_art"
    assert wz.default_text(c, "ring.absorption_loss_db_cm") == "0.0013 (preset state_of_the_art)"
    assert "recommended" in wz.default_text(c, "edge.facet_width_um")         # from the help text
    # once the step has run: its recommendation
    run = _saved_design(tmp_path)
    p = run / "state.json"
    s = json.loads(p.read_text(encoding="utf-8"))
    s["stages"]["edge"] = {"key": "x", "results": {"recommend": {"recommended": {"facet_width_um": 4.75}}}}
    p.write_text(json.dumps(s), encoding="utf-8")
    assert wz.default_text(c, "edge.facet_width_um", run) == "the recommended 4.75"


def test_loss_presets_name_their_papers():
    from tarang.design.assumptions import LOSS_PRESETS
    from tarang.gui import wizard as wz
    assert wz.preset_label("typical_ebl") == "typical_ebl: 0.4 dB/cm, Colación et al., Sci. Rep. (2025)"
    assert "Xuan et al., Optica 3, 1171 (2016)" in wz.preset_label("annealed_low_loss")
    assert "Ji et al., Optica 4, 619 (2017)" in wz.preset_label("state_of_the_art")
    for k in LOSS_PRESETS:
        assert "doi:10." in wz.preset_detail(k)


# ---- Tidy3D jobs ------------------------------------------------------------------------- #
def test_jobs_are_listed_per_step_with_their_links(tmp_path):
    from tarang.gui import wizard as wz
    from tarang.utilities.cloud_run import write_task_record
    run = tmp_path / "d"
    for sub, name, tid, when in [("edge_facet", "facet_a", "fdve-1", "2026-10-22T10:00:00"),
                                 ("edge_taper", "eme_b", "fdve-2", "2026-10-22T11:00:00"),
                                 ("band", "ring_c", "fdve-3", "2026-10-21T09:00:00"),
                                 ("grating", "gc_d", "fdve-4", "2026-10-20T09:00:00")]:
        write_task_record(run / "cloud" / sub / name, task_id=tid, task_name=name, status="downloaded",
                          estimated_cost=0.1, submitted=when)
    all_jobs = wz.jobs(run)
    assert [j["name"] for j in all_jobs] == ["eme_b", "facet_a", "ring_c", "gc_d"]       # newest first
    assert [j["link"].startswith("http") and j["link"].endswith(t) for j, t in
            zip(all_jobs, ["fdve-2", "fdve-1", "fdve-3", "fdve-4"])] == [True] * 4
    assert {j["name"] for j in wz.jobs(run, wz.step("edge"))} == {"facet_a", "eme_b"}
    assert {j["name"] for j in wz.jobs(run, wz.step("checks"))} == {"ring_c", "gc_d"}
    assert wz.jobs(tmp_path / "nothing") == []
    assert wz.is_cloud(wz.step("edge")) and not wz.is_cloud(wz.step("materials"))


# ---- the screens ------------------------------------------------------------------------- #
def _app(tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    import tarang.gui.app as app
    at = AppTest.from_file(app.__file__, default_timeout=120)
    at.session_state["root"] = str(tmp_path)
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def test_a_wrong_film_is_shown_and_no_step_can_run(tmp_path):
    run = _saved_design(tmp_path)
    _set(run, stack__core_height_um=700.0)
    at = _app(tmp_path)
    at.session_state[f"step:{run}"] = "materials"
    at.run()
    _ok(at)
    assert any("did you mean 0.7" in e.value for e in at.error)
    assert next(b for b in at.button if b.label.startswith("Run")).disabled


def test_start_step_lists_the_papers_and_checks_the_film(tmp_path):
    run = _saved_design(tmp_path)
    at = _app(tmp_path)
    at.session_state[f"step:{run}"] = "start"
    at.run()
    _ok(at)
    sb = at.selectbox(key=f"preset:{run}")
    assert "Colación et al." in sb.format_func("typical_ebl")
    assert any("doi:10.1038" in c.value for c in at.caption)
    film = next(t for t in at.text_input if t.label.startswith("Film thickness"))
    film.set_value("700")
    next(b for b in at.button if b.label == "Save and continue").click()
    at.run()
    _ok(at)
    assert any("did you mean 0.7" in e.value for e in at.error)
    assert at.session_state[f"step:{run}"] == "start"


def test_exit_stops_the_gui_and_asks_about_a_running_run(tmp_path, monkeypatch):
    from tarang.gui import data
    called = []
    monkeypatch.setattr(data, "shutdown", lambda delay=1.0: called.append(delay))
    run = _saved_design(tmp_path)
    at = _app(tmp_path)
    at.run()
    at.button(key="exit").click()
    at.run()
    assert called and any("TARANG has stopped" in s.value for s in at.success)
    # with a run going: keep it or stop it
    called.clear()
    monkeypatch.setattr(data, "job", lambda r: {"running": True, "started": "now", "pid": 1, "argv": []})
    stopped = []
    monkeypatch.setattr(data, "stop", lambda r, force=False: stopped.append(r) or True)
    at = _app(tmp_path)
    at.run()
    at.button(key="exit").click()
    at.run()
    assert not called and any("still going" in w.value for w in at.warning)
    at.button(key="exit_stop").click()
    at.run()
    assert called and stopped == [run]


def _all_done(run, extra=None):
    """Every stage of the design marked done with its current settings (plus extra results)."""
    from tarang.design.stages import STAGE_NAMES, applies, settings_key
    from tarang.gui import data
    cfg = data.load_config(run)
    p = run / "state.json"
    s = json.loads(p.read_text(encoding="utf-8"))
    for name in STAGE_NAMES:
        if applies(name, cfg):
            old = (s["stages"].get(name) or {}).get("results") or {}
            s["stages"][name] = {"key": settings_key(cfg, name), "results": {**old, **(extra or {}).get(name, {})}}
    s["gui_choices_saved"] = True
    p.write_text(json.dumps(s), encoding="utf-8")


@pytest.mark.parametrize("goal,coupling", [("comb_ring", "grating"), ("loss_test", "edge")],
                         ids=["path_A_ring_gratings", "path_B_test_waveguides_edge"])
def test_guided_path_walks_every_step(tmp_path, goal, coupling, monkeypatch):
    """Headless walk through every step of a design: each step opens without an error, a done
    step shows its results, cloud steps list their Tidy3D jobs, and Run starts that step's
    own stages."""
    from tarang.gui import data, wizard as wz
    from tarang.utilities.cloud_run import write_task_record
    run = _saved_design(tmp_path)
    cfg = _set(run, run__goal=goal, run__coupling=coupling, stack__core_height_um=0.7)
    rec = {"edge": {"recommend": {"recommended": {"facet_width_um": 4.75, "profile": "parabolic",
                                                  "taper_length_um": 30.0}, "why": "best average"}},
           "waveguides": {"recommend": {"recommended": {"bend_radius_um": 50.0, "lengths_cm": [0, 1, 2, 4]},
                                        "why": "bend loss below a tenth"}}}
    _all_done(run, rec)
    write_task_record(run / "cloud" / ("edge_facet" if coupling == "edge" else "grating") / "job1",
                      task_id="fdve-9", task_name="job1", status="downloaded", estimated_cost=0.5,
                      submitted="2026-10-22T10:00:00")
    started = []
    monkeypatch.setattr(data, "start", lambda r, argv: started.append(argv) or {"pid": 0, "argv": argv})
    steps = wz.steps_for(cfg)
    at = _app(tmp_path)
    for i, s in enumerate(steps, 1):
        at.session_state[f"step:{run}"] = s.key
        at.run()
        _ok(at)
        assert any(h.value == f"Step {i} of {len(steps)}: {s.title}" for h in at.subheader), s.key
        if s.key == "start":
            continue
        assert wz.step_status(run, s) == "done", s.key
        labels = [e.label for e in at.expander]
        if wz.is_cloud(s):
            assert any(x.startswith("Tidy3D jobs of this step") for x in labels), s.key
        if s.key in ("edge", "waveguides"):
            assert any("Recommended" in m.value for m in at.markdown), s.key
    assert any(x.startswith("Tidy3D jobs of this design (1)") for x in [e.label for e in at.expander])
    # Run (free) on a local step starts exactly its stages, repeated
    at.session_state[f"step:{run}"] = "materials"
    at.run()
    next(b for b in at.button if b.label == "Run again (free)").click()
    at.run()
    _ok(at)
    argv = started[-1]
    assert argv[argv.index("--only") + 1] == "materials" and argv[argv.index("--redo") + 1] == "materials"


def test_step_buttons_carry_no_hover_text(tmp_path):
    run = _saved_design(tmp_path)
    at = _app(tmp_path)
    at.run()
    _ok(at)
    nav = [b for b in at.button if b.key and b.key.startswith("nav-")]
    assert nav and all(not b.help for b in nav)
    assert run.exists()


# ---- the benchmark pack ------------------------------------------------------------------- #
def test_benchmark_pack_has_a_comparison_sheet_and_the_versions(tmp_path):
    import csv
    from pathlib import Path
    from test_design import _ctx, _with_cloud_results
    from tarang.design import benchmark, report
    ctx = _with_cloud_results(_ctx(tmp_path))
    ctx.record_versions()
    ctx.cfg.save(ctx.out / "design.toml")
    pack = Path(report.write(ctx)["folder"])
    with open(pack / "comparison.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(pack / "results.csv", encoding="utf-8") as f:
        results = list(csv.DictReader(f))
    assert list(rows[0]) == benchmark.COMPARISON_FIELDS
    assert [r["quantity"] for r in rows] == [r["quantity"] for r in results]
    assert [r["tarang_value"] for r in rows] == [r["value"] for r in results]
    assert all(r["your_value"] == "" for r in rows)
    readme = (pack / "README.md").read_text(encoding="utf-8")
    import tarang
    assert f"Made with TARANG {tarang.__version__}, Tidy3D " in readme
    assert "## Sending your numbers back" in readme
