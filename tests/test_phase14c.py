"""Phase 14C: the guided GUI (steps, settings, status, recommendations, cross-section
choice, price, downloads, GDS preview, drawings, the Streamlit screens) and three fixes
from the 14B run (ring phase beyond the end resonances, the edge coupler's headline value
at the design wavelength, the wording of the device length)."""
from __future__ import annotations

import io
import json
import math
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pytest

from test_phase13 import _saved_design


def _cfg(**kw):
    from tarang.design.config import DesignConfig
    c = DesignConfig()
    for k, v in kw.items():
        sec, key = k.split("__")
        setattr(getattr(c, sec), key, v)
    return c


# ---- the steps ------------------------------------------------------------------------------ #
@pytest.mark.parametrize("goal,coupling,extra,expect", [
    ("comb_ring", "grating", {}, ["start", "materials", "cross_section", "ring", "gap", "grating", "checks",
                                  "taper", "layout", "device", "report"]),
    ("comb_ring", "grating", {"apodise__run": True, "focus__run": True},
     ["start", "materials", "cross_section", "ring", "gap", "grating", "checks", "taper", "apodise", "focus",
      "layout", "device", "report"]),
    ("comb_ring", "edge", {}, ["start", "materials", "cross_section", "ring", "gap", "checks", "edge",
                               "layout", "device", "report"]),
    ("loss_test", "grating", {}, ["start", "materials", "cross_section", "waveguides", "grating", "checks",
                                  "taper", "layout", "device", "report"]),
    ("loss_test", "edge", {"apodise__run": True}, ["start", "materials", "cross_section", "waveguides",
                                                   "checks", "edge", "layout", "device", "report"]),
])
def test_steps_follow_the_goal_and_the_coupling(goal, coupling, extra, expect):
    from tarang.gui import wizard as wz
    cfg = _cfg(run__goal=goal, run__coupling=coupling, **extra)
    assert [s.key for s in wz.steps_for(cfg)] == expect


def test_every_step_has_its_stages_settings_and_a_drawing():
    from tarang.design.stages import STAGE_NAMES
    from tarang.gui import wizard as wz
    covered = {s for st in wz.STEPS for s in st.stages}
    assert covered == set(STAGE_NAMES)                       # every stage belongs to one step
    for goal in ("comb_ring", "loss_test"):
        for coupling in ("grating", "edge"):
            cfg = _cfg(run__goal=goal, run__coupling=coupling, apodise__run=True, focus__run=True)
            for st in wz.STEPS:
                svg = st.picture(cfg) if st.picture else None
                if svg:
                    ET.fromstring(svg)                       # well-formed SVG
                for info in wz.visible_fields(cfg, st):
                    assert info["label"] and info["kind"] in ("bool", "int", "float", "list", "optional",
                                                              "str", "choice")


def test_form_values_are_parsed_in_the_settings_own_type():
    from tarang.gui import wizard as wz
    cfg = _cfg()
    lst = wz.field_info(cfg, "waveguides.lengths_cm")
    assert wz.parse_value(lst, "0, 3, 5.5") == [0, 3, 5.5] and wz.parse_value(lst, "") == []
    opt = wz.field_info(cfg, "edge.facet_width_um")
    assert wz.parse_value(opt, "") is None and wz.parse_value(opt, "4.5") == 4.5
    ch = wz.field_info(cfg, "edge.profile")
    assert ch["kind"] == "choice" and wz.parse_value(ch, "(recommended)") is None
    assert wz.parse_value(wz.field_info(cfg, "run.workers"), 3) == 3
    assert wz.show_value(wz.field_info(cfg, "waveguides.lengths_cm")) == "0, 1, 2, 4"


def test_saving_settings_checks_them_and_a_new_preset_brings_its_losses(tmp_path):
    from tarang.gui import data, wizard as wz
    run = _saved_design(tmp_path)
    assert wz.save_fields(run, {"run.goal": "laser"}).startswith("run.goal")
    assert data.load_config(run).run.goal == "comb_ring"           # refused: nothing changed
    assert wz.save_fields(run, {"ring.scattering_loss_db_cm": 0.3}) == ""
    assert wz.save_fields(run, {"ring.loss_preset": "annealed_low_loss"}) == ""
    cfg = data.load_config(run)
    assert cfg.ring.scattering_loss_db_cm is None and cfg.ring.absorption_loss_db_cm is None


# ---- status, recommendations, the cross-section choice --------------------------------------- #
def _state(run, **stages):
    p = run / "state.json"
    s = json.loads(p.read_text(encoding="utf-8"))
    s["stages"].update(stages)
    p.write_text(json.dumps(s), encoding="utf-8")


def test_step_status_and_the_cross_section_choice(tmp_path):
    from tarang.gui import data, wizard as wz
    run = _saved_design(tmp_path)
    cands = [dict(H_um=0.7, W_um=1.0, D_TE0=360.0, target_neighbours=8),
             dict(H_um=0.7, W_um=1.1, D_TE0=300.0, target_neighbours=6)]
    _state(run, cross_section={"key": "x", "results": dict(skipped=True, awaiting_choice=True, candidates=cands,
                                                           recommend=dict(recommended=dict(height_um=0.7,
                                                                                           width_um=1.0)))})
    st = wz.step("cross_section")
    assert wz.step_status(run, st) == "choose"
    assert wz.candidates(run) == cands
    assert wz.save_fields(run, {"cross_section.height_um": None, "cross_section.width_um": None}) == ""
    wz.prepare(run, st)
    assert data.load_config(run).cross_section.stop_after_map is True
    assert wz.choose_cross_section(run, 0.7, 1.1) == ""
    cfg = data.load_config(run)
    assert (cfg.cross_section.height_um, cfg.cross_section.width_um) == (0.7, 1.1)
    assert cfg.cross_section.stop_after_map is False
    assert "Cross-section" in wz.missing_before(run, wz.step("gap"))


def test_recommendation_is_shown_and_can_be_taken(tmp_path):
    from tarang.gui import data, wizard as wz
    run = _saved_design(tmp_path)
    assert wz.save_fields(run, {"run.goal": "loss_test"}) == ""
    _state(run, waveguides={"key": "x", "results": dict(recommend=dict(
        recommended=dict(bend_radius_um=15.0, lengths_cm=[0, 3, 5.5, 10.5]), why="precision"))})
    rec = wz.recommendation(run, wz.step("waveguides"))
    assert rec["settable"] and rec["why"] == "precision"
    row = next(r for r in rec["rows"] if r["what"].startswith("cutback"))
    assert row["recommended"] == "0, 3, 5.5, 10.5" and row["in use"] == "0, 1, 2, 4"
    assert wz.apply_recommended(run, wz.step("waveguides")) == ""
    cfg = data.load_config(run)
    assert cfg.waveguides.lengths_cm == [0, 3, 5.5, 10.5] and cfg.waveguides.bend_radius_um == 15.0
    _state(run, gap={"key": "x", "results": dict(recommend=dict(recommended=dict(critical_gap_um=0.47),
                                                                why="w"))})
    rec = wz.recommendation(run, wz.step("gap"))
    assert not rec["settable"] and rec["rows"][0]["in use"] == "(result)"


def test_a_step_runs_its_own_stages_and_reads_its_price(tmp_path, monkeypatch):
    from tarang.gui import data, wizard as wz
    run = _saved_design(tmp_path)
    seen = []
    monkeypatch.setattr(data, "start", lambda r, argv: seen.append(argv) or {})
    wz.run_step(run, wz.step("edge"), "price")
    wz.run_step(run, wz.step("gap"), "run", spend_cap=1.2, again=True)
    assert seen[0][seen[0].index("--only") + 1] == "edge" and "--price-only" in seen[0]
    assert seen[1][seen[1].index("--spend-cap") + 1] == "1.2" and "--redo" in seen[1]
    assert wz.price_total("...\n  Total 1.409 FC. Later steps ...") == 1.409
    assert wz.price_total("Price only: no cloud step needs paying for these stages.") == 0.0
    assert wz.price_total("still running") is None


# ---- downloads ----------------------------------------------------------------------------- #
def test_report_html_holds_the_figures_and_tables(tmp_path):
    from tarang.gui import wizard as wz
    run = _saved_design(tmp_path)
    fig = next((run / "figures").glob("*.png"))
    (run / "report.md").write_text(
        "# Design report: x\n\nSome **bold** text with `code`.\n\n## 1. Part\n\n| a | b |\n|---|---|\n"
        f"| 1 | 2 |\n\n- item one\n- item two\n\n![a figure](figures/{fig.name})\n", encoding="utf-8")
    h = wz.report_html(run)
    assert "<h1>Design report: x</h1>" in h and "<b>bold</b>" in h and "<code>code</code>" in h
    assert "<td>1</td><td>2</td>" in h and "<li>item one</li>" in h
    assert "data:image/png;base64," in h and "<figcaption>a figure</figcaption>" in h
    z = zipfile.ZipFile(io.BytesIO(wz.design_zip(run)))
    names = z.namelist()
    assert f"{run.name}/report.html" in names and f"{run.name}/design.toml" in names
    assert any(n.startswith(f"{run.name}/figures/") for n in names)
    assert not any("/cloud/" in n for n in names)


def test_gds_preview_is_drawn_and_reused(tmp_path):
    pytest.importorskip("gdstk")
    from tarang.design import layout
    from tarang.gui import wizard as wz
    from test_phase14b import _edge_ctx, _edge_results
    ctx = _edge_ctx(tmp_path)
    ctx.state["stages"]["edge"] = {"results": _edge_results()}
    out = ctx.out / "layout"
    layout.build_gds(layout.device_parameters(ctx), out)
    p = wz.gds_preview(ctx.out)
    assert p and p.exists() and p.stat().st_size > 5000
    t = p.stat().st_mtime
    assert wz.gds_preview(ctx.out).stat().st_mtime == t           # not redrawn while the GDS is the same


def test_drawings_embed_as_images():
    from tarang.gui import illustrations as il
    for name, fn in il.ILLUSTRATIONS.items():
        svg = fn()
        root = ET.fromstring(svg)
        assert root.tag.endswith("svg"), name
    tag = il.as_img(il.cross_section(0.7, 1.0, 4.0, "air"))
    assert tag.startswith('<img src="data:image/svg+xml;base64,')


# ---- the screens (Streamlit) ------------------------------------------------------------------ #
def _app(tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    import tarang.gui.app as app
    at = AppTest.from_file(app.__file__, default_timeout=120)
    at.session_state["root"] = str(tmp_path)
    return at


def test_guided_view_lists_the_steps_and_opens_the_first_to_do(tmp_path):
    run = _saved_design(tmp_path)
    at = _app(tmp_path)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [b.label for b in at.button]
    for t in ("1. What to design", "4. Ring", "5. Ring–bus gap", "Report and downloads"):
        assert any(t in x for x in labels), t
    assert any(h.value.startswith("Step ") for h in at.subheader)
    assert run.exists()


def test_a_new_design_starts_at_the_two_choices(tmp_path):
    from tarang.gui import data, wizard as wz
    run = data.new_design(tmp_path / "fresh", "fresh", 1.55).parent
    assert wz.step_status(run, wz.step("start")) == "to do"
    wz.mark_choices_saved(run)
    assert wz.step_status(run, wz.step("start")) == "done"
    assert "figure_version" in data.load_state(run)


def test_guided_start_step_saves_the_two_choices(tmp_path):
    from tarang.gui import data
    run = _saved_design(tmp_path)
    at = _app(tmp_path)
    at.session_state[f"step:{run}"] = "start"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    at.radio(key=f"goal:{run}").set_value("loss_test")
    at.radio(key=f"coupling:{run}").set_value("edge")
    at.run()
    assert any("lensed fibre" in m.value for m in at.markdown)
    next(b for b in at.button if b.label == "Save and continue").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    cfg = data.load_config(run)
    assert cfg.run.goal == "loss_test" and cfg.run.coupling == "edge"
    assert at.session_state[f"step:{run}"] == "materials"
    assert any("Test waveguides" in b.label for b in at.button)
    assert not any("Ring–bus gap" in b.label for b in at.button)


def test_guided_cross_section_step_offers_the_candidates(tmp_path):
    run = _saved_design(tmp_path)
    _state(run, cross_section={"key": "x", "results": dict(
        skipped=True, awaiting_choice=True,
        candidates=[dict(H_um=0.7, W_um=1.0, D_TE0=360.0, target_neighbours=8)])})
    at = _app(tmp_path)
    at.session_state[f"step:{run}"] = "cross_section"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(b.label == "Use this cross-section and check it" for b in at.button)
    assert "(recommended)" in at.selectbox(key="xs_pick").format_func(0)


def test_guided_report_step_offers_the_downloads(tmp_path):
    run = _saved_design(tmp_path)
    (run / "report.md").write_text("# Design report: x\n\ntext\n", encoding="utf-8")
    _state(run, report={"key": "x", "results": {}})
    at = _app(tmp_path)
    at.session_state[f"step:{run}"] = "report"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Downloads" in m.value for m in at.markdown)


# ---- fixes from the 14B run ---------------------------------------------------------------- #
def test_ring_phase_continues_past_the_end_resonances():
    from tarang.design.circuit import mode_phase
    om = np.array([10.0, 11.0, 12.0])
    M = np.array([100, 101, 102])
    D1 = np.array([1.0, 1.0, 1.0])
    assert mode_phase(9.5, om, M, D1) / (2 * math.pi) == pytest.approx(99.5)
    assert np.allclose(mode_phase(np.array([12.5, 11.5]), om, M, D1) / (2 * math.pi), [102.5, 101.5])


def test_edge_headline_value_is_at_the_design_wavelength(tmp_path):
    pytest.importorskip("xarray")
    from tarang.design import edge
    from test_phase14b import _edge_ctx, _fake_facet_data
    ctx = _edge_ctx(tmp_path)
    data = _fake_facet_data(0.40, 0.09)
    plane = data["fibre plane"]
    f = plane.Ey.coords["f"].values
    scale = np.sqrt(1 + 0.4 * (f - f.mean()) / (f.max() - f.min()))       # coupling changes with λ
    for c in ("Ey", "Hz"):
        da = getattr(plane, c)
        setattr(plane, c, da * scale[None, None, None, :])
    r = edge.analyse(data, ctx, 3.5, 0.5, 1.0, 1.55)
    sp = r["spectrum"]
    want = float(np.exp(np.interp(1.55, [s["wavelength_um"] for s in sp],
                                  np.log([s["efficiency"] for s in sp]))))
    assert r["eta"] == pytest.approx(want, rel=1e-9)
    assert abs(r["maps_wavelength_um"] - 1.55) > 1e-3 and r["eta_maps"] != pytest.approx(r["eta"], rel=1e-6)


def test_device_length_wording_follows_the_coupling(tmp_path):
    from tarang.design import report
    from test_phase14b import _edge_ctx
    ctx = _edge_ctx(tmp_path)
    dv = dict(resonances=3, coupler="edge coupler: 4.75 µm facet", bus_length_total_um=5000.0,
              band_nm=[1450, 1650], off_resonance_db_at_pump=-8.3, ripple_db_at_pump=2.4, regimes={})
    text = "\n".join(report._device_section(ctx, dv))
    assert "from facet to facet" in text
    dv["coupler"] = "focusing grating with its tip taper (3D FDTD)"
    assert "between the gratings" in "\n".join(report._device_section(ctx, dv))
