"""Phase 15D: the lab's look in the GUI (theme, fonts, labels, status pills), the static
TARANG web page, and your measured values (intrinsic Q or cutback loss, coupler loss, and
measured film, fibre and material values) replacing the assumed ones."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


# ---- theme --------------------------------------------------------------------------------- #
def test_theme_flags_keep_only_options_streamlit_knows():
    from tarang.gui import theme
    known = {"theme.base", "theme.primaryColor", "theme.font"}
    flags = theme.streamlit_flags(known)
    assert flags[::2] == ["--theme.base", "--theme.primaryColor", "--theme.font"]
    assert flags[1::2][:2] == ["dark", theme.VIOLET]
    assert flags[5].startswith("Space Grotesk:https://fonts.googleapis.com/") and flags[5].endswith(", sans-serif")
    assert theme.streamlit_flags(set()) == []


def test_every_theme_option_exists_in_the_installed_streamlit():
    pytest.importorskip("streamlit")
    from tarang.gui import theme
    assert len(theme.streamlit_flags()) == 2 * len(theme.THEME)


def test_css_brand_label_and_pills():
    from tarang.gui import theme
    css = theme.css()
    assert css.startswith("<style>") and css.endswith("</style>")
    for token in (theme.BG, theme.VIOLET, theme.MAGENTA, theme.AMBER, "Space Grotesk", "IBM Plex Mono",
                  "system-ui", "monospace"):                     # with offline fallbacks
        assert token in css, token
    assert 'class="g">TARANG<' in theme.brand("TARANG", "x")
    assert theme.pill("done").count(theme.GREEN) == 1 and theme.pill("choose").count(theme.AMBER) == 1
    assert "&lt;b&gt;" in theme.pill("done", "<b>")              # text is escaped
    assert theme.label("Results").startswith('<div class="tg-label">')
    assert theme.heading("Results") == "**Results**"            # outside a themed app: plain markdown


def test_launcher_passes_the_lab_theme_or_the_plain_look(monkeypatch):
    pytest.importorskip("streamlit")
    from tarang.gui import __main__ as launcher
    seen = []

    class P:
        def __init__(self, cmd):
            seen.append(cmd)

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(launcher.subprocess, "Popen", P)
    launcher.main(["--no-browser"])
    launcher.main(["--no-browser", "--plain"])
    lab, plain = seen
    assert lab[lab.index("--theme.base") + 1] == "dark" and "#7b61ff" in lab
    assert plain[plain.index("--theme.base") + 1] == "light" and "#7b61ff" not in plain


# ---- web page ------------------------------------------------------------------------------ #
def test_web_page_is_self_contained_and_its_pictures_exist():
    page = ROOT / "docs" / "web" / "index.html"
    html = page.read_text(encoding="utf-8")
    assert "<script" not in html                                  # static: no script at all
    for src in re.findall(r'src="([^"]+)"', html):
        assert not src.startswith("http"), src
        assert (page.parent / src).exists(), src
    for section in ('id="steps"', 'id="walkthrough"', 'id="platforms"', 'id="start"'):
        assert section in html
    assert "Lumerical" not in html
    total = sum(p.stat().st_size for p in (page.parent / "assets").iterdir())
    assert total < 3_000_000                                      # light enough for the lab site


# ---- measured values: settings ------------------------------------------------------------- #
def test_measured_section_round_trips_and_is_checked(tmp_path):
    from tarang.design.config import DesignConfig, implausible
    c = DesignConfig()
    assert implausible(c) == {}
    c.measured.source = "chip A2, Nov 2026"
    c.measured.q_intrinsic = 1.2e6
    c.measured.coupler_loss_db = 5.8
    c.measured.keys = ["stack.core_height_um"]
    c.save(tmp_path / "d.toml")
    back = DesignConfig.load(tmp_path / "d.toml")
    assert back.measured == c.measured
    c.measured.propagation_loss_db_cm = 0.3
    assert "both set" in implausible(c)["measured.q_intrinsic"]
    c.measured.propagation_loss_db_cm = None
    c.measured.q_intrinsic = 1.2                                  # a loaded Q typed as 1.2 (million?)
    assert "1e3 and 1e10" in implausible(c)["measured.q_intrinsic"]
    c.measured.q_intrinsic = None
    c.measured.keys = ["stack.film"]
    assert "not a setting" in implausible(c)["measured.keys"]


def test_loss_from_q_is_the_inverse_of_q_intrinsic():
    from tarang.design import physics
    loss = physics.loss_from_q(1.0e6, 2.2227, 1.55)
    assert physics.q_intrinsic(loss, 2.2227, 1.55) == pytest.approx(1.0e6)
    assert loss == pytest.approx(0.391, abs=0.002)               # 1e6 at 1550 nm, n_g 2.22


def test_measured_q_holds_the_bend_loss_and_cutback_does_not():
    from tarang.design.config import DesignConfig
    from tarang.design import physics
    from tarang.design.stages import measured_loss
    c = DesignConfig()
    assert measured_loss(c, 2.2227, 1.55) is None
    c.measured.q_intrinsic = 1.2e6
    m = measured_loss(c, 2.2227, 1.55, bend_db_cm=0.01)
    assert m["straight_db_cm"] + 0.01 == pytest.approx(physics.loss_from_q(1.2e6, 2.2227, 1.55))
    assert m["source"] == "your measurement"
    c.measured.q_intrinsic, c.measured.propagation_loss_db_cm, c.measured.source = None, 0.35, "cutback"
    m = measured_loss(c, 2.2227, 1.55, bend_db_cm=0.01)
    assert m["straight_db_cm"] == 0.35 and m["source"] == "cutback"


def test_only_measured_losses_repeat_the_ring_step():
    from tarang.design.config import DesignConfig
    from tarang.design.stages import settings_key
    c = DesignConfig()
    old = settings_key(c, "ring")
    c.measured.source, c.measured.coupler_loss_db, c.measured.keys = "chip", 5.0, ["stack.box_height_um"]
    assert settings_key(c, "ring") == old                        # labels and the coupler loss: no rerun
    c.measured.q_intrinsic = 1e6
    assert settings_key(c, "ring") != old
    assert settings_key(c, "cross_section") == settings_key(DesignConfig(), "cross_section")


def test_assumptions_table_marks_measured_values():
    from tarang.design.config import DesignConfig
    from tarang.design import assumptions
    c = DesignConfig()
    c.measured.source = "chip A2"
    c.measured.propagation_loss_db_cm = 0.35
    c.measured.coupler_loss_db = 5.8
    c.measured.keys = ["stack.core_height_um", "grating.mfd_um"]
    c.stack.core_height_um = 0.695
    rows = {r["quantity"]: r for r in assumptions.table(c)}
    assert "scattering loss" not in rows
    assert rows["propagation loss (cutback)"]["value"] == 0.35
    assert rows["propagation loss (cutback)"]["source"] == "measured: chip A2"
    assert rows["film thickness"]["value"] == 0.695
    assert rows["fibre mode-field diameter"]["source"] == "measured: chip A2"
    assert rows["coupler insertion loss"]["value"] == 5.8
    assert rows["absorption loss"]["note"].startswith("Kept;")
    assert rows["Kerr index n2 of the core"]["source"].startswith("Ikeda")   # not measured: unchanged


def test_report_shows_the_measured_loss_and_coupler_loss(tmp_path):
    from test_design import _ctx, _with_cloud_results
    from tarang.design import report
    ctx = _with_cloud_results(_ctx(tmp_path))
    ctx.cfg.measured.source = "chip A2"
    ctx.cfg.measured.q_intrinsic = 1.2e6
    ctx.cfg.measured.coupler_loss_db = 5.8
    ring = ctx.state["stages"]["ring"]["results"]
    ring["measured_loss"] = dict(what="measured Q_i = 1.2e+06", source="chip A2", straight_db_cm=0.32)
    ctx.state["stages"].setdefault("checks", {"results": {}})
    ctx.state["stages"]["checks"]["results"]["loss_budget"] = [dict(item="grating, input", loss_db=5.1)]
    ctx.cfg.save(ctx.out / "design.toml")
    report.write(ctx)
    text = (ctx.out / "report.md").read_text(encoding="utf-8")
    assert "| loss from your measurement | measured Q_i = 1.2e+06 (chip A2) |" in text
    assert "| intrinsic Q of the ring | 1200000.0 |" in text and "measured: chip A2" in text
    assert "Measured insertion loss per coupler: 5.80 dB (measured: chip A2)" in text


# ---- measured values: the GUI page --------------------------------------------------------- #
def _app(tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    import tarang.gui.app as app
    at = AppTest.from_file(app.__file__, default_timeout=120)
    at.session_state["root"] = str(tmp_path)
    at.session_state["mode"] = "Your measured values"
    return at


def test_measured_values_page_saves_them_in_design_toml(tmp_path):
    from test_phase13 import _saved_design
    from tarang.gui import data
    run = _saved_design(tmp_path)
    at = _app(tmp_path)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(h.value == "Your measured values" for h in at.subheader)
    at.text_input[0].set_value("chip A2")                       # source
    at.radio(key="mv:kind").set_value("intrinsic Q of a ring")
    by = {t.label: t for t in at.text_input}
    by["Intrinsic Q (at the design wavelength)"].set_value("1.2e6")
    by["Insertion loss per coupler (dB)"].set_value("5.8")
    by["BOX thickness (µm)"].set_value("3.9")
    at.checkbox(key="mk:stack.box_height_um").check()
    next(b for b in at.button if b.label == "Save measured values").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Saved in design.toml" in s.value for s in at.success)
    cfg = data.load_config(run)
    assert cfg.measured.source == "chip A2" and cfg.measured.q_intrinsic == 1.2e6
    assert cfg.measured.coupler_loss_db == 5.8 and cfg.measured.propagation_loss_db_cm is None
    assert cfg.stack.box_height_um == 3.9 and cfg.measured.keys == ["stack.box_height_um"]


def test_measured_values_page_refuses_a_missing_or_wrong_value(tmp_path):
    from test_phase13 import _saved_design
    from tarang.gui import data
    run = _saved_design(tmp_path)
    before = (run / "design.toml").read_text(encoding="utf-8")
    at = _app(tmp_path)
    at.run()
    at.radio(key="mv:kind").set_value("cutback propagation loss")
    next(b for b in at.button if b.label == "Save measured values").click()
    at.run()
    assert any("Propagation loss: give the value" in e.value for e in at.error)
    by = {t.label: t for t in at.text_input}
    by["Propagation loss (dB/cm)"].set_value("0,35 dB")
    next(b for b in at.button if b.label == "Save measured values").click()
    at.run()
    assert any("a number, or empty" in e.value for e in at.error)
    assert (run / "design.toml").read_text(encoding="utf-8") == before
    assert data.load_config(run).measured.propagation_loss_db_cm is None


def test_lab_theme_is_applied_when_streamlit_runs_with_it(tmp_path):
    pytest.importorskip("streamlit")
    from streamlit import config
    from test_phase13 import _saved_design
    from tarang.gui import theme
    _saved_design(tmp_path)
    old = config.get_option("theme.backgroundColor")
    config.set_option("theme.backgroundColor", theme.BG)
    try:
        at = _app(tmp_path)
        at.session_state["mode"] = "Guided design"
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        md = " ".join(m.value for m in at.markdown)
        assert "tg-brand" in md and "<style>" in md and 'class="tg-label">Steps<' in md
    finally:
        config.set_option("theme.backgroundColor", old)
