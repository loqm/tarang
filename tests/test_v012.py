"""v0.1.2: a stage whose priced runs were not run (--price-only, --spend-cap, the budget, or
'skip') is kept but not marked done, so the next run finishes it. Before, a price-only run
recorded such a stage (e.g. checks with its grating runs priced but not run) as done, and the
real run then skipped it."""
from __future__ import annotations

import io

import pytest


@pytest.fixture
def one_stage(monkeypatch):
    """The design run reduced to its checks stage, which prices one 0.5 FC cloud step."""
    import tarang.design as design
    calls = []

    def fake_checks(ctx):
        ran = ctx.ui.proceed("grating checks: 1 simulation", 0.5, 10.0)
        calls.append(ran)
        return {"grating": "done"} if ran else {"mode_families": []}
    stages = [(n, t, k, fake_checks if n == "checks" else f) for n, t, k, f in design.STAGES]
    monkeypatch.setattr(design, "STAGES", stages)
    monkeypatch.setattr(design, "credit_limits", lambda ctx: None)
    return calls


def _run(tmp_path, **ui_kw):
    from tarang.design import run
    from tarang.design.config import DesignConfig
    from tarang.design.ui import Prompter
    cfg = DesignConfig()
    out = io.StringIO()
    ctx = run(cfg, tmp_path / "run", only=["checks"],
              prompter=Prompter(interactive=False, out=out, **ui_kw))
    return ctx, out.getvalue()


def test_price_only_leaves_the_stage_to_finish_next_time(tmp_path, one_stage):
    from tarang.design.stages import settings_key
    ctx, text = _run(tmp_path, price_only=True)
    st = ctx.state["stages"]["checks"]
    assert one_stage == [False] and st["key"] is None
    assert st["not_run"] == ["grating checks: 1 simulation"]
    assert "runs again next time" in text and "Price only: not run." in text
    ctx, text = _run(tmp_path)                                  # the real run
    st = ctx.state["stages"]["checks"]
    assert one_stage == [False, True] and "done earlier" not in text
    assert st["key"] == settings_key(ctx.cfg, "checks") and "not_run" not in st
    assert st["results"] == {"grating": "done"}
    ctx, text = _run(tmp_path)                                  # finished: not again
    assert one_stage == [False, True] and "done earlier with the same settings" in text


def test_spend_cap_also_leaves_the_stage_to_finish(tmp_path, one_stage):
    ctx, _ = _run(tmp_path, spend_cap=0.1)
    assert one_stage == [False] and ctx.state["stages"]["checks"]["key"] is None


def test_gui_shows_a_stage_left_by_price_only(tmp_path, one_stage):
    from tarang.gui import data
    ctx, _ = _run(tmp_path, price_only=True)
    ctx.cfg.save(tmp_path / "run" / "design.toml")
    rows = {r["stage"]: r["status"] for r in data.stage_rows(tmp_path / "run")}
    assert rows["checks"] == "priced only: runs again to finish"
