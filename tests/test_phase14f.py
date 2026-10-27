"""Phase 14F: the v0.1.0 wrap-up audit. A design.toml with a value of the wrong kind is
refused with the setting named (not a traceback later in a stage), the HTML report embeds
only pictures from its own design folder, a saved job whose process number now belongs to
another program is not shown as running, the choice panel says what is running and since
when, and the file-name hashes are marked as not for security (same names as before)."""
from __future__ import annotations

import datetime as dt
import hashlib
import os

import pytest

from test_phase13 import _saved_design


# ---- design.toml values of the wrong kind ---------------------------------------------- #
@pytest.mark.parametrize("line, words", [
    ('wavelength_um = "1.55"', ["[run] wavelength_um", "expected a number"]),
    ("wavelength_um = true", ["[run] wavelength_um", "expected a number"]),
    ("workers = 2.5", ["[run] workers", "expected a whole number"]),
    ('sim_pictures = "yes"', ["[run] sim_pictures", "expected true or false"]),
    ("figure_formats = \"png\"", ["[run] figure_formats", "expected a list"]),
    ("goal = 3", ["[run] goal", "expected text"]),
])
def test_a_value_of_the_wrong_kind_is_refused_with_its_setting(tmp_path, line, words):
    from tarang.design.config import DesignConfig, SettingsError
    p = tmp_path / "design.toml"
    p.write_text("[run]\n" + line + "\n", encoding="utf-8")
    with pytest.raises(SettingsError) as e:
        DesignConfig.load(p)
    assert all(w in str(e.value) for w in words)


def test_values_of_a_close_kind_are_taken(tmp_path):
    from tarang.design.config import DesignConfig
    p = tmp_path / "design.toml"
    p.write_text("[run]\nwavelength_um = 1\nworkers = 3.0\n[cross_section]\nheight_um = 1\n",
                 encoding="utf-8")
    c = DesignConfig.load(p)
    assert c.run.wavelength_um == 1.0 and isinstance(c.run.wavelength_um, float)
    assert c.run.workers == 3 and isinstance(c.run.workers, int)
    assert c.cross_section.height_um == 1.0 and isinstance(c.cross_section.height_um, float)


def test_the_shipped_examples_and_a_saved_default_still_load(tmp_path):
    from pathlib import Path
    from tarang.design.config import DesignConfig
    root = Path(__file__).resolve().parents[1] / "examples"
    for f in sorted(root.glob("*.toml")):
        DesignConfig.load(f)
    c = DesignConfig()
    assert DesignConfig.load(c.save(tmp_path / "design.toml")) == c


def test_design_run_stops_with_exit_code_2(tmp_path, capsys):
    from tarang.design.__main__ import main
    p = tmp_path / "design.toml"
    p.write_text('[run]\nwavelength_um = "0.78 um"\n', encoding="utf-8")
    assert main(["--config", str(p), "--yes", "--only", "materials"]) == 2
    assert "expected a number" in capsys.readouterr().out


# ---- the HTML report ------------------------------------------------------------------- #
def test_report_page_embeds_only_pictures_of_its_own_folder(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tarang.gui import wizard as wz
    run = tmp_path / "d"
    (run / "figures").mkdir(parents=True)
    for p in (run / "figures" / "in.png", tmp_path / "outside.png"):
        plt.figure(figsize=(1, 1)); plt.savefig(p); plt.close()
    (run / "report.md").write_text("# R\n\n![inside](figures/in.png)\n\n![outside](../outside.png)\n",
                                   encoding="utf-8")
    page = wz.report_html(run)
    assert "<figcaption>inside</figcaption>" in page
    assert "outside" not in page


# ---- the job status -------------------------------------------------------------------- #
def test_a_reused_process_number_is_not_a_running_job(tmp_path):
    from tarang.gui import data
    me = os.getpid()
    t0 = data.process_start(me)                 # this test process started at t0
    now = (dt.datetime.fromtimestamp(t0) if t0 else dt.datetime.now()).isoformat(timespec="seconds")
    assert data._same_process(dict(pid=me, started=now))
    assert not data._same_process(dict(pid=me, started="2020-01-01T00:00:00"))
    assert data._same_process(dict(pid=me))                       # older job files: no time
    run = tmp_path / "r"
    run.mkdir()
    (run / data.JOB_FILE).write_text(f'{{"pid": {me}, "started": "2020-01-01T00:00:00"}}',
                                     encoding="utf-8")
    assert data.job(run)["running"] is False
    (run / data.JOB_FILE).write_text(f'{{"pid": {me}, "started": "{now}"}}', encoding="utf-8")
    assert data.job(run)["running"] is True


def test_choice_panel_says_what_is_running_and_since_when(tmp_path, monkeypatch):
    from tarang.gui import data, wizard as wz
    run = _saved_design(tmp_path)
    monkeypatch.setattr(data, "job", lambda r: {})
    assert wz.choice_status(run) is None
    monkeypatch.setattr(data, "job", lambda r: dict(running=True, started="2026-10-27T14:05:09"))
    wz.compare_candidates(run, ["0.7x1", "0.7x0.9"])
    assert wz.choice_status(run).startswith("Comparing 2 candidate(s)")
    assert "since 14:05" in wz.choice_status(run)
    wz.choose_cross_section(run, 0.7, 1.0)
    assert wz.choice_status(run).startswith("Checking 0.7 × 1 µm")


# ---- hashes in file names -------------------------------------------------------------- #
def test_name_hashes_are_unchanged_by_marking_them_not_for_security():
    sig = "H0.7_W1.0_R100"
    assert hashlib.md5(sig.encode(), usedforsecurity=False).hexdigest() == \
        hashlib.md5(sig.encode()).hexdigest()


# ---- a killed worker (out of memory) does not end the map -------------------------------- #
def test_map_goes_on_one_at_a_time_when_a_worker_is_killed(tmp_path, monkeypatch, capsys):
    import json
    import sys
    from concurrent.futures import Future
    from concurrent.futures.process import BrokenProcessPool
    from tarang.waveguide.dispersion_mapping import sweep

    def fake_work(t):
        H, W, kw = t
        return dict(H=H, W=W, steps=kw["steps"], wl0=1.55, D_TE0=100.0, guided_TE=1,
                    guided_TM=1, neff_TE=[1.8], neff_TM=[1.7], secs=0.0)

    class Pool:                       # the first point finishes, then a worker is killed
        def __init__(self, max_workers):
            self.n = 0

        def submit(self, fn, t):
            fu = Future()
            self.n += 1
            if self.n == 1:
                fu.set_result(fn(t))
            else:
                fu.set_exception(BrokenProcessPool("killed"))
            return fu

        def shutdown(self, wait=True, cancel_futures=False):
            pass

    monkeypatch.setattr(sweep, "_work", fake_work)
    monkeypatch.setattr(sweep, "ProcessPoolExecutor", Pool)
    out = tmp_path / "m.jsonl"
    monkeypatch.setattr(sys, "argv", ["sweep", "--heights", "0.7", "--widths", "0.9,1.0,1.1",
                                      "--workers", "2", "--out", str(out)])
    sweep.main()
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert sorted(r["W"] for r in rows) == [0.9, 1.0, 1.1]          # each point once
    text = capsys.readouterr().out
    assert "A worker process was killed" in text and "with one worker" in text
