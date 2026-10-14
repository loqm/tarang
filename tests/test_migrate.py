"""python -m tarang.migrate: updating code written for the old layout (Phase 8). ~2 s.

    python -m pytest tests/test_migrate.py -q

Before v0.1.0 TARANG's packages were top-level (``import media``); now they live in
``tarang`` (``import tarang.media``). The migration tool rewrites import statements in
users' scripts and notebooks. These tests check the rewritten text and, more
importantly, *execute* the rewritten code: every name the old code used must still be
bound afterwards, so the rest of a script needs no edits.
"""
import importlib
import json
import subprocess
import sys

import pytest

from tarang.migrate import MOVED, migrate_file, migrate_source


def _run(code):
    """Execute code and return its namespace."""
    ns = {}
    exec(compile(code, "<migrated>", "exec"), ns)
    return ns


@pytest.mark.parametrize("old,new", [
    ("from media import get_medium\n", "from tarang.media import get_medium\n"),
    ("from media.medium_helper import get_medium\n", "from tarang.media.medium_helper import get_medium\n"),
    ("from gratingCoupler import GratingConfig\n", "from tarang.grating_coupler import GratingConfig\n"),
    ("import geometries.ridge_waveguide as rw\n", "import tarang.geometries.ridge_waveguide as rw\n"),
    ("import waveguide\n", "from tarang import waveguide\n"),
    ("import edgeCoupler\n", "from tarang import edge_coupler as edgeCoupler\n"),
    ("import numpy as np, media\n", "import numpy as np; from tarang import media\n"),
])
def test_rewrites_import_statements(old, new):
    assert migrate_source(old).new_source == new


def test_rewritten_code_binds_the_same_names():
    old = (
        "import media\n"
        "import geometries.ridge_waveguide\n"
        "import gratingCoupler.grating_coupler\n"
        "from utilities.cloud_run import sim_fingerprint\n"
        "from ring_resonator import RingResonator\n"
        "def f():\n"
        "    import waveguide.dispersion_mapping as dm\n"
        "    return dm\n"
    )
    ns = _run(migrate_source(old).new_source)
    assert ns["media"].get_medium is importlib.import_module("tarang.media").get_medium
    assert ns["geometries"].ridge_waveguide.get_passive_simulation           # dotted access works
    assert ns["gratingCoupler"].grating_coupler.GratingCoupler.__name__ == "GratingCoupler"
    assert callable(ns["sim_fingerprint"]) and ns["RingResonator"].__name__ == "RingResonator"
    assert ns["f"]().__name__ == "tarang.waveguide.dispersion_mapping"


def test_multiline_and_indented_imports_keep_their_layout():
    old = (
        "if True:\n"
        "    from media.medium_helper import (\n"
        "        get_medium,          # comment kept\n"
        "        get_medium_dispersion,\n"
        "    )\n"
    )
    new = migrate_source(old).new_source
    assert new == old.replace("from media.medium_helper", "from tarang.media.medium_helper")
    assert callable(_run(new)["get_medium_dispersion"])


@pytest.mark.parametrize("code", [
    "from .media import x\n",                 # relative import inside some package
    "import mediapipe\n",                     # different package, similar name
    "from utilities_extra import y\n",
    "import tarang.media\n",                  # already migrated
    "x = 'media.json'\n",
])
def test_leaves_other_code_alone(code):
    res = migrate_source(code)
    assert res.new_source == code and res.changes == 0 and not res.notes


def test_is_idempotent():
    once = migrate_source("import media\nfrom geometries import ridge_waveguide\n").new_source
    twice = migrate_source(once)
    assert twice.new_source == once and twice.changes == 0


def test_reports_old_paths_in_strings_and_comments_without_changing_them():
    code = ('monkeypatch.setattr("utilities.cloud_run.web", fake)\n'
            "# run: python -m waveguide.dispersion_mapping.sweep\n")
    res = migrate_source(code, "x.py")
    assert res.new_source == code
    assert any("x.py:1" in n and "utilities.cloud_run" in n for n in res.notes)
    assert any("x.py:2" in n and "waveguide.dispersion_mapping" in n for n in res.notes)


def test_every_target_module_exists():
    for new in set(MOVED.values()):
        importlib.import_module(new)


def test_notebook_code_cells_are_migrated_and_magics_kept(tmp_path):
    nb = {"cells": [
        {"cell_type": "markdown", "metadata": {}, "source": ["from media import x  (prose, untouched)\n"]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
         "source": ["%matplotlib inline\n", "from media import get_medium\n", "!pip list\n"]},
    ], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    path = tmp_path / "a.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")
    old, new, res = migrate_file(path)
    cells = json.loads(new)["cells"]
    assert res.changes == 1
    assert cells[0]["source"] == nb["cells"][0]["source"]
    assert cells[1]["source"] == ["%matplotlib inline\n", "from tarang.media import get_medium\n", "!pip list\n"]


def test_command_line_dry_run_then_write_keeps_line_endings(tmp_path):
    script = tmp_path / "scripts" / "old.py"
    script.parent.mkdir()
    script.write_bytes(b"import media\r\nfrom geometries import ridge_waveguide\r\nprint('ok')\r\n")
    broken = tmp_path / "scripts" / "broken.py"
    broken.write_bytes(b"def (:\n")
    before = script.read_bytes()

    dry = subprocess.run([sys.executable, "-m", "tarang.migrate", str(tmp_path), str(tmp_path / "typo")],
                         capture_output=True, text=True, check=True)
    assert script.read_bytes() == before                         # dry run changes nothing
    assert "Would update 1 file(s)" in dry.stdout and "broken.py" in dry.stdout
    assert f"not found: {tmp_path / 'typo'}" in dry.stdout       # a mistyped path is reported

    subprocess.run([sys.executable, "-m", "tarang.migrate", str(tmp_path), "--write"],
                   capture_output=True, text=True, check=True)
    after = script.read_bytes()
    assert after == (b"from tarang import media\r\nfrom tarang.geometries import ridge_waveguide\r\n"
                     b"print('ok')\r\n")
    assert broken.read_bytes() == b"def (:\n"                    # unparsable file untouched
