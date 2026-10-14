"""End-to-end run of the dispersion-mapping command-line tools (Phase 4). Local mode
solver only, no FlexCredits; about 40 s.

    python -m pytest tests/test_dispersion_mapping_cli.py -q

sweep (2 geometries, coarse mesh) -> maps -> maps again -> report, run exactly as a
user would (``python -m tarang.waveguide.dispersion_mapping.<tool>``) in a temporary folder.
The second ``maps`` run must write to a new versioned folder: results are never
overwritten.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.slow


def run(tool, *args, env=None):
    proc = subprocess.run([sys.executable, "-m", f"tarang.waveguide.dispersion_mapping.{tool}", *args],
                          cwd=REPO, capture_output=True, text=True, timeout=900, errors="replace",
                          env={**os.environ, "MPLBACKEND": "Agg", **(env or {})})
    assert proc.returncode == 0, f"{tool} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    return proc.stdout


@pytest.fixture(scope="module")
def sweep_file(tmp_path_factory):
    out = tmp_path_factory.mktemp("dm") / "tiny.jsonl"
    run("sweep", "--heights", "0.70:0.70:0.05", "--widths", "0.90:1.00:0.10",
        "--steps", "20", "--workers", "1", "--out", str(out))
    return out


def test_sweep_writes_one_record_per_geometry(sweep_file):
    rows = [json.loads(line) for line in sweep_file.read_text(encoding="utf-8").splitlines()]
    assert sorted((r["H"], r["W"]) for r in rows) == [(0.7, 0.9), (0.7, 1.0)]
    for r in rows:
        assert r["D_TE0"] > 0                       # 700 nm air-clad SiN: anomalous at 1550 nm
        assert r["guided_TE"] == 1                  # single-mode TE at these widths


def test_sweep_resumes_without_redoing_points(sweep_file):
    before = sweep_file.read_text(encoding="utf-8")
    out = run("sweep", "--heights", "0.70:0.70:0.05", "--widths", "0.90:1.00:0.10",
              "--steps", "20", "--workers", "1", "--out", str(sweep_file))
    assert "2 already done, 0 to run" in out
    assert sweep_file.read_text(encoding="utf-8") == before      # append-only, nothing rewritten


def test_maps_never_overwrite_previous_plots(sweep_file):
    run("maps", "--data", str(sweep_file))
    run("maps", "--data", str(sweep_file))
    dirs = sorted(p.name for p in sweep_file.parent.iterdir() if p.is_dir() and p.name.startswith("plots_tiny"))
    assert len(dirs) == 2, dirs                       # e.g. plots_tiny and plots_tiny_v2
    assert any((sweep_file.parent / d / "target_geometries.csv").exists() for d in dirs)


def test_report_compares_finalists(sweep_file):
    out_dir = sweep_file.parent / "report"
    run("report", "--straight", str(sweep_file), "--finalists", "0.70x1.00,0.70x0.90",
        "--out", str(out_dir))
    files = {p.name for p in out_dir.iterdir()}
    assert {"report.md", "finalists.csv"} <= files
    text = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "baseline = 700x1000 nm" in text and "700x900" in text   # labels are in nm


def test_tools_survive_windows_legacy_console_encoding(sweep_file):
    """On Windows, output redirected to a file or pipe is cp1252, which cannot
    encode the λ/α in the report text. The tools must not crash (they used to)."""
    out_dir = sweep_file.parent / "report_cp1252"
    run("report", "--straight", str(sweep_file), "--finalists", "0.70x1.00", "--out", str(out_dir),
        env={"PYTHONIOENCODING": "cp1252", "PYTHONUTF8": "0"})
    assert (out_dir / "report.md").read_text(encoding="utf-8").count("λ") >= 1   # file stays UTF-8

