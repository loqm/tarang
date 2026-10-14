"""The example notebooks are committed clean and call TARANG correctly (Phase 7). ~1 s.

    python -m pytest tests/test_notebooks.py -q

Notebooks are committed without outputs: stored outputs go stale with every code
change, made the repository about 10 MB larger, and contained paths from the
author's machine. These tests also check every TARANG call in the notebooks against
the real signatures (as tests/test_docs.py does for the Markdown files), that image
links point to committed files, and that materials are fitted over a band that
contains the wavelength each notebook simulates.
"""
import ast
import json
import re
from pathlib import Path

import pytest

from test_docs import _names_and_calls, _resolve

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted(ROOT.glob("*.ipynb"))


def _nb(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _code(path):
    """Code cells joined, with IPython magics and shell escapes removed."""
    cells = [c for c in _nb(path)["cells"] if c["cell_type"] == "code"]
    src = "\n".join("".join(c["source"]) for c in cells)
    return "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("%", "!")))


def test_there_are_notebooks():
    assert len(NOTEBOOKS) >= 5


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_committed_without_outputs(path):
    for cell in _nb(path)["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None, \
                "clear outputs: jupyter nbconvert --clear-output --inplace *.ipynb"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_personal_paths(path):
    text = path.read_text(encoding="utf-8")
    assert not re.search(r"/home/\w+|/Users/\w+|[A-Za-z]:\\\\Users\\\\", text)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_code_compiles_and_uses_real_names(path):
    code = _code(path)
    ast.parse(code)
    names, calls = _names_and_calls(code)
    for dotted in names.values():
        _resolve(dotted)
    import inspect
    for dotted, keywords in calls:
        fn = _resolve(dotted)
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            continue
        if any(p.kind is p.VAR_KEYWORD for p in params.values()):
            continue
        unknown = [k for k in keywords if k not in params]
        assert not unknown, f"{path.name}: {dotted}() has no parameter(s) {unknown}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_image_links_exist(path):
    for cell in _nb(path)["cells"]:
        if cell["cell_type"] == "markdown":
            for link in re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", "".join(cell["source"])):
                if not re.match(r"[a-z]+:", link):
                    assert (ROOT / link).exists(), f"{path.name}: missing image {link}"


_RANGE = re.compile(r'get_medium\(\s*(?:medium_key\s*=\s*)?"(\w+)"\s*,\s*(?:wl_range\s*=\s*)?'
                    r'\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)')
_WAVELENGTH = re.compile(r"wavelength_um\s*=\s*([0-9.]+)")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_materials_fitted_over_the_simulated_wavelength(path):
    """Each code cell that fits a material and builds a device at a literal wavelength
    must fit over a band containing that wavelength (a fit outside its band is an
    extrapolation, and the 1550 nm examples used to fit over 0.6-1.2 um)."""
    cells = [c for c in _nb(path)["cells"] if c["cell_type"] == "code"]
    fitted = {}
    for cell in cells:
        src = "".join(cell["source"])
        for key, lo, hi in _RANGE.findall(src):
            fitted[key] = (float(lo), float(hi))
        for wl in map(float, _WAVELENGTH.findall(src)):
            for key, (lo, hi) in fitted.items():
                if key != "air":
                    assert lo <= wl <= hi, f"{path.name}: {key} fitted over {lo}-{hi} µm, used at {wl} µm"
