"""Consistency checks for the package definition (Phase 5). Local only, ~1 s.

    python -m pytest tests/test_packaging.py -q

pyproject.toml lists by hand which folders, data files and dependencies go into an
installed TARANG. These tests catch the usual ways that list goes out of date: a new
package folder that is not shipped, a material CSV missing from the package data, or
a new third-party import that is not declared as a dependency. Building and
installing the real wheel is done by the "package" job on GitHub.
"""
import ast
import fnmatch
import json
import re
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
PROJECT = PYPROJECT["project"]
SETUPTOOLS = PYPROJECT["tool"]["setuptools"]

NOT_SHIPPED = {"tests", "docs", "examples", "build", "dist", "results"}


def _requirement_name(req):
    """'tidy3d>=2.12,<3' -> 'tidy3d', 'tarang-photonics[plots]' -> 'tarang-photonics'."""
    return re.split(r"[\s<>=!~;\[]", req, maxsplit=1)[0].lower()


def _top_level_packages():
    return sorted(p.parent.name for p in ROOT.glob("*/__init__.py")
                  if p.parent.name not in NOT_SHIPPED and not p.parent.name.startswith("."))


def _shipped(name):
    return any(fnmatch.fnmatch(name, pat) for pat in SETUPTOOLS["packages"]["find"]["include"])


# ---- version and metadata ---------------------------------------------------- #
def test_version_has_a_single_source():
    import tarang
    assert "version" in PROJECT["dynamic"] and "version" not in PROJECT
    assert SETUPTOOLS["dynamic"]["version"] == {"attr": "tarang.__version__"}
    assert re.fullmatch(r"\d+\.\d+\.\d+((a|b|rc)\d+)?", tarang.__version__)


def test_running_python_satisfies_requires_python():
    low = re.fullmatch(r">=(\d+)\.(\d+)", PROJECT["requires-python"]).groups()
    assert sys.version_info[:2] >= tuple(map(int, low))


def test_readme_and_licence_exist():
    assert (ROOT / PROJECT["readme"]).is_file()
    for f in PROJECT["license-files"]:
        assert (ROOT / f).is_file()


# ---- what goes into the installed package -------------------------------------- #
def test_tarang_is_the_only_top_level_package():
    """All code lives inside tarang/, so TARANG adds a single name to an environment
    and cannot clash with other packages called media, utilities, ..."""
    assert _top_level_packages() == ["tarang"]
    assert SETUPTOOLS["packages"]["find"]["include"] == ["tarang*"]


def test_every_package_folder_is_shipped():
    missing = [p for p in _top_level_packages() if not _shipped(p)]
    assert not missing, f"add to [tool.setuptools.packages.find] include: {missing}"


def test_tests_and_docs_are_not_shipped():
    assert not any(_shipped(name) for name in NOT_SHIPPED)


def test_subpackages_have_init_files():
    """setuptools only finds folders with __init__.py; a folder without one is silently
    left out of the wheel while still importing fine from the repository."""
    missing = []
    for pkg in _top_level_packages():
        for d in (ROOT / pkg).rglob("*"):
            if d.is_dir() and d.name != "__pycache__" and any(d.glob("*.py")) \
                    and not (d / "__init__.py").exists():
                missing.append(str(d.relative_to(ROOT)))
    assert not missing


def test_material_files_are_package_data():
    patterns = SETUPTOOLS["package-data"]["tarang.media"]
    assert "media.json" in patterns
    media = ROOT / "tarang" / "media"
    database = json.loads((media / "media.json").read_text(encoding="utf-8"))
    for key, rel in database.items():
        assert not Path(rel).is_absolute(), key
        assert (media / rel).is_file(), f"{key}: {rel} missing"
        assert any(fnmatch.fnmatch(rel, pat) for pat in patterns), f"{key}: {rel} not packaged"


# ---- dependencies --------------------------------------------------------------- #
# import name -> distribution name, where they differ
_DIST_NAME = {}


def _third_party_imports():
    found = {}
    stdlib = set(sys.stdlib_module_names)
    for pkg in _top_level_packages():
        for path in (ROOT / pkg).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module]
                else:
                    continue
                for name in names:
                    top = name.split(".")[0]
                    if top not in stdlib and not (ROOT / top / "__init__.py").exists():
                        found.setdefault(top, set()).add(str(path.relative_to(ROOT)))
    return found


def test_every_third_party_import_is_declared():
    declared = {_requirement_name(r) for r in PROJECT["dependencies"]}
    for extra in PROJECT["optional-dependencies"].values():
        declared |= {_requirement_name(r) for r in extra}
    undeclared = {mod: sorted(files) for mod, files in _third_party_imports().items()
                  if _DIST_NAME.get(mod, mod).lower() not in declared}
    assert not undeclared, f"imported but not in pyproject.toml: {undeclared}"


@pytest.mark.parametrize("module", ["tidy3d", "numpy", "pandas", "pydantic", "matplotlib", "dynesty"])
def test_core_dependencies_are_required_not_optional(module):
    assert module in {_requirement_name(r) for r in PROJECT["dependencies"]}


def test_optional_imports_are_guarded():
    """getdist, corner and pymultinest are extras: TARANG must import without them,
    so they may only be imported inside functions or try blocks, never at module top."""
    optional = {"getdist", "corner", "pymultinest"}
    offenders = []
    for pkg in _top_level_packages():
        for path in (ROOT / pkg).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:                      # module level only
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                else:
                    continue
                if any(n.split(".")[0] in optional for n in names):
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not offenders


def test_test_extra_installs_what_the_test_suite_imports():
    test_extra = {_requirement_name(r) for r in PROJECT["optional-dependencies"]["test"]}
    assert {"pytest", "pytest-cov", "tarang-photonics"} <= test_extra
    plots = {_requirement_name(r) for r in PROJECT["optional-dependencies"]["plots"]}
    assert {"getdist", "corner"} <= plots


# ---- environment.yml and CI agree with pyproject.toml --------------------------- #
def test_environment_yml_installs_this_package():
    text = (ROOT / "environment.yml").read_text(encoding="utf-8")
    assert re.search(r"^\s*- python=3\.1[1-9]\s*$", text, re.M)
    assert re.search(r"^\s*- -e \.\[[a-z,]*test[a-z,]*\]\s*$", text, re.M)


def test_ci_installs_from_pyproject():
    workflow = ROOT / ".github" / "workflows" / "tests.yml"
    if not workflow.exists():
        pytest.skip("no .github folder (running from a source archive)")
    wf = workflow.read_text(encoding="utf-8")
    assert "requirements-test.txt" not in wf
    assert not (ROOT / "requirements-test.txt").exists()
    low = tuple(map(int, re.fullmatch(r">=(\d+)\.(\d+)", PROJECT["requires-python"]).groups()))
    versions = re.search(r'python: \[([^\]]+)\]', wf).group(1)
    for v in re.findall(r'"(\d+\.\d+)"', versions):
        assert tuple(map(int, v.split("."))) >= low
