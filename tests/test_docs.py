"""Keep the documentation true to the code (Phase 6). Local only.

    python -m pytest tests/test_docs.py -q              # ~2 s
    python -m pytest tests/test_docs.py -q -m slow      # also runs the README examples

The pre-v0.1.0 README described an API that no longer existed (wrong parameter names,
a function that was never there, an install procedure that did not work). These tests
read the Markdown files and check them against the code: every Python example
compiles and calls TARANG with real names and keyword arguments, every relative link
points to an existing file or heading, the citation matches the package metadata, and
every material is documented. The slow test runs the README's first-run examples.
"""
import ast
import importlib
import importlib.util
import inspect
import json
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ["Readme.md", "CONTRIBUTING.md", "CHANGELOG.md", "media/data/SOURCES.md",
        "waveguide/dispersion_mapping/README.md", "tests/README.md"]
OWN = {"tarang", "media", "geometries", "waveguide", "ring_resonator", "grating_coupler",
       "edge_coupler", "utilities"}
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _text(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def _blocks(rel, lang):
    return re.findall(rf"```{lang}\n(.*?)```", _text(rel), re.S)


def _python_blocks():
    return [pytest.param(rel, i, code, id=f"{rel}#{i}")
            for rel in DOCS for i, code in enumerate(_blocks(rel, "python"))]


def _shell_lines():
    for rel in DOCS:
        for lang in ("bash", "powershell"):
            for block in _blocks(rel, lang):
                for line in block.splitlines():
                    line = line.split("#")[0].strip()
                    if line:
                        yield rel, line


# ---- Python examples ---------------------------------------------------------------- #
@pytest.mark.parametrize("rel,i,code", _python_blocks())
def test_python_examples_compile(rel, i, code):
    compile(code, f"{rel}[{i}]", "exec")


def _resolve(dotted):
    """Import `a.b.c` as far as it is a module, then follow attributes."""
    parts = dotted.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:cut]))
        except ModuleNotFoundError:
            continue
        for attr in parts[cut:]:
            obj = getattr(obj, attr)
        return obj
    raise ModuleNotFoundError(dotted)


def _names_and_calls(code):
    """Map local names to dotted TARANG paths, and list (dotted callee, keywords)."""
    tree = ast.parse(code)
    names = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in OWN:
            for a in node.names:
                names[a.asname or a.name] = f"{node.module}.{a.name}"
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in OWN:
                    names[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        chain, f = [], node.func
        while isinstance(f, ast.Attribute):
            chain.append(f.attr)
            f = f.value
        if isinstance(f, ast.Name) and f.id in names:
            dotted = ".".join([names[f.id]] + chain[::-1])
            calls.append((dotted, [k.arg for k in node.keywords if k.arg]))
    return names, calls


@pytest.mark.parametrize("rel,i,code", _python_blocks())
def test_python_examples_use_real_names_and_keywords(rel, i, code):
    names, calls = _names_and_calls(code)
    for dotted in names.values():
        _resolve(dotted)                                   # raises if the name is wrong
    for dotted, keywords in calls:
        fn = _resolve(dotted)
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            continue
        if any(p.kind is p.VAR_KEYWORD for p in params.values()):
            continue
        unknown = [k for k in keywords if k not in params]
        assert not unknown, f"{rel}: {dotted}() has no parameter(s) {unknown}"


def test_shell_commands_name_real_modules_and_extras():
    extras = set(PYPROJECT["project"]["optional-dependencies"])
    for rel, line in _shell_lines():
        for module in re.findall(r"python -m (\S+)", line):
            if module.split(".")[0] in OWN:
                assert importlib.util.find_spec(module), f"{rel}: no module {module}"
        for group in re.findall(r'pip install -e "?\.\[([a-z,]+)\]', line):
            missing = set(group.split(",")) - extras
            assert not missing, f"{rel}: unknown extra(s) {missing}"


def test_no_outdated_instructions():
    """Things the pre-v0.1.0 README told users that are no longer true."""
    stale = {
        "tarang-env": "the environment is called 'tarang'",
        "pip install tidy3d pandas getdist": "dependencies come from pyproject.toml",
        "run_gap_optimization_batch": "never existed; use ring_gap_sweep",
        "examples_media.ipynb": "the notebook is media_examples.ipynb",
        "sub_medium=": "use box_medium",
        "substrate_height=": "use box_height",
    }
    for rel in ("Readme.md", "CONTRIBUTING.md", "media/data/SOURCES.md",
                "waveguide/dispersion_mapping/README.md"):
        text = _text(rel)
        for phrase, why in stale.items():
            assert phrase not in text, f"{rel} still says {phrase!r} ({why})"


# ---- links ---------------------------------------------------------------------------- #
def _slug(heading):
    """GitHub's anchor for a Markdown heading."""
    s = re.sub(r"[`*_]", "", heading.strip().lower())
    s = re.sub(r"[^\w\- ]", "", s)
    return s.replace(" ", "-")


def _anchors(rel):
    return {_slug(h) for h in re.findall(r"^#{1,6} (.+)$", _text(rel), re.M)}


@pytest.mark.parametrize("rel", DOCS)
def test_relative_links_resolve(rel):
    here = (ROOT / rel).parent
    links = re.findall(r"\]\(([^)\s]+)\)", _text(rel)) + re.findall(r'src="([^"]+)"', _text(rel))
    for link in links:
        if re.match(r"[a-z]+:", link):
            continue                                   # external URL
        path, _, anchor = link.partition("#")
        target = (here / path).resolve() if path else (ROOT / rel)
        assert target.exists(), f"{rel}: broken link {link}"
        if anchor and target.suffix == ".md":
            assert anchor in _anchors(target.relative_to(ROOT).as_posix()), \
                f"{rel}: no heading for #{anchor} in {target.name}"


# ---- metadata ------------------------------------------------------------------------- #
def test_citation_matches_package():
    import yaml
    import tarang
    cff = yaml.safe_load(_text("CITATION.cff"))
    assert cff["cff-version"] == "1.2.0"
    assert str(cff["version"]) == tarang.__version__
    assert cff["license"] == "MIT"
    cff_authors = [f"{a['given-names']} {a['family-names']}" for a in cff["authors"]]
    assert cff_authors == [a["name"] for a in PYPROJECT["project"]["authors"]]
    released = re.search(rf"^## \[{re.escape(tarang.__version__)}\] - (\S+)$",
                         _text("CHANGELOG.md"), re.M)
    if "date-released" in cff or (released and released.group(1) != "unreleased"):
        assert released and str(cff.get("date-released")) == released.group(1), \
            "CITATION.cff date-released and the CHANGELOG release date differ"


def test_changelog_has_current_version():
    import tarang
    assert re.search(rf"^## \[{re.escape(tarang.__version__)}\]", _text("CHANGELOG.md"), re.M)


def test_every_material_is_documented():
    database = json.loads(_text("media/media.json"))
    sources, readme = _text("media/data/SOURCES.md"), _text("Readme.md")
    for key, rel in database.items():
        assert f"`{key}`" in sources, f"{key} missing from SOURCES.md"
        assert f"`{key}`" in readme, f"{key} missing from the README material table"
        assert Path(rel).name in sources, f"{rel} missing from SOURCES.md"
    for csv in (ROOT / "media" / "data").glob("*.csv"):
        assert csv.name in sources, f"{csv.name} is not documented in SOURCES.md"


# ---- run the README's first steps ---------------------------------------------------- #
@pytest.mark.slow
def test_readme_first_run_examples_execute(tmp_path, monkeypatch, no_show):
    """Steps 1-3 of README section 4 and the logging example, executed in order in a
    temporary folder (local mode solver only). Step 4, a 2 x 4 dispersion sweep, is
    checked for names and keywords above but not run here (several minutes)."""
    readme = _text("Readme.md")
    section4 = readme.split("## 4. ", 1)[1].split("\n## 5. ", 1)[0]
    section8 = readme.split("## 8. ", 1)[1].split("\n## 9. ", 1)[0]
    blocks = re.findall(r"```python\n(.*?)```", section4, re.S)
    blocks = [b for b in blocks if "dispersion_in_geometry" not in b]
    blocks += re.findall(r"```python\n(.*?)```", section8, re.S)
    assert len(blocks) == 4

    monkeypatch.chdir(tmp_path)
    namespace = {}
    for code in blocks:
        exec(compile(code, "Readme.md", "exec"), namespace)

    assert (tmp_path / "example_results" / "si3n4_dispersion.png").exists()
    assert (tmp_path / "example_results" / "waveguide_geometry.png").exists()
    assert (tmp_path / "run.log").exists()
    modes, disp, conf = namespace["modes"], namespace["disp"], namespace["conf"]
    assert modes and disp and conf
    from tarang import configure_logging
    configure_logging("WARNING", console=False)        # detach run.log before tmp_path goes
