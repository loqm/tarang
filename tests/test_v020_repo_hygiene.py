"""Guards against two edits that break the repository's front page without breaking code:
non-breaking spaces in the README (a command copied from it fails in the shell) and a
CITATION.cff that GitHub cannot read (no "Cite this repository")."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
URL = re.compile(r"^https?://\S+$")


def test_docs_have_no_non_breaking_spaces():
    for p in [ROOT / "Readme.md", ROOT / "CHANGELOG.md", ROOT / "CONTRIBUTING.md",
              ROOT / "tarang" / "media" / "data" / "SOURCES.md", *sorted((ROOT / "docs").glob("*.md"))]:
        if p.exists():
            lines = [i for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
                     if " " in line]
            assert not lines, f"{p.name}: non-breaking spaces on lines {lines[:10]}"


def test_citation_cff_is_plain_and_valid():
    cff = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert cff["cff-version"] == "1.2.0"
    for key in ("repository-code", "url"):
        assert URL.match(cff[key]), f"{key}: {cff[key]!r} is not a plain URL"
    pc = cff.get("preferred-citation")
    if pc:
        if "url" in pc:
            assert URL.match(pc["url"]), pc["url"]
        for ident in pc.get("identifiers", []):
            assert ident["type"] in {"doi", "url", "swh", "other"}, ident   # CFF 1.2.0 schema
    try:
        from cffconvert.cli.create_citation import create_citation
    except ImportError:
        return                                     # schema check where cffconvert is installed
    create_citation(str(ROOT / "CITATION.cff"), None).validate()
