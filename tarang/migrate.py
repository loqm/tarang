"""Update scripts and notebooks to the TARANG package layout (``tarang.media`` etc.).

Before the public release, TARANG's packages lived at the top level (``import media``,
``from grating_coupler import GratingCoupler``). They now live inside the ``tarang``
package (``import tarang.media``, ``from tarang.grating_coupler import GratingCoupler``).
This tool rewrites the import statements in your own ``.py`` files and Jupyter
notebooks. It changes only import statements: the names your code uses afterwards stay
exactly the same, so nothing else needs editing.

    python -m tarang.migrate my_scripts/                 # dry run: show what would change
    python -m tarang.migrate my_scripts/ --write         # apply the changes
    python -m tarang.migrate analysis.py notebook.ipynb  # individual files

Examples of what it does::

    from media import get_medium              -> from tarang.media import get_medium
    import geometries.ridge_waveguide as rw   -> import tarang.geometries.ridge_waveguide as rw
    import waveguide                          -> from tarang import waveguide
    from gratingCoupler import GratingConfig  -> from tarang.grating_coupler import GratingConfig

It also lists text that looks like an old module path but is not an import statement,
such as ``"utilities.cloud_run"`` in a string or ``python -m waveguide.dispersion_mapping``
in a comment; check those by hand. Files that are not valid Python are reported and
left unchanged. Without ``--write`` nothing is modified.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# old top-level name -> new dotted name
MOVED = {
    "media": "tarang.media",
    "geometries": "tarang.geometries",
    "waveguide": "tarang.waveguide",
    "ring_resonator": "tarang.ring_resonator",
    "grating_coupler": "tarang.grating_coupler",
    "edge_coupler": "tarang.edge_coupler",
    "utilities": "tarang.utilities",
    # names used before v0.1.0
    "ringResonator": "tarang.ring_resonator",
    "gratingCoupler": "tarang.grating_coupler",
    "edgeCoupler": "tarang.edge_coupler",
}
SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", ".venv", "venv", "site-packages",
             "build", "dist", "node_modules"}
_OLD_PATH = re.compile(r"(?<![\w.])(" + "|".join(MOVED) + r")\.[A-Za-z_]\w*")
_DATA_SUFFIX = re.compile(r"\.(json|csv|png|jpe?g|pdf|svg|txt|md|hdf5|h5|py|ipynb|yml|yaml)\b")


def _new_module(dotted: str) -> str | None:
    top, _, rest = dotted.partition(".")
    if top not in MOVED:
        return None
    return MOVED[top] + ("." + rest if rest else "")


def _rewrite_import(node: ast.Import) -> str | None:
    """`import a.b as c, x` -> equivalent statements binding the same names."""
    parts, binds, changed = [], [], False
    for alias in node.names:
        new = _new_module(alias.name)
        if new is None:
            parts.append(f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""))
            continue
        changed = True
        top = alias.name.split(".")[0]
        sub = new.split(".")[1]                          # e.g. "grating_coupler"
        bind = f"from tarang import {sub}" + (f" as {top}" if top != sub else "")
        if alias.asname:                                 # binds exactly the imported module
            parts.append(f"import {new} as {alias.asname}")
        elif "." in alias.name:                          # binds the top-level name
            parts.append(f"import {new}")
            binds.append(bind)
        else:
            binds.append(bind)
    if not changed:
        return None
    return "; ".join(parts + list(dict.fromkeys(binds)))


def _char_index(lines: list[str], lineno: int, col_byte: int) -> int:
    """Offset into the joined source for an ast (1-based line, UTF-8 byte column)."""
    prefix = sum(len(l) for l in lines[: lineno - 1])
    return prefix + len(lines[lineno - 1].encode("utf-8")[:col_byte].decode("utf-8", "ignore"))


@dataclass
class Result:
    new_source: str
    changes: int = 0
    notes: list[str] = field(default_factory=list)


def migrate_source(source: str, label: str = "<source>") -> Result:
    """Rewrite the import statements in Python source. Raises SyntaxError if unparsable."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    edits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            new = _new_module(node.module)
            if new is None:
                continue
            start = _char_index(lines, node.lineno, node.col_offset)
            end = _char_index(lines, node.end_lineno, node.end_col_offset)
            text = source[start:end]
            replaced = re.sub(r"^(from\s+)" + re.escape(node.module) + r"(?=\s)", r"\g<1>" + new,
                              text, count=1)
            if replaced != text:
                edits.append((start, end, replaced))
        elif isinstance(node, ast.Import):
            new_text = _rewrite_import(node)
            if new_text is not None:
                start = _char_index(lines, node.lineno, node.col_offset)
                end = _char_index(lines, node.end_lineno, node.end_col_offset)
                edits.append((start, end, new_text))
    out = source
    for start, end, text in sorted(edits, reverse=True):
        out = out[:start] + text + out[end:]

    notes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for m in _OLD_PATH.finditer(node.value):
                tail = node.value[m.start():m.end() + 8]
                if not _DATA_SUFFIX.match(node.value[m.start() + len(m.group(1)):]) \
                        and not tail.startswith(tuple(f"{k}.json" for k in MOVED)):
                    notes.append(f"{label}:{node.lineno}: string mentions '{m.group(0)}'")
                    break
    for i, line in enumerate(source.splitlines(), 1):
        comment = line.split("#", 1)[1] if "#" in line else ""
        if comment and _OLD_PATH.search(comment) and not _DATA_SUFFIX.search(comment):
            notes.append(f"{label}:{i}: comment mentions '{_OLD_PATH.search(comment).group(0)}'")
    return Result(out, len(edits), notes)


def _mask_magics(source: str) -> str:
    """IPython magics and shell escapes are not Python: blank them (same length) for parsing."""
    out = []
    for line in source.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith(("%", "!")):
            indent = line[: len(line) - len(stripped)]
            out.append(indent + "#" + stripped[1:])
        else:
            out.append(line)
    return "".join(out)


def migrate_file(path: Path) -> tuple[str, str, Result] | None:
    """Return (old text, new text, result) for a .py or .ipynb file, or None if not handled."""
    with open(path, encoding="utf-8-sig", newline="") as f:      # keep CRLF/LF as they are
        text = f.read()
    if path.suffix == ".py":
        res = migrate_source(text, str(path))
        return text, res.new_source, res
    if path.suffix == ".ipynb":
        nb = json.loads(text)
        total = Result(text)
        for i, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            masked = _mask_magics(src)
            res = migrate_source(masked, f"{path}[cell {i}]")
            if res.changes:
                # apply the edits to the original (unmasked) text: magics have the same length
                new_lines, old_lines = res.new_source.splitlines(keepends=True), src.splitlines(keepends=True)
                merged = [o if o.lstrip().startswith(("%", "!")) and n.lstrip().startswith("#") else n
                          for o, n in zip(old_lines, new_lines)] if len(new_lines) == len(old_lines) else new_lines
                new_src = "".join(merged)
                cell["source"] = new_src.splitlines(keepends=True) if isinstance(cell["source"], list) else new_src
                total.changes += res.changes
            total.notes += res.notes
        new_text = json.dumps(nb, indent=1, ensure_ascii=False) + "\n" if total.changes else text
        return text, new_text, total
    return None


def _files(paths: list[Path]):
    for p in paths:
        if p.is_file():
            yield p
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.suffix in (".py", ".ipynb") and f.is_file() \
                        and not any(part in SKIP_DIRS for part in f.parts):
                    yield f
        else:
            print(f"not found: {p}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tarang.migrate",
                                 description="Update imports to the tarang.* package layout.")
    ap.add_argument("paths", nargs="+", type=Path, help="files or folders (.py and .ipynb)")
    ap.add_argument("--write", action="store_true", help="apply the changes (default: dry run)")
    args = ap.parse_args(argv)
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(errors="replace")

    changed, unreadable, notes = 0, [], []
    for path in _files(args.paths):
        try:
            handled = migrate_file(path)
        except (SyntaxError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            unreadable.append(f"{path}: {type(exc).__name__}: {exc}")
            continue
        if handled is None:
            continue
        old, new, res = handled
        notes += res.notes
        if res.changes:
            changed += 1
            if args.write:
                bom = path.read_bytes().startswith(b"\xef\xbb\xbf")
                with open(path, "w", encoding="utf-8-sig" if bom else "utf-8", newline="") as f:
                    f.write(new)
                print(f"updated  {path}  ({res.changes} import statement(s))")
            elif path.suffix == ".py":
                sys.stdout.writelines(difflib.unified_diff(
                    old.splitlines(keepends=True), new.splitlines(keepends=True),
                    f"{path} (now)", f"{path} (after)"))
            else:
                print(f"would update  {path}  ({res.changes} import statement(s) in code cells)")

    print()
    verb = "Updated" if args.write else "Would update"
    print(f"{verb} {changed} file(s).")
    if notes:
        print("\nCheck these by hand (not import statements):")
        for n in notes:
            print("  " + n)
    if unreadable:
        print("\nNot valid Python, left unchanged:")
        for u in unreadable:
            print("  " + u)
    if not args.write and changed:
        print("\nNothing was changed. Run again with --write to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
