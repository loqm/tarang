"""Questions and answers for the design tool.

Each stage starts with ONE table of its settings (``Prompter.form``): Enter accepts them,
a number changes that setting, ``?`` explains them all, ``q`` (or Ctrl+C) stops; settings
already saved in design.toml are shown without a question unless ``--review`` is given.
Non-interactive (``--yes``, or no terminal): defaults are used and cloud stages run only if
their estimate fits the budget.
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable, Iterable, Optional


class Stop(Exception):
    """The user chose to stop; everything done so far is saved."""


class Prompter:
    def __init__(self, interactive: bool, out=None, inp: Callable[[str], str] = input,
                 show_figures: bool = False, review: bool = False, price_only: bool = False,
                 spend_cap: Optional[float] = None):
        self.interactive = interactive
        self.out = out or sys.stdout
        self.inp = inp
        self.show_figures = show_figures and interactive
        self.review = review and interactive    # ask saved settings again
        self.price_only = price_only            # show every estimate, run nothing priced
        self.spend_cap = spend_cap              # this session spends at most this much
        self.priced: list = []                  # (what, estimate) of every priced step

    # ---- output -------------------------------------------------------------------- #
    def say(self, text: str = "") -> None:
        print(text, file=self.out, flush=True)

    def heading(self, text: str) -> None:
        self.say("")
        self.say("=" * 78)
        self.say(text)
        self.say("=" * 78)

    def table(self, rows: list[dict], columns: Optional[list[str]] = None,
              formats: Optional[dict[str, str]] = None) -> None:
        if not rows:
            self.say("  (none)")
            return
        columns = columns or list(rows[0])
        formats = formats or {}

        def cell(row, c):
            v = row.get(c, "")
            if v is None:
                return "-"
            if c in formats and isinstance(v, (int, float)):
                return format(v, formats[c])
            return str(v)
        body = [[cell(r, c) for c in columns] for r in rows]
        widths = [max(len(c), *(len(b[i]) for b in body)) for i, c in enumerate(columns)]
        self.say("  " + "  ".join(c.rjust(w) for c, w in zip(columns, widths)))
        self.say("  " + "  ".join("-" * w for w in widths))
        for b in body:
            self.say("  " + "  ".join(v.rjust(w) for v, w in zip(b, widths)))

    def figure(self, path: Path) -> None:
        self.say(f"  figure: {path}")
        if self.show_figures:
            open_file(path)

    # ---- input --------------------------------------------------------------------- #
    def _ask(self, question: str, default: str) -> str:
        if not self.interactive:
            return default
        shown = f" [{default}]" if default != "" else ""
        try:
            answer = self.inp(f"{question}{shown}: ").strip()
        except EOFError:
            answer = ""
        if answer.lower() in {"q", "quit", "exit"}:
            raise Stop()
        return answer or default

    def value(self, cfg, section: str, key: str, default: Any = None,
              cast: Optional[Callable] = None, check: Optional[Callable[[Any], str]] = None) -> Any:
        """Return cfg.section.key; ask for it if it was not given in the config file.

        ``default`` replaces an empty default in the config; ``check`` returns an error
        message (or "") for an answer; the answer is stored back into cfg.
        """
        sec = getattr(cfg, section)
        f = {x.name: x for x in fields(sec)}[key]
        current = getattr(sec, key)
        if current in (None, []) and default is not None:
            current = default
        if f"{section}.{key}" in cfg.given:
            setattr(sec, key, current)
            return current
        unit = f" [{f.metadata['unit']}]" if f.metadata.get("unit") else ""
        question = f"{f.metadata['help']}{unit}"
        cast = cast or _caster(current)
        while True:
            text = self._ask(question, _show(current))
            try:
                value = cast(text) if text != "" else current
            except (TypeError, ValueError) as err:
                self.say(f"  Not understood ({err}); try again.")
                continue
            problem = check(value) if check else ""
            if problem:
                if not self.interactive:
                    raise ValueError(f"{section}.{key} = {value!r}: {problem}")
                self.say(f"  {problem}")
                continue
            setattr(sec, key, value)
            cfg.given.add(f"{section}.{key}")
            return value

    def form(self, cfg, title: str, items: list, always: bool = False) -> None:
        """One table for a stage's settings.

        ``items``: dicts with ``section``, ``key`` and optionally ``default`` (a value, or a
        function of cfg evaluated each time the table is shown, so it follows earlier
        settings), ``cast``, ``check`` (returns an error message or ""), ``show`` (bool or
        function of cfg). Accepted values are stored in cfg and marked as given, so the
        stage's later ``value()`` calls return them without asking. ``always``: show the
        table for changes even when every value is saved (interactive sessions only).
        """
        def active():
            return [it for it in items
                    if (it["show"](cfg) if callable(it.get("show")) else it.get("show", True))]

        def current(it):
            sec = getattr(cfg, it["section"])
            v = getattr(sec, it["key"])
            if v in (None, []) and it.get("default") is not None:
                d = it["default"]
                v = d(cfg) if callable(d) else d
            return v

        def meta(it):
            f = {x.name: x for x in fields(getattr(cfg, it["section"]))}[it["key"]]
            return f.metadata

        def accept_all():
            for it in active():
                setattr(getattr(cfg, it["section"]), it["key"], current(it))
                cfg.given.add(f"{it['section']}.{it['key']}")

        saved = {f"{it['section']}.{it['key']}" for it in items} & set(cfg.given)
        pending = [it for it in active() if f"{it['section']}.{it['key']}" not in cfg.given]
        if not self.interactive or (not pending and not self.review and not always):
            accept_all()
            self._form_table(cfg, title, active(), current, meta, saved, compact=True)
            return
        while True:
            rows = active()
            self._form_table(cfg, title, rows, current, meta, saved)
            ans = self._ask("Enter = use these, a number = change it, ? = explain, q = stop", "")
            if ans == "":
                accept_all()
                return
            if ans == "?":
                for i, it in enumerate(rows, 1):
                    m = meta(it)
                    unit = f" [{m['unit']}]" if m.get("unit") else ""
                    self.say(f"  {i}. {it['section']}.{it['key']}{unit}: {m['help']}")
                continue
            try:
                it = rows[int(ans) - 1]
            except (ValueError, IndexError):
                self.say("  Type the number of a setting, ? or press Enter.")
                continue
            key = f"{it['section']}.{it['key']}"
            cfg.given.discard(key)
            sec = getattr(cfg, it["section"])
            if getattr(sec, it["key"]) in (None, []):
                setattr(sec, it["key"], current(it))
            self.value(cfg, it["section"], it["key"], cast=it.get("cast"), check=it.get("check"))
            saved.discard(key)

    def _form_table(self, cfg, title, rows, current, meta, saved, compact=False) -> None:
        self.say(f"{title}:" if not compact else f"{title} (settings; change with --review "
                 "or in design.toml):")
        for i, it in enumerate(rows, 1):
            m = meta(it)
            label = _label(m["help"])
            unit = f" {m['unit']}" if m.get("unit") and m["unit"] != "FlexCredits" else (
                " FC" if m.get("unit") == "FlexCredits" else "")
            where = "saved" if f"{it['section']}.{it['key']}" in saved else ""
            self.say(f"  {i:>2}  {label:<52.52} {_show(current(it)) + unit:<28} {where}")

    def proceed(self, what: str, cost: Optional[float] = None, budget_left: Optional[float] = None,
                per_run_limit: Optional[float] = None) -> bool:
        """Ask whether to run the next stage. Returns False to skip it."""
        line = f"Next: {what}"
        if cost is not None:
            line += f"  (estimate {cost:.3f} FlexCredits"
            if budget_left is not None:
                line += f"; budget left {budget_left:.3f}"
            line += ")"
        self.say("")
        self.say(line)
        if cost is not None:
            self.priced.append((what, float(cost)))
        if self.price_only and cost is not None and cost > 0:
            self.say("  Price only: not run.")
            return False
        if cost is not None and self.spend_cap is not None:
            if cost > self.spend_cap + 1e-9:
                self.say(f"  Skipped: above what this session may still spend ({self.spend_cap:.3f} "
                         "FC, --spend-cap).")
                return False
        if not self.interactive:
            if cost is not None and budget_left is not None and cost > budget_left:
                self.say("  Skipped: the estimate is above the remaining budget "
                         "(raise run.max_total_cost to run it).")
                return False
            self._spend(cost)
            return True
        while True:
            ans = self._ask("Proceed? y = yes, s = skip this stage, q = stop here", "y").lower()
            if ans in {"y", "yes"}:
                if cost is not None and budget_left is not None and cost > budget_left:
                    self.say("  The estimate is above the remaining budget; raise "
                             "run.max_total_cost in the config to run it.")
                    return False
                self._spend(cost)
                return True
            if ans in {"s", "skip", "n", "no"}:
                return False
            self.say("  Please answer y, s or q.")


    def _spend(self, cost: Optional[float]) -> None:
        if cost is not None and self.spend_cap is not None:
            self.spend_cap = max(self.spend_cap - float(cost), 0.0)


def _label(help_text: str) -> str:
    """Short name of a setting: its description up to the first bracket or colon."""
    t = help_text.split(" (")[0].split(": ")[0].split("; ")[0].strip()
    return t if len(t) <= 52 else t[:49] + "..."


def _show(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return ", ".join(_fmt(x) for x in v)
    return _fmt(v)


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def _caster(current: Any) -> Callable[[str], Any]:
    if isinstance(current, bool):
        return lambda s: {"y": True, "yes": True, "true": True, "1": True,
                          "n": False, "no": False, "false": False, "0": False}[s.lower()]
    if isinstance(current, int):
        return int
    if isinstance(current, float) or current is None:
        return _float_or_text
    if isinstance(current, list):
        return parse_list
    return str


def _float_or_text(s: str):
    try:
        return float(s)
    except ValueError:
        return s


def parse_list(text: str) -> list:
    """'0.6, 0.8, 1.0' or a range 'start:stop:step' (stop included) -> list of numbers."""
    text = text.strip().strip("[]")
    if ":" in text:
        start, stop, step = (float(x) for x in text.split(":"))
        n = int(round((stop - start) / step)) + 1
        return [round(start + i * step, 6) for i in range(n)]
    items = [x for x in text.replace(";", ",").split(",") if x.strip()]
    out = []
    for x in items:
        x = x.strip()
        try:
            out.append(int(x) if x.lstrip("-").isdigit() else float(x))
        except ValueError:
            out.append(x)
    return out


def open_file(path: Path) -> None:
    """Open a figure with the system viewer (best effort, never fails the run)."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
    except Exception:
        pass


def rows_from(records: Iterable[dict], keys: list[str]) -> list[dict]:
    return [{k: r.get(k) for k in keys} for r in records]
