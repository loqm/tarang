"""Credits and runs of one design folder: --credits, --budget, --reset-spent, --runs.

Two numbers decide whether a cloud stage may run:

* ``run.max_total_cost`` in design.toml: the LIMIT you set for this design (FlexCredits);
* ``spent`` in state.json: TARANG's running TOTAL of the estimates of everything this
  design has run.

A stage whose estimate is larger than limit - spent is not run. ``--budget`` changes the
limit; ``--reset-spent`` starts the total again at 0 (the history stays in state.json).
Neither changes your Tidy3D account: its balance and expiry are shown alongside.
"""
from __future__ import annotations

import datetime as dt
import json
from collections import defaultdict
from pathlib import Path


def _load_state(out: Path) -> dict:
    p = out / "state.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"stages": {}, "spent": 0.0}


def _save_state(out: Path, state: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "state.json").write_text(json.dumps(state, indent=1, default=str), encoding="utf-8")


def summary(cfg, out: Path, state: dict | None = None) -> dict:
    """Limit, spent, left and the spend history grouped by stage and date (from ``state``,
    or the design folder's state.json)."""
    state = _load_state(Path(out)) if state is None else state
    spent = float(state.get("spent", 0.0))
    log = state.get("spend_log", [])
    since_reset = [e for e in log if e.get("stage") != "reset"]
    resets = [i for i, e in enumerate(log) if e.get("stage") == "reset"]
    current = log[resets[-1] + 1:] if resets else since_reset
    logged = sum(float(e.get("estimate", 0.0)) for e in current if e.get("stage") != "reset")
    by = defaultdict(float)
    for e in current:
        if e.get("stage") != "reset":
            by[(str(e.get("date", ""))[:10], e.get("stage", ""), e.get("what", ""))] += float(e["estimate"])
    rows = [dict(date=d, stage=s, what=w, estimate_FC=round(c, 3)) for (d, s, w), c in sorted(by.items())]
    if spent - logged > 1e-6:
        rows.insert(0, dict(date="(before the log)", stage="", what="runs before Phase 12 "
                            "(no per-stage record)", estimate_FC=round(spent - logged, 3)))
    return dict(limit=float(cfg.run.max_total_cost), spent=spent,
                left=float(cfg.run.max_total_cost) - spent, rows=rows)


def main(cfg, out: Path, budget=None, reset=False, show=True, runs=False) -> None:
    out = Path(out)
    state = _load_state(out)
    if budget is not None:
        if budget < 0:
            raise SystemExit("--budget must be >= 0 FlexCredits")
        cfg.run.max_total_cost = float(budget)
        cfg.given.add("run.max_total_cost")
        cfg.save(out / "design.toml")
        print(f"Credit limit for this design set to {budget:g} FC (run.max_total_cost in "
              f"{out / 'design.toml'}).")
    if reset:
        old = float(state.get("spent", 0.0))
        state.setdefault("spend_log", []).append(dict(
            date=dt.datetime.now().isoformat(timespec="seconds"), stage="reset",
            what=f"spend count reset (was {old:.3f} FC)", estimate=0.0))
        state["spent"] = 0.0
        _save_state(out, state)
        print(f"Spend count reset: was {old:.3f} FC, now 0. Your Tidy3D balance is unchanged.")
    if show:
        s = summary(cfg, out)
        print(f"\nDesign folder: {out.resolve()}")
        print(f"  limit (run.max_total_cost) {s['limit']:.3f} FC")
        print(f"  spent (estimates)          {s['spent']:.3f} FC")
        print(f"  left for cloud stages      {s['left']:.3f} FC")
        if s["rows"]:
            print("\n  What it was spent on:")
            w = max(len(r["what"]) for r in s["rows"])
            for r in s["rows"]:
                print(f"    {r['date']:<16} {r['stage']:<10} {r['what']:<{w}}  {r['estimate_FC']:.3f} FC")
        try:
            from tarang.utilities.cloud_run import account_status
            acc = account_status()
        except Exception:
            acc = None
        if acc:
            exp = acc["expires"].strftime("%Y-%m-%d") if acc.get("expires") else "?"
            bal = f"{acc['credit']:.2f} FC" if acc.get("credit") is not None else "?"
            print(f"\n  Tidy3D account: {bal}, expiring {exp}" +
                  (" (EXPIRED: renew before running)" if acc.get("expired") else ""))
        else:
            print("\n  Tidy3D account: not reachable now (offline or no key).")
    if runs:
        from tarang.utilities.cloud_run import backfill_task_records, collect_runs
        n = backfill_task_records(out / "cloud")
        rows = collect_runs(out / "cloud")
        if n:
            print(f"Links found on Tidy3D for {n} earlier run(s).")
        if not rows:
            print("No cloud runs in this design yet.")
        drawn = draw_run_pictures(out, rows)
        for r in rows:
            link = r["url"] or "(no link: not in the task list of this account)"
            print(f"  {r['name']}\n      {r['status']:<10} {link}")
        missing = sum(1 for r in rows if not r["url"])
        if missing:
            print(f"\n{missing} run(s) have no link: they were made before Phase 12 with a Tidy3D "
                  "account whose task list this API key cannot see (for example an earlier "
                  "account). Their results are on disk; their pages open only in that account.")
        print(f"Geometry pictures: {out / 'figures' / 'sims'}"
              + (f" ({drawn} drawn now)" if drawn else ""))


def load_simulation(path: Path):
    """A saved simulation.json of any Tidy3D simulation type (FDTD, EME, mode solver): the
    file's "type" field names the class."""
    import json

    import tidy3d as td
    kind = json.loads(Path(path).read_text(encoding="utf-8")).get("type", "Simulation")
    cls = getattr(td, kind, None)
    if cls is None or not hasattr(cls, "from_file"):
        cls = td.Simulation
    return cls.from_file(str(path))


def draw_run_pictures(out: Path, rows: list) -> int:
    """A geometry picture (figures/sims/<cloud folder name>.png) for every run that has a
    saved simulation.json and no picture yet; returns how many were drawn."""
    from tarang.design import visuals
    folder = Path(out) / "figures" / "sims"
    n = 0
    for r in rows:
        src = Path(r["folder"]) / "simulation.json"
        dst = folder / f"{Path(r['folder']).name}.png"
        if dst.exists() or not src.exists():
            continue
        try:
            visuals.sim_picture(load_simulation(src), dst, r["name"])
            n += 1
        except Exception as err:                      # noqa: BLE001
            print(f"  (picture of {r['name']} not drawn: {type(err).__name__}: {err})")
    return n
