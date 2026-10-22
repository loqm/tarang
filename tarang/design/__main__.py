"""python -m tarang.design: guided design from materials to layout.

    python -m tarang.design                         # asks everything (defaults shown)
    python -m tarang.design --out runs/sin1550      # choose the run folder
    python -m tarang.design --config runs/sin1550/design.toml --yes   # repeat, no questions
    python -m tarang.design --config design.toml --only gap,grating   # selected stages
    python -m tarang.design --list                  # list the stages
    python -m tarang.design --config <run>/design.toml --review     # ask saved settings again
    python -m tarang.design --config <run>/design.toml --credits    # budget, spent, Tidy3D balance
    python -m tarang.design --config <run>/design.toml --budget 25  # set the credit limit
    python -m tarang.design --config <run>/design.toml --reset-spent  # restart the spend count
    python -m tarang.design --config <run>/design.toml --runs       # every cloud run + Tidy3D link
    python -m tarang.design --config <run>/design.toml --price-only # estimates only, nothing run
    python -m tarang.design --config <run>/design.toml --yes --spend-cap 1.5   # spend <= 1.5 FC

Stop at any time with q (at a question) or Ctrl+C: finished work is kept, and the same
command continues; a cloud run that was in progress is picked up again, not paid twice.
Cloud stages show their FlexCredit estimate and wait for confirmation; with --yes they run
only if the estimate fits run.max_total_cost.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tarang import configure_logging, set_tidy3d_output
from tarang.design import STAGE_NAMES, run
from tarang.design.config import DesignConfig
from tarang.design.stages import STAGES
from tarang.design.ui import Prompter


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tarang.design",
                                 description="Guided device design with TARANG.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--config", type=Path, help="settings file (design.toml) to start from")
    ap.add_argument("--out", type=Path, help="run folder (default: design_runs/<name>)")
    ap.add_argument("--yes", action="store_true",
                    help="no questions: use the config and defaults")
    ap.add_argument("--only", help=f"comma-separated stages to run ({','.join(STAGE_NAMES)})")
    ap.add_argument("--redo", default="", help="comma-separated stages to repeat")
    ap.add_argument("--show", action="store_true", help="open each figure as it is made")
    ap.add_argument("--list", action="store_true", help="list the stages and exit")
    ap.add_argument("--review", action="store_true",
                    help="show every stage's settings for changes, even those saved in design.toml")
    ap.add_argument("--budget", type=float, metavar="FC",
                    help="set run.max_total_cost (FlexCredits for this design) and save it")
    ap.add_argument("--reset-spent", action="store_true",
                    help="restart this design's spend count at 0 (the history is kept)")
    ap.add_argument("--credits", action="store_true",
                    help="show budget, spent, what it was spent on and the Tidy3D balance; exit")
    ap.add_argument("--runs", action="store_true",
                    help="list every cloud run of this design with its Tidy3D link; exit")
    ap.add_argument("--price-only", action="store_true",
                    help="go through the stages and print every cloud estimate; run nothing priced")
    ap.add_argument("--spend-cap", type=float, metavar="FC",
                    help="spend at most this much in this session (cloud steps above it are skipped)")
    a = ap.parse_args(argv)
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(errors="replace")
    if a.list:
        for i, (name, title, kind, _) in enumerate(STAGES, 1):
            print(f"{i}. {name:14s} {title} ({kind})")
        return 0

    configure_logging("WARNING")
    set_tidy3d_output(False)
    cfg = DesignConfig.load(a.config) if a.config else DesignConfig()
    interactive = not a.yes and sys.stdin.isatty()
    ui = Prompter(interactive=interactive, show_figures=a.show, review=a.review,
                  price_only=a.price_only, spend_cap=a.spend_cap)
    _ctrl_break_stops()
    housekeeping = a.budget is not None or a.reset_spent or a.credits or a.runs
    if a.out:
        out = a.out
    elif a.config:
        out = a.config.resolve().parent
    elif housekeeping:
        print("Give the design with --config <run>/design.toml (or --out <run folder>).")
        return 2
    else:
        ui.value(cfg, "run", "name")
        out = Path("design_runs") / cfg.run.name
    if housekeeping:
        from tarang.design import credits
        saved = Path(out) / "design.toml"
        if not a.config and saved.exists():          # --out alone: use that design's settings
            cfg = DesignConfig.load(saved)
        credits.main(cfg, Path(out), budget=a.budget, reset=a.reset_spent,
                     show=a.credits or a.budget is not None or a.reset_spent, runs=a.runs)
        return 0
    only = [s.strip() for s in a.only.split(",")] if a.only else None
    redo = [s.strip() for s in a.redo.split(",") if s.strip()]
    run(cfg, out, interactive=interactive, only=only, redo=redo, prompter=ui)
    return 0


def _ctrl_break_stops() -> None:
    """On Windows, Ctrl+Break (what the GUI's Stop button sends) stops like Ctrl+C, so
    finished work is saved."""
    import signal
    if hasattr(signal, "SIGBREAK"):
        def stop(signum, frame):
            raise KeyboardInterrupt
        try:
            signal.signal(signal.SIGBREAK, stop)
        except ValueError:                     # not the main thread
            pass


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    sys.exit(main())
