"""python -m tarang.design: guided design from materials to layout.

    python -m tarang.design                         # asks everything (defaults shown)
    python -m tarang.design --out runs/sin1550      # choose the run folder
    python -m tarang.design --config runs/sin1550/design.toml --yes   # repeat, no questions
    python -m tarang.design --config design.toml --only gap,grating   # selected stages
    python -m tarang.design --list                  # list the stages

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
    ui = Prompter(interactive=interactive, show_figures=a.show)
    if a.out:
        out = a.out
    elif a.config:
        out = a.config.resolve().parent
    else:
        ui.value(cfg, "run", "name")
        out = Path("design_runs") / cfg.run.name
    only = [s.strip() for s in a.only.split(",")] if a.only else None
    redo = [s.strip() for s in a.redo.split(",") if s.strip()]
    run(cfg, out, interactive=interactive, only=only, redo=redo, prompter=ui)
    return 0


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    sys.exit(main())
