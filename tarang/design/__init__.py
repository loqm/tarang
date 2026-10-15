"""Guided device design: python -m tarang.design

Runs the TARANG pipeline as one session, for any wavelength the material data cover:

    materials -> cross-section (H x W) -> ring (radius, Q, D_int) -> ring-bus gap
    -> grating coupler -> layout (GDS, parameters) -> report

Each stage asks only what it needs (defaults shown; Enter accepts), shows its results and
figures, and, for cloud stages, the FlexCredit estimate before anything runs. Answers are
saved to <run>/design.toml, so

    python -m tarang.design --config <run>/design.toml --yes

repeats the run without questions, reusing every result already computed.

From Python: ``run(DesignConfig(...), out="my_run", interactive=False)``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from tarang.design.config import DesignConfig
from tarang.design.stages import STAGE_NAMES, STAGES, Context, settings_key
from tarang.design.ui import Prompter, Stop

__all__ = ["DesignConfig", "run", "STAGE_NAMES"]


def run(cfg: DesignConfig, out: Path | str, interactive: bool = True,
        only: Optional[Iterable[str]] = None, redo: Iterable[str] = (),
        prompter: Optional[Prompter] = None) -> Context:
    """Run the design stages in order and return the context (results in ctx.state)."""
    ui = prompter or Prompter(interactive=interactive)
    ctx = Context(cfg, ui, Path(out))
    only = set(only or STAGE_NAMES)
    redo = set(redo)
    unknown = (only | redo) - set(STAGE_NAMES)
    if unknown:
        raise ValueError(f"unknown stage(s) {sorted(unknown)}; stages are {STAGE_NAMES}")
    ui.say(f"TARANG design run '{cfg.run.name}' in {ctx.out.resolve()}")
    ui.say("Press Enter to accept a default; type q to stop (everything done so far is kept).")
    stale = False
    try:
        for i, (name, title, kind, func) in enumerate(STAGES, 1):
            if name not in only:
                continue
            saved = ctx.state["stages"].get(name)
            key = settings_key(cfg, name)
            if (saved and saved.get("key") == key and not stale and name not in redo
                    and not saved["results"].get("skipped")):
                ui.say(f"\n[{i}/{len(STAGES)}] {title}: done earlier with the same settings "
                       "(delete it from state.json or pass --redo to repeat)")
                continue
            ui.heading(f"[{i}/{len(STAGES)}] {title}  ({kind}"
                       f"{', priced before it runs' if kind == 'cloud' else ', free'})")
            results = func(ctx)
            ctx.state["stages"][name] = {"key": settings_key(cfg, name), "results": results}
            ctx.save_state()
            stale = True                         # later stages must follow this one
    except Stop:
        ui.say("\nStopped. Settings so far are in design.toml; run again to continue.")
    finally:
        ctx.save_state()
    return ctx
