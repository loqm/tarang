"""Guided device design: python -m tarang.design

Runs the TARANG pipeline as one session, for any wavelength the material data cover:

    materials -> cross-section (H x W) -> ring (radius, Q, D_int) -> ring-bus gap
    -> grating coupler -> checks -> taper -> apodised / focusing grating (optional)
    -> layout (GDS, parameters) -> whole device (circuit model) -> report

Each stage starts with one table of its settings (Enter accepts them, a number changes
one, q or Ctrl+C stops), shows its results and figures, and, for cloud stages, the
FlexCredit estimate before anything runs. Answers are saved to <run>/design.toml, so

    python -m tarang.design --config <run>/design.toml --yes

repeats the run without questions, reusing every result already computed.

From Python: ``run(DesignConfig(...), out="my_run", interactive=False)``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from tarang.design.config import DesignConfig
from tarang.design.stages import STAGE_NAMES, STAGES, Context, applies, pump_mode, settings_key
from tarang.design.ui import Prompter, Stop

#: steps designed for TE00 only so far (a TM00 design stops before them)
TE_ONLY_STAGES = {"grating", "checks", "taper", "apodise", "focus", "edge", "layout", "device"}

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
    ui.say("Each stage shows its settings: Enter accepts them, a number changes one. Type q "
           "(or press Ctrl+C) to stop at any point; everything done so far is kept and the "
           "same command continues from there.")
    ui.say("Figures are saved in figures/ (and opened with --show); the report and a "
           "benchmark_pack/ folder are written at the end.")
    ctx.record_versions()
    for note in ctx.apply_legacy():
        ui.say("  " + note)
    stale = False
    from tarang.design.stages import COUPLINGS, GOALS
    ui.form(cfg, "What to design", [
        dict(section="run", key="goal",
             check=lambda v: "" if v in GOALS else "answer comb_ring or loss_test"),
        dict(section="run", key="coupling",
             check=lambda v: "" if str(v).lower() in COUPLINGS else "answer grating or edge")])
    if cfg.run.goal not in GOALS:
        raise ValueError(f"run.goal must be one of {GOALS}, not {cfg.run.goal!r}")
    cfg.run.coupling = str(cfg.run.coupling or "grating").lower()
    if cfg.run.coupling not in COUPLINGS:
        raise ValueError(f"run.coupling must be one of {COUPLINGS}, not {cfg.run.coupling!r}")
    from tarang.design.config import SettingsError, implausible
    bad = implausible(cfg)
    if bad:                                          # nearly always nm typed for µm
        raise SettingsError("Settings outside their plausible range (fix them in design.toml or "
                         "the GUI, then run again):\n" +
                         "\n".join(f"  {k}: {v}" for k, v in bad.items()))
    _show_assumptions(cfg, ui)                     # after the choice: they depend on the coupling
    try:
        credit_limits(ctx)
        for i, (name, title, kind, func) in enumerate(STAGES, 1):
            if name not in only:
                continue
            if not applies(name, cfg):
                if name in set(only) - {"report"} and len(only) < len(STAGE_NAMES):
                    ui.say(f"\n[{i}/{len(STAGES)}] {title}: not part of a '{cfg.run.goal}' design "
                           f"with {cfg.run.coupling} coupling (run.goal, run.coupling); skipped")
                continue
            saved = ctx.state["stages"].get(name)
            key = settings_key(cfg, name)
            if (saved and saved.get("key") == key and not stale and name not in redo
                    and not saved["results"].get("skipped")):
                ui.say(f"\n[{i}/{len(STAGES)}] {title}: done earlier with the same settings "
                       "(delete it from state.json or pass --redo to repeat)")
                continue
            if name in TE_ONLY_STAGES and pump_mode(cfg) == "TM00":
                raise SettingsError(
                    f"pump_mode = TM00: TARANG designs the cross-section, the ring, the test "
                    f"waveguides and the ring-bus gap for TM00, but the '{name}' step (couplers, "
                    "their checks, the layout and the whole-device model built on them) is "
                    "designed for TE00 only so far. Run the TM00 steps with "
                    "--only materials,cross_section,ring,waveguides,gap,report.")
            ui.heading(f"[{i}/{len(STAGES)}] {title}  ({kind}"
                       f"{', priced before it runs' if kind == 'cloud' else ', free'})")
            ctx.stage = name
            results = func(ctx)
            ctx.state["stages"][name] = {"key": settings_key(cfg, name), "results": results}
            ctx.save_state()
            stale = True                         # later stages must follow this one
            if (results or {}).get("awaiting_choice"):
                ui.say("\nStopped for your choice (see the candidates above); run again to "
                       "continue.")
                break
    except Stop:
        ui.say("\nStopped. Settings so far are in design.toml; run the same command again to "
               "continue.")
    except KeyboardInterrupt:
        ui.say("\nStopped (Ctrl+C). Finished stages and finished cloud runs are kept. A cloud "
               "run that was in progress keeps running on Tidy3D; the same command continues "
               "and picks it up again without paying twice.")
    finally:
        ctx.save_state()
        if getattr(ui, "price_only", False):
            paid = [(w, c) for w, c in ui.priced if c > 0]
            ui.say("\nPrice only: nothing that costs was run. Estimates:" if paid else
                   "\nPrice only: no cloud step needs paying for these stages.")
            if paid:
                ui.table([dict(step=w, estimate_FC=round(c, 3)) for w, c in paid])
                ui.say(f"  Total {sum(c for _, c in paid):.3f} FC. Later steps that depend on these "
                       "runs are priced once those runs exist.")
    return ctx


def limit_checks(spent: float, credit: Optional[float], total_now: float):
    """Checks for the two credit limits: (per-run check, total check). Each returns an
    error message or "". The total may not exceed what this design has spent plus what the
    account holds; a single run may not exceed the total or the account balance."""
    def total(v) -> str:
        v = float(v)
        if v < 0:
            return "a limit cannot be negative"
        if credit is not None and v > spent + credit + 1e-9:
            return (f"the account holds {credit:.2f} FC; with {spent:.3f} FC already spent by this "
                    f"design the limit can be at most {spent + credit:.2f} FC")
        return ""

    def per_run(v) -> str:
        v = float(v)
        if v <= 0:
            return "a run limit must be above 0"
        if v > total_now + 1e-9:
            return f"above this design's total limit ({total_now:g} FC)"
        if credit is not None and v > credit + 1e-9:
            return f"above the account balance ({credit:.2f} FC)"
        return ""
    return per_run, total


def credit_limits(ctx) -> Optional[dict]:
    """One table at the start of every run: the largest cost of a single cloud run and the
    budget of this design, checked against the Tidy3D account balance (when it can be
    read). Returns the account status (or None)."""
    from tarang.utilities import cloud_run as cr
    cfg, ui = ctx.cfg, ctx.ui
    acc = cr.account_status(timeout=8.0) if cr.api_key_source() else None
    credit = acc.get("credit") if acc else None
    ui.say("")
    if acc and credit is not None:
        when = acc["expires"].strftime("%Y-%m-%d") if acc.get("expires") else "?"
        ui.say(f"Tidy3D account: {credit:.2f} FlexCredits, valid until {when}"
               + (" (EXPIRED: cloud runs will be refused)" if acc.get("expired") else "") + ".")
    else:
        ui.say("Tidy3D account: balance not readable now (no API key, or offline); the limits "
               "below are not checked against it.")
    ui.say(f"This design has spent {ctx.spent:.3f} FlexCredits so far (estimates).")
    per_run, total = limit_checks(ctx.spent, credit, float(cfg.run.max_total_cost))
    ui.form(cfg, "Credit limits (Enter keeps them; a row number changes one)", [
        dict(section="run", key="max_total_cost", check=total),
        dict(section="run", key="max_cost_per_run",
             check=lambda v: limit_checks(ctx.spent, credit, float(cfg.run.max_total_cost))[0](v))],
        always=True)
    for msg in (total(cfg.run.max_total_cost),
                limit_checks(ctx.spent, credit, float(cfg.run.max_total_cost))[0](cfg.run.max_cost_per_run)):
        if msg:
            ui.say(f"  Note: {msg}. Cloud stages check the account again before they run.")
    ui.say(f"Credits: limit {cfg.run.max_total_cost:g} FC for this design, {ctx.spent:.3f} spent, "
           f"{ctx.budget_left:.3f} left; at most {cfg.run.max_cost_per_run:g} FC per cloud run "
           "(--credits shows what was spent on what).")
    ctx.save_state()
    return acc


def _show_assumptions(cfg, ui) -> None:
    """The values this design assumes (not computes), with their sources."""
    from tarang.design import assumptions
    try:
        rows = assumptions.table(cfg)
    except ValueError as err:                    # e.g. an unknown loss preset in the file
        ui.say(f"  ({err})")
        return
    ui.say("\nAssumed values (edit them in design.toml, or answer the questions as they "
           "come; your own values replace the published ones):")
    ui.table([dict(quantity=r["quantity"], value=r["value"], unit=r["unit"],
                   source=_short(r["source"])) for r in rows])


def _short(text: str, n: int = 60) -> str:
    text = str(text)
    return text if len(text) <= n else text[: n - 3] + "..."
