"""The stages of a design run, from materials to layout.

Each stage asks only for the settings it needs, does its work, saves its files in the
run folder, prints a summary and returns a dict of results that later stages use. A
stage that has already run with the same settings is not repeated (its saved results
are reused); local stages are free, cloud stages are priced and confirmed first.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from tarang import get_logger
from tarang.design import physics
from tarang.design.config import DesignConfig
from tarang.design.ui import Prompter, parse_list

logger = get_logger(__name__)
C_UM = 2.99792458e14                 # um/s: f = C_UM / wavelength_um gives Hz


# =============================================================================== context
class Context:
    """Everything a stage needs: settings, the prompter, the run folder and saved state."""

    def __init__(self, cfg: DesignConfig, ui: Prompter, out: Path):
        self.cfg, self.ui, self.out = cfg, ui, Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.state_path = self.out / "state.json"
        self.state: dict = (json.loads(self.state_path.read_text(encoding="utf-8"))
                            if self.state_path.exists() else {"stages": {}, "spent": 0.0})
        self.state.setdefault("spend_log", [])
        self._media: dict = {}
        from tarang.design import visuals
        try:
            visuals.configure(cfg.run.figure_formats, cfg.run.figure_dpi)
        except ValueError as err:
            ui.say(f"  ({err}; using png + pdf)")
        self.stage = ""

    # ---- versions and settings added after a run was made ------------------------- #
    def record_versions(self) -> None:
        import datetime as dt
        import platform
        vers = {"python": platform.python_version()}
        for mod in ("tarang", "tidy3d", "numpy"):
            try:
                vers[mod] = __import__(mod).__version__
            except Exception:
                vers[mod] = "?"
        vers["date"] = dt.datetime.now().isoformat(timespec="seconds")
        self.state["versions"] = vers

    def apply_legacy(self) -> list[str]:
        """Use, for settings added after this run's stages were computed, the values that
        reproduce those results (see config.LEGACY); returns notes for the user."""
        from tarang.design.config import LEGACY
        notes = []
        for sec, values in LEGACY.items():
            done = self.state.get("stages", {}).get(sec, {}).get("results", {})
            if not done or done.get("skipped"):
                continue
            for key, old in values.items():
                if f"{sec}.{key}" in self.cfg.given:
                    continue
                setattr(getattr(self.cfg, sec), key, old)
                self.cfg.given.add(f"{sec}.{key}")
                notes.append(f"{sec}.{key} = {old}: kept as in this run's saved {sec} results "
                             "(change it in design.toml to use the new default).")
        return notes

    # ---- persistence ----------------------------------------------------------------- #
    def save_state(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=1, default=_jsonable),
                                   encoding="utf-8")
        self.cfg.save(self.out / "design.toml")

    def results(self, stage: str) -> dict:
        try:
            return self.state["stages"][stage]["results"]
        except KeyError:
            raise RuntimeError(f"Stage '{stage}' has not run yet; run it first.") from None

    @property
    def spent(self) -> float:
        return float(self.state.get("spent", 0.0))

    def add_spent(self, cost: float, what: str = "") -> None:
        import datetime as dt
        cost = float(cost)
        self.state["spent"] = self.spent + cost
        if cost > 0:
            self.state["spend_log"].append(dict(date=dt.datetime.now().isoformat(timespec="seconds"),
                                                stage=self.stage, what=what, estimate=cost))

    @property
    def budget_left(self) -> float:
        return float(self.cfg.run.max_total_cost) - self.spent

    # ---- helpers --------------------------------------------------------------------- #
    def path(self, *parts: str) -> Path:
        p = self.out.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def tool(self, name: str, *args: Any) -> None:
        """Run python -m tarang.waveguide.dispersion_mapping.<name> in the run folder,
        showing its progress lines."""
        cmd = [sys.executable, "-m", f"tarang.waveguide.dispersion_mapping.{name}",
               *map(str, args)]
        self.ui.say(f"  $ {' '.join(cmd[1:])}")
        env = dict(os.environ, PYTHONIOENCODING="utf-8")     # λ, µm on any Windows console
        proc = subprocess.Popen(cmd, cwd=self.out, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace", env=env)
        tail: list[str] = []
        for line in proc.stdout:                                  # type: ignore[union-attr]
            line = line.rstrip()
            if line.strip() and not _noise(line):
                self.ui.say("    " + line)
            tail = (tail + [line])[-30:]
        if proc.wait():
            raise RuntimeError(f"{name} failed:\n" + "\n".join(tail))

    def medium(self, key: str):
        """Material fitted over the run's band (cached for the run)."""
        from tarang.media import get_medium
        if key not in self._media:
            if str(key).lower() in {"air", "none", ""}:
                import tidy3d as td
                self._media[key] = td.Medium(permittivity=1.0)
            else:
                self._media[key] = get_medium(key, wl_range=self.band)
        return self._media[key]

    @property
    def band(self) -> tuple[float, float]:
        wl, half = self.cfg.run.wavelength_um, self.cfg.run.band_um
        return round(wl - half, 4), round(wl + half, 4)

    def n(self, key: str, wl: Optional[float] = None) -> float:
        wl = wl or self.cfg.run.wavelength_um
        return float(np.real(np.sqrt(self.medium(key).eps_model(C_UM / wl))))

    def stack_args(self) -> list[str]:
        m, s = self.cfg.materials, self.cfg.stack
        return ["--core", m.core, "--substrate", m.box, "--cladding", m.cladding,
                "--box", s.box_height_um, "--cladding-height", s.cladding_height_um]

    def cloud_limit(self):
        """Context manager: cap every cloud run at run.max_cost_per_run."""
        return _EnvVar("TARANG_MAX_COST", str(self.cfg.run.max_cost_per_run))

    def cloud_progress(self, label: str, total: int):
        """Context manager printing one line per cloud run of this stage:
        '[3/8] gap 0.6 um: running on the cloud (estimate 0.49 FC) ... done in 2 min 05 s'."""
        from tarang.utilities.cloud_run import progress_listener
        from tarang.design.visuals import fmt_minutes
        count = {"n": 0}
        seen: set = set()
        ui = self.ui

        def show(event: str, info: dict) -> None:
            name = info.get("name", "")
            if event == "cached":
                # a run loaded again (e.g. for its field map) is counted once
                tag = info.get("fingerprint") or name
                if tag in seen:
                    return
                seen.add(tag)
                count["n"] += 1
                ui.say(f"  [{count['n']}/{total}] {label}: {name} - already computed, loaded")
            elif event == "running":
                ui.say(f"  [{count['n'] + 1}/{total}] {label}: {name} - running on the cloud "
                       f"(estimate {info.get('cost', 0):.3f} FC) ...")
                if info.get("url"):
                    ui.say(f"      Tidy3D: {info['url']}")
            elif event == "submitted":                    # one task of a batch
                ui.say(f"      {name}: {info.get('url', '')}")
            elif event == "reattached":
                ui.say(f"  {label}: {name} was submitted before the last stop and is "
                       f"{info.get('status', 'running')} on Tidy3D; following it (no new charge)")
                if info.get("url"):
                    ui.say(f"      Tidy3D: {info['url']}")
            elif event == "done":
                if info.get("fingerprint"):
                    seen.add(info["fingerprint"])
                count["n"] += 1
                ui.say(f"      done in {fmt_minutes(info.get('seconds', 0))}")
        return progress_listener(show)

    def account_ok(self, cost: Optional[float]) -> bool:
        """False (with the reason) if the Tidy3D account cannot pay for ``cost``:
        FlexCredits expired, or fewer left than the estimate. True if fine or unknown."""
        from tarang.utilities.cloud_run import account_status
        acc = account_status()
        if acc is None:
            return True
        if acc["expired"]:
            when = acc["expires"].strftime("%Y-%m-%d") if acc["expires"] else "?"
            self.ui.say(f"  Your FlexCredits expired on {when}, so Tidy3D will refuse to start "
                        "runs. Renew them (or switch account with `tidy3d configure`) and run "
                        "this command again; nothing was run.")
            return False
        if cost is not None and acc["credit"] is not None and acc["credit"] < cost:
            self.ui.say(f"  The account has {acc['credit']:.2f} FlexCredits, less than the "
                        f"estimate ({cost:.2f}); nothing was run.")
            return False
        return True


class _EnvVar:
    def __init__(self, name, value):
        self.name, self.value, self.old = name, value, None

    def __enter__(self):
        self.old = os.environ.get(self.name)
        os.environ[self.name] = self.value

    def __exit__(self, *exc):
        if self.old is None:
            os.environ.pop(self.name, None)
        else:
            os.environ[self.name] = self.old


def _noise(line: str) -> bool:
    """Tidy3D chatter that is not useful in the design log."""
    keys = ("Best weighted RMS", "passivity", "AdvancedFastFitterParam", "consider increasing",
            "subpixel", "tidy3d-extras", "use_local_subpixel", "tidy3d.plugins.mode.web",
            "decay at the plane", "WARNING: ")
    return any(k in line for k in keys)


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _grid(lo: float, hi: float, step: float) -> list[float]:
    n = int(round((hi - lo) / step)) + 1
    return [round(lo + i * step, 4) for i in range(n)]


def _step(x: float) -> float:
    """A round step close to x (0.01, 0.02, 0.025, 0.05, 0.1, ...)."""
    mags = [m * 10 ** e for e in range(-3, 1) for m in (1, 2, 2.5, 5)]
    return min(mags, key=lambda m: abs(math.log(m / x)))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if "error" not in r]


def _losses(cfg) -> tuple[float, float]:
    """Scattering and absorption loss (dB/cm): your values, else the loss preset's."""
    from tarang.design.assumptions import preset
    pre = preset(cfg.ring.loss_preset)
    scat = cfg.ring.scattering_loss_db_cm
    absl = cfg.ring.absorption_loss_db_cm
    return (pre["scattering_db_cm"] if scat is None else float(scat),
            pre["absorption_db_cm"] if absl is None else float(absl))


def _wavelength_ok(v) -> str:
    try:
        return "" if 0.2 < float(v) < 20 else "wavelength must be in um, e.g. 1.55"
    except (TypeError, ValueError):
        return "wavelength must be a number in um, e.g. 1.55"


def _heights_default(cfg) -> list:
    wl, H, dH = cfg.run.wavelength_um, cfg.stack.core_height_um, cfg.cross_section.tolerance_height_um
    if H:
        return [round(H - dH, 4), round(H, 4), round(H + dH, 4)]
    st = _step(0.032 * wl)
    return _grid(physics.round_to(0.32 * wl, st), physics.round_to(0.58 * wl, st), st)


def _widths_default(cfg) -> list:
    wl = cfg.run.wavelength_um
    st = _step(0.065 * wl)
    return _grid(physics.round_to(0.39 * wl, st), physics.round_to(1.16 * wl, st), st)


def _gaps_default(cfg) -> list:
    wl = cfg.run.wavelength_um
    st = _step(0.065 * wl)
    return _grid(physics.round_to(0.19 * wl, st), physics.round_to(0.52 * wl, st), st)


def _budget_items() -> list:
    return [dict(section="run", key="max_cost_per_run"), dict(section="run", key="max_total_cost")]


def _material_ok(key: str) -> str:
    """'' if key is a known material, a CSV file that exists, or air."""
    from tarang.media.medium_helper import _media_database
    k = str(key).strip()
    if k.lower() in {"air", "none"}:
        return ""
    if k.lower().endswith(".csv"):
        return "" if Path(k).expanduser().exists() else f"file {k} not found"
    return "" if k.lower() in _media_database() else (
        f"unknown material {k!r}; use a key from tarang/media/media.json or a .csv path")


# ============================================================================== stages
def stage_materials(ctx: Context) -> dict:
    cfg, ui = ctx.cfg, ctx.ui
    handle_ok = lambda v: "" if str(v).lower() == "none" else _material_ok(v)  # noqa: E731
    ui.form(cfg, "Materials", [
        dict(section="run", key="wavelength_um", check=_wavelength_ok),
        dict(section="run", key="band_um", default=lambda c: round(0.13 * c.run.wavelength_um, 3)),
        dict(section="materials", key="core", check=_material_ok),
        dict(section="materials", key="box", check=_material_ok),
        dict(section="materials", key="cladding", check=_material_ok),
        dict(section="materials", key="handle", check=handle_ok)])
    wl = ui.value(cfg, "run", "wavelength_um", check=_wavelength_ok)
    ui.value(cfg, "run", "band_um", default=round(0.13 * wl, 3))
    for key in ("core", "box", "cladding"):
        ui.value(cfg, "materials", key, check=_material_ok)
    ui.value(cfg, "materials", "handle", check=lambda v: "" if str(v).lower() == "none"
             else _material_ok(v))

    from tarang.media.medium_helper import _material_path
    lo, hi = ctx.band
    rows, curves = [], {}
    wls = np.linspace(lo, hi, 201)
    for role in ("core", "box", "cladding", "handle"):
        key = getattr(cfg.materials, role)
        if str(key).lower() in {"air", "none"}:
            rows.append(dict(role=role, material=key, data_range="-", n_at_wl=1.0 if
                             str(key).lower() == "air" else None, covers_band="yes"))
            continue
        path = _material_path(key)
        data = np.genfromtxt(path, delimiter=",", names=True, encoding="utf-8")
        dmin, dmax = float(np.nanmin(data["wl"])), float(np.nanmax(data["wl"]))
        med = ctx.medium(key)
        nk = med.nk_model(C_UM / wls)
        curves[role] = (key, np.real(nk[0]), np.real(nk[1]))
        rows.append(dict(role=role, material=key, data_range=f"{dmin:.3f}-{dmax:.3f}",
                         n_at_wl=round(ctx.n(key), 5),
                         covers_band="yes" if dmin <= lo and hi <= dmax else "NO (extrapolated)"))
    ui.say(f"Band fitted: {lo}-{hi} um around {wl} um")
    ui.table(rows)
    if any(r["covers_band"].startswith("NO") for r in rows):
        ui.say("  Warning: a material's data do not cover the whole band; its index there is "
               "an extrapolation. Choose another dataset or a narrower band.")

    table = ctx.path("tables", "materials_nk.csv")
    with open(table, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        header = ["wavelength_um"]
        for role, (key, _, _) in curves.items():
            header += [f"n_{role}", f"k_{role}"]
        w.writerow(header)
        for i, x in enumerate(wls):
            row = [round(float(x), 6)]
            for _, n, k in curves.values():
                row += [round(float(n[i]), 8), round(float(k[i]), 10)]
            w.writerow(row)
    fig = _plot_materials(ctx, wls, curves)
    ui.figure(fig)
    ui.say(f"  table : {table}")
    return dict(band=[lo, hi], materials=rows, table=str(table), figure=str(fig))


def _plot_materials(ctx, wls, curves) -> Path:
    from tarang.design import visuals
    plt = visuals._plt()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for role, (key, n, _) in curves.items():
        ax.plot(wls, n, lw=2, label=f"{role}: {Path(str(key)).stem}")
    ax.axvline(ctx.cfg.run.wavelength_um, color="0.5", ls=":")
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("Refractive index n (fitted)")
    ax.grid(alpha=0.3)
    ax.legend()
    from tarang.design import visuals
    return visuals._save(fig, ctx.path("figures", "1_materials.png"))


def stage_cross_section(ctx: Context) -> dict:
    cfg, ui = ctx.cfg, ctx.ui
    wl = cfg.run.wavelength_um
    clad_not_air = str(cfg.materials.cladding).lower() != "air"
    ui.form(cfg, "Cross-section map", [
        dict(section="stack", key="core_height_um"),
        dict(section="stack", key="box_height_um"),
        dict(section="stack", key="cladding_height_um", show=clad_not_air),
        dict(section="cross_section", key="tolerance_width_um"),
        dict(section="cross_section", key="tolerance_height_um"),
        dict(section="cross_section", key="heights_um", default=_heights_default, cast=parse_list),
        dict(section="cross_section", key="widths_um", default=_widths_default, cast=parse_list),
        dict(section="cross_section", key="map_steps",
             check=lambda v: "" if v >= 30 else "use at least 30 (50 for maps)"),
        dict(section="run", key="workers")])
    H_fixed = ui.value(cfg, "stack", "core_height_um")
    ui.value(cfg, "stack", "box_height_um")
    if str(cfg.materials.cladding).lower() != "air":
        ui.value(cfg, "stack", "cladding_height_um")
    dW = ui.value(cfg, "cross_section", "tolerance_width_um")
    dH = ui.value(cfg, "cross_section", "tolerance_height_um")
    if H_fixed:
        hdef = [round(H_fixed - dH, 4), round(H_fixed, 4), round(H_fixed + dH, 4)]
    else:
        st = _step(0.032 * wl)
        hdef = _grid(physics.round_to(0.32 * wl, st), physics.round_to(0.58 * wl, st), st)
    st = _step(0.065 * wl)
    wdef = _grid(physics.round_to(0.39 * wl, st), physics.round_to(1.16 * wl, st), st)
    heights = ui.value(cfg, "cross_section", "heights_um", default=hdef, cast=parse_list)
    widths = ui.value(cfg, "cross_section", "widths_um", default=wdef, cast=parse_list)
    steps = ui.value(cfg, "cross_section", "map_steps", check=lambda v: "" if v >= 30 else
                     "use at least 30 (50 for maps)")
    ui.value(cfg, "run", "workers")
    n_pts = len(heights) * len(widths)
    ui.say(f"Mapping {len(heights)} heights x {len(widths)} widths = {n_pts} cross-sections "
           f"at {steps} steps/λ (local, free; roughly {max(1, n_pts // cfg.run.workers)} min).")

    grid_file = ctx.path("results", f"map_{steps}steps.jsonl")
    ctx.tool("sweep", "--heights", ",".join(map(str, heights)), "--widths",
             ",".join(map(str, widths)), "--wl", wl, "--steps", steps,
             "--workers", cfg.run.workers, "--out", grid_file.relative_to(ctx.out),
             *ctx.stack_args())
    ctx.tool("maps", "--data", grid_file.relative_to(ctx.out), "--tm", "--overwrite")
    plots = ctx.out / "results" / f"plots_{grid_file.stem}"
    ui.figure(plots / "design_map.png")

    rows = [r for r in _read_jsonl(grid_file)
            if r.get("steps") == steps and abs(r.get("wl0", 1.55) - wl) < 1e-9
            and round(r["H"], 4) in {round(h, 4) for h in heights}
            and round(r["W"], 4) in {round(w, 4) for w in widths}]
    ranked = _rank_targets(rows, heights, widths)
    if not ranked:
        ui.say("No cross-section in this map is anomalous AND single-mode. Widen the map "
               "(cross_section.heights_um / widths_um) or change the stack, and rerun.")
        raise RuntimeError("no target geometry found")
    ui.say("\nCandidates (anomalous D, one guided TE mode), most robust first:")
    ui.table(ranked[:8], ["H_um", "W_um", "D_TE0", "ng_TE0", "guided_TE", "guided_TM",
                          "TE_TM_split", "target_neighbours"],
             {"D_TE0": ".1f", "ng_TE0": ".4f", "TE_TM_split": ".4f"})
    ui.say("  Ranked by how many of the 8 neighbouring map points are also targets, then by "
           "TE_TM_split = |n_eff(TE00) - n_eff(TM00)|: a larger split keeps the TE and TM "
           "mode families further from crossing (fewer avoided crossings in the comb).")
    best = ranked[0]
    ui.form(cfg, "Chosen cross-section (default: the most robust candidate)", [
        dict(section="cross_section", key="height_um", default=best["H_um"]),
        dict(section="cross_section", key="width_um", default=best["W_um"]),
        dict(section="cross_section", key="decision_steps")])
    H = ui.value(cfg, "cross_section", "height_um", default=best["H_um"])
    W = ui.value(cfg, "cross_section", "width_um", default=best["W_um"])

    dsteps = ui.value(cfg, "cross_section", "decision_steps")
    ui.say(f"\nChecking {H} x {W} um at {dsteps} steps/λ with its fabrication tolerance "
           f"(W ± {dW}, H ± {dH} um)...")
    fin_file = ctx.path("results", f"finalist_{dsteps}steps.jsonl")
    ctx.tool("sweep", "--points", f"{H}x{W}", "--tol", f"{dW},{dH}", "--wl", wl,
             "--steps", dsteps, "--workers", cfg.run.workers,
             "--out", fin_file.relative_to(ctx.out), *ctx.stack_args())
    ctx.tool("report", "--straight", fin_file.relative_to(ctx.out), "--finalists", f"{H}x{W}",
             "--tol", f"{dW},{dH}", "--alpha-scat", _losses(cfg)[0],
             "--alpha-abs", _losses(cfg)[1], "--n2", cfg.assumptions.n2_m2_w,
             "--out", "results/finalist_report", "--overwrite")
    centre = next(r for r in _read_jsonl(fin_file)
                  if abs(r["H"] - H) < 1e-6 and abs(r["W"] - W) < 1e-6 and r["steps"] == dsteps
                  and r.get("bend_radius") is None)
    with open(ctx.out / "results" / "finalist_report" / "finalists.csv", encoding="utf-8") as f:
        rep = next(csv.DictReader(f))
    robust = str(rep.get("all_neighbours_target")).lower() == "true"
    ui.say(f"\nCross-section {H*1e3:.0f} x {W*1e3:.0f} nm at {dsteps} steps/λ: "
           f"D = {centre['D_TE0']:.1f} ps/(nm km), n_g = {centre['ng_TE0']:.4f}, "
           f"guided TE/TM = {centre['guided_TE']}/{centre['guided_TM']}; worst-case D within the "
           f"tolerance = {float(rep['D_min_tol']):.1f}; whole tolerance box target: "
           f"{'yes' if robust else 'NO'}")
    if not robust:
        ui.say("  Warning: part of the tolerance box is normal-dispersion or multimode. "
               "Consider another candidate.")

    conv_file = ctx.path("results", "convergence.jsonl")
    ladder = ",".join(map(str, sorted({steps, dsteps, dsteps + 20})))
    ctx.tool("convergence", "--points", f"{H}x{W}", "--steps-list", ladder,
             "--wl", wl, "--workers", cfg.run.workers,
             "--data", conv_file.relative_to(ctx.out), "--overwrite", *ctx.stack_args())
    conv = sorted((r for r in _read_jsonl(conv_file)
                   if abs(r["H"] - H) < 1e-6 and abs(r["W"] - W) < 1e-6), key=lambda r: r["steps"])
    if len(conv) >= 2:
        change = abs(conv[-1]["D_TE0"] - conv[-2]["D_TE0"]) / max(abs(conv[-1]["D_TE0"]), 1e-9)
        ui.say("Mesh check: D = " + ", ".join(f"{r['D_TE0']:.1f} ({r['steps']})" for r in conv)
               + f"; last change {change * 100:.1f} %")
    else:
        change = None
    from tarang.design import visuals
    fig_xs = visuals.safe(ctx, "cross-section", visuals.cross_section, ctx, H, W,
                          ctx.path("figures", "2_cross_section.png"))
    fig_mode_profiles(ctx, H, W)
    return dict(height_um=H, width_um=W, steps=dsteps, D=centre["D_TE0"], n_group=centre["ng_TE0"],
                figure=str(fig_xs) if fig_xs else None, convergence=[
                    dict(steps=r["steps"], D=r["D_TE0"]) for r in conv],
                te_tm_split=abs(centre["neff_TE"][0] - centre["neff_TM"][0])
                if centre.get("neff_TM") else None,
                n_eff=centre["neff_TE"][0], n_cut=centre["n_cut"], guided_TE=centre["guided_TE"],
                guided_TM=centre["guided_TM"], aeff_um2=centre.get("Aeff_TE0_um2"),
                D_min_tol=float(rep["D_min_tol"]), D_max_tol=float(rep["D_max_tol"]),
                tolerance_box_target=robust, mesh_change=change,
                map_file=str(grid_file), map_figure=str(plots / "design_map.png"),
                report=str(ctx.out / "results" / "finalist_report" / "report.md"))


# ======================================================== figures shared with the checks stage
def fig_mode_profiles(ctx: Context, H: float, W: float) -> Optional[Path]:
    """TE00 / TM00 field components in the cross-section (free, local mode solver)."""
    from tarang.design import localmodes, visuals
    wl = ctx.cfg.run.wavelength_um

    def draw():
        ms = localmodes.cross_section_modes(ctx, H, W, num_modes=4, group_index=True)
        te, tm = localmodes.te00(ms), localmodes.tm00(ms)
        rows = [("TE00", te)] + ([("TM00", tm)] if tm else [])
        return visuals.mode_profiles(rows, H, W, ctx.path("figures", "2b_mode_profiles.png"),
                                     f"Modes of the {H * 1e3:.0f} × {W * 1e3:.0f} nm core at "
                                     f"{wl} µm (cross-section, y across the width)")
    return visuals.safe(ctx, "mode profiles", draw)


def fig_bent_mode(ctx: Context, H: float, W: float, R: float) -> dict:
    """Straight and bent TE00 side by side; returns the outward shift of the bent mode."""
    from tarang.design import localmodes, visuals
    info: dict = {}

    def draw():
        st = localmodes.te00(localmodes.cross_section_modes(ctx, H, W, num_modes=2))
        bt = localmodes.te00(localmodes.cross_section_modes(ctx, H, W, bend_radius=R, num_modes=4))
        info.update(shift_nm=1e3 * (bt["yc"] - st["yc"]), n_eff_bent=bt["neff"])
        return visuals.mode_profiles([("TE00 straight", st), (f"TE00 bent, R = {R:g} µm", bt)],
                                     H, W, ctx.path("figures", "3b_bent_mode.png"),
                                     f"Bent mode: shifted {info['shift_nm']:+.1f} nm towards the "
                                     "outside of the ring", components=False)
    visuals.safe(ctx, "bent mode", draw)
    return info


def gap_extras(ctx: Context, gap: dict, devices: Optional[dict] = None) -> dict:
    """Coupled-mode check of the gap (local supermodes) and the top-view field map of the
    gap run nearest the critical gap (from its saved data). Returns the check's numbers."""
    from tarang.design import localmodes, visuals
    cfg, ui = ctx.cfg, ctx.ui
    xs, ring = ctx.results("cross_section"), ctx.results("ring")
    H, W, R, wl = xs["height_um"], xs["width_um"], ring["radius_um"], cfg.run.wavelength_um
    g_crit, target, gaps = gap["critical_gap_um"], gap["target_cross_power"], gap["gaps_um"]
    fit = gap.get("fit") or {}
    cmt: dict = {}

    def supermode_check():
        near = sorted(sorted(gaps, key=lambda x: abs(x - g_crit))[:3])
        res = localmodes.coupling_from_supermodes(ctx, H, W, R, near)
        cmt.update(theta0=res["theta0"], gamma_per_um=res["gamma_per_um"], per_gap=res["rows"],
                   critical_gap_um=math.log(res["theta0"] / math.asin(math.sqrt(target)))
                   / res["gamma_per_um"])
        sm = localmodes.coupler_supermodes(ctx, H, W, g_crit)
        th = res["theta0"] * math.exp(-res["gamma_per_um"] * g_crit)
        th_fdtd = fit["theta_0"] * math.exp(-fit["gamma"] * g_crit) if fit else None
        return visuals.supermodes(sm, H, W, ctx.path("figures", "4c_gap_supermodes.png"),
                                  th, th_fdtd)
    visuals.safe(ctx, "gap supermodes", supermode_check)
    if cmt:
        ui.say(f"  Check by coupled-mode theory (local supermodes, free): critical gap "
               f"{cmt['critical_gap_um'] * 1e3:.0f} nm, decay γ = {cmt['gamma_per_um']:.2f} /µm "
               f"(3D FDTD: {g_crit * 1e3:.0f} nm, γ = {fit.get('gamma', float('nan')):.2f} /µm).")
    g_show = min(gaps, key=lambda x: abs(x - g_crit))

    def field_map():
        from tarang.utilities.sweep_helpers import cache_dir as sweep_cache_dir
        d = (devices or {}).get(g_show) or _gap_device(ctx, g_show)
        cache = sweep_cache_dir(Path(gap["folder"]), d)
        if not d.is_cached(cache):                  # a figure never starts a cloud run
            ui.say(f"  (gap field map: no saved results for gap {g_show} µm)")
            return None
        data = d._run_sim(cache)
        return visuals.gap_field_map(data, ctx.path("figures", "4d_gap_field.png"), wl, g_show,
                                     f"|E|² in the middle of the core, gap {g_show} µm "
                                     "(log scale: light entering the ring is visible)")
    visuals.safe(ctx, "gap field map", field_map)
    return cmt


def _gap_device(ctx: Context, gap_um: float):
    """The partial-ring device of a gap with this run's settings (to reload its results)."""
    import warnings as _w
    from tarang.geometries.partial_ring_resonator import get_active_simulation as partial
    cfg = ctx.cfg
    xs, ring = ctx.results("cross_section"), ctx.results("ring")
    clad_is_air = str(cfg.materials.cladding).lower() == "air"
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        return partial(radius=ring["radius_um"], gap=gap_um, wavelength_um=cfg.run.wavelength_um,
                       span_angle_rad=math.radians(cfg.gap.span_angle_deg),
                       wg_width=xs["width_um"], wg_height=xs["height_um"],
                       core_medium=ctx.medium(cfg.materials.core),
                       box_medium=ctx.medium(cfg.materials.box), box_height=cfg.stack.box_height_um,
                       top_cladding_medium=None if clad_is_air else ctx.medium(cfg.materials.cladding),
                       top_cladding_height=0.0 if clad_is_air else cfg.stack.cladding_height_um,
                       fine_steps_per_wvl=cfg.gap.fine_steps, coarse_steps_per_wvl=cfg.gap.coarse_steps,
                       **gap_builder_options(cfg))


def _rank_targets(rows, heights, widths) -> list[dict]:
    """TARGET points (D >= 0, one guided TE) ranked by how many of their 8 neighbours are
    also TARGET (a neighbour outside the map counts as unknown, not as TARGET), then by
    the TE00/TM00 effective-index split (larger = further from a TE/TM crossing)."""
    def is_target(r):
        return r is not None and r["D_TE0"] >= 0 and r["guided_TE"] == 1
    by = {(round(r["H"], 4), round(r["W"], 4)): r for r in rows}
    hs, ws = sorted({round(h, 4) for h in heights}), sorted({round(w, 4) for w in widths})
    out = []
    for (h, w), r in by.items():
        if not is_target(r):
            continue
        i, j = hs.index(h), ws.index(w)
        nb = [by.get((hs[a], ws[b])) for a in (i - 1, i, i + 1) for b in (j - 1, j, j + 1)
              if (a, b) != (i, j) and 0 <= a < len(hs) and 0 <= b < len(ws)]
        split = abs(r["neff_TE"][0] - r["neff_TM"][0]) if r.get("neff_TM") else 0.0
        out.append(dict(H_um=h, W_um=w, D_TE0=r["D_TE0"], ng_TE0=r["ng_TE0"],
                        guided_TE=r["guided_TE"], guided_TM=r["guided_TM"],
                        TE_TM_split=split,
                        target_neighbours=f"{sum(map(is_target, nb))}/8",
                        _score=(sum(map(is_target, nb)), split)))
    out.sort(key=lambda r: r["_score"], reverse=True)
    for r in out:
        r.pop("_score")
    return out


def stage_ring(ctx: Context) -> dict:
    cfg, ui = ctx.cfg, ctx.ui
    xs = ctx.results("cross_section")
    H, W, ng, wl = xs["height_um"], xs["width_um"], xs["n_group"], cfg.run.wavelength_um
    from tarang.design import assumptions
    if "ring.loss_preset" not in cfg.given or ui.review:
        ui.say("Loss presets (values from publications; your own measured values can replace "
               "them below):")
        ui.table([dict(preset=k, scattering_dB_cm=v["scattering_db_cm"],
                       absorption_dB_cm=v["absorption_db_cm"], source=v["source"])
                  for k, v in assumptions.LOSS_PRESETS.items()])
    ui.say(f"n_g = {ng:.4f}: R = c / (2π n_g FSR) gives the radius for the FSR target.")
    pre_of = lambda c: assumptions.preset(c.ring.loss_preset)  # noqa: E731
    ui.form(cfg, "Ring", [
        dict(section="ring", key="fsr_ghz"),
        dict(section="ring", key="radius_um", default=lambda c: physics.round_to(
            physics.radius_for_fsr(c.ring.fsr_ghz, ng), 0.5)),
        dict(section="ring", key="loss_preset",
             check=lambda v: "" if str(v).lower() in assumptions.LOSS_PRESETS else
             "choose one of " + ", ".join(assumptions.LOSS_PRESETS)),
        dict(section="ring", key="scattering_loss_db_cm",
             default=lambda c: pre_of(c)["scattering_db_cm"]),
        dict(section="ring", key="absorption_loss_db_cm",
             default=lambda c: pre_of(c)["absorption_db_cm"]),
        dict(section="ring", key="bend_check")])
    fsr = ui.value(cfg, "ring", "fsr_ghz")
    R0 = physics.round_to(physics.radius_for_fsr(fsr, ng), 0.5)
    ui.say(f"n_g = {ng:.4f} -> R = {physics.radius_for_fsr(fsr, ng):.2f} um for {fsr:g} GHz "
           f"(rounded to {R0} um)")
    R = ui.value(cfg, "ring", "radius_um", default=R0)
    name = ui.value(cfg, "ring", "loss_preset",
                    check=lambda v: "" if str(v).lower() in assumptions.LOSS_PRESETS else
                    "choose one of " + ", ".join(assumptions.LOSS_PRESETS))
    pre = assumptions.preset(name)
    scat = ui.value(cfg, "ring", "scattering_loss_db_cm", default=pre["scattering_db_cm"])
    absl = ui.value(cfg, "ring", "absorption_loss_db_cm", default=pre["absorption_db_cm"])
    ui.say(f"  scattering {scat} dB/cm ({assumptions.loss_source(cfg, 'scattering')})")
    ui.say(f"  absorption {absl} dB/cm ({assumptions.loss_source(cfg, 'absorption')})")
    bend_loss = 0.0
    if ui.value(cfg, "ring", "bend_check"):
        bend_file = ctx.path("results", "bend.jsonl")
        ctx.tool("sweep", "--points", f"{H}x{W}", "--bend-radius", R, "--pml", 12,
                 "--wl", wl, "--steps", cfg.cross_section.map_steps, "--workers", 1,
                 "--out", bend_file.relative_to(ctx.out), *ctx.stack_args())
        rec = next((r for r in _read_jsonl(bend_file)
                    if abs(r["H"] - H) < 1e-6 and abs(r["W"] - W) < 1e-6
                    and r.get("bend_radius") == R), None)
        if rec:
            bend_loss = float(rec.get("loss_TE0_dB_cm", 0.0))
            ui.say(f"Bend loss of TE00 at R = {R} um (local bent-mode solve): {bend_loss:.3g} dB/cm")
    bent_info = fig_bent_mode(ctx, H, W, R)
    total = scat + absl + bend_loss
    qi = physics.q_intrinsic(total, ng, wl)
    fsr_r = physics.fsr_ghz(R, ng)

    lo, hi = ctx.band
    bb_file = ctx.path("results", "broadband.jsonl")
    ctx.tool("broadband", "--points", f"{H}x{W}", "--wl-min", lo, "--wl-max", hi,
             "--pump", wl, "--steps", cfg.cross_section.decision_steps, "--radius", R,
             "--data", bb_file.relative_to(ctx.out), "--overwrite", *ctx.stack_args())
    with open(ctx.out / "results" / "dispersion_profile_plots" / "profiles_summary.csv",
              encoding="utf-8") as f:
        prof = [r for r in csv.DictReader(f) if r["geometry"].startswith(f"{H*1e3:.0f}×{W*1e3:.0f}")]
    prof = prof[-1] if prof else {}
    fig = ctx.out / "results" / "dispersion_profile_plots" / f"profile_H{H*1e3:.0f}_W{W*1e3:.0f}.png"
    ui.say(f"\nRing R = {R} um: FSR = {fsr_r:.1f} GHz, total loss {total:.3g} dB/cm "
           f"(scattering {scat}, absorption {absl}, bend {bend_loss:.2g}) -> Q_int = {qi:.3g}; "
           f"D2/2π = {prof.get('D2_2pi_MHz', '?')} MHz, zero-dispersion wavelengths: "
           f"{prof.get('ZDWs_nm', '?')}")
    ui.figure(fig)
    return dict(radius_um=R, fsr_ghz=fsr_r, n_group=ng, loss_db_cm=total, bend_loss_db_cm=bend_loss,
                scattering_loss_db_cm=scat, absorption_loss_db_cm=absl, q_intrinsic=qi,
                D2_2pi_MHz=_num(prof.get("D2_2pi_MHz")), D3_2pi_kHz=_num(prof.get("D3_2pi_kHz")),
                zdw_nm=prof.get("ZDWs_nm"), profile_figure=str(fig),
                bent_mode_shift_nm=bent_info.get("shift_nm"))


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _estimate(ctx: Context, sims: dict) -> Optional[float]:
    from tarang.utilities.cloud_run import estimate_batch_cost
    try:
        return float(estimate_batch_cost(sims))
    except Exception as err:                                     # no key, no network, ...
        ctx.ui.say(f"  Could not get a cost estimate from Tidy3D ({type(err).__name__}: {err}).")
        ctx.ui.say("  Check `tidy3d configure` and the network, then rerun; nothing was run.")
        return None


def stage_gap(ctx: Context) -> dict:
    cfg, ui = ctx.cfg, ctx.ui
    xs, ring = ctx.results("cross_section"), ctx.results("ring")
    H, W, R = xs["height_um"], xs["width_um"], ring["radius_um"]
    wl = cfg.run.wavelength_um
    ui.form(cfg, "Ring-bus gap", [
        dict(section="gap", key="gaps_um", default=_gaps_default, cast=parse_list),
        dict(section="gap", key="span_angle_deg"),
        dict(section="gap", key="over_coupling"),
        dict(section="gap", key="reference", check=lambda v: "" if v in
             ("ring_port", "bus_only", "source") else "answer ring_port, bus_only or source"),
        dict(section="gap", key="fine_steps"),
        dict(section="gap", key="coarse_steps"),
        dict(section="gap", key="band_points",
             check=lambda v: "" if int(v) >= 1 and int(v) % 2 == 1 else "an odd number >= 1"),
        dict(section="gap", key="ring_inner_box_recess_um",
             check=lambda v: "" if 0 <= float(v) < 1 else "0 <= recess < 1 um"),
        *_budget_items()])
    gaps = ui.value(cfg, "gap", "gaps_um", default=_gaps_default(cfg), cast=parse_list)
    span = ui.value(cfg, "gap", "span_angle_deg")
    K = ui.value(cfg, "gap", "over_coupling")
    reference = ui.value(cfg, "gap", "reference", check=lambda v: "" if v in
                         ("ring_port", "bus_only", "source")
                         else "answer ring_port, bus_only or source")
    fine = ui.value(cfg, "gap", "fine_steps")
    coarse = ui.value(cfg, "gap", "coarse_steps")
    ui.value(cfg, "run", "max_cost_per_run")
    ui.value(cfg, "run", "max_total_cost")

    from tarang.geometries.partial_ring_resonator import get_active_simulation as partial
    from tarang.ring_resonator.ring_gap_sweep import (BUS_ONLY_FOLDER, bus_only_device,
                                                      ring_gap_sweep)
    from tarang.utilities.sweep_helpers import cache_dir as sweep_cache_dir
    from tarang.design import visuals
    clad_is_air = str(cfg.materials.cladding).lower() == "air"
    geo = dict(wavelength_um=wl, span_angle_rad=math.radians(span), wg_width=W, wg_height=H,
               core_medium=ctx.medium(cfg.materials.core), box_medium=ctx.medium(cfg.materials.box),
               box_height=cfg.stack.box_height_um,
               top_cladding_medium=None if clad_is_air else ctx.medium(cfg.materials.cladding),
               top_cladding_height=0.0 if clad_is_air else cfg.stack.cladding_height_um,
               fine_steps_per_wvl=fine, coarse_steps_per_wvl=coarse)
    extra = gap_builder_options(cfg)
    folder = ctx.out / "cloud" / "gap_sweep"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        devices = {g: partial(radius=R, gap=g, **geo, **extra) for g in gaps}
    for msg in sorted({str(w.message) for w in caught}):
        ui.say(f"  Note from the builder: {msg}")
    jobs = {f"gap_{g:g}": (d, sweep_cache_dir(folder, d)) for g, d in devices.items()}
    if reference == "bus_only":
        ref = bus_only_device(devices[max(gaps)])
        jobs["bus_only"] = (ref, folder / BUS_ONLY_FOLDER)
    todo = {k: d.sim for k, (d, cache) in jobs.items() if not d.is_cached(cache)}

    mid = devices[sorted(gaps)[len(gaps) // 2]]
    zc = H / 2
    bus = next((st.geometry.bounds for st in mid.sim.structures if st.name == "Bus_Waveguide"),
               ((0, -W / 2, 0), (0, W / 2, 0)))
    yb = 0.5 * (bus[0][1] + bus[1][1])                 # bus centre line (y)
    visuals.safe(ctx, "gap geometry", visuals.sim_views, mid.sim,
                 ctx.path("figures", "4a_gap_geometry.png"),
                 [dict(z=zc, title=f"top view at z = {zc:.2f} µm (gap {mid.gap} µm)"),
                  dict(z=zc, xlim=(-8, 8), ylim=(yb - 2 * W - 1, yb + 2 * W + mid.gap + 2),
                       title="coupling region"),
                  dict(x=0, ylim=(-2.0, H + 2.0), xlim=(yb - 3 * W, yb + 3 * W + mid.gap),
                       title="cross-section through the coupling point (x = 0)")],
                 f"Partial ring ({span:g}°, R = {R} µm) next to the bus: one of {len(gaps)} gaps",
                 notes=("The ring arcs end inside the simulation box (no absorbers at their ends). "
                        + {"ring_port": "Coupled power: read where light enters the ring arc.",
                           "bus_only": "A bus-only copy (ring removed) is run once as the reference.",
                           "source": "Coupled power: 1 - through power relative to the source."}
                        [reference]))
    ui.say(f"{len(jobs)} simulations ({len(gaps)} gaps" + (" + the bus alone" if reference ==
           "bus_only" else "") + f"), {len(todo)} not computed yet.")
    cost = _estimate(ctx, todo) if todo else 0.0
    if cost is None or (todo and not ctx.account_ok(cost)):
        return dict(skipped=True)
    if not ui.proceed(f"ring-bus gap sweep, {len(todo)} cloud FDTD runs (R = {R} um, "
                      f"{span:g}° partial ring)", cost, ctx.budget_left):
        return dict(skipped=True)

    volume = 2 * math.pi * R * W * H
    with ctx.cloud_limit(), ctx.cloud_progress("gap sweep", len(jobs)), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)        # already shown above
        crit = ring_gap_sweep(gaps, folder, R, ring["q_intrinsic"], volume,
                              n_group=ring["n_group"], reference=reference,
                              **{k: v for k, v in geo.items() if k != "span_angle_rad"},
                              span_angle_rad=math.radians(span), builder_kwargs=extra)
    ctx.add_spent(cost, f"gap sweep, {len(todo)} runs")
    g_crit = crit["critical_gap_um"]
    g_over = physics.gap_for_over_coupling(crit, K) if K and K != 1 else None

    # per-gap table and figure (crit lists follow the order of `gaps`)
    pos = {float(g): i for i, g in enumerate(gaps)}
    g_sorted = np.asarray(crit["sorted_gaps_um"], float)
    idx = [pos[float(g)] for g in g_sorted]
    coupled = 1 - np.asarray(crit["through_power"], float)[idx]
    raw = 1 - np.asarray(crit["through_power_source"], float)[idx]
    ref_power = crit.get("reference_through_power")
    per_gap = [dict(gap_um=float(g), coupled_power=float(c), one_minus_T_source=float(r),
                    theta_rad=float(th), used_in_fit=bool(m))
               for g, c, r, th, m in zip(g_sorted, coupled, raw,
                                         np.asarray(crit["theta_raw_rad"]),
                                         np.asarray(crit["fit_mask"]))]
    ui.table(per_gap, formats={"coupled_power": ".3e", "one_minus_T_source": ".3e",
                               "theta_rad": ".4f"})
    if reference != "source":
        floor = float(np.median(raw[-2:] - coupled[-2:])) if len(raw) >= 2 else 0.0
        ui.say(f"  coupled_power is the light entering the ring ({reference}); 1 - T relative "
               f"to the source carries a gap-independent excess of about {floor:.1e} "
               "(numerical loss on the bus), which is not coupling.")
    target = crit["target_cross_power"]
    targets = {"critical": target}
    if g_over is not None:
        targets[f"over-coupled K = {K:g}"] = min(K * target, 0.999)
    fig = visuals.safe(ctx, "gap coupling", visuals.gap_coupling, g_sorted,
                       raw if reference != "source" else None, coupled, crit.get("fit_params"),
                       targets, ctx.path("figures", "4b_gap_coupling.png"),
                       {"critical": g_crit, f"over-coupled K = {K:g}": g_over},
                       crit.get("fit_r_squared_log_theta"))
    cmt = gap_extras(ctx, dict(critical_gap_um=g_crit, target_cross_power=target, gaps_um=list(gaps),
                               fit=crit.get("fit_params"), folder=str(folder)), devices)
    rows = [dict(regime="critical (K = 1)", gap_um=g_crit, Q_loaded=physics.loaded_q(ring["q_intrinsic"], 1))]
    if g_over is not None:
        rows.append(dict(regime=f"over-coupled (K = {K:g})", gap_um=g_over,
                         Q_loaded=physics.loaded_q(ring["q_intrinsic"], K)))
    ui.table(rows, formats={"gap_um": ".3f", "Q_loaded": ".3g"})
    if crit.get("critical_gap_extrapolated"):
        ui.say("  Note: the critical gap lies outside the fitted gaps (extrapolated); add "
               "gaps around it and rerun this stage.")
    table = ctx.path("tables", "gap_sweep.csv")
    with open(table, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_gap[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(per_gap)
    return dict(critical_gap_um=g_crit, over_coupling=K, over_coupled_gap_um=g_over,
                extrapolated=bool(crit.get("critical_gap_extrapolated")),
                q_loaded_critical=physics.loaded_q(ring["q_intrinsic"], 1),
                q_loaded_over=physics.loaded_q(ring["q_intrinsic"], K) if g_over else None,
                fit=crit.get("fit_params"), fit_r2=crit.get("fit_r_squared_log_theta"),
                fit_gap_range_um=crit.get("fit_gap_range_um"), reference=reference,
                reference_through_power=ref_power, target_cross_power=target,
                span_angle_deg=span, gaps_um=list(gaps), per_gap=per_gap, table=str(table),
                figure=str(fig) if fig else None, estimated_cost=cost, folder=str(folder),
                mesh_steps=[fine, coarse], band_points=int(cfg.gap.band_points),
                ring_inner_box_recess_um=float(cfg.gap.ring_inner_box_recess_um),
                coupled_mode_check=cmt or None)


def gap_builder_options(cfg) -> dict:
    """Partial-ring builder options from the gap settings: the BOX pocket inside the ring
    and the frequencies the ring-port monitors record (odd count, centred exactly on the
    design frequency, +/-8 %). The values of runs made before Phase 12 (0.1 um, one
    frequency) give the builder's original simulation, bit for bit."""
    g = cfg.gap
    opts = dict(inner_box_recess_um=float(g.ring_inner_box_recess_um))
    n = int(g.band_points)
    if n > 1:
        f0 = C_UM / cfg.run.wavelength_um
        opts["monitor_freqs"] = [f0 * (1 + 0.08 * x) for x in np.linspace(-1, 1, n)]
    return opts


def stage_grating(ctx: Context) -> dict:
    """Grating coupler in three small steps, all 2D FDTD with a broadband spectrum:

    A. a coarse map: for each fill factor, periods around its phase-matched period;
    B. retune: shift the period so the peak of the best design lands on the design
       wavelength, and vary the fill factor and the period around it;
    C. fibre position along the grating for the best design.
    Every run records the efficiency spectrum (peak, value at the design wavelength,
    1 dB / 3 dB bandwidths), so a design is judged where it will be used.
    """
    cfg, ui = ctx.cfg, ctx.ui
    xs = ctx.results("cross_section")
    H, wl = xs["height_um"], cfg.run.wavelength_um
    g = cfg.grating
    is_partial = lambda c: c.grating.etch == "partial"  # noqa: E731
    ui.form(cfg, "Grating coupler", [
        dict(section="grating", key="etch",
             check=lambda v: "" if v in ("full", "partial") else "answer full or partial"),
        dict(section="grating", key="etch_depth_um", show=is_partial, default=round(H / 2, 3),
             check=lambda v: "" if 0 < v < H else f"0 < depth < {H}"),
        dict(section="grating", key="fiber_angle_deg"),
        dict(section="grating", key="fiber_tilt",
             check=lambda v: "" if v in ("towards", "away") else "answer towards or away"),
        dict(section="grating", key="mfd_um"),
        dict(section="grating", key="n_teeth"),
        dict(section="grating", key="width_um"),
        dict(section="grating", key="min_feature_um"),
        dict(section="grating", key="bandwidth_um"),
        dict(section="grating", key="fill_factors", cast=parse_list,
             default=lambda c: [0.5, 0.6, 0.7, 0.8, 0.9] if c.grating.etch == "full"
             else [0.4, 0.5, 0.6, 0.7, 0.8]),
        dict(section="grating", key="detuning", cast=parse_list),
        dict(section="grating", key="retune"),
        dict(section="grating", key="centre_peak", show=lambda c: bool(c.grating.retune)),
        dict(section="grating", key="source_x_um", cast=parse_list,
             default=lambda c: [round(c.grating.mfd_um * f, 2) for f in (0.25, 0.375, 0.5, 0.75, 1.0)]),
        dict(section="grating", key="check_3d"),
        *_budget_items()])
    etch = ui.value(cfg, "grating", "etch", check=lambda v: "" if v in ("full", "partial")
                    else "answer full or partial")
    depth = H if etch == "full" else ui.value(cfg, "grating", "etch_depth_um", default=round(H / 2, 3),
                                              check=lambda v: "" if 0 < v < H else f"0 < depth < {H}")
    angle = ui.value(cfg, "grating", "fiber_angle_deg")
    tilt = ui.value(cfg, "grating", "fiber_tilt", check=lambda v: "" if v in ("towards", "away")
                    else "answer towards or away")
    mfd = ui.value(cfg, "grating", "mfd_um")
    teeth = ui.value(cfg, "grating", "n_teeth")
    width = ui.value(cfg, "grating", "width_um")
    min_feat = ui.value(cfg, "grating", "min_feature_um")
    band = ui.value(cfg, "grating", "bandwidth_um")
    ffdef = [0.5, 0.6, 0.7, 0.8, 0.9] if etch == "full" else [0.4, 0.5, 0.6, 0.7, 0.8]
    ffs = ui.value(cfg, "grating", "fill_factors", default=ffdef, cast=parse_list)
    detune = ui.value(cfg, "grating", "detuning", cast=parse_list)
    fixed_periods = g.periods_um if "grating.periods_um" in cfg.given else []
    retune = ui.value(cfg, "grating", "retune")
    centre = bool(retune and ui.value(cfg, "grating", "centre_peak"))

    from tarang.grating_coupler.grating_coupler import phase_matched_period, spectrum_summary
    from tarang.utilities.cloud_run import run_batch_cached
    from tarang.design import visuals
    m = cfg.materials
    n_core, n_box = ctx.n(m.core), ctx.n(m.box)
    n_top = 1.0 if str(m.cladding).lower() == "air" else ctx.n(m.cladding)
    n_slab = physics.slab_te_index(n_core, n_box, n_top, H, wl)
    n_groove = n_top if etch == "full" else physics.slab_te_index(n_core, n_box, n_top, H - depth, wl)

    def p_match(ff: float) -> float:
        return phase_matched_period(ff * n_slab + (1 - ff) * n_groove, wl, angle, tilt, n_top)
    sign = "-" if tilt == "towards" else "+"
    ui.say(f"Slab TE index {n_slab:.4f}, groove {n_groove:.4f}. Fibre {angle:g}° from vertical, "
           f"beam travelling {tilt} the waveguide: first-order period "
           f"λ / (n_avg {sign} n_top sin θ) = " +
           ", ".join(f"{p_match(f):.3f} µm (FF {f:g})" for f in ffs))
    sx_mid = round(0.5 * mfd, 2)
    sxdef = [round(mfd * f, 2) for f in (0.25, 0.375, 0.5, 0.75, 1.0)]
    sx_list = ui.value(cfg, "grating", "source_x_um", default=sxdef, cast=parse_list)
    ui.value(cfg, "run", "max_cost_per_run")
    ui.value(cfg, "run", "max_total_cost")

    folder = ctx.out / "cloud" / "grating"

    def device(period, ff, sx, is_3d=False):
        return grating_device(ctx, period, ff, sx, H=H, depth=depth, is_3d=is_3d)

    def feasible(period, ff) -> bool:
        return min(ff, 1 - ff) * period >= min_feat - 1e-9

    results: list[dict] = []

    def run_set(points, label, step, is_3d=False) -> Optional[list[dict]]:
        points = [pt for pt in dict.fromkeys(points) if feasible(pt[0], pt[1])]
        if not points:
            return []
        devs = {pt: device(*pt, is_3d=is_3d) for pt in points}
        todo = {f"gc_{i}": d.sim for i, (pt, d) in enumerate(devs.items())
                if not d.is_cached(folder)}
        cost = _estimate(ctx, todo) if todo else 0.0
        if cost is None or (todo and not ctx.account_ok(cost)):
            return None
        if not ui.proceed(f"{label}: {len(points)} {'3D' if is_3d else '2D'} grating "
                          f"simulations ({len(todo)} not computed yet)", cost, ctx.budget_left):
            return None
        if todo:
            jobs = {pt: (d.sim, d._readable_name()) for pt, d in devs.items()
                    if not d.is_cached(folder)}
            ui.say(f"  submitting {len(jobs)} runs together (they run in parallel) ...")
            cap = float(cfg.run.max_cost_per_run) * len(jobs)
            with ctx.cloud_progress(label, 1):
                run_batch_cached({f"{pt}": job for pt, job in jobs.items()}, folder, max_cost=cap)
        out = []
        for pt, d in devs.items():
            wls, eta = d.get_spectrum(folder)
            summ = spectrum_summary(wls, eta, wl)
            out.append(dict(step=step, period_um=pt[0], fill_factor=pt[1], source_x_um=pt[2],
                            dim="3D" if is_3d else "2D", **summ))
        ctx.add_spent(cost, f"{label} ({len(todo)} runs)")
        results.extend(out)
        ui.table(sorted(out, key=lambda r: -r["efficiency_at_target"])[:8],
                 ["period_um", "fill_factor", "source_x_um", "peak_wavelength_um",
                  "peak_efficiency", "efficiency_at_target", "bandwidth_1db_nm"],
                 {"peak_wavelength_um": ".4f", "peak_efficiency": ".3f",
                  "efficiency_at_target": ".3f", "bandwidth_1db_nm": ".0f"})
        return out

    # ---- preview of the middle design, before any cost question
    f_mid = ffs[len(ffs) // 2]
    preview = device(p_match(f_mid), f_mid, sx_mid)
    visuals.safe(ctx, "grating geometry", visuals.grating_geometry, preview,
                 ctx.path("figures", "5a_grating_geometry.png"),
                 "Grating coupler, 2D FDTD (one of the designs in step A)")

    # ---- A: coarse map
    if fixed_periods:
        pts = [(round(p, 4), f, sx_mid) for f in ffs for p in fixed_periods]
    else:
        pts = [(round(p_match(f) * d, 4), f, sx_mid) for f in ffs for d in detune]
    stage_a = run_set(pts, "A. period x fill factor map", "A")
    if stage_a is None:
        return dict(skipped=True)
    best = _best_grating(stage_a)
    visuals.safe(ctx, "grating map", visuals.grating_landscape, stage_a, best,
                 ctx.path("figures", "5b_grating_map.png"), wl)
    edge = _edge_notes(best, ffs, sorted({r["period_um"] for r in stage_a
                                          if r["fill_factor"] == best["fill_factor"]}))

    ff_rows, period_rows, align_rows = [], [], []
    final = best
    if retune:
        # ---- B: retune the period onto the design wavelength, vary FF and period
        scale = wl / best["peak_wavelength_um"]
        P_star = best["period_um"] * scale
        if not best["peak_inside"]:
            ui.say(f"  The best design's peak lies outside the simulated band "
                   f"({wl - band:.3f}-{wl + band:.3f} µm); moving the period towards it.")
        f0 = best["fill_factor"]
        ffs_b = [round(f0 + df, 3) for df in (-0.1, -0.05, 0.0, 0.05, 0.1) if 0.05 < f0 + df < 0.95]
        sx0 = best["source_x_um"]
        pts_ff = [(round(P_star * p_match(f) / p_match(f0), 4), f, sx0) for f in ffs_b]
        pts_p = [(round(P_star * k, 4), f0, sx0) for k in (0.98, 0.99, 1.0, 1.01, 1.02)]
        stage_b = run_set(pts_ff + pts_p, "B. period tuned onto the wavelength; fill factor "
                                          "and period around it", "B")
        if stage_b:
            final = max(stage_b, key=lambda r: r["efficiency_at_target"])
            key = lambda r: (round(r["period_um"], 4), r["fill_factor"], r["source_x_um"])  # noqa: E731
            ff_rows = [r for r in stage_b if key(r) in set(pts_ff)]
            period_rows = [r for r in stage_b if key(r) in set(pts_p)]
            periods_tried = [p for p, _, _ in pts_p]
            if abs(final["fill_factor"] - f0) > 1e-9:
                # ---- B2: the best fill factor differs from the centre one; its period was
                # set only by phase matching, so vary the period around it as well
                f1, p1 = final["fill_factor"], final["period_um"]
                pts_p2 = [(round(p1 * k, 4), f1, sx0) for k in (0.98, 0.99, 1.01, 1.02)]
                stage_b2 = run_set(pts_p2, f"B2. period around the best fill factor ({f1:g})",
                                   "B2")
                if stage_b2:
                    period_rows = stage_b2 + [final]
                    periods_tried = [p for p, _, _ in pts_p2] + [p1]
                    final = max(period_rows, key=lambda r: r["efficiency_at_target"])
            if centre:
                # ---- D: period that puts the peak at the design wavelength
                final = _centre_peak(final, results, run_set, wl, ui)
                if {r["fill_factor"] for r in period_rows} == {final["fill_factor"]}:
                    period_rows = [r for r in results if r["dim"] == "2D"
                                   and r["fill_factor"] == final["fill_factor"]
                                   and r["source_x_um"] == final["source_x_um"]]
                    period_rows = list({r["period_um"]: r for r in period_rows}.values())
            # ---- C: fibre position for the best design
            stage_c = run_set([(final["period_um"], final["fill_factor"], x) for x in sx_list],
                              "C. fibre position along the grating", "C")
            if stage_c:
                align_rows = stage_c + ([final] if final["source_x_um"] not in sx_list else [])
                final = max(align_rows, key=lambda r: r["efficiency_at_target"])
            same_ff = {r["fill_factor"] for r in period_rows} == {final["fill_factor"]}
            edge += _edge_notes(final, ffs_b, periods_tried if same_ff else None,
                                sx_list if stage_c else None)
            if not same_ff:
                edge.append(f"the period was varied at fill factor {f0:g} only; the chosen "
                            f"fill factor {final['fill_factor']:g} uses its phase-matched period.")
    best3d = None
    if ui.value(cfg, "grating", "check_3d"):
        r3 = run_set([(final["period_um"], final["fill_factor"], final["source_x_um"])],
                     "3D check of the final design", "3D", is_3d=True)
        if r3:
            best3d = r3[0]

    # ---- figures and table of the final design
    fdev = device(final["period_um"], final["fill_factor"], final["source_x_um"])
    wls, eta = fdev.get_spectrum(folder)
    summ = spectrum_summary(wls, eta, wl)
    summ["target_um"] = wl
    visuals.safe(ctx, "grating spectrum", visuals.grating_spectrum, wls, eta, summ,
                 ctx.path("figures", "5d_grating_spectrum.png"),
                 f"Final grating: P = {final['period_um']:.4f} µm, FF = {final['fill_factor']:.2f}, "
                 f"fibre at {final['source_x_um']} µm")
    if ff_rows or period_rows or align_rows:
        visuals.safe(ctx, "grating sensitivity", visuals.grating_sensitivity, ff_rows,
                     period_rows, align_rows, final, ctx.path("figures", "5c_grating_sensitivity.png"),
                     wl)
    sim_data = fdev._run_sim(folder)
    visuals.safe(ctx, "grating field", visuals.grating_field, sim_data,
                 ctx.path("figures", "5e_grating_field.png"), wl,
                 f"|E| of the final grating at {wl} µm (fibre {tilt} the waveguide)")
    visuals.safe(ctx, "final grating geometry", visuals.grating_geometry, fdev,
                 ctx.path("figures", "5f_grating_final_geometry.png"), "Final grating coupler")
    spec = ctx.path("tables", "grating_spectrum_final.csv")
    with open(spec, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["wavelength_um", "efficiency", "efficiency_db"])
        for x, e in zip(wls, eta):
            w.writerow([round(float(x), 6), float(e), round(10 * math.log10(max(float(e), 1e-12)), 4)])
    table = ctx.path("tables", "grating_sweep.csv")
    with open(table, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(results)
    ui.say(f"\nFinal grating: period {final['period_um']:.4f} µm, fill factor "
           f"{final['fill_factor']:.2f}, fibre at {final['source_x_um']} µm on the grating: "
           f"{summ['efficiency_at_target'] * 100:.1f} % ({-summ['loss_db_at_target']:.2f} dB) "
           f"at {wl} µm; peak {summ['peak_efficiency'] * 100:.1f} % at "
           f"{summ['peak_wavelength_um'] * 1e3:.1f} nm; 1 dB bandwidth "
           f"{summ['bandwidth_1db_nm']:.0f} nm ({summ['band_1db_lo_um'] * 1e3:.0f}-"
           f"{summ['band_1db_hi_um'] * 1e3:.0f} nm).")
    for note in dict.fromkeys(edge):
        ui.say("  Note: " + note)
    return dict(etch=etch, etch_depth_um=depth, fiber_angle_deg=angle, fiber_tilt=tilt,
                mfd_um=mfd, n_teeth=teeth, width_um=width, period_um=final["period_um"],
                fill_factor=final["fill_factor"], source_x_um=final["source_x_um"],
                efficiency_2d=summ["efficiency_at_target"], loss_db_2d=summ["loss_db_at_target"],
                peak_efficiency_2d=summ["peak_efficiency"],
                peak_wavelength_um=summ["peak_wavelength_um"],
                bandwidth_1db_nm=summ["bandwidth_1db_nm"], bandwidth_3db_nm=summ["bandwidth_3db_nm"],
                band_1db_um=[summ["band_1db_lo_um"], summ["band_1db_hi_um"]],
                centre_peak=centre,
                efficiency_3d=best3d and best3d["efficiency_at_target"],
                period_estimate_um=p_match(final["fill_factor"]), n_slab=n_slab, n_groove=n_groove,
                bandwidth_um=band, min_feature_um=min_feat, runs=len(results), table=str(table),
                spectrum_table=str(spec), notes=list(dict.fromkeys(edge)),
                final_sim=str(folder / fdev.get_dir_name() / "simulation.json"))


def grating_device(ctx: Context, period: float, ff: float, sx: float, *, H: float,
                   depth: float, angle: Optional[float] = None, is_3d: bool = False,
                   mesh: float = 1.0):
    """The grating-coupler device of this design (2D unless ``is_3d``). ``angle``
    overrides the fibre angle and ``mesh`` scales the mesh steps (tolerance and mesh
    checks); the defaults give exactly the grating stage's simulations."""
    from tarang.grating_coupler.grating_coupler import GratingConfig, GratingCoupler
    cfg = ctx.cfg
    m, g = cfg.materials, cfg.grating
    handle = None if str(m.handle).lower() == "none" else ctx.medium(m.handle)
    clad = None if str(m.cladding).lower() == "air" else ctx.medium(m.cladding)
    gcfg = GratingConfig(wavelength_um=cfg.run.wavelength_um,
                         theta_deg=abs(g.fiber_angle_deg if angle is None else angle),
                         mfd=g.mfd_um, source_x=sx, source_gap=2.0, core_height=H,
                         top_cladding_height=cfg.stack.cladding_height_um,
                         box_height=cfg.stack.box_height_um, handle_height=3.0,
                         etch_depth=depth, period=round(period, 4), ff=round(ff, 4),
                         n_teeth=g.n_teeth, w_grating=g.width_um, pol_angle=math.pi / 2,
                         buffer=3.0, fiber_tilt=g.fiber_tilt, bandwidth_um=g.bandwidth_um)
    kw = {} if mesh == 1.0 else dict(fine_steps_per_wvl=int(round(15 * mesh)),
                                     coarse_steps_per_wvl=int(round(10 * mesh)))
    return GratingCoupler.from_config(gcfg, core_medium=ctx.medium(m.core),
                                      box_medium=ctx.medium(m.box), handle_medium=handle,
                                      top_cladding_medium=clad, is_3d=is_3d, **kw)


def _centre_peak(design: dict, results: list[dict], run_set, wl: float, ui,
                 tol_um: float = 0.001, max_runs: int = 3) -> dict:
    """Step D: move the period, at the design's fill factor and fibre position, until the
    efficiency peak sits within ``tol_um`` of ``wl``.

    The peak wavelength follows the period almost linearly (first-order Bragg condition
    λ_peak ≈ Λ (n_avg − sin θ)), so the next period is interpolated between the two runs
    whose peaks lie closest to ``wl`` on either side (secant step). Returns the run whose
    peak is closest to ``wl``.
    """
    def same(r):                                # peaks at the band edge are not reliable
        return (r["dim"] == "2D" and r.get("peak_inside", True)
                and r["fill_factor"] == design["fill_factor"]
                and r["source_x_um"] == design["source_x_um"])

    def closest():
        return min((r for r in results if same(r)),
                   key=lambda r: abs(r["peak_wavelength_um"] - wl))

    for _ in range(max_runs):
        best = closest()
        if abs(best["peak_wavelength_um"] - wl) <= tol_um:
            break
        rows = sorted({r["period_um"]: r for r in results if same(r)}.values(),
                      key=lambda r: r["period_um"])
        below = [r for r in rows if r["peak_wavelength_um"] < wl]
        above = [r for r in rows if r["peak_wavelength_um"] > wl]
        if below and above:
            a = max(below, key=lambda r: r["peak_wavelength_um"])
            b = min(above, key=lambda r: r["peak_wavelength_um"])
        elif len(rows) >= 2:
            a, b = sorted(rows, key=lambda r: abs(r["peak_wavelength_um"] - wl))[:2]
        else:                                    # one run only: λ_peak proportional to Λ
            a = b = rows[0]
        if a is b or a["peak_wavelength_um"] == b["peak_wavelength_um"]:
            period = a["period_um"] * wl / a["peak_wavelength_um"]
        else:
            slope = (b["period_um"] - a["period_um"]) / (b["peak_wavelength_um"]
                                                         - a["peak_wavelength_um"])
            period = a["period_um"] + slope * (wl - a["peak_wavelength_um"])
        period = round(period, 4)
        if any(abs(r["period_um"] - period) < 5e-5 for r in rows):
            break                                # the 0.1 nm period grid is reached
        got = run_set([(period, design["fill_factor"], design["source_x_um"])],
                      f"D. period {period:.4f} µm to put the peak at {wl * 1e3:.0f} nm", "D")
        if not got:
            break
    best = closest()
    ui.say(f"  Peak at {best['peak_wavelength_um'] * 1e3:.1f} nm with period "
           f"{best['period_um']:.4f} µm (fill factor {best['fill_factor']:g}).")
    return best


def _best_grating(rows: list[dict]) -> dict:
    """Highest peak efficiency, preferring designs whose peak lies inside the band."""
    inside = [r for r in rows if r["peak_inside"]] or rows
    return max(inside, key=lambda r: r["peak_efficiency"])


def _edge_notes(best: dict, ffs, periods=None, positions=None) -> list[str]:
    """Warnings when the best design sits at the edge of a swept range."""
    notes = []
    if len(ffs) > 1 and best["fill_factor"] in (min(ffs), max(ffs)):
        notes.append(f"the best fill factor ({best['fill_factor']}) is at the edge of the range "
                     f"tried ({min(ffs)}-{max(ffs)}); the optimum may lie beyond it.")
    if periods and len(periods) > 1 and best["period_um"] in (min(periods), max(periods)):
        notes.append(f"the best period ({best['period_um']} µm) is at the edge of the periods "
                     "tried for its fill factor.")
    if positions and len(positions) > 1 and best["source_x_um"] in (min(positions), max(positions)):
        notes.append(f"the best fibre position ({best['source_x_um']} µm) is at the edge of the "
                     "positions tried.")
    return notes


def stage_layout(ctx: Context) -> dict:
    from tarang.design import layout
    return layout.export(ctx)


def stage_report(ctx: Context) -> dict:
    from tarang.design import report
    return report.write(ctx)


STAGES: list[tuple[str, str, str, Callable[[Context], dict]]] = [
    ("materials", "Materials", "local", stage_materials),
    ("cross_section", "Waveguide cross-section (H x W)", "local", stage_cross_section),
    ("ring", "Ring radius, losses, Q, integrated dispersion", "local", stage_ring),
    ("gap", "Ring-bus gap (critical and over-coupled)", "cloud", stage_gap),
    ("grating", "Grating coupler", "cloud", stage_grating),
    ("checks", "Checks: tolerances, mode families, losses, band, grating budget", "cloud",
     lambda ctx: __import__("tarang.design.checks", fromlist=["stage_checks"]).stage_checks(ctx)),
    ("layout", "Layout (GDS) and parameter export", "local", stage_layout),
    ("report", "Report and tables", "local", stage_report),
]
STAGE_NAMES = [s[0] for s in STAGES]

# Settings each stage depends on: a change reruns that stage and the ones after it.
DEPENDS = {
    "materials": ("run.wavelength_um", "run.band_um", "materials"),
    "cross_section": ("stack", "cross_section"),
    "ring": ("ring",),
    "gap": ("gap",),
    "grating": ("grating",),
    "checks": ("checks",),
    "layout": ("layout",),
    "report": (),
}


# Settings that only label where a value came from (the values themselves are hashed):
# changing them must not repeat a stage.
KEY_IGNORE = {"ring": {"loss_preset"}}


def settings_key(cfg: DesignConfig, stage: str) -> str:
    """Hash of every setting this stage and the stages before it depend on."""
    parts = {}
    for name in STAGE_NAMES[:STAGE_NAMES.index(stage) + 1]:
        for dep in DEPENDS[name]:
            sec, _, key = dep.partition(".")
            val = getattr(cfg, sec)
            parts[dep] = getattr(val, key) if key else {
                k: v for k, v in val.__dict__.items() if k not in KEY_IGNORE.get(sec, ())}
    from tarang.design.config import LEGACY
    for sec, values in LEGACY.items():               # settings added later: a value that
        d = parts.get(sec)                            # reproduces older runs is left out
        if isinstance(d, dict):
            for key, old in values.items():
                if key in d and d[key] == old:
                    d.pop(key)
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
