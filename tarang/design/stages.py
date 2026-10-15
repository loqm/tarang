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
C_UM = 2.99792458e14


# =============================================================================== context
class Context:
    """Everything a stage needs: settings, the prompter, the run folder and saved state."""

    def __init__(self, cfg: DesignConfig, ui: Prompter, out: Path):
        self.cfg, self.ui, self.out = cfg, ui, Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.state_path = self.out / "state.json"
        self.state: dict = (json.loads(self.state_path.read_text(encoding="utf-8"))
                            if self.state_path.exists() else {"stages": {}, "spent": 0.0})
        self._media: dict = {}

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

    def add_spent(self, cost: float) -> None:
        self.state["spent"] = self.spent + float(cost)

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
    wl = ui.value(cfg, "run", "wavelength_um", check=lambda v: "" if 0.2 < float(v) < 20
                  else "wavelength must be in um, e.g. 1.55")
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=130)
    for role, (key, n, _) in curves.items():
        ax.plot(wls, n, lw=2, label=f"{role}: {Path(str(key)).stem}")
    ax.axvline(ctx.cfg.run.wavelength_um, color="0.5", ls=":")
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("Refractive index n (fitted)")
    ax.grid(alpha=0.3)
    ax.legend()
    path = ctx.path("figures", "1_materials.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def stage_cross_section(ctx: Context) -> dict:
    cfg, ui = ctx.cfg, ctx.ui
    wl = cfg.run.wavelength_um
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
                          "TE1_margin", "target_neighbours"],
             {"D_TE0": ".1f", "ng_TE0": ".4f", "TE1_margin": "+.4f"})
    best = ranked[0]
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
             "--tol", f"{dW},{dH}", "--alpha-scat", cfg.ring.scattering_loss_db_cm,
             "--alpha-abs", cfg.ring.absorption_loss_db_cm, "--out", "results/finalist_report",
             "--overwrite")
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
    return dict(height_um=H, width_um=W, steps=dsteps, D=centre["D_TE0"], n_group=centre["ng_TE0"],
                n_eff=centre["neff_TE"][0], n_cut=centre["n_cut"], guided_TE=centre["guided_TE"],
                guided_TM=centre["guided_TM"], aeff_um2=centre.get("Aeff_TE0_um2"),
                D_min_tol=float(rep["D_min_tol"]), D_max_tol=float(rep["D_max_tol"]),
                tolerance_box_target=robust, mesh_change=change,
                map_file=str(grid_file), map_figure=str(plots / "design_map.png"),
                report=str(ctx.out / "results" / "finalist_report" / "report.md"))


def _rank_targets(rows, heights, widths) -> list[dict]:
    """TARGET points (D >= 0, one guided TE) ranked by how many of their 8 neighbours are
    also TARGET (a neighbour outside the map counts as unknown, not as TARGET), then by
    the margin of the second TE mode below cut-off."""
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
        te1 = r["neff_TE"][1] if len(r["neff_TE"]) > 1 else r["n_cut"] - 1
        out.append(dict(H_um=h, W_um=w, D_TE0=r["D_TE0"], ng_TE0=r["ng_TE0"],
                        guided_TE=r["guided_TE"], guided_TM=r["guided_TM"],
                        TE1_margin=r["n_cut"] - te1,
                        target_neighbours=f"{sum(map(is_target, nb))}/8",
                        _score=(sum(map(is_target, nb)), r["n_cut"] - te1)))
    out.sort(key=lambda r: r["_score"], reverse=True)
    for r in out:
        r.pop("_score")
    return out


def stage_ring(ctx: Context) -> dict:
    cfg, ui = ctx.cfg, ctx.ui
    xs = ctx.results("cross_section")
    H, W, ng, wl = xs["height_um"], xs["width_um"], xs["n_group"], cfg.run.wavelength_um
    fsr = ui.value(cfg, "ring", "fsr_ghz")
    R0 = physics.round_to(physics.radius_for_fsr(fsr, ng), 0.5)
    ui.say(f"n_g = {ng:.4f} -> R = {physics.radius_for_fsr(fsr, ng):.2f} um for {fsr:g} GHz "
           f"(rounded to {R0} um)")
    R = ui.value(cfg, "ring", "radius_um", default=R0)
    scat = ui.value(cfg, "ring", "scattering_loss_db_cm")
    absl = ui.value(cfg, "ring", "absorption_loss_db_cm")
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
                zdw_nm=prof.get("ZDWs_nm"), profile_figure=str(fig))


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
    st = _step(0.065 * wl)
    gdef = _grid(physics.round_to(0.19 * wl, st), physics.round_to(0.52 * wl, st), st)
    gaps = ui.value(cfg, "gap", "gaps_um", default=gdef, cast=parse_list)
    span = ui.value(cfg, "gap", "span_angle_deg")
    K = ui.value(cfg, "gap", "over_coupling")
    fine = ui.value(cfg, "gap", "fine_steps")
    coarse = ui.value(cfg, "gap", "coarse_steps")
    ui.value(cfg, "run", "max_cost_per_run")
    ui.value(cfg, "run", "max_total_cost")

    from tarang.geometries.partial_ring_resonator import get_active_simulation as partial
    clad_is_air = str(cfg.materials.cladding).lower() == "air"
    geo = dict(wavelength_um=wl, span_angle_rad=math.radians(span), wg_width=W, wg_height=H,
               core_medium=ctx.medium(cfg.materials.core), box_medium=ctx.medium(cfg.materials.box),
               box_height=cfg.stack.box_height_um,
               top_cladding_medium=None if clad_is_air else ctx.medium(cfg.materials.cladding),
               top_cladding_height=0.0 if clad_is_air else cfg.stack.cladding_height_um,
               fine_steps_per_wvl=fine, coarse_steps_per_wvl=coarse)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sims = {f"gap_{g:g}": partial(radius=R, gap=g, **geo).sim for g in gaps}
    for msg in sorted({str(w.message) for w in caught}):
        ui.say(f"  Note from the builder: {msg}")
    cost = _estimate(ctx, sims)
    if cost is None or not ui.proceed(f"ring-bus gap sweep, {len(gaps)} cloud FDTD runs "
                                      f"(R = {R} um, {span:g}° partial ring)", cost,
                                      ctx.budget_left):
        return dict(skipped=True)

    from tarang.ring_resonator.ring_gap_sweep import ring_gap_sweep
    volume = 2 * math.pi * R * W * H
    with ctx.cloud_limit(), warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)        # already shown above
        crit = ring_gap_sweep(gaps, ctx.out / "cloud" / "gap_sweep", R, ring["q_intrinsic"], volume,
                              n_group=ring["n_group"], **{k: v for k, v in geo.items()
                                                          if k != "span_angle_rad"},
                              span_angle_rad=math.radians(span))
    ctx.add_spent(cost)
    g_crit = crit["critical_gap_um"]
    g_over = physics.gap_for_over_coupling(crit, K) if K and K != 1 else None
    rows = [dict(regime="critical (K = 1)", gap_um=g_crit, Q_loaded=physics.loaded_q(ring["q_intrinsic"], 1))]
    if g_over is not None:
        rows.append(dict(regime=f"over-coupled (K = {K:g})", gap_um=g_over,
                         Q_loaded=physics.loaded_q(ring["q_intrinsic"], K)))
    ui.table(rows, formats={"gap_um": ".3f", "Q_loaded": ".3g"})
    if crit.get("critical_gap_extrapolated"):
        ui.say("  Note: the critical gap lies outside the simulated gaps (extrapolated); add "
               "gaps around it and rerun this stage.")
    return dict(critical_gap_um=g_crit, over_coupling=K, over_coupled_gap_um=g_over,
                extrapolated=bool(crit.get("critical_gap_extrapolated")),
                q_loaded_critical=physics.loaded_q(ring["q_intrinsic"], 1),
                q_loaded_over=physics.loaded_q(ring["q_intrinsic"], K) if g_over else None,
                fit=crit.get("fit_params"), estimated_cost=cost,
                folder=str(ctx.out / "cloud" / "gap_sweep"))


def stage_grating(ctx: Context) -> dict:
    cfg, ui = ctx.cfg, ctx.ui
    xs = ctx.results("cross_section")
    H, wl = xs["height_um"], cfg.run.wavelength_um
    etch = ui.value(cfg, "grating", "etch", check=lambda v: "" if v in ("full", "partial")
                    else "answer full or partial")
    depth = H if etch == "full" else ui.value(cfg, "grating", "etch_depth_um", default=round(H / 2, 3),
                                              check=lambda v: "" if 0 < v < H else f"0 < depth < {H}")
    angle = ui.value(cfg, "grating", "fiber_angle_deg")
    mfd = ui.value(cfg, "grating", "mfd_um")
    teeth = ui.value(cfg, "grating", "n_teeth")
    width = ui.value(cfg, "grating", "width_um")
    ffs = ui.value(cfg, "grating", "fill_factors", cast=parse_list)

    m = cfg.materials
    n_core, n_box = ctx.n(m.core), ctx.n(m.box)
    n_top = 1.0 if str(m.cladding).lower() == "air" else ctx.n(m.cladding)
    n_slab = physics.slab_te_index(n_core, n_box, n_top, H, wl)
    n_groove = n_top if etch == "full" else physics.slab_te_index(n_core, n_box, n_top, H - depth, wl)
    p0 = physics.grating_period(n_slab, n_groove, 0.5, n_top, angle, wl)
    ui.say(f"Slab TE index {n_slab:.4f}, groove {n_groove:.4f}: first-order period ≈ {p0:.3f} um "
           f"at fill factor 0.5")
    pdef = [physics.round_to(p0 * f, 0.005) for f in (0.94, 0.97, 1.0, 1.03, 1.06)]
    periods = ui.value(cfg, "grating", "periods_um", default=pdef, cast=parse_list)
    xdef = [round(mfd * f, 2) for f in (0.3, 0.5, 0.7)]
    xs_list = ui.value(cfg, "grating", "source_x_um", default=xdef, cast=parse_list)
    ui.value(cfg, "run", "max_cost_per_run")
    ui.value(cfg, "run", "max_total_cost")

    from tarang.grating_coupler.grating_coupler import GratingConfig, GratingCoupler
    handle = None if str(m.handle).lower() == "none" else ctx.medium(m.handle)
    clad = None if str(m.cladding).lower() == "air" else ctx.medium(m.cladding)

    def device(period, ff, sx, is_3d=False):
        gcfg = GratingConfig(wavelength_um=wl, theta_deg=-abs(angle), mfd=mfd, source_x=sx,
                             source_gap=2.0, core_height=H,
                             top_cladding_height=cfg.stack.cladding_height_um,
                             box_height=cfg.stack.box_height_um, handle_height=3.0,
                             etch_depth=depth, period=period, ff=ff, n_teeth=teeth,
                             w_grating=width, pol_angle=math.pi / 2, buffer=3.0)
        return GratingCoupler.from_config(gcfg, core_medium=ctx.medium(m.core),
                                          box_medium=ctx.medium(m.box), handle_medium=handle,
                                          top_cladding_medium=clad, is_3d=is_3d)

    folder = ctx.out / "cloud" / "grating"
    sx_mid = xs_list[len(xs_list) // 2]
    grid = [(p, f, sx_mid) for p in periods for f in ffs]
    results = []

    def run_batch(points, label, is_3d=False):
        devs = {pt: device(*pt, is_3d=is_3d) for pt in points}
        todo = {f"gc_{i}": d.sim for i, (pt, d) in enumerate(devs.items())
                if not d.is_cached(folder)}
        cost = _estimate(ctx, todo) if todo else 0.0
        if cost is None or not ui.proceed(f"{label}: {len(points)} {'3D' if is_3d else '2D'} "
                                          f"grating simulations ({len(todo)} not cached)",
                                          cost, ctx.budget_left):
            return None
        out = []
        with ctx.cloud_limit():
            for pt, d in devs.items():
                eff = float(d.get_efficiency(folder))
                out.append(dict(period_um=pt[0], fill_factor=pt[1], source_x_um=pt[2],
                                efficiency=eff, loss_db=-10 * math.log10(max(eff, 1e-12)),
                                dim="3D" if is_3d else "2D"))
        ctx.add_spent(cost)
        return out

    stage_a = run_batch(grid, "grating period x fill factor")
    if stage_a is None:
        return dict(skipped=True)
    results += stage_a
    best = max(stage_a, key=lambda r: r["efficiency"])
    others = [x for x in xs_list if x != best["source_x_um"]]
    if others:
        stage_b = run_batch([(best["period_um"], best["fill_factor"], x) for x in others],
                            "fibre position for the best grating")
        if stage_b:
            results += stage_b
            best = max(results, key=lambda r: r["efficiency"])
    ui.say("\nBest 2D designs:")
    ui.table(sorted(results, key=lambda r: -r["efficiency"])[:6],
             ["period_um", "fill_factor", "source_x_um", "efficiency", "loss_db"],
             {"efficiency": ".3f", "loss_db": ".2f"})
    best3d = None
    if ui.value(cfg, "grating", "check_3d"):
        r3 = run_batch([(best["period_um"], best["fill_factor"], best["source_x_um"])],
                       "3D check of the best grating", is_3d=True)
        if r3:
            best3d = r3[0]
            results += r3
            ui.say(f"3D efficiency: {best3d['efficiency']:.3f} ({best3d['loss_db']:.2f} dB)")
    table = ctx.path("tables", "grating_sweep.csv")
    with open(table, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(results)
    return dict(etch=etch, etch_depth_um=depth, fiber_angle_deg=angle, mfd_um=mfd, n_teeth=teeth,
                width_um=width, period_um=best["period_um"], fill_factor=best["fill_factor"],
                source_x_um=best["source_x_um"], efficiency_2d=best["efficiency"],
                loss_db_2d=best["loss_db"], efficiency_3d=best3d and best3d["efficiency"],
                period_estimate_um=p0, n_slab=n_slab, table=str(table))


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
    "layout": ("layout",),
    "report": (),
}


def settings_key(cfg: DesignConfig, stage: str) -> str:
    """Hash of every setting this stage and the stages before it depend on."""
    parts = {}
    for name in STAGE_NAMES[:STAGE_NAMES.index(stage) + 1]:
        for dep in DEPENDS[name]:
            sec, _, key = dep.partition(".")
            val = getattr(cfg, sec)
            parts[dep] = getattr(val, key) if key else val.__dict__
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
