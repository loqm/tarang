"""Stage "checks": how robust and complete the design is, after the gap and grating stages.

Free (local or from saved results):
  * figures that runs made before Phase 12 lack (mode profiles, bent mode, the three-panel
    gap figure, the coupled-mode check, the gap field map);
  * gap tolerance: a width error dW moves both edges, so the gap changes by -dW; with the
    FDTD fit, K(g) = kappa^2(g) / rho gives the coupling ratio, the loaded Q and the depth
    of the resonance dip ((1 - K) / (1 + K))^2 at g_c -/+ dW;
  * FSR over the cross-section's tolerance box (from the group index there);
  * ring mode families: guided modes of the bent waveguide, their group index and how many
    TE00 modes lie between crossings with TE00, N = FSR_00 / |FSR_00 - FSR_i|;
  * fibre-to-fibre loss budget at the design wavelength.
Cloud (priced and confirmed one by one):
  * coupling against wavelength at the critical and over-coupled gaps (2 runs, unless the
    gap runs recorded the band);
  * grating: where the light goes when it is launched from the waveguide (up = towards
    the fibre, down into the substrate, reflected back, past the grating), the fibre angle,
    film thickness and tooth width at +/- tolerance, and a finer mesh;
  * grating efficiency against the fibre angle (``angle_sweep_deg``), with the peak shift
    the grating equation predicts;
  * optionally the critical gap again at a 1.5x finer mesh.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Optional

import numpy as np

from tarang.design import physics
from tarang.design.ui import parse_list
from tarang.design.stages import (C_UM, FIGURE_VERSION, Context, _estimate, fig_bent_mode,
                                  fig_mode_profiles,
                                  gap_extras, grating_device, _gap_device)

LOG10E = 10 * math.log10(math.e)


def _form(ctx: Context) -> None:
    cfg = ctx.cfg
    ctx.ui.form(cfg, "Checks", [
        dict(section="checks", key="band", show=ring_goal),
        dict(section="checks", key="grating_extras", show=grating_coupling),
        dict(section="checks", key="angle_tolerance_deg", show=grating_coupling),
        dict(section="checks", key="angle_sweep_deg", show=grating_coupling, cast=parse_list),
        dict(section="checks", key="litho_bias_um", show=grating_coupling),
        dict(section="checks", key="mesh_grating", show=grating_coupling),
        dict(section="checks", key="mesh_gap", show=ring_goal),
        dict(section="checks", key="mode_families", show=ring_goal)])


def grating_coupling(cfg) -> bool:
    from tarang.design.stages import coupling
    return coupling(cfg) == "grating"


def ring_goal(cfg) -> bool:
    return str(getattr(cfg.run, "goal", "comb_ring")) == "comb_ring"


def _done(ctx: Context, name: str) -> Optional[dict]:
    from tarang.design.stages import applies
    if not applies(name, ctx.cfg):               # e.g. a ring result left from another goal
        return None
    r = ctx.state["stages"].get(name, {}).get("results")
    return None if not r or r.get("skipped") else r


def _write_csv(path: Path, rows: list[dict]) -> Path:
    """Rows as a CSV table; the columns are every key of every row in order of first
    appearance (rows of different kinds may share a table, missing cells stay empty)."""
    if rows:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    return path


# ============================================================================ free checks
def refresh_figures(ctx: Context, gap: Optional[dict]) -> dict:
    """Draw the figures a run made before Phase 12 lacks; returns the coupled-mode check
    if it had to be computed."""
    from tarang.design import visuals
    xs, ring = ctx.results("cross_section"), ctx.results("ring")
    H, W, R = xs["height_um"], xs["width_um"], ring["radius_um"]
    fig = ctx.out / "figures"
    if not (fig / "2b_mode_profiles.png").exists():
        fig_mode_profiles(ctx, H, W)
    if not (fig / "3b_bent_mode.png").exists():
        fig_bent_mode(ctx, H, W, R)
    cmt = {}
    if gap:
        per = gap.get("per_gap") or []
        if per:
            K = gap.get("over_coupling") or 1.5
            targets = {"critical": gap["target_cross_power"]}
            chosen = {"critical": gap["critical_gap_um"]}
            if gap.get("over_coupled_gap_um"):
                targets[f"over-coupled K = {K:g}"] = min(K * gap["target_cross_power"], 0.999)
                chosen[f"over-coupled K = {K:g}"] = gap["over_coupled_gap_um"]
            raw = (np.array([r["one_minus_T_source"] for r in per])
                   if gap.get("reference") != "source" else None)
            visuals.safe(ctx, "gap coupling", visuals.gap_coupling,
                         np.array([r["gap_um"] for r in per]), raw,
                         np.array([r["coupled_power"] for r in per]), gap.get("fit"), targets,
                         ctx.path("figures", "4b_gap_coupling.png"), chosen, gap.get("fit_r2"))
        if not gap.get("coupled_mode_check") or not (fig / "4d_gap_field.png").exists():
            cmt = gap_extras(ctx, gap)
    return cmt


def gap_tolerance(ctx: Context, gap: dict) -> list[dict]:
    """Coupling ratio, loaded Q and resonance depth when the gap is off by the width
    tolerance (both edges move by dW/2, so the gap by dW)."""
    fit = gap.get("fit") or {}
    if not fit:
        return []
    ring = ctx.results("ring")
    dW = float(ctx.cfg.cross_section.tolerance_width_um)
    rho = gap["target_cross_power"]
    qi = ring["q_intrinsic"]
    rows = []
    designs = [("critical", gap["critical_gap_um"])]
    if gap.get("over_coupled_gap_um"):
        designs.append((f"over-coupled K = {gap.get('over_coupling', 1.5):g}",
                        gap["over_coupled_gap_um"]))
    for name, g0 in designs:
        for label, dg in ((f"waveguides {dW * 1e3:.0f} nm wider", -dW), ("as designed", 0.0),
                          (f"waveguides {dW * 1e3:.0f} nm narrower", +dW)):
            g = g0 + dg
            k2 = math.sin(min(fit["theta_0"] * math.exp(-fit["gamma"] * g), math.pi / 2)) ** 2
            K = k2 / rho
            dip_db = -10 * math.log10(max(((1 - K) / (1 + K)) ** 2, 1e-30))
            rows.append(dict(design=name, case=label, gap_um=round(g, 4),
                             coupled_power=k2, K=round(K, 3), Q_loaded=qi / (1 + K),
                             dip_depth_db=round(dip_db, 1) if dip_db < 40 else ">40"))
    return rows


def fsr_tolerance(ctx: Context) -> Optional[dict]:
    """FSR range over the cross-section's tolerance box (group index of its corners)."""
    import json
    xs, ring = ctx.results("cross_section"), ctx.results("ring")
    H, W, steps = xs["height_um"], xs["width_um"], xs["steps"]
    f = ctx.out / "results" / f"finalist_{steps}steps.jsonl"
    if not f.exists():
        return None
    rows = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
    dW = ctx.cfg.cross_section.tolerance_width_um
    dH = ctx.cfg.cross_section.tolerance_height_um
    hs = {round(H + k * dH, 4) for k in (-1, 0, 1)}
    ws = {round(W + k * dW, 4) for k in (-1, 0, 1)}
    ngs = [r["ng_TE0"] for r in rows if "error" not in r and r.get("bend_radius") is None
           and round(r["H"], 4) in hs and round(r["W"], 4) in ws and r.get("ng_TE0")]
    if len(ngs) < 2:
        return None
    R = ring["radius_um"]
    return dict(n_g_min=min(ngs), n_g_max=max(ngs), fsr_min_ghz=physics.fsr_ghz(R, max(ngs)),
                fsr_max_ghz=physics.fsr_ghz(R, min(ngs)))


def mode_families(ctx: Context) -> list[dict]:
    """Guided modes of the bent ring waveguide and how often they cross TE00."""
    from tarang.design import localmodes
    xs, ring = ctx.results("cross_section"), ctx.results("ring")
    H, W, R = xs["height_um"], xs["width_um"], ring["radius_um"]
    wl = ctx.cfg.run.wavelength_um
    n_cut = max(ctx.n(ctx.cfg.materials.box), 1.0 if str(ctx.cfg.materials.cladding).lower()
                == "air" else ctx.n(ctx.cfg.materials.cladding))
    modes = localmodes.cross_section_modes(ctx, H, W, bend_radius=R, num_modes=6,
                                           group_index=True)
    lo, hi = ctx.band
    n_modes_band = (C_UM / lo - C_UM / hi) / (physics.fsr_ghz(R, ring["n_group"]) * 1e9)
    te0 = localmodes.te00(modes)
    rows = []
    counts = {"TE": 0, "TM": 0}
    for m in modes:
        loss = 4 * math.pi * abs(m["k"]) / (wl * 1e-4) * LOG10E          # dB/cm
        guided = m["neff"] > n_cut and loss < 1.0
        if not guided:
            continue
        pol = "TE" if m["te"] > 0.5 else "TM"
        name = f"{pol}{counts[pol]}0"
        counts[pol] += 1
        ng = m.get("ng") or float("nan")
        fsr = physics.fsr_ghz(R, ng)
        row = dict(family=name, n_eff=round(m["neff"], 5), n_g=round(ng, 4),
                   fsr_ghz=round(fsr, 3), loss_db_cm=float(f"{loss:.2g}"))
        if m is not te0 and te0 is not None:
            fsr0 = physics.fsr_ghz(R, te0["ng"])
            N = fsr0 / max(abs(fsr0 - fsr), 1e-12)
            row.update(modes_between_crossings=round(N), crossings_in_band=round(n_modes_band / N, 2))
        else:
            row.update(modes_between_crossings=None, crossings_in_band=None)
        rows.append(row)
    return rows


def arc_reflection(ctx: Context, gap: dict) -> list[dict]:
    """Reflections in the saved gap runs (free, from their mode monitors at the design
    wavelength). The partial-ring arcs end inside the simulation box, so light that entered
    the ring is partly reflected at the arc end and comes back: at the ring port
    'Ring_Out_Right' that is the backward-travelling amplitude, R_arc = |a-|^2 / |a+|^2.
    In the bus, the backward amplitude at the input monitor gives R_bus (what returns
    towards the source, both from the bus end and from the ring)."""
    from tarang.utilities.sweep_helpers import cache_dir as sweep_cache_dir
    rows = []
    f0 = C_UM / ctx.cfg.run.wavelength_um
    folder = Path(gap.get("folder") or ctx.out / "cloud" / "gap_sweep")
    for g in gap.get("gaps_um") or []:
        try:
            d = _gap_device(ctx, g)
            cache = sweep_cache_dir(folder, d)
            if not d.is_cached(cache):
                continue
            row = reflections(d._run_sim(cache), f0)
        except Exception:                                         # noqa: BLE001
            continue
        rows.append(dict(gap_um=g, **row))
    return rows


def reflections(data, f0: float) -> dict:
    """Forward and backward power in the ring port and the bus input of one gap run."""
    def power(mon, direction):
        a = data[mon].amps.sel(direction=direction, mode_index=0)
        return float(np.abs(np.asarray(a.sel(f=f0, method="nearest").values)) ** 2)
    fw, bw = power("Ring_Out_Right", "+"), power("Ring_Out_Right", "-")
    bus_f, bus_b = power("Through_In", "+"), power("Through_In", "-")
    return dict(ring_forward=fw, ring_backward=bw,
                arc_end_reflection=bw / fw if fw > 0 else float("nan"),
                bus_reflection=bus_b / bus_f if bus_f > 0 else float("nan"))


def loss_budget(ctx: Context, grating: Optional[dict]) -> list[dict]:
    """Fibre-to-fibre transmission off resonance at the design wavelength."""
    from tarang.design.layout import coupler_plan
    from tarang.design.stages import _losses
    cfg = ctx.cfg
    ring = (ctx.state["stages"].get("ring") or {}).get("results") or {}
    straight = (ring["scattering_loss_db_cm"] + ring["absorption_loss_db_cm"] if ring
                else sum(_losses(cfg)))
    if not grating_coupling(cfg):
        return _edge_budget(ctx, straight)
    plan = coupler_plan(ctx)
    length_cm = (cfg.layout.bus_length_um + 2 * cfg.grating.taper_length_um) * 1e-4
    tp = ctx.state["stages"].get("taper", {}).get("results") or {}
    if tp and not tp.get("skipped") and tp.get("taper_length_um") and cfg.taper.use_in_layout:
        length_cm = (cfg.layout.bus_length_um + 2 * tp["taper_length_um"]) * 1e-4
    fc = ctx.state["stages"].get("focus", {}).get("results") or {}
    ap = ctx.state["stages"].get("apodise", {}).get("results") or {}
    use_fc = plan["main"] == "focusing"
    label = "grating coupler"
    if plan["main"] == "apodised" and ap.get("efficiency_2d"):
        grating = dict(grating or {}, efficiency_2d=ap["efficiency_2d"])
        label = "apodised grating coupler"
    if use_fc:
        length_cm = (cfg.layout.bus_length_um + 2 * fc["tip_taper_um"]) * 1e-4
        if fc.get("teeth_kind") == "apodised" and ap.get("efficiency_2d"):
            grating = dict(grating or {}, efficiency_2d=ap["efficiency_2d"])
    rows = []
    fc3d = use_fc and fc.get("efficiency_3d")
    if fc3d:
        # the 3D FDTD of the focusing grating already contains the in-plane focusing and the
        # grating's width: only the tip taper is added (the same as the whole-device model)
        db = -10 * math.log10(max(fc["efficiency_3d"], 1e-9))
        rows.append(dict(item=f"focusing grating ({fc.get('teeth_kind', 'uniform')} teeth), input "
                              "(3D FDTD)", loss_db=round(db, 2)))
        rows.append(dict(item="focusing grating, output (same design)", loss_db=round(db, 2)))
    elif grating:
        gc_db = -10 * math.log10(max(grating["efficiency_2d"], 1e-9))
        rows.append(dict(item=f"{label}, input (2D simulation)", loss_db=round(gc_db, 2)))
        rows.append(dict(item=f"{label}, output (same design)", loss_db=round(gc_db, 2)))
    if use_fc:
        if not fc3d:
            rows.append(dict(item=f"focusing into a {fc['tip_width_um']:g} µm tip, 2 x (in-plane model)",
                             loss_db=round(-20 * math.log10(max(fc["eta_focus"], 1e-9)), 3)))
        if fc.get("tip_taper_um") and not fc3d:          # the 3D run contains the tip taper
            rows.append(dict(item=f"tip tapers, 2 x {fc['tip_taper_um']:g} µm (2D model)",
                             loss_db=round(-20 * math.log10(max(fc["tip_taper_T"], 1e-9)), 3)))
        tp = {}                                   # no long taper, no straight-grating overlap
    if tp.get("eta_y"):
        ly = -10 * math.log10(max(tp["eta_y"], 1e-9))
        rows.append(dict(item=f"fibre-grating lateral overlap, 2 x (W = {tp['grating_width_um']:g} µm)",
                         loss_db=round(2 * ly, 3)))
    if tp.get("taper_loss_db") is not None:
        how = tp.get("length_source") or "EME"
        rows.append(dict(item=f"tapers, 2 x {tp['taper_length_um']:g} µm ({how})",
                         loss_db=round(2 * tp["taper_loss_db"], 3)))
    elif tp.get("model_loss_db_at_previous") is not None:
        rows.append(dict(item=f"tapers, 2 x {tp['previous_length_um']:g} µm (2D model)",
                         loss_db=round(2 * tp["model_loss_db_at_previous"], 3)))
    elif tp and not tp.get("skipped"):
        rows.append(dict(item="tapers, 2 x (no length below the accepted loss)", loss_db=0.0))
    elif not use_fc:
        rows.append(dict(item="tapers, 2 x (not simulated: run the taper stage)", loss_db=0.0))
    rows.append(dict(item=f"propagation, {length_cm * 1e4:.0f} µm at {straight:g} dB/cm",
                     loss_db=round(straight * length_cm, 3)))
    total = sum(r["loss_db"] for r in rows)
    rows.append(dict(item="fibre to fibre, off resonance", loss_db=round(total, 2)))
    return rows


def _edge_budget(ctx: Context, straight: float) -> list[dict]:
    """Fibre-to-fibre budget with edge couplers (the edge stage's results; empty until it has
    run, the report recomputes it)."""
    cfg = ctx.cfg
    ed = (ctx.state["stages"].get("edge") or {}).get("results") or {}
    rows = []
    if ed.get("facet_width_um"):
        f3 = ed.get("fdtd")
        eta, how = (f3["eta"], "3D FDTD") if f3 else (ed["eta_free"], "butt-coupling estimate")
        db = -10 * math.log10(max(eta, 1e-9))
        rows.append(dict(item=f"facet, input ({ed['facet_width_um']:g} µm, {how})", loss_db=round(db, 2)))
        rows.append(dict(item="facet, output (same design)", loss_db=round(db, 2)))
        rows.append(dict(item=f"{ed['profile']} tapers, 2 x {ed['taper_length_um']:.0f} µm "
                              f"({ed['taper_loss_source']})", loss_db=round(2 * ed["taper_loss_db"], 3)))
    L_cm = float(cfg.edge.chip_length_um) * 1e-4
    rows.append(dict(item=f"propagation, {L_cm * 1e4:.0f} µm (facet to facet) at {straight:g} dB/cm",
                     loss_db=round(straight * L_cm, 3)))
    total = sum(r["loss_db"] for r in rows)
    rows.append(dict(item="fibre to fibre, off resonance", loss_db=round(total, 2)))
    return rows


# ============================================================================ cloud checks
def _load(sim, folder: Path, name: str):
    import tidy3d as td
    from tarang.utilities.cloud_run import find_cached
    p = find_cached(sim, folder, name)
    return None if p is None else td.SimulationData.from_file(str(p))


def _run_batch(ctx: Context, jobs: dict, folder: Path, label: str) -> Optional[bool]:
    """Price, confirm and run {key: (sim, name)}; True when all results exist, None when
    skipped or not affordable."""
    from tarang.utilities.cloud_run import find_cached, run_batch_cached
    todo = {k: sim for k, (sim, name) in jobs.items() if find_cached(sim, folder, name) is None}
    cost = _estimate(ctx, {f"chk_{i}": s for i, s in enumerate(todo.values())}) if todo else 0.0
    if cost is None or (todo and not ctx.account_ok(cost)):
        return None
    if not ctx.ui.proceed(f"{label}: {len(jobs)} simulations ({len(todo)} not computed yet)",
                          cost, ctx.budget_left):
        return None
    ctx.sim_pictures({f"{label}: {k}": (sim, name) for k, (sim, name) in jobs.items()},
                     z=ctx.results("cross_section")["height_um"] / 2)
    if todo:
        cap = float(ctx.cfg.run.max_cost_per_run) * len(todo)
        with ctx.cloud_limit(), ctx.cloud_progress(label, 1):
            run_batch_cached({k: jobs[k] for k in todo}, folder, max_cost=cap)
        ctx.add_spent(cost, f"{label} ({len(todo)} runs)")
    return True


def band_check(ctx: Context, gap: dict) -> Optional[dict]:
    """Coupling against wavelength at the critical and over-coupled gaps."""
    from tarang.design import visuals
    cfg = ctx.cfg
    designs = {"critical": gap["critical_gap_um"]}
    if gap.get("over_coupled_gap_um"):
        designs[f"over-coupled K = {gap.get('over_coupling', 1.5):g}"] = gap["over_coupled_gap_um"]
    recorded = int(gap.get("band_points") or 1) > 1
    old_points = cfg.gap.band_points
    if not recorded:
        cfg.gap.band_points = 11                      # these two runs record the band
    try:
        devs = {k: _gap_device(ctx, round(g, 4)) for k, g in designs.items()}
    finally:
        cfg.gap.band_points = old_points
    folder = ctx.out / "cloud" / "band"
    jobs = {k: (d.sim, d._readable_name()) for k, d in devs.items()}
    if _run_batch(ctx, jobs, folder, "coupling against wavelength") is None:
        return None
    rho = gap["target_cross_power"]
    curves = {}
    for k, d in devs.items():
        data = _load(d.sim, folder, d._readable_name())
        if data is None:
            continue
        try:
            amps = data["Ring_Out_Right"].amps.sel(direction="+", mode_index=0)
        except Exception as err:                                  # noqa: BLE001
            ctx.ui.say(f"  ({k}: ring-port data unreadable: {type(err).__name__})")
            continue
        f = amps.coords["f"].values
        k2 = np.abs(amps.values) ** 2
        wl = C_UM / f
        o = np.argsort(wl)
        curves[k] = dict(gap_um=designs[k], wavelength_um=wl[o].tolist(), coupled=k2[o].tolist(),
                         K=(k2[o] / rho).tolist())
    if not curves:
        return None

    visuals.safe(ctx, "coupling against wavelength", draw_band, ctx, curves)
    rows = [dict(design=k, gap_um=c["gap_um"], wavelength_um=round(w, 5), coupled_power=p, K=kk)
            for k, c in curves.items() for w, p, kk in zip(c["wavelength_um"], c["coupled"], c["K"])]
    _write_csv(ctx.path("tables", "coupling_vs_wavelength.csv"), rows)
    return dict(curves=curves, recorded_in_gap_runs=recorded)


def k_crossings(wl_um, K) -> list[float]:
    """Wavelengths where K(lambda) crosses 1 (linear interpolation of ln K)."""
    wl, lk = np.asarray(wl_um, float), np.log(np.clip(np.asarray(K, float), 1e-12, None))
    out = []
    for i in range(len(wl) - 1):
        if lk[i] == 0:
            out.append(float(wl[i]))
        elif lk[i] * lk[i + 1] < 0:
            out.append(float(wl[i] + (wl[i + 1] - wl[i]) * (-lk[i]) / (lk[i + 1] - lk[i])))
    return out


def draw_loss_budget(ctx: Context, lb: list[dict]):
    """Figure 7c: the fibre-to-fibre loss as accumulated bars."""
    from tarang.design import visuals
    plt = visuals._plt()
    items = lb[:-1]
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(items) + 1.6))
    left = 0.0
    for i, r in enumerate(items):
        ax.barh(i, max(r["loss_db"], 0.0), left=left, color=visuals.PALETTE[i % len(visuals.PALETTE)])
        ax.text(left + max(r["loss_db"], 0.0) + 0.05, i, f"{r['loss_db']:.2f} dB", va="center",
                fontsize=8.5)
        left += max(r["loss_db"], 0.0)
    ax.set_yticks(range(len(items)), [r["item"] for r in items])
    ax.invert_yaxis()
    ax.set_xlabel("Loss (dB), accumulated")
    ax.set_title(f"Fibre-to-fibre: {lb[-1]['loss_db']:.2f} dB at {ctx.cfg.run.wavelength_um} µm "
                 "(off resonance)")
    ax.set_xlim(0, max(left * 1.25, 1))
    return visuals._save(fig, ctx.path("figures", "7c_loss_budget.png"))


def draw_band(ctx: Context, curves: dict):
    """Figure 7a. ln(kappa^2) is smooth in wavelength (the evanescent decay length grows
    smoothly), so a quadratic fit through the simulated points gives continuous curves; the
    resonance depth ((1-K)/(1+K))^2 is unbounded at K = 1, so it is drawn from the fit
    (finely sampled) rather than by joining the points."""
    from tarang.design import visuals
    plt = visuals._plt()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    wl0 = ctx.cfg.run.wavelength_um
    for n_curve, (k, c) in enumerate(curves.items()):
        wl = np.array(c["wavelength_um"])
        k2 = np.array(c["coupled"])
        K = np.array(c["K"])
        col = visuals.PALETTE[1] if k == "critical" else visuals.PALETTE[0]
        p = np.polyfit(wl - wl0, np.log(np.clip(k2, 1e-12, None)), min(2, len(wl) - 1))
        wf = np.linspace(wl.min(), wl.max(), 600)
        k2f = np.exp(np.polyval(p, wf - wl0))
        Kf = k2f * (K[0] / k2[0])                    # same round-trip loss at every wavelength
        lab = f"{k}: gap {c['gap_um'] * 1e3:.0f} nm"
        axes[0].plot(wf * 1e3, k2f * 100, "-", color=col, lw=1.6)
        axes[0].plot(wl * 1e3, k2 * 100, "o", color=col, ms=5, label=lab)
        axes[1].plot(wf * 1e3, Kf, "-", color=col, lw=1.6)
        axes[1].plot(wl * 1e3, K, "o", color=col, ms=5, label=k)
        dip = -10 * np.log10(np.clip(((1 - Kf) / (1 + Kf)) ** 2, 1e-6, None))
        axes[2].plot(wf * 1e3, dip, "-", color=col, lw=1.6, label=k)
        for x in k_crossings(wf, Kf):
            for ax in axes[1:]:
                ax.axvline(x * 1e3, color=col, ls="--", lw=1)
            axes[1].annotate(f"K = 1 at {x * 1e3:.0f} nm", (x * 1e3, 1),
                             xytext=(8, -16 - 14 * n_curve) if n_curve else (8, 10),
                             textcoords="offset points", color=col, fontsize=8.5,
                             bbox=dict(boxstyle="round,pad=0.15", fc="w", ec="none", alpha=0.85))
    axes[0].set(xlabel="Wavelength (nm)", ylabel="Coupled power per pass κ² (%)",
                title="coupling grows with wavelength (longer evanescent tail)")
    axes[1].axhline(1, color="0.4", ls=":", lw=1)
    axes[1].set(xlabel="Wavelength (nm)", ylabel="K = κ² / round-trip loss", yscale="log",
                title="coupling regime (K > 1 over-coupled, K < 1 under-coupled)")
    from matplotlib.ticker import FuncFormatter, LogLocator
    axes[1].yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 3, 5)))
    axes[1].yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axes[2].set(xlabel="Wavelength (nm)", ylabel="Resonance dip depth (dB)", ylim=(0, 45),
                title="through-port dip ((1 − K)/(1 + K))², from the fit\n(unbounded at K = 1; "
                      "shown up to 45 dB)")
    for ax in axes:
        ax.axvline(wl0 * 1e3, color="0.6", ls=":", lw=1)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle("Ring-bus coupling against wavelength: 3D FDTD (dots) and a smooth fit (lines); "
                 "loss taken the same at every wavelength")
    return visuals._save(fig, ctx.path("figures", "7a_coupling_vs_wavelength.png"))


def draw_grating_checks(ctx: Context, spectra: dict, budget: Optional[dict]):
    """Figure 7b: tolerance spectra and, from the reverse run, where the light goes."""
    from tarang.design import visuals
    if not spectra:
        return None
    plt = visuals._plt()
    wl0 = ctx.cfg.run.wavelength_um
    ncol = 2 if budget else 1
    fig, axes = plt.subplots(1, ncol, figsize=(7.5 * ncol, 4.8), squeeze=False)
    ax = axes[0, 0]
    colours = iter(["#0072B2", "#56B4E9", "#D55E00", "#E69F00", "#009E73", "#CC79A7", "#7F3C8D"])
    for k, (wls, eta) in spectra.items():
        style = (dict(color="k", lw=2.6, ls="-") if k == "final" else
                 dict(color="0.25", lw=1.4, ls="--") if k.startswith("mesh") else
                 dict(color=next(colours), lw=1.4, ls="-"))
        ax.plot(np.asarray(wls) * 1e3, 10 * np.log10(np.clip(eta, 1e-6, None)), label=k, **style)
    ax.axvline(wl0 * 1e3, color="0.6", ls=":")
    top = max(10 * math.log10(max(max(e), 1e-6)) for _, e in spectra.values())
    ax.set(xlabel="Wavelength (nm)", ylabel="Coupling efficiency (dB)", ylim=(top - 10, top + 0.5),
           title="Grating tolerances: fibre angle, film thickness, tooth width, mesh")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, ncol=2)
    if budget:
        ax = axes[0, 1]
        wl = np.array(budget["wavelength_um"]) * 1e3
        up = np.array(budget["up"])
        coupled = np.array(budget["coupled"] if budget.get("coupled") else up)
        parts = [("up, coupled to the fibre", coupled), ("up, not matched to the fibre", up - coupled),
                 ("down, into the substrate", np.array(budget["down"])),
                 ("past the grating", np.array(budget["past"])),
                 ("reflected into the waveguide", np.array(budget["reflected"]))]
        if budget.get("scattered_back") is not None:
            parts.append(("scattered backwards", np.array(budget["scattered_back"])))
        cols = ["#009E73", "#56B4E9", "#D55E00", "#E69F00", "#CC79A7", "#999999"]
        ax.stackplot(wl, *[np.clip(p, 0, None) for _, p in parts], labels=[n for n, _ in parts],
                     colors=cols[:len(parts)], alpha=0.85)
        ax.axvline(wl0 * 1e3, color="k", ls=":")
        ax.set(xlabel="Wavelength (nm)", ylabel="Fraction of the waveguide power",
               ylim=(0, 1.05), title="Where the light goes (launched from the waveguide)")
        ax.legend(fontsize=7.5, loc="upper right")
    return visuals._save(fig, ctx.path("figures", "7b_grating_checks.png"))


def _reverse_sim(device, ctx: Context):
    """The grating simulation run backwards: TE0 launched in the waveguide towards the
    grating, inside a closed box of flux planes: above (towards the fibre), below (into the
    substrate), past the grating and behind the source (light reflected or scattered back),
    plus the waveguide mode monitor for the reflection into the guided mode."""
    import tidy3d as td
    sim = device.sim
    H = device.config.core_height
    box = ctx.cfg.stack.box_height_um
    mm = next(m for m in sim.monitors if isinstance(m, td.ModeMonitor))
    st = sim.sources[0].source_time
    (x0, _, z0), (x1, _, z1) = sim.bounds
    teeth = getattr(device.config, "teeth", None)
    xg = (teeth[-1][0] + teeth[-1][1]) if teeth else device.config.period * device.config.n_teeth
    src = td.ModeSource(center=(mm.center[0] - 1.5, 0, mm.center[2]), size=(0, td.inf, mm.size[2]),
                        source_time=st, direction="+", mode_spec=mm.mode_spec, mode_index=0,
                        name="waveguide_source")
    # a closed box of flux planes: top (towards the fibre), bottom (inside the Si handle),
    # the right side past the grating and the left side behind the source, so every power
    # leaving the box is counted once. The source launches only towards +x, so the left
    # plane sees only light coming back: reflected into the guided mode (the mode monitor)
    # or scattered backwards (the rest).
    x_src = src.center[0]
    x_back = max(x_src - 1.0, x0 + 0.5)
    x_past = min(xg + 1.5, x1 - 0.5)
    z_up, z_down = H + 1.0, -box - 1.0
    fl = dict(freqs=list(mm.freqs))
    xm, xs = (x_back + x_past) / 2, x_past - x_back
    mons = [mm,
            td.FluxMonitor(center=(xm, 0, z_up), size=(xs, td.inf, 0), name="up", **fl),
            td.FluxMonitor(center=(xm, 0, z_down), size=(xs, td.inf, 0),
                           name="down", normal_dir="-", **fl),
            td.FluxMonitor(center=(x_past, 0, (z_up + z_down) / 2), size=(0, td.inf, z_up - z_down),
                           name="past", **fl),
            td.FluxMonitor(center=(x_back, 0, (z_up + z_down) / 2), size=(0, td.inf, z_up - z_down),
                           name="back", normal_dir="-", **fl)]
    del z0, z1
    return sim.updated_copy(sources=[src], monitors=mons)


def grating_checks(ctx: Context, gc: dict) -> Optional[dict]:
    """Power budget (reverse run), tolerances and mesh check of the final grating."""
    from tarang.grating_coupler.grating_coupler import spectrum_summary
    from tarang.design import visuals
    cfg = ctx.cfg
    ch = cfg.checks
    variants = grating_variants(ctx, gc)
    folder = ctx.out / "cloud" / "grating"
    jobs = {k: (d.sim, d._readable_name()) for k, d in variants.items()}
    rev = None
    if ch.grating_extras:
        rev = _reverse_sim(variants["final"], ctx)
        jobs["reverse"] = (rev, variants["final"]._readable_name() + "_reverse")
    sweep = angle_sweep_variants(ctx, gc)
    for ang, d in sweep.items():
        jobs[f"sweep {ang:g}"] = (d.sim, d._readable_name())
    if len(jobs) == 1 or _run_batch(ctx, jobs, folder, "grating checks") is None:
        return None
    wl0 = cfg.run.wavelength_um
    rows, spectra = [], {}
    for k, d in variants.items():
        data = _load(d.sim, folder, d._readable_name())
        if data is None:
            continue
        try:
            wls, eta = d.get_spectrum(folder)
        except Exception as err:                                  # noqa: BLE001
            ctx.ui.say(f"  ({k}: spectrum unreadable: {type(err).__name__})")
            continue
        s = spectrum_summary(wls, eta, wl0)
        spectra[k] = (wls, eta)
        rows.append(dict(case=k, efficiency_at_target=s["efficiency_at_target"],
                         peak_efficiency=s["peak_efficiency"],
                         peak_wavelength_nm=round(s["peak_wavelength_um"] * 1e3, 1),
                         bandwidth_1db_nm=round(s["bandwidth_1db_nm"], 1)))
    base = next((r for r in rows if r["case"] == "final"), None)
    for r in rows:
        if base:
            r["peak_shift_nm"] = round(r["peak_wavelength_nm"] - base["peak_wavelength_nm"], 1)
            r["change_db_at_target"] = round(10 * math.log10(max(r["efficiency_at_target"], 1e-9)
                                                            / max(base["efficiency_at_target"], 1e-9)), 2)
    budget = None
    if rev is not None:
        data = _load(rev, folder, jobs["reverse"][1])
        try:
            budget = _power_budget(data, rev, spectra, wl0) if data is not None else None
        except Exception as err:                                  # noqa: BLE001
            ctx.ui.say(f"  (power budget not computed: {type(err).__name__}: {err})")
            budget = None

    visuals.safe(ctx, "grating checks", draw_grating_checks, ctx, spectra, budget)
    _write_csv(ctx.path("tables", "grating_checks.csv"), rows)
    angle_rows = angle_sweep_rows(ctx, gc, sweep, spectra.get("final"))
    if angle_rows:
        _write_csv(ctx.path("tables", "grating_angle_sweep.csv"), angle_rows)
        visuals.safe(ctx, "grating angle sweep", draw_angle_sweep, ctx, gc, angle_rows)
    return dict(variants=rows, power_budget=budget, angle_sweep=angle_rows)


def angle_sweep_variants(ctx: Context, gc: dict) -> dict:
    """{angle: device} of the final grating at each fibre angle of ``angle_sweep_deg`` other
    than the design's own (its run is the final design)."""
    H = ctx.results("cross_section")["height_um"]
    angle = float(gc["fiber_angle_deg"])
    return {float(a): grating_device(ctx, gc["period_um"], gc["fill_factor"], gc["source_x_um"],
                                     H=H, depth=gc["etch_depth_um"], angle=float(a))
            for a in sorted({float(a) for a in (ctx.cfg.checks.angle_sweep_deg or [])})
            if abs(float(a) - angle) > 1e-9}


def _top_index(ctx: Context) -> float:
    m = ctx.cfg.materials
    return 1.0 if str(m.cladding).lower() == "air" else float(np.real(ctx.n(m.cladding)))


def expected_peak_nm(peak0_nm: float, period_um: float, theta0_deg: float, theta_deg: float,
                     tilt: str, n_top: float = 1.0) -> float:
    """Peak wavelength at another fibre angle from the first-order grating equation
    λ = Λ (n_avg ∓ n_top sin θ) (− for a beam travelling towards the waveguide): the peak
    moves by ∓ Λ n_top (sin θ − sin θ0); n_avg is taken as unchanged."""
    s = 1.0 if tilt == "towards" else -1.0
    return peak0_nm - s * period_um * 1e3 * n_top * (math.sin(math.radians(theta_deg))
                                                    - math.sin(math.radians(theta0_deg)))


def angle_sweep_rows(ctx: Context, gc: dict, sweep: dict, final_spectrum) -> list[dict]:
    """One row per fibre angle (the design's own angle included): efficiency at the design
    wavelength, peak efficiency and wavelength, 1 dB bandwidth, and the peak wavelength the
    grating equation predicts from the design's own run."""
    from tarang.grating_coupler.grating_coupler import spectrum_summary
    if not sweep or final_spectrum is None:
        return []
    folder = ctx.out / "cloud" / "grating"
    wl0 = ctx.cfg.run.wavelength_um
    angle0 = float(gc["fiber_angle_deg"])
    spectra = {angle0: final_spectrum}
    for ang, d in sweep.items():
        try:
            spectra[ang] = d.get_spectrum(folder)
        except Exception:                                         # noqa: BLE001
            continue
    s0 = spectrum_summary(*final_spectrum, wl0)
    n_top = _top_index(ctx)
    rows = []
    for ang in sorted(spectra):
        s = spectrum_summary(*spectra[ang], wl0)
        rows.append(dict(angle_deg=ang, efficiency_at_target=s["efficiency_at_target"],
                         loss_db_at_target=round(s["loss_db_at_target"], 3),
                         peak_efficiency=s["peak_efficiency"],
                         peak_wavelength_nm=round(s["peak_wavelength_um"] * 1e3, 2),
                         peak_inside=bool(s["peak_inside"]),
                         expected_peak_nm=round(expected_peak_nm(
                             s0["peak_wavelength_um"] * 1e3, gc["period_um"], angle0, ang,
                             gc.get("fiber_tilt", "towards"), n_top), 2),
                         bandwidth_1db_nm=round(s["bandwidth_1db_nm"], 1),
                         design_angle=abs(ang - angle0) < 1e-9))
    return rows


def draw_angle_sweep(ctx: Context, gc: dict, rows: list[dict]):
    """Figure 7d: efficiency at the design wavelength and peak efficiency against the fibre
    angle, and the peak wavelength against the grating equation's prediction."""
    from tarang.design import visuals
    plt = visuals._plt()
    a = np.array([r["angle_deg"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(a, [r["efficiency_at_target"] * 100 for r in rows], "o-",
            label=f"at {ctx.cfg.run.wavelength_um * 1e3:.0f} nm")
    ax.plot(a, [r["peak_efficiency"] * 100 for r in rows], "s--", label="peak")
    ax.axvline(float(gc["fiber_angle_deg"]), color="0.5", ls=":", label="design angle")
    ax.set(xlabel="Fibre angle from vertical (°)", ylabel="Coupling efficiency (%)",
           title="Efficiency against the fibre angle (2D FDTD)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    ok = np.array([r["peak_inside"] for r in rows])
    pk = np.array([r["peak_wavelength_nm"] for r in rows])
    ax.plot(a[ok], pk[ok], "o", label="simulated peak")
    if (~ok).any():
        ax.plot(a[~ok], pk[~ok], "o", mfc="none", label="peak at the band edge")
    ax.plot(a, [r["expected_peak_nm"] for r in rows], "-", color="0.4",
            label="grating equation from the design angle")
    ax.axhline(ctx.cfg.run.wavelength_um * 1e3, color="0.6", ls=":")
    ax.set(xlabel="Fibre angle from vertical (°)", ylabel="Peak wavelength (nm)",
           title="Peak wavelength against the fibre angle")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return visuals._save(fig, ctx.path("figures", "7d_grating_angle.png"))


def _power_budget(data, rev, spectra: dict, wl0: float) -> dict:
    """Fractions of the waveguide power (reverse run) against wavelength: up (towards the
    fibre), down (into the substrate), past the grating, reflected into the waveguide;
    with the forward efficiency, the overlap of the upward beam with the fibre mode."""
    f = data["up"].flux.coords["f"].values
    wl = C_UM / f
    o = np.argsort(wl)
    up = np.abs(data["up"].flux.values)[o]
    down = np.abs(data["down"].flux.values)[o]
    past = np.abs(data["past"].flux.values)[o]
    mname = next(m.name for m in rev.monitors if m.name not in ("up", "down", "past", "back"))
    refl = (np.abs(data[mname].amps.sel(direction="-", mode_index=0).values) ** 2).ravel()[o]
    wlo = wl[o]
    eta = np.interp(wlo, *spectra["final"]) if "final" in spectra else None
    back = None
    try:
        back = np.abs(data["back"].flux.values)[o]
    except (KeyError, AttributeError):                # reverse runs from before Phase 12B
        pass
    budget = dict(wavelength_um=wlo.tolist(), up=up.tolist(), down=down.tolist(),
                  past=past.tolist(), reflected=refl.tolist(),
                  coupled=None if eta is None else eta.tolist(),
                  scattered_back=None if back is None else np.clip(back - refl, 0, None).tolist())
    i0 = int(np.argmin(np.abs(wlo - wl0)))
    budget["at_target"] = dict(up=float(up[i0]), down=float(down[i0]), past=float(past[i0]),
                               reflected=float(refl[i0]),
                               directionality=float(up[i0] / max(up[i0] + down[i0], 1e-12)),
                               coupled=None if eta is None else float(eta[i0]),
                               overlap=None if eta is None else float(eta[i0] / max(up[i0], 1e-12)),
                               reflection_db=float(10 * math.log10(max(refl[i0], 1e-12))),
                               scattered_back=None if back is None else float(max(back[i0] - refl[i0], 0.0)),
                               accounted=float(up[i0] + down[i0] + past[i0] +
                                               (back[i0] if back is not None else refl[i0])))
    return budget


def gap_mesh_check(ctx: Context, gap: dict) -> Optional[dict]:
    """The critical gap again at a 1.5x finer mesh."""
    cfg = ctx.cfg
    old = (cfg.gap.fine_steps, cfg.gap.coarse_steps)
    cfg.gap.fine_steps, cfg.gap.coarse_steps = int(round(old[0] * 1.5)), int(round(old[1] * 1.5))
    try:
        d = _gap_device(ctx, round(gap["critical_gap_um"], 4))
    finally:
        cfg.gap.fine_steps, cfg.gap.coarse_steps = old
    folder = ctx.out / "cloud" / "mesh_check"
    if _run_batch(ctx, {"fine": (d.sim, d._readable_name())}, folder,
                  "critical gap at 1.5x finer mesh") is None:
        return None
    data = _load(d.sim, folder, d._readable_name())
    if data is None:
        return None
    try:
        amps = data["Ring_Out_Right"].amps.sel(direction="+", mode_index=0)
    except Exception:                                             # noqa: BLE001
        return None
    f0 = C_UM / cfg.run.wavelength_um
    k2 = float(np.abs(amps.sel(f=f0, method="nearest").values) ** 2)
    return dict(gap_um=round(gap["critical_gap_um"], 4), coupled_fine=k2,
                coupled_fit=gap["target_cross_power"],
                change_percent=100 * (k2 - gap["target_cross_power"]) / gap["target_cross_power"])


# ============================================================================ the stage
def stage_checks(ctx: Context) -> dict:
    from tarang.design import visuals
    _form(ctx)
    cfg, ui = ctx.cfg, ctx.ui
    gap, gc = _done(ctx, "gap"), _done(ctx, "grating")
    out: dict = {}
    ui.say("Free checks (local or from saved results):")
    ring = _done(ctx, "ring")
    cmt = refresh_figures(ctx, gap) if ring else None
    if cmt:
        out["coupled_mode_check"] = cmt
    if gap:
        out["gap_tolerance"] = gap_tolerance(ctx, gap)
        if out["gap_tolerance"]:
            ui.say(f"\nGap tolerance (width ± {cfg.cross_section.tolerance_width_um * 1e3:.0f} nm "
                   "moves the gap by ∓ the same; dip = depth of the through-port resonance, "
                   "unbounded at exactly K = 1):")
            ui.table(out["gap_tolerance"], ["design", "case", "gap_um", "K", "Q_loaded", "dip_depth_db"],
                     {"Q_loaded": ".3g"})
            _write_csv(ctx.path("tables", "gap_tolerance.csv"), out["gap_tolerance"])
    if gap:
        refl = arc_reflection(ctx, gap)
        if refl:
            out["arc_reflection"] = refl
            worst = max(r["arc_end_reflection"] for r in refl)
            ui.say(f"\nReflection at the ends of the partial-ring arc (saved gap runs): "
                   f"{10 * math.log10(max(worst, 1e-12)):.1f} dB at most "
                   f"({worst * 100:.2g} % of the power in the ring comes back); in the bus "
                   f"{10 * math.log10(max(max(r['bus_reflection'] for r in refl), 1e-12)):.1f} dB.")
            ui.say("  The coupled power is the forward-travelling part at the ring port, so the "
                   "reflection only matters through what couples back into the bus (κ² times "
                   "that).")
            _write_csv(ctx.path("tables", "arc_reflection.csv"), refl)
    fsr = fsr_tolerance(ctx) if ring else None
    if fsr:
        out["fsr_tolerance"] = fsr
        ui.say(f"FSR over the cross-section tolerance box: {fsr['fsr_min_ghz']:.1f}-"
               f"{fsr['fsr_max_ghz']:.1f} GHz (n_g {fsr['n_g_min']:.4f}-{fsr['n_g_max']:.4f}).")
    if cfg.checks.mode_families and ring:
        try:
            fam = mode_families(ctx)
        except Exception as err:                                  # noqa: BLE001
            ui.say(f"  (mode families not computed: {type(err).__name__}: {err})")
            fam = []
        if fam:
            out["mode_families"] = fam
            ui.say("\nGuided mode families of the ring waveguide (bent, local solve):")
            ui.table(fam)
            ui.say("  modes_between_crossings: how many TE00 resonances pass between two "
                   "crossings with that family (their resonances drift by FSR_00 - FSR_i per "
                   "mode). Where the crossings fall depends on the exact fabricated geometry.")
            _write_csv(ctx.path("tables", "mode_families.csv"), fam)
    out["loss_budget"] = loss_budget(ctx, gc)
    ui.say("\nFibre-to-fibre loss at the design wavelength, off resonance:")
    ui.table(out["loss_budget"], formats={"loss_db": ".2f"})
    _write_csv(ctx.path("tables", "loss_budget.csv"), out["loss_budget"])

    ui.say("\nCloud checks (each priced first; s skips one, q stops):")
    if gap and cfg.checks.band:
        r = band_check(ctx, gap)
        if r:
            out["band"] = r
            for k, c in r["curves"].items():
                K = np.array(c["K"])
                wl = np.array(c["wavelength_um"])
                ui.say(f"  {k}: K = {K.min():.2f}-{K.max():.2f} over {wl.min() * 1e3:.0f}-"
                       f"{wl.max() * 1e3:.0f} nm")
    if gc and (cfg.checks.grating_extras or cfg.checks.mesh_grating or cfg.checks.angle_sweep_deg):
        r = grating_checks(ctx, gc)
        if r and r["variants"]:
            out["grating"] = r
            ui.table(r["variants"], ["case", "efficiency_at_target", "peak_wavelength_nm",
                                     "peak_shift_nm", "change_db_at_target"],
                     {"efficiency_at_target": ".3f"})
            for row in r.get("angle_sweep") or []:
                ui.say(f"  fibre at {row['angle_deg']:g}°: {row['efficiency_at_target'] * 100:.1f} % "
                       f"at {cfg.run.wavelength_um} µm, peak {row['peak_efficiency'] * 100:.1f} % "
                       f"at {row['peak_wavelength_nm']:.0f} nm (grating equation: "
                       f"{row['expected_peak_nm']:.0f} nm)")
            b = (r.get("power_budget") or {}).get("at_target")
            if b:
                ui.say(f"  At {cfg.run.wavelength_um} µm, light launched from the waveguide: "
                       f"{b['up'] * 100:.0f} % up, {b['down'] * 100:.0f} % into the substrate, "
                       f"{b['past'] * 100:.0f} % past the grating, "
                       + (f"{b['scattered_back'] * 100:.0f} % scattered back, "
                          if b.get("scattered_back") is not None else "")
                       + f"reflection {b['reflection_db']:.1f} dB; "
                       f"directionality {b['directionality'] * 100:.0f} %"
                       + (f", overlap with the fibre {b['overlap'] * 100:.0f} %" if b.get("overlap") else ""))
    if gap and cfg.checks.mesh_gap:
        r = gap_mesh_check(ctx, gap)
        if r:
            out["gap_mesh"] = r
            ui.say(f"  critical gap at 1.5x finer mesh: coupled power {r['coupled_fine']:.4e} "
                   f"vs {r['coupled_fit']:.4e} ({r['change_percent']:+.1f} %)")
    visuals.safe(ctx, "loss budget", draw_loss_budget, ctx, out["loss_budget"])
    return out


# ============================================================================ free redraw


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return {"True": True, "False": False}.get(v, v)


def redraw_saved_figures(ctx: Context, force: bool = False) -> list[str]:
    """Redraw the figures whose drawing improved since this run made them, from its saved
    tables and cached simulation data only (no cloud run, nothing new to pay). Runs once per
    figure version (state 'figure_version'); returns the figures redrawn."""
    from tarang.design import visuals
    from tarang.design.stages import _best_grating
    from tarang.grating_coupler.grating_coupler import spectrum_summary
    if not force and int(ctx.state.get("figure_version", 1)) >= FIGURE_VERSION:
        return []
    st = ctx.state.get("stages", {})
    res = {k: v.get("results") or {} for k, v in st.items()}
    done: list[str] = []
    wl0 = ctx.cfg.run.wavelength_um
    tables = ctx.out / "tables"
    xs, ring = res.get("cross_section"), res.get("ring")
    if xs and ring:
        info = fig_bent_mode(ctx, xs["height_um"], xs["width_um"], ring["radius_um"])
        if info:
            done.append("3b_bent_mode")
    gc = res.get("grating") or {}
    if gc and not gc.get("skipped"):
        rows = [{k: _num(v) for k, v in r.items()} for r in _read_csv(tables / "grating_sweep.csv")]
        stage_a = [r for r in rows if r.get("step") == "A" and r.get("dim") == "2D"]
        if stage_a:
            final = dict(fill_factor=gc["fill_factor"], period_um=gc["period_um"])
            if visuals.safe(ctx, "grating map", visuals.grating_landscape, stage_a,
                            _best_grating(stage_a), ctx.path("figures", "5b_grating_map.png"), wl0,
                            final):
                done.append("5b_grating_map")
        from tarang.design import landscapes
        if visuals.safe(ctx, "grating landscape", landscapes.draw_uniform,
                        ctx.path("figures", "5g_grating_landscape.png"), rows,
                        dict(fill_factor=gc["fill_factor"], period_um=gc["period_um"]), wl0):
            done.append("5g_grating_landscape")
        spec = _read_csv(tables / "grating_spectrum_final.csv")
        if spec:
            wl = np.array([float(r["wavelength_um"]) for r in spec])
            eta = np.array([float(r["efficiency"]) for r in spec])
            summ = spectrum_summary(wl, eta, wl0)
            summ["target_um"] = wl0
            if visuals.safe(ctx, "grating spectrum", visuals.grating_spectrum, wl, eta, summ,
                            ctx.path("figures", "5d_grating_spectrum.png"),
                            f"Final grating: P = {gc['period_um']:.4f} µm, FF = "
                            f"{gc['fill_factor']:.2f}, fibre at {gc['source_x_um']} µm"):
                done.append("5d_grating_spectrum")
    ap = res.get("apodise") or {}
    if ap.get("verified") and (tables / "apodised_check.csv").exists():
        from tarang.design import landscapes
        rows = [{k: _num(v) for k, v in r.items()} for r in _read_csv(tables / "apodised_check.csv")]
        if visuals.safe(ctx, "apodised landscape", landscapes.draw_apodised,
                        ctx.path("figures", "9c_apodised_landscape.png"), rows,
                        (ap.get("source_x_um"), ap.get("scale")), wl0):
            done.append("9c_apodised_landscape")
    band = _read_csv(tables / "coupling_vs_wavelength.csv")
    if band:
        curves: dict = {}
        for r in band:
            c = curves.setdefault(r["design"], dict(gap_um=float(r["gap_um"]), wavelength_um=[],
                                                    coupled=[], K=[]))
            c["wavelength_um"].append(float(r["wavelength_um"]))
            c["coupled"].append(float(r["coupled_power"]))
            c["K"].append(float(r["K"]))
        if visuals.safe(ctx, "coupling against wavelength", draw_band, ctx, curves):
            done.append("7a_coupling_vs_wavelength")
    chk = res.get("checks") or {}
    g = chk.get("grating") or {}
    if g.get("variants") and gc:
        spectra = _grating_check_spectra(ctx, gc, [r["case"] for r in g["variants"]])
        if spectra and visuals.safe(ctx, "grating checks", draw_grating_checks, ctx, spectra,
                                    g.get("power_budget")):
            done.append("7b_grating_checks")
    if g.get("angle_sweep") and gc:
        if visuals.safe(ctx, "grating angle sweep", draw_angle_sweep, ctx, gc, g["angle_sweep"]):
            done.append("7d_grating_angle")
    gap = res.get("gap") or {}
    if gap and not gap.get("skipped") and gap.get("folder"):
        fmap = ctx.path("figures", "4d_gap_field.png")
        before = fmap.stat().st_mtime if fmap.exists() else None
        gap_extras(ctx, gap)                        # field map (cached data only) and check
        if fmap.exists() and fmap.stat().st_mtime != before:
            done.append("4d_gap_field")
    ctx.state["figure_version"] = FIGURE_VERSION
    if done:
        ctx.ui.say("Figures redrawn from saved results (free): " + ", ".join(done))
    return done


def grating_variants(ctx: Context, gc: dict, every: bool = False) -> dict:
    """The final grating and its check variants {case: device}, as the checks stage runs
    them (``every``: all of them whatever the settings)."""
    ch, cfg = ctx.cfg.checks, ctx.cfg
    P, ff, sx = gc["period_um"], gc["fill_factor"], gc["source_x_um"]
    H, depth = ctx.results("cross_section")["height_um"], gc["etch_depth_um"]
    full = gc.get("etch") == "full"
    dH = float(cfg.cross_section.tolerance_height_um)
    b = float(ch.litho_bias_um)
    a = float(ch.angle_tolerance_deg)
    angle = gc["fiber_angle_deg"]
    variants = {"final": grating_device(ctx, P, ff, sx, H=H, depth=depth)}
    if ch.grating_extras or every:
        variants.update({
            f"angle {angle - a:g}°": grating_device(ctx, P, ff, sx, H=H, depth=depth, angle=angle - a),
            f"angle {angle + a:g}°": grating_device(ctx, P, ff, sx, H=H, depth=depth, angle=angle + a),
            f"film −{dH * 1e3:.0f} nm": grating_device(ctx, P, ff, sx, H=H - dH,
                                                      depth=(H - dH) if full else depth),
            f"film +{dH * 1e3:.0f} nm": grating_device(ctx, P, ff, sx, H=H + dH,
                                                      depth=(H + dH) if full else depth),
            f"teeth −{b * 1e3:.0f} nm": grating_device(ctx, P, ff - b / P, sx, H=H, depth=depth),
            f"teeth +{b * 1e3:.0f} nm": grating_device(ctx, P, ff + b / P, sx, H=H, depth=depth)})
    if ch.mesh_grating or every:
        variants["mesh x1.5"] = grating_device(ctx, P, ff, sx, H=H, depth=depth, mesh=1.5)
    return variants


def _grating_check_spectra(ctx: Context, gc: dict, cases: list[str]) -> dict:
    """Spectra of the grating check runs, from their cached data (no new runs)."""
    folder = ctx.out / "cloud" / "grating"
    out = {}
    for case, d in grating_variants(ctx, gc, every=True).items():
        if case not in cases:
            continue
        try:
            if d.is_cached(folder):
                out[case] = d.get_spectrum(folder)
        except Exception:                                         # noqa: BLE001
            continue
    return out
