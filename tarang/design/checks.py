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
  * optionally the critical gap again at a 1.5x finer mesh.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Optional

import numpy as np

from tarang.design import physics
from tarang.design.stages import (C_UM, Context, _estimate, fig_bent_mode, fig_mode_profiles,
                                  gap_extras, grating_device, _gap_device)

LOG10E = 10 * math.log10(math.e)


def _form(ctx: Context) -> None:
    cfg = ctx.cfg
    ctx.ui.form(cfg, "Checks", [
        dict(section="checks", key="band"),
        dict(section="checks", key="grating_extras"),
        dict(section="checks", key="angle_tolerance_deg"),
        dict(section="checks", key="litho_bias_um"),
        dict(section="checks", key="mesh_grating"),
        dict(section="checks", key="mesh_gap"),
        dict(section="checks", key="mode_families"),
        dict(section="run", key="max_cost_per_run"),
        dict(section="run", key="max_total_cost")])


def _done(ctx: Context, name: str) -> Optional[dict]:
    r = ctx.state["stages"].get(name, {}).get("results")
    return None if not r or r.get("skipped") else r


def _write_csv(path: Path, rows: list[dict]) -> Path:
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
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


def loss_budget(ctx: Context, grating: Optional[dict]) -> list[dict]:
    """Fibre-to-fibre transmission off resonance at the design wavelength."""
    cfg, ring = ctx.cfg, ctx.results("ring")
    straight = ring["scattering_loss_db_cm"] + ring["absorption_loss_db_cm"]
    length_cm = (cfg.layout.bus_length_um + 2 * cfg.grating.taper_length_um) * 1e-4
    rows = []
    if grating:
        gc_db = -10 * math.log10(max(grating["efficiency_2d"], 1e-9))
        rows.append(dict(item="grating coupler, input (2D simulation)", loss_db=round(gc_db, 2)))
        rows.append(dict(item="grating coupler, output (same design)", loss_db=round(gc_db, 2)))
    rows.append(dict(item="tapers, 2 x (not simulated yet: Phase 12B)", loss_db=0.0))
    rows.append(dict(item=f"propagation, {length_cm * 1e4:.0f} µm at {straight:g} dB/cm",
                     loss_db=round(straight * length_cm, 3)))
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

    def draw():
        plt = visuals._plt()
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
        for i, (k, c) in enumerate(curves.items()):
            wl = np.array(c["wavelength_um"]) * 1e3
            K = np.array(c["K"])
            col = visuals.PALETTE[1] if k == "critical" else visuals.PALETTE[0]
            axes[0].plot(wl, np.array(c["coupled"]) * 100, "o-", color=col,
                         label=f"{k}: gap {c['gap_um'] * 1e3:.0f} nm")
            axes[1].plot(wl, K, "o-", color=col, label=k)
            dip = ((1 - K) / (1 + K)) ** 2
            axes[2].plot(wl, -10 * np.log10(np.clip(dip, 1e-6, None)), "o-", color=col, label=k)
        axes[0].set(xlabel="Wavelength (nm)", ylabel="Coupled power per pass κ² (%)",
                    title="coupling grows with wavelength (longer evanescent tail)")
        axes[1].axhline(1, color="0.4", ls="--", lw=1)
        axes[1].set(xlabel="Wavelength (nm)", ylabel="K = κ² / round-trip loss",
                    title="coupling regime across the band\n(K > 1 over-coupled, < 1 under-coupled)")
        axes[2].set(xlabel="Wavelength (nm)", ylabel="Resonance dip depth (dB)",
                    title="through-port dip ((1 − K)/(1 + K))²")
        for ax in axes:
            ax.axvline(cfg.run.wavelength_um * 1e3, color="0.6", ls=":", lw=1)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle("Ring-bus coupling against wavelength (loss assumed the same at every "
                     "wavelength)")
        return visuals._save(fig, ctx.path("figures", "7a_coupling_vs_wavelength.png"))
    visuals.safe(ctx, "coupling against wavelength", draw)
    rows = [dict(design=k, gap_um=c["gap_um"], wavelength_um=round(w, 5), coupled_power=p, K=kk)
            for k, c in curves.items() for w, p, kk in zip(c["wavelength_um"], c["coupled"], c["K"])]
    _write_csv(ctx.path("tables", "coupling_vs_wavelength.csv"), rows)
    return dict(curves=curves, recorded_in_gap_runs=recorded)


def _reverse_sim(device, ctx: Context):
    """The grating simulation run backwards: TE0 launched in the waveguide towards the
    grating, with flux planes above (towards the fibre), below (into the substrate), past
    the grating, and the waveguide mode monitor for the reflection."""
    import tidy3d as td
    sim = device.sim
    H = device.config.core_height
    box = ctx.cfg.stack.box_height_um
    mm = next(m for m in sim.monitors if isinstance(m, td.ModeMonitor))
    st = sim.sources[0].source_time
    (x0, _, z0), (x1, _, z1) = sim.bounds
    xg = device.config.period * device.config.n_teeth
    src = td.ModeSource(center=(mm.center[0] - 1.5, 0, mm.center[2]), size=(0, td.inf, mm.size[2]),
                        source_time=st, direction="+", mode_spec=mm.mode_spec, mode_index=0,
                        name="waveguide_source")
    # a closed box of flux planes: top (towards the fibre), bottom (inside the Si handle)
    # and the right side past the grating, so no power is counted twice; what leaves on the
    # left goes back along the waveguide (the mode monitor) or is scattered backwards.
    x_past = min(xg + 1.5, x1 - 0.5)
    z_up, z_down = H + 1.0, -box - 1.0
    fl = dict(freqs=list(mm.freqs))
    mons = [mm,
            td.FluxMonitor(center=((x0 + x_past) / 2, 0, z_up), size=(x_past - x0, td.inf, 0),
                           name="up", **fl),
            td.FluxMonitor(center=((x0 + x_past) / 2, 0, z_down), size=(x_past - x0, td.inf, 0),
                           name="down", normal_dir="-", **fl),
            td.FluxMonitor(center=(x_past, 0, (z_up + z_down) / 2), size=(0, td.inf, z_up - z_down),
                           name="past", **fl)]
    del z0, z1
    return sim.updated_copy(sources=[src], monitors=mons)


def grating_checks(ctx: Context, gc: dict) -> Optional[dict]:
    """Power budget (reverse run), tolerances and mesh check of the final grating."""
    from tarang.grating_coupler.grating_coupler import spectrum_summary
    from tarang.design import visuals
    cfg = ctx.cfg
    ch = cfg.checks
    P, ff, sx = gc["period_um"], gc["fill_factor"], gc["source_x_um"]
    H, depth = ctx.results("cross_section")["height_um"], gc["etch_depth_um"]
    full = gc.get("etch") == "full"
    dH = float(cfg.cross_section.tolerance_height_um)
    b = float(ch.litho_bias_um)
    a = float(ch.angle_tolerance_deg)
    angle = gc["fiber_angle_deg"]
    variants = {"final": grating_device(ctx, P, ff, sx, H=H, depth=depth)}
    if ch.grating_extras:
        variants.update({
            f"angle {angle - a:g}°": grating_device(ctx, P, ff, sx, H=H, depth=depth, angle=angle - a),
            f"angle {angle + a:g}°": grating_device(ctx, P, ff, sx, H=H, depth=depth, angle=angle + a),
            f"film −{dH * 1e3:.0f} nm": grating_device(ctx, P, ff, sx, H=H - dH,
                                                      depth=(H - dH) if full else depth),
            f"film +{dH * 1e3:.0f} nm": grating_device(ctx, P, ff, sx, H=H + dH,
                                                      depth=(H + dH) if full else depth),
            f"teeth −{b * 1e3:.0f} nm": grating_device(ctx, P, ff - b / P, sx, H=H, depth=depth),
            f"teeth +{b * 1e3:.0f} nm": grating_device(ctx, P, ff + b / P, sx, H=H, depth=depth)})
    if ch.mesh_grating:
        variants["mesh x1.5"] = grating_device(ctx, P, ff, sx, H=H, depth=depth, mesh=1.5)
    folder = ctx.out / "cloud" / "grating"
    jobs = {k: (d.sim, d._readable_name()) for k, d in variants.items()}
    rev = None
    if ch.grating_extras:
        rev = _reverse_sim(variants["final"], ctx)
        jobs["reverse"] = (rev, variants["final"]._readable_name() + "_reverse")
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

    def draw():
        if not spectra:
            return None
        plt = visuals._plt()
        ncol = 2 if budget else 1
        fig, axes = plt.subplots(1, ncol, figsize=(7.5 * ncol, 4.6), squeeze=False)
        ax = axes[0, 0]
        for i, (k, (wls, eta)) in enumerate(spectra.items()):
            ax.plot(np.asarray(wls) * 1e3, 10 * np.log10(np.clip(eta, 1e-6, None)),
                    lw=2.4 if k == "final" else 1.3, color="k" if k == "final" else
                    visuals.PALETTE[i % len(visuals.PALETTE)], label=k)
        ax.axvline(wl0 * 1e3, color="0.6", ls=":")
        top = max(10 * math.log10(max(max(e), 1e-6)) for _, e in spectra.values())
        ax.set(xlabel="Wavelength (nm)", ylabel="Coupling efficiency (dB)", ylim=(top - 10, top + 0.5),
               title="Grating tolerances: fibre angle, film thickness, tooth width")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, ncol=2)
        if budget:
            ax = axes[0, 1]
            wl = np.array(budget["wavelength_um"]) * 1e3
            parts = [("up, coupled to the fibre", np.array(budget["coupled"] or budget["up"])),
                     ("up, not matched to the fibre",
                      np.array(budget["up"]) - np.array(budget["coupled"] or budget["up"])),
                     ("down, into the substrate", np.array(budget["down"])),
                     ("past the grating", np.array(budget["past"])),
                     ("reflected into the waveguide", np.array(budget["reflected"]))]
            ax.stackplot(wl, *[np.clip(p, 0, None) for _, p in parts], labels=[n for n, _ in parts],
                         colors=[visuals.PALETTE[i] for i in (2, 5, 1, 4, 3)], alpha=0.85)
            ax.axvline(wl0 * 1e3, color="k", ls=":")
            ax.set(xlabel="Wavelength (nm)", ylabel="Fraction of the waveguide power",
                   ylim=(0, 1.05), title="Where the light goes (launched from the waveguide)")
            ax.legend(fontsize=7.5, loc="upper right")
        return visuals._save(fig, ctx.path("figures", "7b_grating_checks.png"))
    visuals.safe(ctx, "grating checks", draw)
    _write_csv(ctx.path("tables", "grating_checks.csv"), rows)
    return dict(variants=rows, power_budget=budget)


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
    mname = next(m.name for m in rev.monitors if m.name not in ("up", "down", "past"))
    refl = (np.abs(data[mname].amps.sel(direction="-", mode_index=0).values) ** 2).ravel()[o]
    wlo = wl[o]
    eta = np.interp(wlo, *spectra["final"]) if "final" in spectra else None
    budget = dict(wavelength_um=wlo.tolist(), up=up.tolist(), down=down.tolist(),
                  past=past.tolist(), reflected=refl.tolist(),
                  coupled=None if eta is None else eta.tolist())
    i0 = int(np.argmin(np.abs(wlo - wl0)))
    budget["at_target"] = dict(up=float(up[i0]), down=float(down[i0]), past=float(past[i0]),
                               reflected=float(refl[i0]),
                               directionality=float(up[i0] / max(up[i0] + down[i0], 1e-12)),
                               coupled=None if eta is None else float(eta[i0]),
                               overlap=None if eta is None else float(eta[i0] / max(up[i0], 1e-12)),
                               reflection_db=float(10 * math.log10(max(refl[i0], 1e-12))))
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
    cmt = refresh_figures(ctx, gap)
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
    fsr = fsr_tolerance(ctx)
    if fsr:
        out["fsr_tolerance"] = fsr
        ui.say(f"FSR over the cross-section tolerance box: {fsr['fsr_min_ghz']:.1f}-"
               f"{fsr['fsr_max_ghz']:.1f} GHz (n_g {fsr['n_g_min']:.4f}-{fsr['n_g_max']:.4f}).")
    if cfg.checks.mode_families:
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
    if gc and (cfg.checks.grating_extras or cfg.checks.mesh_grating):
        r = grating_checks(ctx, gc)
        if r and r["variants"]:
            out["grating"] = r
            ui.table(r["variants"], ["case", "efficiency_at_target", "peak_wavelength_nm",
                                     "peak_shift_nm", "change_db_at_target"],
                     {"efficiency_at_target": ".3f"})
            b = (r.get("power_budget") or {}).get("at_target")
            if b:
                ui.say(f"  At {cfg.run.wavelength_um} µm, light launched from the waveguide: "
                       f"{b['up'] * 100:.0f} % up, {b['down'] * 100:.0f} % into the substrate, "
                       f"{b['past'] * 100:.0f} % past the grating, reflection {b['reflection_db']:.1f} dB; "
                       f"directionality {b['directionality'] * 100:.0f} %"
                       + (f", overlap with the fibre {b['overlap'] * 100:.0f} %" if b.get("overlap") else ""))
    if gap and cfg.checks.mesh_gap:
        r = gap_mesh_check(ctx, gap)
        if r:
            out["gap_mesh"] = r
            ui.say(f"  critical gap at 1.5x finer mesh: coupled power {r['coupled_fine']:.4e} "
                   f"vs {r['coupled_fit']:.4e} ({r['change_percent']:+.1f} %)")
    lb = out["loss_budget"]

    def draw_budget():
        plt = visuals._plt()
        items = lb[:-1]
        fig, ax = plt.subplots(figsize=(8, 0.55 * len(items) + 1.6))
        left = 0.0
        for i, r in enumerate(items):
            ax.barh(i, r["loss_db"], left=left, color=visuals.PALETTE[i % len(visuals.PALETTE)])
            ax.text(left + r["loss_db"] + 0.05, i, f"{r['loss_db']:.2f} dB", va="center", fontsize=8.5)
            left += r["loss_db"]
        ax.set_yticks(range(len(items)), [r["item"] for r in items])
        ax.invert_yaxis()
        ax.set_xlabel("Loss (dB), accumulated")
        ax.set_title(f"Fibre-to-fibre: {lb[-1]['loss_db']:.2f} dB at {cfg.run.wavelength_um} µm "
                     "(off resonance)")
        ax.set_xlim(0, max(left * 1.25, 1))
        return visuals._save(fig, ctx.path("figures", "7c_loss_budget.png"))
    visuals.safe(ctx, "loss budget", draw_budget)
    return out
