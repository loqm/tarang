"""Sidewall-angle sweep (checks stage, v0.1.3): how the design moves when the etched walls
are not vertical.

From first principles
---------------------
Every design stage assumes vertical walls (90°). A dry etch leaves walls at an angle
alpha (published SiN processes report roughly 85-93°): over the film thickness H each wall
moves sideways by s = H / tan(alpha), so with the top edge on the mask (sidewall reference
'top') the core is w + 2 s wide at the bottom. With H = 0.35 um, 85° gives s = 31 nm; with
H = 0.7 um, 61 nm. That is more core (and, between two cores, a narrower gap at the
bottom), so

* n_eff and n_g rise (alpha < 90°) or fall (> 90°): the FSR = c / (2 pi R n_g) moves,
  by -FSR dn_g / n_g;
* the dispersion D (and D2 = -(c / n_g) D1^2 beta2) moves with the extra width;
* the ring-bus coupling grows (alpha < 90°: the cores come closer at the bottom, where the
  field is strongest for an air-clad core on oxide), so a gap designed as critical turns
  over-coupled;
* grating teeth become trapezoids: the effective fill factor and the scattering strength
  change, so the efficiency and the peak wavelength move.

What is computed
----------------
Free (local mode solves, ``checks.sidewall_sweep_deg``):

* the chosen cross-section at each angle (same mesh, grid snapping and stack as the
  cross-section stage, so the 90° row repeats its numbers): n_eff, n_g, D, beta2 of the
  pump mode and the number of guided modes;
* the ring: FSR from n_g at the ring radius, as the ring stage computes it, and D2 scaled
  from the ring stage's D2 by the ratio of the straight-waveguide estimates
  D2 ~ -(c / n_g) (2 pi FSR)^2 beta2 at the angle and at 90°;
* the coupling: the even/odd supermodes of bus and ring (coupled-mode theory) at the
  critical and over-coupled gaps (grid lines through every core edge, so the gap is
  resolved exactly whatever the angle) give theta_CMT(alpha, g) = theta_0(alpha) exp(-gamma(alpha) g);
  the 3D FDTD fit theta_FDTD(g) stays the anchor and is scaled by the ratio
  theta_CMT(alpha, g) / theta_CMT(90°, g) (the coupled-mode theory's own offset from FDTD
  cancels in the ratio). K = sin^2(theta) / (1 - a^2) at the designed gaps and the gap that
  would be critical at that angle follow.

Cloud (``checks.sidewall_grating_deg``, one small 2D run per angle, priced first): the final
grating with trapezoidal teeth.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from tarang.geometries.sidewall import SNAP_WALLS

C_M_S = 299792458.0


def angles(values, include_vertical: bool = True) -> list[float]:
    out = sorted({round(float(a), 4) for a in (values or [])})
    if include_vertical and out and 90.0 not in out:
        out = sorted(out + [90.0])
    for a in out:
        if not 45.0 < a < 135.0:
            raise ValueError(f"sidewall angle {a}° outside 45-135° (90° = vertical)")
    return out


def d2_estimate_mhz(fsr_ghz: float, n_g: float, beta2_ps2_km: float) -> float:
    """D2 / 2 pi (MHz) of a ring from its FSR and the waveguide's n_g and beta2:
    D2 = -(c / n_g) D1^2 beta2 with D1 = 2 pi FSR (beta2 in ps^2/km = 1e-27 s^2/m)."""
    d1 = 2 * math.pi * fsr_ghz * 1e9
    return -(C_M_S / n_g) * d1 ** 2 * beta2_ps2_km * 1e-27 / (2 * math.pi) / 1e6


# ------------------------------------------------------------------ free: cross-section
def cross_section_sweep(ctx, values) -> list[dict]:
    """One row per sidewall angle: the chosen core's pump-mode numbers and the ring's FSR
    and D2 (see the module notes)."""
    from tarang.design import physics
    from tarang.design.stages import _read_jsonl, pump_fields
    from tarang.geometries import sidewall as sw
    cfg = ctx.cfg
    xs = ctx.results("cross_section")
    H, W, steps = xs["height_um"], xs["width_um"], int(xs["steps"])
    ref = sw.check_reference(cfg.checks.sidewall_reference)
    angs = angles(values)
    if not angs:
        return []
    f = ctx.path("results", f"sidewall_{steps}steps.jsonl")
    ctx.tool("sweep", "--points", f"{H}x{W}", "--wl", cfg.run.wavelength_um, "--steps", steps,
             "--sidewalls", ",".join(f"{a:g}" for a in angs), "--sidewall-ref", ref,
             "--workers", cfg.run.workers, "--out", f.relative_to(ctx.out), *ctx.stack_args())
    snap = bool(cfg.cross_section.snap_grid)
    F = pump_fields(cfg)
    got: dict = {}
    for r in _read_jsonl(f):
        if (abs(r["H"] - H) < 1e-6 and abs(r["W"] - W) < 1e-6 and r["steps"] == steps
                and r.get("bend_radius") is None and bool(r.get("snap", False)) == snap):
            a = round(float(r.get("sidewall_deg", 90.0)), 4)
            if a == 90.0 or (r.get("sidewall_ref", "top") == ref
                             and (not snap or r.get("snap_walls") == SNAP_WALLS)):
                got[a] = r
    ring = (ctx.state["stages"].get("ring") or {}).get("results") or {}
    R = ring.get("radius_um")
    rows = []
    for a in angs:
        r = got.get(a)
        if r is None:
            continue
        wb, wt = sw.core_widths(W, H, a, ref)
        ng = float(r.get(F["ng"] + "_fine") or r[F["ng"]])
        row = dict(sidewall_deg=a, width_bottom_um=round(wb, 4), width_top_um=round(wt, 4),
                   n_eff=float((r.get(F["neff"]) or [float("nan")])[0]), n_g=round(ng, 5),
                   D=float(r[F["D"]]), beta2_ps2_km=float(r[F["beta2"]]),
                   guided=int(r.get(F["guided"]) or 0))
        if R:
            row["fsr_ghz"] = round(physics.fsr_ghz(R, ng), 3)
            row["d2_estimate_mhz"] = round(d2_estimate_mhz(row["fsr_ghz"], ng, row["beta2_ps2_km"]), 4)
        rows.append(row)
    base = next((r for r in rows if r["sidewall_deg"] == 90.0), None)
    for r in rows:
        if base is None:
            break
        r["dD"] = round(r["D"] - base["D"], 2)
        if "fsr_ghz" in r:
            r["dfsr_ghz"] = round(r["fsr_ghz"] - base["fsr_ghz"], 3)
            if ring.get("D2_2pi_MHz") and base.get("d2_estimate_mhz"):
                r["D2_2pi_MHz"] = round(float(ring["D2_2pi_MHz"]) * r["d2_estimate_mhz"]
                                        / base["d2_estimate_mhz"], 3)
    return rows


# ------------------------------------------------------------------ free: coupling
def coupling_sweep(ctx, values) -> dict:
    """K at the designed gaps and the critical gap at each sidewall angle (coupled-mode
    ratio on the FDTD anchor; see the module notes). {} when the gap stage has no fit."""
    from tarang.design import localmodes
    from tarang.geometries import sidewall as sw
    gap = (ctx.state["stages"].get("gap") or {}).get("results") or {}
    ring = (ctx.state["stages"].get("ring") or {}).get("results") or {}
    fit = gap.get("fit") or {}
    if not fit or not ring.get("radius_um") or gap.get("skipped"):
        return {}
    cfg = ctx.cfg
    xs = ctx.results("cross_section")
    H, W, R = xs["height_um"], xs["width_um"], ring["radius_um"]
    ref = sw.check_reference(cfg.checks.sidewall_reference)
    g_c = float(gap["critical_gap_um"])
    g_o = gap.get("over_coupled_gap_um")
    # Three gaps spread over +/-0.1 um (at most +/-40 % of the critical gap): the decay rate
    # comes from a fit over a 0.2 um span. Before v0.1.5 the two design gaps (38 nm apart on
    # the 1550 nm design) were used, so a 2 % error in one supermode split moved gamma by
    # 0.6 /um and put 90 deg off the trend (6.17 against 5.57 and 5.52 /um); over the wide
    # span the same error moves it by about 0.1 /um (5.57, 5.54, 5.50 /um at 89-91 deg).
    spread = round(min(0.1, 0.4 * g_c), 4)
    gaps = [round(g_c - spread, 4), round(g_c, 4), round(g_c + spread, 4)]
    rho = float(gap["target_cross_power"])
    angs = angles(values)
    import json
    steps = int(cfg.cross_section.decision_steps)
    store = ctx.path("results", "sidewall_coupling.jsonl")
    saved: dict = {}
    if store.exists():                       # angles solved before (e.g. by a price-only run)
        for line in store.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if (abs(r["H"] - H) < 1e-9 and abs(r["W"] - W) < 1e-9 and abs(r["R"] - R) < 1e-9
                    and r["gaps"] == gaps and r["ref"] == ref and r["steps"] == steps
                    and r.get("snap") and (abs(float(r["angle"]) - 90.0) < 1e-9
                                           or r.get("snap_walls") == SNAP_WALLS)):
                saved[round(float(r["angle"]), 4)] = r
    cmt: dict = {}
    for a in angs:
        if a in saved:
            cmt[a] = dict(theta0=saved[a]["theta0"], gamma=saved[a]["gamma"], dn=saved[a]["dn"])
            continue
        if min(sw.bottom_gap(g, H, a, ref) for g in gaps) <= 0.0:
            ctx.ui.say(f"  sidewall {a:g}°: the cores touch at the bottom at gap {min(gaps)} µm; "
                       "skipped")
            continue
        try:
            c = localmodes.coupling_from_supermodes(ctx, H, W, R, gaps, sidewall_deg=a,
                                                    sidewall_ref=ref, snap=True)
        except Exception as err:                                   # noqa: BLE001
            ctx.ui.say(f"  sidewall {a:g}°: coupled-mode solve failed ({type(err).__name__}: {err})")
            continue
        cmt[a] = dict(theta0=float(c["theta0"]), gamma=float(c["gamma_per_um"]),
                      dn=[float(r["dn"]) for r in c["rows"]])
        with open(store, "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(H=H, W=W, R=R, gaps=gaps, ref=ref, steps=steps, angle=a,
                                    snap=True, snap_walls=SNAP_WALLS, **cmt[a])) + "\n")
        ctx.ui.say(f"  sidewall {a:g}°: coupled-mode γ = {c['gamma_per_um']:.2f} /µm, "
                   f"Δn = " + ", ".join(f"{r['dn']:.3e}" for r in c["rows"]))
    if 90.0 not in cmt:
        return {}
    b = cmt[90.0]
    th_t = math.asin(math.sqrt(min(rho, 1.0)))

    def theta(a, g):
        c = cmt[a]
        ratio = (c["theta0"] * math.exp(-c["gamma"] * g)) / (b["theta0"] * math.exp(-b["gamma"] * g))
        return fit["theta_0"] * math.exp(-fit["gamma"] * g) * ratio

    rows = []
    designs = [("critical", g_c)] + ([("over-coupled", float(g_o))] if g_o else [])
    for a in sorted(cmt):
        c = cmt[a]
        row = dict(sidewall_deg=a, gamma_cmt_per_um=round(c["gamma"], 3))
        for name, g in designs:
            k2 = math.sin(min(theta(a, g), math.pi / 2)) ** 2
            key = "K_critical_gap" if name == "critical" else "K_over_coupled_gap"
            row[key] = round(k2 / rho, 3)
            if name == "critical":
                row["bottom_gap_um"] = round(sw.bottom_gap(g, H, a, ref), 4)
        # gap that is critical at this angle: ln theta is linear in g
        lnA = (math.log(fit["theta_0"]) + math.log(c["theta0"]) - math.log(b["theta0"]))
        slope = fit["gamma"] + c["gamma"] - b["gamma"]
        row["critical_gap_um"] = round((lnA - math.log(th_t)) / slope, 4) if slope > 0 else None
        rows.append(row)
    return dict(rows=rows, gaps_um=gaps, reference=ref, rho=rho)


# ------------------------------------------------------------------ cloud: grating
def grating_devices(ctx, gc: dict) -> dict:
    """{angle: device} of the final grating with sloped teeth at each angle of
    ``checks.sidewall_grating_deg`` other than 90° (the final design's own run)."""
    from tarang.design.stages import grating_device
    H = ctx.results("cross_section")["height_um"]
    return {a: grating_device(ctx, gc["period_um"], gc["fill_factor"], gc["source_x_um"], H=H,
                              depth=gc["etch_depth_um"], sidewall_deg=a)
            for a in angles(ctx.cfg.checks.sidewall_grating_deg, include_vertical=False)
            if a != 90.0}


def grating_rows(ctx, devices: dict, final_spectrum) -> list[dict]:
    from tarang.grating_coupler.grating_coupler import spectrum_summary
    if not devices or final_spectrum is None:
        return []
    folder = ctx.out / "cloud" / "grating"
    wl0 = ctx.cfg.run.wavelength_um
    spectra = {90.0: final_spectrum}
    for a, d in devices.items():
        try:
            spectra[a] = d.get_spectrum(folder)
        except Exception:                                          # noqa: BLE001
            continue
    s0 = spectrum_summary(*final_spectrum, wl0)
    rows = []
    for a in sorted(spectra):
        s = spectrum_summary(*spectra[a], wl0)
        rows.append(dict(sidewall_deg=a, efficiency_at_target=s["efficiency_at_target"],
                         change_db_at_target=round(10 * math.log10(
                             max(s["efficiency_at_target"], 1e-9) / max(s0["efficiency_at_target"], 1e-9)), 2),
                         peak_efficiency=s["peak_efficiency"],
                         peak_wavelength_nm=round(s["peak_wavelength_um"] * 1e3, 1),
                         peak_shift_nm=round((s["peak_wavelength_um"] - s0["peak_wavelength_um"]) * 1e3, 1),
                         peak_inside=bool(s["peak_inside"]),
                         bandwidth_1db_nm=round(s["bandwidth_1db_nm"], 1)))
    return rows


# ------------------------------------------------------------------ figure and report
def draw(ctx, res: dict):
    """Figure 7e: the design against the sidewall angle."""
    from tarang.design import visuals
    plt = visuals._plt()
    xs_rows = res.get("cross_section") or []
    cp = (res.get("coupling") or {}).get("rows") or []
    gr = res.get("grating") or []
    panels = [p for p, ok in (("D", xs_rows), ("ring", [r for r in xs_rows if "fsr_ghz" in r]),
                              ("K", cp), ("grating", gr)) if ok]
    if not panels:
        return None
    n = len(panels)
    ncol = 2 if n > 1 else 1
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 3.8 * nrow), squeeze=False)
    flat = list(axes.ravel())
    for ax in flat[n:]:
        ax.set_visible(False)
    xlab = f"Sidewall angle (°; 90 = vertical, widths at the {ctx.cfg.checks.sidewall_reference})"
    for ax, p in zip(flat, panels):
        if p == "D":
            a = [r["sidewall_deg"] for r in xs_rows]
            ax.plot(a, [r["D"] for r in xs_rows], "o-", label="D")
            ax.axhline(0, color="0.6", lw=0.8)
            ax.set(xlabel=xlab, ylabel="D (ps/(nm·km))", title="Dispersion")
            tw = ax.twinx()
            tw.plot(a, [r["n_g"] for r in xs_rows], "s--", color="C1", label="n_g")
            tw.set_ylabel("Group index n_g")
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = tw.get_legend_handles_labels()
            tw.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")
        elif p == "ring":
            rr = [r for r in xs_rows if "fsr_ghz" in r]
            a = [r["sidewall_deg"] for r in rr]
            ax.plot(a, [r.get("dfsr_ghz", 0.0) for r in rr], "o-", label="FSR change")
            ax.set(xlabel=xlab, ylabel="FSR − FSR(90°) (GHz)", title="Ring")
            if any("D2_2pi_MHz" in r for r in rr):
                tw = ax.twinx()
                tw.plot(a, [r.get("D2_2pi_MHz") for r in rr], "s--", color="C1", label="D₂/2π")
                tw.set_ylabel("D₂/2π (MHz)")
                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = tw.get_legend_handles_labels()
                tw.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")
        elif p == "K":
            a = [r["sidewall_deg"] for r in cp]
            ax.plot(a, [r["K_critical_gap"] for r in cp], "o-", label="critical gap")
            if any("K_over_coupled_gap" in r for r in cp):
                ax.plot(a, [r.get("K_over_coupled_gap") for r in cp], "s-", label="over-coupled gap")
            ax.axhline(1.0, color="0.6", ls=":", lw=1)
            ax.set_yscale("log")
            ax.set(xlabel=xlab, ylabel="Coupling ratio K", title="Ring–bus coupling")
            ax.legend(fontsize=8)
        else:
            a = [r["sidewall_deg"] for r in gr]
            ax.plot(a, [r["efficiency_at_target"] * 100 for r in gr], "o-",
                    label=f"at {ctx.cfg.run.wavelength_um * 1e3:.0f} nm")
            ax.plot(a, [r["peak_efficiency"] * 100 for r in gr], "s--", label="peak")
            ax.set(xlabel=xlab, ylabel="Coupling efficiency (%)", title="Grating (2D FDTD)")
            tw = ax.twinx()
            tw.plot(a, [r["peak_wavelength_nm"] for r in gr], "^:", color="C2", label="peak λ")
            tw.set_ylabel("Peak wavelength (nm)")
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = tw.get_legend_handles_labels()
            tw.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")
        ax.axvline(90.0, color="0.75", lw=0.8, ls="--")
        ax.grid(alpha=0.3)
    fig.suptitle("Design against the etched sidewall angle")
    fig.tight_layout()
    return visuals._save(fig, ctx.path("figures", "7e_sidewall.png"))


def say(ctx, res: dict) -> None:
    ui = ctx.ui
    if res.get("cross_section"):
        ui.say("\nSidewall angle (local mode solves; widths at the "
               f"{ctx.cfg.checks.sidewall_reference}):")
        ui.table(res["cross_section"], [c for c in ("sidewall_deg", "width_bottom_um", "n_g", "D",
                                                    "dD", "fsr_ghz", "dfsr_ghz", "D2_2pi_MHz",
                                                    "guided") if c in res["cross_section"][0]])
    cp = (res.get("coupling") or {}).get("rows")
    if cp:
        ui.say("  Ring–bus coupling (FDTD fit scaled by coupled-mode theory):")
        ui.table(cp, [c for c in ("sidewall_deg", "bottom_gap_um", "K_critical_gap",
                                  "K_over_coupled_gap", "critical_gap_um") if c in cp[0]])
    if res.get("grating"):
        ui.say("  Grating with sloped teeth (2D FDTD):")
        ui.table(res["grating"], ["sidewall_deg", "efficiency_at_target", "change_db_at_target",
                                  "peak_wavelength_nm", "peak_shift_nm"],
                 {"efficiency_at_target": ".3f"})


def report_lines(ctx, res: dict) -> list[str]:
    from tarang.design.report import _fig, _md_table
    if not res:
        return []
    ref = ctx.cfg.checks.sidewall_reference
    L = [f"**Sidewall angle.** Every design stage uses vertical walls; here the walls slope "
         f"(90° = vertical, below 90° the core widens towards the bottom), the designed widths, "
         f"gaps and tooth lengths holding at the {ref}. Each wall moves sideways by "
         "H / tan(angle) over the film thickness H.", ""]
    if res.get("cross_section"):
        L += ["Cross-section and ring (local mode solves at the cross-section stage's mesh; the "
              "90° row repeats its numbers; FSR from n_g at the ring radius; D₂ scaled from the "
              "ring stage's value by the straight-waveguide estimate −(c/n_g)(2π FSR)²β₂):", ""]
        cols = [c for c in ("sidewall_deg", "width_bottom_um", "n_g", "D", "dD", "fsr_ghz",
                            "dfsr_ghz", "D2_2pi_MHz", "guided") if c in res["cross_section"][0]]
        L += _md_table(res["cross_section"], cols)
    cp = res.get("coupling") or {}
    if cp.get("rows"):
        L += ["Ring–bus coupling: the 3D FDTD fit of the gap stage scaled by the coupled-mode "
              "ratio θ(angle, g)/θ(90°, g) (even/odd supermodes at gaps "
              + ", ".join(f"{g * 1e3:.0f}" for g in cp["gaps_um"])
              + " nm), so the coupled-mode theory's own offset from FDTD cancels. K = 1 is "
              "critical; critical_gap_um is the mask gap that would be critical at that angle.", ""]
        L += _md_table(cp["rows"], [c for c in ("sidewall_deg", "bottom_gap_um", "K_critical_gap",
                                                "K_over_coupled_gap", "critical_gap_um",
                                                "gamma_cmt_per_um") if c in cp["rows"][0]])
    if res.get("grating"):
        L += ["Grating with trapezoidal teeth (2D FDTD, final design otherwise unchanged):", ""]
        L += _md_table(res["grating"], ["sidewall_deg", "efficiency_at_target",
                                        "change_db_at_target", "peak_efficiency",
                                        "peak_wavelength_nm", "peak_shift_nm", "bandwidth_1db_nm"],
                       {"efficiency_at_target": ".3f", "peak_efficiency": ".3f"})
    L += _fig(ctx, ctx.out / "figures" / "7e_sidewall.png", "sidewall angle")
    return L
