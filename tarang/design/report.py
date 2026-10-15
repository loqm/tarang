"""The design report: one Markdown page, a JSON summary, and a table of values to check
in an independent solver.

``tables/check_values.csv`` lists every number the design rests on, with the settings
needed to recompute it elsewhere (any mode solver or FDTD tool): material tables, the
cross-section, the wavelength and the mesh TARANG used.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path


def _rel(ctx, path) -> str:
    try:
        return Path(os.path.relpath(path, ctx.out)).as_posix()
    except (TypeError, ValueError):
        return str(path)


def write(ctx) -> dict:
    cfg = ctx.cfg
    st = {k: v.get("results", {}) for k, v in ctx.state["stages"].items()}
    mat, xs, ring = st.get("materials", {}), st.get("cross_section", {}), st.get("ring", {})
    gap, gc, lay = st.get("gap", {}), st.get("grating", {}), st.get("layout", {})
    wl = cfg.run.wavelength_um

    L = [f"# Design report: {cfg.run.name}", "",
         f"Wavelength {wl} µm; stack {cfg.materials.core} core on {cfg.stack.box_height_um} µm "
         f"{cfg.materials.box}, {cfg.materials.cladding} top. Settings: `design.toml` "
         f"(rerun with `python -m tarang.design --config design.toml`). Cloud estimate spent: "
         f"{ctx.spent:.3f} FlexCredits.", ""]

    if mat:
        L += ["## 1. Materials", "", "| role | material | data range (µm) | n at λ | covers band |",
              "|---|---|---|---|---|"]
        for r in mat["materials"]:
            L.append(f"| {r['role']} | `{r['material']}` | {r['data_range']} | {r['n_at_wl']} | "
                     f"{r['covers_band']} |")
        L += ["", f"![materials]({_rel(ctx, mat['figure'])})", "",
              f"Fitted n, k over the band: `{_rel(ctx, mat['table'])}`.", ""]
    if xs:
        mesh_change = ("-" if xs.get("mesh_change") is None
                       else f"{xs['mesh_change'] * 100:.1f} %")
        L += ["## 2. Cross-section", "",
              f"![design map]({_rel(ctx, xs['map_figure'])})", "",
              f"Chosen: **{xs['height_um'] * 1e3:.0f} nm × {xs['width_um'] * 1e3:.0f} nm** "
              f"({xs['steps']} mesh steps/λ).", "",
              "| quantity | value |", "|---|---|",
              f"| D (TE00) | {xs['D']:.1f} ps/(nm·km) |",
              f"| worst-case D within tolerance | {xs['D_min_tol']:.1f} ps/(nm·km) |",
              f"| n_eff / n_g (TE00) | {xs['n_eff']:.5f} / {xs['n_group']:.4f} |",
              f"| guided TE / TM modes | {xs['guided_TE']} / {xs['guided_TM']} |",
              f"| whole tolerance box anomalous + single-mode | "
              f"{'yes' if xs['tolerance_box_target'] else 'no'} |",
              f"| last mesh refinement changed D by | {mesh_change} |",
              "", f"Finalist report: `{_rel(ctx, xs['report'])}`.", ""]
    if ring:
        L += ["## 3. Ring", "", "| quantity | value |", "|---|---|",
              f"| radius | {ring['radius_um']} µm |",
              f"| FSR | {ring['fsr_ghz']:.1f} GHz |",
              f"| loss (scattering + absorption + bend) | {ring['loss_db_cm']:.3g} dB/cm "
              f"({ring['scattering_loss_db_cm']} + {ring['absorption_loss_db_cm']} + "
              f"{ring['bend_loss_db_cm']:.2g}) |",
              f"| Q_int | {ring['q_intrinsic']:.3g} |",
              f"| D2/2π, D3/2π | {ring.get('D2_2pi_MHz')} MHz, {ring.get('D3_2pi_kHz')} kHz |",
              f"| zero-dispersion wavelengths | {ring.get('zdw_nm')} |", "",
              f"![dispersion profile]({_rel(ctx, ring['profile_figure'])})", ""]
    if gap and not gap.get("skipped"):
        L += ["## 4. Ring–bus gap", "", "| regime | gap | loaded Q |", "|---|---|---|",
              f"| critical | {gap['critical_gap_um']:.3f} µm | {gap['q_loaded_critical']:.3g} |"]
        if gap.get("over_coupled_gap_um"):
            L.append(f"| over-coupled, K = {gap['over_coupling']:g} | "
                     f"{gap['over_coupled_gap_um']:.3f} µm | {gap['q_loaded_over']:.3g} |")
        L += ["", f"Simulation data and fit plots: `{_rel(ctx, gap['folder'])}`." +
              (" The critical gap is extrapolated beyond the simulated gaps." if
               gap.get("extrapolated") else ""), ""]
    elif gap:
        L += ["## 4. Ring–bus gap", "", "Skipped (not confirmed or over budget).", ""]
    if gc and not gc.get("skipped"):
        eff3d = "-" if gc.get("efficiency_3d") is None else f"{gc['efficiency_3d']:.3f}"
        L += ["## 5. Grating coupler", "", "| quantity | value |", "|---|---|",
              f"| etch | {gc['etch']} ({gc['etch_depth_um']} µm) |",
              f"| period / fill factor / teeth | {gc['period_um']} µm / {gc['fill_factor']} / "
              f"{gc['n_teeth']} |",
              f"| fibre angle / MFD / position | {gc['fiber_angle_deg']}° / {gc['mfd_um']} µm / "
              f"{gc['source_x_um']} µm |",
              f"| 2D efficiency | {gc['efficiency_2d']:.3f} ({gc['loss_db_2d']:.2f} dB) |",
              f"| 3D efficiency | {eff3d} |",
              "", f"All simulated designs: `{_rel(ctx, gc['table'])}`.", ""]
    elif gc:
        L += ["## 5. Grating coupler", "", "Skipped (not confirmed or over budget).", ""]
    if lay:
        L += ["## 6. Layout", ""] + [f"- `{_rel(ctx, f)}`" for f in lay.get("files", [])] + [""]

    checks = _check_values(cfg, mat, xs, ring, gap, gc)
    table = ctx.path("tables", "check_values.csv")
    with open(table, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["quantity", "value", "unit", "how TARANG computed it"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(checks)
    L += ["## Checking these numbers elsewhere", "",
          f"`{_rel(ctx, table)}` lists each value with how it was computed; the fitted "
          f"material indices are in `tables/materials_nk.csv` and every dimension is in "
          f"`layout/parameters.json`.", ""]

    md = ctx.path("report.md")
    md.write_text("\n".join(L), encoding="utf-8")
    summary = ctx.path("design_summary.json")
    summary.write_text(json.dumps(st, indent=1, default=str), encoding="utf-8")
    ctx.ui.say(f"Report: {md}")
    return dict(report=str(md), summary=str(summary), check_values=str(table))


def _check_values(cfg, mat, xs, ring, gap, gc) -> list[dict]:
    wl = cfg.run.wavelength_um
    rows = []

    def add(q, v, unit, how):
        if v is not None:
            rows.append({"quantity": q, "value": v, "unit": unit, "how TARANG computed it": how})
    for r in (mat or {}).get("materials", []):
        add(f"n of {r['role']} ({r['material']}) at {wl} um", r["n_at_wl"], "",
            "pole-residue fit of the tabulated data over the band")
    if xs:
        how = (f"local mode solver, {xs['height_um']} x {xs['width_um']} um core, "
               f"{cfg.stack.box_height_um} um box, {xs['steps']} steps/lambda")
        add("n_eff TE00", xs["n_eff"], "", how)
        add("n_g TE00", xs["n_group"], "", how + "; polynomial fit of n_eff(lambda)")
        add("D TE00", xs["D"], "ps/(nm km)", how + "; second derivative of the same fit")
        add("A_eff TE00", xs.get("aeff_um2"), "um^2", how)
        add("guided TE modes", xs["guided_TE"], "", "modes with n_eff above the cladding index")
    if ring:
        add("ring radius", ring["radius_um"], "um", "from the FSR target and n_g")
        add("FSR", round(ring["fsr_ghz"], 3), "GHz", "c / (n_g 2 pi R)")
        add("Q_int", round(ring["q_intrinsic"], 0), "", "2 pi n_g / (lambda alpha), alpha from "
            "the losses in the report")
        add("bend loss TE00", ring["bend_loss_db_cm"], "dB/cm", "local bent-mode solve with PML")
    if gap and not gap.get("skipped"):
        add("critical gap", round(gap["critical_gap_um"], 4), "um",
            "3D FDTD of a partial ring next to the bus at several gaps; exponential fit of the "
            "coupling angle; matched to the round-trip loss from Q_int")
    if gc and not gc.get("skipped"):
        add("grating efficiency (2D)", round(gc["efficiency_2d"], 4), "",
            f"2D FDTD, Gaussian fibre beam (MFD {gc['mfd_um']} um, {gc['fiber_angle_deg']} deg), "
            "power in the waveguide mode")
    return rows
