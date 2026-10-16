"""The design report: one Markdown page with every figure, the assumed values with their
sources, how each number was obtained, a JSON summary, a table of values to check in an
independent solver, and the ``benchmark_pack/`` folder (see ``benchmark.py``).
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path


def _rel(ctx, path) -> str:
    try:
        return Path(os.path.relpath(path, ctx.out)).as_posix()
    except (TypeError, ValueError):
        return str(path)


def _fig(ctx, path, alt) -> list[str]:
    if path and Path(path).exists():
        return [f"![{alt}]({_rel(ctx, path)})", ""]
    return []


def _enrich_cross_section(ctx, xs: dict) -> dict:
    """Fill values that runs before v0.1.0's report did not store (mesh ladder, TE/TM
    split) from the saved result files."""
    import json as _json
    xs = dict(xs)
    H, W = xs.get("height_um"), xs.get("width_um")

    def rows(name):
        f = ctx.out / "results" / name
        if not f.exists():
            return []
        out = []
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                r = _json.loads(line)
            except ValueError:
                continue
            if "error" not in r and abs(r.get("H", -1) - H) < 1e-6 and abs(r.get("W", -1) - W) < 1e-6 \
                    and r.get("bend_radius") is None:
                out.append(r)
        return out
    if not xs.get("convergence"):
        conv = sorted(rows("convergence.jsonl"), key=lambda r: r["steps"])
        if conv:
            xs["convergence"] = [dict(steps=r["steps"], D=r["D_TE0"]) for r in conv]
    if xs.get("te_tm_split") is None:
        fin = [r for r in rows(f"finalist_{xs.get('steps')}steps.jsonl") if r.get("neff_TM")]
        if fin:
            xs["te_tm_split"] = abs(fin[-1]["neff_TE"][0] - fin[-1]["neff_TM"][0])
    return xs


def threshold_mw(xs: dict, ring: dict, n2: float, K: float = 1.0) -> float | None:
    """Comparative parametric-oscillation threshold (mW) of the ring at coupling ratio K
    (1 = critical): 1.54 (pi/2) (Q_c / 2Q_L) n_eff^2 V / (n2 lambda Q_L^2), V = A_eff 2 pi R."""
    try:
        qi, aeff, n = ring["q_intrinsic"], xs["aeff_um2"], xs["n_eff"]
        lam = ring.get("wavelength_um", 1.55)
    except (KeyError, TypeError):
        return None
    if not (qi and aeff and n):
        return None
    ql, qc = qi / (1 + K), qi / K
    V = aeff * 1e-12 * 2 * math.pi * ring["radius_um"] * 1e-6
    return 1e3 * 1.54 * (math.pi / 2) * (qc / (2 * ql)) * n ** 2 * V / (n2 * lam * 1e-6 * ql ** 2)


def write(ctx) -> dict:
    from tarang.design import assumptions, benchmark, visuals
    cfg = ctx.cfg
    st = {k: v.get("results", {}) for k, v in ctx.state["stages"].items()}
    mat, xs, ring = st.get("materials", {}), st.get("cross_section", {}), st.get("ring", {})
    gap, gc, lay = st.get("gap", {}), st.get("grating", {}), st.get("layout", {})
    wl = cfg.run.wavelength_um
    if ring:
        ring = dict(ring, wavelength_um=wl)
    if xs:
        xs = _enrich_cross_section(ctx, xs)
        ctx.state["stages"]["cross_section"]["results"] = xs
        st["cross_section"] = xs
    if xs and not (xs.get("figure") and Path(xs["figure"]).exists()):
        fig = visuals.safe(ctx, "cross-section", visuals.cross_section, ctx, xs["height_um"],
                           xs["width_um"], ctx.path("figures", "2_cross_section.png"))
        if fig:
            xs = dict(xs, figure=str(fig))
    n2 = cfg.assumptions.n2_m2_w
    pth_c = threshold_mw(xs, ring, n2, 1.0) if xs and ring else None
    K = gap.get("over_coupling") if gap and not gap.get("skipped") else None
    pth_o = threshold_mw(xs, ring, n2, K) if K and K != 1 and pth_c else None
    gap_ok = gap and not gap.get("skipped")
    gc_ok = gc and not gc.get("skipped")

    L = [f"# Design report: {cfg.run.name}", "",
         f"Wavelength {wl} µm; {cfg.materials.core} core on {cfg.stack.box_height_um} µm "
         f"{cfg.materials.box} (Si handle below), {cfg.materials.cladding} top. Settings: "
         f"`design.toml` (repeat with `python -m tarang.design --config design.toml`). Cloud "
         f"estimate spent: {ctx.spent:.3f} FlexCredits. Everything needed to check this design "
         f"in another solver is in `benchmark_pack/` (start with its README).", ""]

    # ---- summary
    L += ["## Summary", "", "| quantity | value |", "|---|---|"]
    if xs:
        L.append(f"| core cross-section | {xs['height_um'] * 1e3:.0f} nm × {xs['width_um'] * 1e3:.0f} nm |")
        L.append(f"| D, n_g (TE00, {wl} µm) | {xs['D']:.1f} ps/(nm·km), {xs['n_group']:.4f} |")
    if ring:
        L.append(f"| ring radius, FSR | {ring['radius_um']} µm, {ring['fsr_ghz']:.1f} GHz |")
        L.append(f"| intrinsic Q (assumed losses) | {ring['q_intrinsic']:.3g} |")
        L.append(f"| D2/2π | {ring.get('D2_2pi_MHz')} MHz |")
    if gap_ok:
        L.append(f"| ring–bus gap: critical / over-coupled (K = {gap['over_coupling']:g}) | "
                 f"{gap['critical_gap_um'] * 1e3:.0f} nm / "
                 f"{(gap.get('over_coupled_gap_um') or 0) * 1e3:.0f} nm |")
    if pth_c:
        L.append(f"| threshold estimate (critical / over-coupled) | {pth_c:.0f} mW" +
                 (f" / {pth_o:.0f} mW |" if pth_o else " |"))
    if gc_ok:
        eff = gc.get("efficiency_2d")
        L.append(f"| grating coupler (2D), fibre {gc['fiber_angle_deg']:g}° "
                 f"{gc.get('fiber_tilt', 'away')} the waveguide | "
                 f"{eff * 100:.1f} % ({-gc['loss_db_2d']:.2f} dB) at {wl} µm |")
    L.append("")

    # ---- assumptions
    rows = assumptions.table(cfg)
    L += ["## Assumed values", "",
          "These are inputs, not results. Values from publications can be replaced by your own "
          "measurements in `design.toml`.", "",
          "| quantity | value | unit | source | note |", "|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['quantity']} | {r['value']} | {r['unit']} | {r['source']} | {r['note']} |")
    L.append("")

    # ---- 1 materials
    if mat:
        L += ["## 1. Materials", "",
              "Each material's tabulated n, k is fitted with a pole-residue model over the band "
              f"{mat['band'][0]}-{mat['band'][1]} µm; the fit is checked against the data between "
              "the tabulated points.", "",
              "| role | material | data range (µm) | n at λ | covers band |", "|---|---|---|---|---|"]
        for r in mat["materials"]:
            L.append(f"| {r['role']} | `{r['material']}` | {r['data_range']} | {r['n_at_wl']} | "
                     f"{r['covers_band']} |")
        L += [""] + _fig(ctx, mat.get("figure"), "materials") + [
            f"Fitted n, k over the band: `{_rel(ctx, mat['table'])}`.", ""]

    # ---- 2 cross-section
    if xs:
        conv = xs.get("convergence") or []
        conv_txt = ", ".join(f"{c['D']:.1f} ({c['steps']})" for c in conv) or "-"
        L += ["## 2. Cross-section", "",
              "Method: local finite-difference mode solver over a map of core heights and "
              "widths; at each point the TE00 effective index is solved at 9 wavelengths and "
              "fitted with a 4th-order polynomial, giving n_g and D. A point is a target when D "
              "≥ 0 (anomalous) and exactly one TE mode is guided (n_eff above the higher of the "
              "BOX and top-cladding indices). Candidates are ranked by how many of their 8 "
              "neighbours are targets, then by the TE00/TM00 index split.", ""]
        L += _fig(ctx, xs.get("map_figure"), "design map")
        L += _fig(ctx, xs.get("figure"), "cross-section and TE00 mode")
        mesh_change = ("-" if xs.get("mesh_change") is None else f"{xs['mesh_change'] * 100:.1f} %")
        L += [f"Chosen: **{xs['height_um'] * 1e3:.0f} nm × {xs['width_um'] * 1e3:.0f} nm** "
              f"({xs['steps']} mesh steps per wavelength).", "",
              "| quantity | value |", "|---|---|",
              f"| D (TE00) | {xs['D']:.1f} ps/(nm·km) |",
              f"| D over the tolerance box | {xs['D_min_tol']:.1f} to {xs['D_max_tol']:.1f} ps/(nm·km) |",
              f"| n_eff / n_g (TE00) | {xs['n_eff']:.5f} / {xs['n_group']:.4f} |",
              f"| effective area A_eff | {xs.get('aeff_um2')} µm² |",
              f"| guided TE / TM modes | {xs['guided_TE']} / {xs['guided_TM']} |",
              f"| TE00/TM00 index split | {'-' if xs.get('te_tm_split') is None else round(xs['te_tm_split'], 4)} |",
              f"| whole tolerance box anomalous + single-mode | "
              f"{'yes' if xs['tolerance_box_target'] else 'no'} |",
              f"| D against mesh (steps per wavelength) | {conv_txt}; last change {mesh_change} |",
              "", "D moves by a few per cent between meshes (the core edges fall differently "
              "on each grid); treat it as ±5 % when comparing with another solver. n_eff and "
              "n_g are stable to the third decimal.", "",
              f"Finalist report: `{_rel(ctx, xs['report'])}`.", ""]

    # ---- 3 ring
    if ring:
        L += ["## 3. Ring", "",
              "Radius from the FSR target: R = c / (2π n_g FSR). Bend loss from a local "
              "bent-mode solve at R. Intrinsic Q from the total loss α: Q_i = 2π n_g / (λ α). "
              "D(λ), D2 and D3 from a broadband mode solve of the straight waveguide.", "",
              "| quantity | value |", "|---|---|",
              f"| radius | {ring['radius_um']} µm |",
              f"| FSR | {ring['fsr_ghz']:.1f} GHz |",
              f"| loss: scattering + absorption + bend | {ring['loss_db_cm']:.3g} dB/cm "
              f"({ring['scattering_loss_db_cm']} + {ring['absorption_loss_db_cm']} + "
              f"{ring['bend_loss_db_cm']:.2g}) |",
              f"| Q_i | {ring['q_intrinsic']:.3g} |",
              f"| D2/2π, D3/2π | {ring.get('D2_2pi_MHz')} MHz, {ring.get('D3_2pi_kHz')} kHz |",
              f"| zero-dispersion wavelengths | {ring.get('zdw_nm')} |"]
        if pth_c:
            L.append(f"| threshold, critical coupling (comparative, ±~2×) | {pth_c:.0f} mW |")
        if pth_o:
            L.append(f"| threshold, over-coupled K = {K:g} | {pth_o:.0f} mW |")
        L += [""] + _fig(ctx, ring.get("profile_figure"), "dispersion profile")

    # ---- 4 gap
    if gap_ok:
        ref_txt = {"ring_port": "the power entering the ring arc (mode monitor at the arc's "
                                "output end)",
                   "bus_only": "1 − T/T_bus, with T_bus from the same simulation without the ring",
                   "source": "1 − T, T relative to the source"}.get(gap.get("reference", "source"),
                                                                  gap.get("reference"))
        L += ["## 4. Ring–bus gap", "",
              f"Method: 3D FDTD of a {gap.get('span_angle_deg', 45):g}° partial ring next to the "
              f"straight bus at each gap; coupled power per pass = {ref_txt}. The coupling "
              "angle θ = asin(√coupled) is fitted with θ = θ0·exp(−γ g); the critical gap gives "
              "sin²θ = 1 − a², with a² the round-trip power survival from Q_i; the over-coupled "
              "gap gives K times that.", ""]
        L += _fig(ctx, gap.get("figure"), "gap coupling")
        L += ["| regime | gap | loaded Q |", "|---|---|---|",
              f"| critical | {gap['critical_gap_um'] * 1e3:.0f} nm | {gap['q_loaded_critical']:.3g} |"]
        if gap.get("over_coupled_gap_um"):
            L.append(f"| over-coupled, K = {gap['over_coupling']:g} | "
                     f"{gap['over_coupled_gap_um'] * 1e3:.0f} nm | {gap['q_loaded_over']:.3g} |")
        fit = gap.get("fit") or {}
        L += ["", f"Fit: θ0 = {fit.get('theta_0', float('nan')):.4f}, γ = "
              f"{fit.get('gamma', float('nan')):.3f} /µm over "
              f"{gap.get('fit_gap_range_um')} µm (R² of ln θ = {gap.get('fit_r2')}). "
              f"Per-gap values: `{_rel(ctx, gap.get('table', ''))}`."]
        if gap.get("reference") == "source" or not gap.get("reference"):
            L += ["", "**Caution:** this gap used the source-referenced through power, which "
                  "includes numerical bus losses that do not depend on the gap and biases the "
                  "critical gap wide. Rerun this stage with `gap.reference = \"ring_port\"`."]
        if gap.get("extrapolated"):
            L += ["", "The critical gap lies outside the fitted gaps (extrapolated)."]
        L.append("")
    elif gap:
        L += ["## 4. Ring–bus gap", "", "Skipped (not confirmed, over budget or no credits).", ""]

    # ---- 5 grating
    if gc_ok:
        L += ["## 5. Grating coupler", "",
              f"Method: 2D FDTD of the grating cross-section with a Gaussian fibre beam (MFD "
              f"{gc['mfd_um']} µm, {gc['fiber_angle_deg']:g}° from vertical, travelling "
              f"{gc.get('fiber_tilt', 'away')} the waveguide, TE polarised) launched 2 µm above "
              "the top cladding; efficiency = power in the waveguide's fundamental mode, over a "
              f"±{gc.get('bandwidth_um', 0.01) * 1e3:.0f} nm band. Step A maps period × fill "
              "factor around the phase-matched period of each fill factor; step B moves the "
              "period so the peak lands on the design wavelength and varies fill factor and "
              "period around it; step C moves the fibre along the grating.", ""]
        for key, alt in (("5a_grating_geometry.png", "grating geometry"),
                         ("5b_grating_map.png", "grating map"),
                         ("5c_grating_sensitivity.png", "grating sensitivity"),
                         ("5d_grating_spectrum.png", "grating spectrum"),
                         ("5e_grating_field.png", "grating field")):
            L += _fig(ctx, ctx.out / "figures" / key, alt)
        eff3d = "-" if gc.get("efficiency_3d") is None else f"{gc['efficiency_3d']:.3f}"
        bw1 = gc.get("bandwidth_1db_nm")
        L += ["| quantity | value |", "|---|---|",
              f"| etch | {gc['etch']} ({gc['etch_depth_um']} µm) |",
              f"| period / fill factor / teeth | {gc['period_um']} µm / {gc['fill_factor']} / "
              f"{gc['n_teeth']} |",
              f"| fibre angle / orientation / MFD | {gc['fiber_angle_deg']:g}° / "
              f"{gc.get('fiber_tilt', 'away')} the waveguide / {gc['mfd_um']} µm |",
              f"| fibre position (beam centre on the grating, from its first tooth) | "
              f"{gc['source_x_um']} µm |",
              f"| 2D efficiency at {wl} µm | {gc['efficiency_2d'] * 100:.1f} % "
              f"({-gc['loss_db_2d']:.2f} dB) |"]
        if gc.get("peak_efficiency_2d") is not None:
            L += [f"| peak (2D) | {gc['peak_efficiency_2d'] * 100:.1f} % at "
                  f"{gc['peak_wavelength_um'] * 1e3:.1f} nm |",
                  f"| 1 dB / 3 dB bandwidth | {bw1:.0f} nm / {gc['bandwidth_3db_nm']:.0f} nm |"]
        L += [f"| 3D efficiency | {eff3d} |", "",
              "Not included: the taper between the grating and the waveguide (in the GDS, not "
              "simulated yet), and the 3D beam overlap (the 2D value is usually higher than 3D).",
              ""]
        for note in gc.get("notes") or []:
            L.append(f"- Note: {note}")
        L += ["", f"All simulated designs: `{_rel(ctx, gc['table'])}`.", ""]
    elif gc:
        L += ["## 5. Grating coupler", "", "Skipped (not confirmed, over budget or no credits).", ""]

    # ---- 6 layout
    if lay:
        L += ["## 6. Layout", ""] + _fig(ctx, lay.get("figure"), "layout")
        L += [f"- `{_rel(ctx, f)}`" for f in lay.get("files", [])] + [""]
        p = lay.get("parameters") or {}
        if p.get("fiber_tilt"):
            L += [f"Fibres: {p.get('fiber_angle_deg')}° from vertical, beam travelling "
                  f"{p['fiber_tilt']} the waveguide at both couplers"
                  + (": the two fibres form a V (upper ends lean outward)." if p["fiber_tilt"] ==
                     "towards" else ": the upper ends of the two fibres lean inward."), ""]

    # ---- limits
    L += ["## Known limits", "",
          "- Vertical sidewalls everywhere; the n, k data are tabulated bulk values, not your film.",
          "- Ring: the bent mode is solved for loss and dispersion; avoided crossings between "
          "mode families are not modelled.",
          "- Gap: a partial ring whose arcs end inside the simulation box; the span is not "
          "converged against longer arcs. Inside the ring the builder's air region reaches "
          "0.1 µm into the BOX (a shallow recess the real full etch does not have); it touches "
          "the ring's inner side only, not the gap.",
          "- Grating: 2D, uniform period; no taper; no 3D beam overlap unless the 3D check ran.",
          ""]

    checks = _check_values(cfg, mat, xs, ring, gap, gc, pth_c)
    table = ctx.path("tables", "check_values.csv")
    with open(table, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["quantity", "value", "unit", "how TARANG computed it"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(checks)
    L += ["## Checking these numbers elsewhere", "",
          f"`{_rel(ctx, table)}` lists each value with how it was computed. `benchmark_pack/` "
          "holds everything another solver needs: the step-by-step method, assumed values, "
          "results, the n, k data used, figures, tables, simulation descriptions and the GDS.",
          ""]

    md = ctx.path("report.md")
    md.write_text("\n".join(L), encoding="utf-8")
    summary = ctx.path("design_summary.json")
    summary.write_text(json.dumps(dict(st, threshold_mw={"critical": pth_c, "over_coupled": pth_o}),
                                  indent=1, default=str), encoding="utf-8")
    pack = benchmark.build(ctx, st, checks, pth=(pth_c, pth_o))
    ctx.ui.say(f"Report: {md}")
    ctx.ui.say(f"Benchmark pack: {pack['folder']} (zip: {pack['zip']})")
    return dict(report=str(md), summary=str(summary), check_values=str(table), **pack)


def _check_values(cfg, mat, xs, ring, gap, gc, pth_c=None) -> list[dict]:
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
        add("TE00-TM00 index split", xs.get("te_tm_split"), "", how)
    if ring:
        add("ring radius", ring["radius_um"], "um", "from the FSR target and n_g")
        add("FSR", round(ring["fsr_ghz"], 3), "GHz", "c / (n_g 2 pi R)")
        add("Q_int", round(ring["q_intrinsic"], 0), "", "2 pi n_g / (lambda alpha), alpha from "
            "the assumed losses")
        add("bend loss TE00", ring["bend_loss_db_cm"], "dB/cm", "local bent-mode solve with PML")
        add("D2/2pi", ring.get("D2_2pi_MHz"), "MHz", "broadband mode solve, D_int fit")
        add("threshold (critical, comparative)", None if pth_c is None else round(pth_c, 1), "mW",
            "1.54 (pi/2) n_eff^2 A_eff 2 pi R / (n2 lambda Q_L^2)")
    if gap and not gap.get("skipped"):
        add("critical gap", round(gap["critical_gap_um"], 4), "um",
            f"3D FDTD of a partial ring next to the bus; coupled power measured as "
            f"'{gap.get('reference', 'source')}'; exponential fit of the coupling angle; "
            "matched to the round-trip loss from Q_int")
        if gap.get("over_coupled_gap_um"):
            add(f"over-coupled gap (K = {gap['over_coupling']:g})",
                round(gap["over_coupled_gap_um"], 4), "um", "same fit, K times the critical "
                "coupled power")
        for r in gap.get("per_gap") or []:
            add(f"coupled power at gap {r['gap_um']} um", r["coupled_power"], "",
                "power per pass entering the ring")
    if gc and not gc.get("skipped"):
        add("grating efficiency (2D) at the design wavelength", round(gc["efficiency_2d"], 4), "",
            f"2D FDTD, Gaussian fibre beam (MFD {gc['mfd_um']} um, {gc['fiber_angle_deg']} deg, "
            f"{gc.get('fiber_tilt', 'away')} the waveguide), power in the waveguide mode")
        add("grating peak wavelength (2D)", gc.get("peak_wavelength_um"), "um", "same run")
        add("grating 1 dB bandwidth (2D)", gc.get("bandwidth_1db_nm"), "nm", "same run")
    return rows
