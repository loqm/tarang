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
    from tarang.design import assumptions, benchmark, checks as _checks, visuals
    _checks.redraw_saved_figures(ctx)              # older runs: improved figures, no new runs
    chk_res = ctx.state["stages"].get("checks", {}).get("results")
    if chk_res and chk_res.get("loss_budget") is not None:          # include later stages (taper)
        g_res = ctx.state["stages"].get("grating", {}).get("results") or {}
        lb_now = _checks.loss_budget(ctx, None if g_res.get("skipped") else (g_res or None))
        if lb_now != chk_res["loss_budget"]:
            chk_res["loss_budget"] = lb_now
            visuals.safe(ctx, "loss budget", _checks.draw_loss_budget, ctx, lb_now)
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
        L += _fig(ctx, ctx.out / "figures" / "2b_mode_profiles.png",
                  "TE00 and TM00 mode profiles (|E| and its components)")
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
        L += _fig(ctx, ctx.out / "figures" / "3b_bent_mode.png", "straight and bent TE00")
        if ring.get("bent_mode_shift_nm") is not None:
            L += [f"In the bend the TE00 mode moves {ring['bent_mode_shift_nm']:+.1f} nm towards "
                  "the outside of the ring (local bent-mode solve).", ""]

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
        L += _fig(ctx, ctx.out / "figures" / "4a_gap_geometry.png", "gap simulation geometry")
        L += _fig(ctx, gap.get("figure"), "gap coupling (linear, log and fit)")
        L += ["| regime | gap | loaded Q |", "|---|---|---|",
              f"| critical | {gap['critical_gap_um'] * 1e3:.0f} nm | {gap['q_loaded_critical']:.3g} |"]
        if gap.get("over_coupled_gap_um"):
            L.append(f"| over-coupled, K = {gap['over_coupling']:g} | "
                     f"{gap['over_coupled_gap_um'] * 1e3:.0f} nm | {gap['q_loaded_over']:.3g} |")
        fit = gap.get("fit") or {}
        L += ["", f"Fit: θ0 = {fit.get('theta_0', float('nan')):.4f}, γ = "
              f"{fit.get('gamma', float('nan')):.3f} /µm over "
              f"{gap.get('fit_gap_range_um')} µm (R² of ln θ = {_r(gap.get('fit_r2'), 5)}). "
              f"Per-gap values: `{_rel(ctx, gap.get('table', ''))}`."]
        if gap.get("reference") == "source" or not gap.get("reference"):
            L += ["", "**Caution:** this gap used the source-referenced through power, which "
                  "includes numerical bus losses that do not depend on the gap and biases the "
                  "critical gap wide. Rerun this stage with `gap.reference = \"ring_port\"`."]
        if gap.get("extrapolated"):
            L += ["", "The critical gap lies outside the fitted gaps (extrapolated)."]
        recess = gap.get("ring_inner_box_recess_um", 0.1)
        L += ["", f"Simulation details: {gap.get('band_points', 1)} wavelength(s) recorded at the "
              f"ring ports; inside the ring the air region reaches {recess:g} µm into the BOX"
              + (" (a recess the real full etch does not have; it touches only the ring's inner "
                 "side, not the gap)." if recess else " (no recess: the BOX top is flat).")]
        cmt = gap.get("coupled_mode_check") or st.get("checks", {}).get("coupled_mode_check") or {}
        if cmt.get("critical_gap_um"):
            L += ["", "**Independent check (coupled-mode theory, free).** The even and odd TE "
                  "supermodes of bus and ring side by side give the coupling per length "
                  "κ = π Δn / λ; with the gap growing as x²/2R along the ring, the coupling "
                  "angle per pass is θ = κ(g) √(2πR/γ).", "",
                  "| | critical gap | γ (/µm) |", "|---|---|---|",
                  f"| 3D FDTD fit | {gap['critical_gap_um'] * 1e3:.0f} nm | "
                  f"{fit.get('gamma', float('nan')):.2f} |",
                  f"| coupled-mode theory | {cmt['critical_gap_um'] * 1e3:.0f} nm | "
                  f"{cmt['gamma_per_um']:.2f} |", ""]
            L += _fig(ctx, ctx.out / "figures" / "4c_gap_supermodes.png", "coupler supermodes")
        L += _fig(ctx, ctx.out / "figures" / "4d_gap_field.png", "gap field map")
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
              "period around it (B2: the period again around the best fill factor, when that "
              "differs from the centre one); step C moves the fibre along the grating; step D "
              "(when on) corrects the period once more so the peak of the final spectrum, "
              "refined by a parabola through its top three points, sits on the design "
              "wavelength.", ""]
        for key, alt in (("5a_grating_geometry.png", "grating geometry"),
                         ("5b_grating_map.png", "grating map"),
                         ("5c_grating_sensitivity.png", "grating sensitivity"),
                         ("5d_grating_spectrum.png", "grating spectrum"),
                         ("5e_grating_field.png", "grating field"),
                         ("5f_grating_final_geometry.png", "final grating geometry"),
                         ("5g_grating_landscape.png", "grating efficiency landscape")):
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
              "The grating's width, its taper (section 7b) and its 3D beam overlap enter the "
              "fibre-to-fibre budget in section 7; the 2D value here is usually higher than a "
              "full 3D one.",
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

    L += _checks_section(ctx, st.get("checks", {}))
    L += _taper_section(ctx, st.get("taper", {}))
    L += _apodise_section(ctx, st.get("apodise", {}))
    L += _focus_section(ctx, st.get("focus", {}))
    L += _lle_section(ctx, lay.get("lle"))
    L += _runs_section(ctx)

    # ---- limits (last, after every result)
    L += ["## Known limits", "",
          "- Vertical sidewalls everywhere; the n, k data are tabulated bulk values, not your film.",
          "- Ring: the bent mode is solved for loss and dispersion; how often other mode "
          "families cross TE00 is estimated (checks), but the strength of an avoided crossing "
          "is not modelled.",
          "- Gap: a partial ring whose arcs end inside the simulation box; the span is not "
          "converged against longer arcs.",
          "- Grating: 2D; its width enters only through the lateral overlap with the fibre "
          "(effective-index method, section 7b); no 3D grating run unless the 3D check ran.",
          "- Taper: EME of the straight linear taper (fundamental mode in and out), even modes "
          "only; sidewalls vertical.",
          "- Focusing grating (when designed): the in-plane focusing is a model (the fibre spot's "
          "angular spectrum against the tip mode, effective-index method); the 2D cross-section is "
          "assumed along every radial line. Only the optional 3D FDTD checks both.",
          ""]


    checks = _check_values(cfg, mat, xs, ring, gap, gc, pth_c)
    checks += _check_values_12(st)
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


HEADERS = {"gap_um": "gap (µm)", "Q_loaded": "loaded Q", "dip_depth_db": "dip depth (dB)",
           "n_eff": "n_eff", "n_g": "n_g", "fsr_ghz": "FSR (GHz)", "loss_db_cm": "loss (dB/cm)",
           "modes_between_crossings": "TE00 modes between crossings",
           "crossings_in_band": "crossings in the band", "loss_db": "loss (dB)",
           "estimate_FC": "estimate (FC)", "efficiency_at_target": "efficiency at λ",
           "peak_wavelength_nm": "peak (nm)", "peak_shift_nm": "peak shift (nm)",
           "change_db_at_target": "change at λ (dB)", "wavelengths_nm": "band (nm)",
           "K_min": "K min", "K_max": "K max", "arc_end_reflection_db": "arc-end reflection (dB)",
           "bus_reflection_db": "back towards the source (dB)"}


def _r(v, nd):
    return "-" if v is None else round(float(v), nd)


def _md_table(rows: list[dict], cols: list[str], fmt: dict | None = None) -> list[str]:
    fmt = fmt or {}
    out = ["| " + " | ".join(HEADERS.get(c, c.replace("_", " ")) for c in cols) + " |",
           "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append("-" if v is None else (format(v, fmt[c]) if c in fmt and
                                                isinstance(v, (int, float)) else str(v)))
        out.append("| " + " | ".join(cells) + " |")
    return out + [""]


def _checks_section(ctx, chk: dict) -> list[str]:
    """Section 7: tolerances, mode families, loss budget and the cloud checks."""
    if not chk:
        return []
    wl = ctx.cfg.run.wavelength_um
    L = ["## 7. Checks", "",
         "Free checks (local solves and saved results) and, when confirmed, small cloud runs. "
         "Tables: `tables/`.", ""]
    if chk.get("gap_tolerance"):
        dW = ctx.cfg.cross_section.tolerance_width_um
        L += [f"**Gap tolerance.** Waveguides {dW * 1e3:.0f} nm wider (or narrower) shrink "
              "(or widen) the gap by the same amount. K = coupled power / round-trip loss "
              "(1 = critical); loaded Q = Q_i / (1 + K); dip = depth of the through-port "
              "resonance, ((1 − K)/(1 + K))².", ""]
        L += _md_table(chk["gap_tolerance"], ["design", "case", "gap_um", "K", "Q_loaded",
                                              "dip_depth_db"], {"Q_loaded": ".3g"})
    refl = chk.get("arc_reflection")
    if refl:
        worst = max(r["arc_end_reflection"] for r in refl)
        L += [f"**Reflections in the gap runs.** The arcs of the partial ring end inside the "
              f"simulation box, so some light that entered the ring is reflected back (the "
              f"stripes of period λ/2n_eff in the field map). At the ring port it is at most "
              f"{10 * math.log10(max(worst, 1e-12)):.1f} dB of the forward power. The coupled "
              "power is read from the forward-travelling part only, so the gap result is "
              "affected only through the light that couples back into the bus (κ² times this).",
              ""]
        L += _md_table([dict(r, arc_end_reflection_db=round(10 * math.log10(max(r["arc_end_reflection"], 1e-12)), 1),
                             bus_reflection_db=round(10 * math.log10(max(r["bus_reflection"], 1e-12)), 1))
                        for r in refl], ["gap_um", "arc_end_reflection_db", "bus_reflection_db"])
    f = chk.get("fsr_tolerance")
    if f:
        L += [f"**FSR tolerance.** Over the cross-section's tolerance box the FSR spans "
              f"{f['fsr_min_ghz']:.1f}–{f['fsr_max_ghz']:.1f} GHz (n_g "
              f"{f['n_g_min']:.4f}–{f['n_g_max']:.4f}).", ""]
    if chk.get("mode_families"):
        L += ["**Mode families of the ring waveguide** (bent, local solve). A family with a "
              "different FSR drifts past TE00 by FSR₀₀ − FSRᵢ per mode, so it crosses TE00 every "
              "FSR₀₀ / |FSR₀₀ − FSRᵢ| modes; near a crossing the resonances can shift "
              "(avoided crossing), which locally changes the dispersion.", ""]
        L += _md_table(chk["mode_families"], ["family", "n_eff", "n_g", "fsr_ghz", "loss_db_cm",
                                              "modes_between_crossings", "crossings_in_band"])
    if chk.get("loss_budget"):
        L += [f"**Fibre-to-fibre loss budget** at {wl} µm, off resonance.", ""]
        L += _md_table(chk["loss_budget"], ["item", "loss_db"], {"loss_db": ".2f"})
        L += _fig(ctx, ctx.out / "figures" / "7c_loss_budget.png", "loss budget")
    b = chk.get("band")
    if b and b.get("curves"):
        L += ["**Coupling across the band** (3D FDTD of the critical and over-coupled gaps, "
              "recorded at several wavelengths): the coupling ratio K over the band.", ""]
        rows = []
        for k, c in b["curves"].items():
            K = c["K"]
            rows.append(dict(design=k, K_min=round(min(K), 2), K_max=round(max(K), 2),
                             wavelengths_nm=f"{min(c['wavelength_um']) * 1e3:.0f}–"
                                            f"{max(c['wavelength_um']) * 1e3:.0f}"))
        L += _md_table(rows, ["design", "wavelengths_nm", "K_min", "K_max"])
        L += _fig(ctx, ctx.out / "figures" / "7a_coupling_vs_wavelength.png",
                  "coupling against wavelength")
    g = chk.get("grating")
    if g and g.get("variants"):
        L += ["**Grating tolerances** (2D FDTD): fibre angle, film thickness, tooth width "
              "(lithography bias) and a finer mesh, against the final design.", ""]
        L += _md_table(g["variants"], ["case", "efficiency_at_target", "peak_wavelength_nm",
                                       "peak_shift_nm", "change_db_at_target"],
                       {"efficiency_at_target": ".3f"})
        t = (g.get("power_budget") or {}).get("at_target")
        if t:
            L += [f"Light launched from the waveguide into the grating (reverse run) at {wl} µm: "
                  f"{t['up'] * 100:.0f} % up, {t['down'] * 100:.0f} % into the substrate, "
                  f"{t['past'] * 100:.0f} % past the grating"
                  + (f", {t['scattered_back'] * 100:.0f} % scattered backwards"
                     if t.get("scattered_back") is not None else "")
                  + f"; back-reflection into the waveguide "
                  f"{t['reflection_db']:.1f} dB. Directionality (up / (up + down)) "
                  f"{t['directionality'] * 100:.0f} %"
                  + (f"; overlap of the upward beam with the fibre mode {t['overlap'] * 100:.0f} %."
                     if t.get("overlap") else "."), ""]
        L += _fig(ctx, ctx.out / "figures" / "7b_grating_checks.png", "grating checks")
    m = chk.get("gap_mesh")
    if m:
        L += [f"**Gap mesh check.** At the critical gap with a 1.5× finer mesh the coupled "
              f"power is {m['coupled_fine']:.4e} against {m['coupled_fit']:.4e} from the fit "
              f"({m['change_percent']:+.1f} %).", ""]
    return L


def _taper_section(ctx, tp: dict) -> list[str]:
    """Section 7b: grating width (fibre overlap) and taper length (EME)."""
    if not tp or tp.get("skipped"):
        return []
    L = ["## 7b. Grating width and taper", "",
         "**Width.** The 2D grating simulation is infinitely wide. A grating of width W carries "
         "its fundamental lateral mode ψ(y); the fibre delivers a Gaussian G(y) of waist MFD/2. "
         "Only their overlap η_y = |∫ψG dy|² / (∫|ψ|² dy ∫|G|² dy) couples, so the 3D efficiency is "
         f"about η_2D·η_y. ψ comes from the effective-index method (slab index "
         f"{tp['n_slab']:.4f}).", "",
         "| | width | η_y |", "|---|---|---|",
         f"| set before | {tp['width_before_um']:g} µm | {tp['eta_y_before'] * 100:.1f} % |",
         f"| best for the fibre | {tp['best_width_um']:g} µm | - |",
         f"| **used** | **{tp['grating_width_um']:g} µm** | **{tp['eta_y'] * 100:.1f} %** |", ""]
    if tp.get("efficiency_3d_estimate"):
        e = tp["efficiency_3d_estimate"]
        L += [f"Grating efficiency with the width: about {e * 100:.1f} % "
              f"({-10 * math.log10(e):.2f} dB) per coupler.", ""]
    L += _fig(ctx, ctx.out / "figures" / "8a_grating_width.png", "grating width")
    if tp.get("model_lengths_um") or tp.get("lengths_um"):
        L += ["**Length.** A linear taper from the grating width to the waveguide loses power "
              "into higher modes when it is short. Two calculations: a 2D model of the taper "
              "(effective-index method; the taper cut into many graded cells whose modes are "
              "joined by their overlaps, converged, free) and Tidy3D's 3D eigenmode expansion "
              "(EME) of the same taper. EME replaces the taper by a staircase of straight cells; "
              "too few cells make the loss jump up and down with length, so the 3D result is used "
              "only when its loss does not rise with length by more than the accepted loss.", ""]
        if tp.get("lengths_um"):
            conv = tp.get("eme_converged")
            L += [f"3D EME: {tp.get('eme_cells', '?')} graded cells x {tp.get('eme_modes', '?')} "
                  f"modes; " + ("converged" if conv else
                                (f"**not converged** (the loss rises by {tp.get('eme_rise_db', 0):.2f} dB "
                                 "with length)" if conv is False else "convergence not checked"))
                  + (f"; the 2D model predicts a staircase error up to {tp['staircase_error_db']:.3f} dB "
                     "for this cell count." if tp.get("staircase_error_db") is not None else "."), ""]
        if tp.get("taper_length_um"):
            L += [f"Taper length used: **{tp['taper_length_um']:g} µm** ({tp.get('length_source', 'EME')}, "
                  f"{tp['taper_loss_db']:.3f} dB per taper, accepted {ctx.cfg.taper.max_loss_db:g} dB)."]
            if tp.get("model_loss_db_at_previous") is not None:
                L += [f"The {tp['previous_length_um']:g} µm taper used before: "
                      f"{tp['model_loss_db_at_previous']:.3f} dB (2D model)."]
            L += [""]
        L += _fig(ctx, ctx.out / "figures" / "8b_taper_length.png", "taper length")
    if tp.get("use_in_layout"):
        L += ["The layout uses this width and length (`taper.use_in_layout`).", ""]
    return L


def _apodise_section(ctx, ap: dict) -> list[str]:
    """Section 7c: apodised grating."""
    if not ap or not ap.get("scattering"):
        return []
    sc = ap["scattering"]
    L = ["## 7c. Apodised grating", "",
         "In a uniform grating every tooth scatters the same fraction of the guided light, so "
         "the light leaves with an exponential profile, which overlaps a Gaussian fibre mode by "
         "at most about 80 %. Shaping the scattering strength along the grating, "
         "α(x) = A·G²(x) / (2(1 − A∫₀ˣG²)), makes the emitted profile the fibre's Gaussian "
         "itself. α and the phase-matched period of each fill factor were measured on uniform "
         "gratings (2D FDTD, light launched from the waveguide, field recorded 1.5 µm above the "
         "grating: the decay of |E|² gives α, its phase slope the Bloch index). Only one branch "
         "is used, from the strongest scatterer to the largest fill factor (α falls, the Bloch "
         "index rises), so each α has one fill factor; points are left out when the decay is too "
         "weak to measure, a fit is poor, or the second diffraction order also radiates up.", ""]
    out = [r for r in sc if r.get("why")]
    if out:
        L += ["Left out: " + "; ".join(f"FF {r['fill_factor']:g} ({r['why']})" for r in out) + ".", ""]
    if not ap.get("alpha_range_per_um"):
        L += ["Fewer than three usable fill factors on one branch: no apodised design "
              "(`tables/apodise_scattering.csv`).", ""]
        return L
    a_lo, a_hi = ap["alpha_range_per_um"]
    br = ap.get("branch") or [r["fill_factor"] for r in sc]
    L += [f"Used: fill factors {min(br):g} to {max(br):g}, α {a_lo:.3f} to {a_hi:.3f} per µm "
          f"(`tables/apodise_scattering.csv`). 1D emission model: the uniform grating overlaps the "
          f"fibre by {ap['model_uniform'] * 100:.0f} %, the apodised one by "
          f"{ap['model_apodised'] * 100:.0f} % ({len(ap['teeth'])} teeth, "
          "`tables/apodised_teeth.csv`)."
          + (f" All teeth are etched through the film; trenches {ap['narrowest_trench_um'] * 1e3:.0f}-"
             f"{ap['widest_trench_um'] * 1e3:.0f} nm (narrowest allowed "
             f"{ap['min_trench_um'] * 1e3:.0f} nm, `apodise.min_trench_um`)."
             if ap.get("narrowest_trench_um") else ""), ""]
    L += _fig(ctx, ctx.out / "figures" / "9a_apodise_design.png", "apodisation design")
    if ap.get("verified"):
        cen = (f" The period scale {ap['scale_centred']:.4f} puts the peak on the design wavelength "
               "(round 1); round 2 used it." if ap.get("scale_centred") else "")
        L += [f"**2D FDTD check:** {ap['efficiency_2d'] * 100:.1f} % at "
              f"{ctx.cfg.run.wavelength_um} µm ({-10 * math.log10(max(ap['efficiency_2d'], 1e-9)):.2f} dB), "
              f"peak at {ap['peak_wavelength_nm']} nm, 1 dB band {ap['bandwidth_1db_nm']:.0f} nm; "
              f"uniform {ap['uniform_efficiency_2d'] * 100:.1f} %. Fibre at "
              f"{ap['source_x_um']} µm, period scale {ap['scale']:g} (`tables/apodised_check.csv`)."
              + cen, ""]
        L += _fig(ctx, ctx.out / "figures" / "9b_apodised_spectrum.png", "apodised spectrum")
        L += _fig(ctx, ctx.out / "figures" / "9c_apodised_landscape.png", "apodised landscape")
        if ap.get("better_than_uniform") and ctx.cfg.apodise.use_in_layout:
            L += ["The layout uses the apodised grating (`layout/grating_teeth.csv`).", ""]
        else:
            L += ["The apodised grating does not beat the uniform one here; the layout keeps the "
                  "uniform grating.", ""]
    elif ap.get("verified") is False:
        L += ["The available fill factors cannot scatter weakly enough to shape the beam, so the "
              "apodised design was not run in FDTD. The narrowest trench the etch clears "
              "(`apodise.min_trench_um`) sets the weakest tooth.",
              ""]
    return L


def _focus_section(ctx, fc: dict) -> list[str]:
    """Section 7d: focusing grating."""
    if not fc or not fc.get("curves"):
        return []
    from math import degrees
    L = ["## 7d. Focusing grating", "",
         "Curved teeth (confocal ellipses n_B r − n_top sin θ · r cos φ = const around the "
         "waveguide tip) keep the fibre beam phase-matched along every direction from the tip, "
         "so the light scattered into the film converges onto the tip and a few-µm slab "
         "replaces the long taper. Along the axis the teeth are those of the "
         f"{fc['teeth_kind']} grating (2D efficiency {fc['efficiency_2d'] * 100:.1f} %). The "
         "in-plane focusing is computed from the angular spectrum of the fibre spot on the arc "
         "against the tip's mode (effective-index method), over the distance from the tip to "
         "the first tooth and the tip width; wide light cones are excluded "
         f"(half-angle at most {ctx.cfg.focus.max_half_angle_deg:g}°).", "",
         "| quantity | value |", "|---|---|",
         f"| tip width / tip taper | {fc['tip_width_um']:g} µm / {fc['tip_taper_um']:g} µm |",
         f"| tip to first tooth / to beam centre | {fc['distance_um']:g} µm / {fc['r_c_um']:.1f} µm |",
         f"| sector half-angle | {degrees(fc['half_angle_rad']):.0f}° (teeth within ±{fc['y_max_um']:g} µm) |",
         f"| in-plane focusing × tip taper | {fc['eta_focus'] * 100:.1f} % × {fc['tip_taper_T'] * 100:.1f} % |",
         f"| 3D estimate | {fc['efficiency_3d_estimate'] * 100:.1f} % (straight grating with its "
         f"taper: {fc['efficiency_2d'] * fc['straight_lateral'] * 100:.1f} %) |",
         f"| waveguide end to grating end | {fc['focusing_length_um']:.0f} µm (straight: "
         f"{fc['straight_length_um']:.0f} µm) |"]
    if fc.get("efficiency_3d") is not None:
        L.append(f"| 3D FDTD | {fc['efficiency_3d'] * 100:.1f} % at {ctx.cfg.run.wavelength_um} µm, "
                 f"peak at {fc['peak_wavelength_3d_nm']} nm |")
    L += [""]
    L += _fig(ctx, ctx.out / "figures" / "10a_focusing_landscape.png", "focusing landscape")
    L += _fig(ctx, ctx.out / "figures" / "10b_focusing_geometry.png", "focusing geometry")
    L += ["Curves of every tooth: `tables/focusing_teeth.csv`." +
          (" The layout uses the focusing grating (`focus.use_in_layout`)." if fc.get("use_in_layout")
           and ctx.cfg.focus.run else ""), ""]
    return L


def _check_values_12(st: dict) -> list[dict]:
    """Taper, apodised and focusing gratings in check_values.csv / the benchmark pack."""
    rows = []

    def add(q, v, unit, how):
        if v is not None:
            rows.append({"quantity": q, "value": v, "unit": unit, "how TARANG computed it": how})
    tp = st.get("taper") or {}
    if tp.get("grating_width_um"):
        add("grating width", tp["grating_width_um"], "um", "best lateral overlap of the fibre "
            "Gaussian with the grating's lateral mode (effective-index method)")
        add("lateral overlap eta_y", round(tp["eta_y"], 4), "", "same")
    if tp.get("taper_length_um"):
        add("taper length", tp["taper_length_um"], "um",
            f"shortest linear taper below the accepted loss ({tp.get('length_source')})")
        add("taper loss", round(tp["taper_loss_db"], 4), "dB", tp.get("length_source") or "")
    ap = st.get("apodise") or {}
    if ap.get("verified"):
        add("apodised grating efficiency (2D)", round(ap["efficiency_2d"], 4), "",
            "2D FDTD of the explicit tooth list (tables/apodised_teeth.csv)")
    fc = st.get("focus") or {}
    if fc.get("curves"):
        add("focusing grating: in-plane focusing", round(fc["eta_focus"], 4), "",
            "angular spectrum of the fibre spot against the tip mode (model)")
        add("focusing grating: 3D estimate", round(fc["efficiency_3d_estimate"], 4), "",
            "eta_2D x focusing x tip taper")
        if fc.get("efficiency_3d") is not None:
            add("focusing grating: 3D FDTD", round(fc["efficiency_3d"], 4), "", "3D FDTD")
    return rows


def _lle_section(ctx, lle: dict | None) -> list[str]:
    if not lle:
        return []
    L = ["## 8. Inputs for a comb (LLE) simulation", "",
         "Written to `lle/`: `lle_inputs.json` (every number with units and definitions, and "
         "a block in pyLLE's names), `pyLLE_dispersion.csv` (mode number, resonance frequency; "
         "pyLLE's dispersion-file format) and `integrated_dispersion.csv` (D_int(μ)).", ""]
    if lle.get("gamma_per_W_m"):
        L.append(f"- nonlinear coefficient γ = 2π n₂ / (λ A_eff) = {lle['gamma_per_W_m']:.2f} /(W·m)")
    d = lle.get("dispersion")
    if d:
        L.append(f"- pump mode m = {d['pump_mode_number']} at {d['f_pump_Hz'] / 1e12:.3f} THz; "
                 f"FSR {d['FSR_Hz'] / 1e9:.3f} GHz; D2/2π {d['D2_2pi_Hz'] / 1e6:.2f} MHz")
    for k, r in (lle.get("coupling") or {}).items():
        L.append(f"- {k.replace('_', ' ')}: Q_c {r['Q_c']:.3g}, Q_L {r['Q_L']:.3g}, κ/2π "
                 f"{r['kappa_2pi_Hz'] / 1e6:.1f} MHz, threshold {r['threshold_W'] * 1e3:.1f} mW")
    return L + [""]


def _runs_section(ctx) -> list[str]:
    """Every cloud run with its Tidy3D link (tables/cloud_runs.csv), the spending by stage,
    and the software versions."""
    from tarang.utilities.cloud_run import collect_runs
    from tarang.design import credits
    L = ["## Cloud runs, credits and versions", ""]
    rows = collect_runs(ctx.out / "cloud") if (ctx.out / "cloud").exists() else []
    if rows:
        table = ctx.path("tables", "cloud_runs.csv")
        cols = ["name", "status", "task_id", "url", "estimated_cost", "real_cost", "submitted"]
        with open(table, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        linked = sum(1 for r in rows if r.get("url"))
        L += [f"{len(rows)} cloud simulation(s), {linked} with a link to open them in the Tidy3D "
              f"web interface: `{_rel(ctx, table)}`; a picture of each run's geometry is in "
              "`figures/sims/` (same name as its cloud folder). "
              + (f"{len(rows) - linked} run(s) made before Phase 12 have no link: "
                 "`python -m tarang.design --out <folder> --runs` finds them in the task list of "
                 "the Tidy3D account in use; runs made with another account stay without a link "
                 "(their results are on disk)." if linked < len(rows) else ""), ""]
    s = credits.summary(ctx.cfg, ctx.out, ctx.state)
    L += [f"Credits (estimates): spent {s['spent']:.3f} of the {s['limit']:g} FC limit.", ""]
    if s["rows"]:
        L += _md_table(s["rows"], ["date", "stage", "what", "estimate_FC"])
    v = ctx.state.get("versions")
    if v:
        L += ["Software: " + ", ".join(f"{k} {v[k]}" for k in ("python", "tarang", "tidy3d", "numpy")
                                       if k in v) + f" (recorded {v.get('date', '?')}).", ""]
    return L
