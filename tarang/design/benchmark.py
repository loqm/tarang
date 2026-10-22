"""``benchmark_pack/``: everything another solver needs to repeat this design.

    benchmark_pack/
      README.md          step by step: what was simulated, with every dimension, material,
                         mesh, boundary, source and monitor, how each number was obtained,
                         the result, and what to compare
      assumptions.csv    assumed values with their sources
      results.csv        every computed value with how it was computed
      materials_nk.csv   the n, k actually used (fitted models, sampled over the band)
      design.toml        the settings of the run
      figures/  tables/  layout/   copies of the run's figures, tables and GDS
      simulations/       the simulation files used (JSON) and a plain description of each

A zip of the folder is written next to it. Nothing here is specific to another solver;
the simulation JSON files are TARANG's own record of each run.
"""
from __future__ import annotations

import csv
import glob
import shutil
from pathlib import Path

import numpy as np

C_UM = 2.99792458e14


def _copy(src, dst_dir: Path) -> Path | None:
    if not src:
        return None
    src = Path(src)
    if not src.exists():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return dst


def _fmt(v, nd=4):
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    if isinstance(v, (tuple, list)):
        return "(" + ", ".join(_fmt(x, nd) for x in v) + ")"
    return str(v)


def describe_sim(sim, wavelength_um: float) -> list[str]:
    """A plain-language description of a Tidy3D simulation (lengths in µm)."""
    import tidy3d as td
    out = []
    lo = [c - s / 2 for c, s in zip(sim.center, sim.size)]
    hi = [c + s / 2 for c, s in zip(sim.center, sim.size)]
    out.append(f"- Domain: x {lo[0]:.3f} to {hi[0]:.3f}, y {lo[1]:.3f} to {hi[1]:.3f}, "
               f"z {lo[2]:.3f} to {hi[2]:.3f} µm" + (" (2D: no y extent)" if sim.size[1] == 0 else ""))
    for ax in "xyz":
        b = getattr(sim.boundary_spec, ax)
        desc = []
        for side in ("minus", "plus"):
            bb = getattr(b, side)
            n = getattr(bb, "num_layers", None)
            desc.append(type(bb).__name__ + (f" ({n} layers)" if n else ""))
        out.append(f"- Boundary {ax}: " + (desc[0] if desc[0] == desc[1] else " / ".join(desc)))
    gs = sim.grid_spec
    steps = getattr(gs.grid_x, "min_steps_per_wvl", None)
    out.append(f"- Mesh: automatic non-uniform, at least {steps} cells per wavelength in each "
               f"material (λ = {gs.wavelength or wavelength_um} µm)")
    for o in gs.override_structures or []:
        dl = getattr(o, "dl", None)
        if dl:
            b = o.geometry.bounds
            out.append(f"  - finer mesh {_fmt(tuple(dl))} µm in x {b[0][0]:.2f}-{b[1][0]:.2f}, "
                       f"y {b[0][1]:.2f}-{b[1][1]:.2f}, z {b[0][2]:.2f}-{b[1][2]:.2f}")
    if hasattr(sim, "run_time"):
        out.append(f"- Run time {sim.run_time * 1e12:.3g} ps; stops early when the fields decay "
                   f"below {sim.shutoff:g} of their peak")
    else:                                              # an eigenmode-expansion (EME) simulation
        grid = getattr(sim, "eme_grid", None)
        n_cells = len(grid.boundaries) - 1 if grid is not None else "?"
        sweep = getattr(sim, "sweep_spec", None)
        out.append(f"- Eigenmode expansion along {'xyz'[sim.axis]}: {n_cells} cells"
                   + (f"; cell lengths swept ({type(sweep).__name__})" if sweep is not None else ""))
    out.append("- Structures (later ones override earlier ones where they overlap):")
    f0 = C_UM / wavelength_um
    for s in sim.structures:
        try:
            n = complex(s.medium.nk_model(f0)[0] + 1j * s.medium.nk_model(f0)[1])
            nk = f"n = {n.real:.5f}, k = {n.imag:.2e}"
        except Exception:
            nk = type(s.medium).__name__
        try:
            b = s.geometry.bounds
            bounds = (f"x {_fmt(b[0][0])} to {_fmt(b[1][0])}, y {_fmt(b[0][1])} to "
                      f"{_fmt(b[1][1])}, z {_fmt(b[0][2])} to {_fmt(b[1][2])}")
        except Exception:
            bounds = type(s.geometry).__name__
        out.append(f"  - {s.name or '-'}: {type(s.geometry).__name__}, {bounds}; {nk} at "
                   f"{wavelength_um} µm")
    out.append("- Sources:")
    for src in sim.sources:
        st = src.source_time
        line = (f"  - {src.name}: {type(src).__name__} at {_fmt(tuple(src.center))}, size "
                f"{_fmt(tuple(src.size))}, direction {getattr(src, 'direction', '-')}, "
                f"pulse centre {C_UM / st.freq0:.4f} µm, bandwidth {st.fwidth / st.freq0 * 100:.1f} %")
        if isinstance(src, td.GaussianBeam):
            d = np.round(src._dir_vector, 4)
            line += (f"; waist radius {src.waist_radius} µm at the source plane; angle_theta "
                     f"{np.degrees(src.angle_theta):.2f}° (propagation direction {tuple(d)}); "
                     f"pol_angle {np.degrees(src.pol_angle):.0f}° (90° = E along y, TE)")
        if isinstance(src, td.ModeSource):
            line += f"; mode index {src.mode_index}, target n_eff {src.mode_spec.target_neff}"
        out.append(line)
    out.append("- Monitors:")
    for m in sim.monitors:
        # Tidy3D keeps monitor frequencies as a numpy array (never test its truth value)
        raw = getattr(m, "freqs", None)
        freqs = [] if raw is None else [float(f) for f in np.atleast_1d(np.asarray(raw))]
        band = ""
        if freqs:
            wls = sorted(C_UM / f for f in freqs)
            band = (f", {len(freqs)} wavelengths {wls[0]:.4f}-{wls[-1]:.4f} µm" if len(freqs) > 1
                    else f", at {wls[0]:.4f} µm")
        out.append(f"  - {m.name}: {type(m).__name__} at {_fmt(tuple(m.center))}, size "
                   f"{_fmt(tuple(m.size))}{band}")
    return out


def _load_sim(path):
    from tarang.design.credits import load_simulation
    return load_simulation(path)


def _describe_file(path, wl: float, ui=None) -> list[str]:
    """describe_sim of a saved simulation; a failure is written into the README (and
    shown) instead of stopping the report, since the file itself is in the pack."""
    try:
        return describe_sim(_load_sim(path), wl)
    except Exception as exc:                              # noqa: BLE001
        msg = f"(could not describe {Path(path).name}: {type(exc).__name__}: {exc})"
        if ui is not None:
            ui.say("  Note: " + msg)
        return [f"- {msg} The simulation file itself is in `simulations/`."]


def build(ctx, st: dict, checks: list[dict], pth=(None, None)) -> dict:
    from tarang.design import assumptions
    cfg = ctx.cfg
    wl = cfg.run.wavelength_um
    pack = ctx.out / "benchmark_pack"
    if pack.exists():
        shutil.rmtree(pack)                       # rebuilt from the run every time
    pack.mkdir(parents=True)
    xs, ring = st.get("cross_section", {}), st.get("ring", {})
    gap, gc, mat = st.get("gap", {}), st.get("grating", {}), st.get("materials", {})

    rows = assumptions.table(cfg)
    with open(pack / "assumptions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["quantity", "value", "unit", "source", "note"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    with open(pack / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["quantity", "value", "unit", "how TARANG computed it"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(checks)
    if mat.get("table"):
        _copy(mat["table"], pack)
    _copy(ctx.out / "design.toml", pack)
    for ext in ("png", "pdf", "svg"):                 # every format the run wrote
        for f in sorted((ctx.out / "figures").glob(f"*.{ext}")):
            _copy(f, pack / "figures")
    extra_figs = [xs.get("map_figure"), ring.get("profile_figure")]
    extra_figs += glob.glob(str(ctx.out / "results" / "mesh_convergence_plots" / "*.png"))
    extra_figs += glob.glob(str(ctx.out / "results" / "finalist_report" / "*.png"))
    extra_figs += glob.glob(str(ctx.out / "cloud" / "gap_sweep" / "*.png"))
    extra_figs += glob.glob(str(ctx.out / "cloud" / "gap_sweep" / "field_maps" / "*.png"))
    for f in extra_figs:
        _copy(f, pack / "figures" / "details")
    for f in sorted((ctx.out / "tables").glob("*.csv")):
        _copy(f, pack / "tables")
    for f in (ctx.out / "results" / "finalist_report" / "finalists.csv",
              ctx.out / "results" / "mesh_convergence_plots" / "convergence_table.csv",
              ctx.out / "results" / "dispersion_profile_plots" / "profiles_summary.csv"):
        _copy(f, pack / "tables")
    for f in sorted((ctx.out / "layout").glob("*")):
        _copy(f, pack / "layout")
    for f in sorted((ctx.out / "lle").glob("*")):
        _copy(f, pack / "lle")

    sims_dir = pack / "simulations"
    sims_dir.mkdir()
    sim_sections: list[str] = []
    if gap and not gap.get("skipped"):
        for p in sorted(glob.glob(str(ctx.out / "cloud" / "gap_sweep" / "**" / "simulation.json"),
                                  recursive=True)):
            name = Path(p).parent.name.split("_1.550um")[0].split(f"_{wl:.3f}um")[0]
            dst = sims_dir / f"gap_{name}.json"
            shutil.copy2(p, dst)
        mid = sorted(gap.get("gaps_um") or [])[len(gap.get("gaps_um") or [0]) // 2] if gap.get(
            "gaps_um") else None
        cand = [p for p in sims_dir.glob("gap_*.json") if mid is not None and
                f"gap{mid * 1e3:.0f}nm" in p.name and "BusOnly" not in p.name]
        if cand:
            sim_sections += [f"### Ring–bus gap simulation (gap {mid} µm shown; the others "
                             "differ only in the gap)", ""] + _describe_file(cand[0], wl, ctx.ui) + [""]
    if gc and not gc.get("skipped") and gc.get("final_sim"):
        src = Path(gc["final_sim"])
        if src.exists():
            shutil.copy2(src, sims_dir / "grating_final.json")
            sim_sections += ["### Final grating-coupler simulation (2D)", ""] + _describe_file(
                src, wl, ctx.ui) + [""]
    tp, ap, fc = st.get("taper", {}) or {}, st.get("apodise", {}) or {}, st.get("focus", {}) or {}
    if tp.get("eme_cells"):
        cand = sorted(glob.glob(str(ctx.out / "cloud" / "taper" / f"*eme{tp['eme_cells']}x"
                                     f"{tp['eme_modes']}*" / "simulation.json")))
        if cand:
            shutil.copy2(cand[-1], sims_dir / "taper_eme.json")
            sim_sections += ["### Taper (3D EME)", ""] + _describe_file(cand[-1], wl, ctx.ui) + [""]
    if ap.get("scattering"):
        cand = sorted(glob.glob(str(ctx.out / "cloud" / "apodise" / "uniform_reverse_ff0.750*" /
                                    "simulation.json")))
        if cand:
            shutil.copy2(cand[-1], sims_dir / "apodise_uniform_reverse_ff0.75.json")
    if fc.get("curves"):
        try:
            from tarang.design.focus import focusing_sim
            dev = focusing_sim(ctx, fc)
            dev.sim.to_file(str(sims_dir / "focusing_grating_3d.json"))
            sim_sections += ["### Focusing grating (3D FDTD set-up)", ""] + describe_sim(dev.sim, wl) + [""]
        except Exception as err:                                  # noqa: BLE001
            sim_sections += [f"(focusing grating simulation not written: {type(err).__name__}: {err})", ""]
    readme = _readme(ctx, st, rows, checks, sim_sections, pth)
    (pack / "README.md").write_text("\n".join(readme), encoding="utf-8")
    zip_path = shutil.make_archive(str(ctx.out / "benchmark_pack"), "zip", root_dir=pack)
    return dict(folder=str(pack), zip=str(zip_path))


def _readme(ctx, st, rows, checks, sim_sections, pth) -> list[str]:
    cfg = ctx.cfg
    wl = cfg.run.wavelength_um
    xs, ring = st.get("cross_section", {}), st.get("ring", {})
    gap, gc, mat = st.get("gap", {}), st.get("grating", {}), st.get("materials", {})
    s = cfg.stack
    L = [f"# Benchmark pack: {cfg.run.name}", "",
         "What was designed, how, and with which assumptions, so that every number can be "
         "recomputed in another mode solver or FDTD tool. Lengths in µm unless stated. The "
         "design report (`../report.md`) shows the same results with all figures.", "",
         "## 0. Stack and materials", "",
         f"- Wavelength: {wl} µm (pump / design wavelength)",
         f"- Core: `{cfg.materials.core}`; BOX: `{cfg.materials.box}` {s.box_height_um} µm thick; "
         f"top: `{cfg.materials.cladding}`; handle (grating simulations only): "
         f"`{cfg.materials.handle}`",
         "- Material data: `materials_nk.csv` gives n and k of each layer exactly as used"
         + (f", sampled over {mat['band'][0]}-{mat['band'][1]} µm" if mat.get("band") else "")
         + ". Import these tables rather than a solver's built-in materials, so both solvers "
         "see the same indices.",
         "- Sidewalls: vertical (90°) everywhere.", ""]
    L += ["## Assumed values", "", "| quantity | value | unit | source |", "|---|---|---|---|"]
    L += [f"| {r['quantity']} | {r['value']} | {r['unit']} | {r['source']} |" for r in rows]
    L.append("")
    if xs:
        L += ["## 1. Waveguide cross-section (mode solver)", "",
              f"- Core {xs['height_um']} µm tall × {xs['width_um']} µm wide, rectangular, on the "
              f"BOX; {cfg.materials.cladding} above.",
              f"- Solve the TE00 mode (E mainly along the width) at 9 wavelengths over "
              f"{wl - 0.1:.2f}-{wl + 0.1:.2f} µm; fit n_eff(λ) with a 4th-order polynomial; "
              "n_g = n_eff − λ dn_eff/dλ; D = −(λ/c) d²n_eff/dλ².",
              f"- TARANG: {xs['steps']} mesh cells per wavelength" +
              ("; D against mesh: " + ", ".join(f"{c['D']:.1f} at {c['steps']}"
                                                for c in xs["convergence"]) if xs.get("convergence")
               else "") + ".",
              f"- Results to compare: n_eff = {xs['n_eff']:.5f}, n_g = {xs['n_group']:.4f}, "
              f"D = {xs['D']:.1f} ps/(nm·km) (±5 %), A_eff = {xs.get('aeff_um2')} µm², "
              f"{xs['guided_TE']} guided TE and {xs['guided_TM']} guided TM mode(s) (guided = "
              f"n_eff above {xs.get('n_cut')}).", ""]
    if ring:
        L += ["## 2. Ring", "",
              f"- Radius {ring['radius_um']} µm (to the waveguide centre), same cross-section.",
              f"- Bent-mode solve at R (absorbing layers at the plane edges): TE00 bend loss "
              f"{ring['bend_loss_db_cm']:.2g} dB/cm.",
              f"- FSR = c / (2π R n_g) = {ring['fsr_ghz']:.1f} GHz; Q_i = 2π n_g / (λ α) with α "
              f"= {ring['loss_db_cm']:.3g} dB/cm (assumed scattering + absorption + bend) = "
              f"{ring['q_intrinsic']:.3g}.",
              f"- D2/2π = {ring.get('D2_2pi_MHz')} MHz, D3/2π = {ring.get('D3_2pi_kHz')} kHz from a "
              "broadband solve of the straight waveguide (see `tables/profiles_summary.csv`)."]
        if pth[0]:
            L.append(f"- Threshold estimate (comparative): {pth[0]:.0f} mW at critical coupling"
                     + (f", {pth[1]:.0f} mW over-coupled" if pth[1] else "") + ".")
        L.append("")
    if gap and not gap.get("skipped"):
        L += ["## 3. Ring–bus gap (3D FDTD)", "",
              f"- A {gap.get('span_angle_deg', 45):g}° arc of the ring (radius {ring.get('radius_um')} "
              "µm) above a straight bus; both have the core cross-section above. The gap is the "
              "edge-to-edge distance between bus and ring at the closest point.",
              f"- Gaps simulated: {', '.join(str(g) for g in gap.get('gaps_um') or [])} µm.",
              "- Excitation: fundamental TE mode launched in the bus. Coupled power per pass: "
              f"{gap.get('reference')} (see `tables/gap_sweep.csv`: coupled_power).",
              "- Fit θ = asin(√coupled) = θ0 exp(−γ g). Critical coupling: sin²θ = 1 − a², with "
              f"a² = round-trip power survival from Q_i (target coupled power "
              f"{gap.get('target_cross_power', float('nan')):.4g}).",
              f"- Results: critical gap {gap['critical_gap_um'] * 1e3:.0f} nm; over-coupled "
              f"(K = {gap['over_coupling']:g}) {(gap.get('over_coupled_gap_um') or 0) * 1e3:.0f} nm.",
              "- To compare: the coupled power at each gap (a full ring in the other solver is "
              "fine: compare the per-pass coupling), then the gaps.",
              _recess_line(gap.get("ring_inner_box_recess_um", 0.1)),
              f"- Recorded at {gap.get('band_points', 1)} wavelength(s) over the band.",
              "- Known numerical feature: 1 − T relative to the source shows a gap-independent "
              "excess of about 0.2 % (bus losses in the simulation, not coupling); the power "
              "entering the ring arc agrees with 1 − T/T(widest gap) within about 1 %."]
        cmt = gap.get("coupled_mode_check") or {}
        if cmt.get("critical_gap_um"):
            L.append(f"- Independent estimate without FDTD (coupled-mode theory from the even/odd "
                     f"supermodes of two cores at the gap, θ = (πΔn/λ)·√(2πR/γ)): critical gap "
                     f"{cmt['critical_gap_um'] * 1e3:.0f} nm, γ = {cmt['gamma_per_um']:.2f} /µm. "
                     "A mode solver alone can reproduce this.")
        L.append("")
    if gc and not gc.get("skipped"):
        L += ["## 4. Grating coupler (2D FDTD)", "",
              f"- Uniform grating, {gc['etch']} etch ({gc['etch_depth_um']} µm), period "
              f"{gc['period_um']} µm, fill factor {gc['fill_factor']} (tooth = fill factor × "
              f"period), {gc['n_teeth']} teeth; the core slab continues to the left (−x) as the "
              "waveguide; the first tooth starts at x = 0.",
              f"- Fibre: Gaussian beam, MFD {gc['mfd_um']} µm, waist at the source plane 2 µm above "
              f"the top of the cladding, {gc['fiber_angle_deg']:g}° from vertical, travelling "
              f"{gc.get('fiber_tilt', 'away')} the waveguide; TE polarisation (E along the "
              f"grating lines); beam centre on the grating surface at x = {gc['source_x_um']} µm.",
              f"- BOX {s.box_height_um} µm on a Si handle (3 µm in the simulation).",
              "- Efficiency: power in the waveguide's fundamental mode (travelling −x) divided by "
              "the beam power, against wavelength.",
              f"- Results: {gc['efficiency_2d'] * 100:.1f} % at {wl} µm" +
              (f"; peak {gc['peak_efficiency_2d'] * 100:.1f} % at {gc['peak_wavelength_um'] * 1e3:.1f} "
               f"nm; 1 dB band {gc['bandwidth_1db_nm']:.0f} nm" if gc.get("peak_efficiency_2d")
               else "") + ". Spectrum: `tables/grating_spectrum_final.csv`; all designs: "
              "`tables/grating_sweep.csv`."]
        if (ctx.out / "tables" / "grating_checks.csv").exists():
            L.append("- Tolerances and the reverse run (mode launched from the waveguide: power "
                     "up, into the substrate, past the grating, and the back-reflection): "
                     "`tables/grating_checks.csv`.")
        L.append("")
    tp, ap, fc = st.get("taper", {}) or {}, st.get("apodise", {}) or {}, st.get("focus", {}) or {}
    if tp.get("grating_width_um"):
        L += ["## 4b. Grating width and taper", "",
              f"- Grating width {tp['grating_width_um']:g} µm: lateral overlap of the fibre Gaussian "
              f"(waist MFD/2) with the grating's fundamental lateral mode, effective-index method "
              f"(slab index {tp['n_slab']:.4f}): η_y = {tp['eta_y'] * 100:.1f} %.",
              "- Taper: linear, from the grating width to the waveguide width, Si3N4 film on the "
              "BOX, air above. Transmission of the fundamental (TE0) mode against length: "
              "`tables/taper_length.csv` (3D EME, graded cells) and `tables/taper_length_model.csv` "
              "(2D effective-index mode matching, converged)."
              + (f" Used: {tp['taper_length_um']:g} µm, {tp['taper_loss_db']:.3f} dB "
                 f"({tp.get('length_source')})." if tp.get("taper_length_um") else ""),
              "- To compare: a 3D EME or FDTD of the same taper at a few lengths (e.g. 100, 200, "
              "300 µm).", ""]
    if ap.get("verified"):
        L += ["## 4c. Apodised grating (2D FDTD)", "",
              "- Explicit tooth list `tables/apodised_teeth.csv` (start and length of every tooth, "
              "first tooth at x = 0, full etch), otherwise as section 4.",
              f"- Result: {ap['efficiency_2d'] * 100:.1f} % at {wl} µm, fibre at {ap['source_x_um']} µm"
              f" (uniform: {ap.get('uniform_efficiency_2d', 0) * 100:.1f} %)"
              + (f"; kept recipe: {ap['recipe']}" if ap.get("recipe") else "") + "."
              + (" Its spectrum: `tables/apodised_spectrum.csv`." if (ctx.out / "tables" /
                 "apodised_spectrum.csv").exists() else ""),
              *(["- Linear fill-factor ramp (Marchetti et al. 2017) as a cross-check: "
                 "`tables/apodised_ramp.csv`."] if ap.get("ramp") else []), ""]
    if fc.get("curves"):
        L += ["## 4d. Focusing grating", "",
              "- Teeth are curves n_B r − n_top sin θ · r cos φ = Φ around the waveguide tip (polar "
              "coordinates r, φ from the tip, axis along the grating); Φ and n_B of every tooth "
              "edge in `tables/focusing_teeth.csv`.",
              f"- Tip {fc['tip_width_um']:g} µm wide ({fc['tip_taper_um']:g} µm linear taper from the "
              f"waveguide), first tooth {fc['distance_um']:g} µm from the tip, teeth limited to "
              f"|y| ≤ {fc['y_max_um']:g} µm.",
              f"- TARANG's estimate: {fc['efficiency_3d_estimate'] * 100:.1f} % (2D efficiency × "
              "in-plane focusing model × tip taper)"
              + (f"; 3D FDTD {fc['efficiency_3d'] * 100:.1f} %" if fc.get("efficiency_3d") is not None
                 else "") + ". The full 3D set-up is `simulations/focusing_grating_3d.json`.", ""]
    used = ("a focusing grating (section 4d)" if fc.get("curves") and cfg.focus.run
            and cfg.focus.use_in_layout else
            f"a {tp.get('taper_length_um') or cfg.grating.taper_length_um:g} µm linear taper and "
            "the straight grating")
    L += ["## 5. Layout", "",
          "`layout/` has the GDS files and `parameters.json` (every dimension). Each end of the "
          f"bus has {used}, as in the simulations.", ""]
    extra = [(n, d) for n, d in (
        ("gap_tolerance.csv", "critical and over-coupled gap with the waveguide width ± its tolerance"),
        ("coupling_vs_wavelength.csv", "coupling ratio K of both gaps across the band (3D FDTD)"),
        ("mode_families.csv", "guided mode families of the bent ring waveguide and their crossings"),
        ("loss_budget.csv", "fibre-to-fibre loss at the design wavelength"),
        ("device_resonances.csv", "circuit model: Q_i, Q_c, Q_L, K, dip and D_int of every resonance, "
                                  "both gaps (compare with a measured transmission scan)"),
        ("device_spectrum.csv", "circuit model: fibre-to-fibre transmission around the pump"),
        ("device_envelope.csv", "circuit model: off-resonance transmission and Fabry-Perot ripple bounds"),
        ("cloud_runs.csv", "every cloud simulation with its link in the Tidy3D web interface"))
        if (ctx.out / "tables" / n).exists()]
    if extra:
        L += ["## 6. Checks and runs", ""] + [f"- `tables/{n}`: {d}" for n, d in extra] + [""]
    if (ctx.out / "lle" / "lle_inputs.json").exists():
        L += ["## 7. Comb (LLE) inputs", "",
              "`lle/lle_inputs.json` (all numbers with units), `lle/pyLLE_dispersion.csv` (mode "
              "number, resonance frequency in Hz) and `lle/integrated_dispersion.csv`.", ""]
    if sim_sections:
        L += ["## Simulation details", "",
              "The files in `simulations/` are the exact simulations TARANG ran; below is what "
              "they contain.", ""] + sim_sections
    L += ["## Every number", "", "`results.csv` (quantity, value, unit, how computed).", ""]
    return L


def _recess_line(recess: float) -> str:
    if recess:
        return (f"- Geometry detail to copy (or expect a small difference): inside the ring, the "
                f"air region of TARANG's partial-ring builder reaches {recess:g} µm into the BOX "
                f"(z from −{recess:g} µm); the bus side of the ring and the gap are unaffected.")
    return "- The BOX top is flat everywhere (no recess inside the ring)."
