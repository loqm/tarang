"""Focusing grating coupler: curved teeth that focus the fibre's light straight into the
waveguide, so a short slab region replaces the long linear taper.

From first principles
---------------------
A straight grating of width W makes a beam as wide as the fibre spot, and a long adiabatic
taper (about 290 um for 14.5 -> 1 um) squeezes it into the waveguide. A focusing grating
does the squeezing with phase instead of length: the guided light leaving the waveguide's
end (the focus F) spreads in the film as a cylindrical wave, and every tooth is curved so
that this wave and the tilted fibre beam stay phase-matched along every direction from F.
With F at the origin, the grating axis along +x, Bloch index n_B and s = n_top sin(theta),
phase matching of the q-th tooth edge is

    n_B r - s x = n_B r - s r cos(phi) = Phi_q,         i.e.   r(phi) = Phi_q / (n_B - s cos(phi))

a family of confocal ellipses with one focus at F (Waldhaeusl et al., Appl. Opt. 36, 9383
(1997); Van Laere et al., IEEE PTL 19, 1919 (2007)). On the axis (phi = 0) the edges sit
where the straight grating has them, so the 2D cross-section, and its 2D FDTD efficiency
eta_2D, are those of the straight grating along every radial line.

What the curvature adds is in the plane of the film: the fibre spot (Gaussian, waist w0 =
MFD/2 across the axis) lies on the arc of radius R_c = d + x_c around F (d: distance from F to
the first tooth, x_c: beam centre from the first tooth). Light scattered into the film on
that arc converges to F; an arc point at angle phi sends a plane wave with in-plane wave
number k_y = k_s sin(phi) (k_s = n_slab k0) towards F, with the amplitude the fibre gives it
there, A(phi) = exp(-(R_c phi / w0)^2), for |phi| <= phi_max (the sector). The fraction that
couples into the waveguide's fundamental mode psi (effective-index method, the tip width
w_tip) is the overlap of the two angular spectra

    eta_focus = |int A(k_y) psi~*(k_y) dk_y|^2 / (int |A|^2 dk_y  int |psi~|^2 dk_y)

(A(k_y) = A(phi) / sqrt(k_s cos phi), so that int |A|^2 dk_y = int |A(phi)|^2 dphi). It
peaks when the cone of light 2 w0 / R_c matches the waveguide's own divergence: a 1 um tip
diverges strongly (best R_c about 10 um), a wider tip less (best R_c grows with the tip
width). Large cone angles are where this model is least reliable (the teeth are crossed
obliquely and the 2D cross-section no longer holds exactly), so the half-angle w0 / R_c is
limited (``focus.max_half_angle_deg``). A tip wider than the waveguide needs a short
linear taper (its loss from the 2D taper model). 3D estimate:

    eta_3D ~ eta_2D (teeth along the axis) x eta_focus x T_tip_taper

The teeth along the axis can be the uniform grating or the apodised one (each tooth with
the Bloch index of its own fill factor). A 3D FDTD of the whole focusing coupler is
available (``focus.check_3d``, priced first).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

C_UM = 2.99792458e14


# ------------------------------------------------------------------ in-plane focusing
def tip_spectrum(n_slab: float, n_side: float, w_tip: float, wl: float, n_k: int = 2001):
    """(k_y, angular spectrum of the tip's fundamental mode, k_s): the tip mode as a sum of
    plane waves in the film (effective-index method)."""
    from tarang.design.taper import lateral_mode
    k_s = 2 * math.pi / wl * n_slab
    half = max(4 * w_tip, 6.0)
    y = np.linspace(-half, half, 2401)
    dy = y[1] - y[0]
    _, psi = lateral_mode(n_slab, n_side, w_tip, wl, y)
    ky = np.linspace(-0.999 * k_s, 0.999 * k_s, n_k)
    psi_k = (np.exp(-1j * np.outer(ky, y)) @ psi) * dy
    return ky, psi_k, k_s


def focus_efficiency(n_slab: float, n_side: float, w_tip: float, wl: float, r_c: float, w0: float,
                     half_angle_max: Optional[float] = None, spectrum=None) -> float:
    """eta_focus: overlap of the light converging from the fibre spot (waist ``w0`` across
    the axis, on the arc of radius ``r_c`` around the focus) with the fundamental mode of a
    tip of width ``w_tip`` (effective-index method). ``half_angle_max``: sector half-angle in
    radians (default 2 w0 / r_c, i.e. the whole spot). ``spectrum``: tip_spectrum(...) of the
    same tip, to reuse it over many r_c."""
    ky, psi_k, k_s = spectrum if spectrum is not None else tip_spectrum(n_slab, n_side, w_tip, wl)
    phi = np.arcsin(ky / k_s)
    pm = half_angle_max if half_angle_max is not None else min(2 * w0 / r_c, math.pi / 2)
    a = np.exp(-(r_c * phi / w0) ** 2) * (np.abs(phi) <= pm) / np.sqrt(k_s * np.cos(phi))
    num = abs(np.trapezoid(a * np.conj(psi_k), ky)) ** 2
    den = np.trapezoid(np.abs(a) ** 2, ky) * np.trapezoid(np.abs(psi_k) ** 2, ky)
    return float(num / den) if den > 0 else 0.0


def tip_taper(n_slab: float, n_side: float, w_tip: float, w_wg: float, wl: float,
              max_loss_db: float) -> tuple[float, float]:
    """(length, transmission) of the shortest linear taper from the tip width down to the
    waveguide width with loss below ``max_loss_db`` (2D taper model); (0, 1) when the tip is
    the waveguide itself."""
    from tarang.design.taper import eim_taper, loss_at, shortest_length
    if w_tip <= w_wg + 1e-9:
        return 0.0, 1.0
    lengths = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30]
    curve = eim_taper(n_slab, n_side, w_tip, w_wg, wl, lengths, n_cells=200, n_modes=20,
                      margin=4.0)
    L = shortest_length(curve, max_loss_db)
    if L is None:
        L = float(max(lengths))
    L = float(math.ceil(L))
    return L, min(1.0, 10 ** (-loss_at(curve, L) / 10))


def landscape(n_slab: float, n_side: float, wl: float, w0: float, x_c: float, w_wg: float,
              tip_widths, distances, max_loss_db: float, max_half_angle: float) -> dict:
    """eta_focus x T_tip over (distance focus -> first tooth, tip width); the cone half-angle
    w0 / R_c decides which points are allowed."""
    rows = []
    taper = {w: tip_taper(n_slab, n_side, w, w_wg, wl, max_loss_db) for w in tip_widths}
    for w in tip_widths:
        Lt, T = taper[w]
        spec = tip_spectrum(n_slab, n_side, w, wl)
        for d in distances:
            r_c = d + x_c
            eta = focus_efficiency(n_slab, n_side, w, wl, r_c, w0, spectrum=spec)
            rows.append(dict(tip_width_um=float(w), distance_um=float(d), r_c_um=float(r_c),
                             half_angle_deg=math.degrees(w0 / r_c), eta_focus=eta,
                             tip_taper_um=Lt, tip_taper_T=T, eta_lateral=eta * T,
                             allowed=bool(w0 / r_c <= max_half_angle)))
    return dict(rows=rows, tip_taper=taper)


# ------------------------------------------------------------------ geometry
def ellipse_r(phi, Phi: float, n_b: float, s: float):
    """Radius of the phase-matched curve n_B r - s r cos(phi) = Phi."""
    return Phi / (n_b - s * np.cos(phi))


def _clip_angle(Phi: float, n_b: float, s: float, phi_max: float, y_max: float) -> float:
    """Largest |phi| <= phi_max at which the curve stays within |y| <= y_max."""
    def y(p):
        return float(ellipse_r(p, Phi, n_b, s) * math.sin(p))
    if y(phi_max) <= y_max:
        return phi_max
    lo, hi = 0.0, phi_max
    for _ in range(60):
        m = 0.5 * (lo + hi)
        if y(m) > y_max:
            hi = m
        else:
            lo = m
    return lo


def _arc(Phi, n_b, s, phi_lim, n_pts, reverse=False):
    p = np.linspace(-phi_lim, phi_lim, n_pts)
    if reverse:
        p = p[::-1]
    r = ellipse_r(p, Phi, n_b, s)
    return list(zip((r * np.cos(p)).tolist(), (r * np.sin(p)).tolist()))


def teeth_curves(teeth, distance: float, wl: float, s: float, period_of_tooth=None):
    """For axial teeth ((start, length), ... from the first tooth) placed ``distance`` from
    the focus: (Phi_start, Phi_end, n_B) of every tooth. Each tooth's Bloch index comes from
    its own period (phase matching: n_B = wl / period + s), so apodised teeth keep their
    local phase matching."""
    out = []
    for i, (a, b) in enumerate(teeth):
        if period_of_tooth is not None:
            per = period_of_tooth(i)
        elif i + 1 < len(teeth):
            per = teeth[i + 1][0] - a
        else:
            per = a - teeth[i - 1][0] if i else b
        n_b = wl / per + s
        xs, xe = distance + a, distance + a + b
        out.append(((n_b - s) * xs, (n_b - s) * xe, n_b))
    return out


def polygons(teeth, distance: float, w_tip: float, wl: float, s: float, half_angle: float,
             y_max: float, n_pts: int = 48, period_of_tooth=None) -> dict:
    """Polygons of the focusing grating with the focus at (0, 0) and the axis along +x:
    ``slab`` (the film from the tip to the first tooth, a sector) and ``teeth`` (one
    curved tooth each, between its two phase-matched curves). The sector is limited to
    |phi| <= half_angle and |y| <= y_max (the fibre spot)."""
    curves = teeth_curves(teeth, distance, wl, s, period_of_tooth)
    Phi0, _, nb0 = curves[0]
    p0 = _clip_angle(Phi0, nb0, s, half_angle, y_max)
    slab = [(0.0, -w_tip / 2)] + _arc(Phi0, nb0, s, p0, n_pts) + [(0.0, w_tip / 2)]
    polys = []
    for Ps, Pe, nb in curves:
        ps = _clip_angle(Ps, nb, s, half_angle, y_max)
        pe = _clip_angle(Pe, nb, s, half_angle, y_max)
        polys.append(_arc(Ps, nb, s, ps, n_pts) + _arc(Pe, nb, s, pe, n_pts, reverse=True))
    return dict(slab=slab, teeth=polys, curves=curves)


# ------------------------------------------------------------------ 3D check
def focusing_sim(ctx, design: dict):
    """Tidy3D 3D simulation of the whole focusing coupler: waveguide (and tip taper),
    sector slab, curved teeth (full etch), BOX, handle; fibre beam as in the grating stage;
    the fundamental mode of the waveguide is recorded."""
    import tidy3d as td
    from tarang.design.stages import grating_device
    from tarang.grating_coupler.grating_coupler import GratingCoupler
    cfg = ctx.cfg
    gc = ctx.results("grating")
    H = ctx.results("cross_section")["height_um"]
    W = ctx.results("cross_section")["width_um"]
    teeth = [tuple(t) for t in design["teeth"]]
    d, w_tip, Lt = design["distance_um"], design["tip_width_um"], design["tip_taper_um"]
    sx = design["source_x_um"]
    base = grating_device(ctx, gc["period_um"], gc["fill_factor"], sx, H=H,
                          depth=gc["etch_depth_um"], teeth=teeth)
    src2d = base.sim.sources[0]
    core = ctx.medium(cfg.materials.core)
    box = ctx.medium(cfg.materials.box)
    handle = None if str(cfg.materials.handle).lower() == "none" else ctx.medium(cfg.materials.handle)
    clad = (td.Medium(permittivity=1.0) if str(cfg.materials.cladding).lower() == "air"
            else ctx.medium(cfg.materials.cladding))
    g = polygons(teeth, d, w_tip, cfg.run.wavelength_um, design["s"], design["half_angle_rad"],
                 design["y_max_um"])
    shift = -d                                          # first tooth at x = 0, focus at x = -d
    big = 1e3

    def slab(points, name):
        pts = [(x + shift, y) for x, y in points]
        return td.Structure(geometry=td.PolySlab(vertices=pts, slab_bounds=(0, H), axis=2),
                            medium=core, name=name)
    x_tip = shift
    structs = [td.Structure(geometry=td.Box.from_bounds((-big, -big, -cfg.stack.box_height_um),
                                                         (big, big, 0)), medium=box, name="BOX")]
    if handle is not None:
        structs.insert(0, td.Structure(geometry=td.Box.from_bounds(
            (-big, -big, -cfg.stack.box_height_um - 3.0), (big, big, -cfg.stack.box_height_um)),
            medium=handle, name="Substrate"))
    x_wg_end = x_tip - Lt
    structs.append(td.Structure(geometry=td.Box.from_bounds((-big, -W / 2, 0), (x_wg_end, W / 2, H)),
                                medium=core, name="Waveguide"))
    if Lt > 0:
        structs.append(td.Structure(geometry=td.PolySlab(
            vertices=[(x_wg_end, -W / 2), (x_tip, -w_tip / 2), (x_tip, w_tip / 2), (x_wg_end, W / 2)],
            slab_bounds=(0, H), axis=2), medium=core, name="Tip_taper"))
    structs.append(slab(g["slab"], "Focusing_slab"))
    structs += [slab(p, f"Tooth_{i}") for i, p in enumerate(g["teeth"])]
    allx = [x + shift for p in g["teeth"] for x, _ in p]
    x_max = max(allx) + 3.0
    y_half = design["y_max_um"] + 2.5
    x_min = x_wg_end - 3.0
    z_min = -cfg.stack.box_height_um - (3.0 if handle is not None else 1.0)
    z_max = H + 2.0 + 1.5
    src = src2d.updated_copy(size=(2 * cfg.grating.mfd_um, 2 * cfg.grating.mfd_um, 0))
    allf = np.asarray(next(m for m in base.sim.monitors if m.name == "mode monitor").freqs)
    freqs = list(allf[::5])                             # 21 wavelengths over the band
    f0 = C_UM / cfg.run.wavelength_um
    from tarang.geometries._modes import core_target_neff
    mon = td.ModeMonitor(center=(x_wg_end - 1.5, 0, H / 2), size=(0, W + 4.0, 6 * H), freqs=freqs,
                         mode_spec=td.ModeSpec(num_modes=1, target_neff=core_target_neff(core, f0)),
                         name="mode monitor")
    plane = td.FieldMonitor(center=((x_min + x_max) / 2, 0, H / 2), size=(td.inf, td.inf, 0),
                            freqs=[f0], name="field_z_plane")
    dl = cfg.run.wavelength_um / (2.0 * 12)          # 12 steps per wavelength in the core
    override = td.MeshOverrideStructure(
        geometry=td.Box.from_bounds((x_min, -y_half, -0.5), (x_max, y_half, H + 0.5)),
        dl=[dl, dl, dl], enforce=True)
    sim = td.Simulation(
        center=((x_min + x_max) / 2, 0, (z_min + z_max) / 2),
        size=(x_max - x_min, 2 * y_half, z_max - z_min),
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=8, wavelength=cfg.run.wavelength_um,
                                   override_structures=[override]),
        structures=structs, sources=[src], monitors=[mon, plane], run_time=base.sim.run_time,
        boundary_spec=td.BoundarySpec.all_sides(td.PML()), medium=clad)

    class FocusingCoupler(GratingCoupler):
        def _readable_name(self) -> str:
            import hashlib
            sig = (f"{d:.3f}|{w_tip:.3f}|{Lt:.2f}|" +
                   ",".join(f"{a:.4f}:{b:.4f}" for a, b in teeth))
            h = hashlib.sha256(sig.encode()).hexdigest()[:6]
            return (f"gc3D_focus{len(teeth)}_{h}_d{d:.1f}_tip{w_tip:g}_sx{sx:.2f}_"
                    f"{cfg.run.wavelength_um:.3f}um")
    return FocusingCoupler(sim=sim, config=base.config, core_medium=core, box_medium=box,
                           handle_medium=handle, top_cladding_medium=clad, is_3d=True)


# ------------------------------------------------------------------ the design stage
def _form(ctx) -> None:
    from tarang.design.ui import parse_list
    on = lambda c: bool(c.focus.run)                        # noqa: E731
    ctx.ui.form(ctx.cfg, "Focusing grating", [
        dict(section="focus", key="run"),
        dict(section="focus", key="teeth", show=on),
        dict(section="focus", key="tip_widths_um", cast=parse_list, show=on),
        dict(section="focus", key="max_half_angle_deg", show=on),
        dict(section="focus", key="check_3d", show=on),
        dict(section="focus", key="use_in_layout", show=on),
        dict(section="run", key="max_cost_per_run", show=on),
        dict(section="run", key="max_total_cost", show=on)])


def _teeth_choice(ctx) -> tuple[str, list, float, float]:
    """(kind, axial teeth, eta_2D, beam centre from the first tooth) of the grating to
    curve: the apodised one when asked for (or 'best' and it beats the uniform one)."""
    cfg = ctx.cfg
    gc = ctx.results("grating")
    ap = ctx.state["stages"].get("apodise", {}).get("results") or {}
    ap_ok = bool(ap.get("verified")) and ap.get("teeth")
    want = str(cfg.focus.teeth).lower()
    if want == "apodised" and not ap_ok:
        ctx.ui.say("  No checked apodised grating (apodise stage); the uniform teeth are curved.")
    if ap_ok and (want == "apodised" or (want == "best" and ap.get("better_than_uniform"))):
        return "apodised", [tuple(t) for t in ap["teeth"]], ap["efficiency_2d"], ap["source_x_um"]
    P, ff = gc["period_um"], gc["fill_factor"]
    teeth = [(round(i * P, 4), round(ff * P, 4)) for i in range(int(gc["n_teeth"]))]
    return "uniform", teeth, gc["efficiency_2d"], gc["source_x_um"]


def stage_focus(ctx) -> dict:
    """Focusing grating: in-plane focusing landscape (free), curved teeth for the layout,
    optional 3D FDTD check (priced)."""
    from tarang.design import physics, visuals
    from tarang.design.checks import _run_batch, _write_csv
    _form(ctx)
    cfg, ui = ctx.cfg, ctx.ui
    if not cfg.focus.run:
        ui.say("Focusing grating not requested (focus.run = false): the layout keeps the straight "
               "grating and its taper. Turn it on in this table (or design.toml) whenever you want "
               "it; nothing else is repeated.")
        return dict(disabled=True)
    gc = ctx.state["stages"].get("grating", {}).get("results") or {}
    if not gc or gc.get("skipped"):
        ui.say("The grating stage has no design yet; run it first.")
        return dict(skipped=True)
    wl = cfg.run.wavelength_um
    H = ctx.results("cross_section")["height_um"]
    W = ctx.results("cross_section")["width_um"]
    n_core, n_box = ctx.n(cfg.materials.core), ctx.n(cfg.materials.box)
    n_top = 1.0 if str(cfg.materials.cladding).lower() == "air" else ctx.n(cfg.materials.cladding)
    n_slab = physics.slab_te_index(n_core, n_box, n_top, H, wl)
    n_side = max(n_box, n_top)
    w0 = cfg.grating.mfd_um / 2
    s = n_top * math.sin(math.radians(gc["fiber_angle_deg"]))
    if gc.get("fiber_tilt", "towards") != "towards":
        ui.say("  The focusing law here is written for a beam travelling towards the waveguide "
               "(grating.fiber_tilt = 'towards'); this design uses 'away'. Not designed.")
        return dict(skipped=True)
    kind, teeth, eta2d, x_c = _teeth_choice(ctx)
    max_ha = math.radians(float(cfg.focus.max_half_angle_deg))
    tips = sorted({float(v) for v in (cfg.focus.tip_widths_um or [W, 1.5 * W, 2 * W, 3 * W, 4 * W])})
    tips = [w for w in tips if w >= W - 1e-9]
    dists = np.round(np.arange(0.0, 40.01, 1.0), 2).tolist()
    ui.say(f"In-plane focusing (free): the fibre spot (waist {w0:g} µm across the axis) on the arc "
           f"around the waveguide tip, against the tip's mode (effective index, slab {n_slab:.4f}); "
           f"tip widths {', '.join(f'{w:g}' for w in tips)} µm; cone half-angle at most "
           f"{cfg.focus.max_half_angle_deg:g}°.")
    land = landscape(n_slab, n_side, wl, w0, x_c, W, tips, dists, float(cfg.taper.max_loss_db), max_ha)
    rows = land["rows"]
    for r in rows:
        r["eta_3d_estimate"] = eta2d * r["eta_lateral"]
    _write_csv(ctx.path("tables", "focus_landscape.csv"), rows)
    ok = [r for r in rows if r["allowed"]]
    if not ok:
        ui.say("  No distance within the cone limit; increase focus.max_half_angle_deg.")
        return dict(skipped=True, landscape=rows)
    best = max(ok, key=lambda r: (round(r["eta_lateral"], 3), -r["distance_um"] - r["tip_taper_um"]))
    d, w_tip, Lt = best["distance_um"], best["tip_width_um"], best["tip_taper_um"]
    y_max = 1.6 * w0                       # the fibre spot holds > 99.4 % of its power here
    half = min(math.asin(min(1.0, y_max / max(best["r_c_um"], 1e-9))) + 0.15, math.radians(60))
    geo = polygons(teeth, d, w_tip, wl, s, half, y_max)
    tp = ctx.state["stages"].get("taper", {}).get("results") or {}
    straight_lat = (tp.get("eta_y") or 1.0) * (10 ** (-(tp.get("taper_loss_db") or 0.0) / 10))
    L_g = teeth[-1][0] + teeth[-1][1]
    res = dict(teeth_kind=kind, teeth=[list(t) for t in teeth], efficiency_2d=eta2d,
               source_x_um=x_c, n_slab=n_slab, s=s, distance_um=d, tip_width_um=w_tip,
               tip_taper_um=Lt, r_c_um=best["r_c_um"], half_angle_rad=half, y_max_um=y_max,
               eta_focus=best["eta_focus"], tip_taper_T=best["tip_taper_T"],
               eta_lateral=best["eta_lateral"], efficiency_3d_estimate=eta2d * best["eta_lateral"],
               straight_lateral=straight_lat,
               straight_length_um=(tp.get("taper_length_um") or cfg.grating.taper_length_um) + L_g,
               focusing_length_um=Lt + d + L_g,
               curves=[list(c) for c in geo["curves"]], use_in_layout=bool(cfg.focus.use_in_layout))
    ui.say(f"  Best allowed: tip {w_tip:g} µm (taper {Lt:g} µm from {W:g} µm), first tooth {d:g} µm "
           f"from the tip (beam centre {best['r_c_um']:.1f} µm, cone ±{best['half_angle_deg']:.0f}°): "
           f"focusing {best['eta_focus'] * 100:.1f} %, with the tip taper {best['eta_lateral'] * 100:.1f} %.")
    ui.say(f"  3D estimate with the {kind} teeth: {eta2d * 100:.1f} % x {best['eta_lateral']:.3f} = "
           f"{res['efficiency_3d_estimate'] * 100:.1f} %; straight grating with its taper: "
           f"{eta2d * 100:.1f} % x {straight_lat:.3f} = {eta2d * straight_lat * 100:.1f} %. Length "
           f"from the waveguide to the grating's end: {res['focusing_length_um']:.0f} µm against "
           f"{res['straight_length_um']:.0f} µm.")
    _write_csv(ctx.path("tables", "focusing_teeth.csv"),
               [dict(tooth=i, axial_start_um=round(d + a, 4), axial_length_um=b,
                     phi_start=round(c[0], 5), phi_end=round(c[1], 5), n_bloch=round(c[2], 5))
                for i, ((a, b), c) in enumerate(zip(teeth, geo["curves"]))])
    visuals.safe(ctx, "focusing landscape", draw_landscape, ctx, rows, best, kind, eta2d)
    visuals.safe(ctx, "focusing geometry", draw_top_view, ctx, geo, res, W)
    if not cfg.focus.check_3d:
        return res
    dev = focusing_sim(ctx, res)
    folder = ctx.out / "cloud" / "focus"
    ok3 = _run_batch(ctx, {"focusing grating 3D": (dev.sim, dev._readable_name())}, folder,
                     "focusing grating in 3D FDTD")
    if not ok3:
        return res
    try:
        from tarang.grating_coupler.grating_coupler import spectrum_summary
        wls, eta = dev.get_spectrum(folder)
        sm = spectrum_summary(wls, eta, wl)
        res.update(efficiency_3d=sm["efficiency_at_target"], peak_efficiency_3d=sm["peak_efficiency"],
                   peak_wavelength_3d_nm=round(sm["peak_wavelength_um"] * 1e3, 1))
        ui.say(f"  3D FDTD: {sm['efficiency_at_target'] * 100:.1f} % at {wl} µm (estimate "
               f"{res['efficiency_3d_estimate'] * 100:.1f} %), peak at {res['peak_wavelength_3d_nm']} nm.")
    except Exception as err:                                      # noqa: BLE001
        ui.say(f"  (3D result not read: {type(err).__name__}: {err})")
    return res


# ------------------------------------------------------------------ figures
def draw_landscape(ctx, rows, best, kind, eta2d):
    """Figure 10a: estimated 3D efficiency of the focusing grating over (distance from the
    tip to the first tooth, tip width), with the region beyond the cone limit hatched."""
    from tarang.design import landscapes
    pts = [dict(x=r["distance_um"], y=r["tip_width_um"], z=r["eta_3d_estimate"],
                allowed=r["allowed"]) for r in rows]
    return landscapes.draw(
        ctx.path("figures", "10a_focusing_landscape.png"), [dict(
            points=pts, title=f"Focusing grating ({kind} teeth): estimated 3D efficiency\n"
                              f"η_2D {eta2d * 100:.1f} % × in-plane focusing × tip taper",
            xlabel="Distance from the waveguide tip to the first tooth (µm)",
            ylabel="Tip width (µm)", chosen=(best["distance_um"], best["tip_width_um"]), model=True,
            chosen_label=f"chosen: {best['eta_3d_estimate'] * 100:.1f} %",
            note="hatched: cone half-angle above focus.max_half_angle_deg (model less reliable)")],
        suptitle="Focusing grating landscape (free model; every grid point computed)")


def draw_top_view(ctx, geo, res, W):
    """Figure 10b: top view of the focusing coupler (as drawn in the GDS)."""
    from tarang.design import visuals
    plt = visuals._plt()
    fig, ax = plt.subplots(figsize=(10, 5))
    Lt, d = res["tip_taper_um"], res["distance_um"]
    ax.fill([-Lt - 8, -Lt, -Lt, -Lt - 8], [-W / 2, -W / 2, W / 2, W / 2], color="#0072B2")
    if Lt > 0:
        wt = res["tip_width_um"]
        ax.fill([-Lt, 0, 0, -Lt], [-W / 2, -wt / 2, wt / 2, W / 2], color="#0072B2")
    xs, ys = zip(*geo["slab"])
    ax.fill(xs, ys, color="#0072B2", alpha=0.85)
    for p in geo["teeth"]:
        xs, ys = zip(*p)
        ax.fill(xs, ys, color="#0072B2")
    xc = d + res["source_x_um"]
    t = np.linspace(0, 2 * np.pi, 200)
    w0 = ctx.cfg.grating.mfd_um / 2
    wx = w0 / math.cos(math.radians(ctx.results("grating")["fiber_angle_deg"]))
    ax.plot(xc + wx * np.cos(t), w0 * np.sin(t), "--", color="#D55E00",
            label="fibre spot (1/e field)")
    ax.plot([0], [0], "k+", ms=10, label="focus (waveguide tip)")
    ax.set_aspect("equal")
    ax.set(xlabel="x (µm)", ylabel="y (µm)",
           title=f"Focusing grating, {res['teeth_kind']} teeth: {len(geo['teeth'])} curved teeth, "
                 f"full etch (blue: Si3N4 left after the etch)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    return visuals._save(fig, ctx.path("figures", "10b_focusing_geometry.png"))
