"""Edge coupler with a wide facet, for a lensed fibre in the chip plane (run.coupling = "edge").

From first principles:

1. **What couples.** The lensed fibre focuses a nearly Gaussian spot, waist w0 = MFD / 2, on
   the chip facet. The waveguide's fundamental mode leaves the facet as a field (E1, H1);
   by reciprocity the power the fibre puts into that mode is the power of the emitted field
   that falls into the fibre mode (E2, H2), relative to the power sent towards the facet:

       eta = |a|^2 / P_in,   a = 1/4 * integral (E1 x H2* + E2* x H1) . x dA,

   with the fibre mode normalised to unit power. Where the plane is close to the facet and
   both fields are nearly transverse this is the familiar butt-coupling estimate

       eta ~ T_F * |integral E1 E2* dA|^2 / (integral |E1|^2 dA * integral |E2|^2 dA),

   T_F = 1 - ((n_eff - 1) / (n_eff + 1))^2 the Fresnel transmission of the facet.

2. **Why a wide facet.** The 700 nm film fixes the mode's height (well below the 3.5 um
   spot), so the vertical overlap is what it is; the facet width sets the lateral overlap.
   A 1 um waveguide is far narrower than the spot; widening the waveguide towards the facet
   spreads its mode sideways to match the spot. Too wide, and the mode becomes wider than
   the spot (and the guide carries more higher-order modes, which the adiabatic taper must
   not excite).

3. **Taper.** Between the 1 um waveguide and the facet width the taper must change slowly
   enough for the fundamental mode to follow (adiabatic). The beat length to the next even
   mode grows as the guide widens, so a taper may widen quickly where it is narrow and must
   widen slowly where it is wide: the parabolic (w^2 linear in length) and power-law
   (w ~ s^m, m < 1) shapes reach the same loss in a shorter length than a linear one.
   Loss against length comes from a converged 2D model (effective-index method, free) and
   Tidy3D's eigenmode expansion (3D, priced), as for the grating taper (taper.py).

4. **Facet reflection.** The facet reflects about ((n_eff - 1) / (n_eff + 1))^2 of the guided
   power back into the waveguide (about 7 % for n_eff = 1.75). With two facets the chip is a
   Fabry-Perot cavity: the transmission ripples with period lambda^2 / (2 n_g L). The
   whole-device model (circuit.py) includes it.

The 3D FDTD check launches the waveguide mode towards the facet (the waveguide and its BOX
run through the absorbing boundary behind the source, so nothing reflects from a waveguide
end) and records the field 0.5 um past the facet. The coupling for any fibre
offset, spot size and working distance follows from that one run (the overlap above with a
Gaussian placed anywhere), and the facet reflection from the mode amplitude travelling back.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

from tarang.design.taper import (PROFILES, eim_taper, loss_at, rise_db, shortest_length,
                                 slab_index_te)

C_UM = 2.99792458e14
ETA0 = 376.730313668                 # impedance of free space (ohm)
FIBRE_PLANE_UM = 0.5                 # 3D run: field recorded this far past the facet
EDGE_LIMITS = [
    "Edge coupler: the facet is flat and perpendicular to the waveguide (as cleaved or polished; "
    "no angle, no anti-reflection coating); the lensed fibre's spot is a round Gaussian with its "
    "waist at the working distance; TE polarisation.",
    "Edge coupler: without the 3D run the coupling is the butt-coupling estimate (mode overlap "
    "times the Fresnel transmission at n_eff) and the facet reflection the Fresnel value at "
    "n_eff; the taper loss is that of the design wavelength across the band."]


# ============================================================================ the fibre spot
def gaussian(y, z, y0: float, z0: float, w0: float, wl: float, dist: float = 0.0,
             sign: float = 1.0, n: float = 1.0) -> np.ndarray:
    """Paraxial Gaussian beam (E along y) on the plane (y, z), centred at (y0, z0), waist w0,
    the plane ``dist`` past the waist (negative: before it, the beam still converging);
    normalised to integral |u|^2 dA = 1 over the whole plane. ``sign``: +1 for the exp(-i w t)
    time convention (a diverging wave has the phase + k r^2 / 2R), -1 for exp(+i w t)."""
    k = 2 * math.pi * n / wl
    zr = k * w0 ** 2 / 2
    w = w0 * math.sqrt(1 + (dist / zr) ** 2)
    yy, zz = np.meshgrid(np.asarray(y, float) - y0, np.asarray(z, float) - z0, indexing="ij")
    r2 = yy ** 2 + zz ** 2
    u = math.sqrt(2 / math.pi) / w * np.exp(-r2 / w ** 2)
    if dist != 0.0:
        R = dist * (1 + (zr / dist) ** 2)
        u = u * np.exp(1j * sign * k * r2 / (2 * R))
    return u


def _weights(y, z) -> np.ndarray:
    """Area of each grid point (non-uniform grids: half the distance to each neighbour)."""
    def w1(v):
        v = np.asarray(v, float)
        if len(v) < 2:
            return np.ones_like(v)
        e = np.concatenate([[v[0]], 0.5 * (v[1:] + v[:-1]), [v[-1]]])
        return np.diff(e)
    return np.outer(w1(y), w1(z))


def scalar_overlap(Ey, Ez, y, z, u) -> float:
    """|integral Ey u* dA|^2 / (integral (|Ey|^2 + |Ez|^2) dA * integral |u|^2 dA) with u
    normalised over the whole plane (gaussian()): the butt-coupling overlap of a TE field."""
    dA = _weights(y, z)
    num = abs(np.sum(Ey * np.conj(u) * dA)) ** 2
    den = np.sum((np.abs(Ey) ** 2 + np.abs(Ez) ** 2) * dA)
    return float(num / den) if den > 0 else 0.0


def vector_overlap(Ey, Ez, Hy, Hz, y, z, u, n: float = 1.0) -> tuple[float, float]:
    """(|a|^2, P) on a plane normal to x: the power the field (E, H) puts into the Gaussian
    mode u travelling along +x in a medium of index n (E2 = A u y, H2 = A n u / eta0 z, unit
    power), a = 1/4 integral (E1y H2z* + E2y* H1z) dA, and the power P = 1/2 Re integral
    (E1y H1z* - E1z H1y*) dA crossing the plane. Fields in V/um and A/um (Tidy3D)."""
    dA = _weights(y, z)
    A = math.sqrt(2 * ETA0 / n)
    a = 0.25 * A * np.sum(np.conj(u) * (Ey * n / ETA0 + Hz) * dA)
    P = 0.5 * float(np.real(np.sum((Ey * np.conj(Hz) - Ez * np.conj(Hy)) * dA)))
    return float(abs(a) ** 2), P


def best_height(Ey, Ez, y, z, w0: float, wl: float, z_lo: float, z_hi: float,
                step: float = 0.02) -> tuple[float, float]:
    """(best overlap, fibre height z0) for a Gaussian centred at y = 0, z0 in [z_lo, z_hi]."""
    zs = np.arange(z_lo, z_hi + step / 2, step)
    vals = [scalar_overlap(Ey, Ez, y, z, gaussian(y, z, 0.0, z0, w0, wl)) for z0 in zs]
    i = int(np.argmax(vals))
    return float(vals[i]), float(zs[i])


def fresnel(n_eff: float) -> float:
    """Power reflection of the facet at normal incidence for a mode of index n_eff."""
    return ((n_eff - 1) / (n_eff + 1)) ** 2


# ============================================================================ facet modes
def facet_mode(ctx, width: float, wl: Optional[float] = None, steps: Optional[int] = None,
               with_handle: bool = True) -> dict:
    """TE00 mode of the facet cross-section (the film at ``width``, BOX and the Si handle
    under it; local solve, free). The Si handle carries many slab modes, so the mode is
    picked as the TE mode with the highest index among those confined to the core."""
    import tidy3d as td
    from tidy3d.plugins.mode import ModeSolver
    from tarang.waveguide.dispersion_mapping import core as dm
    cfg = ctx.cfg
    wl = wl or cfg.run.wavelength_um
    H = ctx.results("cross_section")["height_um"]
    handle = with_handle and str(cfg.materials.handle).lower() != "none"
    sim = dm.build_sim(H, width, wl0=wl, steps=int(steps or cfg.cross_section.map_steps),
                       box=cfg.stack.box_height_um, core=cfg.materials.core,
                       substrate=cfg.materials.box, cladding=cfg.materials.cladding,
                       cladding_height=cfg.stack.cladding_height_um, band=ctx.band,
                       margin_y=3.0, si_handle=handle)
    n_slab = slab_index_te(ctx.n(cfg.materials.core), ctx.n(cfg.materials.box),
                           1.0 if str(cfg.materials.cladding).lower() == "air"
                           else ctx.n(cfg.materials.cladding), H, wl)
    plane = td.Box(center=(0, sim.center[1], sim.center[2]), size=(0, sim.size[1], sim.size[2]))
    data = ModeSolver(simulation=sim, plane=plane, freqs=[C_UM / wl],
                      mode_spec=td.ModeSpec(num_modes=8, target_neff=n_slab,
                                            num_pml=(12, 12) if handle else (0, 0))).solve()
    y = data.Ey.coords["y"].values
    z = data.Ey.coords["z"].values
    inside = ((z >= 0) & (z <= H))[None, :] & (np.abs(y) <= width / 2)[:, None]
    best = None
    for m in range(data.n_eff.sizes["mode_index"]):
        f = {c: np.squeeze(getattr(data, c).isel(mode_index=m, f=0).values)
             for c in ("Ex", "Ey", "Ez", "Hy", "Hz")}
        ey2, ez2 = float((np.abs(f["Ey"]) ** 2).sum()), float((np.abs(f["Ez"]) ** 2).sum())
        E2 = np.abs(f["Ex"]) ** 2 + np.abs(f["Ey"]) ** 2 + np.abs(f["Ez"]) ** 2
        conf = float(E2[inside].sum() / max(E2.sum(), 1e-300))
        te = ey2 / (ey2 + ez2 + 1e-300)
        neff = float(np.real(data.n_eff.isel(mode_index=m, f=0)))
        # TE00 is even across the width: Ey at +y and -y have the same sign
        even = float(np.sum(np.real(f["Ey"] * np.conj(f["Ey"][::-1, :])))) > 0
        if te > 0.5 and conf > 0.3 and even and (best is None or neff > best["n_eff"]):
            best = dict(n_eff=neff, k_eff=float(np.real(data.k_eff.isel(mode_index=m, f=0))),
                        confinement=conf, y=y, z=z, **f)
    if best is None:
        raise RuntimeError(f"no confined TE00 mode found at facet width {width} um")
    best.update(width_um=float(width), wavelength_um=float(wl), height_um=float(H))
    return best


def _cache_file(ctx) -> Path:
    return ctx.out / "results" / "edge_facet_map.json"


def facet_map(ctx, widths, mfds) -> list[dict]:
    """Butt-coupling estimate for every facet width and spot size: TE00 of the facet (with
    the Si handle), its best overlap with the spot over the fibre height, the Fresnel
    transmission at n_eff. Solved modes are kept in results/edge_facet_map.json."""
    cfg = ctx.cfg
    wl = cfg.run.wavelength_um
    H = ctx.results("cross_section")["height_um"]
    steps = int(cfg.cross_section.map_steps)
    path = _cache_file(ctx)
    try:
        cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except ValueError:
        cache = {}
    rows = []
    for i, W in enumerate(widths):
        key = f"H{H:.4f}_W{W:.4f}_wl{wl:.4f}_s{steps}_{cfg.materials.handle}"
        entry = cache.get(key, {})
        need = [m for m in mfds if f"{m:.3f}" not in entry.get("mfd", {})]
        if need or "n_eff" not in entry:
            ctx.ui.say(f"    facet {W:g} µm ({i + 1}/{len(widths)}) ...")
            md = facet_mode(ctx, W, wl, steps)
            entry.update(n_eff=md["n_eff"], k_eff=md["k_eff"], confinement=md["confinement"])
            entry.setdefault("mfd", {})
            for m in mfds:
                ov, z0 = best_height(md["Ey"], md["Ez"], md["y"], md["z"], m / 2, wl,
                                     -0.6 * m / 2, H + 0.6 * m / 2)
                entry["mfd"][f"{m:.3f}"] = [ov, z0]
            cache[key] = entry
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
        T = 1 - fresnel(entry["n_eff"])
        row = dict(width_um=float(W), n_eff=round(entry["n_eff"], 5),
                   fresnel_reflection_db=round(10 * math.log10(fresnel(entry["n_eff"])), 2))
        for m in mfds:
            ov, z0 = entry["mfd"][f"{m:.3f}"]
            row[f"eta_mfd{m:g}"] = ov * T
            row[f"z0_mfd{m:g}_um"] = round(z0, 3)
        rows.append(row)
    return rows


def choose_width(rows: list[dict], mfds) -> float:
    """The facet width with the best average coupling over the spot sizes (the nominal MFD
    and its tolerance), refined by a parabola through the best three widths and rounded to
    0.25 um."""
    W = np.array([r["width_um"] for r in rows])
    avg = np.array([np.mean([r[f"eta_mfd{m:g}"] for m in mfds]) for r in rows])
    i = int(np.argmax(avg))
    best = W[i]
    if 0 < i < len(W) - 1:
        p = np.polyfit(W[i - 1:i + 2], avg[i - 1:i + 2], 2)
        if p[0] < 0:
            best = float(np.clip(-p[1] / (2 * p[0]), W[i - 1], W[i + 1]))
    return float(round(best * 4) / 4)


def default_widths(w_wg: float, mfd: float) -> list[float]:
    hi = max(2 * mfd, w_wg + 2.0)
    return [round(float(v), 2) for v in np.arange(w_wg, hi + 1e-9, 0.5)]


# ============================================================================ taper
DEFAULT_LENGTHS = [5, 10, 15, 20, 30, 40, 60, 80, 100, 150, 200, 300]


def taper_models(ctx, w_facet: float, w_wg: float, profiles, m: float) -> dict:
    """Converged 2D model (effective-index method) of each taper shape: loss against length."""
    cfg = ctx.cfg
    wl = cfg.run.wavelength_um
    H = ctx.results("cross_section")["height_um"]
    n_core, n_box = ctx.n(cfg.materials.core), ctx.n(cfg.materials.box)
    n_top = 1.0 if str(cfg.materials.cladding).lower() == "air" else ctx.n(cfg.materials.cladding)
    ns = slab_index_te(n_core, n_box, n_top, H, wl)
    nside = max(n_box, n_top)
    dense = sorted(set(np.round(np.geomspace(2.0, 400.0, 50), 1).tolist()) |
                   {float(v) for v in DEFAULT_LENGTHS})
    out = {}
    for kind in profiles:
        out[kind] = eim_taper(ns, nside, w_facet, w_wg, wl, dense, n_cells=600, n_modes=30,
                              kind=kind, m=m, smooth=True)
    return dict(models=out, n_slab=ns, n_side=nside)


# ============================================================================ 3D facet run
def facet_sim(ctx, w_facet: float, n_te: float, steps: int = 15, w0_max: float = 2.25,
              offset: float = 1.0):
    """3D FDTD of the facet: the facet-width waveguide on its BOX runs through the absorbing
    boundary at -x, ends at x = 0 (the chip edge) and faces air. A mode source
    at x = -4 um launches TE00 towards the facet; a mode monitor at x = -2 um records the
    incident (+) and reflected (-) amplitudes; flux monitors record the power reflected
    back into the chip and the power leaving it; a field plane at x = 0.5 um records the
    emitted field (11 wavelengths) and a side view (y = 0) the field at the design
    wavelength."""
    import tidy3d as td
    cfg = ctx.cfg
    wl = cfg.run.wavelength_um
    H = ctx.results("cross_section")["height_um"]
    core = ctx.medium(cfg.materials.core)
    box = ctx.medium(cfg.materials.box)
    handle = None if str(cfg.materials.handle).lower() == "none" else ctx.medium(cfg.materials.handle)
    air = str(cfg.materials.cladding).lower() == "air"
    clad = td.Medium(permittivity=1.0) if air else ctx.medium(cfg.materials.cladding)
    tbox = cfg.stack.box_height_um
    big = 1e3
    structs = [td.Structure(geometry=td.Box.from_bounds((-big, -big, -tbox), (0, big, 0)),
                            medium=box, name="BOX")]
    if handle is not None:
        structs.insert(0, td.Structure(geometry=td.Box.from_bounds((-big, -big, -tbox - 3.0),
                                                                   (0, big, -tbox)),
                                       medium=handle, name="Si handle"))
    if not air:
        structs.append(td.Structure(geometry=td.Box.from_bounds(
            (-big, -big, 0), (0, big, cfg.stack.cladding_height_um)), medium=clad, name="cladding"))
    structs.append(td.Structure(geometry=td.Box.from_bounds((-big, -w_facet / 2, 0),
                                                            (0, w_facet / 2, H)),
                                medium=core, name="Facet waveguide"))
    bw = float(cfg.grating.bandwidth_um)
    f0 = C_UM / wl
    fmin, fmax = C_UM / (wl + bw), C_UM / (wl - bw)
    freqs = list(np.linspace(fmin, fmax, 101))
    ffield = list(np.linspace(fmin, fmax, 11))
    # The box only has to hold the emitted field: the Gaussian's normalisation is analytic,
    # so the overlap needs the plane only where the emitted field is not negligible (within
    # about 2 um of the facet's core 0.5 um past it). The BOX runs through the boundary below;
    # the Si handle, 4 um down, is outside (the free facet solve with and without it gives the
    # same n_eff to 1e-5) and would force a fine mesh and short time steps.
    x_lo, x_hi = -6.0, FIBRE_PLANE_UM + 1.5
    y_half = w_facet / 2 + 2.5
    z_lo = -2.5
    z_hi = H + 2.5
    handle = None
    pw = (0, w_facet + 4.0, H + 4.0)                      # port planes: core region only
    pc = H / 2
    ms = td.ModeSpec(num_modes=1, target_neff=float(n_te))
    src = td.ModeSource(center=(-4.0, 0, pc), size=pw, direction="+", mode_spec=ms, mode_index=0,
                        num_freqs=7, source_time=td.GaussianPulse(freq0=f0, fwidth=(fmax - fmin) / 2),
                        name="waveguide mode")
    mons = [td.ModeMonitor(center=(-2.0, 0, pc), size=pw, freqs=freqs, mode_spec=ms, name="port"),
            td.FluxMonitor(center=(-5.0, 0, (z_lo + z_hi) / 2), size=(0, td.inf, td.inf),
                           freqs=freqs, name="back"),
            td.FluxMonitor(center=(FIBRE_PLANE_UM, 0, (z_lo + z_hi) / 2), size=(0, td.inf, td.inf),
                           freqs=freqs, name="out"),
            td.FieldMonitor(center=(FIBRE_PLANE_UM, 0, (z_lo + z_hi) / 2), size=(0, td.inf, td.inf),
                            freqs=ffield, fields=["Ey", "Ez", "Hy", "Hz"], colocate=True,
                            name="fibre plane"),
            td.FieldMonitor(center=((x_lo + x_hi) / 2, 0, (z_lo + z_hi) / 2), size=(td.inf, 0, td.inf),
                            freqs=[f0], fields=["Ey"], colocate=True, name="side view")]
    n_core = ctx.n(cfg.materials.core)
    dl = wl / (n_core * int(steps))
    override = td.MeshOverrideStructure(
        geometry=td.Box.from_bounds((x_lo - 1, -w_facet / 2 - 0.5, -0.4), (x_hi + 1, w_facet / 2 + 0.5,
                                                                            H + 0.4)),
        dl=[dl, dl, dl], enforce=True)
    return td.Simulation(
        center=((x_lo + x_hi) / 2, 0, (z_lo + z_hi) / 2),
        size=(x_hi - x_lo, 2 * y_half, z_hi - z_lo),
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=10, wavelength=wl,
                                   override_structures=[override]),
        structures=structs, sources=[src], monitors=mons, run_time=0.6e-12,
        boundary_spec=td.BoundarySpec.all_sides(td.PML()), medium=clad,
        symmetry=(0, -1, 0))                 # TE00: Ey even across y = 0 (PMC plane)


def facet_name(w_facet: float, H: float, wl: float, steps: int) -> str:
    return f"edge3D_facet_W{w_facet:g}_H{H:g}_s{int(steps)}_{wl:.3f}um"


def phase_sign(Ey, z, i_y0: int) -> float:
    """+1 when the recorded field's phase curves like a wave diverging from the facet in the
    exp(-i w t) convention (phase growing away from the axis), -1 otherwise: makes the
    Gaussian's curvature term independent of the solver's time convention."""
    col = Ey[i_y0, :]
    amp = np.abs(col)
    j = int(np.argmax(amp))
    sel = amp > 0.3 * amp[j]
    if sel.sum() < 5:
        return 1.0
    ph = np.unwrap(np.angle(col[sel]))
    zz = z[sel] - z[j]
    c = np.polyfit(zz, ph, 2)[0]
    return 1.0 if c >= 0 else -1.0


def analyse(data, ctx, mfd: float, mfd_tol: float, offset: float, wl0: float,
            sign: Optional[float] = None) -> dict:
    """Everything from one 3D facet run (see facet_sim). ``sign``: the time convention
    (gaussian()); by default read from the recorded field, which diverges from the facet."""
    port = data["port"].amps
    f = np.asarray(port.coords["f"].values, float)
    ap = np.squeeze(np.asarray(port.sel(direction="+", mode_index=0).values))
    am = np.squeeze(np.asarray(port.sel(direction="-", mode_index=0).values))
    P_in = np.abs(ap) ** 2
    R_mode = np.abs(am) ** 2 / P_in
    R_back = -np.asarray(data["back"].flux.values, float) / P_in
    T_out = np.asarray(data["out"].flux.values, float) / P_in
    fp = data["fibre plane"]
    y = np.asarray(fp.Ey.coords["y"].values, float)
    z = np.asarray(fp.Ey.coords["z"].values, float)
    ff = np.asarray(fp.Ey.coords["f"].values, float)

    def comp(name, k):
        return np.squeeze(np.asarray(getattr(fp, name).isel(f=k).values))

    k0 = int(np.argmin(np.abs(ff - C_UM / wl0)))
    Pin_f = np.interp(ff, f, P_in)
    i_y0 = int(np.argmin(np.abs(y)))

    def fields(k):
        return comp("Ey", k), comp("Ez", k), comp("Hy", k), comp("Hz", k)
    Ey0, Ez0, Hy0, Hz0 = fields(k0)
    sgn = phase_sign(Ey0, z, i_y0) if sign is None else float(sign)
    H = ctx.results("cross_section")["height_um"]

    def eta(F, k, y0, z0, w0, g):
        Ey, Ez, Hy, Hz = F
        u = gaussian(y, z, y0, z0, w0, C_UM / ff[k], FIBRE_PLANE_UM - g, sgn)
        a2, _ = vector_overlap(Ey, Ez, Hy, Hz, y, z, u)
        return a2 / Pin_f[k]

    F0 = (Ey0, Ez0, Hy0, Hz0)
    w0 = mfd / 2

    def best_pos(F, k, w0_):
        best = (-1.0, 0.0, 0.0)
        for g in np.arange(0.0, 4.01, 0.5):
            for z0 in np.arange(-0.4, H + 0.41, 0.05):
                e = eta(F, k, 0.0, z0, w0_, g)
                if e > best[0]:
                    best = (e, float(z0), float(g))
        e, z0, g = best                                   # refine
        for g2 in np.arange(max(0.0, g - 0.5), g + 0.51, 0.1):
            for z2 in np.arange(z0 - 0.06, z0 + 0.061, 0.02):
                v = eta(F, k, 0.0, z2, w0_, g2)
                if v > best[0]:
                    best = (v, float(z2), float(g2))
        return best
    e_best, z_best, g_best = best_pos(F0, k0, w0)
    spot = []
    for m in (mfd - mfd_tol, mfd, mfd + mfd_tol):
        e, zz, gg = best_pos(F0, k0, m / 2)
        spot.append(dict(mfd_um=round(m, 3), eta=e, eta_db=10 * math.log10(max(e, 1e-12)),
                         fibre_height_um=round(zz, 3), working_distance_um=round(gg, 2),
                         eta_at_nominal_position=eta(F0, k0, 0.0, z_best, m / 2, g_best)))
    offs = np.round(np.linspace(-offset, offset, 9), 4)
    grid_ = [[eta(F0, k0, float(dy), z_best + float(dz), w0, g_best) for dz in offs] for dy in offs]
    tol = []
    for name, dy, dz in (("y", offset, 0), ("y", -offset, 0), ("z", 0, offset), ("z", 0, -offset)):
        e = eta(F0, k0, dy, z_best + dz, w0, g_best)
        tol.append(dict(direction=name, offset_um=dy + dz, eta=e,
                        change_db=10 * math.log10(max(e, 1e-12) / e_best)))
    gaps = np.arange(0.0, 6.01, 0.5)
    gap_curve = [eta(F0, k0, 0.0, z_best, w0, float(g)) for g in gaps]
    spectrum = [dict(wavelength_um=float(C_UM / ff[k]),
                     efficiency=eta(fields(k), k, 0.0, z_best, w0, g_best)) for k in range(len(ff))]
    spectrum.sort(key=lambda r: r["wavelength_um"])
    wl = C_UM / f
    o = np.argsort(wl)
    side = data["side view"]
    return dict(eta=e_best, eta_db=10 * math.log10(max(e_best, 1e-12)), fibre_height_um=z_best,
                working_distance_um=g_best, phase_sign=sgn,
                reflection_mode=float(np.interp(wl0, wl[o], R_mode[o])),
                reflection_back_total=float(np.interp(wl0, wl[o], R_back[o])),
                transmitted=float(np.interp(wl0, wl[o], T_out[o])),
                spot=spot, tolerance=tol, offsets_um=offs.tolist(), offset_map=grid_,
                gaps_um=gaps.tolist(), gap_curve=gap_curve, spectrum=spectrum,
                reflection_spectrum=dict(wavelength_um=wl[o].tolist(), mode=R_mode[o].tolist(),
                                         total_back=R_back[o].tolist(), out=T_out[o].tolist()),
                side=dict(x=np.asarray(side.Ey.coords["x"].values, float).tolist(),
                          z=np.asarray(side.Ey.coords["z"].values, float).tolist(),
                          Ey_abs=np.abs(np.squeeze(np.asarray(side.Ey.values))).tolist()))


# ============================================================================ the stage
def _form(ctx) -> None:
    from tarang.design.ui import parse_list
    c3 = lambda c: bool(c.edge.check_3d)                      # noqa: E731
    eme = lambda c: bool(c.edge.run_eme)                      # noqa: E731
    ctx.ui.form(ctx.cfg, "Edge coupler (wide facet, lensed fibre)", [
        dict(section="edge", key="fiber_mfd_um"),
        dict(section="edge", key="mfd_tolerance_um"),
        dict(section="edge", key="facet_widths_um", cast=parse_list),
        dict(section="edge", key="facet_width_um"),
        dict(section="edge", key="profiles", cast=parse_list,
             check=lambda v: "" if v and all(x in PROFILES for x in v)
             else "answer one or more of " + ", ".join(PROFILES)),
        dict(section="edge", key="power_exponent"),
        dict(section="edge", key="profile", check=lambda v: "" if v in (None, "") or v in PROFILES
             else "answer " + ", ".join(PROFILES)),
        dict(section="edge", key="taper_length_um"),
        dict(section="edge", key="max_loss_db"),
        dict(section="edge", key="run_eme"),
        dict(section="edge", key="lengths_um", cast=parse_list, show=eme),
        dict(section="edge", key="eme_cells", show=eme),
        dict(section="edge", key="eme_modes", show=eme),
        dict(section="edge", key="check_3d"),
        dict(section="edge", key="mesh_steps", show=c3),
        dict(section="edge", key="offset_um", show=c3),
        dict(section="edge", key="facet_straight_um"),
        dict(section="edge", key="facet_overhang_um"),
        dict(section="edge", key="chip_length_um"),
        dict(section="edge", key="chip_width_um"),
        dict(section="edge", key="dicing_lane_um"),
        dict(section="edge", key="pitch_um")])


def stage_edge(ctx) -> dict:
    """Facet width (free), taper shape and length (free 2D model, priced EME), facet in 3D
    FDTD (priced)."""
    from tarang.design import visuals
    from tarang.design.checks import _run_batch, _write_csv
    from tarang.design.taper import eme_taper, taper_transmission
    from tarang.utilities.cloud_run import _data_class, find_cached
    _form(ctx)
    cfg, ui = ctx.cfg, ctx.ui
    e = cfg.edge
    wl = cfg.run.wavelength_um
    xs = ctx.results("cross_section")
    H, w_wg = xs["height_um"], xs["width_um"]
    mfd, tol = float(e.fiber_mfd_um), float(e.mfd_tolerance_um)
    mfds = [round(mfd - tol, 3), round(mfd, 3), round(mfd + tol, 3)] if tol > 0 else [round(mfd, 3)]
    res = dict(fiber_mfd_um=mfd, mfd_tolerance_um=tol)

    # ---- 1. facet width (free)
    widths = sorted({float(v) for v in (e.facet_widths_um or default_widths(w_wg, mfd))})
    ui.say(f"Facet overlap map (free): TE00 of the {H * 1e3:.0f} nm film at {len(widths)} facet widths "
           f"(local mode solve with the BOX and the Si handle), against the lensed fibre's spot "
           f"(MFD {', '.join(f'{m:g}' for m in mfds)} µm), best fibre height for each:")
    rows = facet_map(ctx, widths, mfds)
    show = [dict(width_um=r["width_um"], n_eff=r["n_eff"],
                 **{f"eta_{m:g}um_pct": 100 * r[f"eta_mfd{m:g}"] for m in mfds},
                 fibre_height_um=r[f"z0_mfd{mfd:g}_um"]) for r in rows]
    ui.table(show, formats={**{f"eta_{m:g}um_pct": ".1f" for m in mfds}, "n_eff": ".4f"})
    _write_csv(ctx.path("tables", "edge_facet_map.csv"), rows)
    w_rec = choose_width(rows, mfds)
    w_use = float(e.facet_width_um) if e.facet_width_um else w_rec
    md = facet_mode(ctx, w_use)
    T_F = 1 - fresnel(md["n_eff"])
    ov, z0 = best_height(md["Ey"], md["Ez"], md["y"], md["z"], mfd / 2, wl, -0.6 * mfd / 2,
                         H + 0.6 * mfd / 2)
    eta_free = ov * T_F
    ui.say(f"  Recommended facet width: {w_rec:g} µm (best average over the spot sizes)"
           + ("" if w_use == w_rec else f"; using your {w_use:g} µm") + f". Butt-coupling estimate: "
           f"{eta_free * 100:.1f} % ({10 * math.log10(eta_free):.2f} dB) per facet with the fibre "
           f"{z0:.2f} µm above the BOX (overlap {ov * 100:.1f} % x Fresnel transmission "
           f"{T_F * 100:.1f} %); facet reflection about {10 * math.log10(fresnel(md['n_eff'])):.1f} dB.")
    leak = 4 * math.pi * abs(md["k_eff"]) / (wl * 1e-4) * 10 / math.log(10)
    res.update(facet_widths_um=widths, facet_map=rows, facet_width_recommended_um=w_rec,
               facet_width_um=w_use, n_eff_facet=md["n_eff"], eta_overlap=ov, fresnel_T=T_F,
               eta_free=eta_free, fibre_height_free_um=z0,
               reflection_free=fresnel(md["n_eff"]), substrate_leakage_db_cm=leak)
    # the free spectrum: the same estimate at the band edges (two more solves; the Si handle,
    # 4 um below, does not change a guided mode's field: left out, the solves are quick)
    bw = float(cfg.grating.bandwidth_um)
    spec_free = []
    for w_ in (wl - bw, wl, wl + bw):
        mw = md if w_ == wl else facet_mode(ctx, w_use, w_, with_handle=False)
        o_, _ = best_height(mw["Ey"], mw["Ez"], mw["y"], mw["z"], mfd / 2, w_, z0, z0)
        spec_free.append(dict(wavelength_um=w_, efficiency=o_ * (1 - fresnel(mw["n_eff"])),
                              reflection=fresnel(mw["n_eff"])))
    res["spectrum_free"] = spec_free
    visuals.safe(ctx, "facet map", draw_facet, ctx, rows, mfds, w_rec, w_use, md, z0)

    # ---- 2. taper shape and length (free 2D model)
    m = float(e.power_exponent)
    profiles = [p for p in (e.profiles or list(PROFILES)) if p in PROFILES] or ["linear"]
    max_loss = float(e.max_loss_db)
    tm = taper_models(ctx, w_use, w_wg, profiles, m)
    short = {}
    for kind, curve in tm["models"].items():
        L = shortest_length(curve, max_loss / 2)     # half the accepted loss: margin, see below
        short[kind] = None if L is None else float(math.ceil(L / 5.0) * 5.0)
    # A short taper's loss oscillates with length, and the fabricated shape differs a little
    # from the drawn one; at the chip edge length costs nothing, so the length is the
    # shortest from which the loss stays below HALF the accepted loss.
    ui.say(f"Taper {w_use:g} → {w_wg:g} µm, 2D model (effective index, free): shortest length "
           f"from which the loss stays below {max_loss / 2:g} dB (half the accepted {max_loss:g} dB, "
           "margin for the fabricated shape): " + "; ".join(
               f"{k} {v:.0f} µm" if v else f"{k} none up to 400 µm" for k, v in short.items()))
    _write_csv(ctx.path("tables", "edge_taper_model.csv"),
               [dict(length_um=L, **{f"loss_db_{k}": c["loss_db"][i] for k, c in tm["models"].items()})
                for i, L in enumerate(next(iter(tm["models"].values()))["length_um"])])
    res.update(profiles=profiles, power_exponent=m, model_lengths_um=short,
               taper_model={k: dict(length_um=c["length_um"], loss_db=c["loss_db"])
                            for k, c in tm["models"].items()})

    # ---- 3. 3D EME per shape (priced)
    eme = {}
    if e.run_eme:
        lengths = sorted({float(v) for v in (e.lengths_um or DEFAULT_LENGTHS)})
        L0 = max(lengths)
        scales = [v / L0 for v in lengths]
        nc, nm = int(e.eme_cells), int(e.eme_modes)
        jobs, sims = {}, {}
        for kind in profiles:
            sim = eme_taper(ctx, w_use, w_wg, L0, scales, nc, nm, kind=kind, m=m)
            name = (f"edge_taper_{kind}{'' if kind != 'power' else f'{m:g}'}_W{w_use:g}to{w_wg:g}um_"
                    f"L{L0:g}um_eme{nc}x{nm}")
            jobs[kind] = (sim, name)
            sims[kind] = (sim, name)
        folder = ctx.out / "cloud" / "edge_taper"
        ok = _run_batch(ctx, jobs, folder, "edge taper loss against length (EME, one run per shape)")
        if ok:
            for kind, (sim, name) in sims.items():
                hit = find_cached(sim, folder, name)
                if hit is None:
                    continue
                curve = taper_transmission(_data_class(sim).from_file(str(hit)), L0, scales)
                rise = rise_db(curve)
                L3 = shortest_length(curve, max_loss / 2)
                eme[kind] = dict(length_um=curve["length_um"], loss_db=curve["loss_db"], rise_db=rise,
                                 converged=rise <= max_loss,
                                 shortest_um=None if L3 is None else float(math.ceil(L3 / 5.0) * 5.0))
            rows3 = [dict(length_um=L, **{f"loss_db_{k}": c["loss_db"][i] for k, c in eme.items()})
                     for i, L in enumerate(lengths)] if eme else []
            if rows3:
                ui.table(rows3, formats={f"loss_db_{k}": ".3f" for k in eme})
                _write_csv(ctx.path("tables", "edge_taper_eme.csv"), rows3)
            for k, c in eme.items():
                ui.say(f"  3D EME, {k}: " + (f"stays below {max_loss / 2:g} dB from {c['shortest_um']:.0f} µm"
                                             if c["shortest_um"] else "no swept length below the accepted loss")
                       + ("" if c["converged"] else f" (loss rises by {c['rise_db']:.2f} dB with length: "
                                                    "not converged, the 2D model is used)"))
    res["eme"] = eme

    def length_of(kind):
        c = eme.get(kind)
        if c and c["converged"] and c["shortest_um"]:
            return c["shortest_um"], "3D EME"
        return short.get(kind), "2D model"
    cands = {k: length_of(k) for k in profiles if length_of(k)[0]}
    rec_kind = min(cands, key=lambda k: cands[k][0]) if cands else profiles[0]
    kind = e.profile if e.profile in PROFILES else rec_kind
    L_rec, src = cands.get(kind, (None, None))
    L_use = float(e.taper_length_um) if e.taper_length_um else (L_rec or 200.0)
    if kind in eme and eme[kind]["converged"]:
        loss_use, src_loss = loss_at(eme[kind], L_use), "3D EME"
    else:
        loss_use, src_loss = loss_at(tm["models"][kind], L_use), "2D model"
    ui.say(f"  Taper used: {kind}{f' (m = {m:g})' if kind == 'power' else ''}, {L_use:.0f} µm "
           f"({loss_use:.3f} dB, {src_loss})" + (f"; recommended: {rec_kind}, "
                                                 f"{cands[rec_kind][0]:.0f} µm" if cands else "") + ".")
    res.update(profile=kind, profile_recommended=rec_kind, taper_length_um=L_use,
               taper_length_recommended_um=cands[rec_kind][0] if cands else None,
               taper_loss_db=loss_use, taper_loss_source=src_loss)
    visuals.safe(ctx, "edge taper", draw_taper, ctx, tm["models"], eme, kind, L_use, w_use, w_wg, m)

    # ---- 4. the facet in 3D FDTD (priced)
    res3 = None
    if e.check_3d:
        steps = int(e.mesh_steps)
        sim = facet_sim(ctx, w_use, md["n_eff"], steps, w0_max=(mfd + tol) / 2, offset=float(e.offset_um))
        name = facet_name(w_use, H, wl, steps)
        folder = ctx.out / "cloud" / "edge_facet"
        ok = _run_batch(ctx, {"facet": (sim, name)}, folder, "edge-coupler facet in 3D FDTD")
        hit = find_cached(sim, folder, name) if ok else None
        if hit is not None:
            import tidy3d as td
            data = td.SimulationData.from_file(str(hit))
            res3 = analyse(data, ctx, mfd, tol, float(e.offset_um), wl)
            ui.say(f"  3D FDTD: {res3['eta'] * 100:.1f} % ({res3['eta_db']:.2f} dB) per facet at {wl} µm, "
                   f"fibre {res3['fibre_height_um']:.2f} µm above the BOX, waist "
                   f"{res3['working_distance_um']:.1f} µm in front of the facet (butt-coupling "
                   f"estimate {eta_free * 100:.1f} %). Facet reflection into the waveguide mode "
                   f"{10 * math.log10(max(res3['reflection_mode'], 1e-12)):.1f} dB; all light sent back "
                   f"into the chip {10 * math.log10(max(res3['reflection_back_total'], 1e-12)):.1f} dB.")
            ui.table([dict(mfd_um=s["mfd_um"], eta_pct=100 * s["eta"], loss_db=-s["eta_db"],
                           fibre_height_um=s["fibre_height_um"], working_distance_um=s["working_distance_um"])
                      for s in res3["spot"]],
                     formats={"eta_pct": ".1f", "loss_db": ".2f"})
            ui.table([dict(direction=t["direction"], offset_um=t["offset_um"],
                           change_db=t["change_db"]) for t in res3["tolerance"]],
                     formats={"change_db": ".2f"})
            _write_csv(ctx.path("tables", "edge_spot_and_offsets.csv"),
                       [dict(kind="spot", **s) for s in res3["spot"]] +
                       [dict(kind="offset", **t) for t in res3["tolerance"]])
            rs = res3["reflection_spectrum"]
            _write_csv(ctx.path("tables", "edge_reflection_3d.csv"),
                       [dict(wavelength_um=a, reflection_mode=b, reflection_all_back=c, transmitted=d)
                        for a, b, c, d in zip(rs["wavelength_um"], rs["mode"], rs["total_back"], rs["out"])])
            _write_spectrum(ctx.path("tables", "edge_spectrum_3d.csv"),
                            [r["wavelength_um"] for r in res3["spectrum"]],
                            [r["efficiency"] for r in res3["spectrum"]])
            visuals.safe(ctx, "edge 3D", draw_3d, ctx, res3, w_use, mfd)
    _write_spectrum(ctx.path("tables", "edge_spectrum_free.csv"),
                    [r["wavelength_um"] for r in spec_free], [r["efficiency"] for r in spec_free])
    if res3:
        res3 = {k: v for k, v in res3.items() if k not in ("side",)}
    res["fdtd"] = res3
    eta_main = res3["eta"] if res3 else eta_free
    T_tp = 10 ** (-loss_use / 10)
    res["eta_coupler"] = eta_main * T_tp
    ui.say(f"  One edge coupler (facet x taper): {res['eta_coupler'] * 100:.1f} % "
           f"({10 * math.log10(res['eta_coupler']):.2f} dB) at {wl} µm.")
    if getattr(ui, "price_only", False) and ((e.run_eme and not eme) or (e.check_3d and not res3)):
        # only priced: the next run must do this stage again (with the cloud parts)
        res.update(skipped=True, price_only=True)
    res["recommend"] = dict(
        choices=[dict(option="facet width", value=w_rec, unit="um"),
                 dict(option="taper shape", value=rec_kind),
                 dict(option="taper length", value=res["taper_length_recommended_um"], unit="um")],
        recommended=dict(facet_width_um=w_rec, profile=rec_kind,
                         taper_length_um=res["taper_length_recommended_um"]),
        why=(f"best average butt coupling for MFD {mfd:g} ± {tol:g} µm; the taper shape that "
             f"stays below half the accepted {max_loss:g} dB from the shortest length"))
    return res


def _write_spectrum(path, wls, eta) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["wavelength_um", "efficiency", "efficiency_db"])
        for a, b in sorted(zip(wls, eta)):
            w.writerow([f"{a:.6f}", f"{b:.6g}", f"{10 * math.log10(max(b, 1e-12)):.4f}"])


# ============================================================================ the coupler model
def coupler(ctx) -> Optional[dict]:
    """eta(lambda), reflection(lambda) and the length on chip of one edge coupler (facet
    straight + taper), from the 3D run when it exists, else the butt-coupling estimate."""
    from tarang.design.circuit import _read_spectrum
    st = ctx.state["stages"]
    ed = (st.get("edge") or {}).get("results") or {}
    if not ed or ed.get("skipped"):
        return None
    tables = ctx.out / "tables"
    T_tp = 10 ** (-float(ed.get("taper_loss_db") or 0.0) / 10)
    length = float(ed.get("taper_length_um") or 0.0) + float(ctx.cfg.edge.facet_straight_um)
    shape = f"{ed.get('profile')} taper {ed.get('taper_length_um'):.0f} µm"
    s3 = _read_spectrum(tables / "edge_spectrum_3d.csv")
    if s3 is not None and ed.get("fdtd"):
        rs = ed["fdtd"]["reflection_spectrum"]
        return dict(wl=s3[0], eta=s3[1] * T_tp,
                    label=f"edge coupler: {ed['facet_width_um']:g} µm facet (3D FDTD) x {shape}",
                    length_um=length, refl_wl=np.asarray(rs["wavelength_um"]),
                    refl=np.asarray(rs["mode"]), refl_source="the 3D FDTD of the facet", notes=[])
    sf = ed.get("spectrum_free") or []
    if not sf:
        return None
    wl = np.array([r["wavelength_um"] for r in sf])
    return dict(wl=wl, eta=np.array([r["efficiency"] for r in sf]) * T_tp,
                label=f"edge coupler: {ed['facet_width_um']:g} µm facet (butt-coupling estimate) x {shape}",
                length_um=length, refl_wl=wl, refl=np.array([r["reflection"] for r in sf]),
                refl_source="the Fresnel reflection at n_eff (no 3D run)", notes=[])


# ============================================================================ figures
def draw_facet(ctx, rows, mfds, w_rec, w_use, md, z0):
    """Figure 12a: coupling against facet width for each spot size, and the chosen facet's
    mode with the fibre spot."""
    from tarang.design import visuals
    plt = visuals._plt()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(width_ratios=[1.1, 1]))
    W = [r["width_um"] for r in rows]
    for k, m in enumerate(mfds):
        ax.plot(W, [100 * r[f"eta_mfd{m:g}"] for r in rows], "o-", ms=4,
                color=visuals.PALETTE[k % len(visuals.PALETTE)], label=f"spot MFD {m:g} µm")
    ax.axvline(w_rec, color="0.4", ls="--", lw=1.2, label=f"recommended {w_rec:g} µm")
    if w_use != w_rec:
        ax.axvline(w_use, color="0.2", ls=":", lw=1.2, label=f"used {w_use:g} µm")
    ax.set(xlabel="Facet width (µm)", ylabel="Coupling per facet (%)",
           title="Butt coupling: mode overlap × Fresnel transmission (best fibre height)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    y, z = md["y"], md["z"]
    I = np.abs(md["Ey"]) ** 2 + np.abs(md["Ez"]) ** 2
    bx.pcolormesh(y, z, (I / I.max()).T, cmap="magma", shading="auto")
    H = md["height_um"]
    bx.add_patch(plt.Rectangle((-w_use / 2, 0), w_use, H, fill=False, ec="w", lw=1))
    bx.axhline(0, color="w", lw=0.6, ls=":")
    w0 = mfds[len(mfds) // 2] / 2
    t = np.linspace(0, 2 * np.pi, 200)
    bx.plot(w0 * np.cos(t), z0 + w0 * np.sin(t), color="#4dd0e1", lw=1.5, ls="--",
            label=f"fibre spot (1/e² radius {w0:g} µm)")
    lim = max(w_use / 2 + 1.0, w0 * 1.3)
    bx.set_xlim(-lim, lim)
    bx.set_ylim(min(-w0 * 1.3 + z0, -1.5), z0 + w0 * 1.3)
    bx.set_aspect("equal")
    bx.set(xlabel="y (µm)", ylabel="z (µm)",
           title=f"TE00 at the {w_use:g} µm facet (n_eff {md['n_eff']:.4f}) and the fibre spot")
    bx.legend(fontsize=7.5, loc="lower right")
    return visuals._save(fig, ctx.path("figures", "12a_edge_facet.png"))


def draw_taper(ctx, models, eme, kind, L_use, w_facet, w_wg, m):
    """Figure 12b: taper loss against length for each shape (2D model lines, 3D EME points),
    and the shapes themselves."""
    from tarang.design import visuals
    from tarang.design.taper import profile_width
    plt = visuals._plt()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=[1.3, 1]))
    for k, (name, c) in enumerate(models.items()):
        col = visuals.PALETTE[k % len(visuals.PALETTE)]
        ax.semilogy(c["length_um"], np.clip(c["loss_db"], 1e-4, None), "-", color=col, lw=1.6,
                    label=f"{name}, 2D model")
        if name in eme:
            c3 = eme[name]
            ax.semilogy(c3["length_um"], np.clip(c3["loss_db"], 1e-4, None), "o" if c3["converged"] else "o",
                        color=col, mfc=col if c3["converged"] else "white",
                        label=f"{name}, 3D EME" + ("" if c3["converged"] else " (not converged)"))
    ax.axhline(ctx.cfg.edge.max_loss_db, color="0.5", ls=":", label=f"accepted {ctx.cfg.edge.max_loss_db:g} dB")
    ax.axvline(L_use, color="0.3", ls="--", lw=1, label=f"used: {kind}, {L_use:.0f} µm")
    ax.set(xlabel="Taper length (µm)", ylabel="Taper loss (dB, one way)",
           title=f"Taper {w_wg:g} → {w_facet:g} µm: loss against length")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5)
    s = np.linspace(0, 1, 200)
    for k, name in enumerate(models):
        w = profile_width(s, w_wg, w_facet, name, m)
        bx.plot(s * L_use, w / 2, color=visuals.PALETTE[k % len(visuals.PALETTE)], label=name)
        bx.plot(s * L_use, -w / 2, color=visuals.PALETTE[k % len(visuals.PALETTE)])
    bx.set(xlabel="Distance from the waveguide (µm)", ylabel="y (µm)",
           title=f"Shapes at {L_use:.0f} µm (y stretched; facet on the right)")
    bx.grid(alpha=0.3)
    bx.legend(fontsize=8)
    return visuals._save(fig, ctx.path("figures", "12b_edge_taper.png"))


def draw_3d(ctx, r, w_facet, mfd):
    """Figure 12c: the 3D facet run: field leaving the facet, coupling against fibre offset,
    against working distance and wavelength, and the facet reflection."""
    from tarang.design import visuals
    plt = visuals._plt()
    fig = plt.figure(figsize=(14, 8))
    g = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(g[0, :2])
    sd = r["side"]
    A = np.asarray(sd["Ey_abs"])
    ax.pcolormesh(sd["x"], sd["z"], (A / A.max()).T ** 2, cmap="magma", shading="auto")
    ax.axvline(0, color="w", lw=0.8, ls="--")
    ax.axvline(FIBRE_PLANE_UM, color="#4dd0e1", lw=0.8, ls=":")
    ax.set(xlabel="x (µm; chip edge at 0)", ylabel="z (µm)",
           title=f"|Ey|² at y = 0, {ctx.cfg.run.wavelength_um} µm: the mode leaving the {w_facet:g} µm facet")
    ax.set_aspect("equal")
    bx = fig.add_subplot(g[0, 2])
    offs = np.asarray(r["offsets_um"])
    M = 10 * np.log10(np.clip(np.asarray(r["offset_map"]), 1e-12, None) / max(r["eta"], 1e-12))
    im = bx.contourf(offs, offs, M.T, levels=np.arange(-6, 0.01, 0.5), cmap="viridis", extend="min")
    bx.contour(offs, offs, M.T, levels=[-1], colors="w", linewidths=1.2)
    fig.colorbar(im, ax=bx, label="change (dB)")
    bx.set(xlabel="fibre offset in y (µm)", ylabel="fibre offset in z (µm)",
           title="Alignment (white: −1 dB)")
    bx.set_aspect("equal")
    cx = fig.add_subplot(g[1, 0])
    cx.plot(r["gaps_um"], 100 * np.asarray(r["gap_curve"]), "o-", color=visuals.PALETTE[0])
    cx.set(xlabel="spot waist in front of the facet (µm)", ylabel="coupling (%)",
           title=f"Working distance (MFD {mfd:g} µm)")
    cx.grid(alpha=0.3)
    dx = fig.add_subplot(g[1, 1])
    sp = r["spectrum"]
    dx.plot([s["wavelength_um"] * 1e3 for s in sp], [100 * s["efficiency"] for s in sp], "o-",
            color=visuals.PALETTE[1])
    dx.set(xlabel="Wavelength (nm)", ylabel="coupling per facet (%)", title="Spectrum (fibre fixed)")
    dx.grid(alpha=0.3)
    ex = fig.add_subplot(g[1, 2])
    rs = r["reflection_spectrum"]
    ex.plot(np.asarray(rs["wavelength_um"]) * 1e3, 10 * np.log10(np.clip(rs["mode"], 1e-12, None)),
            color=visuals.PALETTE[2], label="into the waveguide mode")
    ex.plot(np.asarray(rs["wavelength_um"]) * 1e3, 10 * np.log10(np.clip(rs["total_back"], 1e-12, None)),
            color="0.5", ls="--", label="all light back into the chip")
    ex.set(xlabel="Wavelength (nm)", ylabel="reflection (dB)", title="Facet reflection")
    ex.grid(alpha=0.3)
    ex.legend(fontsize=7.5)
    fig.suptitle(f"Edge coupler in 3D FDTD: {r['eta'] * 100:.1f} % per facet at the best position")
    return visuals._save(fig, ctx.path("figures", "12c_edge_3d.png"))
