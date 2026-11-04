"""Local (free) mode solves used by the figures, the report and the checks stage.

* ``cross_section_modes``: the modes of the chosen core, straight or bent, with their
  fields, TE fraction, effective area and (from three wavelengths) group index.
* ``coupler_supermodes``: the even and odd TE supermodes of the bus and the ring side by
  side at a gap. Their index split gives the coupling per unit length,
  kappa = pi * dn / lambda, and, with the gap growing as g(x) = g0 + x^2 / (2 R) along the
  ring, the coupling angle per pass theta = kappa(g0) * sqrt(2 pi R / gamma), where gamma is
  the decay of kappa with gap (the FDTD fit's gamma). That is an independent estimate of
  the FDTD coupling from first principles (coupled-mode theory).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

C_UM = 2.99792458e14


def _solver(ctx, H, W, wl, *, margin_y=2.0, bend_radius=None, num_modes=4, pml=0,
            extra=None, wls=None, sidewall_deg=90.0, sidewall_ref="top"):
    import tidy3d as td
    from tidy3d.plugins.mode import ModeSolver
    from tarang.waveguide.dispersion_mapping import core as dm
    cfg = ctx.cfg
    sim = dm.build_sim(H, W, wl0=wl, steps=cfg.cross_section.decision_steps,
                       box=cfg.stack.box_height_um, core=cfg.materials.core,
                       substrate=cfg.materials.box, cladding=cfg.materials.cladding,
                       cladding_height=cfg.stack.cladding_height_um, band=ctx.band,
                       margin_y=margin_y, sidewall_deg=sidewall_deg, sidewall_ref=sidewall_ref)
    if extra is not None:
        sim = extra(sim)
    spec = dict(num_modes=num_modes, target_neff=ctx.n(cfg.materials.core))
    if bend_radius:
        spec.update(bend_radius=float(bend_radius), bend_axis=1)
        pml = max(pml, 12)
    if pml:
        spec["num_pml"] = (pml, pml)
    plane = td.Box(center=(0, sim.center[1], sim.center[2]), size=(0, sim.size[1], sim.size[2]))
    freqs = [C_UM / w for w in (wls or [wl])]
    return ModeSolver(simulation=sim, plane=plane, freqs=freqs,
                      mode_spec=td.ModeSpec(**spec)).solve()


def _modes_from(data, fi=0) -> list[dict]:
    out = []
    y = data.Ey.coords["y"].values
    z = data.Ey.coords["z"].values
    for m in range(data.n_eff.sizes["mode_index"]):
        f = {c: np.squeeze(getattr(data, c).isel(mode_index=m, f=fi).values)
             for c in ("Ex", "Ey", "Ez")}
        ey2, ez2 = (np.abs(f["Ey"]) ** 2).sum(), (np.abs(f["Ez"]) ** 2).sum()
        E2 = sum(np.abs(v) ** 2 for v in f.values())
        dA = np.outer(np.gradient(y), np.gradient(z))
        out.append(dict(index=m, neff=float(np.real(data.n_eff.isel(mode_index=m, f=fi))),
                        k=float(np.real(data.k_eff.isel(mode_index=m, f=fi))),
                        te=float(ey2 / (ey2 + ez2 + 1e-30)), fields=f, y=y, z=z,
                        aeff=float(np.sum(E2 * dA) ** 2 / np.sum(E2 ** 2 * dA)),
                        yc=float(np.sum(E2.sum(axis=1) * y) / np.sum(E2)),
                        zc=float(np.sum(E2.sum(axis=0) * z) / np.sum(E2))))
    return out


def cross_section_modes(ctx, H, W, *, bend_radius: Optional[float] = None, num_modes=4,
                        group_index=False) -> list[dict]:
    """Modes at the design wavelength, sorted by n_eff (highest first). With
    ``group_index`` the solve is repeated at wl -/+ 1 % and each mode gets n_g from the
    slope of n_eff, matched to the same mode by field overlap."""
    wl = ctx.cfg.run.wavelength_um
    wls = [wl * 0.99, wl, wl * 1.01] if group_index else [wl]
    data = _solver(ctx, H, W, wl, margin_y=3.0 if bend_radius else 2.0,
                   bend_radius=bend_radius, num_modes=num_modes, wls=wls)
    fi = 1 if group_index else 0
    modes = _modes_from(data, fi)
    if group_index:
        lo, hi = _modes_from(data, 0), _modes_from(data, 2)
        for m in modes:
            def match(cands):
                def ov(c):
                    a = sum(np.abs(np.sum(np.conj(m["fields"][k]) * c["fields"][k]))
                            for k in ("Ey", "Ez"))
                    return a
                return max(cands, key=ov)
            n_lo, n_hi = match(lo)["neff"], match(hi)["neff"]
            dn_dl = (n_hi - n_lo) / (0.02 * wl)
            m["ng"] = m["neff"] - wl * dn_dl
    modes.sort(key=lambda m: -m["neff"])
    return modes


def te00(modes: list[dict]) -> Optional[dict]:
    te = [m for m in modes if m["te"] > 0.5]
    return max(te, key=lambda m: m["neff"]) if te else None


def tm00(modes: list[dict]) -> Optional[dict]:
    tm = [m for m in modes if m["te"] <= 0.5]
    return max(tm, key=lambda m: m["neff"]) if tm else None


def coupler_supermodes(ctx, H, W, gap, steps: Optional[int] = None,
                       sidewall_deg: float = 90.0, sidewall_ref: str = "top",
                       snap: bool = False, snap_sides: int = 1) -> dict:
    """Even and odd TE supermodes of two identical cores (bus and ring at the closest
    point, the ring treated as straight there) separated by ``gap``.

    The split dn is ~1e-3 of n_eff or less, so two things matter:

    * the fine mesh must cover both cores (the single-core simulation's fine region is
      only as wide as one core plus a margin);
    * the solver must return true supermodes. Without help, two nearly degenerate modes
      can come out as mixtures localised in one core each (their index difference is then
      numerical noise, not coupling), which happens at large gaps and depends on the
      platform's linear algebra. The mirror plane y = 0 is therefore imposed: a PEC-type
      symmetry (Ey even) gives the even TE supermode, a PMC-type (Ey odd) the odd one.

    With sloped walls (``sidewall_deg`` != 90) both cores are trapezoids; ``gap`` is the gap
    at the ``sidewall_ref`` plane (the mask gap for 'top'), the centre-to-centre spacing
    W + gap being fixed by the mask. ``snap`` puts grid lines through every core edge
    (both faces of both cores), so the gap is resolved exactly at any mesh: comparisons
    between geometries whose edges sit in different places (the sidewall sweep) then
    measure the geometry, not where the edges fall in the cells. With ``snap_sides`` 2 each
    supermode is solved with the lines just inside and just outside every face and its
    n_eff is the mean of the two (see dispersion_mapping.snapping.snap_offsets).
    """
    from tarang.geometries import sidewall as sw
    import tidy3d as td
    core_medium = ctx.medium(ctx.cfg.materials.core)
    shift = W + gap

    def two_cores(symmetry, offset=0.0):
        def make(sim):
            kept = [st for st in sim.structures if st.name != "core_sloped" and not (
                isinstance(st.geometry, td.Box) and abs(st.geometry.size[1] - W) < 1e-9
                and abs(st.geometry.center[2] - H / 2) < 1e-9)]
            if len(kept) == len(sim.structures):
                raise RuntimeError("core structure not found in the cross-section simulation")
            if sw.is_vertical(sidewall_deg):
                cores = [td.Structure(geometry=td.Box(center=(0, sgn * shift / 2, H / 2),
                                                      size=(td.inf, W, H)), medium=core_medium,
                                      name=f"core_{i}") for i, sgn in enumerate((-1, 1))]
            else:
                if sw.bottom_gap(gap, H, sidewall_deg, sidewall_ref) <= 0:
                    raise ValueError(f"at {sidewall_deg}° the two cores touch at the bottom "
                                     f"(gap {gap} um)")
                cores = [td.Structure(geometry=sw.core_geometry(
                    W, H, sidewall_deg, sidewall_ref, y0=sgn * shift / 2), medium=core_medium,
                    name=f"core_{i}") for i, sgn in enumerate((-1, 1))]
            gs = sim.grid_spec
            overrides = []
            for o in gs.override_structures or []:           # widen the fine region
                c, s = o.geometry.center, o.geometry.size
                overrides.append(o.updated_copy(geometry=td.Box(
                    center=(c[0], 0.0, c[2]), size=(s[0], s[1] + shift, s[2]))))
            gs = gs.updated_copy(override_structures=overrides)
            if snap:
                from tarang.waveguide.dispersion_mapping.core import sloped_snap_points
                pts = [p for sgn in (-1, 1) for p in sloped_snap_points(
                    W, H, sidewall_deg, sidewall_ref, y0=sgn * shift / 2, offset=offset)]
                gs = gs.updated_copy(snapping_points=pts)
            return sim.updated_copy(center=(sim.center[0], 0.0, sim.center[2]),
                                    size=(sim.size[0], sim.size[1] + shift, sim.size[2]),
                                    structures=kept + cores, symmetry=(0, symmetry, 0),
                                    grid_spec=gs)
        return make
    wl = ctx.cfg.run.wavelength_um
    cfg = ctx.cfg
    old = cfg.cross_section.decision_steps
    if steps:
        cfg.cross_section.decision_steps = int(steps)
    from tarang.waveguide.dispersion_mapping.snapping import snap_offsets
    found = {}
    try:
        for name, sym in (("even", -1), ("odd", 1)):
            per_side = []
            for off in snap_offsets(snap, snap_sides):
                data = _solver(ctx, H, W, wl, margin_y=2.5, num_modes=2,
                               extra=two_cores(sym, off), sidewall_deg=sidewall_deg,
                               sidewall_ref=sidewall_ref)
                te = [m for m in _modes_from(data) if m["te"] > 0.5]
                if not te:
                    raise RuntimeError(f"no {name} TE supermode found at gap {gap} um")
                per_side.append(max(te, key=lambda m: m["neff"]))
            mode = dict(per_side[0])
            mode["neff"] = float(np.mean([m["neff"] for m in per_side]))
            mode["neff_sides"] = [m["neff"] for m in per_side]
            found[name] = mode
    finally:
        cfg.cross_section.decision_steps = old
    even, odd = found["even"], found["odd"]
    dn = even["neff"] - odd["neff"]
    if dn <= 0:
        raise RuntimeError(f"odd supermode above the even one at gap {gap} um (dn = {dn:.3g})")
    return dict(even=even, odd=odd, dn=dn, kappa_per_um=math.pi * dn / wl,
                coupling_length_um=wl / (2 * dn), shift_um=shift, gap_um=gap,
                cores=[(-shift / 2 - W / 2, 0.0, W, H), (shift / 2 - W / 2, 0.0, W, H)])


def coupling_from_supermodes(ctx, H, W, radius_um, gaps, steps=None, sidewall_deg=90.0,
                             sidewall_ref="top", snap=False, snap_sides=1) -> dict:
    """Coupled-mode estimate of the coupling per pass at ``gaps``: dn from the supermodes,
    its decay rate gamma from a fit over the gaps, then theta(g) = kappa(g) sqrt(2 pi R /
    gamma). Returns the per-gap values and the fit."""
    rows = []
    for g in gaps:
        sm = coupler_supermodes(ctx, H, W, g, steps=steps, sidewall_deg=sidewall_deg,
                                sidewall_ref=sidewall_ref, snap=snap, snap_sides=snap_sides)
        rows.append(dict(gap_um=g, dn=sm["dn"], kappa_per_um=sm["kappa_per_um"]))
    gs = np.array([r["gap_um"] for r in rows])
    ks = np.array([r["kappa_per_um"] for r in rows])
    slope, icept = np.polyfit(gs, np.log(ks), 1)
    gamma = float(-slope)
    for r in rows:
        r["theta"] = theta_per_pass(r["kappa_per_um"], radius_um, gamma)
        r["coupled_power"] = math.sin(min(r["theta"], math.pi / 2)) ** 2
    th0 = float(np.exp(icept)) * math.sqrt(2 * math.pi * radius_um / gamma)
    return dict(rows=rows, gamma_per_um=gamma, theta0=th0)


def theta_per_pass(kappa_per_um: float, radius_um: float, gamma_per_um: float) -> float:
    """Coupling angle per pass of a ring next to a straight bus when the coupling per unit
    length decays as kappa0 exp(-gamma (g - g0)) and the gap grows as x^2 / (2 R):
    integral of kappa over x = kappa0 sqrt(2 pi R / gamma)."""
    return kappa_per_um * math.sqrt(2 * math.pi * radius_um / gamma_per_um)
