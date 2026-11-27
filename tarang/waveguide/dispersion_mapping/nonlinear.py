"""Full-vectorial nonlinear coefficient of a guided mode, with the Kerr index of each
material where the field actually is.

The scalar form gamma = 2 pi n2 / (lambda A_eff) gives the whole mode the core's n2. In a
high-contrast waveguide a fifth or more of the field sits in the BOX and the cladding,
whose n2 is ten times smaller (silica) or zero (air), and the field is not purely
transverse. Afshar and Monro, Opt. Express 17, 2298 (2009), Eq. 25, give gamma for a mode
with transverse fields e, h and a Kerr index that varies over the cross-section:

    gamma = k0 (eps0/mu0) Int n(r)^2 n2(r) [2 |e|^4 + |e . e|^2] dA / (3 |Int Re(e x h*) . z dA|^2)

(k0 = 2 pi / lambda; e . e without conjugation). For a weakly guiding, linearly polarised
mode in one material it reduces to 2 pi n2 / (lambda A_eff).

The integral splits over materials: gamma = sum_i n2_i G_i, with G_i the overlap of the
mode with material i. ``overlaps`` returns G_i per material key, so a design can apply
its own n2 values (your [assumptions] value for the core, the registry's for the rest)
without solving again.

Exposes: overlaps(), gamma_from(), scalar_gamma().
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6


def _cell_average(a: np.ndarray) -> np.ndarray:
    """Values at grid points -> mean over each cell's four corners."""
    return 0.25 * (a[:-1, :-1] + a[1:, :-1] + a[:-1, 1:] + a[1:, 1:])


def _field(data, comp: str, m: int) -> np.ndarray:
    return np.squeeze(getattr(data, comp).isel(mode_index=m, f=0).values)


def overlaps(height, width, *, wl0=1.55, steps=50, box=4.0, core="si3n4", substrate="sio2",
             cladding=None, cladding_height=2.0, n_shift=0.0, band=None, snap=False,
             snap_sides: int = 2, sidewall_deg=90.0, sidewall_ref="top", profile=None,
             mode: str = "TE00", num_modes: int = 4) -> dict:
    """Per-material nonlinear overlaps G_i (1/m^3; gamma = sum n2_i G_i) of the pump mode at
    wl0, from one mode solve per snapping side (averaged, as every other TARANG number), plus
    the mode's A_eff (um^2) for the scalar form."""
    import tidy3d as td
    from tidy3d.plugins.mode import ModeSolver

    from tarang.waveguide.dispersion_mapping import core as dm
    from tarang.waveguide.dispersion_mapping.snapping import snap_offsets
    lo, hi = band if band else (wl0 * 0.95, wl0 * 1.05)
    keys = {"core": core, "box": substrate,
            "cladding": cladding if cladding else "air"}
    if profile is not None and not profile.is_default and profile.pedestal:
        keys["pedestal"] = profile.pedestal
    if keys["cladding"] != "air":
        keys["air"] = "air"                     # above a finite cladding: the background
    eps_of = {}
    for role, k in keys.items():
        if str(k).lower() in {"air", "none", ""}:
            eps_of[role] = 1.0 + 0j
        else:
            med = dm.shifted(dm.medium_for(k, lo, hi), n_shift, wl0) if role == "core" else \
                dm.medium_for(k, lo, hi)
            eps_of[role] = complex(med.eps_model(td.C_0 / wl0))
    want_te = str(mode).upper().startswith("TE")
    sides = []
    for off in snap_offsets(snap, snap_sides):
        sim = dm.build_sim(height, width, wl0=wl0, steps=steps, box=box, core=core,
                           substrate=substrate, cladding=cladding,
                           cladding_height=cladding_height, n_shift=n_shift, band=(lo, hi),
                           snap=snap, sidewall_deg=sidewall_deg, sidewall_ref=sidewall_ref,
                           snap_offset=off, profile=profile)
        plane = td.Box(center=(0, 0, sim.center[2]), size=(0, sim.size[1], sim.size[2]))
        core_m = dm.shifted(dm.medium_for(core, lo, hi), n_shift, wl0)
        data = ModeSolver(simulation=sim, plane=plane, freqs=[td.C_0 / wl0],
                          mode_spec=td.ModeSpec(num_modes=num_modes,
                                                target_neff=dm.n_of(core_m, wl0))).solve()
        cands = [m for m in range(num_modes)
                 if (dm._te_frac(data, m, 0) > 0.5) == want_te]
        if not cands:
            raise ValueError(f"no {mode}-like mode among the {num_modes} solved")
        m = max(cands, key=lambda i: float(np.real(data.n_eff.isel(f=0, mode_index=i))))
        E = [_field(data, c, m) for c in ("Ex", "Ey", "Ez")]
        H = [_field(data, c, m) for c in ("Hx", "Hy", "Hz")]
        y = np.asarray(data.Ey.coords["y"].values)
        z = np.asarray(data.Ey.coords["z"].values)
        Ec = [_cell_average(a) for a in E]
        Hc = [_cell_average(a) for a in H]
        yc, zc = 0.5 * (y[:-1] + y[1:]), 0.5 * (z[:-1] + z[1:])
        dA = np.outer(np.diff(y), np.diff(z)) * 1e-12                  # m^2
        eps = sim.epsilon(box=plane, coord_key="centers", freq=td.C_0 / wl0)
        ey, ez = eps.coords["y"].values, eps.coords["z"].values
        iy = np.searchsorted(yc, ey - 1e-9)
        iz = np.searchsorted(zc, ez - 1e-9)
        if not (np.allclose(yc[iy], ey, atol=1e-7) and np.allclose(zc[iz], ez, atol=1e-7)):
            raise RuntimeError("field and permittivity grids do not line up")
        sel = np.ix_(iy, iz)
        Ec = [a[sel] for a in Ec]
        Hc = [a[sel] for a in Hc]
        dA = dA[sel]
        e_cell = np.squeeze(eps.values)
        # material of each cell: the role whose permittivity it is (exact: no averaging here)
        roles = list(eps_of)
        dist = np.stack([np.abs(e_cell - eps_of[r]) for r in roles])
        which = np.argmin(dist, axis=0)
        if float(np.min(dist, axis=0).max()) > 1e-6 * max(abs(v) for v in eps_of.values()):
            raise RuntimeError("a cell's permittivity matches none of the stack's materials")
        # The mode propagates along x (the plane is normal to x): power flux (e x h*) . x
        Sx = np.real(Ec[1] * np.conj(Hc[2]) - Ec[2] * np.conj(Hc[1]))
        P = float(np.sum(Sx * dA))
        e2 = sum(np.abs(a) ** 2 for a in Ec)
        ee = sum(a * a for a in Ec)
        dens = np.real(e_cell) * (2 * e2 ** 2 + np.abs(ee) ** 2) * dA  # n^2 = Re(eps)
        k0 = 2 * math.pi / (wl0 * 1e-6)
        G = {}
        for i, r in enumerate(roles):
            G[r] = float(k0 * (EPS0 / MU0) * np.sum(dens[which == i]) / (3 * P ** 2))
        aeff = float(np.sum(e2 * dA) ** 2 / np.sum(e2 ** 2 * dA)) * 1e12        # um^2
        sides.append(dict(G=G, aeff=aeff, n_eff=float(np.real(data.n_eff.isel(f=0, mode_index=m)))))
    out_G = {r: float(np.mean([s["G"][r] for s in sides])) for r in sides[0]["G"]}
    return dict(mode=mode, wl0=wl0, materials=keys, G_per_m3=out_G,
                aeff_um2=float(np.mean([s["aeff"] for s in sides])),
                n_eff=float(np.mean([s["n_eff"] for s in sides])), sides=len(sides))


def gamma_from(G: dict, n2: dict) -> tuple[float, dict]:
    """gamma (1/(W m)) = sum n2_i G_i, and each material's share of it."""
    parts = {r: float(n2.get(r) or 0.0) * g for r, g in G.items()}
    total = sum(parts.values())
    return total, {r: (p / total if total else float("nan")) for r, p in parts.items()}


def scalar_gamma(n2_core: float, wl_um: float, aeff_um2: float) -> float:
    """The core-only form 2 pi n2 / (lambda A_eff), 1/(W m)."""
    return 2 * math.pi * n2_core / (wl_um * 1e-6 * aeff_um2 * 1e-12)
