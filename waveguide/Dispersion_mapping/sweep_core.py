"""
sweep_core.py -- one (H, W) point: polarization-resolved modes + TE00/TM00 dispersion
===================================================================================
Shared by run_air_sweep.py (computes) and plot_air_sweep.py (plots).
Lives in <tarang repo>/scripts/. Builds geometry with TARANG's ridge_waveguide and
materials with TARANG's get_medium(), but replaces the parts of
RidgeWaveguide.get_dispersion() that were found to be unreliable (see README).

Physics
-------
    beta(w) = n_eff(w) * w / c
    D  = -(lambda / c) * d^2 n_eff / d lambda^2          [ps / (nm km)]
    beta2 = -D * lambda^2 / (2 pi c)                      [ps^2 / km]
    n_g = n_eff - lambda * d n_eff / d lambda
    D > 0  <=>  beta2 < 0  <=>  anomalous (bright solitons possible)

Guided-mode criterion (AIR top cladding):
    The highest surrounding index is the SiO2 substrate, so a mode is guided only if
    n_eff > n_SiO2(lambda0).  Modes just above that line are weakly guided and leak in
    bends, so we also report a practical threshold n_SiO2 + margin (default 0.005).

Polarization:
    TE fraction = int|Ey|^2 / (int|Ey|^2 + int|Ez|^2)   (y = width, z = height)
    > 0.6 -> TE-like, < 0.4 -> TM-like, otherwise 'hybrid' (strong TE/TM mixing).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]   # <repo>/scripts/sweep_core.py -> <repo>
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import tidy3d as td
from tidy3d.plugins.mode import ModeSolver

from tarang import configure_logging
from media.medium_helper import get_medium
import geometries.ridge_waveguide as rw

configure_logging(level="WARNING")
try:                                   # tidy3d >= 2.12
    td.config.logging.level = "ERROR"
except Exception:                      # older tidy3d
    try:
        td.config.logging_level = "ERROR"
    except Exception:
        pass

C_UM_PS = td.C_0 / 1e12                # speed of light in um/ps

# ---- Materials ---------------------------------------------------------------
# Si3N4: TARANG's fitted Luke et al. (LPCVD stoichiometric) data.
SIN = get_medium("si3n4", wl_range=(1.1, 2.4))
# SiO2: Malitson fused-silica Sellmeier (n = 1.444 @ 1550 nm).  TARANG's bundled
# Nyakuchena file gives 1.420 and only spans 1.10-1.65 um, so it is NOT used here.
SIO2 = td.Sellmeier(coeffs=[(0.6961663, 0.0684043 ** 2),
                            (0.4079426, 0.1162414 ** 2),
                            (0.8974794, 9.896161 ** 2)])


def n_of(medium, wl_um: float) -> float:
    return float(np.real(np.sqrt(medium.eps_model(td.C_0 / wl_um))))


def _te_frac(data, m: int, f: int) -> float:
    ey = np.abs(data.Ey.isel(mode_index=m, f=f).values) ** 2
    ez = np.abs(data.Ez.isel(mode_index=m, f=f).values) ** 2
    return float(ey.sum() / (ey.sum() + ez.sum() + 1e-30))


def _pol(tf: float) -> str:
    return "TE" if tf > 0.6 else ("TM" if tf < 0.4 else "hybrid")


def _dispersion(wls, neff, wl0):
    """4th-order polynomial fit of n_eff(lambda); analytic derivatives at wl0."""
    p = np.polyfit(wls - wl0, neff, 4)
    d1, d2 = p[-2], 2.0 * p[-3]
    D = -(wl0 / C_UM_PS) * d2 * 1e6                                   # ps/(nm km)
    beta2 = -D * (wl0 * 1e3) ** 2 / (2 * np.pi * C_UM_PS * 1e3)        # ps^2/km
    n0 = float(np.polyval(p, 0.0))
    ng = n0 - wl0 * d1
    resid = float(np.max(np.abs(np.polyval(p, wls - wl0) - neff)))    # fit quality
    return float(D), float(beta2), float(ng), resid


SI = td.Medium(permittivity=3.4757 ** 2)     # Si handle at 1550 nm (only the tail sees it)


def build_sim(height, width, *, wl0=1.55, steps=50, box=4.0, si_handle=False,
              si_thick=2.0, margin_y=2.0):
    """Air / SiN core / SiO2 BOX [/ Si handle].  TARANG's builder stops at the BOX
    bottom; with si_handle=True a Si layer is appended below it (use with PML so the
    Si acts as a semi-infinite, absorbing handle, i.e. substrate leakage is captured)."""
    wg = rw.get_passive_simulation(
        width=width, height=height, length=2.0,
        core_medium=SIN, sub_medium=SIO2, wavelength_um=wl0,
        substrate_height=box, top_cladding_medium=None, top_cladding_height=0.0,
        fine_min_steps_per_wvl=steps, margin_y=margin_y)
    sim = wg.sim
    if si_handle:
        (x0, y0, z0), (x1, y1, z1) = sim.bounds
        z0_new = z0 - si_thick
        si = td.Structure(geometry=td.Box(center=(0, 0, -box - si_thick),
                                          size=(td.inf, td.inf, 2 * si_thick)), medium=SI)
        sim = sim.updated_copy(center=(sim.center[0], sim.center[1], (z0_new + z1) / 2),
                               size=(sim.size[0], sim.size[1], z1 - z0_new),
                               structures=list(sim.structures) + [si])
    return sim


def solve_point(height: float, width: float, *, wl0: float = 1.55,
                span: float = 0.2, npts: int = 9, num_modes: int = 10,
                steps: int = 50, box: float = 4.0, margin: float = 0.005,
                si_handle: bool = False, pml: int = 0,
                bend_radius: float | None = None) -> dict:
    """Air-clad SiN ridge on SiO2 BOX.  All lengths in um.

    si_handle   : add a Si handle below the BOX (real wafer stack)
    pml         : absorbing layers at the mode-plane edges (0 = conducting walls).
                  Needed for leakage / bend-loss values; forced >= 12 for bends.
    bend_radius : radius (um) measured to the waveguide centre; None = straight.
                  n_eff of a bent mode is referenced to this radius, so D computed
                  from it is the effective dispersion of a ring of that radius.
    """
    bent = bend_radius is not None
    if bent or si_handle:
        pml = max(pml, 12)
    sim = build_sim(height, width, wl0=wl0, steps=steps, box=box, si_handle=si_handle,
                    margin_y=(3.0 if bent else 2.0))
    wls = np.linspace(wl0 - span / 2, wl0 + span / 2, npts)
    plane = td.Box(center=(0, 0, sim.center[2]), size=(0, sim.size[1], sim.size[2]))
    spec_kw = dict(num_modes=num_modes, target_neff=n_of(SIN, wl0))
    if pml:
        spec_kw["num_pml"] = (pml, pml)
    if bent:
        spec_kw.update(bend_radius=float(bend_radius), bend_axis=1)  # bend about z
    data = ModeSolver(simulation=sim, plane=plane, freqs=td.C_0 / wls,
                      mode_spec=td.ModeSpec(**spec_kw)).solve()

    # Fundamental of each polarization re-identified at EVERY wavelength
    te0, tm0 = [], []
    for fi in range(npts):
        te, tm = [], []
        for m in range(num_modes):
            n = float(np.real(data.n_eff.isel(f=fi, mode_index=m)))
            (te if _te_frac(data, m, fi) > 0.5 else tm).append(n)
        te0.append(max(te) if te else np.nan)
        tm0.append(max(tm) if tm else np.nan)
    te0, tm0 = np.array(te0), np.array(tm0)
    fc = npts // 2
    m_te0 = max((m for m in range(num_modes) if _te_frac(data, m, fc) > 0.5),
                key=lambda m: float(np.real(data.n_eff.isel(f=fc, mode_index=m))))
    # Tidy3D stores Re(n) in n_eff and Im(n) in k_eff (n_eff itself is real-valued)
    k_te0 = float(np.real(data.k_eff.isel(f=fc, mode_index=m_te0)))
    # power loss: alpha = 4 pi k / lambda ; dB/cm = 10 log10(e) * alpha[1/cm]
    loss_te0_db_cm = 10 * np.log10(np.e) * 4 * np.pi * abs(k_te0) / (wl0 * 1e-4)
    E2 = sum(np.abs(getattr(data, c).isel(mode_index=m_te0, f=fc).values) ** 2
             for c in ("Ex", "Ey", "Ez"))
    y = data.Ey.coords["y"].values; z = data.Ey.coords["z"].values
    E2 = np.squeeze(E2)
    dA = np.outer(np.gradient(y), np.gradient(z)).reshape(E2.shape) if E2.ndim == 2 else 1.0
    aeff = float((np.sum(E2 * dA)) ** 2 / np.sum(E2 ** 2 * dA))   # um^2
    # Confinement factor: fraction of |E|^2 inside the SiN core (|y| < W/2, 0 < z < H)
    if E2.ndim == 2:
        Y, Z = np.meshgrid(y, z, indexing="ij")
        core = (np.abs(Y) <= width / 2) & (Z >= 0) & (Z <= height)
        gamma_core = float(np.sum(E2 * dA * core) / np.sum(E2 * dA))
    else:
        gamma_core = float("nan")
    D_te, b2_te, ng_te, res_te = _dispersion(wls, te0, wl0)
    D_tm, b2_tm, ng_tm, res_tm = _dispersion(wls, tm0, wl0)

    # Mode census at wl0
    n_cut = n_of(SIO2, wl0)
    modes = []
    for m in range(num_modes):
        n = float(np.real(data.n_eff.isel(f=fc, mode_index=m)))
        tf = _te_frac(data, m, fc)
        modes.append(dict(neff=round(n, 5), te_frac=round(tf, 3), pol=_pol(tf),
                          guided=bool(n > n_cut), guided_margin=bool(n > n_cut + margin)))
    modes.sort(key=lambda d: -d["neff"])

    def count(pols, key):
        return sum(1 for d in modes if d["pol"] in pols and d[key])

    te_like = [d for d in modes if d["te_frac"] >= 0.5]
    tm_like = [d for d in modes if d["te_frac"] < 0.5]
    return dict(
        H=round(height, 4), W=round(width, 4), wl0=wl0, steps=steps, n_cut=round(n_cut, 5),
        margin=margin, si_handle=si_handle, pml=pml, bend_radius=bend_radius,
        loss_TE0_dB_cm=float(loss_te0_db_cm), Aeff_TE0_um2=round(aeff, 4),
        Gamma_TE0=round(gamma_core, 4),
        D_TE0=round(D_te, 2), beta2_TE0=round(b2_te, 2), ng_TE0=round(ng_te, 4),
        D_TM0=round(D_tm, 2), beta2_TM0=round(b2_tm, 2), ng_TM0=round(ng_tm, 4),
        fit_resid_TE0=res_te, fit_resid_TM0=res_tm,
        neff_TE=[d["neff"] for d in te_like],      # sorted high -> low: TE0, TE1, ...
        neff_TM=[d["neff"] for d in tm_like],      # TM0, TM1, ...
        hybrid_TE=[d["pol"] == "hybrid" for d in te_like],
        hybrid_TM=[d["pol"] == "hybrid" for d in tm_like],
        # Guided counts: TE-dominant (>= 0.5) incl. hybrids, strict and with margin
        guided_TE=sum(1 for d in te_like if d["guided"]),
        guided_TM=sum(1 for d in tm_like if d["guided"]),
        guided_TE_margin=sum(1 for d in te_like if d["guided_margin"]),
        guided_TM_margin=sum(1 for d in tm_like if d["guided_margin"]),
        guided_hybrid=count(("hybrid",), "guided"),
        # In a bend the conformal-mapped cladding index rises outward, so n_eff > n_SiO2
        # no longer means 'guided' (cladding whispering-gallery modes appear).  Use the
        # STRAIGHT-guide counts for single-mode decisions; bends give D, n_g, loss only.
        guided_counts_valid=not bent,
        te0_gap_to_next=round(float(te_like[0]["neff"] - te_like[1]["neff"]), 5)
        if len(te_like) > 1 else None,
        modes=modes,
    )
