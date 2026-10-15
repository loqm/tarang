"""
tarang.waveguide.dispersion_mapping.core
=================================
One ridge-waveguide cross-section: polarization-resolved modes and robust dispersion.

All materials come from TARANG's media system (``media.json`` keys), so any core,
substrate or top cladding defined there can be used.  Geometry is built with
``tarang.geometries.ridge_waveguide.get_passive_simulation``; modes are solved with Tidy3D's
LOCAL mode solver (no FlexCredits).

Physics
-------
    D     = -(lambda / c) d2 n_eff / d lambda2          [ps / (nm km)]
    beta2 = -D lambda^2 / (2 pi c)                       [ps^2 / km]
    n_g   = n_eff - lambda d n_eff / d lambda
    D > 0  <=>  anomalous (bright solitons possible)

Improvements over ``RidgeWaveguide.get_dispersion()``
------------------------------------------------------
1. The TE00 (and TM00) mode is re-identified at EVERY wavelength, so TE/TM order
   swaps cannot corrupt the second derivative.
2. n_eff(lambda) is fitted with a 4th-order polynomial before differentiating.
3. Guided modes are counted per polarization against the highest surrounding index.
4. Mesh is user-set (>= 50 steps/lambda recommended; TARANG's default 20 is ~25 % off in D).
5. Optional Si handle, absorbing boundaries, bent waveguides, and a core-index shift
   (for uncertainty budgets).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import tidy3d as td
from tidy3d.plugins.mode import ModeSolver

from tarang import configure_logging, set_tidy3d_log_level
from tarang.media.medium_helper import get_medium
from tarang.geometries import ridge_waveguide as rw

configure_logging(level="WARNING")
set_tidy3d_log_level("ERROR")

C_UM_PS = td.C_0 / 1e12                # speed of light in um/ps
SI_HANDLE = td.Medium(permittivity=3.4757 ** 2)
AIR_KEYS = (None, "", "air", "none")

# Defaults: stoichiometric LPCVD Si3N4 (Luke et al.) on Malitson fused silica, air top
DEFAULT_CORE = "si3n4"
DEFAULT_SUBSTRATE = "sio2"


@lru_cache(maxsize=None)
def medium(key: str, wl_min: float = 1.1, wl_max: float = 2.4):
    """TARANG medium (media.json key or CSV path) fitted over [wl_min, wl_max], cached per
    process."""
    return get_medium(key, wl_range=(wl_min, wl_max))


def fit_band(wl_lo: float, wl_hi: float) -> tuple[float, float]:
    """Wavelength range to fit the materials over so that [wl_lo, wl_hi] is covered.

    Telecom work keeps the long-standing 1.1-2.4 um fit (so earlier results are
    reproduced exactly); any band outside it, e.g. around 780 nm, gets a fit over the
    band plus 5 % on each side instead of an extrapolated one."""
    if 1.1 <= wl_lo and wl_hi <= 2.4:
        return 1.1, 2.4
    return round(wl_lo * 0.95, 4), round(wl_hi * 1.05, 4)


def medium_for(key: str, wl_lo: float, wl_hi: float):
    """Medium fitted over a band that covers [wl_lo, wl_hi] (see fit_band)."""
    return medium(key, *fit_band(wl_lo, wl_hi))


def shifted(med, n_shift: float, wl_ref: float = 1.55):
    """Scale a pole-residue medium so its index at wl_ref shifts by n_shift.
    eps(w) is scaled uniformly, which preserves the dispersion shape."""
    if not n_shift:
        return med
    n0 = n_of(med, wl_ref)
    k = ((n0 + n_shift) / n0) ** 2
    return med.updated_copy(eps_inf=med.eps_inf * k, poles=[(a, c * k) for (a, c) in med.poles])


def n_of(med, wl_um: float) -> float:
    return float(np.real(np.sqrt(med.eps_model(td.C_0 / wl_um))))


def is_air(key) -> bool:
    return (key if key is None else str(key).lower()) in AIR_KEYS


def stack_label(core=DEFAULT_CORE, substrate=DEFAULT_SUBSTRATE, cladding=None, box=4.0) -> str:
    top = "air" if is_air(cladding) else str(cladding)
    return f"{core} core / {top} top / {box:g} µm {substrate}"


def build_sim(height, width, *, wl0=1.55, steps=50, box=4.0, core=DEFAULT_CORE,
              substrate=DEFAULT_SUBSTRATE, cladding=None, cladding_height=2.0,
              n_shift=0.0, si_handle=False, si_thick=2.0, margin_y=2.0, band=None):
    """Tidy3D simulation of the cross-section (lengths in um). Materials are fitted over
    a range covering ``band`` (default wl0 +/- 0.1 um)."""
    lo, hi = band or (wl0 - 0.1, wl0 + 0.1)
    core_m = shifted(medium_for(core, lo, hi), n_shift, wl0)
    sub_m = medium_for(substrate, lo, hi)
    top_m = None if is_air(cladding) else medium_for(cladding, lo, hi)
    wg = rw.get_passive_simulation(
        width=width, height=height, length=2.0, core_medium=core_m, box_medium=sub_m,
        wavelength_um=wl0, box_height=box, top_cladding_medium=top_m,
        top_cladding_height=(cladding_height if top_m is not None else 0.0),
        fine_steps_per_wvl=steps, margin_y=margin_y)
    sim = wg.sim
    if si_handle:
        (x0, y0, z0), (x1, y1, z1) = sim.bounds
        z0_new = z0 - si_thick
        si = td.Structure(geometry=td.Box(center=(0, 0, -box - si_thick),
                                          size=(td.inf, td.inf, 2 * si_thick)), medium=SI_HANDLE)
        sim = sim.updated_copy(center=(sim.center[0], sim.center[1], (z0_new + z1) / 2),
                               size=(sim.size[0], sim.size[1], z1 - z0_new),
                               structures=list(sim.structures) + [si])
    return sim


def _te_frac(data, m: int, f: int) -> float:
    ey = np.abs(data.Ey.isel(mode_index=m, f=f).values) ** 2
    ez = np.abs(data.Ez.isel(mode_index=m, f=f).values) ** 2
    return float(ey.sum() / (ey.sum() + ez.sum() + 1e-30))


def _pol(tf: float) -> str:
    return "TE" if tf > 0.6 else ("TM" if tf < 0.4 else "hybrid")


def dispersion_from_neff(wls, neff, wl0):
    """D, beta2, n_g at wl0 from a 4th-order fit of n_eff(lambda)."""
    p = np.polyfit(wls - wl0, neff, 4)
    d1, d2 = p[-2], 2.0 * p[-3]
    D = -(wl0 / C_UM_PS) * d2 * 1e6
    beta2 = -D * (wl0 * 1e3) ** 2 / (2 * np.pi * C_UM_PS * 1e3)
    ng = float(np.polyval(p, 0.0)) - wl0 * d1
    resid = float(np.max(np.abs(np.polyval(p, wls - wl0) - neff)))
    return float(D), float(beta2), float(ng), resid


def solve_point(height: float, width: float, *, wl0: float = 1.55, span: float = 0.2,
                npts: int = 9, num_modes: int = 10, steps: int = 50, box: float = 4.0,
                margin: float = 0.005, core: str = DEFAULT_CORE,
                substrate: str = DEFAULT_SUBSTRATE, cladding=None,
                cladding_height: float = 2.0, n_shift: float = 0.0,
                si_handle: bool = False, pml: int = 0,
                bend_radius: float | None = None) -> dict:
    """Analyse one cross-section.

    core / substrate / cladding : media.json keys (cladding None or "air" = air top)
    n_shift     : shift of the core index at 1550 nm (uncertainty studies)
    si_handle   : add a Si handle below the BOX
    pml         : absorbing layers at the mode-plane edges (forced >= 12 for bends/Si)
    bend_radius : bend radius in um to the waveguide centre (None = straight).  Guided
                  counts are NOT meaningful in bends (cladding modes appear); use D,
                  n_g, loss and TE/TM splitting only.
    """
    bent = bend_radius is not None
    if bent or si_handle:
        pml = max(pml, 12)
    lo, hi = wl0 - span / 2, wl0 + span / 2
    sim = build_sim(height, width, wl0=wl0, steps=steps, box=box, core=core,
                    substrate=substrate, cladding=cladding, cladding_height=cladding_height,
                    n_shift=n_shift, si_handle=si_handle, margin_y=(3.0 if bent else 2.0),
                    band=(lo, hi))
    wls = np.linspace(lo, hi, npts)
    plane = td.Box(center=(0, 0, sim.center[2]), size=(0, sim.size[1], sim.size[2]))
    core_m = shifted(medium_for(core, lo, hi), n_shift, wl0)
    spec_kw = dict(num_modes=num_modes, target_neff=n_of(core_m, wl0))
    if pml:
        spec_kw["num_pml"] = (pml, pml)
    if bent:
        spec_kw.update(bend_radius=float(bend_radius), bend_axis=1)
    data = ModeSolver(simulation=sim, plane=plane, freqs=td.C_0 / wls,
                      mode_spec=td.ModeSpec(**spec_kw)).solve()

    te0, tm0 = [], []
    for fi in range(npts):
        te, tm = [], []
        for m in range(num_modes):
            n = float(np.real(data.n_eff.isel(f=fi, mode_index=m)))
            (te if _te_frac(data, m, fi) > 0.5 else tm).append(n)
        te0.append(max(te) if te else np.nan)
        tm0.append(max(tm) if tm else np.nan)
    D_te, b2_te, ng_te, res_te = dispersion_from_neff(wls, np.array(te0), wl0)
    D_tm, b2_tm, ng_tm, res_tm = dispersion_from_neff(wls, np.array(tm0), wl0)

    fc = npts // 2
    m_te0 = max((m for m in range(num_modes) if _te_frac(data, m, fc) > 0.5),
                key=lambda m: float(np.real(data.n_eff.isel(f=fc, mode_index=m))))
    k_te0 = float(np.real(data.k_eff.isel(f=fc, mode_index=m_te0)))   # Im(n) lives in k_eff
    loss_db_cm = 10 * np.log10(np.e) * 4 * np.pi * abs(k_te0) / (wl0 * 1e-4)
    E2 = np.squeeze(sum(np.abs(getattr(data, c).isel(mode_index=m_te0, f=fc).values) ** 2
                        for c in ("Ex", "Ey", "Ez")))
    y = data.Ey.coords["y"].values
    z = data.Ey.coords["z"].values
    if E2.ndim == 2:
        dA = np.outer(np.gradient(y), np.gradient(z)).reshape(E2.shape)
        Y, Z = np.meshgrid(y, z, indexing="ij")
        in_core = (np.abs(Y) <= width / 2) & (Z >= 0) & (Z <= height)
        aeff = float(np.sum(E2 * dA) ** 2 / np.sum(E2 ** 2 * dA))
        gamma_core = float(np.sum(E2 * dA * in_core) / np.sum(E2 * dA))
    else:
        aeff, gamma_core = float("nan"), float("nan")

    n_sub = n_of(medium_for(substrate, lo, hi), wl0)
    n_top = 1.0 if is_air(cladding) else n_of(medium_for(cladding, lo, hi), wl0)
    n_cut = max(n_sub, n_top)
    modes = []
    for m in range(num_modes):
        n = float(np.real(data.n_eff.isel(f=fc, mode_index=m)))
        tf = _te_frac(data, m, fc)
        modes.append(dict(neff=round(n, 5), te_frac=round(tf, 3), pol=_pol(tf),
                          guided=bool(n > n_cut), guided_margin=bool(n > n_cut + margin)))
    modes.sort(key=lambda d: -d["neff"])
    te_like = [d for d in modes if d["te_frac"] >= 0.5]
    tm_like = [d for d in modes if d["te_frac"] < 0.5]
    return dict(
        H=round(height, 4), W=round(width, 4), wl0=wl0, steps=steps,
        core=core, substrate=substrate, cladding=(None if is_air(cladding) else cladding),
        box=box, n_shift=n_shift, n_cut=round(n_cut, 5), margin=margin,
        si_handle=si_handle, pml=pml, bend_radius=bend_radius,
        loss_TE0_dB_cm=float(loss_db_cm), Aeff_TE0_um2=round(aeff, 4),
        Gamma_TE0=round(gamma_core, 4),
        D_TE0=round(D_te, 2), beta2_TE0=round(b2_te, 2), ng_TE0=round(ng_te, 4),
        D_TM0=round(D_tm, 2), beta2_TM0=round(b2_tm, 2), ng_TM0=round(ng_tm, 4),
        fit_resid_TE0=res_te, fit_resid_TM0=res_tm,
        neff_TE=[d["neff"] for d in te_like], neff_TM=[d["neff"] for d in tm_like],
        hybrid_TE=[d["pol"] == "hybrid" for d in te_like],
        hybrid_TM=[d["pol"] == "hybrid" for d in tm_like],
        guided_TE=sum(1 for d in te_like if d["guided"]),
        guided_TM=sum(1 for d in tm_like if d["guided"]),
        guided_TE_margin=sum(1 for d in te_like if d["guided_margin"]),
        guided_TM_margin=sum(1 for d in tm_like if d["guided_margin"]),
        guided_counts_valid=not bent,
        te0_gap_to_next=(round(te_like[0]["neff"] - te_like[1]["neff"], 5)
                         if len(te_like) > 1 else None),
        modes=modes,
    )
