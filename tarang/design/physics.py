"""Closed-form helpers used between the simulation stages.

Each function is a textbook relation, kept separate so that it can be tested against
known values and reused outside the design tool.
"""
from __future__ import annotations

import math

import numpy as np

C_UM_S = 2.99792458e14          # speed of light, um/s


def radius_for_fsr(fsr_ghz: float, n_group: float) -> float:
    """Ring radius (um) whose FSR is ``fsr_ghz``: FSR = c / (n_g 2 pi R)."""
    return C_UM_S / (n_group * 2 * math.pi * fsr_ghz * 1e9)


def fsr_ghz(radius_um: float, n_group: float) -> float:
    return C_UM_S / (n_group * 2 * math.pi * radius_um) / 1e9


def q_intrinsic(loss_db_cm: float, n_group: float, wavelength_um: float) -> float:
    """Q_i = 2 pi n_g / (lambda alpha), alpha the power attenuation in 1/um."""
    alpha_per_um = loss_db_cm / (10 * math.log10(math.e)) / 1e4
    return 2 * math.pi * n_group / (wavelength_um * alpha_per_um)


def slab_te_index(n_core: float, n_below: float, n_above: float, thickness_um: float,
                  wavelength_um: float) -> float:
    """Effective index of the fundamental TE mode of an asymmetric three-layer slab.

    Solves tan(kappa d) = kappa (g_b + g_a) / (kappa^2 - g_b g_a) on the first branch,
    with kappa = k0 sqrt(n_core^2 - n^2), g_x = k0 sqrt(n^2 - n_x^2).
    """
    k0 = 2 * math.pi / wavelength_um
    n_clad = max(n_below, n_above)
    if n_core <= n_clad:
        raise ValueError("core index must exceed both claddings")

    def f(n):
        kap = k0 * math.sqrt(n_core ** 2 - n ** 2)
        gb = k0 * math.sqrt(n ** 2 - n_below ** 2)
        ga = k0 * math.sqrt(n ** 2 - n_above ** 2)
        # kappa d - atan(gb/kappa) - atan(ga/kappa) = 0 for the fundamental mode
        return kap * thickness_um - math.atan(gb / kap) - math.atan(ga / kap)

    lo, hi = n_clad + 1e-12, n_core - 1e-12
    if f(lo) <= 0:
        raise ValueError("slab is below cut-off")
    for _ in range(200):                      # bisection: f decreases with n
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def grating_period(n_slab: float, n_groove: float, fill_factor: float, n_cover: float,
                   fiber_angle_deg: float, wavelength_um: float) -> float:
    """First-order Bragg period coupling a fibre at ``fiber_angle_deg`` from vertical
    (tilted against the propagation direction): Lambda = lambda / (n_avg - n_cover sin theta),
    with n_avg the fill-factor average of the tooth (slab) and groove indices."""
    n_avg = fill_factor * n_slab + (1 - fill_factor) * n_groove
    return wavelength_um / (n_avg - n_cover * math.sin(math.radians(fiber_angle_deg)))


def gap_for_over_coupling(critical: dict, ratio: float) -> float:
    """Gap giving Q_int / Q_coupling = ``ratio`` from a ring_gap_sweep fit.

    The fit is theta(gap) = theta_0 exp(-gamma gap), with power coupling sin^2(theta);
    critical coupling needs sin^2(theta) = 1 - a^2 (round-trip loss). In the high-Q
    limit the coupling rate scales with sin^2(theta), so K times critical needs
    sin^2(theta) = K (1 - a^2)."""
    theta0, gamma = critical["fit_params"]["theta_0"], critical["fit_params"]["gamma"]
    cross = ratio * critical["target_cross_power"]
    if not 0 < cross < 1:
        raise ValueError(f"over-coupling ratio {ratio} needs a power coupling of {cross:.3g}")
    theta = math.asin(math.sqrt(cross))
    return (math.log(theta0) - math.log(theta)) / gamma


def loaded_q(q_int: float, ratio: float) -> float:
    """Loaded Q for Q_int / Q_coupling = ratio: 1/Q_L = 1/Q_i + 1/Q_c."""
    return q_int / (1 + ratio)


def round_to(x: float, step: float) -> float:
    return float(np.round(x / step) * step)


def thermal_shift_pm_per_K(wavelength_um: float, n_group: float, core_fraction: float,
                           dn_dT_core: float, dn_dT_box: float) -> tuple[float, float]:
    """Resonance shift with temperature, dλ/dT = (λ / n_g) dn_eff/dT, in pm/K.

    dn_eff/dT is the power-weighted sum of the materials' dn/dT (first-order perturbation:
    the fraction of the mode's energy in each material). Only the core fraction Γ is known
    from the mode solve; the rest is split between BOX and air (dn/dT of air about -1e-6,
    taken as 0), so the result is a range: rest all in air (low) to rest all in the BOX
    (high)."""
    g = min(max(float(core_fraction), 0.0), 1.0)
    lo = g * dn_dT_core
    hi = lo + (1.0 - g) * dn_dT_box
    k = wavelength_um * 1e6 / n_group                       # µm -> pm, divided by n_g
    return lo * k, hi * k
