"""Published dispersion formulas behind the tabulated materials of tarang/media/data.

TARANG fits every material from a table of n (and k) against wavelength. For materials
whose source gives a formula instead of a table, the table was written from the
formula below, and the tests check each file against it. Run this module to write them
again:

    python -m tarang.media.index_models

Wavelengths in µm. Each function returns the real refractive index.

Exposes: n_ta2o5_black(), n_gap_wei(), n_algaas_gehrsitz(), e_gamma_gaas_ev(), TABLES.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

#: h c / e in eV µm, as used by Gehrsitz et al. to convert eV to (µm)^-1
EV_UM = 1.239856
#: Boltzmann constant in eV/K (Gehrsitz et al.: 0.0861708 meV/K)
K_B_EV = 0.0861708e-3


def n_ta2o5_black(wl_um):
    """Ion-beam-sputtered Ta2O5: Black et al., Opt. Lett. 46, 817 (2021), Eq. 1, valid
    0.5-5 µm. Poles read as squares of 0.368, 0.1639 and 14.5 µm."""
    l2 = np.asarray(wl_um, dtype=float) ** 2
    return np.sqrt(1 + 0.033 * l2 / (l2 - 0.368 ** 2) + 3.212 * l2 / (l2 - 0.1639 ** 2)
                   + 3.747 * l2 / (l2 - 14.5 ** 2))


def n_gap_wei(wl_um, T_K: float = 295.0):
    """GaP: Wei et al., Opt. Mater. Express 8, 485 (2018), Eq. 3 with Table 2, valid
    0.7-12.5 µm and 78-450 K:  n^2 = A(T) + B(T)/(λ^2 - C) + D(T)/(λ^2 - E)."""
    T = float(T_K)
    l2 = np.asarray(wl_um, dtype=float) ** 2
    A = 10.926 + 7.0787e-4 * T + 1.8594e-7 * T ** 2
    B = 0.53718 + 5.8035e-5 * T + 1.9819e-7 * T ** 2
    C = 0.0911014
    D = 1504 + 0.25935 * T - 0.00023326 * T ** 2
    E = 758.048
    return np.sqrt(A + B / (l2 - C) + D / (l2 - E))


def e_gamma_gaas_ev(T_K: float) -> float:
    """Direct gap of GaAs against temperature, Gehrsitz et al. Eq. 11 (after Proctor)."""
    kT2 = 2 * K_B_EV * float(T_K)
    e_deb, s, e_to, s_to = 15.9e-3, 1.8, 33.6e-3, 1.1
    return (1.5192 + s * e_deb * (1 - 1 / np.tanh(e_deb / kT2))
            + s_to * e_to * (1 - 1 / np.tanh(e_to / kT2)))


def e_gamma_algaas_ev(x: float, T_K: float) -> float:
    """Direct gap of AlxGa1-xAs, Gehrsitz et al. Eq. 14."""
    return e_gamma_gaas_ev(T_K) + 1.36 * x + 0.22 * x ** 2


def n_algaas_gehrsitz(wl_um, x: float, T_K: float = 296.15):
    """AlxGa1-xAs below the direct gap: Gehrsitz et al., J. Appl. Phys. 87, 7825 (2000),
    Eqs. 11-14 with Table IV (composition polynomials) and Table II (GaAs Fit 2
    temperature terms; GaAs- and AlAs-like reststrahl terms). Energies in (µm)^-1:
      n^2 = A + C0/(E0^2 - E^2) + C1/(E1^2 - E^2) + R,  E = 1/λ."""
    T, x = float(T_K), float(x)
    E2 = 1.0 / np.asarray(wl_um, dtype=float) ** 2
    A0 = 5.9613 + 7.178e-4 * T - 0.953e-6 * T ** 2               # Table II, GaAs Fit 2
    E10sq = 4.7171 - 3.237e-4 * T - 1.358e-6 * T ** 2
    A = A0 - 16.159 * x + 43.511 * x ** 2 - 71.317 * x ** 3 + 57.535 * x ** 4 - 17.451 * x ** 5
    C1 = 21.5647 + 113.74 * x - 122.5 * x ** 2 + 108.401 * x ** 3 - 47.318 * x ** 4
    E1sq = E10sq + 11.006 * x - 3.08 * x ** 2
    C0 = 1.0 / (50.535 - 150.7 * x - 62.209 * x ** 2 + 797.16 * x ** 3 - 1125 * x ** 4
                + 503.79 * x ** 5)
    E0 = e_gamma_gaas_ev(T) / EV_UM + 1.1308 * x + 0.1436 * x ** 2
    C2, E2sq_gaas = 1.55e-3, 0.724e-3                            # Table II: GaAs-like phonon
    C3, E3sq_alas = 2.61e-3, 1.331e-3                            # AlAs-like phonon
    R = (1 - x) * C2 / (E2sq_gaas - E2) + x * C3 / (E3sq_alas - E2)
    return np.sqrt(A + C0 / (E0 ** 2 - E2) + C1 / (E1sq - E2) + R)


#: file -> (function of wavelength, first, last, step in µm)
TABLES = {
    "Black_Ta2O5_Sellmeier.csv": (n_ta2o5_black, 0.50, 5.00, 0.01),
    "Wei_GaP_295K.csv": (lambda w: n_gap_wei(w, 295.0), 0.70, 12.50, 0.01),
    "Gehrsitz_Al0.20Ga0.80As_296K.csv": (lambda w: n_algaas_gehrsitz(w, 0.20, 296.15),
                                         0.80, 3.00, 0.005),
}


def grid(first: float, last: float, step: float) -> np.ndarray:
    n = int(round((last - first) / step)) + 1
    return np.round(first + step * np.arange(n), 6)


def write(name: str, folder: Path | None = None) -> Path:
    """Write one table: header 'wl,n', LF line endings, n to 10 decimals."""
    fn, first, last, step = TABLES[name]
    folder = Path(folder) if folder else Path(__file__).with_name("data")
    wl = grid(first, last, step)
    n = fn(wl)
    path = folder / name
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        f.write("wl,n\n")
        for a, b in zip(wl, n):
            f.write(f"{a:.3f},{b:.10f}\n")
    return path


if __name__ == "__main__":
    for name in TABLES:
        print("wrote", write(name))
