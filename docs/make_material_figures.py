"""Regenerate the material-fit figures used by media_examples.ipynb.

    python docs/make_material_figures.py                 # si3n4_hong and si3n4
    python docs/make_material_figures.py sio2 si_franta  # any keys in tarang/media/media.json

Each figure shows a material's tabulated refractive index (points) against TARANG's
pole-residue fit over the whole data range (line), with the fit residual in a second
panel. Figures are written to docs/assets/material_fit_<key>.png. Local only; no
FlexCredits.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tarang.media import get_medium  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets"
C_LIGHT = 2.99792458e14                     # speed of light in µm/s

FIT = "#2a78d6"                             # series colour for the fitted curve
DATA = "#52514e"                            # neutral ink for the measured points
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

LABELS = {
    "si3n4": "Si₃N₄, Luke et al. (2015)",
    "si3n4_hong": "LPCVD SiN, Hong et al. (2021)",
    "sio2": "Fused silica, Malitson (1965)",
    "sio2_nyakuchena": "Silica microspheres, Nyakuchena et al. (2023)",
    "si": "Crystalline Si, Wang et al. (2025)",
    "si_franta": "Float-zone c-Si, Franta et al. (2017)",
}


def figure(key: str) -> Path:
    media = ROOT / "tarang" / "media"
    database = json.loads((media / "media.json").read_text(encoding="utf-8"))
    data = pd.read_csv(media / database[key], skipinitialspace=True)
    wl, n = data["wl"].to_numpy(float), data["n"].to_numpy(float)
    medium = get_medium(key, wl_range=(wl.min(), wl.max()), force_zero_k=True)

    wl_fine = np.linspace(wl.min(), wl.max(), 600)
    n_fine = np.real(medium.nk_model(C_LIGHT / wl_fine)[0])
    residual = np.real(medium.nk_model(C_LIGHT / wl)[0]) - n

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.0, 5.2), sharex=True, dpi=150,
                                      gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.08},
                                      facecolor=SURFACE)
    for ax in (top, bottom):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    top.plot(wl, n, linestyle="none", marker="o", markersize=4.5, markerfacecolor=SURFACE,
             markeredgecolor=DATA, markeredgewidth=1.2, label=f"tabulated data ({len(wl)} points)")
    top.plot(wl_fine, n_fine, color=FIT, linewidth=2, label="TARANG pole-residue fit")
    top.set_ylabel("refractive index n")
    top.legend(frameon=False, labelcolor=INK)
    top.set_title(LABELS.get(key, key), loc="left", color=INK, fontsize=11)

    exponent = int(np.floor(np.log10(np.abs(residual).max() or 1.0)))
    superscript = str(exponent).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    bottom.axhline(0, color=MUTED, linewidth=0.8)
    bottom.plot(wl, residual / 10.0 ** exponent, linestyle="none", marker="o", markersize=3.5,
                color=FIT)
    bottom.ticklabel_format(axis="y", useOffset=False, style="plain")
    bottom.set_ylabel(f"fit − data\n(×10{superscript})")
    bottom.set_xlabel("wavelength (µm)")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"material_fit_{key}.png"
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"{path.relative_to(ROOT)}: max |fit - data| = {np.abs(residual).max():.1e}")
    return path


if __name__ == "__main__":
    for key in sys.argv[1:] or ["si3n4_hong", "si3n4"]:
        figure(key)
