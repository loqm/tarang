"""Regenerate the README's quick-start figure (docs/assets/quickstart_results.png).

    python docs/make_readme_figures.py              # replot from the saved map data (1 min)
    python docs/make_readme_figures.py --recompute  # also re-solve the map: ~30 min, no credits

Three panels, all from TARANG's local tools:
  (a) refractive index of the fitted Si3N4 and SiO2 models (README quick start, step 1);
  (b) the TE00 mode of the quick-start waveguide, 0.7 um x 1.2 um at 1550 nm (steps 2-3);
  (c) a dispersion design map over core height and width (step 4 and the
      dispersion-mapping tools): D of TE00, the D = 0 contour, and the multimode region.
The map uses 50 mesh steps per wavelength, the minimum the dispersion-mapping README
recommends for maps. Its data are kept in docs/assets/quickstart_dispersion_map.csv, so
the figure can be restyled without solving again. The colour scale is clipped at
±600 ps/(nm·km): next to cut-off (0.5 µm × 0.6 µm) D falls far below that.
"""
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patheffects  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "quickstart_results.png"
MAP_DATA = ROOT / "docs" / "assets" / "quickstart_dispersion_map.csv"
D_CLIP = 600.0
C_LIGHT = 2.99792458e14                     # µm/s

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SIN, OXIDE = "#2a78d6", "#d98032"

HEIGHTS = [0.5, 0.6, 0.7, 0.8, 0.9]
WIDTHS = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]
STEPS = 50


def _point(hw):
    from tarang.waveguide.dispersion_mapping import solve_point
    h, w = hw
    r = solve_point(h, w, steps=STEPS)
    return h, w, r["D_TE0"], r["guided_TE"]


def _style(ax):
    ax.set_box_aspect(0.625)
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_color(MUTED)
    ax.tick_params(colors=MUTED)


def _map_points(recompute: bool, workers: int):
    if MAP_DATA.exists() and not recompute:
        with open(MAP_DATA, newline="", encoding="utf-8") as f:
            return [(float(r["H_um"]), float(r["W_um"]), float(r["D_TE0_ps_nm_km"]),
                     int(r["guided_TE"])) for r in csv.DictReader(f)]
    grid = [(h, w) for h in HEIGHTS for w in WIDTHS]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pts = list(pool.map(_point, grid))
    with open(MAP_DATA, "w", newline="", encoding="utf-8") as f:
        out = csv.writer(f, lineterminator="\n")
        out.writerow(["H_um", "W_um", "D_TE0_ps_nm_km", "guided_TE"])
        out.writerows(pts)
    return pts


def main(recompute: bool = False, workers: int = 2) -> Path:
    from tarang.geometries import ridge_waveguide
    from tarang.media import get_medium

    plt.rcParams.update({"font.size": 14, "axes.labelcolor": INK, "axes.titlesize": 15})
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=120, facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1, 1, 1.18], "wspace": 0.28})

    # (a) fitted materials
    sin = get_medium("si3n4", wl_range=(1.2, 2.0))
    sio2 = get_medium("sio2", wl_range=(1.2, 2.0))
    wl = np.linspace(1.2, 2.0, 300)
    ax = axes[0]
    _style(ax)
    ax.grid(True, color=GRID)
    ax.plot(wl, np.real(sin.nk_model(C_LIGHT / wl)[0]), color=SIN, lw=2, label="Si₃N₄ (si3n4)")
    ax.plot(wl, np.real(sio2.nk_model(C_LIGHT / wl)[0]), color=OXIDE, lw=2, label="SiO₂ (sio2)")
    for med, col, dy in ((sin, SIN, -0.07), (sio2, OXIDE, 0.04)):
        n155 = float(np.real(med.nk_model(C_LIGHT / 1.55)[0]))
        ax.plot([1.55], [n155], "o", color=col)
        ax.annotate(f"n(1.55 µm) = {n155:.3f}", (1.55, n155), (1.56, n155 + dy), color=col,
                    fontsize=13)
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("Refractive index n")
    ax.set_title("1  Fitted materials", loc="left", color=INK)
    ax.legend(frameon=False)

    # (b) TE00 mode of the quick-start waveguide
    wg = ridge_waveguide.get_passive_simulation(
        width=1.2, height=0.7, length=2.0, core_medium=sin, box_medium=sio2, box_height=4.0,
        wavelength_um=1.55, fine_steps_per_wvl=50)
    res = wg.get_modes(num_modes=2)
    te = next(m for m in res["modes"] if m["polarization"] == "TE")
    md = res["raw_mode_data"]
    i = te["mode_index"]
    inten = sum(np.abs(np.squeeze(getattr(md, c).isel(mode_index=i, f=0).values)) ** 2
                for c in ("Ex", "Ey", "Ez"))
    y, z = md.Ex.y.values, md.Ex.z.values
    ax = axes[1]
    _style(ax)
    ax.pcolormesh(y, z, (inten / inten.max()).T, cmap="magma", shading="auto")
    ax.add_patch(plt.Rectangle((-0.6, 0), 1.2, 0.7, fill=False, ec="white", lw=1.2))
    ax.axhline(0, color="white", lw=0.8, ls="--")
    ax.text(-1.5, -0.4, "SiO₂", color="white", fontsize=13)
    ax.text(-1.5, 1.05, "air", color="white", fontsize=13)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-0.65, 1.35)                # 2.0 / 3.2 = box aspect 0.625: true proportions
    ax.set_xlabel("y (µm)")
    ax.set_ylabel("z (µm)")
    ax.set_title(f"2-3  TE00 of 0.7 × 1.2 µm\n      n_eff = {te['neff']:.3f}", loc="left", color=INK)

    # (c) dispersion design map
    pts = _map_points(recompute, workers)
    D = np.full((len(HEIGHTS), len(WIDTHS)), np.nan)
    nte = np.zeros_like(D)
    for h, w, d, g in pts:
        D[HEIGHTS.index(h), WIDTHS.index(w)] = d
        nte[HEIGHTS.index(h), WIDTHS.index(w)] = g
    ax = axes[2]
    _style(ax)
    pc = ax.pcolormesh(WIDTHS, HEIGHTS, D, cmap="RdBu_r", vmin=-D_CLIP, vmax=D_CLIP,
                       shading="nearest")
    ax.contour(WIDTHS, HEIGHTS, D, levels=[0], colors=INK, linewidths=1.5)
    for (hi, wi), n in np.ndenumerate(nte):
        if n >= 2:
            ax.text(WIDTHS[wi], HEIGHTS[hi], "MM", ha="center", va="center", fontsize=11, color=INK,
                    fontweight="bold", path_effects=[patheffects.withStroke(linewidth=3,
                                                                           foreground="white")])
    cb = fig.colorbar(pc, ax=ax, pad=0.03, shrink=0.62, extend="both")
    cb.set_label("D of TE00 (ps/(nm·km))")
    ax.set_xlabel("Width W (µm)")
    ax.set_ylabel("Height H (µm)")
    ax.set_title("4  Dispersion map at 1550 nm\n    black line: D = 0;  MM: multimode", loc="left",
                 color=INK)

    fig.suptitle("Quick start, all local and free: air-clad Si₃N₄ on 4 µm SiO₂", x=0.01, y=0.9,
                 ha="left", color=INK, fontsize=16, fontweight="bold")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return OUT


if __name__ == "__main__":
    print(main(recompute="--recompute" in sys.argv))
