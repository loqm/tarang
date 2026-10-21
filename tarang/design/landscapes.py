"""Efficiency landscapes: contour maps of the coupling efficiency over two design
parameters, drawn from simulations that were already run (free).

Contours are linear interpolation between the simulated points (Delaunay triangles), so
they are only as good as the points under them: every simulated point is drawn as a dot,
and the best simulated point is labelled with its value. The chosen design is a red X.

- 5g: uniform grating, period x fill factor (each point: the best fibre position
  simulated there), efficiency at the design wavelength and peak efficiency.
- 9c: apodised grating, fibre position x period scale (all teeth stretched by one factor).
- 10a: focusing grating, distance from the waveguide tip to the first tooth x tip width
  (free in-plane model; see tarang.design.focus).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def _panel(ax, fig, panel: dict):
    import matplotlib.tri as mtri
    pts = panel["points"]
    x = np.array([p["x"] for p in pts], float)
    y = np.array([p["y"] for p in pts], float)
    z = np.array([p["z"] for p in pts], float) * 100
    vmax = max(float(np.nanmax(z)), 1e-3)
    from matplotlib.ticker import MaxNLocator
    levels = MaxNLocator(nbins=14).tick_values(0, vmax)
    if levels[-1] < vmax:
        levels = np.append(levels, levels[-1] + (levels[1] - levels[0]))
    drawn = None
    if len(pts) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0:
        try:
            # scale the axes to similar ranges so the triangles are not slivers
            tri = mtri.Triangulation((x - x.min()) / np.ptp(x), (y - y.min()) / np.ptp(y))
            tri = mtri.Triangulation(x, y, triangles=tri.triangles)
            drawn = ax.tricontourf(tri, z, levels=levels, cmap="viridis")
            allowed = np.array([1.0 if p.get("allowed", True) else 0.0 for p in pts])
            if (allowed < 1).any():
                ax.tricontourf(tri, allowed, levels=[-0.5, 0.5], colors="none", hatches=["///"])
        except (RuntimeError, ValueError):
            drawn = None
    sc = ax.scatter(x, y, c=z, cmap="viridis", vmin=0, vmax=vmax, s=18 if drawn else 140,
                    edgecolors="w" if drawn else "k", linewidths=0.6, zorder=3,
                    label="simulated points" if not panel.get("model") else "computed points")
    ok = np.array([bool(p.get("allowed", True)) for p in pts])
    k = int(np.nanargmax(np.where(ok, z, -np.inf))) if ok.any() else int(np.nanargmax(z))
    ax.annotate(f"{z[k]:.1f} %", (x[k], y[k]), xytext=(6, 6), textcoords="offset points",
                fontsize=8, color="w" if drawn else "k", zorder=4,
                bbox=dict(boxstyle="round,pad=0.2", fc="k", alpha=0.35, lw=0) if drawn else None)
    if panel.get("chosen") is not None:
        cx, cy = panel["chosen"]
        ax.scatter([cx], [cy], marker="x", s=130, color="red", linewidths=2.6, zorder=5,
                   label=panel.get("chosen_label", "chosen design"))
    ax.set(xlabel=panel["xlabel"], ylabel=panel["ylabel"])
    ax.set_title(panel["title"], fontsize=10)
    fig.colorbar(drawn if drawn is not None else sc, ax=ax, label="Efficiency (%)")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    if panel.get("note"):
        ax.text(0.0, -0.32, panel["note"], transform=ax.transAxes, fontsize=7.5, color="0.35")


def draw(path: Path, panels: list[dict], suptitle: str = "") -> Path:
    """One figure with a contour panel per entry of ``panels`` (points: dicts with x, y, z
    (efficiency 0-1) and optionally allowed; title, xlabel, ylabel, chosen (x, y),
    chosen_label, note)."""
    from tarang.design import visuals
    plt = visuals._plt()
    fig, axes = plt.subplots(1, len(panels), figsize=(6.6 * len(panels), 5.6), squeeze=False)
    for ax, panel in zip(axes[0], panels):
        _panel(ax, fig, panel)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.subplots_adjust(bottom=0.27, wspace=0.3)
    return visuals._save(fig, Path(path))


# ------------------------------------------------------------------ per design
def uniform_points(rows: list[dict]) -> list[dict]:
    """2D grating runs grouped by (period, fill factor): the best fibre position at each."""
    best: dict = {}
    for r in rows:
        if r.get("dim", "2D") != "2D":
            continue
        key = (round(float(r["period_um"]), 4), round(float(r["fill_factor"]), 4))
        e = float(r["efficiency_at_target"])
        if key not in best or e > best[key]["efficiency_at_target"]:
            best[key] = dict(period_um=key[0], fill_factor=key[1], efficiency_at_target=e,
                             peak_efficiency=float(r.get("peak_efficiency", e)))
    return list(best.values())


def draw_uniform(path: Path, rows: list[dict], final: Optional[dict], wl: float) -> Optional[Path]:
    """Figure 5g: the uniform grating's landscape over fill factor and period."""
    pts = uniform_points(rows)
    if len(pts) < 3:
        return None
    note = ("each point: the best fibre position simulated at that period and fill factor;\n"
            "contours interpolate between the dots (2D FDTD)")
    chosen = (final["fill_factor"], final["period_um"]) if final else None
    lab = (f"final design (FF {final['fill_factor']:g}, P {final['period_um']:.4f} µm)"
           if final else "")
    panels = [dict(points=[dict(x=p["fill_factor"], y=p["period_um"], z=p["efficiency_at_target"])
                           for p in pts],
                   title=f"Efficiency at {wl} µm", xlabel="Fill factor (tooth / period)",
                   ylabel="Period (µm)", chosen=chosen, chosen_label=lab, note=note),
              dict(points=[dict(x=p["fill_factor"], y=p["period_um"], z=p["peak_efficiency"])
                           for p in pts],
                   title="Peak efficiency (at any wavelength in the band)",
                   xlabel="Fill factor (tooth / period)", ylabel="Period (µm)", chosen=chosen,
                   chosen_label=lab)]
    return draw(path, panels, "Uniform grating coupler: efficiency landscape (period × fill factor)")


def draw_apodised(path: Path, rows: list[dict], chosen: Optional[tuple], wl: float) -> Optional[Path]:
    """Figure 9c: the apodised grating's landscape over fibre position and period scale."""
    if len(rows) < 3:
        return None
    note = ("period scale: every tooth and gap stretched by one factor;\n"
            "contours interpolate between the dots (2D FDTD)")
    mk = lambda key: [dict(x=float(r["source_x_um"]), y=float(r["scale"]), z=float(r[key]))  # noqa: E731
                      for r in rows]
    panels = [dict(points=mk("efficiency_at_target"), title=f"Efficiency at {wl} µm",
                   xlabel="Fibre position on the grating (µm)", ylabel="Period scale",
                   chosen=chosen, chosen_label="chosen", note=note),
              dict(points=mk("peak_efficiency"), title="Peak efficiency (any wavelength)",
                   xlabel="Fibre position on the grating (µm)", ylabel="Period scale",
                   chosen=chosen, chosen_label="chosen")]
    return draw(path, panels, "Apodised grating coupler: efficiency landscape")
