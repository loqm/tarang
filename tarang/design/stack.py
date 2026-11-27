"""The layer stack as the solver sees it: a drawing and a text summary, made from the very
cross-section simulation the mode solver gets (materials, core, slab, BOX recess and
pedestal, grid lines), so what is shown is what is solved.

Shown at the start of every design (the cross-section stage: ``figures/2_stack.png`` and
a printed summary) and drawn again for the chosen geometry with its mode.

Exposes: simulation(), summary(), draw(), preview().
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from tarang.design.config import profile_of

C_UM = 299792458.0 * 1e6   # um/s, for the frequency of a wavelength in um


def simulation(ctx, H: float, W: float, steps: Optional[int] = None):
    """The cross-section simulation of this design at H x W, exactly as the mode solves
    build it (materials, profile, snapping)."""
    from tarang.waveguide.dispersion_mapping import core as dm
    cfg = ctx.cfg
    return dm.build_sim(H, W, wl0=cfg.run.wavelength_um,
                        steps=int(steps or cfg.cross_section.decision_steps),
                        box=cfg.stack.box_height_um, core=cfg.materials.core,
                        substrate=cfg.materials.box, cladding=cfg.materials.cladding,
                        cladding_height=cfg.stack.cladding_height_um, band=ctx.band,
                        snap=bool(cfg.cross_section.snap_grid), profile=profile_of(cfg))


def summary(cfg, H: Optional[float], W: Optional[float]) -> list[str]:
    """The stack from top to bottom, one line per layer."""
    m, st = cfg.materials, cfg.stack
    p = profile_of(cfg)
    clad_air = str(m.cladding).lower() in {"air", "none", ""}
    size = (f"{H * 1e3:.0f} nm thick × {W * 1e3:.0f} nm wide" if H and W else
            (f"{H * 1e3:.0f} nm film" if H else "film thickness from the map"))
    lines = [f"top      {'air' if clad_air else f'{m.cladding}, {st.cladding_height_um:g} µm'}",
             f"core     {m.core}, {size}"]
    if p.slab_um:
        lines.append(f"slab     {m.core}, {p.slab_um * 1e3:g} nm"
                     + (" across the chip (rib)" if p.slab_width_um is None
                        else f", {p.slab_width_um:g} µm wide (pedestal)"))
    if p.recess_um:
        fill = "air" if clad_air else m.cladding
        ped = p.pedestal or m.box
        lines.append(f"recess   BOX etched {p.recess_um * 1e3:g} nm beside the core "
                     f"(filled with {fill}); undercut {p.undercut_um * 1e3:g} nm per edge "
                     f"-> {ped} pedestal"
                     + (f" {(W - 2 * p.undercut_um) * 1e3:.0f} nm wide" if W else ""))
    lines.append(f"BOX      {m.box}, {st.box_height_um:g} µm")
    lines.append(f"handle   {m.handle}" if str(m.handle).lower() != "none" else "handle   none")
    return lines


def draw(ax, sim, H: float, W: float, wl: float, *, title: str = "", grid: bool = True,
         labels: Optional[dict] = None) -> None:
    """Refractive index of ``sim`` in its cross-section plane, cell by cell as the local
    solver samples it, with the grid lines (``grid``) and the core's dimensions."""
    import matplotlib.pyplot as plt
    import tidy3d as td
    plane = td.Box(center=(0, 0, sim.center[2]), size=(0, sim.size[1], sim.size[2]))
    eps = sim.epsilon(box=plane, coord_key="centers", freq=C_UM / wl)
    n = np.sqrt(np.real(np.squeeze(eps.values)))
    ys = np.asarray(sim.grid.boundaries.y)
    zs = np.asarray(sim.grid.boundaries.z)
    yb = ys[(ys >= sim.bounds[0][1] - 1e-9) & (ys <= sim.bounds[1][1] + 1e-9)]
    zb = zs[(zs >= sim.bounds[0][2] - 1e-9) & (zs <= sim.bounds[1][2] + 1e-9)]
    if n.shape != (len(yb) - 1, len(zb) - 1):        # fall back to the cell centres
        yc = np.asarray(eps.coords["y"].values)
        zc = np.asarray(eps.coords["z"].values)
        im = ax.pcolormesh(yc, zc, n.T, shading="nearest", cmap="Greys", vmin=1.0)
    else:
        im = ax.pcolormesh(yb, zb, n.T, shading="flat", cmap="Greys", vmin=1.0,
                           edgecolors=(0.55, 0.55, 0.55, 0.25) if grid else "none",
                           linewidth=0.1 if grid else 0)
    plt.colorbar(im, ax=ax, shrink=0.8, label=f"n at {wl:g} µm")
    ax.annotate("", xy=(-W / 2, H + 0.12), xytext=(W / 2, H + 0.12),
                arrowprops=dict(arrowstyle="<->", color="C3"))
    ax.text(0, H + 0.17, f"W = {W * 1e3:.0f} nm", ha="center", va="bottom", color="C3")
    ax.annotate("", xy=(W / 2 + 0.12, 0), xytext=(W / 2 + 0.12, H),
                arrowprops=dict(arrowstyle="<->", color="C3"))
    ax.text(W / 2 + 0.17, H / 2, f"H = {H * 1e3:.0f} nm", va="center", color="C3")
    for (txt, y, z) in (labels or {}).values():
        ax.text(y, z, txt, fontsize=7.5, color="C0", ha="left", va="center",
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))
    half = max(1.5 * W, 1.2)
    ax.set_xlim(-half, half)
    ax.set_ylim(-min(1.0, sim.size[2]), H + 0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("y (µm)")
    ax.set_ylabel("z (µm)")
    if title:
        ax.set_title(title, fontsize=10)


def _labels(cfg, H: float, W: float) -> dict:
    """Material names placed in their regions of the drawing."""
    m = cfg.materials
    p = profile_of(cfg)
    clad_air = str(m.cladding).lower() in {"air", "none", ""}
    half = max(1.5 * W, 1.2)
    out = {"core": (m.core, -W / 2 + 0.02, H * 0.5),
           "box": (f"BOX: {m.box}", -half + 0.05, -0.75),
           "top": ("air" if clad_air else str(m.cladding), -half + 0.05, H + 0.45)}
    if p.slab_um:
        out["slab"] = (f"slab: {m.core}", -half + 0.05, p.slab_um / 2)
    if p.recess_um:
        out["recess"] = ("recess", -half + 0.05, -p.recess_um / 2)
    return out


def preview(ctx, H: float, W: float, path: Path, note: str = "") -> Path:
    """Print the stack and save its drawing (made from the simulation itself)."""
    from tarang.design import visuals
    cfg = ctx.cfg
    ctx.ui.say("Stack as the mode solver sees it" + (f" ({note})" if note else "") + ":")
    for line in summary(cfg, H, W):
        ctx.ui.say("  " + line)
    plt = visuals._plt()
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    sim = simulation(ctx, H, W)
    p = profile_of(cfg)
    draw(ax, sim, H, W, cfg.run.wavelength_um, labels=_labels(cfg, H, W),
         title=(f"Stack at {H * 1e3:.0f} × {W * 1e3:.0f} nm" + (f": {p.label()}" if p.label()
                                                               else "") +
                ("\n" + note if note else "")))
    out = visuals._save(fig, path)
    ctx.ui.say(f"  figure: {out}")
    return out
