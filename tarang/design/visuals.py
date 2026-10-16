"""Figures of the design tool: what is about to be simulated, and what came out.

Every function writes one PNG into ``<run>/figures`` and returns its path. They never
stop a run: a figure that cannot be drawn is reported and skipped (``safe``).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Optional

import numpy as np

C_UM = 2.99792458e14


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def safe(ctx, label: str, fn: Callable, *args, **kwargs) -> Optional[Path]:
    """Draw a figure; on any error say so and carry on."""
    try:
        path = fn(*args, **kwargs)
    except Exception as err:                                    # noqa: BLE001
        ctx.ui.say(f"  (figure '{label}' not drawn: {type(err).__name__}: {err})")
        return None
    if path:
        ctx.ui.figure(path)
    return path


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    _plt().close(fig)
    return path


# =============================================================== cross-section
def cross_section(ctx, H: float, W: float, path: Path, mode: bool = True) -> Path:
    """Stack drawing with every dimension, and the TE00 field of the chosen core."""
    plt = _plt()
    cfg = ctx.cfg
    clad_air = str(cfg.materials.cladding).lower() == "air"
    box = cfg.stack.box_height_um
    ncols = 2 if mode else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6.2 * ncols, 4.6))
    axes = np.atleast_1d(axes)
    ax = axes[0]
    span = max(3 * W, 4.0)
    show_box = min(box, 2.5)
    ax.add_patch(plt.Rectangle((-span / 2, -show_box), span, show_box, color="#9ecae1",
                               label=f"BOX: {cfg.materials.box}, {box:g} µm"))
    if not clad_air:
        ax.add_patch(plt.Rectangle((-span / 2, 0), span, cfg.stack.cladding_height_um,
                                   color="#fdd0a2", alpha=0.6,
                                   label=f"cladding: {cfg.materials.cladding}"))
    ax.add_patch(plt.Rectangle((-W / 2, 0), W, H, color="#e6550d",
                               label=f"core: {cfg.materials.core}"))
    ax.annotate("", xy=(-W / 2, H + 0.15), xytext=(W / 2, H + 0.15),
                arrowprops=dict(arrowstyle="<->"))
    ax.text(0, H + 0.22, f"W = {W * 1e3:.0f} nm", ha="center", va="bottom")
    ax.annotate("", xy=(W / 2 + 0.15, 0), xytext=(W / 2 + 0.15, H),
                arrowprops=dict(arrowstyle="<->"))
    ax.text(W / 2 + 0.22, H / 2, f"H = {H * 1e3:.0f} nm", va="center")
    if show_box < box:
        ax.text(-span / 2 + 0.1, -show_box + 0.1, f"(BOX continues to {box:g} µm; Si handle "
                "below)", ha="left", fontsize=8)
    ax.set_xlim(-span / 2, span / 2)
    ax.set_ylim(-show_box, H + 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("y (µm)")
    ax.set_ylabel("z (µm)")
    ax.set_title(f"Cross-section ({'air' if clad_air else cfg.materials.cladding} top)")
    ax.legend(loc="upper right", fontsize=8)
    if mode:
        _mode_panel(ctx, axes[1], H, W)
    return _save(fig, path)


def _mode_panel(ctx, ax, H, W) -> None:
    """|E|^2 of the fundamental TE mode from the local mode solver (free, seconds)."""
    import tidy3d as td
    from tidy3d.plugins.mode import ModeSolver
    from tarang.waveguide.dispersion_mapping import core as dm
    cfg = ctx.cfg
    wl = cfg.run.wavelength_um
    sim = dm.build_sim(H, W, wl0=wl, steps=cfg.cross_section.decision_steps,
                       box=cfg.stack.box_height_um, core=cfg.materials.core,
                       substrate=cfg.materials.box, cladding=cfg.materials.cladding,
                       cladding_height=cfg.stack.cladding_height_um, band=ctx.band)
    plane = td.Box(center=(0, 0, sim.center[2]), size=(0, sim.size[1], sim.size[2]))
    n_core = ctx.n(cfg.materials.core)
    data = ModeSolver(simulation=sim, plane=plane, freqs=[C_UM / wl],
                      mode_spec=td.ModeSpec(num_modes=2, target_neff=n_core)).solve()
    m = max(range(2), key=lambda i: dm._te_frac(data, i, 0))
    e2 = sum(np.abs(np.squeeze(getattr(data, c).isel(mode_index=m, f=0).values)) ** 2
             for c in ("Ex", "Ey", "Ez"))
    y = data.Ey.coords["y"].values
    z = data.Ey.coords["z"].values
    im = ax.pcolormesh(y, z, np.asarray(e2).T, shading="auto", cmap="magma")
    ax.add_patch(_plt().Rectangle((-W / 2, 0), W, H, fill=False, ec="w", lw=1))
    ax.axhline(0, color="w", lw=0.6, ls=":")
    ax.set_xlim(-1.5 * W, 1.5 * W)
    ax.set_ylim(-1.2, H + 0.8)
    ax.set_aspect("equal")
    neff = float(np.real(data.n_eff.isel(mode_index=m, f=0)))
    ax.set_title(f"TE00 |E|² at {wl} µm (n_eff = {neff:.4f})")
    ax.set_xlabel("y (µm)")
    _plt().colorbar(im, ax=ax, shrink=0.8)


# =============================================================== simulation views
def sim_views(sim, path: Path, views: list[dict], title: str,
              notes: Optional[str] = None) -> Path:
    """Tidy3D's own drawing of ``sim`` in the given planes, e.g. [dict(z=0.35), dict(y=0)].
    A view may carry 'xlim' / 'ylim' / 'title'."""
    plt = _plt()
    fig, axes = plt.subplots(len(views), 1, figsize=(10, 3.8 * len(views)))
    axes = np.atleast_1d(axes)
    for ax, view in zip(axes, views):
        view = dict(view)
        xlim, ylim, sub = view.pop("xlim", None), view.pop("ylim", None), view.pop("title", None)
        sim.plot(ax=ax, **view)
        if xlim:
            ax.set_xlim(*xlim)
        if ylim:
            ax.set_ylim(*ylim)
        if sub:
            ax.set_title(sub)
    fig.suptitle(title)
    if notes:
        fig.text(0.01, 0.005, notes, fontsize=8, va="bottom")
    return _save(fig, path)


def grating_geometry(device, path: Path, title: str) -> Path:
    """Side view of a grating simulation with the fibre beam drawn as an arrow from the
    source plane to where its centre meets the grating (``source_x``)."""
    plt = _plt()
    sim, cfg = device.sim, device.config
    src = sim.sources[0]
    d = np.array(src._dir_vector)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    sim.plot(y=0, ax=ax)
    x0, z0 = src.center[0], src.center[2]
    t = (z0 - cfg.core_height) / max(-d[2], 1e-9)
    ax.annotate("", xy=(x0 + t * d[0], cfg.core_height), xytext=(x0, z0),
                arrowprops=dict(arrowstyle="-|>", color="red", lw=2))
    ax.plot([cfg.source_x], [cfg.core_height], "o", color="red")
    ax.annotate(f"fibre beam: {abs(cfg.theta_deg):g}° from vertical, travelling "
                f"{cfg.fiber_tilt} the waveguide\nbeam centre on the grating at x = "
                f"{cfg.source_x:.2f} µm", xy=(x0, z0), xytext=(x0, z0 + 0.4), color="red",
                fontsize=9, ha="center")
    ax.text(-cfg.buffer, cfg.core_height + 0.25, "← waveguide", fontsize=9)
    ax.set_title(title + f"\nperiod {cfg.period:.4f} µm, fill factor {cfg.ff:.2f}, etch "
                 f"{cfg.etch_depth:.3f} µm, {cfg.n_teeth} teeth")
    return _save(fig, path)


# =============================================================== gap
def gap_coupling(gaps, cross_raw, cross, fit, targets: dict, path: Path,
                 chosen: dict) -> Path:
    """Coupled power per pass against gap (log scale): as measured, after the bus-only
    normalisation, the exponential fit, and the critical / over-coupled targets."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    g = np.asarray(gaps, float)
    if cross_raw is not None:
        ax.semilogy(g, np.clip(cross_raw, 1e-8, None), "s", mfc="none", color="0.5",
                    label="1 − T relative to the source (includes bus losses)")
    ax.semilogy(g, np.clip(cross, 1e-8, None), "o", color="C0",
                label="power coupled into the ring")
    if fit:
        picks = [x for x in chosen.values() if x]
        gg = np.linspace(min([g.min(), *picks]), max([g.max(), *picks]), 200)
        theta = fit["theta_0"] * np.exp(-fit["gamma"] * gg)
        ax.semilogy(gg, np.sin(np.clip(theta, 0, np.pi / 2)) ** 2, "-", color="C0",
                    label=f"fit: θ = {fit['theta_0']:.3f}·exp(−{fit['gamma']:.2f} g)")
    for label, level in targets.items():
        ax.axhline(level, ls="--", lw=1, color="C3" if "critical" in label else "C2")
        ax.text(g.max(), level, f" {label}", va="bottom", ha="right", fontsize=8)
    for label, gap in chosen.items():
        if gap:
            ax.axvline(gap, ls=":", color="C3" if "critical" in label else "C2")
            ax.text(gap, ax.get_ylim()[0] * 3, f" {label}\n {gap * 1e3:.0f} nm", fontsize=8)
    ax.set_xlabel("Ring–bus gap (µm)")
    ax.set_ylabel("Power coupled into the ring per pass")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Ring–bus coupling against gap")
    return _save(fig, path)


# =============================================================== grating
def grating_landscape(rows: list[dict], best: dict, path: Path, target_um: float) -> Path:
    """Peak efficiency and efficiency at the design wavelength over (fill factor,
    period), with the peak wavelength written at every point."""
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ff = np.array([r["fill_factor"] for r in rows])
    P = np.array([r["period_um"] for r in rows])
    for ax, key, title in ((axes[0], "peak_efficiency", "Peak efficiency (any wavelength)"),
                           (axes[1], "efficiency_at_target",
                            f"Efficiency at {target_um} µm")):
        v = np.array([r[key] for r in rows])
        sc = ax.scatter(ff, P, c=v, s=520, marker="s", cmap="viridis", vmin=0,
                        vmax=max(v.max(), 1e-3))
        for r in rows:
            ax.annotate(f"{r[key] * 100:.0f}%\n{r['peak_wavelength_um'] * 1e3:.0f}",
                        (r["fill_factor"], r["period_um"]), ha="center", va="center",
                        fontsize=6.5, color="w")
        ax.plot(best["fill_factor"], best["period_um"], "x", color="red", ms=14, mew=3)
        ax.set_xlabel("Fill factor (tooth / period)")
        ax.set_ylabel("Period (µm)")
        ax.set_title(title + "\n(label: efficiency, peak wavelength in nm)", fontsize=10)
        plt.colorbar(sc, ax=ax, label="efficiency")
    return _save(fig, path)


def grating_sensitivity(ff_rows, period_rows, align_rows, best, path: Path,
                        target_um: float) -> Path:
    """Efficiency at the design wavelength against fill factor, period and fibre
    position around the final design (one parameter changed at a time)."""
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    panels = ((axes[0], ff_rows, "fill_factor", "Fill factor", f"P = {best['period_um']:.4f} µm"),
              (axes[1], period_rows, "period_um", "Period (µm)", f"FF = {best['fill_factor']:.2f}"),
              (axes[2], align_rows, "source_x_um", "Fibre position on the grating (µm)",
               "alignment"))
    for ax, rows, key, label, sub in panels:
        rows = sorted(rows, key=lambda r: r[key])
        if rows:
            ax.plot([r[key] for r in rows], [r["efficiency_at_target"] for r in rows], "o-")
            ax.axvline(best[key], color="red", ls="--", lw=1, label="chosen")
        ax.set_xlabel(label)
        ax.set_ylabel(f"Efficiency at {target_um} µm")
        ax.set_title(f"{label.split(' (')[0]} sensitivity ({sub})", fontsize=10)
        ax.grid(alpha=0.3)
    return _save(fig, path)


def grating_spectrum(wl, eta, summary: dict, path: Path, title: str) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    db = 10 * np.log10(np.clip(eta, 1e-6, None))
    ax.plot(np.asarray(wl) * 1e3, db, lw=2)
    peak_db = 10 * np.log10(max(summary["peak_efficiency"], 1e-6))
    ax.axhline(peak_db - 1, ls=":", color="0.5")
    ax.axhline(peak_db - 3, ls=":", color="0.7")
    ax.axvline(summary.get("target_um", 1.55) * 1e3, ls="--", color="red", lw=1)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Coupling efficiency (dB)")
    ax.set_ylim(peak_db - 12, peak_db + 1)
    ax.grid(alpha=0.3)
    bw1 = summary["bandwidth_1db_nm"]
    bw3 = summary["bandwidth_3db_nm"]
    ax.set_title(f"{title}\npeak {summary['peak_efficiency'] * 100:.1f}% "
                 f"({-summary['peak_loss_db']:.2f} dB) at {summary['peak_wavelength_um'] * 1e3:.1f}"
                 f" nm; 1 dB band {bw1:.0f} nm{'' if summary['bandwidth_1db_closed'] else '+'}, "
                 f"3 dB band {bw3:.0f} nm{'' if summary['bandwidth_3db_closed'] else '+'}",
                 fontsize=10)
    return _save(fig, path)


def grating_field(sim_data, path: Path, wavelength_um: float, title: str) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(11, 3.8))
    sim_data.plot_field("field_y_plane", "E", "abs", f=C_UM / wavelength_um, ax=ax)
    ax.set_title(title)
    return _save(fig, path)


# =============================================================== layout
def layout_views(gds_path: Path, p: dict, path: Path) -> Path:
    """The ring + bus + couplers cell from the GDS: whole device, coupling region and
    one grating."""
    import gdstk
    plt = _plt()
    lib = gdstk.read_gds(str(gds_path))
    cell = next((c for c in lib.cells if "_gc" in c.name), lib.top_level()[0])
    polys = cell.get_polygons()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8),
                             gridspec_kw=dict(width_ratios=[1.6, 1, 1]))
    for ax in axes:
        for poly in polys:
            pts = np.asarray(poly.points)
            ax.fill(pts[:, 0], pts[:, 1], color="#3182bd", lw=0)
        ax.set_aspect("equal")
        ax.set_xlabel("x (µm)")
    L, W = p["bus_length_um"], p["waveguide_width_um"]
    axes[0].set_title(cell.name)
    axes[0].autoscale()
    gap = p.get("gap_critical_um") or 0.5
    axes[1].set_xlim(-6 * W - 3, 6 * W + 3)
    axes[1].set_ylim(-3 * W, 2 * W + gap + 4)
    axes[1].set_title(f"coupling region (gap {gap * 1e3:.0f} nm)")
    x0 = L / 2 + p["taper_length_um"]
    if p.get("grating_period_um"):
        n = int(p["grating_teeth"]) * p["grating_period_um"]
        axes[2].set_xlim(x0 - 15, x0 + n + 5)
        axes[2].set_ylim(-p["grating_width_um"], p["grating_width_um"])
        axes[2].set_title(f"grating: {p['grating_teeth']} × {p['grating_period_um']:.4f} µm, "
                          f"FF {p['grating_fill_factor']:.2f}")
    return _save(fig, path)


def stage_figure_path(ctx, name: str) -> Path:
    return ctx.path("figures", name)


def fmt_minutes(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m} min {s:02d} s" if m else f"{s} s"


def deg(x: float) -> float:
    return math.degrees(x)
