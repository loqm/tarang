"""Figures of the design tool: what is about to be simulated, and what came out.

Every function writes one figure into ``<run>/figures`` (PNG at ``run.figure_dpi``, plus
PDF and/or SVG per ``run.figure_formats``: vector files for papers and slides, with text
kept as text) and returns the PNG path. They never stop a run: a figure that cannot be
drawn is reported and skipped (``safe``); a file that cannot be replaced because another
program has it open is written next to it as ``<name>_new.<ext>``.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np

C_UM = 2.99792458e14

# Okabe-Ito colours: distinguishable with every common colour-vision deficiency
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442",
           "#000000"]
SETTINGS = {"formats": ["png", "pdf"], "dpi": 300}
_LAST_NOTES: list[str] = []


def configure(formats=None, dpi=None) -> None:
    """Figure files to write (png, pdf, svg) and the PNG resolution."""
    if formats:
        fm = [str(f).lower().lstrip(".") for f in formats]
        bad = set(fm) - {"png", "pdf", "svg"}
        if bad:
            raise ValueError(f"unknown figure format(s) {sorted(bad)}; use png, pdf, svg")
        SETTINGS["formats"] = fm
    if dpi:
        SETTINGS["dpi"] = int(dpi)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cycler = matplotlib.cycler
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8.5,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "lines.linewidth": 1.8,
        "lines.markersize": 6, "axes.grid": False, "grid.alpha": 0.3,
        "axes.prop_cycle": cycler(color=PALETTE), "figure.dpi": 110,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",   # editable text
        "image.cmap": "magma"})
    return plt


def safe(ctx, label: str, fn: Callable, *args, **kwargs) -> Optional[Path]:
    """Draw a figure; on any error say so and carry on."""
    try:
        _LAST_NOTES.clear()
        path = fn(*args, **kwargs)
    except Exception as err:                                    # noqa: BLE001
        ctx.ui.say(f"  (figure '{label}' not drawn: {type(err).__name__}: {err})")
        return None
    for note in _LAST_NOTES:
        ctx.ui.say("  " + note)
    if path:
        ctx.ui.figure(path)
    return path


def _write(fig, target: Path, **kw) -> Path:
    """Save via a temporary file and swap it in; if the old file is held open by another
    program (e.g. a Windows image viewer), keep it and write <name>_new.<ext>."""
    tmp = target.with_name(target.stem + ".tmp" + target.suffix)
    fig.savefig(tmp, **kw)
    try:
        os.replace(tmp, target)
        return target
    except OSError:
        alt = target.with_name(target.stem + "_new" + target.suffix)
        try:
            os.replace(tmp, alt)
        except OSError:
            tmp.unlink(missing_ok=True)
            raise
        _LAST_NOTES.append(f"{target.name} is open in another program; saved as {alt.name}")
        return alt


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.tight_layout()
    except Exception:
        pass
    out = None
    for fmt in SETTINGS["formats"]:
        target = path.with_suffix("." + fmt)
        kw = {"dpi": SETTINGS["dpi"]} if fmt == "png" else {}
        written = _write(fig, target, **kw)
        if fmt == "png" or out is None:
            out = written
    _plt().close(fig)
    return out


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


def mode_profiles(rows: list, H: float, W: float, path: Path, title: str,
                  cores: Optional[list] = None, components: bool = True) -> Path:
    """Mode profiles in the cross-section plane (y across the width, z up): one row per
    mode, with |E|^2 and (``components``) |Ex|, |Ey|, |Ez|, each normalised to the mode's
    largest field component, so the weak components show how polarised the mode is.

    ``rows``: (label, mode dict from localmodes). ``cores``: core rectangles (y0, z0, w, h)
    to outline (default: one core at the origin)."""
    plt = _plt()
    cores = cores or [(-W / 2, 0.0, W, H)]
    ncol = 4 if components else 1
    fig, axes = plt.subplots(len(rows), ncol, figsize=(4.1 * ncol, 3.2 * len(rows) + 0.4),
                             squeeze=False)
    ymin = min(c[0] for c in cores) - 1.2
    ymax = max(c[0] + c[2] for c in cores) + 1.2
    for r, (label, m) in enumerate(rows):
        f = m["fields"]
        amp = {k: np.abs(v) for k, v in f.items()}
        top = max(a.max() for a in amp.values())
        panels = [("|E|²", sum(a ** 2 for a in amp.values()) / top ** 2, "magma")]
        if components:
            panels += [(f"|{k}|", amp[k] / top, "viridis") for k in ("Ex", "Ey", "Ez")]
        for c, (name, val, cmap) in enumerate(panels):
            ax = axes[r, c]
            im = ax.pcolormesh(m["y"], m["z"], np.asarray(val).T, shading="auto", cmap=cmap,
                               vmin=0, vmax=max(float(np.max(val)), 1e-12) if c == 0 else 1)
            for (y0, z0, w, h) in cores:
                ax.add_patch(plt.Rectangle((y0, z0), w, h, fill=False, ec="w", lw=1))
            ax.axhline(0, color="w", lw=0.5, ls=":")
            ax.set_xlim(ymin, ymax)
            ax.set_ylim(-1.2, H + 1.0)
            ax.set_aspect("equal")
            ax.set_xlabel("y (µm)")
            if c == 0:
                ax.set_ylabel(f"{label}\nz (µm)")
            ax.set_title(name if r else name, fontsize=10)
            fig.colorbar(im, ax=ax, shrink=0.75)
        extra = f", n_g {m['ng']:.4f}" if m.get("ng") else ""
        axes[r, 0].text(0.02, 0.97, f"n_eff {m['neff']:.5f}{extra}\nA_eff {m['aeff']:.3f} µm²\n"
                        f"TE fraction {m['te']:.3f}", transform=axes[r, 0].transAxes,
                        va="top", fontsize=7.5, color="w")
    fig.suptitle(title)
    return _save(fig, path)


def bent_mode(straight: dict, bent: dict, H: float, W: float, R: float, path: Path) -> Path:
    """Straight and bent TE00 (|E|^2 maps) and a cut through the middle of the core: the
    bend pushes the mode towards the outside of the ring by a few nm, too little to see in
    the maps but clear in the cut and its centroid."""
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    shift = 1e3 * (bent["yc"] - straight["yc"])
    for ax, (lab, m) in zip(axes[:2], (("straight", straight), (f"bent, R = {R:g} µm", bent))):
        e2 = sum(np.abs(v) ** 2 for v in m["fields"].values())
        im = ax.pcolormesh(m["y"], m["z"], (e2 / e2.max()).T, shading="auto", cmap="magma",
                           vmin=0, vmax=1)
        ax.add_patch(plt.Rectangle((-W / 2, 0), W, H, fill=False, ec="w", lw=1))
        ax.axvline(m["yc"], color="#56B4E9", lw=1, ls="--")
        ax.set(xlim=(-W / 2 - 1.2, W / 2 + 1.2), ylim=(-1.0, H + 0.9), aspect="equal",
               xlabel="y (µm), outside of the ring → +y", ylabel="z (µm)",
               title=f"TE00 {lab}: n_eff {m['neff']:.5f}")
        fig.colorbar(im, ax=ax, shrink=0.75, label="|E|² (normalised)")
    ax = axes[2]
    cuts = {}
    for lab, m, col, ls in (("straight", straight, "k", "-"), ("bent", bent, "#D55E00", "--")):
        e2 = sum(np.abs(v) ** 2 for v in m["fields"].values())
        iz = int(np.argmin(np.abs(np.asarray(m["z"]) - H / 2)))
        cut = e2[:, iz] / e2[:, iz].max()
        cuts[lab] = (np.asarray(m["y"]), cut)
        ax.plot(m["y"], cut, ls, color=col, lw=1.8, label=f"{lab}: centroid {m['yc'] * 1e3:+.1f} nm")
        ax.axvline(m["yc"], color=col, ls=ls, lw=0.8)
    ys, cs = cuts["straight"]
    diff = np.interp(ys, *cuts["bent"]) - cs           # what the bend moves: out of -y, into +y
    tw = ax.twinx()
    tw.fill_between(ys, diff, color="#56B4E9", alpha=0.45, lw=0, label="bent − straight")
    tw.axhline(0, color="#56B4E9", lw=0.6)
    lim = max(float(np.abs(diff).max()), 1e-9) * 1.15
    tw.set(ylim=(-lim, lim), ylabel="bent − straight (right axis)")
    tw.legend(fontsize=8, loc="lower right")
    ax.axvspan(-W / 2, W / 2, color="0.9", zorder=0)
    ax.set(xlim=(-W / 2 - 1.2, W / 2 + 1.2), ylim=(0, 1.05), xlabel="y (µm)",
           ylabel="|E|² through the middle of the core", title=f"outward shift {shift:+.1f} nm")
    ax.legend(fontsize=8)
    fig.suptitle(f"Bent mode: the bend (R = {R:g} µm) shifts TE00 by {shift:+.1f} nm towards the "
                 "outside of the ring")
    return _save(fig, path)


def supermodes(sm: dict, H: float, W: float, path: Path, theta_est: Optional[float],
               theta_fdtd: Optional[float]) -> Path:
    """Even and odd TE supermodes of the bus and the ring at one gap (cross-section at the
    closest point), and what their index split says about the coupling."""
    plt = _plt()
    cores = sm["cores"]
    half = sm["shift_um"] / 2 + W + 1
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9))
    for ax, key in zip(axes[:2], ("even", "odd")):
        m = sm[key]
        ey = np.real(m["fields"]["Ey"])
        ey = ey / np.max(np.abs(ey))
        im = ax.pcolormesh(m["y"], m["z"], ey.T, shading="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        for (y0, z0, w, h) in cores:
            ax.add_patch(plt.Rectangle((y0, z0), w, h, fill=False, ec="k", lw=1))
        ax.set_xlim(-half, half)
        ax.set_ylim(-1, H + 0.9)
        ax.set_aspect("equal")
        ax.set_xlabel("y (µm)")
        ax.set_ylabel("z (µm)")
        ax.set_title(f"{key} supermode, Re(Ey): n_eff {m['neff']:.5f}")
        fig.colorbar(im, ax=ax, shrink=0.8)
    ax = axes[2]
    m = sm["even"]
    zi = int(np.argmin(np.abs(m["z"] - H / 2)))
    for key, col in (("even", PALETTE[0]), ("odd", PALETTE[1])):
        e2 = np.sum([np.abs(sm[key]["fields"][k][:, zi]) ** 2 for k in ("Ex", "Ey", "Ez")], axis=0)
        ax.semilogy(sm[key]["y"], e2 / e2.max(), color=col, label=f"{key} |E|²")
    for (y0, _, w, _) in cores:
        ax.axvspan(y0, y0 + w, color="0.85", lw=0)
    ax.set_ylim(1e-5, 2)
    ax.set_xlim(-half, half)
    ax.set_xlabel("y (µm), through the middle of the cores")
    ax.set_title("evanescent overlap across the gap (log)")
    ax.legend(fontsize=8)
    txt = (f"gap {sm['gap_um'] * 1e3:.0f} nm: Δn = {sm['dn']:.2e}, κ = πΔn/λ = "
           f"{sm['kappa_per_um'] * 1e3:.3f} /mm, full transfer in L_c = λ/2Δn = "
           f"{sm['coupling_length_um']:.0f} µm")
    if theta_est is not None:
        txt += (f"\nper pass along the ring (coupled-mode theory): θ = κ·√(2πR/γ) = "
                f"{theta_est:.4f}" + (f";  3D FDTD: {theta_fdtd:.4f}" if theta_fdtd else ""))
    fig.suptitle(txt, fontsize=10)
    return _save(fig, path)


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


def sim_picture(sim, path: Path, title: str, z: Optional[float] = None) -> Path:
    """Geometry of one simulation as Tidy3D draws it (structures, sources, monitors,
    boundaries): the simulation plane for a 2D run; a top view (at height ``z``) and the
    cross-section x = centre for a 3D run. PNG only, 150 dpi: one per cloud run."""
    plt = _plt()
    zero = [i for i, s in enumerate(sim.size) if s == 0]
    if zero:
        ax_name = "xyz"[zero[0]]
        views = [({ax_name: sim.center[zero[0]]}, f"{ax_name} = {sim.center[zero[0]]:g} µm plane")]
    else:
        zc = sim.center[2] if z is None else z
        views = [(dict(z=zc), f"top view at z = {zc:.2f} µm"),
                 (dict(x=sim.center[0]), f"cross-section at x = {sim.center[0]:g} µm")]
    fig, axes = plt.subplots(len(views), 1, figsize=(10, 3.6 * len(views)), squeeze=False)
    for ax, (view, sub) in zip(axes[:, 0], views):
        sim.plot(ax=ax, **view)
        ax.set_title(sub, fontsize=9)
    fig.suptitle(title, fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = _write(fig, path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


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
                 chosen: dict, fit_r2: Optional[float] = None) -> Path:
    """Coupled power per pass against gap on three panels:

    1. linear scale (%): where the useful coupling lies and where the critical and
       over-coupled targets cross the curve;
    2. log scale: the exponential evanescent decay is a straight line; 1 - T relative to the
       source levels off at the gap-independent bus loss;
    3. the fit space: theta = asin(sqrt(coupled)) against gap on a log axis (straight line
       theta0 exp(-gamma g)), with the through power T = 1 - coupled on a linear right axis.
    """
    plt = _plt()
    g = np.asarray(gaps, float)
    k2 = np.clip(np.asarray(cross, float), 1e-12, 1)
    picks = [x for x in chosen.values() if x]
    lo, hi = min([g.min(), *picks]) - 0.05, max([g.max(), *picks]) + 0.05
    gg = np.linspace(max(lo, 0.0), hi, 400)
    fitc = (np.sin(np.clip(fit["theta_0"] * np.exp(-fit["gamma"] * gg), 0, np.pi / 2)) ** 2
            if fit else None)
    cols = {"critical": PALETTE[1], "over": "0.45"}

    def colour(label):
        return cols["critical"] if "critical" in label else cols["over"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, log in ((axes[0], False), (axes[1], True)):
        sc = 1.0 if log else 100.0
        if fitc is not None:
            ax.plot(gg, fitc * sc, "-", color=PALETTE[2],
                    label=f"fit: θ = {fit['theta_0']:.3f}·exp(−{fit['gamma']:.2f} g)")
        ax.plot(g, k2 * sc, "o", color=PALETTE[2], label="power coupled into the ring")
        if cross_raw is not None:
            ax.plot(g, np.clip(cross_raw, 1e-12, None) * sc, "s", mfc="none", color="0.5",
                    label="1 − T vs source (includes bus loss)")
        for label, level in targets.items():
            ax.axhline(level * sc, ls="--", lw=1, color=colour(label))
        for label, gap in chosen.items():
            if gap:
                ax.axvline(gap, ls=":", lw=1.2, color=colour(label))
        ax.set_xlabel("Ring–bus gap (µm)")
        ax.grid(alpha=0.3, which="both")
        if log:
            ax.set_yscale("log")
            ax.set_ylabel("Coupled power per pass κ²")
            ax.set_title("log scale: straight line = exponential decay")
        else:
            top = max(targets.values()) if targets else k2.max()
            ax.set_ylim(0, 2.6 * top * 100)
            ax.set_ylabel("Coupled power per pass κ² (%)")
            ax.set_title("linear scale")
            y = 2.45 * top * 100
            for label, gap in chosen.items():
                if gap:
                    lvl = next((v for k, v in targets.items() if (("critical" in k) ==
                                                                  ("critical" in label))), None)
                    txt = f"{label}: {gap * 1e3:.0f} nm" + (f"\n(κ² = {lvl * 100:.3f} %)"
                                                            if lvl else "")
                    ax.annotate(txt, (gap, (lvl or 0) * 100), (g.min() + 0.55 * (g.max() - g.min()), y),
                                color=colour(label), fontsize=8.5, va="top",
                                arrowprops=dict(arrowstyle="->", color=colour(label), lw=1))
                    y -= 0.42 * top * 100
            ax.legend(loc="center right", fontsize=8)
    ax = axes[2]
    th = np.arcsin(np.sqrt(k2))
    if fit:
        ax.semilogy(gg, fit["theta_0"] * np.exp(-fit["gamma"] * gg), "-", color=PALETTE[3],
                    label="θ₀·exp(−γ g)")
    ax.semilogy(g, th, "s", color=PALETTE[0], label="θ = asin(√κ²)")
    for label, level in targets.items():
        if "critical" in label:
            ax.axhline(math.asin(math.sqrt(level)), ls="--", lw=1, color=colour(label),
                       label=f"target θ (critical) = {math.asin(math.sqrt(level)):.4f}")
    for label, gap in chosen.items():
        if gap and "critical" in label:
            ax.axvline(gap, ls=":", lw=1.2, color=colour(label),
                       label=f"critical gap = {gap * 1e3:.1f} nm")
    ax2 = ax.twinx()
    if fitc is not None:
        ax2.plot(gg, 1 - fitc, "--", color="0.6", lw=1)
    ax2.plot(g, 1 - k2, "o", color="0.4", ms=5)
    ax2.set_ylabel("Through power T = 1 − κ² (linear)", color="0.4")
    tmin = float(np.min(1 - k2))
    ax2.set_ylim(max(0.0, tmin - 0.25 * (1 - tmin)), 1 + 0.02 * (1 - tmin))
    ax.set_xlabel("Ring–bus gap (µm)")
    ax.set_ylabel("Coupling angle θ (rad)")
    ax.set_title("fit space" + (f" (R² of ln θ = {fit_r2:.5f})" if fit_r2 else ""))
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Ring–bus coupling against gap")
    return _save(fig, path)


def _field_e2(data, monitor: str, f0: float):
    """|E|^2 of a field monitor at the frequency nearest f0, and its two in-plane axes."""
    mon = data[monitor]
    comps = [getattr(mon, c, None) for c in ("Ex", "Ey", "Ez")]
    comps = [c for c in comps if c is not None]
    e2 = None
    for c in comps:
        a = np.abs(c.sel(f=f0, method="nearest").values) ** 2
        e2 = a if e2 is None else e2 + a
    coords = comps[0].coords
    axes = [ax for ax in ("x", "y", "z") if coords[ax].size > 1]
    return np.squeeze(e2), {a: coords[a].values for a in axes}, axes


def gap_field_map(data, path: Path, wavelength_um: float, gap_um: float, title: str) -> Path:
    """Top view (xy, middle of the core) of |E|^2 in a gap run, on a log scale so the
    evanescent field across the gap and the light in the ring are both visible."""
    plt = _plt()
    e2, c, axes_ = _field_e2(data, "coupling_field_map", C_UM / wavelength_um)
    x, y = c[axes_[0]], c[axes_[1]]
    v = np.log10(e2 / e2.max() + 1e-7)
    fig, axs = plt.subplots(2, 1, figsize=(12, 6.2), gridspec_kw=dict(height_ratios=[1, 1.2]))
    for ax, zoom in zip(axs, (False, True)):
        im = ax.pcolormesh(x, y, v.T, shading="auto", cmap="magma", vmin=-5, vmax=0)
        ax.set_aspect("equal")
        ax.set_xlabel(f"{axes_[0]} (µm)")
        ax.set_ylabel(f"{axes_[1]} (µm)")
        if zoom:
            yb = y[np.argmax(e2[np.argmin(np.abs(x - x.min() - 0.1 * (x.max() - x.min())))])]
            ax.set_xlim(max(-12, float(x.min())), min(12, float(x.max())))
            ax.set_ylim(max(yb - 2.5, float(y.min())), min(yb + 4.5, float(y.max())))
            ax.set_title("coupling region (stripes of period λ/2n_eff mean light travelling "
                         "both ways, i.e. a reflection)", fontsize=9.5)
        else:
            ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.85, label="log₁₀ |E|² (normalised)")
    return _save(fig, path)


# =============================================================== grating
def grating_landscape(rows: list[dict], best: dict, path: Path, target_um: float,
                      final: Optional[dict] = None) -> Path:
    """Peak efficiency and efficiency at the design wavelength over (fill factor,
    period), with the peak wavelength written at every point. The best point of the map is
    outlined; the final design (after the later steps), if given, is a star."""
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    ff = np.array([r["fill_factor"] for r in rows])
    P = np.array([r["period_um"] for r in rows])
    for ax, key, title in ((axes[0], "peak_efficiency", "Peak efficiency (any wavelength)"),
                           (axes[1], "efficiency_at_target",
                            f"Efficiency at {target_um} µm")):
        v = np.array([r[key] for r in rows])
        vmax = max(v.max(), 1e-3)
        sc = ax.scatter(ff, P, c=v, s=520, marker="s", cmap="viridis", vmin=0, vmax=vmax)
        for r in rows:                         # dark text on the bright end of the colour map
            ax.annotate(f"{r[key] * 100:.0f}%\n{r['peak_wavelength_um'] * 1e3:.0f}",
                        (r["fill_factor"], r["period_um"]), ha="center", va="center",
                        fontsize=6.5, color="k" if r[key] > 0.6 * vmax else "w")
        ax.scatter([best["fill_factor"]], [best["period_um"]], s=760, marker="s",
                   facecolors="none", edgecolors="#D55E00", linewidths=2.2,
                   label="best of this map")
        if final:
            ax.scatter([final["fill_factor"]], [final["period_um"]], s=260, marker="*",
                       color="#D55E00", edgecolors="k", linewidths=0.8, zorder=5,
                       label=f"final design (FF {final['fill_factor']:g}, "
                             f"P {final['period_um']:.4f} µm)")
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2,
                  markerscale=0.45, frameon=False)
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

    def held(rows, key, fmt):
        vals = sorted({r[key] for r in rows})
        return f"{vals[0]:{fmt}}" if len(vals) == 1 else "varied"

    # Each panel states what was held fixed in ITS runs, which need not be the final design.
    panels = ((axes[0], ff_rows, "fill_factor", "Fill factor",
               "period kept phase-matched for each fill factor"),
              (axes[1], period_rows, "period_um", "Period (µm)",
               f"FF = {held(period_rows, 'fill_factor', '.2f')}" if period_rows else ""),
              (axes[2], align_rows, "source_x_um", "Fibre position on the grating (µm)",
               f"P = {held(align_rows, 'period_um', '.4f')} µm, "
               f"FF = {held(align_rows, 'fill_factor', '.2f')}" if align_rows else ""))
    for ax, rows, key, label, sub in panels:
        rows = sorted(rows, key=lambda r: r[key])
        if rows:
            ax.plot([r[key] for r in rows], [r["efficiency_at_target"] for r in rows], "o-")
            # the final period and fill factor (at this panel's fibre position)
            mine = [r for r in rows if r["period_um"] == best["period_um"]
                    and r["fill_factor"] == best["fill_factor"]
                    and (key != "source_x_um" or r["source_x_um"] == best["source_x_um"])]
            if mine:
                ax.plot([best[key]], [mine[0]["efficiency_at_target"]], "o", ms=11,
                        mfc="none", mec="red", mew=2, label="final design")
                ax.legend(fontsize=8, loc="lower center")
            else:
                ax.text(0.5, 0.03, "final design not in this sweep", transform=ax.transAxes,
                        ha="center", fontsize=8, color="0.4")
        ax.set_xlabel(label)
        ax.set_ylabel(f"Efficiency at {target_um} µm")
        ax.set_title(f"{label.split(' (')[0]} sensitivity\n({sub})", fontsize=10)
        ax.grid(alpha=0.3)
    return _save(fig, path)


def grating_spectrum(wl, eta, summary: dict, path: Path, title: str) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    db = 10 * np.log10(np.clip(eta, 1e-6, None))
    ax.plot(np.asarray(wl) * 1e3, db, lw=2)
    peak_db = 10 * np.log10(max(summary["peak_efficiency"], 1e-6))
    ax.axhline(peak_db - 1, ls=":", color="0.45", label="1 dB below the peak")
    ax.axhline(peak_db - 3, ls="-.", color="0.6", lw=1, label="3 dB below the peak")
    ax.axvline(summary.get("target_um", 1.55) * 1e3, ls="--", color="red", lw=1,
               label="design wavelength")
    if "band_1db_lo_um" in summary:
        ax.axvspan(summary["band_1db_lo_um"] * 1e3, summary["band_1db_hi_um"] * 1e3,
                   color="C0", alpha=0.12, label="1 dB band")
    ax.axvline(summary["peak_wavelength_um"] * 1e3, ls=":", color="C0", lw=1.5, label="peak")
    ax.legend(fontsize=8, loc="lower center")
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
    """Side view (xz) of the field at the design wavelength: |E| (linear) and |E|^2 on a
    log scale, which shows the beam coming in, the light diffracted down into the BOX and
    the guided mode leaving towards the waveguide."""
    plt = _plt()
    e2, c, axes_ = _field_e2(sim_data, "field_y_plane", C_UM / wavelength_um)
    x, z = c[axes_[0]], c[axes_[1]]
    fig, axs = plt.subplots(2, 1, figsize=(12, 6.0))
    im0 = axs[0].pcolormesh(x, z, np.sqrt(e2).T, shading="auto", cmap="magma")
    im1 = axs[1].pcolormesh(x, z, np.log10(e2.T / e2.max() + 1e-6), shading="auto",
                            cmap="magma", vmin=-4, vmax=0)
    for ax, im, lab in ((axs[0], im0, "|E|"), (axs[1], im1, "log₁₀ |E|² (normalised)")):
        ax.set_aspect("equal")
        ax.set_xlabel(f"{axes_[0]} (µm)")
        ax.set_ylabel(f"{axes_[1]} (µm)")
        fig.colorbar(im, ax=ax, shrink=0.85, label=lab)
    axs[0].set_title(title)
    axs[1].set_title("same field, log scale")
    return _save(fig, path)


# =============================================================== layout
def layout_views(gds_path: Path, p: dict, path: Path) -> Path:
    """The ring + bus + couplers cell from the GDS: the whole device (every shape outlined,
    so the 1 µm bus stays visible at this scale) and three zooms: coupling region,
    waveguide-to-taper joint and grating."""
    import gdstk
    plt = _plt()
    lib = gdstk.read_gds(str(gds_path))
    cell = next((c for c in lib.cells if any(t in c.name for t in ("_gc", "_agc", "_fgc"))),
                lib.top_level()[0])
    polys = [np.asarray(q.points) for q in cell.get_polygons()]
    colour = "#3182bd"
    fig = plt.figure(figsize=(15, 8.5))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])
    top = fig.add_subplot(grid[0, :])
    zooms = [fig.add_subplot(grid[1, i]) for i in range(3)]
    for ax in [top] + zooms:
        for pts in polys:
            # an outline as well as the fill: thin shapes are drawn at least one line wide
            ax.fill(pts[:, 0], pts[:, 1], facecolor=colour, edgecolor=colour,
                    lw=1.2 if ax is top else 0)
        ax.set_aspect("equal")
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
    L, W = p["bus_length_um"], p["waveguide_width_um"]
    R = p.get("ring_radius_um") or 0.0
    gap = p.get("gap_critical_um") or 0.5
    focusing = bool(p.get("grating_focusing"))
    if focusing:                                  # tip taper, then the focus and the sector
        taper = p.get("focus_tip_taper_um") or 0.0
        x0 = L / 2 + taper + p["focus_distance_um"]
        tl = p.get("focus_teeth_list") or []
        n = (tl[-1][0] + tl[-1][1]) if tl else 0.0
    else:
        taper = p.get("taper_length_um") or 0.0
        x0 = L / 2 + taper                        # start of the grating
        tl = p.get("grating_teeth_list") or []
        n = ((tl[-1][0] + tl[-1][1]) if tl else
             int(p.get("grating_teeth") or 0) * (p.get("grating_period_um") or 0.0))
    top.autoscale()
    ymin, ymax = top.get_ylim()
    top.set_ylim(ymin - 0.12 * (ymax - ymin), ymax + 0.05 * (ymax - ymin))
    top.set_title(f"{cell.name}: whole device (outlines widened so the {W:g} µm "
                  "waveguide is visible)")
    label = dict(fontsize=9, ha="center", va="top", color="0.25")
    dy = 0.06 * (ymax - ymin)
    top.text(0, -dy, f"bus waveguide, W = {W:g} µm, {L:g} µm long", **label)
    if R:
        top.text(0, W / 2 + gap + R, f"ring R = {R:g} µm", fontsize=9, ha="center",
                 va="center", color="0.25")
    if taper and not focusing:                     # a few-µm tip taper is shown in the zoom
        for sgn in (-1, 1):
            top.text(sgn * (L / 2 + taper / 2), -dy, f"taper {taper:g} µm", **label)
    if n:
        for sgn in (-1, 1):
            top.text(sgn * (x0 + n / 2), -2.2 * dy,
                     "focusing grating" if focusing else "grating", **label)
    zooms[0].set_xlim(-6 * W - 3, 6 * W + 3)
    zooms[0].set_ylim(-3 * W, 2 * W + gap + 4)
    zooms[0].set_title(f"coupling region (gap {gap * 1e3:.0f} nm)")
    gw = (2 * p["focus_y_max_um"] if focusing else p.get("grating_width_um")) or 4 * W
    zooms[1].set_xlim(L / 2 - 10, L / 2 + min(taper, 30) + 2)
    zooms[1].set_ylim(-max(3 * W, gw / 4), max(3 * W, gw / 4))
    zooms[1].axvline(L / 2, color="0.5", ls=":", lw=1)
    zooms[1].set_aspect("auto")                    # y stretched: the taper widens slowly
    zooms[1].set_title(f"waveguide ({W:g} µm) into the {'tip taper and focusing slab' if focusing else 'taper'}"
                       "\n(y axis stretched)")
    if n:
        zooms[2].set_xlim(x0 - 15, x0 + n + 5)
        zooms[2].set_ylim(-gw, gw)
        if focusing:
            zooms[2].set_xlim(L / 2 + taper - 3, x0 + n + 3)
            zooms[2].set_aspect("equal")
            zooms[2].set_title(f"focusing grating: {len(tl)} curved teeth, focus "
                               f"{p['focus_distance_um']:g} µm before the first")
        elif tl:
            zooms[2].set_title(f"apodised grating: {len(tl)} teeth")
        else:
            zooms[2].set_title(f"grating: {p['grating_teeth']} × {p['grating_period_um']:.4f} µm, "
                               f"FF {p['grating_fill_factor']:.2f}")
    return _save(fig, path)


def cutback_views(spiral_gds: Path, ref_gds: Path, p: dict, path: Path) -> Path:
    """Figure 6 for test waveguides: the reference waveguide and every cutback spiral with
    its couplers (outlines widened so the waveguide stays visible), and a zoom on the
    centre of the longest spiral."""
    import gdstk
    plt = _plt()
    cells = list(gdstk.read_gds(str(spiral_gds)).top_level())
    if Path(ref_gds).exists():
        cells = list(gdstk.read_gds(str(ref_gds)).top_level()) + cells
    n = len(cells)
    fig, axes = plt.subplots(1, n + 1, figsize=(4.2 * (n + 1), 4.6), squeeze=False)
    colour = "#3182bd"
    for ax, cell in zip(axes[0], cells + [cells[-1]]):
        for q in cell.get_polygons():
            pts = np.asarray(q.points)
            ax.fill(pts[:, 0], pts[:, 1], facecolor=colour, edgecolor=colour, lw=0.6)
        ax.set_aspect("equal")
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        ax.set_title(cell.name, fontsize=9)
    zoom = axes[0][-1]
    r = 1.6 * (p.get("spiral_bend_radius_um") or 50.0)
    zoom.set_xlim(-r * 1.4, r * 1.4)
    zoom.set_ylim(-r, r)
    zoom.set_title(f"centre of {cells[-1].name}: S-bend, smallest radius "
                   f"{p.get('spiral_bend_radius_um') or 0:g} µm", fontsize=9)
    fig.suptitle(f"Cutback set: {', '.join(f'{x:g}' for x in p.get('cutback_lengths_cm') or [])} cm "
                 f"(pitch {p.get('spiral_pitch_um') or 0:g} µm), {p.get('main_coupler')} couplers")
    return _save(fig, path)


def stage_figure_path(ctx, name: str) -> Path:
    return ctx.path("figures", name)


def fmt_minutes(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m} min {s:02d} s" if m else f"{s} s"


def deg(x: float) -> float:
    return math.degrees(x)
