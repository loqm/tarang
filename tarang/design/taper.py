"""Taper between the grating and the waveguide: width of the grating end and taper length.

Two questions, from first principles:

1. **How wide should the grating be?** The 2D grating simulation treats the grating as
   infinitely wide. A real grating of width W guides light in its fundamental lateral mode
   psi(y) (close to cos(pi y / W) inside, decaying outside), while the fibre delivers a
   Gaussian G(y) = exp(-y^2 / w0^2), w0 = MFD / 2. Only the overlap couples:

       eta_y(W) = |integral psi G dy|^2 / (integral |psi|^2 dy * integral |G|^2 dy)

   so the 3D efficiency is about eta_2D * eta_y. psi comes from the effective-index method:
   the film (Si3N4 on the BOX, the top cladding above) gives the slab index n_slab; the
   lateral profile is the fundamental mode of a slab of index n_slab and width W in the
   cladding (E across the width is normal to the side walls: TM-like boundary condition).
   Too narrow cuts the Gaussian, too wide spreads psi; the optimum is near W = 2.7 w0.

2. **How long should the taper be?** A linear taper from W to the waveguide width w is
   lossless only if it changes slowly enough for the fundamental mode to follow
   (adiabatic); a short taper scatters power into higher modes and radiation. The
   eigenmode-expansion (EME) solver of Tidy3D finds the local modes of the taper in a
   number of cells and joins them with their overlaps, which gives the transmission of the
   fundamental mode, |S21|^2. Rescaling the cell lengths re-uses the same modes, so one run
   gives |S21|^2 against taper length (EMELengthSweep). The mirror plane y = 0 is imposed
   (only the even modes the symmetric taper can excite).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

C_UM = 2.99792458e14


# ------------------------------------------------------------------ slab modes (EIM)
def _brent(f, a, b, tol=1e-13, it=200):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("root not bracketed")
    for _ in range(it):
        m = 0.5 * (a + b)
        fm = f(m)
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
        if b - a < tol:
            break
    return 0.5 * (a + b)


def slab_index_te(n_core: float, n_below: float, n_above: float, d: float, wl: float) -> float:
    """Effective index of the fundamental TE mode (E parallel to the layers) of an
    asymmetric slab: k d = atan(g_below / k) + atan(g_above / k)."""
    k0 = 2 * math.pi / wl

    def f(n):
        k = k0 * math.sqrt(n_core ** 2 - n ** 2)
        g2 = k0 * math.sqrt(n ** 2 - n_below ** 2)
        g3 = k0 * math.sqrt(n ** 2 - n_above ** 2)
        return k * d - math.atan(g2 / k) - math.atan(g3 / k)
    lo = max(n_below, n_above) + 1e-12
    return _brent(f, lo, n_core - 1e-12)


def lateral_mode(n_slab: float, n_side: float, width: float, wl: float, y: np.ndarray):
    """Fundamental lateral mode of a symmetric slab (index n_slab, width ``width``) in
    n_side, TM-like (field normal to the walls): tan(k W/2) = (n_slab/n_side)^2 g / k.
    Returns (effective index, profile on ``y`` of the continuous field component)."""
    k0 = 2 * math.pi / wl
    r = (n_slab / n_side) ** 2

    def f(n):
        k = k0 * math.sqrt(n_slab ** 2 - n ** 2)
        g = k0 * math.sqrt(n ** 2 - n_side ** 2)
        return math.tan(k * width / 2) - r * g / k
    lo = math.sqrt(max(n_slab ** 2 - (math.pi / (k0 * width)) ** 2, n_side ** 2)) + 1e-12
    n = _brent(f, lo, n_slab - 1e-12)
    k = k0 * math.sqrt(n_slab ** 2 - n ** 2)
    g = k0 * math.sqrt(n ** 2 - n_side ** 2)
    a = np.abs(y)
    psi = np.where(a <= width / 2, np.cos(k * y), math.cos(k * width / 2) * np.exp(-g * (a - width / 2)))
    return n, psi


def lateral_overlap(n_slab: float, n_side: float, width: float, mfd: float, wl: float) -> float:
    """eta_y: power overlap of the fibre's Gaussian (waist MFD/2) with the grating's
    fundamental lateral mode."""
    w0 = mfd / 2
    half = max(3 * width, 4 * w0)
    y = np.linspace(-half, half, 40001)
    _, psi = lateral_mode(n_slab, n_side, width, wl, y)
    g = np.exp(-(y / w0) ** 2)
    num = np.trapezoid(psi * g, y) ** 2
    return float(num / (np.trapezoid(psi ** 2, y) * np.trapezoid(g ** 2, y)))


def width_sweep(ctx, widths=None) -> dict:
    """eta_y over grating widths; the optimum width and eta_y at the current width."""
    cfg = ctx.cfg
    wl = cfg.run.wavelength_um
    H = ctx.results("cross_section")["height_um"]
    n_core, n_box = ctx.n(cfg.materials.core), ctx.n(cfg.materials.box)
    n_top = 1.0 if str(cfg.materials.cladding).lower() == "air" else ctx.n(cfg.materials.cladding)
    n_slab = slab_index_te(n_core, n_box, n_top, H, wl)
    n_side = max(n_box, n_top)                 # no film beside the grating: the claddings
    mfd = cfg.grating.mfd_um
    widths = list(widths) if widths else [round(float(w), 2) for w in
                                           np.linspace(0.8 * mfd, 2.2 * mfd, 29)]
    rows = [dict(width_um=w, eta_y=lateral_overlap(n_slab, n_side, w, mfd, wl)) for w in widths]
    for r in rows:
        r["loss_db"] = -10 * math.log10(max(r["eta_y"], 1e-12))
    fine = np.linspace(min(widths), max(widths), 141)
    etas = np.array([lateral_overlap(n_slab, n_side, w, mfd, wl) for w in fine])
    w_best = float(fine[int(np.argmax(etas))])
    now = float(cfg.grating.width_um)
    return dict(rows=rows, n_slab=n_slab, n_side=n_side, mfd_um=mfd, best_width_um=round(w_best, 2),
                best_eta_y=float(etas.max()),
                current_width_um=now, current_eta_y=lateral_overlap(n_slab, n_side, now, mfd, wl))


# ------------------------------------------------------------------ EME taper
def eme_taper(ctx, w_wide: float, w_narrow: float, length0: float, scale_factors,
              n_cells: int = 30, n_modes: int = 8, steps: int = 15):
    """Tidy3D EMESimulation of a linear taper from w_wide (x = 0) to w_narrow (x = length0)
    with short straight sections at both ends; one EME run, cell lengths rescaled by
    ``scale_factors`` (lengths = length0 * scale)."""
    import tidy3d as td
    cfg = ctx.cfg
    H = ctx.results("cross_section")["height_um"]
    wl = cfg.run.wavelength_um
    core = ctx.medium(cfg.materials.core)
    box = ctx.medium(cfg.materials.box)
    clad_air = str(cfg.materials.cladding).lower() == "air"
    straight = 2.0
    margin = 2.0
    structs = [td.Structure(geometry=td.Box(center=(length0 / 2, 0, -cfg.stack.box_height_um / 2),
                                            size=(td.inf, td.inf, cfg.stack.box_height_um)),
                            medium=box, name="BOX")]
    if not clad_air:
        structs.append(td.Structure(geometry=td.Box(
            center=(length0 / 2, 0, cfg.stack.cladding_height_um / 2),
            size=(td.inf, td.inf, cfg.stack.cladding_height_um)),
            medium=ctx.medium(cfg.materials.cladding), name="top cladding"))
    structs += [
        td.Structure(geometry=td.Box(center=(-straight, 0, H / 2), size=(2 * straight, w_wide, H)),
                     medium=core, name="grating end"),
        td.Structure(geometry=td.PolySlab(vertices=[(0, -w_wide / 2), (length0, -w_narrow / 2),
                                                    (length0, w_narrow / 2), (0, w_wide / 2)],
                                          slab_bounds=(0, H), axis=2), medium=core, name="taper"),
        td.Structure(geometry=td.Box(center=(length0 + straight, 0, H / 2),
                                     size=(2 * straight, w_narrow, H)), medium=core,
                     name="waveguide")]
    ms = td.EMEModeSpec(num_modes=int(n_modes), target_neff=ctx.n(cfg.materials.core))
    grid = td.EMECompositeGrid(
        subgrids=[td.EMEUniformGrid(num_cells=1, mode_spec=ms),
                  td.EMEUniformGrid(num_cells=int(n_cells), mode_spec=ms),
                  td.EMEUniformGrid(num_cells=1, mode_spec=ms)],
        subgrid_boundaries=[0.0, length0])
    z_lo, z_hi = -1.5, H + 1.5
    return td.EMESimulation(
        center=(length0 / 2, 0, (z_lo + z_hi) / 2),
        size=(length0 + 2 * straight, w_wide + 2 * margin, z_hi - z_lo),
        structures=structs, axis=0, freqs=[C_UM / wl], eme_grid_spec=grid,
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=int(steps), wavelength=wl),
        symmetry=(0, -1, 0),                    # even modes only (E across the width even)
        boundary_spec=td.BoundarySpec.all_sides(td.PECBoundary()),
        sweep_spec=td.EMELengthSweep(scale_factors=[float(s) for s in scale_factors]))


def taper_transmission(data, length0: float, scale_factors) -> dict:
    """|S21|^2 of the fundamental mode against taper length."""
    s21 = data.smatrix.S21
    if "f" in s21.dims:
        s21 = s21.isel(f=0)
    t = np.abs(np.asarray(s21.isel(mode_index_out=0, mode_index_in=0).values)) ** 2
    lengths = [length0 * float(s) for s in scale_factors]
    t = np.atleast_1d(t).astype(float)
    return dict(length_um=lengths, transmission=t.tolist(),
                loss_db=[-10 * math.log10(max(v, 1e-12)) for v in t])


def loss_at(curve: dict, length: float) -> float:
    """Taper loss at ``length``, interpolating log(loss) between the swept lengths."""
    L = np.array(curve["length_um"], float)
    o = np.argsort(L)
    lg = np.log(np.clip(np.array(curve["loss_db"], float)[o], 1e-12, None))
    return float(np.exp(np.interp(length, L[o], lg)))


def shortest_length(curve: dict, max_loss_db: float) -> Optional[float]:
    """Shortest length whose loss stays below ``max_loss_db`` from there on. Between two
    swept lengths the loss of an adiabatic taper falls roughly exponentially with length, so
    log(loss) is interpolated linearly."""
    L = np.array(curve["length_um"])
    loss = np.array(curve["loss_db"])
    o = np.argsort(L)
    L, loss = L[o], loss[o]
    ok = loss <= max_loss_db
    if not ok.any():
        return None
    # last index where it is still above the target, then everything after is below
    bad = np.where(~ok)[0]
    if not len(bad):
        return float(L[0])
    i = bad[-1]
    if i == len(L) - 1:
        return None
    a, b = math.log(max(loss[i], 1e-12)), math.log(max(loss[i + 1], 1e-12))
    t = (a - math.log(max_loss_db)) / (a - b) if a != b else 1.0
    return float(L[i] + t * (L[i + 1] - L[i]))


# ------------------------------------------------------------------ the design stage
DEFAULT_LENGTHS = [5, 10, 15, 20, 30, 40, 60, 80, 100, 150, 200, 300, 400, 500]


def _form(ctx) -> None:
    from tarang.design.ui import parse_list
    ctx.ui.form(ctx.cfg, "Taper", [
        dict(section="taper", key="grating_width_um"),
        dict(section="taper", key="widths_um", cast=parse_list),
        dict(section="taper", key="run_eme"),
        dict(section="taper", key="lengths_um", cast=parse_list,
             show=lambda c: bool(c.taper.run_eme)),
        dict(section="taper", key="max_loss_db", show=lambda c: bool(c.taper.run_eme)),
        dict(section="taper", key="eme_cells", show=lambda c: bool(c.taper.run_eme)),
        dict(section="taper", key="eme_modes", show=lambda c: bool(c.taper.run_eme)),
        dict(section="taper", key="use_in_layout"),
        dict(section="run", key="max_cost_per_run"),
        dict(section="run", key="max_total_cost")])


def draw_width(ctx, ws: dict, chosen: float):
    from tarang.design import visuals
    plt = visuals._plt()
    cfg = ctx.cfg
    wl = cfg.run.wavelength_um
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    w = np.array([r["width_um"] for r in ws["rows"]])
    eta = np.array([r["eta_y"] for r in ws["rows"]])
    ax = axes[0]
    ax.plot(w, eta * 100, "-", color=visuals.PALETTE[0], lw=2)
    ax.axvline(ws["current_width_um"], color="0.5", ls=":", lw=1.2,
               label=f"set before: {ws['current_width_um']:g} µm, {ws['current_eta_y'] * 100:.1f} %")
    ax.axvline(chosen, color=visuals.PALETTE[1], ls="--", lw=1.5,
               label=f"chosen: {chosen:g} µm, "
                     f"{lateral_overlap(ws['n_slab'], ws['n_side'], chosen, ws['mfd_um'], wl) * 100:.1f} %")
    ax.set(xlabel="Grating width W (µm)", ylabel="Lateral overlap η_y (%)",
           title=f"fibre (MFD {ws['mfd_um']:g} µm) against the grating's lateral mode")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower center")
    ax = axes[1]
    half = max(1.6 * chosen, 2.2 * ws["mfd_um"])
    y = np.linspace(-half, half, 2001)
    _, psi = lateral_mode(ws["n_slab"], ws["n_side"], chosen, wl, y)
    g = np.exp(-(y / (ws["mfd_um"] / 2)) ** 2)
    ax.plot(y, psi / np.abs(psi).max(), color=visuals.PALETTE[1], lw=2,
            label=f"grating mode, W = {chosen:g} µm (effective-index method)")
    ax.plot(y, g, "--", color="k", lw=1.5, label=f"fibre mode (Gaussian, w0 = {ws['mfd_um'] / 2:g} µm)")
    ax.axvspan(-chosen / 2, chosen / 2, color="0.92", zorder=0)
    ax.set(xlabel="y across the grating (µm)", ylabel="field (normalised)",
           title="the two lateral profiles")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Grating width: the 3D efficiency is about η_2D × η_y")
    return visuals._save(fig, ctx.path("figures", "8a_grating_width.png"))


def draw_length(ctx, curve: dict, chosen: Optional[float], w_wide: float, w_narrow: float,
                layout_now: float):
    from tarang.design import visuals
    plt = visuals._plt()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    L = np.array(curve["length_um"])
    loss = np.array(curve["loss_db"])
    o = np.argsort(L)
    ax.semilogy(L[o], np.clip(loss[o], 1e-4, None), "o-", color=visuals.PALETTE[0], lw=1.8,
                label="EME (fundamental mode in → out)")
    ax.axhline(ctx.cfg.taper.max_loss_db, color="0.5", ls=":", label=f"accepted loss "
               f"{ctx.cfg.taper.max_loss_db:g} dB")
    if chosen:
        ax.axvline(chosen, color=visuals.PALETTE[1], ls="--", label=f"shortest accepted: {chosen:.0f} µm")
    ax.axvline(layout_now, color="0.6", ls="-.", lw=1, label=f"layout until now: {layout_now:g} µm")
    ax.set(xlabel="Taper length (µm)", ylabel="Taper loss (dB)",
           title=f"Linear taper {w_wide:g} → {w_narrow:g} µm (one way)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    return visuals._save(fig, ctx.path("figures", "8b_taper_length.png"))


def stage_taper(ctx) -> dict:
    """Grating width (free) and taper length (EME, priced)."""
    from tarang.design import visuals
    from tarang.design.checks import _run_batch, _write_csv
    from tarang.utilities.cloud_run import _data_class, find_cached
    _form(ctx)
    cfg, ui = ctx.cfg, ctx.ui
    xs = ctx.results("cross_section")
    w_narrow = xs["width_um"]
    ws = width_sweep(ctx, cfg.taper.widths_um or None)
    chosen_w = float(cfg.taper.grating_width_um) if cfg.taper.grating_width_um else \
        round(ws["best_width_um"] * 2) / 2                     # nearest 0.5 um
    eta_chosen = lateral_overlap(ws["n_slab"], ws["n_side"], chosen_w, ws["mfd_um"], cfg.run.wavelength_um)
    ui.say(f"Lateral overlap of the fibre with the grating (effective-index method, slab index "
           f"{ws['n_slab']:.4f}): best width {ws['best_width_um']:.1f} µm (η_y "
           f"{ws['best_eta_y'] * 100:.1f} %); {ws['current_width_um']:g} µm gives "
           f"{ws['current_eta_y'] * 100:.1f} %; using {chosen_w:g} µm ({eta_chosen * 100:.1f} %).")
    _write_csv(ctx.path("tables", "grating_width.csv"), ws["rows"])
    visuals.safe(ctx, "grating width", draw_width, ctx, ws, chosen_w)
    res = dict(grating_width_um=chosen_w, eta_y=eta_chosen, eta_y_before=ws["current_eta_y"],
               width_before_um=ws["current_width_um"], best_width_um=ws["best_width_um"],
               n_slab=ws["n_slab"], use_in_layout=bool(cfg.taper.use_in_layout))
    gc = ctx.state["stages"].get("grating", {}).get("results") or {}
    if gc.get("efficiency_2d"):
        res["efficiency_3d_estimate"] = gc["efficiency_2d"] * eta_chosen
        ui.say(f"  grating efficiency estimate with the width: {gc['efficiency_2d'] * 100:.1f} % x "
               f"{eta_chosen:.3f} = {res['efficiency_3d_estimate'] * 100:.1f} % "
               f"({-10 * math.log10(res['efficiency_3d_estimate']):.2f} dB)")
    if not cfg.taper.run_eme:
        return res
    lengths = sorted({float(v) for v in (cfg.taper.lengths_um or DEFAULT_LENGTHS)})
    L0 = max(lengths)
    scales = [v / L0 for v in lengths]
    sim = eme_taper(ctx, chosen_w, w_narrow, L0, scales, cfg.taper.eme_cells, cfg.taper.eme_modes)
    name = f"taper_W{chosen_w:g}to{w_narrow:g}um_L{L0:g}um_eme{cfg.taper.eme_cells}x{cfg.taper.eme_modes}"
    folder = ctx.out / "cloud" / "taper"
    ok = _run_batch(ctx, {"taper EME": (sim, name)}, folder, "taper loss against length (EME)")
    hit = find_cached(sim, folder, name)
    if not ok or hit is None:
        ui.say("  Taper length not computed (skipped or not run); the layout keeps its taper.")
        return res
    data = _data_class(sim).from_file(str(hit))
    curve = taper_transmission(data, L0, scales)
    L_ok = shortest_length(curve, float(cfg.taper.max_loss_db))
    rows = [dict(length_um=l, transmission=t, loss_db=d)
            for l, t, d in zip(curve["length_um"], curve["transmission"], curve["loss_db"])]
    ui.table(rows, formats={"transmission": ".4f", "loss_db": ".3f"})
    _write_csv(ctx.path("tables", "taper_length.csv"), rows)
    layout_now = float(cfg.grating.taper_length_um)
    loss_now = loss_at(curve, layout_now)
    chosen_L = None if L_ok is None else float(math.ceil(L_ok / 5.0) * 5.0)   # round up to 5 um
    res.update(lengths_um=curve["length_um"], loss_db=curve["loss_db"],
               taper_length_um=chosen_L,
               taper_loss_db=loss_at(curve, chosen_L) if chosen_L else None,
               loss_db_at_previous_length=loss_now, previous_length_um=layout_now)
    if chosen_L:
        ui.say(f"  Shortest taper with loss below {cfg.taper.max_loss_db:g} dB: {chosen_L:.0f} µm "
               f"({res['taper_loss_db']:.3f} dB); the {layout_now:g} µm taper of the layout so far "
               f"loses {loss_now:.3f} dB.")
    else:
        ui.say(f"  No swept length reaches {cfg.taper.max_loss_db:g} dB; add longer lengths "
               "(taper.lengths_um) and rerun this stage.")
    visuals.safe(ctx, "taper length", draw_length, ctx, curve, chosen_L, chosen_w, w_narrow, layout_now)
    return res
