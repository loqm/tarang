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

   EME replaces the smooth taper by a staircase of straight cells. Each step scatters a
   little power into higher modes; for a smooth taper these small amplitudes cancel, but
   only if every cell is short compared to the beat length between the fundamental and the
   next even mode. With 30 equal cells the step at the narrow end is 0.45 um on a 1 um
   waveguide and the scattered amplitudes no longer cancel: the loss then depends on the
   cell phases and jumps up and down with length. So (a) the cells are graded (constant
   width ratio per cell: short cells where the width changes fastest relative to itself),
   and (b) a free 2D model of the same taper (effective-index method, mode matching with
   many cells) gives the converged curve, predicts the staircase error of the 3D cell
   count, and is used for the length when the 3D result is not converged.
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


# ------------------------------------------------------------------ taper shapes
PROFILES = ("linear", "parabolic", "power")


def profile_width(s, w_narrow: float, w_wide: float, kind: str = "linear", m: float = 0.6):
    """Width of a taper at s = distance from its narrow end / length (0 ... 1).

    linear     w = w_n + (w_w - w_n) s
    parabolic  w^2 = w_n^2 + (w_w^2 - w_n^2) s   (the classic adiabatic shape: the mode of a
               wide guide changes slowly with width, so the taper may widen fast where it is
               narrow and must widen slowly where it is wide)
    power      w = w_n + (w_w - w_n) s^m"""
    s = np.clip(np.asarray(s, float), 0.0, 1.0)
    if kind == "linear":
        return w_narrow + (w_wide - w_narrow) * s
    if kind == "parabolic":
        return np.sqrt(w_narrow ** 2 + (w_wide ** 2 - w_narrow ** 2) * s)
    if kind == "power":
        return w_narrow + (w_wide - w_narrow) * s ** float(m)
    raise ValueError(f"unknown taper shape {kind!r}; use one of {PROFILES}")


def profile_fraction(w, w_narrow: float, w_wide: float, kind: str = "linear", m: float = 0.6):
    """Inverse of profile_width: s (0 at the narrow end, 1 at the wide end) where the width is w."""
    u = np.clip((np.asarray(w, float) - w_narrow) / (w_wide - w_narrow), 0.0, 1.0)
    if kind == "linear":
        return u
    if kind == "parabolic":
        return np.clip((np.asarray(w, float) ** 2 - w_narrow ** 2) / (w_wide ** 2 - w_narrow ** 2), 0, 1)
    if kind == "power":
        return u ** (1.0 / float(m))
    raise ValueError(f"unknown taper shape {kind!r}; use one of {PROFILES}")


# ------------------------------------------------------------------ taper cells
def cell_boundaries(w_wide: float, w_narrow: float, n_cells: int, length: float = 1.0,
                    kind: str = "linear", m: float = 0.6) -> np.ndarray:
    """Positions (0 ... length, the wide end at 0) of the cell boundaries of a taper, graded
    so that the width changes by the same ratio in every cell: w_i = W (w/W)^(i/N). Near the
    narrow end, where a fixed width step is a large change of the mode, the cells are short.
    ``kind`` / ``m``: the taper shape (profile_width); the boundaries sit where the shape
    reaches those widths."""
    i = np.arange(int(n_cells) + 1) / int(n_cells)
    wb = w_wide * (w_narrow / w_wide) ** i
    if kind == "linear":
        x = (w_wide - wb) / (w_wide - w_narrow) * length
    else:
        x = (1.0 - profile_fraction(wb, w_narrow, w_wide, kind, m)) * length
    x[0], x[-1] = 0.0, length
    return x


# ------------------------------------------------------------------ 2D model (free)
def eim_modes(width: float, n_core: float, n_side: float, wl: float, half_domain: float,
              dy: float = 0.02, n_modes: int = 30, smooth: bool = False):
    """Even lateral modes of a slab of index n_core and width ``width`` in n_side, on
    0 <= y <= half_domain (mirror at y = 0, metal wall at the edge), for the in-plane
    field normal to the walls (TM-like, as the TE mode of the film sees the side walls).
    Solves d/dy(1/eps dH/dy) + k0^2 H = beta^2 H / eps for H = H_z with finite differences.
    Returns beta (1/um, propagating modes only), the profiles H (normalised to unit power:
    integral beta H^2 / eps dy = 1) and eps on the grid."""
    y = np.arange(dy / 2, half_domain, dy)
    eps = np.where(y <= width / 2, n_core ** 2, n_side ** 2)
    if smooth:
        # the grid cell that the wall crosses gets the average of 1/eps over the cell (the
        # field component is normal to the wall): the width then changes continuously
        # instead of in steps of 2 dy, which matters for long tapers
        frac = np.clip((width / 2 - (y - dy / 2)) / dy, 0.0, 1.0)
        eps = 1.0 / (frac / n_core ** 2 + (1 - frac) / n_side ** 2)
    k0 = 2 * math.pi / wl
    w = 0.5 * (1 / eps[1:] + 1 / eps[:-1]) / dy ** 2          # 1/eps between grid points
    d = np.full(len(y), k0 ** 2)
    d[:-1] -= w
    d[1:] -= w                                                 # dH/dy = 0 at both ends
    # symmetric form: H = sqrt(eps) u turns  A H = beta^2 H / eps  into  S u = beta^2 u
    ds = d * eps
    es = w * np.sqrt(eps[1:] * eps[:-1])
    n = len(y)
    m = min(int(n_modes), n)
    S = np.diag(ds) + np.diag(es, 1) + np.diag(es, -1)
    vals, vecs = np.linalg.eigh(S)                             # ascending
    vals, vecs = vals[::-1][:m], vecs[:, ::-1][:, :m]
    keep = vals > 0
    beta = np.sqrt(vals[keep])
    H = vecs[:, keep] * np.sqrt(eps)[:, None]
    for i in range(H.shape[1]):
        H[:, i] /= math.sqrt(np.sum(beta[i] / eps * H[:, i] ** 2) * dy)
        if H[0, i] < 0:
            H[:, i] *= -1
    return beta, H, eps


def eim_taper(n_slab: float, n_side: float, w_wide: float, w_narrow: float, wl: float, lengths,
              n_cells: int = 600, n_modes: int = 30, margin: float = 2.0, dy: float = 0.02,
              kind: str = "linear", m: float = 0.6, smooth: bool = False) -> dict:
    """Transmission of the fundamental mode through a taper (2D, effective-index method;
    shape ``kind`` / ``m``, see profile_width): the taper is cut into ``n_cells`` graded
    cells (cell_boundaries), each a straight section of its mid width; the modes of
    neighbouring cells are joined by their overlaps (transmission
    t_ij = (<E_j, H_i> + <E_i, H_j>) / 2, reflections neglected) and propagate with
    exp(i beta L) in between. The same cells and modes serve every length."""
    half = w_wide / 2 + margin
    xb = cell_boundaries(w_wide, w_narrow, n_cells, kind=kind, m=m)
    wb = (w_wide + (w_narrow - w_wide) * xb if kind == "linear"
          else profile_width(1.0 - xb, w_narrow, w_wide, kind, m))
    widths = 0.5 * (wb[1:] + wb[:-1])
    fracs = np.diff(xb)
    secs = [eim_modes(v, n_slab, n_side, wl, half, dy, n_modes, smooth)
            for v in [w_wide, *widths, w_narrow]]
    T = []
    for (b1, h1, e1), (b2, h2, e2) in zip(secs[:-1], secs[1:]):
        T.append(0.5 * dy * ((h2.T / e2) @ h1 * b1[None, :] + (h2.T / e1) @ h1 * b2[:, None]))
    trans = []
    for L in lengths:
        a = np.zeros(len(secs[0][0]), complex)
        a[0] = 1.0
        a = T[0] @ a
        for k in range(len(widths)):
            a = T[k + 1] @ (np.exp(1j * secs[k + 1][0] * fracs[k] * float(L)) * a)
        trans.append(float(abs(a[0]) ** 2))
    trans = np.clip(np.array(trans), 1e-12, None)
    return dict(length_um=[float(v) for v in lengths], transmission=trans.tolist(),
                loss_db=(-10 * np.log10(trans)).tolist(), cells=int(n_cells), modes=int(n_modes))


def rise_db(curve: dict) -> float:
    """Largest increase of the loss with length (0 for a curve that only falls): a smooth
    taper's loss falls with length (apart from a small ripple), so a large rise means the
    EME staircase is not converged."""
    L = np.array(curve["length_um"], float)
    loss = np.array(curve["loss_db"], float)[np.argsort(L)]
    best, rise = math.inf, 0.0
    for v in loss:
        rise = max(rise, v - best)
        best = min(best, v)
    return float(rise)


# ------------------------------------------------------------------ EME taper (3D, cloud)
def eme_taper(ctx, w_wide: float, w_narrow: float, length0: float, scale_factors,
              n_cells: int = 120, n_modes: int = 20, steps: int = 15, kind: str = "linear",
              m: float = 0.6):
    """Tidy3D EMESimulation of a taper (shape ``kind`` / ``m``, profile_width) from w_wide
    (x = 0) to w_narrow (x = length0) with short straight sections at both ends, the taper
    cut into ``n_cells`` graded cells (cell_boundaries); one EME run, the taper cells
    rescaled by ``scale_factors`` (taper length = length0 * scale; the straight sections
    keep their length; the shape scales with the length)."""
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
        td.Structure(geometry=td.PolySlab(vertices=taper_outline(w_wide, w_narrow, length0, kind, m),
                                          slab_bounds=(0, H), axis=2), medium=core, name="taper"),
        td.Structure(geometry=td.Box(center=(length0 + straight, 0, H / 2),
                                     size=(2 * straight, w_narrow, H)), medium=core,
                     name="waveguide")]
    ms = td.EMEModeSpec(num_modes=int(n_modes), target_neff=ctx.n(cfg.materials.core))
    inner = cell_boundaries(w_wide, w_narrow, n_cells, length0, kind, m)[1:-1]
    grid = td.EMECompositeGrid(
        subgrids=[td.EMEUniformGrid(num_cells=1, mode_spec=ms),
                  td.EMEExplicitGrid(mode_specs=[ms] * int(n_cells),
                                     boundaries=[float(v) for v in inner]),
                  td.EMEUniformGrid(num_cells=1, mode_spec=ms)],
        subgrid_boundaries=[0.0, length0])
    # per cell: the straight end sections keep their length, the taper cells are scaled
    scales = [[1.0] + [float(s)] * int(n_cells) + [1.0] for s in scale_factors]
    z_lo, z_hi = -1.5, H + 1.5
    return td.EMESimulation(
        center=(length0 / 2, 0, (z_lo + z_hi) / 2),
        size=(length0 + 2 * straight, w_wide + 2 * margin, z_hi - z_lo),
        structures=structs, axis=0, freqs=[C_UM / wl], eme_grid_spec=grid,
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=int(steps), wavelength=wl),
        symmetry=(0, -1, 0),                    # even modes only (E across the width even)
        boundary_spec=td.BoundarySpec.all_sides(td.PECBoundary()),
        sweep_spec=td.EMELengthSweep(scale_factors=scales))


def taper_outline(w_wide: float, w_narrow: float, length: float, kind: str = "linear",
                  m: float = 0.6, n: int = 120) -> list:
    """Polygon (x, y) of a taper, wide end at x = 0, narrow end at x = length; a linear
    taper is the four corners, a curved one ``n`` points per side (denser near the narrow
    end, where a parabolic or power-law taper bends most)."""
    if kind == "linear":
        return [(0, -w_wide / 2), (length, -w_narrow / 2), (length, w_narrow / 2), (0, w_wide / 2)]
    s = np.linspace(0, 1, int(n)) ** 2                    # from the narrow end
    w = profile_width(s, w_narrow, w_wide, kind, m)
    x = (1 - s) * length
    upper = [(float(a), float(b) / 2) for a, b in zip(x, w)]          # narrow -> wide
    lower = [(float(a), -float(b) / 2) for a, b in zip(x[::-1], w[::-1])]  # wide -> narrow
    return upper + lower


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
DEFAULT_LENGTHS = [10, 20, 30, 40, 60, 80, 100, 150, 200, 250, 300, 400]
MODEL_CELLS = 600            # cells of the converged 2D model (600 agrees with 1200 to < 0.01 dB)


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
        dict(section="taper", key="use_in_layout")])


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


def draw_length(ctx, model: dict, curve: Optional[dict], chosen: Optional[float], source: str,
                w_wide: float, w_narrow: float, layout_now: float, converged: Optional[bool] = None):
    """Figure 8b: taper loss against length; the converged 2D model as a line, the 3D EME
    as points (open when it is not converged)."""
    from tarang.design import visuals
    plt = visuals._plt()
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    L = np.array(model["length_um"])
    ax.semilogy(L, np.clip(model["loss_db"], 1e-4, None), "-", color="0.35", lw=1.8,
                label=f"2D model, effective index ({model['cells']} cells, converged)")
    if curve:
        L3 = np.array(curve["length_um"])
        o = np.argsort(L3)
        good = converged is not False
        ax.semilogy(L3[o], np.clip(np.array(curve["loss_db"])[o], 1e-4, None), "o-" if good else "o--",
                    color=visuals.PALETTE[0], lw=1.5, mfc=visuals.PALETTE[0] if good else "white",
                    label=f"3D EME, {curve.get('cells', '?')} cells x {curve.get('modes', '?')} modes"
                          + ("" if good else " (NOT converged)"))
    ax.axhline(ctx.cfg.taper.max_loss_db, color="0.5", ls=":", label=f"accepted loss "
               f"{ctx.cfg.taper.max_loss_db:g} dB")
    if chosen:
        ax.axvline(chosen, color=visuals.PALETTE[1], ls="--", label=f"chosen: {chosen:.0f} µm ({source})")
    ax.axvline(layout_now, color="0.6", ls="-.", lw=1, label=f"layout until now: {layout_now:g} µm")
    ax.set(xlabel="Taper length (µm)", ylabel="Taper loss (dB)",
           title=f"Linear taper {w_wide:g} → {w_narrow:g} µm (one way, fundamental mode)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5)
    return visuals._save(fig, ctx.path("figures", "8b_taper_length.png"))


def stage_taper(ctx) -> dict:
    """Grating width (free), taper length from the 2D model (free) and Tidy3D EME (priced)."""
    from tarang.design import visuals
    from tarang.design.checks import _run_batch, _write_csv
    from tarang.utilities.cloud_run import _data_class, find_cached
    _form(ctx)
    cfg, ui = ctx.cfg, ctx.ui
    wl = cfg.run.wavelength_um
    xs = ctx.results("cross_section")
    w_narrow = xs["width_um"]
    ws = width_sweep(ctx, cfg.taper.widths_um or None)
    chosen_w = float(cfg.taper.grating_width_um) if cfg.taper.grating_width_um else \
        round(ws["best_width_um"] * 2) / 2                     # nearest 0.5 um
    eta_chosen = lateral_overlap(ws["n_slab"], ws["n_side"], chosen_w, ws["mfd_um"], wl)
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

    # ---- taper length, 2D model (free, converged)
    max_loss = float(cfg.taper.max_loss_db)
    lengths = sorted({float(v) for v in (cfg.taper.lengths_um or DEFAULT_LENGTHS)})
    L0 = max(lengths)
    dense = sorted(set(np.round(np.geomspace(5.0, max(L0, 500.0), 60), 1).tolist()) | set(lengths))
    ns, nside = ws["n_slab"], ws["n_side"]
    model = eim_taper(ns, nside, chosen_w, w_narrow, wl, dense, n_cells=MODEL_CELLS)
    L_2d = shortest_length(model, max_loss)
    L_2d = None if L_2d is None else float(math.ceil(L_2d / 10.0) * 10.0)   # round up to 10 um
    layout_now = float(cfg.grating.taper_length_um)
    res.update(model_lengths_um=model["length_um"], model_loss_db=model["loss_db"],
               model_length_um=L_2d, model_loss_db_at_length=loss_at(model, L_2d) if L_2d else None,
               previous_length_um=layout_now, model_loss_db_at_previous=loss_at(model, layout_now))
    ui.say(f"  Taper length, 2D model (effective index, {model['cells']} graded cells, free): "
           + (f"shortest below {max_loss:g} dB: {L_2d:.0f} µm ({res['model_loss_db_at_length']:.3f} dB)"
              if L_2d else f"no length up to {max(dense):g} µm below {max_loss:g} dB")
           + f"; the {layout_now:g} µm taper so far: {res['model_loss_db_at_previous']:.3f} dB.")
    _write_csv(ctx.path("tables", "taper_length_model.csv"),
               [dict(length_um=a, loss_db=b) for a, b in zip(model["length_um"], model["loss_db"])])
    chosen_L, source, curve, converged = L_2d, "2D model", None, None

    # ---- 3D EME (priced): the same cells, checked against the 2D staircase error first
    if cfg.taper.run_eme:
        nc, nm = int(cfg.taper.eme_cells), int(cfg.taper.eme_modes)
        stair = eim_taper(ns, nside, chosen_w, w_narrow, wl, lengths, n_cells=nc, n_modes=min(nm, 30))
        ref = [loss_at(model, v) for v in lengths]
        err = float(np.max(np.abs(np.array(stair["loss_db"]) - np.array(ref))))
        res["staircase_error_db"] = err
        ui.say(f"  3D EME with {nc} cells x {nm} modes: the same staircase in the 2D model differs from "
               f"the converged curve by up to {err:.3f} dB over {min(lengths):g}-{L0:g} µm"
               + (" (more cells, eme_cells, would bring it below the accepted loss)."
                  if err > max_loss else "."))
        scales = [v / L0 for v in lengths]
        sim = eme_taper(ctx, chosen_w, w_narrow, L0, scales, nc, nm)
        name = f"taper_W{chosen_w:g}to{w_narrow:g}um_L{L0:g}um_eme{nc}x{nm}g"
        folder = ctx.out / "cloud" / "taper"
        ok = _run_batch(ctx, {"taper EME": (sim, name)}, folder, "taper loss against length (EME)")
        hit = find_cached(sim, folder, name)
        if ok and hit is not None:
            data = _data_class(sim).from_file(str(hit))
            curve = taper_transmission(data, L0, scales)
            curve.update(cells=nc, modes=nm)
            rise = rise_db(curve)
            converged = rise <= max_loss
            rows = [dict(length_um=l, transmission=t, loss_db=d, model_2d_db=loss_at(model, l))
                    for l, t, d in zip(curve["length_um"], curve["transmission"], curve["loss_db"])]
            ui.table(rows, formats={"transmission": ".4f", "loss_db": ".3f", "model_2d_db": ".3f"})
            _write_csv(ctx.path("tables", "taper_length.csv"), rows)
            L_3d = shortest_length(curve, max_loss)
            L_3d = None if L_3d is None else float(math.ceil(L_3d / 10.0) * 10.0)
            res.update(lengths_um=curve["length_um"], loss_db=curve["loss_db"], eme_cells=nc,
                       eme_modes=nm, eme_rise_db=rise, eme_converged=converged, eme_length_um=L_3d,
                       loss_db_at_previous_length=loss_at(curve, layout_now))
            if not converged:
                ui.say(f"  The 3D loss rises by {rise:.2f} dB somewhere with increasing length: the EME "
                       "staircase is not converged (more eme_cells / eme_modes). The length is taken "
                       "from the 2D model.")
            elif L_3d:
                chosen_L, source = L_3d, "3D EME"
                ui.say(f"  3D EME: shortest below {max_loss:g} dB: {L_3d:.0f} µm "
                       f"({loss_at(curve, L_3d):.3f} dB).")
            else:
                ui.say(f"  3D EME: no swept length below {max_loss:g} dB (longest {L0:g} µm); the "
                       "length is taken from the 2D model.")
        else:
            ui.say("  3D EME not run; the length is taken from the 2D model.")
    if chosen_L:
        loss_c = loss_at(curve, chosen_L) if source == "3D EME" else loss_at(model, chosen_L)
        res.update(taper_length_um=chosen_L, taper_loss_db=loss_c, length_source=source)
        ui.say(f"  Taper length used: {chosen_L:.0f} µm ({source}, {loss_c:.3f} dB per taper).")
    else:
        res.update(taper_length_um=None, taper_loss_db=None, length_source=None)
        ui.say("  No taper length reaches the accepted loss; the layout keeps its taper.")
    visuals.safe(ctx, "taper length", draw_length, ctx, model, curve, chosen_L, source, chosen_w,
                 w_narrow, layout_now, converged)
    return res
