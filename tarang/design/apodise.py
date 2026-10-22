"""Apodised grating coupler: the scattering strength varies along the grating so that the
light it sends up has the shape of the fibre mode.

From first principles
---------------------
In a uniform grating every tooth scatters the same fraction of the light still guided.
With a scattering strength alpha (field, per um) the guided power decays as
P(x) = exp(-2 alpha x) and the emitted intensity S(x) = 2 alpha P(x) is an exponential,
which overlaps a Gaussian fibre mode by about 80 % at best (and less when alpha is far from
its optimum, as for a full etch). If instead

    alpha(x) = A G^2(x) / (2 (1 - A integral_0^x G^2))         (energy conservation)

then S(x) = A G^2(x): the emitted field has the fibre's Gaussian shape G(x) =
exp(-(x - x_c)^2 / w_x^2), with w_x = w0 / cos(theta) the beam's footprint on the grating.
A < 1 leaves a little power unscattered at the end and keeps alpha finite. Real teeth
give alpha only between alpha_min and alpha_max, so alpha(x) is clipped and the emitted
profile is computed with the clipped alpha.

Each tooth's alpha is set by its fill factor ff. Changing ff also changes the grating's
Bloch index n_B(ff), so each period is chosen for phase matching:
Lambda(ff) = lambda / (n_B(ff) - n_cover sin(theta)) for a beam travelling towards the
waveguide (the TARANG default; + for 'away').

alpha(ff) and n_B(ff) come from one small 2D run per fill factor: TE0 is launched from the
waveguide into a uniform grating and the field is recorded on a line above it, high enough
(1.5 um) that the evanescent diffraction orders have died out. Along that line the light is
the radiated beam only: |E|^2 decays as exp(-2 alpha x), and the phase advances as k_x x
with k_x = beta_B - 2 pi / Lambda, so n_B = (k_x + 2 pi / Lambda) / k0.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

C_UM = 2.99792458e14


# ------------------------------------------------------------------ uniform-grating runs
def uniform_reverse_sim(ctx, ff: float, period: float, n_teeth: int = 30, z_above: float = 1.5):
    """TE0 launched from the waveguide into a uniform grating, with a line field monitor
    ``z_above`` um above the top of the core along the whole grating (one frequency)."""
    import tidy3d as td
    from tarang.design.stages import grating_device
    gc = ctx.results("grating")
    H = ctx.results("cross_section")["height_um"]
    dev = grating_device(ctx, period, ff, gc["source_x_um"], H=H, depth=gc["etch_depth_um"])
    sim = dev.sim
    mm = next(m for m in sim.monitors if isinstance(m, td.ModeMonitor))
    f0 = C_UM / ctx.cfg.run.wavelength_um
    st = sim.sources[0].source_time
    src = td.ModeSource(center=(mm.center[0] - 1.5, 0, mm.center[2]), size=(0, td.inf, mm.size[2]),
                        source_time=st, direction="+", mode_spec=mm.mode_spec, mode_index=0,
                        name="waveguide_source")
    L = period * dev.config.n_teeth
    line = td.FieldMonitor(center=(L / 2, 0, H + z_above), size=(L + 2.0, 0, 0), freqs=[f0],
                           name="above", fields=["Ex", "Ey", "Ez"])
    return sim.updated_copy(sources=[src], monitors=[line]), dev


def _two_wave(x, E, period: float, length: float, alpha: float, kx: float, harmonics=(-1, 0, 1)):
    """Least-squares fit of E(x) on the grating to a forward wave decaying from the start
    and a backward wave decaying from the end (light reflected at the grating's end), each
    with its spatial harmonics k_x + m K (the near field of the teeth). Linear in the six
    complex amplitudes for given (alpha, k_x); returns (relative residual, amplitudes)."""
    K = 2 * math.pi / period
    cols = []
    for m in harmonics:
        cols.append(np.exp((1j * (kx + m * K) - alpha) * x))
        cols.append(np.exp(-alpha * (length - x) + 1j * (-kx + m * K) * x))
    A = np.array(cols).T
    c, *_ = np.linalg.lstsq(A, E, rcond=None)
    r = E - A @ c
    return float(np.vdot(r, r).real / max(np.vdot(E, E).real, 1e-300)), c


def alpha_and_index(x_um, e_fields: dict, period: float, wl: float, length: Optional[float] = None,
                    edge_periods: int = 2, kx_sign: float = 1.0) -> dict:
    """alpha (field decay, 1/um) and Bloch index from the field on the line above a uniform
    grating of length ``length`` (teeth from x = 0; default: the line is 1 um longer than the
    grating at both ends). ``e_fields``: complex field components along x; the dominant one
    is fitted.

    The light above the grating is not one wave: the guided light that reaches the end of
    the grating is partly reflected, and the backward guided wave also scatters upwards,
    with k_x of the opposite sign. Their beating makes |E|^2 far from exponential (deep
    fringes, about lambda / (2 sin theta) apart). So the complex field between
    ``edge_periods`` periods from each end is fitted to

        E(x) = e^(-alpha x) sum_m A_m e^(i (k_x + m K) x) + e^(-alpha (L - x)) sum_m B_m e^(i (-k_x + m K) x)

    (m = -1, 0, 1: the near-field harmonics of the teeth, K = 2 pi / period), linear in
    A_m, B_m; alpha and k_x by a grid search and refinement. ``residual`` is the part of
    the field this model leaves unexplained (|E - fit|^2 / |E|^2): large when another
    diffraction order also radiates."""
    x = np.asarray(x_um, float)
    comps = {k: np.asarray(v, complex).ravel() for k, v in e_fields.items() if v is not None}
    main = max(comps.values(), key=lambda v: float(np.sum(np.abs(v) ** 2)))
    L = float(length) if length is not None else float(x.max() - 1.0)
    sel = (x >= edge_periods * period) & (x <= L - edge_periods * period)
    if sel.sum() < 20:
        raise ValueError("too few points on the grating to fit the field")
    xs, Es = x[sel], main[sel]
    k0 = 2 * math.pi / wl
    # starting k_x: the strongest spatial frequency of the radiated light (|k_x| < k0)
    n_fft = 1 << int(np.ceil(np.log2(len(xs) * 8)))
    dx = float(np.median(np.diff(xs)))
    spec = np.abs(np.fft.fft(Es * np.hanning(len(xs)), n_fft))
    kk = 2 * math.pi * np.fft.fftfreq(n_fft, dx)
    inside = np.abs(kk) < k0
    kx0 = float(kk[inside][np.argmax(spec[inside])])
    sub = slice(None, None, max(1, len(xs) // 400))               # coarse search on fewer points
    best = None
    for a in np.linspace(0.0, 0.4, 81):
        for kx in np.linspace(kx0 - 0.3, kx0 + 0.3, 31):
            v, _ = _two_wave(xs[sub], Es[sub], period, L, a, kx)
            if best is None or v < best[0]:
                best = (v, a, kx)
    v, a, kx = _two_wave(xs, Es, period, L, best[1], best[2])[0], best[1], best[2]
    da, dk = 0.005, 0.02
    for _ in range(4):                                            # refine on all points
        for a2 in np.linspace(max(a - da, 0.0), a + da, 11):
            for k2 in np.linspace(kx - dk, kx + dk, 11):
                v2, _ = _two_wave(xs, Es, period, L, a2, k2)
                if v2 < v:
                    v, a, kx = v2, a2, k2
        da, dk = da / 5, dk / 5
    v, c = _two_wave(xs, Es, period, L, a, kx)
    amp = np.abs(c).reshape(-1, 2)                                # rows m = -1, 0, 1; fwd, bwd
    alpha = max(float(a), 1e-6)
    n_bloch = (kx_sign * kx + 2 * math.pi / period) / k0
    return dict(alpha_per_um=alpha, kx_per_um=float(kx), n_bloch=float(n_bloch),
                decay_length_um=float(1 / (2 * alpha)), window_um=float(xs[-1] - xs[0]),
                residual=float(v), backward_ratio=float(amp[1, 1] / max(amp[1, 0], 1e-30)))


def second_order_up(n_bloch: float, period: float, wl: float, n_top: float,
                    margin: float = 0.1) -> bool:
    """True when the second diffraction order (k_x = beta - 2 K) also radiates into the top
    cladding, |n_B - 2 wl / period| < n_top, or is evanescent but so close to radiating
    (within ``margin``) that it has not died out on the line 1.5 um above the grating.
    Two waves above the grating spoil the phase (Bloch index) and decay measured there."""
    return abs(n_bloch - 2 * wl / period) < n_top + margin


def validate(rows: list, wl: float, n_top: float, n_low: float, n_high: float,
             min_drop: float = 0.2, max_residual: float = 0.2) -> list:
    """Mark each measured fill factor ``ok`` (with ``why`` when not). A point is rejected when
    the decay is too weak to measure over the grating (intensity drop exp(-2 alpha L) below
    ``min_drop``), the forward + backward wave model leaves more than ``max_residual`` of the
    field unexplained, the Bloch index lies outside the two film indices, or the second
    diffraction order also radiates upward."""
    for r in rows:
        why = []
        if 2 * r["alpha_per_um"] * r.get("window_um", 0.0) < min_drop:
            why.append("decay too weak to measure")
        if r.get("residual", 0.0) > max_residual:
            why.append(f"field not two waves ({r['residual'] * 100:.0f} % unexplained)")
        if not (n_low < r["n_bloch"] < n_high):
            why.append("Bloch index outside the film indices")
        if second_order_up(r["n_bloch"], r["period_run_um"], wl, n_top):
            why.append("2nd order radiates up")
        r["ok"] = not why
        r["why"] = "; ".join(why)
    return rows


def weak_branch(rows: list) -> list:
    """The usable fill factors: valid points from the strongest scatterer up to the largest
    fill factor, keeping alpha falling and the Bloch index rising with fill factor (one
    monotonic branch, so each alpha has exactly one fill factor). Narrow trenches (large
    fill factor) give the weak scattering the start of the grating needs."""
    good = sorted([r for r in rows if r.get("ok", True)], key=lambda r: r["fill_factor"])
    if len(good) < 2:
        return good
    k = int(np.argmax([r["alpha_per_um"] for r in good]))
    branch = [good[k]]
    for r in good[k + 1:]:
        if r["alpha_per_um"] < branch[-1]["alpha_per_um"] and r["n_bloch"] > branch[-1]["n_bloch"]:
            branch.append(r)
    for r in rows:
        r["on_branch"] = any(r is b for b in branch)
    return branch


# ------------------------------------------------------------------ design
def beam_profile(x, x_c: float, w_x: float):
    """Field of the fibre beam along the grating surface (Gaussian, footprint w_x)."""
    return np.exp(-((np.asarray(x) - x_c) / w_x) ** 2)


def emission(x, alpha_of_x):
    """Emitted intensity S(x) = 2 alpha P(x), dP/dx = -2 alpha P, P(0) = 1, on grid x."""
    x = np.asarray(x, float)
    a = np.asarray(alpha_of_x, float)
    dx = np.diff(x, prepend=x[0])
    cum = np.cumsum(2 * a * dx)
    P = np.exp(-cum)
    return 2 * a * P, P


def overlap(x, S, x_c: float, w_x: float) -> float:
    """Coupling of an emitted intensity S(x) (flat phase, total = power scattered) to the
    fibre beam G(x) = exp(-((x - x_c)/w_x)^2), relative to the power launched:
    |int sqrt(S) G dx|^2 / int G^2 dx, the Gaussian normalised over ALL x (w_x sqrt(pi/2)),
    so the part of the beam that misses the grating counts as lost."""
    G = beam_profile(x, x_c, w_x)
    return float(np.trapezoid(np.sqrt(np.clip(S, 0, None)) * G, x) ** 2 / (w_x * math.sqrt(math.pi / 2)))


def target_alpha(x, x_c, w_x, A, a_min, a_max):
    """alpha(x) that emits A G^2(x), clipped to [a_min, a_max]."""
    G2 = beam_profile(x, x_c, w_x) ** 2
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (G2[1:] + G2[:-1]) * np.diff(x))])
    norm = w_x * math.sqrt(math.pi / 2)            # integral of G^2 over all x
    rem = np.clip(1 - A * cum / norm, 1e-6, None)
    return np.clip(A * (G2 / norm) / (2 * rem), a_min, a_max)


def best_profile(a_min: float, a_max: float, w_x: float, length: Optional[float] = None) -> dict:
    """Beam centre x_c and fraction A that maximise the 1D model's coupling with alpha in
    [a_min, a_max]; also the best uniform grating (constant alpha) for comparison."""
    length = length or 4 * w_x
    x = np.linspace(0, length, 4001)
    best = None
    for x_c in np.linspace(0.5 * w_x, 2.5 * w_x, 41):
        for A in (0.80, 0.88, 0.94, 0.97, 0.99):
            a = target_alpha(x, x_c, w_x, A, a_min, a_max)
            S, _ = emission(x, a)
            eta = overlap(x, S, x_c, w_x)
            if best is None or eta > best["eta_model"]:
                best = dict(x_c=float(x_c), A=A, eta_model=eta)
    uni = None
    for a in np.geomspace(max(a_min, 1e-4), a_max, 40):
        S, _ = emission(x, np.full_like(x, a))
        for x_c in np.linspace(0.2 * w_x, 2.5 * w_x, 47):
            eta = overlap(x, S, x_c, w_x)
            if uni is None or eta > uni["eta_model"]:
                uni = dict(alpha=float(a), x_c=float(x_c), eta_model=eta)
    best["uniform_best"] = uni
    return best


def build_teeth(alpha_ff, period_ff, ffs, x_c, w_x, A, a_min, a_max, max_teeth=80,
                end_power=0.01) -> tuple[list, list]:
    """Tooth list ((x_start, length), ...) and the fill factor of each tooth: march along
    the grating; each tooth's fill
    factor gives the alpha the profile needs there (interpolated, on the monotonic branch),
    and its period phase-matches that fill factor."""
    ffs = np.asarray(ffs, float)
    al = np.asarray([alpha_ff(f) for f in ffs])
    if len(al) > 1 and not (np.all(np.diff(al) < 0) or np.all(np.diff(al) > 0)):
        raise ValueError("alpha must change monotonically with fill factor (use weak_branch)")
    o = np.argsort(al)
    al_s, ff_s = al[o], ffs[o]
    teeth, ff_list = [], []
    x, P_rem = 0.0, 1.0
    norm = w_x * math.sqrt(math.pi / 2)
    P_typ = period_ff(float(np.median(ffs)))
    for _ in range(max_teeth):
        # emit A G^2 per unit length from the power still guided: alpha = A G^2 / (2 P_rem)
        xm = x + P_typ / 2
        want = A * beam_profile(xm, x_c, w_x) ** 2 / norm / (2 * max(P_rem, 1e-6))
        want = float(np.clip(want, a_min, a_max))
        ff = float(np.interp(want, al_s, ff_s))
        P = period_ff(ff)
        teeth.append((round(x, 4), round(ff * P, 4)))
        ff_list.append(ff)
        P_rem *= math.exp(-2 * alpha_ff(ff) * P)
        x += P
        if (P_rem < end_power and x > x_c + w_x) or x > x_c + 2.5 * w_x:   # beam has ended
            break
    return teeth, ff_list


def teeth_model(teeth, alpha_ff_of_length, x_c, w_x, dx=0.01):
    """1D model of an explicit tooth list: piecewise-constant alpha per period."""
    xs, al = [], []
    for i, (x0, ln) in enumerate(teeth):
        x1 = teeth[i + 1][0] if i + 1 < len(teeth) else x0 + (x0 - teeth[i - 1][0] if i else ln)
        seg = np.arange(x0, x1, dx)
        xs.append(seg)
        al.append(np.full_like(seg, alpha_ff_of_length(i)))
    x = np.concatenate(xs)
    a = np.concatenate(al)
    S, P = emission(x, a)
    return dict(x=x, S=S, remaining=float(P[-1]), eta_model=overlap(x, S, x_c, w_x))


# ------------------------------------------------------------------ the design stage
def read_line(data, f0: float):
    """x and the E components on the 'above' line monitor at the frequency nearest f0."""
    mon = data["above"]
    comps, x = {}, None
    for c in ("Ex", "Ey", "Ez"):
        arr = getattr(mon, c, None)
        if arr is None:
            continue
        a = arr.sel(f=f0, method="nearest")
        comps[c] = np.squeeze(a.values)
        x = np.asarray(a.coords["x"].values, float)
    return x, comps


def _form(ctx) -> None:
    from tarang.design.ui import parse_list
    on = lambda c: bool(c.apodise.run)                      # noqa: E731
    ctx.ui.form(ctx.cfg, "Apodised grating", [
        dict(section="apodise", key="run"),
        dict(section="apodise", key="fill_factors", cast=parse_list, show=on),
        dict(section="apodise", key="min_trench_um", show=on),
        dict(section="apodise", key="verify", show=on),
        dict(section="apodise", key="max_teeth", show=on),
        dict(section="apodise", key="prefer", show=on,
             check=lambda v: "" if v in ("efficiency", "bandwidth") else "answer efficiency or bandwidth"),
        dict(section="apodise", key="ramp_check", show=on),
        dict(section="apodise", key="reflection_check", show=on),
        dict(section="apodise", key="use_in_layout", show=on)])


def default_fill_factors(period: float, min_feature: float) -> list[float]:
    hi = 1 - min_feature / period
    ffs = [round(f, 3) for f in np.arange(0.5, hi - 1e-9, 0.05)] + [round(hi - 1e-3, 3)]
    return sorted(set(ffs))


def stage_apodise(ctx) -> dict:
    """Measure alpha(ff) and the Bloch index (small 2D runs), design the apodised tooth list
    (free 1D model), then check it in 2D FDTD (priced)."""
    from tarang.design import physics, visuals
    from tarang.design.checks import _load, _run_batch, _write_csv
    from tarang.design.stages import grating_device
    from tarang.grating_coupler.grating_coupler import phase_matched_period, spectrum_summary
    _form(ctx)
    cfg, ui = ctx.cfg, ctx.ui
    if not cfg.apodise.run:
        ui.say("Apodised grating not requested (apodise.run = false): the layout uses the uniform "
               "grating. Turn it on in this table (or design.toml) whenever you want it; nothing "
               "else is repeated.")
        return dict(disabled=True)
    gc = ctx.state["stages"].get("grating", {}).get("results") or {}
    if not gc or gc.get("skipped"):
        ui.say("The grating stage has no design yet; run it first.")
        return dict(skipped=True)
    wl = cfg.run.wavelength_um
    f0 = C_UM / wl
    H = ctx.results("cross_section")["height_um"]
    P0, ff0, depth = gc["period_um"], gc["fill_factor"], gc["etch_depth_um"]
    angle, tilt = gc["fiber_angle_deg"], gc.get("fiber_tilt", "towards")
    n_top = 1.0 if str(cfg.materials.cladding).lower() == "air" else ctx.n(cfg.materials.cladding)
    n_core, n_box = ctx.n(cfg.materials.core), ctx.n(cfg.materials.box)
    n_slab = physics.slab_te_index(n_core, n_box, n_top, H, wl)
    n_groove = n_top if gc.get("etch") == "full" else physics.slab_te_index(n_core, n_box, n_top,
                                                                              H - depth, wl)

    def p_model(ff):
        return phase_matched_period(ff * n_slab + (1 - ff) * n_groove, wl, angle, tilt, n_top)
    scale = P0 / p_model(ff0)                     # calibrated on the final uniform design
    # full etch: a trench narrower than the etch can clear (aspect-ratio-dependent etching)
    # is not etched through, so the weakest usable tooth is set by the narrowest trench
    trench = float(cfg.apodise.min_trench_um or cfg.grating.min_feature_um)
    ffs = [float(f) for f in (cfg.apodise.fill_factors or
                              default_fill_factors(P0, max(trench, float(cfg.grating.min_feature_um))))]
    # no runs for fill factors whose trench is too narrow (the final design's is always run:
    # it fixes the sign convention of the phase)
    ffs = [f for f in ffs if (1 - f) * p_model(f) * scale >= trench - 1e-6 or abs(f - ff0) < 1e-9]
    if ff0 not in ffs:
        ffs = sorted(ffs + [ff0])
    folder = ctx.out / "cloud" / "apodise"
    sims, lengths = {}, {}
    for f in ffs:
        P = round(p_model(f) * scale, 4)
        sim, dev = uniform_reverse_sim(ctx, f, P)
        sims[f] = (sim, f"uniform_reverse_ff{f:.3f}_p{P:.4f}", P)
        lengths[f] = P * dev.config.n_teeth
    ui.say(f"Narrowest trench allowed: {trench * 1e3:.0f} nm (apodise.min_trench_um; empty: "
           f"grating.min_feature_um), i.e. fill factor up to about {1 - trench / P0:.3f} at the "
           f"{P0:g} µm period.")
    ui.say(f"Scattering strength of uniform gratings at {len(ffs)} fill factors "
           f"({', '.join(f'{f:g}' for f in ffs)}): TE0 launched from the waveguide, field recorded "
           "1.5 µm above the grating.")
    ok = _run_batch(ctx, {f"FF {f:g}": (s, n) for f, (s, n, _) in sims.items()}, folder,
                    "apodisation: scattering strength per fill factor")
    if not ok:
        return dict(skipped=True)
    rows = []
    for f, (sim, name, P) in sims.items():
        data = _load(sim, folder, name)
        if data is None:
            continue
        try:
            x, comps = read_line(data, f0)
            r = alpha_and_index(x, comps, P, wl, length=lengths[f])
        except Exception as err:                                  # noqa: BLE001
            ui.say(f"  (FF {f:g}: not analysed: {type(err).__name__}: {err})")
            continue
        r.update(fill_factor=f, period_run_um=P)
        rows.append(r)
    if len(rows) < 2:
        ui.say("  Fewer than two fill factors analysed; no apodised design.")
        return dict(skipped=True, scattering=rows)
    rows.sort(key=lambda r: r["fill_factor"])
    # the final uniform design is phase-matched to the fibre: its emitted wave travels along
    # +x for a beam 'towards' the waveguide (k_x = +k0 sin theta), along -x for 'away'. That
    # fixes the sign convention of the measured phase slopes for every fill factor.
    ref = min(rows, key=lambda r: abs(r["fill_factor"] - ff0))
    want = 1.0 if tilt == "towards" else -1.0
    conv = want * (1.0 if ref["kx_per_um"] >= 0 else -1.0)
    k0 = 2 * math.pi / wl
    for r in rows:
        r["n_bloch"] = (conv * r["kx_per_um"] + 2 * math.pi / r["period_run_um"]) / k0
    validate(rows, wl, n_top, n_groove, n_slab)
    for r in rows:
        r["trench_um"] = (1 - r["fill_factor"]) * r["period_run_um"]
        if r["trench_um"] < trench - 1e-6:
            r["ok"] = False
            r["why"] = "; ".join(v for v in (r["why"], f"trench {r['trench_um'] * 1e3:.0f} nm < "
                                                       f"{trench * 1e3:.0f} nm") if v)
    branch = weak_branch(rows)
    if not any(abs(r["fill_factor"] - ff0) < 1e-9 for r in branch):
        ui.say(f"  The final design's fill factor {ff0:g} is not on the usable branch (see why); no "
               "apodised design.")
        branch = []
    # smooth Bloch index along the branch (quadratic in ff with 4+ points), then the period
    fb = np.array([r["fill_factor"] for r in branch])
    nb = np.array([r["n_bloch"] for r in branch])
    deg = 2 if len(fb) >= 4 else 1
    nfit = np.polyfit(fb, nb, deg) if len(fb) >= 2 else None
    for r in rows:
        n_use = float(np.polyval(nfit, r["fill_factor"])) if (nfit is not None and r.get("on_branch")) \
            else r["n_bloch"]
        r["period_matched_um"] = phase_matched_period(n_use, wl, angle, tilt, n_top)
    ui.table(rows, ["fill_factor", "trench_um", "alpha_per_um", "decay_length_um", "n_bloch",
                    "period_matched_um", "on_branch", "why"],
             {"trench_um": ".3f", "alpha_per_um": ".4f", "decay_length_um": ".2f", "n_bloch": ".4f",
              "period_matched_um": ".4f"})
    _write_csv(ctx.path("tables", "apodise_scattering.csv"), rows)
    res = dict(scattering=rows, branch=[r["fill_factor"] for r in branch])
    if len(branch) < 3:
        ui.say("  Fewer than three usable fill factors on one branch (alpha falling, Bloch index rising "
               "with fill factor); no apodised design. Add fill factors (apodise.fill_factors) "
               "between the strongest scatterer and the largest fill factor.")
        res.update(skipped=True, verified=False)
        return res
    ui.say(f"  Usable branch: fill factors {fb.min():g}-{fb.max():g} (alpha falls and the Bloch "
           "index rises with fill factor). Points off it are listed with the reason.")
    fgrid = fb
    agrid = np.array([r["alpha_per_um"] for r in branch])
    pgrid = np.array([r["period_matched_um"] for r in branch])
    # the measured periods carry the same small model error at every ff: re-anchor them on
    # the final uniform design, whose peak is on the design wavelength
    p_at_ff0 = float(np.interp(ff0, fgrid, pgrid))
    pgrid = pgrid * (P0 / p_at_ff0)

    def alpha_ff(f):
        return float(np.interp(f, fgrid, agrid))

    def period_ff(f):
        return float(np.interp(f, fgrid, pgrid))
    w_x = cfg.grating.mfd_um / 2 / math.cos(math.radians(angle))
    a_min, a_max = float(agrid.min()), float(agrid.max())
    best = best_profile(a_min, a_max, w_x)
    a0 = alpha_ff(ff0)
    x = np.linspace(0, 4 * w_x, 4001)
    S0, _ = emission(x, np.full_like(x, a0))
    xc_grid = np.linspace(0, 3 * w_x, 121)
    uni_eta = max(overlap(x, S0, xc, w_x) for xc in xc_grid)
    teeth, ffs_t = build_teeth(alpha_ff, period_ff, fgrid, best["x_c"], w_x, best["A"], a_min,
                               a_max, max_teeth=int(cfg.apodise.max_teeth))
    model = teeth_model(teeth, lambda i: alpha_ff(ffs_t[i]), best["x_c"], w_x)
    ui.say(f"  1D emission model (fibre footprint {w_x:.2f} µm): this uniform grating (α "
           f"{a0:.3f}/µm) overlaps the fibre by {uni_eta * 100:.0f} %; with the scattering shaped "
           f"along the grating ({len(teeth)} teeth, fill factor {min(ffs_t):.2f}-{max(ffs_t):.2f}, "
           f"α {a_min:.3f}-{a_max:.3f}/µm): {model['eta_model'] * 100:.0f} %.")
    res.update(alpha_range_per_um=[a_min, a_max], footprint_um=w_x,
               uniform_alpha_per_um=a0, model_uniform=uni_eta, model_apodised=model["eta_model"],
               x_centre_um=best["x_c"], fraction_A=best["A"], teeth=teeth,
               fill_factors_along=ffs_t, period_scale_to_final=P0 / p_at_ff0)
    trows = []
    for i, ((a, b), f) in enumerate(zip(teeth, ffs_t)):
        per = (teeth[i + 1][0] - a) if i + 1 < len(teeth) else period_ff(f)
        trows.append(dict(tooth=i, x_start_um=a, length_um=b, trench_um=round(per - b, 4),
                          fill_factor=round(f, 4), period_um=round(per, 4)))
    _write_csv(ctx.path("tables", "apodised_teeth.csv"), trows)
    res.update(narrowest_trench_um=min(r["trench_um"] for r in trows),
               widest_trench_um=max(r["trench_um"] for r in trows), min_trench_um=trench)
    ui.say(f"  Trenches of the apodised grating: {res['narrowest_trench_um'] * 1e3:.0f}-"
           f"{res['widest_trench_um'] * 1e3:.0f} nm (all etched through the film).")
    visuals.safe(ctx, "apodisation design", draw_design, ctx, rows, teeth, ffs_t, model, best,
                 w_x, a0, uni_eta)
    if model["eta_model"] < uni_eta + 0.02:
        ui.say("  The fill factors available here cannot scatter weakly enough to shape the beam "
               "(the weakest tooth already scatters strongly), so apodisation gains little; "
               "not checked in FDTD. The narrowest trench (apodise.min_trench_um) sets the weakest "
               "tooth: narrower trenches scatter more weakly, if the etch clears them.")
        res["verified"] = False
        return res
    if not cfg.apodise.verify:
        return res
    # ---- verification in 2D FDTD, two rounds. Round 1: the period scale around the design
    # (0.98, 1, 1.02) at the model's fibre position, and the fibre moved by -1/+1 um at scale 1;
    # the peak wavelength against scale gives the scale that puts the peak on the design
    # wavelength. Round 2: that scale at three fibre positions.
    gfolder = ctx.out / "cloud" / "grating"

    def device(s_, sx):
        tl = [(round(a * s_, 4), round(b * s_, 4)) for a, b in teeth]
        return grating_device(ctx, P0, ff0, round(sx, 2), H=H, depth=depth, teeth=tl)

    def run_round(keys, label):
        devs = {k: device(*k) for k in keys}
        ok = _run_batch(ctx, {f"scale {k[0]:g}, fibre {k[1]:.2f} µm": (d.sim, d._readable_name())
                              for k, d in devs.items()}, gfolder, label)
        out = {}
        if not ok:
            return out
        for k, d in devs.items():
            try:
                wls, eta = d.get_spectrum(gfolder)
            except Exception:                                     # noqa: BLE001
                continue
            out[k] = (wls, eta, spectrum_summary(wls, eta, wl))
        return out

    xc = best["x_c"]
    r1 = run_round([(0.98, 0.98 * xc), (1.0, xc), (1.02, 1.02 * xc), (1.0, xc - 1.0), (1.0, xc + 1.0)],
                   "apodised grating in 2D FDTD (round 1: period scale)")
    if not r1:
        return res
    centre = [(k[0], v[2]["peak_wavelength_um"]) for k, v in r1.items()
              if abs(k[1] - k[0] * xc) < 1e-6]
    results = dict(r1)
    if len(centre) >= 2:
        sc, pk = np.array(centre).T
        slope, icpt = np.polyfit(sc, pk, 1)
        s_star = float(np.clip((wl - icpt) / slope, 0.9, 1.1)) if slope > 0 else 1.0
        s_star = round(s_star, 4)
        res["scale_centred"] = s_star
        ui.say(f"  Peak wavelength moves {slope * 1e3 / 100:.2f} nm per 1 % of period; the peak sits on "
               f"{wl * 1e3:g} nm at period scale {s_star:.4f}.")
        if all(abs(s_star - k[0]) > 0.002 for k in r1):
            r2 = run_round([(s_star, s_star * xc + d) for d in (-1.0, 0.0, 1.0)],
                           "apodised grating in 2D FDTD (round 2: peak centred)")
            results.update(r2)
    vrows, spectra = [], {}
    for (s_, sx), (wls, eta, sm) in results.items():
        spectra[(s_, sx)] = (wls, eta)
        vrows.append(dict(scale=s_, source_x_um=round(sx, 2),
                          efficiency_at_target=sm["efficiency_at_target"],
                          peak_efficiency=sm["peak_efficiency"],
                          peak_wavelength_nm=round(sm["peak_wavelength_um"] * 1e3, 1),
                          bandwidth_1db_nm=round(sm["bandwidth_1db_nm"], 1)))
    ui.table(sorted(vrows, key=lambda r: -r["efficiency_at_target"]),
             formats={"efficiency_at_target": ".3f", "peak_efficiency": ".3f"})
    _write_csv(ctx.path("tables", "apodised_check.csv"), vrows)
    top = max(vrows, key=lambda r: r["efficiency_at_target"])
    key = next(k for k in results if k[0] == top["scale"] and round(k[1], 2) == top["source_x_um"])
    cands = [dict(recipe="measured scattering", eff=top["efficiency_at_target"],
                  bw=top["bandwidth_1db_nm"], peak=top["peak_efficiency"],
                  peak_nm=top["peak_wavelength_nm"], source_x_um=round(key[1], 2), scale=key[0],
                  teeth=[(round(a * key[0], 4), round(b * key[0], 4)) for a, b in teeth],
                  spectrum=spectra[key])]
    s_ramp = res.get("scale_centred", 1.0)
    if cfg.apodise.ramp_check:
        ramp = ramp_check(ctx, fgrid, period_ff, s_ramp, best["x_c"], P0, ff0, H, depth, gfolder)
        if ramp:
            res["ramp"] = {k: v for k, v in ramp.items() if k != "spectra"}
            b = ramp["best"]
            cands.append(dict(recipe=f"linear ramp R = {b['R_per_um']:g}/µm", eff=b["efficiency_at_target"],
                              bw=b["bandwidth_1db_nm"], peak=b["peak_efficiency"],
                              peak_nm=b["peak_wavelength_nm"], source_x_um=b["source_x_um"],
                              scale=s_ramp, teeth=b["teeth"], spectrum=ramp["spectra"][b["key"]]))
    eff_u = gc["efficiency_2d"]
    bw_u = gc.get("bandwidth_1db_nm") or 0.0
    prefer = str(cfg.apodise.prefer).lower()
    chosen, better = choose(cands, eff_u, bw_u, prefer)
    res.update(verified=True, check=vrows, efficiency_2d=chosen["eff"], recipe=chosen["recipe"],
               peak_efficiency=chosen["peak"], peak_wavelength_nm=chosen["peak_nm"],
               bandwidth_1db_nm=chosen["bw"], source_x_um=chosen["source_x_um"],
               scale=chosen["scale"], teeth=chosen["teeth"], uniform_efficiency_2d=eff_u,
               uniform_bandwidth_1db_nm=bw_u, prefer=prefer, better_than_uniform=better,
               candidates=[{k: v for k, v in c.items() if k not in ("teeth", "spectrum")} for c in cands],
               recommend=dict(choices=[{k: v for k, v in c.items() if k not in ("teeth", "spectrum")}
                                       for c in cands],
                              recommended=dict(recipe=chosen["recipe"], use_in_layout=better),
                              why=("higher efficiency at the design wavelength" if prefer == "efficiency"
                                   else "wider 1 dB band, at most 0.5 dB below the uniform grating")))
    from tarang.design.focus import _write_spectrum
    _write_spectrum(ctx.path("tables", "apodised_spectrum.csv"), *chosen["spectrum"])
    if cfg.apodise.reflection_check:
        rb = reflection(ctx, chosen, P0, ff0, H, depth, gfolder)
        if rb:
            res["power_budget"] = rb
            t = rb["at_target"]
            ui.say(f"  Back-reflection of the kept apodised grating into the waveguide at {wl} µm: "
                   f"{t['reflection_db']:.1f} dB (uniform grating: see the checks stage); "
                   f"{t['up'] * 100:.0f} % up, {t['down'] * 100:.0f} % into the substrate.")
    why = ("higher efficiency at the design wavelength" if prefer == "efficiency" else
           "a wider 1 dB band with at most 0.5 dB less at the design wavelength")
    ui.say(f"  Apodised ({chosen['recipe']}): {chosen['eff'] * 100:.1f} % at {wl} µm "
           f"({-10 * math.log10(max(chosen['eff'], 1e-9)):.2f} dB), peak at {chosen['peak_nm']} nm, "
           f"1 dB band {chosen['bw']:.0f} nm; uniform: {eff_u * 100:.1f} %, {bw_u:.0f} nm. "
           + (f"The layout uses it ({why}; apodise.prefer = '{prefer}')." if better else
              f"The layout keeps the uniform grating (apodise.prefer = '{prefer}': {why})."))
    try:
        uni_dev = grating_device(ctx, P0, ff0, gc["source_x_um"], H=H, depth=depth)
        uni = uni_dev.get_spectrum(gfolder)
    except Exception:                                             # noqa: BLE001
        uni = None
    top_row = dict(efficiency_at_target=chosen["eff"])
    visuals.safe(ctx, "apodised spectrum", draw_check, ctx, chosen["spectrum"], uni, top_row, eff_u)
    from tarang.design import landscapes
    visuals.safe(ctx, "apodised landscape", landscapes.draw_apodised,
                 ctx.path("figures", "9c_apodised_landscape.png"), vrows, (round(key[1], 2), key[0]), wl)
    return res


def reflection(ctx, chosen: dict, P0, ff0, H, depth, folder) -> Optional[dict]:
    """The kept apodised grating run backwards (TE0 from the waveguide, closed box of flux
    planes, as the checks stage does for the uniform grating): where the light goes and the
    reflection into the guided mode against wavelength. One small 2D run, priced first."""
    from tarang.design.checks import _load, _power_budget, _reverse_sim, _run_batch
    from tarang.design.stages import grating_device
    dev = grating_device(ctx, P0, ff0, round(chosen["source_x_um"], 2), H=H, depth=depth,
                         teeth=[tuple(t) for t in chosen["teeth"]])
    rev = _reverse_sim(dev, ctx)
    name = dev._readable_name() + "_reverse"
    if not _run_batch(ctx, {"reverse": (rev, name)}, folder,
                      "apodised grating: back-reflection (reverse run, 2D)"):
        return None
    data = _load(rev, folder, name)
    if data is None:
        return None
    try:
        b = _power_budget(data, rev, {"final": chosen["spectrum"]}, ctx.cfg.run.wavelength_um)
    except Exception as err:                                      # noqa: BLE001
        ctx.ui.say(f"  (reflection of the apodised grating not read: {type(err).__name__}: {err})")
        return None
    return b


def choose(cands: list, eff_u: float, bw_u: float, prefer: str) -> tuple[dict, bool]:
    """The apodised design to keep and whether it beats the uniform grating: 'efficiency' =
    highest efficiency at the design wavelength; 'bandwidth' = widest 1 dB band among
    designs at most 0.5 dB below the uniform grating at the design wavelength."""
    floor = eff_u * 10 ** (-0.05)
    if prefer == "bandwidth":
        ok = [c for c in cands if c["eff"] >= floor]
        if ok:
            c = max(ok, key=lambda c: (c["bw"], c["eff"]))
            return c, c["bw"] > bw_u
        return max(cands, key=lambda c: c["eff"]), False
    c = max(cands, key=lambda c: c["eff"])
    return c, c["eff"] > eff_u


RAMP_RATES = (0.004, 0.008, 0.012, 0.016, 0.024)      # 1/um


def ramp_teeth(f0: float, f_min: float, rate: float, n: int, period_ff, scale: float = 1.0) -> list:
    """Linear fill-factor ramp F(x) = F0 - R x (Marchetti et al., Sci. Rep. 7, 16670 (2017)),
    clipped to [f_min, F0]; every tooth gets the phase-matched period of its fill factor."""
    teeth, x = [], 0.0
    for _ in range(int(n)):
        f = min(max(f0 - rate * x, f_min), f0)
        p = period_ff(f) * scale
        teeth.append((round(x, 4), round(f * p, 4)))
        x += p
    return teeth


def ramp_check(ctx, fgrid, period_ff, scale, x_c, P0, ff0, H, depth, folder) -> Optional[dict]:
    """Second recipe: linear ramps from the weakest usable tooth (largest fill factor), at
    five ramp rates and three fibre positions, in 2D FDTD (priced). Figure 9d."""
    from tarang.design import visuals
    from tarang.design.checks import _run_batch, _write_csv
    from tarang.design.stages import grating_device
    from tarang.grating_coupler.grating_coupler import spectrum_summary
    cfg, ui = ctx.cfg, ctx.ui
    n = int(ctx.results("grating")["n_teeth"])
    f0, f_min = float(np.max(fgrid)), float(np.min(fgrid))
    devs = {}
    for R in RAMP_RATES:
        tl = ramp_teeth(f0, f_min, R, n, period_ff, scale)
        for dx in (-1.5, 0.0, 1.5):
            sx = round(x_c + dx, 2)
            devs[(R, sx)] = (grating_device(ctx, P0, ff0, sx, H=H, depth=depth, teeth=tl), tl)
    ui.say(f"  Linear ramp (Marchetti et al. 2017): fill factor from {f0:g} falling at "
           f"{', '.join(f'{r:g}' for r in RAMP_RATES)} per µm (down to {f_min:g}), {n} teeth, fibre at "
           f"{x_c - 1.5:.1f}/{x_c:.1f}/{x_c + 1.5:.1f} µm.")
    ok = _run_batch(ctx, {f"ramp {k[0]:g}/µm, fibre {k[1]:g} µm": (d.sim, d._readable_name())
                          for k, (d, _) in devs.items()}, folder, "apodised grating: linear ramps (2D)")
    if not ok:
        return None
    rows, spectra = [], {}
    for (R, sx), (d, tl) in devs.items():
        try:
            wls, eta = d.get_spectrum(folder)
        except Exception:                                         # noqa: BLE001
            continue
        sm = spectrum_summary(wls, eta, cfg.run.wavelength_um)
        spectra[(R, sx)] = (wls, eta)
        rows.append(dict(R_per_um=R, source_x_um=sx, efficiency_at_target=sm["efficiency_at_target"],
                         peak_efficiency=sm["peak_efficiency"],
                         peak_wavelength_nm=round(sm["peak_wavelength_um"] * 1e3, 1),
                         bandwidth_1db_nm=round(sm["bandwidth_1db_nm"], 1)))
    if not rows:
        return None
    ui.table(sorted(rows, key=lambda r: -r["efficiency_at_target"]),
             formats={"efficiency_at_target": ".3f", "peak_efficiency": ".3f"})
    _write_csv(ctx.path("tables", "apodised_ramp.csv"), rows)
    b = max(rows, key=lambda r: r["efficiency_at_target"])
    key = (b["R_per_um"], b["source_x_um"])
    best = dict(b, key=key, teeth=devs[key][1])
    from tarang.design import landscapes
    visuals.safe(ctx, "ramp landscape", landscapes.draw, ctx.path("figures", "9d_apodised_ramp_landscape.png"),
                 [dict(points=[dict(x=r["source_x_um"], y=r["R_per_um"], z=r["efficiency_at_target"])
                               for r in rows],
                       title=f"Efficiency at {cfg.run.wavelength_um} µm", xlabel="Fibre position on the grating (µm)",
                       ylabel="Ramp rate R (1/µm)", chosen=(key[1], key[0]), chosen_label="best ramp",
                       note="F(x) = F0 − R x, each tooth phase-matched; 2D FDTD"),
                  dict(points=[dict(x=r["source_x_um"], y=r["R_per_um"], z=r["peak_efficiency"])
                               for r in rows],
                       title="Peak efficiency (any wavelength)", xlabel="Fibre position on the grating (µm)",
                       ylabel="Ramp rate R (1/µm)", chosen=(key[1], key[0]), chosen_label="best ramp")],
                 "Apodised grating, linear fill-factor ramp: efficiency landscape")
    return dict(rows=rows, best=best, spectra=spectra, f0=f0, f_min=f_min, rates=list(RAMP_RATES))


def draw_design(ctx, rows, teeth, ffs_t, model, best, w_x, a0, uni_eta):
    """Figure 9a: measured alpha(ff) and period(ff); the tooth fill factors along the grating;
    the emitted profile of the uniform and apodised designs against the fibre beam."""
    from tarang.design import visuals
    plt = visuals._plt()
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    ff = np.array([r["fill_factor"] for r in rows])
    on = np.array([bool(r.get("on_branch", True)) for r in rows])
    al = np.array([r["alpha_per_um"] for r in rows])
    pm = np.array([r["period_matched_um"] for r in rows])
    ax = axes[0]
    ax.plot(ff, al, ":", color=visuals.PALETTE[0], lw=1)
    ax.plot(ff[on], al[on], "o-", color=visuals.PALETTE[0], label="α (field, per µm), used")
    if (~on).any():
        ax.plot(ff[~on], al[~on], "o", mfc="white", color=visuals.PALETTE[0],
                label="α, not used (see the table)")
    ax.set(xlabel="fill factor (tooth / period)", ylabel="scattering strength α (1/µm)",
           title="uniform gratings (2D FDTD, reverse runs)")
    tw = ax.twinx()
    tw.plot(ff[on], pm[on], "s--", color=visuals.PALETTE[1], label="phase-matched period (used)")
    tw.set_ylabel("phase-matched period (µm)")
    ax.grid(alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = tw.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center")
    ax = axes[1]
    xs = [a + b / 2 for a, b in teeth]
    ax.step(xs, ffs_t, where="mid", color=visuals.PALETTE[2])
    ax.plot(xs, ffs_t, ".", color=visuals.PALETTE[2])
    ax.set(xlabel="position along the grating (µm)", ylabel="fill factor of each tooth",
           title=f"apodised grating: {len(teeth)} teeth")
    ax.grid(alpha=0.3)
    ax = axes[2]
    x = model["x"]
    G2 = beam_profile(x, best["x_c"], w_x) ** 2
    S0, _ = emission(x, np.full_like(x, a0))
    ax.plot(x, G2 / G2.max(), "k--", lw=1.5, label="fibre beam |G|² (target)")
    ax.plot(x, S0 / S0.max(), color="0.55", lw=1.5, label=f"uniform: overlap {uni_eta * 100:.0f} %")
    ax.plot(x, model["S"] / model["S"].max(), color=visuals.PALETTE[1], lw=1.8,
            label=f"apodised: overlap {model['eta_model'] * 100:.0f} %")
    ax.set(xlabel="position along the grating (µm)", ylabel="emitted intensity (normalised)",
           title="1D emission model")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle("Apodisation: scattering strength shaped so the emitted light matches the fibre mode")
    return visuals._save(fig, ctx.path("figures", "9a_apodise_design.png"))


def draw_check(ctx, apod, uniform, top, eff_u):
    """Figure 9b: spectra of the apodised (best of the check) and the uniform grating."""
    from tarang.design import visuals
    plt = visuals._plt()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    wl0 = ctx.cfg.run.wavelength_um
    for (w, e), lab, col in ((apod, f"apodised: {top['efficiency_at_target'] * 100:.1f} % at {wl0} µm",
                              visuals.PALETTE[1]),
                             (uniform, f"uniform: {eff_u * 100:.1f} %", "0.4")):
        if w is None:
            continue
        ax.plot(np.asarray(w) * 1e3, 10 * np.log10(np.clip(e, 1e-6, None)), lw=2, color=col, label=lab)
    ax.axvline(wl0 * 1e3, color="0.6", ls=":")
    ax.set(xlabel="Wavelength (nm)", ylabel="Coupling efficiency (dB)",
           title="Grating coupler (2D FDTD): apodised against uniform")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return visuals._save(fig, ctx.path("figures", "9b_apodised_spectrum.png"))
