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


def alpha_and_index(x_um, e_fields: dict, period: float, wl: float, skip_periods: int = 3,
                    floor: float = 1e-3, kx_sign: float = 1.0) -> dict:
    """alpha (field decay, 1/um) and Bloch index from the field on the line above a
    uniform grating. ``e_fields``: complex arrays of the field components along x. The
    dominant component's phase gives k_x; |E|^2 averaged over each period gives the decay."""
    x = np.asarray(x_um, float)
    comps = {k: np.asarray(v, complex).ravel() for k, v in e_fields.items() if v is not None}
    e2 = sum(np.abs(v) ** 2 for v in comps.values())
    main = max(comps.values(), key=lambda v: float(np.sum(np.abs(v) ** 2)))
    # per-period averages, starting a few periods in (the source end is not yet periodic)
    x0, x1 = x.min(), x.max()
    edges = np.arange(x0 + skip_periods * period, x1 - period, period)
    centres, means = [], []
    for a in edges:
        m = (x >= a) & (x < a + period)
        if m.sum() >= 2:
            centres.append(a + period / 2)
            means.append(float(e2[m].mean()))
    centres, means = np.array(centres), np.array(means)
    ok = means > floor * means.max() if len(means) else np.array([], bool)
    if ok.sum() < 3:
        raise ValueError("too few periods above the noise floor to fit the decay")
    slope, icpt = np.polyfit(centres[ok], np.log(means[ok]), 1)
    alpha = max(-slope / 2, 1e-6)
    fit_rms = float(np.sqrt(np.mean((np.log(means[ok]) - (slope * centres[ok] + icpt)) ** 2)))
    # phase slope over the same region (unwrapped), weighted by intensity
    sel = (x >= centres[ok][0] - period / 2) & (x <= centres[ok][-1] + period / 2)
    ph = np.unwrap(np.angle(main[sel]))
    w = np.abs(main[sel])
    kx, c0 = np.polyfit(x[sel], ph, 1, w=w)
    kx = float(kx)
    res_ph = ph - (kx * x[sel] + c0)
    phase_rms = float(np.sqrt(np.sum(w ** 2 * res_ph ** 2) / max(np.sum(w ** 2), 1e-30)))
    window = float(centres[ok][-1] - centres[ok][0] + period)
    k0 = 2 * math.pi / wl
    # kx_sign converts the measured phase slope to the physical k_x along +x (the guided
    # light's direction); it depends on the solver's time convention, see stage_apodise
    n_bloch = (kx_sign * kx + 2 * math.pi / period) / k0
    return dict(alpha_per_um=float(alpha), kx_per_um=kx, n_bloch=float(n_bloch),
                decay_length_um=float(1 / (2 * alpha)), periods_fitted=int(ok.sum()),
                window_um=window, fit_rms=fit_rms, phase_rms_rad=phase_rms)


def second_order_up(n_bloch: float, period: float, wl: float, n_top: float,
                    margin: float = 0.1) -> bool:
    """True when the second diffraction order (k_x = beta - 2 K) also radiates into the top
    cladding, |n_B - 2 wl / period| < n_top, or is evanescent but so close to radiating
    (within ``margin``) that it has not died out on the line 1.5 um above the grating.
    Two waves above the grating spoil the phase (Bloch index) and decay measured there."""
    return abs(n_bloch - 2 * wl / period) < n_top + margin


def validate(rows: list, wl: float, n_top: float, n_low: float, n_high: float,
             min_drop: float = 0.2, max_phase_rms: float = 0.6, max_fit_rms: float = 0.35) -> list:
    """Mark each measured fill factor ``ok`` (with ``why`` when not). A point is rejected when
    the decay is too weak to measure over the grating (intensity drop exp(-2 alpha L) below
    ``min_drop``), the log-intensity or phase fit is poor, the Bloch index lies outside the
    two film indices, or the second diffraction order also radiates upward."""
    for r in rows:
        why = []
        if 2 * r["alpha_per_um"] * r.get("window_um", 0.0) < min_drop:
            why.append("decay too weak to measure")
        if r.get("fit_rms", 0.0) > max_fit_rms:
            why.append("intensity not exponential")
        if r.get("phase_rms_rad", 0.0) > max_phase_rms:
            why.append("phase not linear")
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
        dict(section="apodise", key="use_in_layout", show=on),
        dict(section="run", key="max_cost_per_run", show=on),
        dict(section="run", key="max_total_cost", show=on)])


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
    sims = {}
    for f in ffs:
        P = round(p_model(f) * scale, 4)
        sim, dev = uniform_reverse_sim(ctx, f, P)
        sims[f] = (sim, f"uniform_reverse_ff{f:.3f}_p{P:.4f}", P)
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
            r = alpha_and_index(x, comps, P, wl)
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
    final_teeth = [(round(a * key[0], 4), round(b * key[0], 4)) for a, b in teeth]
    eff_u = gc["efficiency_2d"]
    res.update(verified=True, check=vrows, efficiency_2d=top["efficiency_at_target"],
               peak_efficiency=top["peak_efficiency"], peak_wavelength_nm=top["peak_wavelength_nm"],
               bandwidth_1db_nm=top["bandwidth_1db_nm"], source_x_um=round(key[1], 2), scale=key[0],
               teeth=final_teeth, uniform_efficiency_2d=eff_u,
               better_than_uniform=top["efficiency_at_target"] > eff_u)
    ui.say(f"  Apodised: {top['efficiency_at_target'] * 100:.1f} % at {wl} µm "
           f"({-10 * math.log10(max(top['efficiency_at_target'], 1e-9)):.2f} dB), peak at "
           f"{top['peak_wavelength_nm']} nm; uniform: {eff_u * 100:.1f} %."
           + ("" if res["better_than_uniform"] else " The layout keeps the uniform grating."))
    try:
        uni_dev = grating_device(ctx, P0, ff0, gc["source_x_um"], H=H, depth=depth)
        uni = uni_dev.get_spectrum(gfolder)
    except Exception:                                             # noqa: BLE001
        uni = None
    visuals.safe(ctx, "apodised spectrum", draw_check, ctx, spectra[key], uni, top, eff_u)
    return res


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
