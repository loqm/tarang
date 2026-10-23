"""Whole-device circuit model: fibre-to-fibre transmission of grating + taper + bus + ring +
taper + grating, every resonance in the band, from the stages' results (free, local).

Why a circuit model
-------------------
The device is millimetres long and its resonances are picometres wide: one FDTD run of
all of it would need ~10^11 cells and ~10^7 periods. Instead each part is simulated once
on its own (the earlier stages) and described by its scattering matrix S(omega), the
2 x 2 matrix of amplitudes into and out of its two ports. Joining the parts in a chain is
then exact for linear, single-mode light (the same idea as Lumerical INTERCONNECT, the
Ansys/Tidy3D "photonic circuit" tools, or the KLayout/SiEPIC compact models).

From first principles
---------------------
* Two 2-ports A then B joined at one port: the light bounces between them, and summing
  all the round trips (a geometric series) gives the Redheffer star product
      D   = 1 - A22 B11
      S11 = A11 + A12 B11 A21 / D        S21 = B21 A21 / D
      S22 = B22 + B21 A22 B12 / D        S12 = A12 B12 / D
  (``star``). The chain fibre -> grating -> bus -> ring -> bus -> grating -> fibre is built
  with it, so the Fabry-Perot cavity formed by the two gratings' back-reflections (the
  ripple seen on every measured spectrum) comes out by itself.
* A straight waveguide of length L: t = exp(i beta L - alpha L / 2), beta = 2 pi n_eff / lambda,
  alpha the power loss per metre.
* The ring on its bus (all-pass): with self-coupling r = sqrt(1 - kappa^2), round-trip
  amplitude a and round-trip phase phi,
      t = (r - a e^{i phi}) / (1 - r a e^{i phi}),
      |t|^2 = (a^2 - 2 r a cos phi + r^2) / (1 - 2 r a cos phi + r^2 a^2).
  Resonances sit at phi = 2 pi m. phi(omega) is taken from the ring stage's resonance list
  (omega_m of every mode m, from the broadband TE00 effective index, the same list the
  LLE files hold), so the resonance positions carry the full dispersion D_int(mu).
* Around one resonance: FWHM in phase 2 (1 - r a) / sqrt(r a), and d phi / d omega =
  2 pi / D1 (D1 the local FSR in rad/s), so Q_L = omega D1^-1 2 pi sqrt(r a) / (2 (1 - r a)).
  With r = 1 this is Q_i, with a = 1 it is Q_c; K = Q_i / Q_c; the dip on resonance is
  T_min = (r - a)^2 / (1 - r a)^2 (zero at critical coupling, r = a).

What feeds the model (all saved by earlier stages)
-------------------------------------------------
* kappa^2(lambda) at the critical and over-coupled gaps: the checks stage's band curves
  (3D FDTD of the coupler); without them, the value at the design wavelength for all.
* a: the ring's loss (dB/cm, scattering + absorption + bend) over 2 pi R.
* Coupler efficiency eta(lambda): the 2D grating spectrum used by the layout (uniform or
  apodised), times the fibre's lateral overlap and the taper; for the focusing grating,
  its 3D FDTD spectrum (if run) times its tip taper, else the 2D spectrum times the
  in-plane focusing model.
* Grating reflection R(lambda): the checks stage's reverse run of the uniform grating
  (an approximation for the apodised or focusing grating). Its phase is not known, so the
  ripple's position is arbitrary; its depth and period are not.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Optional

import numpy as np

C = 299792458.0


# ------------------------------------------------------------------ S-matrix algebra
def star(A: tuple, B: tuple) -> tuple:
    """Redheffer star product of two 2-ports (S11, S12, S21, S22), element-wise arrays."""
    A11, A12, A21, A22 = A
    B11, B12, B21, B22 = B
    D = 1.0 - A22 * B11
    return (A11 + A12 * B11 * A21 / D, A12 * B12 / D, B21 * A21 / D, B22 + B21 * A22 * B12 / D)


def two_port(t, r1=0.0, r2=0.0) -> tuple:
    """A reciprocal 2-port: transmission t both ways, reflections r1 (port 1), r2 (port 2)."""
    t = np.asarray(t, complex)
    z = np.zeros_like(t)
    return (z + r1, t, t, z + r2)


def all_pass(phi, r, a):
    """Amplitude transmission of a ring on a bus (all-pass)."""
    e = a * np.exp(1j * np.asarray(phi))
    return (r - e) / (1 - r * e)


def resonance_numbers(D1: float, r: float, a: float, omega: float) -> dict:
    """Q_i, Q_c, Q_L, K and dip of one resonance (D1: local FSR in rad/s)."""
    def q(x):
        x = min(max(x, 1e-15), 1 - 1e-15)
        return omega * 2 * math.pi * math.sqrt(x) / (D1 * 2 * (1 - x))
    qi, qc, ql = q(a), q(r), q(r * a)
    tmin = (r - a) ** 2 / (1 - r * a) ** 2
    return dict(Q_i=qi, Q_c=qc, Q_L=ql, K=qi / qc, T_min=tmin,
                dip_db=-10 * math.log10(max(tmin, 1e-6)), linewidth_Hz=omega / ql / (2 * math.pi))


def chain(f_Hz, eta_in, eta_out, refl, n_eff, L_bus_m: float, alpha_per_m: float, t_ring):
    """Fibre-to-fibre amplitude: grating, bus (with the ring halfway), grating."""
    f = np.asarray(f_Hz, float)
    beta = 2 * np.pi * f * np.asarray(n_eff) / C
    half = np.exp(1j * beta * L_bus_m / 2 - alpha_per_m * L_bus_m / 4)
    rg = np.sqrt(np.clip(refl, 0, 1))
    g_in = two_port(np.sqrt(np.clip(eta_in, 0, 1)), 0.0, rg)
    g_out = two_port(np.sqrt(np.clip(eta_out, 0, 1)), rg, 0.0)
    S = star(star(star(star(g_in, two_port(half)), two_port(t_ring)), two_port(half)), g_out)
    return S[2]


# ------------------------------------------------------------------ inputs from the stages
def interp_eta(wl_um, wl_data, eta_data):
    """Coupler efficiency between the simulated wavelengths, interpolated in log(eta): the
    tails of a coupler spectrum fall roughly exponentially, and a linear interpolation would
    draw V-shaped notches in dB between two simulated points."""
    w = np.asarray(wl_data, float)
    o = np.argsort(w)
    lg = np.log(np.clip(np.asarray(eta_data, float)[o], 1e-12, None))
    return np.exp(np.interp(wl_um, w[o], lg))


def coupling_kind(ctx) -> str:
    from tarang.design.stages import coupling
    return coupling(ctx.cfg)


def _read_spectrum(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    wl = np.array([float(r["wavelength_um"]) for r in rows])
    e = np.array([float(r["efficiency"]) for r in rows])
    o = np.argsort(wl)
    return wl[o], e[o]


def coupler_model(ctx) -> Optional[dict]:
    """eta(lambda) of one coupler (grating + whatever joins it to the bus), its label and the
    length of waveguide it adds; None if the grating stage has not run."""
    st = ctx.state["stages"]
    cfg = ctx.cfg
    from tarang.design.stages import coupling
    if coupling(cfg) == "edge":
        from tarang.design import edge
        return edge.coupler(ctx)
    gc = (st.get("grating") or {}).get("results") or {}
    if not gc or gc.get("skipped"):
        return None
    tables = ctx.out / "tables"
    tp = (st.get("taper") or {}).get("results") or {}
    fc = (st.get("focus") or {}).get("results") or {}
    from tarang.design.layout import coupler_plan
    plan = coupler_plan(ctx)
    use_ap = plan["main"] == "apodised"
    use_fc = plan["main"] == "focusing"
    uniform = _read_spectrum(tables / "grating_spectrum_final.csv")
    apod = _read_spectrum(tables / "apodised_spectrum.csv")
    notes = []
    if use_fc:
        s3 = _read_spectrum(tables / "focusing_spectrum_3d.csv")
        if s3 is not None:
            # the 3D run launches into the tip taper and records the 1 um waveguide behind it:
            # the tip taper is already in the spectrum
            return dict(wl=s3[0], eta=s3[1], label="focusing grating with its tip taper (3D FDTD)",
                        length_um=float(fc.get("tip_taper_um") or 0.0), notes=notes)
        shape = apod if (fc.get("teeth_kind") == "apodised" and apod is not None) else uniform
        if shape is None:
            return None
        return dict(wl=shape[0], eta=shape[1] * float(fc.get("eta_lateral") or 1.0),
                    label=f"focusing grating ({fc.get('teeth_kind', 'uniform')} teeth, 2D x in-plane model)",
                    length_um=float(fc.get("tip_taper_um") or 0.0), notes=notes)
    shape = apod if (use_ap and apod is not None) else uniform
    if shape is None:
        return None
    if use_ap and apod is None:
        notes.append("apodised spectrum not saved (rerun the apodise stage, free from the cache): "
                     "the uniform grating's spectrum is used")
    lat = float(tp.get("eta_y") or 1.0) if tp and not tp.get("skipped") else 1.0
    tl = tp.get("taper_loss_db")
    if tl is None:
        tl = tp.get("model_loss_db_at_previous") or 0.0
    length = tp.get("taper_length_um") or cfg.grating.taper_length_um
    label = ("apodised" if shape is apod else "uniform") + " grating (2D) x lateral overlap x taper"
    return dict(wl=shape[0], eta=shape[1] * lat * 10 ** (-float(tl) / 10), label=label,
                length_um=float(length), notes=notes)


def coupler_reflection(ctx) -> tuple[dict, Optional[str]]:
    """Reverse-run power budget of the grating the main coupler uses: the apodised
    grating's own (apodise.reflection_check) when its teeth are used, else the uniform
    grating's (checks stage). Returns (budget, where it came from)."""
    from tarang.design.layout import coupler_plan
    from tarang.design.stages import coupling
    if coupling(ctx.cfg) == "edge":
        cp = coupler_model(ctx)
        if cp is None or cp.get("refl") is None:
            return {}, None
        return dict(wavelength_um=list(map(float, cp["refl_wl"])),
                    reflected=list(map(float, cp["refl"]))), cp["refl_source"]
    st = ctx.state["stages"]
    plan = coupler_plan(ctx)
    ap = (st.get("apodise") or {}).get("results") or {}
    fc = (st.get("focus") or {}).get("results") or {}
    apod_teeth = plan["main"] == "apodised" or (plan["main"] == "focusing"
                                                and fc.get("teeth_kind") == "apodised")
    if apod_teeth and (ap.get("power_budget") or {}).get("reflected"):
        return ap["power_budget"], "the apodised grating's reverse run"
    gc_res = ((st.get("checks") or {}).get("results") or {}).get("grating") or {}
    b = gc_res.get("power_budget") or {}
    if b.get("reflected"):
        return b, ("the uniform grating's reverse run" + (" (the apodised grating's own was not "
                   "simulated: apodise.reflection_check)" if apod_teeth else ""))
    return {}, None


def coupling_curves(ctx) -> dict:
    """kappa^2(lambda) of each gap regime: {name: (gap_um, wl_um array or None, k2 array)}."""
    st = ctx.state["stages"]
    gap = (st.get("gap") or {}).get("results") or {}
    if not gap or gap.get("skipped") or not gap.get("target_cross_power"):
        return {}
    rho = float(gap["target_cross_power"])
    K = gap.get("over_coupling")
    band = (((st.get("checks") or {}).get("results") or {}).get("band") or {}).get("curves") or {}
    out = {}
    for name, g, k2 in (("critical", gap.get("critical_gap_um"), rho),
                        (f"over-coupled K = {K:g}" if K else None, gap.get("over_coupled_gap_um"),
                         min((K or 1) * rho, 0.999))):
        if not name or g is None:
            continue
        c = band.get(name)
        if c and len(c.get("wavelength_um", [])) > 1:
            out[name] = (float(g), np.array(c["wavelength_um"], float), np.array(c["coupled"], float))
        else:
            out[name] = (float(g), None, np.array([k2]))
    return out


def _k2_at(curve, wl_um):
    _, w, k = curve
    if w is None:
        return np.full_like(np.asarray(wl_um, float), float(k[0]))
    o = np.argsort(w)
    return np.interp(wl_um, w[o], k[o])


# ------------------------------------------------------------------ the model
def model(ctx) -> Optional[dict]:
    """Every number of the device model (no files, no figures)."""
    from tarang.design.lle import _broadband_record
    from tarang.waveguide.dispersion_mapping.broadband import analyse
    st = ctx.state["stages"]
    cfg = ctx.cfg
    xs = (st.get("cross_section") or {}).get("results")
    ring = (st.get("ring") or {}).get("results")
    if not xs or not ring:
        return None
    H, W, R = xs["height_um"], xs["width_um"], ring["radius_um"]
    wl0 = cfg.run.wavelength_um
    rec = _broadband_record(ctx, H, W)
    if rec is None:
        return None
    A = analyse(dict(rec, pump=wl0), R)
    om = np.asarray(A["om_mu"], float)
    M = A["m0"] + np.asarray(A["mus"], int)
    o = np.argsort(om)
    om, M, mus = om[o], M[o], np.asarray(A["mus"], int)[o]
    Dint = np.asarray(A["Dint"], float)[o]
    cp = coupler_model(ctx)
    curves = coupling_curves(ctx)
    notes = list(cp["notes"]) if cp else []
    loss_ring = float(ring["loss_db_cm"])
    L_ring = 2 * math.pi * R * 1e-6
    a = 10 ** (-loss_ring * L_ring * 100 / 20)
    straight = float(ring["scattering_loss_db_cm"]) + float(ring["absorption_loss_db_cm"])
    alpha = straight * math.log(10) / 10 * 100                  # power, per metre
    from tarang.design.layout import bus_length_um
    L_bus = (bus_length_um(ctx) + 2 * (cp["length_um"] if cp else 0.0)) * 1e-6
    # resonances inside the coupler's band
    wl_res = 2 * math.pi * C / om * 1e6
    if cp is not None:
        lo, hi = float(cp["wl"].min()), float(cp["wl"].max())
        keep = (wl_res >= lo) & (wl_res <= hi)
        if keep.sum() >= 3:
            om, M, mus, Dint, wl_res = om[keep], M[keep], mus[keep], Dint[keep], wl_res[keep]
    D1_loc = np.gradient(om, M.astype(float))                   # rad/s per mode
    budget, refl_src = coupler_reflection(ctx)
    edge_c = coupling_kind(ctx) == "edge"
    if refl_src:
        notes.append(f"{'facet' if edge_c else 'grating'} reflection from {refl_src}")
    use_refl = bool(cfg.device.reflections) and bool(budget.get("reflected"))
    if cfg.device.reflections and not budget.get("reflected"):
        notes.append("no grating reflection saved (checks stage, grating extras): no Fabry-Perot ripple")
    n_wf = (np.asarray(A["wf"]), np.asarray(A["nf"]))

    def n_eff(wl_um):
        return np.interp(wl_um, *n_wf)

    def eta(wl_um):
        if cp is None:
            return np.ones_like(np.asarray(wl_um, float))
        return interp_eta(wl_um, cp["wl"], cp["eta"])

    def refl(wl_um):
        if not use_refl:
            return np.zeros_like(np.asarray(wl_um, float))
        w = np.asarray(budget["wavelength_um"], float)
        oo = np.argsort(w)
        return np.interp(wl_um, w[oo], np.asarray(budget["reflected"], float)[oo])

    def phase(omega):
        return 2 * np.pi * np.interp(omega, om, M.astype(float))

    regimes = {}
    for name, curve in curves.items():
        k2 = np.clip(_k2_at(curve, wl_res), 1e-12, 1 - 1e-12)
        r = np.sqrt(1 - k2)
        rows = []
        for i in range(len(om)):
            n = resonance_numbers(float(D1_loc[i]), float(r[i]), a, float(om[i]))
            off = float(eta(wl_res[i]) ** 2 * math.exp(-alpha * L_bus))
            rows.append(dict(regime=name, mu=int(mus[i]), mode_number=int(M[i]),
                             wavelength_nm=round(float(wl_res[i]) * 1e3, 4),
                             frequency_THz=round(float(om[i]) / (2 * math.pi) / 1e12, 6),
                             Dint_2pi_MHz=round(float(Dint[i]) / (2 * math.pi) / 1e6, 3),
                             kappa2=float(k2[i]), Q_i=n["Q_i"], Q_c=n["Q_c"], Q_L=n["Q_L"],
                             K=round(n["K"], 4), linewidth_MHz=round(n["linewidth_Hz"] / 1e6, 3),
                             dip_db=round(n["dip_db"], 2),
                             off_resonance_db=round(10 * math.log10(max(off, 1e-12)), 3),
                             on_resonance_db=round(10 * math.log10(max(off * n["T_min"], 1e-15)), 3)))
        regimes[name] = dict(gap_um=curve[0], band_curve=curve[1] is not None, r=r, rows=rows)
    if curves and not all(v["band_curve"] for v in regimes.values()):
        notes.append("coupling taken as constant over the band for regimes without a checks-stage "
                     "band curve (checks.band)")
    if not curves:
        notes.append("gap stage not run: no ring coupling, transmission of the bus only")
    return dict(om=om, M=M, mus=mus, Dint=Dint, wl_res=wl_res, D1=D1_loc, a=a, alpha=alpha,
                L_bus=L_bus, L_ring=L_ring, loss_ring=loss_ring, n_eff=n_eff, eta=eta, refl=refl,
                phase=phase, regimes=regimes, coupler=cp, notes=notes, use_refl=use_refl,
                n_g=float(ring["n_group"]), R_um=R, wl0=wl0, m0=int(A["m0"]))


def grid(md: dict, regime: Optional[str], f_lo: float, f_hi: float, dense: bool = True) -> np.ndarray:
    """Frequencies (Hz): fine enough for the Fabry-Perot ripple everywhere, and 61 points
    across +-8 linewidths of every resonance."""
    fsr_fp = C / (2 * md["n_g"] * max(md["L_bus"], 1e-6))
    fsr_ring = float(np.median(md["D1"])) / (2 * math.pi)
    step = min(fsr_fp / 14, fsr_ring / 40) if dense else fsr_ring / 40
    n = int(min(max((f_hi - f_lo) / step, 200), 400000))
    parts = [np.linspace(f_lo, f_hi, n)]
    if regime and regime in md["regimes"]:
        u = np.tan(np.linspace(-1.45, 1.45, 61)) / math.tan(1.45)
        for row in md["regimes"][regime]["rows"]:
            f0, lw = row["frequency_THz"] * 1e12, row["linewidth_MHz"] * 1e6
            if f_lo <= f0 <= f_hi:
                parts.append(f0 + 8 * lw * u)
    f = np.unique(np.concatenate(parts))
    return f[(f >= f_lo) & (f <= f_hi)]


def transmission(md: dict, regime: Optional[str], f: np.ndarray) -> np.ndarray:
    """Fibre-to-fibre power transmission at frequencies f (Hz)."""
    wl = C / f * 1e6
    if regime and regime in md["regimes"]:
        rr = md["regimes"][regime]
        k2 = np.clip(np.interp(f, md["om"] / (2 * math.pi), [x["kappa2"] for x in rr["rows"]]), 1e-12, 1 - 1e-12)
        t_ring = all_pass(md["phase"](2 * math.pi * f), np.sqrt(1 - k2), md["a"])
    else:
        t_ring = np.ones_like(f, complex)
    e = md["eta"](wl)
    t = chain(f, e, e, md["refl"](wl), md["n_eff"](wl), md["L_bus"], md["alpha"], t_ring)
    return np.abs(t) ** 2


def envelope(md: dict, wl_um: np.ndarray) -> dict:
    """Off-resonance fibre-to-fibre transmission: no ripple, ripple maxima and minima."""
    e = md["eta"](wl_um)
    p = math.exp(-md["alpha"] * md["L_bus"])
    Rr = md["refl"](wl_um) * p
    base = e ** 2 * p
    return dict(mean=base, hi=base / (1 - Rr) ** 2, lo=base / (1 + Rr) ** 2,
                ripple_db=20 * np.log10((1 + Rr) / np.clip(1 - Rr, 1e-12, None)))


# ------------------------------------------------------------------ the stage
def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main_regime(md: dict) -> Optional[str]:
    names = list(md["regimes"])
    over = [n for n in names if n.startswith("over")]
    return over[0] if over else (names[0] if names else None)


def stage_device(ctx) -> dict:
    from tarang.design import visuals
    cfg, ui = ctx.cfg, ctx.ui
    ring_goal = str(getattr(cfg.run, "goal", "comb_ring")) == "comb_ring"
    ui.form(cfg, "Whole device (circuit model, free)", [
        dict(section="device", key="run"),
        dict(section="device", key="reflections", show=lambda c: c.device.run),
        dict(section="device", key="zoom_fsr", show=lambda c: c.device.run and ring_goal)])
    if not cfg.device.run:
        ui.say("  Device model not run (device.run = false).")
        return dict(skipped=True)
    if not ring_goal:
        return cutback_device(ctx)
    md = model(ctx)
    if md is None:
        ui.say("  Needs the cross-section and ring stages (broadband solve); not run.")
        return dict(skipped=True)
    cp = md["coupler"]
    ui.say(f"Circuit model: fibre -> {cp['label'] if cp else 'ideal coupler'} -> "
           f"{md['L_bus'] * 1e6:.0f} µm of waveguide with the ring halfway -> the same coupler -> "
           f"fibre; {len(md['om'])} resonances from {md['wl_res'].min() * 1e3:.0f} to "
           f"{md['wl_res'].max() * 1e3:.0f} nm; ring round trip a = {md['a']:.5f} "
           f"({md['loss_ring']:.3g} dB/cm over {md['L_ring'] * 1e6:.0f} µm).")
    for n in md["notes"]:
        ui.say("  Note: " + n)
    rows_all, summary = [], {}
    i0 = int(np.argmin(np.abs(md["mus"])))
    for name, rr in md["regimes"].items():
        rows_all += rr["rows"]
        p = rr["rows"][i0]
        Q = [x["Q_L"] for x in rr["rows"]]
        dips = [x["dip_db"] for x in rr["rows"]]
        summary[name] = dict(gap_um=rr["gap_um"], Q_L_pump=p["Q_L"], Q_i_pump=p["Q_i"], Q_c_pump=p["Q_c"],
                             K_pump=p["K"], dip_db_pump=p["dip_db"], linewidth_MHz_pump=p["linewidth_MHz"],
                             Q_L_range=[min(Q), max(Q)], dip_db_range=[min(dips), max(dips)],
                             band_curve=rr["band_curve"])
    if summary:
        ui.table([dict(regime=k, gap_um=v["gap_um"], Q_L=v["Q_L_pump"], K=v["K_pump"],
                       dip_db=v["dip_db_pump"], linewidth_MHz=v["linewidth_MHz_pump"],
                       Q_L_band=f"{v['Q_L_range'][0]:.3g}-{v['Q_L_range'][1]:.3g}")
                  for k, v in summary.items()],
                 formats={"gap_um": ".3f", "Q_L": ".3g"})
    env0 = envelope(md, np.array([md["wl0"]]))
    off0 = float(10 * np.log10(max(env0["mean"][0], 1e-12)))
    rip0 = float(env0["ripple_db"][0])
    ui.say(f"  At {md['wl0']} µm, off resonance: {off0:.2f} dB fibre to fibre"
           + (f", Fabry-Perot ripple {rip0:.2f} dB peak to peak (period "
              f"{(md['wl0'] * 1e-6) ** 2 / (2 * md['n_g'] * md['L_bus']) * 1e12:.0f} pm)" if md["use_refl"] else "")
           + ".")
    _write_csv(ctx.path("tables", "device_resonances.csv"), rows_all)
    reg = main_regime(md)
    f_p = float(md["om"][i0]) / (2 * math.pi)
    fsr = float(md["D1"][i0]) / (2 * math.pi)
    zf = grid(md, reg, f_p - cfg.device.zoom_fsr * fsr, f_p + cfg.device.zoom_fsr * fsr)
    spec_rows = []
    zoom = {}
    for name in (md["regimes"] or {None: None}):
        T = transmission(md, name, zf)
        zoom[name] = T
        spec_rows += [dict(regime=name or "bus only", frequency_THz=round(float(x) / 1e12, 9),
                           wavelength_nm=round(C / float(x) * 1e9, 6), transmission=float(t),
                           transmission_db=round(10 * math.log10(max(float(t), 1e-15)), 4))
                      for x, t in zip(zf, T)]
    _write_csv(ctx.path("tables", "device_spectrum.csv"), spec_rows)
    wl_env = np.linspace(md["wl_res"].min(), md["wl_res"].max(), 1500)
    env = envelope(md, wl_env)
    _write_csv(ctx.path("tables", "device_envelope.csv"),
               [dict(wavelength_nm=round(float(w) * 1e3, 4),
                     off_resonance_db=round(10 * math.log10(max(float(m), 1e-12)), 4),
                     ripple_max_db=round(10 * math.log10(max(float(h), 1e-12)), 4),
                     ripple_min_db=round(10 * math.log10(max(float(lo_), 1e-12)), 4))
                for w, m, h, lo_ in zip(wl_env, env["mean"], env["hi"], env["lo"])])
    full_f = grid(md, reg, float(md["om"].min()) / (2 * math.pi) - fsr / 2,
                  float(md["om"].max()) / (2 * math.pi) + fsr / 2)
    full_T = transmission(md, reg, full_f)
    fig = visuals.safe(ctx, "device spectrum", draw_spectrum, ctx, md, reg, full_f, full_T, zf, zoom,
                       wl_env, env)
    fig2 = visuals.safe(ctx, "device resonances", draw_resonances, ctx, md)
    return dict(coupler=cp["label"] if cp else None, coupler_length_um=cp["length_um"] if cp else None,
                bus_length_total_um=md["L_bus"] * 1e6, round_trip_a=md["a"], resonances=len(md["om"]),
                band_nm=[float(md["wl_res"].min() * 1e3), float(md["wl_res"].max() * 1e3)],
                off_resonance_db_at_pump=off0, ripple_db_at_pump=rip0 if md["use_refl"] else None,
                reflections=md["use_refl"], regimes=summary, main_regime=reg, notes=md["notes"],
                tables=["tables/device_resonances.csv", "tables/device_spectrum.csv",
                        "tables/device_envelope.csv"],
                figures=[str(p) for p in (fig, fig2) if p])


# ------------------------------------------------------------------ figures
def draw_spectrum(ctx, md, reg, full_f, full_T, zf, zoom, wl_env, env):
    """Figure 11a: fibre-to-fibre spectrum over the band, and a close-up at the pump."""
    from tarang.design import visuals
    plt = visuals._plt()
    fig = plt.figure(figsize=(11, 7.6))
    ax = fig.add_subplot(2, 1, 1)
    wl = C / full_f * 1e9
    ax.plot(wl, 10 * np.log10(np.clip(full_T, 1e-15, None)), lw=0.4, color=visuals.PALETTE[0],
            rasterized=True, label=f"fibre to fibre ({reg or 'bus only'})")
    ax.plot(wl_env * 1e3, 10 * np.log10(np.clip(env["mean"], 1e-12, None)), lw=1.4,
            color=visuals.PALETTE[1], label="off resonance, no ripple")
    cp = md.get("coupler")
    if cp:
        ax.plot(np.asarray(cp["wl"]) * 1e3, 10 * np.log10(np.clip(cp["eta"], 1e-12, None)), "o--",
                ms=2.5, lw=0.8, color="0.45", label="one coupler (simulated points)")
    ax.axvline(md["wl0"] * 1e3, color="0.5", ls=":", lw=1)
    ax.set(xlabel="Wavelength (nm)", ylabel="Transmission (dB)",
           title=f"Whole device (circuit model): {len(md['om'])} resonances")
    lo = float(np.nanmin(10 * np.log10(np.clip(full_T, 1e-15, None))))
    ax.set_ylim(max(lo - 2, -60), None)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower center")
    ax2 = fig.add_subplot(2, 2, 3)
    for k, (name, T) in enumerate(zoom.items()):
        ax2.plot(C / zf * 1e9, 10 * np.log10(np.clip(T, 1e-15, None)), lw=1,
                 color=visuals.PALETTE[k % len(visuals.PALETTE)], label=name or "bus only")
    ax2.set(xlabel="Wavelength (nm)", ylabel="Transmission (dB)", title="Close-up around the pump")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7.5)
    ax3 = fig.add_subplot(2, 2, 4)
    if reg:
        rr = md["regimes"][reg]["rows"]
        f0 = np.array([r["frequency_THz"] for r in rr])
        i = int(np.argmin(np.abs(f0 - md["om"][int(np.argmin(np.abs(md["mus"])))] / 2e12 / math.pi)))
        r0 = rr[i]
        f = r0["frequency_THz"] * 1e12 + np.linspace(-6, 6, 801) * r0["linewidth_MHz"] * 1e6
        T = transmission(md, reg, f)
        ax3.plot((f - r0["frequency_THz"] * 1e12) / 1e6, T / T.max(), color=visuals.PALETTE[0])
        ax3.set(xlabel="Detuning from the pump resonance (MHz)", ylabel="Transmission (normalised)",
                title=f"Pump resonance: Q_L = {r0['Q_L']:.3g}, dip {r0['dip_db']:.1f} dB")
        ax3.grid(alpha=0.3)
    return visuals._save(fig, ctx.path("figures", "11a_device_spectrum.png"))


def draw_resonances(ctx, md):
    """Figure 11b: Q_L, Q_i, Q_c and the dip depth of every resonance."""
    from tarang.design import visuals
    if not md["regimes"]:
        return None
    plt = visuals._plt()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.4))
    first = True
    for k, (name, rr) in enumerate(md["regimes"].items()):
        rows = rr["rows"]
        wl = [r["wavelength_nm"] for r in rows]
        col = visuals.PALETTE[k % len(visuals.PALETTE)]
        ax.semilogy(wl, [r["Q_L"] for r in rows], "-", color=col, label=f"Q_L, {name}")
        ax.semilogy(wl, [r["Q_c"] for r in rows], "--", color=col, label=f"Q_c, {name}")
        if first:
            ax.semilogy(wl, [r["Q_i"] for r in rows], ":", color="0.3", label="Q_i")
            first = False
        bx.plot(wl, [r["dip_db"] for r in rows], "-", color=col, label=name)
    for a_ in (ax, bx):
        a_.axvline(md["wl0"] * 1e3, color="0.6", ls=":")
        a_.grid(alpha=0.3)
        a_.legend(fontsize=7.5)
    ax.set(xlabel="Wavelength (nm)", ylabel="Quality factor", title="Q of every resonance")
    bx.set(xlabel="Wavelength (nm)", ylabel="Resonance dip (dB)",
           title="Dip depth (deepest where Q_c = Q_i)")
    return visuals._save(fig, ctx.path("figures", "11b_device_resonances.png"))


# ------------------------------------------------------------------ test waveguides
def cutback_device(ctx) -> dict:
    """Expected measurement of the cutback set (run.goal = loss_test): each structure is
    coupler + waveguide of length L + coupler, T(dB) = 2 eta_dB - alpha L, with the coupling
    scatter (waveguides.coupler_scatter_db) as error bars and the Fabry-Perot ripple of the
    two couplers' reflections (cavity = the whole structure)."""
    from tarang.design import visuals
    cfg, ui = ctx.cfg, ctx.ui
    st = ctx.state["stages"]
    wg = (st.get("waveguides") or {}).get("results") or {}
    xs = (st.get("cross_section") or {}).get("results") or {}
    if not wg or not xs:
        ui.say("  Needs the cross-section and test-waveguide stages; not run.")
        return dict(skipped=True)
    cp = coupler_model(ctx)
    if cp is None:
        ui.say("  Needs a coupler (grating or edge-coupler stage); not run.")
        return dict(skipped=True)
    alpha = float(wg["alpha_db_cm"])
    sigma = float(cfg.waveguides.coupler_scatter_db)
    budget, refl_src = coupler_reflection(ctx)
    use_refl = bool(cfg.device.reflections) and bool(budget.get("reflected"))
    wl0 = cfg.run.wavelength_um
    n_g = float(xs.get("n_group") or 2.0)
    lengths = [(0.0 if x == 0 else float(x)) for x in wg["lengths_cm"]]
    from tarang.design.layout import bus_length_um
    L_ref_cm = bus_length_um(ctx) * 1e-4
    extra_cm = 2 * cp["length_um"] * 1e-4                # tapers / tip tapers, same in every one
    edge_c = coupling_kind(ctx) == "edge"
    eta0 = float(interp_eta(wl0, cp["wl"], cp["eta"]))
    wl = np.linspace(float(cp["wl"].min()), float(cp["wl"].max()), 400)
    eta = interp_eta(wl, cp["wl"], cp["eta"])
    R = (np.interp(wl, np.sort(budget["wavelength_um"]),
                   np.asarray(budget["reflected"], float)[np.argsort(budget["wavelength_um"])])
         if use_refl else np.zeros_like(wl))
    rows, curves = [], []
    for x in lengths:
        # edge coupling: the cutback lengths are facet to facet (couplers included)
        L_cm = (L_ref_cm + extra_cm) if x == 0 else (x if edge_c else x + extra_cm)
        p = 10 ** (-alpha * L_cm / 10)
        T0 = eta0 ** 2 * p
        Rp = (float(np.interp(wl0, wl, R)) * p) if use_refl else 0.0
        ripple = 20 * math.log10((1 + Rp) / (1 - Rp)) if Rp else 0.0
        fsr_pm = (wl0 * 1e-6) ** 2 / (2 * n_g * L_cm * 1e-2) * 1e12
        rows.append(dict(structure="reference" if x == 0 else f"spiral {x:g} cm",
                         length_cm=round(L_cm, 4), transmission_db=round(10 * math.log10(T0), 3),
                         ripple_pp_db=round(ripple, 3), ripple_period_pm=round(fsr_pm, 2)))
        base = eta ** 2 * p
        Rw = R * p
        curves.append((rows[-1]["structure"], base, base / (1 - Rw) ** 2, base / (1 + Rw) ** 2))
    Ls = np.array([r["length_cm"] for r in rows])
    sig_a = sigma / math.sqrt(float(np.sum((Ls - Ls.mean()) ** 2))) if len(Ls) > 1 else math.inf
    ui.say(f"Cutback set with the {cp['label']}: expected transmission at {wl0} µm (±{sigma:g} dB "
           f"coupling scatter per structure):")
    ui.table(rows)
    ui.say(f"  A straight-line fit gives the loss ({alpha:g} dB/cm expected) to ±{sig_a:.3g} dB/cm "
           f"and the coupling loss per coupler ({-10 * math.log10(eta0):.2f} dB) from the intercept.")
    if use_refl:
        ui.say(f"  Ripple from the couplers' reflection ({refl_src}): up to "
               f"{max(r['ripple_pp_db'] for r in rows):.2f} dB peak to peak; average over a few "
               "ripple periods when measuring.")
        if edge_c:
            ui.say("  With edge couplers each structure is a Fabry-Perot cavity between its two facets: "
                   "the ripple contrast K = (Tmax - Tmin)/(Tmax + Tmin) of one waveguide also gives its "
                   "loss, R exp(-alpha L) = (1 - sqrt(1 - K^2)) / K with the facet reflection R from "
                   "this design (Regener and Sohler, Appl. Phys. B 36, 143 (1985)); sweep the laser in "
                   "steps well below the ripple period.")
    from tarang.design.checks import _write_csv
    _write_csv(ctx.path("tables", "device_cutback.csv"), rows)
    fig = visuals.safe(ctx, "cutback", draw_cutback, ctx, rows, sigma, alpha, eta0, wl, curves, cp)
    return dict(goal="loss_test", coupler=cp["label"], rows=rows, alpha_db_cm=alpha,
                sigma_alpha_db_cm=sig_a, coupling_loss_db=-10 * math.log10(eta0),
                reflections=use_refl, reflection_source=refl_src,
                tables=["tables/device_cutback.csv"], figures=[str(fig)] if fig else [])


def draw_cutback(ctx, rows, sigma, alpha, eta0, wl, curves, cp=None):
    """Figure 11a (test waveguides): expected cutback line with the coupling scatter, and
    the spectrum of every structure with its ripple band."""
    from tarang.design import visuals
    plt = visuals._plt()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.6))
    L = np.array([r["length_cm"] for r in rows])
    T = np.array([r["transmission_db"] for r in rows])
    ax.errorbar(L, T, yerr=sigma, fmt="o", color=visuals.PALETTE[0], capsize=4,
                label=f"expected, ±{sigma:g} dB coupling scatter")
    xx = np.linspace(0, L.max() * 1.05, 50)
    ax.plot(xx, 20 * math.log10(eta0) - alpha * xx, color="0.4", ls="--",
            label=f"slope −{alpha:g} dB/cm, intercept 2 × coupler")
    ax.set(xlabel="Waveguide length (cm)", ylabel="Transmission at the pump (dB)",
           title="Cutback: transmission against length")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    for k, (name, base, hi, lo) in enumerate(curves):
        col = visuals.PALETTE[k % len(visuals.PALETTE)]
        bx.plot(wl * 1e3, 10 * np.log10(np.clip(base, 1e-12, None)), color=col, label=name)
        bx.fill_between(wl * 1e3, 10 * np.log10(np.clip(lo, 1e-12, None)),
                        10 * np.log10(np.clip(hi, 1e-12, None)), color=col, alpha=0.18, lw=0)
    if cp:
        bx.plot(np.asarray(cp["wl"]) * 1e3, 10 * np.log10(np.clip(cp["eta"], 1e-12, None)), "o--",
                ms=2.5, lw=0.8, color="0.45", label="one coupler (simulated points)")
    bx.axvline(ctx.cfg.run.wavelength_um * 1e3, color="0.6", ls=":")
    bx.set(xlabel="Wavelength (nm)", ylabel="Transmission (dB)",
           title="Spectra (band: Fabry-Perot ripple between the couplers)")
    bx.grid(alpha=0.3)
    bx.legend(fontsize=8)
    return visuals._save(fig, ctx.path("figures", "11a_cutback.png"))
