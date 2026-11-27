"""Inputs for a Lugiato-Lefever (LLE) comb simulation, written by the layout stage into
``<run>/lle/`` so a comb solver can start from this design without retyping anything.

Files:
* ``lle_inputs.json``: every number an LLE needs, with units and definitions:
  pump frequency, FSR (D1/2pi), D2/2pi, D3/2pi, Q_i, and for the critical and the
  over-coupled gap: Q_c, Q_L, the loss rates kappa_0, kappa_ex, kappa (as /2pi, Hz), the
  threshold; n_eff, n_g, A_eff, n2 and gamma. gamma is the full-vectorial coefficient of
  Afshar and Monro (Opt. Express 17, 2298 (2009), Eq. 39) with the Kerr index of each
  material where the field is ([assumptions] gamma_model = "vectorial", the default), or the
  core-only 2 pi n2 / (lambda A_eff) ("core", as before v0.2.0); both are written. It also holds a
  ``pyLLE`` block in that solver's own names (R, Qi, Qc, gamma, dispfile).
* ``pyLLE_dispersion.csv``: azimuthal mode number, resonance frequency (Hz), one mode per
  line, no header: the dispersion-file format pyLLE reads.
* ``integrated_dispersion.csv``: mu, wavelength, frequency and D_int/2pi of each mode
  around the pump (with header), for any other solver.

The resonances come from the broadband TE00 effective index of the straight cross-section
(the ring stage's solve): f_m solves m = n_eff(f) 2 pi R f / c. TARANG does not run an LLE
itself; this is the interface to one.
"""
from __future__ import annotations

import csv
import json
import math
from typing import Optional

C = 299792458.0


def _broadband_record(ctx, H: float, W: float) -> Optional[dict]:
    f = ctx.out / "results" / "broadband.jsonl"
    if not f.exists():
        return None
    recs = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
    recs = [r for r in recs if "error" not in r and abs(r["H"] - H) < 1e-6 and abs(r["W"] - W) < 1e-6
            and not r.get("bend_radius")]
    return recs[-1] if recs else None


def regime(qi: float, K: float, f_pump: float, n_eff: float, aeff_um2: float, R_um: float,
           n2: float, wl_um: float, gamma: Optional[float] = None) -> dict:
    """Loss rates and threshold at coupling ratio K = Q_i / Q_c. The threshold
    1.54 (pi/2) (Q_c/2Q_L) n_eff^2 V / (n2 lambda Q_L^2), V = A_eff 2 pi R, depends on the
    mode through n2 / A_eff = gamma lambda / (2 pi): with ``gamma`` given it is used there
    (the vectorial coefficient), otherwise n2 / A_eff."""
    qc, ql = qi / K, qi / (1 + K)
    k0, kex = f_pump / qi, f_pump / qc
    n2_per_area = (gamma * wl_um * 1e-6 / (2 * math.pi)) if gamma else n2 / (aeff_um2 * 1e-12)
    L = 2 * math.pi * R_um * 1e-6
    pth = 1.54 * (math.pi / 2) * (qc / (2 * ql)) * n_eff ** 2 * L / (n2_per_area * wl_um * 1e-6
                                                                     * ql ** 2)
    return dict(K=K, Q_c=qc, Q_L=ql, kappa0_2pi_Hz=k0, kappa_ex_2pi_Hz=kex,
                kappa_2pi_Hz=k0 + kex, linewidth_Hz=k0 + kex, threshold_W=pth)


def nonlinear(ctx, H: float, W: float) -> Optional[dict]:
    """Vectorial nonlinear overlaps of the pump mode (one local mode solve per snapping side,
    at the design's mesh, profile and materials), saved in results/nonlinear_overlaps.json
    and reused while those inputs stay the same; gamma with this design's n2 values."""
    from tarang.design.assumptions import n2 as _n2
    from tarang.design.config import profile_of
    from tarang.media import materials
    from tarang.waveguide.dispersion_mapping import nonlinear as nl
    cfg = ctx.cfg
    snap = bool(cfg.cross_section.snap_grid)
    prof = profile_of(cfg)
    inputs = dict(H=H, W=W, wl=cfg.run.wavelength_um, steps=int(cfg.cross_section.decision_steps),
                  box=cfg.stack.box_height_um, core=cfg.materials.core, substrate=cfg.materials.box,
                  cladding=cfg.materials.cladding, cladding_height=cfg.stack.cladding_height_um,
                  snap=snap, snap_sides=(ctx.snap_sides if snap else 1),
                  sidewall_deg=float(getattr(cfg.cross_section, "sidewall_deg", 90.0) or 90.0),
                  profile=prof.record(), mode=str(getattr(cfg.cross_section, "pump_mode", "TE00")),
                  band=list(ctx.band), version=1)
    f = ctx.out / "results" / "nonlinear_overlaps.json"
    saved = json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
    if saved and saved.get("inputs") == inputs:
        ov = saved["overlaps"]
    else:
        clad = None if str(cfg.materials.cladding).lower() in {"air", "none", ""} else cfg.materials.cladding
        ctx.ui.say("Vectorial nonlinear coefficient: one local mode solve of the pump mode"
                   + (" per snapping side" if snap and ctx.snap_sides == 2 else "") + "...")
        ov = nl.overlaps(H, W, wl0=inputs["wl"], steps=inputs["steps"], box=inputs["box"],
                         core=inputs["core"], substrate=inputs["substrate"], cladding=clad,
                         cladding_height=inputs["cladding_height"], band=ctx.band, snap=snap,
                         snap_sides=inputs["snap_sides"], sidewall_deg=inputs["sidewall_deg"],
                         profile=prof, mode=inputs["mode"])
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(dict(inputs=inputs, overlaps=ov), indent=1), encoding="utf-8")
    n2 = {}
    notes = []
    for role, key in ov["materials"].items():
        if role == "core":
            n2[role] = _n2(cfg)
            continue
        inf = materials.info(key)
        v = inf.n2_m2_w if inf else None
        if v is None:
            notes.append(f"n2 of {key} ({role}) not recorded: counted as 0")
        n2[role] = float(v or 0.0)
    g, shares = nl.gamma_from(ov["G_per_m3"], n2)
    g_core = nl.scalar_gamma(n2["core"], inputs["wl"], ov["aeff_um2"])
    return dict(gamma_vectorial_per_W_m=g, gamma_core_only_per_W_m=g_core,
                shares=shares, n2_by_material={ov["materials"][r]: v for r, v in n2.items()},
                aeff_um2=ov["aeff_um2"], notes=notes, overlaps_file=str(f),
                reference="Afshar and Monro, Opt. Express 17, 2298 (2009), Eq. 39")


def export(ctx) -> Optional[dict]:
    """Write the LLE inputs; None (with a note) if the ring stage has not run."""
    from tarang.waveguide.dispersion_mapping.broadband import analyse
    st = ctx.state["stages"]
    xs = st.get("cross_section", {}).get("results")
    ring = st.get("ring", {}).get("results")
    if not xs or not ring:
        return None
    gap = st.get("gap", {}).get("results") or {}
    cfg = ctx.cfg
    H, W, R = xs["height_um"], xs["width_um"], ring["radius_um"]
    wl = cfg.run.wavelength_um
    out = ctx.out / "lle"
    out.mkdir(parents=True, exist_ok=True)
    rec = _broadband_record(ctx, H, W)
    disp = None
    if rec is not None:
        rec = dict(rec, pump=wl)
        A = analyse(rec, R)
        m0 = int(A["m0"])
        rows = sorted(((m0 + int(mu), float(om) / (2 * math.pi), float(di) / (2 * math.pi),
                        float(lam)) for mu, om, di, lam in zip(A["mus"], A["om_mu"], A["Dint"],
                                                                A["lam_mu"])))
        with open(out / "pyLLE_dispersion.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            for m, fr, _, _ in rows:
                w.writerow([m, repr(fr)])
        with open(out / "integrated_dispersion.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["mu", "mode_number", "wavelength_nm", "frequency_Hz", "Dint_2pi_Hz"])
            for m, fr, di, lam in rows:
                w.writerow([m - m0, m, round(lam, 4), repr(fr), repr(di)])
        disp = dict(pump_mode_number=m0, f_pump_Hz=A["om_pump"] / (2 * math.pi),
                    FSR_Hz=A["D1"] / (2 * math.pi), D2_2pi_Hz=A["D2_2pi_MHz"] * 1e6,
                    D3_2pi_Hz=A["D3_2pi_kHz"] * 1e3, modes=len(rows))
    f_pump = disp["f_pump_Hz"] if disp else C / (wl * 1e-6)
    from tarang.design.assumptions import n2 as _n2
    n2 = _n2(cfg)
    aeff = xs.get("aeff_um2")
    qi = ring["q_intrinsic"]
    g_core = 2 * math.pi * n2 / (wl * 1e-6 * aeff * 1e-12) if aeff else None
    model = str(getattr(cfg.assumptions, "gamma_model", "vectorial") or "vectorial")
    vec = None
    if model == "vectorial":
        vec = nonlinear(ctx, H, W)
    gamma = vec["gamma_vectorial_per_W_m"] if vec else g_core
    regimes = {}
    if aeff:
        regimes["critical"] = regime(qi, 1.0, f_pump, xs["n_eff"], aeff, R, n2, wl,
                                     gamma=gamma if vec else None)
        K = gap.get("over_coupling")
        if K and K != 1:
            regimes[f"over_coupled_K{K:g}"] = regime(qi, float(K), f_pump, xs["n_eff"], aeff, R,
                                                     n2, wl, gamma=gamma if vec else None)
    over = next((v for k, v in regimes.items() if k.startswith("over")), regimes.get("critical"))
    data = {
        "_about": "Inputs for a Lugiato-Lefever comb simulation of this TARANG design. "
                  "Frequencies in Hz (rates as /2pi), lengths in m unless stated.",
        "wavelength_um": wl, "pump_frequency_Hz": f_pump, "radius_m": R * 1e-6,
        "n_eff": xs["n_eff"], "n_g": ring["n_group"], "A_eff_m2": aeff * 1e-12 if aeff else None,
        "n2_m2_per_W": n2, "gamma_per_W_m": gamma, "gamma_model": model,
        "gamma_core_only_per_W_m": g_core,
        "gamma_vectorial": vec or "not computed (gamma_model = core)",
        "Q_intrinsic": qi,
        "loss_dB_per_cm": ring.get("loss_db_cm"),
        "dispersion": disp or "not available (ring stage broadband solve missing)",
        "coupling": regimes,
        "gaps_um": {"critical": gap.get("critical_gap_um"), "over_coupled": gap.get("over_coupled_gap_um")},
        "definitions": {
            "D_int(mu)": "omega_mu - omega_0 - D1 mu = D2 mu^2/2 + D3 mu^3/6 + ...",
            "pyLLE D2": "pyLLE's dispersion analyser prints the mu^2 coefficient of D_int, i.e. "
                        "D2/2 in this file's convention; it simulates from the resonance "
                        "frequencies in pyLLE_dispersion.csv, so the physics is the same",
            "K": "Q_i / Q_c = kappa_ex / kappa_0 (1 = critical, > 1 over-coupled)",
            "kappa": "kappa_0 + kappa_ex = omega / Q_L (full linewidth, here as /2pi in Hz)",
            "threshold_W": "1.54 (pi/2) (Q_c / 2 Q_L) n_eff^2 V / (n2 lambda Q_L^2), V = A_eff 2 pi R; "
                           "with the vectorial model n2 / A_eff = gamma lambda / (2 pi)",
            "gamma (vectorial)": "k0 (eps0/mu0) Int n^2 n2(r) [2|e|^4 + |e.e|^2] dA / "
                                 "(3 |Int Re(e x h*).z dA|^2): Afshar and Monro, Opt. Express "
                                 "17, 2298 (2009), Eq. 39; n2(r) of each material",
            "gamma (core only)": "2 pi n2_core / (lambda A_eff), A_eff = (Int|E|^2)^2 / Int|E|^4"},
        "pyLLE": {"res": {"R": R * 1e-6, "Qi": qi, "Qc": over["Q_c"] if over else None,
                          "gamma": gamma, "dispfile": "pyLLE_dispersion.csv"},
                  "note": "pyLLE's resonator dict; its simulation dict (pump power, detuning "
                          "scan, modes to simulate) is chosen by the user."},
    }
    (out / "lle_inputs.json").write_text(json.dumps(data, indent=1, default=float), encoding="utf-8")
    files = [str(out / "lle_inputs.json")]
    if disp:
        files += [str(out / "pyLLE_dispersion.csv"), str(out / "integrated_dispersion.csv")]
    return dict(files=files, gamma_per_W_m=gamma, gamma_model=model,
                gamma_core_only_per_W_m=g_core,
                gamma_shares=(vec or {}).get("shares"), dispersion=disp, coupling=regimes)
