"""Proof of soliton formation for a TARANG design, with pyLLE (not part of the package).

TARANG designs the resonator and writes everything a Lugiato-Lefever simulation needs into
``<design>/lle/`` (``lle_inputs.json`` and ``pyLLE_dispersion.csv``). This script hands
those to pyLLE (G. Moille, NIST; https://github.com/gregmoille/pyLLE) twice:

1. **Detuning sweep** from the blue to the red side of the resonance, as in an experiment:
   the comb power shows modulation instability, chaos and then soliton steps.
   ``pylle_power_vs_step`` (with the normalised detuning), ``pylle_spectrum`` and
   ``pylle_temporal`` at the chosen step, and the number of solitons on that step.
2. **One soliton, held**: a single sech pulse is seeded at a fixed red detuning inside the
   soliton existence range and evolved for many photon lifetimes. If it keeps one pulse and
   a constant power, the design supports a stable single soliton (``pylle_single_soliton``).

Plus ``pylle_dint``: D_int as pyLLE reads it from TARANG's resonance frequencies, against
TARANG's own ``integrated_dispersion.csv``; and ``pylle_summary.json`` with the numbers.

The normalised quantities (Herr et al., Nat. Photon. 8, 145 (2014)): detuning
zeta = 2 (omega_0 - omega_p) / kappa; pump f^2 = 8 kappa_ex g0 P / (kappa^3 hbar omega), here
computed as theta P gamma L / alpha^3 with theta = kappa_ex t_R and alpha = kappa t_R / 2; a
soliton exists up to zeta_max = pi^2 f^2 / 8 and has the width sech(phi sqrt(zeta kappa / D2)).

Each figure is saved with its numbers (``_data.csv``) and settings (``_meta.json``) by
``tarang.figures``, in ``<design>/validation/pylle/``.

Requirements (once): Julia >= 1.8 (https://julialang.org/downloads/) and ``pip install
pyLLE``. pyLLE installs its Julia packages itself on Linux and macOS; on Windows install them
first, in Julia:  ``import Pkg; Pkg.add(["HDF5", "FFTW", "LinearAlgebra", "ProgressMeter"])``.
pyLLE 4.1 still calls ``np.string_`` (removed in NumPy 2); this script restores the name.

Run:
    python validation/pylle_soliton/run_pylle.py design_runs/<name> --julia <path to julia>
    python validation/pylle_soliton/run_pylle.py design_runs/<name> --dry-run   # inputs only

Physics of the defaults: the pump power is 4x the threshold TARANG computed for the
over-coupled gap (``--power-over-threshold``); the sweep runs from 2 half-linewidths blue to
12 red of the cold resonance; ``--modes`` either side of the pump are simulated; the single
soliton is held at the middle of [sqrt(3), zeta_max] (``--hold-zeta`` sets it).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def load_inputs(design: Path) -> dict:
    """lle_inputs.json of a TARANG design, with the regime the pyLLE block uses."""
    p = Path(design) / "lle" / "lle_inputs.json"
    if not p.exists():
        raise SystemExit(f"{p} not found: run the design's layout step first "
                         "(it writes the lle/ folder).")
    data = json.loads(p.read_text(encoding="utf-8"))
    block = (data.get("pyLLE") or {}).get("res") or {}
    missing = [k for k in ("R", "Qi", "Qc", "gamma") if not block.get(k)]
    if missing:
        raise SystemExit(f"{p}: the pyLLE block lacks {missing} (needs the ring and gap steps).")
    disp = Path(design) / "lle" / block.get("dispfile", "pyLLE_dispersion.csv")
    if not disp.exists():
        raise SystemExit(f"{disp} not found (needs the ring step's broadband solve).")
    over = next((v for k, v in (data.get("coupling") or {}).items() if k.startswith("over")),
                (data.get("coupling") or {}).get("critical"))
    return dict(data=data, res=block, dispfile=disp, regime=over or {})


def settings(inp: dict, power_mw=None, power_over_threshold=4.0, modes=200, steps=5000,
             det_start=2.0, det_end=-12.0) -> tuple[dict, dict]:
    """pyLLE's resonator and simulation dicts (SI units, rates in rad/s)."""
    data, res, reg = inp["data"], inp["res"], inp["regime"]
    kappa = 2 * math.pi * float(reg.get("kappa_2pi_Hz") or data["pump_frequency_Hz"] / res["Qi"])
    pth = reg.get("threshold_W")
    pin = power_mw * 1e-3 if power_mw else (power_over_threshold * pth if pth else 0.2)
    r = dict(R=float(res["R"]), Qi=float(res["Qi"]), Qc=float(res["Qc"]),
             γ=float(res["gamma"]), dispfile=str(inp["dispfile"]))
    s = dict(Pin=[pin], f_pmp=[float(data["pump_frequency_Hz"])], φ_pmp=[0], δω=[None],
             Tscan=float(steps) * 1e2, μ_sim=[-int(modes), int(modes)],
             δω_init=det_start * kappa / 2, δω_end=det_end * kappa / 2, num_probe=int(steps))
    return r, s


def soliton_step(pcomb) -> int:
    """The middle of the soliton step of the sweep: after the chaotic maximum, the longest run
    of steps whose power changes by less than 1 % from one step to the next (a soliton state
    is steady; chaotic and breathing states and the jumps between steps are not)."""
    import numpy as np
    p = np.asarray(pcomb, float)
    k = int(np.argmax(p))
    q = p[k:]
    on = q > 0.1 * p[k]
    calm = on[1:] & on[:-1] & (np.abs(np.diff(q)) < 0.01 * np.maximum(q[:-1], 1e-30))
    best, run, start = (0, 0), 0, 0
    for i, c in enumerate(calm):
        if c:
            if run == 0:
                start = i
            run += 1
            if run > best[1] - best[0]:
                best = (start, i + 1)
        else:
            run = 0
    if best[1] - best[0] < 10:
        return int(len(p) * 0.8)
    return int(k + (best[0] + best[1]) // 2)


def fwhm(theta, intensity) -> float:
    """Full width at half maximum of the highest pulse (same units as ``theta``), with the
    half-maximum crossings interpolated between samples."""
    import numpy as np
    y = np.asarray(intensity, float)
    x = np.asarray(theta, float)
    k = int(np.argmax(y))
    half = 0.5 * y[k]
    i = k
    while i > 0 and y[i - 1] > half:
        i -= 1
    j = k
    while j < len(y) - 1 and y[j + 1] > half:
        j += 1
    left = x[i] if i == 0 else x[i - 1] + (half - y[i - 1]) / (y[i] - y[i - 1]) * (x[i] - x[i - 1])
    right = x[j] if j == len(y) - 1 else x[j] + (y[j] - half) / (y[j] - y[j + 1]) * (x[j + 1] - x[j])
    return float(right - left)


def expected_fwhm_fs(nrm: dict, zeta: float) -> float:
    """FWHM of the soliton's intensity, sech²(φ √(ζ κ / D2)), in time: the sech² FWHM
    2 arcsech(1/√2) = 1.7627 divided by √(ζ κ / D2), as a fraction of the round trip."""
    a = math.sqrt(zeta * nrm["kappa_2pi_Hz"] / nrm["D2_2pi_Hz"])
    return 1.7627 / a / (2 * math.pi) * nrm["t_R_s"] * 1e15


def normalised(inp: dict, pin_W: float) -> dict:
    """Herr-normalised pump f^2, the soliton existence limit zeta_max = pi^2 f^2 / 8 and the
    rates needed to build a soliton (None when lle_inputs.json lacks them)."""
    data, reg = inp["data"], inp["regime"]
    disp = data.get("dispersion") or {}
    need = (disp.get("FSR_Hz"), disp.get("D2_2pi_Hz"), data.get("radius_m"),
            data.get("gamma_per_W_m"), reg.get("kappa_2pi_Hz"), reg.get("kappa_ex_2pi_Hz"))
    if any(v is None for v in need):
        return {}
    fsr, d2, radius, gamma, k2, kex2 = (float(v) for v in need)
    t_r, length = 1.0 / fsr, 2 * math.pi * radius
    theta, alpha = 2 * math.pi * kex2 * t_r, math.pi * k2 * t_r
    f2 = theta * pin_W * gamma * length / alpha ** 3
    return dict(f2=f2, zeta_max=math.pi ** 2 * f2 / 8, kappa_2pi_Hz=k2, D2_2pi_Hz=d2,
                t_R_s=t_r, alpha=alpha, gamma_L=gamma * length)


def seed_soliton(nrm: dict, zeta: float, n: int):
    """One bright soliton on its CW background around the ring (field in sqrt(W), n points),
    the initial state of the held run: psi = Phi0 + sqrt(2 zeta) e^(-i phi0) sech(phi sqrt(zeta
    kappa / D2)). The phase sign follows pyLLE's detuning convention (red = negative)."""
    import numpy as np
    f = math.sqrt(nrm["f2"])
    phi0 = math.acos(min(1.0, math.sqrt(8 * zeta) / (math.pi * f)))
    phi = np.linspace(-math.pi, math.pi, n)
    background = f / zeta ** 2 - 1j * f / zeta
    psi = background + math.sqrt(2 * zeta) * np.exp(-1j * phi0) / np.cosh(
        phi * math.sqrt(zeta * nrm["kappa_2pi_Hz"] / nrm["D2_2pi_Hz"]))
    return math.sqrt(nrm["alpha"] / nrm["gamma_L"]) * psi


def count_pulses(intensity, rel=0.3) -> int:
    """Pulses around the ring: maxima above rel x the highest (none on a flat background)."""
    import numpy as np
    y = np.asarray(intensity, float)
    if y.size < 3 or y.max() <= 0 or y.max() < 2.0 * max(np.median(y), 1e-30):
        return 0
    yy = np.concatenate([y[-1:], y, y[:1]])                 # periodic ring
    peak = (yy[1:-1] > yy[:-2]) & (yy[1:-1] >= yy[2:]) & (y > rel * y.max())
    return int(peak.sum())


def _safe_console() -> None:
    """Replace characters the console cannot show (γ, ±, μ on a Windows cp1252 pipe) with '?'
    instead of stopping with UnicodeEncodeError. Files are always written as UTF-8."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv=None) -> int:
    _safe_console()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("design", type=Path, help="the TARANG design folder (holds lle/)")
    ap.add_argument("--power-mw", type=float, default=None, help="pump power in the bus (mW)")
    ap.add_argument("--power-over-threshold", type=float, default=4.0,
                    help="pump power as a multiple of the threshold (when --power-mw is not set)")
    ap.add_argument("--modes", type=int, default=200, help="modes simulated either side")
    ap.add_argument("--steps", type=int, default=5000, help="saved steps of the sweep")
    ap.add_argument("--step", type=int, default=None, help="step to show (default: on the "
                                                          "single-soliton plateau)")
    ap.add_argument("--hold-zeta", type=float, default=None,
                    help="normalised detuning of the held single soliton (default: the middle "
                         "of [sqrt(3), zeta_max])")
    ap.add_argument("--hold-roundtrips", type=float, default=2e4,
                    help="length of the held run in round trips")
    ap.add_argument("--no-sweep", action="store_true", help="only the held single soliton")
    ap.add_argument("--julia", default=None, help="path to the julia executable")
    ap.add_argument("--dry-run", action="store_true", help="check and print the inputs only")
    a = ap.parse_args(argv)

    inp = load_inputs(a.design)
    res, sim = settings(inp, a.power_mw, a.power_over_threshold, a.modes, a.steps)
    print("pyLLE resonator:", {k: v for k, v in res.items()})
    print(f"pump {sim['Pin'][0] * 1e3:.0f} mW at {sim['f_pmp'][0] / 1e12:.3f} THz, "
          f"modes ±{a.modes}, detuning {sim['δω_init'] / 2 / math.pi / 1e6:.0f} -> "
          f"{sim['δω_end'] / 2 / math.pi / 1e6:.0f} MHz")
    nrm = normalised(inp, sim["Pin"][0])
    if nrm:
        print(f"normalised pump f^2 = {nrm['f2']:.2f}; solitons exist up to zeta_max = "
              f"pi^2 f^2 / 8 = {nrm['zeta_max']:.2f}")
    if a.dry_run:
        return 0
    import numpy as np
    if not hasattr(np, "string_"):      # pyLLE 4.1 still calls np.string_ (gone in NumPy 2)
        np.string_ = np.bytes_
    try:
        import pyLLE
    except ImportError:
        print("pyLLE is not installed: pip install pyLLE (and Julia >= 1.8, see the top of "
              "this file).")
        return 1
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tarang import figures

    out = Path(a.design) / "validation" / "pylle"
    out.mkdir(parents=True, exist_ok=True)
    figures.configure(formats=["png", "pdf"], dpi=300)
    meta = dict(source="pyLLE " + getattr(pyLLE, "__version__", "?"), resonator=res,
                pump_W=sim["Pin"][0], pump_Hz=sim["f_pmp"][0], design=str(a.design))

    summary = dict(pump_mW=sim["Pin"][0] * 1e3, **{k: nrm[k] for k in ("f2", "zeta_max")
                                                     if k in nrm})

    def solve(sim_):
        solver = pyLLE.LLEsolver(sim=sim_, res=res, debug=False)
        solver.Analyze(plot=False)
        solver.Setup(verbose=False)
        solver.SolveTemporal(**({"bin": a.julia} if a.julia else {}))
        solver.RetrieveData()
        return solver

    def lam_nm(freq):
        return 299792458.0 / np.asarray(freq, float) * 1e9

    def spectrum_panel(ax, sol, k):
        spec = np.abs(np.asarray(sol.Ewg)[:, k]) ** 2
        db = 10 * np.log10(np.maximum(spec / spec.max(), 1e-12))
        lam = lam_nm(sol.freq)
        ax.vlines(lam, -200, db, lw=0.8)
        shown = lam[db > -100]
        if shown.size:
            pad = 0.05 * (shown.max() - shown.min() + 1)
            ax.set_xlim(shown.min() - pad, shown.max() + pad)
        ax.set(xlabel="wavelength (nm)", ylabel="power in the bus (dB, rel.)", ylim=(-100, 5))

    k2 = nrm.get("kappa_2pi_Hz")
    if not a.no_sweep:
        solver = solve(sim)
        sol = solver.sol
        pcomb = np.asarray(sol.Pcomb, float)
        k = a.step if a.step is not None else soliton_step(pcomb)
        det = np.asarray(getattr(sol, "δfreq", getattr(sol, "dfreq", [])), float).ravel()
        zeta = -2 * det / k2 if (k2 and det.size == pcomb.size) else None
        fig, ax = plt.subplots(figsize=(7, 4))
        x = zeta if zeta is not None else np.arange(len(pcomb))
        ax.plot(x, pcomb * 1e3, lw=1.2, label="intracavity comb power")
        ax.axvline(x[k], color="crimson", ls="--", lw=1, label="state shown")
        if zeta is not None and "zeta_max" in nrm:
            ax.axvline(nrm["zeta_max"], color="0.4", ls=":", lw=1,
                       label="soliton limit \u03b6_max = \u03c0\u00b2f\u00b2/8")
        ax.set(xlabel=("normalised detuning \u03b6 = 2(\u03c9\u2080 \u2212 \u03c9_p)/\u03ba"
                       if zeta is not None else "step of the detuning sweep (blue to red)"),
               ylabel="comb power (mW)")
        ax.legend(fontsize=8, loc="upper left")
        figures.save(fig, out / "pylle_power_vs_step.png", meta=meta)
        fig, ax = plt.subplots(figsize=(7, 4))
        spectrum_panel(ax, sol, k)
        figures.save(fig, out / "pylle_spectrum.png", meta=meta)
        acav = np.asarray(sol.Acav)
        theta = np.linspace(-np.pi, np.pi, acav.shape[0])
        inten = np.abs(acav[:, k]) ** 2
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(theta, inten / inten.max(), lw=1.2)
        ax.set(xlabel="angle around the ring (rad)", ylabel="intracavity intensity (norm.)")
        figures.save(fig, out / "pylle_temporal.png", meta=meta)
        lost = np.where(pcomb > 0.1 * pcomb.max())[0]
        summary.update(sweep_state_step=int(k), sweep_solitons=count_pulses(inten),
                       sweep_state_zeta=(float(zeta[k]) if zeta is not None else None),
                       sweep_zeta_soliton_lost=(float(zeta[lost[-1]]) if zeta is not None
                                                and lost.size else None))
        dint_csv = Path(a.design) / "lle" / "integrated_dispersion.csv"
        try:                     # D_int as pyLLE interpolated it (rad/s on its mode grid)
            d_p = np.asarray(solver._sim["Dint"], float).ravel()
            lo, hi = solver._sim["mu_sim"][0], solver._sim["mu_sim"][1]
            m_p = np.arange(lo, hi + 1) if hi - lo + 1 == d_p.size else \
                np.arange(d_p.size) - d_p.size // 2
            d_p = (d_p - d_p[m_p == 0][0]) / (2 * np.pi)          # Hz, 0 at the pump
        except Exception as err:                                      # noqa: BLE001
            print(f"  D_int of pyLLE not found ({type(err).__name__}: {err}); no D_int figure")
            d_p = None
        if d_p is not None and dint_csv.exists():
            import csv as _csv
            with open(dint_csv, encoding="utf-8") as f:
                rows = list(_csv.DictReader(f))
            m_t = np.array([float(r["mu"]) for r in rows])
            d_t = np.array([float(r["Dint_2pi_Hz"]) for r in rows])
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.plot(m_t, d_t / 1e9, lw=1.8, label="TARANG (lle/integrated_dispersion.csv)")
            ax.plot(m_p, d_p / 1e9, "--", lw=1.2, label="as pyLLE interpolated it")
            outside = (m_p < m_t.min()) | (m_p > m_t.max())
            if outside.any():                     # modes beyond the solved wavelengths
                for lo_, hi_ in ((m_p.min(), m_t.min()), (m_t.max(), m_p.max())):
                    if hi_ > lo_:
                        ax.axvspan(lo_, hi_, color="0.85", lw=0)
                ax.plot([], [], color="0.85", lw=6, label="extrapolated by pyLLE")
                summary["modes_extrapolated"] = int(outside.sum())
                print(f"  Note: {int(outside.sum())} of the {m_p.size} simulated modes lie beyond "
                      f"TARANG's D_int (mu {m_t.min():.0f} to {m_t.max():.0f}); pyLLE extrapolates "
                      "there. Keep the comb lines that matter inside it (or lower --modes).")
            ax.set(xlabel="mode number \u03bc from the pump", ylabel="D_int / 2\u03c0 (GHz)")
            ax.legend(fontsize=8)
            figures.save(fig, out / "pylle_dint.png", meta=meta)
            common = np.intersect1d(m_t, m_p)
            if common.size > 3:
                a_t = np.interp(common, m_t, d_t)
                a_p = np.interp(common, m_p, d_p)
                summary["dint_max_difference_MHz"] = float(np.abs(a_t - a_p).max() / 1e6)
                summary["dint_span_GHz"] = float(np.abs(a_t).max() / 1e9)

    if nrm and nrm["zeta_max"] > math.sqrt(3) + 0.2:
        z = a.hold_zeta or 0.5 * (math.sqrt(3) + nrm["zeta_max"])
        hold = dict(sim, δω_init=-z * math.pi * k2, δω_end=-z * math.pi * k2,
                    Tscan=float(a.hold_roundtrips), num_probe=200)
        hold["DKS_init"] = seed_soliton(nrm, z, 2 * a.modes + 1)
        sol = solve(hold).sol
        p_h = np.asarray(sol.Pcomb, float)
        acav = np.asarray(sol.Acav)
        inten = np.abs(acav[:, -1]) ** 2
        n_end = count_pulses(inten)
        steady = bool(p_h[-1] > 0 and abs(p_h[-1] - p_h[len(p_h) // 2]) < 0.02 * p_h[-1])
        theta = np.linspace(-np.pi, np.pi, acav.shape[0])
        fwhm_rad = fwhm(theta, inten) if n_end == 1 else None
        fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
        axs[0].plot(np.linspace(0, a.hold_roundtrips, p_h.size), p_h * 1e3, lw=1.2)
        axs[0].set(xlabel="round trips", ylabel="comb power (mW)",
                   title=f"held at \u03b6 = {z:.2f}")
        axs[1].plot(theta, inten / inten.max(), lw=1.2)
        axs[1].set(xlabel="angle around the ring (rad)", ylabel="intensity (norm.)",
                   title=f"{n_end} pulse(s) at the end")
        spectrum_panel(axs[2], sol, -1)
        axs[2].set_title("comb spectrum")
        figures.save(fig, out / "pylle_single_soliton.png", meta=dict(meta, hold_zeta=z))
        summary.update(hold_zeta=z, hold_pulses=n_end, hold_steady=steady,
                       single_soliton_stable=bool(n_end == 1 and steady),
                       hold_comb_power_mW=float(p_h[-1] * 1e3),
                       pulse_fwhm_fs=(fwhm_rad / (2 * np.pi) * nrm["t_R_s"] * 1e15
                                      if fwhm_rad else None),
                       pulse_fwhm_expected_fs=expected_fwhm_fs(nrm, z))
        print(("Single soliton: stable" if summary["single_soliton_stable"] else
               "Single soliton: NOT stable") + f" at zeta = {z:.2f} ({n_end} pulse(s), "
              f"comb power {p_h[-1] * 1e3:.1f} mW)")
    elif nrm:
        print(f"No single-soliton check: zeta_max = {nrm['zeta_max']:.2f} is too close to sqrt(3); "
              "raise the pump (--power-over-threshold).")
    (out / "pylle_summary.json").write_text(json.dumps(summary, indent=2, default=float),
                                           encoding="utf-8")
    print(f"Figures -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
