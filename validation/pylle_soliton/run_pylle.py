"""Proof of soliton formation for a TARANG design, with pyLLE (not part of the package).

TARANG designs the resonator and writes everything a Lugiato-Lefever simulation needs into
``<design>/lle/`` (``lle_inputs.json`` and ``pyLLE_dispersion.csv``). This script hands
those to pyLLE (G. Moille, NIST; https://github.com/gregmoille/pyLLE), sweeps the pump
detuning across the resonance and saves the figures of a soliton comb:

* ``pylle_power_vs_step``: intracavity power along the detuning sweep (the soliton steps),
* ``pylle_spectrum``: the comb spectrum at the chosen step (sech^2 envelope, dispersive waves),
* ``pylle_temporal``: the intracavity intensity around the ring at that step,
* ``pylle_dint``: D_int as pyLLE reads it from TARANG's resonance frequencies.

Each figure is saved with its numbers (``_data.csv``) and settings (``_meta.json``) by
``tarang.figures``, in ``<design>/validation/pylle/``.

Requirements (once): Julia >= 1.8 on the PATH (https://julialang.org/downloads/) and
``pip install pyLLE``. pyLLE installs its Julia packages itself on Linux and macOS; on
Windows install them first, in Julia:  ``import Pkg; Pkg.add(["HDF5", "FFTW", "LinearAlgebra",
"ProgressMeter"])``.

Run:
    python validation/pylle_soliton/run_pylle.py design_runs/<name> --power-mw 200
    python validation/pylle_soliton/run_pylle.py design_runs/<name> --dry-run   # check inputs only

Physics of the defaults: the pump power is a multiple of the threshold TARANG computed for
the over-coupled gap (``--power-over-threshold``, default 4; solitons need a few times the
threshold), the detuning sweeps from 2 linewidths blue to 12 linewidths red of the cold
resonance (the soliton existence range grows with power), and the simulated modes cover
``--modes`` either side of the pump.
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
    """A step on the single-soliton plateau: past the chaotic maximum, the last step where
    the power is still above a tenth of that maximum (before the soliton is lost)."""
    import numpy as np
    p = np.asarray(pcomb, float)
    k = int(np.argmax(p))
    after = np.where(p[k:] > 0.1 * p[k])[0]
    return int(k + after[-1] - 5) if len(after) > 10 else int(len(p) * 0.8)


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
    ap.add_argument("--julia", default=None, help="path to the julia executable")
    ap.add_argument("--dry-run", action="store_true", help="check and print the inputs only")
    a = ap.parse_args(argv)

    inp = load_inputs(a.design)
    res, sim = settings(inp, a.power_mw, a.power_over_threshold, a.modes, a.steps)
    print("pyLLE resonator:", {k: v for k, v in res.items()})
    print(f"pump {sim['Pin'][0] * 1e3:.0f} mW at {sim['f_pmp'][0] / 1e12:.3f} THz, "
          f"modes ±{a.modes}, detuning {sim['δω_init'] / 2 / math.pi / 1e6:.0f} -> "
          f"{sim['δω_end'] / 2 / math.pi / 1e6:.0f} MHz")
    if a.dry_run:
        return 0
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

    solver = pyLLE.LLEsolver(sim=sim, res=res, debug=False)
    solver.Analyze(plot=False)
    solver.Setup(verbose=False)
    if a.julia:
        solver.SolveTemporal(bin=a.julia)
    else:
        solver.SolveTemporal()
    solver.RetrieveData()
    sol = solver.sol
    pcomb = np.asarray(sol.Pcomb, float)
    k = a.step if a.step is not None else soliton_step(pcomb)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(len(pcomb)), pcomb * 1e3, lw=1.2, label="intracavity comb power")
    ax.axvline(k, color="crimson", ls="--", lw=1, label=f"step {k} (shown)")
    ax.set(xlabel="step of the detuning sweep (blue to red)", ylabel="comb power (mW)")
    ax.legend(fontsize=8)
    figures.save(fig, out / "pylle_power_vs_step.png", meta=meta)

    spec = np.abs(np.asarray(sol.Ewg)[:, k]) ** 2 if hasattr(sol, "Ewg") else None
    freq = np.asarray(getattr(sol, "freq", []), float)
    if spec is not None and len(freq) == len(spec):
        fig, ax = plt.subplots(figsize=(7, 4))
        db = 10 * np.log10(np.maximum(spec / spec.max(), 1e-12))
        ax.vlines(299792458.0 / freq * 1e9, -200, db, lw=0.8)
        ax.set(xlabel="wavelength (nm)", ylabel="power in the bus (dB, rel.)",
               ylim=(-100, 5))
        figures.save(fig, out / "pylle_spectrum.png", meta=meta)
    acav = np.asarray(getattr(sol, "Acav", np.empty((0, 0))))
    if acav.size:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        theta = np.linspace(-np.pi, np.pi, acav.shape[0])
        ax.plot(theta, np.abs(acav[:, k]) ** 2, lw=1.2)
        ax.set(xlabel="angle around the ring (rad)", ylabel="intracavity intensity (arb.)")
        figures.save(fig, out / "pylle_temporal.png", meta=meta)
    print(f"Figures -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
