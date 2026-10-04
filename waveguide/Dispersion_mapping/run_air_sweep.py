"""
run_air_sweep.py -- height x width sweep for AIR-clad SiN ridges on SiO2
=======================================================================
Run from the tarang repo root (Windows PowerShell or any shell):

    python scripts/run_air_sweep.py                                  # default grid
    python scripts/run_air_sweep.py --heights 0.60:0.90:0.05 --widths 0.6:1.6:0.1
    python scripts/run_air_sweep.py --heights 0.70,0.80 --widths 1.0,1.1 --steps 70
    python scripts/run_air_sweep.py --workers 4

Ranges are start:stop:step (stop INCLUDED) or comma lists, in micrometres.
Results are appended line-by-line to results/air_sweep_<steps>steps.jsonl, so you can
stop (Ctrl+C) and rerun the same command: finished points are skipped.

Runtime guide (local Tidy3D mode solver, no FlexCredits): ~20-60 s per point at
--steps 50 on one core; roughly 2x longer at --steps 70.  Default grid = 77 points.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def parse_range(s: str) -> list[float]:
    if ":" in s:
        a, b, st = (float(x) for x in s.split(":"))
        n = int(round((b - a) / st)) + 1
        return [round(a + i * st, 4) for i in range(n)]
    return [round(float(x), 4) for x in s.split(",") if x.strip()]


def _work(args):
    H, W, kw = args
    import waveguide.Dispersion_mapping.sweep_core as sweep_core  # imported inside each worker process

    t0 = time.time()
    try:
        r = sweep_core.solve_point(H, W, **kw)
    except Exception as e:  # record failure, keep the sweep alive
        r = dict(H=H, W=W, steps=kw["steps"], error=f"{type(e).__name__}: {e}")
    r["secs"] = round(time.time() - t0, 1)
    return r


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--heights", default="0.60:0.90:0.05", help="SiN thickness list, um"
    )
    ap.add_argument(
        "--widths", default="0.60:1.60:0.10", help="waveguide width list, um"
    )
    ap.add_argument("--wl", type=float, default=1.55, help="centre wavelength, um")
    ap.add_argument(
        "--steps",
        type=int,
        default=50,
        help="mesh steps/wavelength in core; 50 ~ +/-2%% D, 70+ converged",
    )
    ap.add_argument("--box", type=float, default=4.0, help="BOX thickness, um")
    ap.add_argument(
        "--margin",
        type=float,
        default=0.005,
        help="n_eff margin above n_SiO2 for 'robustly guided'",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, (os.cpu_count() or 2) // 2)),
        help="parallel processes (each needs ~1 GB RAM)",
    )
    ap.add_argument(
        "--points",
        default=None,
        help='finalists "HxW,HxW" in um, e.g. "0.80x0.80,0.70x1.00" (overrides grid)',
    )
    ap.add_argument(
        "--tol",
        default=None,
        help='with --points: fabrication tolerance "dW,dH" in um, e.g. "0.05,0.02"; '
        "runs the 3x3 neighbourhood of every point",
    )
    ap.add_argument(
        "--si-handle",
        action="store_true",
        help="add Si handle below the BOX (with absorbing boundaries)",
    )
    ap.add_argument(
        "--pml",
        type=int,
        default=0,
        help="absorbing layers in the mode plane (needed for loss values)",
    )
    ap.add_argument(
        "--bend-radius",
        type=float,
        default=None,
        help="ring radius in um (bent-waveguide mode solve); omit = straight",
    )
    ap.add_argument("--out", default=None, help="output .jsonl path")
    a = ap.parse_args()

    if a.points:
        pts = []
        for tok in a.points.split(","):
            H0, W0 = (float(v) for v in tok.lower().split("x"))
            if a.tol:
                dW, dH = (float(v) for v in a.tol.split(","))
                pts += [
                    (round(H0 + i * dH, 4), round(W0 + j * dW, 4))
                    for i in (-1, 0, 1)
                    for j in (-1, 0, 1)
                ]
            else:
                pts.append((round(H0, 4), round(W0, 4)))
        pts = list(dict.fromkeys(pts))  # dedupe, keep order
        Hs = sorted({p[0] for p in pts})
        Ws = sorted({p[1] for p in pts})
    else:
        Hs, Ws = parse_range(a.heights), parse_range(a.widths)
        pts = [(H, W) for H in Hs for W in Ws]
    tag = f"air_sweep_{a.steps}steps"
    if abs(a.wl - 1.55) > 1e-9:  # different wavelength -> different file
        tag += f"_wl{a.wl * 1e3:.0f}nm"
    if a.si_handle:
        tag += "_si"
    if a.bend_radius is not None:
        tag += f"_R{a.bend_radius:g}um"
    if a.points:
        tag += "_finalists"
    out = Path(a.out) if a.out else Path("results") / f"{tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                same = (
                    r.get("steps") == a.steps
                    and abs(r.get("wl0", 1.55) - a.wl) < 1e-9
                    and r.get("bend_radius") == a.bend_radius
                    and bool(r.get("si_handle", False)) == a.si_handle
                )
                if "error" not in r and same:
                    done.add((round(r["H"], 4), round(r["W"], 4)))

    kw = dict(
        wl0=a.wl,
        steps=a.steps,
        box=a.box,
        margin=a.margin,
        si_handle=a.si_handle,
        pml=a.pml,
        bend_radius=a.bend_radius,
    )
    todo = [(H, W, kw) for (H, W) in pts if (H, W) not in done]
    print(
        f"{len(pts)} points requested; "
        f"{len(done)} already done, {len(todo)} to run with {a.workers} worker(s)."
    )
    print(f"Output: {out.resolve()}")
    if not todo:
        return

    # Fit/cache the materials ONCE in the parent so workers only read the cache
    import waveguide.Dispersion_mapping.sweep_core as sweep_core  # noqa: F401

    t_start = time.time()
    with open(out, "a", encoding="utf-8") as f:
        if a.workers <= 1:
            iterator = (_work(t) for t in todo)
        else:
            pool = ProcessPoolExecutor(max_workers=a.workers)
            iterator = (
                fut.result()
                for fut in as_completed([pool.submit(_work, t) for t in todo])
            )
        for i, r in enumerate(iterator, 1):
            f.write(json.dumps(r) + "\n")
            f.flush()
            if "error" in r:
                msg = f"ERROR {r['error']}"
            else:
                msg = (
                    f"D_TE0={r['D_TE0']:8.1f}  guided TE/TM={r['guided_TE']}/{r['guided_TM']}"
                    f"  n_TE0={r['neff_TE'][0]:.4f}  n_TM0={r['neff_TM'][0]:.4f}"
                )
            eta = (time.time() - t_start) / i * (len(todo) - i)
            print(
                f"[{i:3d}/{len(todo)}] H={r['H']:.3f} W={r['W']:.3f}  {msg}  "
                f"({r['secs']:.0f}s, ETA {eta/60:.0f} min)",
                flush=True,
            )
        if a.workers > 1:
            pool.shutdown()
    print("Done. Now run:  python scripts/plot_air_sweep.py --data", out)


if __name__ == "__main__":
    mp.freeze_support()  # required for multiprocessing on Windows
    main()
