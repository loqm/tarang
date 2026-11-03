"""Why does D of the 780 nm design (350 x 500 nm) move by +/-37 ps/(nm km) between meshes
while the 1550 nm design (700 x 1000 nm) stays within about +/-8?

D comes from the second derivative of n_eff(lambda): solve_point fits a 4th-order polynomial
to n_eff at ``npts`` wavelengths over ``span`` um around the pump. The span is 0.2 um at
every wavelength, i.e. +/-6.5 % of 1550 nm but +/-13 % of 780 nm, where the material and
the mode change much faster. Two things can make D jump between meshes:

1. the fit: over a wide relative band a quartic leaves a systematic residual (2.7e-7 in
   n_eff at 780 nm against 4.6e-9 at 1550 nm in the paper runs), and how that residual is
   shared between the polynomial's terms can change with small changes of n_eff(lambda);
2. the mesh itself: where the graded cells fall around the thin core.

This study separates them. For each mesh (steps per wavelength in the core) it solves the
same cross-section with the paper setting (span 0.2 um, 9 points) and with the band scaled
to the wavelength (span 0.1 um at 780 nm, the same +/-6.5 % as 0.2 um at 1550 nm), and with
more points. If D settles with the scaled band, the fix is the band (cheap, free); if it
still jumps, the mesh is the cause.

Local mode solves only (free). About 10-20 min per solve at 780 nm on a laptop; with 4
workers the default 15 solves take 1-1.5 h. Results are appended to the --out file, so a
stopped run continues where it left off.

    python validation/mesh_study/mesh_study.py                      # 780 nm paper design
    python validation/mesh_study/mesh_study.py --wl 1.55 --h 0.7 --w 1.0 --out mesh_1550.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

CASES = [  # (label, span_um at 780 nm scaled with the wavelength, npts)
    ("paper: 0.2 um, 9 pts", 0.2, 9),
    ("scaled: 0.1 um, 9 pts", 0.1, 9),
    ("scaled: 0.1 um, 13 pts", 0.1, 13),
]


def _one(args):
    H, W, wl, steps, span, npts, label = args
    from tarang.waveguide.dispersion_mapping import core
    t0 = time.time()
    r = core.solve_point(H, W, wl0=wl, steps=steps, span=span, npts=npts, snap=True)
    keep = {k: r[k] for k in ("D_TE0", "ng_TE0", "ng_TE0_fine", "fit_resid_TE0", "neff_TE")}
    return dict(H=H, W=W, wl=wl, steps=steps, span=span, npts=npts, case=label,
                secs=round(time.time() - t0, 1), **keep)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wl", type=float, default=0.78)
    ap.add_argument("--h", type=float, default=0.35)
    ap.add_argument("--w", type=float, default=0.5)
    ap.add_argument("--steps", default="50,60,70,80,90")
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--out", default="mesh_study_780.jsonl")
    a = ap.parse_args()
    scale = a.wl / 0.78
    jobs = [(a.h, a.w, a.wl, int(s), round(span * scale, 4), n, label)
            for s in a.steps.split(",") for label, span, n in CASES]
    out = Path(a.out)
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["steps"], r["span"], r["npts"]))
    todo = [j for j in jobs if (j[3], j[4], j[5]) not in done]
    print(f"{len(jobs)} solves, {len(jobs) - len(todo)} already done, {len(todo)} to run "
          f"with {a.workers} worker(s) -> {out.resolve()}")
    from tarang.waveguide.dispersion_mapping import core  # noqa: F401  (fit materials once)
    with ProcessPoolExecutor(max_workers=a.workers) as ex, open(out, "a", encoding="utf-8") as f:
        futs = [ex.submit(_one, j) for j in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"[{i:2d}/{len(todo)}] steps {r['steps']:3d}  {r['case']:24s} D = {r['D_TE0']:8.2f}"
                  f"  n_g = {r['ng_TE0']:.4f}  ({r['secs']:.0f} s)")
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
    print("\nD_TE0 [ps/(nm km)] against the mesh:")
    labels = sorted({r["case"] for r in rows}, key=lambda c: [x[0] for x in CASES].index(c)
                    if c in [x[0] for x in CASES] else 99)
    steps = sorted({r["steps"] for r in rows})
    print(f"  {'case':24s}" + "".join(f"{s:>9d}" for s in steps) + "   median +/- spread")
    for c in labels:
        Ds = {r["steps"]: r["D_TE0"] for r in rows if r["case"] == c}
        vals = [Ds[s] for s in steps if s in Ds]
        med = sorted(vals)[len(vals) // 2] if vals else float("nan")
        spread = max(abs(v - med) for v in vals) if vals else float("nan")
        print(f"  {c:24s}" + "".join(f"{Ds[s]:9.1f}" if s in Ds else f"{'-':>9s}" for s in steps)
              + f"   {med:.1f} +/- {spread:.1f}")


if __name__ == "__main__":
    main()
