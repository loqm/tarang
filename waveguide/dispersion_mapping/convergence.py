"""
waveguide.dispersion_mapping.convergence -- show that D, n_eff, n_g are mesh-converged
======================================================================================
    python -m waveguide.dispersion_mapping.convergence --points 0.80x0.80,0.70x1.00 --steps-list 20,30,40,50,70,90 --workers 4
    python -m waveguide.dispersion_mapping.convergence --points 0.80x0.80 --bend-radius 25   # bends too
    python -m waveguide.dispersion_mapping.convergence --plot-only                            # replot

Why this matters
----------------
D = -(lambda/c) d2n_eff/dlambda2 is a SECOND derivative: tiny mesh errors in n_eff(lambda)
get amplified.  TARANG's default mesh (20 steps/lambda in the core, ~39 nm cells) moves
n_eff in the 3rd decimal and D by ~25 %.  This script proves where the numbers stop
changing, which justifies the --steps used for the finalists (70).

Data  : results/mesh_convergence.jsonl        (append-only; finished runs are skipped)
Plots : results/mesh_convergence_plots[_vN]/  (a new _vN folder instead of overwriting)
    conv_H800_W800[_R25um].png  D, n_eff, n_g vs cell size for one geometry
    conv_summary.png            |relative D error| vs cell size, all geometries
    convergence_table.csv       every value + relative error vs the finest mesh

Cell size in the core: dl = lambda / (n_SiN * steps)  (n_SiN ~ 1.996 at 1550 nm)
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

N_SIN_1550 = 1.996


def unique_dir(base: Path, overwrite: bool = False) -> Path:
    base = Path(base)
    if overwrite or not base.exists() or not any(base.iterdir()):
        return base
    k = 2
    while True:
        cand = base.with_name(f"{base.name}_v{k}")
        if not cand.exists() or not any(cand.iterdir()):
            return cand
        k += 1


def _work(job):
    H, W, steps, R, wl, mat = job
    from waveguide.dispersion_mapping import core as sweep_core
    t0 = time.time()
    try:
        r = sweep_core.solve_point(H, W, steps=steps, bend_radius=R, wl0=wl, **mat)
        r.pop("modes", None)
    except Exception as e:
        r = dict(H=H, W=W, steps=steps, bend_radius=R, wl0=wl, error=f"{type(e).__name__}: {e}")
    r["secs"] = round(time.time() - t0, 1)
    return r


def key(r):
    return (round(r["H"], 4), round(r["W"], 4), r.get("bend_radius"), round(r.get("wl0", 1.55), 4),
            r.get("core", "si3n4"), r.get("substrate", "sio2"), r.get("cladding") or None)


def plot(data_path: Path, overwrite: bool, tol_pct: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    groups = defaultdict(dict)
    for line in data_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if "error" not in r:
                groups[key(r)][r["steps"]] = r
    out = unique_dir(data_path.parent / "mesh_convergence_plots", overwrite)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    fig_s, ax_s = plt.subplots(figsize=(8, 5), dpi=140)
    for (H, W, R, wl, *_mat), byS in sorted(groups.items(),
                                           key=lambda kv: (kv[0][2] or 0, kv[0][:2])):
        S = sorted(byS)
        if len(S) < 2:
            continue
        dl = np.array([wl * 1e3 / (N_SIN_1550 * s) for s in S])        # nm
        D = np.array([byS[s]["D_TE0"] for s in S])
        ne = np.array([byS[s]["neff_TE"][0] for s in S])
        ng = np.array([byS[s]["ng_TE0"] for s in S])
        gte = [byS[s]["guided_TE"] for s in S]
        Dref, neref, ngref = D[-1], ne[-1], ng[-1]
        name = f"{H*1e3:.0f}×{W*1e3:.0f} nm" + (f", R = {R:g} µm" if R else " (straight)")
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), dpi=140)
        for ax, y, ref, lab, band in (
                (axs[0], D, Dref, "D(TE00) [ps/(nm·km)]", abs(Dref) * tol_pct / 100),
                (axs[1], ne, neref, "n_eff(TE00)", 1e-3),
                (axs[2], ng, ngref, "n_g(TE00)", 1e-3)):
            ax.axhspan(ref - band, ref + band, color="#b8e6b0", alpha=0.6,
                       label=(f"±{tol_pct:g} % of finest" if ax is axs[0] else "±0.001 of finest"))
            ax.plot(dl, y, "-o", color="k", ms=6)
            for x, v, s in zip(dl, y, S):
                ax.annotate(f"{s}", (x, v), textcoords="offset points", xytext=(4, 6), fontsize=7,
                            color="0.35")
            ax.invert_xaxis()
            ax.set_xlabel("core cell size (nm)   [labels: steps/λ]  → finer")
            ax.set_ylabel(lab); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="best")
        axs[0].axvline(1550 / (N_SIN_1550 * 20), color="crimson", ls="--", lw=1)
        axs[0].text(1550 / (N_SIN_1550 * 20), axs[0].get_ylim()[0], " TARANG\n default",
                    color="crimson", fontsize=7, va="bottom")
        fig.suptitle(f"Mesh convergence — {name}, λ = {wl*1e3:.0f} nm   "
                     f"(guided TE count per mesh: {gte})", fontsize=11)
        fig.tight_layout()
        tag = f"conv_H{H*1e3:.0f}_W{W*1e3:.0f}" + (f"_R{R:g}um" if R else "") + \
              (f"_wl{wl*1e3:.0f}nm" if abs(wl - 1.55) > 1e-9 else "")
        fig.savefig(out / f"{tag}.png"); plt.close(fig)
        err = np.abs(D - Dref) / abs(Dref) * 100
        ax_s.semilogy(dl[:-1], np.maximum(err[:-1], 1e-2), "-o", label=name)
        for s, x, d, n1, n2, e, g in zip(S, dl, D, ne, ng, err, gte):
            rows.append(dict(H_nm=H * 1e3, W_nm=W * 1e3, R_um=R or "", wl_nm=wl * 1e3, steps=s,
                             cell_nm=round(x, 2), D=d, neff=n1, ng=n2,
                             D_rel_err_pct=round(e, 2), dneff=round(n1 - neref, 5),
                             guided_TE=g))
    ax_s.axhline(tol_pct, color="g", ls="--", label=f"{tol_pct:g} % target")
    ax_s.axvline(1550 / (N_SIN_1550 * 20), color="crimson", ls=":", label="TARANG default (20)")
    ax_s.invert_xaxis(); ax_s.grid(alpha=0.3, which="both")
    ax_s.set_xlabel("core cell size (nm)  → finer"); ax_s.set_ylabel("|D − D_finest| / |D_finest| (%)")
    ax_s.set_title("Dispersion error vs mesh (relative to the finest mesh run)", fontsize=10)
    ax_s.legend(fontsize=7); fig_s.tight_layout(); fig_s.savefig(out / "conv_summary.png")
    if rows:
        with open(out / "convergence_table.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"Plots -> {out.resolve()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--core", default="si3n4", help="core medium key in media.json")
    ap.add_argument("--substrate", default="sio2", help="substrate (BOX) medium key")
    ap.add_argument("--cladding", default="air", help='top cladding medium key, or "air"')
    ap.add_argument("--points", default="0.80x0.80,0.70x1.00", help='"HxW,HxW" in um')
    ap.add_argument("--steps-list", default="20,30,40,50,70,90")
    ap.add_argument("--bend-radius", type=float, default=None)
    ap.add_argument("--wl", type=float, default=1.55)
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--data", default="results/mesh_convergence.jsonl")
    ap.add_argument("--tol-pct", type=float, default=2.0, help="acceptance band for D (%%)")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="reuse the plots folder")
    a = ap.parse_args()
    clad = None if str(a.cladding).lower() in ("", "air", "none") else a.cladding
    mat = dict(core=a.core, substrate=a.substrate, cladding=clad)
    data = Path(a.data); data.parent.mkdir(parents=True, exist_ok=True)

    if not a.plot_only:
        pts = [tuple(float(v) for v in t.lower().split("x")) for t in a.points.split(",")]
        steps = [int(s) for s in a.steps_list.split(",")]
        done = set()
        if data.exists():
            for line in data.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    if "error" not in r:
                        done.add(key(r) + (r["steps"],))
        jobs = [(H, W, s, a.bend_radius, a.wl, mat) for (H, W) in pts for s in steps
                if (round(H, 4), round(W, 4), a.bend_radius, round(a.wl, 4),
                    a.core, a.substrate, clad, s) not in done]
        print(f"{len(pts)} geometries x {len(steps)} meshes; {len(jobs)} runs to do "
              f"(finest meshes are slowest: ~1-3 min each at 90 steps).")
        if jobs:
            from waveguide.dispersion_mapping import core as sweep_core  # noqa: F401  (fit/cache materials once)
            with open(data, "a", encoding="utf-8") as f:
                if a.workers <= 1:
                    it = (_work(j) for j in jobs)
                else:
                    pool = ProcessPoolExecutor(max_workers=a.workers)
                    it = (fu.result() for fu in as_completed([pool.submit(_work, j) for j in jobs]))
                for i, r in enumerate(it, 1):
                    f.write(json.dumps(r) + "\n"); f.flush()
                    msg = r.get("error") or f"D={r['D_TE0']:.1f} n_eff={r['neff_TE'][0]:.5f}"
                    print(f"[{i}/{len(jobs)}] H={r['H']} W={r['W']} steps={r['steps']} "
                          f"R={r.get('bend_radius')}  {msg}  ({r['secs']}s)", flush=True)
                if a.workers > 1:
                    pool.shutdown()
    plot(data, a.overwrite, a.tol_pct)


if __name__ == "__main__":
    mp.freeze_support()
    main()
