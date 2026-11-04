"""
tarang.waveguide.dispersion_mapping.convergence -- show that D, n_eff, n_g are mesh-converged
======================================================================================
    python -m tarang.waveguide.dispersion_mapping.convergence --points 0.80x0.80,0.70x1.00 --steps-list 20,30,40,50,70,90 --workers 4
    python -m tarang.waveguide.dispersion_mapping.convergence --points 0.80x0.80 --bend-radius 25   # bends too
    python -m tarang.waveguide.dispersion_mapping.convergence --plot-only                            # replot

Why this matters
----------------
D = -(lambda/c) d2n_eff/dlambda2 is a SECOND derivative: tiny mesh errors in n_eff(lambda)
get amplified. At 20 steps/lambda (~39 nm cells) n_eff moves in the 3rd decimal and D by
~25 %. Finer meshes do not converge monotonically either: n_eff jitters by ~1e-3 from grid
to grid (where the core edges fall in their cells; with --snap, grid lines go through the
edges, which removes most of it), and D by a few ps/(nm km). So D is reported as the median
over a ladder of meshes with the largest distance of any mesh from it (``mesh_band``).

Measured for 700 x 1000 nm air-clad SiN at 1550 nm, 50/60/70/80/90/100 steps:
    not snapped: 378.7, 373.6, 361.6, 372.5, 374.6        -> 373.6 +/- 12.0
    snapped    : 374.0, 374.0, 376.8, 373.8, 370.9, 373.7 -> 373.9 +/- 3.0

Data  : results/mesh_convergence.jsonl        (append-only; finished runs are skipped)
Plots : results/mesh_convergence_plots[_vN]/  (a new _vN folder instead of overwriting)
    conv_H800_W800[_R25um].png  D, n_eff, n_g vs cell size, with median +/- spread
    conv_summary.png            distance of D from its median vs cell size, all geometries
    convergence_table.csv       every value, its distance from the median (D) and the finest (n_eff)
    mesh_band.csv               per geometry: median and spread of D, n_eff and n_g

Cell size in the core: dl = lambda / (n_core(lambda) * steps)
"""
from __future__ import annotations

from tarang.figures import save as save_fig

from tarang.waveguide.dispersion_mapping._console import safe_console

import argparse
import csv
import json
import multiprocessing as mp
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
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
    H, W, steps, R, wl, mat = job      # mat holds the materials and snap
    from tarang.waveguide.dispersion_mapping import core as sweep_core
    t0 = time.time()
    try:
        r = sweep_core.solve_point(H, W, steps=steps, bend_radius=R, wl0=wl, **mat)
        r.pop("modes", None)
    except Exception as e:
        r = dict(H=H, W=W, steps=steps, bend_radius=R, wl0=wl, error=f"{type(e).__name__}: {e}")
    r["secs"] = round(time.time() - t0, 1)
    return r


def _core_index(core: str, wl: float) -> float:
    """Core index at ``wl`` for the cell size on the plots (falls back to SiN at 1550 nm)."""
    try:
        from tarang.waveguide.dispersion_mapping import core as sc
        return float(sc.n_of(sc.medium_for(core, wl - 0.05, wl + 0.05), wl))
    except Exception:                                         # noqa: BLE001
        return N_SIN_1550


def mesh_band(values: dict) -> dict:
    """A quantity over the mesh ladder ({steps: value}): the median and the largest distance
    of any mesh from it. The mode solver's n_eff jitters by about 1e-3 from grid to grid even
    when the grid is refined (where the grading from fine to coarse cells falls changes), so
    D does not converge monotonically: the median of several meshes with this band is the
    honest number, not the value at any single mesh."""
    import numpy as np
    S = sorted(values)
    v = np.array([float(values[s]) for s in S])
    med = float(np.median(v))
    return dict(value=med, band=float(np.max(np.abs(v - med))) if len(v) > 1 else None,
                meshes=S)


def snap_sides_of(r):
    from tarang.waveguide.dispersion_mapping.snapping import snap_sides_of as f
    return f(r)


def key(r):
    return (round(r["H"], 4), round(r["W"], 4), r.get("bend_radius"), round(r.get("wl0", 1.55), 4),
            r.get("core", "si3n4"), r.get("substrate", "sio2"), r.get("cladding") or None,
            bool(r.get("snap", False)), snap_sides_of(r))


def plot(data_path: Path, overwrite: bool, tol_pct: float, plots_dir=None, pump="TE00"):
    """Plots and tables of the mesh check, for the pump mode ``pump`` (TE00 or TM00)."""
    from tarang.waveguide.dispersion_mapping.core import PUMP_FIELDS, pol_of
    F = PUMP_FIELDS[pol_of(pump)]
    pump = "TE00" if pol_of(pump) == "TE" else "TM00"
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
    out = unique_dir(Path(plots_dir) if plots_dir else data_path.parent / "mesh_convergence_plots", overwrite)
    out.mkdir(parents=True, exist_ok=True)
    rows, summary = [], []
    fig_s, ax_s = plt.subplots(figsize=(8, 5), dpi=140)
    for (H, W, R, wl, *_mat), byS in sorted(groups.items(),
                                           key=lambda kv: (kv[0][2] or 0, kv[0][:2])):
        S = sorted(byS)
        if len(S) < 2:
            continue
        n_c = _core_index(_mat[0] if _mat else "si3n4", wl)
        dl = np.array([wl * 1e3 / (n_c * s) for s in S])               # nm
        D = np.array([byS[s][F["D"]] for s in S])
        ne = np.array([byS[s][F["neff"]][0] for s in S])
        ng = np.array([byS[s][F["ng"]] for s in S])
        gte = [byS[s][F["guided"]] for s in S]
        Dref, neref, ngref = D[-1], ne[-1], ng[-1]
        bands = {q: mesh_band(dict(zip(S, y))) for q, y in (("D", D), ("neff", ne), ("ng", ng))}
        name = f"{H*1e3:.0f}×{W*1e3:.0f} nm" + (f", R = {R:g} µm" if R else " (straight)")
        snapped = bool(_mat[-1]) if _mat else False
        label = name + ("" if snapped else ", not snapped")      # in the summary's legend
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), dpi=140)
        for ax, y, q, lab, fmt in (
                (axs[0], D, "D", f"D({pump}) [ps/(nm·km)]", "{:.1f}"),
                (axs[1], ne, "neff", f"n_eff({pump})", "{:.5f}"),
                (axs[2], ng, "ng", f"n_g({pump})", "{:.4f}")):
            b = bands[q]
            ax.axhspan(b["value"] - (b["band"] or 0), b["value"] + (b["band"] or 0),
                       color="#b8e6b0", alpha=0.6)
            ax.axhline(b["value"], color="#2e7d32", lw=1,
                       label=("median ± spread: " + fmt.format(b["value"]) + " ± "
                              + fmt.format(b["band"] or 0)))
            ax.plot(dl, y, "-o", color="k", ms=6)
            for x, v, s in zip(dl, y, S):
                ax.annotate(f"{s}", (x, v), textcoords="offset points", xytext=(4, 6), fontsize=7,
                            color="0.35")
            ax.ticklabel_format(axis="y", useOffset=False)      # plain values, no "+3.74e2"
            ax.invert_xaxis()
            ax.set_xlabel("core cell size (nm)   [labels: steps/λ]  → finer")
            ax.set_ylabel(lab); ax.grid(alpha=0.3)
            ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
        fig.suptitle(f"Mesh convergence — {name}, λ = {wl*1e3:.0f} nm, grid "
                     f"{'snapped to the core edges' if snapped else 'not snapped'}   "
                     f"(guided {pump[:2]} count per mesh: {gte})", fontsize=11)
        summary.append(dict(H_nm=H * 1e3, W_nm=W * 1e3, R_um=R or "", wl_nm=wl * 1e3,
                            pump=pump, snapped=snapped, meshes=" ".join(map(str, S)),
                            D=round(bands["D"]["value"], 2), D_band=round(bands["D"]["band"], 2),
                            neff=round(bands["neff"]["value"], 5),
                            neff_band=round(bands["neff"]["band"], 5),
                            ng=round(bands["ng"]["value"], 5), ng_band=round(bands["ng"]["band"], 5)))
        fig.tight_layout()
        tag = f"conv_H{H*1e3:.0f}_W{W*1e3:.0f}" + (f"_R{R:g}um" if R else "") + \
              (f"_wl{wl*1e3:.0f}nm" if abs(wl - 1.55) > 1e-9 else "") + \
              ("" if snapped else "_unsnapped") + ("" if pump == "TE00" else "_TM00")
        save_fig(fig, out / f"{tag}.png", layout=False)
        Dmed = bands["D"]["value"]
        err = np.abs(D - Dmed) / abs(Dmed) * 100
        ax_s.semilogy(dl, np.maximum(err, 1e-2), "-o", label=label)
        for s, x, d, n1, n2, e, g in zip(S, dl, D, ne, ng, err, gte):
            rows.append(dict(H_nm=H * 1e3, W_nm=W * 1e3, R_um=R or "", wl_nm=wl * 1e3,
                             snapped=snapped, steps=s, cell_nm=round(x, 2), D=d, neff=n1, ng=n2,
                             D_from_median_pct=round(e, 2), dneff_from_finest=round(n1 - neref, 5),
                             pump=pump, guided=g))
    ax_s.axhline(tol_pct, color="g", ls="--", label=f"{tol_pct:g} % target")
    ax_s.invert_xaxis(); ax_s.grid(alpha=0.3, which="both")
    ax_s.set_xlabel("core cell size (nm)  → finer"); ax_s.set_ylabel("|D − D_median| / |D_median| (%)")
    ax_s.set_title("Distance of D from its median over the mesh ladder", fontsize=10)
    ax_s.legend(fontsize=7); fig_s.tight_layout(); save_fig(fig_s, out / "conv_summary.png", layout=False)
    if rows:
        with open(out / "convergence_table.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    if summary:
        with open(out / "mesh_band.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    print(f"Plots -> {out.resolve()}")


def main():
    safe_console()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--core", default="si3n4", help="core medium key in media.json")
    ap.add_argument("--substrate", default="sio2", help="substrate (BOX) medium key")
    ap.add_argument("--cladding", default="air", help='top cladding medium key, or "air"')
    ap.add_argument("--box", type=float, default=4.0, help="BOX thickness, um")
    ap.add_argument("--cladding-height", type=float, default=2.0, help="top cladding thickness, um")
    ap.add_argument("--points", default="0.80x0.80,0.70x1.00", help='"HxW,HxW" in um')
    ap.add_argument("--steps-list", default="20,30,40,50,70,90")
    ap.add_argument("--bend-radius", type=float, default=None)
    ap.add_argument("--wl", type=float, default=1.55)
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--data", default="results/mesh_convergence.jsonl")
    ap.add_argument("--tol-pct", type=float, default=2.0, help="acceptance band for D (%%)")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="reuse the plots folder")
    ap.add_argument("--plots-dir", default=None,
                    help="write the plots here (default: mesh_convergence_plots next to the data)")
    ap.add_argument("--pump", default="TE00", choices=["TE00", "TM00"],
                    help="the mode whose D, n_eff and n_g are checked")
    ap.add_argument("--snap", action="store_true",
                    help="grid lines through the four core edges (steadier D from mesh to mesh)")
    ap.add_argument("--snap-sides", type=int, default=2, choices=[1, 2],
                    help="with --snap: 2 (default) averages two solves with the lines just "
                         "inside and just outside the core's faces (same result on every "
                         "computer); 1 reproduces results before v0.1.5")
    a = ap.parse_args()
    clad = None if str(a.cladding).lower() in ("", "air", "none") else a.cladding
    mat = dict(core=a.core, substrate=a.substrate, cladding=clad, box=a.box,
               cladding_height=a.cladding_height, snap=a.snap, snap_sides=a.snap_sides)
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
                    a.core, a.substrate, clad, a.snap,
                    (a.snap_sides if a.snap else 1), s) not in done]
        print(f"{len(pts)} geometries x {len(steps)} meshes; {len(jobs)} runs to do "
              f"(finest meshes are slowest: ~1-3 min each at 90 steps).")
        if jobs:
            from tarang.waveguide.dispersion_mapping import core as sweep_core  # noqa: F401  (fit/cache materials once)
            with open(data, "a", encoding="utf-8") as f:
                saved: set = set()

                def record(i, r):
                    f.write(json.dumps(r) + "\n"); f.flush()
                    saved.add((r["H"], r["W"], r["steps"]))
                    msg = r.get("error") or f"D_TE0={r['D_TE0']:.1f} n_TE0={r['neff_TE'][0]:.5f}"
                    print(f"[{i}/{len(jobs)}] H={r['H']} W={r['W']} steps={r['steps']} "
                          f"R={r.get('bend_radius')}  {msg}  ({r['secs']}s)", flush=True)

                if a.workers <= 1:
                    for i, j in enumerate(jobs, 1):
                        record(i, _work(j))
                else:
                    pool = ProcessPoolExecutor(max_workers=a.workers)
                    i = 0
                    try:
                        for fu in as_completed([pool.submit(_work, j) for j in jobs]):
                            i += 1
                            record(i, fu.result())
                        pool.shutdown()
                    except BrokenProcessPool:
                        # a worker was killed (nearly always out of memory): finish one at a time
                        pool.shutdown(wait=False, cancel_futures=True)
                        rest = [j for j in jobs if (j[0], j[1], j[2]) not in saved]
                        print(f"A worker process was killed (usually out of memory). Going on "
                              f"with the other {len(rest)} run(s) one at a time.", flush=True)
                        for j in rest:
                            i += 1
                            record(i, _work(j))
    plot(data, a.overwrite, a.tol_pct, a.plots_dir, a.pump)


if __name__ == "__main__":
    mp.freeze_support()
    main()
