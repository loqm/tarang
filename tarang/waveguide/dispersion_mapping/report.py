"""
tarang.waveguide.dispersion_mapping.report -- compare candidate geometries against a baseline
=====================================================================================
    python -m tarang.waveguide.dispersion_mapping.report --straight results/air_sweep_70steps_finalists.jsonl --bends "results/air_sweep_70steps_R*um_finalists.jsonl" --finalists 0.80x0.80,0.80x1.00,0.70x1.00,0.65x1.10 --tol 0.05,0.02

The FIRST finalist is the baseline (your initial 800 x 800 nm selection); every other
geometry is reported relative to it.  Writes results/finalist_report/:
  report.md                 tables + automatic verdicts (paste into slides / notes)
  finalists_comparison.png  4-panel figure
  bend_dispersion.png       D_bend, dD vs straight, D2/2pi and TE/TM splitting vs R
  finalists.csv             all numbers

What is compared
----------------
Straight waveguide (at 1550 nm):
  D, beta2, n_g, guided TE/TM counts
  TE/TM splitting  |n_TE0 - n_TM0|      (small -> TE/TM hybridise in bends)
  TE1 / TM1 cutoff margin  n_cut - n_1  (positive = 2nd-order mode NOT guided)
  Fabrication tolerance: D and regime at W +/- dW, H +/- dH (worst case over 3x3)
    - exact if the runner was used with --tol; otherwise interpolated from the grid
Bent waveguide (per radius R, from --bend-radius runs):
  D_bend, FSR = c / (2 pi n_g R), D2/2pi = -(c/n_g) D1^2 beta2 / 2pi,
  bend + substrate radiation loss (from Im n_eff),
  Q_i = 2 pi n_g / (lambda alpha_total),  alpha_total = alpha_scat + alpha_abs + alpha_bend
  P_th ~ 1.54 (pi/2) n_eff^2 V / (n2 lambda Q_L^2),  critical coupling Q_L = Q_i / 2,
         V = A_eff * 2 pi R            (Kippenberg et al.; comparative, +/- factor ~2)

alpha_scat / alpha_abs defaults (0.3 / 0.1 dB/cm, i.e. Q_i ~ 1e6) are placeholders
typical of air-clad LPCVD SiN rings; replace with your cutback / ring measurements.
"""
from __future__ import annotations

from tarang.waveguide.dispersion_mapping._console import safe_console

import argparse
import csv
import glob
import json
import math
from pathlib import Path

import numpy as np

C = 299792458.0

def unique_dir(base: Path, overwrite: bool = False) -> Path:
    """Never overwrite earlier results: if `base` exists and is non-empty, return
    base_v2, base_v3, ... (first free one).  --overwrite restores the old behaviour."""
    base = Path(base)
    if overwrite or not base.exists() or not any(base.iterdir()):
        return base
    k = 2
    while True:
        cand = base.with_name(f"{base.name}_v{k}")
        if not cand.exists() or not any(cand.iterdir()):
            return cand
        k += 1



def load(paths):
    rows = {}
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if "error" not in r:
                    rows[(round(r["H"], 4), round(r["W"], 4), r.get("bend_radius"))] = r
    return rows


def classify(r, dmargin, deg_tol):
    if r["D_TE0"] < 0:
        return "NORMAL"
    if r["D_TE0"] < dmargin:
        return "MARGINAL"
    if r["guided_TE"] != 1:
        return "MULTIMODE"
    if abs(r["neff_TE"][0] - r["neff_TM"][0]) < deg_tol:
        return "DEGENERATE"
    return "TARGET"


def interp_D(straight, H, W):
    """D at (H, W): exact if computed, else linear interpolation along W, then H."""
    key = (round(H, 4), round(W, 4), None)
    if key in straight:
        return straight[key]["D_TE0"], "exact"

    def along_W(h):
        pts = sorted((k[1], v["D_TE0"]) for k, v in straight.items() if k[0] == round(h, 4))
        xs = [p[0] for p in pts]
        if len(xs) >= 2 and xs[0] <= W <= xs[-1]:
            return float(np.interp(W, xs, [p[1] for p in pts]))
        return None
    d = along_W(H)
    if d is not None:
        return d, "interp"
    hs = sorted({k[0] for k in straight})
    lo = max([h for h in hs if h <= H], default=None)
    hi = min([h for h in hs if h >= H], default=None)
    if lo is not None and hi is not None:
        dl, dh = along_W(lo), along_W(hi)
        if dl is not None and dh is not None:
            return (dl if hi == lo else dl + (dh - dl) * (H - lo) / (hi - lo)), "interp"
    return float("nan"), "missing"


def bend_metrics(r, a):
    R = r["bend_radius"] * 1e-6
    ng = r["ng_TE0"]
    beta2 = r["beta2_TE0"] * 1e-27            # ps^2/km -> s^2/m
    D1 = C / (ng * R)                          # rad/s
    fsr = D1 / (2 * math.pi)
    D2 = -(C / ng) * D1 ** 2 * beta2           # rad/s
    a_bend = r.get("loss_TE0_dB_cm", 0.0)
    a_tot = a.alpha_scat + a.alpha_abs + a_bend
    alpha = a_tot * 100 / (10 * math.log10(math.e))       # 1/m (power)
    lam = r["wl0"] * 1e-6
    Qi = 2 * math.pi * ng / (lam * alpha)
    QL = Qi / 2
    V = r["Aeff_TE0_um2"] * 1e-12 * 2 * math.pi * R
    n = r["neff_TE"][0]
    Pth = 1.54 * (math.pi / 2) * n ** 2 * V / (a.n2 * lam * QL ** 2)
    return dict(R_um=r["bend_radius"], D_bend=r["D_TE0"], FSR_GHz=fsr / 1e9,
                D2_2pi_MHz=D2 / (2 * math.pi) / 1e6, bend_loss_dB_cm=a_bend,
                Qi=Qi, Pth_mW=Pth * 1e3,
                dn_TE_TM=r["neff_TE"][0] - r["neff_TM"][0],
                steps=r["steps"])


def tol_text(r):
    t = r["all_neighbours_target"]
    if t is None:
        return "not computed (run with --tol)"
    return "yes" if t else "NO: " + r["neighbour_classes"]


def main():
    safe_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--straight", nargs="+", required=True,
                    help="straight-waveguide jsonl file(s); grid and/or finalists runs")
    ap.add_argument("--bends", default=None, help='glob, e.g. "results/*_R*um_finalists.jsonl"')
    ap.add_argument("--finalists", required=True, help="HxW list in um; first = baseline")
    ap.add_argument("--tol", default="0.05,0.02", help="dW,dH in um")
    ap.add_argument("--dmargin", type=float, default=0.0,
                    help="optional user safety margin on D [ps/(nm km)]; 0 (default) = none")
    ap.add_argument("--deg-tol", dest="deg_tol", type=float, default=0.01)
    ap.add_argument("--alpha-scat", type=float, default=0.3, help="dB/cm")
    ap.add_argument("--alpha-abs", type=float, default=0.1, help="dB/cm")
    ap.add_argument("--n2", type=float, default=2.4e-19, help="Kerr index of the core, m^2/W (default: Si3N4)")
    ap.add_argument("--out", default="results/finalist_report")
    ap.add_argument("--overwrite", action="store_true",
                    help="reuse --out instead of creating _v2, _v3, ...")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    straight = {k: v for k, v in load(a.straight).items() if k[2] is None}
    bends = load(glob.glob(a.bends)) if a.bends else {}
    dW, dH = (float(x) for x in a.tol.split(","))
    fins = [tuple(float(v) for v in t.lower().split("x")) for t in a.finalists.split(",")]
    out = unique_dir(Path(a.out), a.overwrite); out.mkdir(parents=True, exist_ok=True)
    print(f"Writing report to {out.resolve()}")

    rows, bend_rows = [], []
    for (H, W) in fins:
        key = (round(H, 4), round(W, 4), None)
        if key not in straight:
            raise SystemExit(f"Finalist {H}x{W} missing from --straight data; run it first.")
        r = straight[key]
        cut = r["n_cut"]
        te1 = r["neff_TE"][1] if len(r["neff_TE"]) > 1 else float("nan")
        tm1 = r["neff_TM"][1] if len(r["neff_TM"]) > 1 else float("nan")
        # tolerance neighbourhood (3x3)
        Ds, srcs, worst_cls = [], set(), []
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                d, src = interp_D(straight, H + i * dH, W + j * dW)
                Ds.append(d); srcs.add(src)
                nb = straight.get((round(H + i * dH, 4), round(W + j * dW, 4), None))
                # Mode counts / degeneracy cannot be interpolated honestly -> UNKNOWN
                worst_cls.append(classify(nb, a.dmargin, a.deg_tol) if nb else "UNKNOWN")
        dplus, _ = interp_D(straight, H, W + dW); dminus, _ = interp_D(straight, H, W - dW)
        hplus, _ = interp_D(straight, H + dH, W); hminus, _ = interp_D(straight, H - dH, W)
        rows.append(dict(
            name=f"{H*1e3:.0f}x{W*1e3:.0f}", H_nm=H * 1e3, W_nm=W * 1e3,
            D=r["D_TE0"], beta2=r["beta2_TE0"], ng=r["ng_TE0"], steps=r["steps"],
            guided_TE=r["guided_TE"], guided_TM=r["guided_TM"],
            n_TE0=r["neff_TE"][0], n_TM0=r["neff_TM"][0],
            dn_TE_TM=r["neff_TE"][0] - r["neff_TM"][0],
            TE1_margin=cut - te1, TM1_margin=cut - tm1,
            D_min_tol=np.nanmin(Ds), D_max_tol=np.nanmax(Ds),
            dD_dW=(dplus - dminus) / (2 * dW * 1e3), dD_dH=(hplus - hminus) / (2 * dH * 1e3),
            class_center=classify(r, a.dmargin, a.deg_tol),
            all_neighbours_target=(None if "UNKNOWN" in worst_cls
                                   else all(c == "TARGET" for c in worst_cls)),
            neighbour_classes=",".join(sorted(set(worst_cls))),
            tol_source="+".join(sorted(srcs)), Aeff=r.get("Aeff_TE0_um2")))
        for k, b in sorted(bends.items(), key=lambda kv: kv[0][2]):
            if k[0] == round(H, 4) and k[1] == round(W, 4):
                bm = bend_metrics(b, a); bm["name"] = rows[-1]["name"]
                if b["steps"] != r["steps"]:
                    print(f"WARNING {rows[-1]['name']} R={b['bend_radius']}: bend run at "
                          f"{b['steps']} steps vs straight at {r['steps']}; dD_vs_straight "
                          "mixes mesh error with bend effect -- use equal --steps.")
                bm["dD_vs_straight"] = bm["D_bend"] - r["D_TE0"]
                bend_rows.append(bm)

    base = rows[0]
    r0 = next(iter(straight.values()))
    stack = (f"{r0.get('core', 'si3n4')} core / {r0.get('cladding') or 'air'} top / "
             f"{r0.get('box', 4.0):g} µm {r0.get('substrate', 'sio2')}")
    # ---------------- CSV
    with open(out / "finalists.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    if bend_rows:
        with open(out / "finalists_bend.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(bend_rows[0]))
            w.writeheader(); w.writerows(bend_rows)

    # ---------------- figure
    fig, axs = plt.subplots(2, 2, figsize=(13, 9), dpi=140)
    names = [r["name"] for r in rows]
    x = np.arange(len(rows))
    cols = ["0.35"] + list(plt.cm.tab10.colors[:len(rows) - 1])
    ax = axs[0, 0]
    for i, r in enumerate(rows):
        ax.errorbar(i, r["D"], yerr=[[r["D"] - r["D_min_tol"]], [r["D_max_tol"] - r["D"]]],
                    fmt="o", ms=9, capsize=8, color=cols[i], lw=2)
        ax.annotate(f"{r['D']:.0f}\n[{r['D_min_tol']:.0f}–{r['D_max_tol']:.0f}]",
                    (i, r["D"]), xytext=(12, -10), textcoords="offset points", fontsize=8)
    if a.dmargin > 0:
        ax.axhline(a.dmargin, ls=":", color="k")
    ax.axhline(0, ls="--", color="b")
    ax.set_xticks(x, names); ax.set_ylabel("D(TE00) straight [ps/(nm·km)]")
    ax.set_title(f"Straight D with fabrication range (W±{dW*1e3:.0f} nm, H±{dH*1e3:.0f} nm)",
                 fontsize=10)
    ax = axs[0, 1]
    wbar = 0.38
    ax.bar(x - wbar / 2, [abs(r["dn_TE_TM"]) for r in rows], wbar, label="|n_TE0 − n_TM0|",
           color="purple", alpha=0.7)
    ax.bar(x + wbar / 2, [r["TE1_margin"] for r in rows], wbar, label="TE1 cutoff margin",
           color="deepskyblue", alpha=0.8)
    ax.axhline(a.deg_tol, color="purple", ls=":", label=f"degeneracy threshold {a.deg_tol}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, names); ax.set_ylabel("index difference")
    ax.set_title("Distance from hazards (bigger = safer; TE1 margin < 0 → multimode)",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax = axs[1, 0]; ax2 = axs[1, 1]
    if bend_rows:
        for i, r in enumerate(rows):
            br = [b for b in bend_rows if b["name"] == r["name"]]
            if not br:
                continue
            Rs = [b["R_um"] for b in br]
            ax.plot(Rs, [b["D_bend"] for b in br], "-o", color=cols[i], label=r["name"])
            ax.axhline(r["D"], color=cols[i], ls=":", lw=1)
            ax2.plot(Rs, [b["Pth_mW"] for b in br], "-o", color=cols[i], label=r["name"])
        if a.dmargin > 0:
            ax.axhline(a.dmargin, ls=":", color="k")
        ax.axhline(0, ls="--", color="b")
        ax.set_xscale("log"); ax.set_xlabel("Ring radius R (µm)")
        ax.set_ylabel("D(TE00) in bend [ps/(nm·km)]")
        ax.set_title("Bend dispersion (dotted = straight value)", fontsize=10)
        ax.legend(fontsize=8)
        ax2.set_xscale("log"); ax2.set_yscale("log"); ax2.set_xlabel("Ring radius R (µm)")
        ax2.set_ylabel("P_th estimate (mW, on-chip)")
        ax2.set_title(f"Threshold estimate, α_scat+α_abs = {a.alpha_scat + a.alpha_abs:.2f} dB/cm "
                      "+ bend", fontsize=10)
        ax2.legend(fontsize=8)
    else:
        for axx in (ax, ax2):
            axx.text(0.5, 0.5, "no bend data yet\n(run with --bend-radius)", ha="center",
                     va="center", transform=axx.transAxes)
    for axx in axs.flat:
        axx.grid(alpha=0.3)
    fig.suptitle(f"Finalists vs baseline {base['name']} nm ({stack}, "
                 f"{r0.get('wl0', 1.55) * 1e3:.0f} nm)", fontsize=12)
    fig.tight_layout(); fig.savefig(out / "finalists_comparison.png")

    # ---------------- dedicated bend-dispersion figure
    if bend_rows:
        fb, bx = plt.subplots(2, 2, figsize=(13, 9), dpi=140)
        for i, r in enumerate(rows):
            br = sorted((b for b in bend_rows if b["name"] == r["name"]), key=lambda b: b["R_um"])
            if not br:
                continue
            Rs = [b["R_um"] for b in br]
            kw = dict(color=cols[i], marker="o", label=r["name"] + (" (baseline)" if i == 0 else ""))
            bx[0, 0].plot(Rs, [b["D_bend"] for b in br], "-", **kw)
            bx[0, 0].axhline(r["D"], color=cols[i], ls=":", lw=1)
            bx[0, 1].plot(Rs, [b["dD_vs_straight"] for b in br], "-", **kw)
            bx[1, 0].plot(Rs, [b["D2_2pi_MHz"] for b in br], "-", **kw)
            bx[1, 1].plot(Rs, [abs(b["dn_TE_TM"]) for b in br], "-", **kw)
            bx[1, 1].axhline(abs(r["dn_TE_TM"]), color=cols[i], ls=":", lw=1)
        if a.dmargin > 0:
            bx[0, 0].axhline(a.dmargin, color="k", ls="--", lw=0.8)
        bx[0, 0].set_ylabel("D(TE00) in bend [ps/(nm·km)]")
        bx[0, 0].set_title("Bend dispersion (dotted = straight guide)", fontsize=10, pad=34)
        bx[0, 1].axhline(0, color="k", lw=0.8)
        bx[0, 1].set_ylabel("D_bend − D_straight [ps/(nm·km)]")
        bx[0, 1].set_title("Bend-induced change of D (same mesh required)", fontsize=10)
        bx[1, 0].set_yscale("log"); bx[1, 0].set_ylabel("D2/2π (MHz)")
        bx[1, 0].set_title("Ring integrated-dispersion coefficient D2 (>0 = anomalous)",
                           fontsize=10)
        bx[1, 1].axhline(a.deg_tol, color="purple", ls="--", lw=0.8,
                         label=f"degeneracy threshold {a.deg_tol}")
        bx[1, 1].set_ylabel("|n_TE0 − n_TM0| in bend")
        bx[1, 1].set_title("TE/TM splitting in the bend (dotted = straight)", fontsize=10)
        for axx in bx.flat:
            axx.set_xscale("log"); axx.set_xlabel("Ring radius R (µm)")
            axx.grid(alpha=0.3, which="both"); axx.legend(fontsize=8)
        # FSR on a secondary axis of the first panel (FSR ~ 1/R)
        ng0 = rows[0]["ng"]
        sec = bx[0, 0].secondary_xaxis(
            "top", functions=(lambda R: C / (2 * math.pi * ng0 * np.maximum(R, 1e-3) * 1e-6) / 1e9,
                              lambda f: C / (2 * math.pi * ng0 * np.maximum(f, 1e-3) * 1e9) * 1e6))
        sec.set_xlabel(f"≈ FSR (GHz) for n_g = {ng0:.2f}")
        fb.suptitle(f"Bent-waveguide dispersion of the finalists ({stack})", fontsize=12)
        fb.tight_layout(); fb.savefig(out / "bend_dispersion.png")

    # ---------------- markdown report
    L = [f"# Finalist comparison (baseline = {base['name']} nm)", "",
         f"Straight-waveguide data: {base['steps']} steps/λ. Tolerance: W ± {dW*1e3:.0f} nm, "
         f"H ± {dH*1e3:.0f} nm (source: {', '.join(sorted({r['tol_source'] for r in rows}))}).",
         (f"Classes use D ≥ {a.dmargin:g} ps/(nm·km) (user margin) as 'safely anomalous' and "
          if a.dmargin > 0 else "Classes use D ≥ 0 as anomalous (no margin applied) and ") +
         f"|Δn(TE0,TM0)| ≥ {a.deg_tol} as 'non-degenerate'.", "",
         "| Geometry | D | worst-case D | dD/dW (per nm) | dD/dH (per nm) | guided TE/TM | "
         "|Δn TE–TM| | TE1 margin | centre class | all neighbours TARGET? |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['name']}{' (baseline)' if r is base else ''} | {r['D']:.0f} | "
                 f"{r['D_min_tol']:.0f} | {r['dD_dW']:.2f} | {r['dD_dH']:.2f} | "
                 f"{r['guided_TE']}/{r['guided_TM']} | {abs(r['dn_TE_TM']):.4f} | "
                 f"{r['TE1_margin']:+.4f} | {r['class_center']} | "
                 f"{tol_text(r)} |")
    if bend_rows:
        L += ["", "## Bent waveguide", "",
              "| Geometry | R (µm) | FSR (GHz) | D_bend | ΔD vs straight | D2/2π (MHz) | "
              "bend loss (dB/cm) | Q_i | P_th (mW) | Δn TE–TM (bend) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for b in bend_rows:
            L.append(f"| {b['name']} | {b['R_um']:g} | {b['FSR_GHz']:.0f} | {b['D_bend']:.0f} | "
                     f"{b['dD_vs_straight']:+.0f} | {b['D2_2pi_MHz']:.2f} | "
                     f"{b['bend_loss_dB_cm']:.2e} | {b['Qi']:.2e} | {b['Pth_mW']:.1f} | "
                     f"{b['dn_TE_TM']:+.4f} |")
    L += ["", "## Automatic verdicts relative to the baseline", ""]
    for r in rows[1:]:
        v = []
        v.append(f"D {r['D'] - base['D']:+.0f} ps/(nm·km) at centre, "
                 f"worst case {r['D_min_tol'] - base['D_min_tol']:+.0f}")
        v.append(f"width sensitivity {abs(r['dD_dW']):.2f} vs {abs(base['dD_dW']):.2f} per nm "
                 f"({'less' if abs(r['dD_dW']) < abs(base['dD_dW']) else 'more'} sensitive)")
        v.append(f"TE/TM splitting {abs(r['dn_TE_TM']):.4f} vs {abs(base['dn_TE_TM']):.4f}")
        v.append(f"TE1 cutoff margin {r['TE1_margin']:+.4f} vs {base['TE1_margin']:+.4f}")
        v.append({True: "tolerance box fully TARGET",
                  None: "tolerance box not computed exactly (run the runner with --tol)"}
                 .get(r["all_neighbours_target"],
                      f"tolerance box touches {r['neighbour_classes']}"))
        L.append(f"- **{r['name']}**: " + "; ".join(v) + ".")
    cores = ", ".join(sorted({str(straight[(round(H, 4), round(W, 4), None)].get("core", "si3n4"))
                              for H, W in fins}))
    L += ["", f"Loss assumptions: α_scat = {a.alpha_scat} dB/cm, α_abs = {a.alpha_abs} dB/cm, "
          f"n2 = {a.n2:g} m²/W. P_th is a comparative estimate (±~2×).",
          "Model limits: vertical sidewalls, the tabulated index of the core material "
          f"({cores}), no avoided mode crossings between ring mode families."]
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n-> {out.resolve()}")


if __name__ == "__main__":
    main()
