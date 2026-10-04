"""
plot_air_sweep.py -- figures from run_air_sweep.py results
==========================================================
    python scripts/plot_air_sweep.py --data results/air_sweep_50steps.jsonl
    python scripts/plot_air_sweep.py --data results/air_sweep_50steps.jsonl --dmargin 100 --show

Writes to results/plots_<name>/ :
  D_vs_width/H_XXXnm.png    one per height  (dispersion regime map along width)
  D_vs_height/W_XXXXnm.png  one per width   (dispersion regime map along thickness)
  modes_vs_width/H_XXXnm.png   TE0/TE1/TM0/TM1 n_eff vs width, per height
  modes_vs_height/W_XXXXnm.png TE0/TE1/TM0/TM1 n_eff vs height, per width
  design_map.png            2-D (W, H) map: D contours + regime classes
  target_geometries.csv     every point classified TARGET, sorted by margin

Classification of each (H, W) point  (TE00 is the pump mode):
  NORMAL      D < 0
  MARGINAL    0 <= D < dmargin        (sign not trustworthy within model error)
  TARGET      D >= dmargin and exactly 1 guided TE mode
  MULTIMODE   D >= dmargin and >= 2 guided TE modes
  Extra flag  |n_TE0 - n_TM0| < deg_tol  -> TE/TM near-degenerate (hybridises in bends)

dmargin default 100 ps/(nm km): straight-waveguide model vs NIST measurement differed
by ~80 ps/(nm km) (bending, sidewall angle, film index are not modelled).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

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


CLASS_COL = {"NORMAL": "#d9d9d9", "MARGINAL": "#fff2b3",
             "TARGET": "#b8e6b0", "MULTIMODE": "#f7b6b2"}
CLASS_LBL = {"NORMAL": "Normal dispersion (D < 0)",
             "MARGINAL": "Anomalous but within model error",
             "TARGET": "TARGET: anomalous + single TE mode",
             "MULTIMODE": "Anomalous + multimode (TE)"}


def load(path: Path):
    rows, errs = {}, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "error" in r:
            errs.append(r)
        else:
            rows[(r["H"], r["W"])] = r          # later lines overwrite earlier
    return list(rows.values()), errs


def classify(r, dmargin, sm_key):
    D = r["D_TE0"]
    if D < 0:
        return "NORMAL"
    if D < dmargin:
        return "MARGINAL"
    return "TARGET" if r[sm_key] == 1 else "MULTIMODE"


def degenerate(r, tol):
    return abs(r["neff_TE"][0] - r["neff_TM"][0]) < tol


def bands(xs):
    xs = np.asarray(xs, float)
    if len(xs) == 1:
        return [(xs[0] - 0.025, xs[0] + 0.025)]
    mids = (xs[1:] + xs[:-1]) / 2
    edges = np.concatenate([[xs[0] - (mids[0] - xs[0])], mids, [xs[-1] + (xs[-1] - mids[-1])]])
    return list(zip(edges[:-1], edges[1:]))


def shade(ax, xs, classes):
    for (a, b), c in zip(bands(xs), classes):
        ax.axvspan(a, b, color=CLASS_COL[c], alpha=0.8, lw=0, zorder=0)


def legend_patches():
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    h = [Patch(color=CLASS_COL[k], label=CLASS_LBL[k]) for k in CLASS_COL]
    h.append(Line2D([], [], marker="v", ls="", color="purple", label="TE/TM near-degenerate"))
    return h


def plot_D(series, xkey, fixed_lbl, fname, a, plt):
    series = sorted(series, key=lambda r: r[xkey])
    xs = [r[xkey] for r in series]
    D = [r["D_TE0"] for r in series]
    cls = [classify(r, a.dmargin, a.sm_key) for r in series]
    fig, ax = plt.subplots(figsize=(10.5, 5), dpi=140)
    shade(ax, xs, cls)
    ax.axhline(0, color="b", ls="--", lw=1)
    for y in (a.dmargin, -a.dmargin):
        ax.axhline(y, color="0.4", ls=":", lw=1)
    ax.plot(xs, D, "k-o", lw=1.8, ms=5, label="TE00", zorder=3)
    if a.tm:
        ax.plot(xs, [r["D_TM0"] for r in series], "--s", color="crimson", ms=4, lw=1.2,
                label="TM00", zorder=3)
    for x, d, r in zip(xs, D, series):
        if degenerate(r, a.deg_tol):
            ax.plot(x, d, "v", color="purple", ms=10, zorder=4)
        ax.annotate(f"{r[a.sm_key]}", (x, d), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7, color="0.25")
    ax.set_xlabel(("Waveguide width" if xkey == "W" else "SiN thickness") + " (µm)")
    ax.set_ylabel(r"$D$ at %.0f nm [ps/(nm·km)]" % (series[0]["wl0"] * 1e3))
    ax.set_title(f"Air-clad Si$_3$N$_4$ on SiO$_2$ — {fixed_lbl}   "
                 f"(numbers = guided TE modes, {series[0]['steps']} steps/λ)", fontsize=10)
    ax.grid(alpha=0.3)
    h = legend_patches()
    ax.legend(handles=ax.get_legend_handles_labels()[0] + h, fontsize=7,
              loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig) if not a.show else None


def plot_modes(series, xkey, fixed_lbl, fname, a, plt):
    series = sorted(series, key=lambda r: r[xkey])
    xs = np.array([r[xkey] for r in series])
    ncut = series[0]["n_cut"]
    margin = series[0]["margin"]
    fig, ax = plt.subplots(figsize=(10.5, 5), dpi=140)
    cls = [classify(r, a.dmargin, a.sm_key) for r in series]
    for (lo, hi), c, r in zip(bands(xs), cls, series):
        if r[a.sm_key] == 1:
            ax.axvspan(lo, hi, color="#b8e6b0", alpha=0.35, lw=0, zorder=0)
    specs = [("neff_TE", "hybrid_TE", 0, "TE00 (fund. TE)", "navy", "-", "o"),
             ("neff_TE", "hybrid_TE", 1, "TE1 (2nd-order TE)", "deepskyblue", "-", "^"),
             ("neff_TM", "hybrid_TM", 0, "TM00 (fund. TM)", "crimson", "--", "s"),
             ("neff_TM", "hybrid_TM", 1, "TM1 (2nd-order TM)", "orange", "--", "x")]
    for key, hkey, idx, lbl, col, ls, mk in specs:
        y = np.array([r[key][idx] if len(r[key]) > idx else np.nan for r in series])
        hyb = np.array([r[hkey][idx] if len(r[hkey]) > idx else False for r in series])
        g = y > ncut
        # guided part: solid; below cutoff: faded (solver returns substrate/radiation
        # modes there -- NOT physical guided modes, don't read meaning into kinks)
        yg = np.where(g, y, np.nan)
        ax.plot(xs, yg, ls=ls, marker=mk, color=col, lw=1.8, ms=6, label=lbl)
        ax.plot(xs[~g], y[~g], ls="", marker=mk, color=col, alpha=0.25, ms=5)
        if hyb.any():
            ax.plot(xs[hyb], y[hyb], "o", mfc="none", mec="k", ms=11,
                    label=None if "hyb" in ax.get_legend_handles_labels()[1] else "hyb")
    ax.axhline(ncut, color="k", lw=1.2, label=f"SiO$_2$ cutoff (n = {ncut:.3f})")
    ax.axhline(ncut + margin, color="k", lw=0.8, ls=":", label=f"cutoff + {margin}")
    h, l = ax.get_legend_handles_labels()
    l = ["hybrid (TE/TM mixed, 0.4<TE frac<0.6)" if x == "hyb" else x for x in l]
    ax.legend(h, l, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_xlabel(("Waveguide width" if xkey == "W" else "SiN thickness") + " (µm)")
    ax.set_ylabel(r"Effective index $n_\mathrm{eff}$")
    ax.set_title(f"Modal analysis, air-clad — {fixed_lbl}  "
                 f"(green = single guided TE mode; faded = below cutoff)", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig) if not a.show else None


def plot_map(rows, outdir, a, plt):
    Hs = sorted({r["H"] for r in rows})
    Ws = sorted({r["W"] for r in rows})
    idx = {(r["H"], r["W"]): r for r in rows}
    Dm = np.full((len(Hs), len(Ws)), np.nan)
    Cm = np.full((len(Hs), len(Ws)), np.nan)
    order = list(CLASS_COL)
    for i, H in enumerate(Hs):
        for j, W in enumerate(Ws):
            r = idx.get((H, W))
            if r:
                Dm[i, j] = r["D_TE0"]
                Cm[i, j] = order.index(classify(r, a.dmargin, a.sm_key))
    from matplotlib.colors import ListedColormap, TwoSlopeNorm
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), dpi=140)
    # Clip the colour scale: near-cutoff points reach -2000+ and would wash out the
    # anomalous region.  Scale to the largest positive D (the design-relevant side).
    lim = max(float(np.nanmax(Dm)), a.dmargin) * 1.1
    im = axs[0].pcolormesh(Ws, Hs, np.clip(Dm, -lim, lim), cmap="RdBu_r",
                           norm=TwoSlopeNorm(0, -lim, lim), shading="nearest")
    if len(Hs) > 1 and len(Ws) > 1:
        cs = axs[0].contour(Ws, Hs, Dm, levels=[0, a.dmargin], colors=["k", "k"],
                            linestyles=["-", "--"])
        axs[0].clabel(cs, fmt="%g", fontsize=8)
    fig.colorbar(im, ax=axs[0], extend="min", label="D(TE00) [ps/(nm·km)], clipped")
    axs[0].set_title("D(TE00): solid = 0, dashed = dmargin", fontsize=10)
    axs[1].pcolormesh(Ws, Hs, Cm, cmap=ListedColormap([CLASS_COL[k] for k in order]),
                      vmin=-0.5, vmax=len(order) - 0.5, shading="nearest")
    for (H, W), r in idx.items():
        t = f"{r['D_TE0']:.0f}\n{r[a.sm_key]}TE"
        axs[1].text(W, H, t, ha="center", va="center", fontsize=6)
        if degenerate(r, a.deg_tol):
            axs[1].plot(W + ((Ws[1] - Ws[0]) * 0.33 if len(Ws) > 1 else 0.03), H + ((Hs[1] - Hs[0]) * 0.30 if len(Hs) > 1 else 0.015), "v", color="purple", ms=7)
    axs[1].set_title("Regime (text: D, guided TE count; ▽ TE/TM degenerate)", fontsize=10)
    axs[1].legend(handles=legend_patches(), fontsize=7, loc="upper center",
                  bbox_to_anchor=(0.5, -0.13), ncol=3)
    for ax in axs:
        ax.set_xlabel("Width W (µm)")
        ax.set_ylabel("SiN thickness H (µm)")
    fig.suptitle(f"Air-clad Si3N4 design map at {rows[0]['wl0']*1e3:.0f} nm", fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / "design_map.png")
    plt.close(fig) if not a.show else None


# ============================ extra physics maps ==============================
def _grid(rows):
    Hs = sorted({r["H"] for r in rows}); Ws = sorted({r["W"] for r in rows})
    idx = {(r["H"], r["W"]): r for r in rows}
    return Hs, Ws, idx


def _mat(Hs, Ws, idx, f):
    M = np.full((len(Hs), len(Ws)), np.nan)
    for i, H in enumerate(Hs):
        for j, W in enumerate(Ws):
            r = idx.get((H, W))
            if r is not None:
                try:
                    M[i, j] = f(r)
                except (KeyError, IndexError, TypeError):
                    pass
    return M


def _heat(ax, Ws, Hs, M, cmap, norm, cbar_label, fig, fmt=None, contours=(), extend="neither"):
    im = ax.pcolormesh(Ws, Hs, M, cmap=cmap, norm=norm, shading="nearest")
    fig.colorbar(im, ax=ax, label=cbar_label, extend=extend)
    if len(Hs) > 1 and len(Ws) > 1:
        for level, color, ls in contours:
            if np.nanmin(M) < level < np.nanmax(M):
                cs = ax.contour(Ws, Hs, M, levels=[level], colors=[color], linestyles=[ls],
                                linewidths=1.6)
                ax.clabel(cs, fmt=lambda v: f"{v:g}", fontsize=7)
    if fmt:
        for i, H in enumerate(Hs):
            for j, W in enumerate(Ws):
                if np.isfinite(M[i, j]):
                    ax.text(W, H, fmt(M[i, j]), ha="center", va="center", fontsize=6)
    ax.set_xlabel("Width W (µm)"); ax.set_ylabel("SiN thickness H (µm)")


def robust_class(rows, a):
    """TARGET (and non-degenerate) whose grid neighbours in +/-W and +/-H are also
    TARGET -> ROBUST; TARGET with a failing neighbour -> EDGE; neighbour missing -> BORDER."""
    Hs, Ws, idx = _grid(rows)
    def ok(r):
        return (r is not None and classify(r, a.dmargin, a.sm_key) == "TARGET"
                and not degenerate(r, a.deg_tol))
    out = {}
    for i, H in enumerate(Hs):
        for j, W in enumerate(Ws):
            r = idx.get((H, W))
            if r is None:
                continue
            if not ok(r):
                out[(H, W)] = "NOT_TARGET"; continue
            nbrs = []
            for di, dj in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                ii, jj = i + di, j + dj
                nbrs.append(idx.get((Hs[ii], Ws[jj])) if 0 <= ii < len(Hs) and 0 <= jj < len(Ws)
                            else "border")
            if any(n == "border" for n in nbrs):
                out[(H, W)] = "BORDER" if all(ok(n) for n in nbrs if n != "border") else "EDGE"
            else:
                out[(H, W)] = "ROBUST" if all(ok(n) for n in nbrs) else "EDGE"
    return out


def plot_extra_maps(rows, outdir, a, plt):
    from matplotlib.colors import TwoSlopeNorm, LogNorm, ListedColormap
    md = outdir / "maps"; md.mkdir(exist_ok=True)
    Hs, Ws, idx = _grid(rows)
    if len(Hs) < 2 or len(Ws) < 2:
        print("maps: need >= 2 heights and >= 2 widths, skipped"); return
    made = {}

    def save(fig, name):
        fig.tight_layout(); fig.savefig(md / name); made[name] = True
        if not a.show:
            plt.close(fig)

    wl = rows[0]["wl0"]
    Dm = _mat(Hs, Ws, idx, lambda r: r["D_TE0"])
    lim = max(float(np.nanmax(Dm)), a.dmargin) * 1.1

    # 1  D map
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=140)
    _heat(ax, Ws, Hs, np.clip(Dm, -lim, lim), "RdBu_r", TwoSlopeNorm(0, -lim, lim),
          "D(TE00) [ps/(nm·km)], clipped", fig, fmt=lambda v: f"{v:.0f}",
          contours=[(0, "k", "-"), (a.dmargin, "k", "--")], extend="min")
    ax.set_title("1. Dispersion D of the pump mode (solid 0, dashed = margin)", fontsize=10)
    save(fig, "map1_dispersion_D.png")

    # 2  TE-TM splitting (signed)
    dn = _mat(Hs, Ws, idx, lambda r: r["neff_TE"][0] - r["neff_TM"][0])
    L = max(0.02, float(np.nanmax(np.abs(dn))))
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=140)
    _heat(ax, Ws, Hs, dn, "PuOr", TwoSlopeNorm(0, -L, L), "n_TE0 − n_TM0", fig,
          fmt=lambda v: f"{v:+.3f}",
          contours=[(0, "k", "-"), (a.deg_tol, "k", "--"), (-a.deg_tol, "k", "--")])
    ax.set_title("2. TE/TM splitting (<0: TM is fundamental; |Δn|<0.01 dashed: hybridises)",
                 fontsize=10)
    save(fig, "map2_TE_TM_splitting.png")

    # 3  second-order-mode cutoff margin
    m1 = _mat(Hs, Ws, idx, lambda r: r["n_cut"] - r["neff_TE"][1])
    m1tm = _mat(Hs, Ws, idx, lambda r: r["n_cut"] - r["neff_TM"][1])
    L = float(np.nanmax(np.abs(m1)))
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=140)
    _heat(ax, Ws, Hs, m1, "BrBG", TwoSlopeNorm(0, -L, L),
          "n_SiO2 − n_TE1  (>0: TE1 not guided)", fig, fmt=lambda v: f"{v:+.3f}",
          contours=[(0, "k", "-")])
    if np.nanmin(m1tm) < 0 < np.nanmax(m1tm):
        cs = ax.contour(Ws, Hs, m1tm, levels=[0], colors=["crimson"], linestyles=["--"])
        ax.clabel(cs, fmt=lambda v: "TM1 cutoff", fontsize=7)
    ax.set_title("3. Distance from the 2nd TE mode's cutoff (solid: TE1 cutoff; red: TM1)",
                 fontsize=10)
    save(fig, "map3_second_mode_margin.png")

    # 4/5  sensitivity of D to width and thickness errors
    Wn = np.array(Ws) * 1e3; Hn = np.array(Hs) * 1e3
    dDdW = np.abs(np.gradient(Dm, Wn, axis=1)); dDdH = np.abs(np.gradient(Dm, Hn, axis=0))
    for k, (M, lab, nm) in enumerate(((dDdW, "|∂D/∂W|", "width"), (dDdH, "|∂D/∂H|", "thickness")),
                                     start=4):
        fig, ax = plt.subplots(figsize=(7.5, 5), dpi=140)
        _heat(ax, Ws, Hs, np.clip(M, 0.02, 50), "magma_r", LogNorm(0.02, 50),
              f"{lab}  [ps/(nm·km) per nm]", fig, fmt=lambda v: f"{v:.2g}", extend="both")
        ax.set_title(f"{k}. Sensitivity of D to a {nm} error (darker = more fragile)", fontsize=10)
        save(fig, f"map{k}_sensitivity_{nm}.png")

    # 6  robustness map
    rc = robust_class(rows, a)
    order = ["NOT_TARGET", "EDGE", "BORDER", "ROBUST"]
    cols = ["#e6e6e6", "#ffd27f", "#c9e4f5", "#2e9e44"]
    C = _mat(Hs, Ws, idx, lambda r: order.index(rc[(r["H"], r["W"])]))
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=140)
    ax.pcolormesh(Ws, Hs, C, cmap=ListedColormap(cols), vmin=-0.5, vmax=3.5, shading="nearest")
    for (H, W), r in idx.items():
        ax.text(W, H, f"{r['D_TE0']:.0f}", ha="center", va="center", fontsize=6,
                color="w" if rc[(H, W)] == "ROBUST" else "k")
    from matplotlib.patches import Patch
    lbl = {"NOT_TARGET": "not target (normal/marginal/multimode/degenerate)",
           "EDGE": "target, but a grid neighbour fails", "BORDER": "target at grid border",
           "ROBUST": "ROBUST: target and all 4 neighbours target"}
    ax.legend(handles=[Patch(color=c, label=lbl[o]) for o, c in zip(order, cols)], fontsize=7,
              loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    dWg = (Ws[1] - Ws[0]) * 1e3; dHg = (Hs[1] - Hs[0]) * 1e3
    ax.set_xlabel("Width W (µm)"); ax.set_ylabel("SiN thickness H (µm)")
    ax.set_title(f"6. Robust design window (neighbours ±{dWg:.0f} nm W, ±{dHg:.0f} nm H)",
                 fontsize=10)
    save(fig, "map6_robust_window.png")

    # 7  group index
    ng = _mat(Hs, Ws, idx, lambda r: r["ng_TE0"])
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=140)
    _heat(ax, Ws, Hs, ng, "viridis", None, "n_g(TE00)", fig, fmt=lambda v: f"{v:.3f}")
    ax.set_title("7. Group index (sets FSR = c / (2π n_g R) for a chosen ring radius)",
                 fontsize=10)
    save(fig, "map7_group_index.png")

    # 8/9  mode area, nonlinear coefficient, confinement (only if recorded)
    Ae = _mat(Hs, Ws, idx, lambda r: r["Aeff_TE0_um2"])
    if np.isfinite(Ae).any():
        g = 2 * np.pi * a.n2 / (wl * 1e-6 * Ae * 1e-12)          # 1/(W m)
        fig, ax = plt.subplots(figsize=(7.5, 5), dpi=140)
        _heat(ax, Ws, Hs, g, "cividis", None, "γ = 2π n2 / (λ A_eff)  [1/(W·m)]", fig,
              fmt=lambda v: f"{v:.2f}")
        ax.set_title("8. Kerr nonlinear coefficient γ (higher = lower threshold)", fontsize=10)
        save(fig, "map8_nonlinear_gamma.png")
    Gm = _mat(Hs, Ws, idx, lambda r: r["Gamma_TE0"])
    if np.isfinite(Gm).any():
        fig, ax = plt.subplots(figsize=(7.5, 5), dpi=140)
        _heat(ax, Ws, Hs, Gm * 100, "YlGn", None, "Γ = power fraction in SiN core (%)", fig,
              fmt=lambda v: f"{v:.0f}")
        ax.set_title("9. Core confinement Γ (more field outside → more sidewall scattering)",
                     fontsize=10)
        save(fig, "map9_confinement.png")

    # dashboard of the six decision maps
    fig, axs = plt.subplots(2, 3, figsize=(20, 10.5), dpi=120)
    panels = [("map1_dispersion_D.png", "1"), ("map2_TE_TM_splitting.png", "2"),
              ("map3_second_mode_margin.png", "3"), ("map4_sensitivity_width.png", "4"),
              ("map6_robust_window.png", "6"), ("map7_group_index.png", "7")]
    for ax, (fn, _) in zip(axs.flat, panels):
        ax.imshow(plt.imread(md / fn)); ax.axis("off")
    fig.suptitle("Air-clad Si3N4 / 4 µm SiO2 — decision dashboard at "
                 f"{wl*1e3:.0f} nm", fontsize=14)
    fig.tight_layout(); fig.savefig(md / "dashboard.png")
    if not a.show:
        plt.close(fig)
    print(f"maps: {len(made) + 1} files -> {md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--dmargin", type=float, default=100.0,
                    help="min D [ps/(nm km)] counted as safely anomalous")
    ap.add_argument("--deg-tol", dest="deg_tol", type=float, default=0.01,
                    help="|n_TE0 - n_TM0| below this is flagged degenerate")
    ap.add_argument("--single-mode", dest="sm", choices=["strict", "margin"], default="strict",
                    help="strict: any TE mode above n_SiO2 counts; margin: above n_SiO2+margin")
    ap.add_argument("--tm", action="store_true", help="also plot D of TM00")
    ap.add_argument("--n2", type=float, default=2.4e-19, help="Si3N4 Kerr index, m^2/W")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--overwrite", action="store_true",
                    help="reuse the existing plots folder instead of creating _v2, _v3, ...")
    a = ap.parse_args()
    a.sm_key = "guided_TE" if a.sm == "strict" else "guided_TE_margin"
    if not a.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, errs = load(a.data)
    if errs:
        print(f"WARNING: {len(errs)} failed points skipped, e.g. {errs[0]}")
    if not rows:
        raise SystemExit("No valid results in " + str(a.data))
    out = unique_dir(a.data.parent / f"plots_{a.data.stem}", a.overwrite)
    print(f"Writing plots to {out.resolve()}")
    for sub in ("D_vs_width", "D_vs_height", "modes_vs_width", "modes_vs_height"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    byH, byW = defaultdict(list), defaultdict(list)
    for r in rows:
        byH[r["H"]].append(r)
        byW[r["W"]].append(r)
    for H, s in sorted(byH.items()):
        if len(s) > 1:
            lbl = f"H = {H*1e3:.0f} nm"
            plot_D(s, "W", lbl, out / "D_vs_width" / f"H_{H*1e3:04.0f}nm.png", a, plt)
            plot_modes(s, "W", lbl, out / "modes_vs_width" / f"H_{H*1e3:04.0f}nm.png", a, plt)
    for W, s in sorted(byW.items()):
        if len(s) > 1:
            lbl = f"W = {W*1e3:.0f} nm"
            plot_D(s, "H", lbl, out / "D_vs_height" / f"W_{W*1e3:04.0f}nm.png", a, plt)
            plot_modes(s, "H", lbl, out / "modes_vs_height" / f"W_{W*1e3:04.0f}nm.png", a, plt)
    plot_map(rows, out, a, plt)
    plot_extra_maps(rows, out, a, plt)

    tgt = [r for r in rows if classify(r, a.dmargin, a.sm_key) == "TARGET"]
    tgt.sort(key=lambda r: -r["D_TE0"])
    with open(out / "target_geometries.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["H_nm", "W_nm", "D_TE0", "beta2_TE0_ps2_km", "ng_TE0", "n_TE0", "n_TM0",
                    "dn_TE0_TM0", "TE_TM_degenerate", "guided_TE", "guided_TM", "steps"])
        for r in tgt:
            dn = r["neff_TE"][0] - r["neff_TM"][0]
            w.writerow([round(r["H"]*1e3), round(r["W"]*1e3), r["D_TE0"], r["beta2_TE0"],
                        r["ng_TE0"], r["neff_TE"][0], r["neff_TM"][0], round(dn, 4),
                        degenerate(r, a.deg_tol), r["guided_TE"], r["guided_TM"], r["steps"]])
    print(f"{len(rows)} points plotted -> {out.resolve()}")
    print(f"{len(tgt)} TARGET geometries -> {out / 'target_geometries.csv'}")
    if a.show:
        plt.show()


if __name__ == "__main__":
    main()
