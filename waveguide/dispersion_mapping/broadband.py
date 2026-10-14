"""
waveguide.dispersion_mapping.broadband -- broadband dispersion D(lambda) and ring D_int
=========================================================================================
    python -m waveguide.dispersion_mapping.broadband --points 0.80x0.80,0.70x1.00 --radius 100
    python -m waveguide.dispersion_mapping.broadband --points 0.70x1.00 --bend-radius 100 --radius 100
    python -m waveguide.dispersion_mapping.broadband --plot-only

Why: D at 1550 nm says whether solitons can form at the pump.  How broad the comb can be,
and where dispersive waves appear, is set by how D (and the ring's integrated dispersion)
behaves over the WHOLE band, e.g. 1200-2000 nm.

Physics
-------
One mode solve over many wavelengths; TE00 re-identified at every wavelength; n_eff(lambda)
fitted with a smooth polynomial, then
    n_g(lambda) = n_eff - lambda dn/dlambda
    D(lambda)   = -(lambda/c) d2n/dlambda2
    ZDW         = zero-dispersion wavelengths (D = 0)
Ring integrated dispersion for radius R (azimuthal mode number m, pump m0):
    m(omega)  = n_eff(omega) * omega * R / c      (resonance condition)
    D_int(mu) = omega_mu - omega_0 - D1 * mu,  mu = m - m0,  D1 = 2*pi*FSR
    ~ D2 mu^2 / 2 + D3 mu^3 / 6 + ...
D_int > 0 around mu = 0  <=>  anomalous; where D_int crosses 0 again, dispersive waves can
be emitted (phase matching with the soliton).  With --bend-radius the n_eff itself comes
from a bent-waveguide solve, so D_int includes bending.

Data : results/dispersion_profiles.jsonl  (append-only)
Plots: results/dispersion_profile_plots[_vN]/
    profile_H700_W1000[_R100um].png   n_eff, n_g, D(lambda) and D_int(mu) for one geometry
    compare_D_lambda.png              D(lambda) of all geometries (baseline first)
    compare_Dint.png                  D_int(mu) of all geometries for the chosen radius
    profiles_summary.csv              ZDWs, D at pump, D2, D3 per geometry
"""
from __future__ import annotations

from waveguide.dispersion_mapping._console import safe_console

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

C = 299792458.0


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


def solve_profile(H, W, wl_min, wl_max, nwl, steps, bend_radius, num_modes=10, mat=None):
    mat = dict(mat or {})
    import tidy3d as td
    from tidy3d.plugins.mode import ModeSolver
    from waveguide.dispersion_mapping import core as sc
    sim = sc.build_sim(H, W, wl0=(wl_min + wl_max) / 2, steps=steps,
                       margin_y=(3.0 if bend_radius else 2.0), **mat)
    core_m = sc.medium(mat.get("core", sc.DEFAULT_CORE))
    sub_m = sc.medium(mat.get("substrate", sc.DEFAULT_SUBSTRATE))
    top_key = mat.get("cladding")
    top_m = None if sc.is_air(top_key) else sc.medium(top_key)
    wls = np.linspace(wl_min, wl_max, nwl)
    plane = td.Box(center=(0, 0, sim.center[2]), size=(0, sim.size[1], sim.size[2]))
    kw = dict(num_modes=num_modes, target_neff=sc.n_of(core_m, 1.55))
    if bend_radius:
        kw.update(bend_radius=float(bend_radius), bend_axis=1, num_pml=(12, 12))
    data = ModeSolver(simulation=sim, plane=plane, freqs=td.C_0 / wls,
                      mode_spec=td.ModeSpec(**kw)).solve()
    neff, tefrac = [], []
    for fi in range(nwl):
        best, bt = None, None
        for m in range(num_modes):
            tf = sc._te_frac(data, m, fi)
            n = float(np.real(data.n_eff.isel(f=fi, mode_index=m)))
            if tf > 0.5 and (best is None or n > best):
                best, bt = n, tf
        neff.append(best); tefrac.append(bt)
    ncut = [max(sc.n_of(sub_m, w), sc.n_of(top_m, w) if top_m is not None else 1.0)
            for w in wls]
    return dict(H=H, W=W, steps=steps, bend_radius=bend_radius,
                core=mat.get("core", sc.DEFAULT_CORE),
                substrate=mat.get("substrate", sc.DEFAULT_SUBSTRATE),
                cladding=None if sc.is_air(top_key) else top_key,
                wl=list(map(float, wls)),
                neff_TE0=neff, te_frac_TE0=tefrac, n_cut=ncut)


def analyse(rec, R_um, deg=8, mu_max=None):
    wl = np.array(rec["wl"]); n = np.array(rec["neff_TE0"], float)
    good = np.isfinite(n) & (n > np.array(rec["n_cut"]))          # guided only
    wl, n = wl[good], n[good]
    x = (wl - wl.mean()) / (wl.max() - wl.min())                     # scaled for conditioning
    p = np.polynomial.Polynomial.fit(x, n, deg)
    s = 1.0 / (wl.max() - wl.min())
    wf = np.linspace(wl.min(), wl.max(), 800)
    xf = (wf - wl.mean()) * s
    nf = p(xf); d1 = p.deriv(1)(xf) * s; d2 = p.deriv(2)(xf) * s ** 2
    c_um_ps = C * 1e6 / 1e12
    ng = nf - wf * d1
    D = -(wf / c_um_ps) * d2 * 1e6                                    # ps/(nm km)
    resid = float(np.max(np.abs(p(x) - n)))
    zdw = [float(wf[i] - D[i] * (wf[i + 1] - wf[i]) / (D[i + 1] - D[i]))
           for i in range(len(wf) - 1) if D[i] * D[i + 1] < 0]
    # ---- ring integrated dispersion
    R = R_um * 1e-6
    om = 2 * np.pi * C / (wf * 1e-6)
    m_of = nf * om * R / C                                            # decreasing in wf
    order = np.argsort(om); om_s, m_s = om[order], m_of[order]
    om0 = 2 * np.pi * C / 1.55e-6
    m0 = int(round(np.interp(om0, om_s, m_s)))
    om_p = np.interp(m0, m_s, om_s)                                   # exact pump resonance
    mus = np.arange(int(np.ceil(m_s.min())) - m0, int(np.floor(m_s.max())) - m0 + 1)
    if mu_max:
        mus = mus[np.abs(mus) <= mu_max]
    om_mu = np.interp(m0 + mus, m_s, om_s)
    D1 = (np.interp(m0 + 1, m_s, om_s) - np.interp(m0 - 1, m_s, om_s)) / 2
    Dint = om_mu - om_p - D1 * mus
    q = np.polyfit(mus[np.abs(mus) <= max(5, len(mus) // 20)],
                   Dint[np.abs(mus) <= max(5, len(mus) // 20)], 3)
    D2, D3 = 2 * q[1], 6 * q[0]
    lam_mu = 2 * np.pi * C / om_mu * 1e9
    i0 = np.argmin(np.abs(wf - 1.55))
    return dict(wf=wf, nf=nf, ng=ng, D=D, zdw=zdw, resid=resid, mus=mus, Dint=Dint,
                lam_mu=lam_mu, FSR_GHz=D1 / (2 * np.pi) / 1e9, D2_2pi_MHz=D2 / (2 * np.pi) / 1e6,
                D3_2pi_kHz=D3 / (2 * np.pi) / 1e3, D_pump=float(D[i0]), ng_pump=float(ng[i0]),
                wl_raw=wl, n_raw=n)


def plot_all(data_path, R_um, overwrite, deg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    recs = {}
    for line in Path(data_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if "error" not in r:
                recs[(r["H"], r["W"], r["bend_radius"], r["steps"], r.get("core", "si3n4"), r.get("substrate", "sio2"), r.get("cladding") or None)] = r
    out = unique_dir(Path(data_path).parent / "dispersion_profile_plots", overwrite)
    out.mkdir(parents=True, exist_ok=True)
    figD, axD = plt.subplots(figsize=(9, 5.5), dpi=140)
    figI, axI = plt.subplots(figsize=(9, 5.5), dpi=140)
    rows = []
    for k, rec in recs.items():
        H, W, Rb, st = k[:4]
        A = analyse(rec, Rb or R_um, deg)
        name = f"{H*1e3:.0f}×{W*1e3:.0f}" + (f" (bent R={Rb:g})" if Rb else "")
        fig, ax = plt.subplots(2, 2, figsize=(13, 9), dpi=140)
        ax[0, 0].plot(A["wl_raw"], A["n_raw"], "o", ms=4, label="solver")
        ax[0, 0].plot(A["wf"], A["nf"], "-", label=f"poly fit (max resid {A['resid']:.1e})")
        ax[0, 0].plot(rec["wl"], rec["n_cut"], "k--", lw=1, label="cladding cutoff")
        ax[0, 0].set_ylabel("n_eff(TE00)"); ax[0, 0].legend(fontsize=8)
        ax[0, 1].plot(A["wf"], A["ng"]); ax[0, 1].set_ylabel("n_g(TE00)")
        ax[1, 0].plot(A["wf"], A["D"], "k-"); ax[1, 0].axhline(0, color="b", ls="--")
        ax[1, 0].axvline(1.55, color="0.5", ls=":")
        for z in A["zdw"]:
            ax[1, 0].axvline(z, color="crimson", ls="--", lw=1)
            ax[1, 0].text(z, ax[1, 0].get_ylim()[1] * 0.9, f" ZDW {z*1e3:.0f} nm",
                          color="crimson", fontsize=8)
        ax[1, 0].set_ylabel("D [ps/(nm·km)]")
        for a_ in (ax[0, 0], ax[0, 1], ax[1, 0]):
            a_.set_xlabel("wavelength (µm)"); a_.grid(alpha=0.3)
        ax[1, 1].plot(A["lam_mu"], A["Dint"] / (2 * np.pi) / 1e9, "-", color="teal")
        ax[1, 1].axhline(0, color="k", lw=0.8); ax[1, 1].axvline(1550, color="0.5", ls=":")
        ax[1, 1].set_xlabel("resonance wavelength (nm)")
        ax[1, 1].set_ylabel("D_int / 2π (GHz)")
        ax[1, 1].set_title(f"Ring R = {Rb or R_um:g} µm: FSR {A['FSR_GHz']:.0f} GHz, "
                           f"D2/2π {A['D2_2pi_MHz']:.1f} MHz", fontsize=10)
        ax[1, 1].grid(alpha=0.3)
        fig.suptitle(f"Dispersion profile {name} nm, {rec.get('core', 'si3n4')} core / "
                     f"{rec.get('cladding') or 'air'} top / "
                     f"{rec.get('substrate', 'sio2')} ({st} steps/λ)",
                     fontsize=12)
        fig.tight_layout()
        tag = f"profile_H{H*1e3:.0f}_W{W*1e3:.0f}" + (f"_R{Rb:g}um" if Rb else "")
        fig.savefig(out / f"{tag}.png"); plt.close(fig)
        axD.plot(A["wf"], A["D"], lw=2, label=name)
        axI.plot(A["lam_mu"], A["Dint"] / (2 * np.pi) / 1e9, lw=2, label=name)
        rows.append(dict(geometry=name, steps=st, D_pump=round(A["D_pump"], 1),
                         ng_pump=round(A["ng_pump"], 4),
                         ZDWs_nm=";".join(f"{z*1e3:.0f}" for z in A["zdw"]) or "none in band",
                         R_um=Rb or R_um, FSR_GHz=round(float(A["FSR_GHz"]), 1),
                         D2_2pi_MHz=round(float(A["D2_2pi_MHz"]), 2),
                         D3_2pi_kHz=round(float(A["D3_2pi_kHz"]), 1),
                         fit_resid=float(A["resid"])))
    # Titles are set BEFORE tight_layout so they are inside the saved canvas
    axD.set_title("D(λ) of all profiled geometries", fontsize=11)
    axI.set_title(f"Ring integrated dispersion (straight ones use R = {R_um:g} µm)", fontsize=11)
    for axx, f, yl, fn in ((axD, figD, "D [ps/(nm·km)]", "compare_D_lambda.png"),
                           (axI, figI, "D_int / 2π (GHz)", "compare_Dint.png")):
        axx.axhline(0, color="k", lw=0.8); axx.grid(alpha=0.3); axx.legend(fontsize=8)
        axx.set_ylabel(yl)
        axx.set_xlabel("wavelength (µm)" if axx is axD else "resonance wavelength (nm)")
        f.tight_layout(); f.savefig(out / fn)
    if rows:
        with open(out / "profiles_summary.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        for r in rows:
            print(r)
    print(f"Plots -> {out.resolve()}")


def main():
    safe_console()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--core", default="si3n4", help="core medium key in media.json")
    ap.add_argument("--substrate", default="sio2", help="substrate (BOX) medium key")
    ap.add_argument("--cladding", default="air", help='top cladding medium key, or "air"')
    ap.add_argument("--points", default="0.80x0.80,0.70x1.00")
    ap.add_argument("--wl-min", type=float, default=1.20)
    ap.add_argument("--wl-max", type=float, default=2.00)
    ap.add_argument("--nwl", type=int, default=33, help="wavelength samples (one solve)")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--bend-radius", type=float, default=None,
                    help="solve the BENT waveguide at this radius (um)")
    ap.add_argument("--radius", type=float, default=100.0,
                    help="ring radius (um) used for D_int of straight-guide profiles")
    ap.add_argument("--fit-degree", type=int, default=8)
    ap.add_argument("--data", default="results/dispersion_profiles.jsonl")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    clad = None if str(a.cladding).lower() in ("", "air", "none") else a.cladding
    data = Path(a.data); data.parent.mkdir(parents=True, exist_ok=True)
    if not a.plot_only:
        done = set()
        if data.exists():
            for line in data.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    done.add((r["H"], r["W"], r["bend_radius"], r["steps"], r.get("core", "si3n4"), r.get("substrate", "sio2"), r.get("cladding") or None))
        for tok in a.points.split(","):
            H, W = (round(float(v), 4) for v in tok.lower().split("x"))
            if (H, W, a.bend_radius, a.steps, a.core, a.substrate, clad) in done:
                print(f"skip {H}x{W} (done)"); continue
            t = time.time()
            try:
                rec = solve_profile(H, W, a.wl_min, a.wl_max, a.nwl, a.steps, a.bend_radius,
                                    mat=dict(core=a.core, substrate=a.substrate, cladding=clad))
            except Exception as e:
                rec = dict(H=H, W=W, steps=a.steps, bend_radius=a.bend_radius,
                           error=f"{type(e).__name__}: {e}")
            rec["secs"] = round(time.time() - t, 1)
            with open(data, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"{H}x{W} R={a.bend_radius}: {rec.get('error', 'ok')} ({rec['secs']}s)",
                  flush=True)
    plot_all(data, a.radius, a.overwrite, a.fit_degree)


if __name__ == "__main__":
    main()
