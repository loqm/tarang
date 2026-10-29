"""Compare TARANG's dispersion with published Si3N4 microresonators.

For each row of ``papers.csv`` TARANG solves the paper's cross-section (same width, height,
top cladding and mode, at the pump wavelength; SiO2 below, 4 µm) and computes the group
velocity dispersion β2 and the group index n_g. With the paper's FSR (D1 = 2π FSR) the
dispersion of the resonance frequencies ω_μ = ω0 + D1 μ + D2 μ²/2 follows as

    D2 = −(c / n_g) D1² β2

(the mode of the bent waveguide when the paper's radius is given, column bend_radius_um;
else of the straight one, fine for large rings)

and D2/2π is compared with the paper's value. Each geometry is also solved with the film
10 nm thinner and thicker, the usual run-to-run spread of LPCVD films, to show how much of a
difference that alone explains. Rows of kind ``window`` check that the dispersion stays
anomalous over a range of widths the paper reports.

Run:  python validation/literature/compare.py [--steps 70] [--only id1,id2] [--out DIR]
Writes ``literature_comparison.csv`` and ``literature_comparison.json`` (default: next to
this file, under results/) and prints a table. Free: local mode solver only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
C = 299792458.0
DH = 0.010                                       # film thickness spread for the band (µm)


def read_papers(path: Path = HERE / "papers.csv") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def d2_2pi_MHz(beta2_ps2_km: float, ng: float, fsr_GHz: float) -> float:
    """D2/2π (MHz) from β2 (ps²/km), the group index and the FSR (GHz)."""
    beta2 = beta2_ps2_km * 1e-27                 # s²/m
    d1 = 2 * math.pi * fsr_GHz * 1e9
    return -(C / ng) * d1 ** 2 * beta2 / (2 * math.pi) / 1e6


def _solve(H, W, row, steps, cladding=None):
    from tarang.waveguide.dispersion_mapping import core
    wl = float(row["pump_nm"]) / 1e3
    clad = cladding or row["cladding"]
    bend = float(row["bend_radius_um"]) if row.get("bend_radius_um") else None
    r = core.solve_point(H, W, wl0=wl, span=min(0.2, 0.13 * wl), steps=steps,
                         cladding=None if clad == "air" else clad, snap=True, bend_radius=bend)
    pol = "TE0" if row["mode"].upper().startswith("TE") else "TM0"
    return r[f"beta2_{pol}"], r[f"ng_{pol}"], r[f"D_{pol}"]


def compare_row(row: dict, steps: int = 50) -> dict:
    H = float(row["height_um"])
    out = dict(id=row["id"], reference=row["reference"], mode=row["mode"],
               cladding=row["cladding"], kind=row["kind"])
    if row["kind"] == "window":
        lo, hi = float(row["window_lo_um"]), float(row["window_hi_um"])
        n = 5
        ws = [round(lo + k * (hi - lo) / (n - 1), 4) for k in range(n)]
        Ds = [_solve(H, w, row, steps)[2] for w in ws]
        out.update(height_um=H, widths_um=ws, D_ps_nm_km=[round(d, 1) for d in Ds],
                   tarang_all_anomalous=all(d > 0 for d in Ds),
                   agrees=all(d > 0 for d in Ds))
        return out
    W, fsr, ref = float(row["width_um"]), float(row["fsr_GHz"]), float(row["d2_2pi_MHz"])
    if row["cladding"] == "oxide_sides_air_top":
        # TARANG's cross-section has one cladding around the core: SiO2 beside it and air on
        # top (a Damascene waveguide) lies between all-air and all-SiO2
        d2s = {}
        for clad in ("air", "sio2"):
            b2, ng, _ = _solve(H, W, row, steps, clad)
            d2s[clad] = d2_2pi_MHz(b2, ng, fsr)
        lo, hi = sorted(d2s.values())
        out.update(kind="bracket", width_um=W, height_um=H, pump_nm=float(row["pump_nm"]),
                   fsr_GHz=fsr, d2_paper_MHz=ref, d2_tarang_air_MHz=round(d2s["air"], 4),
                   d2_tarang_sio2_MHz=round(d2s["sio2"], 4), within_bracket=lo <= ref <= hi,
                   agrees=bool(lo <= ref <= hi))
        return out
    vals = {}
    for tag, h in (("nominal", H), ("thinner", H - DH), ("thicker", H + DH)):
        b2, ng, D = _solve(h, W, row, steps)
        vals[tag] = dict(beta2_ps2_km=b2, ng=ng, D_ps_nm_km=D, d2=d2_2pi_MHz(b2, ng, fsr))
    d2 = vals["nominal"]["d2"]
    band = sorted([vals["thinner"]["d2"], vals["thicker"]["d2"]])
    out.update(width_um=W, height_um=H, pump_nm=float(row["pump_nm"]), fsr_GHz=fsr,
               d2_paper_MHz=ref, d2_tarang_MHz=round(d2, 4),
               d2_tarang_film_pm10nm_MHz=[round(band[0], 4), round(band[1], 4)],
               difference_percent=round(100 * (d2 - ref) / abs(ref), 1),
               same_sign=(d2 > 0) == (ref > 0),
               within_film_spread=band[0] <= ref <= band[1],
               D_ps_nm_km=round(vals["nominal"]["D_ps_nm_km"], 1),
               beta2_ps2_km=round(vals["nominal"]["beta2_ps2_km"], 2),
               ng=round(vals["nominal"]["ng"], 4))
    if row["kind"] == "sign":          # near-zero GVD (a small ring): only the sign is robust
        out["agrees"] = bool(out["same_sign"])
    else:
        out["agrees"] = bool(out["same_sign"] and (out["within_film_spread"]
                                                   or abs(out["difference_percent"]) <= 20))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=70,
                    help="mesh steps per wavelength (70: the ±10 nm film values are smooth)")
    ap.add_argument("--only", default="", help="comma-separated ids from papers.csv")
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args(argv)
    rows = read_papers()
    if a.only:
        keep = set(a.only.split(","))
        rows = [r for r in rows if r["id"] in keep]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = []
    for r in rows:
        print(f"{r['id']}: {r['reference']} ...", flush=True)
        res.append(compare_row(r, a.steps))
        x = res[-1]
        if x["kind"] == "bracket":
            print(f"  D2/2π: paper {x['d2_paper_MHz']} MHz; TARANG all-SiO2 "
                  f"{x['d2_tarang_sio2_MHz']:.3f} MHz, all-air {x['d2_tarang_air_MHz']:.3f} MHz -> "
                  f"{'inside' if x['within_bracket'] else 'OUTSIDE'} the bracket")
        elif x["kind"] == "window":
            print(f"  D at W = {x['widths_um']} µm: {x['D_ps_nm_km']} ps/(nm km) -> "
                  f"{'all anomalous' if x['tarang_all_anomalous'] else 'NOT all anomalous'}")
        else:
            print(f"  D2/2π: paper {x['d2_paper_MHz']} MHz, TARANG {x['d2_tarang_MHz']:.3f} MHz "
                  f"({x['difference_percent']:+.1f} %; film ±10 nm: "
                  f"{x['d2_tarang_film_pm10nm_MHz'][0]:.3f}-{x['d2_tarang_film_pm10nm_MHz'][1]:.3f})")
    (out / "literature_comparison.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    flat = [{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()} for r in res]
    keys = list(dict.fromkeys(k for r in flat for k in r))
    with open(out / "literature_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(flat)
    print(f"{sum(r['agrees'] for r in res)} of {len(res)} agree -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
