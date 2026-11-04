"""Is D of the 780 nm design platform-dependent?

The same one-sided snapped solve (350 x 500 nm, 780 nm, 50 steps per wavelength) gave
D = 198.8 ps/(nm km) on Linux and 235.3 on the Windows PC that ran the paper design. The
grids and permittivities of the two computers agree to 1e-13 um; what differs is which
material the field samples lying exactly on a core face get, which is decided by rounding
(a 1e-12 um change of the width moves n_eff by 3e-3). Since v0.1.5 a snapped solve is the
mean of two, with the grid lines just inside and just outside the faces
(tarang.waveguide.dispersion_mapping.snapping); this prints both sides, their mean and the
numerical libraries in use, so two computers can be compared number by number. Local mode
solves only (free), a few minutes.

    python validation/mesh_study/platform_check.py                  # two-sided, 50 steps
    python validation/mesh_study/platform_check.py --snap-sides 1   # as before v0.1.5
    python validation/mesh_study/platform_check.py --steps 60
"""
from __future__ import annotations

import argparse
import platform
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--wl", type=float, default=0.78)
    ap.add_argument("--h", type=float, default=0.35)
    ap.add_argument("--w", type=float, default=0.5)
    ap.add_argument("--num-modes", type=int, default=10)
    ap.add_argument("--snap-sides", type=int, default=2, choices=[1, 2])
    a = ap.parse_args()
    import scipy
    import tidy3d as td
    from tidy3d.plugins.mode import ModeSolver
    from tarang.waveguide.dispersion_mapping import core
    from tarang.waveguide.dispersion_mapping.snapping import snap_offsets
    print(f"python {platform.python_version()} on {platform.system()} {platform.machine()}; "
          f"tidy3d {td.__version__}, numpy {np.__version__}, scipy {scipy.__version__}")
    try:
        from threadpoolctl import threadpool_info
        for lib in threadpool_info():
            print(f"  {lib.get('internal_api')} {lib.get('version')} threads={lib.get('num_threads')}")
    except Exception:                                                  # noqa: BLE001
        print("  (threadpoolctl not installed: BLAS not listed)")
    lo, hi = a.wl - 0.1, a.wl + 0.1
    wls = np.linspace(lo, hi, 9)
    target = core.n_of(core.medium_for(core.DEFAULT_CORE, lo, hi), a.wl)
    core_m = core.medium_for(core.DEFAULT_CORE, lo, hi)
    box_m = core.medium_for(core.DEFAULT_SUBSTRATE, lo, hi)
    print("materials as fitted here (the fit is cached on disk; a different cached fit "
          "changes D):")
    for w in wls:
        print(f"  {w:.4f} um  n_core = {core.n_of(core_m, w):.9f}  n_box = {core.n_of(box_m, w):.9f}")
    sides = []
    for off in snap_offsets(True, a.snap_sides):
        sim = core.build_sim(a.h, a.w, wl0=a.wl, steps=a.steps, snap=True, band=(lo, hi),
                             snap_offset=off)
        plane = td.Box(center=(0, 0, sim.center[2]), size=(0, sim.size[1], sim.size[2]))
        where = ("exactly on the faces" if off == 0 else
                 f"{abs(off):g} um {'inside' if off < 0 else 'outside'} the faces")
        print(f"\ngrid lines {where}: {len(sim.grid.boundaries.y) - 1} x "
              f"{len(sim.grid.boundaries.z) - 1} cells; target n_eff {target:.6f}")
        t = time.time()
        d = ModeSolver(simulation=sim, plane=plane, freqs=td.C_0 / wls,
                       mode_spec=td.ModeSpec(num_modes=a.num_modes, target_neff=target)).solve()
        te0 = []
        for fi in range(len(wls)):
            te = [float(np.real(d.n_eff.isel(f=fi, mode_index=m)))
                  for m in range(a.num_modes) if core._te_frac(d, m, fi) > 0.5]
            te0.append(max(te))
        te0 = np.array(te0)
        sides.append(te0)
        D, _, ng, res = core.dispersion_from_neff(wls, te0, a.wl)
        print(f"  D = {D:.2f} ps/(nm km), n_g = {ng:.6f}, fit residual {res:.2e} "
              f"({time.time() - t:.0f} s)")
        for w, n in zip(wls, te0):
            print(f"  {w:.4f} um  n_eff = {n:.9f}")
    if len(sides) == 2:
        mean = np.mean(sides, axis=0)
        D, _, ng, res = core.dispersion_from_neff(wls, mean, a.wl)
        print(f"\nmean of the two sides: D = {D:.2f} ps/(nm km), n_g = {ng:.6f}")
        for w, n in zip(wls, mean):
            print(f"  {w:.4f} um  n_eff = {n:.9f}")


if __name__ == "__main__":
    main()
