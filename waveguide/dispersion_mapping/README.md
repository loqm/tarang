# `waveguide.dispersion_mapping`

Robust dispersion mapping and design selection for ridge waveguides, built on TARANG's geometry (`geometries.ridge_waveguide`) and media system (`media/media.json`). Everything runs on Tidy3D's **local** mode solver, so no FlexCredits are used.

## What it adds to TARANG

| Capability | Module |
|---|---|
| Per-wavelength TE00/TM00 tracking, polynomial-fit D, β₂, n_g; per-polarization mode counts; A_eff, Γ, loss | `core` |
| Any core / substrate / top cladding from `media.json`; optional Si handle; bent waveguides; core-index shift for uncertainty budgets | `core` |
| Height × width grids, or finalists with a fabrication-tolerance box; resumable, parallel, append-only results | `sweep` |
| Per-height and per-width modal and dispersion plots, design map, 9 physics maps (TE/TM splitting, second-mode margin, sensitivities, robust window, n_g, γ, Γ), dashboard | `maps` |
| Mesh-convergence study and plots | `convergence` |
| Broadband D(λ), zero-dispersion wavelengths, ring integrated dispersion D_int(μ), D₂, D₃ | `broadband` |
| Finalists against a baseline: tolerance, hazards, bends, Q and threshold estimate; markdown report | `report` |

## Usage (from the TARANG repository root)

```powershell
python -m waveguide.dispersion_mapping.sweep --heights 0.60:0.90:0.05 --widths 0.60:1.60:0.10 --steps 50 --workers 4
python -m waveguide.dispersion_mapping.maps --data results/air_sweep_50steps.jsonl --tm
python -m waveguide.dispersion_mapping.convergence --points 0.80x0.80,0.70x1.00 --steps-list 20,30,40,50,70,90
python -m waveguide.dispersion_mapping.sweep --points 0.80x0.80,0.70x1.00 --tol 0.05,0.02 --steps 70
python -m waveguide.dispersion_mapping.broadband --points 0.80x0.80,0.70x1.00 --steps 70 --radius 100
python -m waveguide.dispersion_mapping.sweep --points 0.80x0.80,0.70x1.00 --steps 70 --bend-radius 100
python -m waveguide.dispersion_mapping.report --straight results/air_sweep_70steps_finalists.jsonl --bends "results/air_sweep_70steps_R*um_finalists.jsonl" --finalists 0.80x0.80,0.70x1.00
```

**Material options** (all tools): `--core si3n4`, `--substrate sio2`, `--cladding air`. Any key in `media/media.json` works, e.g. `--core si3n4_hong` or `--cladding sio2`.

**Sweep-only options:** `--n-shift ±0.02` (core-index uncertainty), `--si-handle`, `--bend-radius R`, `--wl 1.30`.

Python API:

```python
from waveguide.dispersion_mapping import solve_point
r = solve_point(0.70, 1.00, steps=70)                       # air-clad Si3N4 on SiO2
r = solve_point(0.74, 1.425, cladding="sio2")      # oxide-clad
r = solve_point(0.70, 1.00, n_shift=+0.02)                  # index-uncertainty study
print(r["D_TE0"], r["guided_TE"], r["neff_TE"][0] - r["neff_TM"][0])
```

## Conventions

- **Data safety.** Result files are append-only and named by their settings (cladding, core, mesh, wavelength, bend radius, index shift), and finished points are skipped on rerun. Plot and report folders get `_v2`, `_v3`, … instead of being overwritten unless `--overwrite` is given.
- **Mesh.** At least 50 steps/λ in the core for maps and 70 for decisions. The TARANG default of 20 is about 25 % off in D.
- **SiO₂ data.** `sio2` now points to the standard Malitson fused-silica data (n = 1.444 at 1550 nm). The previous dataset (Nyakuchena; n = 1.420, 1.10–1.65 µm only) remains available as `sio2_nyakuchena` for reproducing older results.
- **Bends.** Guided-mode counts are not meaningful in bends (`guided_counts_valid: false`), because the solver returns cladding modes there. Take single-mode decisions from straight guides.

## Validation

- **Published devices:** the model reproduces the measured dispersion sign of NIST (603 × 1760 nm, air), Chalmers (600 × 1850 nm normal and 740 × 2000 nm anomalous, SiO₂) and EPFL (740 × 1425 nm, SiO₂).
- **Index sensitivity:** a ±0.02 change of the core index moves D by about ±50 ps/(nm·km) at 700 × 1000 nm.
- **Smoke test:** `python -m pytest tests/test_dispersion_mapping.py`.
