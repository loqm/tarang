<div align="center">

# TARANG

**T**oolkit for **A**utomated **R**esonator **A**nalysis and **N**onlinear-comb **G**eneration

</div>

<p align="center">
  <img src="docs/assets/status.svg" alt="Status: active development. Python 3.11. Tidy3D cloud FDTD backend. MIT licence." width="100%">
</p>

> [!IMPORTANT]
> **This repository is under active development.** Function signatures, returned dictionary
> schemas and default parameters can change between commits without notice. Pin a commit hash
> (`git checkout <sha>`) for any result destined for a thesis or a paper, and record the
> `tidy3d` version alongside it — the mode-solver API has changed across major releases.
> Open items are listed in [§8 Known limitations](#8-known-limitations).
>
> **Packaging is in progress.** TARANG is not yet distributed as an installable package, so for
> now it is used by cloning the repository and working from its root (§3.5). Once the package is
> released, the installation and import instructions below will be replaced.

TARANG is a design and characterisation framework for integrated photonics, built on the
[Tidy3D](https://docs.flexcompute.com/projects/tidy3d/) FDTD and mode-solving backend. It takes
measured refractive-index data at one end and returns fabrication-ready device parameters at the
other, with material fitting, parameter sweeps, result caching and Bayesian optimisation handled
for you.

---

## 1. The pipeline

<p align="center">
  <img src="docs/assets/pipeline.svg" alt="TARANG pipeline: Materials, Waveguide, Ring radius, Ring-bus gap, Grating coupler, Edge coupler, with an automated sweep loop feeding back into the waveguide stage, all running on the Tidy3D cloud FDTD engine." width="100%">
</p>

Each stage consumes the output of the one before it. Material fits set the dispersion used by the
waveguide solver; the waveguide cross-section sets the group index that predicts FSR; the intrinsic
*Q* from the radius sweep sets the coupling condition targeted by the gap sweep. The dashed return
path is the optimisation loop — sweeps and the nested sampler re-parametrise the geometry and
resubmit on their own.

The colour split matters for your budget. Blue stages run the **local** mode solver and cost
nothing; orange stages submit **cloud FDTD** jobs metered in FlexCredits. Develop a geometry
entirely in blue before spending anything on orange.

---

## 2. Capabilities

| Stage | What it produces | Entry point |
|---|---|---|
| Materials | `PoleResidue` media fitted to measured *n*, *k*, cached and hash-validated | `media.get_medium` |
| Waveguide | *n*<sub>eff</sub>, TE/TM classification, *D* [ps/(nm·km)], Γ, *A*<sub>eff</sub> | `waveguide.RidgeWaveguide` |
| Waveguide sweeps | Width × height maps with regime shading and single-mode boundaries | `modes_in_geometry`, `dispersion_in_geometry`, `confinement_in_geometry` |
| Ring radius | Bend loss by cutback across arc angles, *Q*<sub>int</sub>, *n*<sub>g</sub>, FSR | `ringResonator.ring_radius_sweep` |
| Ring–bus gap | Coupling against gap, critical-coupling point | `ringResonator.run_gap_optimization_batch` |
| Grating coupler | 2D/3D coupling efficiency; nested-sampling design search | `gratingCoupler.GratingCoupler`, `nested_sampler` |
| Edge coupler | Inverse-taper efficiency, insertion loss, tip × taper sweep matrix | `edgeCoupler` |
| Visualisation | Geometry slices and projections, GetDist corner plots | `utilities` |

---

## 3. Setup

<p align="center">
  <img src="docs/assets/setup.svg" alt="Five setup steps: create the conda environment, add the pip-only packages, authenticate with Tidy3D, register the Jupyter kernel, and work from the repository root." width="100%">
</p>

### 3.1 Environment

```bash
conda env create -f environment.yml
conda activate tarang-env
```

This pins Python 3.11 with NumPy, SciPy, Matplotlib, MPICH and — the awkward one —
**MultiNest + PyMultiNest**. The Fortran sampler is why this is an `environment.yml` rather than a
`requirements.txt`; installing it from conda-forge avoids a source build.

### 3.2 Pip-only packages

`environment.yml` does not yet list every module the code imports. Inside the activated
environment:

```bash
pip install tidy3d pandas getdist python-dotenv
```

| Package | Needed by |
|---|---|
| `tidy3d` | every module — the FDTD and mode-solving backend |
| `pandas` | `media/medium_helper.py`, `media/format_csv.py` |
| `getdist` | `utilities/nested_sampling_post_process.py` |
| `python-dotenv` | optional `.env` loading in `gratingCoupler/grating_coupler.py` |

Pin `tidy3d` to the release you validate against.

### 3.3 Credentials

```bash
tidy3d configure        # writes ~/.tidy3d/config
```

Or place a `.env` in the repository root (already git-ignored):

```
TIDY3D_API_KEY=your_key_here
```

The code accepts either `TIDY3D_KEY` or `TIDY3D_API_KEY`, and logs a warning before any cloud
submission if neither is set.

### 3.4 Jupyter kernel

```bash
python -m ipykernel install --user --name tarang-env --display-name "TARANG"
```

### 3.5 Always run from the repository root

There is no `pyproject.toml` or `setup.py` yet — a packaged release is being prepared, and until
it lands, packages resolve by directory name. Two things break
from any other working directory:

1. `import geometries`, `import waveguide`, … fail with `ModuleNotFoundError`.
2. `media/medium_helper.py` opens `media.json` at a **relative** path and raises `FileNotFoundError`.

Launch `jupyter lab` from `tarang-main/`, and run scripts as `python nested_sampling_example.py`
from the same place.

---

## 4. First run — about fifteen minutes, zero credits

Everything below runs locally. No FDTD job is submitted.

**Step 1 — fit a material.**

```python
from media import get_medium, get_medium_dispersion

SiN  = get_medium("si3n4", wl_range=(1.35, 1.75), tolerance=1e-5)
SiO2 = get_medium("sio2",  wl_range=(1.35, 1.75), tolerance=5e-5)

get_medium_dispersion(
    medium_key="si3n4",
    output_path="example_results/si3n4/dispersion.png",
    force_zero_k=True,
    show=True,
)
```

The first call runs a pole-residue fit and writes it to `media/fitted_media/` with a manifest entry
keyed on CSV hash, wavelength range, pole count and RMS tolerance. Compatible later calls are cache
hits — watch for `[Medium] Cache HIT` in the log. Pass `use_cache=False` to force a refit.
`get_medium("air")` short-circuits the pipeline and returns a dispersionless `td.Medium`.

**Step 2 — build a waveguide and inspect it.**

```python
from geometries import ridge_waveguide
from utilities.sim_plotter import display_3d

wg = ridge_waveguide.get_passive_simulation(
    width=0.800, height=0.800, length=2.0,
    core_medium=SiN, sub_medium=SiO2,
    substrate_height=4.0,
    wavelength_um=1.550,
)

display_3d(wg, output_path="example_results/waveguide/geometry.png", save=True)
```

`get_passive_simulation` gives structures and mesh only. `get_active_simulation` adds a
`ModeSource` and `ModeMonitor` for a full FDTD run.

**Step 3 — extract the physics.**

```python
from pprint import pprint

pprint(wg.get_modes(num_modes=6))    # n_eff, TE fraction, TE/TM label, guided flag
pprint(wg.get_dispersion())          # D [ps/(nm·km)], regime, single-mode flag
pprint(wg.get_confinement())         # Γ and A_eff [µm²] for the fundamental TE00 mode
```

These call `ModeSolver.solve()` locally — free and fast. Only `web.run(...)`, reached through
`run_sim()`, `get_efficiency()`, the gap and radius sweeps and the nested sampler, spends credits.

**Step 4 — run a sweep.**

```python
import waveguide

waveguide.dispersion_in_geometry(
    sweep_heights=[0.60, 0.80],
    sweep_widths=[0.9, 1.2, 1.5, 1.8],
    core_medium=SiN, sub_medium=SiO2,
    wavelength_um=1.550,
    substrate_height=4.0,
    output_dir_name="example_results/dispersion_sweep",
    preview_geometry=True,
)
```

One figure per height, one point per width, normal and anomalous regions shaded, single-mode
boundary marked. `modes_in_geometry` and `confinement_in_geometry` take the same argument shape.

**Step 5 — then go to the cloud.** Work through the notebooks in this order:

| # | Notebook | Credits |
|---|---|---|
| 1 | `examples_media.ipynb` — material fitting, curves checked against published data | none |
| 2 | `waveguide_example.ipynb` — modes, dispersion, confinement, sweeps | none |
| 3 | `ring_resonator_example.ipynb` — ring geometries, type conversion, FSR, gap and radius optimisation | yes |
| 4 | `edge_coupler_example.ipynb`, `grating_coupler_example.ipynb` — I/O structures | yes |
| 5 | `nested_sampling_example.py` — Bayesian design search | yes, capped |

---

## 5. Repository layout

<p align="center">
  <img src="docs/assets/architecture.svg" alt="Architecture: media fits materials, geometries builds Tidy3D simulations, device packages add physics and sweeps, utilities produce figures and posteriors." width="100%">
</p>

Every device follows the same two-layer split. `geometries/*` builds a raw `td.Simulation` and
wraps it in a physics object; the device package holds the analysis. Wrappers expose `.sim`, and
`RidgeWaveguide` forwards unknown attributes to the underlying simulation through `__getattr__`,
so `wg.plot(z=0)` reaches Tidy3D directly.

```
tarang-main/
│
├── media.json                      # material key → CSV path registry (read from CWD)
├── environment.yml                 # conda environment (see §3.2 for the pip gap)
├── LICENSE
│
├── media/                          # material models
│   ├── medium_helper.py            #   get_medium(), get_medium_dispersion()
│   ├── format_csv.py               #   repairs stacked refractiveindex.info CSVs in place
│   ├── data/                       #   raw n/k datasets (see §6)
│   └── fitted_media/               #   auto-created fit cache + manifest (git-ignored)
│
├── geometries/                     # Tidy3D geometry builders — no analysis logic
│   ├── ridge_waveguide.py          #   get_passive_simulation / get_active_simulation
│   ├── grating_coupler.py
│   ├── edge_coupler.py
│   ├── half_ring_resonator.py
│   ├── full_ring_resonator.py
│   ├── partial_ring_resonator.py
│   └── resonator_arc.py
│
├── waveguide/                      # ridge-waveguide physics
│   ├── waveguide.py                #   RidgeWaveguide: get_modes / get_dispersion /
│   │                               #   get_dispersion_vs_wavelength / get_confinement
│   ├── mode_sweep.py               #   modes_in_geometry()
│   ├── dispersion_sweep.py         #   dispersion_in_geometry()
│   └── confinement_sweep.py        #   confinement_in_geometry()
│
├── ringResonator/                  # resonator physics
│   ├── ring_resonator.py           #   BaseResonator, halfringResonator, ringResonator,
│   │                               #   partialRingResonator (+ .to_half_ring()/.to_full_ring())
│   ├── resonator_arc.py            #   resonatorArc — bend-loss test structures
│   ├── ring_radius_sweep.py        #   radius → bend loss, Q_int, FSR
│   └── ring_gap_sweep.py           #   gap → coupling regime
│
├── gratingCoupler/
│   ├── grating_coupler.py          #   GratingConfig (dataclass) + GratingCoupler
│   └── nested_sampler.py           #   GratingCouplerBayesianOptimizer (PyMultiNest)
│
├── edgeCoupler/
│   ├── edge_coupler.py             #   edgeCoupler: efficiency, insertion loss, fields
│   └── optimization_sweep.py       #   parametric_optimization_sweep()
│
├── utilities/
│   ├── sim_plotter.py              #   display_2d(), display_3d()
│   └── nested_sampling_post_process.py
│
├── docs/assets/                    # figures used by this README
│
└── *_example.ipynb                 # one notebook per device family
```

---

## 6. Material data

Registered in `media.json`. Verified coverage of the shipped datasets:

| Key | File | Range (µm) | Columns |
|---|---|---|---|
| `si3n4` | `Luke_Si3N4.csv` | 0.310 – 5.504 | wl, n |
| `si3n4_hong` | `Hong_Si3N4_LPCVD.csv` | 0.423 – 1.633 | wl, n |
| `sio2` | `Nyakuchena_fusedSilica.csv` | **1.100 – 1.650** | wl, n |
| `si` | `Wang_Si_25C.csv` | **0.400 – 1.305** | wl, n, k |
| `si` | `_Franta_Si_300K.csv` | 0.031 – 309.96 | wl, n, k |
| `air` | — | any | handled internally, *n* = 1 |


To add a material: drop a CSV with columns `wl, n[, k]` (wavelength in µm, strictly increasing)
into `media/data/`, add a lowercase key to `media.json`, and call `get_medium`. If the file is a
stacked refractiveindex.info export (an `n` block followed by a `wl,k` block),
`format_csv.fix_stacked_csv` runs automatically and **rewrites the file in place** — keep a
pristine copy.

---

## 7. Cost control

Cloud FDTD is metered, and the repository is built around not paying twice for the same result.

- **Fitted media** cache in `media/fitted_media/`, validated against the CSV hash.
- **Simulation results** are written as `.hdf5` into the output directory you pass; analysis
  methods read the local file instead of resubmitting. `BaseResonator.load_sim_data(path)` reloads
  one by hand.
- **Sweep manifests** (`intrinsic_physics_manifest.json`, `analytic_coupling_manifest.json`)
  short-circuit an entire sweep on re-run. Delete the manifest, or pass `force_rerun=True` where
  available, to actually recompute.
- **The nested sampler** enforces a hard cap. `GratingCouplerBayesianOptimizer(..., credit_limit=…)`
  and `execute_search_with_budget(credit_limit=…)` estimate the next simulation's cost and stop
  before submitting if it would exceed the cap, returning `status="PAUSED_BUDGET_EXCEEDED"`.
  Raising the limit and re-running against the same output directory resumes MultiNest from its
  existing chains.

Start every device in 2D (`is_3d=False` for grating couplers) with coarse `fine_steps_per_wvl` and
`coarse_steps_per_wvl`; refine only once the geometry is sane. Outputs, `*.png`, `*.hdf5`,
`results/`, `*example_results/` and `pipeline.log` are all git-ignored.

---

## 8. Known limitations

Open items as of this revision — worth checking before you rely on a result.

- `environment.yml` omits `tidy3d`, `pandas`, `getdist` and `python-dotenv` (§3.2).
- The repository is not yet installable as a package, so the working directory must be the root
  (§3.5). Packaging is under way; this section will be revised when it ships.
- Grating-coupler simulation caches are keyed on geometry only, **not on the media used to build
  the simulation** — a cached result may be served for a different material stack. This is flagged
  in `grating_coupler_example.ipynb`. Clear the cache directory when you change media.
- `get_medium` validates only that *some* data falls inside the requested `wl_range`, so a partially
  covered window extrapolates silently (§6).
- When no guided mode exists, `RidgeWaveguide.get_modes` logs an error and returns anyway rather
  than raising; the returned row is physically meaningless.
- `format_csv.fix_stacked_csv` mutates source data files in place.

---

## 9. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `FileNotFoundError: media.json` | Not running from the repository root. See §3.5. |
| `ModuleNotFoundError: geometries` | Same cause — the root must be on `sys.path`. |
| `ModuleNotFoundError: tidy3d` / `pandas` / `getdist` | Missing from `environment.yml`. See §3.2. |
| `TIDY3D_API_KEY not found in environment` (warning) | Run `tidy3d configure`, or add the key to `.env`. |
| `WaveguideModeError: TE00 mode not found` | No mode with TE fraction > 0.5 in the solver window. Widen the core, raise `num_modes`, or enlarge `sim_size_y` / `sim_size_z`. |
| `"No guided modes found"` in the log, no exception | *n*<sub>eff</sub> ≤ *n*<sub>substrate</sub> for every mode — the geometry is below cutoff. |
| `ValueError: sim_size_z must exceed …` | The fixed vertical domain is smaller than substrate + core + cladding. |
| `Requested wl_range has ZERO data points` | The material CSV does not cover that window. See §6. |
| Sweep returns instantly, no simulations | Manifest cache hit. Delete the manifest JSON or pass `force_rerun=True`. |
| PyMultiNest import or link errors | The MultiNest Fortran library did not install cleanly. Reinstall from conda-forge inside the environment; do not mix in a pip `pymultinest`. |

Everything logs to `pipeline.log` in the working directory as well as to stdout, configured in
`media/medium_helper.py`.

---

## 10. Data provenance

Bundled datasets are named for their published sources. As cited in `examples_media.ipynb`:

- Luke *et al.*, "Broadband mid-infrared frequency comb generation in a Si₃N₄ microresonator,"
  *Opt. Lett.* **40**, 4823 (2015) — [10.1364/OL.40.004823](https://doi.org/10.1364/OL.40.004823) → `si3n4`
- Hong *et al.*, "Dispersion Optimization of Silicon Nitride Waveguides for Efficient Four-Wave
  Mixing," *Photonics* **8**, 161 (2021) — [10.3390/photonics8050161](https://doi.org/10.3390/photonics8050161) → `si3n4_hong`

Tooling: [Tidy3D](https://docs.flexcompute.com/projects/tidy3d/) ·
[PyMultiNest](https://johannesbuchner.github.io/PyMultiNest/) ·
Feroz, Hobson & Bridges, *MNRAS* **398**, 1601 (2009) for MultiNest ·
[GetDist](https://getdist.readthedocs.io/) for posterior plotting.

---

## 11. Licence and authors

MIT — © 2026 Laboratory of Optics of Quantum Materials.

Esraaj Sarkar Gupta, Divyanshu Tiwari and Anshuman Kumar.
