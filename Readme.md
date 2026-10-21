<div align="center">

# TARANG

**T**oolkit for **A**utomated **R**esonator **A**nalysis and **N**onlinear-comb **G**eneration

</div>

<p align="center">
  <img src="docs/assets/status.svg" alt="Status: active development. Python 3.11 and newer. Tidy3D cloud FDTD backend. MIT licence." width="100%">
</p>

TARANG designs integrated photonic devices on the [Tidy3D](https://docs.flexcompute.com/projects/tidy3d/)
mode solver and FDTD engine: from measured refractive-index data to waveguide, ring-resonator and
coupler parameters, with material fitting, sweeps, result caching, cost limits and Bayesian
optimisation built in. The current focus is silicon-nitride microresonators for Kerr frequency
combs. Record the TARANG version (or commit) and the `tidy3d` version with every result you
publish.

| Section | What it gives you |
|---|---|
| [Where to start](#where-to-start) | the path from installation to your first cloud run |
| [Pipeline](#pipeline) and [Capabilities](#capabilities) | what each stage computes, and which stages cost credits |
| [Installation](#installation) | one block of commands |
| [Quick start](#quick-start-local-no-credits) | five steps, all local and free, with the figures they produce |
| [Guided design](#guided-design-one-command) | one command from materials to GDS, at any wavelength |
| [Cloud runs and cost control](#cloud-runs-and-cost-control) | caching, cost limits, resuming |
| [Materials](#materials), [Logging](#logging), [Repository layout](#repository-layout) | reference, including the two `tarang.migrate` commands for old scripts |
| [Known limitations](#known-limitations), [Troubleshooting](#troubleshooting) | what to watch for, and fixes for common errors |
| [Contributing, citing, licence](#contributing-citing-licence) | how to contribute, what to cite, MIT licence; changes per release in [CHANGELOG.md](CHANGELOG.md) |

---

## Where to start

<p align="center">
  <img src="docs/assets/start_here.svg" alt="Where to start: 1 install; 2 the README quick start, local and free; 3 the local notebooks media_examples, waveguide_example and dispersion_mapping_example, free; 4 the cloud notebooks cloud_basics_example, ring_resonator_example, grating_coupler_example, edge_coupler_example and nested_sampling_example.py, with a cost cap." width="100%">
</p>

## Pipeline

<p align="center">
  <img src="docs/assets/pipeline.svg" alt="TARANG pipeline: Materials, Waveguide, Ring radius, Ring-bus gap, Grating coupler, Edge coupler, with an automated sweep loop feeding back into the waveguide stage, all running on the Tidy3D cloud FDTD engine." width="100%">
</p>

Material fits set the waveguide dispersion; the waveguide's group index sets the ring's FSR; the
intrinsic *Q* from the radius sweep sets the coupling target for the gap sweep. Blue stages use
the **local** mode solver and are free; orange stages run **cloud FDTD**, paid in FlexCredits.
Settle a geometry in the blue stages before running orange ones.

## Capabilities

| Stage | What it produces | Entry point |
|---|---|---|
| Materials | Pole-residue media fitted to measured *n*, *k*; cached | `tarang.media.get_medium`, `get_medium_dispersion` |
| Waveguide | *n*<sub>eff</sub>, TE/TM labels, *D* [ps/(nm·km)], Γ, *A*<sub>eff</sub> | `tarang.waveguide.RidgeWaveguide` |
| Waveguide sweeps | Width × height maps of modes, dispersion, confinement | `tarang.waveguide.modes_in_geometry`, `dispersion_in_geometry`, `confinement_in_geometry` |
| Dispersion mapping | Converged *D* maps, *D*(λ), *D*<sub>int</sub>, finalist reports (command line) | `tarang.waveguide.dispersion_mapping` ([README](tarang/waveguide/dispersion_mapping/README.md)) |
| Rings | Full, half and partial rings; FSR, group index, mode fluxes | `tarang.ring_resonator.RingResonator`, `HalfRingResonator`, `PartialRingResonator` |
| Ring radius | Bend loss over arc angles, *Q*<sub>int</sub>, FSR | `tarang.ring_resonator.ring_radius_sweep.radius_sweep` |
| Ring–bus gap | Coupling against gap, critical-coupling gap | `tarang.ring_resonator.ring_gap_sweep.ring_gap_sweep`, `radius_gap_sweep` |
| Grating coupler | 2D/3D coupling efficiency and spectrum, fibre towards or away from the waveguide; nested-sampling design search | `tarang.grating_coupler.GratingCoupler`, `tarang.grating_coupler.nested_sampler` |
| Edge coupler | Inverse-taper efficiency; tip × taper sweep | `tarang.edge_coupler.EdgeCoupler`, `tarang.edge_coupler.parametric_optimization_sweep` |
| Plots | Geometry views, corner plots | `tarang.utilities.sim_plotter`, `tarang.utilities.plot_corner` |
| Guided design | All stages in one session: cross-section, ring, gap, grating coupler, GDS, report | `python -m tarang.design` |

---

## Installation

Python 3.11 or newer on Windows, Linux or macOS. TARANG is installed from this repository
(not from PyPI), in editable mode, so local edits take effect immediately.

```bash
git clone https://github.com/loqm/tarang
cd tarang
conda env create -f environment.yml      # environment "tarang", with all extras
conda activate tarang
```

Without conda, into an existing environment: `pip install -e ".[plots,notebooks]"`.

| Extra | Adds |
|---|---|
| (none) | `tidy3d` ≥ 2.12, < 3; `numpy`, `pandas`, `pydantic`, `matplotlib`, `dynesty` |
| `[plots]` | `getdist`, `corner` (posterior plots) |
| `[notebooks]` | `jupyterlab`, `ipykernel`, `ipython` |
| `[layout]` | `gdstk` (GDS export of the guided design) |
| `[test]` | `pytest`, `pytest-cov`, plus `[plots]` and `[layout]` |
| `[multinest]` | `pymultinest` (needs the MultiNest library, below) |

Then, once per machine:

```bash
tidy3d configure                                   # Tidy3D API key, needed for cloud runs only
python -m ipykernel install --user --name tarang --display-name "TARANG"
python -m pytest -m "not slow"                     # check: about a minute, no credits
```

Optional, Linux and macOS only: `conda install -c conda-forge multinest pymultinest` adds the
MultiNest sampler; otherwise the nested sampler uses dynesty. After pulling new code, run
`conda env update -f environment.yml`.

---

## Quick start (local, no credits)

Everything here runs on your computer; files are written relative to the current folder.
Steps 1–4 produce the three results below (generated by `docs/make_readme_figures.py`).

<p align="center">
  <img src="docs/assets/quickstart_results.png" alt="Quick-start results: fitted refractive index of Si3N4 and SiO2 from 1.2 to 2.0 um; the TE00 mode intensity of a 0.7 um by 1.2 um air-clad Si3N4 waveguide at 1550 nm; a dispersion map over core height and width with the zero-dispersion line and the multimode region marked." width="100%">
</p>

**1. Fit the materials.**

```python
from tarang.media import get_medium, get_medium_dispersion

SiN = get_medium("si3n4", wl_range=(1.35, 1.75))
SiO2 = get_medium("sio2", wl_range=(1.35, 1.75))

get_medium_dispersion("si3n4", output_path="example_results/si3n4_dispersion.png",
                      wl_range=(1.35, 1.75), force_zero_k=True)
```

Fits are cached in `~/.cache/tarang/fitted_media` (change with `TARANG_CACHE_DIR`) and reused
by later calls; `use_cache=False` forces a refit. Fit over the band you will simulate.

**2. Build a waveguide.**

```python
from tarang.geometries import ridge_waveguide
from tarang.utilities.sim_plotter import display_3d

wg = ridge_waveguide.get_passive_simulation(
    width=1.2, height=0.7, length=2.0,
    core_medium=SiN, box_medium=SiO2, box_height=4.0,
    wavelength_um=1.55,
    fine_steps_per_wvl=50,      # converged dispersion needs >= 50 (default 20 is ~25 % off)
)

display_3d(wg, output_path="example_results/waveguide_geometry.png", show=False, save=True)
```

Lengths are in µm. Layer arguments are the same for every device: `core_medium`;
`box_medium`, `box_height` (buried oxide); `top_cladding_medium`, `top_cladding_height` (air
if omitted); `handle_medium`, `handle_height` (Si wafer, grating coupler only).
`get_active_simulation` also adds a mode source and monitors for FDTD.

**3. Compute the mode physics.**

```python
modes = wg.get_modes(num_modes=4)     # n_eff, TE fraction, TE/TM label, guided flag
disp = wg.get_dispersion()            # D [ps/(nm km)], regime, single-mode flag
conf = wg.get_confinement()           # Γ and A_eff [µm²] of the TE00 mode
```

**4. Sweep the cross-section.**

```python
from tarang import waveguide

waveguide.dispersion_in_geometry(
    sweep_heights=[0.6, 0.8],
    sweep_widths=[0.9, 1.2, 1.5, 1.8],
    core_medium=SiN, box_medium=SiO2, box_height=4.0,
    wavelength_um=1.55,
    output_dir_name="example_results/dispersion_sweep",
)
```

`modes_in_geometry` and `confinement_in_geometry` take the same arguments. For converged maps,
*D*(λ) and finalist selection, use the [dispersion-mapping tools](tarang/waveguide/dispersion_mapping/README.md):

```bash
python -m tarang.waveguide.dispersion_mapping.sweep --heights 0.60:0.90:0.05 --widths 0.6:1.6:0.1
```

**5. Continue with the notebooks** in `examples/`, in this order (stored without outputs;
each starts with a table of the functions it shows):

| Notebook | Covers | Credits |
|---|---|---|
| `media_examples.ipynb` | material fits against published data | none |
| `waveguide_example.ipynb` | modes, Γ, *D*, *D*(λ), guided-mode check, width × height sweeps | none |
| `dispersion_mapping_example.ipynb` | choosing *H* and *W*: design map, mesh convergence, tolerance report, *D*<sub>int</sub> | none |
| `cloud_basics_example.ipynb` | fingerprints, cache, cost limits (one optional capped run) | none by default |
| `ring_resonator_example.ipynb` | ring geometries, FSR, bend loss and *Q*<sub>int</sub>, critical-coupling gap | yes, capped |
| `grating_coupler_example.ipynb`, `edge_coupler_example.ipynb` | coupler efficiency, fields, taper sweep | yes, capped |
| `nested_sampling_example.py` | Bayesian grating-coupler search | yes, capped |

---

## Guided design (one command)

`python -m tarang.design` runs the whole pipeline as one session, for any wavelength the
material data cover:

```
materials -> cross-section -> ring -> ring-bus gap -> grating coupler -> checks -> taper -> apodised grating -> focusing grating -> layout -> report
  local          local        local      cloud             cloud        free+cloud  free+cloud  cloud (opt-in)   free (opt-in)     local     local
```

Each stage asks only what it needs (the default is shown; Enter accepts it), then prints
its results and saves its figures and tables. Cloud stages first show the FlexCredit
estimate and wait for your confirmation. Your answers are saved in `design.toml`, so the
same design can be repeated, or continued after a stop, without questions:

Each stage shows **one table** of its settings: press Enter to use them, type a row number
to change that value, `?` to explain every row, `q` to stop. Values already saved in
`design.toml` are shown in a compact list and not asked again (`--review` asks anyway).

### Every command

Below, `D` stands for your design's settings file, e.g. `design_runs/my_ring/design.toml`
(on Windows `design_runs\my_ring\design.toml`).

| To | Run |
|---|---|
| start a new design (asks everything) | `python -m tarang.design` |
| continue a stopped design | `python -m tarang.design --config D` |
| repeat with no questions | `python -m tarang.design --config D --yes` |
| look at every setting again | `python -m tarang.design --config D --review` |
| run some stages only | `python -m tarang.design --config D --only gap,grating` |
| run a stage again (new results beside the old) | `python -m tarang.design --config D --only grating --redo grating` |
| run the extra checks (tolerances, band, mode families, loss budget) | `python -m tarang.design --config D --only checks` |
| design the taper and the apodised grating, then the layout and report | `python -m tarang.design --config D --only taper,apodise,layout,report` (set `apodise.run = true` first: the apodised grating is opt-in) |
| design a focusing grating (curved teeth, no long taper) | `python -m tarang.design --config D --only focus,layout,report` (set `focus.run = true` first; opt-in, free unless `focus.check_3d`) |
| redraw a design's figures in the current style (free) | `python -m tarang.design --config D --only report --redo report` |
| open each figure as it is made | add `--show` |
| list the stages | `python -m tarang.design --list` |
| credits: limit, spent, what on, account balance | `python -m tarang.design --config D --credits` |
| set this design's credit limit to 20 FC | `python -m tarang.design --config D --budget 20` |
| start the spend count at 0 again (e.g. before a new phase) | `python -m tarang.design --config D --reset-spent` |
| list every cloud run with its Tidy3D web link (and draw a picture of each one's geometry) | `python -m tarang.design --config D --runs` |
| figure formats and resolution | in `D`: `run.figure_formats = ["png", "pdf"]` (also `"svg"`), `run.figure_dpi = 300` |

**Stopping is always safe.** Type `q` at any question, or press Ctrl+C at any time
(also during a cloud run). Finished stages and finished cloud runs are kept, and your
answers are in `design.toml`. A cloud run that was in progress keeps running on Tidy3D;
the same command later picks it up again (it is not submitted or paid twice). Closing
the terminal has the same effect as Ctrl+C.

**Credits.** Two numbers decide whether a cloud stage may run: the design's limit
`run.max_total_cost` (FlexCredits, in `D`) and TARANG's running total of the estimates
it has spent (`spent` in `state.json`). A stage whose estimate is more than what is left
is not run. `--budget` changes the limit, `--reset-spent` starts the total at 0 again
(the history stays in `state.json`; `--credits` shows it by stage and date). Neither
changes your Tidy3D account. `run.max_cost_per_run` caps any single simulation.

**Web links and pictures.** Every cloud run is recorded in a `task.json` next to its
results, with its Tidy3D task id and a link that opens the simulation (geometry, sources,
monitors, results) in the Tidy3D web interface; the links are printed as runs start and
listed in `tables/cloud_runs.csv` and the report. A picture of each run's geometry is saved
in `figures/sims/`, named like its cloud folder (`run.sim_pictures`). `--runs` also finds
the links of runs made before these records existed, as long as they were made with the
Tidy3D account of the current API key (a link only opens in the account that ran it).

| Stage | Asks for | Produces |
|---|---|---|
| Materials | wavelength, core, box, cladding (a key, or your own CSV of measured *n*, *k*) | fitted *n*(λ), *k*(λ) table and plot |
| Cross-section | film thickness if fixed, map ranges, tolerances | design map, ranked candidates, tolerance and mesh check of your choice |
| Ring | FSR target, losses (a published preset, or your own values) | radius, *Q*<sub>int</sub>, bend loss, *D*(λ), *D*<sub>int</sub>, threshold estimate |
| Gap | gaps, over-coupling ratio | coupled power per gap (read where light enters the ring), critical and over-coupled gap, loaded *Q* |
| Grating coupler | etch, fibre angle, orientation and MFD, smallest feature | spectra of a period × fill-factor map (and its efficiency landscape, 5g), the period tuned onto your wavelength, fill-factor / period / fibre-position sensitivity, 1 dB and 3 dB bandwidths; optional 3D check |
| Checks | which checks (each cloud check is priced first) | gap tolerance (K, loaded Q, dip depth), coupled-mode-theory check of the gap, FSR tolerance, ring mode families and crossings, fibre-to-fibre loss budget; optional: coupling across the band, grating angle / film / tooth-width tolerances, back-reflection and directionality, mesh checks |
| Taper | grating width (or the best for the fibre), taper lengths to try, loss accepted | lateral overlap of the fibre with the grating against its width (effective-index method, free), taper loss against length (one Tidy3D EME run), the shortest taper below the accepted loss; the layout uses both |
| Apodised grating (opt-in, `apodise.run`; off: the uniform grating is used and nothing is paid) | fill factors to measure, narrowest trench the full etch clears, whether to check in FDTD | scattering strength and phase-matched period of each fill factor (small 2D runs), a tooth list whose scattering follows the fibre's Gaussian, the predicted gain (1D model, before anything else is paid), a 2D FDTD check; the layout uses it when it beats the uniform grating |
| Focusing grating (opt-in, `focus.run`) | teeth (uniform, apodised or the better one), tip widths, largest light-cone angle, optional 3D check | curved teeth (confocal ellipses around the waveguide tip) on the straight grating's cross-section, the in-plane focusing efficiency against tip-to-grating distance and tip width (free model, landscape figure), the 3D estimate against the straight grating with its taper; optional 3D FDTD (priced); the GDS gets `*_fgc` cells with it beside the straight-grating `*_gc` cells |
| Layout | layer numbers, tone (structure or trench) | `waveguide.gds`, `ring_waveguide.gds`, `ring_waveguide_gc.gds`, `parameters.json`/`.m`/`.csv`, layout figure; `lle/` inputs for a comb (Lugiato-Lefever) solver, incl. pyLLE's dispersion-file format |
| Report | | `report.md` with every figure, `benchmark_pack/` (and its zip), `design_summary.json`, `tables/cloud_runs.csv` |

While it runs, the tool shows what it is about to simulate (a picture of each geometry,
before any cost question), a line for every cloud run, and the figures of each stage
(`--show` opens them). It checks that your FlexCredits are valid before a cloud stage. At the
start it lists the values it assumes (losses, *n*<sub>2</sub>, fibre MFD, tolerances) with
their published sources; your own values replace them.

`benchmark_pack/` holds everything another solver needs: a step-by-step README (geometry,
materials, mesh, sources, monitors, how each number was obtained), `assumptions.csv`,
`results.csv`, the exact *n*, *k* data used, all figures, tables, simulation files and the GDS.
GDS export needs the `layout` extra (`pip install -e ".[layout]"`; included in
`environment.yml`).

**Fibre orientation.** The fibre beam can travel *towards* the waveguide (the default:
the fibre's upper end leans away from it, and two fibres feeding the two mirrored couplers
form a V) or *away* from it:

```
   towards (default)                  away
                  / fibre             fibre \
                 /                           \
   waveguide ===|||||||       waveguide ===|||||||
   period  λ / (n_eff − sin θ)          λ / (n_eff + sin θ)
```

`GratingConfig(..., theta_deg=12, fiber_tilt="towards")`; `theta_deg` is the angle from
vertical (its sign is ignored) and `source_x` the beam centre on the grating surface.

---

## Cloud runs and cost control

<p align="center">
  <img src="docs/assets/cloud_flow.svg" alt="Every cloud run: build the simulation, compute its fingerprint; if a verified result is on disk, load it for free; otherwise upload a draft and get the cost estimate; above the limit, delete the draft and raise CostLimitExceededError with nothing charged; otherwise run and store the result under its fingerprint, so the same simulation later is found on disk." width="100%">
</p>

* **Cached by simulation.** Each result is stored under a SHA-256 fingerprint of the complete
  Tidy3D simulation, next to its `simulation.json`, in the output folder you pass. An identical
  simulation is loaded from disk; any change, including a material, runs again. Results are
  never overwritten.
* **Priced before it starts.** A job whose estimate exceeds its limit is deleted unstarted and
  raises `CostLimitExceededError`. Set the limit per device (`device.max_cost = 0.5`), per
  batch (`parametric_optimization_sweep(..., max_cost=...)`), or for the whole session with
  the environment variable `TARANG_MAX_COST`, which also covers sweeps that build devices
  internally.
* **Sweeps resume.** Cached points are skipped; `force_rerun=True` recomputes.
* **Capped optimisation.** The nested sampler stops before exceeding `credit_limit` and
  returns `status="PAUSED_BUDGET_EXCEEDED"`; raise the limit and rerun in the same folder to
  continue from the cache.

Start in 2D (`is_3d=False` for grating couplers) with a coarse mesh; refine once the geometry
is right.

---

## Materials

| Key | Material | Range (µm) |
|---|---|---|
| `si3n4` | Stoichiometric Si₃N₄ (Luke 2015) | 0.31 – 5.50 |
| `si3n4_hong` | LPCVD SiN (Hong 2021) | 0.42 – 1.63 |
| `sio2` | Fused silica (Malitson 1965) | 0.40 – 3.00 |
| `sio2_nyakuchena` | Silica microspheres (Nyakuchena 2023) | 1.10 – 1.65 |
| `si` | Crystalline Si film, 25 °C (Wang 2025) | 0.40 – 1.31 |
| `si_franta` | Float-zone c-Si, 300 K (Franta 2017) | 0.03 – 310 |
| `air` | n = 1 | any |

Choose a dataset whose range covers your whole simulation band; outside it the fit is an
extrapolation and TARANG warns. Your own measured data (for example ellipsometry of your
film) can be used directly by path: `get_medium("data/my_film.csv", wl_range=(1.3, 1.8))`,
with columns `wl` (µm), `n` and optionally `k`. References, recommendations per wavelength and how to add a
material: [SOURCES.md](tarang/media/data/SOURCES.md).

## Logging

```python
from tarang import configure_logging, set_tidy3d_log_level

configure_logging("INFO")                          # to the console
configure_logging("INFO", log_file="run.log")      # and to a UTF-8 file
set_tidy3d_log_level("WARNING")                    # Tidy3D's own messages
```

---

## Repository layout

<p align="center">
  <img src="docs/assets/architecture.svg" alt="Architecture: media fits materials, geometries builds Tidy3D simulations, device packages add physics and sweeps, utilities produce figures and posteriors." width="100%">
</p>

```
tarang/                 the Python package
├── media/              material data, fitting and cache
├── geometries/         Tidy3D simulation builders (no analysis)
├── waveguide/          RidgeWaveguide, sweeps, dispersion_mapping/
├── ring_resonator/     ring classes, radius and gap sweeps
├── grating_coupler/    GratingCoupler, nested sampler
├── edge_coupler/       EdgeCoupler, taper sweep
├── utilities/          cloud runs (cache, cost guard), samplers, plotting
├── design/             python -m tarang.design: the guided design tool
└── migrate.py          updates scripts written for the old layout
examples/               notebooks and nested_sampling_example.py
tests/                  test suite (tests/README.md)
docs/                   figures
```

Builders in `tarang.geometries` return a wrapper holding the `td.Simulation` on `.sim`; the
analysis lives in the device package.

### Upgrading old scripts

Before v0.1.0 the packages were top-level (`from media import ...`, `import gratingCoupler`).
This rewrites the imports in your scripts and notebooks; the rest of the code needs no change:

```bash
python -m tarang.migrate path/to/your/scripts            # dry run: shows the changes
python -m tarang.migrate path/to/your/scripts --write    # apply
```

---

## Known limitations

* **Fitted materials make fingerprints machine-specific.** Fits differ between operating
  systems in the last digits, so a cache copied between Windows and Linux is not reused.
* **Deprecated names** (`ringResonator`, `sub_medium`, ...) work with a warning until v0.2.0
  ([CHANGELOG.md](CHANGELOG.md#upgrading-from-pre-010-code)).
* **Tidy3D 2.x only**, tested with 2.12. **MultiNest** is not available on Windows (dynesty is
  used).
* The resonator-arc builder triggers Tidy3D's warning `Override structures should be 'Box'`.
  Results are unaffected; it is kept so that cached arc results remain valid.
* The partial-ring builder's air region inside the ring reaches 0.1 µm into the BOX (the
  ring's inner side only; the gap is unaffected). It is kept so cached gap results stay
  valid, and noted in the design report and benchmark pack.
* Grating couplers: the taper between grating and waveguide is drawn in the GDS but not
  simulated yet.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No module named 'media'` (or `geometries`, `gratingCoupler`, ...) | Old layout: run `python -m tarang.migrate` ([Upgrading old scripts](#upgrading-old-scripts)). |
| `No module named 'tarang'` | Activate the environment, or run `pip install -e .` in the repository. |
| A dependency is found only in some runs | It is in your user folder, not the environment: `python -s -m pip install <package>`. |
| `No Tidy3D API key found` | Run `tidy3d configure` (or set `SIMCLOUD_APIKEY`). |
| `CostLimitExceededError` | Estimate above `max_cost` / `TARANG_MAX_COST`; nothing was charged. Raise the limit or coarsen the mesh. |
| `WaveguideModeError: TE00 mode not found` | Widen the core, raise `num_modes`, or enlarge `sim_size_y` / `sim_size_z`. |
| `No guided modes` in the log | Below cut-off; `require_guided=True` makes it an error. |
| `ValueError: sim_size_z (...) must exceed ...` | The vertical domain is smaller than box + core + cladding; enlarge `sim_size_z`. |
| `Requested wl_range ...` (no data / beyond the data) | Pick a dataset that covers the band ([Materials](#materials)). |
| A sweep returns instantly | Every point is cached; `force_rerun=True` recomputes. |
| `DeprecationWarning: ... removed in TARANG v0.2.0` | Use the new name given in the message. |

---

## Contributing, citing, licence

**Contributing:** [CONTRIBUTING.md](CONTRIBUTING.md) (setup, review rules, conventions, adding
a material or device); every pull request runs the tests on Linux and Windows
([tests/README.md](tests/README.md)).

**Citing:** cite TARANG using [CITATION.cff](CITATION.cff) (**Cite this repository** on GitHub);
a paper is in preparation. Also cite the material data you used
([SOURCES.md](tarang/media/data/SOURCES.md)), [Tidy3D](https://docs.flexcompute.com/projects/tidy3d/),
and for nested sampling [dynesty](https://dynesty.readthedocs.io/) (Speagle, *MNRAS* **493**,
3132 (2020)) or MultiNest (Feroz, Hobson & Bridges, *MNRAS* **398**, 1601 (2009)), and
[GetDist](https://getdist.readthedocs.io/) for posterior plots.

**Authors:** Esraaj Sarkar Gupta and Divyanshu Tiwari. Developed at the
[Laboratory of Optics of Quantum Materials (LOQM)](https://loqm.tech), Indian Institute of
Technology Bombay. **Licence:** MIT, © 2026 Laboratory of Optics of Quantum Materials
([LICENSE](LICENSE)).
