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
| [First design](#first-design-gui-or-one-command) | a ring with grating couplers, from the browser or one command |
| [Where to start](#where-to-start) | the path from installation to your first cloud run |
| [Pipeline](#pipeline) and [Capabilities](#capabilities) | what each stage computes, and which stages cost credits |
| [Installation](#installation) | one block of commands |
| [Quick start](#quick-start-local-no-credits) | five steps, all local and free, with the figures they produce |
| [Guided design](#guided-design-one-command) | one command from materials to GDS, at any wavelength; every stage and command |
| [Design in the browser](#design-in-the-browser-gui) | the GUI: guided steps, results, files, credits |
| [User guide](docs/USER_GUIDE.md) | a design step by step in the GUI, with what each step computes and screenshots |
| [Cloud runs and cost control](#cloud-runs-and-cost-control) | caching, cost limits, resuming |
| [Materials](#materials), [Logging](#logging), [Repository layout](#repository-layout) | reference, including the two `tarang.migrate` commands for old scripts |
| [Known limitations](#known-limitations), [Troubleshooting](#troubleshooting) | what to watch for, and fixes for common errors |
| [Contributing, citing, licence](#contributing-citing-licence) | how to contribute, what to cite, MIT licence; changes per release in [CHANGELOG.md](CHANGELOG.md) |

---

## First design (GUI or one command)

After [installing](#installation) (and `tidy3d configure` for cloud runs):

```bash
python -m tarang.gui                      # opens the design GUI in your browser
```

In the GUI: **New design** (left), then the guided view takes you through it one step at a
time. Step 1 asks the two choices (a ring for frequency combs or test waveguides for
propagation loss; grating or edge couplers); every later step shows a drawing of what it
designs, its settings, **Run** (cloud steps: **Estimate cost** first), and then what it
recommends, its results and figures. The same from a terminal:

```bash
mkdir design_runs/my_ring
cp examples/design_sin1550.toml design_runs/my_ring/design.toml
python -m tarang.design --config design_runs/my_ring/design.toml --price-only
python -m tarang.design --config design_runs/my_ring/design.toml
```

(Windows PowerShell: `mkdir design_runs\my_ring`, `copy examples\design_sin1550.toml
design_runs\my_ring\design.toml`, and backslashes in the paths.) The terminal version asks
each stage's settings as it comes (Enter accepts them). A run
can be stopped at any point and continued with the same command; nothing is paid twice.

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
| Guided design | All stages in one session: cross-section, ring, gap, grating coupler (uniform, apodised, focusing), taper, checks, GDS, whole-device spectrum, report | `python -m tarang.design` |
| Whole device | Fibre-to-fibre spectrum of grating + taper + bus + ring + taper + grating, every resonance (Q, dip, *D*<sub>int</sub>), Fabry-Perot ripple (S-matrix circuit model, free) | `tarang.design.circuit` |
| GUI | Run, stop and inspect designs in the browser | `python -m tarang.gui` |

---

## Installation

Python 3.11 or newer on Windows, Linux or macOS. TARANG is installed from this repository
in editable mode, so local edits take effect immediately. (A PyPI release, `pip install
tarang-photonics`, will follow the paper.)

**Windows** (Anaconda Prompt or PowerShell with conda):

```powershell
git clone https://github.com/loqm/tarang
cd tarang
conda env create -f environment.yml      # environment "tarang", with all extras
conda activate tarang
```

**Linux / macOS:**

```bash
git clone https://github.com/loqm/tarang
cd tarang
conda env create -f environment.yml      # environment "tarang", with all extras
conda activate tarang
```

Without conda, into an existing environment (any system): `pip install -e ".[plots,layout,gui]"`.
Existing environment, new code: `git pull`, then `pip install -e ".[plots,layout,gui]"` (or
`conda env update -f environment.yml`).

| Extra | Adds |
|---|---|
| (none) | `tidy3d` ≥ 2.12, < 3; `numpy`, `pandas`, `pydantic`, `matplotlib`, `dynesty` |
| `[plots]` | `getdist`, `corner` (posterior plots) |
| `[notebooks]` | `jupyterlab`, `ipykernel`, `ipython` |
| `[layout]` | `gdstk` (GDS export of the guided design) |
| `[gui]` | `streamlit` (the design GUI, `python -m tarang.gui`) |
| `[test]` | `pytest`, `pytest-cov`, plus `[plots]`, `[layout]` and `[gui]` |
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
materials -> cross-section -> ring | test waveguides -> ring-bus gap -> grating coupler -> checks -> taper -> apodised grating -> focusing grating -> edge coupler -> layout -> device -> report
  local          local        local     local (loss_test)   cloud             cloud        free+cloud  free+cloud  cloud (opt-in)   free (opt-in)    free+cloud     local    local     local
```

The grating stages (grating, taper, apodised, focusing) run with `run.coupling = "grating"`,
the edge-coupler stage with `run.coupling = "edge"`.

The first question is the goal (`run.goal`): **`comb_ring`**, a ring resonator for
frequency combs with its bus and couplers, or **`loss_test`**, test waveguides for
propagation loss (cutback spirals with couplers; the ring and gap stages are left out).
The second is how light gets on and off the chip (`run.coupling`): **`grating`** couplers
(fibre from above) or **`edge`** couplers at the chip facets (a lensed fibre in the chip
plane; the waveguide widens to a wide facet, which suits an air-clad film).
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
| start from an example | copy `examples/design_sin1550.toml` (or `design_sin780.toml`) to `design_runs/<name>/design.toml`, then `python -m tarang.design --config D` |
| see what a run would cost, running nothing that costs | `python -m tarang.design --config D --price-only` |
| run, spending at most 1.5 FC in this session | `python -m tarang.design --config D --yes --spend-cap 1.5` |
| test waveguides for propagation loss instead of a ring | in `D`: `run.goal = "loss_test"` (and `cross_section.height_um` / `width_um` to skip the map) |
| which couplers go on the mask | in `D`: `layout.coupler = "auto"`, `"uniform"`, `"apodised"`, `"focusing"` or `"all"` |
| edge couplers (lensed fibre at the chip facets) instead of gratings | in `D`: `run.coupling = "edge"`, the spot size `edge.fiber_mfd_um` and the chip `edge.chip_length_um` / `chip_width_um`; then `--only edge,layout,device,report` |
| extra ring variants around the designed gaps | in `D`: `layout.gap_offsets_um = [-0.05, 0.05]` |
| the whole-device spectrum and every resonance (free) | `python -m tarang.design --config D --only device,report` |
| the same in the browser | `python -m tarang.gui` |
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

**Credits.** Every run starts with one table of the two limits, the design's total and the
largest single cloud run, beside the Tidy3D balance and expiry when the account can be
read; Enter keeps them, a row number changes one, and neither can be set above what the
account holds. Two numbers decide whether a cloud stage may run: the design's limit
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
| Apodised grating (opt-in, `apodise.run`; off: the uniform grating is used and nothing is paid) | fill factors to measure, narrowest trench the full etch clears, whether to check in FDTD, what "better" means (`apodise.prefer`: efficiency, or bandwidth within 0.5 dB), a second recipe (`apodise.ramp_check`: linear fill-factor ramp, Marchetti et al. 2017) | scattering strength and phase-matched period of each fill factor (small 2D runs), a tooth list whose scattering follows the fibre's Gaussian, the predicted gain (1D model, before anything else is paid), a 2D FDTD check; the layout uses it when it beats the uniform grating |
| Focusing grating (opt-in, `focus.run`) | teeth (uniform, apodised or the better one), tip widths, largest light-cone angle, optional 3D check | curved teeth (confocal ellipses around the waveguide tip) on the straight grating's cross-section, the in-plane focusing efficiency against tip-to-grating distance and tip width (free model, landscape figure), the 3D estimate against the straight grating with its taper; optional 3D FDTD (priced); the GDS gets `*_fgc` cells with it beside the straight-grating `*_gc` cells |
| Edge coupler (`run.coupling = "edge"`) | spot MFD and its tolerance, facet widths, taper shapes (linear, parabolic, power law), accepted taper loss, optional EME and 3D facet run, facet straight, chip size, dicing lane, facet pitch | the facet's TE00 (local solve with the Si handle) against the lensed fibre's spot over facet width and fibre height (butt coupling: overlap x Fresnel transmission, free) and the recommended width; taper loss against length for each shape (converged 2D model, free; Tidy3D EME per shape, priced) and the shortest length that stays below half the accepted loss; one small 3D FDTD of the facet (priced): coupling for any fibre offset, spot size and working distance from one run, and the facet reflection (figures 12a-12c) |
| Test waveguides (`loss_test` only, free) | expected loss, cutback lengths, coupling scatter per structure, bend radii, spiral gap | bend loss against radius (local bent-mode solve) and the smallest spiral radius, the precision of the fitted loss, σ_α = σ / √Σ(L − L̄)², and the lengths that reach the wanted precision; spiral sizes |
| Layout | layer numbers, tone (structure or trench), couplers on the mask (`layout.coupler`), reference waveguide, gap variants | `waveguide.gds`, `ring_waveguide.gds`, `ring_waveguide_gc.gds` (and `*_agc` / `*_fgc` for the apodised and focusing gratings), `waveguide_gc.gds` (the same couplers without a ring: the reference that gives the coupler loss), `spiral_cutback_gc.gds` (test waveguides), `parameters.json`/`.m`/`.csv`, layout figure; `lle/` inputs for a comb (Lugiato-Lefever) solver, incl. pyLLE's dispersion-file format |
| Whole device (`device.run`, free) | whether to include the gratings' back-reflection | fibre-to-fibre transmission over the band from the stages' results (S-matrix circuit model: coupler spectra, taper, bus loss, ring all-pass with the resonance frequencies of the dispersion solve and κ²(λ) of the gap runs), every resonance's Q<sub>i</sub>, Q<sub>c</sub>, Q<sub>L</sub>, dip and *D*<sub>int</sub> for both gaps, the Fabry-Perot ripple between the couplers (figures 11a, 11b; `tables/device_*.csv`) |
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

## Design in the browser (GUI)

```bash
python -m tarang.gui                         # designs under ./design_runs
python -m tarang.gui --root D:/my/designs    # another folder (any path style)
```

The GUI opens in your browser (it runs only on your computer, at `http://localhost:8501`).
**Exit TARANG** at the bottom of the left panel closes it (or Ctrl+C in the terminal); a
run still going can go on without it or be stopped first. It shows the same design
folders the command line writes, so both can be used on one design in turn. The
[user guide](docs/USER_GUIDE.md) walks through a design step by step, with screenshots.

It opens in the dark look of the LOQM software pages (`--plain` for Streamlit's light
look). **Your measured values** (under View) takes numbers from your own chips: a ring's
intrinsic Q or a cutback loss replaces the assumed propagation loss, a measured coupler
loss is shown next to the simulated one, and measured film, fibre and material values
replace the assumed ones; the report lists them as measured, with their source.

**Guided design** (the default view): the steps of your design on the left, with their
status (done, waiting for your choice, settings changed, not run yet); the chosen step on
the right:

| Step part | What |
|---|---|
| drawing | what the step designs (ring, spirals, the three gratings, the edge coupler and lensed fibre, the cross-section of your stack, ...) |
| settings | the step's settings as a form, each with its explanation (hover); an empty field shows the value it stands for; **All settings** below holds the whole `design.toml`; values that cannot be right (e.g. a film of 700 µm, nm typed for µm) are refused with the value meant |
| run | local steps: **Run (free)**; cloud steps: **Estimate cost**, then **Run (up to ... FC)** with that spend cap; **Stop** keeps finished work; the output scrolls while it runs |
| recommended | what the step recommends and why, beside what is in use; **Use the recommended values** where they are settings (cross-section, ring radius, spiral radius and cutback lengths, the edge coupler's facet width, taper shape and length) |
| choice | the cross-section map stops with its candidates (most robust first): pick one, it is then checked at a finer mesh |
| results | the step's key numbers and figures; the layout step shows `design.gds` as it goes to the mask |
| report | the design in numbers, the report, and downloads: report as HTML (figures inside; print it to PDF) or Markdown, everything as one zip, the benchmark pack |
| Tidy3D jobs | every cloud simulation of the step (and, in the left panel, of the design) with its status, estimate, billed cost and a link to it in the Tidy3D web interface |

**All results** (the other view):

| Where | What |
|---|---|
| left | the design (any folder with a `design.toml`), **New design** (defaults or an example), FlexCredits spent and left, **Check Tidy3D balance**, the two credit limits (checked against the balance), what was spent on what |
| top | stages to run, stages to repeat, **Spend at most** for this run; **Estimate cost** (every cloud estimate, nothing that costs runs), **Run**, **Stop** (like Ctrl+C: finished work is kept); the run's output scrolls below |
| Summary | headline numbers (geometry, Q, gaps, grating, device), loss budget, status of every stage |
| Whole device | figures 11a/11b and every resonance |
| Figures, Tables | every figure by stage; every table, with a CSV download |
| Report | `report.md` with its figures |
| Files | every result file with its full path; **Open the design folder** |
| Settings | `design.toml` with every explanation; checked before it is saved |
| Log | the full output of the last run started from the GUI (`gui_log.txt`) |

Runs started from the GUI run as `python -m tarang.design --config ... --yes` in the
background, so cloud steps run only within the design's limits and the spend cap (there are
no questions to answer). Closing the browser tab does not stop them.

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
* The whole-device spectrum is a circuit model: each part comes from its own simulation
  and the parts are joined by their scattering matrices. Backscattering inside the ring
  (resonance splitting), the coupler's own phase and the taper's reflection are left out,
  and the gratings' reflection phase is unknown (the ripple's depth and period are right,
  its position is not). For the apodised and focusing gratings the reflection of the
  uniform grating is used.
* The focusing grating's efficiency outside 3D FDTD comes from an in-plane model; the
  3D check (`focus.check_3d`) is the reference.

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
| `... outside 0.02–5 µm (the setting is in µm); did you mean 0.7 µm (700 nm)?` | Lengths are in µm: type 0.7 for a 700 nm film (in the GUI's step 1, or in `design.toml`). |
| `ValueError: sim_size_z (...) must exceed ...` | The vertical domain is smaller than box + core + cladding; enlarge `sim_size_z`. |
| `Requested wl_range ...` (no data / beyond the data) | Pick a dataset that covers the band ([Materials](#materials)). |
| A sweep returns instantly | Every point is cached; `force_rerun=True` recomputes. |
| `DeprecationWarning: ... removed in TARANG v0.2.0` | Use the new name given in the message. |

---

## Contributing, citing, licence

**Contributing:** [CONTRIBUTING.md](CONTRIBUTING.md) (setup, review rules, conventions, adding
a material or device); every pull request runs the tests on Linux and Windows
([tests/README.md](tests/README.md)). Release steps (tag, wheel, later PyPI):
[docs/RELEASING.md](docs/RELEASING.md).

**Citing:** cite TARANG using [CITATION.cff](CITATION.cff) (**Cite this repository** on GitHub);
a paper is in preparation. Also cite the material data you used
([SOURCES.md](tarang/media/data/SOURCES.md)), [Tidy3D](https://docs.flexcompute.com/projects/tidy3d/),
and for nested sampling [dynesty](https://dynesty.readthedocs.io/) (Speagle, *MNRAS* **493**,
3132 (2020)) or MultiNest (Feroz, Hobson & Bridges, *MNRAS* **398**, 1601 (2009)), and
[GetDist](https://getdist.readthedocs.io/) for posterior plots.

**Built on:** [Tidy3D](https://docs.flexcompute.com/projects/tidy3d/) (FDTD, mode solver,
EME), [NumPy](https://numpy.org/), [pandas](https://pandas.pydata.org/),
[Matplotlib](https://matplotlib.org/), [dynesty](https://dynesty.readthedocs.io/),
[gdstk](https://heitzmann.github.io/gdstk/) (GDS), [Streamlit](https://streamlit.io/) (GUI).
Published methods used in the design stages are cited where they are used (report,
benchmark pack, CHANGELOG): e.g. the linear apodisation ramp of Marchetti et al.,
*Sci. Rep.* **7**, 16670 (2017). The `lle/` files follow the dispersion-file format of
[pyLLE](https://github.com/gregmoille/pyLLE) (Moille, Li, Lu, Srinivasan, *J. Res. NIST*
**124**, 124012 (2019)).

**Authors:**  Divyanshu Tiwari and Esraaj Sarkar Gupta. Developed at the
[Laboratory of Optics of Quantum Materials (LOQM)](https://loqm.tech), Indian Institute of
Technology Bombay. **Licence:** MIT, © 2026 Laboratory of Optics of Quantum Materials
([LICENSE](LICENSE)).
