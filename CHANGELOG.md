# Changelog

All notable changes to TARANG are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and version numbers follow
[Semantic Versioning](https://semver.org/) (before 1.0, a minor release may contain
breaking changes; they are announced here first).

## [0.1.0] - unreleased

First public release. Compared with the pre-release code on `main`, TARANG is now
installable, tested on Linux and Windows, safer with cloud credits, and uses
consistent names inside a single `tarang` package. Old class and parameter names keep
working until v0.2.0; old imports are updated with one command (see
[Upgrading](#upgrading-from-pre-010-code)).

### Added

- **User guide** ([docs/USER_GUIDE.md](docs/USER_GUIDE.md)): a design in the GUI step by
  step for both paths (ring with grating couplers; test waveguides with edge couplers), what
  each step computes from first principles, what to set and check, credits and Tidy3D jobs,
  the files and the benchmark pack, units and common mistakes; with screenshots.
- **GUI: Exit button** (bottom of the left panel) stops the GUI; a run still going can go on
  without it or be stopped first. **Tidy3D jobs**: every cloud simulation of a step (and of
  the design, left panel) with status, estimate, billed cost, date and a link to it in the
  Tidy3D web interface (from each run's `task.json`; **Find the missing links** asks Tidy3D
  for runs saved before task records). **Empty settings show their value** (greyed in the
  field and written under it: the default worked out from your other settings, or the
  step's recommendation once it has run). The **loss presets** name their paper and total
  loss in the drop-down, with the full source and note below it. Step buttons show the whole
  title (no hover text over them) and a legend of the status icons.
- **Plausible values** (`config.RANGES_UM`, `range_problem`, `implausible`): film, BOX, core
  height and width (and the mapped ones), lensed-fibre spot and fibre MFD must lie in a
  plausible range in µm; a value outside (nearly always nm typed for µm) is refused by the
  GUI with the value meant ("did you mean 0.7 µm (700 nm)?"), shown on every step with Run
  disabled, asked again in the terminal forms, and stops a design run before its first stage
  (`SettingsError`, a message instead of a traceback).
- **Choosing the cross-section from every candidate, after comparing them.** The map's
  choice lists every target geometry (not only the top 8). The candidates you tick (the top
  four by default) are compared side by side before the choice (`cross_section.compare`,
  local, free): each at the check mesh with its tolerance box, TE/TM split and second-mode
  margin, D and loss in a bend (the ring radius for the FSR, or the spiral radius), TE00/TM00
  mode profiles, D over the band and D_int (`figures/2_candidates/`,
  `tables/candidates_*.csv`). The chosen cross-section's finalist report now includes its
  bend (its bend panels were empty).
- **Dispersion-engineering figures and tables with the others.** The map, candidates,
  finalists, mesh convergence and the ring's dispersion profile are written to
  `figures/2_dispersion_map/`, `figures/2_candidates/`, `figures/2_finalists/`,
  `figures/2_mesh_convergence/`, `figures/3_dispersion_profile/`; their tables to
  `tables/`; the raw solver data stays in `results/` (what makes a repeated solve free).
  `--plots-dir` on `dispersion_mapping.maps`, `.broadband` and `.convergence` (the default is
  unchanged). Older designs keep `results/plots_*`; the GUI, report and benchmark pack read
  both.
- **GUI: the dispersion-engineering suite** is shown where it belongs: the cross-section
  step (also while it waits for your choice) shows the design map, the decision dashboard,
  the nine maps (D, TE/TM splitting, second-mode margin, sensitivity to width and to
  thickness, robust window, group index, γ, confinement), D and the guided modes against
  width and height, the mesh convergence, the finalists and the table of target geometries;
  the ring step its dispersion profile. They were saved under `results/` but not shown. The
  All results Figures and Tables tabs and the zip download include them (not the raw solver
  data).
- **Benchmark pack:** `comparison.csv` (every number of `results.csv` with empty columns for
  another solver's value, its settings, the difference and notes, to fill in and send back),
  and the TARANG, Tidy3D, NumPy and Python versions in its README.
- **Guided design in the GUI** (the default view of `python -m tarang.gui`): the steps of the
  design one at a time. Step 1 asks the two choices (ring for combs or test waveguides;
  grating or edge couplers; which gratings, or the lensed fibre's spot) with drawings of
  each option; every later step shows a drawing of what it designs, its settings as a form,
  Run (cloud steps priced first, then run with that spend cap), and once run what it
  recommends and why (one click writes the recommended values where they are settings), its
  key numbers and figures. The cross-section map stops for your choice of candidate. The
  layout step shows `design.gds` as it goes to the mask; the last step offers the report as
  one HTML page with its figures (print it to PDF), everything as a zip and the benchmark
  pack. `tarang.gui.wizard` holds the steps and their logic without Streamlit;
  `tarang.gui.illustrations` the drawings. The earlier view stays as "All results".
- **Edge coupling** (`run.coupling = "edge"`, asked with the goal; `[edge]` settings): a wide
  facet for a lensed fibre, the right choice for an air-clad film. New stage `edge` (after
  the focusing grating): the facet's TE00 (local solve with the BOX and the Si handle)
  against the fibre spot (MFD ± tolerance) over facet width and fibre height, as butt
  coupling (mode overlap x Fresnel transmission, free; figure 12a); taper loss against
  length for linear, parabolic (w² linear) and power-law shapes (converged 2D model, free;
  one Tidy3D EME per shape, priced; figure 12b), the length being the shortest from which
  the loss stays below half the accepted loss; one small 3D FDTD of the facet (priced): the
  waveguide runs through the absorbing boundary, the field 0.5 µm past the facet gives the
  coupling for any fibre offset (y, z), spot size and working distance by reciprocity, and
  the backward mode amplitude the facet reflection (figure 12c). On the mask the waveguides
  run facet to facet across the chip (`edge.chip_length_um`), with the taper, a straight at
  the facet width and an overhang into the dicing lane; the chip outline and the lanes are
  drawn on their own layers (not etched), devices at least `edge.pitch_um` apart
  (`layout/chip.json`); cutback spirals have the requested facet-to-facet length. The
  whole-device model treats the chip as a Fabry-Perot cavity between the two facets; the
  loss budget, the report (section 5), the benchmark pack and the GUI summary include it.
- Taper shapes (`taper.profile_width`, used by the edge stage): linear, parabolic and power
  law, in the 2D model and the EME; the 2D model can average the permittivity of the grid
  cell at the wall (`smooth`), so the width changes continuously (without it, long tapers
  show a spurious loss from width steps of two grid cells).
- **Design goal** (`run.goal`, asked first): `comb_ring` (as before) or `loss_test`, test
  waveguides for propagation loss. Stages of the other goal are left out (the ring and gap
  for `loss_test`; their settings do not affect the other stages' saved results). With
  `loss_test` and a given cross-section the map is skipped; otherwise the map ranks
  single-mode cross-sections.
- **Test-waveguide stage** (`waveguides`, local, free): bend loss against radius (local
  bent-mode solve) and the smallest spiral radius (bend loss below a tenth of the expected
  loss); the precision of the cutback fit, σ_α = σ / √Σ(L − L̄)² with σ the coupling scatter
  per structure, and the lengths that reach `waveguides.target_precision` (for 0.4 dB/cm and
  0.3 dB scatter: 0, 3, 5.5 and 10.5 cm for ±10 %); figure 3w. The whole-device stage then
  gives the expected cutback line and every structure's spectrum with its Fabry-Perot
  ripple (figure 11a, `tables/device_cutback.csv`).
- **Cutback spirals in the GDS**: double Archimedean spirals (two interleaved arms, a centre
  S-bend at the smallest radius, leads to the couplers), each of exactly the requested
  length (`layout/spirals.csv`), with the couplers at both ends.
- **Reference waveguide beside every ring**: `waveguide_gc.gds` (the same couplers without a
  ring) is now drawn for ring designs too (`layout.reference`); measuring it gives the
  coupler loss. Before, it was drawn only when no gap had been designed.
- **Couplers on the mask** (`layout.coupler`): `auto` (as before), `uniform`, `apodised`,
  `focusing`, or `all` (uniform `*_gc`, apodised `*_agc` and focusing `*_fgc` side by side);
  the loss budget and the whole-device model follow the main coupler.
- **Gap variants** (`layout.gap_offsets_um`): extra rings with the gap moved by these amounts.
- **Back-reflection of the kept apodised grating** (`apodise.reflection_check`, one small 2D
  reverse run, priced first); the whole-device ripple uses the reflection of the grating
  actually drawn.
- **Recommendations** in each stage's results (`recommend`: the options, the recommended one
  and why), for the GUI. `cross_section.stop_after_map` stops after the map so a candidate
  can be chosen, then the stage is run again.

- **Design GUI** (`python -m tarang.gui`, extra `[gui]`: Streamlit). In the browser, on your
  own computer: pick or create a design (defaults or an example file), see the FlexCredits
  spent and left and the Tidy3D balance, change the two credit limits (checked like the
  run's own limit table), choose stages, **Estimate cost** (runs nothing that costs),
  **Run** (in the background, `--yes` with a spend cap) and **Stop** (like Ctrl+C, work done
  is kept), with the run's output live. Tabs: summary, whole device, figures, tables (with
  download), the report with its figures, every file with its path, `design.toml` (checked
  before saving) and the full log. `tarang.gui.data` holds the data layer without Streamlit.
- **Whole-device stage** (`--only device`, free, on by default; `[device]` settings). A
  circuit model joins each part's 2 x 2 scattering matrix (Redheffer star product): coupler
  spectrum (uniform or apodised 2D grating x lateral overlap x taper, or the focusing
  grating's 3D FDTD / model x tip taper), the bus with its loss, the ring as an all-pass
  filter whose resonances are the dispersion solve's mode frequencies (so *D*<sub>int</sub> is
  included) with κ²(λ) from the gap/checks FDTD, and the gratings' back-reflection from the
  checks stage (Fabry-Perot ripple). Gives the fibre-to-fibre spectrum over the band, a
  close-up at the pump and every resonance's Q<sub>i</sub>, Q<sub>c</sub>, Q<sub>L</sub>, K,
  linewidth and dip for the critical and over-coupled gaps (figures 11a, 11b;
  `tables/device_resonances.csv`, `device_spectrum.csv`, `device_envelope.csv`; report
  section 7e; benchmark pack).
- **Credit limits at the start of every run**: one table with the design's total limit and
  the largest single cloud run, beside the Tidy3D balance and expiry; neither can be set
  above what the account holds (the per-stage limit rows are gone). `--price-only` goes
  through the stages and lists every cloud estimate with a total, running nothing that
  costs (cached, free steps still run); `--spend-cap FC` limits what one session may spend.
- **Apodisation preference and a second recipe.** `apodise.prefer = "efficiency"` (default)
  keeps the apodised grating when it is more efficient at the design wavelength;
  `"bandwidth"` when its 1 dB band is wider and it is at most 0.5 dB below the uniform
  grating there. `apodise.ramp_check = true` also tries the linear fill-factor ramp of
  Marchetti et al. (*Sci. Rep.* **7**, 16670 (2017)) from the weakest usable tooth, five ramp
  rates x three fibre positions in 2D FDTD (priced; figure 9d, `tables/apodised_ramp.csv`),
  and keeps whichever design the preference favours. The kept design's spectrum is saved
  (`tables/apodised_spectrum.csv`; rerunning the stage reads it from the cache, free).
- `focus.mesh_steps` (default 12) for the focusing grating's 3D check; its spectrum is
  saved (`tables/focusing_spectrum_3d.csv`).
- **Example design files** `examples/design_sin1550.toml` and `examples/design_sin780.toml`
  (700 nm Si3N4 / mapped film, 4 µm BOX, air top, full-etch gratings, 200 nm narrowest
  trench, apodised and focusing gratings on).

- **Taper stage** (`--only taper`). The grating width that best matches the fibre: the
  lateral overlap η_y of the fibre's Gaussian with the grating's fundamental lateral mode
  (effective-index method; free), so the 3D efficiency is about η_2D·η_y (for the 1550 nm
  design the optimum is about 14.5 µm, η_y 99 % against 97 % at 12 µm). The taper length:
  a free 2D model of the linear taper (effective-index method, 600 graded cells joined by
  mode overlaps, converged) and one Tidy3D EME run (even modes, 120 graded cells x 20
  modes, EMELengthSweep) give the fundamental-mode transmission against length; the
  shortest taper below `taper.max_loss_db` is chosen from the 3D EME when its loss does not
  rise with length (converged), otherwise from the 2D model (about 270 µm for 14.5 → 1 µm
  at 0.05 dB). The 2D model also predicts the staircase error of the 3D cell count before
  the run is priced. The layout and the fibre-to-fibre loss budget use both
  (`taper.use_in_layout`). (In the first version 30 equal cells and 8 modes made the 3D loss
  jump between 0.2 and 1.1 dB with length; the 2D model reproduces that with the same cells.)
- **Apodised grating stage** (`--only apodise`; opt-in with `apodise.run = true`, so a first
  design run uses the uniform grating only and pays nothing here). The scattering strength α and the
  phase-matched period of each fill factor are measured on uniform gratings (small 2D
  runs: TE0 launched from the waveguide, field 1.5 µm above the grating; the decay of |E|²
  gives α, its phase slope the Bloch index). The tooth list follows
  α(x) = A·G²/(2(1 − A∫G²)) so the emitted light has the fibre's Gaussian shape; a 1D
  emission model (uniform grating: the textbook 80 % ceiling) predicts the gain before the
  2D FDTD check is priced. α and n_B are fitted to the complex field above the grating as
  a forward wave decaying from the start plus a backward wave decaying from the end (the
  light reflected at the grating's end, which also scatters up and makes |E|² fringe), each
  with its near-field harmonics; on the 1550 nm design this explains the field to within
  2-13 % for fill factors 0.65-0.908 (a single exponential fit had read 0.094/µm at FF 0.65
  and zero above FF 0.85). Only one monotonic branch of fill factors is used (α falling and
  the Bloch index rising with fill factor, from the strongest scatterer up); points are
  rejected when the decay is too weak to measure, the fits are poor, or the second
  diffraction order also radiates up (its field above the grating spoils the phase). The
  2D check runs in two rounds: the period scale that puts the peak on the design wavelength
  (5 runs), then that scale at three fibre positions (3 runs). Every tooth is etched through
  the film (one full etch, as for the rest of the chip); `apodise.min_trench_um` sets the
  narrowest trench the etch clears (narrow trenches etch more slowly in ICP-RIE), which
  limits how weakly a tooth can scatter. Fill factors with narrower trenches are neither run
  nor used, and the report lists the trench widths of the design. When the apodised grating beats the uniform one, the layout uses
  it (`layout/grating_teeth.csv`, and the teeth as a matrix in `parameters.m`).
  `GratingConfig(teeth=...)` builds a grating from an explicit tooth list.
- **Focusing grating stage** (`--only focus`; opt-in with `focus.run = true`). The teeth of
  the straight grating (uniform, or apodised when it was checked and is better) are curved
  into confocal ellipses n_B r − n_top sin θ · r cos φ = Φ around the waveguide tip, so the
  light scattered into the film converges onto the tip and a few-µm slab replaces the long
  taper (for the 1550 nm design about 45 µm instead of about 320 µm from the waveguide to
  the grating's end). The in-plane focusing is computed for free from the angular spectrum
  of the fibre spot on the arc against the tip's mode (effective-index method) over the
  tip-to-grating distance and the tip width (a wider tip, with a short taper from the 2D
  taper model, allows a narrower light cone; `focus.max_half_angle_deg` keeps the cone
  where the model holds). The layout draws the tip taper, the sector and the curved teeth
  in extra `*_fgc` cells beside the straight-grating `*_gc` cells (both in `design.gds`, so
  one mask can carry both; `tables/focusing_teeth.csv`); with `focus.use_in_layout` the
  focusing grating is the main coupler and the loss budget uses the focusing efficiency
  instead of the lateral overlap and the long taper. A 3D FDTD of the whole focusing
  coupler is optional (`focus.check_3d`, priced first).
- **Efficiency landscapes** (contour maps over two design parameters, from runs already
  made; every simulated point drawn, the chosen design as a red X): uniform grating, period ×
  fill factor (5g; older runs get it from their sweep table); apodised grating, fibre
  position × period scale (9c); focusing grating, tip-to-grating distance × tip width (10a).
- **Reflections.** The arc-end reflection of the partial ring and the bus reflection are
  read from the saved gap runs (free; checks stage, report). The reverse grating run is a
  closed box: a plane behind the source adds the light scattered backwards.
- **A picture of every cloud run's geometry** in `figures/sims/`, named like its cloud
  folder; `--runs` draws them for earlier runs too. Batch links name their case
  (e.g. `angle 11° (gc2D_...)`), so runs with the same name can be told apart.
- `inner_box_recess_um` for the full- and half-ring builders as well (0.1 keeps the old
  geometry bit for bit; 0 gives a flat BOX).
- EME simulations go through the same verified cache as FDTD runs, and `--runs` draws their
  pictures too.

- **Every cloud run has a web link and survives Ctrl+C.** Each run writes `task.json`
  (Tidy3D task id, a link that opens the simulation in the Tidy3D web interface, status,
  estimated and billed cost, date) next to its results; links are printed as runs start and
  listed in `tables/cloud_runs.csv` and the report. A run interrupted while it was running
  on Tidy3D is picked up again on the next start instead of being submitted (and paid)
  twice; a run that failed on Tidy3D is submitted again. Batch runs do the same per task.
  `python -m tarang.design --runs` lists every run and finds the links of runs made before
  these records.
- **Credits commands.** `--credits` (limit, spent, what it was spent on by stage and date,
  account balance and expiry), `--budget FC` (this design's limit), `--reset-spent` (start
  the count at 0; the history is kept). Every cost is now recorded with its stage and date.
- **One table per stage.** Each stage shows its settings in one table: Enter accepts, a row
  number changes that value, `?` explains, `q` stops. Saved settings are listed, not asked
  again; `--review` asks anyway.
- **Checks stage** (`--only checks`; free parts are local or from saved results, cloud
  parts priced first):
  gap tolerance (coupling ratio K, loaded Q and resonance depth with the waveguides ± the
  width tolerance); an independent check of the gap by coupled-mode theory (even/odd
  supermodes, κ = πΔn/λ, θ = κ√(2πR/γ): 466 nm against 470 nm from 3D FDTD for the 1550 nm
  design); FSR over the tolerance box; guided mode families of the bent ring waveguide and
  how often they cross TE00; fibre-to-fibre loss budget; optional coupling across the band
  (both gaps, 3D FDTD recorded at `gap.band_points` wavelengths); grating tolerances (fibre
  angle ± `checks.angle_tolerance_deg`, film thickness, tooth width ±
  `checks.litho_bias_um`, 1.5× finer mesh) and a reverse run giving the power up, into the
  substrate and past the grating, the back-reflection and the directionality; optional
  gap mesh check.
- **Figures for publication.** Every figure is written as 300 dpi PNG and PDF (fonts
  embedded; `run.figure_formats` adds SVG, `run.figure_dpi` sets the resolution) in a
  colour-blind-safe palette. New: TE00/TM00 mode profiles with |E| and each component,
  straight against bent mode, gap coupling on linear and log scales with the fit, coupler
  supermodes, a top-view field map of the gap run nearest the critical gap, coupling against
  wavelength, grating checks and the loss budget. A figure held open by another program is
  saved beside it as `<name>_new.png` instead of failing.
- **Inputs for a comb (LLE) solver.** The layout stage writes `lle/lle_inputs.json` (pump,
  FSR, D2, D3, Q_i, Q_c, Q_L, loss rates, threshold, γ = 2πn₂/(λA_eff), with units and
  definitions), `lle/pyLLE_dispersion.csv` (mode number, resonance frequency: pyLLE's
  dispersion-file format) and `lle/integrated_dispersion.csv`.
- `gap.ring_inner_box_recess_um`: how far the air inside the ring reaches into the BOX in
  the gap simulation (0 for new designs: a flat BOX, as fabricated; earlier designs keep
  0.1 µm so their results stay valid). `gap.band_points`: wavelengths recorded at the ring
  ports.
- The report and `benchmark_pack/` gain the checks, the coupled-mode check, the LLE inputs,
  the cloud runs with links, the spending by stage and the software versions; the pack
  includes the PDF figures and `lle/`.

- **Design runs show and record everything.** Before each stage the tool draws what it is
  about to simulate (stack and TE00 mode, partial ring and bus, grating with the fibre
  beam), before any cost question; every cloud run prints a progress line; each stage saves
  its figures (gap coupling against gap; grating period × fill-factor map, fill-factor /
  period / fibre-position sensitivity, spectrum, field; layout views). The run starts with
  the table of assumed values (losses, *n*<sub>2</sub>, fibre MFD, tolerances) and their
  published sources (`ring.loss_preset`: `typical_ebl`, `annealed_low_loss`,
  `state_of_the_art`); your own values replace them. The report shows every figure and a
  threshold estimate, and `benchmark_pack/` (plus a zip) holds a step-by-step README for
  repeating the design in another solver, `assumptions.csv`, `results.csv`, the n, k data
  used, figures, tables, simulation files and the GDS.
- Grating couplers: `GratingConfig.fiber_tilt` (`"towards"` or `"away"` the waveguide) and
  `bandwidth_um` (half-width of the recorded spectrum, default 0.1 µm);
  `GratingCoupler.get_spectrum()` / `get_spectrum_summary()` (peak, value at the design
  wavelength, 1 dB / 3 dB bandwidths); `phase_matched_period()` and `spectrum_summary()`.
- Design tool grating stage in three steps: a period × fill-factor map around each fill
  factor's phase-matched period (all runs submitted together), the period moved so the
  efficiency peak lands on the design wavelength with fill factor and period varied around
  it, then the fibre position. Designs below `grating.min_feature_um` are not simulated;
  a best design at the edge of a swept range is reported.
- `ring_gap_sweep(reference=...)`: `"ring_port"` reads the power entering the ring arc,
  `"bus_only"` divides the through power by that of the bus alone (one extra run);
  `bus_only_device()`.
- `tarang.utilities.cloud_run.progress_listener()` (events for every cloud run) and
  `account_status()` (FlexCredit balance and expiry); the design tool stops before a cloud
  stage when the credits have expired or are short.

- **Guided design** (`python -m tarang.design`): materials → cross-section → ring →
  ring–bus gap → grating coupler → layout → report in one session, at any wavelength the
  material data cover. Each stage asks only what it needs, shows its results, and prices
  cloud stages before asking to run them; answers are saved to `design.toml` so a design
  can be repeated or resumed with `--config ... --yes`. Exports GDS files of the
  waveguide, ring + bus and ring + bus + grating couplers (`gdstk`, new `layout` extra),
  the dimensions as JSON, MATLAB and CSV, and solver-neutral tables of materials and
  key values for checking the design elsewhere.
- `get_medium` accepts the path of your own CSV (`wl`, `n`, optional `k`), e.g. ellipsometry
  of your film, without editing `media.json`; it is fitted and cached like the built-in
  materials.
- Dispersion mapping: `broadband --pump` (pump wavelength for *D*<sub>int</sub>), and
  `--box` / `--cladding-height` for `broadband` and `convergence`, as `sweep` already had.
- **Dispersion mapping** (`tarang.waveguide.dispersion_mapping`): command-line tools for robust
  height × width dispersion sweeps, maps, mesh-convergence studies, broadband *D*(λ) and
  integrated dispersion, and finalist reports against a baseline. Local mode solver
  only, no FlexCredits; results are appended line by line, so interrupted sweeps resume.
- **Verified result cache** (`tarang/utilities/cloud_run.py`): every cloud result is stored
  under a SHA-256 fingerprint of the complete simulation (geometry, materials, mesh,
  sources, monitors), with `simulation.json` saved next to `raw_simulation_data.hdf5`.
  Each cache hit is checked against the stored simulation.
- **Cost guard**: each job is priced before it starts and stopped if it exceeds
  `max_cost` (per device or sweep) or `$TARANG_MAX_COST`; `CostLimitExceededError` is
  raised and nothing is charged.
- **Nested sampling on every OS**: dynesty backend next to MultiNest
  (`tarang/utilities/nested_backends.py`); `backend="auto"` picks MultiNest when its library
  loads and dynesty otherwise. Both write MultiNest-format chains. Seeded dynesty runs
  resume by replaying cached evaluations.
- **Installable package**: `pyproject.toml` (distribution `tarang-photonics`) with extras
  `[plots]`, `[notebooks]`, `[multinest]`, `[test]`; cross-platform `environment.yml`.
- **Material `si_franta`**: float-zone crystalline silicon (Franta 2017, 0.03–310 µm),
  for silicon at 1550 nm, where the `si` dataset has no data.
- **Documentation**: a rewritten README (contents table, where-to-start diagram, quick
  start with the figures it produces, cost control, troubleshooting), this changelog, `CONTRIBUTING.md`, `CITATION.cff`,
  `tarang/media/data/SOURCES.md` (reference, sample and range of every material dataset), and
  material fit-versus-data figures (`docs/make_material_figures.py`).
- **Example notebooks** `dispersion_mapping_example.ipynb` (choosing core height and
  width: design map, mesh convergence, tolerance report, integrated dispersion; local) and
  `cloud_basics_example.ipynb` (fingerprints, cache, cost limits). The other notebooks
  open with a table of the functions they show, set a cost cap for their cloud cells, and
  now also cover `get_dispersion_vs_wavelength`, the guided-mode check, the FDTD radius
  sweep and the critical-coupling gap sweep.
- **Tests and CI**: 200+ tests (fake cloud, no FlexCredits) run on every pull request on
  Ubuntu and Windows with Python 3.11 and 3.12, plus a MultiNest job, a job that builds
  and installs the package, a conda-environment check, and an optional real-cloud smoke
  test started by hand.
- `tarang.set_tidy3d_log_level()`; `RidgeWaveguide.get_modes(require_guided=True)`;
  `MonitorDataError` explains whether a monitor is undefined or has no data.

### Changed

- **Figures.** 3b bent mode with a cut and the bent − straight difference (the 4 nm shift
  is visible); 4d field map zoom kept inside the data, with the standing-wave stripes
  explained; 5b grating map: the best point outlined, the final design starred, readable
  labels; 5d: the 1 dB and 3 dB lines labelled; 7a: a smooth fit of κ²(λ), the K = 1
  wavelengths marked, the resonance depth drawn from the fit (the sampled points made the
  peaks look finite); 7b: distinct line styles. Runs made before keep their results and get
  the new figures, free, when their report is next written.
- The report ends with "Known limits"; the loss budget includes the taper and the lateral
  overlap once the taper stage has run.

- **Grating couplers are fed "towards" the waveguide by default.** Before, any `theta_deg`
  was made negative, which always gave a beam travelling away from the waveguide
  (period λ / (n_eff + sin θ)); the usual two-fibre setup (fibres on two holders forming a
  V over mirrored couplers) needs λ / (n_eff − sin θ). `theta_deg` is now the angle from
  vertical (its sign is ignored) and `fiber_tilt="away"` gives the old orientation.
  `source_x` is now the beam centre on the grating surface (it was the centre at the
  source plane, 3 µm higher), and the mode monitor records ±100 nm instead of ±10 nm.
  Grating simulations therefore change, and earlier cached grating results are not reused.
  The example notebooks and `nested_sampling_example.py` use the "towards" equivalent of
  their earlier design (period 1.052 µm instead of 0.820 µm).
- Design tool: the ring–bus gap stage measures the coupled power where light enters the
  ring (`gap.reference = "ring_port"`), grating defaults are 30 teeth, fill factors
  0.5–0.9 (full etch) and five fibre positions, and cross-section candidates with the same
  robustness are ordered by the TE00/TM00 index split instead of a second-mode margin that
  measured a radiation mode.

- Dispersion-mapping figures and the finalist report name the core material in use instead
  of assuming Si₃N₄ (Luke) data.
- **One package.** All code moved into the `tarang` package (`tarang.media`,
  `tarang.geometries`, `tarang.waveguide`, `tarang.ring_resonator`,
  `tarang.grating_coupler`, `tarang.edge_coupler`, `tarang.utilities`), so installing
  TARANG adds a single name to an environment instead of eight generic ones. The example
  notebooks and `nested_sampling_example.py` moved to `examples/`. The dispersion-mapping
  tools are now `python -m tarang.waveguide.dispersion_mapping.<tool>`.
- **Names** follow one convention (PEP 8): packages `grating_coupler`, `edge_coupler`,
  `ring_resonator`; classes in CapWords; one parameter name per physical layer
  (`box_*`, `handle_*`, `top_cladding_*`) and `wavelength_um` everywhere. Old class and
  parameter names are deprecated aliases (table below).
- **`sio2` now means bulk fused silica** (Malitson 1965, n = 1.444 at 1550 nm). It
  previously pointed to a silica-microsphere dataset (n = 1.420), which remains available
  as `sio2_nyakuchena`. Simulations with `sio2` therefore change (and run again).
- Mode sources and monitors target the core material's index instead of a fixed 2.0, so
  the fundamental mode is found for any core material.
- `RidgeWaveguide.get_dispersion()` identifies TE00 at every wavelength (previously the
  mode index from the centre wavelength was reused, which corrupted *D* where TE and TM
  cross) and derives *D* and *n*<sub>g</sub> from a polynomial fit of *n*<sub>eff</sub>(λ).
  New keys: `group_index`, `guided_te_count`, `guided_tm_count`, `n_cutoff`.
- `is_singlemode` now means exactly one guided TE mode, counting guided modes per
  polarization against the higher of the box and top-cladding indices (previously mode
  index 2 was compared with the substrate index only).
- The three waveguide sweeps default to 50 mesh steps per wavelength (20 gave *D* about
  25 % low) and accept `top_cladding_medium` / `top_cladding_height`. Device geometry
  defaults, and therefore FDTD costs, are unchanged.

### Fixed

- GUI: the Materials step showed the figures of the focusing grating, the whole device and
  the edge coupler too (its figure filter `^1` also matched `10a_`, `11a_`, `12a_`).

- Whole-device spectrum: beyond the first and last computed resonance (half an FSR at each
  end of the band) the ring's phase was held at the end resonance, so the trace sat on a dip
  there (e.g. −21.6 dB at 1450 nm); it now continues with the local FSR.
- Edge coupler: the headline coupling is the spectrum at the design wavelength; the
  alignment and spot-size maps are at the recorded wavelength nearest to it (now said so;
  before, that wavelength's value was reported as the design wavelength's).

- Whole-device model and loss budget with the focusing grating's 3D FDTD: the 3D run already
  contains the tip taper, which was counted a second time (0.02 dB per coupler); the loss
  budget now uses the 3D result (it used the 2D grating with the in-plane model, so the
  budget and the device model differed by about 0.5 dB).
- Coupler spectra are interpolated in log(η) between the simulated wavelengths (a linear
  interpolation drew V-shaped notches in dB far from the peak); the device figures show the
  single coupler's simulated points.
- Layout figure with gap variants: it showed the first variant in the file (e.g. gap 381 nm)
  while labelling the critical gap; it now shows the critical ring.
- The report of a test-waveguide design no longer lists the ring's checks (gap tolerance,
  FSR, mode families, coupling across the band) or the ring's known limits; the summary
  table shows the coupler actually on the mask, the fibre-to-fibre transmission and the
  cutback precision.
- Bend losses below the solver's resolution are printed as "< 0.001" (the csv keeps the
  numbers).

- **Bend loss in tight bends** (ring stage, test waveguides): the solver's highest-index
  TE-like solution in a tight bend is a whispering-gallery mode of the substrate, not the
  waveguide mode, and was reported as the bend loss (about 9000 dB/cm at R = 15 µm for a
  700 × 1000 nm guide). The bent TE00 is now picked by its overlap with the straight one,
  in a domain that reaches past the radiation caustic; losses below 0.001 dB/cm are
  reported as below the solver's resolution. Rings of the existing 1550 nm design
  (R = 107.5 µm) were not affected.

- Stopping a design run (q, Ctrl+C, the GUI's Stop) while the local mode solver maps the
  cross-section left the solver's worker processes running until every queued point was
  done; they now stop with it (finished points are kept and skipped next time).
- The benchmark pack could not describe the taper's EME simulation (`'EMESimulation' object
  has no attribute 'run_time'`); it now lists its cells and the length sweep.

- Runs made with another Tidy3D account are reported as such (their links cannot be found
  with the current API key) instead of suggesting `--runs` would find them.
- `--budget`, `--credits` and the other credit commands given only `--out` now use that
  design's saved settings (before, `--budget` with `--out` alone wrote default settings
  into its `design.toml`).
- Reading the Tidy3D account (credits, expiry) gives up after 15 s when offline instead of
  waiting for Tidy3D's retries.
- The coupled-mode check solves the even and odd supermodes separately with the mirror
  symmetry imposed and the fine mesh covering both cores. Without that, the solver could
  return modes localised in one core each (on Windows already at 0.4-0.6 µm gaps), and
  the check gave a critical gap of 5.6 µm.
- Settings added in this release do not repeat finished stages: a design whose gap ran
  before keeps the values its results were computed with (the session says so), and its
  stage fingerprints are unchanged.

- **Ring–bus critical gap too wide.** The gap sweep took the coupled power as 1 − T with T
  relative to the source, and the simulated bus loses about 0.2 % of the power whatever
  the gap (normalisation and boundaries), which near critical coupling is a third of the
  coupled power. For the 1550 nm design (R = 107.5 µm, 0.4 dB/cm) this gave 532 nm instead
  of 470 nm (over-coupled 464 nm instead of 431 nm), and the fit used only four gaps. The
  power entering the ring and the bus-only reference agree to about 1 %.
- Design tool: settings files that start with a byte-order mark (Windows PowerShell 5)
  are read; the grating stage no longer prints a phase-matched period for the wrong
  orientation.
- Design tool report: building the benchmark pack stopped with "The truth value of an
  array ... is ambiguous" when describing the grating simulation (Tidy3D stores monitor
  frequencies as an array). The grating stage now varies the period again (step B2) when
  step B picks a fill factor other than its centre one, so the final fill factor's period
  is optimised too; the sensitivity figure names what each panel's runs held fixed; a
  gap run loaded twice (data, then field map) is counted once in the progress lines.
- Design tool, second 1550 nm run: the benchmark pack stopped again ("cannot access local
  variable 'np'") on the ring simulation, because numpy was imported only inside the
  Gaussian-beam branch; a simulation that cannot be described is now noted in the README
  instead of stopping the report. **The grating's efficiency peak is put at the design
  wavelength** (`grating.centre_peak`, default true): after steps B/B2, step D moves the
  period (secant steps on the peak wavelength, usually one or two runs) until the peak
  is within 1 nm of it; before, the design with the highest efficiency at the wavelength
  was kept, which can sit on the flank of a peak 8 nm away. `spectrum_summary` finds the
  peak between samples (parabola) and gives the 1 dB / 3 dB band edges; the spectrum
  figure shades the 1 dB band. The layout figure draws every shape with an outline, so
  the 1 µm bus is visible beside the 1 mm device, labels the parts, and adds a zoom of
  the waveguide-to-taper joint.

- **Material fits that swing between sparse data points.** A pole fit can pass through
  every tabulated point and still oscillate between them: `si3n4` fitted over 0.68–0.88 µm
  (three data points) gave n = 3.8 and 1.5 between the points instead of 2.03. Every fit,
  new or cached, is now checked against the data between the points; one that strays more
  than 0.002 (or 5 × the requested tolerance) is refitted with fewer poles and, if needed,
  over a wider stretch of the data, with a warning. Fits that pass, including all the
  1550 nm fits of `si3n4`, `sio2` and `si_franta`, are unchanged, and so are fingerprints.
- **Dispersion mapping away from 1550 nm.** The tools always fitted the materials over
  1.1–2.4 µm and computed *D*<sub>int</sub> around a 1550 nm pump, so a 780 nm map used
  extrapolated indices. Materials are now fitted over a range that covers the solved band
  and the pump lies in the band. Telecom results are unchanged (same fit, same fingerprints);
  at 780 nm the change for `si3n4` was 0.03 ps/(nm·km), larger for narrower datasets.
- **Full-etch grating couplers** (`etch_depth = core_height`) no longer contain a
  zero-thickness floor structure; an etch deeper than the film is refused.
- **Result cache missed results run in the session that fitted the materials.** A fresh
  fit returned the fitter's own object, whose index differed from the saved copy in the
  last digit. The mode solver's `target_neff` comes from that index, so a simulation built
  in the fitting session had a different fingerprint from the same simulation in any later
  session, and a cloud result run then was not found again. `get_medium` now returns the
  saved copy in both cases. Results run in later sessions are unaffected.
- Example notebooks: the waveguide and grating-coupler notebooks fitted their materials
  over 0.6–1.2 µm but simulated at 1.55 µm. (For these smooth materials the extrapolated
  index happened to agree to four decimals, so earlier results stand.) They now fit over
  1.35–1.75 µm; the grating coupler uses `si_franta` for silicon; the 1550 nm waveguide
  uses a converged mesh; the nested-sampling cells read the folder
  `nested_sampling_example.py` writes and do not fail before it has run. The notebooks are
  stored without outputs, which also removed about 10 MB of stale results and local paths.

- Grating-coupler results were cached by geometry only, so changing a material reused
  results computed with the old one. All devices are now keyed on the full simulation.
- The nested sampler uploaded every design twice (once only to price it).
- Downloads are written to a temporary file and renamed when complete, so an interrupted
  download is never mistaken for a result.
- `get_medium` warns when `wl_range` extends beyond the data, instead of extrapolating
  silently.
- Stacked refractiveindex.info CSVs are read in memory; source data files are never
  modified.
- Tidy3D API key: Tidy3D reads `SIMCLOUD_APIKEY` or the `tidy3d configure` file, not
  `TIDY3D_API_KEY` as earlier documentation said. TARANG passes the old variable on with a
  warning.
- Command-line tools no longer crash on Windows consoles that cannot print λ or α.
- `import pymultinest` without its native library no longer ends the Python process.
- Notebook calls to settings removed in Tidy3D 2.12.

### Deprecated

The old class, parameter and attribute names in the table below work in v0.1.0 with a
`DeprecationWarning` naming the replacement, and will be **removed in v0.2.0**.

### Removed

- `requirements-test.txt` (replaced by the `[test]` extra).
- The root `media.json`: an outdated copy of `tarang/media/media.json` that nothing read (it still
  pointed `sio2` to the microsphere data).
- The top-level packages (`media`, `geometries`, `waveguide`, `ring_resonator`,
  `grating_coupler`, `edge_coupler`, `utilities`) and the pre-v0.1.0 package names
  (`gratingCoupler`, `edgeCoupler`, `ringResonator`). They are not aliased:
  `python -m tarang.migrate` rewrites old imports (below).

### Upgrading from pre-0.1.0 code

**Imports.** Run the migration tool on your scripts and notebooks; it rewrites only the
import statements, so the rest of your code keeps working unchanged:

```bash
python -m tarang.migrate path/to/your/scripts            # dry run: shows what would change
python -m tarang.migrate path/to/your/scripts --write    # apply
```

| Old import | New import |
|---|---|
| `from media import get_medium` | `from tarang.media import get_medium` |
| `import geometries.ridge_waveguide as rw` | `import tarang.geometries.ridge_waveguide as rw` |
| `import waveguide` | `from tarang import waveguide` |
| `from grating_coupler import GratingCoupler` | `from tarang.grating_coupler import GratingCoupler` |
| `import gratingCoupler`, `edgeCoupler`, `ringResonator` | `tarang.grating_coupler`, `tarang.edge_coupler`, `tarang.ring_resonator` |
| `python -m waveguide.dispersion_mapping.sweep` | `python -m tarang.waveguide.dispersion_mapping.sweep` |

**Example results.** The example notebooks are now in `examples/` and save their results
in `examples/example_results/`. If you ran them before, move your old `example_results/`
folder into `examples/` so that their cached cloud results are reused rather than run
(and paid for) again. Cached results are matched by simulation fingerprint, so a moved
folder is safe to reuse.

**Names.** These still work, with a warning, until v0.2.0:

| Old | New |
|---|---|
| `edgeCoupler` (class) | `EdgeCoupler` |
| `ringResonator`, `halfringResonator`, `partialRingResonator` | `RingResonator`, `HalfRingResonator`, `PartialRingResonator` |
| `resonatorArc` | `ResonatorArc` |
| `sub_medium`, `substrate_medium` (oxide under the core) | `box_medium` |
| `substrate_height` (oxide thickness) | `box_height` |
| `substrate_medium`, `substrate_height` in the **grating coupler** (Si wafer) | `handle_medium`, `handle_height` |
| `cladding_medium`, `cladding_height` | `top_cladding_medium`, `top_cladding_height` |
| `wavelength0`, `wavelength0_um` | `wavelength_um` |
| `fine_min_steps_per_wvl`, `coarse_min_steps_per_wvl` | `fine_steps_per_wvl`, `coarse_steps_per_wvl` |

The grating coupler used `substrate_*` for the silicon handle while the other devices
used it for the oxide, so each function maps its own old names to their old meaning:
existing calls keep their behaviour. Passing an old and a new name together raises
`TypeError`.

Cached results from before v0.1.0 are reused automatically when the stored simulation
matches the new one exactly; otherwise TARANG explains what differs and runs again. Old
result folders are never modified or deleted.
