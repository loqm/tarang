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

- **Taper stage** (`--only taper`). The grating width that best matches the fibre: the
  lateral overlap η_y of the fibre's Gaussian with the grating's fundamental lateral mode
  (effective-index method; free), so the 3D efficiency is about η_2D·η_y (for the 1550 nm
  design the optimum is about 14.5 µm, η_y 99 % against 97 % at 12 µm). The taper length:
  one Tidy3D EME run of the linear taper (even modes, 30 cells) gives the fundamental-mode
  transmission against length (EMELengthSweep), and the shortest taper below
  `taper.max_loss_db` is chosen. The layout and the fibre-to-fibre loss budget use both
  (`taper.use_in_layout`).
- **Apodised grating stage** (`--only apodise`). The scattering strength α and the
  phase-matched period of each fill factor are measured on uniform gratings (small 2D
  runs: TE0 launched from the waveguide, field 1.5 µm above the grating; the decay of |E|²
  gives α, its phase slope the Bloch index). The tooth list follows
  α(x) = A·G²/(2(1 − A∫G²)) so the emitted light has the fibre's Gaussian shape; a 1D
  emission model (uniform grating: the textbook 80 % ceiling) predicts the gain before the
  2D FDTD check is priced. When the apodised grating beats the uniform one, the layout uses
  it (`layout/grating_teeth.csv`, and the teeth as a matrix in `parameters.m`).
  `GratingConfig(teeth=...)` builds a grating from an explicit tooth list.
- **Reflections.** The arc-end reflection of the partial ring and the bus reflection are
  read from the saved gap runs (free; checks stage, report). The reverse grating run is a
  closed box: a plane behind the source adds the light scattered backwards.
- **A picture of every cloud run's geometry** in `figures/sims/`, named like its cloud
  folder; `--runs` draws them for earlier runs too. Batch links name their case
  (e.g. `angle 11° (gc2D_...)`), so runs with the same name can be told apart.
- `inner_box_recess_um` for the full- and half-ring builders as well (0.1 keeps the old
  geometry bit for bit; 0 gives a flat BOX).
- EME simulations go through the same verified cache as FDTD runs.

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
