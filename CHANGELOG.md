# Changelog

All notable changes to TARANG are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and version numbers follow
[Semantic Versioning](https://semver.org/) (before 1.0, a minor release may contain
breaking changes; they are announced here first).

## [0.1.0] - 2026-10-16

First public release. Compared with the pre-release code on `main`, TARANG is now
installable, tested on Linux and Windows, safer with cloud credits, and uses
consistent names. Old names keep working until v0.2.0 (see
[Upgrading](#upgrading-from-pre-010-code)).

### Added

- **Dispersion mapping** (`waveguide.dispersion_mapping`): command-line tools for robust
  height × width dispersion sweeps, maps, mesh-convergence studies, broadband *D*(λ) and
  integrated dispersion, and finalist reports against a baseline. Local mode solver
  only, no FlexCredits; results are appended line by line, so interrupted sweeps resume.
- **Verified result cache** (`utilities/cloud_run.py`): every cloud result is stored
  under a SHA-256 fingerprint of the complete simulation (geometry, materials, mesh,
  sources, monitors), with `simulation.json` saved next to `raw_simulation_data.hdf5`.
  Each cache hit is checked against the stored simulation.
- **Cost guard**: each job is priced before it starts and stopped if it exceeds
  `max_cost` (per device or sweep) or `$TARANG_MAX_COST`; `CostLimitExceededError` is
  raised and nothing is charged.
- **Nested sampling on every OS**: dynesty backend next to MultiNest
  (`utilities/nested_backends.py`); `backend="auto"` picks MultiNest when its library
  loads and dynesty otherwise. Both write MultiNest-format chains. Seeded dynesty runs
  resume by replaying cached evaluations.
- **Installable package**: `pyproject.toml` (distribution `tarang-photonics`) with extras
  `[plots]`, `[notebooks]`, `[multinest]`, `[test]`; cross-platform `environment.yml`.
- **Material `si_franta`**: float-zone crystalline silicon (Franta 2017, 0.03–310 µm),
  for silicon at 1550 nm, where the `si` dataset has no data.
- **Documentation**: this changelog, `CONTRIBUTING.md`, `CITATION.cff`,
  `media/data/SOURCES.md` (reference, sample and range of every material dataset), and
  material fit-versus-data figures (`docs/make_material_figures.py`).
- **Tests and CI**: 160+ tests (fake cloud, no FlexCredits) run on every pull request on
  Ubuntu and Windows with Python 3.11 and 3.12, plus a MultiNest job, a job that builds
  and installs the package, a conda-environment check, and an optional real-cloud smoke
  test started by hand.
- `tarang.set_tidy3d_log_level()`; `RidgeWaveguide.get_modes(require_guided=True)`;
  `MonitorDataError` explains whether a monitor is undefined or has no data.

### Changed

- **Names** follow one convention (PEP 8): packages `grating_coupler`, `edge_coupler`,
  `ring_resonator`; classes in CapWords; one parameter name per physical layer
  (`box_*`, `handle_*`, `top_cladding_*`) and `wavelength_um` everywhere. Old names are
  deprecated aliases (table below).
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

All names below work in v0.1.0 with a `DeprecationWarning` naming the replacement, and
will be **removed in v0.2.0**.

### Removed

- `requirements-test.txt` (replaced by the `[test]` extra).
- The root `media.json`: an outdated copy of `media/media.json` that nothing read (it still
  pointed `sio2` to the microsphere data).

### Upgrading from pre-0.1.0 code

| Old | New |
|---|---|
| `import gratingCoupler`, `edgeCoupler`, `ringResonator` | `grating_coupler`, `edge_coupler`, `ring_resonator` |
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
