# TARANG test suite

```bash
python -m pytest                     # whole suite, ~4 min, no FlexCredits
python -m pytest -m "not slow"       # quick check, ~20 s (skips 11 mode-solve / fitting tests)
python -m pytest tests/test_geometries.py -q    # one file
python -m pytest --run-cloud tests/test_cloud_smoke.py -v   # REAL cloud run, ~0.03 FlexCredits
```

Expected on Windows: `138 passed, 2 skipped`. The two skips are the real-cloud
smoke test (opt-in) and the MultiNest test, which needs the native MultiNest
library (normally Linux/macOS only). Settings live in `pytest.ini` at the
repository root.

The same suite runs automatically on GitHub for every push and pull request
(`.github/workflows/tests.yml`): Linux and Windows, Python 3.11 and 3.12, plus a
Linux job with MultiNest installed, in which the MultiNest test must run, and a
"package" job that builds the wheel, installs it into a clean environment and
runs `installed_smoke.py` from a folder outside the repository. A second
workflow (`.github/workflows/conda-env.yml`) checks `environment.yml` on Windows
and Linux whenever that file or `pyproject.toml` changes. The real-cloud test
runs on GitHub only when started by hand.

Install the test dependencies with `python -m pip install -e ".[test]"` (they are
the `test` extra in `pyproject.toml`; `environment.yml` already includes them).

## How the tests work

A test is a function that sets up a situation, runs TARANG code and `assert`s
something that must be true. `pytest` collects every `test_*` function in this
folder. Each test is written so that it **fails if a specific bug returns**, so
these are regression tests. The audit IDs (C1, C2, ..., R1) refer to the v0.1.0
release audit.

Apart from the opt-in cloud test, the suite costs nothing to run:

1. **The local mode solver only.** Tidy3D's mode solver runs on your machine for
   free.
2. **A fake cloud.** `FakeWeb` in `conftest.py` (the `fake_web` fixture) replaces
   `tidy3d.web`. It records each call (upload, estimate, start, monitor, load,
   delete, and the same for batches) and "runs" a task by writing an empty
   result file. The tests check *what TARANG would ask the cloud to do*, and in
   which order.
3. **Analytic stand-ins.** Constant-index materials (n = 1.996, 1.444, 3.476) are
   used where real dispersion fits don't matter, and Gaussian likelihoods with a
   known evidence replace FDTD in the sampler tests.

Physics tests check **relations**: orderings (TM00 above TE00), signs (D > 0),
counts (one guided TE mode) and trends. They don't check 4-decimal values,
because coarse-mesh n_eff shifts by ~0.005 between Tidy3D versions.

**Markers** (declared in `pytest.ini`):

* `slow`: needs a local mode solve or a fresh material fit (tens of seconds).
* `cloud`: real Tidy3D job; runs only with `--run-cloud` and an API key.

**Warnings policy** (`pytest.ini`): TARANG calling one of its own deprecated
names is an **error**, so the old names can never creep back into the code.
Third-party warnings that are not TARANG's to fix (getdist on matplotlib 3.11)
are hidden, so a new warning stands out.

## What the tests don't cover

Whether a real FDTD result (for example a grating efficiency or a ring Q) is
physically right, and the analysis inside the ring gap/radius sweeps, which
needs real monitor data. `test_cloud_smoke.py --run-cloud` saves a real result
in `results/cloud_smoke/` that later offline tests can use.

---

## `test_waveguide_fixes.py`: waveguide physics (4 tests, PR #1)

| Test | Checks |
|---|---|
| `test_fix5_sio2_is_fused_silica` | `sio2` gives n = 1.444 at 1550 nm (Malitson). The old data is still available as `sio2_nyakuchena` (n = 1.420) |
| `test_fix1_fix3_get_dispersion_air` | 800×800 nm air-clad SiN: TM00 lies above TE00 (near-square core). `get_dispersion()` returns TE00, not TM00; D > 0 (anomalous); single-mode; cutoff index 1.444 |
| `test_fix3_top_cladding_in_mode_count` | With an SiO₂ top cladding, guided modes are counted against max(n_sub, n_top), so the same core is multimode |
| `test_fix2_fix4_dispersion_in_geometry_options` | The sweep accepts a top cladding and a mesh setting, and skips failed points instead of crashing |

## `test_dispersion_mapping.py`: dispersion-mapping package (3 tests, PR #1)

| Test | Checks |
|---|---|
| `test_media_registry_has_malitson` | The package reads the corrected SiO₂ (1.444) |
| `test_reference_point_800x800` | n_TE in 1.65–1.68, TM above TE, one guided TE mode, D > 0, smooth n_eff(λ) fit |
| `test_index_shift_raises_dispersion` | Raising the core index by 0.02 increases D, the expected physical trend |

## `test_cloud_run.py`: verified cache, cost guard and API key (15 tests, Phases 3a and 4)

| Test | Checks |
|---|---|
| `test_fingerprint_tracks_media_and_ignores_version` | Identical simulations give the same fingerprint, a different SiO₂ index gives a different one, and the Tidy3D version string is ignored |
| `test_c1_grating_cache_key_includes_media` | **C1 regression:** the old grating-coupler folder name was identical for two BOX materials, and the new one differs |
| `test_dir_names_are_short_enough_for_windows` | Cache folder names stay under 80 characters (Windows path limit) |
| `test_miss_then_hit` | First run: upload → estimate → start → monitor → load, then `simulation.json` + `.hdf5` are saved and no partial file is left. Second identical device: zero cloud calls |
| `test_r4_task_name_carries_fingerprint` | Cloud task names carry the fingerprint and fit the dashboard |
| `test_c6_max_cost_blocks_before_start` | Estimate 0.25 vs `max_cost` 0.1: error *before* `start`, the draft task is deleted, nothing is saved |
| `test_c6_env_max_cost` | The same limit set through `TARANG_MAX_COST` blocks or allows correctly |
| `test_cost_error_is_not_a_runtime_error` | The sweeps' `except RuntimeError` can't swallow a budget stop |
| `test_legacy_cache_reused_only_if_simulation_matches` | A pre-v0.1.0 folder is reused only if its stored simulation matches. Otherwise it is untouched and a warning names what differs |
| `test_partial_download_is_not_a_cache_hit` | An interrupted download is never treated as a result |
| `test_nested_sampler_budget_uses_single_upload` | One upload per design (it used to be two), a repeated design is free, and the credit limit stops a run before `start` |
| `test_api_key_found_in_simcloud_variable`, `..._in_tidy3d_config` (2) | The key is found where Tidy3D really looks: `SIMCLOUD_APIKEY` or the `tidy3d configure` file |
| `test_old_tarang_key_variable_is_passed_on` | `TIDY3D_API_KEY` (from earlier TARANG docs, ignored by Tidy3D) is passed on as `SIMCLOUD_APIKEY`, with a message |
| `test_missing_key_warns_with_the_fix` | No key: the warning says to run `tidy3d configure` |

## `test_phase3a_fixes.py`: correctness fixes (24 tests, Phase 3a)

| Test | Checks |
|---|---|
| `test_every_wrapper_cache_key_tracks_box_medium[edge/ring/arc/gc]` (4) | Every cloud device's cache key changes with the BOX (under-core oxide) material |
| `test_every_wrapper_runs_through_cost_guard[edge/ring/arc/gc]` (4) | Every cloud device stops before `start` when over `max_cost` |
| `test_c2_target_neff_is_core_index[...]` (6) | **C2:** for SiN (1.996) and Si (3.476) cores, every mode monitor and source in the edge coupler, grating coupler and ring targets the core index, so the solver picks the fundamental mode |
| `test_c2_edge_in_coupling_monitor_also_fixed` | Same for the edge coupler's in-coupling layout |
| `test_c2_no_hard_coded_targets_left` | No `target_neff=2.0` remains in `geometries/` |
| `test_c3_get_medium_warns_on_extrapolation` | **C3:** a 1.20–2.00 µm request on data ending at 1.65 µm warns; 1.20–1.60 µm does not |
| `test_c5_read_stacked_in_memory` | **C5:** a stacked n/k CSV is parsed in memory and the source is unchanged |
| `test_c5_fix_writes_copy_and_never_overwrites` | A merged copy is written, never overwritten, and never written onto the source |
| `test_c5_merged_copy_has_unix_newlines_on_any_platform` | Imitates Windows line endings: the copy has `\n` endings and no blank rows (the bug found on Windows) |
| `test_c5_get_medium_reads_stacked_file_without_modifying_it` | `get_medium` fits a stacked file (n = 1.99 ± 0.005) without touching it |
| `test_c4_get_modes_flags_unguided` | **C4:** a 150×100 nm core reports `has_guided_mode = False`, and `require_guided=True` raises |
| `test_r1_log_level_helper` | **R1:** `set_tidy3d_log_level` works on Tidy3D 2.12 |
| `test_r1_notebooks_use_no_removed_config` | No notebook assigns the removed `td.config.logging_level` / `show_progress_bars` |

## `test_nested_backends.py`: nested sampling on every OS (11 tests, Phase 3b)

Analytic problem: a 3-parameter Gaussian (σ = 0.1) inside a box prior [−1, 2]³,
so the exact evidence is ln Z = 3 ln(√(2π)·0.1) − 3 ln 3 = −7.447.

| Test | Checks |
|---|---|
| `test_auto_falls_back_to_dynesty` | Without MultiNest, `backend="auto"` picks dynesty |
| `test_explicit_multinest_unavailable_gives_clear_error` | Requesting MultiNest where it can't load gives an error that points to dynesty |
| `test_unknown_backend_rejected` | A misspelt backend name is rejected |
| `test_probe_survives_pymultinest_systemexit` | Without its library, `import pymultinest` raises `SystemExit` (it would quit Python). The probe reports "unavailable" instead |
| `test_dynesty_recovers_analytic_evidence_and_posterior` | dynesty's ln Z is within 3σ (+0.2) of the exact value. The posterior mean is within 0.03 and the width within 30 %. The MultiNest-format files are read by TARANG's existing loader |
| `test_same_seed_same_run` | The same seed gives an identical run (needed for resuming) |
| `test_interrupted_run_resumes_by_exact_replay` | A stop after 150 calls, then resume: the resumed run re-requests **exactly the same 150 points in the same order** (cache hits for FDTD) and ends with the same ln Z as an uninterrupted run |
| `test_mixing_backends_in_one_directory_is_refused` | Resuming a MultiNest directory with dynesty (or the reverse) is refused |
| `test_multinest_backend_same_interface` | *(skipped without MultiNest)* Real MultiNest recovers the same ln Z and posterior through the same interface |
| `test_optimizer_search_with_dynesty` | Full `execute_search(backend="dynesty")` on an analytic efficiency surface: converges, finds period 0.82 µm and FF 0.51, and reports 60 % peak efficiency, the backend and ln Z |
| `test_budget_pause_and_resume_reuses_cache` | Budget 0.375 at 0.125 per run: pauses after 3 paid runs. Resuming at 0.75 replays those 3 from the cache (no new uploads) and pays for exactly 3 new ones |

## `test_api_compat.py`: v0.1.0 names and deprecated aliases (20 tests, Phase 3c)

The old names keep working until v0.2.0, each with a `DeprecationWarning` that
names the replacement.

| Test | Checks |
|---|---|
| `test_old_package_names_alias_the_new_modules[...]` (3) | `import gratingCoupler` / `edgeCoupler` / `ringResonator` warn, and `oldPkg.module` is the **same module object** as `new_pkg.module` (isinstance checks and monkeypatching keep working) |
| `test_old_class_names_are_the_new_classes[...]` (6) | `edgeCoupler`, `ringResonator`, `halfringResonator`, `partialRingResonator`, `resonatorArc` warn and are the new classes |
| `test_ring_cache_labels_keep_pre_rename_names` | Ring cache folders and cloud task names still use the pre-rename labels, so existing results stay findable |
| `test_old_keywords_build_identical_simulation[edge/ring/arc/waveguide]` (4) | Old keywords (`substrate_medium`, `sub_medium`, `wavelength0`, `fine_min_steps_per_wvl`, ...) build a simulation with the **same fingerprint** as the new ones, and each warning names old and new |
| `test_grating_substrate_medium_still_means_the_si_handle` | **Safety-critical:** in the grating coupler the old `substrate_medium` was the Si handle. It maps to `handle_medium`, never to `box_medium`; same for the `GratingConfig` heights |
| `test_old_and_new_keyword_together_is_an_error` | Passing both names raises `TypeError` |
| `test_old_attribute_names_forward_with_warning` | Old attribute names read and write through to the new ones (including `RidgeWaveguide`, which forwards other attributes to its Simulation) |
| `test_nested_sampler_accepts_old_dictionary_keys` | Old `fixed_params` / `materials` keys work with a warning; giving both raises |
| `test_no_public_signature_uses_old_names` | No public function in the device packages still takes an old name |
| `test_every_alias_points_at_a_real_parameter` | Every alias maps to a parameter that exists (guards against typos) |

## `test_geometries.py`: every builder, validated (28 tests, Phase 4)

| Test | Checks |
|---|---|
| `test_builder_passes_tidy3d_pre_upload_validation[...]` (16) | All 16 builders (ridge, edge passive/out/in, full/half/partial ring passive/active, arc passive/active, grating 2D, 2D with Si handle, 3D) pass Tidy3D's own server-side check `validate_pre_upload()`, so a broken geometry is caught before any upload. Active builds have a source |
| `test_analysis_monitor_names_exist_in_builder[...]` (8) | Every monitor the analysis methods read by name (e.g. `field_xy`, `mode monitor`, `Mode Monitor In`) is created by the matching builder |
| `test_grating_handle_layer_present_only_when_requested` | The Si handle appears only when `handle_medium` is given |
| `test_ring_conversions_keep_geometry` | `to_half_ring()` / `to_full_ring()` return the right classes with unchanged radius, gap and width |
| `test_fsr_formulas` | FSR = λ²/(n_g·2πR) and c/(n_g·2πR), consistent with each other |
| `test_monitor_data_explains_what_is_missing` | A missing monitor is reported as "not defined" or "defined but no data", and old `except KeyError` / `except ValueError` handlers still catch it |

## `test_edge_sweep.py`: batch sweeps (4 tests, Phase 4)

| Test | Checks |
|---|---|
| `test_batch_runs_uncached_points_once_then_reuses_them` | A 2×2 sweep is one batch of 4; repeating it starts nothing; adding a tip width runs only the 2 new points |
| `test_batch_over_budget_is_not_started` | Estimate 0.4 > `max_cost` 0.3: the batch is deleted before it starts and nothing is cached |
| `test_sequential_mode_caps_each_run` | With `batching=False`, `max_cost` applies to each run |
| `test_sweep_returns_zero_efficiency_for_unreadable_results` | Results without data give efficiency 0 and a warning, not a crash |

## `test_dispersion_mapping_cli.py`: command-line tools end to end (5 tests, slow, Phase 4)

| Test | Checks |
|---|---|
| `test_sweep_writes_one_record_per_geometry` | `sweep` on 700 nm × {900, 1000} nm writes two records; both anomalous (D > 0) and single-mode TE |
| `test_sweep_resumes_without_redoing_points` | Re-running `sweep` reports "2 already done, 0 to run" and leaves the file byte-identical |
| `test_maps_never_overwrite_previous_plots` | Running `maps` twice creates two versioned plot folders |
| `test_report_compares_finalists` | `report` writes `report.md` and `finalists.csv` for the requested finalists |
| `test_tools_survive_windows_legacy_console_encoding` | With Windows' legacy console encoding (cp1252), `report` no longer crashes on λ/α, and its files stay UTF-8 |

## `test_waveguide_sweeps.py`: mode and confinement sweeps (2 tests, Phase 4)

| Test | Checks |
|---|---|
| `test_confinement_rises_with_width` | Confinement is between 0 and 100 % and rises from 800 to 1000 nm width |
| `test_mode_sweep_writes_its_figure` | `modes_in_geometry` runs and saves its figure |

## `test_utilities.py`: shared helpers (5 tests, Phase 4)

| Test | Checks |
|---|---|
| `test_positive`, `test_integer`, `test_values_rejects_duplicates_and_short_lists` | Input validation used by every sweep |
| `test_to_jsonable_handles_numpy_infinities_and_tidy3d_objects` | Arrays, infinities and Tidy3D objects become strict JSON, without pydantic deprecation warnings |
| `test_save_json_creates_folders_and_replaces_atomically` | Reports are written atomically into new folders |

## `test_cloud_smoke.py`: real Tidy3D cloud (1 test, opt-in, Phase 4)

| Test | Checks |
|---|---|
| `test_real_cloud_run_then_cache_hit` | A small 2D grating coupler (`max_cost` 0.1 FlexCredits) goes through TARANG's real upload → estimate → start → monitor → download sequence, returns real monitor data and an efficiency in [0, 1], and the second request is a free cache hit. A copy of the result is kept in `results/cloud_smoke/` |


## `test_packaging.py`: the package definition (18 tests, Phase 5)

`pyproject.toml` lists by hand what an installed TARANG contains. These tests
fail when that list falls behind the code. Each one was checked by breaking a copy
of the repository on purpose (dropping a package from the list, adding an
undeclared import, importing an optional library at module top, adding a material
file outside `media/data/`).

| Test | Checks |
|---|---|
| `test_version_has_a_single_source` | The version is read from `tarang.__version__` only, and has release form (`0.1.0`, `0.2.0rc1`) |
| `test_running_python_satisfies_requires_python` | The Python running the tests is one TARANG declares support for (3.11+) |
| `test_readme_and_licence_exist` | The README and licence named in the package metadata exist |
| `test_every_package_folder_is_shipped` | Every top-level code folder (including the deprecated old names) is in the package list, so `pip install` does not silently drop one |
| `test_tests_and_docs_are_not_shipped` | `tests/`, `docs/` and `results/` stay out of the installed package |
| `test_subpackages_have_init_files` | Every code folder has an `__init__.py`; without one, setuptools leaves it out of the wheel even though it imports fine from the repository |
| `test_material_files_are_package_data` | Every material in `media/media.json` points to an existing CSV that is covered by the package-data patterns |
| `test_every_third_party_import_is_declared` | Every non-standard-library import in the code is a declared dependency or extra |
| `test_core_dependencies_are_required_not_optional` (×6) | tidy3d, numpy, pandas, pydantic, matplotlib and dynesty are required, not extras |
| `test_optional_imports_are_guarded` | getdist, corner and pymultinest are never imported at module top, so TARANG imports without the extras |
| `test_test_extra_installs_what_the_test_suite_imports` | `.[test]` brings pytest, pytest-cov and the plotting libraries |
| `test_environment_yml_installs_this_package` | `environment.yml` pins Python 3.11+ and installs TARANG with the test extra |
| `test_ci_installs_from_pyproject` | The GitHub workflow installs from `pyproject.toml` (not the removed `requirements-test.txt`) and tests only supported Python versions. Skipped when run from a source archive |

## `installed_smoke.py`: the installed package (script, not collected by pytest)

Run with `python -I tests/installed_smoke.py` **from a folder outside the
repository**, after `pip install`. `-I` makes sure Python uses the installed copy,
not the repository folders. It checks that every module imports, every packaged
material loads and fits, a ridge-waveguide simulation passes Tidy3D's server-side
validation, the dynesty backend is available and the old package names still work.
No network, no FlexCredits.
