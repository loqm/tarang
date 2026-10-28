# TARANG test suite

```bash
python -m pytest                     # whole suite, ~15-20 min, no FlexCredits
python -m pytest -m "not slow"       # quick check, ~2-3 min (skips 21 mode-solve / fitting tests)
python -m pytest tests/test_geometries.py -q    # one file
python -m pytest --run-cloud tests/test_cloud_smoke.py -v   # REAL cloud run, ~0.03 FlexCredits
```

Expected on Windows: `507 passed, 2 skipped` (quick check: `486 passed, 2 skipped, 21 deselected`). The two skips are the real-cloud
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
4. **Fast stand-ins in the design-tool tests** (autouse fixtures in `conftest.py`): the
   design tool's local mode solves for figures and checks (`tarang.design.localmodes`)
   return a synthetic TE00/TM00 pair and supermodes, and figures are written as 72 dpi
   PNG only. Tests marked `real_modes` or `real_figures` use the real ones.

Physics tests check **relations**: orderings (TM00 above TE00), signs (D > 0),
counts (one guided TE mode) and trends. They don't check 4-decimal values,
because coarse-mesh n_eff shifts by ~0.005 between Tidy3D versions.

**Markers** (declared in `pytest.ini`):

* `slow`: needs a local mode solve or a fresh material fit (tens of seconds).
* `cloud`: real Tidy3D job; runs only with `--run-cloud` and an API key.
* `real_modes`: the design tool's real local mode solver (slow tests).
* `real_figures`: figures with the configured formats and resolution.

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

## `test_dispersion_mapping.py`: dispersion-mapping package (4 tests, PR #1 and Phase 10)

| Test | Checks |
|---|---|
| `test_media_registry_has_malitson` | The package reads the corrected SiO₂ (1.444) |
| `test_reference_point_800x800` | n_TE in 1.65–1.68, TM above TE, one guided TE mode, D > 0, smooth n_eff(λ) fit |
| `test_index_shift_raises_dispersion` | Raising the core index by 0.02 increases D, the expected physical trend |
| `test_materials_are_fitted_over_the_band_that_is_solved` | Telecom bands keep the 1.1–2.4 µm material fit (results unchanged); a 780 nm band gets a fit that covers it; the D_int pump lies in the solved band |

## `test_cloud_run.py`: verified cache, cost guard and API key (16 tests, Phases 3a, 4 and 9)

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
| `test_fresh_fit_and_cached_fit_give_the_same_fingerprint` (slow) | A simulation built right after a material is fitted has the same fingerprint as in a later session that reads the fit from the cache, so a result run in the fitting session is found again (it was not before Phase 9) |

## `test_phase3a_fixes.py`: correctness fixes (28 tests, Phases 3a and 10)

| Test | Checks |
|---|---|
| `test_every_wrapper_cache_key_tracks_box_medium[edge/ring/arc/gc]` (4) | Every cloud device's cache key changes with the BOX (under-core oxide) material |
| `test_every_wrapper_runs_through_cost_guard[edge/ring/arc/gc]` (4) | Every cloud device stops before `start` when over `max_cost` |
| `test_c2_target_neff_is_core_index[...]` (6) | **C2:** for SiN (1.996) and Si (3.476) cores, every mode monitor and source in the edge coupler, grating coupler and ring targets the core index, so the solver picks the fundamental mode |
| `test_c2_edge_in_coupling_monitor_also_fixed` | Same for the edge coupler's in-coupling layout |
| `test_c2_no_hard_coded_targets_left` | No `target_neff=2.0` remains in `tarang/geometries/` |
| `test_c3_get_medium_warns_on_extrapolation` | **C3:** a 1.20–2.00 µm request on data ending at 1.65 µm warns; 1.20–1.60 µm does not |
| `test_c5_read_stacked_in_memory` | **C5:** a stacked n/k CSV is parsed in memory and the source is unchanged |
| `test_c5_fix_writes_copy_and_never_overwrites` | A merged copy is written, never overwritten, and never written onto the source |
| `test_c5_merged_copy_has_unix_newlines_on_any_platform` | Imitates Windows line endings: the copy has `\n` endings and no blank rows (the bug found on Windows) |
| `test_c5_get_medium_reads_stacked_file_without_modifying_it` | `get_medium` fits a stacked file (n = 1.99 ± 0.005) without touching it |
| `test_c4_get_modes_flags_unguided` | **C4:** a 150×100 nm core reports `has_guided_mode = False`, and `require_guided=True` raises |
| `test_r1_log_level_helper` | **R1:** `set_tidy3d_log_level` works on Tidy3D 2.12 |
| `test_r1_notebooks_use_no_removed_config` | No notebook assigns the removed `td.config.logging_level` / `show_progress_bars` |
| `test_get_medium_accepts_your_own_csv_file` (slow) | A material given as a CSV path is fitted and cached like a built-in one; a missing file is reported |
| `test_fit_follows_sparse_data_between_the_points` (slow) | `si3n4` over 0.68–0.88 µm (three data points) follows the data between the points, by whichever route the fit takes on that machine; the 1550 nm fit is kept unchanged |
| `test_swinging_fit_is_refitted` | A fit that strays between the points (forced, so it runs the same everywhere) is refitted with fewer poles, flagged `smoothness_fallback`, reported, and reused from the cache |
| `test_fit_that_never_follows_the_data_is_refused` | With no acceptable fit, `get_medium` tries fewer poles, then wider ranges, then raises and caches nothing (stand-in fitter, instant) |

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

## `test_api_compat.py`: v0.1.0 names and deprecated aliases (17 tests, Phase 3c)

The old class, parameter and attribute names keep working until v0.2.0, each with a
`DeprecationWarning` that names the replacement. (The old package names and the old
top-level layout are not aliased: `python -m tarang.migrate` rewrites imports instead,
see `test_migrate.py`.)

| Test | Checks |
|---|---|
| `test_old_class_names_are_the_new_classes[...]` (6) | `edgeCoupler`, `ringResonator`, `halfringResonator`, `partialRingResonator`, `resonatorArc` warn and are the new classes |
| `test_ring_cache_labels_keep_pre_rename_names` | Ring cache folders and cloud task names still use the pre-rename labels, so existing results stay findable |
| `test_old_keywords_build_identical_simulation[edge/ring/arc/waveguide]` (4) | Old keywords (`substrate_medium`, `sub_medium`, `wavelength0`, `fine_min_steps_per_wvl`, ...) build a simulation with the **same fingerprint** as the new ones, and each warning names old and new |
| `test_grating_substrate_medium_still_means_the_si_handle` | **Safety-critical:** in the grating coupler the old `substrate_medium` was the Si handle. It maps to `handle_medium`, never to `box_medium`; same for the `GratingConfig` heights |
| `test_old_and_new_keyword_together_is_an_error` | Passing both names raises `TypeError` |
| `test_old_attribute_names_forward_with_warning` | Old attribute names read and write through to the new ones (including `RidgeWaveguide`, which forwards other attributes to its Simulation) |
| `test_nested_sampler_accepts_old_dictionary_keys` | Old `fixed_params` / `materials` keys work with a warning; giving both raises |
| `test_no_public_signature_uses_old_names` | No public function in the device packages still takes an old name |
| `test_every_alias_points_at_a_real_parameter` | Every alias maps to a parameter that exists (guards against typos) |

## `test_geometries.py`: every builder, validated (31 tests, Phases 4 and 10)

| Test | Checks |
|---|---|
| `test_builder_passes_tidy3d_pre_upload_validation[...]` (18) | All 18 builds (ridge, edge passive/out/in, full/half/partial ring passive/active, arc passive/active, grating 2D, 2D with Si handle, 3D, and full-etch 2D and 3D) pass Tidy3D's own server-side check `validate_pre_upload()`, so a broken geometry is caught before any upload. Active builds have a source |
| `test_analysis_monitor_names_exist_in_builder[...]` (8) | Every monitor the analysis methods read by name (e.g. `field_xy`, `mode monitor`, `Mode Monitor In`) is created by the matching builder |
| `test_grating_handle_layer_present_only_when_requested` | The Si handle appears only when `handle_medium` is given |
| `test_full_etch_grating_has_no_floor_and_deeper_etch_is_refused` | A full etch (etch depth = film thickness) leaves no core under the grooves; an etch deeper than the film raises |
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


## `test_packaging.py`: the package definition (19 tests, Phases 5 and 8)

`pyproject.toml` lists by hand what an installed TARANG contains. These tests
fail when that list falls behind the code. Each one was checked by breaking a copy
of the repository on purpose (dropping a package from the list, adding an
undeclared import, importing an optional library at module top, adding a material
file outside `tarang/media/data/`).

| Test | Checks |
|---|---|
| `test_version_has_a_single_source` | The version is read from `tarang.__version__` only, and has release form (`0.1.0`, `0.2.0rc1`) |
| `test_running_python_satisfies_requires_python` | The Python running the tests is one TARANG declares support for (3.11+) |
| `test_readme_and_licence_exist` | The README and licence named in the package metadata exist |
| `test_every_package_folder_is_shipped` | Every code folder is in the package list, so `pip install` does not silently drop one |
| `test_tarang_is_the_only_top_level_package` | `tarang` is the only top-level package in the repository and in the package list, so an installed TARANG adds no generic names such as `media` or `utilities` to Python |
| `test_tests_and_docs_are_not_shipped` | `tests/`, `docs/`, `examples/` and `results/` stay out of the installed package |
| `test_subpackages_have_init_files` | Every code folder has an `__init__.py`; without one, setuptools leaves it out of the wheel even though it imports fine from the repository |
| `test_material_files_are_package_data` | Every material in `tarang/media/media.json` points to an existing CSV that is covered by the package-data patterns |
| `test_every_third_party_import_is_declared` | Every non-standard-library import in the code is a declared dependency or extra |
| `test_core_dependencies_are_required_not_optional` (×6) | tidy3d, numpy, pandas, pydantic, matplotlib and dynesty are required, not extras |
| `test_optional_imports_are_guarded` | getdist, corner, pymultinest and gdstk are never imported at module top, so TARANG imports without the extras |
| `test_test_extra_installs_what_the_test_suite_imports` | `.[test]` brings pytest, pytest-cov, the plotting libraries and gdstk (`layout` extra) |
| `test_environment_yml_installs_this_package` | `environment.yml` pins Python 3.11+ and installs TARANG with the test extra |
| `test_ci_installs_from_pyproject` | The GitHub workflow installs from `pyproject.toml` (not the removed `requirements-test.txt`) and tests only supported Python versions. Skipped when run from a source archive |

## `installed_smoke.py`: the installed package (script, not collected by pytest)

Run with `python -I tests/installed_smoke.py` **from a folder outside the
repository**, after `pip install`. `-I` makes sure Python uses the installed copy,
not the repository folders. It checks that every module imports, every packaged
material loads and fits, a ridge-waveguide simulation passes Tidy3D's server-side
validation, the dynesty backend is available, and `python -m tarang.migrate` and `python -m tarang.design` work.
No network, no FlexCredits.

## `test_docs.py`: the documentation matches the code (25 tests, Phases 6 and 14D)

The pre-v0.1.0 README described parameters and a function that no longer existed and an
install procedure that did not work. These tests read the Markdown files (README,
CONTRIBUTING, CHANGELOG, `tarang/media/data/SOURCES.md`, the dispersion-mapping README and this
file) and check them against the code. Each check was confirmed to fail on a
deliberately broken copy (misspelt keyword, renamed function, broken link, undocumented
material, unknown pip extra).

| Test | Checks |
|---|---|
| `test_python_examples_compile[...]` (6) | Every Python example in the docs is valid Python |
| `test_python_examples_use_real_names_and_keywords[...]` (6) | Every TARANG name an example imports exists, and every keyword argument it passes to a TARANG function is a real parameter |
| `test_shell_commands_name_real_modules_and_extras` | `python -m ...` commands name existing TARANG modules; `pip install -e ".[...]"` names existing extras |
| `test_no_outdated_instructions` | Known-wrong pre-v0.1.0 instructions (`tarang-env`, the manual `pip install tidy3d pandas ...`, `sub_medium=`, a function that never existed) do not come back, and no code block imports or runs a module by its pre-v0.1.0 top-level name (`from media ...`, `python -m waveguide...`) |
| `test_relative_links_resolve[...]` (7) | Every relative link and image points to an existing file, and every `#anchor` to an existing heading |
| `test_citation_matches_package` | `CITATION.cff` is CFF 1.2.0 with the package's version, licence and authors, and its release date matches the CHANGELOG |
| `test_changelog_has_current_version` | `CHANGELOG.md` has a section for `tarang.__version__` |
| `test_every_material_is_documented` | Every key in `tarang/media/media.json` and every CSV in `tarang/media/data/` is described in `SOURCES.md`, and every key is in the README's material table. Fails if the data folder holds no CSVs, so a moved folder cannot make it pass vacuously |
| `test_readme_first_run_examples_execute` (slow) | README quick-start steps 1–3 and the logging example run as written in an empty folder: materials fit, the waveguide builds, modes, dispersion and confinement are computed, and the files the README mentions are written |

## `test_notebooks.py`: the example notebooks (43 tests, Phases 7–9)

The notebooks live in `examples/`. They are committed without outputs: stored outputs go stale with every code change,
had grown the repository by about 10 MB, and contained paths from the author's machine.
Each check below was confirmed to fail on the notebooks as they were before Phase 7.

| Test | Checks |
|---|---|
| `test_there_are_notebooks` | The example notebooks are found (seven) |
| `test_committed_without_outputs[...]` (7) | No outputs or execution counts are stored |
| `test_no_personal_paths[...]` (7) | No `/home/<user>`, `/Users/<user>` or `C:\\Users\\...` paths |
| `test_code_compiles_and_uses_real_names[...]` (7) | Code cells are valid Python; every TARANG name exists and every keyword argument is a real parameter |
| `test_image_links_exist[...]` (7) | Every image a notebook shows is committed |
| `test_markdown_links_resolve[...]` (7) | Every relative link in a notebook's text points to an existing file (links are relative to `examples/`) |
| `test_materials_fitted_over_the_simulated_wavelength[...]` (7) | A material is fitted over a band that contains the wavelength the same notebook simulates (the 1550 nm examples used to fit over 0.6–1.2 µm) |

## `test_migrate.py`: updating old scripts (19 tests, Phase 8)

Before v0.1.0 TARANG's packages were top-level (`import media`); now they live in the
`tarang` package (`import tarang.media`). `python -m tarang.migrate` rewrites the imports
in users' scripts and notebooks. These tests run the rewritten code, so a migrated script
needs no other edits.

| Test | Checks |
|---|---|
| `test_rewrites_import_statements[...]` (7) | `from media import ...`, dotted, aliased, camelCase (`gratingCoupler`, `edgeCoupler`) and mixed (`import numpy as np, media`) imports are rewritten correctly |
| `test_rewritten_code_binds_the_same_names` | The rewritten imports are executed: every name the old code used (`media`, `geometries.ridge_waveguide`, `gratingCoupler`, imports inside functions) is still bound to the right module or object |
| `test_multiline_and_indented_imports_keep_their_layout` | A parenthesised import inside a block keeps its indentation and comments; only the module name changes |
| `test_leaves_other_code_alone[...]` (5) | Relative imports, other packages with similar names (`mediapipe`), already migrated imports and strings such as `'media.json'` are not changed |
| `test_is_idempotent` | Running the tool twice changes nothing the second time |
| `test_reports_old_paths_in_strings_and_comments_without_changing_them` | Old module paths in strings (e.g. `monkeypatch.setattr("utilities.cloud_run.web", ...)`) and comments are listed with file and line for checking by hand, and left unchanged |
| `test_every_target_module_exists` | Every new module name in the tool's mapping imports |
| `test_notebook_code_cells_are_migrated_and_magics_kept` | In a notebook only code cells change; `%matplotlib` and `!pip` lines and Markdown cells are kept |
| `test_command_line_dry_run_then_write_keeps_line_endings` | Without `--write` nothing is changed and the files that would change are listed; with `--write` Windows (CRLF) line endings are kept; a file that is not valid Python is reported and left untouched; a mistyped path is reported |

## `test_design.py`: the guided design tool (17 tests, Phases 10 and 11)

The local stages call the dispersion-mapping tools, tested above; these tests check the
tool around them on a design whose earlier results are filled in by hand, with the fake
cloud for the cloud stages.

| Test | Checks |
|---|---|
| `test_settings_file_round_trip_and_unknown_keys` | `design.toml` is written and read back exactly; unset values stay unset; a misspelt setting is an error |
| `test_list_and_range_answers` | Answers such as `0.6, 0.8` and `0.60:0.90:0.10` become lists |
| `test_non_interactive_answers_use_defaults_and_are_recorded` | With `--yes`, defaults are used and written to the settings |
| `test_interactive_answer_is_checked_and_stored` | A wrong answer is refused with a message and asked again |
| `test_a_settings_change_reruns_that_stage_and_later_ones` | Changing the FSR target reruns the ring stage and every later one, not the cross-section |
| `test_physics_helpers` | R from FSR and back; slab TE index of 220 nm Si (2.848); Q_int from loss; loaded Q |
| `test_over_coupled_gap_is_narrower_and_k1_is_critical` | The over-coupled gap from the coupling fit is narrower, and K = 1 gives the critical gap |
| `test_candidates_are_ranked_by_robustness` | Map points that are anomalous and single-mode are ranked by how many neighbours are too |
| `test_gap_stage_prices_every_gap_then_runs` | Every gap is priced (uploaded and deleted) before the sweep runs; the cost is added to the budget; per-gap table and coupling figure written |
| `test_gap_stage_over_budget_is_skipped_without_running` | Above the budget nothing starts and the stage is marked skipped |
| `test_grating_stage_maps_retunes_then_places_the_fibre` | With made-up spectra: full etch uses the film thickness; the map, the retune and the fibre-position steps each run as one batch; the final peak lands on 1550 nm; table and figures written |
| `test_grating_stage_flags_a_best_design_at_the_edge` | A best fill factor at the end of the range is reported |
| `test_grating_stage_skips_features_below_the_lithography_limit` | No tooth or groove narrower than `min_feature_um` is simulated |
| `test_layout_reproduces_the_design_dimensions` | GDS files exist; ring outer radius R + W/2 and the gap above the bus are exact; both couplers have all teeth; parameters in JSON and MATLAB |
| `test_trench_tone_draws_the_etched_regions` | Trench tone draws a trench of the requested width around the core |
| `test_report_lists_values_to_check_elsewhere` | The report has every section; `check_values.csv` and `design_summary.json` are written, with no solver-specific content |
| `test_command_line_lists_the_stages` | `python -m tarang.design --list` works |

## `test_phase11.py`: fibre orientation, spectra, coupling measures, benchmark pack (23 tests, Phase 11)

| Test | Checks |
|---|---|
| `test_fiber_tilt_sets_the_beam_direction_and_where_it_lands[towards/away]` (2) | The beam travels towards -x (the waveguide) for "towards" and +x for "away", 12° from vertical, TE polarised, and its centre meets the core at `source_x` |
| `test_towards_is_the_default_and_the_sign_of_theta_is_ignored` | "towards" is the default; -12° and 12° give the same simulation; an unknown orientation is refused |
| `test_the_mode_monitor_records_a_band_around_the_wavelength` | The spectrum covers 1.45-1.65 µm by default; a zero band is refused |
| `test_phase_matching_formula_for_both_orientations` | λ/(n − sin θ) and λ/(n + sin θ); the old 0.820 µm "away" example is the 1.052 µm "towards" grating |
| `test_spectrum_summary_peak_value_and_bandwidths` | On a Gaussian: peak, value at 1550 nm and 1 dB / 3 dB widths match the formulas; a peak outside the band is flagged |
| `test_ring_port_and_bus_only_remove_the_floor_that_biases_the_gap` | With a 0.2 % gap-independent bus loss, "ring_port" and "bus_only" find the true critical gap (within 2 nm) and "source" puts it at least 30 nm too wide |
| `test_progress_listener_sees_runs_and_cache_hits` | A run gives "estimated", "running", "done"; the same simulation again gives "cached" |
| `test_expired_credits_stop_a_cloud_stage_before_anything_is_uploaded` | Expired credits or too few credits stop a cloud stage with the reason |
| `test_settings_file_with_a_byte_order_mark_loads` | A `design.toml` saved by Windows PowerShell 5 (byte-order mark) is read |
| `test_loss_presets_come_with_sources_and_your_values_are_marked` | Every loss preset has a source; your own value is labelled as yours |
| `test_candidates_tie_break_on_the_te_tm_split` | Equally robust candidates are ordered by the TE00/TM00 index split |
| `test_nested_sampler_passes_the_fibre_orientation` | The nested sampler builds "towards" couplers unless told otherwise |
| `test_report_writes_a_complete_benchmark_pack` | `benchmark_pack/` has its README (every section), assumptions, results with the threshold, settings and a zip; nothing solver-specific |
| `test_describe_sim_reads_monitor_bands_stored_as_arrays` | A saved grating simulation (monitor wavelengths stored as an array by Tidy3D) is described in the benchmark README with its 1.45-1.65 µm band |
| `test_grating_stage_through_report_and_benchmark_pack` | Gap, grating and layout stages, then the report: the benchmark pack describes and contains the final grating simulation |
| `test_period_is_varied_again_when_the_best_fill_factor_moves` | When step B picks another fill factor, step B2 varies the period around it (four batches: A, B, B2, C) |
| `test_sensitivity_panels_name_what_their_runs_held_fixed` | Each sensitivity panel names the fill factor or period its own runs used, not the final design's |
| `test_a_run_loaded_twice_is_counted_once` | A cached run loaded again (for its field map) is counted once in the progress lines |
| `test_describe_sim_without_a_gaussian_beam` | A ring simulation (mode source, one wavelength) is described as well as a grating one |
| `test_spectrum_summary_finds_the_peak_between_samples_and_the_band_edges` | On 2 nm samples the peak wavelength is found to 0.2 nm; the 1 dB band edges match the Gaussian |
| `test_centring_puts_the_peak_on_the_design_wavelength` | With a peak wavelength nonlinear in the period and a skewed spectrum, step D puts the peak within 1 nm of 1550 nm inside the 1 dB band; `centre_peak = False` keeps the highest efficiency at 1550 nm |
| `test_a_simulation_that_cannot_be_described_does_not_stop_the_pack` | An unreadable simulation file is noted in the README instead of stopping the report |

## `test_phase12.py`: run links, credits, one-table forms, checks, LLE inputs (22 tests, Phase 12A)

| Test | Checks |
|---|---|
| `test_a_run_records_its_task_and_web_link` | A cloud run writes `task.json` with the task id, a web link containing it, its estimate and date; `collect_runs` lists it |
| `test_an_interrupted_run_is_reattached_not_paid_again` | A run stopped (Ctrl+C) while running on Tidy3D is picked up again: one upload, one start, the "reattached" event carries the link |
| `test_a_failed_task_is_submitted_again` | A task that failed on Tidy3D is not reattached but submitted again |
| `test_batch_runs_record_each_task_and_reattach` | Each task of an interrupted batch has a record and a link; the rerun reattaches without starting a new batch |
| `test_links_of_earlier_runs_are_filled_in_from_the_task_list` | Results saved without a record get their task id and link from the account's task list (matched by fingerprint), once |
| `test_supermodes_reproduce_the_fdtd_critical_gap` (slow) | Coupled-mode theory from the real supermodes gives the 0.7 × 1.0 µm, R = 107.5 µm critical gap within 10 nm of 3D FDTD (469.6 nm) and γ within 5 % |
| `test_supermodes_are_true_even_and_odd_modes_at_a_wide_gap` (slow) | At a 0.9 µm gap the supermodes are mirror-even and mirror-odd (not localised in one core, as seen once on Windows), straddle the single-core n_eff, and split less than at 0.6 µm |
| `test_mode_profiles_tell_te_from_tm` (slow) | The real solve finds TE00 above TM00, n_g = 2.223, and the bent mode shifted outwards by less than 50 nm |
| `test_a_stage_form_accepts_with_enter_and_edits_by_number` | A stage table: a row number edits that value, Enter accepts all, values are saved; a second time nothing is asked |
| `test_q_stops_a_form_and_the_question_mark_explains` | `?` lists every setting with its explanation; `q` stops |
| `test_credit_commands_set_limit_reset_and_list_spending` | Spending is grouped by stage and date with the earlier total as its own row; `--budget` saves the limit; `--reset-spent` starts at 0; a negative budget is refused |
| `test_settings_added_later_keep_old_results_and_fingerprints` | A design whose gap already ran keeps the old recess and band points (and says so); a real change gives a new fingerprint; a new design uses the new defaults |
| `test_gap_tolerance_moves_the_coupling_ratio_the_right_way` | Wider waveguides (smaller gap) over-couple, narrower under-couple; K = 1 and K = 1.5 at the design gaps; loaded Q = Q_i / (1 + K) |
| `test_mode_family_crossings_follow_from_the_fsr_difference` | TE00 is first; TM00 crosses it every FSR₀₀ / |FSR₀₀ − FSR_TM| modes |
| `test_loss_budget_adds_up` | Two grating couplers at 31.3 % give 2 × 5.04 dB; the total is the sum of the items |
| `test_power_budget_of_the_reverse_run_gives_directionality_and_overlap` | From up/down/past flux and the reflected mode: directionality = up / (up + down), overlap = efficiency / up, reflection in dB |
| `test_lle_inputs_are_written_in_pylle_format` | `lle_inputs.json` has γ, Q_c, Q_L; `pyLLE_dispersion.csv` has no header, increasing mode numbers and frequencies; the FSR matches c / (2πR n_g) of the input dispersion to 0.1 % |
| `test_lle_export_waits_for_the_ring_stage` | Without the ring stage nothing is written |
| `test_figures_are_written_in_every_format_at_the_chosen_resolution` | PNG, PDF and SVG at the chosen dpi; an unknown format is refused |
| `test_a_figure_open_elsewhere_is_saved_beside_it` | A figure locked by another program is saved as `<name>_new.png` with a note; the old file and no temporary file are left |
| `test_an_unreachable_account_does_not_hang` | Reading the account gives up after its timeout |
| `test_budget_with_only_the_folder_keeps_the_saved_settings` | `--out <run> --budget 20` changes only the limit in that run's `design.toml`; without a design the command is refused |

## `test_phase12b.py`: taper, apodised grating, reflections, figures and run pictures (36 tests, Phase 12B)

| Test | Checks |
|---|---|
| `test_asymmetric_slab_reduces_to_the_symmetric_one` | The slab solver satisfies tan(kd/2) = g/k for equal claddings; air on top lowers n_eff |
| `test_lateral_overlap_peaks_near_2p7_waists_and_never_exceeds_one` | η_y peaks between 2.4 and 3.0 fibre waists (above 97 %), falls for narrower and wider gratings; the lateral mode is a cosine, continuous at the wall, decaying outside |
| `test_shortest_taper_length_interpolates_between_the_swept_lengths` | The shortest accepted taper interpolates log(loss); none when no length is good enough |
| `test_taper_stage_picks_width_and_length_and_the_layout_uses_them` | With a fake EME (T = 1 − e^(−L/40 µm), converged) the stage picks 13-15 µm and the exact length for 0.05 dB (rounded up to 10 µm) from the 3D EME; the 2D model and its staircase estimate are reported; figures and tables written; the layout and the loss budget use them; the cached EME run is not repeated |
| `test_taper_off_in_layout_keeps_the_previous_values` | Without the 3D EME nothing goes to the cloud and the length comes from the 2D model; `taper.use_in_layout = false` leaves the grating width and taper length as before |
| `test_eme_simulation_is_valid_and_cached_like_any_other` | The EME taper is a valid `EMESimulation` (even-mode symmetry); it is run, cached and reloaded as `EMESimulationData`; lengths follow the scale factors |
| `test_a_new_stage_after_checks_does_not_repeat_the_checks` | Taper settings change the layout's key, not the checks' (no repeated cloud runs) |
| `test_scattering_strength_and_bloch_index_are_read_from_the_field_above` | On a synthetic radiated field with the periodic near-field ripple, α to 3 % and n_B to 2e-3, in either time convention |
| `test_uniform_gratings_reach_80_percent_and_shaping_reaches_more` | The 1D model gives the textbook 80 % for the best uniform grating (at α w ≈ 0.7) and above 95 % when shaped; no gain when every tooth scatters strongly |
| `test_teeth_scatter_weakly_first_and_strongly_later` | The tooth list starts with the weakest teeth, scatters most just past the beam centre, stops when the beam has ended; each tooth's length is its fill factor times its period |
| `test_apodise_stage_designs_checks_and_feeds_the_layout` | Scattering runs, design, 2D check (5 runs; the peak is already on the wavelength), choice of the best; periods phase-matched; the layout uses the teeth (parameters, `grating_teeth.csv`, the `.m` matrix, one GDS polygon per tooth) |
| `test_apodisation_is_not_run_when_teeth_cannot_scatter_weakly` | Strong teeth only: the model predicts no gain, and the FDTD check is not run (nothing more paid) |
| `test_apodised_teeth_build_a_valid_simulation_and_uniform_names_are_unchanged` | Explicit teeth are built where they belong; uniform gratings keep their names (and fingerprints) |
| `test_arc_end_and_bus_reflections_from_mode_amplitudes` | Reflections are \|a−\|²/\|a+\|² at the ring port and the bus input |
| `test_the_reverse_grating_run_is_a_closed_box` | The reverse run has a plane behind the source; the top and bottom planes span exactly from it to the plane past the grating |
| `test_power_budget_includes_what_is_scattered_backwards` | Scattered back = flux through the back plane minus the guided reflection; all planes add up |
| `test_k_equals_one_is_found_between_samples` | The K = 1 wavelength is interpolated in log K; none when K stays above 1 |
| `test_old_runs_get_their_figures_redrawn_once_from_saved_tables` | A run made before gets 3b and 7a redrawn from its tables, once; new runs never need it |
| `test_every_cloud_run_gets_a_geometry_picture_named_like_its_folder` | One picture per simulation in `figures/sims/`, named like its cloud folder, kept afterwards; off with `run.sim_pictures = false` |
| `test_batch_links_name_the_case_they_belong_to` | Two runs with the same name are told apart by their case in the link list |
| `test_runs_without_links_are_explained_and_get_pictures` | A run without a task record is listed as "not in the task list of this account", with the reason, and gets its picture |
| `test_apodise_stage_reads_periods_right_in_either_time_convention` | The sign convention of the phase slope is fixed by the final uniform design, so the periods are right either way |
| `test_new_stages_run_through_the_design_session` | `--only taper,apodise` runs both stages as stages 8 and 9 of 13; the apodised grating is off by default (no cloud run, uniform grating used); turned on later it runs without repeating the taper |
| `test_every_ring_builder_can_have_a_flat_box_and_keeps_its_old_geometry[...]` (3) | Full, half and partial ring: 0.1 µm gives the old simulation bit for bit, 0 a flat BOX |
| `test_graded_cells_change_the_width_by_the_same_ratio` | The EME cell boundaries keep w_(i+1)/w_i constant; the cells are short at the narrow end |
| `test_2d_taper_model_is_converged_and_falls_with_length` (slow) | The 2D taper model with 600 cells agrees with 1200 cells to 0.01 dB, its loss falls with length, and a 20-cell staircase is visibly wrong (the cause of the jumping 3D result of Phase 12B v1) |
| `test_rise_finds_a_loss_that_increases_with_length` | A falling curve has no rise; the 30-cell EME of the v1 run rises by 0.88 dB |
| `test_unconverged_eme_falls_back_to_the_2d_model` | A 3D EME whose loss jumps with length is flagged not converged; the length and the loss budget come from the 2D model |
| `test_only_one_monotonic_branch_of_fill_factors_is_used` | On the measured 1550 nm table: fill factors 0.5-0.6 (2nd order radiating up, field not two waves) are rejected; the branch is 0.65-0.908; the teeth start weak and stay on it; mixing both branches is refused; the report explains a stage with too few usable points |
| `test_the_wave_reflected_at_the_grating_end_does_not_spoil_the_fit` | A weak grating whose end reflection makes deep fringes in \|E\|²: α to 5 %, n_B to 2e-3, and the backward/forward ratio are recovered |
| `test_fit_quality_is_reported_for_a_clean_and_a_messy_field` | An extra, unrelated wave above the grating leaves more than 20 % of the field unexplained; the 2nd-order test on two measured points |
| `test_apodised_peak_is_moved_onto_the_design_wavelength` | A design whose peak is 17 nm long: round 1 finds the period scale that centres it, round 2 (3 runs) uses it, and the chosen grating peaks within 1.5 nm of 1550 nm |
| `test_run_pictures_load_eme_simulations` | `--runs` loads a saved EME simulation as `EMESimulation` and draws its picture |
| `test_the_narrowest_trench_limits_the_teeth_and_is_not_paid_for` | With `apodise.min_trench_um = 0.2`, fill factors whose trench would be narrower are not run (no cost) and not used; the teeth's trenches stay at 200 nm or wider; the limit is printed |

## `test_phase12c.py`: focusing grating, efficiency landscapes (11 tests, Phase 12C)

| Test | Checks |
|---|---|
| `test_focusing_peaks_where_the_light_cone_matches_the_tip` | The in-plane focusing efficiency stays in (0, 1], peaks above 97 % at a beam-centre radius of 8-12 µm for a 1 µm tip, further out for a 3 µm tip, and falls when the spot is too far |
| `test_tip_taper_is_free_for_the_waveguide_itself_and_short_otherwise` | No taper for a tip as wide as the waveguide; 2 → 1 µm needs a few µm below 0.05 dB |
| `test_teeth_are_phase_matched_ellipses_around_the_tip` | Every tooth edge lies on the axis where the straight grating has it and obeys n_B r − s r cos φ = Φ off the axis; the polygons stay within the sector and the fibre strip; the slab starts at the tip |
| `test_focusing_is_opt_in_and_free` | Off by default: one line, nothing run |
| `test_focusing_stage_picks_an_allowed_design_and_writes_its_figures` | The chosen design respects the light-cone limit, its 3D estimate is η_2D × focusing × tip taper, it is far shorter than the straight grating with its taper; figures 10a/10b and the landscape table are written; no cloud run |
| `test_apodised_teeth_are_curved_when_they_are_better` | With a checked, better apodised grating its teeth (and fibre position, η_2D) are curved; `focus.teeth = "uniform"` overrides |
| `test_focusing_3d_check_is_a_valid_priced_simulation` | The 3D focusing coupler passes Tidy3D's pre-upload check, has its waveguide, slab, teeth and BOX, stays below 60 M cells; its result is read |
| `test_layout_and_budget_use_the_focusing_grating` | The `*_fgc` cells have bus + 2 × (tip taper, slab, 30 curved teeth) and the `*_gc` cells keep the straight grating, both in `design.gds`; `parameters.m` has the focusing values; the budget replaces the lateral overlap and long taper by the focusing rows; the report section, check values and benchmark-pack sections 4b/4d exist; with `focus.use_in_layout = false` the straight grating is the main coupler again (the focusing cells stay) |
| `test_uniform_landscape_takes_the_best_fibre_position_per_design` | 5g groups 2D runs by (period, fill factor) with the best fibre position, leaves 3D runs out, needs 3 points; 9c draws from the apodised check |
| `test_old_runs_get_the_uniform_landscape_from_their_sweep_table` | A run made before Phase 12C gets 5g drawn from `grating_sweep.csv` at its next report (free) |
| `test_focusing_stage_runs_through_the_design_session` | `--only focus` runs it as stage 10 of 13 |

## `test_phase13.py`: credit limits, whole-device circuit model, GUI (24 tests, Phase 13)

| Test | Checks |
|---|---|
| `test_limits_are_capped_by_the_account_and_each_other` | The total limit may not exceed spent + balance; a single run may not exceed the total or the balance; no cap when the balance is unknown |
| `test_price_only_shows_estimates_and_runs_nothing_that_costs` | `--price-only` records every estimate and refuses priced steps; free (cached) steps still run |
| `test_spend_cap_counts_down_and_skips_what_does_not_fit` | `--spend-cap` falls with each accepted run and skips a run that does not fit |
| `test_command_line_has_price_only_and_spend_cap` | The command line parses (help) |
| `test_benchmark_describes_an_eme_simulation` | The benchmark pack describes an EME simulation (cells, length sweep) instead of failing on `run_time` |
| `test_efficiency_preference_keeps_the_most_efficient` | `apodise.prefer = "efficiency"`: the most efficient design, kept only when above the uniform grating (the 1550 nm case: 29.9 % < 31.3 %, not kept) |
| `test_bandwidth_preference_takes_the_wider_band_within_half_a_db` | `"bandwidth"`: the 1550 nm apodised grating (0.2 dB less, 53 against 35 nm) is kept; 1 dB less is refused; the widest acceptable design wins |
| `test_linear_ramp_falls_from_the_weakest_tooth_and_is_phase_matched` | The ramp's fill factor falls from F0 to its floor, each tooth with the phase-matched period of its fill factor |
| `test_star_product_gives_the_fabry_perot_formula` | Two reflectors through the star product: maxima η²/(1−R)², minima η²/(1+R)²; without reflection exactly η² |
| `test_ring_numbers_match_the_coupled_mode_picture` | Q_i from the round-trip loss equals `physics.q_intrinsic`; critical coupling: K = 1, Q_L = Q_i/2, zero transmission; K = 1.5: dip ((1−K)/(1+K))² |
| `test_device_stage_writes_every_resonance_the_spectrum_and_figures` | Resonances across the coupler band, both gaps, off-resonance loss = 2 × (grating, overlap, taper) + bus loss, ripple = 20 log((1+R')/(1−R')), tables and figures 11a/11b written |
| `test_device_uses_the_coupling_against_wavelength_when_measured` | With a checks-stage κ²(λ) the coupling ratio K changes across the band; without reflection data there is no ripple (noted) |
| `test_device_resonance_shape_matches_its_q` | The computed resonance's width is its linewidth (2 %) and its depth its dip (0.1 dB) |
| `test_device_is_a_stage_between_layout_and_report` | Stage order and the `[device]` settings round trip |
| `test_device_section_in_report_and_check_values` | Report section 7e and the check values (off-resonance loss, loaded Q per gap) |
| `test_device_off_needs_nothing` | `device.run = false`, or no dispersion solve: skipped, nothing fails |
| `test_gui_finds_designs_and_reads_their_results` | Design folders are found; stage status, summary, figure groups, tables and files are read |
| `test_gui_limits_follow_the_same_checks` | Limits set from the GUI are checked like the run's own table and saved in `design.toml` |
| `test_gui_command_lines` | Estimate = `--yes --price-only`; Run = `--yes --spend-cap`; stages and repeats passed on |
| `test_gui_new_design_from_an_example` | A new design from `examples/design_sin780.toml` gets its name and wavelength; an existing one is not overwritten |
| `test_gui_starts_and_stops_a_background_run` | A background run writes its log, a second start is refused, Stop ends it (Ctrl+Break on Windows, Ctrl+C elsewhere) |
| `test_gui_app_renders_a_design` | The Streamlit app runs headless (AppTest) on a saved design without errors: eight tabs, the run and limit buttons |
| `test_gui_launcher_explains_a_missing_streamlit` | Without Streamlit, `python -m tarang.gui` says how to install it |
| `test_apodise_stage_tries_the_ramp_and_keeps_the_preferred_design` | With `apodise.ramp_check` 15 ramp runs (5 rates × 3 positions), table and figure 9d; `prefer = "efficiency"` keeps the more efficient measured recipe, `"bandwidth"` the wider-band ramp; the kept spectrum is saved |

## `test_phase14a.py`: design goal, test waveguides, mask contents (25 tests, Phase 14A)

| Test | Checks |
|---|---|
| `test_each_goal_runs_its_own_stages_and_old_keys_stay` | Ring and gap belong to `comb_ring`, the test-waveguide stage to `loss_test`; settings of the other goal leave every saved stage key unchanged |
| `test_unknown_goal_is_refused` | A goal other than `comb_ring` / `loss_test` stops the run |
| `test_checks_ignore_results_of_the_other_goal` | A ring or gap result left in the folder is ignored by a `loss_test` design; the loss budget works without a ring |
| `test_run_stops_when_a_choice_is_awaited` | A stage waiting for a choice (cross-section candidates) ends the run; later stages do not start |
| `test_single_mode_ranking_for_test_waveguides` | Test waveguides need one guided TE mode, not anomalous dispersion |
| `test_cutback_precision_is_the_least_squares_slope_error` | σ_α = σ / √Σ(L − L̄)²; recommended lengths scaled up (never down) to the target precision; smallest radius below a tenth of the loss, robust to a glitch at one radius |
| `test_waveguides_stage_recommends_radius_and_lengths` | The stage's radius, lengths, spiral sizes, figure 3w, table and recommendation |
| `test_spiral_has_the_length_radius_and_no_crossings[...]` (3) | Each spiral has exactly the requested length, no bend tighter than the radius anywhere (three-point curvature), leads along x at both ends, and no two parts of the path closer than the pitch |
| `test_coupler_setting_decides_what_is_drawn[...]` (5) | `layout.coupler` = auto, uniform, apodised, focusing, all: the cells drawn and the main coupler |
| `test_missing_coupler_kinds_fall_back_with_a_note` | A coupler that was not designed falls back to the straight grating, with a note; an unknown setting means auto |
| `test_ring_mask_has_reference_waveguides_and_gap_variants` | Gap variants around both gaps; ring and reference cells for every coupler kind; apodised cells have 18 teeth, uniform 30; the top cell places every variant; `layout.reference = false` leaves the reference out |
| `test_loss_test_mask_has_spirals_with_couplers` | A test-waveguide mask: reference and spirals with couplers, no ring cells, exact spiral lengths in `spirals.csv`, layout figure |
| `test_parameters_files_hold_the_new_lists` | The coupler list, the cutback lengths and the main coupler in `parameters.m` / `.json` |
| `test_circuit_uses_the_apodised_gratings_own_reflection` | The whole-device ripple uses the apodised grating's reflection when its teeth are used, else the uniform grating's (with a note) |
| `test_reverse_run_of_explicit_teeth_covers_the_whole_grating` | The reverse run's "past" plane lies beyond the last explicit tooth |
| `test_apodise_stage_measures_the_kept_gratings_reflection` | The kept apodised grating is run backwards (mode source, its own teeth); off with `apodise.reflection_check = false` |
| `test_expected_cutback_follows_the_loss_and_the_couplers` | Expected transmission = 2 × coupler − α L for every structure, slope −α, ripple from the couplers' reflection, shorter ripple period for longer structures |
| `test_report_has_the_test_waveguide_sections` | Report sections 3w and 7e for test waveguides |
| `test_bend_loss_follows_the_waveguide_mode_not_a_substrate_mode` (slow) | Real local solves of the 700 × 1000 nm guide: bend loss tens of dB/cm at 5 µm, falling more than tenfold by 10 µm, below the solver's floor by 20 µm (the bent TE00 tracked by overlap, not the substrate whispering-gallery modes) |

## `test_phase14b.py`: edge coupler, chip-edge layout, facet cavity (28 tests, Phase 14B)

| Test | Checks |
|---|---|
| `test_coupling_choice_picks_the_stages_and_keeps_old_keys` | `run.coupling = "grating"` keeps the grating, taper, apodise and focus stages and every earlier stage key; `"edge"` runs the edge stage instead (just before layout) |
| `test_unknown_coupling_is_refused` | A coupling other than grating / edge stops the run |
| `test_edge_settings_round_trip_through_the_toml` | The `[edge]` settings (shapes, facet width, layers) are saved and read back |
| `test_gaussian_is_normalised_and_overlaps_match_the_closed_forms` | The fibre Gaussian carries unit power; the overlap of two Gaussians and of a shifted one equal their closed forms; the vector overlap of a plane wave is 1; Fresnel reflection ((n−1)/(n+1))² |
| `test_facet_width_is_chosen_for_the_average_over_the_spot_sizes` | The facet width with the best mean coupling over MFD ± tolerance (4.75 µm), rounded to 0.25 µm; the default widths |
| `test_taper_shapes_and_their_inverse[...]` (3) | Linear, parabolic and power-law widths run from the waveguide to the facet, rise monotonically and invert exactly; cell boundaries and outline |
| `test_linear_cells_are_unchanged` | The linear taper's EME cells are those of Phase 12B (saved results stay valid) |
| `test_smoothed_2d_taper_model_falls_with_length` | With sub-pixel smoothing the 2D taper loss falls with length and stays below 0.005 dB when long; without it, grid steps make a spurious rise |
| `test_facet_sim_runs_the_waveguide_through_the_boundary` | The 3D facet simulation: waveguide and BOX through the absorbing boundary, port, back, out, fibre-plane (11 wavelengths) and side monitors, 101 port wavelengths, symmetry, fewer than 5 million cells |
| `test_facet_analysis_finds_the_spot_and_its_tolerances` | From a synthetic emitted Gaussian: coupling, best fibre height, working distance, reflection, transmitted power, other spot sizes and the offset losses against the Gaussian formula |
| `test_time_convention_is_read_from_a_diverging_field` | The sign of the phase curvature of a diverging field is found either way |
| `test_edge_stage_without_cloud_runs` | Free parts only: facet width = recommended, taper shape and length below the accepted loss, coupler = facet × taper, figures 12a/12b, cached solves; price-only marks the stage skipped and keeps it off the mask |
| `test_edge_stage_with_its_cloud_runs_writes_every_table` | With the EME and the 3D run answered from saved runs: EME per shape, 3D coupling and reflection, the spot-and-offset table (the 14B v1 crash), figure 12c, the coupler and reflection models used downstream |
| `test_csv_tables_take_rows_of_different_kinds` | A CSV from rows with different fields takes the union of the fields |
| `test_edge_coupler_model_and_budget` | Coupler model (butt estimate × taper), its length, the facet's Fresnel reflection, the bus length between the facets, the loss budget |
| `test_edge_mask_runs_to_the_chip_edge` | Edge cells only, waveguides from facet to facet with the overhang, chip outline and dicing lanes on their own layers, devices at least the pitch apart, one top cell; a chip too small is reported |
| `test_edge_spirals_have_the_facet_to_facet_length` | Cutback spirals have the requested facet-to-facet length; a length shorter than the chip is refused |
| `test_edge_cutback_and_ring_device` | Edge cutback lengths, the facet Fabry–Perot ripple 20 log((1+R)/(1−R)), the Regener–Sohler note |
| `test_report_has_the_edge_section_and_no_grating_parts` | Report section 5 (edge coupler), no grating limits or checks |
| `test_budget_uses_the_3d_focusing_result_without_double_counting` | The focusing grating's 3D FDTD value is used alone (no extra tip-taper or in-plane factor) |
| `test_focusing_3d_spectrum_is_not_multiplied_by_the_tip_taper` | The same in the whole-device coupler spectrum |
| `test_coupler_spectrum_is_interpolated_in_log` | Coupler efficiency between simulated wavelengths is interpolated in dB |
| `test_layout_figure_shows_the_critical_ring` | The layout figure shows the critical-gap ring cell |
| `test_loss_test_report_has_no_ring_checks` | A test-waveguide report has no gap, mode-family or FSR checks, and no ring limits |
| `test_bend_loss_below_the_floor_is_printed_as_a_bound` | Bend loss below the solver's floor prints as "< 0.001"; the CSV keeps the number |
| `test_facet_mode_with_the_si_handle_is_the_core_mode` (slow) | Real local solve of the 4.75 µm facet with the Si handle: the core TE00 (n_eff 1.83–1.85, confinement > 0.8), H/E ratio = n_eff/η0, overlap 40–46 % |

## `test_phase14c.py`: the guided GUI, three fixes from the 14B run (22 tests, Phase 14C)

| Test | Checks |
|---|---|
| `test_steps_follow_the_goal_and_the_coupling[...]` (5) | The steps of every goal × coupling (and with apodised / focusing gratings on) |
| `test_every_step_has_its_stages_settings_and_a_drawing` | Every stage belongs to exactly one step; every drawing is valid SVG; every form field has a label and a kind |
| `test_form_values_are_parsed_in_the_settings_own_type` | Lists, optional numbers, choices ("(recommended)" = empty), integers |
| `test_saving_settings_checks_them_and_a_new_preset_brings_its_losses` | A wrong value is refused and nothing changes; a new loss preset clears the old preset's losses |
| `test_step_status_and_the_cross_section_choice` | "Waiting for your choice" with the candidates; choosing one saves H and W and clears the stop; later steps list the missing earlier ones |
| `test_recommendation_is_shown_and_can_be_taken` | Recommended vs in use; "Use the recommended values" writes the cutback lengths and radius; information-only results are not settable |
| `test_a_step_runs_its_own_stages_and_reads_its_price` | Estimate = `--only <stage> --price-only`; Run = `--spend-cap` and `--redo`; the total read from the output |
| `test_report_html_holds_the_figures_and_tables` | The HTML report: headings, bold, code, tables, lists, embedded figures; the zip holds the report, settings and figures, not the cloud data |
| `test_gds_preview_is_drawn_and_reused` | The GDS picture is drawn and not redrawn while the GDS is unchanged |
| `test_drawings_embed_as_images` | Every drawing is SVG and embeds as an image |
| `test_guided_view_lists_the_steps_and_opens_the_first_to_do` | The guided view (AppTest) lists the steps and opens one |
| `test_a_new_design_starts_at_the_two_choices` | A new design opens at step 1 until its choices are saved |
| `test_guided_start_step_saves_the_two_choices` | Step 1 saves goal and coupling, opens step 2 and the steps follow the choice |
| `test_guided_cross_section_step_offers_the_candidates` | The cross-section step offers the candidates and the check button |
| `test_guided_report_step_offers_the_downloads` | The report step offers the downloads |
| `test_ring_phase_continues_past_the_end_resonances` | The ring phase beyond the first and last resonances follows the local FSR (figure 11a band ends) |
| `test_edge_headline_value_is_at_the_design_wavelength` | The edge coupling is the spectrum's value at the design wavelength; the spot/offset maps state their own wavelength |
| `test_device_length_wording_follows_the_coupling` | "from facet to facet" for edge, "between the gratings" for gratings |

## `test_phase14d.py`: plausible values, defaults, papers, Tidy3D jobs, Exit, guided paths, dispersion-map suite (16 tests, Phase 14D)

| Test | Checks |
|---|---|
| `test_a_film_in_nanometres_is_caught_with_the_value_meant` | A film of 700 µm is refused with "did you mean 0.7 µm (700 nm)"; lists are checked element by element; a plausible design has no problem |
| `test_a_design_run_stops_before_any_stage_on_such_a_value` | `python -m tarang.design` exits with code 2 and a message (no traceback) before the first stage |
| `test_the_gui_refuses_it_and_takes_the_right_one` | `save_fields` refuses 700 and changes nothing; 0.7 is saved |
| `test_empty_settings_show_their_value` | Band half-width 0.13 λ, heights around the fixed film, widths, the preset's losses, the help text, and a done step's recommendation |
| `test_loss_presets_name_their_papers` | Each preset's drop-down label has its total loss and paper; the detail has the DOI |
| `test_jobs_are_listed_per_step_with_their_links` | `task.json` records under `cloud/` listed newest first with their links, filtered by step (checks: band and grating folders) |
| `test_a_wrong_film_is_shown_and_no_step_can_run` | The red box on a step and Run disabled |
| `test_start_step_lists_the_papers_and_checks_the_film` | Step 1: the paper in the drop-down and below it; 700 refused, the step stays open |
| `test_exit_stops_the_gui_and_asks_about_a_running_run` | Exit ends the server at once; with a run going it asks, and "Stop run and exit" stops the run first |
| `test_guided_path_walks_every_step[path_A_ring_gratings]`, `[path_B_test_waveguides_edge]` | Headless walk through every step of both paths: each opens without an error, shows done, cloud steps list their Tidy3D jobs, recommendations shown; Run again starts that step's own stages |
| `test_step_buttons_carry_no_hover_text` | The step buttons have no tooltip covering their neighbours |
| `test_benchmark_pack_has_a_comparison_sheet_and_the_versions` | `comparison.csv` holds every number of `results.csv` with empty columns for another solver; the README gives the versions and how to send numbers back |
| `test_each_step_shows_only_its_own_figures` | The materials step shows only figure 1 (its filter matched 10a, 11a and 12a before); no figure belongs to two steps |
| `test_dispersion_map_suite_is_shown_listed_and_zipped` | The dispersion-engineering figures under `results/` (design map, dashboard, the nine maps, D and modes against width and height, convergence, finalists, ring profile) are grouped for the cross-section and ring steps and the Figures tab; their tables are listed; the zip holds them but not the raw solver data |
| `test_cross_section_step_shows_the_map_while_waiting_for_the_choice` | While the cross-section waits for the choice, the map overview and the target geometries are shown; "8/8 neighbours also targets" |

## `test_phase14e.py`: choosing the cross-section after comparing candidates; figures with the others (9 tests, Phase 14E)

| Test | Checks |
|---|---|
| `test_compare_setting_round_trips_and_does_not_change_the_stage_keys` | `cross_section.compare` is saved and read back, and changes no stage's settings key (nothing reruns) |
| `test_compare_radius_follows_the_goal` | Bend radius of the comparison: the ring radius for the FSR (107.5 µm at 200 GHz) or the one set; for test waveguides the spiral radius set, else 20 µm at 1550 nm |
| `test_candidates_are_compared_side_by_side` | The comparison runs the tolerance sweep, the bend sweep, the report (to `figures/2_candidates`) and the broadband profiles (raw data in `results/`); its tables are moved to `tables/`; the numbers per candidate; mode-profile figures |
| `test_chosen_cross_section_gets_its_bend_and_its_tables` | The chosen cross-section's finalist report includes its bend; figures to `figures/2_finalists` and `figures/2_mesh_convergence`, tables to `tables/` |
| `test_map_tools_write_their_plots_where_they_are_asked` | `dispersion_mapping.maps --plots-dir` writes the design map and dashboard there and nothing next to the data |
| `test_suite_reads_the_new_folders_and_the_old_ones` | The GUI finds the map in `figures/2_dispersion_map` (else an older `results/plots_map_*`), and the candidates, finalists, convergence and profile figures in their new folders |
| `test_compare_choice_is_saved_for_the_next_run` | Ticked candidates are saved as `cross_section.compare` and the cross-section is emptied |
| `test_choice_lists_every_candidate_and_compares_the_ticked_ones` | All 10 candidates in the choice (not only 8); the top four pre-ticked; Compare saves them and reruns only the cross-section, which stops again for the choice |
| `test_choice_shows_the_comparison` | The comparison table and figures in the choice; compared candidates marked; not shown twice |

## `test_phase14f.py`: the v0.1.0 wrap-up audit (14 tests, Phase 14F)

| Test | Checks |
|---|---|
| `test_a_value_of_the_wrong_kind_is_refused_with_its_setting` (6 cases) | A word or `true` for a length, `2.5` workers, `"yes"` for a switch, a single value for a list, a number for text: refused when `design.toml` is read, naming the setting and what it expects |
| `test_values_of_a_close_kind_are_taken` | `wavelength_um = 1` is read as 1.0, `workers = 3.0` as 3 |
| `test_the_shipped_examples_and_a_saved_default_still_load` | `examples/*.toml` load; a saved default reads back equal |
| `test_design_run_stops_with_exit_code_2` | `python -m tarang.design` with such a file prints the message and returns 2 (no traceback) |
| `test_report_page_embeds_only_pictures_of_its_own_folder` | The HTML report embeds `figures/in.png` but not `../outside.png` |
| `test_a_reused_process_number_is_not_a_running_job` | A saved job whose process started at another time (pid reused) is not running; older job files without a time are still checked by pid |
| `test_choice_panel_says_what_is_running_and_since_when` | "Comparing 2 candidate(s) … since 14:05", "Checking 0.7 × 1 µm …"; nothing when no run |
| `test_name_hashes_are_unchanged_by_marking_them_not_for_security` | `usedforsecurity=False` gives the same MD5 (same file names as before) |
| `test_map_goes_on_one_at_a_time_when_a_worker_is_killed` | When a mode-solver worker is killed (out of memory), the map says so and finishes the remaining points with one worker; each point is saved once |

## `test_phase15a.py`: figures for papers and the mesh band of D (12 tests, Phase 15A)

| Test | Checks |
|---|---|
| `test_figure_is_saved_with_its_numbers_and_labels` | A two-panel figure (line, axhline, axvline, error bars, bars, coloured points, a map with a colour bar) gives `_data.csv` with every series and kind, the error-bar ends, the colour values; `_meta.json` with two panels (the colour bar is not one), labels, title, versions, the caller's settings; the map in `_data.npz` |
| `test_numbers_can_be_switched_off` | With `figure_data` off only the figure is written |
| `test_paper_style_is_journal_sized_and_keeps_the_screen_figures` | Paper style: 8 pt text, 3.4 in wide for one column of panels, 7 in for three; written to `figures_paper/` (the screen figures stay); back to 10 pt in screen style |
| `test_settings_reach_the_tools_and_are_checked` | Formats, dpi, style and data travel as environment variables; unknown style or format refused |
| `test_design_settings_round_trip_and_leave_stage_keys_alone` | `run.figure_style` and `run.figure_data` are saved and read back and change no stage key |
| `test_map_tool_writes_pdf_and_numbers_when_asked` | `dispersion_mapping.maps` started with the design's settings writes `design_map.png`, `.pdf`, `_data.csv`, `_meta.json` |
| `test_grid_can_be_snapped_to_the_core_edges` | With `snap`, grid lines pass through the four core edges at 50 and 70 steps; without it the top edge falls inside a cell |
| `test_landscape_legend_keys_are_neutral` | The landscape legend shows a grey dot for the points and a red X for the chosen design (not a colour of the colour map) |
| `test_mesh_band_is_the_median_and_the_largest_distance` | The measured 700 × 1000 nm ladders: snapped 373.9 ± 3.0, unsnapped ± 12.0 ps/(nm·km) |
| `test_convergence_plot_shows_the_band_and_writes_its_table` | The mesh-check tool writes `mesh_band.csv` (median, band, snapped) and draws the band, named in the legend |
| `test_snap_and_ladder_settings` | `snap_grid` on and the 50–90 ladder by default; the ladder leaves the stage key alone, snapping changes it; the tools get `--snap` |
| `test_paper_style_moves_titles_to_the_caption_and_keeps_legends_readable` | Paper style: the figure and panel titles go to the JSON's caption, dense in-cell annotations are hidden, a 4-column legend under a panel becomes 2 columns, every text is 6–8.5 pt |

## `test_phase15b.py`: the simulation figures of a comb paper and the TM00 pump (16 tests, Phase 15B)

| Test | Checks |
|---|---|
| `test_dispersive_waves_are_the_zeros_of_dint_away_from_the_pump` | D_int = 500 μ² − 0.5 μ⁴ (MHz) has its dispersive waves at μ = ±√1000; none for a purely anomalous D_int, none within the pump region (\|μ\| < 3) |
| `test_thermal_shift_is_the_power_weighted_dn_dT_over_n_g` | dλ/dT = (λ/n_g)(Γ dn/dT_core + rest × dn/dT_BOX): 13.7 to 15.0 pm/K for Γ = 0.8 at 1550 nm |
| `test_thermo_optic_values_are_in_the_assumptions_with_their_source` | Si₃N₄ 2.45e-5 /K and SiO₂ 0.95e-5 /K in the assumptions table, citing Arbabi and Goddard |
| `test_gamma_over_the_band` | γ = 2π n₂ / (λ A_eff) at each guided wavelength; unguided points left out; nothing drawn without A_eff |
| `test_te_tm_anomalous_map` | With the measured 780 nm values: 350 nm film TE-only anomalous at W 0.5–0.6 µm, normal at 0.7; 400 nm film both anomalous |
| `test_pylle_inputs_come_from_the_design` | The pyLLE script takes R, Q_i, Q_c, γ and the dispersion file from `lle/`; 4× the over-coupled threshold by default, or `--power-mw`; detuning from +2 to −12 half-linewidths; ±200 modes |
| `test_pylle_script_dry_run_and_missing_inputs` | `--dry-run` prints the inputs and returns 0; a design without `lle/` says to run the layout step first |
| `test_soliton_step_is_on_the_plateau_after_the_chaotic_maximum` | The step shown by default lies on the single-soliton plateau of a rise / chaos / plateau / loss power trace |
| `test_pump_fields_name_the_records_fields_of_each_mode` | `core.pump(r, mode)` gives D, n_g, n_eff, guided count of TE00 or TM00 under neutral names; an unknown mode is refused |
| `test_tm00_targets_are_ranked_by_their_own_rule` | With TM00: anomalous D_TM0 and one guided TM mode make a target (TE-multimode allowed); candidates carry D_TM0 / ng_TM0; the split is to the nearest guided TE mode |
| `test_map_tool_judges_targets_by_the_pump_mode` | `maps --pump TM00` lists the TM00 targets with D_TM0 columns |
| `test_ring_gap_fdtd_launches_tm00_only_for_tm_designs` | The partial-ring FDTD launches and records TM00 (4 modes sorted by TM fraction) with `mode_pol="tm"`; TE keeps the one-mode spec (same simulations, same cached results) |
| `test_pump_mode_setting_checks_and_keeps_old_keys` | TE00 by default; TM00 changes the cross-section key and asks the gap for TM; an unknown mode is flagged |
| `test_tm00_design_stops_before_the_te_only_steps` | A TM00 design stops before the grating step with exit code 2 and says which steps to run |
| `test_gui_candidate_columns_follow_the_pump_mode` | The GUI's candidate table shows D_TM0 / ng_TM0 for TM00 designs; the pump mode is a choice in the cross-section step |
| `test_broadband_records_of_either_mode_are_read_by_their_own_fields` | Broadband records of TE00 (old and new) and TM00 are read by their own fields |

## `test_phase15d.py`: the lab's look, the web page and your measured values (14 tests, Phase 15D)

| Test | Checks |
|---|---|
| `test_theme_flags_keep_only_options_streamlit_knows` | Only the `--theme.*` options the installed Streamlit knows are passed; dark base, violet primary, Space Grotesk with a sans-serif fallback |
| `test_every_theme_option_exists_in_the_installed_streamlit` | Every option of the lab theme is known to the Streamlit in use |
| `test_css_brand_label_and_pills` | The style block holds the lab's colours and fonts with system fallbacks; the brand, labels and pills are HTML with escaped text; outside a themed app a heading is plain bold Markdown |
| `test_launcher_passes_the_lab_theme_or_the_plain_look` | `python -m tarang.gui` starts Streamlit with the lab theme; `--plain` with the light look |
| `test_web_page_is_self_contained_and_its_pictures_exist` | `docs/web/index.html` has no script and no external picture, every picture exists, the four sections are there, the assets stay under 3 MB |
| `test_measured_section_round_trips_and_is_checked` | `[measured]` is saved and read back; Q_i and a cutback loss together, a Q below 1e3 and an unknown key are flagged |
| `test_loss_from_q_is_the_inverse_of_q_intrinsic` | α from Q_i and Q_i from α agree; Q_i = 10⁶ at 1550 nm and n_g 2.22 is 0.39 dB/cm |
| `test_measured_q_holds_the_bend_loss_and_cutback_does_not` | A measured Q_i gives straight loss + bend = α(Q_i); a cutback loss is the straight loss itself; the source defaults to "your measurement" |
| `test_only_measured_losses_repeat_the_ring_step` | The source, the coupler loss and the measured keys leave the ring's stage key alone; a measured Q changes it; the cross-section key is untouched |
| `test_assumptions_table_marks_measured_values` | A cutback loss replaces the scattering row; measured film thickness and fibre MFD say "measured: <source>"; the coupler loss is added; unmeasured rows keep their publication |
| `test_report_shows_the_measured_loss_and_coupler_loss` | The report's ring table names the measured loss and its source; the assumptions table lists the Q; the measured coupler loss follows the loss budget |
| `test_measured_values_page_saves_them_in_design_toml` | The GUI page saves the source, Q_i, coupler loss and a ticked BOX thickness into `design.toml` |
| `test_measured_values_page_refuses_a_missing_or_wrong_value` | A chosen cutback without a value, or text that is not a number, is refused and nothing is saved |
| `test_lab_theme_is_applied_when_streamlit_runs_with_it` | With the lab theme, the GUI writes the style block, the gradient brand and the mono step labels |
