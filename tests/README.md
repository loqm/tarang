# TARANG test suite

```bash
python -m pytest                     # whole suite, ~4 min, no FlexCredits
python -m pytest -m "not slow"       # quick check, under a minute (skips 15 mode-solve / fitting tests)
python -m pytest tests/test_geometries.py -q    # one file
python -m pytest --run-cloud tests/test_cloud_smoke.py -v   # REAL cloud run, ~0.03 FlexCredits
```

Expected on Windows: `267 passed, 2 skipped`. The two skips are the real-cloud
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

## `test_docs.py`: the documentation matches the code (24 tests, Phase 6)

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
| `test_relative_links_resolve[...]` (6) | Every relative link and image points to an existing file, and every `#anchor` to an existing heading |
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

## `test_phase11.py`: fibre orientation, spectra, coupling measures, benchmark pack (19 tests, Phase 11)

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
