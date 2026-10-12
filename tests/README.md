# TARANG test suite

```bash
python -m pytest tests -q          # whole suite, ~4 min, no FlexCredits
python -m pytest tests/test_nested_backends.py -q   # one file
```

Expected on Windows: `72 passed, 1 skipped`. The skipped test needs the native
MultiNest library, which is normally installed only on Linux/macOS. With MultiNest
installed, all 73 pass.

## How the tests work

A test is a function that sets up a situation, runs TARANG code and `assert`s
something that must be true. `pytest` collects every `test_*` function in this
folder. Each test is written so that it **fails if a specific bug returns**, so
these are regression tests. The audit IDs (C1, C2, ..., R1) refer to the v0.1.0
release audit.

The suite costs nothing to run:

1. **The local mode solver only.** Tidy3D's mode solver runs on your machine for
   free. No test starts a cloud FDTD run.
2. **A fake cloud.** `FakeWeb` in `test_cloud_run.py` replaces `tidy3d.web`. It
   records each call (upload, estimate, start, monitor, load, delete) and "runs"
   a task by writing an empty result file. The tests check *what TARANG would ask
   the cloud to do*, and in which order.
3. **Analytic stand-ins.** Constant-index materials (n = 1.996, 1.444, 3.476) are
   used where real dispersion fits don't matter, and Gaussian likelihoods with a
   known evidence replace FDTD in the sampler tests.

Physics tests check **relations**: orderings (TM00 above TE00), signs (D > 0),
counts (one guided TE mode) and trends. They don't check 4-decimal values,
because coarse-mesh n_eff shifts by ~0.005 between Tidy3D versions.

## What the tests don't cover

Whether a real FDTD result (for example a grating efficiency) is physically
right. That needs saved reference results and is planned for Phase 4.

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

## `test_cloud_run.py`: verified cache and cost guard (11 tests, Phase 3a)

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

