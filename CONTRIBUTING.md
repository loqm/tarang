# Contributing to TARANG

Thank you for improving TARANG. This page covers the development setup, how changes are
reviewed, the conventions the code follows, and the rules that protect simulation
results and cloud credits.

## Development setup

```bash
git clone https://github.com/loqm/tarang
cd tarang
conda env create -f environment.yml      # or: pip install -e ".[plots,notebooks,test]"
conda activate tarang
python -m pytest -m "not slow"           # about a minute
```

TARANG is installed in editable mode, so your changes take effect immediately. Make sure
all of TARANG's dependencies are inside the environment, not only in your personal Python
folder: `python -I -c "import tidy3d, pandas, tarang"` must work.

## Branches and review

- Never push to `main`. Work on a branch (`feature/<topic>`, `fix/<topic>`) and open a
  pull request into `main`.
- Keep a pull request to one topic. Large work goes in reviewable commits, each with a
  message that says what changed and why.
- Every pull request runs the test suite on Ubuntu and Windows, Python 3.11 and 3.12, plus
  a MultiNest job and a packaging job. All of them must pass before merging.
- Merge with **Create a merge commit**, so the individual commits stay visible. Do not
  squash or rebase a branch that another open pull request is built on.
- Stage files by name (`git add path/to/file`), not with `git add .`, so that local
  results, plots and editor settings stay out of commits.

## Tests

```bash
python -m pytest                          # full suite, a few minutes, no FlexCredits
python -m pytest -m "not slow"            # skips local mode solves and material fits
python -m pytest tests/test_geometries.py # one file
python -m pytest --run-cloud tests/test_cloud_smoke.py   # one real cloud job, ~0.03 FlexCredits
```

The suite never contacts Tidy3D unless you pass `--run-cloud`: a fake web API in
`tests/conftest.py` stands in for the cloud and records what TARANG would ask it to do.
[tests/README.md](tests/README.md) explains every test.

- Add a test with every bug fix that fails without the fix.
- A new geometry builder needs an entry in `tests/test_geometries.py` (Tidy3D's
  `validate_pre_upload()` must pass, and every monitor the analysis reads must exist).
- Physics tests check relations (orderings, signs, counts, trends), not 4-decimal values,
  which shift between Tidy3D versions.
- TARANG's own deprecation warnings are errors in the test suite, so internal code cannot
  use an old name.

## Conventions

**Names.** Packages and modules `lower_case`, classes `CapWords`, functions and arguments
`lower_case`. One name per physical layer, used by every device:

| Layer | Arguments |
|---|---|
| waveguide core | `core_medium` |
| buried oxide under the core | `box_medium`, `box_height` |
| cladding above the core | `top_cladding_medium`, `top_cladding_height` |
| silicon wafer under the box (grating coupler) | `handle_medium`, `handle_height` |
| wavelength | `wavelength_um` |
| mesh | `fine_steps_per_wvl`, `coarse_steps_per_wvl` |

**Units.** Lengths and wavelengths in µm, frequencies in Hz, dispersion *D* in
ps/(nm·km), losses in dB/cm. Put the unit in the name when it is not µm
(`loss_db_cm`, `fsr_ghz`).

**Renaming a public name.** Keep the old name working for one minor release with a
`DeprecationWarning`, using the helpers in `tarang/_compat.py` (`renamed_params`,
`RenamedAttributes`, `deprecated_module_names`). Moving or renaming a module is different:
add the old -> new mapping to `MOVED` in `tarang/migrate.py`, so old scripts can be
updated with `python -m tarang.migrate`. Add every rename to the upgrading table in
[CHANGELOG.md](CHANGELOG.md).

**Layout.** All code lives in the `tarang` package and imports itself as `tarang.<package>`;
examples go in `examples/`, tests in `tests/`, documentation figures in `docs/`.

**Logging.** Use `tarang.get_logger(__name__)` in library code, not `print`. Configure
logging (`tarang.configure_logging`) only in command-line tools, scripts and notebooks,
which may also print their results.

**Changelog.** Add a line under the unreleased version in `CHANGELOG.md` for every change
a user would notice.

## Results and cloud credits

- **Results are append-only.** Never overwrite or delete a cached result or a results
  file from code. New results go to new files or folders.
- **Cache keys follow the simulation.** Results are cached under a fingerprint of the
  complete Tidy3D simulation. A change to a builder that alters the simulation (a default
  dimension, mesh, monitor or material) changes the fingerprint, so every affected cached
  result runs again and costs credits. Say so in the pull request, with an estimate of
  what will rerun. Changes that leave simulations identical must leave fingerprints
  identical (`tests/test_cloud_run.py`, `tests/test_api_compat.py`).
- **Every cloud path must go through `tarang/utilities/cloud_run.py`**, so the cost estimate, the
  `max_cost` guard and the verified cache always apply. Never call `tidy3d.web.run`
  directly.
- Commit notebooks **without outputs**
  (`jupyter nbconvert --clear-output --inplace examples/*.ipynb`); `tests/test_notebooks.py`
  checks this. Figures that belong to the documentation go in `docs/assets/` and are
  generated by a script there (`docs/make_material_figures.py`,
  `docs/make_readme_figures.py`).
- Never commit results, `.hdf5` files, generated plots, notebook outputs with personal
  paths, API keys or `.env` files. `.gitignore` covers the usual locations.

## Adding a material

See [tarang/media/data/SOURCES.md](tarang/media/data/SOURCES.md#adding-a-material): add the CSV, register
the key in `tarang/media/media.json`, document the reference and range in `SOURCES.md`, and run
`tests/test_packaging.py` and `tests/test_docs.py`. Never edit an existing data file in
place: its bytes are part of the cache key.

## Adding a device

1. A builder in `tarang/geometries/` that returns a wrapper object holding the `td.Simulation`
   on `.sim`, with `get_passive_simulation` / `get_active_simulation` where both make
   sense.
2. Analysis in the device package, with cloud runs through the `CloudCachedDevice` mixin
   in `tarang/utilities/cloud_run.py`.
3. Tests in `tests/test_geometries.py`, and an example notebook in `examples/`.
4. A row in the README's capabilities table and a CHANGELOG entry.

## Secrets

Never commit API keys or `.env` files. Before a release, scan the whole history with
[gitleaks](https://github.com/gitleaks/gitleaks): `gitleaks git . --redact -v`. Findings
reviewed and confirmed harmless are listed in `.gitleaksignore` with a comment saying why.

## Line endings

`.gitattributes` stores code and text with LF line endings on every OS; Git converts on
checkout and commit as needed. Material CSVs are excluded on purpose (their exact bytes
are part of the cache key).
