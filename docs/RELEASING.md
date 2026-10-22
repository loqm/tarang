# Releasing TARANG

The repository stays private until the paper is out. Publishing on PyPI makes the code
public (anyone can download the sdist), so the PyPI steps below wait for the paper and for
the decision of Esraaj Sarkar Gupta and Prof. Kumar. Until then a release is a tag and the
wheel built by CI.

## 1. Before any release

1. `main` is green on GitHub (Linux, Windows, Python 3.11 and 3.12, MultiNest, package job).
2. `tarang/__init__.py` `__version__`, `CITATION.cff` `version` and the top entry of
   `CHANGELOG.md` agree (`tests/test_docs.py` checks this). Replace `unreleased` in the
   changelog heading by the date.
3. The benchmark pack of the reference designs (1550 nm, 780 nm) is regenerated with the
   release code and the collaborators' values are entered.

## 2. Private release (now)

```bash
git checkout main
git pull
git tag -a v0.1.0 -m "TARANG 0.1.0"
git push origin v0.1.0
```

On GitHub: **Releases → Draft a new release**, choose the tag, paste the changelog section.
The wheel and sdist of the commit are on its CI run's page (**Actions → the run →
Artifacts → tarang-dist**); attach them to the release. Collaborators with access install
from the wheel: `pip install tarang_photonics-0.1.0-py3-none-any.whl`, or from the
repository: `pip install -e ".[plots,layout,gui]"`.

## 3. Public release with the paper (later)

1. Make the repository public (GitHub **Settings → General → Danger zone**), after checking
   that no keys, private data or unpublished results are in the history.
2. Check the name is free on PyPI (`tarang-photonics`), then try TestPyPI first:

   ```bash
   python -m pip install --upgrade build twine
   python -m build
   python -m twine check --strict dist/*
   python -m twine upload --repository testpypi dist/*
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple tarang-photonics
   ```

3. Upload to PyPI (an account with two-factor authentication; or a GitHub "trusted
   publisher" workflow, which needs no stored token): `python -m twine upload dist/*`.
4. Anyone can then install with `pip install tarang-photonics` (extras as in the README,
   e.g. `pip install "tarang-photonics[layout,gui]"`). Update the README's installation
   section accordingly, and add the DOI (Zenodo archive of the tag, and the paper) to
   `CITATION.cff`.

A version on PyPI can never be replaced, only followed by a new one (0.1.1, ...).
