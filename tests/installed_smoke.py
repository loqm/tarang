"""Smoke test for an INSTALLED TARANG (wheel or pip install), run from outside the repo.

    cd <any folder that is not the repository>
    python -I <repo>/tests/installed_smoke.py

`-I` stops Python from importing the repository's folders by accident, so this checks
exactly what `pip install` put into site-packages: the code, the material data and the
declared dependencies. It needs no network and spends no FlexCredits. The GitHub
"package" job runs it on a freshly built wheel. (Not collected by pytest: the file name
does not start with test_.)
"""
import importlib
import json
import os
import pkgutil
import site
import sys
import tempfile
from importlib import resources

PACKAGES = ["tarang"]                     # every TARANG module lives inside tarang/
OWN = {"tarang"}


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{(': ' + detail) if detail else ''}")
    if not ok:
        sys.exit(1)


def _missing_module(exc):
    """Name of the module behind an ImportError (pandas wraps the real one)."""
    while exc is not None:
        if isinstance(exc, ModuleNotFoundError) and exc.name:
            return exc.name.split(".")[0]
        exc = exc.__cause__ or exc.__context__
    return None


def _explain_missing(exc):
    """A dependency that pip sees but this run cannot: usually one installed only in
    the user's personal folder (pip install --user), which `python -I` ignores and
    which nobody else using the environment has."""
    name = _missing_module(exc)
    print(f"  FAIL a dependency cannot be imported: {name or exc}")
    user = site.getusersitepackages()
    if name and os.path.isdir(user):
        for info in sorted(os.listdir(user)):
            if not info.endswith(".dist-info"):
                continue
            try:
                with open(os.path.join(user, info, "RECORD"), encoding="utf-8") as f:
                    owns = any(line.startswith((f"{name}/", f"{name}.py")) for line in f)
            except OSError:
                continue
            if owns:
                dist = info[: -len(".dist-info")]
                print(f"       '{name}' is installed only in your user folder\n"
                      f"         {user}\n"
                      f"       and not in this environment. Install it into the environment:\n"
                      f"         python -s -m pip install {dist.replace('-', '==', 1)}")
                break
    sys.exit(1)


def main():
    os.environ.setdefault("TARANG_CACHE_DIR", tempfile.mkdtemp(prefix="tarang-smoke-"))
    from importlib import metadata
    import tarang
    print(f"TARANG {tarang.__version__} on Python {sys.version.split()[0]}")
    try:
        installed = metadata.version("tarang-photonics")
    except metadata.PackageNotFoundError:
        installed = None
    check("tarang-photonics is installed", installed == tarang.__version__,
          f"pip metadata {installed}, code {tarang.__version__}")
    here = os.path.dirname(os.path.dirname(os.path.abspath(tarang.__file__)))
    check("not imported from the current folder", os.path.abspath(os.getcwd()) != here,
          f"code in {here}")

    # 1. every module imports with the installed dependencies
    failed = []
    for top in PACKAGES:
        try:
            pkg = importlib.import_module(top)
        except ImportError as exc:
            _explain_missing(exc)
        for info in pkgutil.walk_packages(pkg.__path__, top + "."):
            try:
                importlib.import_module(info.name)
            except ImportError as exc:
                if _missing_module(exc) not in OWN | {None}:
                    _explain_missing(exc)                 # a third-party dependency
                failed.append(f"{info.name} ({type(exc).__name__}: {exc})")
            except Exception as exc:                      # report all, not just the first
                failed.append(f"{info.name} ({type(exc).__name__}: {exc})")
    check("all modules import", not failed, "; ".join(failed))

    # 2. packaged material data: every material loads and fits
    from tarang.media.medium_helper import get_medium
    database = json.loads((resources.files("tarang.media") / "media.json").read_text(encoding="utf-8"))
    freq = 2.99792458e14 / 1.55                           # 1.55 um, in Hz
    for key in database:
        n = float(get_medium(key, wl_range=(1.2, 2.0)).nk_model(freq)[0])
        check(f"material {key}", 1.0 < n < 4.0, f"n(1.55 um) = {n:.4f}")

    # 3. a device simulation builds and passes Tidy3D's server-side checks
    import tarang.geometries.ridge_waveguide as ridge
    from tarang.utilities.cloud_run import sim_fingerprint
    sim = ridge.get_active_simulation(
        width=0.9, height=0.7, length=3.0, wavelength_um=1.55,
        core_medium=get_medium("si3n4", wl_range=(1.2, 2.0)),
        box_medium=get_medium("sio2", wl_range=(1.2, 2.0)),
        fine_steps_per_wvl=12, coarse_steps_per_wvl=8)
    sim.validate_pre_upload()
    check("ridge waveguide simulation validates", True, sim_fingerprint(sim)[:12])

    # 4. nested sampling works without MultiNest
    from tarang.utilities.nested_backends import available_backends
    check("dynesty backend available", "dynesty" in available_backends(), str(available_backends()))

    # 5. the migration tool for pre-layout scripts is installed and works
    from tarang.migrate import migrate_source
    new = migrate_source("from media import get_medium\nimport geometries.ridge_waveguide as rw\n")
    check("python -m tarang.migrate available", new.changes == 2
          and "from tarang.media import get_medium" in new.new_source)

    # 6. the guided design tool is installed and starts
    import subprocess
    listing = subprocess.run([sys.executable, "-m", "tarang.design", "--list"],
                             capture_output=True, text=True)
    check("python -m tarang.design available", listing.returncode == 0
          and "materials" in listing.stdout, listing.stderr[-300:])

    print("INSTALLED-PACKAGE SMOKE TEST PASSED")


if __name__ == "__main__":
    try:
        main()
    except ImportError as exc:
        _explain_missing(exc)
