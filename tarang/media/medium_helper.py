"""
medium_helper.py

Stores the chromatic dispersion data on the list of
materials relevant to the system.

Returns td.Medium objects when called.

Author: Esraaj Sarkar Gupta
Date: June, 2026

Exposes: get_file_hash(), write_json_atomic(), get_medium(), get_medium_dispersion().
"""

from tarang import get_logger

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from pydantic import ValidationError
import json
import os
import hashlib
import uuid
from datetime import datetime

import tidy3d as td
from tidy3d.plugins.dispersion import FastDispersionFitter as Fitter

from tarang.media import format_csv

# Resolve bundled inputs independently of the application's working directory.
_RESOURCE_ROOT = Path(__file__).resolve().parent
MEDIUM_FILE_PATH = Path(
    os.environ.get(
        "TARANG_MEDIA_FILE",
        _RESOURCE_ROOT / "media.json",
    )
)
FITTED_DIR = (
    Path(
        os.environ.get(
            "TARANG_CACHE_DIR",
            Path(
                os.environ.get(
                    "XDG_CACHE_HOME",
                    Path.home() / ".cache",
                )
            )
            / "tarang",
        )
    )
    / "fitted_media"
)
MEDIA_DB = None


logger = get_logger(__name__)


def _media_database():
    """Load the material registry on first use.
    Relative CSV paths are resolved against the registry's parent directory.
    """
    global MEDIA_DB
    if MEDIA_DB is None:
        with MEDIUM_FILE_PATH.open(encoding="utf-8") as file:
            MEDIA_DB = json.load(file)
    return MEDIA_DB


def _material_path(medium_key):
    """Resolve a material CSV from the lazy registry.
    Report unknown materials before attempting any file reads or fitting.
    """
    database = _media_database()
    key = medium_key.lower()
    if key not in database:
        raise ValueError(f"Material {medium_key!r} not found in {MEDIUM_FILE_PATH}.")
    path = Path(database[key])
    return path if path.is_absolute() else MEDIUM_FILE_PATH.parent / path


# ---- Utility Functions ---- #


def get_file_hash(filepath: Path) -> str:
    """Computes a quick MD5 hash of a file to check for staleness.
    Read the input in chunks and return its hexadecimal digest.
    """
    hasher = hashlib.md5()
    with open(
        filepath,
        "rb",
    ) as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()


def write_json_atomic(
    filepath: Path,
    data: dict,
) -> None:
    """Writes JSON to a temporary file then atomically renames it to prevent
    corruption.
    """
    tmp_path = filepath.with_suffix(".tmp")
    with open(
        tmp_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=4,
        )
    tmp_path.replace(filepath)


# ---- Main Helper Function ---- #


def get_medium(
    medium_key: str,
    wl_range: tuple = None,
    tolerance: float = 1e-6,
    max_num_poles: int = 10,
    force_zero_k: bool = False,
    use_cache: bool = True,
    try_reformat: bool = True,
) -> td.Medium:
    """Fit or load a material medium using the requested bounds and tolerances.
    Infer omitted wavelength bounds from the CSV; use_cache=False forces a new fit.
    """

    # -- Case-Insensitive Air Override Gate -- #
    if medium_key.lower() == "air":
        logger.info(
            " Bypassing file system for air. Returning dispersionless vacuum medium (n=1.0)..."
        )
        return td.Medium(permittivity=1.0)

    csv_path = _material_path(medium_key)

    if not csv_path.exists():
        raise FileNotFoundError(f"Raw data CSV not found at {csv_path}!")

    # 1. Safely Parse Numeric CSV Data (Catches stacked headers/strings early).
    #    A stacked n/k export is merged in memory; the source file is never
    #    modified (audit C5). Use tarang.media.format_csv.fix_stacked_csv() to write a
    #    clean copy if you want one on disk.
    try:
        df = pd.read_csv(csv_path)
        wvl_um = pd.to_numeric(df["wl"], errors="raise").to_numpy(dtype=float)
        n_data = pd.to_numeric(df["n"], errors="raise").to_numpy(dtype=float)
        k_col = None if (force_zero_k or "k" not in df.columns) else df["k"]
        k_data = (np.zeros_like(n_data) if k_col is None
                  else pd.to_numeric(k_col, errors="raise").to_numpy(dtype=float))
    except Exception as e:
        if not try_reformat:
            raise RuntimeError(
                f"[{medium_key}] Unrecoverable numeric parsing error in {csv_path.name}: {e}"
            ) from e
        logger.warning(
            f"[{medium_key}] {csv_path.name} is not a plain wl,n,k table ({e}); "
            "trying the stacked refractiveindex.info layout (file is not modified)..."
        )
        df = format_csv.read_stacked_csv(csv_path)
        if df is None:
            raise RuntimeError(
                f"[{medium_key}] Could not read {csv_path.name} as a plain or stacked "
                "wl,n,k table. Fix the file or pass force_zero_k=True."
            ) from e
        wvl_um = df["wl"].to_numpy(dtype=float)
        n_data = df["n"].to_numpy(dtype=float)
        k_data = (np.zeros_like(n_data) if force_zero_k
                  else df["k"].fillna(0.0).to_numpy(dtype=float))
        ok = np.isfinite(wvl_um) & np.isfinite(n_data)
        wvl_um, n_data, k_data = wvl_um[ok], n_data[ok], k_data[ok]

    # 2. Resolve Dynamic Wavelength Range & Validate Coverage
    data_wl_min, data_wl_max = float(np.min(wvl_um)), float(np.max(wvl_um))

    if wl_range is None:
        wl_range = (data_wl_min, data_wl_max)
        logger.info(
            f" Auto-detected full wavelength range for {medium_key}: {wl_range}"
        )
    else:
        # Check that requested range contains at least some data points
        eps_bounds = 1e-5
        in_range_mask = (wvl_um >= wl_range[0] - eps_bounds) & (
            wvl_um <= wl_range[1] + eps_bounds
        )
        if not np.any(in_range_mask):
            raise ValueError(
                f"[{medium_key}] Requested wl_range {wl_range} μm has ZERO data points in '{csv_path.name}'. "
                f"The available dataset wavelength range is [{data_wl_min:.4f}, {data_wl_max:.4f}] μm."
            )
        # Partial coverage: the pole fit is only constrained where there is data,
        # and outside it the fitted index is an extrapolation (audit C3).
        if wl_range[0] < data_wl_min - eps_bounds or wl_range[1] > data_wl_max + eps_bounds:
            logger.warning(
                f"[{medium_key}] Requested wl_range ({wl_range[0]:.4f}, {wl_range[1]:.4f}) μm "
                f"extends beyond the data in '{csv_path.name}' "
                f"([{data_wl_min:.4f}, {data_wl_max:.4f}] μm). Outside the data the fitted "
                f"index is an extrapolation and may be unphysical; prefer a dataset that "
                f"covers the full simulation band."
            )

    # 3. Manifest and Hashing Setup
    current_csv_hash = get_file_hash(csv_path)
    manifest_path: Path = FITTED_DIR / f"{csv_path.stem}_index.json"

    manifest = {
        "fits": [],
    }
    if manifest_path.exists():
        try:
            with open(
                manifest_path,
                "r",
                encoding="utf-8",
            ) as f:
                manifest = json.load(f)
        except json.JSONDecodeError:
            logger.warning(
                f" Manifest {manifest_path.name} corrupted. Generating fresh."
            )

    # 4. Cache Validation Rules
    valid_entries = []
    rejection_reasons = []
    eps = 1e-5

    req_wl_min, req_wl_max = wl_range

    for entry in manifest.get(
        "fits",
        [],
    ):
        c_wl_min, c_wl_max = entry["wl_range"]

        if entry["csv_hash"] != current_csv_hash:
            rejection_reasons.append("csv_hash_mismatch")
            continue
        if entry["force_zero_k"] != force_zero_k:
            rejection_reasons.append("force_zero_k_mismatch")
            continue
        if entry["max_num_poles"] < max_num_poles:
            rejection_reasons.append("insufficient_max_poles")
            continue
        if entry["rms_err"] > tolerance:
            rejection_reasons.append("insufficient_accuracy")
            continue
        if not (c_wl_min <= req_wl_min + eps and c_wl_max >= req_wl_max - eps):
            rejection_reasons.append("wl_range_out_of_bounds")
            continue

        valid_entries.append(entry)

    # 5. Cache Hit Evaluation
    if use_cache and valid_entries:
        best_match = sorted(
            valid_entries,
            key=lambda x: (x["wl_range"][1] - x["wl_range"][0], x["max_num_poles"]),
        )[0]

        target_fit_path = FITTED_DIR / best_match["fit_filename"]
        if target_fit_path.exists():
            logger.info(
                f" Cache HIT for {medium_key}. Reusing {best_match['fit_filename']} "
                f"(Fit range: {best_match['wl_range']}, RMS: {best_match['rms_err']:.2e}, "
                f"Poles allowed: {best_match['max_num_poles']})."
            )
            return td.PoleResidue.from_file(str(target_fit_path))
        else:
            logger.warning(
                f" Valid manifest entry points to missing file {target_fit_path.name}. Refitting..."
            )

    else:
        if not use_cache:
            logger.info(
                f" Cache BYPASSED for {medium_key} (use_cache=False). Refitting data..."
            )
        elif not manifest.get("fits"):
            logger.info(
                f" Cache MISS for {medium_key}. No manifest found. Fitting data..."
            )
        else:
            logger.info(
                f" Cache MISS for {medium_key}. Existing entries failed validation: {list(set(rejection_reasons))}. Fitting data..."
            )

    # 6. The Inquisition (Sanity Checks)
    if np.isnan(n_data).any() or np.isnan(k_data).any():
        logger.error(f"[{medium_key}] Error! Data contains NaNs.")
    if (k_data < 0).any():
        logger.error(
            f"[{medium_key}] Error! Negative k values detected. Physical impossibilities"
        )
    if not np.all(np.diff(wvl_um) > 0):
        logger.error(
            f"[{medium_key}] Error! Wavelengths do not increase monotonically!"
        )

    # 7. Initialize Fitter
    try:
        fitter = Fitter(
            wvl_um=wvl_um,
            n_data=n_data,
            k_data=k_data,
            wvl_range=wl_range,
        )
    except ValidationError as e:
        raise RuntimeError(
            f"[{medium_key}] Dispersion fitter rejected the data in {csv_path.name}: {e}"
        ) from e

    # 8. Execute Local Fit
    logger.info(
        f" Running local fitter for {medium_key} (up to {max_num_poles} poles)..."
    )
    medium, rms_err = fitter.fit(
        tolerance_rms=tolerance,
        max_num_poles=max_num_poles,
    )

    logger.info(f" {medium_key} fit complete! RMS Error: {rms_err:.2e}")

    # 9. Save Unique Fit and Update Manifest
    unique_id = uuid.uuid4().hex[:8]
    fit_filename = f"{csv_path.stem}_{unique_id}.json"
    fit_path = FITTED_DIR / fit_filename

    FITTED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    medium.to_file(str(fit_path))
    # Return the saved copy, not the fitter's object: the fitter's poles are numpy
    # scalars and give indices that differ from the saved copy in the last digit. Every
    # later session reads the saved copy, so returning it here keeps simulation
    # fingerprints (and the result cache) the same in the fitting session and after.
    medium = td.PoleResidue.from_file(str(fit_path))

    new_entry = {
        "timestamp": datetime.now().isoformat(),
        "fit_filename": fit_filename,
        "csv_hash": current_csv_hash,
        "wl_range": wl_range,
        "max_num_poles": max_num_poles,
        "tolerance": tolerance,
        "rms_err": float(rms_err),
        "force_zero_k": force_zero_k,
    }

    manifest["fits"].append(new_entry)
    write_json_atomic(
        manifest_path,
        manifest,
    )

    logger.info(
        f" {medium_key} cached new fit to {fit_filename} and updated manifest.\n"
    )
    return medium


def get_medium_dispersion(
    medium_key: str,
    output_path: Path | str,
    wl_range: tuple = None,
    points_plotted: int = 500,
    loglog: bool = False,
    show: bool = False,
    tolerance: float = 1e-8,
    max_num_poles: int = 10,
    force_zero_k: bool = False,
    use_cache: bool = True,
    *,
    plot_dpi: int = 300,  # Resolution of displayed and saved figures (dots per inch).
) -> dict:
    """
    Fetches a Tidy3D Medium object and generates the chromatic dispersion
    graph for it.
    """
    if isinstance(
        output_path,
        str,
    ):
        output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 1. Delegate fitting, reformatting, and dynamic range resolution to get_medium FIRST
    medium: td.Medium = get_medium(
        medium_key=medium_key,
        wl_range=wl_range,
        tolerance=tolerance,
        max_num_poles=max_num_poles,
        force_zero_k=force_zero_k,
        use_cache=use_cache,
        try_reformat=True,
    )

    # 2. Safely read bounds for plotting AFTER get_medium guarantees the CSV is clean
    if wl_range is None:
        if medium_key.lower() == "air":
            raise ValueError("wl_range is required to plot dispersionless air.")
        csv_path = _material_path(medium_key)
        df = pd.read_csv(csv_path)
        wvl_um = (
            pd.to_numeric(
                df["wl"],
                errors="coerce",
            )
            .dropna()
            .to_numpy()
        )
        wl_range = (float(np.min(wvl_um)), float(np.max(wvl_um)))

    wl_linspace = np.linspace(
        wl_range[0],
        wl_range[1],
        points_plotted,
    )
    freq_linspace = td.C_0 / wl_linspace

    eps_complex = medium.eps_model(freq_linspace)
    n_complex = np.sqrt(eps_complex)

    n_arr = np.real(n_complex)

    fig, ax1 = plt.subplots(
        figsize=(10, 6),
        dpi=plot_dpi,
    )
    plot_fn = ax1.loglog if loglog else ax1.plot

    ax1.set_xlabel("Wavelength (μm)")
    ax1.set_ylabel(
        "Real Index (n)",
        color="tab:blue",
    )
    line1 = plot_fn(
        wl_linspace,
        n_arr,
        color="tab:blue",
        label="n (real)",
    )
    ax1.tick_params(
        axis="y",
        labelcolor="tab:blue",
    )

    if not force_zero_k:
        k_arr = np.imag(n_complex)
        ax2 = ax1.twinx()
        plot_fn_2 = ax2.loglog if loglog else ax2.plot
        ax2.set_ylabel(
            "Extinction Coefficient (k)",
            color="tab:red",
        )
        line2 = plot_fn_2(
            wl_linspace,
            k_arr,
            color="tab:red",
            linestyle="--",
            label="k (imag)",
        )
        ax2.tick_params(
            axis="y",
            labelcolor="tab:red",
        )

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(
            lines,
            labels,
            loc="best",
        )
    else:
        ax1.legend(loc="best")

    ax1.set_title(f"Chromatic Dispersion Profile: {medium_key}")
    ax1.grid(
        True,
        which="both" if loglog else "major",
        linestyle=":",
        alpha=0.6,
    )

    fig.tight_layout()

    plt.savefig(
        output_path,
        dpi=plot_dpi,
    )
    if show:
        plt.show()
    else:
        plt.close(fig)

    if force_zero_k:
        return dict(
            {
                "wl": wl_linspace,
                "wavelength": wl_linspace,
                "n": n_arr,
            }
        )
    else:
        return dict(
            {
                "wl": wl_linspace,
                "wavelength": wl_linspace,
                "n": n_arr,
                "k": k_arr,
            }
        )
