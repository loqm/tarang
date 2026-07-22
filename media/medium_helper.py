"""
medium_helper.py

Stores the chromatic dispersion data on the list of
materials relevant to the system.

Returns td.Medium objects when called.

Author: Esraaj Sarkar Gupta
Date: June, 2026
"""

import numpy as np
import pandas as pd

from pathlib import Path
from pydantic import ValidationError
import json
import sys
import logging


import tidy3d as td
from tidy3d.plugins.dispersion import FastDispersionFitter as Fitter

from media import format_csv

# Configure the logger -- this needs to sit in the main
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log", mode="a"), # Appends to file
        logging.StreamHandler(sys.stdout)             # Prints live to terminal/console!
    ]
)

logger = logging.getLogger(__name__)

MEDIUM_FILE_PATH : Path = Path("media.json")
FITTED_DIR = Path("media/fitted_media/")
FITTED_DIR.mkdir(parents=True, exist_ok=True)

# ---- Load Media Configuration ---- #
try:
    with open (MEDIUM_FILE_PATH, "r", encoding="utf-8") as file:
        MEDIA_DB = json.load(file)       
except FileNotFoundError:
    raise FileNotFoundError(f"Failed to find media JSON file at {MEDIUM_FILE_PATH}!")

# ---- Main Helper Function ---- #
def get_medium(
        medium_key : str,
        wl_range : tuple,
        tolerance : float = 1e-8,
        force_zero_k : bool = False,
        try_reformat : bool = True
) -> td.Medium :
    """
    Docstring for get_medium
    
    :param medium_key: Description
    :type medium_key: str
    :param wl_range: Description
    :type wl_range: tuple
    :param tolerance: Description
    :type tolerance: float
    :param force_zero_k: Description
    :type force_zero_k: bool
    :param try_reformat: Description
    :type try_reformat: bool
    :return: Description
    :rtype: Medium

    Fetches a Tidy3D Medium object. Loads from JSON cache if available, 
    otherwise reads the CSV, fits the data, caches it, and returns the Medium.
    Bypasses file system entirely for standard air background requests.
    """

     # -- Case-Insensitive Air Override Gate -- #
    if medium_key.lower() == "air":
        logger.info("[Medium] Bypassing file system for air. Returning dispersionless vacuum medium (n=1.0)...")
        return td.Medium(permittivity=1.0)
    
    if medium_key.lower() not in MEDIA_DB:
        raise ValueError(f"Material '{medium_key}' not found in media data JSON file. Are you sure you spelt that correctly?")
    
    csv_path : Path = Path(MEDIA_DB[medium_key.lower()])
    
    # Check if the path listed exists
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw data CSV not found at {csv_path}!")
    
    # Generate the JSON filename dynamically from the CSV name
    json_path : Path = FITTED_DIR / f"{csv_path.stem}.json"
    
    # Load from cache if it exists
    if json_path.exists():
        logger.info(f"[Medium] Loading {medium_key} cached fit from {json_path.name}...")
        return td.PoleResidue.from_file(str(json_path))
    
    # Fit if no cache exists
    logger.info(f"[Medium] No {medium_key} cache found. Fitting data from {csv_path.name}...")
    df = pd.read_csv(csv_path)

    wvl_um = np.array(df['wl'].values)
    n_data = np.array(df['n'].values)

    # Handle zero 'k' cases automatically
    if force_zero_k or 'k' not in df.columns:
        k_data = np.zeros_like(n_data)
    else:
        k_data = np.array(df['k'].values)

    if try_reformat:
        try:    
            fitter = Fitter(
                wvl_um=wvl_um,
                n_data=n_data,
                k_data=k_data,
                wvl_range=wl_range
            )
        except ValidationError as e:
            logger.error(f"[Medium]: Error during fitter initialization. Rerunning with reformatted data.")
            
            # Format the csv file in place
            format_question_mark = format_csv.fix_stacked_csv(str(csv_path))

            # If formatting was successful
            # Prevent reformatting a second time to prevent infinite recursion
            if format_question_mark:
                return get_medium(
                    medium_key=medium_key,
                    wl_range=wl_range,
                    tolerance=tolerance,
                    force_zero_k=force_zero_k,
                    try_reformat=False
                )
            else:
                raise RuntimeError(f"[Medium]: Failed to reformat CSV for {medium_key}. You're on your own here.")
    else:
        fitter = Fitter(
                wvl_um=wvl_um,
                n_data=n_data,
                k_data=k_data,
                wvl_range=wl_range
            )

    
    medium, rms_err = fitter.fit(tolerance_rms=tolerance)
    logger.info(f"[Medium] {medium_key} fit complete! RMS Error: {rms_err:.2e}")

    medium.to_file(str(json_path))
    logger.info(f"[Medium] {medium_key} cached fit to {json_path.name}.\n")

    return medium