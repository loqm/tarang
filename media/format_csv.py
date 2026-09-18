"""
materials.format_csv.py

Used to reformat csv files from refractiveindex.info that have stacked data
into a useable form. This mutates the csv file in place, so will only be called
once for any data file.

Exposes: fix_stacked_csv().
"""

from tarang import get_logger

import pandas as pd
import io
import os

logger = get_logger(__name__)


def fix_stacked_csv(file_path: str) -> bool:
    """
    Detects if a CSV file has stacked data (separated by 'wl,k'),
    reformats it into a clean merged dataframe, and overwrites the file.
    Returns True if successfully fixed, False otherwise.
    """
    if not os.path.exists(file_path):
        logger.error(
            f"[CSV Formatter] Error: File '{file_path}' does not exist. How did we get here?"
        )
        return False

    with open(
        file_path,
        "r",
    ) as file:
        raw_text = file.read()

    # Check if it actually needs fixing
    if "wl,k" not in raw_text:
        logger.error(
            f"[CSV Formatter] 'wl,k' header not found in '{file_path}'. File might already be clean."
        )
        return False

    try:
        parts = raw_text.split("wl,k")
        n_data_raw = parts[0].strip()
        k_data_raw = "wl,k\n" + parts[1].strip()

        df_n = pd.read_csv(io.StringIO(n_data_raw))
        df_k = pd.read_csv(io.StringIO(k_data_raw))

        # Merge on the wavelength column
        clean_df = pd.merge(
            df_n,
            df_k,
            on="wl",
            how="outer",
        )
        clean_df = clean_df.sort_values(by="wl").reset_index(drop=True)

        # Overwrite the broken file
        clean_df.to_csv(
            file_path,
            index=False,
        )
        logger.info(
            f"[CSV Formatter] Success! '{file_path}' has been reformatted and overwritten."
        )
        return True

    except Exception as e:
        logger.error(
            f"[CSV Formatter] Failed to reformat file: {e}.\nIf you wish to attempt to reformat this by yourself, \
keep in mind that this framework expects wavelength, n and k to form a single table.\
Better yet, if you're willing to compromise, try running with the flag `force_zer_k` set to True!"
        )

        return False
