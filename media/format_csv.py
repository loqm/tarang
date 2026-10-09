"""
media.format_csv

Read refractive-index CSV files exported from refractiveindex.info, where the
n table and the k table are stacked one after the other::

    wl,n
    0.50,2.05
    ...
    wl,k
    0.50,0.0001
    ...

and merge them into one ``wl, n, k`` table.

Source data files are never modified (audit C5): ``read_stacked_csv`` parses in
memory, and ``fix_stacked_csv`` writes a corrected *copy*.

Exposes: read_stacked_csv(), fix_stacked_csv().
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from tarang import get_logger

logger = get_logger(__name__)

PathLike = Union[str, Path]


def read_stacked_csv(file_path: PathLike) -> Optional[pd.DataFrame]:
    """Parse a stacked n/k CSV into one DataFrame with columns wl, n, k.

    Returns None if the file does not contain a stacked ``wl,k`` block or cannot
    be parsed. The file itself is only read.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"[CSV Formatter] File '{file_path}' does not exist.")
        return None

    raw_text = file_path.read_text(encoding="utf-8", errors="replace")
    if "wl,k" not in raw_text:
        logger.info(f"[CSV Formatter] No stacked 'wl,k' block in '{file_path.name}'.")
        return None

    try:
        n_part, k_part = raw_text.split("wl,k", 1)
        df_n = pd.read_csv(io.StringIO(n_part.strip()))
        df_k = pd.read_csv(io.StringIO("wl,k\n" + k_part.strip()))
        merged = pd.merge(df_n, df_k, on="wl", how="outer")
        merged = merged.sort_values(by="wl").reset_index(drop=True)
        for col in ("wl", "n", "k"):
            merged[col] = pd.to_numeric(merged[col], errors="raise")
        return merged
    except Exception as exc:
        logger.error(
            f"[CSV Formatter] Could not parse stacked data in '{file_path.name}': {exc}. "
            "TARANG expects wavelength (um), n and k in one table with columns wl,n,k; "
            "alternatively call get_medium(..., force_zero_k=True) to ignore k."
        )
        return None


def fix_stacked_csv(file_path: PathLike, output_path: Optional[PathLike] = None) -> bool:
    """Write a merged ``wl,n,k`` copy of a stacked CSV. The original is untouched.

    Parameters
    ----------
    file_path : stacked CSV to read.
    output_path : where to write the merged table. Defaults to
        ``<stem>_merged.csv`` next to the source. Must differ from ``file_path``.
        An existing output file is never replaced: if its content already matches,
        nothing is written; otherwise FileExistsError is raised.

    Returns True if a merged table is available at ``output_path``.
    """
    src = Path(file_path)
    dst = Path(output_path) if output_path is not None else src.with_name(f"{src.stem}_merged.csv")
    if dst.resolve() == src.resolve():
        raise ValueError("output_path must differ from file_path; source data is never overwritten.")

    merged = read_stacked_csv(src)
    if merged is None:
        return False

    # Fixed "\n" line endings and byte-level I/O: pandas otherwise ends lines with
    # os.linesep ("\r\n" on Windows), which text-mode writing turns into "\r\r\n"
    # (a blank line after every row) and which breaks the identical-content check.
    data = merged.to_csv(index=False, lineterminator="\n").encode("utf-8")
    if dst.exists():
        if dst.read_bytes() == data:
            logger.info(f"[CSV Formatter] '{dst.name}' already holds the merged table.")
            return True
        raise FileExistsError(
            f"'{dst}' exists with different content; choose another output_path."
        )

    tmp = dst.with_name(dst.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, dst)
    logger.info(f"[CSV Formatter] Wrote merged copy '{dst}'. Original left unchanged.")
    return True
