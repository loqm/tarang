"""Configuration-based cache directories and sweep fit statistics.

Exposes: cache_dir(), r_squared().
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from .json_helpers import snapshot


def cache_dir(
    output_dir,
    obj,
):
    # The supplied wrapper cache names omit mesh/absorber/source settings.
    # Namespace their native HDF5 cache using the complete generated object.
    """Derive a cache directory from the complete wrapper configuration.
    Hash the serialized settings to distinguish geometry and solver changes.
    """
    signature = json.dumps(
        snapshot(obj),
        sort_keys=True,
        allow_nan=False,
    )
    digest = hashlib.sha256(signature.encode()).hexdigest()[:24]
    return Path(output_dir) / "raw_cache" / digest


def r_squared(
    observed,
    predicted,
):
    """Compute the coefficient of determination for a fitted prediction.
    Return None when the observed data have no variance.
    """
    ss_total = float(np.sum((observed - np.mean(observed)) ** 2))
    return (
        None
        if ss_total == 0
        else float(1 - np.sum((observed - predicted) ** 2) / ss_total)
    )
