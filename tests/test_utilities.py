"""Unit tests for the small shared helpers in utilities/ (Phase 4). Instant, local.

    python -m pytest tests/test_utilities.py -q
"""
import json
import math
import warnings

import numpy as np
import pytest
import tidy3d as td

from utilities import json_helpers as jh
from utilities import validation as v


# ---- validation -------------------------------------------------------------- #
def test_positive():
    assert v.positive("x", 2) == 2.0 and isinstance(v.positive("x", np.float32(1.5)), float)
    assert v.positive("x", 0, allow_zero=True) == 0.0
    for bad in (0, -1, math.inf, math.nan):
        with pytest.raises(ValueError, match="x must be finite"):
            v.positive("x", bad)


def test_integer():
    assert v.integer("n", np.int64(3)) == 3
    for bad in (True, 2.0, 0):
        with pytest.raises(ValueError, match="n must be an integer"):
            v.integer("n", bad)


def test_values_rejects_duplicates_and_short_lists():
    assert v.values("w", [0.5, 0.6]) == [0.5, 0.6]
    with pytest.raises(ValueError):
        v.values("w", [0.5, 0.5])
    with pytest.raises(ValueError):
        v.values("w", [0.5], minimum=2)


# ---- json helpers ------------------------------------------------------------ #
def test_to_jsonable_handles_numpy_infinities_and_tidy3d_objects():
    out = jh.to_jsonable({"a": np.arange(3), "b": np.float64(2.5), "c": math.inf, 4: None})
    assert out == {"a": [0, 1, 2], "b": 2.5, "c": "inf", "4": None}
    with warnings.catch_warnings():
        warnings.simplefilter("error")                 # no pydantic deprecation warnings
        med = jh.to_jsonable(td.Medium(permittivity=2.0))
    assert med["permittivity"] == 2.0
    json.dumps(med, allow_nan=False)                  # strict JSON


def test_save_json_creates_folders_and_replaces_atomically(tmp_path):
    target = tmp_path / "a" / "b" / "report.json"
    jh.save_json(target, {"x": np.array([1.0, math.inf])})
    assert json.loads(target.read_text()) == {"x": [1.0, "inf"]}
    assert not target.with_suffix(".tmp").exists()
