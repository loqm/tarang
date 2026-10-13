"""JSON conversion, report logging, and atomic report persistence.

Exposes: to_jsonable(), log_json(), save_json(), snapshot().
"""

import json
from pathlib import Path

import numpy as np


def to_jsonable(value):
    """
    Convert arrays, media, and nonfinite values into JSON-compatible data.
    Preserve infinite geometry bounds as strings for strict JSON serialization.
    """
    if isinstance(
        value,
        dict,
    ):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(
        value,
        (list, tuple, np.ndarray),
    ):
        return [to_jsonable(v) for v in value]
    if isinstance(
        value,
        np.generic,
    ):
        return to_jsonable(value.item())
    if isinstance(
        value,
        float,
    ) and not np.isfinite(value):
        return str(value)  # Preserve infinite geometry bounds in strict JSON.
    if value is None or isinstance(
        value,
        (str, int, float, bool),
    ):
        return value
    if hasattr(value, "model_dump_json"):          # pydantic v2 models (Tidy3D objects)
        return to_jsonable(json.loads(value.model_dump_json()))
    if hasattr(
        value,
        "json",
    ):
        return to_jsonable(json.loads(value.json()))  # older pydantic
    return str(value)


def log_json(
    value,
    *,
    logger,
):
    """
    Log a JSON-formatted sweep report at INFO level.
    Use the supplied logger so applications control its destination.
    """
    logger.info(
        json.dumps(
            to_jsonable(value),
            indent=2,
            allow_nan=False,
        ),
    )


def save_json(
    path,
    report,
):
    """
    Write a JSON report using a temporary file and atomic replacement.
    Create the destination directory when necessary.
    """
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            to_jsonable(report),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    temporary.replace(path)


def snapshot(obj):
    """Serialize the stored attributes of a simulation wrapper.
    Use this snapshot in reports and cache signatures.
    """
    return to_jsonable(vars(obj))
