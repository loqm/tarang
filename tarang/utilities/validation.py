"""Numeric and geometry validation shared by photonics sweeps.

Exposes: positive(), integer(), values(), check_object(), radius_value(), resolve_span().
"""

import numpy as np


def positive(
    name,
    value,
    allow_zero=False,
):
    """
    Validate and return a finite positive floating-point value.
    Accept zero only when allow_zero is enabled.
    """
    value = float(value)
    if not np.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(
            f'{name} must be finite and {"nonnegative" if allow_zero else "positive"}.'
        )
    return value


def integer(
    name,
    value,
    minimum=1,
):
    """Validate an integer count and return a native int.
    Reject booleans and fractional values; accept NumPy integer scalars.
    """
    if (
        isinstance(
            value,
            (bool, np.bool_),
        )
        or not isinstance(
            value,
            (int, np.integer),
        )
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def values(
    name,
    values,
    minimum=1,
    allow_zero=False,
):
    """
    Validate a sweep sequence for finite values and distinct samples.
    Require the requested minimum number of points.
    """
    values = [
        positive(
            name,
            v,
            allow_zero,
        )
        for v in values
    ]
    if len(values) < minimum or len(set(values)) != len(values):
        raise ValueError(
            f"{name} needs at least {minimum} distinct values; duplicates are not allowed."
        )
    return values


def check_object(
    obj,
    **expected,
):
    """Validate builder geometry against the requested sweep parameters.
    Require an active simulation containing at least one source.
    """
    for name, value in expected.items():
        if not np.isclose(
            float(
                getattr(
                    obj,
                    name,
                )
            ),
            value,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(f"Builder returned incorrect {name}: expected {value}.")
    if not obj.sim.sources:
        raise ValueError(
            "The builder must return an active resonator object with a source."
        )


def radius_value(
    name,
    values,
    radius,
    allow_zero=False,
):
    """Resolve a scalar, radius mapping, or callable and validate its value."""
    if callable(values):
        value = values(radius)
    elif hasattr(
        values,
        "keys",
    ):
        value = values[radius] if radius in values else values[str(float(radius))]
    else:
        value = values
    return positive(
        name,
        value,
        allow_zero,
    )


def resolve_span(
    span_angle_deg,
    span_angle_rad,
):
    """Resolve a partial-ring span from degrees or radians.
    Reject conflicting inputs and angles outside the supported half-circle.
    """
    if span_angle_deg is not None and span_angle_rad is not None:
        raise ValueError("Supply only span_angle_deg or span_angle_rad, not both.")
    if span_angle_rad is not None:
        span = positive(
            "span_angle_rad",
            span_angle_rad,
        )
    else:
        degrees = (
            45.0
            if span_angle_deg is None
            else positive(
                "span_angle_deg",
                span_angle_deg,
            )
        )
        span = float(np.deg2rad(degrees))
    if span > np.pi:
        raise ValueError("The partial-ring builder supports spans in (0, 180] degrees.")
    return span
