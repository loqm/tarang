"""Sloped (non-vertical) etched sidewalls.

From first principles
---------------------
A dry etch rarely leaves a vertical wall. The sidewall angle ``alpha`` is measured inside
the core, between the wall and the substrate plane:

* alpha = 90°: vertical wall (every simulation before v0.1.3);
* alpha < 90°: the core widens towards the bottom (the usual, "positive" taper of a dry
  etch: the mask erodes sideways while the etch goes down);
* alpha > 90°: re-entrant (undercut) wall, narrower at the bottom.

Over an etched height h each wall moves sideways by

    s = h / tan(alpha)          (s > 0 for alpha < 90°),

so a core of width w_top at the top is w_bottom = w_top + 2 s wide at the bottom. Which of
the two is "the width" depends on what sets it:

* ``"top"`` (default): the top edge follows the mask (resist) edge, which is also what a
  top-down SEM measures; the foot spreads (alpha < 90°) or is undercut (alpha > 90°). The
  designed gap of a coupler is then the gap at the top; at the bottom it is g - 2 s.
* ``"bottom"``: the bottom width is the designed one.
* ``"middle"``: the width at mid-height is the designed one (keeps the cross-section's area,
  so it isolates the effect of the slope itself).

The same rule applies to grating teeth (a tooth is a core of width "tooth length" along
the propagation direction) and to the trenches between them.
"""
from __future__ import annotations

import math

REFERENCES = ("top", "bottom", "middle")
# How sloped walls are snapped to the mode-solver grid (v0.1.4: one line per wall at
# mid-height; see tarang.waveguide.dispersion_mapping.core.sloped_snap_points). Recorded
# with sloped-wall results so results snapped the older way are not reused.
SNAP_WALLS = "mid"


def wall_shift(height: float, angle_deg: float) -> float:
    """Sideways shift s of one wall over the height (um; > 0 when the core widens downwards)."""
    a = float(angle_deg)
    if not 45.0 < a < 135.0:
        raise ValueError(f"sidewall angle {a}° outside 45-135° (90° = vertical)")
    if abs(a - 90.0) < 1e-12:
        return 0.0
    return float(height) / math.tan(math.radians(a))


def check_reference(reference: str) -> str:
    r = str(reference or "top").strip().lower()
    if r not in REFERENCES:
        raise ValueError(f"sidewall reference {reference!r}: use one of {', '.join(REFERENCES)}")
    return r


def core_widths(width: float, height: float, angle_deg: float = 90.0,
                reference: str = "top") -> tuple[float, float]:
    """(bottom width, top width) of a core whose designed width is ``width`` at the
    ``reference`` plane."""
    s = wall_shift(height, angle_deg)
    r = check_reference(reference)
    if r == "top":
        wb, wt = width + 2 * s, width
    elif r == "bottom":
        wb, wt = width, width - 2 * s
    else:
        wb, wt = width + s, width - s
    if wb <= 0 or wt <= 0:
        raise ValueError(f"sidewall angle {angle_deg}° leaves no core: width {width} um, "
                         f"height {height} um gives {wb:.4f} um at the bottom and "
                         f"{wt:.4f} um at the top")
    return float(wb), float(wt)


def is_vertical(angle_deg) -> bool:
    return angle_deg is None or abs(float(angle_deg) - 90.0) < 1e-9


def trapezoid(y0: float, width: float, z0: float, height: float, angle_deg: float = 90.0,
              reference: str = "top") -> list[tuple[float, float]]:
    """Vertices (counter-clockwise) of the core's cross-section centred on y0, from z0 to
    z0 + height."""
    wb, wt = core_widths(width, height, angle_deg, reference)
    return [(y0 - wb / 2, z0), (y0 + wb / 2, z0), (y0 + wt / 2, z0 + height),
            (y0 - wt / 2, z0 + height)]


def core_geometry(width: float, height: float, angle_deg: float = 90.0,
                  reference: str = "top", y0: float = 0.0, z0: float = 0.0):
    """Tidy3D geometry of a straight core along x: the usual Box when the wall is vertical
    (the very geometry of earlier versions), otherwise a trapezoid extruded along x."""
    import tidy3d as td
    if is_vertical(angle_deg):
        return td.Box(center=(0, y0, z0 + height / 2), size=(td.inf, width, height))
    return td.PolySlab(vertices=trapezoid(y0, width, z0, height, angle_deg, reference),
                       slab_bounds=(-1e4, 1e4), axis=0)


def tooth_vertices(x0: float, length: float, z0: float, height: float, angle_deg: float,
                   reference: str = "top", vertical_left: bool = False
                   ) -> list[tuple[float, float]]:
    """Vertices in the (x, z) plane of a grating tooth that is ``length`` long at the
    reference plane, starting at x0. ``vertical_left``: the tooth's left side is not an
    etched face (the first tooth continues the unetched waveguide) and stays vertical."""
    wb, wt = core_widths(length, height, angle_deg, reference)
    xc = x0 + length / 2
    bl, br = xc - wb / 2, xc + wb / 2
    tl, tr = xc - wt / 2, xc + wt / 2
    if vertical_left:          # joined to the waveguide on its left (which ends at x0):
        bl = tl = min(bl, tl, x0)        # no etched face, no notch between the two
    return [(bl, z0), (br, z0), (tr, z0 + height), (tl, z0 + height)]


def tooth_geometry(x0: float, length: float, z0: float, height: float, angle_deg: float,
                   reference: str = "top", y_size: float | None = None,
                   vertical_left: bool = False):
    """Tidy3D geometry of a grating tooth: a Box when vertical, else a trapezoid in the
    (x, z) plane extruded along y over ``y_size`` (None or inf: infinite, 2D)."""
    import tidy3d as td
    inf = y_size is None or not math.isfinite(float(y_size))
    if is_vertical(angle_deg):
        return td.Box(center=(x0 + length / 2, 0, z0 + height / 2),
                      size=(length, td.inf if inf else float(y_size), height))
    half = 1e4 if inf else float(y_size) / 2
    return td.PolySlab(vertices=tooth_vertices(x0, length, z0, height, angle_deg, reference,
                                               vertical_left),
                       slab_bounds=(-half, half), axis=1)


def bottom_gap(gap: float, height: float, angle_deg: float, reference: str = "top") -> float:
    """Gap between two cores at the bottom when ``gap`` is the gap at the reference plane
    (the centre-to-centre spacing is fixed by the mask)."""
    s = wall_shift(height, angle_deg)
    r = check_reference(reference)
    return gap - (2 * s if r == "top" else 0.0 if r == "bottom" else s)
