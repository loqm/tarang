"""Two-sided grid snapping of the cross-section mode solves (no Tidy3D import: the sweep
tools use these to match saved records without loading the solver)."""
from __future__ import annotations

#: How far the snapped grid lines sit from the core's faces in a two-sided solve (um).
#: Far below a cell (8 nm at 50 steps in SiN at 780 nm), far above rounding (1e-13 um).
SNAP_OFFSET_UM = 1e-6
#: Default number of snapped solves averaged (2 since v0.1.5; 1 reproduces older results).
SNAP_SIDES = 2


def snap_offsets(snap: bool, snap_sides: int = SNAP_SIDES) -> tuple:
    """Offsets of the snapped grid lines from the core's faces, one solve each.

    The local mode solver gives every field sample a single material (no sub-pixel
    averaging). With grid lines exactly on a face, the samples of the field component
    normal to that face sit exactly on it, and which material they get is decided by the
    last bits of floating-point coordinates: a 1e-12 um change of the width moved n_eff of
    the 780 nm core by 3e-3, and D at 50 steps came out 198.8 on Linux and 235.3 on Windows
    from the same code. With the lines just inside the core every such sample is SiN; just
    outside, cladding. The two bracket the answer with errors of opposite sign and first
    order in the cell size, and their mean is steady from mesh to mesh (780 nm core: D =
    195.5 and 196.1 at 50 and 60 steps; inside 240.4 / 233.5, outside 150.7 / 158.7).

    ``snap_sides`` 2: (-SNAP_OFFSET_UM, +SNAP_OFFSET_UM), negative = lines inside the core;
    1: (0.0,), the one-sided snapping of v0.1.0-0.1.4; no snapping: (0.0,)."""
    if not snap or int(snap_sides) == 1:
        return (0.0,)
    if int(snap_sides) != 2:
        raise ValueError(f"snap_sides must be 1 or 2, not {snap_sides!r}")
    return (-SNAP_OFFSET_UM, SNAP_OFFSET_UM)


def snap_sides_of(r: dict) -> int:
    """Snap sides of a saved record (records before v0.1.5 have none: 1)."""
    return int(r.get("snap_sides", 1)) if r.get("snap") else 1


def same_snap(r: dict, snap: bool, snap_sides: int = SNAP_SIDES) -> bool:
    """A saved record was solved with this snapping (on/off, and one- or two-sided)."""
    if bool(r.get("snap", False)) != bool(snap):
        return False
    return (not snap) or snap_sides_of(r) == int(snap_sides)
