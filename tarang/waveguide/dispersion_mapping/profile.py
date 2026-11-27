"""Cross-section profile: what an etch leaves around the core, beyond a plain strip.

Every option is a number that is 0 (or empty) by default, and the default leaves the
simulation exactly as before v0.2.0, byte for byte, so its fingerprint and every result
are unchanged. Lengths in µm; the core is the film of thickness ``height`` from z = 0 to
z = height on the BOX top (z = 0), as everywhere in TARANG.

Partial etch (``slab_um``)
    A layer of core material from z = 0 to z = slab_um left by an etch that stops short of
    the BOX. ``slab_width_um`` empty: the layer runs across the whole cross-section (a rib
    or ridge waveguide, etch depth height - slab_um). A number: the layer is that wide,
    centred under the core (a pedestal of core material, as in Black et al., Opt. Lett. 46,
    817 (2021)).

BOX recess with undercut (``recess_um``, ``undercut_um``, ``pedestal``)
    The BOX top etched down by ``recess_um`` everywhere except under the core, and
    ``undercut_um`` back under each core edge, so the core stands on a pedestal of width
    (core bottom width - 2 undercut). The pedestal is the BOX material unless ``pedestal``
    names another; the recess is filled by the top cladding (air when there is none).
    This is the oxide window of the Applied Nanotools air-clad process. depth and
    undercut are independent: an isotropic etch gives undercut = depth.

The two cannot be combined in one cross-section (v0.2.0).

Grid snapping: every new face gets grid lines, as the core's faces do: the slab's top face
(and edges when it is finite), the recess floor and the pedestal walls.

Exposes: Profile, apply(), from_args(), add_args(), as_args().
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Optional

#: fine-mesh region below the BOX top of the ridge builder (its buffer_z)
_FINE_BELOW_UM = 0.5


@dataclass(frozen=True)
class Profile:
    slab_um: float = 0.0
    slab_width_um: Optional[float] = None
    recess_um: float = 0.0
    undercut_um: float = 0.0
    pedestal: str = ""

    # ---- what it is ---------------------------------------------------------------------- #
    @property
    def is_default(self) -> bool:
        """True for a plain strip: the simulation is exactly as before v0.2.0."""
        return self == Profile()

    def check(self, width_bottom: float, height: float, box: float) -> None:
        """ValueError for a cross-section that cannot be built."""
        s, d, u = float(self.slab_um), float(self.recess_um), float(self.undercut_um)
        if min(s, d, u) < 0:
            raise ValueError("slab, recess and undercut must be >= 0")
        if s and not s < height:
            raise ValueError(f"slab {s:g} um must be thinner than the core film {height:g} um")
        if self.slab_width_um is not None and s and not self.slab_width_um > width_bottom:
            raise ValueError(f"slab width {self.slab_width_um:g} um must exceed the core's "
                             f"bottom width {width_bottom:g} um")
        if s and (d or u):
            raise ValueError("a partial-etch slab and a BOX recess cannot be combined (v0.2.0)")
        if u and not d:
            raise ValueError("an undercut needs a recess (recess_um > 0)")
        if d and not d < box:
            raise ValueError(f"recess {d:g} um must be shallower than the BOX {box:g} um")
        if u and not width_bottom - 2 * u > 0:
            raise ValueError(f"undercut {u:g} um removes the whole pedestal under a "
                             f"{width_bottom:g} um wide core")
        if self.pedestal and not d:
            raise ValueError("a pedestal material needs a recess (recess_um > 0)")

    def label(self) -> str:
        """Short text for tables and figure titles; '' for a plain strip."""
        parts = []
        if self.slab_um:
            parts.append(f"slab {self.slab_um * 1e3:g} nm" + (
                "" if self.slab_width_um is None else f" × {self.slab_width_um:g} µm"))
        if self.recess_um:
            parts.append(f"BOX recess {self.recess_um * 1e3:g} nm, undercut "
                         f"{self.undercut_um * 1e3:g} nm"
                         + (f", {self.pedestal} pedestal" if self.pedestal else ""))
        return "; ".join(parts)

    def record(self) -> Optional[dict]:
        """The dict stored with a result; None for a plain strip (records unchanged)."""
        return None if self.is_default else asdict(self)

    @classmethod
    def from_record(cls, d) -> "Profile":
        return cls() if not d else cls(**{k: d[k] for k in asdict(cls()) if k in d})


def of_record(r: dict) -> Profile:
    """Profile of a solver record (records without one are plain strips)."""
    return Profile.from_record(r.get("profile"))


# ---- building it into a cross-section simulation ---------------------------------------- #
def apply(sim, profile: Optional[Profile], *, width_bottom: float, height: float,
          box: float, core_medium, box_medium, fill_medium=None, pedestal_medium=None,
          snap: bool = False, snap_offset: float = 0.0, y0: float = 0.0):
    """``sim`` with the profile's structures added (after every existing one, so they
    override the BOX and cladding where they overlap) and, with ``snap``, grid lines
    through their faces. A plain strip returns ``sim`` itself, untouched.

    fill_medium: what fills the recess (the top cladding; None = air).
    pedestal_medium: the pedestal's material (None = the BOX's)."""
    import tidy3d as td
    if profile is None or profile.is_default:
        return sim
    profile.check(width_bottom, height, box)
    structs = list(sim.structures)
    gs = sim.grid_spec
    points = list(gs.snapping_points or [])
    overrides = list(gs.override_structures or [])
    s = float(profile.slab_um)
    if s:
        sw = profile.slab_width_um
        structs.append(td.Structure(
            geometry=td.Box(center=(0, y0, s / 2), size=(td.inf, td.inf if sw is None else sw, s)),
            medium=core_medium, name="slab"))
        if snap:
            points.append((None, None, s + snap_offset))
            if sw is not None:
                half = sw / 2 + snap_offset
                points += [(None, y0 - half, None), (None, y0 + half, None)]
    d, u = float(profile.recess_um), float(profile.undercut_um)
    if d:
        fill = fill_medium if fill_medium is not None else td.Medium(permittivity=1.0)
        ped_w = width_bottom - 2 * u
        structs.append(td.Structure(
            geometry=td.Box(center=(0, 0, -d / 2), size=(td.inf, td.inf, d)),
            medium=fill, name="box_recess"))
        structs.append(td.Structure(
            geometry=td.Box(center=(0, y0, -d / 2), size=(td.inf, ped_w, d)),
            medium=pedestal_medium if pedestal_medium is not None else box_medium,
            name="box_pedestal"))
        if snap:
            points += [(None, y0 - ped_w / 2, -d), (None, y0 + ped_w / 2, -d)]
        if d + 0.1 > _FINE_BELOW_UM:                  # keep the recess in the fine mesh
            overrides = [_reach_down(o, d + 0.1) for o in overrides]
    grid = {}
    if points != list(gs.snapping_points or []):
        grid["snapping_points"] = points
    if overrides != list(gs.override_structures or []):
        grid["override_structures"] = overrides
    return sim.updated_copy(structures=structs,
                            grid_spec=gs.updated_copy(**grid) if grid else gs)


def _reach_down(o, depth: float):
    """A mesh-override box extended so its bottom lies at z = -depth or below."""
    import tidy3d as td
    (cx, cy, cz), (sx, sy, sz) = o.geometry.center, o.geometry.size
    bottom, top = cz - sz / 2, cz + sz / 2
    if bottom <= -depth:
        return o
    bottom = -depth
    return o.updated_copy(geometry=td.Box(center=(cx, cy, (bottom + top) / 2),
                                          size=(sx, sy, top - bottom)))


# ---- command-line tools ------------------------------------------------------------------ #
def add_args(ap) -> None:
    """The profile options of the sweep, broadband and convergence tools."""
    ap.add_argument("--slab", type=float, default=0.0,
                    help="partial etch: core-material slab left under and beside the core, um")
    ap.add_argument("--slab-width", type=float, default=None,
                    help="width of that slab, um (default: across the whole cross-section)")
    ap.add_argument("--recess", type=float, default=0.0,
                    help="BOX recess depth around the core, um")
    ap.add_argument("--undercut", type=float, default=0.0,
                    help="undercut of the BOX under each core edge, um")
    ap.add_argument("--pedestal", default="",
                    help="pedestal material key (default: the BOX material)")


def from_args(a) -> Profile:
    return Profile(slab_um=float(a.slab), slab_width_um=a.slab_width,
                   recess_um=float(a.recess), undercut_um=float(a.undercut),
                   pedestal=str(a.pedestal or ""))


def as_args(p: Profile) -> list:
    """Command-line options for ``p``; [] for a plain strip (commands unchanged)."""
    if p is None or p.is_default:
        return []
    out = []
    if p.slab_um:
        out += ["--slab", p.slab_um]
    if p.slab_width_um is not None:
        out += ["--slab-width", p.slab_width_um]
    if p.recess_um:
        out += ["--recess", p.recess_um]
    if p.undercut_um:
        out += ["--undercut", p.undercut_um]
    if p.pedestal:
        out += ["--pedestal", p.pedestal]
    return out


def tag(p: Profile) -> str:
    """File-name suffix for results of a non-default profile ('' otherwise)."""
    if p is None or p.is_default:
        return ""
    return "_profile" + __import__("hashlib").sha256(
        json.dumps(p.record(), sort_keys=True).encode()).hexdigest()[:8]
