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

In 3D (the cloud FDTD and EME runs: ring-bus gap, taper, edge facet) ``apply_3d`` builds
the same profile under every core feature of the device, curved or straight: each feature
(a ring, a bus, a taper polygon) is copied into the layer below (recess) or the bottom
of the film (finite slab), its outline moved in by the undercut (or out to the slab width),
and the holes between features (the inside of a ring) moved the other way. The 2D grating
runs are invariant across the width, so the lateral profile does not enter them.

Exposes: Profile, apply(), apply_3d(), from_args(), add_args(), as_args().
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
          snap: bool = False, snap_offset: float = 0.0, y0=0.0):
    """``sim`` with the profile's structures added (after every existing one, so they
    override the BOX and cladding where they overlap) and, with ``snap``, grid lines
    through their faces. A plain strip returns ``sim`` itself, untouched.

    fill_medium: what fills the recess (the top cladding; None = air).
    pedestal_medium: the pedestal's material (None = the BOX's).
    y0: the core's centre, or a tuple of centres for several identical cores (the two
    cores of a coupler: one recess, a pedestal or finite slab under each)."""
    import tidy3d as td
    if profile is None or profile.is_default:
        return sim
    ys = tuple(float(v) for v in y0) if isinstance(y0, (tuple, list)) else (y0,)
    profile.check(width_bottom, height, box)
    structs = list(sim.structures)
    gs = sim.grid_spec
    points = list(gs.snapping_points or [])
    overrides = list(gs.override_structures or [])
    s = float(profile.slab_um)
    if s:
        sw = profile.slab_width_um
        for i, y in enumerate(ys if sw is not None else ys[:1]):
            structs.append(td.Structure(
                geometry=td.Box(center=(0, y, s / 2), size=(td.inf, td.inf if sw is None else sw, s)),
                medium=core_medium, name="slab" + (f"_{i}" if i else "")))
        if snap:
            points.append((None, None, s + snap_offset))
            if sw is not None:
                half = sw / 2 + snap_offset
                for y in ys:
                    points += [(None, y - half, None), (None, y + half, None)]
    d, u = float(profile.recess_um), float(profile.undercut_um)
    if d:
        fill = fill_medium if fill_medium is not None else td.Medium(permittivity=1.0)
        ped_w = width_bottom - 2 * u
        structs.append(td.Structure(
            geometry=td.Box(center=(0, 0, -d / 2), size=(td.inf, td.inf, d)),
            medium=fill, name="box_recess"))
        for i, y in enumerate(ys):
            structs.append(td.Structure(
                geometry=td.Box(center=(0, y, -d / 2), size=(td.inf, ped_w, d)),
                medium=pedestal_medium if pedestal_medium is not None else box_medium,
                name="box_pedestal" + (f"_{i}" if i else "")))
        if snap:
            for y in ys:
                points += [(None, y - ped_w / 2, -d), (None, y + ped_w / 2, -d)]
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


# ---- building it into a 3D device ------------------------------------------------------- #
def as_profile(p) -> Optional["Profile"]:
    """A Profile from a Profile, its record (dict) or None."""
    if p is None or isinstance(p, Profile):
        return p
    return Profile.from_record(p)


def apply_3d(sim, profile, *, height: float, width_bottom: float, features, holes=(),
             box_name: str, core_medium, box_medium, fill_medium=None, pedestal_medium=None):
    """``sim`` (a 3D Simulation or EMESimulation) with the profile built under its core
    features. A plain strip (or None) returns ``sim`` itself, untouched.

    features: names of the core structures (film from z = 0 to height), in any order;
    holes: names of structures cut out of them (the inside of a ring), whose outline
    moves the other way; box_name: the BOX structure, whose lateral extent the recess
    and a full-width slab take (an edge-coupler BOX ends at the chip edge).
    width_bottom: the waveguide's bottom width (for a finite slab: the slab reaches
    (slab_width - width_bottom) / 2 past every core edge).

    New structures go after every existing one, in the order of the originals:
      recess:  'BOX recess' (fill, the BOX's lateral extent, z -d..0), then each feature as
               '<name> pedestal' (pedestal medium, outline moved in by the undercut) and
               each hole as '<name> recess' (fill, moved out by the undercut);
      slab:    full width: 'partial-etch slab' (core material, z 0..s, the BOX's extent);
               finite: each feature as '<name> slab' (core material, z 0..s, moved out
               by (slab_width - width_bottom) / 2) and each hole moved in by as much.
    Mesh: overrides that hold the core reach down to 0.1 um below the recess floor, and
    with snapping the recess floor / slab top get grid lines."""
    import numpy as np
    import tidy3d as td
    profile = as_profile(profile)
    if profile is None or profile.is_default:
        return sim
    by_name = {st.name: st for st in sim.structures}
    missing = [n for n in (*features, *holes, box_name) if n not in by_name]
    if missing:
        raise ValueError(f"profile: no structure named {missing} in the simulation")
    box_st = by_name[box_name]
    (bx0, by0, _), (bx1, by1, _) = box_st.geometry.bounds
    box_depth = -box_st.geometry.bounds[0][2]
    profile.check(width_bottom, height, box_depth)
    fill = fill_medium if fill_medium is not None else td.Medium(permittivity=1.0)
    keep = set(features) | set(holes)
    ordered = [(st, st.name in features) for st in sim.structures if st.name in keep]

    def lateral_box(za, zb):
        c = [(bx0 + bx1) / 2 if np.isfinite(bx1 - bx0) else 0.0,
             (by0 + by1) / 2 if np.isfinite(by1 - by0) else 0.0, (za + zb) / 2]
        sz = [bx1 - bx0 if np.isfinite(bx1 - bx0) else td.inf,
              by1 - by0 if np.isfinite(by1 - by0) else td.inf, zb - za]
        return td.Box(center=c, size=sz)

    new, z_lines = [], []
    s = float(profile.slab_um)
    d, u = float(profile.recess_um), float(profile.undercut_um)
    if s and profile.slab_width_um is None:
        new.append(td.Structure(geometry=lateral_box(0.0, s), medium=core_medium,
                                name="partial-etch slab"))
        z_lines.append(s)
    elif s:
        e = (float(profile.slab_width_um) - width_bottom) / 2
        for st, is_core in ordered:
            g = _layer(st.geometry, 0.0, s, e if is_core else -e)
            if g is not None:
                new.append(td.Structure(geometry=g, medium=core_medium if is_core else st.medium,
                                        name=f"{st.name} slab"))
        z_lines.append(s)
    if d:
        ped = pedestal_medium if pedestal_medium is not None else box_medium
        new.append(td.Structure(geometry=lateral_box(-d, 0.0), medium=fill, name="BOX recess"))
        for st, is_core in ordered:
            g = _layer(st.geometry, -d, 0.0, -u if is_core else u)
            if g is not None:
                new.append(td.Structure(geometry=g, medium=ped if is_core else fill,
                                        name=f"{st.name} {'pedestal' if is_core else 'recess'}"))
        z_lines.append(-d)
    upd = dict(structures=list(sim.structures) + new)
    gs = sim.grid_spec
    if gs is not None:
        grid = {}
        overrides = list(getattr(gs, "override_structures", None) or [])
        if d and overrides:
            moved = [_reach_down(o, d + 0.1) if _holds_core(o, height) else o for o in overrides]
            if moved != overrides:
                grid["override_structures"] = moved
        if hasattr(gs, "snapping_points"):
            grid["snapping_points"] = list(gs.snapping_points or []) + \
                [(None, None, z) for z in z_lines]
        if grid:
            upd["grid_spec"] = gs.updated_copy(**grid)
    return sim.updated_copy(**upd)


def _holds_core(o, height: float) -> bool:
    (_, _, z0), (_, _, z1) = o.geometry.bounds
    return z0 <= 1e-9 and z1 >= height - 1e-9


def _min_width(verts) -> float:
    """Smallest width of a polygon (the narrowest of its projections on its edge normals:
    exact for a convex polygon, an upper bound otherwise, as for a curved tooth)."""
    import numpy as np
    v = np.asarray(verts, float)
    e = np.roll(v, -1, axis=0) - v
    n = np.stack([-e[:, 1], e[:, 0]], axis=1)
    n = n[np.linalg.norm(n, axis=1) > 1e-12]
    n /= np.linalg.norm(n, axis=1)[:, None]
    proj = v @ n.T
    return float(np.min(proj.max(axis=0) - proj.min(axis=0)))


def _layer(g, za: float, zb: float, delta: float, cutter: bool = False):
    """The footprint of ``g`` at its bottom face (z = 0), extruded from za to zb, with its
    outline moved out by ``delta`` (in when negative); None when nothing is left.

    Box: across its narrower lateral side (a waveguide's width; its length is kept);
    Cylinder: the radius; PolySlab: the polygon (dilated); clip operations and groups:
    their parts (the second operand of a difference moves the other way; a box that
    clips a part, such as the box that keeps an arc of a ring, grows on every side)."""
    import numpy as np
    import tidy3d as td
    zc, zl = (za + zb) / 2, zb - za
    if isinstance(g, td.Box):
        c, s = list(g.center), list(g.size)
        c[2], s[2] = zc, zl
        if delta:
            # a waveguide: across its narrower side only (its length is kept); a box that
            # clips another part (the box that keeps an arc of a ring): grown on every side
            # whichever way the part moves, so the part's own outline decides (a hole
            # moved in must still cover all of the core moved out)
            for ax in ((0, 1) if cutter else (0 if s[0] < s[1] else 1,)):
                if np.isfinite(s[ax]):
                    s[ax] += 2 * (abs(delta) if cutter else delta)
                    if s[ax] <= 0:
                        return None
        return td.Box(center=c, size=s)
    if isinstance(g, td.Cylinder):
        if g.axis != 2:
            raise NotImplementedError("profile: only vertical (z) cylinders")
        r = float(g.radius_bottom) + delta
        if r <= 0:
            return None
        return td.Cylinder(center=(g.center[0], g.center[1], zc), radius=r, length=zl, axis=2)
    if isinstance(g, td.PolySlab):
        if g.axis == 2:
            verts = np.asarray(g.base_polygon)
            if delta < 0 and 2 * -delta >= _min_width(verts):
                return None                            # fully undercut: no pedestal left
            try:
                return td.PolySlab(vertices=verts.tolist(), slab_bounds=(za, zb), axis=2,
                                   dilation=float(delta))
            except Exception as err:                   # a curved part narrower than 2 u
                if delta < 0 and "fully eroded" in str(err):
                    return None
                raise
        # a sloped-wall waveguide: trapezoid in the cross-section plane, extruded along x or y
        v = np.asarray(g.vertices)                     # (lateral, z) in the plane
        lat = v[np.isclose(v[:, 1], v[:, 1].min()), 0]
        lo, hi = float(lat.min()) - delta, float(lat.max()) + delta
        if hi <= lo:
            return None
        a, b = g.slab_bounds
        length = b - a if np.isfinite(b - a) else td.inf
        mid = (a + b) / 2 if np.isfinite(b - a) else 0.0
        if g.axis == 0:
            return td.Box(center=(mid, (lo + hi) / 2, zc), size=(length, hi - lo, zl))
        return td.Box(center=((lo + hi) / 2, mid, zc), size=(hi - lo, length, zl))
    if isinstance(g, td.ClipOperation):
        a = _layer(g.geometry_a, za, zb, delta, cutter=isinstance(g.geometry_a, td.Box)
                   and not isinstance(g.geometry_b, td.Box))
        flip = -delta if g.operation == "difference" else delta
        b = _layer(g.geometry_b, za, zb, flip, cutter=isinstance(g.geometry_b, td.Box))
        if a is None:
            return None
        if b is None:
            return None if g.operation == "intersection" else a
        return td.ClipOperation(operation=g.operation, geometry_a=a, geometry_b=b)
    if isinstance(g, td.GeometryGroup):
        parts = [p for p in (_layer(x, za, zb, delta) for x in g.geometries) if p is not None]
        return td.GeometryGroup(geometries=parts) if parts else None
    raise NotImplementedError(f"profile: {type(g).__name__} geometries are not supported")


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
