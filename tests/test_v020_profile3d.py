"""v0.2.0, phase D: the cross-section profile (slab / BOX recess / undercut / pedestal) in
the 3D cloud runs: ring-bus gap, taper (EME), edge facet, focusing grating. A plain strip
must leave every simulation exactly as it was (same cloud fingerprint)."""
from __future__ import annotations

import io
import math
import warnings

import numpy as np
import pytest

td = pytest.importorskip("tidy3d")

from tarang.waveguide.dispersion_mapping.profile import Profile, _layer, apply_3d  # noqa: E402

SIO2 = td.Medium(permittivity=1.444 ** 2)
SIN = td.Medium(permittivity=2.0 ** 2)
R, W, H, GAP = 20.0, 0.55, 0.40, 0.30
F0 = td.C_0 / 0.78


def _ring(**kw):
    from tarang.geometries.partial_ring_resonator import get_active_simulation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return get_active_simulation(radius=R, gap=GAP, wg_width=W, wg_height=H, core_medium=SIN,
                                     box_medium=SIO2, wavelength_um=0.78,
                                     span_angle_rad=math.radians(30), inner_box_recess_um=0.0,
                                     **kw)


def _eps_line(sim, z, x=0.0):
    """Re(eps) along y at (x, z), and a lookup by y."""
    eps = sim.epsilon(box=td.Box(center=(x, 0, z), size=(0, td.inf, 0)), coord_key="centers",
                      freq=F0)
    y = eps.coords["y"].values
    e = np.real(np.squeeze(eps.values))
    return y, e, (lambda yy: float(e[np.argmin(np.abs(y - yy))]))


# ---- a plain strip changes nothing ------------------------------------------------------- #
def test_plain_strip_leaves_the_ring_simulation_and_its_cache_untouched():
    from tarang.utilities.cloud_run import cache_paths
    from tarang.utilities.sweep_helpers import cache_dir
    a = _ring()
    for p in (None, Profile(), Profile().record(), {}):
        b = _ring(profile=p)
        assert b.sim == a.sim
        assert cache_paths("x", "n", b.sim) == cache_paths("x", "n", a.sim)
        assert cache_dir("x", b) == cache_dir("x", a)


def test_apply_3d_returns_the_same_object_for_a_plain_strip():
    sim = _ring().sim
    kw = dict(height=H, width_bottom=W, features=("Ring_Outer",), box_name="Substrate",
              core_medium=SIN, box_medium=SIO2)
    assert apply_3d(sim, None, **kw) is sim
    assert apply_3d(sim, Profile(), **kw) is sim


# ---- recess and undercut under a curved ring and a straight bus -------------------------- #
def test_recess_with_undercut_under_ring_and_bus():
    d = _ring(profile=Profile(recess_um=0.3, undercut_um=0.1).record())
    names = [s.name for s in d.sim.structures]
    assert names[-4:] == ["BOX recess", "Ring_Outer pedestal", "Ring_Inner_Hole recess",
                          "Bus_Waveguide pedestal"]
    yb = -(R + W + GAP)                         # bus centre; the ring's centre line is y = -R
    box, air, core = 1.444 ** 2, 1.0, 4.0
    _, _, at = _eps_line(d.sim, -0.15)          # inside the recess
    assert at(yb) == pytest.approx(box, abs=1e-3)                  # bus pedestal
    assert at(yb + W / 2 - 0.05) == air                            # under the bus edge: undercut
    assert at(yb + W / 2 - 0.15) == pytest.approx(box, abs=1e-3)
    assert at(yb + W / 2 + GAP / 2) == air                         # the gap
    assert at(-R) == pytest.approx(box, abs=1e-3)                  # ring pedestal
    assert at(-R + W / 2 - 0.05) == air                            # under the ring's inner edge
    assert at(-R + W / 2 + 0.05) == air                            # inside the ring
    _, _, at = _eps_line(d.sim, -0.35)                             # below the floor: BOX
    assert at(-R + 2.0) == pytest.approx(box, abs=1e-3)
    _, _, at = _eps_line(d.sim, H / 2)                             # the core layer as before
    assert at(yb) == core and at(-R) == core and at(-R + 2.0) == air
    # mesh: the fine region reaches 0.1 um below the floor; the floor gets a grid line
    assert d.sim.grid_spec.override_structures[0].geometry.bounds[0][2] == pytest.approx(-0.4)
    assert (None, None, -0.3) in d.sim.grid_spec.snapping_points


def test_bus_only_reference_drops_the_ring_pedestal_too():
    from tarang.ring_resonator.ring_gap_sweep import bus_only_device
    d = _ring(profile=Profile(recess_um=0.3, undercut_um=0.1).record())
    assert [s.name for s in bus_only_device(d).sim.structures] == [
        "Substrate", "Bus_Waveguide", "BOX recess", "Bus_Waveguide pedestal"]


def test_pedestal_of_another_material():
    ta = td.Medium(permittivity=2.05 ** 2)
    d = _ring(profile=Profile(recess_um=0.2, pedestal="ta2o5").record(), pedestal_medium=ta)
    _, _, at = _eps_line(d.sim, -0.1)
    assert at(-R) == pytest.approx(2.05 ** 2, abs=1e-3)


# ---- slabs ------------------------------------------------------------------------------- #
def test_full_width_slab_fills_the_ring_hole_bottom():
    d = _ring(profile=Profile(slab_um=0.1).record())
    _, e, _ = _eps_line(d.sim, 0.05)
    assert float(e.min()) == 4.0                                   # core material everywhere
    _, _, at = _eps_line(d.sim, 0.2)
    assert at(-R + 2.0) == 1.0                                     # above it: as before


def test_finite_slab_reaches_past_every_edge():
    d = _ring(profile=Profile(slab_um=0.1, slab_width_um=W + 0.4).record())
    y, e, at = _eps_line(d.sim, 0.05)
    core = y[e > 3.9]
    yb = -(R + W + GAP)
    assert core.min() == pytest.approx(yb - W / 2 - 0.2, abs=0.03)
    assert core.max() == pytest.approx(-R + W / 2 + 0.2, abs=0.03)
    # gap 0.3 um < 2 x 0.2 um: the bus and ring slabs touch, one region
    assert int(np.sum(np.diff((e > 3.9).astype(int)) != 0)) == 2


@pytest.mark.parametrize("p, msg", [(Profile(recess_um=0.2, undercut_um=0.3), "undercut"),
                                    (Profile(slab_um=0.5), "thinner")])
def test_impossible_profiles_are_refused(p, msg):
    with pytest.raises(ValueError, match=msg):
        _ring(profile=p.record())


# ---- footprints ---------------------------------------------------------------------------- #
def test_layer_footprints():
    box = td.Box(center=(0, 1, H / 2), size=(td.inf, 0.6, H))
    g = _layer(box, -0.3, 0.0, -0.1)
    assert g.size[1] == pytest.approx(0.4) and g.size[0] == td.inf
    assert g.bounds[0][2] == pytest.approx(-0.3) and g.bounds[1][2] == pytest.approx(0.0)
    cyl = td.Cylinder(center=(0, 0, H / 2), radius=5.0, length=H, axis=2, sidewall_angle=0.2,
                      reference_plane="top")
    assert _layer(cyl, -0.3, 0, 0.1).radius == pytest.approx(cyl.radius_bottom + 0.1)
    poly = td.PolySlab(vertices=[(0, -1), (4, -0.2), (4, 0.2), (0, 1)], slab_bounds=(0, H), axis=2)
    pg = _layer(poly, -0.3, 0, -0.05)
    assert pg.dilation == pytest.approx(-0.05) and pg.slab_bounds == (-0.3, 0)
    # a sloped-wall waveguide (trapezoid in y-z, along x): its bottom width
    trap = td.PolySlab(vertices=[(-0.4, 0), (0.4, 0), (0.3, H), (-0.3, H)], slab_bounds=(-td.inf,
                                                                                       td.inf),
                       axis=0)
    tb = _layer(trap, -0.3, 0, -0.1)
    assert tb.size[1] == pytest.approx(0.6)
    # clip: a cylinder kept within a box -> the box grows on every side, whichever way the
    # part moves (its edge can coincide with the ring's outer edge, as in the partial ring)
    clip = td.Cylinder(center=(0, 0, H / 2), radius=5, length=H, axis=2) & \
        td.Box(center=(0, -4, H / 2), size=(4, 2, H + 0.4))
    cg = _layer(clip, -0.3, 0, -0.1)
    assert cg.geometry_a.radius == pytest.approx(4.9)
    assert cg.geometry_b.size[:2] == pytest.approx((4.2, 2.2))
    diff = td.ClipOperation(operation="difference", geometry_a=clip.geometry_a,
                            geometry_b=td.Cylinder(center=(0, 0, H / 2), radius=4, length=H, axis=2))
    dg = _layer(diff, -0.3, 0, -0.1)
    assert (dg.geometry_a.radius, dg.geometry_b.radius) == pytest.approx((4.9, 4.1))


# ---- the design's 3D stages --------------------------------------------------------------- #
def _ctx(tmp_path, **stack):
    from tarang.design.config import DesignConfig
    from tarang.design.stages import Context
    from tarang.design.ui import Prompter
    cfg = DesignConfig()
    cfg.run.wavelength_um, cfg.run.band_um = 0.78, 0.05
    for k, v in stack.items():
        setattr(cfg.stack, k, v)
    ctx = Context(cfg, Prompter(interactive=False, out=io.StringIO()), tmp_path / "run")
    ctx.state["stages"] = {"cross_section": {"results": dict(height_um=H, width_um=W)}}
    return ctx


def test_gap_builder_options_keep_defaults_and_carry_the_profile(tmp_path):
    from tarang.design.stages import gap_builder_options
    ctx = _ctx(tmp_path)
    plain = gap_builder_options(ctx.cfg, ctx.medium)
    assert "profile" not in plain and "pedestal_medium" not in plain
    assert plain == gap_builder_options(ctx.cfg)
    ctx = _ctx(tmp_path, box_recess_um=0.3, box_undercut_um=0.1)
    o = gap_builder_options(ctx.cfg, ctx.medium)
    assert o["profile"]["recess_um"] == 0.3 and o["inner_box_recess_um"] == 0.0
    ctx = _ctx(tmp_path, box_recess_um=0.3, pedestal_material="sio2")
    assert "pedestal_medium" in gap_builder_options(ctx.cfg, ctx.medium)
    with pytest.raises(ValueError, match="pedestal"):
        gap_builder_options(ctx.cfg)


def test_edge_facet_and_taper(tmp_path):
    from tarang.design.edge import facet_sim
    from tarang.design.taper import eme_taper
    plain = _ctx(tmp_path)
    rec = _ctx(tmp_path, box_recess_um=0.3, box_undercut_um=0.1)
    a, b = facet_sim(plain, 0.3, 1.5), facet_sim(rec, 0.3, 1.5)
    assert a == facet_sim(_ctx(tmp_path), 0.3, 1.5)
    assert [s.name for s in b.structures][-2:] == ["BOX recess", "Facet waveguide pedestal"]
    _, _, at = _eps_line(b, -0.15, x=-3.0)
    box = float(np.real(rec.medium("sio2").eps_model(F0)))
    assert at(0) == pytest.approx(box, abs=1e-3) and at(0.13) == 1.0     # 0.1 um pedestal
    # the recess ends with the BOX at the chip edge (x = 0)
    assert b.structures[-2].geometry.bounds[1][0] == pytest.approx(0.0)
    ta, tb = eme_taper(plain, 3.0, W, 10.0, [1.0]), eme_taper(rec, 3.0, W, 10.0, [1.0])
    assert ta == eme_taper(_ctx(tmp_path), 3.0, W, 10.0, [1.0])
    assert [s.name for s in tb.structures][-4:] == ["BOX recess", "grating end pedestal",
                                                     "taper pedestal", "waveguide pedestal"]
    assert tb.bounds[0][2] <= -1.3


def test_focusing_grating_gets_pedestals_under_every_tooth(tmp_path, fake_web, monkeypatch):
    """The focusing grating's 3D run: a pedestal under the waveguide, the tip taper, the
    slab and each curved tooth; a plain strip leaves the simulation as it was."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from test_phase12c import _focus_ctx
    from tarang.design import focus
    from tarang.grating_coupler.grating_coupler import GratingCoupler
    monkeypatch.setattr(GratingCoupler, "get_spectrum", lambda self, folder: (
        np.linspace(1.45, 1.65, 21), 0.29 * np.exp(-((np.linspace(1.45, 1.65, 21) - 1.552) / 0.04) ** 2)))
    ctx = _focus_ctx(tmp_path / "a", focus__check_3d=True, run__max_total_cost=10.0)
    res = focus.stage_focus(ctx)
    plain = focus.focusing_sim(ctx, res).sim
    ctx.cfg.stack.box_recess_um, ctx.cfg.stack.box_undercut_um = 0.5, 0.1
    sim = focus.focusing_sim(ctx, res).sim
    assert sim.structures[:len(plain.structures)] == plain.structures
    new = [s.name for s in sim.structures[len(plain.structures):]]
    assert new[0] == "BOX recess" and "Waveguide pedestal" in new
    assert "Focusing_slab pedestal" in new and "Tooth_29 pedestal" in new
    sim.validate_pre_upload()


@pytest.mark.real_modes
def test_coupler_supermodes_put_a_pedestal_under_each_core(tmp_path, monkeypatch):
    """The supermode check of the gap (two cores side by side): the profile's recess and
    pedestal were built for one core at y = 0 (under the gap); now one under each core."""
    from tarang.design import localmodes
    from tarang.design.config import profile_of
    from tarang.waveguide.dispersion_mapping import core as dm
    ctx = _ctx(tmp_path, box_recess_um=0.3, box_undercut_um=0.1)
    cfg = ctx.cfg
    built = {}

    class Stop(Exception):
        pass

    def fake_solver(ctx_, H_, W_, wl, **kw):
        sim = dm.build_sim(H_, W_, wl0=wl, steps=20, box=cfg.stack.box_height_um,
                           core=cfg.materials.core, substrate=cfg.materials.box,
                           cladding=cfg.materials.cladding, band=ctx.band,
                           margin_y=kw["margin_y"], profile=profile_of(cfg))
        built["sim"] = kw["extra"](sim)
        raise Stop
    monkeypatch.setattr(localmodes, "_solver", fake_solver)
    with pytest.raises(Stop):
        localmodes.coupler_supermodes(ctx, H, W, 0.3)
    sim = built["sim"]
    assert [s.name for s in sim.structures][-3:] == ["box_recess", "box_pedestal", "box_pedestal_1"]
    box = float(np.real(ctx.medium("sio2").eps_model(F0)))
    _, _, at = _eps_line(sim, -0.15)
    shift = W + 0.3
    assert at(-shift / 2) == pytest.approx(box, abs=1e-3)
    assert at(shift / 2) == pytest.approx(box, abs=1e-3)
    assert at(0.0) == 1.0                                          # under the gap: recess
    assert at(shift / 2 + W / 2 - 0.05) == 1.0                     # undercut
    _, _, at = _eps_line(sim, H / 2)
    assert at(-shift / 2) > 3.5 and at(shift / 2) > 3.5 and at(0.0) == 1.0


def test_2d_profile_with_one_core_is_unchanged_by_the_tuple_form():
    from tarang.waveguide.dispersion_mapping import core as dm
    p = Profile(recess_um=0.3, undercut_um=0.1)
    a = dm.build_sim(H, W, wl0=0.78, steps=20, box=2.0, band=(0.73, 0.83), snap=True,
                     snap_offset=1e-6, profile=p)
    from tarang.waveguide.dispersion_mapping.profile import apply
    plain = dm.build_sim(H, W, wl0=0.78, steps=20, box=2.0, band=(0.73, 0.83), snap=True,
                         snap_offset=1e-6)
    kw = dict(width_bottom=W, height=H, box=2.0, core_medium=SIN, box_medium=SIO2, snap=True,
              snap_offset=1e-6)
    assert apply(plain, p, y0=0.0, **kw) == apply(plain, p, y0=(0.0,), **kw)
    assert [s.name for s in a.structures][-2:] == ["box_recess", "box_pedestal"]
