"""v0.1.5 two-sided grid snapping: every snapped cross-section solve is the mean of two,
the grid lines just inside and just outside the core's faces, so a result does not depend
on how a computer rounds the coordinates of the samples that lie on a face."""
from __future__ import annotations

import io
import subprocess
import sys

import numpy as np
import pytest


# ------------------------------------------------------------- the offsets and records
def test_snap_offsets():
    from tarang.waveguide.dispersion_mapping.snapping import SNAP_OFFSET_UM, snap_offsets
    assert snap_offsets(False) == (0.0,) and snap_offsets(False, 2) == (0.0,)
    assert snap_offsets(True, 1) == (0.0,)
    assert snap_offsets(True) == snap_offsets(True, 2) == (-SNAP_OFFSET_UM, SNAP_OFFSET_UM)
    assert 0 < SNAP_OFFSET_UM <= 1e-5
    with pytest.raises(ValueError):
        snap_offsets(True, 3)


def test_saved_records_match_their_snapping():
    from tarang.waveguide.dispersion_mapping.snapping import same_snap, snap_sides_of
    old = dict(snap=True)                      # a record from before v0.1.5: one-sided
    new = dict(snap=True, snap_sides=2)
    off = dict(snap=False, snap_sides=1)
    assert snap_sides_of(old) == 1 and snap_sides_of(new) == 2 and snap_sides_of({}) == 1
    assert same_snap(old, True, 1) and not same_snap(old, True, 2)
    assert same_snap(new, True, 2) and not same_snap(new, True, 1)
    assert same_snap(off, False, 2) and same_snap({}, False, 1) and not same_snap(off, True, 2)


def test_snap_points_move_outside_or_inside_the_core():
    from tarang.waveguide.dispersion_mapping import core as dm
    d = 1e-6
    out = dm.sloped_snap_points(0.5, 0.35, 90.0, offset=d)
    ins = dm.sloped_snap_points(0.5, 0.35, 90.0, offset=-d)
    assert [p[1] for p in out] == pytest.approx([-0.25 - d, 0.25 + d], abs=1e-12)
    assert [p[2] for p in out] == pytest.approx([-d, 0.35 + d], abs=1e-12)
    assert [p[1] for p in ins] == pytest.approx([-0.25 + d, 0.25 - d], abs=1e-12)
    assert [p[2] for p in ins] == pytest.approx([d, 0.35 - d], abs=1e-12)
    two = dm.sloped_snap_points(0.5, 0.35, 90.0, y0=0.4, offset=d)          # a second core
    assert [p[1] for p in two] == pytest.approx([0.15 - d, 0.65 + d], abs=1e-12)


def test_build_sim_puts_grid_lines_at_the_offset_faces():
    import tidy3d as td
    from tarang.waveguide.dispersion_mapping import core as dm
    for off in (-1e-6, 1e-6):
        sim = dm.build_sim(0.35, 0.5, wl0=0.78, steps=30, snap=True, band=(0.73, 0.83),
                           snap_offset=off)
        plane = td.Box(center=(0, 0, sim.center[2]), size=(0, sim.size[1], sim.size[2]))
        g = sim.discretize(plane).boundaries
        y, z = np.asarray(g.y), np.asarray(g.z)
        for v, arr in ((0.25 + off, y), (-0.25 - off, y), (-off, z), (0.35 + off, z)):
            assert np.min(np.abs(arr - v)) < 1e-9


# ------------------------------------------------------------- the solves
@pytest.fixture(scope="module")
def two_sided():
    from tarang.waveguide.dispersion_mapping import solve_point
    kw = dict(wl0=0.78, span=0.04, npts=5, num_modes=2, steps=30, snap=True)
    return solve_point(0.35, 0.5, **kw), kw


def test_two_sided_solve_is_the_mean_of_its_two_sides(two_sided):
    r, _ = two_sided
    s = r["snap_split"]
    assert r["snap_sides"] == 2 and s["offsets_um"] == [-1e-6, 1e-6]
    assert r["neff_TE"][0] == pytest.approx(np.mean(s["neff_TE0"]), abs=1e-5)
    # D and n_g are linear in n_eff: the mean's D is the mean of the two D values
    assert r["D_TE0"] == pytest.approx(np.mean(s["D_TE0"]), abs=0.02)
    assert r["ng_TE0"] == pytest.approx(np.mean(s["ng_TE0"]), abs=1e-4)
    # lines inside the core: every sample on a face is SiN, so n_eff is higher
    assert s["neff_TE0"][0] > s["neff_TE0"][1]


def test_two_sided_solve_does_not_depend_on_rounding(two_sided):
    """The point of v0.1.5: a 1e-12 um change of the width (rounding) leaves it unchanged."""
    from tarang.waveguide.dispersion_mapping import solve_point
    r, kw = two_sided
    r2 = solve_point(0.35, 0.5 + 1e-12, **kw)
    assert r2["neff_TE"][0] == pytest.approx(r["neff_TE"][0], abs=1e-7)
    assert r2["D_TE0"] == pytest.approx(r["D_TE0"], abs=0.05)


def test_one_sided_solve_is_unchanged():
    """snap_sides=1 is the solve of v0.1.0-0.1.4: no split recorded, one solve."""
    from tarang.waveguide.dispersion_mapping import solve_point
    r = solve_point(0.35, 0.5, wl0=0.78, span=0.04, npts=5, num_modes=2, steps=30, snap=True,
                    snap_sides=1)
    assert r["snap_sides"] == 1 and r["snap_split"] is None
    u = solve_point(0.35, 0.5, wl0=0.78, span=0.04, npts=5, num_modes=2, steps=30)
    assert u["snap"] is False and u["snap_sides"] == 1 and u["snap_split"] is None


# ------------------------------------------------------------- the design settings
def test_snap_sides_setting_and_legacy_key():
    from tarang.design.config import LEGACY, DesignConfig
    from tarang.design.stages import STAGE_NAMES, settings_key
    cfg = DesignConfig()
    assert cfg.cross_section.snap_sides == 2 and LEGACY["cross_section"]["snap_sides"] == 1
    cfg.cross_section.snap_sides = 1                      # what an older run folder loads as
    keys = {s: settings_key(cfg, s) for s in STAGE_NAMES}
    cfg.cross_section.__dict__.pop("snap_sides")          # as if the setting did not exist
    assert {s: settings_key(cfg, s) for s in STAGE_NAMES} == keys
    cfg.cross_section.snap_sides = 2
    assert settings_key(cfg, "cross_section") != keys["cross_section"]
    assert settings_key(cfg, "materials") == keys["materials"]


def test_tools_get_the_snap_sides(tmp_path):
    from tarang.design.config import DesignConfig
    from tarang.design.stages import Context
    from tarang.design.ui import Prompter
    cfg = DesignConfig()
    ctx = Context(cfg, Prompter(interactive=False, out=io.StringIO()), tmp_path / "run")
    args = [str(a) for a in ctx.stack_args()]
    assert args[-3:] == ["--snap", "--snap-sides", "2"] and ctx.snap_sides == 2
    cfg.cross_section.snap_sides = 1
    assert [str(a) for a in ctx.stack_args()][-1] == "1"
    cfg.cross_section.snap_grid = False
    assert "--snap" not in ctx.stack_args()


@pytest.mark.parametrize("tool", ["sweep", "convergence", "broadband"])
def test_command_line_tools_take_snap_sides(tool):
    out = subprocess.run([sys.executable, "-m", f"tarang.waveguide.dispersion_mapping.{tool}",
                          "--help"], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and "--snap-sides" in out.stdout


@pytest.mark.real_modes
def test_supermodes_two_sided(tmp_path):
    """The coupler's even and odd supermodes are each the mean of their two sides."""
    from tarang.design import localmodes
    from tarang.design.config import DesignConfig
    from tarang.design.stages import Context
    from tarang.design.ui import Prompter
    cfg = DesignConfig()
    cfg.run.wavelength_um, cfg.run.band_um = 0.78, 0.1
    cfg.cross_section.decision_steps = 20
    ctx = Context(cfg, Prompter(interactive=False, out=io.StringIO()), tmp_path / "run")
    sm = localmodes.coupler_supermodes(ctx, 0.35, 0.5, 0.25, snap=True, snap_sides=2)
    for m in (sm["even"], sm["odd"]):
        assert len(m["neff_sides"]) == 2 and m["neff"] == pytest.approx(np.mean(m["neff_sides"]))
    assert sm["dn"] == pytest.approx(sm["even"]["neff"] - sm["odd"]["neff"]) and sm["dn"] > 0
