"""v0.1.4: diverged runs are reported, never used; a choice of absorbing boundary for the
grating FDTD; sloped walls snapped continuously into the vertical case."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest


# ------------------------------------------------------------- sloped-wall snapping
def test_sloped_snap_points_meet_the_vertical_ones_at_90_degrees():
    from tarang.waveguide.dispersion_mapping import core as dm
    vert = dm.build_sim(0.7, 1.0, wl0=1.55, steps=20, snap=True).grid_spec.snapping_points
    for a in (89.0, 89.9, 89.999):
        pts = dm.sloped_snap_points(1.0, 0.7, a, "top")
        assert len(pts) == 2 and pts[0][2] == 0.0 and pts[1][2] == 0.7
        s = 0.7 / math.tan(math.radians(a))                 # wall shift over the height
        assert pts[1][1] == pytest.approx(0.5 + s / 2)     # the wall at mid-height
    # the limit is the vertical core's own lines: no sliver cell near 90 degrees
    near = dm.sloped_snap_points(1.0, 0.7, 89.999999, "top")
    assert [p[1] for p in near] == pytest.approx([p[1] for p in vert], abs=1e-6)


def test_sloped_cross_section_has_one_grid_line_per_wall():
    from tarang.waveguide.dispersion_mapping import core as dm
    sim = dm.build_sim(0.35, 0.5, wl0=0.78, steps=30, snap=True, sidewall_deg=87.0)
    ys = sorted(p[1] for p in sim.grid_spec.snapping_points)
    wb = 0.5 + 2 * 0.35 / math.tan(math.radians(87.0))
    assert ys == pytest.approx([-(wb + 0.5) / 4, (wb + 0.5) / 4])


def test_snap_walls_marker_is_shared():
    from tarang.geometries.sidewall import SNAP_WALLS
    from tarang.design import walls
    from tarang.waveguide.dispersion_mapping import core, sweep
    assert SNAP_WALLS == "mid" and walls.SNAP_WALLS is core.SNAP_WALLS is sweep.SNAP_WALLS


# ------------------------------------------------------------- grating boundary
def _gc(**kw):
    import tidy3d as td
    from tarang.grating_coupler.grating_coupler import GratingConfig, GratingCoupler
    c = GratingConfig(wavelength_um=0.78, theta_deg=12, mfd=5, source_x=2.5, source_gap=2.0,
                      core_height=0.35, top_cladding_height=0.0, box_height=4.0,
                      handle_height=3.0, etch_depth=0.35, period=0.56, ff=0.7, n_teeth=30,
                      w_grating=7, buffer=3.0, pol_angle=math.pi / 2, **kw)
    return GratingCoupler.from_config(c, core_medium=td.Medium(permittivity=4.0),
                                      box_medium=td.Medium(permittivity=2.1),
                                      handle_medium=None, top_cladding_medium=None)


def test_default_boundary_is_the_old_pml_so_saved_results_stay_valid():
    import tidy3d as td
    d = _gc()
    b = d.sim.boundary_spec
    assert b.x == td.Boundary.pml() and b.z == td.Boundary.pml()
    assert b.y == td.Boundary.periodic()                        # 2D
    assert not d._readable_name().endswith(("_spml", "_abs"))


@pytest.mark.parametrize("kind,cls,tag", [("stable_pml", "StablePML", "_spml"),
                                          ("absorber", "Absorber", "_abs")])
def test_other_boundaries_change_the_simulation_and_its_name(kind, cls, tag):
    from tarang.utilities.cloud_run import sim_fingerprint
    d, base = _gc(boundary=kind), _gc()
    assert type(d.sim.boundary_spec.x.plus).__name__ == cls
    assert type(d.sim.boundary_spec.z.minus).__name__ == cls
    assert d._readable_name().endswith(tag)
    assert sim_fingerprint(d.sim) != sim_fingerprint(base.sim)


def test_unknown_boundary_is_refused():
    with pytest.raises(ValueError, match="grating boundary"):
        _gc(boundary="pec")


def test_design_config_passes_the_boundary():
    from tarang.design.config import Grating
    assert Grating().boundary == "pml"


# ------------------------------------------------------------- diverged results
def test_check_diverged_raises_with_the_task_link(tmp_path):
    from tarang.utilities.cloud_run import (SimulationDivergedError, check_diverged,
                                            write_task_record)
    check_diverged(SimpleNamespace(diverged=False), "fine")              # nothing happens
    write_task_record(tmp_path, task_id="fdve-123", url="https://example.org/t")
    with pytest.raises(SimulationDivergedError) as e:
        check_diverged(SimpleNamespace(diverged=True), "gc2D_x", tmp_path)
    assert "gc2D_x" in str(e.value) and e.value.url == "https://example.org/t"


def test_mesh_ladder_reports_a_diverged_scale_instead_of_dropping_it(tmp_path):
    from tarang.design import checks
    from tarang.utilities.cloud_run import SimulationDivergedError
    wl = np.linspace(0.73, 0.83, 51)

    def spec(peak, eta):
        return wl, eta * np.exp(-((wl - peak) / 0.03) ** 2)

    class Ok:
        def get_spectrum(self, folder):
            return spec(0.779, 0.236)

    class Diverged:
        def get_spectrum(self, folder):
            raise SimulationDivergedError("gc2D_ladder", "https://example.org/x")

    ctx = SimpleNamespace(out=tmp_path, cfg=SimpleNamespace(run=SimpleNamespace(wavelength_um=0.78),
                                                            grating=SimpleNamespace(mesh_scale=1.0)))
    failed: list = []
    rows = checks.mesh_ladder_rows(ctx, {}, {2.0: Ok(), 2.5: Diverged(), 3.0: Diverged()},
                                   {"final": spec(0.7776, 0.168)}, failed=failed)
    assert [r["mesh_scale"] for r in rows] == [1.0, 2.0]
    assert [(f["mesh_scale"], f["cells_per_wavelength"], f["reason"]) for f in failed] == [
        (2.5, 38, "diverged"), (3.0, 45, "diverged")]
    assert failed[0]["url"] == "https://example.org/x"


def test_new_boundary_setting_keeps_older_runs_done():
    from tarang.design.config import DesignConfig
    from tarang.design.stages import settings_key
    cfg = DesignConfig()
    before = settings_key(cfg, "grating")
    del cfg.grating.__dict__["boundary"]                     # a design.toml from v0.1.3
    assert settings_key(cfg, "grating") == before
    cfg = DesignConfig()
    cfg.grating.boundary = "stable_pml"
    assert settings_key(cfg, "grating") != before            # a new choice runs again
