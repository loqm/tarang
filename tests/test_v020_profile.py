"""v0.2.0, phase B: the cross-section profile (partial-etch slab, BOX recess with undercut
and pedestal material) and the stack preview.

Reference for the recess: the ANT oxide-window scan of 26 Nov 2026
(design_runs/ant_recess/box_recess_scan.py, 12 points: W 0.50-0.60 um, d = u 0-150 nm, H
400 nm, 780 nm, BOX 4.5 um, 50 steps, two-sided snapping). The profile must build exactly
the simulation that script built, so its solved values (recess_scan.csv) carry over."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

td = pytest.importorskip("tidy3d")

from tarang.waveguide.dispersion_mapping import core as dm                 # noqa: E402
from tarang.waveguide.dispersion_mapping.profile import (Profile, as_args,  # noqa: E402
                                                         from_args, add_args, tag)


def _eps(sim, wl):
    plane = td.Box(center=(0, 0, sim.center[2]), size=(0, sim.size[1], sim.size[2]))
    return sim.epsilon(box=plane, coord_key="centers", freq=td.C_0 / wl)


def _same_sim(a, b, wl) -> None:
    for ax in "xyz":
        assert np.array_equal(getattr(a.grid.boundaries, ax), getattr(b.grid.boundaries, ax))
    assert np.array_equal(_eps(a, wl).values, _eps(b, wl).values)


# ---- a plain strip is exactly as before -------------------------------------------------- #
@pytest.mark.parametrize("kw", [dict(), dict(snap=True, snap_offset=1e-6),
                                dict(snap=True, sidewall_deg=85.0),
                                dict(cladding="sio2", si_handle=True)])
def test_default_profile_leaves_the_simulation_untouched(kw):
    base = dm.build_sim(0.7, 1.2, wl0=1.55, steps=30, **kw)
    for p in (None, Profile()):
        assert dm.build_sim(0.7, 1.2, wl0=1.55, steps=30, profile=p, **kw) == base


def test_settings_keys_of_older_designs_are_unchanged():
    """New [stack] settings at their defaults leave every stage's key as in v0.1.5/phase A,
    so finished stages of earlier designs are reused (keys pinned from a4c486e)."""
    from tarang.design.config import DesignConfig
    from tarang.design.stages import settings_key
    want = {"materials": "43d432ed1a29892c", "cross_section": "f1e466bd665d21f1",
            "ring": "4f09d62c40425764", "gap": "8bef2f354c8f9e4f",
            "grating": "0c8c636c8ac73586", "focus": "b1fa6c45bdd2b3dc",
            "device": "8c431494430bbe93"}
    c = DesignConfig()
    assert {s: settings_key(c, s) for s in want} == want


def test_stack_args_and_records_unchanged_for_a_plain_strip():
    from tarang.design.config import DesignConfig, profile_of
    c = DesignConfig()
    assert profile_of(c).is_default and as_args(profile_of(c)) == [] and tag(Profile()) == ""
    assert Profile().record() is None


# ---- the ANT window: exactly the scan's simulation ---------------------------------------- #
def _script_sim(w, recess, off):
    """box_recess_scan.py's patched build_sim, verbatim in its logic."""
    H, BOX, WL0 = 0.40, 4.5, 0.78
    sim = dm.build_sim(H, w, wl0=WL0, steps=50, box=BOX, cladding=None,
                       band=(WL0 - 0.05, WL0 + 0.05), snap=True, snap_offset=off)
    if recess <= 0:
        return sim
    box_medium = next(st.medium for st in sim.structures if isinstance(st.geometry, td.Box)
                      and st.geometry.center[2] + st.geometry.size[2] / 2 <= 1e-9
                      and st.geometry.size[2] > 1)
    air = td.Structure(geometry=td.Box(center=(0, 0, -recess / 2), size=(td.inf, td.inf, recess)),
                       medium=td.Medium(permittivity=1.0))
    ped_w = w - 2 * recess
    ped = td.Structure(geometry=td.Box(center=(0, 0, -recess / 2), size=(td.inf, ped_w, recess)),
                       medium=box_medium)
    gs = sim.grid_spec
    snaps = list(gs.snapping_points or []) + [(None, -ped_w / 2, -recess), (None, ped_w / 2, -recess)]
    return sim.updated_copy(structures=list(sim.structures) + [air, ped],
                            grid_spec=gs.updated_copy(snapping_points=snaps))


@pytest.mark.parametrize("w", [0.50, 0.55, 0.60])
@pytest.mark.parametrize("d", [0.0, 0.020, 0.085, 0.150])
def test_ant_window_builds_the_scans_simulation(w, d):
    """All 12 points of the scan, both snapping sides: same grid, same permittivity in
    every cell, so recess_scan.csv holds for the profile (e.g. W 0.50, d 150 nm: D_TE0
    724.05 ps/(nm km))."""
    for off in (-1e-6, 1e-6):
        new = dm.build_sim(0.40, w, wl0=0.78, steps=50, box=4.5, cladding=None,
                           band=(0.73, 0.83), snap=True, snap_offset=off,
                           profile=Profile(recess_um=d, undercut_um=d))
        _same_sim(new, _script_sim(w, d, off), 0.78)


# ---- the other options --------------------------------------------------------------------- #
def _n_at(sim, y, z, wl=1.55):
    e = _eps(sim, wl)
    yi = int(np.argmin(np.abs(e.coords["y"].values - y)))
    zi = int(np.argmin(np.abs(e.coords["z"].values - z)))
    return float(np.sqrt(np.real(np.squeeze(e.values)[yi, zi])))


def test_rib_slab_is_core_material_across_the_chip():
    sim = dm.build_sim(0.7, 1.2, wl0=1.55, steps=30, snap=True, profile=Profile(slab_um=0.1))
    n_core = dm.n_of(dm.medium_for("si3n4", 1.45, 1.65), 1.55)
    assert _n_at(sim, 1.5, 0.05) == pytest.approx(n_core, abs=1e-3)   # slab beside the core
    assert _n_at(sim, 1.5, 0.30) == pytest.approx(1.0)                # air above the slab
    assert (None, None, 0.1) in list(sim.grid_spec.snapping_points)


def test_finite_slab_is_a_pedestal_of_core_material():
    sim = dm.build_sim(0.57, 1.56, wl0=1.55, steps=30, snap=True,
                       profile=Profile(slab_um=0.05, slab_width_um=2.0))
    n_core = dm.n_of(dm.medium_for("si3n4", 1.45, 1.65), 1.55)
    assert _n_at(sim, 0.9, 0.025) == pytest.approx(n_core, abs=1e-3)  # inside the 2 um slab
    assert _n_at(sim, 1.5, 0.025) == pytest.approx(1.0)               # beyond it: air


def test_pedestal_material_and_cladding_fill():
    p = Profile(recess_um=0.1, undercut_um=0.05, pedestal="si3n4")
    sim = dm.build_sim(0.4, 1.0, wl0=1.55, steps=30, cladding="sio2", profile=p)
    n_core = dm.n_of(dm.medium_for("si3n4", 1.45, 1.65), 1.55)
    n_ox = dm.n_of(dm.medium_for("sio2", 1.45, 1.65), 1.55)
    assert _n_at(sim, 0.0, -0.05) == pytest.approx(n_core, abs=1e-3)  # pedestal material
    assert _n_at(sim, 1.5, -0.05) == pytest.approx(n_ox, abs=1e-3)    # recess filled by cladding


def test_deep_recess_stays_in_the_fine_mesh():
    sim = dm.build_sim(0.4, 1.0, wl0=1.55, steps=40, profile=Profile(recess_um=0.8, undercut_um=0.1))
    o = sim.grid_spec.override_structures[0].geometry
    assert o.center[2] - o.size[2] / 2 <= -0.9 + 1e-9


@pytest.mark.parametrize("p, msg", [
    (Profile(slab_um=-0.1), ">= 0"),
    (Profile(slab_um=0.8), "thinner than the core film"),
    (Profile(slab_um=0.1, recess_um=0.1), "cannot be combined"),
    (Profile(undercut_um=0.05), "needs a recess"),
    (Profile(recess_um=0.1, undercut_um=0.6), "removes the whole pedestal"),
    (Profile(recess_um=5.0), "shallower than the BOX"),
    (Profile(pedestal="si3n4"), "needs a recess"),
    (Profile(slab_um=0.1, slab_width_um=0.5), "must exceed the core's bottom width")])
def test_impossible_profiles_are_refused(p, msg):
    with pytest.raises(ValueError, match=msg):
        p.check(width_bottom=1.0, height=0.7, box=4.0)


def test_command_line_round_trip():
    import argparse
    p = Profile(recess_um=0.085, undercut_um=0.06, pedestal="sio2")
    ap = argparse.ArgumentParser()
    add_args(ap)
    assert from_args(ap.parse_args([str(x) for x in as_args(p)])) == p
    assert from_args(ap.parse_args([])) == Profile()
    assert tag(p).startswith("_profile") and tag(p) != tag(Profile(recess_um=0.085))
    assert Profile.from_record(p.record()) == p


def test_profile_from_design_settings():
    from tarang.design.config import DesignConfig, profile_of
    c = DesignConfig()
    c.stack.box_recess_um, c.stack.box_undercut_um = 0.085, 0.085
    assert profile_of(c) == Profile(recess_um=0.085, undercut_um=0.085)
    assert profile_of(c).label() == "BOX recess 85 nm, undercut 85 nm"


# ---- the stack preview ------------------------------------------------------------------- #
class _UI:
    def __init__(self):
        self.lines = []

    def say(self, s=""):
        self.lines.append(str(s))


def test_stack_preview_is_drawn_from_the_solved_simulation(tmp_path, no_show):
    from tarang.design import stack
    from tarang.design.config import DesignConfig
    c = DesignConfig()
    c.run.wavelength_um, c.stack.box_height_um = 0.78, 4.5
    c.stack.box_recess_um = c.stack.box_undercut_um = 0.085
    ctx = SimpleNamespace(cfg=c, ui=_UI(), band=(0.7, 0.86))
    out = stack.preview(ctx, 0.40, 0.55, tmp_path / "2_stack.png", note="test")
    assert out.exists()
    text = "\n".join(ctx.ui.lines)
    assert "recess   BOX etched 85 nm" in text and "pedestal 380 nm wide" in text
    sim = stack.simulation(ctx, 0.40, 0.55)
    names = [s.name for s in sim.structures]
    assert "box_recess" in names and "box_pedestal" in names


def test_stack_summary_of_a_plain_strip():
    from tarang.design import stack
    from tarang.design.config import DesignConfig
    lines = stack.summary(DesignConfig(), 0.7, 1.2)
    assert lines[0] == "top      air"
    assert lines[1] == "core     si3n4, 700 nm thick × 1200 nm wide"
    assert lines[2] == "BOX      sio2, 4 µm"
    assert not any(l.startswith(("slab", "recess")) for l in lines)


def test_mesh_check_labels_unsnapped_runs_and_keeps_profiles_apart(tmp_path):
    """The convergence plot reads the snapping flag itself (it read the snap-side count,
    never 0, and called every run snapped), and runs of different profiles are separate
    groups."""
    import csv
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    data = tmp_path / "conv.jsonl"
    with open(data, "w", encoding="utf-8") as f:
        for s, D in ((50, 373.99), (70, 376.81), (90, 370.88)):
            base = dict(H=0.7, W=1.0, wl0=1.55, steps=s, bend_radius=None, D_TE0=D,
                        neff_TE=[1.71], ng_TE0=2.22, guided_TE=1)
            f.write(json.dumps(dict(base, snap=False)) + "\n")
            f.write(json.dumps(dict(base, snap=True, snap_sides=2, D_TE0=D + 10,
                                    profile=Profile(recess_um=0.1, undercut_um=0.1).record())) + "\n")
    env = dict(os.environ, PYTHONPATH=str(root), MPLBACKEND="Agg", TARANG_FIGURE_FORMATS="png",
               TARANG_FIGURE_DPI="50")
    r = subprocess.run([sys.executable, "-m", "tarang.waveguide.dispersion_mapping.convergence",
                        "--data", str(data), "--plot-only", "--plots-dir", str(tmp_path / "p")],
                       cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr
    rows = list(csv.DictReader(open(tmp_path / "p" / "mesh_band.csv", encoding="utf-8")))
    assert sorted(row["snapped"] for row in rows) == ["False", "True"]
