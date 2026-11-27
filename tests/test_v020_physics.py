"""v0.2.0, phase C: mode families from the field, the mesh-index rule, default ranges for
high-contrast cores, and Kerr indices of the command-line tools (audit items H1-H4, V3)."""
from __future__ import annotations

import numpy as np
import pytest

td = pytest.importorskip("tidy3d")


# ---- mode families ------------------------------------------------------------------------ #
def test_nodes_ignore_the_tails():
    from tarang.waveguide.dispersion_mapping.core import _nodes
    x = np.linspace(-1, 1, 201)
    assert _nodes(np.exp(-x ** 2 / 0.1)) == 0
    assert _nodes(x * np.exp(-x ** 2 / 0.1)) == 1
    assert _nodes((4 * x ** 2 - 1) * np.exp(-x ** 2 / 0.5)) == 2
    tail = np.exp(-x ** 2 / 0.1) + 0.02 * np.sin(40 * x)          # ripple below 10 % of peak
    assert _nodes(tail) == 0


def _families(W):
    from tidy3d.plugins.mode import ModeSolver
    from tarang.waveguide.dispersion_mapping import core as dm
    sim = dm.build_sim(0.40, W, wl0=0.78, steps=50, box=4.5, cladding=None, band=(0.73, 0.83),
                       snap=True, snap_offset=1e-6)
    plane = td.Box(center=(0, 0, sim.center[2]), size=(0, sim.size[1], sim.size[2]))
    d = ModeSolver(simulation=sim, plane=plane, freqs=[td.C_0 / 0.78],
                   mode_spec=td.ModeSpec(num_modes=6, target_neff=2.0)).solve()
    n_cut = 1.45367
    out = []
    for m in range(6):
        n = float(np.real(d.n_eff.isel(f=0, mode_index=m)))
        if n > n_cut:
            out.append((n, dm._family(d, m, 0)))
    return [f for _, f in sorted(out, reverse=True)]


def test_ant_widths_count_modes_by_family():
    """The ANT window cross-sections (H 400 nm, 780 nm, air top), one snapping side (+1e-6):
    the guided modes by field grow with the width, 2 -> 3 -> 4. The third is TM10 (Ez with
    one node across the width), the fourth, at 0.60 um, TE10. (At 0.50 um TM10 is just at
    its cutoff: guided on the other snapping side only; see the pairing test below.)"""
    assert _families(0.50) == ["TE00", "TM00"]
    assert _families(0.55) == ["TE00", "TM00", "TM10"]
    assert _families(0.60) == ["TE00", "TM00", "TM10", "TE10"]


# ---- the mesh-index rule (H1, H4) ---------------------------------------------------------- #
@pytest.mark.parametrize("key, wl, want", [("si3n4", 1.55, 2.0), ("si3n4", 0.78, 2.0),
                                           ("ta2o5", 1.064, 2.0), ("gap", 1.55, 3.0551),
                                           ("al0.2ga0.8as", 1.55, 3.2654)])
def test_mesh_index_keeps_si3n4_and_follows_high_index_cores(key, wl, want):
    from tarang.geometries._modes import mesh_index
    from tarang.waveguide.dispersion_mapping.core import medium_for
    m = medium_for(key, wl * 0.9, wl * 1.1)
    assert mesh_index(m, td.C_0 / wl) == pytest.approx(want, abs=2e-3)


def test_no_silent_index_fall_back_left():
    """The ring and gap geometries read an attribute fitted media do not have and fell back
    to n = 2.0 for every core; they now use mesh_index (no try/except around it)."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "tarang"
    hits = [p.name for p in root.rglob("*.py") if "n_complex(freq0)" in p.read_text(encoding="utf-8")]
    assert hits == []


# ---- default ranges (H2) ------------------------------------------------------------------- #
def test_default_ranges_keep_si3n4_and_shrink_for_high_contrast():
    from tarang.design.config import DesignConfig
    from tarang.design.stages import _heights_default, _range_scale, _widths_default
    c = DesignConfig()
    assert _range_scale(c) == 1.0
    assert (_heights_default(c)[0], _heights_default(c)[-1]) == (0.5, 0.9)     # as before
    assert (_widths_default(c)[0], _widths_default(c)[-1]) == (0.6, 1.8)
    c.materials.core = "ta2o5"
    assert _range_scale(c) == 1.0
    c.materials.core = "gap"
    s = _range_scale(c)
    assert s == pytest.approx(np.sqrt(4 - 1.444 ** 2) / np.sqrt(3.0551 ** 2 - 1.444 ** 2), abs=0.01)
    assert _heights_default(c)[-1] < 0.5 and _widths_default(c)[-1] < 1.0


# ---- Kerr index of the command-line tools (V3) -------------------------------------------- #
def test_command_line_n2_comes_from_the_records_core():
    from tarang.media.materials import n2_of_core
    assert n2_of_core("si3n4") == 2.4e-19
    assert n2_of_core("gap") == 1.1e-17
    assert n2_of_core("gap", 5e-18) == 5e-18
    with pytest.raises(SystemExit, match="--n2"):
        n2_of_core("si_franta")


def test_snapping_sides_are_paired_by_family_not_by_rank():
    """W = 0.50 um of the ANT scan: TM10 is guided on one snapping side (1.4682) and absent
    on the other, whose third mode is a TE01 at 1.4513. Pairing by rank averaged the two into
    a 'second guided TE mode' at 1.4598 (the scan's guided_TE = 2); pairing by family keeps
    TM10 as it is, guided on one side only, and TE01 below cutoff."""
    from tarang.waveguide.dispersion_mapping.core import _pair_modes
    minus = [(1.7687, 1.0, "TE00"), (1.7556, 0.01, "TM00"), (1.4682, 0.31, "TM10"),
             (1.4512, 0.97, "TE00"), (1.4505, 0.0, "TM00")]
    plus = [(1.7558, 0.99, "TE00"), (1.7442, 0.01, "TM00"), (1.4513, 0.93, "TE01"),
            (1.4505, 0.0, "TM00"), (1.4486, 1.0, "TE10")]
    out = _pair_modes([minus, plus])
    n_cut = 1.45367
    guided = [(n, fam, both) for n, tf, fam, both in out if n > n_cut]
    assert [(f, b) for _, f, b in guided] == [("TE00", True), ("TM00", True), ("TM10", False)]
    assert [n for n, _, _ in guided] == pytest.approx([1.76225, 1.7499, 1.4682])
    tm10 = next(t for t in out if t[2] == "TM10")
    assert tm10[1] == 0.31                                   # its own TE fraction: TM-like
    assert all(not (fam == "TE01" and n > n_cut) for n, _, fam, _ in out)
