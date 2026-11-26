"""v0.1.6 fixes. Dispersion maps with no anomalous point: the colour scale of the D map
used to be zero for such a map and matplotlib refused it, so the cross-section stage
stopped with a traceback instead of reporting that no target geometry exists."""
from __future__ import annotations

import numpy as np


def test_colour_limit_with_anomalous_points_is_unchanged():
    from tarang.waveguide.dispersion_mapping.maps import d_colour_limit
    D = np.array([[-2400.0, -150.0], [40.0, 120.0]])
    assert d_colour_limit(D) == 120.0 * 1.1          # as before: the largest anomalous D
    assert d_colour_limit(D, 200.0) == 200.0 * 1.1   # or the margin, if larger


def test_colour_limit_of_an_all_normal_map_is_positive():
    from tarang.waveguide.dispersion_mapping.maps import d_colour_limit
    D = np.array([[-1247.7, -155.4], [-399.1, np.nan]])     # oxide-clad 800 nm film at 780 nm
    assert d_colour_limit(D) == 2 * 155.4                   # twice the least normal D
    assert d_colour_limit(np.full((2, 2), np.nan)) == 1.0   # nothing solved: still drawable
    assert d_colour_limit(np.zeros((1, 2))) == 1.0


def test_all_normal_map_is_drawn(tmp_path):
    """The design map and map 1 draw without error when every D is negative."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from tarang.waveguide.dispersion_mapping.maps import d_colour_limit
    D = np.array([[-1247.7, -155.4, -183.6], [-1235.5, -156.0, -178.2]])
    lim = d_colour_limit(D)
    fig, ax = plt.subplots()
    ax.pcolormesh([0.3, 0.6, 0.7], [0.78, 0.80], np.clip(D, -lim, lim), cmap="RdBu_r",
                  norm=TwoSlopeNorm(0, -lim, lim), shading="nearest")
    fig.savefig(tmp_path / "map.png")
    plt.close(fig)
    assert (tmp_path / "map.png").stat().st_size > 0


# ------------------------------------------------- a diverged edge-coupler facet run
def test_diverged_edge_facet_run_is_reported_and_not_used(tmp_path, monkeypatch, capsys):
    """The 3D FDTD facet run of the edge stage was read without the divergence check that
    every other FDTD result gets; a diverged run would have been analysed as if it were
    physical. It is now reported (with the link to its record) and the butt-coupling
    estimate stands instead."""
    import math
    import types

    import pytest
    pytest.importorskip("xarray")
    import tidy3d as td
    from test_phase14b import _edge_ctx, _fake_facet_data, _patched_edge
    from tarang.design import checks, taper
    from tarang.utilities import cloud_run
    edge = _patched_edge(monkeypatch)
    ctx = _edge_ctx(tmp_path, edge__facet_widths_um=[2.0, 4.0, 6.0])
    monkeypatch.setattr(checks, "_run_batch", lambda *a, **k: True)
    monkeypatch.setattr(cloud_run, "find_cached", lambda sim, folder, name: tmp_path / "fake.hdf5")
    monkeypatch.setattr(cloud_run, "_data_class", lambda sim: types.SimpleNamespace(
        from_file=lambda p: "eme"))

    def fake_eme(data, L0, scales):
        L = [L0 * s for s in scales]
        loss = [2.0 * math.exp(-x / 6.0) + 1e-3 for x in L]
        return dict(length_um=L, transmission=[10 ** (-v / 10) for v in loss], loss_db=loss)
    monkeypatch.setattr(taper, "taper_transmission", fake_eme)

    class DivergedData(dict):                                   # the saved run says it diverged
        diverged = True

    def diverged(cls, p):
        return DivergedData(_fake_facet_data(0.395, 0.148))
    monkeypatch.setattr(td.SimulationData, "from_file", classmethod(diverged))
    res = edge.stage_edge(ctx)
    assert res["fdtd"] is None                                   # the diverged run is not used
    assert res["eta_coupler"] == pytest.approx(res["eta_free"] * 10 ** (-res["taper_loss_db"] / 10))
    assert not (ctx.out / "figures" / "12c_edge_3d.png").exists()
    assert "diverged (its results are not used)" in capsys.readouterr().out
