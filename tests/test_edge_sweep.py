"""Edge-coupler parametric sweep through the batch path (Phase 4). Fake cloud, no credits.

    python -m pytest tests/test_edge_sweep.py -q

The fake cloud returns empty results, so the efficiencies come back as 0 with a
logged warning. These tests check the bookkeeping, not the physics: which points
are sent, what they cost, and what is reused.
"""
import numpy as np
import pytest
import tidy3d as td

import utilities.cloud_run as cr

SIN, SIO2 = td.Medium(permittivity=1.996 ** 2), td.Medium(permittivity=1.444 ** 2)


def sweep(out, lengths, widths, **kw):
    from edge_coupler import parametric_optimization_sweep
    return parametric_optimization_sweep(
        sweep_taper_lengths=lengths, sweep_tip_widths=widths, output_dir=out,
        wg_width=0.8, wg_height=0.7, core_medium=SIN, box_medium=SIO2, wavelength_um=1.55,
        fine_steps_per_wvl=8, coarse_steps_per_wvl=6, out_coupling=True,
        show=False, save=False, **kw)


def test_batch_runs_uncached_points_once_then_reuses_them(tmp_path, fake_web):
    fake_web.cost = 0.1
    eff = sweep(tmp_path, [10.0, 20.0], [0.15, 0.25], max_cost=1.0)
    assert eff.shape == (2, 2)
    assert fake_web.names().count("batch_init") == 1 and fake_web.names().count("batch_start") == 1
    started = [c for c in fake_web.calls if c[0] == "batch_start"][0][1]
    assert started == 4                                  # 2 x 2 points in one batch
    cached = sorted(p.parent.name for p in tmp_path.glob("*/raw_simulation_data.hdf5"))
    assert len(cached) == 4
    assert all((tmp_path / d / cr.SIM_FILE).exists() for d in cached)

    fake_web.calls.clear()
    sweep(tmp_path, [10.0, 20.0], [0.15, 0.25], max_cost=1.0)
    assert "batch_start" not in fake_web.names()         # everything came from the cache

    fake_web.calls.clear()
    sweep(tmp_path, [10.0, 20.0], [0.15, 0.25, 0.35], max_cost=1.0)
    started = [c for c in fake_web.calls if c[0] == "batch_start"][0][1]
    assert started == 2                                   # only the new tip width


def test_batch_over_budget_is_not_started(tmp_path, fake_web):
    fake_web.cost = 0.1
    with pytest.raises(cr.CostLimitExceededError) as err:
        sweep(tmp_path, [10.0, 20.0], [0.15, 0.25], max_cost=0.3)    # 4 x 0.1 > 0.3
    assert err.value.estimated == pytest.approx(0.4)
    assert "batch_start" not in fake_web.names() and "batch_delete" in fake_web.names()
    assert not list(tmp_path.glob("*/raw_simulation_data.hdf5"))


def test_sequential_mode_caps_each_run(tmp_path, fake_web):
    """batching=False runs the points one by one; max_cost then caps each run."""
    fake_web.cost = 0.1
    sweep(tmp_path, [10.0], [0.15, 0.25], batching=False, max_cost=0.15)
    assert fake_web.names().count("start") == 2 and "batch_start" not in fake_web.names()
    fake_web.cost = 0.2
    with pytest.raises(cr.CostLimitExceededError):
        sweep(tmp_path / "b", [10.0], [0.15], batching=False, max_cost=0.15)


def test_sweep_returns_zero_efficiency_for_unreadable_results(tmp_path, fake_web, caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="tarang"):
        eff = sweep(tmp_path, [10.0], [0.15], max_cost=1.0)
    assert np.all(eff == 0.0) and "Failed to extract efficiency" in caplog.text
