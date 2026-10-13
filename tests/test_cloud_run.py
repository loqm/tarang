"""Tests for the shared cloud-run cache (audit C1, C6, R4). No network, no FlexCredits:
the Tidy3D web API is replaced by an in-memory fake.

    python -m pytest tests/test_cloud_run.py -q
"""
import json
import logging

import numpy as np
import pytest
import tidy3d as td

import utilities.cloud_run as cr
from grating_coupler.grating_coupler import GratingConfig, GratingCoupler


def _medium(n):
    return td.Medium(permittivity=n ** 2)


SIN, SIO2_A, SIO2_B, AIR = _medium(1.996), _medium(1.444), _medium(1.420), _medium(1.0)

CFG = GratingConfig(
    wavelength_um=1.55, theta_deg=-12.0, mfd=10.4, source_x=3.84, source_gap=2.0,
    core_height=0.8, top_cladding_height=2.0, box_height=4.0, handle_height=8.0,
    etch_depth=0.5, period=0.82, ff=0.51, n_teeth=22, w_grating=10.4,
    pol_angle=np.pi / 2, buffer=3.0,
)


def make_gc(box=SIO2_A):
    return GratingCoupler.from_config(CFG, SIN, box, None, AIR,
                                      fine_steps_per_wvl=10, coarse_steps_per_wvl=8)


# ---- fingerprint ---------------------------------------------------------- #
def test_fingerprint_tracks_media_and_ignores_version():
    a, a2, b = make_gc(SIO2_A), make_gc(SIO2_A), make_gc(SIO2_B)
    assert cr.sim_fingerprint(a.sim) == cr.sim_fingerprint(a2.sim)
    assert cr.sim_fingerprint(a.sim) != cr.sim_fingerprint(b.sim)
    d = json.loads(a.sim.model_dump_json())
    assert "version" in d and "version" not in cr._sim_dict(a.sim)


def test_c1_grating_cache_key_includes_media():
    """Before v0.1.0 both coupler variants mapped to the same directory (the bug)."""
    a, b = make_gc(SIO2_A), make_gc(SIO2_B)
    assert a._legacy_dir_name() == b._legacy_dir_name()     # documents the old bug
    assert a.get_dir_name() != b.get_dir_name()              # fixed


def test_dir_names_are_short_enough_for_windows():
    assert len(make_gc().get_dir_name()) < 80


# ---- run_cached ------------------------------------------------------------ #
def test_miss_then_hit(tmp_path, fake_web):
    gc = make_gc()
    gc._run_sim(tmp_path)
    assert fake_web.names() == ["upload", "estimate", "start", "monitor", "load"]
    d = tmp_path / gc.get_dir_name()
    assert (d / cr.DATA_FILE).exists() and (d / cr.SIM_FILE).exists()
    assert not (d / "partial_download.hdf5").exists()
    assert gc.last_run.estimated_cost == pytest.approx(0.25) and not gc.last_run.from_cache

    fake_web.calls.clear()
    gc2 = make_gc()
    assert gc2.is_cached(tmp_path)
    gc2._run_sim(tmp_path)
    assert fake_web.calls == [] and gc2.last_run.from_cache


def test_r4_task_name_carries_fingerprint(tmp_path, fake_web):
    gc = make_gc()
    gc._run_sim(tmp_path)
    task = fake_web.calls[0][1]
    fp = cr.sim_fingerprint(gc.sim)[: cr.FINGERPRINT_LEN]
    assert task.startswith("gc_") and task.endswith(fp) and len(task) <= 80


def test_c6_max_cost_blocks_before_start(tmp_path, fake_web):
    gc = make_gc()
    gc.max_cost = 0.1
    with pytest.raises(cr.CostLimitExceededError) as ei:
        gc._run_sim(tmp_path)
    assert ei.value.estimated == pytest.approx(0.25)
    assert "start" not in fake_web.names() and "delete" in fake_web.names()
    assert not (tmp_path / gc.get_dir_name() / cr.DATA_FILE).exists()


def test_c6_env_max_cost(tmp_path, fake_web, monkeypatch):
    monkeypatch.setenv("TARANG_MAX_COST", "0.2")
    with pytest.raises(cr.CostLimitExceededError):
        make_gc()._run_sim(tmp_path)
    monkeypatch.setenv("TARANG_MAX_COST", "1.0")
    make_gc()._run_sim(tmp_path)
    assert "start" in fake_web.names()


def test_cost_error_is_not_a_runtime_error():
    """Sweeps catch RuntimeError per point; a budget stop must not be swallowed."""
    assert not issubclass(cr.CostLimitExceededError, RuntimeError)


def test_legacy_cache_reused_only_if_simulation_matches(tmp_path, fake_web, caplog):
    a, b = make_gc(SIO2_A), make_gc(SIO2_B)
    legacy = tmp_path / a._legacy_dir_name()
    legacy.mkdir()
    td.SimulationData(simulation=a.sim, data=()).to_file(str(legacy / cr.DATA_FILE))

    # Same simulation -> reused from the old directory, no cloud call.
    a._run_sim(tmp_path)
    assert fake_web.calls == [] and a.last_run.path.parent == legacy

    # Different SiO2 -> old results ignored with an explanation, fresh run elsewhere.
    with caplog.at_level(logging.WARNING, logger="tarang"):
        b._run_sim(tmp_path)
    assert "start" in fake_web.names()
    assert b.last_run.path.parent != legacy
    assert "structures" in caplog.text
    assert (legacy / cr.DATA_FILE).exists()          # old results untouched


def test_partial_download_is_not_a_cache_hit(tmp_path, fake_web):
    gc = make_gc()
    d = tmp_path / gc.get_dir_name()
    d.mkdir()
    (d / "partial_download.hdf5").write_bytes(b"truncated")
    assert not gc.is_cached(tmp_path)


def test_nested_sampler_budget_uses_single_upload(tmp_path, fake_web):
    """The sampler used to upload each design twice (once only to price it)."""
    pytest.importorskip("matplotlib")
    from grating_coupler.nested_sampler import GratingCouplerBayesianOptimizer, BudgetExceededError

    fixed = dict(wavelength_um=1.55, theta_deg=-12.0, mfd=10.4, source_gap=2.0,
                 core_height=0.8, top_cladding_height=2.0, box_height=4.0, handle_height=8.0,
                 w_grating=10.4, buffer=3.0, pol_angle=np.pi / 2, fine_steps=10, coarse_steps=8)
    mats = dict(core_medium=SIN, box_medium=SIO2_A, handle_medium=None, top_cladding_medium=AIR)
    bounds = dict(period=[0.80, 0.84], fill_factor=[0.48, 0.55], etch_depth=[0.48, 0.55],
                  num_periods=[18.0, 26.0], source_x=[3.5, 4.2])
    opt = GratingCouplerBayesianOptimizer(fixed_params=fixed, materials=mats,
                                          output_dir=tmp_path, param_bounds=bounds,
                                          sigma=0.02, is_3d=False, credit_limit=0.3)
    cube = [0.82, 0.51, 0.5, 22.0, 3.84]
    opt.log_likelihood(list(cube), 5, 5)
    assert fake_web.names().count("upload") == 1
    assert opt.total_credits_used == pytest.approx(0.25)

    fake_web.calls.clear()
    opt.log_likelihood(list(cube), 5, 5)             # cached: free, no upload
    assert fake_web.calls == []

    with pytest.raises(BudgetExceededError):        # 0.05 left < 0.25 needed
        opt.log_likelihood([0.81, 0.50, 0.5, 20.0, 3.9], 5, 5)
    assert "start" not in [c[0] for c in fake_web.calls]


# ---- API key discovery ------------------------------------------------------ #
@pytest.fixture
def no_key(monkeypatch):
    """No key anywhere: clean environment and a stub Tidy3D config without a key."""
    import types
    for var in ("SIMCLOUD_APIKEY", "TIDY3D_API_KEY", "TIDY3D_KEY"):
        monkeypatch.delenv(var, raising=False)
    stub = types.SimpleNamespace(config=types.SimpleNamespace(web=types.SimpleNamespace(apikey=None)))
    monkeypatch.setattr(cr, "td", stub)
    return monkeypatch


def test_api_key_found_in_simcloud_variable(no_key):
    no_key.setenv("SIMCLOUD_APIKEY", "abc")
    assert cr.api_key_source() == "SIMCLOUD_APIKEY"


def test_api_key_found_in_tidy3d_config(no_key):
    cr.td.config.web.apikey = "abc"                     # what `tidy3d configure` provides
    assert cr.api_key_source() == "tidy3d config"


def test_old_tarang_key_variable_is_passed_on(no_key, caplog):
    """Earlier TARANG docs said TIDY3D_API_KEY, which Tidy3D never reads."""
    import logging
    import os
    no_key.setenv("TIDY3D_API_KEY", "abc")
    assert cr.api_key_source() is None
    with caplog.at_level(logging.WARNING, logger="tarang"):
        assert cr.ensure_api_key() == "TIDY3D_API_KEY"
    assert os.environ["SIMCLOUD_APIKEY"] == "abc" and "SIMCLOUD_APIKEY" in caplog.text


def test_missing_key_warns_with_the_fix(no_key, caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="tarang"):
        assert cr.ensure_api_key() is None
    assert "tidy3d configure" in caplog.text
