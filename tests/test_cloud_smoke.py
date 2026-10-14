"""REAL Tidy3D cloud smoke test (Phase 4). Opt-in: costs a few hundredths of a FlexCredit.

    python -m pytest tests/test_cloud_smoke.py --run-cloud -v

Skipped unless ``--run-cloud`` is given and a Tidy3D API key is configured. Every
other test uses a fake cloud. This one checks that TARANG's real submission
sequence (upload -> cost estimate -> start -> monitor -> download) works against
the live service, and that the second request for the same simulation is served
from the verified cache without contacting the cloud.

The job: a small 2D grating coupler, coarse mesh, capped at max_cost = 0.1
FlexCredits (it is not started if the estimate is higher). A copy of the
result is kept in ``results/cloud_smoke/`` (git-ignored) for later offline
regression tests.
"""
import shutil
from pathlib import Path

import numpy as np
import pytest
import tidy3d as td

import tarang.utilities.cloud_run as cr

pytestmark = pytest.mark.cloud
REPO = Path(__file__).resolve().parents[1]
MAX_COST = 0.1


def _tiny_grating():
    from tarang.grating_coupler.grating_coupler import GratingConfig, GratingCoupler
    from tarang.media import get_medium
    cfg = GratingConfig(wavelength_um=1.55, theta_deg=-12.0, mfd=10.4, source_x=3.84, source_gap=2.0,
                        core_height=0.7, top_cladding_height=2.0, box_height=4.0, handle_height=2.0,
                        etch_depth=0.35, period=0.9, ff=0.5, n_teeth=12, w_grating=10.4,
                        pol_angle=np.pi / 2, buffer=3.0)
    return GratingCoupler.from_config(cfg, core_medium=get_medium("si3n4"), box_medium=get_medium("sio2"),
                                      top_cladding_medium=get_medium("air"), is_3d=False,
                                      fine_steps_per_wvl=10, coarse_steps_per_wvl=8)


def test_real_cloud_run_then_cache_hit(tmp_path):
    gc = _tiny_grating()
    gc.max_cost = MAX_COST
    eff = gc.get_efficiency(output_dir=tmp_path)
    run = gc.last_run
    assert run is not None and not run.from_cache
    assert 0.0 < run.estimated_cost <= MAX_COST
    assert run.path.exists() and (run.path.parent / cr.SIM_FILE).exists()
    assert 0.0 <= eff <= 1.0
    data = td.SimulationData.from_file(str(run.path))
    assert "mode monitor" in {d.monitor.name for d in data.data}   # real monitor data came back

    keep = REPO / "results" / "cloud_smoke" / run.path.parent.name
    keep.mkdir(parents=True, exist_ok=True)
    for f in run.path.parent.iterdir():
        shutil.copy2(f, keep / f.name)

    gc2 = _tiny_grating()                             # same simulation, new object
    assert gc2.get_efficiency(output_dir=tmp_path) == pytest.approx(eff, abs=1e-12)
    assert gc2.last_run.from_cache and gc2.last_run.estimated_cost == 0.0
