"""Shared pytest fixtures and options for the TARANG test suite.

* ``fake_web``: replaces the Tidy3D web API used by ``utilities.cloud_run`` with an
  in-memory fake. It records every call and "runs" a task by writing an empty
  SimulationData, so no test uploads anything or spends FlexCredits.
* ``--run-cloud``: tests marked ``cloud`` submit REAL Tidy3D jobs and spend a few
  hundredths of a FlexCredit. They are skipped unless this option is given and an
  API key is configured.

Markers (declared in pytest.ini):
  slow   needs a local mode solve or a fresh material fit (tens of seconds)
  cloud  real Tidy3D cloud run; opt-in with --run-cloud
"""
from __future__ import annotations

import os

import pytest
import tidy3d as td


def pytest_addoption(parser):
    parser.addoption("--run-cloud", action="store_true", default=False,
                     help="also run tests marked 'cloud' (submits real Tidy3D jobs; costs FlexCredits)")


def _have_tidy3d_key() -> bool:
    from utilities.cloud_run import api_key_source, _LEGACY_KEY_VARS
    return bool(api_key_source() or any(os.environ.get(v) for v in _LEGACY_KEY_VARS))


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-cloud"):
        if not _have_tidy3d_key():
            skip = pytest.mark.skip(reason="--run-cloud given but no Tidy3D API key is configured")
            for item in items:
                if "cloud" in item.keywords:
                    item.add_marker(skip)
        return
    skip = pytest.mark.skip(reason="real cloud run; enable with --run-cloud (costs FlexCredits)")
    for item in items:
        if "cloud" in item.keywords:
            item.add_marker(skip)


# --------------------------------------------------------------------------- #
# Fake Tidy3D web API
# --------------------------------------------------------------------------- #
class FakeWeb:
    """Records calls; 'runs' a task by writing an empty SimulationData."""

    def __init__(self, cost: float = 0.25):
        self.cost = cost
        self.calls: list[tuple] = []
        self.sims: dict = {}
        fake = self

        class Batch:                                    # minimal tidy3d.web.Batch
            def __init__(self, simulations, folder_name="default", verbose=False, **kw):
                self.simulations = dict(simulations)
                fake.calls.append(("batch_init", tuple(self.simulations)))

            def upload(self):
                fake.calls.append(("batch_upload", len(self.simulations)))

            def estimate_cost(self, verbose=False):
                fake.calls.append(("batch_estimate", len(self.simulations)))
                return fake.cost * len(self.simulations)

            def delete(self):
                fake.calls.append(("batch_delete", len(self.simulations)))

            def start(self):
                fake.calls.append(("batch_start", len(self.simulations)))

            def monitor(self):
                fake.calls.append(("batch_monitor", len(self.simulations)))

            def load(self, path_dir=".", **kw):
                fake.calls.append(("batch_load", len(self.simulations)))
                return {name: td.SimulationData(simulation=sim, data=())
                        for name, sim in self.simulations.items()}

        self.Batch = Batch

    def upload(self, sim, task_name=None, folder_name="default", verbose=False, **kw):
        tid = f"tid{len(self.sims)}"
        self.sims[tid] = sim
        self.calls.append(("upload", task_name))
        return tid

    def estimate_cost(self, tid, verbose=False, **kw):
        self.calls.append(("estimate", tid))
        return self.cost

    def start(self, tid, **kw):
        self.calls.append(("start", tid))

    def monitor(self, tid, verbose=False, **kw):
        self.calls.append(("monitor", tid))

    def load(self, tid, path=None, verbose=False, **kw):
        self.calls.append(("load", tid))
        data = td.SimulationData(simulation=self.sims[tid], data=())
        data.to_file(str(path))
        return data

    def delete(self, tid, **kw):
        self.calls.append(("delete", tid))

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]


@pytest.fixture
def fake_web(monkeypatch):
    """A FakeWeb (cost 0.25 per task; change with ``fake_web.cost = ...``)."""
    import utilities.cloud_run as cr

    fw = FakeWeb()
    monkeypatch.setattr(cr, "web", fw)
    monkeypatch.delenv("TARANG_MAX_COST", raising=False)
    return fw


@pytest.fixture
def no_show(monkeypatch):
    """Use the non-interactive matplotlib backend and make plt.show() a no-op."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield
    plt.close("all")
