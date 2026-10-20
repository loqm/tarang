"""Shared pytest fixtures and options for the TARANG test suite.

* ``fake_web``: replaces the Tidy3D web API used by ``tarang.utilities.cloud_run`` with an
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

import numpy as np
import pytest
import tidy3d as td


def pytest_addoption(parser):
    parser.addoption("--run-cloud", action="store_true", default=False,
                     help="also run tests marked 'cloud' (submits real Tidy3D jobs; costs FlexCredits)")


def _have_tidy3d_key() -> bool:
    from tarang.utilities.cloud_run import api_key_source, _LEGACY_KEY_VARS
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
        self.status: dict = {}          # task id -> Tidy3D status ("running", "success", ...)
        self.tasks: list[dict] = []     # what get_tasks() lists
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
                self.jobs = {}
                for name, sim in self.simulations.items():
                    tid = f"tid{len(fake.sims)}"
                    fake.sims[tid] = sim
                    fake.status[tid] = "running"
                    self.jobs[name] = type("Job", (), {"task_id": tid})()

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
        self.status[tid] = "running"

    def get_info(self, tid, verbose=False, **kw):
        self.calls.append(("get_info", tid))
        return type("TaskInfo", (), {"status": self.status.get(tid, "deleted")})()

    def real_cost(self, tid, verbose=False, **kw):
        return self.cost

    def get_tasks(self, num_tasks=None, order="new", folder="default"):
        self.calls.append(("get_tasks", folder))
        return list(self.tasks)

    def monitor(self, tid, verbose=False, **kw):
        self.calls.append(("monitor", tid))

    def load(self, tid, path=None, verbose=False, **kw):
        self.calls.append(("load", tid))
        self.status[tid] = "success"
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
    import tarang.utilities.cloud_run as cr

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


# ---- fast stand-in for the local mode solver in workflow tests -------------------------- #
def _fake_mode(neff, te, shift=0.0, ng=2.22):
    y = np.linspace(-3.0, 3.0 + shift, 61)
    z = np.linspace(-2.0, 2.7, 48)
    Y, Z = np.meshgrid(y, z, indexing="ij")
    g = np.exp(-((Y - shift / 2) / 0.6) ** 2 - ((Z - 0.35) / 0.45) ** 2)
    main, weak = (g + 0j, 0.05 * g + 0j)
    f = {"Ex": 0.1j * g, "Ey": main if te else weak, "Ez": weak if te else main}
    return dict(index=0, neff=neff, k=0.0, te=0.99 if te else 0.01, fields=f, y=y, z=z,
                aeff=0.75, yc=0.0, zc=0.35, ng=ng)


@pytest.fixture(autouse=True)
def fast_local_modes(request, monkeypatch):
    """Workflow tests use a synthetic mode solver (milliseconds instead of minutes); tests
    marked ``real_modes`` keep Tidy3D's."""
    if request.node.get_closest_marker("real_modes"):
        return
    import math
    try:
        from tarang.design import localmodes as lm
    except Exception:                       # the design package is not importable here
        return

    def modes(ctx, H, W, bend_radius=None, num_modes=4, group_index=False):
        return [_fake_mode(1.7066, True, ng=2.2227), _fake_mode(1.6812, False, ng=2.2252)]

    def supermodes(ctx, H, W, gap, steps=None):
        dn = 3.5e-3 * math.exp(-5.4 * (gap - 0.47))
        s = W + gap
        return dict(even=_fake_mode(1.7064, True), odd=_fake_mode(1.7064 - dn, True), dn=dn,
                    kappa_per_um=math.pi * dn / 1.55, coupling_length_um=1.55 / (2 * dn),
                    shift_um=s, gap_um=gap,
                    cores=[(-s / 2 - W / 2, 0.0, W, H), (s / 2 - W / 2, 0.0, W, H)])

    def coupling(ctx, H, W, radius_um, gaps, steps=None):
        rows = []
        for g in gaps:
            k = math.pi * supermodes(ctx, H, W, g)["dn"] / 1.55
            th = lm.theta_per_pass(k, radius_um, 5.4)
            rows.append(dict(gap_um=g, dn=k * 1.55 / math.pi, kappa_per_um=k, theta=th,
                             coupled_power=math.sin(th) ** 2))
        th0 = rows[0]["theta"] * math.exp(5.4 * rows[0]["gap_um"])
        return dict(rows=rows, gamma_per_um=5.4, theta0=th0)
    monkeypatch.setattr(lm, "cross_section_modes", modes)
    monkeypatch.setattr(lm, "coupler_supermodes", supermodes)
    monkeypatch.setattr(lm, "coupling_from_supermodes", coupling)


@pytest.fixture(autouse=True)
def fast_figures(request, monkeypatch):
    """Figures at 300 dpi as PNG + PDF cost ~0.6 s each; tests write them at 72 dpi as PNG
    only, except tests marked ``real_figures`` (which check the publication settings)."""
    if request.node.get_closest_marker("real_figures"):
        yield
        return
    from tarang.design import visuals
    monkeypatch.setitem(visuals.SETTINGS, "formats", ["png"])
    monkeypatch.setitem(visuals.SETTINGS, "dpi", 72)
    monkeypatch.setattr(visuals, "configure", lambda formats=None, dpi=None: None)
    yield
