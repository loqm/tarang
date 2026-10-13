"""Tests for the nested-sampling backends (Phase 3b). Local only, no FlexCredits:
analytic likelihoods replace FDTD, and the Tidy3D web API is faked.

    python -m pytest tests/test_nested_backends.py -q

The MultiNest test runs only where PyMultiNest and libmultinest load (Linux/macOS
with MultiNest installed); on Windows it is reported as skipped.
"""
import json
import sys
import textwrap

import numpy as np
import pytest

import utilities.nested_backends as nb
from utilities.corner_plot import load_multinest_samples

pytest.importorskip("dynesty")

# 3-parameter Gaussian likelihood inside a box prior [-1, 2]^3 (volume 27):
# Z = (sqrt(2 pi) s)^3 / 27 exactly (the Gaussian lies far inside the box).
MU = np.array([0.2, 0.5, 0.9])
S = 0.1
LO, HI = -1.0, 2.0
EXACT_LNZ = 3 * np.log(np.sqrt(2 * np.pi) * S) - 3 * np.log(HI - LO)


def prior(cube, ndim, nparams):                     # MultiNest convention: in place
    for i in range(ndim):
        cube[i] = LO + cube[i] * (HI - LO)


def loglike(cube, ndim, nparams):
    x = np.array([cube[i] for i in range(ndim)])
    return float(-0.5 * np.sum((x - MU) ** 2) / S ** 2)


class Stop(Exception):
    """Stands in for BudgetExceededError."""


def counting(limit=None, log=None):
    def f(cube, ndim, nparams):
        if log is not None:
            log.append(tuple(round(float(cube[i]), 12) for i in range(ndim)))
        if limit is not None and log is not None and len(log) > limit:
            raise Stop()
        return loglike(cube, ndim, nparams)
    return f


# ---- backend selection ---------------------------------------------------- #
def test_auto_falls_back_to_dynesty(monkeypatch):
    monkeypatch.setattr(nb, "_MULTINEST_OK", False)
    assert nb.resolve_backend("auto") == "dynesty"
    assert "dynesty" in nb.available_backends()


def test_explicit_multinest_unavailable_gives_clear_error(monkeypatch):
    monkeypatch.setattr(nb, "_MULTINEST_OK", False)
    with pytest.raises(ImportError, match="dynesty"):
        nb.resolve_backend("multinest")


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        nb.resolve_backend("emcee")


def test_probe_survives_pymultinest_systemexit(tmp_path, monkeypatch):
    """Without libmultinest, `import pymultinest` raises SystemExit (it would end
    Python). The probe must report 'unavailable' instead."""
    pkg = tmp_path / "pymultinest"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(textwrap.dedent("""
        print("ERROR: Could not load MultiNest library")
        raise SystemExit(1)
    """))
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "pymultinest", raising=False)
    monkeypatch.delitem(sys.modules, "pymultinest.run", raising=False)
    monkeypatch.setattr(nb, "_MULTINEST_OK", None)
    assert nb.multinest_available() is False


# ---- dynesty correctness -------------------------------------------------- #
def test_dynesty_recovers_analytic_evidence_and_posterior(tmp_path):
    base = str(tmp_path / "chains" / "g_")
    r = nb.run_nested(loglike, prior, 3, base, backend="dynesty", n_live=80,
                      evidence_tolerance=0.1, seed=7, resume=False)
    assert r.completed and r.backend == "dynesty"
    assert abs(r.log_evidence - EXACT_LNZ) < 3 * r.log_evidence_err + 0.2

    # MultiNest-format files are read by the existing TARANG loader
    chain = load_multinest_samples(tmp_path, source="weighted")
    w = chain.weights / chain.weights.sum()
    mean = np.average(chain.samples, weights=w, axis=0)
    std = np.sqrt(np.average((chain.samples - mean) ** 2, weights=w, axis=0))
    assert np.allclose(mean, MU, atol=0.03)
    assert np.allclose(std, S, rtol=0.3)
    eq = load_multinest_samples(tmp_path, source="equal")
    assert eq.samples.shape[1] == 3 and np.allclose(eq.samples.mean(0), MU, atol=0.04)
    summary = json.loads((tmp_path / "chains" / "g_dynesty_summary.json").read_text())
    assert summary["seed"] == 7


def test_same_seed_same_run(tmp_path):
    a = nb.run_nested(loglike, prior, 3, str(tmp_path / "a" / "g_"), backend="dynesty",
                      n_live=40, evidence_tolerance=0.5, seed=3, resume=False)
    b = nb.run_nested(loglike, prior, 3, str(tmp_path / "b" / "g_"), backend="dynesty",
                      n_live=40, evidence_tolerance=0.5, seed=3, resume=False)
    assert a.log_evidence == b.log_evidence and a.n_calls == b.n_calls


def test_interrupted_run_resumes_by_exact_replay(tmp_path):
    """A stop (budget) mid-run, then resume: the resumed run first re-requests
    exactly the same points, in the same order (cache hits for FDTD), and ends
    with the same result as an uninterrupted run."""
    ref = nb.run_nested(loglike, prior, 3, str(tmp_path / "ref" / "g_"), backend="dynesty",
                        n_live=40, evidence_tolerance=0.5, seed=11, resume=False)

    base = str(tmp_path / "run" / "g_")
    first = []
    with pytest.raises(Stop):
        nb.run_nested(counting(limit=150, log=first), prior, 3, base, backend="dynesty",
                      n_live=40, evidence_tolerance=0.5, seed=11, resume=False)
    first = first[:150]
    # A snapshot was written for inspection / live corner plots
    assert (tmp_path / "run" / "g_phys_live.points").exists()

    second = []
    res = nb.run_nested(counting(log=second), prior, 3, base, backend="dynesty",
                        n_live=40, evidence_tolerance=0.5, resume=True)   # seed from file
    assert second[:150] == first
    assert res.seed == 11
    assert res.log_evidence == pytest.approx(ref.log_evidence, abs=1e-12)


def test_mixing_backends_in_one_directory_is_refused(tmp_path):
    base = str(tmp_path / "g_")
    (tmp_path / "g_backend.json").write_text(json.dumps({"backend": "multinest"}))
    with pytest.raises(ValueError, match="multinest"):
        nb.run_nested(loglike, prior, 3, base, backend="dynesty", n_live=20, resume=True)

    base2 = str(tmp_path / "h_")
    (tmp_path / "h_resume.dat").write_text("1")          # an old MultiNest run
    with pytest.raises(ValueError, match="MultiNest"):
        nb.run_nested(loglike, prior, 3, base2, backend="dynesty", n_live=20, resume=True)


@pytest.mark.skipif(not nb.multinest_available(), reason="PyMultiNest/libmultinest not installed")
def test_multinest_backend_same_interface(tmp_path):
    base = str(tmp_path / "chains" / "g_")
    r = nb.run_nested(loglike, prior, 3, base, backend="multinest", n_live=80,
                      evidence_tolerance=0.1, seed=5, resume=False)
    assert r.backend == "multinest" and r.n_calls > 0
    assert abs(r.log_evidence - EXACT_LNZ) < 3 * r.log_evidence_err + 0.5
    chain = load_multinest_samples(tmp_path, source="weighted")
    w = chain.weights / chain.weights.sum()
    assert np.allclose(np.average(chain.samples, weights=w, axis=0), MU, atol=0.03)


# ---- grating-coupler optimizer, end to end ---------------------------------- #
FIXED = dict(wavelength_um=1.55, theta_deg=-12.0, mfd=10.4, source_gap=2.0,
             core_height=0.8, top_cladding_height=2.0, box_height=4.0, handle_height=8.0,
             w_grating=10.4, buffer=3.0, pol_angle=np.pi / 2, fine_steps=8, coarse_steps=6)
BOUNDS = dict(period=[0.80, 0.84], fill_factor=[0.48, 0.55], etch_depth=[0.48, 0.55],
              num_periods=[18.0, 26.0], source_x=[3.5, 4.2])


def _materials():
    import tidy3d as td
    m = lambda n: td.Medium(permittivity=n ** 2)
    return dict(core_medium=m(1.996), box_medium=m(1.444), handle_medium=None, top_cladding_medium=m(1.0))


def _optimizer(tmp_path, **kw):
    from grating_coupler.nested_sampler import GratingCouplerBayesianOptimizer
    return GratingCouplerBayesianOptimizer(fixed_params=FIXED, materials=_materials(),
                                           output_dir=tmp_path, param_bounds=BOUNDS, sigma=0.02,
                                           is_3d=False, show_corner_plots=False, **kw)


def test_optimizer_search_with_dynesty(tmp_path, monkeypatch):
    """execute_search(backend='dynesty') end to end with an analytic efficiency
    surface (peak 60 % at period 0.82 um, FF 0.51) instead of FDTD."""
    opt = _optimizer(tmp_path)

    def fake_loglike(cube, ndim, nparams):
        eff = 0.6 * np.exp(-((cube[0] - 0.82) / 0.01) ** 2 - ((cube[1] - 0.51) / 0.02) ** 2)
        return -0.5 * ((1.0 - eff) / opt.sigma) ** 2

    monkeypatch.setattr(opt, "log_likelihood", fake_loglike)
    out = opt.execute_search(n_live_points=30, evidence_tolerance=0.5, backend="dynesty", seed=2)
    assert out["status"] == "CONVERGED"
    assert out["sampler_backend"] == "dynesty" and out["sampler_seed"] == 2
    assert np.isfinite(out["log_evidence"])
    v = out["parameter_space_variance"]
    assert v["period"]["marginalized_mean"] == pytest.approx(0.82, abs=0.005)
    assert v["fill_factor"]["marginalized_mean"] == pytest.approx(0.51, abs=0.01)
    assert out["max_efficiency"] == pytest.approx(0.6, abs=0.03)


def test_budget_pause_and_resume_reuses_cache(tmp_path, fake_web):
    """Budget stop after 3 paid runs; resuming with a higher limit replays those
    3 points from the cache (no new uploads) and pays only for new points."""
    fw = fake_web
    fw.cost = 0.125

    opt = _optimizer(tmp_path)
    out = opt.execute_search_with_budget(credit_limit=0.375, n_live_points=12,
                                         evidence_tolerance=0.5, backend="dynesty", seed=4)
    assert out["status"] == "PAUSED_BUDGET_EXCEEDED"
    assert fw.names().count("start") == 3
    assert opt.total_credits_used == pytest.approx(0.375)

    fw.calls.clear()
    opt2 = _optimizer(tmp_path)                       # new session, same directory
    out2 = opt2.execute_search_with_budget(credit_limit=0.75, n_live_points=12,
                                           evidence_tolerance=0.5, backend="dynesty")
    assert out2["status"] == "PAUSED_BUDGET_EXCEEDED"
    assert fw.names().count("start") == 3             # 3 NEW runs only; first 3 were cache hits
    assert opt2.total_credits_used == pytest.approx(0.75)
