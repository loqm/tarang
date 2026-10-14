"""
tarang.utilities.nested_backends

One entry point, ``run_nested()``, for nested sampling with either

* **MultiNest** (via PyMultiNest): a Fortran library. It is fast and well tested,
  but it needs the native ``libmultinest``, which is readily available only on
  Linux and macOS (conda-forge).
* **dynesty**: pure Python (``pip install dynesty``), so it runs on Windows as well.

Both backends use the MultiNest calling convention for the user's callbacks:

    prior(cube, ndim, nparams)          # transforms the unit cube IN PLACE
    loglike(cube, ndim, nparams) -> float

so existing TARANG optimizers work unchanged. Both also leave **MultiNest-format
files** under ``basename`` (``<basename>.txt``, ``post_equal_weights.dat``,
``phys_live.points``), so ``tarang.utilities.corner_plot`` and
``nested_sampling_post_process`` read either backend's output the same way.

Background: nested sampling in one paragraph
--------------------------------------------
Bayes' theorem gives the posterior p(θ|D) = L(θ) π(θ) / Z, where the evidence
Z = ∫ L(θ) π(θ) dθ. Nested sampling keeps ``n_live`` points drawn from the prior.
At each step it discards the point with the lowest likelihood L* and replaces it
with a new prior draw that has L > L*. The prior volume enclosed by the
likelihood contour shrinks by ≈ exp(-1/n_live) per step, so Z ≈ Σ L_i ΔX_i.
The discarded points, weighted by L_i ΔX_i / Z, are posterior samples. The run
stops when the evidence the live points could still add, Δln Z, falls below
``evidence_tolerance``. Every likelihood call is one FDTD simulation here, so the
number of calls is the cost.

Resuming dynesty runs
---------------------
MultiNest resumes from its own ``resume.dat``. dynesty's checkpoint pickles the
likelihood function itself, which here holds the optimizer's credit ledger, so
restoring it would revive a stale ledger. TARANG resumes dynesty by **replay**
instead. The seed is stored in ``<basename>dynesty_state.json`` and the run
restarts with the same seed. With identical likelihood values the sampler makes
identical proposals, and every point evaluated before the stop is now a
verified cache hit (no cloud run, no cost). The run therefore retraces its path
for free and continues from where it stopped.

Exposes: available_backends(), multinest_available(), resolve_backend(),
run_nested(), NestedRunResult, DYNESTY_DEFAULTS.
"""

from __future__ import annotations

import contextlib
import io
import json
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from tarang import get_logger

logger = get_logger(__name__)

BACKENDS = ("auto", "multinest", "dynesty")

# Tuned on 5-parameter test problems (unimodal and bimodal Gaussians, 60 live
# points, dlogz = 0.3). With these settings dynesty needed about as many likelihood
# calls as MultiNest (≈1.7k vs ≈1.5k on the bimodal case) and recovered the
# evidence and both modes at least as reliably. dynesty's library defaults needed
# 2-4x more calls. Each call is a paid FDTD run, so the call count matters.
DYNESTY_DEFAULTS = dict(
    bound="multi",       # multi-ellipsoid bounds (MultiNest's approach), handles multiple modes
    sample="unif",       # uniform sampling inside the bounds: fewest calls for <= ~10 parameters
    enlarge=1.1,         # ellipsoid enlargement factor (dynesty default 1.25)
    # first_update is set per run: start using bounds once sampling from the
    # unit cube falls below 50 % efficiency, after n_live calls.
)


@dataclass
class NestedRunResult:
    """Summary of one run_nested() call."""

    backend: str
    completed: bool
    log_evidence: Optional[float] = None
    log_evidence_err: Optional[float] = None
    n_calls: int = 0                      # likelihood calls made in this session
    seed: Optional[int] = None
    files: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Backend discovery
# --------------------------------------------------------------------------- #
_MULTINEST_OK: Optional[bool] = None


def multinest_available() -> bool:
    """True if PyMultiNest imports AND its native library loads.

    Without libmultinest, ``import pymultinest`` prints an error and raises
    SystemExit, which would end the Python process. It is probed here with
    output captured and SystemExit caught.
    """
    global _MULTINEST_OK
    if _MULTINEST_OK is None:
        sink = io.StringIO()
        try:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                import pymultinest  # noqa: F401
                import pymultinest.run  # noqa: F401
            _MULTINEST_OK = True
        except (ImportError, OSError, SystemExit):
            _MULTINEST_OK = False
    return _MULTINEST_OK


def _dynesty_available() -> bool:
    try:
        import dynesty  # noqa: F401
        return True
    except ImportError:
        return False


def available_backends() -> list[str]:
    """Backends usable in this Python environment."""
    out = []
    if multinest_available():
        out.append("multinest")
    if _dynesty_available():
        out.append("dynesty")
    return out


def resolve_backend(backend: str = "auto") -> str:
    """Turn 'auto' into a concrete backend and check that it is installed.

    'auto' prefers MultiNest when it works (results comparable with earlier
    TARANG runs) and otherwise uses dynesty, the usual case on Windows.
    """
    backend = str(backend).lower()
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if backend == "auto":
        if multinest_available():
            return "multinest"
        if _dynesty_available():
            return "dynesty"
        raise ImportError(
            "No nested-sampling backend is installed. Install dynesty "
            "(`pip install dynesty`, works on every OS) or, on Linux/macOS, "
            "MultiNest + PyMultiNest from conda-forge."
        )
    if backend == "multinest" and not multinest_available():
        raise ImportError(
            "backend='multinest' requested, but PyMultiNest or its native "
            "libmultinest library is not available. On Windows use "
            "backend='dynesty' (or 'auto')."
        )
    if backend == "dynesty" and not _dynesty_available():
        raise ImportError("backend='dynesty' requested; install it with `pip install dynesty`.")
    return backend


# --------------------------------------------------------------------------- #
# Backend bookkeeping (refuse to mix backends in one chain directory)
# --------------------------------------------------------------------------- #
def _backend_file(basename: str) -> Path:
    return Path(basename + "backend.json")


def _check_and_record_backend(basename: str, backend: str, resume: bool) -> None:
    path = _backend_file(basename)
    if resume and path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8")).get("backend")
        except (OSError, ValueError):
            previous = None
        if previous and previous != backend:
            raise ValueError(
                f"The chains in '{Path(basename).parent}' were written by {previous!r}; "
                f"resuming them with {backend!r} would mix two samplers' files. "
                f"Use backend={previous!r} or a new output directory."
            )
    elif resume and Path(basename + "resume.dat").exists() and backend != "multinest":
        raise ValueError(
            f"'{Path(basename).parent}' holds a MultiNest run (resume.dat). "
            "Use backend='multinest' or a new output directory."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"backend": backend}), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Adapters from the MultiNest calling convention
# --------------------------------------------------------------------------- #
class _PriorAdapter:
    """dynesty wants x = prior_transform(u); MultiNest priors modify u in place."""

    def __init__(self, prior: Callable, ndim: int):
        self.prior, self.ndim = prior, ndim

    def __call__(self, u):
        cube = [float(v) for v in u]
        self.prior(cube, self.ndim, self.ndim)
        return np.asarray(cube, dtype=float)


class _LikeAdapter:
    def __init__(self, loglike: Callable, ndim: int):
        self.loglike, self.ndim, self.calls = loglike, ndim, 0

    def __call__(self, x):
        self.calls += 1
        return float(self.loglike([float(v) for v in x], self.ndim, self.ndim))


# --------------------------------------------------------------------------- #
# MultiNest-format output from dynesty results
# --------------------------------------------------------------------------- #
def _write_rows(path: Path, rows: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        np.savetxt(fh, rows, fmt="%.15E")
    tmp.replace(path)


def _write_multinest_files(basename: str, samples, logl, logwt, live=None, rng=None) -> dict:
    """Write <basename>.txt, post_equal_weights.dat and (optionally) phys_live.points."""
    files = {}
    samples = np.atleast_2d(np.asarray(samples, float))
    logl = np.asarray(logl, float)
    logwt = np.asarray(logwt, float)
    if len(logl):
        w = np.exp(logwt - np.max(logwt))
        w /= w.sum()
        # MultiNest .txt: weight, -2 lnL, parameters
        p = Path(basename + ".txt")
        _write_rows(p, np.column_stack([w, -2.0 * logl, samples]))
        files["weighted"] = str(p)
        # Equal-weight posterior: systematic resampling (as in dynesty.utils.resample_equal)
        rng = rng or np.random.default_rng(0)
        n = len(w)
        positions = (rng.random() + np.arange(n)) / n
        idx = np.minimum(np.searchsorted(np.cumsum(w), positions), n - 1)
        p = Path(basename + "post_equal_weights.dat")
        _write_rows(p, np.column_stack([samples[idx], logl[idx]]))
        files["equal"] = str(p)
    if live is not None and len(live[1]):
        lv, ll = np.atleast_2d(np.asarray(live[0], float)), np.asarray(live[1], float)
        p = Path(basename + "phys_live.points")
        _write_rows(p, np.column_stack([lv, ll, np.ones(len(ll))]))   # params, lnL, mode
        files["live"] = str(p)
    return files


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def run_nested(
    loglike: Callable,
    prior: Callable,
    ndim: int,
    basename: str,
    *,
    backend: str = "auto",
    n_live: int = 150,
    evidence_tolerance: float = 0.3,
    resume: bool = True,
    seed: Optional[int] = None,
    multimodal: bool = True,
    dynesty_kwargs: Optional[dict] = None,
) -> NestedRunResult:
    """Run nested sampling with MultiNest or dynesty (see module docstring).

    Parameters
    ----------
    loglike, prior : callbacks in the MultiNest convention.
    ndim : number of parameters.
    basename : output prefix, e.g. ``.../chains/grating_``.
    backend : 'auto' (MultiNest if installed, else dynesty), 'multinest' or 'dynesty'.
    n_live : live points. More points explore more carefully and cost more calls
        (roughly proportional).
    evidence_tolerance : stop when the remaining ln Z contribution is below this
        (MultiNest ``evidence_tolerance`` = dynesty ``dlogz``).
    resume : continue an interrupted run in ``basename``.
    seed : random seed. For dynesty a new run picks one and stores it; resuming
        reuses the stored seed (needed for replay).
    multimodal : MultiNest mode separation (dynesty's multi-ellipsoid bounds
        handle several modes anyway).
    dynesty_kwargs : overrides for DYNESTY_DEFAULTS / NestedSampler arguments.

    Any exception raised by ``loglike`` (for example a budget stop) propagates.
    For dynesty, partial MultiNest-format files are written first, so the run can
    be inspected or plotted.
    """
    backend = resolve_backend(backend)
    Path(basename).parent.mkdir(parents=True, exist_ok=True)
    _check_and_record_backend(basename, backend, resume)
    if backend == "multinest":
        return _run_multinest(loglike, prior, ndim, basename, n_live,
                              evidence_tolerance, resume, seed, multimodal)
    return _run_dynesty(loglike, prior, ndim, basename, n_live, evidence_tolerance,
                        resume, seed, dynesty_kwargs or {})


def _run_multinest(loglike, prior, ndim, basename, n_live, tol, resume, seed, multimodal):
    import pymultinest

    calls = [0]

    def counted(cube, nd, npar):
        calls[0] += 1
        return loglike(cube, nd, npar)

    pymultinest.run(
        LogLikelihood=counted,
        Prior=prior,
        n_dims=ndim,
        n_params=ndim,
        outputfiles_basename=basename,
        n_live_points=n_live,
        evidence_tolerance=tol,
        importance_nested_sampling=False,
        multimodal=multimodal,
        resume=resume,
        verbose=False,
        seed=-1 if seed is None else int(seed),
    )
    logz = logzerr = None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            stats = pymultinest.Analyzer(n_params=ndim, outputfiles_basename=basename).get_stats()
        logz, logzerr = float(stats["global evidence"]), float(stats["global evidence error"])
    except Exception as exc:  # stats are a convenience; chains are already on disk
        logger.warning(f"Could not read MultiNest evidence: {exc}")
    return NestedRunResult("multinest", True, logz, logzerr, calls[0], seed,
                           {"weighted": basename + ".txt"})


def _run_dynesty(loglike, prior, ndim, basename, n_live, tol, resume, seed, overrides):
    import dynesty

    state_path = Path(basename + "dynesty_state.json")
    settings = {"n_live": int(n_live), "evidence_tolerance": float(tol),
                "ndim": int(ndim), "overrides": repr(sorted(overrides.items()))}
    if resume and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if seed is not None and int(seed) != state["seed"]:
            logger.warning(f"Resuming with the stored seed {state['seed']} (ignoring seed={seed}).")
        seed = int(state["seed"])
        if state.get("settings") != settings:
            logger.warning(
                "dynesty settings differ from the interrupted run "
                f"({state.get('settings')} -> {settings}). The run is still valid, "
                "but it will not retrace the earlier path, so fewer cached "
                "simulations will be reused."
            )
        logger.info(f"Resuming dynesty by replay with seed {seed} "
                    "(earlier evaluations come from the simulation cache).")
    else:
        seed = int(seed) if seed is not None else secrets.randbelow(2**31 - 1)
    state_path.write_text(json.dumps({"seed": seed, "settings": settings}, indent=1),
                          encoding="utf-8")

    kwargs = dict(DYNESTY_DEFAULTS)
    kwargs["first_update"] = {"min_ncall": int(n_live), "min_eff": 50.0}
    kwargs.update(overrides)
    rng = np.random.default_rng(seed)
    like = _LikeAdapter(loglike, ndim)
    sampler = dynesty.NestedSampler(like, _PriorAdapter(prior, ndim), ndim,
                                    nlive=int(n_live), rstate=rng, **kwargs)
    try:
        sampler.run_nested(dlogz=float(tol), print_progress=False)
    except BaseException:
        files = _write_partial(sampler, basename)
        logger.warning(f"dynesty stopped after {like.calls} likelihood calls; "
                       f"partial chains written ({', '.join(files) or 'none'}).")
        raise
    res = sampler.results
    files = _write_multinest_files(basename, res.samples, res.logl, res.logwt,
                                   live=None, rng=np.random.default_rng(seed))
    summary = {"backend": "dynesty", "dynesty_version": dynesty.__version__, "seed": seed,
               "log_evidence": float(res.logz[-1]), "log_evidence_err": float(res.logzerr[-1]),
               "n_iterations": int(res.niter), "n_calls_this_session": like.calls,
               "n_live": int(n_live), "evidence_tolerance": float(tol)}
    Path(basename + "dynesty_summary.json").write_text(json.dumps(summary, indent=1),
                                                       encoding="utf-8")
    logger.info(f"dynesty converged: ln Z = {summary['log_evidence']:.3f} "
                f"± {summary['log_evidence_err']:.3f} after {like.calls} likelihood calls.")
    return NestedRunResult("dynesty", True, summary["log_evidence"],
                           summary["log_evidence_err"], like.calls, seed, files)


def _write_partial(sampler, basename: str) -> dict:
    """Best-effort snapshot after an interruption: dead points + live points."""
    files = {}
    try:
        live = (np.asarray(sampler.live_v), np.asarray(sampler.live_logl))
    except Exception:
        live = None
    try:
        saved_logl = np.asarray(sampler.saved_run["logl"])
        if len(saved_logl):
            res = sampler.results
            files = _write_multinest_files(basename, res.samples, res.logl, res.logwt, live=live)
        elif live is not None:
            files = _write_multinest_files(basename, np.empty((0, live[0].shape[1])), [], [], live=live)
    except Exception as exc:
        logger.warning(f"Could not write partial dynesty chains: {exc}")
    return files
