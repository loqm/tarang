"""
utilities.cloud_run

One place where TARANG sends a Tidy3D simulation to the cloud.

Why this module exists
----------------------
Every device wrapper (grating coupler, edge coupler, ring, arc) caches the
downloaded ``SimulationData`` in ``<output_dir>/<dir_name>/raw_simulation_data.hdf5``
and skips the cloud run when that file exists. Before v0.1.0 each wrapper built
``dir_name`` from a hand-picked list of parameters; the grating coupler left the
materials out, so changing ``sio2`` silently reused results computed with the old
index (audit C1). The fix is to key the cache on the simulation itself:

* ``sim_fingerprint(sim)`` hashes the complete Simulation description (geometry,
  media, mesh, sources, monitors, run time), excluding only the Tidy3D version
  string. Two simulations share a fingerprint only if Tidy3D would run the same
  physics.
* ``run_cached(...)`` stores results in ``<readable_name>_<fingerprint[:10]>``,
  writes ``simulation.json`` beside the data for provenance, and verifies the
  fingerprint on every cache hit.
* Results that already exist under the pre-v0.1.0 directory name are reused only
  if the simulation stored inside the HDF5 has the same fingerprint; otherwise a
  warning explains what differs and a fresh run goes to the new directory. Old
  results are never modified or deleted.

Every cloud submission is preceded by a FlexCredit estimate that is logged, and
``max_cost`` (FlexCredits) aborts before the task is started (audit C6). The task
name carries the fingerprint so runs are distinguishable in the web UI (audit R4).

Exposes: sim_fingerprint(), cache_paths(), find_cached(), run_cached(),
run_batch_cached(), store_result(), estimate_batch_cost(), resolve_max_cost(),
CloudCachedDevice, CostLimitExceededError, CachedRun.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Union

import tidy3d as td
from tidy3d import web

from tarang import get_logger
from tarang._logging import tidy3d_verbose

logger = get_logger(__name__)

PathLike = Union[str, Path]

DATA_FILE = "raw_simulation_data.hdf5"
SIM_FILE = "simulation.json"
FINGERPRINT_LEN = 10          # hex characters used in directory and task names
_TASK_NAME_MAX = 80           # keep task names readable in the Tidy3D dashboard


class CostLimitExceededError(Exception):
    """Raised before starting a cloud task whose estimated cost exceeds ``max_cost``."""

    def __init__(self, estimated: float, limit: float, task_name: str):
        self.estimated = float(estimated)
        self.limit = float(limit)
        self.task_name = task_name
        super().__init__(
            f"Estimated cost {self.estimated:.4f} FlexCredits for task '{task_name}' "
            f"exceeds max_cost={self.limit:.4f}. Nothing was run or charged. "
            f"Raise max_cost, coarsen the mesh, or shorten run_time."
        )


def resolve_max_cost(max_cost: Optional[float]) -> Optional[float]:
    """Explicit max_cost wins; otherwise use $TARANG_MAX_COST (FlexCredits) if set.

    The environment variable gives a session-wide safety net for code paths that
    build device wrappers internally (sweeps, notebooks).
    """
    if max_cost is not None:
        if max_cost < 0:
            raise ValueError(f"max_cost must be >= 0, got {max_cost}")
        return float(max_cost)
    env = os.environ.get("TARANG_MAX_COST", "").strip()
    if not env:
        return None
    try:
        return float(env)
    except ValueError:
        raise ValueError(f"TARANG_MAX_COST must be a number of FlexCredits, got {env!r}")


@dataclass
class CachedRun:
    """What run_cached() did: where the data lives, whether it came from cache,
    and the FlexCredit estimate for a fresh run (0.0 for a cache hit)."""

    data: td.SimulationData
    path: Path
    from_cache: bool
    estimated_cost: float
    fingerprint: str


# --------------------------------------------------------------------------- #
# Fingerprinting
# --------------------------------------------------------------------------- #
def _sim_dict(sim: td.Simulation) -> dict:
    """Return the Simulation as a plain dict without the Tidy3D version field."""
    dump = getattr(sim, "model_dump_json", None) or sim.json
    d = json.loads(dump())
    d.pop("version", None)
    return d


def sim_fingerprint(sim: td.Simulation) -> str:
    """SHA-256 of the full Simulation description, version string excluded.

    Any change that alters the physics (a material pole, a mesh step, a monitor,
    run_time, symmetry, ...) changes the fingerprint. Cosmetic Tidy3D upgrades
    that keep the same serialized simulation do not.
    """
    canon = json.dumps(_sim_dict(sim), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def _diff_top_level(a: dict, b: dict) -> list[str]:
    """Top-level Simulation fields that differ, used to explain cache misses."""
    keys = sorted(set(a) | set(b))
    return [k for k in keys if a.get(k) != b.get(k)]


def _explain(field_names: Iterable[str]) -> str:
    hints = {
        "structures": "structures (geometry or material media)",
        "medium": "background medium",
        "grid_spec": "mesh",
        "sources": "sources",
        "monitors": "monitors",
        "run_time": "run_time",
        "symmetry": "symmetry",
        "boundary_spec": "boundaries",
        "size": "domain size",
        "center": "domain centre",
    }
    names = [hints.get(f, f) for f in field_names]
    return ", ".join(names) if names else "no top-level field (numerical detail)"


# --------------------------------------------------------------------------- #
# Cache layout
# --------------------------------------------------------------------------- #
def cache_paths(output_dir: PathLike, readable_name: str, sim: td.Simulation) -> tuple[Path, str]:
    """Return (directory, fingerprint) for this simulation's cache entry."""
    fp = sim_fingerprint(sim)
    return Path(output_dir) / f"{readable_name}_{fp[:FINGERPRINT_LEN]}", fp


def _stored_fingerprint(directory: Path) -> Optional[str]:
    """Fingerprint of the simulation saved in a cache directory, or None."""
    sim_json = directory / SIM_FILE
    data_file = directory / DATA_FILE
    try:
        if sim_json.exists():
            return sim_fingerprint(td.Simulation.from_file(str(sim_json)))
        if data_file.exists():
            return sim_fingerprint(td.Simulation.from_hdf5(str(data_file), group_path="simulation"))
    except Exception as exc:  # unreadable or partial file
        logger.warning(f"Could not read the simulation stored in {directory}: {exc}")
    return None


def find_cached(
    sim: td.Simulation,
    output_dir: PathLike,
    readable_name: str,
    legacy_names: Iterable[str] = (),
) -> Optional[Path]:
    """Return the HDF5 path of a verified cache entry for ``sim``, or None.

    Looks in the fingerprinted directory first, then in any pre-v0.1.0 directory
    names. A legacy entry is accepted only if the simulation stored with it has
    the same fingerprint as ``sim``.
    """
    directory, fp = cache_paths(output_dir, readable_name, sim)
    candidates = [directory] + [Path(output_dir) / n for n in legacy_names if n]
    for cand in candidates:
        data_file = cand / DATA_FILE
        if not data_file.exists():
            continue
        stored = _stored_fingerprint(cand)
        if stored == fp:
            return data_file
        if stored is None:
            continue
        try:
            old = (td.Simulation.from_file(str(cand / SIM_FILE)) if (cand / SIM_FILE).exists()
                   else td.Simulation.from_hdf5(str(data_file), group_path="simulation"))
            changed = _explain(_diff_top_level(_sim_dict(old), _sim_dict(sim)))
        except Exception:
            changed = "unknown"
        logger.warning(
            f"Existing results in '{cand}' were computed for a different simulation "
            f"(differs in: {changed}). They are left untouched and will NOT be reused."
        )
    return None


def _task_name(prefix: str, readable_name: str, fp: str) -> str:
    tail = f"_{fp[:FINGERPRINT_LEN]}"
    head = f"{prefix}_{readable_name}" if prefix else readable_name
    return head[: _TASK_NAME_MAX - len(tail)] + tail


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #
def run_cached(
    sim: td.Simulation,
    output_dir: PathLike,
    readable_name: str,
    *,
    task_prefix: str = "",
    legacy_names: Iterable[str] = (),
    max_cost: Optional[float] = None,
    force_rerun: bool = False,
    verbose: bool = False,
    folder_name: str = "default",
) -> CachedRun:
    """Load ``sim`` results from the local cache or run them on the Tidy3D cloud.

    Parameters
    ----------
    sim : the Tidy3D Simulation to run.
    output_dir : root folder for cached results.
    readable_name : human-readable directory stem (the fingerprint is appended).
    task_prefix : prefix for the cloud task name (e.g. "gc_sim").
    legacy_names : pre-v0.1.0 directory names to check for reusable results.
    max_cost : FlexCredit ceiling for this run. If the estimate exceeds it, the
        task is not started and CostLimitExceededError is raised. When None, the
        environment variable TARANG_MAX_COST is used if set.
    force_rerun : ignore any cache and run again (results go to the same
        fingerprinted directory; the previous file there is replaced).
    verbose : show Tidy3D progress output.
    folder_name : Tidy3D web folder.
    """
    directory, fp = cache_paths(output_dir, readable_name, sim)
    max_cost = resolve_max_cost(max_cost)

    if not force_rerun:
        hit = find_cached(sim, output_dir, readable_name, legacy_names)
        if hit is not None:
            logger.info(f"Cache hit (fingerprint {fp[:FINGERPRINT_LEN]}). Loading {hit} ...")
            return CachedRun(td.SimulationData.from_file(str(hit)), hit, True, 0.0, fp)

    if "TIDY3D_KEY" not in os.environ and "TIDY3D_API_KEY" not in os.environ:
        logger.warning(
            "TIDY3D_API_KEY not found in environment variables; relying on the "
            "Tidy3D config file for authentication."
        )

    directory.mkdir(parents=True, exist_ok=True)
    sim.to_file(str(directory / SIM_FILE))          # provenance, written before upload
    task_name = _task_name(task_prefix, readable_name, fp)

    logger.info(f"Cache miss. Uploading '{task_name}' to Tidy3D ...")
    task_id = web.upload(sim, task_name=task_name, folder_name=folder_name,
                         verbose=tidy3d_verbose(verbose))
    cost = float(web.estimate_cost(task_id, verbose=tidy3d_verbose(verbose)))
    logger.info(f"Estimated cost for '{task_name}': {cost:.4f} FlexCredits.")

    if max_cost is not None and cost > max_cost:
        try:
            web.delete(task_id)                     # remove the unstarted draft
        except Exception:
            pass
        raise CostLimitExceededError(cost, max_cost, task_name)

    web.start(task_id)
    web.monitor(task_id, verbose=tidy3d_verbose(verbose))

    # Download to a temporary name and rename only when complete, so an
    # interrupted download is never mistaken for a valid cache entry.
    target = directory / DATA_FILE
    partial = directory / "partial_download.hdf5"
    data = web.load(task_id, path=str(partial), verbose=tidy3d_verbose(verbose))
    os.replace(partial, target)
    logger.info(f"Run complete. Saved {target}")
    return CachedRun(data, target, False, cost, fp)


def store_result(
    sim: td.Simulation,
    data: td.SimulationData,
    output_dir: PathLike,
    readable_name: str,
) -> Path:
    """Save externally obtained results (e.g. from a web.Batch) in the cache layout.

    Never replaces an existing verified entry.
    """
    directory, _ = cache_paths(output_dir, readable_name, sim)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / DATA_FILE
    if not target.exists():
        sim.to_file(str(directory / SIM_FILE))
        partial = directory / "partial_download.hdf5"
        data.to_file(str(partial))
        os.replace(partial, target)
    return target


def run_batch_cached(
    jobs: dict[str, tuple[td.Simulation, str]],
    output_dir: PathLike,
    *,
    legacy_names: Optional[dict[str, str]] = None,
    max_cost: Optional[float] = None,
    verbose: bool = False,
    folder_name: str = "default",
) -> dict[str, Path]:
    """Run several simulations as one Tidy3D Batch, skipping cached ones.

    ``jobs`` maps a key to (simulation, readable_name). Returns key -> HDF5 path.
    The total FlexCredit estimate for the uncached jobs is logged, and the batch is
    not started if it exceeds ``max_cost``.
    """
    legacy_names = legacy_names or {}
    max_cost = resolve_max_cost(max_cost)
    out: dict[str, Path] = {}
    pending: dict[str, td.Simulation] = {}
    names: dict[str, tuple[str, str]] = {}
    for key, (sim, readable) in jobs.items():
        hit = find_cached(sim, output_dir, readable, [legacy_names.get(key, "")])
        if hit is not None:
            out[key] = hit
            continue
        fp = sim_fingerprint(sim)
        task = _task_name("", readable, fp)
        pending[task] = sim
        names[task] = (key, readable)

    if not pending:
        logger.info("All batch simulations found in cache; nothing to run.")
        return out

    batch = web.Batch(simulations=pending, folder_name=folder_name,
                      verbose=tidy3d_verbose(verbose))
    batch.upload()
    cost = float(batch.estimate_cost(verbose=tidy3d_verbose(verbose)))
    logger.info(f"Estimated cost for {len(pending)} batch task(s): {cost:.4f} FlexCredits.")
    if max_cost is not None and cost > max_cost:
        try:
            batch.delete()
        except Exception:
            pass
        raise CostLimitExceededError(cost, max_cost, f"batch of {len(pending)}")

    batch.start()
    batch.monitor()
    results = batch.load(path_dir=str(Path(output_dir) / "batch_downloads"))
    for task, sim_data in results.items():
        key, readable = names[task]
        out[key] = store_result(pending[task], sim_data, output_dir, readable)
    return out


class CloudCachedDevice:
    """Mixin giving a device wrapper verified caching and cost control.

    A subclass provides ``self.sim``, ``_readable_name()`` and, to reuse
    pre-v0.1.0 results, ``_legacy_dir_name()``. Set ``max_cost`` (FlexCredits) on
    an instance to cap every cloud run it makes; ``last_run`` records the most
    recent run's estimate and cache status.
    """

    _task_prefix: str = ""
    _folder_name: str = "default"
    max_cost: Optional[float] = None
    last_run: Optional[CachedRun] = None

    def _readable_name(self) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def _legacy_dir_name(self) -> Optional[str]:
        return None

    def get_dir_name(self) -> str:
        """Cache directory name: readable parameters + simulation fingerprint."""
        return cache_paths("", self._readable_name(), self.sim)[0].name

    def is_cached(self, output_dir: PathLike) -> bool:
        """True if verified results for this exact simulation exist locally."""
        return find_cached(self.sim, output_dir, self._readable_name(),
                           [self._legacy_dir_name() or ""]) is not None

    def _run_sim(
        self,
        output_dir: PathLike,
        verbose: bool = False,
        force_rerun: bool = False,
        max_cost: Optional[float] = None,
    ) -> td.SimulationData:
        """Load cached results or run on the Tidy3D cloud (see run_cached)."""
        run = run_cached(
            self.sim,
            output_dir,
            self._readable_name(),
            task_prefix=self._task_prefix,
            legacy_names=[self._legacy_dir_name() or ""],
            max_cost=self.max_cost if max_cost is None else max_cost,
            force_rerun=force_rerun,
            verbose=verbose,
            folder_name=self._folder_name,
        )
        self.last_run = run
        return run.data


def estimate_batch_cost(sims: dict[str, td.Simulation], verbose: bool = False) -> float:
    """Upload ``sims`` as drafts, sum their FlexCredit estimates, delete the drafts.

    Use this before a large sweep when you want a budget figure without running.
    """
    total = 0.0
    for name, sim in sims.items():
        tid = web.upload(sim, task_name=name, verbose=tidy3d_verbose(verbose))
        try:
            total += float(web.estimate_cost(tid, verbose=tidy3d_verbose(verbose)))
        finally:
            try:
                web.delete(tid)
            except Exception:
                pass
    return total
