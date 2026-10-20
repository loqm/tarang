"""
tarang.utilities.cloud_run

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

Every submitted task is recorded in ``task.json`` beside its results (task id, web link,
estimate, status, date) the moment it is uploaded. If a run is interrupted (Ctrl+C, closed
terminal, lost network) the task keeps running on Tidy3D; the next call finds the record
and re-attaches to that task instead of submitting and paying for it again.

Exposes: sim_fingerprint(), cache_paths(), find_cached(), run_cached(),
run_batch_cached(), store_result(), estimate_batch_cost(), resolve_max_cost(),
monitor_data(), api_key_source(), ensure_api_key(), progress_listener(),
account_status(), task_url(), read_task_record(), collect_runs(), backfill_task_records(),
CloudCachedDevice, CostLimitExceededError, MonitorDataError, CachedRun.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

import tidy3d as td
from tidy3d import web

from tarang import get_logger
from tarang._logging import tidy3d_verbose

logger = get_logger(__name__)

PathLike = Union[str, Path]

DATA_FILE = "raw_simulation_data.hdf5"
SIM_FILE = "simulation.json"
TASK_FILE = "task.json"
WEB_URL = "https://tidy3d.simulation.cloud/workbench?taskId={}"
# Tidy3D task states from which no results will come: submit again
DEAD_STATES = {"error", "diverged", "deleted", "aborted", "abort", "failed", "cancelled",
               "canceled", "draft", "initialized"}
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


_LEGACY_KEY_VARS = ("TIDY3D_API_KEY", "TIDY3D_KEY")


def api_key_source() -> Optional[str]:
    """Where Tidy3D will find its API key, or None.

    Tidy3D reads the environment variable SIMCLOUD_APIKEY at request time, and
    otherwise its config (``tidy3d configure`` writes ~/.tidy3d/config;
    TIDY3D_WEB__APIKEY overrides it). It does not read TIDY3D_API_KEY.
    """
    if os.environ.get("SIMCLOUD_APIKEY"):
        return "SIMCLOUD_APIKEY"
    try:
        key = td.config.web.apikey
    except Exception:
        key = None
    if key is not None and str(getattr(key, "get_secret_value", lambda: key)()):
        return "tidy3d config"
    return None


def ensure_api_key() -> Optional[str]:
    """Make a key visible to Tidy3D before a cloud call; return its source.

    Earlier TARANG docs suggested TIDY3D_API_KEY / TIDY3D_KEY, which Tidy3D
    ignores. If only one of those is set, it is passed on as SIMCLOUD_APIKEY.
    """
    source = api_key_source()
    if source:
        return source
    for name in _LEGACY_KEY_VARS:
        if os.environ.get(name):
            os.environ["SIMCLOUD_APIKEY"] = os.environ[name]
            logger.warning(
                f"Tidy3D does not read {name}; passing it on as SIMCLOUD_APIKEY. "
                "Prefer `tidy3d configure` (once per machine) or set SIMCLOUD_APIKEY."
            )
            return name
    logger.warning(
        "No Tidy3D API key found. Run `tidy3d configure` once on this machine, or set "
        "the SIMCLOUD_APIKEY environment variable; the cloud request will fail otherwise."
    )
    return None


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


class MonitorDataError(KeyError, ValueError):
    """A requested monitor is missing from a simulation or from its results.

    Subclasses both KeyError and ValueError, the two types TARANG raised for this
    before v0.1.0, so existing ``except`` clauses keep working.
    """

    def __str__(self):                      # KeyError would otherwise add quotes
        return str(self.args[0]) if self.args else ""


def monitor_data(sim_data: td.SimulationData, name: str):
    """Return ``sim_data[name]`` or explain precisely why it is unavailable."""
    defined = [m.name for m in sim_data.simulation.monitors]
    if name not in defined:
        raise MonitorDataError(
            f"Monitor '{name}' is not defined in this simulation. Defined monitors: {defined}"
        )
    try:
        return sim_data[name]
    except KeyError:
        raise MonitorDataError(
            f"Monitor '{name}' is defined, but the results contain no data for it "
            f"(incomplete or failed run?). Re-run with force_rerun=True."
        ) from None


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


# --------------------------------------------------------------------------- #
# Progress and account
# --------------------------------------------------------------------------- #
_LISTENERS: list[Callable[[str, dict], None]] = []


@contextlib.contextmanager
def progress_listener(callback: Callable[[str, dict], None]):
    """Call ``callback(event, info)`` for every ``run_cached`` inside the block.

    Events: "cached" (loaded from disk), "estimated" (uploaded and priced; info has
    ``cost``), "running" (started on the cloud), "done" (downloaded; info has
    ``seconds`` and ``cost``). ``info`` always has ``name`` (the readable name) and
    ``fingerprint``. A failing callback never stops a run.
    """
    _LISTENERS.append(callback)
    try:
        yield callback
    finally:
        _LISTENERS.remove(callback)


def _emit(event: str, **info) -> None:
    for cb in list(_LISTENERS):
        try:
            cb(event, info)
        except Exception as exc:                      # progress display must never fail a run
            logger.debug(f"progress listener failed: {exc}")


def account_status(timeout: float = 15.0) -> Optional[dict]:
    """FlexCredit balance and expiry of the configured Tidy3D account, or None when it
    cannot be read within ``timeout`` seconds (no network, older Tidy3D; Tidy3D's own
    retries can otherwise take minutes offline). Keys: ``credit`` (float or None),
    ``expires`` (datetime or None), ``expired`` (bool)."""
    import datetime as _dt
    import threading
    box: dict = {}

    def ask():
        try:
            try:
                box["acc"] = web.account(verbose=False)
            except TypeError:                         # Tidy3D without the verbose option
                box["acc"] = web.account()
        except Exception as exc:                      # noqa: BLE001 (reported below)
            box["err"] = exc
    th = threading.Thread(target=ask, daemon=True)
    th.start()
    th.join(timeout)
    if "acc" not in box:
        logger.debug(f"account information unavailable: {box.get('err', 'timed out')}")
        return None
    acc = box["acc"]
    credit = getattr(acc, "credit", None)
    expires = getattr(acc, "credit_expiration", None)
    expired = False
    if isinstance(expires, _dt.datetime):
        now = _dt.datetime.now(expires.tzinfo or _dt.timezone.utc)
        expired = expires < now
    return {"credit": None if credit is None else float(credit), "expires": expires,
            "expired": bool(expired)}


def _task_name(prefix: str, readable_name: str, fp: str) -> str:
    tail = f"_{fp[:FINGERPRINT_LEN]}"
    head = f"{prefix}_{readable_name}" if prefix else readable_name
    return head[: _TASK_NAME_MAX - len(tail)] + tail


# --------------------------------------------------------------------------- #
# Task records (task.json) and web links
# --------------------------------------------------------------------------- #
def task_url(task_id: str) -> str:
    """Link to the task in the Tidy3D web interface (simulation setup, mesh, results)."""
    try:
        from tidy3d.web.api.task_api import _get_url
        url = _get_url(task_id)
        if url.startswith("http"):
            return url
    except Exception:
        pass
    return WEB_URL.format(task_id)


def _now() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def read_task_record(directory: PathLike) -> Optional[dict]:
    """The task.json of a cache directory, or None."""
    f = Path(directory) / TASK_FILE
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_task_record(directory: PathLike, **fields) -> dict:
    """Create or update task.json (keeps fields it already has)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rec = read_task_record(directory) or {}
    rec.update({k: v for k, v in fields.items() if v is not None})
    if rec.get("task_id") and "url" not in fields:
        rec["url"] = task_url(rec["task_id"])
    rec["updated"] = _now()
    tmp = directory / (TASK_FILE + ".tmp")
    tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    os.replace(tmp, directory / TASK_FILE)
    return rec


def _task_status(task_id: str) -> Optional[str]:
    """Current Tidy3D status of a task ("queued", "running", "success", ...), or None."""
    try:
        info = web.get_info(task_id, verbose=False)
    except TypeError:
        info = web.get_info(task_id)
    except Exception as exc:
        logger.debug(f"status of {task_id} unavailable: {exc}")
        return None
    st = getattr(info, "status", None)
    return str(st).lower() if st is not None else None


def _live_record(directory: Path) -> Optional[dict]:
    """The record of a task submitted earlier for this directory that will still give
    results (queued, running or finished on Tidy3D), or None."""
    rec = read_task_record(directory)
    if not rec or not rec.get("task_id") or rec.get("status") == "downloaded":
        return None
    st = _task_status(rec["task_id"])
    if st is None or st in DEAD_STATES:
        return None
    rec["remote_status"] = st
    return rec


def _real_cost(task_id: str) -> Optional[float]:
    try:
        c = web.real_cost(task_id, verbose=False)
    except TypeError:
        c = web.real_cost(task_id)
    except Exception:
        return None
    return None if c is None else float(c)


def collect_runs(root: PathLike) -> list[dict]:
    """Every cloud run under ``root`` (folders with results or a task record): name,
    folder, task id, web link, status, estimate, billed cost and date."""
    rows = []
    for d in sorted({p.parent for p in Path(root).rglob(DATA_FILE)} |
                    {p.parent for p in Path(root).rglob(TASK_FILE)}):
        rec = read_task_record(d) or {}
        rows.append(dict(name=d.name, folder=str(d), task_id=rec.get("task_id"),
                         url=rec.get("url"), status=rec.get("status") or
                         ("downloaded" if (d / DATA_FILE).exists() else "unknown"),
                         estimated_cost=rec.get("estimated_cost"), real_cost=rec.get("real_cost"),
                         submitted=rec.get("submitted"), backfilled=rec.get("backfilled", False)))
    return rows


def backfill_task_records(root: PathLike, folders: Iterable[str] = ("default",)) -> int:
    """Find the Tidy3D task of results saved before task records existed (matching the
    fingerprint at the end of the task name) and write their task.json. Returns the number
    of records written; never fails (no network = 0)."""
    missing = [p.parent for p in Path(root).rglob(DATA_FILE)
               if read_task_record(p.parent) is None]
    if not missing:
        return 0
    tasks: list[dict] = []
    for folder in folders:
        try:
            tasks += web.get_tasks(num_tasks=None, order="new", folder=folder) or []
        except Exception as exc:
            logger.debug(f"task list of folder {folder!r} unavailable: {exc}")
    by_tail: dict[str, dict] = {}
    for t in tasks:
        name = t.get("task_name") or t.get("taskName") or ""
        tid = t.get("task_id") or t.get("taskId")
        if name and tid and "_" in name:
            by_tail.setdefault(name.rsplit("_", 1)[-1], dict(t, _id=tid, _name=name))
    n = 0
    for d in missing:
        tail = d.name.rsplit("_", 1)[-1]
        t = by_tail.get(tail)
        if not t:
            continue
        created = t.get("created_at") or t.get("createdAt")
        write_task_record(d, task_id=t["_id"], task_name=t["_name"], status="downloaded",
                          remote_status=str(t.get("status") or "").lower() or None,
                          real_cost=t.get("real_flex_unit") or t.get("realCost"),
                          submitted=str(created) if created else None, backfilled=True)
        n += 1
    return n


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
            _emit("cached", name=readable_name, fingerprint=fp[:FINGERPRINT_LEN])
            return CachedRun(td.SimulationData.from_file(str(hit)), hit, True, 0.0, fp)

    ensure_api_key()

    directory.mkdir(parents=True, exist_ok=True)
    sim.to_file(str(directory / SIM_FILE))          # provenance, written before upload
    task_name = _task_name(task_prefix, readable_name, fp)
    short = fp[:FINGERPRINT_LEN]

    rec = None if force_rerun else _live_record(directory)
    if rec is not None:
        # submitted earlier (the run was interrupted): follow that task, do not pay again
        task_id, cost = rec["task_id"], float(rec.get("estimated_cost") or 0.0)
        logger.info(f"Re-attaching to task {task_id} ({rec['remote_status']}) for '{task_name}'.")
        _emit("reattached", name=readable_name, fingerprint=short, cost=cost,
              task_id=task_id, url=rec.get("url"), status=rec["remote_status"])
        started = time.monotonic()
    else:
        logger.info(f"Cache miss. Uploading '{task_name}' to Tidy3D ...")
        task_id = web.upload(sim, task_name=task_name, folder_name=folder_name,
                             verbose=tidy3d_verbose(verbose))
        cost = float(web.estimate_cost(task_id, verbose=tidy3d_verbose(verbose)))
        logger.info(f"Estimated cost for '{task_name}': {cost:.4f} FlexCredits.")
        _emit("estimated", name=readable_name, fingerprint=short, cost=cost, task_id=task_id,
              url=task_url(task_id))

        if max_cost is not None and cost > max_cost:
            try:
                web.delete(task_id)                     # remove the unstarted draft
            except Exception:
                pass
            raise CostLimitExceededError(cost, max_cost, task_name)

        started = time.monotonic()
        web.start(task_id)
        write_task_record(directory, task_id=task_id, task_name=task_name, folder=folder_name,
                          estimated_cost=cost, status="running", submitted=_now(),
                          fingerprint=short)
        _emit("running", name=readable_name, fingerprint=short, cost=cost, task_id=task_id,
              url=task_url(task_id))
    web.monitor(task_id, verbose=tidy3d_verbose(verbose))

    # Download to a temporary name and rename only when complete, so an
    # interrupted download is never mistaken for a valid cache entry.
    target = directory / DATA_FILE
    partial = directory / "partial_download.hdf5"
    data = web.load(task_id, path=str(partial), verbose=tidy3d_verbose(verbose))
    os.replace(partial, target)
    write_task_record(directory, status="downloaded", downloaded=_now(),
                      real_cost=_real_cost(task_id))
    logger.info(f"Run complete. Saved {target}")
    _emit("done", name=readable_name, fingerprint=short, cost=cost,
          seconds=time.monotonic() - started, task_id=task_id, url=task_url(task_id))
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
    if read_task_record(directory):
        write_task_record(directory, status="downloaded", downloaded=_now())
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
    reattach: dict[str, tuple[str, dict]] = {}
    for key, (sim, readable) in jobs.items():
        hit = find_cached(sim, output_dir, readable, [legacy_names.get(key, "")])
        if hit is not None:
            out[key] = hit
            continue
        directory, fp = cache_paths(output_dir, readable, sim)
        task = _task_name("", readable, fp)
        rec = _live_record(directory)
        if rec is not None:                     # submitted before an interruption
            reattach[task] = (key, dict(rec, _sim=sim))
            names[task] = (key, readable)
            continue
        pending[task] = sim
        names[task] = (key, readable)

    if not pending and not reattach:
        logger.info("All batch simulations found in cache; nothing to run.")
        return out

    ensure_api_key()
    started = time.monotonic()
    for task, (key, rec) in reattach.items():
        _emit("reattached", name=task, fingerprint=task.rsplit("_", 1)[-1],
              cost=float(rec.get("estimated_cost") or 0.0), task_id=rec["task_id"],
              url=rec.get("url"), status=rec["remote_status"])
    cost = 0.0
    batch = None
    if pending:
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
        each = cost / max(len(pending), 1)
        for task, sim in pending.items():
            key, readable = names[task]
            directory, fp = cache_paths(output_dir, readable, sim)
            directory.mkdir(parents=True, exist_ok=True)
            sim.to_file(str(directory / SIM_FILE))
            try:
                tid = batch.jobs[task].task_id
            except Exception:
                tid = None
            if tid:
                write_task_record(directory, task_id=tid, task_name=task, folder=folder_name,
                                  estimated_cost=each, status="running", submitted=_now(),
                                  fingerprint=fp[:FINGERPRINT_LEN])
                _emit("submitted", name=readable, fingerprint=fp[:FINGERPRINT_LEN], cost=each,
                      task_id=tid, url=task_url(tid))
        _emit("running", name=f"batch of {len(pending)}", fingerprint="", cost=cost)
    for task, (key, rec) in reattach.items():
        _, readable = names[task]
        web.monitor(rec["task_id"], verbose=tidy3d_verbose(verbose))
        sim = rec["_sim"]
        directory, _ = cache_paths(output_dir, readable, sim)
        partial = directory / "partial_download.hdf5"
        data = web.load(rec["task_id"], path=str(partial), verbose=tidy3d_verbose(verbose))
        out[key] = store_result(sim, data, output_dir, readable)
        try:
            partial.unlink()
        except OSError:
            pass
    if batch is not None:
        batch.monitor()
        results = batch.load(path_dir=str(Path(output_dir) / "batch_downloads"))
        for task, sim_data in results.items():
            key, readable = names[task]
            out[key] = store_result(pending[task], sim_data, output_dir, readable)
    _emit("done", name=f"batch of {len(pending) + len(reattach)}", fingerprint="", cost=cost,
          seconds=time.monotonic() - started)
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
