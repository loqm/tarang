"""What the GUI shows, read from a design folder; and starting / stopping design runs.

Nothing here imports Streamlit, so it is usable (and tested) on its own. A design folder
is any folder holding a ``design.toml`` (written by ``python -m tarang.design``); its
``state.json`` holds every stage's results, ``figures/``, ``tables/``, ``layout/`` and
``lle/`` the files.

Runs are started as a separate process (``python -m tarang.design --config ... --yes``),
with everything they print written to ``<run>/gui_log.txt``, so the GUI stays responsive
and a run keeps going if the browser tab is closed. Stop sends Ctrl+Break (Windows) or
Ctrl+C (Linux/macOS), which TARANG treats like q: finished stages and finished cloud runs
are kept, and the same run continues next time.
"""
from __future__ import annotations

import csv
import math
import datetime as dt
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from typing import Optional

JOB_FILE = "gui_job.json"
LOG_FILE = "gui_log.txt"
_PROCS: dict[str, subprocess.Popen] = {}         # run folder -> process started here


# ------------------------------------------------------------------ design folders
def find_runs(root: Path | str, depth: int = 3) -> list[Path]:
    """Design folders (holding design.toml) under ``root``, newest first."""
    root = Path(root)
    found = []
    if (root / "design.toml").exists():
        found.append(root)
    for d in range(1, depth + 1):
        for p in root.glob("/".join(["*"] * d) + "/design.toml"):
            if "benchmark_pack" not in p.parts and p.parent not in found:
                found.append(p.parent)

    def mtime(p: Path) -> float:
        s = p / "state.json"
        return (s if s.exists() else p / "design.toml").stat().st_mtime
    return sorted(found, key=mtime, reverse=True)


def load_state(run: Path) -> dict:
    p = Path(run) / "state.json"
    if not p.exists():
        return {"stages": {}, "spent": 0.0}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"stages": {}, "spent": 0.0}


def load_config(run: Path):
    from tarang.design.config import DesignConfig
    return DesignConfig.load(Path(run) / "design.toml")


def stage_rows(run: Path) -> list[dict]:
    """One row per stage: number, name, title, kind, status."""
    from tarang.design.stages import STAGES, Context, applies, settings_key
    from tarang.design.ui import Prompter
    import io
    run = Path(run)
    state = load_state(run)
    try:
        cfg = load_config(run)
        ctx = Context(cfg, Prompter(False, out=io.StringIO()), run)
        ctx.apply_legacy()
    except Exception:                                             # noqa: BLE001
        cfg = None
    rows = []
    for i, (name, title, kind, _) in enumerate(STAGES, 1):
        st = state.get("stages", {}).get(name)
        if cfg is not None and not applies(name, cfg):
            status = "not part of this goal"
        elif not st:
            status = "not run"
        else:
            res = st.get("results") or {}
            if res.get("skipped") or res.get("disabled"):
                status = "skipped / off"
            elif cfg is not None and st.get("key") != settings_key(cfg, name):
                status = "settings changed: will rerun"
            else:
                status = "done"
        rows.append(dict(no=i, stage=name, title=title, kind="cloud (priced)" if kind == "cloud"
                         else "local (free)", status=status))
    return rows


def credits(run: Path, account: bool = False) -> dict:
    """Limit, spent, left, the spend history; with ``account`` also the Tidy3D balance
    (only when an API key is configured; at most 8 s)."""
    from tarang.design import credits as cr_design
    run = Path(run)
    cfg = load_config(run)
    out = cr_design.summary(cfg, run, load_state(run))
    out["per_run"] = float(cfg.run.max_cost_per_run)
    out["account"] = None
    if account:
        try:
            from tarang.utilities import cloud_run as cr
            if cr.api_key_source():
                acc = cr.account_status(timeout=8.0)
                if acc:
                    out["account"] = dict(credit=acc.get("credit"), expired=bool(acc.get("expired")),
                                          expires=acc["expires"].strftime("%Y-%m-%d")
                                          if acc.get("expires") else None)
        except Exception:                                         # noqa: BLE001
            pass
    return out


def set_limits(run: Path, total: Optional[float] = None, per_run: Optional[float] = None,
               credit: Optional[float] = None) -> str:
    """Change run.max_total_cost / run.max_cost_per_run in design.toml, with the same checks
    as the run's own limit table; returns "" or the reason it was refused."""
    from tarang.design import limit_checks
    run = Path(run)
    cfg = load_config(run)
    spent = float(load_state(run).get("spent", 0.0))
    new_total = float(cfg.run.max_total_cost if total is None else total)
    per_chk, tot_chk = limit_checks(spent, credit, new_total)
    if total is not None:
        msg = tot_chk(total)
        if msg:
            return "Total limit: " + msg
        cfg.run.max_total_cost = float(total)
    if per_run is not None:
        msg = per_chk(per_run)
        if msg:
            return "Per-run limit: " + msg
        cfg.run.max_cost_per_run = float(per_run)
    cfg.save(run / "design.toml")
    return ""


# ------------------------------------------------------------------ results
def _fmt(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


def summary_rows(run: Path) -> list[dict]:
    """The design's headline numbers, one row each (quantity, value, from)."""
    st = {k: (v or {}).get("results") or {} for k, v in load_state(run).get("stages", {}).items()}
    rows = []

    def add(q, v, unit="", src=""):
        if v is not None and v != "":
            rows.append(dict(quantity=q, value=_fmt(v) if not isinstance(v, str) else v, unit=unit,
                             stage=src))
    xs, ring, gap, gc = st.get("cross_section", {}), st.get("ring", {}), st.get("gap", {}), st.get("grating", {})
    add("core height x width", f"{xs['height_um']:g} x {xs['width_um']:g}" if xs.get("height_um") else None,
        "um", "cross_section")
    add("dispersion D at the pump", xs.get("D"), "ps/(nm km)", "cross_section")
    if xs.get("D_mesh") is not None and xs.get("D_mesh_band") is not None:
        add("D over the mesh check (median ± spread)",
            f"{xs['D_mesh']:.1f} ± {xs['D_mesh_band']:.1f}", "ps/(nm km)", "cross_section")
    add("effective / group index", f"{xs['n_eff']:.4f} / {xs['n_group']:.4f}" if xs.get("n_eff") else None,
        "", "cross_section")
    add("ring radius", ring.get("radius_um"), "um", "ring")
    add("FSR", ring.get("fsr_ghz"), "GHz", "ring")
    add("intrinsic Q", ring.get("q_intrinsic"), "", "ring")
    add("D2/2pi", ring.get("D2_2pi_MHz"), "MHz", "ring")
    add("critical gap", gap.get("critical_gap_um"), "um", "gap")
    if gap.get("over_coupled_gap_um"):
        add(f"over-coupled gap (K = {gap.get('over_coupling'):g})", gap["over_coupled_gap_um"], "um", "gap")
    if gc.get("period_um"):
        add("grating period / fill factor", f"{gc['period_um']:.4f} / {gc['fill_factor']:.2f}", "um / -", "grating")
        add("grating efficiency (2D) at the pump", gc.get("efficiency_2d"), "", "grating")
        add("grating 1 dB bandwidth (2D)", gc.get("bandwidth_1db_nm"), "nm", "grating")
    tp = st.get("taper", {})
    add("grating width / taper length",
        f"{tp['grating_width_um']:g} / {tp.get('taper_length_um') or '-'}" if tp.get("grating_width_um") else None,
        "um", "taper")
    ap = st.get("apodise", {})
    if ap.get("verified"):
        add("apodised grating (2D)", f"{ap['efficiency_2d'] * 100:.1f} % ({ap.get('recipe', 'measured scattering')}"
            f"{', used' if ap.get('better_than_uniform') else ', not used'})", "", "apodise")
    fc = st.get("focus", {})
    if fc.get("curves"):
        add("focusing grating", f"estimate {fc['efficiency_3d_estimate'] * 100:.1f} %"
            + (f", 3D FDTD {fc['efficiency_3d'] * 100:.1f} %" if fc.get("efficiency_3d") is not None else ""),
            "", "focus")
    ed = st.get("edge", {})
    if ed.get("facet_width_um"):
        add("edge coupler: facet width / taper", f"{ed['facet_width_um']:g} / {ed['profile']} "
            f"{ed['taper_length_um']:g}", "um", "edge")
        f3 = ed.get("fdtd") or {}
        add("edge coupler per facet", f"{ed['eta_coupler'] * 100:.1f} %"
            + (" (3D FDTD)" if f3 else " (butt-coupling estimate)"), "", "edge")
        if f3.get("reflection_mode") is not None:
            add("facet reflection", 10 * math.log10(max(f3["reflection_mode"], 1e-12)), "dB", "edge")
    wg = st.get("waveguides", {})
    if wg.get("lengths_cm"):
        add("expected propagation loss", wg.get("alpha_db_cm"), "dB/cm", "waveguides")
        add("smallest spiral bend radius", wg.get("bend_radius_um"), "um", "waveguides")
        add("cutback lengths", ", ".join(f"{x:g}" for x in wg["lengths_cm"]), "cm", "waveguides")
        add("loss known to", wg.get("sigma_alpha_db_cm"), "dB/cm", "waveguides")
    dv = st.get("device", {})
    for r in dv.get("rows") or []:
        add(f"expected transmission, {r['structure']}", r["transmission_db"], "dB", "device")
    if dv.get("resonances"):
        add("fibre to fibre, off resonance", dv.get("off_resonance_db_at_pump"), "dB", "device")
        for k, v in (dv.get("regimes") or {}).items():
            add(f"loaded Q / dip ({k})", f"{v['Q_L_pump']:.3g} / {v['dip_db_pump']:.1f} dB", "", "device")
    return rows


def loss_rows(run: Path) -> list[dict]:
    return read_csv(Path(run) / "tables" / "loss_budget.csv")


def read_csv(path: Path, limit: Optional[int] = None) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        rows = []
        for i, r in enumerate(csv.DictReader(f)):
            if limit is not None and i >= limit:
                break
            rows.append(r)
    return rows


FIGURE_GROUPS = [("1", "Materials"), ("2", "Cross-section"), ("3", "Ring and test waveguides"), ("4", "Gap"),
                 ("5", "Grating"), ("6", "Layout"), ("7", "Checks and budget"), ("8", "Taper"),
                 ("9", "Apodised grating"), ("10", "Focusing grating"), ("11", "Whole device"),
                 ("12", "Edge coupler")]


def figures(run: Path) -> list[tuple[str, list[Path]]]:
    """PNG figures grouped by their number prefix (the report's order), then the
    dispersion-engineering figures under results/ (:func:`dispersion_suite`)."""
    figs = sorted((Path(run) / "figures").glob("*.png"))
    groups = []
    for pre, name in FIGURE_GROUPS:
        g = [f for f in figs if (re.match(r"\d+", f.name) or [None])[0] == pre]
        if g:
            groups.append((name, g))
    rest = [f for f in figs if all(f not in g for _, g in groups)]
    if rest:
        groups.append(("Other", rest))
    return groups + dispersion_suite(run)


#: sub-folders of a dispersion map's plots folder, in the order they are shown
MAP_GROUPS = [("D_vs_width", "Dispersion map: D against width, one figure per height"),
              ("D_vs_height", "Dispersion map: D against height, one figure per width"),
              ("modes_vs_width", "Dispersion map: guided modes against width, per height"),
              ("modes_vs_height", "Dispersion map: guided modes against height, per width")]


def _first_dir(*paths: Path) -> Optional[Path]:
    """The first of these folders that holds a PNG."""
    for p in paths:
        if p is not None and p.is_dir() and any(p.rglob("*.png")):
            return p
    return None


def latest_map(run: Path) -> Optional[Path]:
    """The plots folder of the cross-section map: figures/2_dispersion_map (since 14E), else
    the latest results/plots_map_<steps>steps of an older run."""
    run = Path(run)
    old = [p for p in (run / "results").glob("plots_map_*") if p.is_dir()]
    return _first_dir(run / "figures" / "2_dispersion_map",
                      max(old, key=lambda p: p.stat().st_mtime) if old else None)


def dispersion_suite(run: Path) -> list[tuple[str, list[Path]]]:
    """Every figure of the dispersion engineering, in groups: the map overview (design map and
    dashboard), the nine maps, D and the guided modes against width and height, the candidates
    compared before the choice, the mesh convergence and the finalists of the chosen
    cross-section, and the ring's dispersion profile. New runs keep them in figures/2_* and
    figures/3_*; older runs under results/."""
    run = Path(run)
    res, fig = run / "results", run / "figures"
    groups: list[tuple[str, list[Path]]] = []
    m = latest_map(run)
    if m is not None:
        groups.append(("Dispersion map: overview (design map, decision dashboard)",
                       [p for p in (m / "design_map.png", m / "maps" / "dashboard.png") if p.exists()]))
        groups.append(("Dispersion map: the nine maps", sorted((m / "maps").glob("map*.png"))))
        groups += [(title, sorted((m / sub).glob("*.png"))) for sub, title in MAP_GROUPS]
    cand = fig / "2_candidates"
    if cand.is_dir():
        groups.append(("Candidates compared", [p for p in (
            cand / "finalists_comparison.png", cand / "bend_dispersion.png",
            cand / "profiles" / "compare_D_lambda.png", cand / "profiles" / "compare_Dint.png")
            if p.exists()] + sorted(cand.glob("modes_*.png"))))
    conv = _first_dir(fig / "2_mesh_convergence", res / "mesh_convergence_plots")
    groups.append(("Mesh convergence of the chosen cross-section", sorted(conv.glob("*.png")) if conv else []))
    fin = _first_dir(fig / "2_finalists", res / "finalist_report")
    groups.append(("Finalists compared", sorted(fin.glob("*.png")) if fin else []))
    prof = _first_dir(fig / "3_dispersion_profile", res / "dispersion_profile_plots")
    groups.append(("Ring: dispersion profile", sorted(prof.glob("*.png")) if prof else []))
    return [(t, g) for t, g in groups if g]


def tables(run: Path) -> list[Path]:
    """Every table of the design: tables/ and the dispersion-map tables under results/."""
    run = Path(run)
    extra = []
    m = latest_map(run)
    if m is not None:
        extra += sorted(m.glob("*.csv"))
    for sub in ("mesh_convergence_plots", "finalist_report", "dispersion_profile_plots"):
        extra += sorted((run / "results" / sub).glob("*.csv"))
    return sorted((run / "tables").glob("*.csv")) + extra


def files(run: Path) -> list[dict]:
    """Every result file of the design (not cloud data), with folder, size and date."""
    run = Path(run)
    rows = []
    for sub in ("", "figures", "tables", "layout", "lle", "benchmark_pack"):
        d = run / sub if sub else run
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.is_file():
                s = p.stat()
                rows.append(dict(folder=sub or ".", file=p.name, size_kB=round(s.st_size / 1024, 1),
                                 modified=dt.datetime.fromtimestamp(s.st_mtime).strftime("%Y-%m-%d %H:%M"),
                                 path=str(p.resolve())))
    return rows


# ------------------------------------------------------------------ runs
def command(run: Path, only: list[str], mode: str = "run", redo: Optional[list[str]] = None,
            spend_cap: Optional[float] = None) -> list[str]:
    """The command line of a run. ``mode``: 'run' (cloud steps within the limits and the
    spend cap) or 'price' (estimates only, nothing priced runs)."""
    argv = [sys.executable, "-m", "tarang.design", "--config", str(Path(run) / "design.toml"), "--yes"]
    if only:
        argv += ["--only", ",".join(only)]
    if redo:
        argv += ["--redo", ",".join(redo)]
    if mode == "price":
        argv.append("--price-only")
    elif spend_cap is not None:
        argv += ["--spend-cap", f"{float(spend_cap):g}"]
    return argv


def start(run: Path, argv: list[str]) -> dict:
    """Start a design run in the background; its output goes to <run>/gui_log.txt."""
    run = Path(run)
    if job(run).get("running"):
        raise RuntimeError("a run of this design is already going")
    log = run / LOG_FILE
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    kw: dict = {}
    if os.name == "nt":
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP    # so Ctrl+Break reaches it
    else:
        kw["start_new_session"] = True
    fh = open(log, "w", encoding="utf-8")
    fh.write("$ " + " ".join(argv) + "\n\n")
    fh.flush()
    proc = subprocess.Popen(argv, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            cwd=str(Path.cwd()), env=env, **kw)
    fh.close()
    _PROCS[str(run.resolve())] = proc
    info = dict(pid=proc.pid, argv=argv, started=dt.datetime.now().isoformat(timespec="seconds"),
                log=str(log))
    (run / JOB_FILE).write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32                              # type: ignore[attr-defined]
        h = k32.OpenProcess(0x1000, False, int(pid))              # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259                     # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:                                       # a finished child not yet reaped is a zombie
        with open(f"/proc/{int(pid)}/stat", encoding="utf-8") as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return True


def process_start(pid: int) -> Optional[float]:
    """When the process ``pid`` started (seconds since the epoch), or None if unknown."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            k32 = ctypes.windll.kernel32                          # type: ignore[attr-defined]
            h = k32.OpenProcess(0x1000, False, int(pid))
            if not h:
                return None
            try:
                t = [wintypes.FILETIME() for _ in range(4)]
                if not k32.GetProcessTimes(h, *[ctypes.byref(x) for x in t]):
                    return None
                ticks = (t[0].dwHighDateTime << 32) | t[0].dwLowDateTime   # 100 ns since 1601
                return ticks / 1e7 - 11644473600.0
            finally:
                k32.CloseHandle(h)
        if os.path.exists("/proc/stat"):
            with open(f"/proc/{int(pid)}/stat", encoding="utf-8") as f:
                ticks = int(f.read().split(")")[-1].split()[19])          # field 22: starttime
            with open("/proc/stat", encoding="utf-8") as f:
                boot = next(int(x.split()[1]) for x in f if x.startswith("btime"))
            return boot + ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, StopIteration, AttributeError):
        return None
    return None


def _same_process(info: dict) -> bool:
    """False when the saved pid now belongs to another program (after a restart of the
    computer the number is reused): its start time is far from the job's."""
    try:
        started = dt.datetime.fromisoformat(info["started"]).timestamp()
    except (KeyError, TypeError, ValueError):
        return True
    t = process_start(int(info.get("pid", 0)))
    return t is None or abs(t - started) < 120


def job(run: Path) -> dict:
    """The last run started from the GUI: its info, whether it is still going and its exit
    code when known."""
    run = Path(run)
    p = run / JOB_FILE
    if not p.exists():
        return {}
    try:
        info = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    proc = _PROCS.get(str(run.resolve()))
    if proc is not None and proc.pid == info.get("pid"):
        code = proc.poll()
        info.update(running=code is None, returncode=code)
    else:
        pid = int(info.get("pid", 0))
        info.update(running=pid_alive(pid) and _same_process(info), returncode=None)
    return info


def stop(run: Path, force: bool = False) -> bool:
    """Ask the running design run to stop as with Ctrl+C (work done so far is kept); with
    ``force``, end it at once (what was finished is kept too; a stage in progress is lost)."""
    info = job(run)
    if not info.get("running"):
        return False
    pid = int(info["pid"])
    try:
        if os.name == "nt":
            os.kill(pid, signal.SIGTERM if force else signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
        else:
            os.killpg(os.getpgid(pid), signal.SIGKILL if force else signal.SIGINT)
    except OSError:
        return False
    return True


def log_tail(run: Path, lines: int = 400) -> str:
    p = Path(run) / LOG_FILE
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(text[-lines:])


def new_design(folder: Path, name: str, wavelength_um: float, example: Optional[str] = None) -> Path:
    """A new design folder with a design.toml (from an example file, or the defaults)."""
    from tarang.design.config import DesignConfig
    folder = Path(folder)
    if (folder / "design.toml").exists():
        raise FileExistsError(f"{folder / 'design.toml'} exists already")
    cfg = DesignConfig.load(example) if example else DesignConfig()
    cfg.run.name = name
    cfg.run.wavelength_um = float(wavelength_um)
    return cfg.save(folder / "design.toml")


def shutdown(delay: float = 1.0) -> None:
    """End this GUI server after ``delay`` seconds (time for the page to show that it
    stopped). Runs started from the GUI are separate processes and keep going."""
    import os
    import signal
    import threading

    threading.Timer(delay, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
