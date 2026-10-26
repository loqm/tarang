"""TARANG design GUI (Streamlit). Start it with ``python -m tarang.gui``.

Left: pick a design folder, the view, its credit limits and the Tidy3D balance.
Guided design (the default view, ``guided.py``): one step at a time: what to design, then
each stage with a drawing, its settings, Run (cloud steps priced first), and what it
recommends, its results and figures.
All results: choose stages, then Estimate cost (runs nothing that costs), Run, or Stop;
tabs Summary, Whole device, Figures, Tables, Report, Files, Settings, Log.

Everything shown is read from the design folder (state.json, figures/, tables/, ...), the
same files the command line writes, so the GUI and ``python -m tarang.design`` can be
used on the same design in turn.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from tarang.gui import data

try:
    import streamlit as st
except ImportError:                    # the gui extra is missing; python -m tarang.gui says how
    st = None

IMG = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$")


def _args():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--root", default=os.environ.get("TARANG_GUI_ROOT", "design_runs"))
    a, _ = ap.parse_known_args(sys.argv[1:])
    return a


def _open_folder(path: Path) -> None:
    try:
        if os.name == "nt":
            os.startfile(str(path))                                   # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as err:                                          # noqa: BLE001
        st.warning(f"Could not open the folder: {err}")


def _table(rows, **kw):
    if rows:
        st.dataframe(rows, hide_index=True, **kw)
    else:
        st.caption("(nothing yet)")


# ------------------------------------------------------------------ sidebar
def sidebar(root_default: str):
    st.sidebar.title("TARANG")
    st.sidebar.caption("Microrings and test waveguides, with grating or edge couplers")
    root = st.sidebar.text_input("Folder with your designs", value=st.session_state.get("root", root_default),
                                 help="The folder that holds your design folders (each has a design.toml).")
    st.session_state["root"] = root
    runs = data.find_runs(root) if Path(root).exists() else []
    if not runs:
        st.sidebar.info("No design.toml found under this folder. Create a design below, or "
                        "type the folder where your design_runs are.")
        run = None
    else:
        labels = [f"{p.name}  ({p.parent})" for p in runs]
        i = st.sidebar.selectbox("Design", range(len(runs)), format_func=lambda k: labels[k])
        run = runs[i]
    with st.sidebar.expander("New design"):
        name = st.text_input("Name", value="my_design")
        wl = st.number_input("Wavelength (µm)", value=1.55, min_value=0.3, max_value=5.0, step=0.01,
                             format="%.3f")
        ex = sorted((Path(__file__).resolve().parents[2] / "examples").glob("design_*.toml"))
        pick = st.selectbox("Start from", ["defaults"] + [e.name for e in ex])
        if st.button("Create"):
            try:
                example = None if pick == "defaults" else next(e for e in ex if e.name == pick)
                p = data.new_design(Path(root) / name, name, wl, example)
                st.success(f"Created {p}. Check its Settings tab, then Run.")
                st.rerun()
            except Exception as err:                                  # noqa: BLE001
                st.error(str(err))
    if run is not None:
        st.sidebar.radio("View", ["Guided design", "All results"], key="mode",
                         help="Guided: one step at a time, with what each step recommends. All "
                              "results: every stage, figure, table and file at once.")
        credits_panel(run)
        from tarang.gui import guided
        guided.jobs_box(run, where=st.sidebar)
    exit_panel(run)
    return run


def exit_panel(run) -> None:
    """Exit: stops the GUI (as Ctrl+C in its window does). A run started from the GUI keeps
    going unless it is stopped here too."""
    st.sidebar.divider()
    running = bool(run is not None and data.job(run).get("running"))
    if st.sidebar.button("Exit TARANG", key="exit", help="Stops the GUI; your designs and results "
                         "stay in their folders. Start it again with python -m tarang.gui."):
        if not running:
            _exit_now()
        st.session_state["exit_ask"] = True
    if st.session_state.get("exit_ask") and running:
        st.sidebar.warning("A run of this design is still going. It can keep going without the GUI "
                           "(its output goes on into gui_log.txt; open the GUI again to watch it).")
        a, b = st.sidebar.columns(2)
        if a.button("Exit, run goes on", key="exit_keep"):
            _exit_now()
        if b.button("Stop run and exit", key="exit_stop"):
            data.stop(run)
            _exit_now()
        if st.sidebar.button("Cancel", key="exit_cancel"):
            st.session_state.pop("exit_ask", None)
            st.rerun()


def _exit_now() -> None:
    st.session_state.pop("exit_ask", None)
    st.success("TARANG has stopped. You can close this browser tab; start it again with "
               "`python -m tarang.gui`.")
    data.shutdown()
    st.stop()


def credits_panel(run: Path):
    st.sidebar.subheader("FlexCredits")
    try:
        c = data.credits(run, account=False)
    except Exception as err:                                          # noqa: BLE001
        st.sidebar.error(f"design.toml not readable: {err}")
        return
    acc = st.session_state.get(f"acc:{run}")
    a, b = st.sidebar.columns(2)
    a.metric("Spent", f"{c['spent']:.2f}")
    b.metric("Left of limit", f"{c['left']:.2f}")
    if st.sidebar.button("Check Tidy3D balance", help="Asks Tidy3D (a few seconds; needs your API key)"):
        acc = data.credits(run, account=True)["account"] or {"credit": None}
        st.session_state[f"acc:{run}"] = acc
    if acc:
        if acc.get("credit") is None:
            st.sidebar.caption("Balance not readable (no API key, or offline).")
        else:
            st.sidebar.caption(f"Account: **{acc['credit']:.2f} FC**, valid until {acc.get('expires') or '?'}"
                               + (" (EXPIRED)" if acc.get("expired") else ""))
    with st.sidebar.form("limits"):
        tot = st.number_input("Limit for this design (FC)", value=float(c["limit"]), min_value=0.0, step=0.5)
        per = st.number_input("Largest single cloud run (FC)", value=float(c["per_run"]), min_value=0.01, step=0.25)
        if st.form_submit_button("Save limits"):
            credit = (acc or {}).get("credit")
            msg = data.set_limits(run, total=tot, per_run=per, credit=credit)
            (st.error(msg) if msg else st.success("Saved in design.toml."))
    with st.sidebar.expander("Spent on"):
        _table(c["rows"])


# ------------------------------------------------------------------ run panel
def run_panel(run: Path):
    rows = data.stage_rows(run)
    names = [r["stage"] for r in rows]
    todo = [r["stage"] for r in rows if r["status"] != "done"] or ["report"]
    job = data.job(run)
    running = bool(job.get("running"))
    with st.container(border=True):
        st.markdown("**Run stages**")
        c1, c2, c3 = st.columns([3, 2, 1.3])
        only = c1.multiselect("Stages", names, default=[n for n in names if n in todo],
                              help="Done stages with the same settings are not repeated.")
        redo = c2.multiselect("Repeat even if done", names, default=[],
                              help="Repeating a cloud stage reuses its saved cloud results (free).")
        try:
            left = max(data.credits(run)["left"], 0.0)
        except Exception:                                             # noqa: BLE001
            left = 0.0
        cap = c3.number_input("Spend at most (FC)", value=round(min(left, 2.0), 2), min_value=0.0,
                              step=0.25, help="Cloud steps that would take the spending of this run "
                                              "above this are skipped.")
        b1, b2, b3, _ = st.columns([1.3, 1, 1, 3])
        if b1.button("Estimate cost", disabled=running or not only,
                     help="Goes through the stages and prints every cloud estimate; nothing that "
                          "costs is run (free local stages still run)."):
            data.start(run, data.command(run, only, "price", redo))
            st.rerun()
        if b2.button("Run", type="primary", disabled=running or not only):
            data.start(run, data.command(run, only, "run", redo, cap))
            st.rerun()
        if b3.button("Stop", disabled=not running,
                     help="Like Ctrl+C: finished stages and cloud runs are kept."):
            data.stop(run)
            st.session_state[f"stopped:{run}"] = job.get("pid")
        if running and st.session_state.get(f"stopped:{run}") == job.get("pid"):
            if st.button("Force stop", help="If Stop does not end the run within a minute: ends it "
                                            "at once (finished stages and cloud runs are still kept)."):
                data.stop(run, force=True)
        st.fragment(run_every=2)(live_log)(str(run))


def live_log(run_s: str):
    run = Path(run_s)
    job = data.job(run)
    was = st.session_state.get("was_running")
    if job.get("running"):
        st.info(f"Running since {job['started']} … (output below; it is also in {run / data.LOG_FILE})")
    elif job:
        code = job.get("returncode")
        st.caption(f"Last run started {job['started']}"
                   + (f", finished (exit code {code})" if code is not None else ", finished") + ".")
    tail = data.log_tail(run, 25)
    if tail and job.get("running"):
        st.code(tail, language=None)
    elif tail:
        with st.expander("Output of the last run (all of it: Log tab)"):
            st.code(tail, language=None)
    st.session_state["was_running"] = bool(job.get("running"))
    if was and not job.get("running"):
        st.rerun(scope="app")                    # the run ended: refresh results everywhere


# ------------------------------------------------------------------ tabs
def tab_summary(run: Path):
    a, b = st.columns([3, 2])
    with a:
        st.markdown("**Design**")
        _table(data.summary_rows(run))
    with b:
        st.markdown("**Fibre-to-fibre loss at the design wavelength**")
        _table(data.loss_rows(run))
    st.markdown("**Stages**")
    _table(data.stage_rows(run))


def tab_device(run: Path):
    figs = dict(data.figures(run)).get("Whole device", [])
    if not figs:
        st.info("Run the 'device' stage (free) for the fibre-to-fibre spectrum and every resonance.")
        return
    for f in figs:
        st.image(str(f), caption=f.name, width="stretch")
    rows = data.read_csv(run / "tables" / "device_resonances.csv")
    st.markdown("**Every resonance** (`tables/device_resonances.csv`)")
    _table(rows, height=320)


def tab_figures(run: Path):
    groups = data.figures(run)
    if not groups:
        st.info("No figures yet.")
        return
    names = [g for g, _ in groups]
    pick = st.selectbox("Group", names, help="figures/ in the report's order, then the "
                        "dispersion-engineering figures kept under results/")
    for f in dict(groups)[pick]:
        st.image(str(f), caption=str(f), width="stretch")


def tab_tables(run: Path):
    tabs = data.tables(run)
    if not tabs:
        st.info("No tables yet.")
        return
    i = st.selectbox("Table", range(len(tabs)), format_func=lambda k: tabs[k].relative_to(run).as_posix())
    p = tabs[i]
    st.caption(str(p))
    _table(data.read_csv(p, limit=5000), height=420)
    st.download_button("Download CSV", p.read_bytes(), file_name=p.name, mime="text/csv")


def tab_report(run: Path):
    rep = run / "report.md"
    if not rep.exists():
        st.info("The report is written by the 'report' stage.")
        return
    buf: list[str] = []
    for line in rep.read_text(encoding="utf-8").splitlines():
        m = IMG.match(line.strip())
        if m:
            if buf:
                st.markdown("\n".join(buf))
                buf = []
            p = (run / m.group(2)).resolve()
            if p.exists():
                st.image(str(p), caption=m.group(1), width="stretch")
        else:
            buf.append(line)
    if buf:
        st.markdown("\n".join(buf))


def tab_files(run: Path):
    st.markdown(f"Design folder: `{run.resolve()}`")
    if st.button("Open the design folder"):
        _open_folder(run.resolve())
    _table(data.files(run), height=480)


def tab_settings(run: Path):
    p = run / "design.toml"
    st.caption(f"{p.resolve()}: every setting with its explanation. Changing a stage's settings "
               "makes that stage and the ones after it run again.")
    text = st.text_area("design.toml", p.read_text(encoding="utf-8"), height=520)
    if st.button("Save settings"):
        from tarang.design.config import DesignConfig
        tmp = run / ".design.toml.check"
        tmp.write_text(text, encoding="utf-8")
        try:
            DesignConfig.load(tmp)
        except Exception as err:                                      # noqa: BLE001
            st.error(f"Not saved: {err}")
        else:
            p.write_text(text, encoding="utf-8")
            st.success("Saved.")
        finally:
            tmp.unlink(missing_ok=True)


def tab_log(run: Path):
    st.caption(str(run / data.LOG_FILE))
    st.code(data.log_tail(run, 3000) or "(no run started from the GUI yet)", language=None)


def main():
    st.set_page_config(page_title="TARANG design", page_icon="〰", layout="wide")
    run = sidebar(_args().root)
    if run is None:
        st.title("TARANG design")
        st.write("Choose the folder with your designs on the left (each design folder holds a "
                 "`design.toml`), or create a new design.")
        return
    try:
        cfg = data.load_config(run)
        st.title(f"{cfg.run.name}  ·  {cfg.run.wavelength_um:g} µm")
    except Exception as err:                                          # noqa: BLE001
        st.title(run.name)
        st.error(f"design.toml not readable: {err}")
    st.caption(f"Folder: {run.resolve()}")
    if st.session_state.get("mode", "Guided design") == "Guided design":
        from tarang.gui import guided
        guided.page(run, live_log)
        return
    run_panel(run)
    tabs = st.tabs(["Summary", "Whole device", "Figures", "Tables", "Report", "Files", "Settings", "Log"])
    for t, fn in zip(tabs, (tab_summary, tab_device, tab_figures, tab_tables, tab_report, tab_files,
                            tab_settings, tab_log)):
        with t:
            fn(run)


if __name__ == "__main__":              # streamlit runs this file as __main__
    main()
