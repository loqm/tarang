"""The guided design screens of the GUI (Streamlit); the logic is in ``wizard.py``.

Left: the steps of this design with their status. Right: the chosen step: a drawing of
what it designs, its settings, Run (local steps) or Estimate cost then Run (cloud steps),
and once it has run, what it recommends, its key numbers and its figures.
"""
from __future__ import annotations

from pathlib import Path

from tarang.gui import data, illustrations as il, wizard as wz

try:
    import streamlit as st
except ImportError:                    # the gui extra is missing; python -m tarang.gui says how
    st = None

ICON = {"done": "✅", "choose": "🟡", "changed": "🔁", "skipped": "⏭️", "to do": "⚪"}
STATUS_TEXT = {"done": "done", "choose": "waiting for your choice", "changed": "settings changed: run again",
               "skipped": "skipped or off", "to do": "not run yet"}


# step buttons: the whole title on one or two lines, left-aligned (no "…")
NAV_CSS = """<style>
div[class*="st-key-nav-"] button {justify-content: flex-start; text-align: left; height: auto;
  min-height: 2.5rem; padding-top: .35rem; padding-bottom: .35rem}
div[class*="st-key-nav-"] button p, div[class*="st-key-nav-"] button div {white-space: normal;
  overflow: visible; text-overflow: clip; text-align: left}
[data-testid="stCheckbox"] label p, [data-testid="stCheckbox"] label div {white-space: normal;
  overflow: visible; text-overflow: clip}
</style>"""


def _img(svg: str, width: int = 320) -> None:
    st.markdown(il.as_img(svg, width), unsafe_allow_html=True)


def _table(rows, **kw):
    if rows:
        st.dataframe(rows, hide_index=True, **kw)


def _current(run: Path, steps, statuses) -> str:
    key = f"step:{run}"
    keys = [s.key for s in steps]
    cur = st.session_state.get(key)
    if cur not in keys:
        cur = next((s.key for s in steps if statuses[s.key] != "done"), keys[-1])
        st.session_state[key] = cur
    return cur


def _go(run: Path, key: str) -> None:
    st.session_state[f"step:{run}"] = key


# ------------------------------------------------------------------ the page
def page(run: Path, live_log) -> None:
    try:
        cfg = data.load_config(run)
    except Exception as err:                                          # noqa: BLE001
        st.error(f"design.toml not readable: {err}")
        return
    steps = wz.steps_for(cfg)
    rows = data.stage_rows(run)
    statuses = {s.key: wz.step_status(run, s, rows) for s in steps}
    cur = _current(run, steps, statuses)
    st.markdown(NAV_CSS, unsafe_allow_html=True)
    nav, main = st.columns([1.15, 3.3], gap="medium")
    with nav:
        st.markdown("**Steps**")
        for i, s in enumerate(steps, 1):
            label = f"{ICON[statuses[s.key]]} {i}. {s.title}"
            if st.button(label, key=f"nav-{s.key}", width="stretch",
                         type="primary" if s.key == cur else "secondary"):
                _go(run, s.key)
                st.rerun()
        done = sum(1 for s in steps if statuses[s.key] in ("done", "skipped"))
        st.progress(done / len(steps), text=f"{done} of {len(steps)} steps done")
        st.caption("  \n".join(f"{ICON[k]} {STATUS_TEXT[k]}" for k in ICON))
    with main:
        s = wz.step(cur)
        i = [x.key for x in steps].index(cur)
        st.subheader(f"Step {i + 1} of {len(steps)}: {s.title}")
        st.caption(s.explain)
        bad = wz.problems(cfg)
        if bad:
            st.error("**A setting cannot be right; fix it before running a step:**  \n"
                     + "  \n".join(f"`{k}`: {v}" for k, v in bad.items())
                     + ("" if s.key == "start" else "  \n(The film, BOX and fibre are set in step 1.)"))
        if s.key == "start":
            start_step(run, cfg)
        else:
            stage_step(run, cfg, s, statuses[s.key], live_log)
        st.divider()
        b1, _, b2 = st.columns([1, 3, 1])
        if i > 0 and b1.button("← Back", key="back"):
            _go(run, steps[i - 1].key)
            st.rerun()
        if i < len(steps) - 1 and b2.button("Next →", key="next", type="primary"
                                            if statuses[cur] in ("done", "skipped") else "secondary"):
            _go(run, steps[i + 1].key)
            st.rerun()


# ------------------------------------------------------------------ step 1
def start_step(run: Path, cfg) -> None:
    goal, coupling = wz._goal(cfg), wz._coupling(cfg)
    st.markdown("##### 1. What is the chip for?")
    a, b = st.columns(2)
    with a:
        _img(il.ring_bus())
    with b:
        _img(il.spirals())
    goal_new = st.radio("Goal", ["comb_ring", "loss_test"], index=0 if goal == "comb_ring" else 1,
                        format_func=lambda k: wz.CHOICE_LABELS[k], horizontal=True,
                        label_visibility="collapsed", key=f"goal:{run}")
    st.markdown("##### 2. How does light get on and off the chip?")
    a, b = st.columns(2)
    with a:
        _img(il.grating_uniform())
    with b:
        _img(il.edge_coupler())
    coupling_new = st.radio("Coupling", ["grating", "edge"], index=0 if coupling == "grating" else 1,
                            format_func=lambda k: wz.CHOICE_LABELS[k], horizontal=True,
                            label_visibility="collapsed", key=f"coupling:{run}")
    values = {"run.goal": goal_new, "run.coupling": coupling_new}
    if coupling_new == "grating":
        st.markdown("##### 3. Which grating couplers?")
        st.caption("The uniform grating is always designed. The apodised and focusing gratings are "
                   "extra steps (small priced runs); the better one is used on the mask, and "
                   "`layout.coupler` can put every kind side by side.")
        a, b, c = st.columns(3)
        with a:
            _img(il.grating_uniform(), 260)
            st.checkbox("Uniform", value=True, disabled=True, key="gc_u")
        with b:
            _img(il.grating_apodised(), 260)
            values["apodise.run"] = st.checkbox("Apodised", value=bool(cfg.apodise.run), key="gc_a")
        with c:
            _img(il.grating_focusing(), 260)
            values["focus.run"] = st.checkbox("Focusing", value=bool(cfg.focus.run), key="gc_f")
    else:
        st.markdown("##### 3. Your lensed fibre")
        a, b = st.columns([1, 1.2])
        with a:
            _img(il.lensed_fibre(float(cfg.edge.fiber_mfd_um)), 300)
        with b:
            values["edge.fiber_mfd_um"] = st.number_input(
                "Spot size, MFD (µm)", value=float(cfg.edge.fiber_mfd_um), min_value=0.5, step=0.1,
                help="Mode-field diameter of the lensed fibre's spot (datasheet).")
            values["edge.mfd_tolerance_um"] = st.number_input(
                "Its tolerance ± (µm)", value=float(cfg.edge.mfd_tolerance_um), min_value=0.0, step=0.1)
            st.caption("The waveguide widens to a wide facet at the chip edge (no top cladding "
                       "needed).")
    st.markdown("##### 4. Wavelength, film and materials")
    a, b, c = st.columns(3)
    values["run.name"] = a.text_input("Design name", value=str(cfg.run.name))
    values["run.wavelength_um"] = b.number_input("Design wavelength (µm)", value=float(cfg.run.wavelength_um),
                                                 min_value=0.3, max_value=5.0, step=0.01, format="%.3f")
    film = c.text_input("Film thickness, if fixed (µm)", value="" if cfg.stack.core_height_um is None
                        else f"{cfg.stack.core_height_um:g}", placeholder="e.g. 0.7; empty: heights are mapped",
                        help="In µm (0.7 for a 700 nm film): the wafer you have or ordered. Empty: "
                             "several heights are mapped in the Cross-section step.")
    a, b, c = st.columns(3)
    values["stack.box_height_um"] = a.number_input("BOX thickness (µm)", value=float(cfg.stack.box_height_um),
                                                   min_value=0.5, step=0.5)
    values["materials.core"] = b.text_input("Core material", value=str(cfg.materials.core),
                                            help="A key in tarang/media/media.json or a CSV path.")
    values["materials.cladding"] = c.text_input("Top cladding", value=str(cfg.materials.cladding),
                                                help="air for an air-clad chip")
    presets = wz._choices()["ring.loss_preset"]
    values["ring.loss_preset"] = st.selectbox(
        "Propagation loss to start from (each from a publication; your measured values replace them)",
        presets, index=presets.index(cfg.ring.loss_preset) if cfg.ring.loss_preset in presets else 0,
        format_func=wz.preset_label, key=f"preset:{run}")
    st.caption(wz.preset_detail(values["ring.loss_preset"])
               + " For an air-clad, fully etched film, typical_ebl is the realistic start; the "
                 "other two papers are oxide-clad.")
    if st.button("Save and continue", type="primary"):
        try:
            values["stack.core_height_um"] = float(film) if film.strip() else None
        except ValueError:
            st.error("Film thickness: a number in µm (0.7 for 700 nm), or empty.")
            return
        msg = wz.save_fields(run, values)
        if msg:
            st.error(msg)
        else:
            wz.mark_choices_saved(run)
            new = data.load_config(run)
            steps = wz.steps_for(new)
            _go(run, steps[1].key)
            st.rerun()


# ------------------------------------------------------------------ the other steps
def settings_form(run: Path, cfg, s) -> None:
    infos = wz.visible_fields(cfg, s)
    if not infos:
        return
    with st.form(f"form:{s.key}"):
        raw = {}
        cols = st.columns(2)
        for k, info in enumerate(infos):
            c = cols[k % 2]
            label = info["label"]
            key = f"f:{info['dotted']}"
            if info["kind"] == "bool":
                raw[info["dotted"]] = c.checkbox(label, value=bool(info["value"]), help=info["help"], key=key)
            elif info["kind"] == "choice":
                opts = info["choices"]
                v = info["value"] if info["value"] is not None else opts[0]
                raw[info["dotted"]] = c.selectbox(label, opts, index=opts.index(v) if v in opts else 0,
                                                  help=info["help"], key=key)
            elif info["kind"] == "int":
                raw[info["dotted"]] = c.number_input(label, value=int(info["value"]), step=1,
                                                     help=info["help"], key=key)
            elif info["kind"] == "float":
                raw[info["dotted"]] = c.number_input(label, value=float(info["value"]), help=info["help"],
                                                     format="%g", key=key)
            else:
                dflt = wz.default_text(cfg, info["dotted"], run)
                empty = not wz.show_value(info)
                raw[info["dotted"]] = c.text_input(
                    label, value=wz.show_value(info), placeholder=f"empty: {dflt}" if dflt else "",
                    help=info["help"] + (f"  Empty: {dflt}." if dflt else ""), key=key)
                if dflt and empty:
                    c.caption(f"Empty, so: **{dflt}**")
        if st.form_submit_button("Save settings"):
            try:
                vals = {i["dotted"]: wz.parse_value(i, raw[i["dotted"]]) for i in infos}
            except (TypeError, ValueError) as err:
                st.error(f"Not saved: {err}")
                return
            msg = wz.save_fields(run, vals)
            st.error(f"Not saved: {msg}") if msg else st.success("Saved in design.toml.")


def run_controls(run: Path, s, status: str, live_log) -> None:
    job = data.job(run)
    running = bool(job.get("running")) or bool(wz.problems(data.load_config(run)))
    missing = wz.missing_before(run, s)
    if missing and not running:
        st.warning("Needs the earlier step(s) first: " + ", ".join(missing))
    cloud = wz.is_cloud(s)
    pkey = f"price:{run}:{s.key}"
    # a finished price-only run of this step: read its total
    if not running and job and "--price-only" in job.get("argv", []) and \
            ",".join(s.stages) in " ".join(job.get("argv", [])):
        tot = wz.price_total(data.log_tail(run, 400))
        if tot is not None:
            st.session_state[pkey] = tot
    price = st.session_state.get(pkey)
    again = status in ("done", "skipped")
    c1, c2, c3 = st.columns([1.3, 1.6, 1])
    if cloud:
        if c1.button("Estimate cost", disabled=running, key=f"est:{s.key}",
                     help="Goes through the step and prints every cloud estimate; nothing that costs "
                          "runs (free local parts do run)."):
            wz.run_step(run, s, "price", again=again)
            st.rerun()
        if price is None:
            c2.caption("Estimate the cost first.")
        label = (f"Run (up to {price:.3f} FC)" if price else "Run (free: saved results)") \
            if price is not None else "Run"
        if c2.button(label, type="primary", disabled=running or price is None, key=f"run:{s.key}"):
            wz.run_step(run, s, "run", spend_cap=round(price * 1.05 + 0.005, 3), again=again)
            st.session_state.pop(pkey, None)
            st.rerun()
    else:
        if c1.button("Run again (free)" if again else "Run (free)", type="primary", disabled=running,
                     key=f"run:{s.key}"):
            wz.run_step(run, s, "run", again=again)
            st.rerun()
    if c3.button("Stop", disabled=not job.get("running"), key=f"stop:{s.key}",
                 help="Like Ctrl+C: finished stages and cloud runs are kept."):
        data.stop(run)
    st.fragment(run_every=2)(live_log)(str(run))


def cross_section_choice(run: Path) -> None:
    cands = wz.candidates(run)
    if not cands:
        return
    st.markdown("**Choose the cross-section** (candidates from the map, most robust first)")
    _table(cands)
    labels = [f"{c['H_um']} × {c['W_um']} µm: D = {c.get('D_TE0', 0):.0f} ps/(nm·km), "
              f"{c.get('target_neighbours', '?')} neighbours also targets" for c in cands]
    i = st.selectbox("Cross-section", range(len(cands)), format_func=lambda k: labels[k]
                     + ("  (recommended)" if k == 0 else ""), key="xs_pick")
    if st.button("Use this cross-section and check it", type="primary", disabled=bool(data.job(run).get("running"))):
        msg = wz.choose_cross_section(run, cands[i]["H_um"], cands[i]["W_um"])
        if msg:
            st.error(msg)
        else:
            wz.run_step(run, wz.step("cross_section"), "run")
            st.rerun()


def results_panel(run: Path, s, status: str) -> None:
    if s.key == "cross_section" and status == "choose":
        cross_section_choice(run)
        return
    if status not in ("done", "changed", "skipped"):
        return
    rec = wz.recommendation(run, s)
    if rec:
        with st.container(border=True):
            st.markdown("**Recommended**")
            _table(rec["rows"])
            st.caption("Why: " + rec["why"])
            differs = any(r["in use"] not in (r["recommended"], "(result)")
                          and not r["in use"].startswith("(empty") for r in rec["rows"])
            if rec["settable"] and differs:
                if st.button("Use the recommended values", key=f"rec:{s.key}"):
                    msg = wz.apply_recommended(run, s)
                    st.error(msg) if msg else st.success("Saved. Run this step again to use them.")
    nums = wz.key_numbers(run, s)
    if nums:
        st.markdown("**Results**")
        _table(nums)
    if s.key == "cross_section" and wz.candidates(run):
        with st.expander("Choose another cross-section"):
            cross_section_choice(run)
    figs = wz.step_figures(run, s)
    if s.key == "layout":
        p = wz.gds_preview(run)
        if p:
            st.image(str(p), caption="design.gds as it goes to the mask", width="stretch")
            gds = run / "layout" / "design.gds"
            st.download_button("Download design.gds", gds.read_bytes(), file_name=f"{run.name}_design.gds")
    if figs:
        with st.expander(f"Figures ({len(figs)})", expanded=True):
            for f in figs:
                st.image(str(f), caption=f.name, width="stretch")
    if s.key == "report":
        downloads(run)
        rows = data.summary_rows(run)
        if rows:
            st.markdown("**The design in numbers**")
            _table([dict(quantity=r["quantity"], value=r["value"], unit=r["unit"]) for r in rows])
        if (run / "report.md").exists():
            with st.expander("Read the report here"):
                from tarang.gui.app import tab_report
                tab_report(run)


def downloads(run: Path) -> None:
    st.markdown("**Downloads**")
    c1, c2, c3, c4 = st.columns(4)
    h = wz.report_html(run)
    if h:
        c1.download_button("Report (HTML)", h.encode("utf-8"), file_name=f"{run.name}_report.html",
                           mime="text/html")
        c2.download_button("Report (Markdown)", (run / "report.md").read_bytes(), file_name=f"{run.name}_report.md")
    c3.download_button("Everything (zip)", wz.design_zip(run), file_name=f"{run.name}.zip", mime="application/zip")
    bp = run / "benchmark_pack.zip"
    if bp.exists():
        c4.download_button("Benchmark pack (zip)", bp.read_bytes(), file_name=f"{run.name}_benchmark_pack.zip",
                           mime="application/zip")
    st.caption("The HTML report holds every figure; open it in a browser and print it (Save as PDF) for a "
               "PDF.")


def stage_step(run: Path, cfg, s, status: str, live_log) -> None:
    a, b = st.columns([1, 1.6], gap="large")
    with a:
        if s.picture:
            _img(s.picture(cfg), 330)
        st.markdown(f"Status: **{STATUS_TEXT[status]}**")
    with b:
        settings_form(run, cfg, s)
    run_controls(run, s, status, live_log)
    if wz.is_cloud(s):
        jobs_box(run, s)
    results_panel(run, s, status)
    with st.expander("All settings (design.toml, advanced)"):
        p = run / "design.toml"
        text = st.text_area("design.toml", p.read_text(encoding="utf-8"), height=360, key=f"toml:{s.key}")
        if st.button("Save design.toml", key=f"tomlsave:{s.key}"):
            from tarang.design.config import DesignConfig
            tmp = run / ".design.toml.check"
            tmp.write_text(text, encoding="utf-8")
            try:
                DesignConfig.load(tmp)
            except Exception as err:                                  # noqa: BLE001
                st.error(f"Not saved: {err}")
            else:
                p.write_text(text, encoding="utf-8")
                st.success("Saved.")
            finally:
                tmp.unlink(missing_ok=True)


def jobs_box(run: Path, s=None, where=None) -> None:
    """The Tidy3D jobs of this step (or of the whole design), each with its link."""
    where = where or st
    rows = wz.jobs(run, s)
    title = (f"Tidy3D jobs of this step ({len(rows)})" if s is not None
             else f"Tidy3D jobs of this design ({len(rows)})")
    with where.expander(title):
        if not rows:
            st.caption("None yet. Each cloud run appears here with its link once it is submitted.")
        else:
            if s is not None:
                rows = [{k: v for k, v in r.items() if k != "step"} for r in rows]
            st.dataframe(rows, hide_index=True, width="stretch",
                         column_config={"link": st.column_config.LinkColumn("Tidy3D", display_text="open ↗")})
            st.caption("A link opens the run in the Tidy3D web interface (set-up, mesh, results; "
                       "log in with your Tidy3D account).")
        if any(not r["link"] for r in rows):
            st.caption("Runs saved before TARANG kept task records have no link yet.")
            if st.button("Find the missing links (asks Tidy3D)", key=f"backfill:{s.key if s else 'all'}"):
                from tarang.utilities.cloud_run import backfill_task_records
                n = backfill_task_records(run / "cloud")
                st.success(f"{n} link(s) found.") if n else st.info("No more found (offline, or no API key).")
