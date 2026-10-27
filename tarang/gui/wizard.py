"""The guided design flow of the GUI: the steps, their settings, status, results,
recommendations and downloads (no Streamlit here, so everything is testable on its own).

A design goes through the steps one at a time. Each step is one or more stages of
``python -m tarang.design``: the GUI saves the step's settings in ``design.toml``, runs
the step's stages in the background (cloud stages are priced first), then shows the
step's results, its figures and what it recommends. Where a step offers a choice (the
cross-section from the map, the bend radius and cutback lengths, the edge coupler's facet
width, taper shape and length, the ring radius), the recommended value can be taken with
one click, or another value typed, and the step run again.
"""
from __future__ import annotations

import base64
import dataclasses
import datetime as dt
import html
import io
import json
import re
import zipfile
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Callable, Optional

from tarang.gui import data, illustrations as il


# ------------------------------------------------------------------ the steps
@dataclass
class Step:
    key: str
    title: str
    stages: tuple
    explain: str
    fields: list = field(default_factory=list)
    figures: str = ""                       # regex on figure file names
    picture: Optional[Callable] = None       # cfg -> SVG
    when: Optional[Callable] = None          # cfg -> bool, beyond the stages' goal/coupling


def _goal(cfg) -> str:
    return str(getattr(cfg.run, "goal", "comb_ring") or "comb_ring")


def _coupling(cfg) -> str:
    return str(getattr(cfg.run, "coupling", "grating") or "grating").lower()


def _start_picture(cfg) -> str:
    return il.ring_bus() if _goal(cfg) == "comb_ring" else il.spirals()


def _xs_picture(cfg) -> str:
    H = cfg.cross_section.height_um or cfg.stack.core_height_um or 0.7
    W = cfg.cross_section.width_um or 1.0
    return il.cross_section(float(H), float(W), float(cfg.stack.box_height_um), cfg.materials.cladding)


STEPS: list[Step] = [
    Step("start", "What to design", (),
         "Choose what the chip is for and how light gets on and off it. Everything after this "
         "follows from these choices; they can be changed later (the steps they affect run again).",
         ["run.name", "run.goal", "run.coupling", "run.wavelength_um", "stack.core_height_um",
          "stack.box_height_um", "materials.core", "materials.box", "materials.cladding",
          "ring.loss_preset", "apodise.run", "focus.run", "edge.fiber_mfd_um"],
         picture=_start_picture),
    Step("materials", "Materials", ("materials",),
         "Refractive index of every material over the band, fitted for the solvers (local, free, "
         "seconds).", ["run.band_um"], r"^1[a-z]?_", lambda c: il.materials()),
    Step("cross_section", "Cross-section", ("cross_section",),
         "A map of core heights × widths (local mode solver, free, minutes). For a comb the "
         "waveguide needs anomalous dispersion and one guided TE mode; for test waveguides one "
         "guided TE mode. Pick one of the recommended candidates; it is then checked at a finer "
         "mesh with its fabrication tolerance.",
         ["cross_section.heights_um", "cross_section.widths_um", "cross_section.tolerance_width_um",
          "cross_section.tolerance_height_um", "cross_section.map_steps",
          "cross_section.decision_steps", "run.workers"], r"^2", _xs_picture),
    Step("ring", "Ring", ("ring",),
         "Ring radius from the free spectral range, intrinsic Q from the losses, bend loss and "
         "integrated dispersion (local, free).",
         ["ring.fsr_ghz", "ring.radius_um", "ring.scattering_loss_db_cm",
          "ring.absorption_loss_db_cm", "ring.bend_check"], r"^3[a-v]?_", lambda c: il.ring_bus()),
    Step("waveguides", "Test waveguides", ("waveguides",),
         "Bend loss against radius, the smallest spiral radius, and the cutback lengths that "
         "measure the loss to the precision you want (local, free).",
         ["waveguides.lengths_cm", "waveguides.coupler_scatter_db", "waveguides.target_precision",
          "waveguides.bend_radius_um", "waveguides.bend_radii_um", "waveguides.spiral_gap_um",
          "ring.scattering_loss_db_cm", "ring.absorption_loss_db_cm"], r"^3w", lambda c: il.spirals()),
    Step("gap", "Ring–bus gap", ("gap",),
         "3D FDTD of the coupling region at several gaps (cloud, priced first): the critical and "
         "over-coupled gaps.",
         ["gap.gaps_um", "gap.over_coupling", "gap.span_angle_deg", "gap.band_points",
          "gap.fine_steps"], r"^4", lambda c: il.gap_detail()),
    Step("grating", "Grating coupler", ("grating",),
         "2D FDTD of the grating for your fibre angle (cloud, priced first): period, fill factor "
         "and fibre position for the highest efficiency at the design wavelength.",
         ["grating.etch", "grating.etch_depth_um", "grating.fiber_angle_deg", "grating.fiber_tilt",
          "grating.mfd_um", "grating.n_teeth", "grating.min_feature_um", "grating.check_3d"], r"^5",
         lambda c: il.grating_uniform()),
    Step("checks", "Checks", ("checks",),
         "Fabrication tolerances, mode families of the ring, coupling across the band, the "
         "grating's back-reflection and the loss budget (free parts and small cloud runs).",
         ["checks.band", "checks.grating_extras", "checks.angle_tolerance_deg", "checks.litho_bias_um",
          "checks.mesh_grating", "checks.mesh_gap", "checks.mode_families"], r"^7",
         lambda c: il.checks_picture()),
    Step("taper", "Grating width and taper", ("taper",),
         "Grating width for the fibre (free) and the taper length to the waveguide (free 2D "
         "model; one priced EME run).",
         ["taper.grating_width_um", "taper.run_eme", "taper.max_loss_db", "taper.eme_cells",
          "taper.eme_modes", "taper.use_in_layout"], r"^8", lambda c: il.taper_picture()),
    Step("apodise", "Apodised grating", ("apodise",),
         "Teeth that change along the grating so the light leaves as a Gaussian like the fibre "
         "mode (small priced 2D runs).",
         ["apodise.min_trench_um", "apodise.verify", "apodise.prefer", "apodise.ramp_check",
          "apodise.reflection_check", "apodise.use_in_layout"], r"^9",
         lambda c: il.grating_apodised(), lambda c: bool(c.apodise.run)),
    Step("focus", "Focusing grating", ("focus",),
         "Curved teeth that focus the light into the waveguide tip (free model; optional 3D "
         "FDTD, priced).",
         ["focus.teeth", "focus.max_half_angle_deg", "focus.check_3d", "focus.mesh_steps",
          "focus.use_in_layout"], r"^10", lambda c: il.grating_focusing(), lambda c: bool(c.focus.run)),
    Step("edge", "Edge coupler", ("edge",),
         "Facet width for your lensed fibre (free), taper shape and length (free 2D model; priced "
         "EME), and one small 3D FDTD of the facet (priced): coupling, alignment and reflection.",
         ["edge.fiber_mfd_um", "edge.mfd_tolerance_um", "edge.facet_width_um", "edge.profiles",
          "edge.profile", "edge.taper_length_um", "edge.max_loss_db", "edge.run_eme", "edge.check_3d",
          "edge.offset_um"], r"^12", lambda c: il.edge_coupler()),
    Step("layout", "Layout (GDS)", ("layout",),
         "The mask: every device with its couplers, reference waveguides, gap variants or "
         "cutback spirals; parameters for other layout tools (local, free).",
         ["layout.coupler", "layout.reference", "layout.gap_offsets_um", "layout.bus_length_um",
          "layout.tone", "edge.chip_length_um", "edge.chip_width_um", "edge.dicing_lane_um",
          "edge.pitch_um", "edge.facet_straight_um", "edge.facet_overhang_um"], r"^6",
         lambda c: il.layout_chip(_coupling(c) == "edge")),
    Step("device", "Whole device", ("device",),
         "Fibre-to-fibre transmission over the band from every stage's result: each resonance's "
         "Q and dip, the ripple between the couplers, or the expected cutback measurement "
         "(local, free).", ["device.reflections", "device.zoom_fsr"], r"^11",
         lambda c: il.device_spectrum()),
    Step("report", "Report and downloads", ("report",),
         "The report with every figure and how each number was obtained, the tables, the GDS and "
         "the benchmark pack for another solver (local, free).", [], "", lambda c: il.report_picture()),
]
STEP_KEYS = [s.key for s in STEPS]

# fields that only make sense with a goal or a coupling, or with another field on
FIELD_WHEN = {
    "apodise.run": lambda c: _coupling(c) == "grating",
    "focus.run": lambda c: _coupling(c) == "grating",
    "edge.fiber_mfd_um": lambda c: _coupling(c) == "edge",
    "layout.coupler": lambda c: _coupling(c) == "grating",
    "layout.bus_length_um": lambda c: _coupling(c) == "grating",
    "layout.gap_offsets_um": lambda c: _goal(c) == "comb_ring",
    "layout.reference": lambda c: _goal(c) == "comb_ring",
    "grating.etch_depth_um": lambda c: str(c.grating.etch) == "partial",
    "checks.band": lambda c: _goal(c) == "comb_ring",
    "checks.mesh_gap": lambda c: _goal(c) == "comb_ring",
    "checks.mode_families": lambda c: _goal(c) == "comb_ring",
    "checks.grating_extras": lambda c: _coupling(c) == "grating",
    "checks.angle_tolerance_deg": lambda c: _coupling(c) == "grating",
    "checks.litho_bias_um": lambda c: _coupling(c) == "grating",
    "checks.mesh_grating": lambda c: _coupling(c) == "grating",
    "device.zoom_fsr": lambda c: _goal(c) == "comb_ring",
    "focus.mesh_steps": lambda c: bool(c.focus.check_3d),
    "taper.eme_cells": lambda c: bool(c.taper.run_eme),
    "taper.eme_modes": lambda c: bool(c.taper.run_eme),
}
for _k in ("edge.chip_length_um", "edge.chip_width_um", "edge.dicing_lane_um", "edge.pitch_um",
           "edge.facet_straight_um", "edge.facet_overhang_um"):
    FIELD_WHEN[_k] = lambda c: _coupling(c) == "edge"


# short form labels (the full explanation is the field's help, shown on hover)
LABELS = {
    "run.band_um": "Band half-width for the material fit (µm)",
    "cross_section.heights_um": "Core heights to map (µm)", "cross_section.widths_um": "Core widths to map (µm)",
    "cross_section.tolerance_width_um": "Width tolerance ± (µm)",
    "cross_section.tolerance_height_um": "Thickness tolerance ± (µm)",
    "cross_section.map_steps": "Mesh steps per wavelength (map)",
    "cross_section.decision_steps": "Mesh steps per wavelength (check)", "run.workers": "Parallel workers",
    "ring.fsr_ghz": "Free spectral range (GHz)", "ring.radius_um": "Ring radius (µm; empty: from the FSR)",
    "ring.scattering_loss_db_cm": "Scattering loss (dB/cm)", "ring.absorption_loss_db_cm": "Absorption loss (dB/cm)",
    "ring.bend_check": "Check the bend loss",
    "waveguides.lengths_cm": "Cutback lengths (cm)", "waveguides.coupler_scatter_db": "Coupling scatter per structure (dB)",
    "waveguides.target_precision": "Wanted precision of the loss (0.1 = 10 %)",
    "waveguides.bend_radius_um": "Smallest spiral radius (µm; empty: recommended)",
    "waveguides.bend_radii_um": "Bend radii to check (µm)", "waveguides.spiral_gap_um": "Gap between spiral turns (µm)",
    "gap.gaps_um": "Gaps to simulate (µm)", "gap.over_coupling": "Over-coupling K of the second gap",
    "gap.span_angle_deg": "Ring arc simulated (deg)", "gap.band_points": "Wavelengths at the ring ports",
    "gap.fine_steps": "Mesh steps per wavelength (waveguide)",
    "grating.etch": "Etch", "grating.etch_depth_um": "Etch depth (µm, partial etch)",
    "grating.fiber_angle_deg": "Fibre angle from vertical (deg)", "grating.fiber_tilt": "Beam travels … the waveguide",
    "grating.mfd_um": "Fibre MFD (µm)", "grating.n_teeth": "Number of teeth",
    "grating.min_feature_um": "Smallest feature (µm)", "grating.check_3d": "Also simulate in 3D",
    "checks.band": "Coupling across the band", "checks.grating_extras": "Grating tolerances and reflection",
    "checks.angle_tolerance_deg": "Fibre angle tolerance ± (deg)", "checks.litho_bias_um": "Lithography bias ± (µm)",
    "checks.mesh_grating": "Grating mesh check", "checks.mesh_gap": "Gap mesh check (3D, costly)",
    "checks.mode_families": "Ring mode families",
    "taper.grating_width_um": "Grating width (µm; empty: best for the fibre)", "taper.run_eme": "3D EME of the taper",
    "taper.max_loss_db": "Accepted taper loss (dB)", "taper.eme_cells": "EME cells", "taper.eme_modes": "EME modes",
    "taper.use_in_layout": "Use in the layout",
    "apodise.min_trench_um": "Narrowest trench (µm)", "apodise.verify": "Check in 2D FDTD",
    "apodise.prefer": "Better means", "apodise.ramp_check": "Also try the linear ramp",
    "apodise.reflection_check": "Simulate its back-reflection", "apodise.use_in_layout": "Use in the layout if better",
    "focus.teeth": "Teeth along the axis", "focus.max_half_angle_deg": "Largest light-cone half-angle (deg)",
    "focus.check_3d": "Check in 3D FDTD", "focus.mesh_steps": "3D mesh steps per wavelength",
    "focus.use_in_layout": "Main coupler on the mask",
    "edge.fiber_mfd_um": "Lensed-fibre spot MFD (µm)", "edge.mfd_tolerance_um": "Spot size tolerance ± (µm)",
    "edge.facet_width_um": "Facet width (µm; empty: recommended)", "edge.profiles": "Taper shapes to compare",
    "edge.profile": "Taper shape used", "edge.taper_length_um": "Taper length (µm; empty: recommended)",
    "edge.max_loss_db": "Accepted taper loss (dB)", "edge.run_eme": "3D EME of the tapers",
    "edge.check_3d": "3D FDTD of the facet", "edge.offset_um": "Fibre offsets for the tolerance ± (µm)",
    "edge.chip_length_um": "Chip length (µm)", "edge.chip_width_um": "Chip width (µm)",
    "edge.dicing_lane_um": "Dicing lane (µm)", "edge.pitch_um": "Spacing between facets (µm)",
    "edge.facet_straight_um": "Straight before the chip edge (µm)",
    "edge.facet_overhang_um": "Overhang into the dicing lane (µm)",
    "layout.coupler": "Couplers on the mask", "layout.reference": "Reference waveguide beside every ring",
    "layout.gap_offsets_um": "Extra ring variants: gap changed by (µm)", "layout.bus_length_um": "Bus length (µm)",
    "layout.tone": "Polygon tone", "device.reflections": "Include the couplers' reflections",
    "device.zoom_fsr": "Close-up half-width (FSRs)",
}


def _choices() -> dict:
    from tarang.design.assumptions import LOSS_PRESETS
    from tarang.design.layout import COUPLER_SETTINGS
    from tarang.design.stages import COUPLINGS, GOALS
    from tarang.design.taper import PROFILES
    return {"run.goal": list(GOALS), "run.coupling": list(COUPLINGS),
            "grating.etch": ["full", "partial"], "grating.fiber_tilt": ["towards", "away"],
            "apodise.prefer": ["bandwidth", "efficiency"], "focus.teeth": ["best", "uniform", "apodised"],
            "layout.coupler": list(COUPLER_SETTINGS), "layout.tone": ["structure", "trench"],
            "gap.reference": ["ring_port", "bus_only", "source"],
            "ring.loss_preset": list(LOSS_PRESETS), "edge.profile": ["(recommended)", *PROFILES]}


CHOICE_LABELS = {
    "comb_ring": "Ring resonator for frequency combs",
    "loss_test": "Test waveguides for propagation loss",
    "grating": "Grating couplers (fibre from above)",
    "edge": "Edge couplers (lensed fibre at the chip facet)",
}


def is_cloud(step: Step) -> bool:
    """True if any of the step's stages runs on Tidy3D (priced first)."""
    from tarang.design.stages import STAGES
    return any(kind == "cloud" for name, _, kind, _ in STAGES if name in step.stages)


def applies(step: Step, cfg) -> bool:
    from tarang.design.stages import applies as stage_applies
    if any(not stage_applies(s, cfg) for s in step.stages):
        return False
    return step.when(cfg) if step.when else True


def steps_for(cfg) -> list[Step]:
    """The steps of this design, in order."""
    return [s for s in STEPS if applies(s, cfg)]


def step(key: str) -> Step:
    return STEPS[STEP_KEYS.index(key)]


# ------------------------------------------------------------------ settings
def field_info(cfg, dotted: str) -> dict:
    """What the form needs for one setting: value, explanation, unit, kind, choices."""
    sec, key = dotted.split(".")
    f = {x.name: x for x in fields(getattr(cfg, sec))}[key]
    value = getattr(getattr(cfg, sec), key)
    default = f.default if f.default is not dataclasses.MISSING else f.default_factory()
    choices = _choices().get(dotted)
    if choices:
        kind = "choice"
    elif isinstance(default, bool):
        kind = "bool"
    elif isinstance(default, int) and not isinstance(default, bool):
        kind = "int"
    elif isinstance(default, float):
        kind = "float"
    elif isinstance(default, list):
        kind = "list"
    elif default is None:
        kind = "optional"
    else:
        kind = "str"
    unit = f.metadata.get("unit", "")
    label = LABELS.get(dotted) or (f.metadata.get("help", key).split(" (")[0][:60]
                                   + (f" [{unit}]" if unit else ""))
    return dict(dotted=dotted, section=sec, key=key, value=value, label=label,
                help=f.metadata.get("help", ""),
                unit=f.metadata.get("unit", ""), kind=kind, choices=choices)


def visible_fields(cfg, st: Step) -> list[dict]:
    return [field_info(cfg, d) for d in st.fields if FIELD_WHEN.get(d, lambda c: True)(cfg)]


def parse_value(info: dict, raw):
    """A form value in the setting's own type ("" = empty for optional values and lists)."""
    from tarang.design.ui import parse_list
    kind = info["kind"]
    if kind == "choice":
        return None if raw == "(recommended)" else raw
    if kind == "bool":
        return bool(raw)
    if kind == "int":
        return int(raw)
    if kind == "float":
        return float(raw)
    if kind == "list":
        if isinstance(raw, (list, tuple)):
            return list(raw)
        raw = str(raw).strip()
        return parse_list(raw) if raw else []
    if kind == "optional":
        raw = "" if raw is None else str(raw).strip()
        if raw == "":
            return None
        try:
            return float(raw)
        except ValueError:
            return raw
    return str(raw)


def show_value(info: dict) -> str:
    v = info["value"]
    if v is None:
        return ""
    if isinstance(v, list):
        return ", ".join(f"{x:g}" if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x)
                         for x in v)
    return str(v)


def _list_text(v: list) -> str:
    return ", ".join(f"{x:g}" for x in v)


def _recommended(run: Optional[Path], dotted: str):
    """The value a done step recommends for this setting, or None."""
    if run is None:
        return None
    for stage, mapping in RECOMMEND_FIELDS.items():
        for k, d in mapping.items():
            if d == dotted:
                rec = (results(run, stage).get("recommend") or {}).get("recommended") or {}
                return rec.get(k)
    return None


def default_text(cfg, dotted: str, run: Optional[Path] = None) -> str:
    """What an empty setting stands for, with its value where it can be worked out now
    (shown greyed in the empty field)."""
    from tarang.design import assumptions, stages as sg
    from tarang.design.waveguides import default_radii
    wl = float(cfg.run.wavelength_um)
    rec = _recommended(run, dotted)
    if rec not in (None, [], ""):
        return f"the recommended {_list_text(rec) if isinstance(rec, list) else _fmt(rec)}"
    try:
        pre = assumptions.preset(cfg.ring.loss_preset)
    except ValueError:
        pre = {}
    known = {
        "run.band_um": lambda: f"{round(0.13 * wl, 3):g} (0.13 × wavelength)",
        "cross_section.heights_um": lambda: _list_text(sg._heights_default(cfg)),
        "cross_section.widths_um": lambda: _list_text(sg._widths_default(cfg)),
        "gap.gaps_um": lambda: _list_text(sg._gaps_default(cfg)),
        "waveguides.bend_radii_um": lambda: _list_text(default_radii(wl)),
        "ring.scattering_loss_db_cm": lambda: f"{pre['scattering_db_cm']:g} (preset {cfg.ring.loss_preset})",
        "ring.absorption_loss_db_cm": lambda: f"{pre['absorption_db_cm']:g} (preset {cfg.ring.loss_preset})",
        "ring.radius_um": lambda: "from the FSR and the group index (shown when the step runs)",
        "apodise.min_trench_um": lambda: f"{cfg.grating.min_feature_um:g} (grating.min_feature_um)",
        "stack.core_height_um": lambda: "heights are mapped",
        "grating.etch_depth_um": lambda: f"{(cfg.cross_section.height_um or 0) / 2:g} (half the core)",
    }
    if dotted in known:
        try:
            return known[dotted]()
        except Exception:                                             # noqa: BLE001
            pass
    sec, key = dotted.split(".")
    f = {x.name: x for x in fields(getattr(cfg, sec))}[key]
    m = re.search(r"\((?:default|empty)[:]?\s*([^)]*)\)", f.metadata.get("help", ""))
    return m.group(1).strip() if m else ""


def preset_label(name: str) -> str:
    """'typical_ebl: 0.40 dB/cm, Colación et al., Sci. Rep. (2025)' for the loss drop-down."""
    from tarang.design.assumptions import LOSS_PRESETS
    p = LOSS_PRESETS[name]
    tot = p["scattering_db_cm"] + p["absorption_db_cm"]
    return f"{name}: {tot:.3g} dB/cm, {p['short']}"


def preset_detail(name: str) -> str:
    from tarang.design.assumptions import LOSS_PRESETS
    p = LOSS_PRESETS[name]
    return (f"Scattering {p['scattering_db_cm']:g} + absorption {p['absorption_db_cm']:g} dB/cm. "
            f"{p['note']} Source: {p['source']}.")


def problems(cfg) -> dict:
    """Settings outside their plausible range (e.g. nm typed for µm)."""
    from tarang.design.config import implausible
    return implausible(cfg)


def save_fields(run: Path, values: dict) -> str:
    """Set {"section.key": value} in design.toml (checked by reading the file back); returns
    "" or why it was refused (then nothing is changed)."""
    from tarang.design.config import DesignConfig
    run = Path(run)
    cfg = data.load_config(run)
    if "ring.loss_preset" in values and values["ring.loss_preset"] != cfg.ring.loss_preset:
        # a new preset: its losses replace the old preset's (unless typed in the same form)
        for k in ("ring.scattering_loss_db_cm", "ring.absorption_loss_db_cm"):
            values.setdefault(k, None)
    from tarang.design.config import range_problem
    for dotted, v in values.items():
        msg = range_problem(dotted, v)
        if msg:
            return msg
    for dotted, v in values.items():
        sec, key = dotted.split(".")
        setattr(getattr(cfg, sec), key, v)
    tmp = run / ".design.toml.check"
    try:
        cfg.save(tmp)
        back = DesignConfig.load(tmp)
        from tarang.design.stages import COUPLINGS, GOALS
        if back.run.goal not in GOALS:
            return f"run.goal must be one of {', '.join(GOALS)}"
        if str(back.run.coupling).lower() not in COUPLINGS:
            return f"run.coupling must be one of {', '.join(COUPLINGS)}"
    except Exception as err:                                      # noqa: BLE001
        return f"{type(err).__name__}: {err}"
    finally:
        tmp.unlink(missing_ok=True)
    cfg.save(run / "design.toml")
    return ""


# ------------------------------------------------------------------ status and results
def mark_choices_saved(run: Path) -> None:
    """Remember (in state.json, which the design runs keep as it is) that step 1 was done."""
    p = Path(run) / "state.json"
    state = data.load_state(run)
    if not p.exists():                               # as a design run starts it
        from tarang.design.stages import FIGURE_VERSION
        state = {"stages": {}, "spent": 0.0, "figure_version": FIGURE_VERSION}
    state["gui_choices_saved"] = True
    p.write_text(json.dumps(state, indent=1), encoding="utf-8")


def results(run: Path, stage: str) -> dict:
    return ((data.load_state(run).get("stages") or {}).get(stage) or {}).get("results") or {}


def step_status(run: Path, st: Step, rows: Optional[list] = None) -> str:
    """'done', 'choose' (a result waits for your choice), 'changed' (settings changed since
    it ran), 'skipped', or 'to do'."""
    if not st.stages:                               # step 1: done once its choices were saved
        state = data.load_state(run)
        return "done" if state.get("gui_choices_saved") or state.get("stages") else "to do"
    rows = rows if rows is not None else data.stage_rows(run)
    by = {r["stage"]: r["status"] for r in rows}
    if any(results(run, s).get("awaiting_choice") for s in st.stages):
        return "choose"
    stat = [by.get(s, "not run") for s in st.stages]
    if all(s == "done" for s in stat):
        return "done"
    if any(s.startswith("settings changed") for s in stat):
        return "changed"
    if all(s in ("done", "skipped / off") for s in stat) and any(s == "skipped / off" for s in stat):
        return "skipped"
    return "to do"


def missing_before(run: Path, st: Step, cfg=None) -> list[str]:
    """Titles of the earlier steps of this design that are not done (their results are
    needed by this one)."""
    cfg = cfg or data.load_config(run)
    rows = data.stage_rows(run)
    out = []
    for s in steps_for(cfg):
        if s.key == st.key:
            break
        if s.stages and step_status(run, s, rows) not in ("done", "skipped"):
            out.append(s.title)
    return out


def step_figures(run: Path, st: Step) -> list[Path]:
    if not st.figures:
        return []
    figs = sorted((Path(run) / "figures").glob("*.png"))
    rx = re.compile(st.figures)
    return [f for f in figs if rx.search(f.name)]


def suite(run: Path, st: Step) -> list[tuple[str, list[Path]]]:
    """The step's extra figures kept under results/ (the dispersion-engineering suite of the
    cross-section map, the ring's dispersion profile), in groups."""
    groups = data.dispersion_suite(run)
    keep = {"cross_section": lambda t: not t.startswith("Ring"), "ring": lambda t: t.startswith("Ring")}
    if st.key not in keep:
        return []
    return [(t, g) for t, g in groups if keep[st.key](t)]


def key_numbers(run: Path, st: Step) -> list[dict]:
    rows = [r for r in data.summary_rows(run) if r.get("stage") in st.stages]
    return [dict(quantity=r["quantity"], value=r["value"], unit=r["unit"]) for r in rows]


# how a stage's recommendation maps onto settings (the rest is information only)
RECOMMEND_FIELDS = {
    "cross_section": {"height_um": "cross_section.height_um", "width_um": "cross_section.width_um"},
    "ring": {"radius_um": "ring.radius_um"},
    "waveguides": {"bend_radius_um": "waveguides.bend_radius_um", "lengths_cm": "waveguides.lengths_cm"},
    "edge": {"facet_width_um": "edge.facet_width_um", "profile": "edge.profile",
             "taper_length_um": "edge.taper_length_um"},
}
NICE = {"height_um": "core height (µm)", "width_um": "core width (µm)", "radius_um": "ring radius (µm)",
        "bend_radius_um": "smallest bend radius (µm)", "lengths_cm": "cutback lengths (cm)",
        "facet_width_um": "facet width (µm)", "profile": "taper shape",
        "taper_length_um": "taper length (µm)", "critical_gap_um": "critical gap (µm)",
        "over_coupled_gap_um": "over-coupled gap (µm)", "over_coupling": "over-coupling K",
        "period_um": "grating period (µm)", "fill_factor": "fill factor",
        "source_x_um": "fibre position (µm)", "recipe": "apodisation recipe",
        "use_in_layout": "use in the layout", "tip_width_um": "tip width (µm)",
        "distance_um": "tip to first tooth (µm)"}


def recommendation(run: Path, st: Step) -> Optional[dict]:
    """{rows: [{what, recommended, in use}], why, settable} for the step's stages."""
    cfg = data.load_config(run)
    for s in st.stages:
        rec = results(run, s).get("recommend")
        if not rec or not rec.get("recommended"):
            continue
        mapping = RECOMMEND_FIELDS.get(s, {})
        rows, settable = [], False
        for k, v in rec["recommended"].items():
            use = None
            if k in mapping:
                sec, key = mapping[k].split(".")
                use = getattr(getattr(cfg, sec), key)
                settable = True
            rows.append({"what": NICE.get(k, k), "recommended": _fmt(v),
                         "in use": _fmt(use) if k in mapping else "(result)"})
        return dict(stage=s, rows=rows, why=rec.get("why", ""), settable=settable)
    return None


def _fmt(v) -> str:
    if v is None:
        return "(empty: the recommended one)"
    if isinstance(v, list):
        return ", ".join(_fmt(x) for x in v)
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def apply_recommended(run: Path, st: Step) -> str:
    """Write the step's recommended values into design.toml (those that are settings)."""
    rec = recommendation(run, st)
    if not rec or not rec["settable"]:
        return "nothing to set"
    src = results(run, rec["stage"])["recommend"]["recommended"]
    vals = {RECOMMEND_FIELDS[rec["stage"]][k]: v for k, v in src.items()
            if k in RECOMMEND_FIELDS[rec["stage"]] and v is not None}
    return save_fields(run, vals)


def candidates(run: Path) -> list[dict]:
    """The cross-section map's candidates, most robust first."""
    r = results(run, "cross_section")
    return list(r.get("candidates") or (r.get("recommend") or {}).get("choices") or [])


#: columns of the candidates table in the choice
CANDIDATE_COLUMNS = ("H_um", "W_um", "D_TE0", "ng_TE0", "guided_TE", "guided_TM", "TE_TM_split",
                     "target_neighbours")


def hw_key(c) -> str:
    """'0.7x1' for a candidate or comparison row (or [H, W])."""
    if isinstance(c, (list, tuple)):
        h, w = c
    else:
        h, w = c["H_um"], c["W_um"]
    return f"{float(h):g}x{float(w):g}"


def compare_list(run: Path) -> list[str]:
    return [hw_key([float(v) for v in str(t).lower().replace("×", "x").split("x")])
            for t in (data.load_config(run).cross_section.compare or [])]


def compare_candidates(run: Path, keys: list[str]) -> str:
    """Ask the next cross-section run to compare these candidates before the choice."""
    return save_fields(run, {"cross_section.compare": list(keys), "cross_section.height_um": None,
                             "cross_section.width_um": None})


def comparison(run: Path) -> Optional[dict]:
    """The comparison of candidates the cross-section stage made, if any."""
    return results(run, "cross_section").get("comparison") or None


def comparison_figures(run: Path) -> list[Path]:
    d = Path(run) / "figures" / "2_candidates"
    if not d.is_dir():
        return []
    fixed = [d / "finalists_comparison.png", d / "bend_dispersion.png",
             d / "profiles" / "compare_D_lambda.png", d / "profiles" / "compare_Dint.png"]
    return [p for p in fixed if p.exists()] + sorted(d.glob("modes_*.png"))


def choice_status(run: Path) -> Optional[str]:
    """While the cross-section step runs from the choice panel: what it is doing and since
    when (the buttons are greyed until it finishes)."""
    info = data.job(run)
    if not info.get("running"):
        return None
    try:
        since = dt.datetime.fromisoformat(info["started"]).strftime("%H:%M")
    except (KeyError, TypeError, ValueError):
        since = "?"
    xs = data.load_config(run).cross_section
    if xs.height_um and xs.width_um:
        what = (f"Checking {xs.height_um:g} × {xs.width_um:g} µm (tolerance box, bend, mesh "
                "convergence)")
    elif xs.compare:
        what = f"Comparing {len(xs.compare)} candidate(s) (tolerance box, bend, profiles each)"
    else:
        what = "Running the cross-section step"
    return (f"{what} since {since}. The buttons come back when it finishes; the log shows "
            "progress. A tolerance point takes about 1 to 2 min on a laptop, so this can take "
            "15 to 20 min per cross-section.")


def choose_cross_section(run: Path, H: float, W: float) -> str:
    return save_fields(run, {"cross_section.height_um": float(H), "cross_section.width_um": float(W),
                             "cross_section.stop_after_map": False})


def prepare(run: Path, st: Step) -> None:
    """Before a step runs: the cross-section map stops for your choice when no cross-section
    has been chosen yet (test waveguides with a given one skip the map)."""
    if st.key == "cross_section":
        cfg = data.load_config(run)
        if not (cfg.cross_section.height_um and cfg.cross_section.width_um):
            save_fields(run, {"cross_section.stop_after_map": True})


def run_step(run: Path, st: Step, mode: str = "run", spend_cap: Optional[float] = None,
             again: bool = False) -> dict:
    """Start the step's stages in the background (``mode`` 'price': estimates only)."""
    prepare(run, st)
    stages = list(st.stages)
    return data.start(run, data.command(run, stages, mode, stages if again else None, spend_cap))


def price_total(log: str) -> Optional[float]:
    """The total of a price-only run from its output: FC, 0 when nothing costs, None when
    the run has not printed its estimates (yet)."""
    m = re.findall(r"Total ([0-9.]+) FC", log)
    if m:
        return float(m[-1])
    if "no cloud step needs paying" in log:
        return 0.0
    return None


# ------------------------------------------------------------------ Tidy3D jobs
#: cloud/ sub-folder -> the step whose runs it holds (the grating folder also holds the
#: checks' grating variants and the apodise step's reference grating)
CLOUD_FOLDERS = {"gap_sweep": "gap", "grating": "grating", "band": "checks", "mesh_check": "checks",
                 "taper": "taper", "apodise": "apodise", "focus": "focus", "edge_taper": "edge",
                 "edge_facet": "edge"}


def jobs(run: Path, st: Optional[Step] = None) -> list[dict]:
    """The Tidy3D jobs of this design (of one step when ``st`` is given), newest first:
    step, name, status, estimate, billed cost, submitted, link (from each run's task.json;
    nothing is asked of Tidy3D)."""
    from tarang.utilities.cloud_run import collect_runs
    root = Path(run) / "cloud"
    if not root.exists():
        return []
    out = []
    for r in collect_runs(root):
        rel = Path(r["folder"]).relative_to(root).parts
        step_key = CLOUD_FOLDERS.get(rel[0], rel[0]) if rel else ""
        if st is not None and step_key != st.key and not (st.key == "checks" and rel and rel[0] == "grating"):
            continue
        out.append(dict(link=r.get("url") or "", step=step_key, name=r["name"], status=r["status"],
                        estimate_FC=r.get("estimated_cost"), billed_FC=r.get("real_cost"),
                        submitted=(r.get("submitted") or "")[:16].replace("T", " ")))
    out.sort(key=lambda r: r["submitted"], reverse=True)
    return out


# ------------------------------------------------------------------ downloads
def _md_inline(text: str) -> str:
    t = html.escape(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2">\1</a>', t)
    return t


def report_html(run: Path) -> Optional[str]:
    """report.md as one self-contained HTML page (figures embedded): open it in a browser and
    print it to PDF, or send it as it is."""
    run = Path(run)
    rep = run / "report.md"
    if not rep.exists():
        return None
    out, para, table, in_list = [], [], [], False

    def flush():
        nonlocal para, table, in_list
        if para:
            out.append("<p>" + " ".join(_md_inline(x) for x in para) + "</p>")
            para = []
        if table:
            rows = [r for r in table if not re.match(r"^\|\s*:?-{2,}", r)]
            cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
            h = "".join(f"<th>{_md_inline(c)}</th>" for c in cells[0])
            b = "".join("<tr>" + "".join(f"<td>{_md_inline(c)}</td>" for c in r) + "</tr>" for r in cells[1:])
            out.append(f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>")
            table = []
        if in_list:
            out.append("</ul>")
            in_list = False
    for line in rep.read_text(encoding="utf-8").splitlines():
        s = line.rstrip()
        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", s.strip())
        if m:
            flush()
            p = (run / m.group(2)).resolve()
            # only pictures inside the design folder: a report.md edited by hand must not
            # pull any other file of the computer into the page
            if p.is_relative_to(run.resolve()) and p.exists() and p.suffix.lower() == ".png":
                b64 = base64.b64encode(p.read_bytes()).decode("ascii")
                out.append(f'<figure><img src="data:image/png;base64,{b64}"/>'
                           f"<figcaption>{html.escape(m.group(1))}</figcaption></figure>")
            continue
        if s.startswith("#"):
            flush()
            n = len(s) - len(s.lstrip("#"))
            out.append(f"<h{n}>{_md_inline(s[n:].strip())}</h{n}>")
        elif s.startswith("|"):
            if para:
                flush()
            table.append(s)
        elif s.startswith("- "):
            if para or table:
                flush()
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_md_inline(s[2:])}</li>")
        elif not s.strip():
            flush()
        else:
            if table:
                flush()
            para.append(s)
    flush()
    css = ("body{font-family:Helvetica,Arial,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;"
           "color:#1f2937;line-height:1.45}table{border-collapse:collapse;margin:8px 0 16px;font-size:13px}"
           "th,td{border:1px solid #cbd5e1;padding:4px 8px;text-align:left}th{background:#eef2f7}"
           "img{max-width:100%}figure{margin:12px 0}figcaption{color:#64748b;font-size:12px}"
           "code{background:#f1f5f9;padding:0 3px}h1{font-size:26px}h2{margin-top:28px;"
           "border-bottom:1px solid #e2e8f0}@media print{h2{break-before:auto}figure,table{break-inside:avoid}}")
    title = html.escape(rep.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip())
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>"
            f"<style>{css}</style></head><body>{''.join(out)}</body></html>")


def design_zip(run: Path) -> bytes:
    """The design's results in one zip: settings, report (Markdown and HTML), figures,
    tables, layout, LLE inputs, the benchmark pack and the dispersion-map figures and tables
    under results/ (not the cloud data, not the raw solver data)."""
    run = Path(run)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("design.toml", "report.md", "design_summary.json"):
            if (run / name).exists():
                z.write(run / name, f"{run.name}/{name}")
        h = report_html(run)
        if h:
            z.writestr(f"{run.name}/report.html", h)
        for sub in ("figures", "tables", "layout", "lle", "benchmark_pack"):
            d = run / sub
            if d.exists():
                for p in sorted(d.rglob("*")):
                    if p.is_file():
                        z.write(p, f"{run.name}/{p.relative_to(run).as_posix()}")
        res = run / "results"                      # the dispersion-engineering suite (no raw data)
        if res.exists():
            for p in sorted(res.rglob("*")):
                if p.is_file() and p.suffix.lower() in (".png", ".csv", ".md"):
                    z.write(p, f"{run.name}/{p.relative_to(run).as_posix()}")
    return buf.getvalue()


def gds_preview(run: Path) -> Optional[Path]:
    """A picture of layout/design.gds as it will be written to the mask (every polygon of
    the top cell, coloured by layer), redrawn when the GDS changes."""
    run = Path(run)
    gds = run / "layout" / "design.gds"
    png = run / "layout" / "design_preview.png"
    if not gds.exists():
        return None
    if png.exists() and png.stat().st_mtime >= gds.stat().st_mtime:
        return png
    try:
        import gdstk
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
    except ImportError:
        return None
    lib = gdstk.read_gds(str(gds))
    tops = lib.top_level()
    if not tops:
        return None
    polys = tops[0].get_polygons()
    colours = {}
    palette = ["#2b6cb0", "#dd6b20", "#2f855a", "#9b2c2c", "#6b46c1", "#718096"]
    groups: dict = {}
    for q in polys:
        groups.setdefault((q.layer, q.datatype), []).append(q.points)
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, (ld, pts) in enumerate(sorted(groups.items())):
        c = colours.setdefault(ld, palette[i % len(palette)])
        etched = ld[0] < 90
        ax.add_collection(PolyCollection(pts, facecolors=c if etched else "none", edgecolors=c,
                                         linewidths=0.8 if etched else 1.2,
                                         linestyles="solid" if etched else "dashed",
                                         label=f"layer {ld[0]}/{ld[1]}" + ("" if etched else " (not etched)")))
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set(xlabel="x (µm)", ylabel="y (µm)", title=f"{gds.name}: top cell {tops[0].name}")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return png


def summary_json(run: Path) -> str:
    """Every step's key numbers as JSON (for a lab notebook)."""
    cfg = data.load_config(run)
    out = {s.title: key_numbers(run, s) for s in steps_for(cfg) if s.stages}
    return json.dumps(out, indent=1, ensure_ascii=False)
