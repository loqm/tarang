"""GDS layout and parameter export of a finished design.

Writes, in <run>/layout/:
  waveguide.gds            the bus waveguide alone
  ring_waveguide.gds       ring + bus at the critical gap (and the over-coupled gap)
  ring_waveguide_gc.gds    the same with a taper and a grating coupler at both ends
  (without a gap result, the ring files are left out; waveguide_gc.gds has the couplers)
  design.gds               all of the above, side by side
  parameters.json / .m / .csv   every dimension, for other layout tools (e.g. MATLAB)

Coordinates are in um. The bus runs along x, centred at y = 0; the ring sits above it.
A grating starts where its taper ends: tooth i occupies [i P, i P + ff P] along the
propagation direction, as in the simulation, so the layout reproduces the simulated
grating. The fibre orientation (``fiber_tilt``) and the fibre position on the grating
(from the start of the first tooth) are in parameters.json. gdstk is an optional dependency: pip install "tarang-photonics[layout]".
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


COUPLER_SETTINGS = ("auto", "uniform", "apodised", "focusing", "all")
SUFFIX = {"uniform": "gc", "apodised": "agc", "focusing": "fgc"}


def coupler_plan(ctx) -> dict:
    """Which couplers the mask gets and which one is the main coupler (loss budget, layout
    figure, whole-device model), from layout.coupler and what the stages designed.
    Returns kinds (list of (cell suffix, kind)), main, straight (the straight grating used
    when the main coupler is straight) and notes."""
    cfg = ctx.cfg
    st = ctx.state["stages"]
    from tarang.design.stages import coupling
    if coupling(cfg) == "edge":
        ed = (st.get("edge") or {}).get("results") or {}
        have = bool(ed.get("facet_width_um")) and not ed.get("skipped")
        return dict(kinds=[("ec", "edge")] if have else [], main="edge" if have else None,
                    straight=None, setting="edge", notes=[] if have else
                    ["run.coupling = edge, but the edge-coupler stage has not run: no couplers drawn"])
    gc = (st.get("grating") or {}).get("results") or {}
    ap = (st.get("apodise") or {}).get("results") or {}
    fc = (st.get("focus") or {}).get("results") or {}
    have_u = bool(gc.get("period_um"))
    have_a = bool(ap.get("verified")) and bool(ap.get("teeth"))
    have_f = bool(fc.get("curves")) and bool(cfg.focus.run)
    use_a = have_a and bool(ap.get("better_than_uniform")) and bool(cfg.apodise.use_in_layout)
    use_f = have_f and bool(cfg.focus.use_in_layout)
    setting = str(getattr(cfg.layout, "coupler", "auto") or "auto").lower()
    notes = []
    straight = "apodised" if use_a else "uniform"
    kinds = [straight] + (["focusing"] if have_f else [])
    main = "focusing" if use_f else straight
    if setting == "uniform":
        straight, kinds, main = "uniform", ["uniform"], "uniform"
    elif setting == "apodised":
        if have_a:
            straight, kinds, main = "apodised", ["apodised"], "apodised"
        else:
            notes.append("layout.coupler = apodised, but no checked apodised grating exists "
                         "(apodise.run): the uniform grating is drawn")
            straight, kinds, main = "uniform", ["uniform"], "uniform"
    elif setting == "focusing":
        if have_f:
            kinds, main = ["focusing"], "focusing"
        else:
            notes.append("layout.coupler = focusing, but no focusing grating was designed "
                         "(focus.run): the straight grating is drawn")
            kinds, main = [straight], straight
    elif setting == "all":
        kinds = ["uniform"] + (["apodised"] if have_a else []) + (["focusing"] if have_f else [])
    elif setting != "auto":
        notes.append(f"layout.coupler = {setting!r} is not one of {', '.join(COUPLER_SETTINGS)}; "
                     "'auto' is used")
    if not have_u:
        kinds, main = [], None
    # cell suffix: the straight grating is *_gc (uniform or apodised) unless both are drawn
    pairs = [("agc" if (k == "apodised" and "uniform" in kinds) else SUFFIX[k] if k != "apodised"
              else "gc", k) for k in kinds]
    return dict(kinds=pairs, main=main, straight=straight, setting=setting, notes=notes)


def device_parameters(ctx) -> dict:
    """All layout dimensions, from the earlier stages and the layout settings."""
    cfg = ctx.cfg
    st = ctx.state["stages"]
    xs = ctx.results("cross_section")
    ring = (st.get("ring") or {}).get("results") or {}
    gap = (st.get("gap") or {}).get("results") or {}
    gc = (st.get("grating") or {}).get("results") or {}
    tp = (st.get("taper") or {}).get("results") or {}
    use_tp = bool(tp) and not tp.get("skipped") and bool(cfg.taper.use_in_layout)
    ap = (st.get("apodise") or {}).get("results") or {}
    fc = (st.get("focus") or {}).get("results") or {}
    wg = (st.get("waveguides") or {}).get("results") or {}
    plan = coupler_plan(ctx)
    kinds = [k for _, k in plan["kinds"]]
    use_ap = plan["straight"] == "apodised" and "apodised" in kinds
    have_fc = "focusing" in kinds
    use_fc = plan["main"] == "focusing"
    lay = cfg.layout
    goal = str(getattr(cfg.run, "goal", "comb_ring"))
    ed = (st.get("edge") or {}).get("results") or {}
    from tarang.design.stages import coupling
    edge_on = coupling(cfg) == "edge" and bool(ed.get("facet_width_um"))
    if coupling(cfg) == "edge":                 # grating results left from a grating design
        gc, tp, ap, fc = {}, {}, {}, {}
        use_tp = use_ap = have_fc = use_fc = False
    gaps = [g for g in (gap.get("critical_gap_um"), gap.get("over_coupled_gap_um")) if g]
    variants = sorted({round(g + d, 4) for g in gaps for d in [0.0, *map(float, lay.gap_offsets_um)]
                       if g + d > 0.05})
    return {
        "name": cfg.run.name,
        "goal": goal,
        "wavelength_um": cfg.run.wavelength_um,
        "core_material": cfg.materials.core,
        "core_height_um": xs["height_um"],
        "waveguide_width_um": xs["width_um"],
        "ring_radius_um": ring.get("radius_um") if goal == "comb_ring" else None,
        "ring_width_um": xs["width_um"],
        "gap_critical_um": _r(gap.get("critical_gap_um")) if goal == "comb_ring" else None,
        "gap_over_coupled_um": _r(gap.get("over_coupled_gap_um")) if goal == "comb_ring" else None,
        "over_coupling_ratio": gap.get("over_coupling") if goal == "comb_ring" else None,
        "gap_variants_um": variants if goal == "comb_ring" else [],
        "reference_waveguide": bool(lay.reference) or goal == "loss_test",
        "couplers": [list(k) for k in plan["kinds"]],
        "main_coupler": plan["main"],
        "coupler_setting": plan["setting"],
        "cutback_lengths_cm": [float(x) for x in wg.get("lengths_cm", [])] if goal == "loss_test" else [],
        "spiral_bend_radius_um": wg.get("bend_radius_um") if goal == "loss_test" else None,
        "spiral_pitch_um": wg.get("spiral_pitch_um") if goal == "loss_test" else None,
        "coupling": coupling(cfg),
        "bus_length_um": edge_bus_length_um(cfg, ed) if edge_on else lay.bus_length_um,
        "edge_facet_width_um": ed.get("facet_width_um") if edge_on else None,
        "edge_taper_profile": ed.get("profile") if edge_on else None,
        "edge_power_exponent": ed.get("power_exponent") if edge_on else None,
        "edge_taper_length_um": ed.get("taper_length_um") if edge_on else None,
        "edge_facet_straight_um": cfg.edge.facet_straight_um if edge_on else None,
        "edge_facet_overhang_um": cfg.edge.facet_overhang_um if edge_on else None,
        "edge_fiber_mfd_um": cfg.edge.fiber_mfd_um if edge_on else None,
        "edge_fiber_height_um": (((ed.get("fdtd") or {}).get("fibre_height_um")
                                  or ed.get("fibre_height_free_um")) if edge_on else None),
        "chip_length_um": cfg.edge.chip_length_um if edge_on else None,
        "chip_width_um": cfg.edge.chip_width_um if edge_on else None,
        "dicing_lane_um": cfg.edge.dicing_lane_um if edge_on else None,
        "edge_pitch_um": cfg.edge.pitch_um if edge_on else None,
        "grating_period_um": gc.get("period_um"),
        "grating_fill_factor": gc.get("fill_factor"),
        "grating_tooth_length_um": _r(gc["period_um"] * gc["fill_factor"]) if gc.get("period_um") else None,
        "grating_teeth": len(ap["teeth"]) if use_ap else gc.get("n_teeth"),
        "uniform_teeth": gc.get("n_teeth"),
        "grating_apodised": use_ap,
        "grating_teeth_list": ([list(t) for t in ap["teeth"]] if "apodised" in kinds and ap.get("teeth")
                               else None),
        "apodised_fiber_position_um": ap.get("source_x_um") if "apodised" in kinds else None,
        "grating_width_um": (tp["grating_width_um"] if use_tp and tp.get("grating_width_um")
                             else gc.get("width_um")),
        "grating_etch": gc.get("etch"),
        "grating_etch_depth_um": gc.get("etch_depth_um"),
        "fiber_angle_deg": gc.get("fiber_angle_deg"),
        "fiber_tilt": gc.get("fiber_tilt", "away" if gc.get("period_um") else None),
        "fiber_position_um": ap["source_x_um"] if use_ap else gc.get("source_x_um"),
        "taper_length_um": (tp["taper_length_um"] if use_tp and tp.get("taper_length_um")
                            else cfg.grating.taper_length_um),
        "grating_focusing": use_fc,               # the main coupler (budget, layout figure)
        "focus_available": have_fc,               # *_fgc cells are drawn as well
        "focus_teeth_kind": fc.get("teeth_kind") if have_fc else None,
        "focus_distance_um": fc.get("distance_um") if have_fc else None,
        "focus_tip_width_um": fc.get("tip_width_um") if have_fc else None,
        "focus_tip_taper_um": fc.get("tip_taper_um") if have_fc else None,
        "focus_half_angle_deg": _r(math.degrees(fc["half_angle_rad"]), 3) if have_fc else None,
        "focus_y_max_um": fc.get("y_max_um") if have_fc else None,
        "focus_s": _r(fc.get("s"), 6) if have_fc else None,
        "focus_teeth_list": fc.get("teeth") if have_fc else None,
        "tone": lay.tone,
        "trench_width_um": lay.trench_width_um,
        "layers": {"waveguide": list(lay.waveguide_layer), "ring": list(lay.ring_layer),
                   "taper": list(lay.taper_layer), "grating": list(lay.grating_layer),
                   "chip": list(cfg.edge.chip_layer), "lane": list(cfg.edge.lane_layer)},
    }


def edge_coupler_length_um(cfg, ed: dict) -> float:
    """Length on the chip of one edge coupler: taper plus the straight at the facet width."""
    return float(ed.get("taper_length_um") or 0.0) + float(cfg.edge.facet_straight_um)


def edge_bus_length_um(cfg, ed: dict) -> float:
    """The 1 um waveguide between the two edge-coupler tapers: the chip length less both
    couplers (the waveguides run from facet to facet)."""
    return float(cfg.edge.chip_length_um) - 2 * edge_coupler_length_um(cfg, ed)


def bus_length_um(ctx) -> float:
    """Waveguide between the couplers' tapers on the device: layout.bus_length_um for
    grating couplers, the chip length less both edge couplers for edge coupling."""
    from tarang.design.stages import coupling
    ed = (ctx.state["stages"].get("edge") or {}).get("results") or {}
    if coupling(ctx.cfg) == "edge" and ed.get("facet_width_um"):
        return edge_bus_length_um(ctx.cfg, ed)
    return float(ctx.cfg.layout.bus_length_um)


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


def export(ctx) -> dict:
    ui = ctx.ui
    cfg = ctx.cfg
    from tarang.design.ui import parse_list
    from tarang.design.stages import coupling
    grating = lambda c: coupling(c) == "grating"              # noqa: E731
    ui.form(cfg, "Layout", [
        dict(section="layout", key="bus_length_um", show=grating),
        dict(section="layout", key="tone", check=lambda v: "" if v in ("structure", "trench")
             else "answer structure or trench"),
        dict(section="layout", key="trench_width_um", show=lambda c: c.layout.tone == "trench"),
        *[dict(section="layout", key=k, cast=parse_list) for k in
          ("waveguide_layer", "ring_layer", "taper_layer", "grating_layer")],
        dict(section="grating", key="taper_length_um", show=grating),
        dict(section="layout", key="coupler", show=grating,
             check=lambda v: "" if str(v).lower() in COUPLER_SETTINGS
             else "answer " + ", ".join(COUPLER_SETTINGS)),
        dict(section="layout", key="reference",
             show=lambda c: str(getattr(c.run, "goal", "comb_ring")) == "comb_ring"),
        dict(section="layout", key="gap_offsets_um", cast=parse_list,
             show=lambda c: str(getattr(c.run, "goal", "comb_ring")) == "comb_ring")])
    for key in ("bus_length_um", "tone", "trench_width_um", "waveguide_layer", "ring_layer",
                "taper_layer", "grating_layer"):
        if key == "trench_width_um" and cfg.layout.tone != "trench":
            continue
        if key == "bus_length_um" and not grating(cfg):
            continue
        ui.value(cfg, "layout", key)
    if grating(cfg):
        ui.value(cfg, "grating", "taper_length_um")
    p = device_parameters(ctx)
    for note in coupler_plan(ctx)["notes"]:
        ui.say("  Note: " + note)
    ui.say("  Couplers on the mask: " + (", ".join(f"{k} (*_{s})" for s, k in p["couplers"]) or "none")
           + (f"; main coupler: {p['main_coupler']}" if p.get("main_coupler") else "")
           + f" (layout.coupler = {p['coupler_setting']!r}).")
    if p.get("gap_variants_um") and len(p["gap_variants_um"]) > 2:
        ui.say("  Ring variants at gaps " + ", ".join(f"{g * 1e3:.0f}" for g in p["gap_variants_um"])
               + " nm (layout.gap_offsets_um).")
    if p.get("focus_available"):
        ui.say(f"  Focusing grating ({p['focus_teeth_kind']} teeth): first tooth "
               f"{p['focus_distance_um']:g} µm from a {p['focus_tip_width_um']:g} µm tip "
               f"({p['focus_tip_taper_um']:g} µm tip taper), drawn in the *_fgc cells; "
               + ("it is the main coupler (budget, layout figure)." if p["grating_focusing"] else
                  "the straight grating stays the main coupler (focus.use_in_layout = false)."))
    if p.get("edge_facet_width_um"):
        ui.say(f"  Edge couplers (*_ec cells): {p['edge_taper_profile']} taper "
               f"{p['edge_taper_length_um']:g} µm from the {p['waveguide_width_um']:g} µm waveguide to the "
               f"{p['edge_facet_width_um']:g} µm facet, then {p['edge_facet_straight_um']:g} µm straight to "
               f"the chip edge and {p['edge_facet_overhang_um']:g} µm on into the dicing lane; the "
               f"waveguides run across the {p['chip_length_um']:g} µm chip "
               f"({p['bus_length_um']:g} µm between the tapers).")
    if p.get("grating_period_um"):
        src = ("taper stage" if p["taper_length_um"] != cfg.grating.taper_length_um
               else "grating.taper_length_um")
        ui.say(f"  Grating: straight, {'apodised' if p['grating_apodised'] else 'uniform'}, "
               f"{p['grating_width_um']:g} µm wide; taper {p['taper_length_um']:g} µm (from the "
               f"{src}" + ("; the table's value is the fallback)" if src == "taper stage" else ")"))
    out = ctx.out / "layout"
    out.mkdir(parents=True, exist_ok=True)
    files = _write_parameters(p, out)
    lle = _lle(ctx)
    import importlib.util
    if importlib.util.find_spec("gdstk") is None:
        ui.say("gdstk is not installed, so no GDS was written (parameters were). Install it "
               'with: pip install "tarang-photonics[layout]" and rerun this stage.')
        return dict(parameters=p, files=files, gds=None, lle=lle)
    try:
        files += build_gds(p, out)
    except ValueError as err:                       # a spiral that does not fit the chip
        ui.say(f"  Layout not written: {err}")
        return dict(parameters=p, files=[str(f) for f in files], gds=None, lle=lle, error=str(err))
    chip = out / "chip.json"
    if p.get("edge_facet_width_um") and chip.exists():
        info = json.loads(chip.read_text(encoding="utf-8"))
        ui.say(f"  Chip {info['chip_length_um']:g} × {info['chip_width_um']:g} µm (outline on layer "
               f"{p['layers']['chip'][0]}/{p['layers']['chip'][1]}, dicing lanes of "
               f"{info['dicing_lane_um']:g} µm on {p['layers']['lane'][0]}/{p['layers']['lane'][1]}): "
               f"{len(info['devices'])} devices, facets at least {info['pitch_um']:g} µm apart, "
               f"{info['height_needed_um']:.0f} µm of the width used"
               + ("." if info["fits"] else " — MORE than the chip is wide: use a wider chip "
                                            "(edge.chip_width_um) or fewer variants."))
    ui.say("Layout files:")
    for f in files:
        ui.say(f"  {f}")
    fig = None
    from tarang.design import visuals
    suf = next((a for a, k in p.get("couplers") or [] if k == p.get("main_coupler")),
               "ec" if p.get("edge_facet_width_um") else "gc")
    if p.get("goal") == "loss_test":
        sp = next((f for f in files if Path(f).name == f"spiral_cutback_{suf}.gds"), None)
        if sp:
            fig = visuals.safe(ctx, "cutback layout", visuals.cutback_views, sp,
                               out / f"waveguide_{suf}.gds", p, ctx.path("figures", "6_layout.png"))
    else:
        main = (f"ring_waveguide_{suf}.gds", f"waveguide_{suf}.gds")
        gc_file = next((f for f in files if Path(f).name in main), None)
        if gc_file:
            fig = visuals.safe(ctx, "layout", visuals.layout_views, gc_file, p,
                               ctx.path("figures", "6_layout.png"))
    if p.get("edge_facet_width_um"):
        h = p.get("edge_fiber_height_um")
        ui.say(f"  Fibres: lensed, MFD {p['edge_fiber_mfd_um']:g} µm, in the chip plane facing the facets"
               + (f", spot centre {h:.2f} µm above the BOX top" if h is not None else "") + ".")
    if p.get("grating_period_um") and p.get("fiber_tilt"):
        ui.say(f"  Fibres: {p['fiber_angle_deg']}° from vertical, beam travelling "
               f"{p['fiber_tilt']} the waveguide at both couplers"
               + (" (upper ends of the two fibres lean outward: a V)." if p["fiber_tilt"] == "towards"
                  else " (upper ends of the two fibres lean inward, over the chip)."))
    if lle:
        ui.say("Comb-simulation (LLE) inputs: " + ", ".join(Path(f).name for f in lle["files"])
               + f" in {ctx.out / 'lle'}")
    return dict(parameters=p, files=[str(f) for f in files], figure=str(fig) if fig else None,
                lle=lle)


def _lle(ctx):
    """LLE inputs (lle/); a failure here never stops the layout. Rings only."""
    from tarang.design import lle
    if str(getattr(ctx.cfg.run, "goal", "comb_ring")) != "comb_ring":
        return None
    try:
        return lle.export(ctx)
    except Exception as err:                                      # noqa: BLE001
        ctx.ui.say(f"  (LLE inputs not written: {type(err).__name__}: {err})")
        return None


def _write_parameters(p: dict, out: Path) -> list[Path]:
    flat = {k: v for k, v in p.items() if k not in ("layers", "grating_teeth_list",
                                                     "focus_teeth_list", "couplers")}
    flat["couplers"] = " ".join(f"{a}:{b}" for a, b in p.get("couplers") or [])
    teeth = p.get("grating_teeth_list")
    flat.update({f"layer_{k}": v[0] for k, v in p["layers"].items()})
    flat.update({f"datatype_{k}": v[1] for k, v in p["layers"].items()})
    js = out / "parameters.json"
    js.write_text(json.dumps(p, indent=2), encoding="utf-8")
    m = out / "parameters.m"
    lines = ["% TARANG design parameters (lengths in um). Load with: run('parameters.m')",
             "p = struct();"]
    for k, v in flat.items():
        if v is None:
            lines.append(f"p.{k} = [];")
        elif isinstance(v, (list, tuple)):
            lines.append(f"p.{k} = [" + " ".join(str(x) for x in v) + "];")
        elif isinstance(v, str):
            lines.append(f"p.{k} = '{v}';")
        else:
            lines.append(f"p.{k} = {v};")
    if teeth:      # apodised grating: one row per tooth, [x_start, length] in um
        lines.append("p.grating_teeth_list = [" + "; ".join(f"{a} {b}" for a, b in teeth) + "];")
        flat["grating_teeth_list"] = "grating_teeth.csv"
    m.write_text("\n".join(lines) + "\n", encoding="utf-8")
    c = out / "parameters.csv"
    with open(c, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["parameter", "value"])
        w.writerows(flat.items())
    files = [js, m, c]
    if teeth:
        t = out / "grating_teeth.csv"
        with open(t, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["tooth", "x_start_um", "length_um"])
            w.writerows([i, a, b] for i, (a, b) in enumerate(teeth))
        files.append(t)
    return files


# ------------------------------------------------------------------------------- GDS
def build_gds(p: dict, out: Path) -> list[Path]:
    import gdstk
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    W, R, L = p["waveguide_width_um"], p["ring_radius_um"], p["bus_length_um"]
    lay = {k: {"layer": v[0], "datatype": v[1]} for k, v in p["layers"].items()}
    tol = 1e-3

    def bus(cell):
        cell.add(gdstk.rectangle((-L / 2, -W / 2), (L / 2, W / 2), **lay["waveguide"]))

    def ring(cell, gap):
        yc = R + W + gap
        cell.add(gdstk.ellipse((0, yc), R + W / 2, inner_radius=R - W / 2, tolerance=tol,
                               **lay["ring"]))

    def focusing(cell, side):
        """Tip taper, focusing slab and curved teeth (focus at the tip, axis along +x)."""
        from tarang.design.focus import polygons
        x0, Lt, wt = L / 2, p["focus_tip_taper_um"], p["focus_tip_width_um"]
        shapes = []
        if Lt > 0:
            shapes.append(gdstk.Polygon([(x0, -W / 2), (x0 + Lt, -wt / 2), (x0 + Lt, wt / 2),
                                         (x0, W / 2)], **lay["taper"]))
        F = x0 + Lt
        g = polygons(p["focus_teeth_list"], p["focus_distance_um"], wt, p["wavelength_um"],
                     p["focus_s"], math.radians(p["focus_half_angle_deg"]), p["focus_y_max_um"],
                     n_pts=96)
        shapes.append(gdstk.Polygon([(F + x, y) for x, y in g["slab"]], **lay["taper"]))
        shapes += [gdstk.Polygon([(F + x, y) for x, y in t], **lay["grating"]) for t in g["teeth"]]
        for poly in shapes:
            if side < 0:
                poly.mirror((0, 0), (0, 1))
            cell.add(poly)

    def edge(cell):
        """Edge coupler at the right end of the bus: taper (its shape from the edge stage)
        from the waveguide to the facet width, the straight at the facet width up to the
        chip edge, and on into the dicing lane."""
        from tarang.design.taper import taper_outline
        x0, Lt = L / 2, float(p["edge_taper_length_um"])
        Wf = float(p["edge_facet_width_um"])
        pts = taper_outline(Wf, W, Lt, p.get("edge_taper_profile") or "linear",
                            float(p.get("edge_power_exponent") or 0.6))
        cell.add(gdstk.Polygon([(x0 + Lt - x, y) for x, y in pts], **lay["taper"]))
        x1 = x0 + Lt + float(p["edge_facet_straight_um"]) + float(p["edge_facet_overhang_um"])
        cell.add(gdstk.rectangle((x0 + Lt, -Wf / 2), (x1, Wf / 2), **lay["taper"]))

    def coupler(cell, side, kind="uniform", end=None):
        """Taper and grating at the right (side = +1) or left (-1) end of the bus (or at the
        waveguide end ``end`` = (x, y), the waveguide arriving along +x for side +1 and -x
        for side -1). kind: 'uniform', 'apodised', 'focusing' or 'edge'."""
        if kind == "edge":
            if not p.get("edge_facet_width_um"):
                return
        elif not p.get("grating_period_um"):
            return
        sub = gdstk.Cell("_tmp")
        if kind == "edge":
            edge(sub)
        elif kind == "focusing":
            focusing(sub, +1)
        else:
            straight(sub, kind)
        for poly in sub.polygons:
            if side < 0:
                poly.mirror((0, 0), (0, 1))
            if end is not None:
                poly.translate(end[0] - side * L / 2, end[1])
            cell.add(poly)

    def straight(cell, kind):
        Lt, Wg = p["taper_length_um"], p["grating_width_um"]
        P, ff = p["grating_period_um"], p["grating_fill_factor"]
        n = int(p.get("uniform_teeth") or p["grating_teeth"])
        x0 = L / 2
        taper = gdstk.Polygon([(x0, -W / 2), (x0 + Lt, -Wg / 2), (x0 + Lt, Wg / 2), (x0, W / 2)],
                              **lay["taper"])
        g0 = x0 + Lt
        tl = p.get("grating_teeth_list") if kind == "apodised" else None
        if tl:                             # apodised: explicit teeth from x = 0 at the grating start
            ends = [a + b for a, b in tl]
            L_g = tl[-1][0] + (tl[-1][0] - tl[-2][0] if len(tl) > 1 else tl[-1][1])
            if p.get("grating_etch") == "full":
                shapes = [gdstk.rectangle((g0 + a, -Wg / 2), (g0 + a + b, Wg / 2), **lay["grating"])
                          for a, b in tl]
            else:
                shapes = [gdstk.rectangle((g0, -Wg / 2), (g0 + L_g, Wg / 2), **lay["taper"])]
                starts = [a for a, _ in tl[1:]] + [L_g]
                shapes += [gdstk.rectangle((g0 + e, -Wg / 2), (g0 + s_, Wg / 2), **lay["grating"])
                           for e, s_ in zip(ends, starts) if s_ > e]
        elif p.get("grating_etch") == "full":
            # full etch: the teeth are the remaining core; grooves are etched through
            shapes = [gdstk.rectangle((g0 + i * P, -Wg / 2), (g0 + i * P + ff * P, Wg / 2),
                                      **lay["grating"]) for i in range(n)]
        else:
            # partial etch: the slab under the whole grating stays (taper layer, full
            # etch around it); the grooves are a second, shallow etch (grating layer)
            shapes = [gdstk.rectangle((g0, -Wg / 2), (g0 + n * P, Wg / 2), **lay["taper"])]
            shapes += [gdstk.rectangle((g0 + i * P + ff * P, -Wg / 2), (g0 + (i + 1) * P, Wg / 2),
                                       **lay["grating"]) for i in range(n)]
        for poly in [taper, *shapes]:
            cell.add(poly)

    def finish(cell):
        if p["tone"] != "trench":
            return cell
        t = p["trench_width_um"]
        out_cell = gdstk.Cell(cell.name)
        allpolys = cell.get_polygons()
        grown = gdstk.offset(allpolys, t, join="miter", tolerance=tol)
        trench = gdstk.boolean(grown, allpolys, "not", precision=tol)
        layer = lay["waveguide"]
        for poly in trench:
            poly.layer, poly.datatype = layer["layer"], layer["datatype"]
            out_cell.add(poly)
        return out_cell

    lib_all = gdstk.Library(unit=1e-6, precision=1e-9)
    files = []
    kinds = [tuple(k) for k in (p.get("couplers") or [])]
    gaps = [g for g in (p.get("gap_variants_um") or [])] or \
        [g for g in (p.get("gap_critical_um"), p.get("gap_over_coupled_um")) if g]
    variants: dict = {"waveguide": [dict(label="waveguide")]}
    if gaps and R:                                  # the ring needs the gap stage's result
        variants["ring_waveguide"] = [dict(label=f"ring_waveguide_gap{g * 1e3:.0f}nm", gap=g)
                                      for g in gaps]
        for suf, kind in kinds:
            variants[f"ring_waveguide_{suf}"] = [
                dict(label=f"ring_waveguide_{suf}_gap{g * 1e3:.0f}nm", gap=g, kind=kind) for g in gaps]
    if kinds and (p.get("reference_waveguide") or not (gaps and R)):
        for suf, kind in kinds:
            variants[f"waveguide_{suf}"] = [dict(label=f"waveguide_{suf}", kind=kind)]
    lengths = [x for x in (p.get("cutback_lengths_cm") or []) if x > 0]
    if lengths and kinds and p.get("spiral_bend_radius_um"):
        for suf, kind in kinds:
            variants[f"spiral_cutback_{suf}"] = [
                dict(label=f"spiral_{x:g}cm_{suf}".replace(".", "p"), kind=kind, spiral_cm=x)
                for x in lengths]
    top = gdstk.Cell(f"{p['name']}_TOP")
    y_off = 0.0
    spiral_info = []
    is_edge = any(k == "edge" for _, k in kinds)
    placed = []                                       # (cell, its y offset) for the edge chip
    for fname, items in variants.items():
        lib = gdstk.Library(unit=1e-6, precision=1e-9)
        for it in items:
            cell = gdstk.Cell(it["label"])
            kind = it.get("kind")
            if it.get("spiral_cm"):
                r0 = 2 * float(p["spiral_bend_radius_um"])
                if kind == "edge":                  # total facet-to-facet length = the target
                    pts, info = edge_spiral(it["spiral_cm"] * 1e4, r0, float(p["spiral_pitch_um"]),
                                            L / 2, float(p["chip_length_um"]))
                else:
                    pts, info = spiral_path(it["spiral_cm"] * 1e4, r0, float(p["spiral_pitch_um"]))
                cell.add(gdstk.FlexPath(pts, W, tolerance=tol, **lay["waveguide"]))
                info.update(label=it["label"], kind=kind)
                spiral_info.append(info)
                coupler(cell, +1, kind, end=tuple(pts[-1]))
                coupler(cell, -1, kind, end=tuple(pts[0]))
            else:
                bus(cell)
                if it.get("gap"):
                    ring(cell, it["gap"])
                if kind:
                    coupler(cell, +1, kind)
                    coupler(cell, -1, kind)
            cell = finish(cell)
            lib.add(cell)
            if not (is_edge and kind != "edge"):     # the chip holds only cells with facets
                lib_all.add(cell)
            (x0b, y0b), (x1b, y1b) = cell.bounding_box() or ((0, 0), (0, 0))
            if is_edge:
                if kind == "edge":            # only cells with facets go on the chip
                    placed.append((cell, y0b, y1b))
                continue
            top.add(gdstk.Reference(cell, (0, y_off - y1b)))
            y_off -= (y1b - y0b) + 50.0
        path = out / f"{fname}.gds"
        lib.write_gds(path)
        files.append(path)
    if is_edge and placed:
        chip = edge_chip(top, placed, p, lay)
        with open(out / "chip.json", "w", encoding="utf-8") as f:
            json.dump(chip, f, indent=1)
        files.append(out / "chip.json")
    lib_all.add(top)
    path = out / "design.gds"
    lib_all.write_gds(path)
    files.append(path)
    if spiral_info:
        with open(out / "spirals.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["label", "kind", "length_um", "turns", "r0", "r_out",
                                              "min_bend_radius_um"], lineterminator="\n",
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(spiral_info)
        files.append(out / "spirals.csv")
    return files


def _arc(cx, cy, r, a0, a1, n=None):
    n = n or max(16, int(abs(a1 - a0) * r / 0.5))
    a = np.linspace(a0, a1, n)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)


def _arm(r0, a, theta, n):
    t = np.linspace(0.0, theta, n)
    r = r0 + a * t
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def _length(pts) -> float:
    return float(np.sum(np.hypot(*np.diff(pts, axis=0).T)))


def _spiral(r0, pitch, turns, lead):
    """Points of the double spiral: arm B inwards, centre S-bend (two half circles of radius
    r0 / 2), arm A outwards; each arm then turns (radius r0 / 2) onto a straight lead along
    x: the waveguide enters at the left lead (heading +x) and leaves at the right one."""
    a = 2 * pitch / (2 * math.pi)                 # each arm grows by 2 pitches per turn
    theta = 2 * math.pi * turns
    n = max(64, int(theta * (r0 + a * theta) / 1.0))
    A = _arm(r0, a, theta, n)                     # outwards, counter-clockwise
    rb = r0 / 2
    s_bend = np.concatenate([_arc(-r0 / 2, 0, rb, math.pi, 0.0),        # over the top
                             _arc(r0 / 2, 0, rb, math.pi, 2 * math.pi)])  # under the bottom
    rA = r0 + a * theta
    exit_a = _arc(rA + rb, 0.0, rb, math.pi, math.pi / 2)    # heading +y -> +x
    xe = rA + rb
    lead_a = np.array([[xe, rb], [xe + lead, rb]])
    B = -A[::-1]                                  # arm B, run inwards
    exit_b = -exit_a[::-1]
    lead_b = -lead_a[::-1]
    pts = np.concatenate([lead_b, exit_b, B, s_bend, A, exit_a, lead_a])
    keep = np.concatenate([[True], np.hypot(*np.diff(pts, axis=0).T) > 1e-6])
    return pts[keep], dict(r0=r0, r_out=rA, turns=turns)


def spiral_path(length_um: float, r0_min: float, pitch: float, lead: float = 20.0):
    """A double spiral of the given total length (from the start of one lead to the end of
    the other), smallest bend radius r0_min / 2: the whole number of turns that fits,
    then the centre radius r0 raised until the length is reached. Returns (points, info)."""
    def total(r0, turns):
        return _length(_spiral(r0, pitch, turns, lead)[0])
    turns = 0
    while total(r0_min, turns + 1) <= length_um:
        turns += 1
    lo, hi = r0_min, r0_min
    if total(lo, turns) > length_um:              # shorter than the smallest spiral: as is
        pts, info = _spiral(lo, pitch, turns, lead)
    else:
        while total(hi, turns) < length_um:
            hi *= 1.5
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if total(mid, turns) < length_um:
                lo = mid
            else:
                hi = mid
        pts, info = _spiral(hi, pitch, turns, lead)
    info.update(length_um=round(_length(pts), 2), min_bend_radius_um=round(info["r0"] / 2, 3),
                r0=round(info["r0"], 3), r_out=round(info["r_out"], 3))
    return pts, info


def edge_spiral(total_um: float, r0_min: float, pitch: float, x_taper: float, chip_um: float):
    """Spiral whose waveguide, run out horizontally to the edge-coupler tapers at +-x_taper,
    has the given facet-to-facet length (couplers included, chip_um long in total): the
    spiral's own length s satisfies s + 2 (x_taper - x_spiral_end) + (chip_um - 2 x_taper)
    = total, solved by a few fixed-point steps (x_spiral_end grows slowly with s)."""
    xe = 0.0
    pts = info = None
    for _ in range(6):
        s = total_um - chip_um + 2 * xe
        if s <= 0:
            raise ValueError(f"{total_um / 1e4:g} cm is shorter than the chip ({chip_um / 1e4:g} cm) "
                             "plus a spiral: use longer cutback lengths or a shorter chip")
        pts, info = spiral_path(s, r0_min, pitch)
        xe_new = float(pts[-1][0])
        if abs(xe_new - xe) < 1e-3:
            break
        xe = xe_new
    if float(pts[-1][0]) > x_taper:
        raise ValueError(f"the {total_um / 1e4:g} cm spiral ({2 * pts[-1][0]:.0f} µm across) does not "
                         f"fit between the edge couplers ({2 * x_taper:.0f} µm apart): use a longer chip")
    start = np.array([[-x_taper, pts[0][1]]])
    end = np.array([[x_taper, pts[-1][1]]])
    full = np.concatenate([start, pts, end])
    info = dict(info, length_um=round(_length(full) + chip_um - 2 * x_taper, 2),
                spiral_length_um=round(_length(pts), 2))
    return full, info


def edge_chip(top, placed, p: dict, lay: dict) -> dict:
    """Place the edge-coupled devices on the chip, one above the other with at least
    edge_pitch_um between neighbouring facets, centred on the chip; draw the chip outline
    and the dicing lanes (on their own layers, not etched). Returns the chip summary."""
    import gdstk
    pitch = float(p.get("edge_pitch_um") or 127.0)
    Lc, Wc = float(p["chip_length_um"]), float(p["chip_width_um"])
    lane = float(p.get("dicing_lane_um") or 0.0)
    ys, y = [], 0.0
    prev_bottom = None
    for cell, y0b, y1b in placed:
        if prev_bottom is None:
            y = -y1b
        else:
            # the bus (y = 0 in the cell) at least one pitch below the last one, and no overlap
            y = min(ys[-1] - pitch, prev_bottom - 30.0 - y1b)
        ys.append(y)
        prev_bottom = y + y0b
    top_y = ys[0] + placed[0][2]
    shift = -(top_y + prev_bottom) / 2                # centre the stack on the chip
    for (cell, _, _), yy in zip(placed, ys):
        top.add(gdstk.Reference(cell, (0, yy + shift)))
    needed = top_y - prev_bottom
    top.add(gdstk.rectangle((-Lc / 2, -Wc / 2), (Lc / 2, Wc / 2), **lay["chip"]))
    if lane > 0:
        for sgn in (-1, 1):
            x_in, x_out = sgn * Lc / 2, sgn * (Lc / 2 + lane)
            top.add(gdstk.rectangle((min(x_in, x_out), -Wc / 2), (max(x_in, x_out), Wc / 2),
                                    **lay["lane"]))
    return dict(chip_length_um=Lc, chip_width_um=Wc, dicing_lane_um=lane, pitch_um=pitch,
                devices=[c.name for c, _, _ in placed],
                facet_y_um=[round(yy + shift, 3) for yy in ys], height_needed_um=round(needed, 1),
                fits=bool(needed <= Wc))
