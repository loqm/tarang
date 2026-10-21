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


def device_parameters(ctx) -> dict:
    """All layout dimensions, from the earlier stages and the layout settings."""
    cfg = ctx.cfg
    xs, ring = ctx.results("cross_section"), ctx.results("ring")
    gap = ctx.state["stages"].get("gap", {}).get("results", {})
    gc = ctx.state["stages"].get("grating", {}).get("results", {})
    tp = ctx.state["stages"].get("taper", {}).get("results", {}) or {}
    use_tp = bool(tp) and not tp.get("skipped") and bool(cfg.taper.use_in_layout)
    ap = ctx.state["stages"].get("apodise", {}).get("results", {}) or {}
    use_ap = (bool(ap.get("verified")) and bool(ap.get("better_than_uniform"))
              and bool(cfg.apodise.use_in_layout))
    fc = ctx.state["stages"].get("focus", {}).get("results", {}) or {}
    have_fc = bool(fc.get("curves")) and bool(cfg.focus.run)
    use_fc = have_fc and bool(cfg.focus.use_in_layout)
    lay = cfg.layout
    return {
        "name": cfg.run.name,
        "wavelength_um": cfg.run.wavelength_um,
        "core_material": cfg.materials.core,
        "core_height_um": xs["height_um"],
        "waveguide_width_um": xs["width_um"],
        "ring_radius_um": ring["radius_um"],
        "ring_width_um": xs["width_um"],
        "gap_critical_um": _r(gap.get("critical_gap_um")),
        "gap_over_coupled_um": _r(gap.get("over_coupled_gap_um")),
        "over_coupling_ratio": gap.get("over_coupling"),
        "bus_length_um": lay.bus_length_um,
        "grating_period_um": gc.get("period_um"),
        "grating_fill_factor": gc.get("fill_factor"),
        "grating_tooth_length_um": _r(gc["period_um"] * gc["fill_factor"]) if gc.get("period_um") else None,
        "grating_teeth": len(ap["teeth"]) if use_ap else gc.get("n_teeth"),
        "grating_apodised": use_ap,
        "grating_teeth_list": [list(t) for t in ap["teeth"]] if use_ap else None,
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
                   "taper": list(lay.taper_layer), "grating": list(lay.grating_layer)},
    }


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


def export(ctx) -> dict:
    ui = ctx.ui
    cfg = ctx.cfg
    from tarang.design.ui import parse_list
    ui.form(cfg, "Layout", [
        dict(section="layout", key="bus_length_um"),
        dict(section="layout", key="tone", check=lambda v: "" if v in ("structure", "trench")
             else "answer structure or trench"),
        dict(section="layout", key="trench_width_um", show=lambda c: c.layout.tone == "trench"),
        *[dict(section="layout", key=k, cast=parse_list) for k in
          ("waveguide_layer", "ring_layer", "taper_layer", "grating_layer")],
        dict(section="grating", key="taper_length_um")])
    for key in ("bus_length_um", "tone", "trench_width_um", "waveguide_layer", "ring_layer",
                "taper_layer", "grating_layer"):
        if key == "trench_width_um" and cfg.layout.tone != "trench":
            continue
        ui.value(cfg, "layout", key)
    ui.value(cfg, "grating", "taper_length_um")
    p = device_parameters(ctx)
    if p.get("focus_available"):
        ui.say(f"  Focusing grating ({p['focus_teeth_kind']} teeth): first tooth "
               f"{p['focus_distance_um']:g} µm from a {p['focus_tip_width_um']:g} µm tip "
               f"({p['focus_tip_taper_um']:g} µm tip taper), drawn in the *_fgc cells; "
               + ("it is the main coupler (budget, layout figure)." if p["grating_focusing"] else
                  "the straight grating stays the main coupler (focus.use_in_layout = false)."))
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
    files += build_gds(p, out)
    ui.say("Layout files:")
    for f in files:
        ui.say(f"  {f}")
    fig = None
    main = ("ring_waveguide_fgc.gds", "waveguide_fgc.gds") if p.get("grating_focusing") else \
        ("ring_waveguide_gc.gds", "waveguide_gc.gds")
    gc_file = next((f for f in files if Path(f).name in main), None)
    if gc_file:
        from tarang.design import visuals
        fig = visuals.safe(ctx, "layout", visuals.layout_views, gc_file, p,
                           ctx.path("figures", "6_layout.png"))
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
    """LLE inputs (lle/); a failure here never stops the layout."""
    from tarang.design import lle
    try:
        return lle.export(ctx)
    except Exception as err:                                      # noqa: BLE001
        ctx.ui.say(f"  (LLE inputs not written: {type(err).__name__}: {err})")
        return None


def _write_parameters(p: dict, out: Path) -> list[Path]:
    flat = {k: v for k, v in p.items() if k not in ("layers", "grating_teeth_list",
                                                     "focus_teeth_list")}
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

    def coupler(cell, side, focus=False):
        """Taper and grating at the right (side = +1) or left (-1) end of the bus; the
        focusing grating instead when ``focus``."""
        if not p.get("grating_period_um"):
            return
        if focus:
            return focusing(cell, side)
        Lt, Wg = p["taper_length_um"], p["grating_width_um"]
        P, ff, n = p["grating_period_um"], p["grating_fill_factor"], int(p["grating_teeth"])
        x0 = L / 2
        taper = gdstk.Polygon([(x0, -W / 2), (x0 + Lt, -Wg / 2), (x0 + Lt, Wg / 2), (x0, W / 2)],
                              **lay["taper"])
        g0 = x0 + Lt
        tl = p.get("grating_teeth_list")
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
            if side < 0:
                poly.mirror((0, 0), (0, 1))
            cell.add(poly)

    def finish(cell):
        if p["tone"] != "trench":
            return cell
        t = p["trench_width_um"]
        out_cell = gdstk.Cell(cell.name)
        allpolys = cell.polygons
        grown = gdstk.offset(allpolys, t, join="miter", tolerance=tol)
        trench = gdstk.boolean(grown, allpolys, "not", precision=tol)
        layer = lay["waveguide"]
        for poly in trench:
            poly.layer, poly.datatype = layer["layer"], layer["datatype"]
            out_cell.add(poly)
        return out_cell

    lib_all = gdstk.Library(unit=1e-6, precision=1e-9)
    files = []
    gaps = [g for g in (p["gap_critical_um"], p["gap_over_coupled_um"]) if g]
    variants = {"waveguide": [(None, False, False)]}
    fgc = bool(p.get("focus_available") and p.get("focus_teeth_list"))
    if gaps:                                   # the ring needs the gap stage's result
        variants["ring_waveguide"] = [(g, False, False) for g in gaps]
        variants["ring_waveguide_gc"] = [(g, True, False) for g in gaps]
        if fgc:                                # the same with focusing gratings
            variants["ring_waveguide_fgc"] = [(g, True, True) for g in gaps]
    elif p.get("grating_period_um"):
        variants["waveguide_gc"] = [(None, True, False)]
        if fgc:
            variants["waveguide_fgc"] = [(None, True, True)]
    top = gdstk.Cell(f"{p['name']}_TOP")
    y_off = 0.0
    for fname, items in variants.items():
        lib = gdstk.Library(unit=1e-6, precision=1e-9)
        for gap, with_gc, focus in items:
            label = fname + (f"_gap{gap * 1e3:.0f}nm" if gap else "")
            cell = gdstk.Cell(label)
            bus(cell)
            if gap and fname != "waveguide":
                ring(cell, gap)
            if with_gc:
                coupler(cell, +1, focus)
                coupler(cell, -1, focus)
            cell = finish(cell)
            lib.add(cell)
            lib_all.add(cell)
            top.add(gdstk.Reference(cell, (0, y_off)))
            y_off -= (2 * R + 4 * W + 50.0) if gap else 50.0
        path = out / f"{fname}.gds"
        lib.write_gds(path)
        files.append(path)
    lib_all.add(top)
    path = out / "design.gds"
    lib_all.write_gds(path)
    files.append(path)
    return files
