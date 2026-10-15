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
grating. gdstk is an optional dependency: pip install "tarang-photonics[layout]".
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


def device_parameters(ctx) -> dict:
    """All layout dimensions, from the earlier stages and the layout settings."""
    cfg = ctx.cfg
    xs, ring = ctx.results("cross_section"), ctx.results("ring")
    gap = ctx.state["stages"].get("gap", {}).get("results", {})
    gc = ctx.state["stages"].get("grating", {}).get("results", {})
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
        "grating_teeth": gc.get("n_teeth"),
        "grating_width_um": gc.get("width_um"),
        "grating_etch": gc.get("etch"),
        "grating_etch_depth_um": gc.get("etch_depth_um"),
        "fiber_angle_deg": gc.get("fiber_angle_deg"),
        "fiber_position_um": gc.get("source_x_um"),
        "taper_length_um": cfg.grating.taper_length_um,
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
    for key in ("bus_length_um", "tone", "trench_width_um", "waveguide_layer", "ring_layer",
                "taper_layer", "grating_layer"):
        if key == "trench_width_um" and cfg.layout.tone != "trench":
            continue
        ui.value(cfg, "layout", key)
    ui.value(cfg, "grating", "taper_length_um")
    p = device_parameters(ctx)
    out = ctx.out / "layout"
    out.mkdir(parents=True, exist_ok=True)
    files = _write_parameters(p, out)
    import importlib.util
    if importlib.util.find_spec("gdstk") is None:
        ui.say("gdstk is not installed, so no GDS was written (parameters were). Install it "
               'with: pip install "tarang-photonics[layout]" and rerun this stage.')
        return dict(parameters=p, files=files, gds=None)
    files += build_gds(p, out)
    ui.say("Layout files:")
    for f in files:
        ui.say(f"  {f}")
    return dict(parameters=p, files=[str(f) for f in files])


def _write_parameters(p: dict, out: Path) -> list[Path]:
    flat = {k: v for k, v in p.items() if k != "layers"}
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
    m.write_text("\n".join(lines) + "\n", encoding="utf-8")
    c = out / "parameters.csv"
    with open(c, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["parameter", "value"])
        w.writerows(flat.items())
    return [js, m, c]


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

    def coupler(cell, side):
        """Taper and grating at the right (side = +1) or left (-1) end of the bus."""
        if not p.get("grating_period_um"):
            return
        Lt, Wg = p["taper_length_um"], p["grating_width_um"]
        P, ff, n = p["grating_period_um"], p["grating_fill_factor"], int(p["grating_teeth"])
        x0 = L / 2
        taper = gdstk.Polygon([(x0, -W / 2), (x0 + Lt, -Wg / 2), (x0 + Lt, Wg / 2), (x0, W / 2)],
                              **lay["taper"])
        g0 = x0 + Lt
        if p.get("grating_etch") == "full":
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
    variants = {"waveguide": [(None, False)]}
    if gaps:                                   # the ring needs the gap stage's result
        variants["ring_waveguide"] = [(g, False) for g in gaps]
        variants["ring_waveguide_gc"] = [(g, True) for g in gaps]
    elif p.get("grating_period_um"):
        variants["waveguide_gc"] = [(None, True)]
    top = gdstk.Cell(f"{p['name']}_TOP")
    y_off = 0.0
    for fname, items in variants.items():
        lib = gdstk.Library(unit=1e-6, precision=1e-9)
        for gap, with_gc in items:
            label = fname + (f"_gap{gap * 1e3:.0f}nm" if gap else "")
            cell = gdstk.Cell(label)
            bus(cell)
            if gap and fname != "waveguide":
                ring(cell, gap)
            if with_gc:
                coupler(cell, +1)
                coupler(cell, -1)
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
