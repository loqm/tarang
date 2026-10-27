"""Saving figures for papers: every figure with its numbers.

``save(fig, path)`` writes the figure in each format of ``SETTINGS["formats"]`` (png at
``SETTINGS["dpi"]``; pdf and svg keep text as text) and, with ``SETTINGS["data"]``, two
files next to it so the figure can be redrawn in any plotting program:

* ``<name>_data.csv``: every plotted line, point set, bar set and error bar, one row per
  point, in long form: ``panel, series, kind, x, y, z, lo, hi`` (``panel`` counts the plot
  panels from 0, left to right and top to bottom; ``kind`` is line, points, bars, errorbar,
  hline or vline). Values are in the units of the axis labels, which are in the JSON.
* ``<name>_meta.json``: the axis labels, scales and limits of each panel, the series names,
  the package versions, the time, and whatever the caller adds (the settings that made it).
  Maps and images (2-D arrays) go to ``<name>_data.npz`` (x, y, z per panel), listed here.

``SETTINGS["style"]``: ``"screen"`` (as drawn) or ``"paper"``: saved 3.4 in wide for one
column of panels and 7 in otherwise, with text (6-8.5 pt), lines and markers scaled to that
size, titles moved to the caption in the JSON, and written to ``figures_paper/`` instead
of ``figures/`` (``to_paper``).

The settings travel to the dispersion-mapping tools, which run as separate processes,
through the environment variables ``TARANG_FIGURE_FORMATS``, ``TARANG_FIGURE_DPI``,
``TARANG_FIGURE_STYLE`` and ``TARANG_FIGURE_DATA``.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
import platform
from pathlib import Path
from typing import Optional

STYLES = ("screen", "paper")
FORMATS = ("png", "pdf", "svg")
ONE_COLUMN_IN, TWO_COLUMN_IN, MAX_HEIGHT_IN = 3.4, 7.0, 9.0

#: rcParams of the paper style (applied before drawing)
PAPER_RC = {"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
            "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.2,
            "lines.markersize": 4, "axes.linewidth": 0.8, "xtick.major.width": 0.8,
            "ytick.major.width": 0.8, "legend.frameon": False}


def _env_list(name: str, default: list[str]) -> list[str]:
    v = os.environ.get(name, "")
    return [x.strip().lower() for x in v.split(",") if x.strip()] or default


SETTINGS: dict = {
    "formats": _env_list("TARANG_FIGURE_FORMATS", ["png"]),
    "dpi": int(os.environ.get("TARANG_FIGURE_DPI", "0") or 0) or 300,
    "style": os.environ.get("TARANG_FIGURE_STYLE", "screen") or "screen",
    "data": os.environ.get("TARANG_FIGURE_DATA", "1") != "0",
}


def configure(formats=None, dpi=None, style=None, data=None) -> None:
    """Set what ``save`` writes (``env()`` hands it to the tools this process starts)."""
    if formats:
        fm = [str(f).lower().lstrip(".") for f in formats]
        bad = set(fm) - set(FORMATS)
        if bad:
            raise ValueError(f"unknown figure format(s) {sorted(bad)}; use png, pdf, svg")
        SETTINGS["formats"] = fm
    if dpi:
        SETTINGS["dpi"] = int(dpi)
    if style is not None:
        if style not in STYLES:
            raise ValueError(f"unknown figure style {style!r}; use 'screen' or 'paper'")
        SETTINGS["style"] = style
    if data is not None:
        SETTINGS["data"] = bool(data)


def env() -> dict:
    """The settings as environment variables, for a tool started as a separate process."""
    return dict(TARANG_FIGURE_FORMATS=",".join(SETTINGS["formats"]),
                TARANG_FIGURE_DPI=str(SETTINGS["dpi"]), TARANG_FIGURE_STYLE=SETTINGS["style"],
                TARANG_FIGURE_DATA="1" if SETTINGS["data"] else "0")


def apply_style(plt) -> None:
    """Put the paper style's sizes into matplotlib (call before drawing)."""
    if SETTINGS["style"] == "paper":
        plt.rcParams.update(PAPER_RC)


# ----------------------------------------------------------------------- extraction
def _panels(fig) -> list:
    """The plot panels (colour bars left out), left to right, top to bottom."""
    axes = [ax for ax in fig.axes if ax.get_visible() and not hasattr(ax, "_colorbar")
            and getattr(ax, "get_label", lambda: "")() != "<colorbar>"]
    return sorted(axes, key=lambda ax: (-round(ax.get_position().y1, 2),
                                        round(ax.get_position().x0, 2)))


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    return "" if math.isnan(f) else f"{f:.10g}"           # 10 significant digits


def extract(fig) -> tuple[list[dict], list[dict], dict]:
    """(rows of the CSV, panel descriptions for the JSON, 2-D arrays for the npz)."""
    import numpy as np
    from matplotlib.collections import PathCollection, QuadMesh
    from matplotlib.container import BarContainer, ErrorbarContainer
    from matplotlib.image import AxesImage

    rows: list[dict] = []
    panels: list[dict] = []
    grids: dict = {}

    def add(p, series, kind, xs, ys, zs=None, lo=None, hi=None):
        n = len(xs)
        for k in range(n):
            rows.append(dict(panel=p, series=series, kind=kind, x=_num(xs[k]), y=_num(ys[k]),
                             z=_num(zs[k]) if zs is not None else "",
                             lo=_num(lo[k]) if lo is not None else "",
                             hi=_num(hi[k]) if hi is not None else ""))

    for p, ax in enumerate(_panels(fig)):
        names: list[str] = []
        in_err: set = set()
        for c in ax.containers:
            if isinstance(c, ErrorbarContainer):
                line, _caps, bars = c.lines
                if line is not None:
                    in_err.add(id(line))
                label = c.get_label()
                label = label if label and not label.startswith("_") else f"errorbar {len(names)}"
                names.append(label)
                xs = np.asarray(line.get_xdata(), float) if line is not None else []
                ys = np.asarray(line.get_ydata(), float) if line is not None else []
                lo = hi = None
                if bars:
                    segs = bars[0].get_segments()
                    if len(segs) == len(xs) and all(abs(s[0][0] - s[1][0]) < 1e-12 for s in segs):
                        lo = [min(s[0][1], s[1][1]) for s in segs]
                        hi = [max(s[0][1], s[1][1]) for s in segs]
                add(p, label, "errorbar", xs, ys, lo=lo, hi=hi)
            elif isinstance(c, BarContainer):
                label = c.get_label()
                label = label if label and not label.startswith("_") else f"bars {len(names)}"
                names.append(label)
                xs = [r.get_x() + r.get_width() / 2 for r in c.patches]
                ys = [r.get_height() for r in c.patches]
                add(p, label, "bars", xs, ys)
        for j, line in enumerate(ax.get_lines()):
            if id(line) in in_err or not line.get_visible():
                continue
            xs, ys = line.get_xdata(), line.get_ydata()
            try:
                xs, ys = np.asarray(xs, float), np.asarray(ys, float)
            except (TypeError, ValueError):
                continue                                     # dates or text: not numbers
            label = line.get_label()
            label = label if label and not label.startswith("_") else f"line {j}"
            tr = line.get_transform()
            if tr == ax.get_yaxis_transform():               # axhline: y in data units
                add(p, label, "hline", [""], ys[:1])
            elif tr == ax.get_xaxis_transform():             # axvline: x in data units
                add(p, label, "vline", xs[:1], [""])
            elif len(xs):
                add(p, label, "line", xs, ys)
            names.append(label)
        for j, col in enumerate(ax.collections):
            label = col.get_label()
            label = label if label and not label.startswith("_") else f"set {j}"
            if isinstance(col, QuadMesh):
                xy = col.get_coordinates()
                z = np.asarray(col.get_array(), float).reshape(xy.shape[0] - 1, xy.shape[1] - 1)
                xc = 0.25 * (xy[:-1, :-1] + xy[1:, :-1] + xy[:-1, 1:] + xy[1:, 1:])
                grids[f"p{p}_{len(grids)}"] = dict(x=xc[..., 0], y=xc[..., 1], z=z)
                names.append(label + " (map, in the npz)")
            elif isinstance(col, PathCollection):
                off = np.asarray(col.get_offsets(), float)
                if off.ndim != 2 or not len(off):
                    continue
                arr = col.get_array()
                z = np.asarray(arr, float) if arr is not None and len(arr) == len(off) else None
                add(p, label, "points", off[:, 0], off[:, 1], z)
                names.append(label)
        for j, im in enumerate(ax.images):
            if isinstance(im, AxesImage):
                z = np.asarray(im.get_array(), float)
                if z.ndim != 2:
                    continue                                 # an RGB picture, not data
                x0, x1, y0, y1 = im.get_extent()
                ny, nx = z.shape
                xs = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
                ys = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
                if im.origin == "upper":
                    ys = ys[::-1]
                X, Y = np.meshgrid(xs, ys)
                grids[f"p{p}_{len(grids)}"] = dict(x=X, y=Y, z=z)
                names.append(f"image {j} (in the npz)")
        panels.append(dict(panel=p, title=ax.get_title(), xlabel=ax.get_xlabel(),
                           ylabel=ax.get_ylabel(), xscale=ax.get_xscale(),
                           yscale=ax.get_yscale(), xlim=[float(v) for v in ax.get_xlim()],
                           ylim=[float(v) for v in ax.get_ylim()], series=names))
    return rows, panels, grids


def versions() -> dict:
    out = {"python": platform.python_version()}
    for mod in ("tarang", "tidy3d", "numpy", "matplotlib"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:                                     # noqa: BLE001
            out[mod] = "?"
    return out


def write_data(fig, path: Path, meta: Optional[dict] = None) -> list[Path]:
    """The CSV (and npz) of what ``fig`` plots and its JSON, named after ``path``."""
    import numpy as np
    path = Path(path)
    rows, panels, grids = extract(fig)
    written = []
    csv_path = path.with_name(path.stem + "_data.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["panel", "series", "kind", "x", "y", "z", "lo", "hi"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    written.append(csv_path)
    if grids:
        npz = path.with_name(path.stem + "_data.npz")
        np.savez_compressed(npz, **{f"{k}_{c}": v[c] for k, v in grids.items() for c in "xyz"})
        written.append(npz)
    info = dict(figure=path.name, made=dt.datetime.now().isoformat(timespec="seconds"),
                style=SETTINGS["style"], versions=versions(), panels=panels,
                data=[p.name for p in written],
                columns="panel, series, kind (line, points, bars, errorbar, hline, vline), "
                        "x, y, z (colour value of points), lo and hi (error bar ends)",
                **(meta or {}))
    js = path.with_name(path.stem + "_meta.json")
    js.write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
    written.append(js)
    return written


# --------------------------------------------------------------------------- saving
def paper_size(fig) -> tuple[float, float]:
    """Width for a journal: one column (3.4 in) for a single column of panels drawn up to 9 in
    wide, else two columns (7 in); the height keeps the figure's proportions."""
    xs = {round(ax.get_position().x0, 2) for ax in _panels(fig)}
    w0, h0 = fig.get_size_inches()
    width = ONE_COLUMN_IN if len(xs) <= 1 and w0 <= 9 else TWO_COLUMN_IN
    return width, min(MAX_HEIGHT_IN, h0 * width / w0)


def _write(fig, target: Path, notes: Optional[list] = None, **kw) -> Path:
    """Save via a temporary file and swap it in; if the old file is held open by another
    program (e.g. a Windows image viewer), keep it and write <name>_new.<ext>."""
    tmp = target.with_name(target.stem + ".tmp" + target.suffix)
    fig.savefig(tmp, **kw)
    try:
        os.replace(tmp, target)
        return target
    except OSError:
        alt = target.with_name(target.stem + "_new" + target.suffix)
        try:
            os.replace(tmp, alt)
        except OSError:
            tmp.unlink(missing_ok=True)
            raise
        if notes is not None:
            notes.append(f"{target.name} is open in another program; saved as {alt.name}")
        return alt


def to_paper(fig) -> dict:
    """Make a figure drawn for the screen journal-sized: resize it (``paper_size``) and scale
    every text, line and marker by the same factor, within journal limits (text 6-8.5 pt).
    Titles go: a paper puts them in the caption (returned, for the JSON); so do dense in-cell
    annotations of maps (more than 15 texts in one panel), which cannot stay legible."""
    from matplotlib.lines import Line2D
    from matplotlib.text import Text
    w0, _ = fig.get_size_inches()
    w, h = paper_size(fig)
    k = w / w0
    caption = {}
    sup = getattr(fig, "_suptitle", None)
    if sup is not None and sup.get_text():
        caption["figure"] = sup.get_text()
        sup.set_visible(False)
    for i, ax in enumerate(_panels(fig)):
        if ax.get_title():
            caption[f"panel {i}"] = ax.get_title()
            ax.set_title("")
        if len(ax.texts) > 15:
            for t in ax.texts:
                t.set_visible(False)
    # legends under a panel: at most 2 columns, and moved down clear of the smaller labels
    below = []
    for ax in _panels(fig):
        leg = ax.get_legend()
        if leg is None:
            continue
        try:
            bb = leg.get_bbox_to_anchor()
            xa, ya = ax.transAxes.inverted().transform((bb.x0 + bb.width / 2, bb.y0))
        except Exception:                                     # noqa: BLE001
            continue
        if ya < 0:
            below.append((ax, leg, xa, ya))
    fig.set_size_inches(w, h)
    for ax, leg, xa, ya in below:
        if getattr(leg, "_ncols", 1) > 2:   # a legend's columns are fixed when it is made
            leg = ax.legend(handles=leg.legend_handles,
                            labels=[t.get_text() for t in leg.get_texts()], ncols=2,
                            loc=leg._loc, fontsize=leg.get_texts()[0].get_fontsize(),
                            frameon=leg.get_frame_on())
        leg.set_bbox_to_anchor((xa, ya - 0.10), transform=ax.transAxes)
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text():
            t.set_fontsize(min(8.5, max(6.0, t.get_fontsize() * k)))
    for ln in fig.findobj(Line2D):
        ln.set_linewidth(min(1.5, max(0.6, ln.get_linewidth() * k)))
        ln.set_markersize(min(5.0, max(2.0, ln.get_markersize() * k)))
    return caption


def paper_path(path: Path) -> Path:
    """Where the paper style writes: ``figures_paper/`` instead of ``figures/``, so the
    screen figures (shown by the GUI and the report) are kept."""
    parts = list(Path(path).parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "figures":
            parts[i] = "figures_paper"
            return Path(*parts)
    return Path(path)


def save(fig, path: Path, meta: Optional[dict] = None, notes: Optional[list] = None,
         close: bool = True, layout: bool = True) -> Path:
    """Write ``fig`` in every format (and its data); returns the png path (or the first).
    ``layout=False`` keeps the figure's own layout (the paper style always re-lays it out
    after resizing)."""
    path = Path(path)
    paper = SETTINGS["style"] == "paper"
    if paper:
        path = paper_path(path)
        caption = to_paper(fig)
        if caption:
            meta = dict(meta or {}, caption=caption)
    path.parent.mkdir(parents=True, exist_ok=True)
    if layout or paper:
        try:
            fig.tight_layout()
        except Exception:                                     # noqa: BLE001
            pass
    out = None
    for fmt in SETTINGS["formats"]:
        target = path.with_suffix("." + fmt)
        kw = {"dpi": SETTINGS["dpi"]} if fmt == "png" else {}
        kw.update(bbox_inches="tight", pad_inches=0.03)       # nothing cut off at the edges
        written = _write(fig, target, notes, **kw)
        if fmt == "png" or out is None:
            out = written
    if SETTINGS["data"]:
        try:
            write_data(fig, path, meta)
        except Exception as err:                              # noqa: BLE001
            if notes is not None:
                notes.append(f"numbers of {path.name} not saved: {type(err).__name__}: {err}")
    if close:
        import matplotlib.pyplot as plt
        plt.close(fig)
    return out
