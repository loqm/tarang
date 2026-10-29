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
column of panels and 7 in otherwise, with text (6-8 pt by role), lines and markers scaled to
that size, titles and notes moved to the caption in the JSON, panel letters (a), (b), ...,
legends kept off the data, and written to ``figures_paper/`` instead of ``figures/``
(``to_paper``).

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
    # rows: a panel joins a row when its height overlaps every panel of the row by more than
    # half the shorter one (a tall panel beside two short rows belongs to the first)
    def same_row(p, q):
        return min(p.y1, q.y1) - max(p.y0, q.y0) > 0.5 * min(p.height, q.height)
    rows: list[list] = []
    for ax in sorted(axes, key=lambda a: (-round(a.get_position().y1, 3), a.get_position().x0)):
        p = ax.get_position()
        for row in rows:
            if all(same_row(p, a.get_position()) for a in row):
                row.append(ax)
                break
        else:
            rows.append([ax])
    return [ax for row in rows for ax in sorted(row, key=lambda a: round(a.get_position().x0, 3))]


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


#: font sizes of the paper style, by role (pt): (smallest, largest)
PAPER_FONTS = {"tick": (6.0, 7.0), "label": (6.5, 8.0), "legend": (6.0, 7.0),
               "letter": (8.0, 8.0), "text": (6.0, 8.0)}
SHORT_TITLE = 6          # titles up to this many characters stay on the panel (|E|², Ex, TE00)


def keep_title(ax) -> None:
    """Keep this panel's title on the figure in the paper style (other titles go to the
    caption): for column headings such as |E|², Ex, Ey."""
    ax.title.set_gid("tarang-keep")


def _clamp(v: float, role: str) -> float:
    lo, hi = PAPER_FONTS[role]
    return min(hi, max(lo, v))


def _inside(bb, outer, tol=1.5) -> bool:
    return (bb.x0 >= outer.x0 - tol and bb.x1 <= outer.x1 + tol and bb.y0 >= outer.y0 - tol
            and bb.y1 <= outer.y1 + tol)


def _data_points(ax):
    """Display coordinates of what an axes plots (lines, densified; scatter points)."""
    import numpy as np
    pts = []
    for ln in ax.get_lines():
        if not ln.get_visible():
            continue
        try:
            xy = np.asarray(ln.get_xydata(), float)
        except (TypeError, ValueError):
            continue
        xy = xy[np.isfinite(xy).all(axis=1)] if xy.ndim == 2 else xy[:0]
        if len(xy) == 0:
            continue
        if 1 < len(xy) < 400 and ln.get_linestyle() not in ("None", "", " ", "none"):
            t = np.linspace(0, 1, 12)[:-1]
            seg = xy[:-1, None, :] + (xy[1:, None, :] - xy[:-1, None, :]) * t[None, :, None]
            xy = np.vstack([seg.reshape(-1, 2), xy[-1:]])
        pts.append(ln.get_transform().transform(xy))
    from matplotlib.container import BarContainer
    for c in ax.containers:                           # bars: their top edge and middle
        if isinstance(c, BarContainer):
            for r in c.patches:
                x0, y0, w, h = r.get_x(), r.get_y(), r.get_width(), r.get_height()
                xs = x0 + w * np.linspace(0, 1, 7)
                pts.append(r.get_transform().transform(np.r_[
                    np.c_[xs, np.full(7, y0 + h)], np.c_[xs, np.full(7, y0 + h / 2)]]))
    for col in ax.collections:
        off = getattr(col, "get_offsets", lambda: None)()
        if off is not None and len(off) and col.get_visible() and hasattr(col, "get_sizes") \
                and type(col).__name__ == "PathCollection":
            pts.append(col.get_offset_transform().transform(np.asarray(off, float)))
    return np.vstack(pts) if pts else np.zeros((0, 2))


def _clashes(ax, bb, renderer, skip=()) -> int:
    """How many plotted points and panel texts fall inside the box ``bb`` (display units)."""
    import numpy as np
    pts = _data_points(ax)
    n = 0
    if len(pts):
        n += int(np.count_nonzero((pts[:, 0] > bb.x0 + 1) & (pts[:, 0] < bb.x1 - 1)
                                  & (pts[:, 1] > bb.y0 + 1) & (pts[:, 1] < bb.y1 - 1)))
    for t in ax.texts:
        if t in skip or not t.get_visible() or not t.get_text():
            continue
        if t.get_window_extent(renderer).overlaps(bb):
            n += 5
    return n


def _below_offset(ax, renderer) -> float:
    """How far (points) the panel's x tick labels and x label reach below it: a legend put
    that far under the panel stays clear of them whatever the layout does to the panel."""
    pos = tuple(round(v, 3) for v in ax.get_position().bounds)
    twins = [a for a in ax.figure.axes                # a twin axis carries the x labels
             if tuple(round(v, 3) for v in a.get_position().bounds) == pos]
    legs = [a.get_legend() for a in twins if a.get_legend() is not None]
    for leg in legs:
        leg.set_visible(False)
    y0 = min(a.get_tightbbox(renderer, for_layout_only=True).y0 for a in twins)
    for leg in legs:
        leg.set_visible(True)
    return (ax.bbox.y0 - y0) * 72.0 / ax.figure.dpi + 2.0


def _set_loc(leg, loc: str) -> None:
    try:
        leg.set_loc(loc)                                      # matplotlib >= 3.8
    except AttributeError:
        from matplotlib.legend import Legend
        leg._loc = Legend.codes[loc]


def _relegend(ax, leg, ncols, **kw):
    """The same legend made again (its columns are fixed when a legend is made)."""
    return ax.legend(handles=leg.legend_handles, labels=[t.get_text() for t in leg.get_texts()],
                     ncols=ncols, fontsize=leg.get_texts()[0].get_fontsize(),
                     frameon=leg.get_frame_on(), **kw)


def _room(fig, ax) -> float:
    """Width (display units) a legend under ``ax`` may take: the whole figure when the panel
    is alone in its row, else the share of the row between its neighbours' middles."""
    p = ax.get_position()
    row = sorted({round(a.get_position().x0, 3): a for a in _panels(fig)
                  if min(p.y1, a.get_position().y1) - max(p.y0, a.get_position().y0)
                  > 0.5 * min(p.height, a.get_position().height)}.values(),
                 key=lambda a: a.get_position().x0)
    if len(row) <= 1:
        return fig.bbox.width * 0.98
    mids = [0.0] + [0.5 * (a.get_position().x1 + b.get_position().x0)
                    for a, b in zip(row, row[1:])] + [1.0]
    i = next(k for k, a in enumerate(row) if round(a.get_position().x0, 3) == round(p.x0, 3))
    return (mids[i + 1] - mids[i]) * fig.bbox.width


def _tidy_legends(fig, renderer) -> None:
    """A legend drawn on its panel's data moves to the least crowded corner; if every corner
    is crowded it goes under the panel. A legend under a panel that is wider than the whole
    figure is set in one column."""
    import numpy as np
    for ax in _panels(fig):
        leg = ax.get_legend()
        if leg is None or not leg.get_visible():
            continue
        bb = leg.get_window_extent(renderer)
        if bb.overlaps(ax.bbox) and _clashes(ax, bb, renderer):
            n_best = None
            for loc in ("upper right", "upper left", "lower left", "lower right", "center right",
                        "center left", "upper center", "lower center"):
                _set_loc(leg, loc)
                leg.set_bbox_to_anchor(None)
                c = _clashes(ax, leg.get_window_extent(renderer), renderer)
                if n_best is None or c < n_best[0]:
                    n_best = (c, loc)
                if c == 0:
                    break
            if n_best[0] == 0:
                _set_loc(leg, n_best[1])
            else:                                            # under the panel
                from matplotlib.transforms import offset_copy
                below = offset_copy(ax.transAxes, fig=fig, y=-_below_offset(ax, renderer),
                                    units="points")
                leg = _relegend(ax, leg, min(2, len(leg.get_texts())), loc="upper center",
                                bbox_to_anchor=(0.5, 0.0), bbox_transform=below)
        bb = leg.get_window_extent(renderer)
        if bb.y1 < ax.bbox.y0 + 1:                 # under the panel: clear of its x labels
            from matplotlib.transforms import offset_copy
            anchor = leg.get_bbox_to_anchor()
            xa = ax.transAxes.inverted().transform((anchor.x0 + anchor.width / 2, 0))[0]
            below = offset_copy(ax.transAxes, fig=fig, y=-_below_offset(ax, renderer),
                                units="points")
            ncols = getattr(leg, "_ncols", 1)
            if bb.width > _room(fig, ax) and ncols > 1:
                ncols = 1
            _relegend(ax, leg, ncols, loc="upper center", bbox_to_anchor=(xa, 0.0),
                      bbox_transform=below)


def _tidy_texts(fig, renderer, caption: dict) -> None:
    """Notes set on a panel that spill over its edge get smaller (down to 5.5 pt); if they
    still spill, they go to the caption."""
    for i, ax in enumerate(_panels(fig)):
        box = ax.bbox
        for t in ax.texts:
            if not t.get_visible() or not t.get_text() or t.get_gid() == "tarang-letter":
                continue
            x, y = t.get_transform().transform(t.get_position())
            if not (box.x0 <= x <= box.x1 and box.y0 <= y <= box.y1):
                if len(t.get_text()) > 40:        # a note written beside the panel
                    caption.setdefault(f"note {_letter(i)}", []).append(t.get_text())
                    t.set_visible(False)
                continue                          # a short label placed outside on purpose
            fs = t.get_fontsize()
            while not _inside(t.get_window_extent(renderer), box) and fs > 5.5:
                fs = max(5.5, fs - 0.5)
                t.set_fontsize(fs)
            if not _inside(t.get_window_extent(renderer), box):
                caption.setdefault(f"note {_letter(i)}", []).append(t.get_text())
                t.set_visible(False)


def _letter(i: int) -> str:
    return f"({chr(ord('a') + i)})" if i < 26 else f"({i})"


def to_paper(fig) -> dict:
    """Make a figure drawn for the screen journal-sized and tidy, and return its caption.

    1. Size: 3.4 in wide for one column of panels, 7 in otherwise (``paper_size``); lines
       and markers scaled with it; text by role (``PAPER_FONTS``: ticks 6-7 pt, axis labels
       6.5-8 pt, legends 6-7 pt, other text 6-8 pt).
    2. Caption: the figure title, panel titles longer than ``SHORT_TITLE`` characters (unless
       ``keep_title``) and notes written on the figure go to the caption (returned, for the
       JSON); panels get letters (a), (b), ... when there are several, so the caption can
       name them. Dense in-cell annotations of maps (more than 15 texts in a panel) are hidden.
    3. Layout: a legend that covers plotted data moves to a free corner, else under its panel;
       a legend under a panel set in more columns than fit goes to fewer; notes spilling over
       a panel's edge shrink, else go to the caption."""
    from matplotlib.lines import Line2D
    from matplotlib.text import Text
    w0, _ = fig.get_size_inches()
    w, h = paper_size(fig)
    k = w / w0
    caption: dict = {}
    sup = getattr(fig, "_suptitle", None)
    if sup is not None and sup.get_text():
        caption["figure"] = sup.get_text()
        sup.set_visible(False)
    notes = [t for t in fig.texts if t is not sup and t.get_visible() and t.get_text()]
    for t in notes:
        caption.setdefault("notes", []).append(t.get_text())
        t.set_visible(False)
    panels = _panels(fig)
    seen, distinct = set(), []
    for ax in panels:                                  # twin axes share one letter
        key = tuple(round(v, 3) for v in ax.get_position().bounds)
        if key not in seen:
            seen.add(key)
            distinct.append(ax)
    lettered = len(distinct) > 1
    for i, ax in enumerate(distinct):
        twins = [a for a in panels if tuple(round(v, 3) for v in a.get_position().bounds)
                 == tuple(round(v, 3) for v in ax.get_position().bounds)]
        for a in twins:
            title = a.get_title()
            if title and (a.title.get_gid() != "tarang-keep"
                          and (len(title) > SHORT_TITLE or "\n" in title)):
                caption[_letter(i) if lettered else "panel"] = title
                a.set_title("")
            if len(a.texts) > 15:
                for t in a.texts:
                    t.set_visible(False)
        if lettered:
            ax.annotate(_letter(i), xy=(0, 1), xycoords="axes fraction", xytext=(-2, 4),
                        textcoords="offset points", ha="right", va="bottom",
                        fontweight="bold", gid="tarang-letter", annotation_clip=False)
    # legends already under a panel: at most 2 columns, moved down clear of the smaller labels
    below = []
    for ax in panels:
        leg = ax.get_legend()
        if leg is None:
            continue
        try:
            bb = leg.get_bbox_to_anchor()
            xa, ya = ax.transAxes.inverted().transform((bb.x0 + bb.width / 2, bb.y0))
        except Exception:                                     # noqa: BLE001
            continue
        if ya < -0.01:          # not a rounding error of a legend anchored to its panel
            below.append((ax, leg, xa, ya))
    fig.set_size_inches(w, h)
    for ax, leg, xa, ya in below:
        if getattr(leg, "_ncols", 1) > 2:
            leg = _relegend(ax, leg, 2, loc=leg._loc)
        leg.set_bbox_to_anchor((xa, ya - 0.10), transform=ax.transAxes)   # refined below
    # text sizes by role
    special: dict = {}
    for ax in fig.axes:
        for t in [ax.xaxis.label, ax.yaxis.label]:
            special[id(t)] = "label"
        for t in ax.get_xticklabels() + ax.get_yticklabels() + \
                [ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text()]:
            special[id(t)] = "tick"
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts() + [leg.get_title()]:
                special[id(t)] = "legend"
        for t in ax.texts:
            if t.get_gid() == "tarang-letter":
                special[id(t)] = "letter"
    for ax in fig.axes:
        fs = ax.xaxis.get_ticklabels()[0].get_fontsize() if ax.xaxis.get_ticklabels() else 8
        ax.tick_params(labelsize=_clamp(fs * k, "tick"))
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text():
            t.set_fontsize(_clamp(t.get_fontsize() * k, special.get(id(t), "text")))
    for ln in fig.findobj(Line2D):
        ln.set_linewidth(min(1.5, max(0.6, ln.get_linewidth() * k)))
        ln.set_markersize(min(5.0, max(2.0, ln.get_markersize() * k)))
    # layout checks on the drawn figure: legends first (moving one changes the layout), then
    # the notes, measured against the panels as they will be saved
    _layout(fig)
    try:
        _tidy_legends(fig, fig.canvas.get_renderer())
    except Exception:                                         # noqa: BLE001
        pass                    # a figure the checks cannot read keeps its own layout
    _layout(fig)
    try:
        _tidy_texts(fig, fig.canvas.get_renderer(), caption)
    except Exception:                                         # noqa: BLE001
        pass
    return caption


def _layout(fig) -> None:
    try:
        fig.tight_layout()
    except Exception:                                         # noqa: BLE001
        pass


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
    if layout and not paper:                                  # to_paper lays the figure out
        _layout(fig)
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
