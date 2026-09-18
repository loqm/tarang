"""Read MultiNest chains and draw corner plots without importing the native solver.

Exposes ChainData, load_multinest_samples(), and plot_corner().
Adapted from the supplied corner.py reference; TARANG integration, 2026.
"""

from dataclasses import dataclass
from pathlib import Path
import os

import numpy as np

from tarang import get_logger

logger = get_logger(__name__)

DEFAULT_PARAM_NAMES = ["period", "fill_factor", "etch_depth", "num_periods", "source_x"]
DEFAULT_PARAM_LABELS = [
    r"\Lambda\ (\mu\mathrm{m})",
    r"\mathrm{FF}",
    r"t_{\mathrm{etch}}\ (\mu\mathrm{m})",
    r"N",
    r"X_{\mathrm{src}}\ (\mu\mathrm{m})",
]
_SUFFIXES = {
    "weighted": (".txt",),
    "equal": ("post_equal_weights.dat",),
    "live": ("phys_live.points", "physlive.dat", "phys_live.points.txt"),
}


@dataclass(frozen=True)
class ChainData:
    """A read-only snapshot description with physical samples and log likelihoods."""

    samples: np.ndarray
    weights: np.ndarray
    log_likelihood: np.ndarray
    path: Path
    prefix: str
    kind: str
    skipped_rows: int = 0


def _read_rows(path):
    """Read complete numeric rows from one bounded snapshot of a chain file.
    Ignore an unfinished final line; reject malformed complete rows.
    """
    with path.open("rb") as stream:
        content = stream.read(os.fstat(stream.fileno()).st_size)
    lines = content.splitlines(keepends=True)
    skipped = 0
    if lines and not lines[-1].endswith((b"\n", b"\r")):
        lines.pop()
        skipped = 1
    rows = []
    for number, line in enumerate(
        lines,
        start=1,
    ):
        line = line.split(
            b"#",
            1,
        )[0].strip()
        if not line:
            continue
        try:
            row = [
                float(
                    value.replace(
                        b"D",
                        b"E",
                    ).replace(
                        b"d",
                        b"e",
                    )
                )
                for value in line.split()
            ]
        except ValueError as exc:
            raise ValueError(f"Non-numeric row {number} in {path}") from exc
        if rows and len(row) != len(rows[0]):
            raise ValueError(f"Inconsistent column count at row {number} in {path}")
        rows.append(row)
    if not rows:
        return np.empty((0, 0)), skipped
    return (
        np.asarray(
            rows,
            dtype=float,
        ),
        skipped,
    )


def _find_prefix(directory):
    """Find a unique MultiNest prefix in a run directory or its chains folder.
    Require an explicit prefix if multiple chain families are present.
    """
    prefixes = set()
    for path in directory.iterdir():
        if not path.is_file() or path.stat().st_size == 0:
            continue
        matched = False
        for suffix in (*_SUFFIXES["equal"], *_SUFFIXES["live"]):
            if path.name.endswith(suffix):
                prefixes.add(path.name[: -len(suffix)])
                matched = True
                break
        if (
            matched
            or not path.name.endswith(".txt")
            or path.name.endswith("summary.txt")
        ):
            continue
        # Weighted chains have at least three numeric columns. Do not treat
        # arbitrary README or parameter-name text files as a posterior chain.
        with path.open("rb") as stream:
            first = next(
                (
                    line
                    for line in stream
                    if line.strip() and not line.lstrip().startswith(b"#")
                ),
                b"",
            )
        try:
            values = [
                float(
                    value.replace(
                        b"D",
                        b"E",
                    )
                )
                for value in first.split()
            ]
        except ValueError:
            continue
        if len(values) >= 3:
            prefixes.add(path.name[:-4])
    if not prefixes:
        raise FileNotFoundError(f"No supported MultiNest chains found in {directory}")
    if len(prefixes) > 1:
        raise ValueError(
            f"Multiple chain prefixes found: {sorted(prefixes)}. Specify prefix explicitly."
        )
    return prefixes.pop()


def load_multinest_samples(
    directory: str | Path,
    *,
    prefix: str | None = None,
    source: str = "auto",
) -> ChainData:
    """Load physical samples from weighted, equal-weight, or live-point chains.
    Auto mode prefers posterior files, then falls back to live diagnostic points.
    """
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Run or chains directory does not exist: {root}")
    if source not in ("auto", "weighted", "equal", "live"):
        raise ValueError("source must be 'auto', 'weighted', 'equal', or 'live'")
    chain_dir = root / "chains" if (root / "chains").is_dir() else root
    prefix = _find_prefix(chain_dir) if prefix is None else prefix
    if (
        not isinstance(
            prefix,
            str,
        )
        or "/" in prefix
        or "\\" in prefix
        or prefix in (".", "..")
    ):
        raise ValueError(
            "prefix must be a filename prefix without directory components"
        )
    kinds = ("weighted", "equal", "live") if source == "auto" else (source,)
    for kind in kinds:
        for suffix in _SUFFIXES[kind]:
            path = chain_dir / f"{prefix}{suffix}"
            if not path.is_file():
                continue
            data, skipped = _read_rows(path)
            if data.size == 0:
                continue
            trailing = 1 if kind == "equal" else 2
            if data.shape[1] <= trailing:
                raise ValueError(f"No parameter columns found in {path}")
            if kind == "weighted":
                weights, loglike, samples = data[:, 0], -0.5 * data[:, 1], data[:, 2:]
                if np.any(weights < 0):
                    raise ValueError(f"Negative sample weights in {path}")
            else:
                samples = data[:, :-trailing]
                loglike = data[:, -trailing]
                weights = np.ones(len(data))
            valid = (
                np.all(
                    np.isfinite(samples),
                    axis=1,
                )
                & np.isfinite(loglike)
                & np.isfinite(weights)
                & (weights > 0)
            )
            skipped += int(np.count_nonzero(~valid))
            if not np.any(valid):
                continue
            if skipped:
                logger.warning(
                    "Skipped %s unfinished, nonfinite, or zero-weight rows in %s",
                    skipped,
                    path,
                )
            return ChainData(
                samples[valid],
                weights[valid],
                loglike[valid],
                path,
                prefix,
                kind,
                skipped,
            )
    raise FileNotFoundError(
        f"No complete usable {source} chain rows for prefix {prefix!r} in {chain_dir}. Retry after the next sampler flush."
    )


def _parameter_labels(
    chain,
    names,
    labels,
):
    """Resolve labels from explicit arguments, a paramnames file, or defaults.
    Validate all names against the number of physical parameter columns.
    """
    count = chain.samples.shape[1]
    inferred_labels = None
    if names is None:
        name_file = chain.path.parent / f"{chain.prefix}.paramnames"
        if not name_file.exists():
            name_file = chain.path.parent / f"{chain.prefix}paramnames"
        if name_file.is_file():
            entries = [
                line.strip().split(maxsplit=1)
                for line in name_file.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
            names = [entry[0].rstrip("*") for entry in entries]
            inferred_labels = [
                entry[1] if len(entry) == 2 else entry[0] for entry in entries
            ]
        elif count == 5 and chain.prefix.startswith("grating"):
            names, inferred_labels = DEFAULT_PARAM_NAMES, DEFAULT_PARAM_LABELS
        else:
            names = [f"p{i + 1}" for i in range(count)]
    names = list(names)
    labels = list(
        labels
        if labels is not None
        else (inferred_labels if inferred_labels is not None else names)
    )
    if len(names) != count or len(labels) != count or len(set(names)) != count:
        raise ValueError(f"Provide {count} unique parameter names and {count} labels")
    return names, labels


def _draw_corner(
    chain,
    names,
    labels,
    backend,
    filled,
    *,
    plot_dpi: int = 150,  # Resolution of displayed and saved figures (dots per inch).
):
    """Render variable parameters using GetDist or the optional corner backend.
    Drop fixed columns from the plot while preserving them in loaded chain data.
    """
    import matplotlib.pyplot as plt

    varying = (
        np.ptp(
            chain.samples,
            axis=0,
        )
        > 0
    )
    if not np.any(varying):
        raise ValueError(
            "All sampled parameters are constant; no corner distribution can be plotted"
        )
    if not np.all(varying):
        logger.warning(
            "Omitting constant parameters from corner plot: %s",
            [
                name
                for name, keep in zip(
                    names,
                    varying,
                )
                if not keep
            ],
        )
    samples = chain.samples[:, varying]
    names = [
        name
        for name, keep in zip(
            names,
            varying,
        )
        if keep
    ]
    labels = [
        label
        for label, keep in zip(
            labels,
            varying,
        )
        if keep
    ]
    if backend in ("auto", "getdist"):
        try:
            from getdist import MCSamples, plots
        except ImportError:
            if backend == "getdist":
                raise ImportError(
                    'Install plotting dependencies with pip install "tarang-photonics[plots]"'
                ) from None
            backend = "corner"
        else:
            gd_samples = MCSamples(
                samples=samples,
                weights=chain.weights,
                names=names,
                labels=labels,
                sampler="nested",
                settings={
                    "ignore_rows": 0,
                },
            )
            with plt.rc_context({"figure.dpi": plot_dpi}):
                plotter = plots.get_subplot_plotter(width_inch=10)
                plotter.triangle_plot(
                    gd_samples,
                    filled=filled,
                )
            plotter.fig.set_dpi(plot_dpi)
            return plotter.fig, "getdist", names
    try:
        import corner
    except ImportError:
        raise ImportError(
            'Install plotting dependencies with pip install "tarang-photonics[plots]"'
        ) from None
    with plt.rc_context({"figure.dpi": plot_dpi}):
        figure = corner.corner(
            samples,
            weights=chain.weights,
            labels=labels,
            show_titles=True,
            quantiles=[0.16, 0.5, 0.84],
            fill_contours=filled,
        )
    figure.set_dpi(plot_dpi)
    return figure, "corner", names


def plot_corner(
    directory: str | Path,
    *,
    prefix: str | None = None,
    source: str = "auto",
    param_names: list[str] | None = None,
    param_labels: list[str] | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
    save: bool = True,
    backend: str = "auto",
    filled: bool = True,
    dpi: int | None = None,  # Legacy alias; when supplied, overrides plot_dpi.
    plot_dpi: int = 150,  # Resolution of displayed and saved figures (dots per inch).
) -> dict:
    """Display or save a corner plot from a completed run or current chain snapshot.
    Return the figure and source metadata; chain files and optimizer state stay untouched.
    """
    import matplotlib.pyplot as plt

    if backend not in ("auto", "getdist", "corner"):
        raise ValueError("backend must be 'auto', 'getdist', or 'corner'")
    if dpi is not None:
        plot_dpi = dpi
    if not np.isfinite(plot_dpi) or plot_dpi <= 0:
        raise ValueError("plot_dpi must be finite and positive")
    chain = load_multinest_samples(
        directory,
        prefix=prefix,
        source=source,
    )
    if len(chain.samples) < 5:
        raise ValueError(
            f"Only {len(chain.samples)} usable samples; at least 5 are required for a corner plot"
        )
    names, labels = _parameter_labels(
        chain,
        param_names,
        param_labels,
    )
    figure, used_backend, plotted_names = _draw_corner(
        chain,
        names,
        labels,
        backend,
        filled,
        plot_dpi=plot_dpi,
    )
    title = (
        "Live-point diagnostic snapshot"
        if chain.kind == "live"
        else "Posterior chain snapshot"
    )
    figure.suptitle(title)
    target = None
    try:
        if save:
            root = Path(directory)
            root = root.parent if root.name == "chains" else root
            target = (
                Path(output_path)
                if output_path is not None
                else root
                / ("corner_live.png" if chain.kind == "live" else "corner_plot.png")
            )
            if target.resolve() == chain.path.resolve():
                raise ValueError("output_path must not overwrite the input chain")
            target.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            figure.savefig(
                target,
                dpi=plot_dpi,
                bbox_inches="tight",
            )
            logger.info(
                "Corner plot saved to %s",
                target,
            )
        if show:
            plt.show()
    finally:
        if not show:
            plt.close(figure)
    return {
        "figure": figure,
        "output_path": str(target) if target is not None else None,
        "chain_file": str(chain.path),
        "source": chain.kind,
        "sample_count": len(chain.samples),
        "parameters": plotted_names,
        "backend": used_backend,
        "skipped_rows": chain.skipped_rows,
    }
