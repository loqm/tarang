"""
utilities.nested_sampling_post_process.py

Standalone Post-Processing and Visualizer for MultiNest Grating Coupler Runs.

Provides post_process_run() to parse output directories, recover raw chains from
completed or manually killed runs, render GetDist corner plots, and extract
posterior summary statistics as a dictionary.

Author: Esraaj Sarkar Gupta
Date: 2026

Exposes: post_process_run().
"""

from tarang import get_logger

import json
from pathlib import Path
import numpy as np

from .corner_plot import load_multinest_samples, plot_corner

DEFAULT_PARAM_NAMES = ["period", "fill_factor", "etch_depth", "num_periods", "source_x"]
DEFAULT_PARAM_LABELS = [
    r"\Lambda\ (\mu\mathrm{m})",
    r"\mathrm{FF}",
    r"t_{\mathrm{etch}}\ (\mu\mathrm{m})",
    r"N",
    r"X_{\mathrm{src}}\ (\mu\mathrm{m})",
]


logger = get_logger(__name__)


def post_process_run(
    output_dir: Path | str,
    param_names: list[str] = None,
    param_labels: list[str] = None,
    sigma: float = 0.02,
    plot_filename: str = "bayesian_structural_corner_plot.png",
    show: bool = True,
    *,
    plot_dpi: int = 150,  # Resolution of displayed and saved figures (dots per inch).
) -> dict:
    """Parse MultiNest chains and generate a GetDist corner plot.
    Return a consistent posterior summary for complete or interrupted runs.
    """
    out_path = Path(output_dir)
    if not out_path.exists():
        raise FileNotFoundError(
            f"Specified output directory does not exist: {out_path}"
        )

    param_names = param_names or DEFAULT_PARAM_NAMES
    param_labels = param_labels or DEFAULT_PARAM_LABELS

    # Pre-populate complete schema defaults to prevent KeyError upstream
    results_summary = {
        "status": "UNKNOWN",
        "output_dir": str(out_path),
        "total_samples_collected": 0,
        "total_evaluations": 0,
        "total_credits_used": 0.0,
        "max_efficiency": 0.0,
        "peak_loss_db": -120.0,
        "optimal_vector": [],
        "parameter_space_variance": {},
        "top_candidates": [],
        "corner_plot_path": None,
        "error": None,
    }

    # 1. Recover Tracker / Budget Metadata if present
    tracker_file = out_path / "budget_cache" / "tracker.json"
    top_10_file = out_path / "top_10_configurations.json"

    if tracker_file.exists():
        try:
            with open(
                tracker_file,
                "r",
            ) as f:
                t_data = json.load(f)
                results_summary["total_credits_used"] = float(
                    t_data.get(
                        "total_credits_used",
                        0.0,
                    )
                )
                results_summary["total_evaluations"] = int(
                    t_data.get(
                        "total_evaluations",
                        0,
                    )
                )
                results_summary["top_candidates"] = t_data.get(
                    "top_10_runs",
                    [],
                )
        except Exception as e:
            logger.warning(f"Could not load tracker file: {e}")

    if not results_summary["top_candidates"] and top_10_file.exists():
        try:
            with open(
                top_10_file,
                "r",
            ) as f:
                s_data = json.load(f)
                results_summary["top_candidates"] = s_data.get(
                    "top_10_candidates",
                    [],
                )
        except Exception as e:
            logger.warning(f"Could not load top 10 file: {e}")

    # Pre-seed metrics from leaderboard if available (useful if chains are missing/empty)
    if results_summary["top_candidates"]:
        best_cand = results_summary["top_candidates"][0]
        results_summary["max_efficiency"] = float(
            best_cand.get(
                "efficiency",
                0.0,
            )
        )
        results_summary["peak_loss_db"] = float(
            best_cand.get(
                "peak_loss_db",
                -120.0,
            )
        )
        cfg = best_cand.get(
            "config",
            {},
        )
        if cfg:
            results_summary["optimal_vector"] = [
                float(
                    cfg.get(
                        "period",
                        0.0,
                    )
                ),
                float(
                    cfg.get(
                        "ff",
                        0.0,
                    )
                ),
                float(
                    cfg.get(
                        "etch_depth",
                        0.0,
                    )
                ),
                float(
                    cfg.get(
                        "n_teeth",
                        0.0,
                    )
                ),
                float(
                    cfg.get(
                        "source_x",
                        0.0,
                    )
                ),
            ]

    # Use the same reader for completed posterior files and live snapshots.
    try:
        chain = load_multinest_samples(out_path)
    except FileNotFoundError as exc:
        results_summary["status"] = "NO_CHAIN_DATA_FOUND"
        results_summary["error"] = str(exc)
        logger.warning(
            "%s",
            exc,
        )
        return results_summary
    except ValueError as exc:
        results_summary["status"] = "CHAIN_READ_ERROR"
        results_summary["error"] = str(exc)
        return results_summary

    samples_matrix = chain.samples
    weights = chain.weights / np.sum(chain.weights)
    results_summary["total_samples_collected"] = len(samples_matrix)
    results_summary["chain_source"] = chain.kind
    results_summary["chain_file"] = str(chain.path)
    if len(param_names) != samples_matrix.shape[1]:
        raise ValueError("Parameter names do not match the chain's physical columns")
    if len(samples_matrix) < 5:
        results_summary["status"] = "INSUFFICIENT_SAMPLES"
        results_summary["error"] = "At least 5 complete samples are required."
        return results_summary

    try:
        plotted = plot_corner(
            out_path,
            prefix=chain.prefix,
            source=chain.kind,
            param_names=param_names,
            param_labels=param_labels,
            output_path=out_path / plot_filename,
            show=show,
            plot_dpi=plot_dpi,
        )
        results_summary["corner_plot_path"] = plotted["output_path"]
    except (ImportError, ValueError, RuntimeError, OSError) as exc:
        results_summary["corner_plot_error"] = str(exc)
        logger.warning(
            "Corner plot could not be generated: %s",
            exc,
        )

    mean = np.average(
        samples_matrix,
        weights=weights,
        axis=0,
    )
    variance = np.average(
        (samples_matrix - mean) ** 2,
        weights=weights,
        axis=0,
    )
    results_summary["parameter_space_variance"] = {
        name: {
            "marginalized_mean": float(mean[index]),
            "standard_deviation": float(np.sqrt(variance[index])),
        }
        for index, name in enumerate(param_names)
    }
    best_index = int(np.argmax(chain.log_likelihood))
    max_loglike = float(chain.log_likelihood[best_index])
    results_summary["optimal_vector"] = samples_matrix[best_index].tolist()
    if max_loglike <= 0:
        efficiency = float(
            np.clip(
                1.0 - sigma * np.sqrt(-2.0 * max_loglike),
                0.0,
                1.0,
            )
        )
        results_summary["max_efficiency"] = efficiency
        results_summary["peak_loss_db"] = float(10.0 * np.log10(efficiency + 1e-12))
    results_summary["status"] = "LIVE_SNAPSHOT" if chain.kind == "live" else "RECOVERED"

    parsed_json_file = out_path / "post_processed_results.json"
    parsed_json_file.write_text(
        json.dumps(
            results_summary,
            indent=4,
        ),
        encoding="utf-8",
    )
    return results_summary
