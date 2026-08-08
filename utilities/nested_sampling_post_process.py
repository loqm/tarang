"""
utilities.nested_sampling_post_process.py

Standalone Post-Processing and Visualizer for MultiNest Grating Coupler Runs.

Provides post_process_run() to parse output directories, recover raw chains from 
completed or manually killed runs, render GetDist corner plots, and extract 
posterior summary statistics as a dictionary.

Author: Esraaj Sarkar Gupta
Date: 2026
"""

import json
import logging
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pymultinest
from getdist import mcsamples, plots

logger = logging.getLogger(__name__)

DEFAULT_PARAM_NAMES = ['period', 'fill_factor', 'etch_depth', 'num_periods', 'source_x']
DEFAULT_PARAM_LABELS = [
    r'\Lambda\ (\mu\mathrm{m})', 
    r'\mathrm{FF}', 
    r't_{\mathrm{etch}}\ (\mu\mathrm{m})', 
    r'N', 
    r'X_{\mathrm{src}}\ (\mu\mathrm{m})'
]


def post_process_run(
    output_dir: Path | str,
    param_names: list[str] = None,
    param_labels: list[str] = None,
    sigma: float = 0.02,
    plot_filename: str = "bayesian_structural_corner_plot.png"
) -> dict:
    """
    Parses a MultiNest output directory and extracts posterior results.
    
    Generates a GetDist triangle/corner plot and returns a results dictionary,
    guaranteeing standard key schema even if the run was killed or incomplete.
    """
    out_path = Path(output_dir)
    if not out_path.exists():
        raise FileNotFoundError(f"Specified output directory does not exist: {out_path}")

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
        "error": None
    }

    # 1. Recover Tracker / Budget Metadata if present
    tracker_file = out_path / "budget_cache" / "tracker.json"
    top_10_file = out_path / "top_10_configurations.json"

    if tracker_file.exists():
        try:
            with open(tracker_file, "r") as f:
                t_data = json.load(f)
                results_summary["total_credits_used"] = float(t_data.get("total_credits_used", 0.0))
                results_summary["total_evaluations"] = int(t_data.get("total_evaluations", 0))
                results_summary["top_candidates"] = t_data.get("top_10_runs", [])
        except Exception as e:
            logger.warning(f"Could not load tracker file: {e}")

    if not results_summary["top_candidates"] and top_10_file.exists():
        try:
            with open(top_10_file, "r") as f:
                s_data = json.load(f)
                results_summary["top_candidates"] = s_data.get("top_10_candidates", [])
        except Exception as e:
            logger.warning(f"Could not load top 10 file: {e}")

    # Pre-seed metrics from leaderboard if available (useful if chains are missing/empty)
    if results_summary["top_candidates"]:
        best_cand = results_summary["top_candidates"][0]
        results_summary["max_efficiency"] = float(best_cand.get("efficiency", 0.0))
        results_summary["peak_loss_db"] = float(best_cand.get("peak_loss_db", -120.0))
        cfg = best_cand.get("config", {})
        if cfg:
            results_summary["optimal_vector"] = [
                float(cfg.get("period", 0.0)),
                float(cfg.get("ff", 0.0)),
                float(cfg.get("etch_depth", 0.0)),
                float(cfg.get("n_teeth", 0.0)),
                float(cfg.get("source_x", 0.0))
            ]

    # 2. Locate Chain Files with fallback logic
    chn_dir = out_path / "chains"
    basename = str(chn_dir / "grating_") if chn_dir.exists() else str(out_path / "grating_")
    
    candidate_files = [
        Path(f"{basename}.txt"),
        Path(f"{basename}_post_equal_weights.dat"),
        Path(f"{basename}_live.points")
    ]
    
    chain_file = next((f for f in candidate_files if f.exists() and f.stat().st_size > 0), None)

    if chain_file is None:
        print(f"[PostProcessor] Warning: No non-empty chain files found in {out_path}")
        results_summary["status"] = "NO_CHAIN_DATA_FOUND"
        results_summary["error"] = "Missing or empty chain files."
        return results_summary

    # 3. Load Raw Chain Data
    try:
        raw_chain_data = np.loadtxt(chain_file)
        if raw_chain_data.ndim == 1 and raw_chain_data.size > 0:
            raw_chain_data = raw_chain_data.reshape(1, -1)

        if raw_chain_data.ndim < 2 or raw_chain_data.shape[0] < 5:
            results_summary["status"] = "INSUFFICIENT_SAMPLES"
            results_summary["total_samples_collected"] = int(raw_chain_data.shape[0]) if raw_chain_data.ndim > 0 else 0
            results_summary["error"] = f"Only {results_summary['total_samples_collected']} samples collected; GetDist requires >= 5."
            return results_summary
    except Exception as e:
        results_summary["status"] = "CHAIN_READ_ERROR"
        results_summary["error"] = str(e)
        return results_summary

    weights = raw_chain_data[:, 0]
    loglikes = -0.5 * raw_chain_data[:, 1]
    samples_matrix = raw_chain_data[:, 2:]
    results_summary["total_samples_collected"] = int(samples_matrix.shape[0])

    # 4. Generate GetDist Corner Plot
    corner_plot_path = out_path / plot_filename
    plot_generated = False
    
    try:
        gd_samples = mcsamples.MCSamples(
            samples=samples_matrix,
            weights=weights,
            loglikes=loglikes,
            names=param_names,
            labels=param_labels,
            name_tag="GratingCoupler"
        )

        g = plots.get_subplot_plotter(width_inch=10)
        g.settings.num_plot_contours = 2
        g.triangle_plot(gd_samples, filled=True, cmap="viridis")
        g.export(str(corner_plot_path))
        plot_generated = True
        results_summary["corner_plot_path"] = str(corner_plot_path)
        print(f"[PostProcessor] Corner plot rendered: {corner_plot_path.name}")
    except Exception as e:
        print(f"[PostProcessor] GetDist plot generation failed: {e}")

    # 5. Extract Marginal Posterior Statistics
    space_variance_measures = {}
    try:
        marge_stats = gd_samples.getMargeStats()
        for name in param_names:
            p_stat = marge_stats.parWithName(name)
            space_variance_measures[name] = {
                "marginalized_mean": float(p_stat.mean),
                "standard_deviation": float(p_stat.err)
            }
        results_summary["parameter_space_variance"] = space_variance_measures
    except Exception as e:
        logger.warning(f"Failed to compute marginal statistics: {e}")

    # 6. Isolate Maximum Likelihood Coordinates
    max_lnL = float(np.max(loglikes))
    try:
        analyzer = pymultinest.Analyzer(n_params=len(param_names), outputfiles_basename=basename)
        best_stats = analyzer.get_best_fit()
        results_summary["optimal_vector"] = [float(x) for x in best_stats['parameters']]
        max_lnL = float(best_stats['log_likelihood'])
    except Exception:
        best_idx = np.argmax(loglikes)
        results_summary["optimal_vector"] = [float(x) for x in samples_matrix[best_idx, :len(param_names)]]

    if max_lnL <= 0:
        results_summary["max_efficiency"] = float(1.0 - (sigma * np.sqrt(-2.0 * max_lnL)))
        results_summary["peak_loss_db"] = float(10.0 * np.log10(results_summary["max_efficiency"] + 1e-12))

    results_summary["status"] = "RECOVERED"

    # Save recovered summary to disk
    parsed_json_file = out_path / "post_processed_results.json"
    with open(parsed_json_file, "w") as f:
        json.dump(results_summary, f, indent=4)

    return results_summary