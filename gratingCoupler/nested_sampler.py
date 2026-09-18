"""
gratingCoupler/nested_sampler.py

Bayesian Parameter Space Explorer for Grating Couplers using MultiNest Nested Sampling.

Available Search Pipelines:
---------------------------
1. execute_search():
    Traditional nested sampling execution. Runs until statistical evidence
    convergence (or native MultiNest stopping criteria) without budget caps.

2. execute_search_with_budget():
    Budget-aware nested sampling execution. Enforces a strict FlexCredit limit,
    tracks credit usage per evaluation in a persistent ledger, maintains a
    Top-10 leaderboard on disk, and cleanly pauses when credits are exhausted.

Author: Esraaj Sarkar Gupta
Date: 2026

Exposes: BudgetExceededError, GratingCouplerBayesianOptimizer,
GratingCouplerBayesianOptimizer.plot_corner(),
GratingCouplerBayesianOptimizer.prior(),
GratingCouplerBayesianOptimizer.log_likelihood(),
GratingCouplerBayesianOptimizer.report_top_configurations(),
GratingCouplerBayesianOptimizer.execute_search(),
GratingCouplerBayesianOptimizer.execute_search_with_budget().
"""

from tarang import get_logger
from tarang._logging import tidy3d_verbose

import json
import time
import sys
import io
import contextlib
from pathlib import Path
from dataclasses import asdict
import numpy as np

import matplotlib

# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tidy3d as td
from tidy3d import web

from gratingCoupler.grating_coupler import GratingConfig, GratingCoupler
from geometries.grating_coupler import get_ideal_coords
from utilities.sim_plotter import display_2d, display_3d
from utilities.nested_sampling_post_process import post_process_run
from utilities.corner_plot import plot_corner

logger = get_logger(__name__)


class BudgetExceededError(Exception):
    """Raised when the FlexCredit usage limit is reached or exceeded."""

    pass


class GratingCouplerBayesianOptimizer:
    """
    Bayesian parameter space optimizer for SiN/Silicon grating couplers using MultiNest.
    """

    def __init__(
        self,
        fixed_params: dict,
        materials: dict,
        output_dir: Path,
        param_bounds: dict = None,
        sigma: float = 0.02,
        is_3d: bool = False,
        credit_limit: float | None = None,
        show_corner_plots: bool = True,
    ):
        """Initialize search bounds, output directories, and budget tracking.
        Restore previous credit usage when a tracker exists.
        """
        self.fixed_params = fixed_params
        self.materials = materials
        self.output_dir = Path(output_dir)
        self.sigma = sigma
        self.is_3d = is_3d
        self.dimensions = "3D" if is_3d else "2D"
        self.credit_limit = credit_limit
        self.show_corner_plots = show_corner_plots

        self.cache_dir = self.output_dir / "fdtd_cache"
        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.chn_dir = self.output_dir / "chains"
        self.chn_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.basename = str(self.chn_dir / "grating_")

        self.budget_dir = self.output_dir / "budget_cache"
        self.budget_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.tracker_file = self.budget_dir / "tracker.json"

        self.total_credits_used = 0.0
        self.total_evaluations = 0
        self.top_10_runs: list[dict] = []
        self._load_tracker_state()

        self.param_bounds = {
            "period": [0.600, 1.200],
            "fill_factor": [0.20, 0.80],
            "etch_depth": [0.100, 0.700],
            "num_periods": [15.0, 45.0],
            "source_x": [0.00, 25.00],
        }

        if param_bounds is not None:
            logger.info(
                f"[Bayesian Engine] Custom search envelopes registered for {self.dimensions} workspace."
            )
            self.param_bounds.update(param_bounds)

    def plot_corner(
        self,
        *,
        plot_dpi: int = 150,  # Resolution of displayed and saved figures (dots per inch).
        **kwargs,
    ) -> dict:
        """Display a corner plot from the latest files in this optimizer's directory.
        Accept the standalone plot_corner options, including source="live".
        """
        kwargs.setdefault(
            "show",
            self.show_corner_plots,
        )
        return plot_corner(
            self.output_dir,
            plot_dpi=plot_dpi,
            **kwargs,
        )

    def _load_tracker_state(self) -> None:
        """Restore credit usage and the best runs from the saved tracker.
        Log a warning if the existing tracker cannot be parsed.
        """
        if self.tracker_file.exists():
            try:
                with open(
                    self.tracker_file,
                    "r",
                ) as f:
                    data = json.load(f)
                    self.total_credits_used = data.get(
                        "total_credits_used",
                        0.0,
                    )
                    self.total_evaluations = data.get(
                        "total_evaluations",
                        0,
                    )
                    self.top_10_runs = data.get(
                        "top_10_runs",
                        [],
                    )
                    if self.credit_limit is None and "credit_limit" in data:
                        self.credit_limit = data["credit_limit"]
                logger.info(
                    f"[Budget Warden] Resumed state: {self.total_credits_used:.4f} credits spent across {self.total_evaluations} runs."
                )
            except Exception as e:
                logger.warning(f"Failed to parse budget tracker file: {e}")

    def _record_run(
        self,
        config: GratingConfig,
        efficiency: float,
        cost: float,
        dir_name: str,
    ) -> None:
        """Record an evaluation and update the credit ledger and leaderboard.
        Persist the ten highest-efficiency configurations to the tracker file.
        """
        self.total_evaluations += 1
        self.total_credits_used += cost

        entry = {
            "efficiency": float(efficiency),
            "peak_loss_db": float(10.0 * np.log10(efficiency + 1e-12)),
            "cost": float(cost),
            "dir_name": dir_name,
            "config": asdict(config),
        }

        existing_idx = next(
            (
                i
                for i, r in enumerate(self.top_10_runs)
                if r.get("dir_name") == dir_name
            ),
            None,
        )
        if existing_idx is not None:
            if efficiency > self.top_10_runs[existing_idx]["efficiency"]:
                self.top_10_runs[existing_idx] = entry
        else:
            self.top_10_runs.append(entry)

        self.top_10_runs.sort(
            key=lambda x: x["efficiency"],
            reverse=True,
        )
        self.top_10_runs = self.top_10_runs[:10]

        for idx, item in enumerate(
            self.top_10_runs,
            start=1,
        ):
            item["rank"] = idx

        tracker_data = {
            "total_credits_used": self.total_credits_used,
            "credit_limit": self.credit_limit,
            "total_evaluations": self.total_evaluations,
            "top_10_runs": self.top_10_runs,
        }

        with open(
            self.tracker_file,
            "w",
        ) as f:
            json.dump(
                tracker_data,
                f,
                indent=4,
            )

    def prior(
        self,
        cube,
        ndim,
        nparams,
    ):
        """Transform a unit-cube sample into the configured physical bounds.
        Modify the MultiNest cube in place and constrain etching to the core depth.
        """
        b = self.param_bounds["period"]
        cube[0] = b[0] + cube[0] * (b[1] - b[0])

        b = self.param_bounds["fill_factor"]
        cube[1] = b[0] + cube[1] * (b[1] - b[0])

        b = self.param_bounds["etch_depth"]
        max_allowable_etch = min(
            b[1],
            self.fixed_params["core_height"] - 0.05,
        )
        cube[2] = b[0] + cube[2] * (max_allowable_etch - b[0])

        b = self.param_bounds["num_periods"]
        cube[3] = b[0] + cube[3] * (b[1] - b[0])

        b = self.param_bounds["source_x"]
        cube[4] = b[0] + cube[4] * (b[1] - b[0])

    def log_likelihood(
        self,
        cube,
        ndim,
        nparams,
    ):
        """Evaluate a sampled grating configuration and its likelihood.
        Reuse cached simulations and enforce the configured cloud-credit budget.
        """
        p_period = float(cube[0])
        p_dc = float(cube[1])
        p_etch = float(cube[2])
        p_num_p = int(np.round(float(cube[3])))
        p_src_x = float(cube[4])

        try:
            config = GratingConfig(
                wavelength_um=self.fixed_params["wavelength0_um"],
                theta_deg=self.fixed_params["theta_deg"],
                mfd=self.fixed_params["mfd"],
                source_gap=self.fixed_params["source_gap"],
                core_height=self.fixed_params["core_height"],
                cladding_height=self.fixed_params["cladding_height"],
                box_height=self.fixed_params["box_height"],
                substrate_height=self.fixed_params["substrate_height"],
                w_grating=self.fixed_params["w_grating"],
                buffer=self.fixed_params["buffer"],
                period=p_period,
                ff=p_dc,
                etch_depth=p_etch,
                n_teeth=p_num_p,
                source_x=p_src_x,
                pol_angle=self.fixed_params.get(
                    "pol_angle",
                    np.pi / 2.0,
                ),
            )

            sim_wrapper = GratingCoupler.from_config(
                config=config,
                core_medium=self.materials.get("core_medium"),
                box_medium=self.materials.get("box_medium"),
                substrate_medium=self.materials.get("sub_medium"),
                cladding_medium=self.materials.get("cladding_medium"),
                is_3d=self.is_3d,
                fine_steps_per_wvl=self.fixed_params.get(
                    "fine_steps",
                    15,
                ),
                coarse_steps_per_wvl=self.fixed_params.get(
                    "coarse_steps",
                    10,
                ),
            )

            dir_name = sim_wrapper.get_dir_name()
            target_hdf5 = self.cache_dir / dir_name / "raw_simulation_data.hdf5"

            if target_hdf5.exists():
                cost = 0.0
            else:
                with contextlib.redirect_stdout(
                    io.StringIO()
                ), contextlib.redirect_stderr(io.StringIO()):
                    task_id = web.upload(
                        sim_wrapper.sim,
                        task_name=f"gc_eval_{dir_name[:25]}",
                        verbose=tidy3d_verbose(False),
                    )
                    cost = float(
                        web.estimate_cost(
                            task_id,
                            verbose=tidy3d_verbose(),
                        )
                    )

                if (
                    self.credit_limit is not None
                    and (self.total_credits_used + cost) > self.credit_limit
                ):
                    logger.warning(
                        f"\n[Budget Warden] Credit limit reached! Next run needs {cost:.4f}, but only "
                        f"{(self.credit_limit - self.total_credits_used):.4f} credits remain."
                    )
                    raise BudgetExceededError(
                        f"FlexCredit limit of {self.credit_limit} exceeded! Current spent: {self.total_credits_used:.4f} credits."
                    )

            efficiency = sim_wrapper.get_efficiency(
                output_dir=self.cache_dir,
                verbose=False,
            )

            logger.info(
                f"[FDTD Run Audit] Period: {p_period:.4f} μm | FF: {p_dc:.4f} | "
                f"Etch: {p_etch:.4f} μm | Teeth: {p_num_p} | Source X: {p_src_x:.4f} μm "
                f"==> Efficiency: {efficiency * 100.0:.3f}%"
            )

            self._record_run(
                config=config,
                efficiency=efficiency,
                cost=cost,
                dir_name=dir_name,
            )

            credit_str = (
                f"{self.credit_limit:.3f}"
                if self.credit_limit is not None
                else "Unlimited"
            )
            logger.info(
                f"Nested Sampling -> Efficiency: {efficiency*100:.3f}% | Spent: {self.total_credits_used:.3f}/{credit_str} credits"
            )

        except BudgetExceededError:
            raise
        except Exception as e:
            logger.error(f"\n[CRITICAL FAILURE] FDTD engine crashed. Error: {e}")
            return -1e10

        ln_L = -0.5 * ((1.0 - efficiency) / self.sigma) ** 2
        return ln_L

    def report_top_configurations(self) -> list[dict]:
        """Log and export the ten best recorded grating configurations.
        Return the ranked configuration records for further analysis.
        """
        report_file = self.output_dir / "top_10_configurations.json"

        summary = {
            "total_credits_used": self.total_credits_used,
            "credit_limit": self.credit_limit,
            "total_evaluations": self.total_evaluations,
            "top_10_candidates": self.top_10_runs,
        }

        with open(
            report_file,
            "w",
        ) as f:
            json.dump(
                summary,
                f,
                indent=4,
            )

        logger.info(f"\n{'='*75}")
        logger.info(
            f"[TOP 10 GRATING COUPLER CONFIGURATIONS] (Spent: {self.total_credits_used:.4f} FlexCredits)"
        )
        logger.info(
            f"{'Rank':<5} | {'Efficiency':<10} | {'Loss (dB)':<10} | {'Period (um)':<11} | {'FF':<6} | {'Etch (um)':<9} | {'Teeth':<6}"
        )
        logger.info("-" * 75)

        for item in self.top_10_runs:
            cfg = item["config"]
            rank = item.get(
                "rank",
                "-",
            )
            logger.info(
                f"{rank:<5} | "
                f"{item['efficiency']*100.0:>8.2f}% | "
                f"{item['peak_loss_db']:>8.2f} dB | "
                f"{cfg['period']:>10.4f} | "
                f"{cfg['ff']:>5.2f} | "
                f"{cfg['etch_depth']:>8.4f} | "
                f"{cfg['n_teeth']:>5}"
            )
        logger.info(f"{'='*75}")
        logger.info(f"Full dataclass parameters exported to: {report_file.name}\n")

        return self.top_10_runs

    def execute_search(
        self,
        n_live_points: int = 150,
        evidence_tolerance: float = 0.3,
        *,
        plot_dpi: int = 150,  # Resolution of displayed and saved figures (dots per inch).
    ) -> dict:
        """Execute an unbudgeted MultiNest search or load existing results.
        Post-process completed chains into the returned summary dictionary.
        """
        self.credit_limit = None

        result_file = self.output_dir / "bayesian_optimum_results.json"
        if result_file.exists():
            logger.info(
                f"\n[Bayesian Engine] Results already exist at {result_file.name}. Bypassing run."
            )
            with open(
                result_file,
                "r",
            ) as f:
                return json.load(f)

        logger.info(
            f"\n[Bayesian Engine] Launching standard MultiNest space extraction ({self.dimensions})..."
        )

        import pymultinest

        pymultinest.run(
            LogLikelihood=self.log_likelihood,
            Prior=self.prior,
            n_dims=5,
            n_params=5,
            outputfiles_basename=self.basename,
            n_live_points=n_live_points,
            evidence_tolerance=evidence_tolerance,
            importance_nested_sampling=False,
            multimodal=True,
            resume=True,
            verbose=False,
        )

        return self._post_process_results(
            result_file=result_file,
            plot_dpi=plot_dpi,
        )

    def execute_search_with_budget(
        self,
        credit_limit: float,
        n_live_points: int = 150,
        evidence_tolerance: float = 0.3,
        *,
        plot_dpi: int = 150,  # Resolution of displayed and saved figures (dots per inch).
    ) -> dict:
        """Run or resume MultiNest within the supplied credit limit.
        Save progress and return a paused status when the budget is exhausted.
        """
        self.credit_limit = credit_limit

        result_file = self.output_dir / "bayesian_optimum_results.json"
        if result_file.exists():
            logger.info(
                f"\n[Bayesian Engine] Results already exist at {result_file.name}. Bypassing run."
            )
            with open(
                result_file,
                "r",
            ) as f:
                return json.load(f)

        logger.info(
            f"\n[Bayesian Engine] Launching budget-aware MultiNest space extraction ({self.dimensions})..."
        )
        logger.info(
            f"[Budget Warden] Active Cap: {self.credit_limit:.4f} FlexCredits | Spent So Far: {self.total_credits_used:.4f} FlexCredits"
        )

        import pymultinest

        budget_halted = False
        try:
            pymultinest.run(
                LogLikelihood=self.log_likelihood,
                Prior=self.prior,
                n_dims=5,
                n_params=5,
                outputfiles_basename=self.basename,
                n_live_points=n_live_points,
                evidence_tolerance=evidence_tolerance,
                importance_nested_sampling=False,
                multimodal=True,
                resume=True,
                verbose=False,
            )
        except BudgetExceededError as e:
            budget_halted = True
            logger.warning(f"\n[Budget Warden] Execution paused cleanly: {e}")
            logger.info(
                "[Budget Warden] MultiNest chains and credit ledger are saved. Pass a higher limit to resume."
            )

        self.report_top_configurations()

        if budget_halted:
            summary = post_process_run(
                output_dir=self.output_dir,
                sigma=self.sigma,
                plot_filename="bayesian_structural_corner_plot_paused.png",
                show=self.show_corner_plots,
                plot_dpi=plot_dpi,
            )
            summary["status"] = "PAUSED_BUDGET_EXCEEDED"
            summary["credit_limit"] = self.credit_limit
            return summary

        return self._post_process_results(
            result_file=result_file,
            plot_dpi=plot_dpi,
        )

    def _post_process_results(
        self,
        result_file: Path,
        *,
        plot_dpi: int = 150,  # Resolution of displayed and saved figures (dots per inch).
    ) -> dict:
        """Summarize posterior chains and export the best grating geometry.
        Save the final result dictionary and return it to the caller.
        """
        self.report_top_configurations()
        logger.info(
            "[Bayesian Engine] Delegating chain parsing and corner plot to PostProcessor..."
        )

        summary = post_process_run(
            output_dir=self.output_dir,
            sigma=self.sigma,
            plot_filename="bayesian_structural_corner_plot.png",
            show=self.show_corner_plots,
            plot_dpi=plot_dpi,
        )

        if summary["status"] != "RECOVERED":
            logger.warning(
                "Search ended without a usable posterior summary: %s",
                summary["status"],
            )
            return summary

        space_variance = summary.get(
            "parameter_space_variance",
            {},
        )
        required_keys = [
            "period",
            "fill_factor",
            "etch_depth",
            "num_periods",
            "source_x",
        ]

        if space_variance and all(k in space_variance for k in required_keys):
            try:
                opt_config = GratingConfig(
                    wavelength_um=self.fixed_params["wavelength0_um"],
                    theta_deg=self.fixed_params["theta_deg"],
                    mfd=self.fixed_params["mfd"],
                    source_gap=self.fixed_params["source_gap"],
                    core_height=self.fixed_params["core_height"],
                    cladding_height=self.fixed_params["cladding_height"],
                    box_height=self.fixed_params["box_height"],
                    substrate_height=self.fixed_params["substrate_height"],
                    w_grating=self.fixed_params["w_grating"],
                    buffer=self.fixed_params["buffer"],
                    period=space_variance["period"]["marginalized_mean"],
                    ff=space_variance["fill_factor"]["marginalized_mean"],
                    etch_depth=space_variance["etch_depth"]["marginalized_mean"],
                    n_teeth=int(
                        np.round(space_variance["num_periods"]["marginalized_mean"])
                    ),
                    source_x=space_variance["source_x"]["marginalized_mean"],
                    pol_angle=self.fixed_params.get(
                        "pol_angle",
                        np.pi / 2.0,
                    ),
                )

                verification_wrapper = GratingCoupler.from_config(
                    config=opt_config,
                    core_medium=self.materials.get("core_medium"),
                    box_medium=self.materials.get("box_medium"),
                    substrate_medium=self.materials.get("sub_medium"),
                    cladding_medium=self.materials.get("cladding_medium"),
                    is_3d=self.is_3d,
                    fine_steps_per_wvl=25,
                    coarse_steps_per_wvl=15,
                )

                x_c, y_c, z_c = get_ideal_coords(verification_wrapper.sim)
                max_efficiency = summary.get(
                    "max_efficiency",
                    0.0,
                )

                if self.is_3d:
                    display_3d(
                        device=verification_wrapper,
                        output_path=self.output_dir
                        / "geometry_3d_opt_verification.png",
                        title=f"3D Grating Audit (Posterior Means)\nEst. Max $\\eta$: {max_efficiency*100.0:.2f}%",
                        coordinates=(x_c, y_c, z_c),
                        save=True,
                        plot_dpi=plot_dpi,
                    )
                else:
                    display_2d(
                        device=verification_wrapper,
                        output_path=self.output_dir
                        / "geometry_2d_opt_verification.png",
                        title=f"2D Grating Audit (Posterior Means)\nEst. Max $\\eta$: {max_efficiency*100.0:.2f}%",
                        slice_axis="y",
                        slice_coord=y_c,
                        save=True,
                        plot_dpi=plot_dpi,
                    )
            except Exception as e:
                logger.warning(
                    f"[Warning] Failed to generate FDTD geometry verification plot: {e}"
                )

        summary["status"] = "CONVERGED"
        with open(
            result_file,
            "w",
        ) as f:
            json.dump(
                summary,
                f,
                indent=4,
            )

        return summary
