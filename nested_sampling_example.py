"""Budgeted grating-coupler search using Tarang's material library.

The optimizer exposes ``execute_search_with_budget`` for a cumulative credit
cap, ``execute_search`` for an uncapped search, ``report_top_configurations``
for the recorded leaderboard, and ``plot_corner`` for saved-chain diagnostics.
Its ``prior`` and ``log_likelihood`` methods are nested-sampling callbacks
(MultiNest convention); calling the latter directly can submit a paid
simulation. See their docstrings in ``grating_coupler.nested_sampler`` for
inputs, outputs, and side effects.

Sampler backend: ``backend="auto"`` uses MultiNest where PyMultiNest and its
native library are installed (Linux/macOS) and dynesty otherwise (installed
with TARANG; the only option on Windows). Pass ``backend="dynesty"`` or ``"multinest"`` to
choose explicitly. Do not switch backends within one output directory.

The geometry, priors, and sampler settings retain the original example values.
Library-fitted Si3N4 and SiO2 replace the constant-index approximations, so
efficiencies can change. Cached FDTD results are reused only when the stored
simulation (geometry, materials, mesh) matches exactly; see
``utilities.cloud_run``.

Exposes: main().
"""

from tarang import get_logger

from pathlib import Path
import numpy as np

logger = get_logger(__name__)


def main(
    *,
    plot_dpi: int = 150,  # Resolution of generated figures (dots per inch).
):
    """Run or resume the 2D search, report candidates, then plot saved chains.

    ``plot_dpi`` controls figure resolution only; it does not affect the mesh
    or sampling. The run uses 25 live points, evidence tolerance 0.5, sigma
    0.02, and a cumulative 0.5-FlexCredit cap including the saved ledger.
    Material fits are loaded or computed locally by ``media.get_medium``;
    uncached FDTD evaluations submit cloud jobs and can consume credits.

    Outputs go to ``example_results/bayesian_demo_budgeted`` relative to the
    working directory. Existing final results bypass sampling; compatible
    unfinished runs resume from the saved chains (MultiNest resume file, or a
    dynesty replay from the stored seed). The final corner plot
    prefers posterior chains and falls back to live points, which are only
    diagnostics, not a converged posterior. Missing or insufficient samples
    and plotting failures are logged without discarding the run's reports.
    Figures are saved and displayed; an interactive display may block until
    closed. Returns None.
    """
    from media import get_medium
    from grating_coupler.nested_sampler import GratingCouplerBayesianOptimizer
    from tarang import configure_logging, set_tidy3d_output

    configure_logging()
    set_tidy3d_output(False)
    # Restore the previous Tidy3D output settings with set_tidy3d_output(True).
    # Load Tarang's registered optical data with the helper's default fit/cache
    # settings. Air remains dispersionless; solids use their tabulated data.
    SiN = get_medium("si3n4")
    SiO2 = get_medium("sio2")
    Air = get_medium("air")

    # Package Fixed Physical & Domain Parameters
    # These map directly to the GratingConfig dataclass inside the sampler
    fixed_params = {
        "wavelength_um": 1.550,  # Central excitation wavelength (μm)
        "theta_deg": -12.0,  # Fiber angle off-vertical (degrees)
        "mfd": 10.4,  # Mode Field Diameter of fiber source (μm)
        "source_gap": 2.0,  # Gap above cladding for source plane (μm)

        # Layer Heights
        "core_height": 0.800,  # Silicon Nitride core thickness (μm)
        "top_cladding_height": 2.0,  # Cladding height buffer (μm)
        "box_height": 4.0,  # Buried Oxide (BOX) thickness (μm)
        "handle_height": 8.0,  # Silicon handle substrate thickness (μm)

        # 3D Domain & Numerical Resolution
        "w_grating": 10.4,  # Transverse width for 3D runs (μm)
        "buffer": 3.0,  # Spatial buffer margin around domain (μm)
        "pol_angle": np.pi / 2.0,  # TE polarization (90 degrees)
        "fine_steps": 15,  # Grid steps per wavelength in fine mesh region
        "coarse_steps": 10,  # Grid steps per wavelength in background mesh
    }

    # Package Material Media Dictionary
    materials = {
        "core_medium": SiN,
        "box_medium": SiO2,
        "handle_medium": None,  # Terminated at BOX layer (set Si medium to include handle)
        "top_cladding_medium": Air,  # Cladding medium (Air or SiO2)
    }

    # Define Tight Prior Search Envelopes Around Design Point
    # Narrow priors focus the search but do not guarantee high efficiency.
    tight_param_bounds = {
        "period": [0.800, 0.840],  # Centered near Lambda = 0.8201 μm
        "fill_factor": [0.480, 0.550],  # Centered near Duty Cycle = 0.5136
        "etch_depth": [0.480, 0.550],  # Centered near Etch Depth = 0.5136 μm
        "num_periods": [18.0, 26.0],  # Centered near N = 22 teeth
        "source_x": [3.500, 4.200],  # Centered near Fiber X = 3.8419 μm
    }

    # Output directory for chains, budget ledger, and leaderboard
    demo_2_out = Path("example_results/bayesian_demo_budgeted")

    # Initialize the Budget-Aware Optimizer Engine
    optimizer_budgeted = GratingCouplerBayesianOptimizer(
        fixed_params=fixed_params,
        materials=materials,
        output_dir=demo_2_out,
        param_bounds=tight_param_bounds,
        sigma=0.02,
        is_3d=False,
        credit_limit=0.5,
    )

    # Execute Budget-Capped Nested Sampling
    # Checks estimated cost after upload, before running an uncached simulation.
    results_budgeted = optimizer_budgeted.execute_search_with_budget(
        credit_limit=0.5,  # Hard cap: Max FlexCredits allowed for this entire run session
        n_live_points=25,  # Number of live points
        evidence_tolerance=0.5,
        backend="auto",  # MultiNest if installed, else dynesty (Windows)
        plot_dpi=plot_dpi,
    )

    # Check run termination status
    if results_budgeted.get("status") == "PAUSED_BUDGET_EXCEEDED":
        logger.warning(
            f"\n[Run Paused] Spent {results_budgeted['total_credits_used']:.4f} / {results_budgeted['credit_limit']:.4f} credits."
        )
        logger.info(
            "Pass a higher credit_limit to this directory to resume the search."
        )
    elif results_budgeted.get("status") == "CONVERGED":
        logger.info(
            f"\n[Run Converged] Max Efficiency: {results_budgeted['max_efficiency']*100:.2f}%"
        )
    else:
        logger.warning("Run ended with status: %s", results_budgeted.get("status"))

    # Print the Top 10 configurations collected before hitting the budget cap
    optimizer_budgeted.report_top_configurations()

    # Explicitly plot at the end, including when sampling loaded saved results.
    try:
        optimizer_budgeted.plot_corner(source="auto", plot_dpi=plot_dpi)
    except (ImportError, ValueError, RuntimeError, OSError) as exc:
        logger.warning("Final corner plot could not be generated: %s", exc)

    # For an ongoing run, use this in a separate notebook/kernel (no optimizer needed):
    # from utilities import plot_corner (see grating_coupler_example.ipynb)
    # plot_corner(
    #     directory=demo_2_out,
    #     source="live",
    #     show=True,
    # )


if __name__ == "__main__":
    main()
