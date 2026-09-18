"""
edge_coupler.optimization_sweep.py

Defines the 2D sweep (taper length & tip width) for optimal
edge coupler in-coupling/ out-coupling

Authors: Esraaj Sarkar Gupta, Divyanshu Tiwari
Date: 2026

Exposes: parametric_optimization_sweep().
"""

from tarang import get_logger
from utilities.validation import positive as _positive, values as _values
from tarang._logging import tidy3d_verbose

import tidy3d as td
from tidy3d import web
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from pathlib import Path

from media import get_medium

logger = get_logger(__name__)


def parametric_optimization_sweep(
    sweep_taper_lengths: list[float],
    sweep_tip_widths: list[float],
    output_dir: Path | str,
    wg_width: float,
    wg_height: float,
    core_medium: td.Medium,
    substrate_medium: td.Medium,
    wavelength0: float,
    top_cladding_medium: td.Medium = get_medium("air"),
    substrate_height: float = 4.0,
    top_cladding_height: float = 0.0,
    fine_steps_per_wvl: int = 20,
    coarse_steps_per_wvl: int = 15,
    settling_buffer_length: Optional[float] = None,
    in_coupling: Optional[bool] = None,
    out_coupling: Optional[bool] = None,
    batching: bool = True,  # Run the sweep simulations in a single batch?
    show: bool = True,  # Display the efficiency matrix?
    save: bool = True,
    *,
    plot_dpi: int = 300,  # Resolution of displayed and saved figures (dots per inch).
) -> np.ndarray:
    """
    This function carries out a parametric sweep of edge coupler taper lengths and
    tip widths.
    """

    sweep_taper_lengths = _values("sweep_taper_lengths", sweep_taper_lengths)
    sweep_tip_widths = _values("sweep_tip_widths", sweep_tip_widths)

    wg_width = _positive("wg_width", wg_width)
    wg_height = _positive("wg_height", wg_height)
    wavelength0 = _positive("wavelength0", wavelength0)
    substrate_height = _positive("substrate_height", substrate_height)

    fine_steps_per_wvl = int(_positive("fine_steps_per_wvl", fine_steps_per_wvl))
    coarse_steps_per_wvl = int(_positive("coarse_steps_per_wvl", coarse_steps_per_wvl))
    plot_dpi = int(_positive("plot_dpi", plot_dpi))

    top_cladding_height = _positive("top_cladding_height", top_cladding_height, allow_zero=True)
    if settling_buffer_length is not None:
        settling_buffer_length = _positive("settling_buffer_length", settling_buffer_length, allow_zero=True)
        
    
    # Local import to prevent circular dependency with geometries.edge_coupler
    from geometries.edge_coupler import get_active_simulation

    if (in_coupling is None) and (out_coupling is None):
        raise ValueError(
            "A coupling type must be specified for a sweep! "
            "Declare either argument out_coupling or in_coupling to be True."
        )

    if in_coupling is not None and out_coupling is not None and in_coupling == out_coupling:
        raise ValueError("Exactly one of in_coupling or out_coupling must be True.")

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Define parameter space and instantiate simulations
    couplers: list = list([])
    sim_dict: dict = dict({})

    logger.info(
        f"Initializing {len(sweep_taper_lengths) * len(sweep_tip_widths)} simulations for the sweep..."
    )
    for L in sweep_taper_lengths:
        row = []
        for W in sweep_tip_widths:
            ec = get_active_simulation(
                tip_width=W,
                taper_length=L,
                wg_width=wg_width,
                wg_height=wg_height,
                core_medium=core_medium,
                substrate_medium=substrate_medium,
                wavelength0=wavelength0,
                top_cladding_medium=top_cladding_medium,
                substrate_height=substrate_height,
                top_cladding_height=top_cladding_height,
                fine_steps_per_wvl=fine_steps_per_wvl,
                coarse_steps_per_wvl=coarse_steps_per_wvl,
                settling_buffer_length=settling_buffer_length,
                in_coupling=in_coupling,
                out_coupling=out_coupling,
            )
            row.append(ec)

            # Map for batching: Use the task name the wrapper expects
            task_name = f"edge_coupler_{ec.get_dir_name()}"
            sim_dict[task_name] = ec.sim

        couplers.append(row)

    # Carry out the simulations
    if batching:
        logger.info("Submitting batch to Tidy3D cloud infrastructure...")
        batch = web.Batch(
            simulations=sim_dict,
            verbose=tidy3d_verbose(True),
        )
        batch_results = batch.run(path_dir=output_dir / "batch_downloads")

        # Distribute downloaded data to cache locations expected by the wrapper
        for task_name, sim_data in batch_results.items():
            param_dir = task_name.replace(
                "edge_coupler_",
                "",
            )
            target_path = output_dir / param_dir / "raw_simulation_data.hdf5"
            target_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if not target_path.exists():
                sim_data.to_file(str(target_path))

    # Extract efficiencies
    efficiencies = np.zeros((len(sweep_taper_lengths), len(sweep_tip_widths)))
    for i, L in enumerate(sweep_taper_lengths):
        for j, W in enumerate(sweep_tip_widths):
            try:
                efficiencies[i, j] = couplers[i][j].get_efficiency(
                    output_dir=output_dir
                )
            except RuntimeError as e:
                logger.warning(
                    f"Warning: Failed to extract efficiency for L={L}, W={W}: {e}"
                )
                efficiencies[i, j] = 0.0

    # Generate efficiency matrix visualization
    if show or save:
        eff_pct = efficiencies * 100.0
        fig, ax = plt.subplots(
            figsize=(8, 6),
            dpi=plot_dpi,
        )

        im = ax.imshow(
            eff_pct,
            cmap="magma",
            origin="lower",
            aspect="auto",
        )
        cbar = fig.colorbar(
            im,
            ax=ax,
        )
        cbar.set_label(
            "Coupling Efficiency (%)",
            rotation=270,
            labelpad=15,
        )

        # Set tick labels to exact parameter sweep values
        ax.set_xticks(np.arange(len(sweep_tip_widths)))
        ax.set_yticks(np.arange(len(sweep_taper_lengths)))
        ax.set_xticklabels([f"{w:.3f}" for w in sweep_tip_widths])
        ax.set_yticklabels([f"{l:.1f}" for l in sweep_taper_lengths])

        # Annotate matrix cells with percentage values
        vmax = np.max(eff_pct) if np.max(eff_pct) > 0 else 1.0
        for i in range(len(sweep_taper_lengths)):
            for j in range(len(sweep_tip_widths)):
                val = eff_pct[i, j]
                text_color = "black" if val > (0.6 * vmax) else "white"
                ax.text(
                    j,
                    i,
                    f"{val:.1f}%",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontweight="bold",
                    fontsize=9,
                )

        mode_label = "In-Coupling" if in_coupling else "Out-Coupling"
        ax.set_title(f"Edge Coupler Efficiency Matrix ({mode_label})")
        ax.set_xlabel("Tip Width (μm)")
        ax.set_ylabel("Taper Length (μm)")
        fig.tight_layout()

        if save:
            save_path = output_dir / f"efficiency_matrix_{mode_label.lower()}.png"
            plt.savefig(
                save_path,
                dpi=plot_dpi,
            )
            logger.info(f"Matrix plot saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    # Return efficiency matrix
    return efficiencies
