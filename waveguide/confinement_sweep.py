"""
waveguide.confinement_sweep.py

This script allows us to evaluate and visualize spatial mode confinement metrics 
(Confinement Factor and Effective Mode Area) across a geometric sweep.

Author: Divyanshu Tiwari, Esraaj Sarkar Gupta
Date: 2026
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

import tidy3d as td

import geometries
import utilities

if TYPE_CHECKING:
    from .waveguide import RidgeWaveguide

logger = logging.getLogger(__name__)
ConfinementSweepResult = Dict[float, Tuple[List[float], List[float], List[float]]]


def confinement_in_geometry(
        sweep_heights: list[float],
        sweep_widths: list[float],

        core_medium: td.Medium,       
        sub_medium: td.Medium,

        output_dir_name: Path,
        
        wavelength_um: float,
        substrate_height: float,

        preview_geometry: bool = False,
        display_results: bool = True
) -> ConfinementSweepResult:
    """
    Evaluates and visualizes TE00 mode confinement metrics across a combined height and width sweep.
    Generates a unique width-sweep summary figure for each individual height in `sweep_heights`.
    """
    if output_dir_name:
        output_path = Path(output_dir_name)
        output_path.mkdir(parents=True, exist_ok=True)

    if preview_geometry and output_dir_name:
        geom_dir = output_dir_name / Path("geometries/")
        geom_dir.mkdir(parents=True, exist_ok=True)

        mean_height = float(np.mean(sweep_heights))
        mean_width = float(np.mean(sweep_widths))

        logger.info(f"     [Confinement] Plotting mean geometry (Width: {mean_width:.2f} μm, Height: {mean_height:.2f} μm)...")

        preview_waveguide: RidgeWaveguide = geometries.ridge_waveguide.get_passive_simulation(
            width=mean_width,
            height=mean_height,
            length=(5*mean_width + 5*mean_height)/2,
            core_medium=core_medium,
            sub_medium=sub_medium,
            wavelength_um=wavelength_um,
            substrate_height=substrate_height
        )

        utilities.sim_plotter.display_3d(
            device=preview_waveguide,
            output_path=geom_dir / Path("example_geometry.png"),
            title="Ridge Waveguide Confinement Sweep Example Geometry",
            show=display_results,
            save=True
        )

    logger.info(f"Starting confinement sweep across {len(sweep_heights)} heights and {len(sweep_widths)} widths.")

    all_results: ConfinementSweepResult = {}

    for h in sweep_heights:
        results_w: list = []
        results_gamma: list = []
        results_Aeff: list = []

        print(f"    [Confinement] Running geometric width sweep for fixed H = {h:.3f} μm at λ = {wavelength_um:.3f} μm...")

        for w in sweep_widths:
            waveguide_obj: RidgeWaveguide = geometries.ridge_waveguide.get_passive_simulation(
                width=w,
                height=h,
                length=2,
                core_medium=core_medium,
                sub_medium=sub_medium,
                wavelength_um=wavelength_um,
                substrate_height=substrate_height
            )

            # Get confinement metrics for the current waveguide geometry
            conf_data = waveguide_obj.get_confinement()

            # TODO: Implement error handling in the waveguide object.
            if "error" in conf_data:
                print(f"  Width {w:.2f} μm (H={h:.2f} μm): Fundamental TE00 mode not found. Skipping.")
                continue

            results_w.append(w)
            results_gamma.append(conf_data["confinement_percent"])
            results_Aeff.append(conf_data["effective_mode_area_um2"])

            print(f"    [Confinement] Width {w:.2f} μm: Gamma = {conf_data['confinement_percent']:.1f}%, A_eff = {conf_data['effective_mode_area_um2']:.3f} μm²")

        if not results_w:
            logger.warning(f"   [Confinement] No valid modes identified for height H = {h:.3f} μm across all widths.")
            continue

        # Plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=200)
        fig.suptitle(f"TE00 Spatial Confinement Metrics (Fixed H = {h:.3f} μm)", fontsize=14, fontweight='bold', y=1.02)

        ax1.plot(results_w, results_gamma, 's-', color='teal', linewidth=2, markersize=6)
        ax1.set_title("Optical Confinement Factor ($\Gamma$)")
        ax1.set_xlabel("Width (μm)")
        ax1.set_ylabel("Power in Core (%)")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(40, 101)

        ax2.plot(results_w, results_Aeff, 'o-', color='indigo', linewidth=2, markersize=6, label='$A_{eff}$')
        geometric_area = [width * h for width in results_w]
        ax2.plot(results_w, geometric_area, '--', color='gray', label='Physical Core Area')
        ax2.set_title("Effective Mode Area ($A_{eff}$)")
        ax2.set_xlabel("Width (μm)")
        ax2.set_ylabel("Area (μm$^2$)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        fig.tight_layout()

        if output_dir_name:
            path = Path(output_dir_name) / f"confinement_metrics_H{h:.3f}um.png"
            plt.savefig(path, dpi=200, bbox_inches='tight')
            print(f"    --> Saved plot: {path}")

        if display_results:
            plt.show()
        else:
            plt.close(fig)
        
        all_results[h] = (results_w, results_gamma, results_Aeff)
    
    return all_results