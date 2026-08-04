"""
waveguide.dispersion.py

This script allows us to visualize the different regimes of dispersion
within the waveguide, either by varying the geometry, or by varying
the wavelength.

Author: Esraaj Sarkar Gupta, Divyanshu Tiwari
Date: 2026
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import tidy3d as td

import geometries
import utilities

if TYPE_CHECKING:
    from .waveguide import RidgeWaveguide

logger = logging.getLogger(__name__)
DispersionSweepResult = Dict[float, Tuple[List[float], List[float], List[bool]]]


def dispersion_in_geometry(
        sweep_heights: list[float],
        sweep_widths: list[float],

        core_medium: td.Medium,
        sub_medium: td.Medium,

        output_dir_name: Path,

        wavelength_um: float,
        substrate_height: float,

        preview_geometry: bool = False,
        display_results: bool = True
) -> DispersionSweepResult:
    """
    Evaluates and visualizes Chromatic Dispersion (D) and Single-Mode boundaries 
    across a combined height and width sweep.
    """
    if output_dir_name:
        output_path = Path(output_dir_name)
        output_path.mkdir(parents=True, exist_ok=True)

    if preview_geometry and output_dir_name:
        geom_dir = output_dir_name / Path("geometries")
        geom_dir.mkdir(parents=True, exist_ok=True)

        mean_height = float(np.mean(sweep_heights))
        mean_width = float(np.mean(sweep_widths))

        logger.info(f"     [Dispersion] Plotting mean geometry (Width: {mean_width:.2f} μm, Height: {mean_height:.2f} μm)...")

        preview_waveguide: RidgeWaveguide = geometries.ridge_waveguide.get_passive_simulation(
            width=mean_width,
            height=mean_height,
            length=(5 * mean_width + 5 * mean_height) / 2,
            core_medium=core_medium,
            sub_medium=sub_medium,
            wavelength_um=wavelength_um,
            substrate_height=substrate_height
        )

        utilities.sim_plotter.display_3d(
            device=preview_waveguide,
            output_path=geom_dir / Path("example_dispersion_geometry.png"),
            title="Ridge Waveguide Dispersion Sweep Example Geometry",
            show=display_results,
            save=True
        )

    logger.info(f"Starting dispersion sweep across {len(sweep_widths)} widths and {len(sweep_heights)} heights.")

    all_results: DispersionSweepResult = {}

    for w in sweep_widths:
        results_h: list = []
        results_D: list = []
        results_is_singlemode: list = []

        print(f"    [Dispersion] Sweeping heights for fixed W = {w:.3f} μm at λ = {wavelength_um:.3f} μm...")

        for h in sweep_heights:
            waveguide_obj: RidgeWaveguide = geometries.ridge_waveguide.get_passive_simulation(
                width=w,
                height=h,
                length=2,
                core_medium=core_medium,
                sub_medium=sub_medium,
                wavelength_um=wavelength_um,
                substrate_height=substrate_height
            )

            # Get dispersion data for the current waveguide geometry,
            # 11 points spanning 0.1um is the default.
            disp_data = waveguide_obj.get_dispersion(delta_wvl_um=0.1, num_points=11)

            # TODO: Implement error handling in the waveguide object.
            if "error" in disp_data:
                print(f"  Height {h:.2f} μm (W={w:.2f} μm): Fundamental TE00 mode not found. Skipping.")
                continue
                
            D_val = disp_data["dispersion_ps_nm_km"]
            is_singlemode = disp_data["is_singlemode"]

            results_h.append(h)
            results_D.append(D_val)
            results_is_singlemode.append(is_singlemode)

            sm_status = "SINGLE MODE" if is_singlemode else "MULTIMODE"
            print(f"    [Dispersion] Height {h:.2f} μm: D = {D_val:.2f} ps/(nm·km) [{sm_status}]")

        if not results_h:
            logger.warning(f"   [Dispersion] No valid modes identified for width W = {w:.3f} μm across all heights.")
            continue

        # Plotting
        fig, ax = plt.subplots(figsize=(9, 6), dpi=200)
        ax.plot(results_h, results_D, 'o-', color='black', linewidth=2, label='Dispersion ($D$)')
        ax.axhline(0, color='blue', linestyle='--', linewidth=1.2, label='Zero Dispersion Line')

        for i in range(len(results_h) - 1):
            h_start, h_end = results_h[i], results_h[i+1]
            D_avg = (results_D[i] + results_D[i+1]) / 2.0
            is_sm = results_is_singlemode[i] and results_is_singlemode[i+1]

            if D_avg > 0:
                color_zone = 'lightgreen' if is_sm else 'salmon'
            else:
                color_zone = 'lightgray'

            ax.axvspan(h_start, h_end, color=color_zone, alpha=0.3)

        ax.set_xlabel('Waveguide Height ($\mu$m)')
        ax.set_ylabel('Dispersion Parameter $D$ [ps / (nm · km)]')
        ax.set_title(f'Geometric Optimization Space (Fixed W = {w:.3f} μm, λ = {wavelength_um:.3f} μm)', fontsize=12, fontweight='bold')

        legend_elements = [
            Patch(facecolor='lightgray', edgecolor='gray', alpha=0.5, label='Normal Dispersion ($D < 0$)'),
            Patch(facecolor='lightgreen', edgecolor='green', alpha=0.5, label='TARGET: Anomalous + Single Mode'),
            Patch(facecolor='salmon', edgecolor='red', alpha=0.5, label='Anomalous + Multimode Hazard')
        ]
        ax.legend(handles=legend_elements, loc='best')
        ax.grid(True, alpha=0.2)
        fig.tight_layout()

        if output_dir_name:
            path = Path(output_dir_name) / f"dispersion_metrics_W{w:.3f}um.png"
            plt.savefig(path, dpi=200, bbox_inches='tight')
            print(f"  --> Saved plot: {path}")

        if display_results:
            plt.show()
        else:
            plt.close(fig)

        all_results[w] = (results_h, results_D, results_is_singlemode)

    return all_results