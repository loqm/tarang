"""
waveguide.confinement_sweep.py

This script allows us to evaluate and visualize spatial mode confinement metrics 
(Confinement Factor and Effective Mode Area) across a geometric sweep.

Author: Divyanshu Tiwari, Esraaj Sarkar Gupta
Date: 2026
"""
import os
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

import tidy3d as td
from tidy3d.plugins.mode import ModeSolver

import geometries
import utilities

logger = logging.getLogger(__name__)
ConfinementSweepResult = Dict[float, Tuple[List[float], List[float], List[float]]]


def get_te_fraction(mode_data, m_idx: int) -> float:
    """
    Docstring for get_te_fraction
    
    :param mode_data: Description
    :param m_idx: Description
    :type m_idx: int
    :return: Description
    :rtype: float

    Computes the TE polarization fraction for a
    waveguide propagating along X. TE dominance is
    determined by the horizontal transverse field
    component (Ey).
    """

    Ey = mode_data.Ey.isel(mode_index=m_idx, f=0).values
    Ez = mode_data.Ez.isel(mode_index=m_idx, f=0).values
    Iy = np.sum(np.abs(Ey)**2)
    Iz = np.sum(np.abs(Ez)**2)
    return float(Iy / (Iy + Iz + 1e-10))


def confinement_in_geometry(
        sweep_heights : list[float],
        sweep_widths : list[float],

        core_medium: td.Medium,       
        sub_medium: td.Medium,

        output_dir_name : Path,
        
        wavelength_um: float,

        preview_geometry : bool = False,
        display_results : bool = True
) -> ConfinementSweepResult :
    """
    Docstring for confinement_in_geometry
    
    :param sweep_heights: Description
    :type sweep_heights: list[float]
    :param sweep_widths: Description
    :type sweep_widths: list[float]
    :param core_medium: Description
    :type core_medium: td.Medium
    :param sub_medium: Description
    :type sub_medium: td.Medium
    :param output_dir_name: Description
    :type output_dir_name: Path
    :param wavelength_um: Description
    :type wavelength_um: float
    :param preview_geometry: Description
    :type preview_geometry: bool
    :param display_results: Description
    :type display_results: bool
    :return: Description
    :rtype: ConfinementSweepResult

    Evaluates and visualizes TE00 mode confinement metrics
    across a combined height and width sweep. Generates a unique
    width-sweep summary figure for each individual height in
    `sweep_heights`.
    """
    freq0 = td.C_0 / wavelength_um
    n_core_guess = float(np.real(np.sqrt(core_medium.eps_model(freq0))))

    if output_dir_name:
        output_path = Path(output_dir_name)
        output_path.mkdir(parents=True, exist_ok=True)

    # Optional visual check -- DO THIS ONCE before wasting time on 50 simulations
    if preview_geometry and output_dir_name:
        geom_dir = output_dir_name / Path("geometries/")
        geom_dir.mkdir(parents=True, exist_ok=True)

        mean_height = float(np.mean(sweep_heights))
        mean_width = float(np.mean(sweep_widths))

        logger.info(f"     [Confinement] Plotting mean geometry (Width: {mean_width:.2f} μm, Height: {mean_height:.2f} μm)...")

        preview_sim = geometries.ridge_waveguide.get_passive_simulation(
            width = mean_width,
            height = mean_height,
            length = (5*mean_width + 5*mean_height)/2,
            core_medium = core_medium,
            sub_medium = sub_medium,
            wavelength_um = wavelength_um,
        )

        utilities.sim_plotter.display_3d(
            sim = preview_sim,
            output_path = geom_dir / Path("example_geometry.png"),
            title = "Ridge Waveguide Confinement Sweep Example Geometry",
            show = display_results,
            save = True
        )

    logger.info(f"Starting confinement sweep across {len(sweep_heights)} heights and {len(sweep_widths)} widths.")

    all_results: ConfinementSweepResult = {}

    for h in sweep_heights:
        results_w : list = []
        results_gamma : list = []
        results_Aeff : list = []

        print(f"    [Confinement] Running geometric width sweep for fixed H = {h:.3f} μm at λ = {wavelength_um:.3f} μm...")

        for w in sweep_widths:
            sim = geometries.ridge_waveguide.get_passive_simulation(
                width = w,
                height = h,
                length = 2,
                core_medium = core_medium,
                sub_medium = sub_medium,
                wavelength_um = wavelength_um
            )

            mode_spec = td.ModeSpec(num_modes=6, target_neff=n_core_guess)
            plane = td.Box(center=(0, 0, h/2), size=(0, sim.size[1], sim.size[2]))

            solver = ModeSolver(simulation=sim, plane=plane, mode_spec=mode_spec, freqs=[freq0])
            data = solver.solve()

            te00_idx = -1
            for m in range(len(data.n_eff.mode_index)):
                if get_te_fraction(data, m) > 0.5:
                    te00_idx = m
                    break

            if te00_idx == -1:
                print(f"  Width {w:.2f} μm (H={h:.2f} μm): Fundamental TE00 mode not found. Skipping.")
                continue

            Ex = data.Ex.isel(mode_index=te00_idx, f=0).values.squeeze()
            Ey = data.Ey.isel(mode_index=te00_idx, f=0).values.squeeze()
            Ez = data.Ez.isel(mode_index=te00_idx, f=0).values.squeeze()

            intensity = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2

            y_vals = data.Ex.y.values
            z_vals = data.Ex.z.values
            Y, Z = np.meshgrid(y_vals, z_vals, indexing='ij')

            core_mask = (np.abs(Y) <= w / 2) & (Z >= 0) & (Z <= h)

            gamma = np.sum(intensity[core_mask]) / np.sum(intensity)

            dy = y_vals[1] - y_vals[0]
            dz = z_vals[1] - z_vals[0]
            dA = dy * dz

            numerator = (np.sum(intensity) * dA)**2
            denominator = np.sum(intensity**2) * dA
            A_eff = numerator / denominator

            results_w.append(w)
            results_gamma.append(gamma * 100)
            results_Aeff.append(A_eff)

            print(f"    [Confinement] Width {w:.2f} μm: Gamma = {gamma*100:.1f}%, A_eff = {A_eff:.3f} μm²")

        if not results_w:
            logger.warning(f"   [Confinement] No valid modes identified for height H = {h:.3f} μm across all widths.")
            continue

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