"""
mode_sweep.py

Carries out modal analysis on specified geometries of specific materials.
Saves plots into a specified results directory.

Author: Divyanshu Tiwari, Esraaj Sarkar Gupta
Date: 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

import tidy3d as td
from tidy3d.plugins.mode import ModeSolver
from tidy3d.constants import C_0

import geometries
import utilities

logger = logging.getLogger(__name__)

def modes_in_geometry(
        sweep_heights : list[float],
        sweep_widths : list[float],

        core_medium: td.Medium,       
        sub_medium: td.Medium,

        output_dir_name : Path,
        
        wavelength_um: float,
        num_modes : int = 8,

        preview_geometry : bool = False,
        display_results : bool = True
) -> None :
    
    # Check the output directory
    output_dir = Path(output_dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    freq0 = C_0 / wavelength_um

    # Establish substrate indexing for boundary lines
    n_sub = float(np.real(sub_medium.eps_model(freq0))**0.5)
    n_core = float(np.real(core_medium.eps_model(freq0))**0.5)
    wl_nm = int(wavelength_um * 1000)

    # --- Preview Geometry --- #
    geom_dir = output_dir / "geometries"
    geom_dir.mkdir(parents=True, exist_ok=True)

    mean_height = float(np.mean(sweep_heights))
    mean_width = float(np.mean(sweep_widths))

    # If requested, generate a dummy simulation
    if preview_geometry:
        geom_dir = output_dir / "geometries"
        geom_dir.mkdir(parents=True, exist_ok=True)

        mean_height = float(np.mean(sweep_heights))
        mean_width = float(np.mean(sweep_widths))

        logging.info(f"\n[Mode Analysis] Plotting mean geometry (Width: {mean_width:.2f} um, Height: {mean_height:.2f} um)...")

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
            output_path = output_dir_name / Path("example_geometry.png"),
            title = "Ridge Waveguide Sweep Example Geometry",
            show = True,
            save = True
        )

    for height in sweep_heights:
        logging.info(f"\n[Mode Analysis] | Height: {height:.2f} um at {wl_nm} nm")
        
        # Separate tracking dictionaries for the modes
        te_tracks : dict = {i: {"widths": [], "neffs": []} for i in range(num_modes)}
        tm_tracks : dict = {i: {"widths": [], "neffs": []} for i in range(num_modes)}

        for width in sweep_widths:
            # Dynamically resize the simulation box boundary
            sim_y_size = float(width + 3.0)
            sim_z_size = float(height + 4.0)
            
            # Pull pre-built geometry from the geometry module
            geometry_sim = geometries.ridge_waveguide.get_passive_simulation(
                width = width,
                height = height,

                length = 2,     # This doesn't really matter

                core_medium=core_medium,
                sub_medium=sub_medium,
                
                wavelength_um=wavelength_um,
            )

            # -- Setup the Mode Solver -- #
            # YZ cross-section (X size = 0)
            mode_plane = td.Box(
                center=(0, 0, height/2),
                size=(0, width + 2.0, height + 2.0) 
            )

            mode_spec = td.ModeSpec(
                num_modes = num_modes,
                target_neff = n_core
            )

            solver = ModeSolver(
                simulation=geometry_sim,
                plane=mode_plane,
                mode_spec=mode_spec,
                freqs=[freq0]
            )

            mode_data = solver.solve()

            current_te_neffs : list = []
            current_tm_neffs : list = []

            # Sort the messy modal output into neat polarization baskets
            for i in range(mode_spec.num_modes):
                n_eff = np.real(mode_data.n_eff[0, i])
                te_frac = mode_data.pol_fraction['te'][0, i]
                
                if te_frac > 0.5:
                    current_te_neffs.append(n_eff)
                else:
                    current_tm_neffs.append(n_eff)

            for rank, neff in enumerate(current_te_neffs):
                if rank < num_modes:
                    te_tracks[rank]["widths"].append(width)
                    te_tracks[rank]["neffs"].append(neff)
                    
            for rank, neff in enumerate(current_tm_neffs):
                if rank < num_modes:
                    tm_tracks[rank]["widths"].append(width)
                    tm_tracks[rank]["neffs"].append(neff)

        # ---- Plotting ---- #
        plt.figure(figsize=(12, 6))

        plt.axhline(y=n_core, color='black', linestyle='-', linewidth=1.5, label=f'Core')
        plt.axhline(y=n_sub, color='black', linestyle='--', linewidth=1.5, label='Substrate')
        plt.fill_between(sweep_widths, n_sub, n_core, color='purple', alpha=0.03, label='Guided Domain')

        colors = plt.get_cmap('tab10')

        for rank in range(num_modes):
            if te_tracks[rank]["widths"]: 
                plt.plot(te_tracks[rank]["widths"], te_tracks[rank]["neffs"], 
                         color=colors(rank % 10), marker='o', linestyle='-', markersize=5, 
                         label=f'TE$_{rank}$', zorder=3)

        for rank in range(num_modes):
            if tm_tracks[rank]["widths"]: 
                plt.plot(tm_tracks[rank]["widths"], tm_tracks[rank]["neffs"], 
                         color=colors((rank + 4) % 10), marker='s', linestyle='--', markersize=5, 
                         label=f'TM$_{rank}$', zorder=3)

        plt.title(f"Dispersion Profile | (h={height:.2f} $\mu$m) at $\lambda$={wavelength_um*1000:.0f}nm", fontsize=12, pad=12)
        plt.xlabel("Waveguide Width ($\mu$m)", fontsize=11)
        plt.ylabel("Effective Index ($n_{eff}$)", fontsize=11)

        plt.ylim(n_sub - 0.05, n_core + 0.05)
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)

        plt.tight_layout()
        
        # Formatted name emphasizing substrate height and wavelength
        file_name = f"{wl_nm}nm_wgW_{width:.2f}_wgH_{height:.2f}um.png"
        
        plt.savefig(output_dir / file_name, dpi=150)
        if display_results:
            plt.show()
        plt.close()
        
        logging.info(f"[Mode Analysis] Visualized distribution out to {file_name}")
        

    




