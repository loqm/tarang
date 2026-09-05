"""
ringResonator.resonator_arc

Author: Esraaj Sarkar Gupta
Date: 2026
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Union, Any
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import tidy3d as td
from tidy3d import web

from ringResonator.ring_resonator import BaseResonator

logger = logging.getLogger(__name__)

class resonatorArc(BaseResonator):
    def __init__(
        self,
        sim: td.Simulation,
        radius: float,
        wg_width: float,
        wg_height: float,
        core_medium: td.Medium,
        cladding_medium: td.Medium,
        substrate_medium: td.Medium,
        bend_angle_deg: float,  # Updated to float for continuous angle support
        fine_steps_per_wvl: float,
        coarse_steps_per_wvl: float,
        wavelength0: float,
    ):
        super().__init__(
            sim=sim,
            radius=radius,
            gap=0.0,
            wg_width=wg_width,
            wg_height=wg_height,
            core_medium=core_medium,
            substrate_medium=substrate_medium,
            substrate_height=2.0,
            top_cladding_medium=cladding_medium,
            top_cladding_height=10.0,
            fine_steps_per_wvl=fine_steps_per_wvl,
            coarse_steps_per_wvl=coarse_steps_per_wvl,
            wg_length=0.0,
            wavelength0=wavelength0
        )
        self.bend_angle_deg = bend_angle_deg
        self.cladding_medium = cladding_medium

    def get_dir_name(self) -> str:
        """Generates a clean, unique parameter-based identifier for caching directory paths."""
        # Added .1f formatting to bend_angle_deg to ensure clean directory names
        return f"arc_R{self.radius:.2f}um_deg{self.bend_angle_deg:.1f}_{self.wavelength0:.3f}um"

    def run_sim(
        self, 
        output_dir: Union[str, Path], 
        verbose: bool = False,
        force_rerun: bool = False
    ) -> td.SimulationData:
        """
        Checks for local cached HDF5 simulation data.
        If missing (or if force_rerun=True), executes the FDTD simulation on the 
        Tidy3D cloud infrastructure and serializes the dataset to disk.
        """
        if "TIDY3D_KEY" not in os.environ and "TIDY3D_API_KEY" not in os.environ:
            logger.warning("[Warning] TIDY3D_API_KEY not found in environment variables.")

        param_dir = self.get_dir_name()
        full_output_dir = Path(output_dir) / param_dir
        full_output_dir.mkdir(parents=True, exist_ok=True)
        
        hdf5_target_path = full_output_dir / "raw_simulation_data.hdf5"

        if hdf5_target_path.exists() and not force_rerun:
            logger.info(f"[Profiler] Cache hit. Loading existing simulation data from {hdf5_target_path}...")
            return td.SimulationData.from_file(str(hdf5_target_path))
        
        logger.info(f"[Profiler] Cache miss. Uploading {param_dir} to Tidy3D cloud infrastructure...")
        task_name = f"arc_sim_{param_dir}"
        sim_data = web.run(self.sim, task_name=task_name, verbose=verbose)
        
        logger.info(f"[Profiler] Execution complete. Serializing data to {hdf5_target_path}...")
        sim_data.to_file(str(hdf5_target_path))
        
        return sim_data

    def get_mode_data(
        self,
        output_dir: Union[str, Path],
        monitor_name: str = "Mode Monitor",
        force_rerun: bool = False,
        verbose: bool = False,
    ) -> Any:
        """
        Retrieves mode monitor data from cached HDF5 file or triggers simulation execution.
        """
        sim_data = self.run_sim(output_dir=output_dir, verbose=verbose, force_rerun=force_rerun)
        
        if monitor_name not in sim_data.simulation.monitors:
            raise KeyError(f"Monitor '{monitor_name}' not found in simulation monitors.")
            
        return sim_data[monitor_name]

    def plot_field(
        self, 
        sim_data: td.SimulationData,
        monitor_name: str = "field_map", 
        output_path: Optional[Union[str, Path]] = None, 
        show: bool = True, 
        save: bool = False
    ) -> None:
        """Extracts and visualizes the E-field magnitude for the arc monitor."""
        if monitor_name not in sim_data.simulation.monitors:
            raise ValueError(f"Monitor '{monitor_name}' doesn't exist in the simulation monitors.")
            
        field_data = sim_data[monitor_name]
        
        fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
        
        E_mag = np.sqrt(
            np.abs(field_data.Ex)**2 + 
            np.abs(field_data.Ey)**2 + 
            np.abs(field_data.Ez)**2
        )
        
        E_mag.isel(f=0).plot(x='x', y='y', ax=ax, cmap='magma', robust=True)
        
        # Formatted bend_angle_deg to .1f for cleaner plot titles
        ax.set_title(f"Field Intensity: {monitor_name} (R={self.radius}μm, {self.bend_angle_deg:.1f}°)", fontsize=10)
        ax.set_aspect('equal')
        fig.tight_layout()
        
        if save:
            if output_path is None or str(output_path).strip() == "":
                raise ValueError("Cannot save plot: 'save' is True, but no valid 'output_path' was provided.")
            
            path = Path(output_path)
            if path.is_dir() or str(output_path).endswith(('/', '\\')) or not path.suffix:
                # Formatted bend_angle_deg to .1f and radius to .2f for robust file naming
                path = path / f"bend_{self.bend_angle_deg:.1f}deg_R{self.radius:.2f}_{monitor_name}.png"
                
            if path.parent:
                path.parent.mkdir(parents=True, exist_ok=True)
                
            plt.savefig(path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved field plot to {path}")
            
        if show:
            plt.show()
        else:
            plt.close(fig)