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
import hashlib

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
        bend_angle_deg: float, 
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
            substrate_height=2.5,       # Fixed to match Z-bounds of passive sim (-2.5 to 1.5)
            top_cladding_medium=cladding_medium,
            top_cladding_height=1.5,    # Fixed to match Z-bounds
            fine_steps_per_wvl=fine_steps_per_wvl,
            coarse_steps_per_wvl=coarse_steps_per_wvl,
            wg_length=0.0,
            wavelength0=wavelength0
        )
        self.bend_angle_deg = bend_angle_deg
        self.cladding_medium = cladding_medium
        self.mode_direction = "+"

    def get_dir_name(self) -> str:
        """Generates a clean, unique parameter-based identifier containing an MD5 hash for cache safety."""
        sig = f"{self.wg_width:.4f}_{self.wg_height:.4f}_{self.core_medium}_{self.cladding_medium}_{self.substrate_medium}"
        h = hashlib.md5(sig.encode()).hexdigest()[:8]
        return f"arc_R{self.radius:.2f}um_deg{self.bend_angle_deg:.1f}_{self.wavelength0:.3f}um_{h}"

    def _run_sim(
        self, 
        output_dir: Union[str, Path], 
        verbose: bool = False,
        force_rerun: bool = False
    ) -> td.SimulationData:
        """
        Private method: Checks for local cached HDF5 simulation data.
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

    # --------------------------------
    # Execution and Analysis Methods
    # --------------------------------

    def get_mode_data(
        self,
        output_dir: Union[str, Path],
        monitor_name: str = "Mode Monitor",
        force_rerun: bool = False,
        verbose: bool = False,
    ) -> Any:
        sim_data = self._run_sim(output_dir=output_dir, verbose=verbose, force_rerun=force_rerun)
        
        try:
            return sim_data[monitor_name]
        except KeyError:
            raise KeyError(f"Monitor '{monitor_name}' not found in simulation monitors.")

    def get_efficiency(
        self,
        output_dir: Union[str, Path],
        monitor_name: str = "Mode Monitor",
        verbose: bool = False,
        force_rerun: bool = False
    ) -> float:
        sim_data = self._run_sim(output_dir=output_dir, verbose=verbose, force_rerun=force_rerun)
        
        logger.info(f"[Analytics] Extracting transmission coefficients from '{monitor_name}'...")
        try:
            direction = getattr(self, "mode_direction", "+")
            mode_amps_array = sim_data[monitor_name].amps.sel(direction=direction, mode_index=0)
            target_freq = td.C_0 / self.wavelength0
            
            efficiency = float(np.abs(mode_amps_array.sel(f=target_freq, method="nearest"))**2)
            
            logger.info(f"[Analytics] Target mode audit completed. Coupling Efficiency: {efficiency*100.0:.2f}%")          
            return efficiency
            
        except Exception as e:
            logger.error(f"[Analytics Error] Failed to parse mode data cleanly: {e}")
            return 0.0

    def get_loss(
        self,
        output_dir: Union[str, Path],
        monitor_name: str = "Mode Monitor",
        verbose: bool = False,
        force_rerun: bool = False
    ) -> float:
        efficiency = self.get_efficiency(
            output_dir=output_dir, 
            monitor_name=monitor_name, 
            verbose=verbose, 
            force_rerun=force_rerun
        )
        loss_db = -10.0 * np.log10(efficiency + 1e-12)
        logger.info(f"[Analytics] Calculated Insertion Loss: {loss_db:.2f} dB")
        return loss_db

    def plot_field(
        self, 
        output_dir: Union[str, Path],
        monitor_name: str = "field_map", 
        output_path: Optional[Union[str, Path]] = None, 
        show: bool = True, 
        save: bool = False,
        verbose: bool = False,
        force_rerun: bool = False
    ) -> None:
        sim_data = self._run_sim(output_dir=output_dir, verbose=verbose, force_rerun=force_rerun)
        
        try:
            field_data = sim_data[monitor_name]
        except KeyError:
            raise ValueError(f"Monitor '{monitor_name}' doesn't exist in the simulation monitors.")
            
        fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
        
        E_mag = np.sqrt(
            np.abs(field_data.Ex)**2 + 
            np.abs(field_data.Ey)**2 + 
            np.abs(field_data.Ez)**2
        )
        
        E_mag.isel(f=0).squeeze().plot(x='x', y='y', ax=ax, cmap='magma', robust=True)
        
        ax.set_title(f"Field Intensity: {monitor_name} (R={self.radius}μm, {self.bend_angle_deg:.1f}°)", fontsize=10)
        ax.set_aspect('equal')
        fig.tight_layout()
        
        if save:
            param_dir = self.get_dir_name()
            target_dir = Path(output_dir) / param_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            
            if output_path is None or str(output_path).strip() == "":
                path = target_dir / f"spatial_field_{monitor_name}.png"
            else:
                path = Path(output_path)
                if path.is_dir() or str(output_path).endswith(('/', '\\')) or not path.suffix:
                    path = path / f"bend_{self.bend_angle_deg:.1f}deg_R{self.radius:.2f}_{monitor_name}.png"
                
                if path.parent:
                    path.parent.mkdir(parents=True, exist_ok=True)
                
            plt.savefig(path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved field plot to {path}")
            
        if show:
            plt.show()
        else:
            plt.close(fig)