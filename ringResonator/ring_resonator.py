"""
ringResonator.ring_resonator

Author: Esraaj Sarkar Gupta
Date: 2026
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, TYPE_CHECKING, Union

import numpy as np
import matplotlib.pyplot as plt
import tidy3d as td
from tidy3d import web

if TYPE_CHECKING:
    from geometries.half_ring_resonator import (
        get_active_simulation as get_active_half_sim,
        get_passive_simulation as get_passive_half_sim,
    )
    from geometries.full_ring_resonator import (
        get_active_simulation as get_active_full_sim,
        get_passive_simulation as get_passive_full_sim,
    )

logger = logging.getLogger(__name__)


class BaseResonator:
    def __init__(
        self,
        sim: td.Simulation,
        radius: float,
        gap: float,
        wg_width: float,
        wg_height: float,
        core_medium: td.Medium,
        substrate_medium: td.Medium,
        substrate_height: float,
        top_cladding_medium: td.Medium,
        top_cladding_height: float,
        fine_steps_per_wvl: float,
        coarse_steps_per_wvl: float,
        wg_length: float,
        wavelength0: float,
    ):
        self.sim = sim
        self.radius = radius
        self.gap = gap
        self.wg_width = wg_width
        self.wg_height = wg_height
        self.wg_length = wg_length
        self.core_medium = core_medium
        self.substrate_medium = substrate_medium
        self.top_cladding_medium = top_cladding_medium
        self.substrate_height = substrate_height
        self.top_cladding_height = top_cladding_height
        self.fine_steps_per_wvl = fine_steps_per_wvl
        self.coarse_steps_per_wvl = coarse_steps_per_wvl
        self.wavelength0 = wavelength0

    def run_sim(
        self, 
        output_dir: Union[str, Path], 
        task_name: str = "Resonator_Sim"
    ) -> td.SimulationData:
        """
        Yeets the simulation to the Tidy3D cloud and downloads the HDF5 results.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        hdf5_path = output_dir / f"{task_name}.hdf5"
        
        logger.info(f"Submitting {task_name} to the void... I mean, cloud.")
        sim_data = web.run(self.sim, task_name=task_name, path=str(hdf5_path))
        return sim_data

    @staticmethod
    def load_sim_data(filepath: Union[str, Path]) -> td.SimulationData:
        """
        Resurrects simulation data from a local .hdf5 cache file.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Are you hallucinating? There is no file at {filepath}")
        return td.SimulationData.from_file(str(filepath))

    def plot_field(
        self, 
        sim_data: td.SimulationData, 
        monitor_name: str = "coupling_field_map", 
        output_path: Optional[Union[str, Path]] = None, 
        show: bool = True, 
        save: bool = False
    ) -> None:
        """
        Extracts and plots the E-field magnitude for a specific monitor.
        Because staring at raw arrays is just depressing.
        """
        if monitor_name not in sim_data.simulation.monitors:
            raise ValueError(f"Monitor '{monitor_name}' doesn't exist. Did you even run an active sim?")
            
        field_data = sim_data[monitor_name]
        
        fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
        
        # Calculate full E-field magnitude
        E_mag = np.sqrt(
            np.abs(field_data.Ex)**2 + 
            np.abs(field_data.Ey)**2 + 
            np.abs(field_data.Ez)**2
        )
        
        E_mag.isel(f=0).plot(x='x', y='y', ax=ax, cmap='magma', robust=True)
        
        ax.set_title(f"Field Intensity: {monitor_name} (Gap={self.gap*1000:.0f}nm)", fontsize=10)
        ax.set_aspect('equal')
        fig.tight_layout()
        
        if save:
            if output_path is None or str(output_path).strip() == "":
                raise ValueError("Cannot save plot: 'save' is True, but no valid 'output_path' was provided. Try using actual letters.")
            
            path = Path(output_path)
            # If they just pass a directory, generate a cute filename for them
            if path.is_dir() or str(output_path).endswith(('/', '\\')) or not path.suffix:
                path = path / f"{monitor_name}_field.png"
                
            if path.parent:
                path.parent.mkdir(parents=True, exist_ok=True)
                
            plt.savefig(path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved field plot to {path}")
            
        if show:
            plt.show()
        else:
            plt.close(fig)


class halfringResonator(BaseResonator):
    def to_full_ring(self, active: Optional[bool] = None) -> ringResonator:
        """
        Converts this half-ring simulation object to a full-ring simulation object.
        Auto-detects active/passive state if `active` is not explicitly provided.
        """
        from geometries.full_ring_resonator import (
            get_active_simulation,
            get_passive_simulation,
        )

        is_active = active if active is not None else (len(self.sim.sources) > 0)
        builder = get_active_simulation if is_active else get_passive_simulation

        return builder(
            radius=self.radius,
            gap=self.gap,
            wg_width=self.wg_width,
            wg_height=self.wg_height,
            core_medium=self.core_medium,
            substrate_medium=self.substrate_medium,
            wavelength0=self.wavelength0,
            substrate_height=self.substrate_height,
            top_cladding_medium=self.top_cladding_medium,
            top_cladding_height=self.top_cladding_height,
            fine_steps_per_wvl=self.fine_steps_per_wvl,
            coarse_steps_per_wvl=self.coarse_steps_per_wvl,
            wg_length=self.wg_length,
        )


class ringResonator(BaseResonator):
    def to_half_ring(self, active: Optional[bool] = None) -> halfringResonator:
        """
        Converts this full-ring simulation object to a truncated half-ring simulation object.
        Auto-detects active/passive state if `active` is not explicitly provided.
        """
        from geometries.half_ring_resonator import (
            get_active_simulation,
            get_passive_simulation,
        )

        is_active = active if active is not None else (len(self.sim.sources) > 0)
        builder = get_active_simulation if is_active else get_passive_simulation

        return builder(
            radius=self.radius,
            gap=self.gap,
            wg_width=self.wg_width,
            wg_height=self.wg_height,
            core_medium=self.core_medium,
            substrate_medium=self.substrate_medium,
            wavelength0=self.wavelength0,
            substrate_height=self.substrate_height,
            top_cladding_medium=self.top_cladding_medium,
            top_cladding_height=self.top_cladding_height,
            fine_steps_per_wvl=self.fine_steps_per_wvl,
            coarse_steps_per_wvl=self.coarse_steps_per_wvl,
            wg_length=self.wg_length,
        )