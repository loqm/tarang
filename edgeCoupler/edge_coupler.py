"""
edgeCoupler.edge_coupler

Defines the edgeCoupler wrapper class for photonics simulations.

Author: Esraaj Sarkar Gupta, Divyanshu Tiwari
Date: 2026
"""

from __future__ import annotations
import logging
import os
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, TYPE_CHECKING, Union, Tuple

import numpy as np
import matplotlib.pyplot as plt
import tidy3d as td
from tidy3d import web

logger = logging.getLogger(__name__)


class edgeCoupler:
    """
    Wrapper class representing an edge coupler simulation domain.

    Simulation Modes:
        - In-Coupling  (`in_coupling=True, out_coupling=False`):
            Light launches from a lensed fiber source into the edge taper (+X).
        - Out-Coupling (`in_coupling=False, out_coupling=True`):
            Light launches from a guided mode source out towards the tip (-X).
        - Passive      (`in_coupling=False, out_coupling=False`):
            Geometry-only structural domain with no active sources. Useful for
            visualizing refractive index cross-sections or boundary layouts.
    """

    def __init__(
            self,
            sim : td.Simulation,

            tip_width : float,
            taper_length : float,

            wg_width : float,
            wg_height : float,

            core_medium : td.Medium,
            substrate_medium : td.Medium,
            top_cladding_medium : td.Medium,

            wavelength0 : float,
      
            substrate_height : float,
            top_cladding_height : float,

            fine_steps_per_wvl : int,
            coarse_steps_per_wvl : int,

            settling_buffer_length : float,
            substrate_length : float,

            in_coupling : bool = False,
            out_coupling : bool = False,
            
            symmetry : Tuple[int, int, int] = (0, 0, 0)
    ):  
        # Generally optimizable degrees of freedom
        self.tip_width = tip_width
        self.taper_length = taper_length
        
        # Waveguide characteristics
        self.wg_width = wg_width
        self.wg_height = wg_height

        # Device media
        self.core_medium = core_medium
        self.substrate_medium = substrate_medium
        self.top_cladding_medium = top_cladding_medium

        # Wavelength of interest
        self.wavelength0 = wavelength0

        # Isolating layer thicknesses
        self.substrate_height = substrate_height
        self.top_cladding_height = top_cladding_height

        # Meshing resolution
        self.fine_steps_per_wvl = fine_steps_per_wvl
        self.coarse_steps_per_wvl = coarse_steps_per_wvl

        # Buffer length for the mode monitor
        self.settling_buffer_length = settling_buffer_length

        # Substrate length (total physical length of BOX layer)
        self.substrate_length = substrate_length
        
        # Simulation boundary symmetry (x, y, z)
        self.symmetry = symmetry

        # Validation: raise error only if BOTH active modes are True simultaneously
        if in_coupling and out_coupling:
            raise ValueError(
                f"Invalid directional configuration: in_coupling={in_coupling}, out_coupling={out_coupling}. "
                "Both 'in_coupling' and 'out_coupling' cannot be True simultaneously. "
                "Set one to True for active simulations, or set both to False for a passive simulation."
            )

        self.in_coupling = in_coupling
        self.out_coupling = out_coupling
        self.is_passive = not (in_coupling or out_coupling)

        # The Simulation
        self.sim = sim

    # --------------------------------
    # Cache & Execution Management
    # --------------------------------

    def get_dir_name(self) -> str:
        """Generates a clean, unique parameter-based identifier containing an MD5 hash for cache safety."""
        if self.in_coupling:
            mode_str = "incouple"
        elif self.out_coupling:
            mode_str = "outcouple"
        else:
            mode_str = "passive"

        sig = (
            f"{self.tip_width:.6f}_{self.taper_length:.6f}_{self.wg_width:.6f}_{self.wg_height:.6f}_"
            f"{self.core_medium}_{self.substrate_medium}_{self.top_cladding_medium}_{mode_str}_"
            f"{self.wavelength0:.6f}_{self.substrate_height:.6f}_{self.top_cladding_height:.6f}_"
            f"{self.fine_steps_per_wvl}_{self.coarse_steps_per_wvl}_{self.settling_buffer_length:.6f}_"
            f"{self.symmetry[0]}_{self.symmetry[1]}_{self.symmetry[2]}"
        )
        h = hashlib.md5(sig.encode()).hexdigest()[:8]
        
        sym_str = f"sym{self.symmetry[0]}{self.symmetry[1]}{self.symmetry[2]}"
        return f"coupler_Wtip{self.tip_width:.2f}um_Ltaper{self.taper_length:.2f}um_{mode_str}_{sym_str}_{self.wavelength0:.3f}um_{h}"

    def _run_sim(
        self, 
        output_dir: Union[str, Path], 
        verbose: bool = False,
        force_rerun: bool = False
    ) -> td.SimulationData:
        """
        Private method: Checks for local cached HDF5 simulation data or executes via cloud.
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
        task_name = f"edge_coupler_{param_dir}"
        sim_data = web.run(self.sim, task_name=task_name, verbose=verbose)
        
        logger.info(f"[Profiler] Execution complete. Serializing data to {hdf5_target_path}...")
        sim_data.to_file(str(hdf5_target_path))
        
        return sim_data

    # Alias for method name consistency
    _run_simulation = _run_sim

    # --------------------------------
    # Execution and Analysis Methods
    # --------------------------------

    def get_mode_data(
        self,
        output_dir: Union[str, Path],
        monitor_name: Optional[str] = None,
        force_rerun: bool = False,
        verbose: bool = False,
    ) -> Any:
        """Retrieves raw monitor data for a specified monitor."""
        if self.is_passive:
            raise ValueError("Cannot extract mode transmission data from a passive simulation (no sources present).")

        if monitor_name is None:
            monitor_name = "mode_transmission" if self.in_coupling else "flux_fiber"

        sim_data = self._run_sim(output_dir=output_dir, verbose=verbose, force_rerun=force_rerun)
        
        try:
            return sim_data[monitor_name]
        except KeyError:
            raise KeyError(f"Monitor '{monitor_name}' not found in simulation monitors.")

    def get_efficiency(
        self,
        output_dir: Union[str, Path],
        monitor_name: Optional[str] = None,
        verbose: bool = False,
        force_rerun: bool = False
    ) -> float:
        """
        Calculates power coupling efficiency (0.0 to 1.0).
        Dynamically adjusts target monitor and data field based on coupling direction.
        
        Raises:
            RuntimeError: If monitor extraction or evaluation fails due to missing keys or shape mismatch.
        """
        if self.is_passive:
            raise ValueError("Cannot calculate coupling efficiency on a passive simulation (no sources present).")

        if monitor_name is None:
            monitor_name = "mode_transmission" if self.in_coupling else "flux_fiber"

        sim_data = self._run_sim(output_dir=output_dir, verbose=verbose, force_rerun=force_rerun)
        
        logger.info(f"[Analytics] Extracting transmission metrics from '{monitor_name}'...")
        try:
            target_freq = td.C_0 / self.wavelength0
            monitor_data = sim_data[monitor_name]

            if hasattr(monitor_data, "amps"):
                # ModeMonitor: extract amplitude squared for target mode
                direction = "+" if self.in_coupling else "-"
                mode_amps = monitor_data.amps.sel(direction=direction, mode_index=0)
                efficiency = float(np.abs(mode_amps.sel(f=target_freq, method="nearest"))**2)
            elif hasattr(monitor_data, "flux"):
                # FluxMonitor: extract normalized net flux across plane
                flux_val = float(monitor_data.flux.sel(f=target_freq, method="nearest"))
                efficiency = float(np.abs(flux_val))
            else:
                raise ValueError(f"Monitor '{monitor_name}' contains neither 'amps' nor 'flux' fields.")
            
            logger.info(f"[Analytics] Transmission evaluation complete. Efficiency: {efficiency * 100.0:.2f}%")
            return efficiency
            
        except (KeyError, ValueError) as e:
            logger.error(f"[Analytics Error] Failed to parse transmission efficiency cleanly: {e}")
            raise RuntimeError(
                f"Failed to extract coupling efficiency from monitor '{monitor_name}' "
                f"for simulation '{self.get_dir_name()}': {e}"
            ) from e

    def get_loss_db(
        self,
        output_dir: Union[str, Path],
        monitor_name: Optional[str] = None,
        verbose: bool = False,
        force_rerun: bool = False
    ) -> float:
        """Calculates Insertion Loss in decibels (dB)."""
        efficiency = self.get_efficiency(
            output_dir=output_dir, 
            monitor_name=monitor_name, 
            verbose=verbose, 
            force_rerun=force_rerun
        )
        loss_db = float(-10.0 * np.log10(efficiency + 1e-12))
        logger.info(f"[Analytics] Calculated Insertion Loss: {loss_db:.2f} dB")
        return loss_db

    def plot_fields(
        self, 
        output_dir: Union[str, Path],
        monitor_name: str = "field_xy", 
        output_path: Optional[Union[str, Path]] = None, 
        show: bool = True, 
        save: bool = False,
        verbose: bool = False,
        force_rerun: bool = False
    ) -> None:
        """Plots spatial electric field intensity profiles for FieldMonitors."""
        sim_data = self._run_sim(output_dir=output_dir, verbose=verbose, force_rerun=force_rerun)
        
        try:
            field_data = sim_data[monitor_name]
        except KeyError:
            raise ValueError(f"Monitor '{monitor_name}' does not exist in simulation monitors.")
            
        fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
        
        E_mag = np.sqrt(
            np.abs(field_data.Ex)**2 + 
            np.abs(field_data.Ey)**2 + 
            np.abs(field_data.Ez)**2
        )
        
        # Determine cross-section slice coordinates
        x_label, y_label = ('x', 'y') if 'xy' in monitor_name else ('x', 'z')
        E_mag.isel(f=0).squeeze().plot(x=x_label, y=y_label, ax=ax, cmap='magma', robust=True)
        
        if self.in_coupling:
            mode_label = "In-Coupling"
        elif self.out_coupling:
            mode_label = "Out-Coupling"
        else:
            mode_label = "Passive"

        ax.set_title(
            f"Field Intensity: {monitor_name} ({mode_label}, Wtip={self.tip_width}μm, Ltaper={self.taper_length}μm)",
            fontsize=10
        )
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
                    path = path / f"coupler_{mode_label}_Wtip{self.tip_width:.2f}_{monitor_name}.png"
                
                if path.parent:
                    path.parent.mkdir(parents=True, exist_ok=True)
                
            plt.savefig(path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved field plot to {path}")
            
        if show:
            plt.show()
        else:
            plt.close(fig)