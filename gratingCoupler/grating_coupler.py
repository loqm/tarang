"""
grating_coupler.grating_coupler

Defines the GratingConfig dataclass and GratingCoupler wrapper class.

Author: Esraaj Sarkar Gupta, Divyanshu Tiwari
Date: 2026
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

import tidy3d as td
from tidy3d import web

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    raise FileNotFoundError("Failed to find .env file! I'm not sure where to\
search for your API key right now!")

if TYPE_CHECKING:
    from geometries.grating_coupler import get_active_simulation

logger = logging.getLogger(__name__)

@dataclass
class GratingConfig:
    wavelength_um: float

    theta_deg: float
    mfd: float
    source_x: float
    source_gap: float

    # Substrate geometry
    core_height: float
    cladding_height: float
    box_height: float        # Isolating oxide layer beneath core
    substrate_height: float  # Deep silicon handle

    # Coupler geometry
    etch_depth: float
    period: float
    ff: float                # Fill Factor
    n_teeth: int
    w_grating: float

    # Source
    pol_angle: float

    buffer: float


class GratingCoupler:
    """
    Wraps a Tidy3D Simulation object for grating couplers, binding
    its GratingConfig parameters and material media directly to the object.
    """

    def __init__(
            self,
            sim: td.Simulation,
            config: GratingConfig,

            core_medium: td.Medium,
            box_medium: td.Medium,
            substrate_medium: td.Medium,
            cladding_medium: td.Medium | None = None,   # Defaults to ``air" if None

            is_3d: bool = False,

            fine_steps_per_wvl: int = 20,
            coarse_steps_per_wvl: int = 15
    ):
        self.sim = sim
        self.config = config

        self.core_medium = core_medium
        self.box_medium = box_medium
        self.substrate_medium = substrate_medium
        self.cladding_medium = cladding_medium

        self.is_3d = is_3d

        self.fine_steps_per_wvl = fine_steps_per_wvl
        self.coarse_steps_per_wvl = coarse_steps_per_wvl

    # This is a questionable design choice
    """
    def __getattr__(self, item):
        #Forwards unknown attributes/methods directly to self.sim
        return getattr(self.sim, item)
    """

    @classmethod
    def from_config(
            cls,
            config: GratingConfig,
            core_medium: td.Medium,
            box_medium: td.Medium,
            substrate_medium: td.Medium | None = None,
            cladding_medium: td.Medium | None = None,
            is_3d: bool = False,
            fine_steps_per_wvl: int = 15,
            coarse_steps_per_wvl: int = 10
    ) -> GratingCoupler:
        """
        Convenience factory method to build and wrap a simulation in one step.
        """
        from geometries.grating_coupler import get_active_simulation

        return get_active_simulation(
            config=config,
            core_medium=core_medium,
            box_medium=box_medium,
            substrate_medium=substrate_medium,
            cladding_medium=cladding_medium,
            is_3d=is_3d,
            fine_steps_per_wvl=fine_steps_per_wvl,
            coarse_steps_per_wvl=coarse_steps_per_wvl
        )
    
    # --------------------------------
    # Execution and Analysis Methods
    # --------------------------------
    def get_dir_name(self) -> str:
        dim = "3D" if self.is_3d else "2D"

        # 2. Construct the hyper-specific directory name
        # Using 4 decimal places for floats to keep path readable yet precise
        cfg = self.config
        param_dir_name = (
            f"{dim}_p_{cfg.period:.4f}_dc_{cfg.ff:.4f}_e_{cfg.etch_depth:.4f}_"
            f"n_{cfg.n_teeth}_w_{cfg.w_grating:.4f}_ch_{cfg.core_height:.4f}_"
            f"clh_{cfg.cladding_height:.4f}_bh_{cfg.box_height:.4f}_sh_{cfg.substrate_height:.4f}_"
            f"t_{cfg.theta_deg:.4f}_mfd_{cfg.mfd:.4f}_sx_{cfg.source_x:.4f}_sg_{cfg.source_gap:.4f}_"
            f"wl_{cfg.wavelength_um:.4f}_pol_{cfg.pol_angle:.4f}_b_{cfg.buffer:.4f}"
        )
        return param_dir_name

    def _run_sim(self, output_dir: Path) -> td.SimulationData:
        """
        Private method: Checks for API key, ensures the output directory exists,
        and either runs the FDTD simulation on the cloud or loads existing HDF5 data.
        """

        if "TIDY3D_KEY" not in os.environ and "TIDY3D_API_KEY" not in os.environ:
            logging.warning("[Warning] TIDY3D_API_KEY not found in environment. The web run will likely fail unless configured globally.")

        # 1. Construct specific directory name
        param_dir_name = self.get_dir_name()

        # 2. Form child path and ensure it exists
        full_output_dir = Path(output_dir) / param_dir_name
        full_output_dir.mkdir(parents=True, exist_ok=True)
        
        hdf5_target_path = full_output_dir / "raw_simulation_data.hdf5"

        # 3. Cache hit check
        if hdf5_target_path.exists():
            logging.info(f"[Profiler] Cache hit. Loading existing simulation data from {hdf5_target_path}...")
            return td.SimulationData.from_file(str(hdf5_target_path))
        
        # 4. Cache miss: Run on cloud
        logging.info("[Profiler] Cache miss. Uploading architecture to Tidy3D cloud infrastructure...")
        task_name = f"gc_sim_{cfg.wavelength_um}um"
        sim_data = web.run(self.sim, task_name=task_name, verbose=True)
        
        logging.info(f"[Profiler] Execution complete. Serializing data to {hdf5_target_path}...")
        sim_data.to_file(str(hdf5_target_path))
        
        return sim_data
    
    def get_efficiency(
            self,
            output_dir: Path,
        ) -> float:
        """
        Executes the simulation (or loads from cache) and extracts the coupling 
        efficiency for the fundamental mode at the target wavelength.
        """
        sim_data = self._run_sim(output_dir)
        
        logging.info("[Analytics] Extracting scattering coefficients from 'mode monitor'...")
        try:
            # Robust frequency selection using nearest-neighbor interpolation
            mode_amps_array = sim_data['mode monitor'].amps.sel(direction="-", mode_index=0)
            target_freq = td.C_0 / self.config.wavelength_um
            
            efficiency = float(np.abs(mode_amps_array.sel(f=target_freq, method="nearest"))**2)
            peak_loss = 10.0 * np.log10(efficiency + 1e-12)
            
            logging.info(f"[Analytics] Target mode audit completed.\n\
\tCoupling Efficiency: {efficiency*100.0:.2f}%\n\
\tInsertion Loss:      {peak_loss:.2f} dB")          
            return efficiency
            
        except Exception as e:
            logging.error(f"[Analytics Error] Failed to parse mode data cleanly: {e}")
            return 0.0
        
    def get_fields(self, output_dir: Path, show: bool = True) -> None:
        """
        Executes the simulation (or loads from cache) and plots spatial field 
        distribution maps for valid 2D FieldMonitors.
        """
        # 1. Reconstruct the exact child parameter directory where HDF5 lives
        param_dir_name = self.get_dir_name()
        target_dir = Path(output_dir) / param_dir_name
        
        # 2. Run / Load simulation data
        sim_data = self._run_sim(output_dir)
        target_freq = td.C_0 / cfg.wavelength_um

        print(f"[Analytics] Processing spatial field distributions ({dim_str} mode)...")

        # 3. Filter monitors based on dimensionality
        for monitor in self.sim.monitors:
            # Ignore ModeMonitors explicitly (they don't hold spatial field slices)
            if isinstance(monitor, td.ModeMonitor):
                logging.debug(f"[Analytics] Skipping ModeMonitor '{monitor.name}' during spatial field plotting.")
                continue

            if isinstance(monitor, td.FieldMonitor):
                # In 2D, only the y-plane (x-z cross section) has non-zero 2D area
                if not self.is_3d and monitor.name != "field_y_plane":
                    logging.info(f"[Analytics] Skipping 3D-only slice '{monitor.name}' for 2D simulation.")
                    continue

                # Additional sanity check: ensure monitor is actually a 2D slice (exactly one axis size == 0)
                non_zero_axes = sum(1 for axis_len in monitor.size if axis_len > 0)
                if non_zero_axes < 2:
                    logging.warning(f"[Analytics] Skipping '{monitor.name}': degenerate slice with <2 spatial dimensions.")
                    continue

                try:
                    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

                    # Compute total electric field intensity (|E|^2)
                    sim_data.plot_field(
                        field_monitor_name=monitor.name,
                        field_name="E",
                        val="abs^2",
                        f=target_freq,
                        ax=ax
                    )

                    ax.set_title(f"Field Intensity Map (|E|²): {monitor.name} ({dim_str})", fontweight="bold")

                    save_path = target_dir / f"spatial_field_{monitor.name}.png"
                    plt.savefig(save_path, bbox_inches="tight")
                    logging.info(f"[Analytics] Exported field profile slice: {save_path.name}")

                    if show:
                        plt.show()

                    plt.close(fig)

                except Exception as e:
                    logging.error(f"[Analytics Error] Could not render field monitor plot '{monitor.name}': {e}")

        