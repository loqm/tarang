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

from tidy3d.plugins.mode import ModeSolver
from tidy3d.plugins.mode.web import run as run_mode_solver

if TYPE_CHECKING:
    from geometries.half_ring_resonator import (
        get_active_simulation as get_active_half_sim,
        get_passive_simulation as get_passive_half_sim,
    )
    from geometries.full_ring_resonator import (
        get_active_simulation as get_active_full_sim,
        get_passive_simulation as get_passive_full_sim,
    )
    from geometries.partial_ring_resonator import (
        get_active_simulation as get_active_partial_sim,
        get_passive_simulation as get_passive_partial_sim,
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
        available_monitors = [m.name for m in sim_data.simulation.monitors]
        if monitor_name not in available_monitors:
            raise ValueError(
                f"Monitor '{monitor_name}' doesn't exist in simulation data. "
                f"Available monitors: {available_monitors}"
            )
            
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
                raise ValueError("Cannot save plot: 'save' is True, but no valid 'output_path' was provided.")
            
            path = Path(output_path)
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

    # ---- The "Physicsy" Object Methods ---- #
    def compute_fsr_nm(
        self, 
        n_group: float, 
        radius_um: Optional[float] = None, 
        wavelength_um: Optional[float] = None
    ) -> float:
        """Analytical FSR (nm) for a ring resonator of given radius and group index."""
        r = radius_um if radius_um is not None else self.radius
        wvl = wavelength_um if wavelength_um is not None else self.wavelength0
        L_um = 2 * np.pi * r
        fsr_um = (wvl ** 2) / (n_group * L_um)
        return fsr_um * 1000.0

    def compute_fsr_ghz(
        self, 
        n_group: float, 
        radius_um: Optional[float] = None
    ) -> float:
        """
        Calculates FSR directly in GHz from group index and cavity length.
        Defaults to self.radius if radius_um is not explicitly provided.
        """
        r = radius_um if radius_um is not None else self.radius
        L_um = 2 * np.pi * r
        
        fsr_hz = td.C_0 / (n_group * L_um)
        return fsr_hz / 1e9

    def get_group_index(
        self,
        target_neff: Optional[float] = None,
        mode_index: int = 0,
        use_bend_radius: bool = True,
        local: bool = False,
        domain_padding: float = 2.0,
    ) -> float:
        """
        Solves for the fundamental mode's group index at wavelength0 using a
        standalone mode solver on the waveguide cross-section.
        """
        freq0 = td.C_0 / self.wavelength0

        air_medium = td.Medium(permittivity=1.0)
        top_medium = self.top_cladding_medium if self.top_cladding_medium is not None else air_medium
        top_height = self.top_cladding_height if self.top_cladding_height > 0.0 else 2.0

        size_y = self.wg_width + 2 * domain_padding
        size_z = self.substrate_height + self.wg_height + top_height

        z_min = -self.substrate_height
        substrate = td.Structure(
            geometry=td.Box(
                center=(0, 0, z_min / 2.0),
                size=(td.inf, td.inf, self.substrate_height),
            ),
            medium=self.substrate_medium,
            name="mode_solve_substrate",
        )

        cladding = td.Structure(
            geometry=td.Box(
                center=(0, 0, top_height / 2.0),
                size=(td.inf, td.inf, top_height),
            ),
            medium=top_medium,
            name="mode_solve_cladding",
        )

        core = td.Structure(
            geometry=td.Box(
                center=(0, 0, self.wg_height / 2.0),
                size=(td.inf, self.wg_width, self.wg_height),
            ),
            medium=self.core_medium,
            name="mode_solve_core",
        )

        dummy_sim = td.Simulation(
            center=(0, 0, (self.wg_height / 2.0)),
            size=(1.0, size_y, size_z),
            grid_spec=td.GridSpec.auto(
                min_steps_per_wvl=self.fine_steps_per_wvl,
                wavelength=self.wavelength0,
            ),
            structures=[substrate, cladding, core],
            run_time=1e-12,
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
            medium=air_medium,
        )

        plane = td.Box(center=(0, 0, self.wg_height / 2.0), size=(0, size_y, size_z))

        mode_spec = td.ModeSpec(
            num_modes=mode_index + 1,
            target_neff=target_neff,
            precision="double",
            group_index_step=True,
            bend_radius=self.radius if use_bend_radius else None,
            bend_axis=0,
        )

        mode_solver = ModeSolver(
            simulation=dummy_sim,
            plane=plane,
            mode_spec=mode_spec,
            freqs=[freq0],
        )

        data = mode_solver.solve() if local else run_mode_solver(mode_solver)
        return float(data.n_group.sel(mode_index=mode_index).item())

    def get_fsr_nm(self, **kwargs) -> float:
        """Solves group index and computes FSR in nm."""
        n_g = self.get_group_index(**kwargs)
        return self.compute_fsr_nm(n_g, self.radius, self.wavelength0)

    def get_fsr_ghz(self, **kwargs) -> float:
        """Solves group index and computes FSR directly in GHz."""
        n_g = self.get_group_index(**kwargs)
        return self.compute_fsr_ghz(n_g, self.radius)


class halfringResonator(BaseResonator):
    def to_full_ring(self, active: Optional[bool] = None) -> ringResonator:
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


class partialRingResonator(BaseResonator):
    def __init__(self, span_angle_rad: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.span_angle_rad = span_angle_rad

    def to_half_ring(self, active: Optional[bool] = None) -> halfringResonator:
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
            # We don't want to directly pass the domain length
            #wg_length=self.wg_length
        )

    def to_full_ring(self, active: Optional[bool] = None) -> ringResonator:
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
            # We don't want to directly pass the domain length
            # wg_length=self.wg_length,
        )