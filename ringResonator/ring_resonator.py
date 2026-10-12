"""
ringResonator.ring_resonator

Wrap ring geometries with simulation caching, modal analysis, and FSR estimates.

Author: Esraaj Sarkar Gupta
Date: 2026

Exposes: BaseResonator, BaseResonator.get_dir_name(), BaseResonator.load_sim_data(),
BaseResonator.get_mode_data(), BaseResonator.get_mode_fluxes(),
BaseResonator.plot_field(), BaseResonator.compute_fsr_nm(),
BaseResonator.compute_fsr_ghz(), BaseResonator.get_group_index(),
BaseResonator.get_fsr_nm(), BaseResonator.get_fsr_ghz(), halfringResonator,
halfringResonator.to_full_ring(), ringResonator, ringResonator.to_half_ring(),
partialRingResonator, partialRingResonator.to_half_ring(),
partialRingResonator.to_full_ring().
"""

from __future__ import annotations
from tarang import get_logger
from tarang._logging import tidy3d_verbose
from utilities.cloud_run import CloudCachedDevice

import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
import tidy3d as td

from tidy3d.plugins.mode import ModeSolver
from tidy3d.plugins.mode.web import run as run_mode_solver


logger = get_logger(__name__)


class BaseResonator(CloudCachedDevice):
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
        """Initialize the device wrapper with its simulation and physical settings.
        Store the parameters used by subsequent analysis and geometry conversion.
        """
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

    def _readable_name(self) -> str:
        """Short, human-readable cache stem; the fingerprint is appended."""
        return (f"{self.__class__.__name__}_R{self.radius:.2f}um_"
                f"gap{self.gap*1000:.0f}nm_{self.wavelength0:.3f}um")

    def _legacy_dir_name(self) -> str:
        """Directory name used before v0.1.0 (verified before reuse)."""
        res_type = self.__class__.__name__

        # Capture span_angle_rad if dealing with a partialRingResonator
        span_angle = getattr(
            self,
            "span_angle_rad",
            0.0,
        )

        sig = (
            f"{self.radius:.4f}_{self.gap:.6f}_{self.wg_width:.4f}_{self.wg_height:.4f}_"
            f"{self.wg_length:.4f}_{self.substrate_height:.4f}_{self.top_cladding_height:.4f}_"
            f"{span_angle:.4f}_{self.core_medium}_{self.substrate_medium}_{self.top_cladding_medium}"
        )
        h = hashlib.md5(sig.encode()).hexdigest()[:8]
        return f"{res_type}_R{self.radius:.2f}um_gap{self.gap*1000:.0f}nm_{self.wavelength0:.3f}um_{h}"

    @staticmethod
    def load_sim_data(filepath: Union[str, Path]) -> td.SimulationData:
        """Resurrects simulation data from a local .hdf5 cache file.
        Raise FileNotFoundError if the supplied cache path does not exist.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(
                f"Are you hallucinating? There is no file at {filepath}"
            )
        return td.SimulationData.from_file(str(filepath))

    # --------------------------------
    # Execution and Analysis Methods
    # --------------------------------

    def get_mode_data(
        self,
        output_dir: Union[str, Path],
        monitor_name: str,
        force_rerun: bool = False,
        verbose: bool = False,
    ) -> Any:
        """Load or run the simulation and return the requested mode monitor.
        Raise an error when the named monitor is absent.
        """
        sim_data = self._run_sim(
            output_dir=output_dir,
            verbose=verbose,
            force_rerun=force_rerun,
        )
        try:
            return sim_data[monitor_name]
        except KeyError:
            raise KeyError(
                f"Monitor '{monitor_name}' not found in simulation monitors."
            )

    def get_mode_fluxes(
        self,
        output_dir: Union[str, Path],
        verbose: bool = False,
        force_rerun: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        """Run or load the simulation and extract fluxes from every mode monitor.
        Report absolute modal powers referenced to the normalized source.
        """
        sim_data = self._run_sim(
            output_dir=output_dir,
            verbose=verbose,
            force_rerun=force_rerun,
        )

        target_freq = td.C_0 / self.wavelength0
        flux_results = {}

        logger.info(
            "[Analytics] Extracting flux coefficients across all mode monitors..."
        )
        for monitor in self.sim.monitors:
            if isinstance(
                monitor,
                td.ModeMonitor,
            ):
                try:
                    # Extract directional amplitudes
                    amps_p = sim_data[monitor.name].amps.sel(
                        direction="+",
                        mode_index=0,
                    )
                    amps_m = sim_data[monitor.name].amps.sel(
                        direction="-",
                        mode_index=0,
                    )

                    # Convert to power (Flux)
                    flux_p = float(
                        np.abs(
                            amps_p.sel(
                                f=target_freq,
                                method="nearest",
                            )
                        )
                        ** 2
                    )
                    flux_m = float(
                        np.abs(
                            amps_m.sel(
                                f=target_freq,
                                method="nearest",
                            )
                        )
                        ** 2
                    )

                    # Evaluate the dominant signal path for summary metrics
                    dominant_flux = max(
                        flux_p,
                        flux_m,
                    )

                    flux_results[monitor.name] = {
                        "absolute_flux": dominant_flux,
                        "efficiency": dominant_flux,
                        "efficiency_percent": dominant_flux * 100.0,
                        "insertion_loss_db": -10.0 * np.log10(dominant_flux + 1e-12),
                        "flux_forward": flux_p,
                        "flux_backward": flux_m,
                    }
                except Exception as e:
                    logger.error(
                        f"[Analytics Error] Failed to parse mode data cleanly for '{monitor.name}': {e}"
                    )

        return flux_results

    def plot_field(
        self,
        output_dir: Union[str, Path],
        monitor_name: str = "coupling_field_map",
        output_path: Optional[Union[str, Path]] = None,
        show: bool = True,
        save: bool = False,
        verbose: bool = False,
        force_rerun: bool = False,
        *,
        plot_dpi: int = 300,  # Resolution of displayed and saved figures (dots per inch).
    ) -> None:
        """Executes the simulation (or loads from cache) and visualizes the E-field
        magnitude.
        """
        sim_data = self._run_sim(
            output_dir=output_dir,
            verbose=verbose,
            force_rerun=force_rerun,
        )

        try:
            field_data = sim_data[monitor_name]
        except KeyError:
            available = [m.name for m in sim_data.simulation.monitors]
            raise ValueError(
                f"Monitor '{monitor_name}' doesn't exist. Available: {available}"
            )

        fig, ax = plt.subplots(
            figsize=(10, 4),
            dpi=plot_dpi,
        )

        E_mag = np.sqrt(
            np.abs(field_data.Ex) ** 2
            + np.abs(field_data.Ey) ** 2
            + np.abs(field_data.Ez) ** 2
        )

        E_mag.isel(f=0).squeeze().plot(
            x="x",
            y="y",
            ax=ax,
            cmap="magma",
            robust=True,
        )

        ax.set_title(
            f"Field Intensity: {monitor_name} (Gap={self.gap*1000:.0f}nm)",
            fontsize=10,
        )
        ax.set_aspect("equal")
        fig.tight_layout()

        if save:
            param_dir = self.get_dir_name()
            target_dir = Path(output_dir) / param_dir
            target_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            if output_path is None or str(output_path).strip() == "":
                path = target_dir / f"spatial_field_{monitor_name}.png"
            else:
                path = Path(output_path)
                if (
                    path.is_dir()
                    or str(output_path).endswith(("/", "\\"))
                    or not path.suffix
                ):
                    path = (
                        path
                        / f"R{self.radius:.2f}um_gap{self.gap*1000:.0f}nm_{monitor_name}.png"
                    )

                if path.parent:
                    path.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

            plt.savefig(
                path,
                dpi=plot_dpi,
                bbox_inches="tight",
            )
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
        wavelength_um: Optional[float] = None,
    ) -> float:
        """Analytical FSR (nm) for a ring resonator of given radius and group index.
        Default to this device's radius and central wavelength when overrides are omitted.
        """
        r = radius_um if radius_um is not None else self.radius
        wvl = wavelength_um if wavelength_um is not None else self.wavelength0
        L_um = 2 * np.pi * r
        fsr_um = (wvl**2) / (n_group * L_um)
        return fsr_um * 1000.0

    def compute_fsr_ghz(
        self,
        n_group: float,
        radius_um: Optional[float] = None,
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
        top_medium = (
            self.top_cladding_medium
            if self.top_cladding_medium is not None
            else air_medium
        )
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

        plane = td.Box(
            center=(0, 0, self.wg_height / 2.0),
            size=(0, size_y, size_z),
        )

        mode_spec = td.ModeSpec(
            num_modes=mode_index + 1,
            target_neff=target_neff,
            precision="double",
            group_index_step=True,
            bend_radius=self.radius if use_bend_radius else None,
            bend_axis=1,
        )

        mode_solver = ModeSolver(
            simulation=dummy_sim,
            plane=plane,
            mode_spec=mode_spec,
            freqs=[freq0],
        )

        data = (
            mode_solver.solve()
            if local
            else run_mode_solver(
                mode_solver,
                verbose=tidy3d_verbose(),
            )
        )
        return float(data.n_group.sel(mode_index=mode_index).item())

    def get_fsr_nm(
        self,
        **kwargs,
    ) -> float:
        """Solves group index and computes FSR in nm.
        Forward solver keyword arguments to get_group_index before conversion.
        """
        n_g = self.get_group_index(**kwargs)
        return self.compute_fsr_nm(
            n_g,
            self.radius,
            self.wavelength0,
        )

    def get_fsr_ghz(
        self,
        **kwargs,
    ) -> float:
        """Solves group index and computes FSR directly in GHz.
        Forward solver keyword arguments to get_group_index before conversion.
        """
        n_g = self.get_group_index(**kwargs)
        return self.compute_fsr_ghz(
            n_g,
            self.radius,
        )


class halfringResonator(BaseResonator):
    def to_full_ring(
        self,
        active: Optional[bool] = None,
    ) -> ringResonator:
        """Construct a full-ring device using this resonator configuration.
        Preserve the current excitation state unless active is explicitly supplied.
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
    def to_half_ring(
        self,
        active: Optional[bool] = None,
    ) -> halfringResonator:
        """Construct a half-ring device using this resonator configuration.
        Preserve the current excitation state unless active is explicitly supplied.
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


class partialRingResonator(BaseResonator):
    def __init__(
        self,
        span_angle_rad: float,
        *args,
        **kwargs,
    ):
        """Initialize the device wrapper with its simulation and physical settings.
        Store the parameters used by subsequent analysis and geometry conversion.
        """
        super().__init__(
            *args,
            **kwargs,
        )
        self.span_angle_rad = span_angle_rad

    def to_half_ring(
        self,
        active: Optional[bool] = None,
    ) -> halfringResonator:
        """Construct a half-ring device using this resonator configuration.
        Preserve the current excitation state unless active is explicitly supplied.
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
        )

    def to_full_ring(
        self,
        active: Optional[bool] = None,
    ) -> ringResonator:
        """Construct a full-ring device using this resonator configuration.
        Preserve the current excitation state unless active is explicitly supplied.
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
        )
