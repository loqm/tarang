"""
grating_coupler.grating_coupler

Defines the GratingConfig dataclass and GratingCoupler wrapper class.

Author: Esraaj Sarkar Gupta, Divyanshu Tiwari
Date: 2026

Exposes: GratingConfig, GratingCoupler, GratingCoupler.from_config(),
GratingCoupler.get_dir_name(), GratingCoupler.is_cached(),
GratingCoupler.get_efficiency(), GratingCoupler.get_fields().
Set ``GratingCoupler.max_cost`` (FlexCredits) to cap cloud runs.
"""

from __future__ import annotations
from tarang import get_logger
from utilities.cloud_run import CloudCachedDevice

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import tidy3d as td

if TYPE_CHECKING:
    from geometries.grating_coupler import get_active_simulation


logger = get_logger(__name__)


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
    box_height: float  # Isolating oxide layer beneath core
    substrate_height: float  # Deep silicon handle

    # Coupler geometry
    etch_depth: float
    period: float
    ff: float  # Fill Factor
    n_teeth: int
    w_grating: float

    # Source
    pol_angle: float

    buffer: float


class GratingCoupler(CloudCachedDevice):
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
        cladding_medium: td.Medium | None = None,  # Defaults to "air" if None
        is_3d: bool = False,
        fine_steps_per_wvl: int = 20,
        coarse_steps_per_wvl: int = 15,
    ):
        """Initialize the grating device from its simulation and configuration.
        Store the output settings used for simulation and efficiency analysis.
        """
        self.sim = sim
        self.config = config

        self.core_medium = core_medium
        self.box_medium = box_medium
        self.substrate_medium = substrate_medium
        self.cladding_medium = cladding_medium

        self.is_3d = is_3d

        self.fine_steps_per_wvl = fine_steps_per_wvl
        self.coarse_steps_per_wvl = coarse_steps_per_wvl

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
        coarse_steps_per_wvl: int = 10,
    ) -> GratingCoupler:
        """Convenience factory method to build and wrap a simulation in one step.
        Build the active grating geometry without submitting a cloud simulation.
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
            coarse_steps_per_wvl=coarse_steps_per_wvl,
        )

    # --------------------------------
    # Execution and Analysis Methods
    # --------------------------------
    _task_prefix = "gc"

    def _readable_name(self) -> str:
        """Short, human-readable cache stem; the full parameter set is stored in
        simulation.json beside the results and in the fingerprint."""
        cfg = self.config
        dim = "3D" if self.is_3d else "2D"
        return (
            f"gc{dim}_p{cfg.period:.4f}_ff{cfg.ff:.4f}_e{cfg.etch_depth:.4f}_"
            f"n{cfg.n_teeth}_sx{cfg.source_x:.4f}_{cfg.wavelength_um:.3f}um"
        )

    def _legacy_dir_name(self) -> str:
        """Directory name used before v0.1.0 (geometry only, no media). Results
        found there are reused only after their stored simulation is verified."""
        dim = "3D" if self.is_3d else "2D"
        cfg = self.config
        return (
            f"{dim}_p_{cfg.period:.4f}_dc_{cfg.ff:.4f}_e_{cfg.etch_depth:.4f}_"
            f"n_{cfg.n_teeth}_w_{cfg.w_grating:.4f}_ch_{cfg.core_height:.4f}_"
            f"clh_{cfg.cladding_height:.4f}_bh_{cfg.box_height:.4f}_sh_{cfg.substrate_height:.4f}_"
            f"t_{cfg.theta_deg:.4f}_mfd_{cfg.mfd:.4f}_sx_{cfg.source_x:.4f}_sg_{cfg.source_gap:.4f}_"
            f"wl_{cfg.wavelength_um:.4f}_pol_{cfg.pol_angle:.4f}_b_{cfg.buffer:.4f}"
        )

    # get_dir_name(), is_cached() and _run_sim() come from CloudCachedDevice:
    # the cache key is the full simulation fingerprint (geometry, media, mesh,
    # sources, monitors), and every cloud run is cost-estimated first.

    def get_efficiency(
        self,
        output_dir: Path,
        verbose: bool = False,
    ) -> float:
        """
        Executes the simulation (or loads from cache) and extracts the coupling
        efficiency for the fundamental mode at the target wavelength.
        """
        sim_data = self._run_sim(
            output_dir,
            verbose=verbose,
        )

        logger.info(
            "[Analytics] Extracting scattering coefficients from 'mode monitor'..."
        )
        try:
            mode_amps_array = sim_data["mode monitor"].amps.sel(
                direction="-",
                mode_index=0,
            )
            target_freq = td.C_0 / self.config.wavelength_um

            efficiency = float(
                np.abs(
                    mode_amps_array.sel(
                        f=target_freq,
                        method="nearest",
                    )
                )
                ** 2
            )
            peak_loss = 10.0 * np.log10(efficiency + 1e-12)

            logger.info(f"[Analytics] Target mode audit completed.\n\
\tCoupling Efficiency: {efficiency*100.0:.2f}%\n\
\tInsertion Loss:      {peak_loss:.2f} dB")
            return efficiency

        except Exception as e:
            logger.error(f"[Analytics Error] Failed to parse mode data cleanly: {e}")
            return 0.0

    def get_fields(
        self,
        output_dir: Path,
        show: bool = True,
        *,
        plot_dpi: int = 300,  # Resolution of displayed and saved figures (dots per inch).
    ) -> None:
        """
        Executes the simulation (or loads from cache) and plots spatial field
        distribution maps for valid 2D FieldMonitors.
        """
        cfg = self.config
        dim_str = "3D" if self.is_3d else "2D"

        param_dir_name = self.get_dir_name()
        target_dir = Path(output_dir) / param_dir_name

        sim_data = self._run_sim(
            output_dir,
            verbose=False,
        )
        target_freq = td.C_0 / cfg.wavelength_um

        logger.info(
            f"[Analytics] Processing spatial field distributions ({dim_str} mode)..."
        )

        for monitor in self.sim.monitors:
            if isinstance(
                monitor,
                td.ModeMonitor,
            ):
                logger.debug(
                    f"[Analytics] Skipping ModeMonitor '{monitor.name}' during spatial field plotting."
                )
                continue

            if isinstance(
                monitor,
                td.FieldMonitor,
            ):
                if not self.is_3d and monitor.name != "field_y_plane":
                    logger.info(
                        f"[Analytics] Skipping 3D-only slice '{monitor.name}' for 2D simulation."
                    )
                    continue

                non_zero_axes = sum(1 for axis_len in monitor.size if axis_len > 0)
                if non_zero_axes < 2:
                    logger.warning(
                        f"[Analytics] Skipping '{monitor.name}': degenerate slice with <2 spatial dimensions."
                    )
                    continue

                try:
                    fig, ax = plt.subplots(
                        figsize=(10, 5),
                        dpi=plot_dpi,
                    )

                    sim_data.plot_field(
                        field_monitor_name=monitor.name,
                        field_name="E",
                        val="abs^2",
                        f=target_freq,
                        ax=ax,
                    )

                    ax.set_title(
                        f"Field Intensity Map (|E|²): {monitor.name} ({dim_str})",
                        fontweight="bold",
                    )

                    save_path = target_dir / f"spatial_field_{monitor.name}.png"
                    plt.savefig(
                        save_path,
                        bbox_inches="tight",
                        dpi=plot_dpi,
                    )
                    logger.info(
                        f"[Analytics] Exported field profile slice: {save_path.name}"
                    )

                    if show:
                        plt.show()

                    plt.close(fig)

                except Exception as e:
                    logger.error(
                        f"[Analytics Error] Could not render field monitor plot '{monitor.name}': {e}"
                    )
