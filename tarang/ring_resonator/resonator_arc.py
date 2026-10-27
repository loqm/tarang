"""
tarang.ring_resonator.resonator_arc

Author: Esraaj Sarkar Gupta
Date: 2026

Role: ANALYSIS WRAPPER for a waveguide bend. Holds a built Simulation, runs it
through the verified cloud cache, and extracts transmission and bend loss. The
Simulation itself is built by ``tarang.geometries.resonator_arc`` (same file name,
different role: geometries/* builds, the device packages analyse).

Exposes: ResonatorArc, ResonatorArc.get_dir_name(), ResonatorArc.get_mode_data(),
ResonatorArc.get_efficiency(), ResonatorArc.get_loss(), ResonatorArc.plot_field().
"""

from __future__ import annotations
from tarang import get_logger
from tarang._compat import deprecated_module_names, renamed_params

from pathlib import Path
from typing import Optional, Union, Any
import hashlib

import matplotlib.pyplot as plt
import numpy as np
import tidy3d as td

from tarang.ring_resonator.ring_resonator import BaseResonator
from tarang.utilities.cloud_run import monitor_data

logger = get_logger(__name__)


class ResonatorArc(BaseResonator):
    _renamed_attrs = {**BaseResonator._renamed_attrs, "cladding_medium": "top_cladding_medium"}

    @renamed_params(substrate_medium="box_medium", cladding_medium="top_cladding_medium", wavelength0="wavelength_um")
    def __init__(
        self,
        sim: td.Simulation,
        radius: float,
        wg_width: float,
        wg_height: float,
        core_medium: td.Medium,
        top_cladding_medium: td.Medium,
        box_medium: td.Medium,
        bend_angle_deg: float,
        fine_steps_per_wvl: float,
        coarse_steps_per_wvl: float,
        wavelength_um: float,
    ):
        """Initialize the device wrapper with its simulation and physical settings.
        Store the parameters used by subsequent analysis and geometry conversion.
        """
        super().__init__(
            sim=sim,
            radius=radius,
            gap=0.0,
            wg_width=wg_width,
            wg_height=wg_height,
            core_medium=core_medium,
            box_medium=box_medium,
            box_height=2.5,  # Fixed to match Z-bounds of passive sim (-2.5 to 1.5)
            top_cladding_medium=top_cladding_medium,
            top_cladding_height=1.5,  # Fixed to match Z-bounds
            fine_steps_per_wvl=fine_steps_per_wvl,
            coarse_steps_per_wvl=coarse_steps_per_wvl,
            wg_length=0.0,
            wavelength_um=wavelength_um,
        )
        self.bend_angle_deg = bend_angle_deg
        self.top_cladding_medium = top_cladding_medium
        self.mode_direction = "+"

    _task_prefix = "arc_sim"

    def _readable_name(self) -> str:
        """Short, human-readable cache stem; the fingerprint is appended."""
        return f"arc_R{self.radius:.2f}um_deg{self.bend_angle_deg:.1f}_{self.wavelength_um:.3f}um"

    def _legacy_dir_name(self) -> str:
        """Directory name used before v0.1.0 (verified before reuse)."""
        sig = f"{self.wg_width:.4f}_{self.wg_height:.4f}_{self.core_medium}_{self.top_cladding_medium}_{self.box_medium}"
        h = hashlib.md5(sig.encode(), usedforsecurity=False).hexdigest()[:8]
        return f"arc_R{self.radius:.2f}um_deg{self.bend_angle_deg:.1f}_{self.wavelength_um:.3f}um_{h}"

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
        """Load or run the simulation and return the requested mode monitor.
        Raise an error when the named monitor is absent.
        """
        sim_data = self._run_sim(
            output_dir=output_dir,
            verbose=verbose,
            force_rerun=force_rerun,
        )

        return monitor_data(sim_data, monitor_name)

    def get_efficiency(
        self,
        output_dir: Union[str, Path],
        monitor_name: str = "Mode Monitor",
        verbose: bool = False,
        force_rerun: bool = False,
    ) -> float:
        """Compute output-to-input fundamental modal power transmission.
        Select matching frequencies and the outgoing arc propagation direction.
        """
        sim_data = self._run_sim(
            output_dir=output_dir,
            verbose=verbose,
            force_rerun=force_rerun,
        )

        target_freq = td.C_0 / self.wavelength_um
        output_monitor = sim_data[monitor_name]
        normal_axis = output_monitor.monitor.size.index(0.0)

        theta = np.deg2rad(self.bend_angle_deg)
        tangent = (np.cos(theta), -np.sin(theta), 0.0)
        direction = "+" if tangent[normal_axis] > 0 else "-"

        a_in = (
            sim_data["Mode Monitor In"]
            .amps.sel(
                direction="+",
                mode_index=0,
            )
            .sel(
                f=target_freq,
                method="nearest",
            )
        )

        a_out = output_monitor.amps.sel(
            direction=direction,
            mode_index=0,
        ).sel(
            f=target_freq,
            method="nearest",
        )

        if float(a_in.f) != float(a_out.f):
            raise ValueError("Input and output frequencies do not match.")

        p_in = float(np.abs(a_in) ** 2)
        p_out = float(np.abs(a_out) ** 2)

        if not np.isfinite(p_in) or p_in <= 0:
            raise ValueError("Input modal power must be finite and positive.")
        if not np.isfinite(p_out):
            raise ValueError("Output modal power must be finite.")

        return p_out / p_in

    def get_loss(
        self,
        output_dir: Union[str, Path],
        monitor_name: str = "Mode Monitor",
        verbose: bool = False,
        force_rerun: bool = False,
        per_cm: bool = False,
    ) -> float:
        """Convert modal transmission into positive insertion loss in decibels.
        Optionally normalize by the arc length to report loss per centimeter.
        """
        efficiency = self.get_efficiency(
            output_dir=output_dir,
            monitor_name=monitor_name,
            verbose=verbose,
            force_rerun=force_rerun,
        )

        loss_db = float("inf") if efficiency == 0 else -10 * np.log10(efficiency)

        if per_cm:
            arc_length_cm = self.radius * np.deg2rad(self.bend_angle_deg) * 1e-4
            if arc_length_cm <= 0:
                raise ValueError("Arc length must be positive.")
            return float(loss_db / arc_length_cm)

        return float(loss_db)

    def plot_field(
        self,
        output_dir: Union[str, Path],
        monitor_name: str = "field_map",
        output_path: Optional[Union[str, Path]] = None,
        show: bool = True,
        save: bool = False,
        verbose: bool = False,
        force_rerun: bool = False,
        *,
        plot_dpi: int = 300,  # Resolution of displayed and saved figures (dots per inch).
    ) -> None:
        """Render the electric-field intensity from the requested field monitor.
        Optionally save the plot and display it after loading or running simulation data.
        """
        sim_data = self._run_sim(
            output_dir=output_dir,
            verbose=verbose,
            force_rerun=force_rerun,
        )

        field_data = monitor_data(sim_data, monitor_name)

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
            f"Field Intensity: {monitor_name} (R={self.radius}μm, {self.bend_angle_deg:.1f}°)",
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
                        / f"bend_{self.bend_angle_deg:.1f}deg_R{self.radius:.2f}_{monitor_name}.png"
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


# Pre-v0.1.0 class names keep working (with a DeprecationWarning) until v0.2.0.
__getattr__ = deprecated_module_names(globals(), {'resonatorArc': 'ResonatorArc'})
