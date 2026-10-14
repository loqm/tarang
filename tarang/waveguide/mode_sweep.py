"""
tarang.waveguide.mode_sweep.py

Carries out modal analysis on specified geometries of specific materials.
Saves plots into a specified results directory.

Author: Divyanshu Tiwari, Esraaj Sarkar Gupta
Date: 2026

Exposes: modes_in_geometry().
"""

from __future__ import annotations
from tarang import get_logger
from tarang._compat import renamed_params
from tarang.utilities.validation import positive as _positive, values as _values, integer as _integer

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt

import tidy3d as td
from tidy3d.constants import C_0

from tarang import geometries
from tarang import utilities

if TYPE_CHECKING:
    from .waveguide import RidgeWaveguide


logger = get_logger(__name__)


@renamed_params(sub_medium="box_medium", substrate_height="box_height", fine_min_steps_per_wvl="fine_steps_per_wvl")
def modes_in_geometry(
    sweep_heights: list[float],
    sweep_widths: list[float],
    core_medium: td.Medium,
    box_medium: td.Medium,
    output_dir_name: Path,
    wavelength_um: float,
    box_height: float,
    num_modes: int = 8,
    preview_geometry: bool = False,
    display_results: bool = True,
    *,
    top_cladding_medium: td.Medium | None = None,  # None = air top cladding
    top_cladding_height: float = 0.0,  # Top cladding thickness (um); used if a medium is given
    fine_steps_per_wvl: int = 50,  # Core mesh; >= 50 for converged n_eff and D (local solver, no FlexCredits)
    plot_dpi: int = 150,  # Resolution of displayed and saved figures (dots per inch).
) -> None:
    """
    Generates width vs effective index plots at every width sweep point,
    with a figure for every height sweep point.
    """

    sweep_heights = _values("sweep_heights", sweep_heights)
    sweep_widths = _values("sweep_widths", sweep_widths)

    wavelength_um = _positive("wavelength_um", wavelength_um)
    box_height = _positive("box_height", box_height)
    plot_dpi = int(_positive("plot_dpi", plot_dpi))
    num_modes = _integer("num_modes", num_modes)

    output_dir = Path(output_dir_name)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    freq0 = C_0 / wavelength_um

    n_sub = float(np.real(box_medium.eps_model(freq0)) ** 0.5)
    if top_cladding_medium is not None and top_cladding_height > 0:
        # Guiding cutoff is the highest surrounding index (substrate or top cladding)
        n_sub = max(n_sub, float(np.real(top_cladding_medium.eps_model(freq0)) ** 0.5))
    n_core = float(np.real(core_medium.eps_model(freq0)) ** 0.5)
    wl_nm = int(wavelength_um * 1000)

    if preview_geometry:
        geom_dir = output_dir / "geometries"
        geom_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        mean_height = float(np.mean(sweep_heights))
        mean_width = float(np.mean(sweep_widths))

        logger.info(
            f"\n[Mode Analysis] Plotting mean geometry (Width: {mean_width:.2f} um, Height: {mean_height:.2f} um)..."
        )

        preview_waveguide: RidgeWaveguide = (
            geometries.ridge_waveguide.get_passive_simulation(
                width=mean_width,
                height=mean_height,
                length=(5 * mean_width + 5 * mean_height) / 2,
                core_medium=core_medium,
                box_medium=box_medium,
                wavelength_um=wavelength_um,
                box_height=box_height,
                top_cladding_medium=top_cladding_medium,
                top_cladding_height=(top_cladding_height if top_cladding_medium is not None else 0.0),
                fine_steps_per_wvl=fine_steps_per_wvl,
            )
        )

        utilities.sim_plotter.display_3d(
            device=preview_waveguide,
            output_path=output_dir_name / Path("example_geometry.png"),
            title="Ridge Waveguide Mode Sweep Example Geometry",
            show=True,
            save=True,
            plot_dpi=plot_dpi,
        )

    for height in sweep_heights:
        logger.info(f"\n[Mode Analysis] | Height: {height:.2f} um at {wl_nm} nm")

        te_tracks: dict = {
            i: {
                "widths": [],
                "neffs": [],
            }
            for i in range(num_modes)
        }
        tm_tracks: dict = {
            i: {
                "widths": [],
                "neffs": [],
            }
            for i in range(num_modes)
        }

        for width in sweep_widths:
            waveguide_obj: RidgeWaveguide = (
                geometries.ridge_waveguide.get_passive_simulation(
                    width=width,
                    height=height,
                    length=2,
                    core_medium=core_medium,
                    box_medium=box_medium,
                    wavelength_um=wavelength_um,
                    box_height=box_height,
                    top_cladding_medium=top_cladding_medium,
                    top_cladding_height=(top_cladding_height if top_cladding_medium is not None else 0.0),
                    fine_steps_per_wvl=fine_steps_per_wvl,
                )
            )

            # Get mode data for the current waveguide geometry
            modes_data = waveguide_obj.get_modes(num_modes=num_modes)

            current_te_neffs: list = []
            current_tm_neffs: list = []

            for mode in modes_data["modes"]:
                if mode["polarization"] == "TE":
                    current_te_neffs.append(mode["neff"])
                else:
                    current_tm_neffs.append(mode["neff"])

            for rank, neff in enumerate(current_te_neffs):
                if rank < num_modes:
                    te_tracks[rank]["widths"].append(width)
                    te_tracks[rank]["neffs"].append(neff)

            for rank, neff in enumerate(current_tm_neffs):
                if rank < num_modes:
                    tm_tracks[rank]["widths"].append(width)
                    tm_tracks[rank]["neffs"].append(neff)

        # ---- Plotting ---- #
        plt.figure(
            figsize=(12, 6),
            dpi=plot_dpi,
        )

        plt.axhline(
            y=n_core,
            color="black",
            linestyle="-",
            linewidth=1.5,
            label="Core",
        )
        plt.axhline(
            y=n_sub,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="Substrate",
        )
        plt.fill_between(
            sweep_widths,
            n_sub,
            n_core,
            color="purple",
            alpha=0.03,
            label="Guided Domain",
        )

        colors = plt.get_cmap("tab10")

        for rank in range(num_modes):
            if te_tracks[rank]["widths"]:
                plt.plot(
                    te_tracks[rank]["widths"],
                    te_tracks[rank]["neffs"],
                    color=colors(rank % 10),
                    marker="o",
                    linestyle="-",
                    markersize=5,
                    label=f"TE$_{rank}$",
                    zorder=3,
                )

        for rank in range(num_modes):
            if tm_tracks[rank]["widths"]:
                plt.plot(
                    tm_tracks[rank]["widths"],
                    tm_tracks[rank]["neffs"],
                    color=colors((rank + 4) % 10),
                    marker="s",
                    linestyle="--",
                    markersize=5,
                    label=f"TM$_{rank}$",
                    zorder=3,
                )

        plt.title(
            rf"Dispersion Profile | (h={height:.2f} $\mu$m) at $\lambda$={wavelength_um*1000:.0f}nm",
            fontsize=12,
            pad=12,
        )
        plt.xlabel(
            r"Waveguide Width ($\mu$m)",
            fontsize=11,
        )
        plt.ylabel(
            "Effective Index ($n_{eff}$)",
            fontsize=11,
        )

        plt.ylim(
            n_sub - 0.05,
            n_core + 0.05,
        )
        plt.grid(
            True,
            linestyle=":",
            alpha=0.5,
        )
        plt.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=9,
        )

        plt.tight_layout()

        file_name = f"{wl_nm}nm_wgW_{width:.2f}_wgH_{height:.2f}um.png"
        plt.savefig(
            output_dir / file_name,
            dpi=plot_dpi,
        )

        if display_results:
            plt.show()
        plt.close()

        logger.info(f"Visualized distribution out to {file_name}")
