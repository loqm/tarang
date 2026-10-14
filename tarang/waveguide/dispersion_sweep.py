"""Dispersion sweep.

Exposes: dispersion_in_geometry().
"""

from __future__ import annotations
from tarang import get_logger
from tarang._compat import renamed_params
from tarang.utilities.validation import positive as _positive, values as _values

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import tidy3d as td

from tarang import geometries
from tarang import utilities

from .waveguide import WaveguideModeError

if TYPE_CHECKING:
    from .waveguide import RidgeWaveguide

DispersionSweepResult = Dict[float, Tuple[List[float], List[float], List[bool]]]


logger = get_logger(__name__)


@renamed_params(sub_medium="box_medium", substrate_height="box_height", fine_min_steps_per_wvl="fine_steps_per_wvl")
def dispersion_in_geometry(
    sweep_heights: list[float],
    sweep_widths: list[float],
    core_medium: td.Medium,
    box_medium: td.Medium,
    output_dir_name: Path,
    wavelength_um: float,
    box_height: float,
    preview_geometry: bool = False,
    display_results: bool = True,
    *,
    top_cladding_medium: td.Medium | None = None,  # None = air top cladding
    top_cladding_height: float = 0.0,  # Top cladding thickness (um); used if a medium is given
    fine_steps_per_wvl: int = 50,  # Core mesh; >= 50 for converged n_eff and D (local solver, no FlexCredits)
    plot_dpi: int = 200,  # Resolution of displayed and saved figures (dots per inch).
) -> DispersionSweepResult:
    """
    Evaluates and visualizes Chromatic Dispersion (D) and Single-Mode boundaries
    across a combined height and width sweep (sweeping width for each fixed height).
    """

    sweep_heights = _values("sweep_heights", sweep_heights)
    sweep_widths = _values("sweep_widths", sweep_widths)

    wavelength_um = _positive("wavelength_um", wavelength_um)
    box_height = _positive("box_height", box_height)
    plot_dpi = int(_positive("plot_dpi", plot_dpi))

    if output_dir_name:
        output_path = Path(output_dir_name)
        output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

    if preview_geometry and output_dir_name:
        geom_dir = output_dir_name / Path("geometries")
        geom_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        mean_height = float(np.mean(sweep_heights))
        mean_width = float(np.mean(sweep_widths))

        logger.info(
            f" Plotting mean geometry (Width: {mean_width:.2f} μm, Height: {mean_height:.2f} μm)..."
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
            output_path=geom_dir / Path("example_dispersion_geometry.png"),
            title="Ridge Waveguide Dispersion Sweep Example Geometry",
            show=display_results,
            save=True,
            plot_dpi=plot_dpi,
        )

    logger.info(
        f"Starting dispersion sweep across {len(sweep_heights)} heights and {len(sweep_widths)} widths."
    )

    all_results: DispersionSweepResult = {}

    for h in sweep_heights:
        results_w: list = []
        results_D: list = []
        results_is_singlemode: list = []

        logger.info(
            f"Sweeping widths for fixed H = {h:.3f} μm at λ = {wavelength_um:.3f} μm..."
        )

        for w in sweep_widths:
            waveguide_obj: RidgeWaveguide = (
                geometries.ridge_waveguide.get_passive_simulation(
                    width=w,
                    height=h,
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

            # Get dispersion data for the current waveguide geometry,
            # 11 points spanning 0.1um is the default.
            try:
                disp_data = waveguide_obj.get_dispersion(
                    delta_wvl_um=0.1,
                    num_points=11,
                )
            except WaveguideModeError as e:
                # get_dispersion() raises (it never returns an "error" key), so a failed
                # geometry is logged and skipped instead of aborting the whole sweep.
                logger.warning(f"  Width {w:.2f} μm (H={h:.2f} μm): {e} Skipping.")
                continue

            D_val = disp_data["dispersion_ps_nm_km"]
            is_singlemode = disp_data["is_singlemode"]

            results_w.append(w)
            results_D.append(D_val)
            results_is_singlemode.append(is_singlemode)

            sm_status = "SINGLE MODE" if is_singlemode else "MULTIMODE"
            logger.info(
                f"Width {w:.2f} μm: D = {D_val:.2f} ps/(nm·km) [{sm_status}]"
            )

        if not results_w:
            logger.warning(
                f"No valid modes identified for height H = {h:.3f} μm across all widths."
            )
            continue

        # Plotting
        fig, ax = plt.subplots(
            figsize=(9, 6),
            dpi=plot_dpi,
        )
        ax.plot(
            results_w,
            results_D,
            "o-",
            color="black",
            linewidth=2,
            label="Dispersion ($D$)",
        )
        ax.axhline(
            0,
            color="blue",
            linestyle="--",
            linewidth=1.2,
            label="Zero Dispersion Line",
        )

        for i in range(len(results_w) - 1):
            w_start, w_end = results_w[i], results_w[i + 1]
            D_avg = (results_D[i] + results_D[i + 1]) / 2.0
            is_sm = results_is_singlemode[i] and results_is_singlemode[i + 1]

            if D_avg > 0:
                color_zone = "lightgreen" if is_sm else "salmon"
            else:
                color_zone = "lightgray"

            ax.axvspan(
                w_start,
                w_end,
                color=color_zone,
                alpha=0.3,
            )

        ax.set_xlabel(r"Waveguide Width ($\mu$m)")
        ax.set_ylabel("Dispersion Parameter $D$ [ps / (nm · km)]")
        ax.set_title(
            f"Geometric Optimization Space (Fixed H = {h:.3f} μm, λ = {wavelength_um:.3f} μm)",
            fontsize=12,
            fontweight="bold",
        )

        legend_elements = [
            Patch(
                facecolor="lightgray",
                edgecolor="gray",
                alpha=0.5,
                label="Normal Dispersion ($D < 0$)",
            ),
            Patch(
                facecolor="lightgreen",
                edgecolor="green",
                alpha=0.5,
                label="TARGET: Anomalous + Single Mode",
            ),
            Patch(
                facecolor="salmon",
                edgecolor="red",
                alpha=0.5,
                label="Anomalous + Multimode Hazard",
            ),
        ]
        ax.legend(
            handles=legend_elements,
            loc="best",
        )
        ax.grid(
            True,
            alpha=0.2,
        )
        fig.tight_layout()

        if output_dir_name:
            path = Path(output_dir_name) / f"dispersion_metrics_H{h:.3f}um.png"
            plt.savefig(
                path,
                dpi=plot_dpi,
                bbox_inches="tight",
            )
            logger.info(f"  --> Saved plot: {path}")

        if display_results:
            plt.show()
        else:
            plt.close(fig)

        all_results[h] = (results_w, results_D, results_is_singlemode)

    return all_results
