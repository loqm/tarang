"""
waveguide.dispersion.py

This script allows us to visualize the different regimes of dispersion
within the waveguide, either by varying the geometry, or by varying
the wavelength.

Author: Esraaj Sarkar Gupta,  Divyanshu Tiwari
Date: 2026
"""
from __future__ import annotations
import os
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import Dict, Tuple, List

import tidy3d as td
from tidy3d.plugins.mode import ModeSolver
from matplotlib.patches import Patch

import geometries
import utilities

if TYPE_CHECKING:
    from .waveguide import RidgeWaveguide

logger = logging.getLogger(__name__)
DispersionSweepResult = Dict[float, Tuple[List[float], List[float], List[bool]]]

# Deriving speed of light in um/ps from Tidy3D's native um/s constant
C_UM_PS = td.C_0 / 1e12  # Result: 299.792458

def get_te_fraction(mode_data, m_idx: int, f_idx: int = 2) -> float:
    """
    Computes the TE polarization fraction for a waveguide propagating along X.
    TE dominance is determined by the horizontal transverse field component (Ey).
    """
    Ey = mode_data.Ey.isel(mode_index=m_idx, f=f_idx).values
    Ez = mode_data.Ez.isel(mode_index=m_idx, f=f_idx).values
    Iy = np.sum(np.abs(Ey)**2)
    Iz = np.sum(np.abs(Ez)**2)
    return float(Iy / (Iy + Iz + 1e-10))

def dispersion_in_geometry(
        sweep_heights : list[float],
        sweep_widths : list[float],

        core_medium : td.Medium,
        sub_medium : td.Medium,

        output_dir_name : Path,

        wavelength_um : float,

        preview_geometry : bool = False,
        display_results : bool = True
) -> DispersionSweepResult:
    """
    Evaluates and visualizes Chromatic Dispersion (D) and Single-Mode boundaries 
    across a combined height and width sweep.
    
    Generates a unique figure per fixed width in `sweep_widths`, plotting $D$ 
    against varying heights with shaded operational regime zones.
    """
    freq0 = td.C_0 / wavelength_um
    n_core_guess = float(np.real(np.sqrt(core_medium.eps_model(freq0))))
    n_sub_guess = float(np.real(np.sqrt(sub_medium.eps_model(freq0))))

    # Wavelength span for derivative evaluation
    _index = 5
    wavelengths: np.ndarray = np.linspace(wavelength_um - 0.05, wavelength_um + 0.05, _index*2 + 1)
    freqs = td.C_0 / wavelengths

    if output_dir_name:
        output_path = Path(output_dir_name)
        output_path.mkdir(parents=True, exist_ok=True)

    # Optional geometry check
    if preview_geometry and output_dir_name:
        geom_dir = output_dir_name / Path("geometries")
        geom_dir.mkdir(parents=True, exist_ok=True)

        mean_height = float(np.mean(sweep_heights))
        mean_width = float(np.mean(sweep_widths))

        logger.info(f"     [Dispersion] Plotting mean geometry (Width: {mean_width:.2f} μm, Height: {mean_height:.2f} μm)...")

        preview_waveguide : RidgeWaveguide = geometries.ridge_waveguide.get_passive_simulation(
            width = mean_width,
            height = mean_height,
            length = (5 * mean_width + 5 * mean_height) / 2,
            core_medium = core_medium,
            sub_medium = sub_medium,
            wavelength_um = wavelength_um,
        )

        utilities.sim_plotter.display_3d(
            device = preview_waveguide,
            output_path = geom_dir / Path("example_dispersion_geometry.png"),
            title = "Ridge Waveguide Dispersion Sweep Example Geometry",
            show = display_results,
            save = True
        )

    logger.info(f"Starting dispersion sweep across {len(sweep_widths)} widths and {len(sweep_heights)} heights.")

    all_results: DispersionSweepResult = {}

    # Outer loop over widths: generates one figure per width
    for w in sweep_widths:
        results_h = []
        results_D = []
        results_is_singlemode = []

        print(f"    [Dispersion] Sweeping heights for fixed W = {w:.3f} μm at λ = {wavelength_um:.3f} μm...")

        for h in sweep_heights:
            waveguide_obj : RidgeWaveguide = geometries.ridge_waveguide.get_passive_simulation(
                width = w,
                height = h,
                length = 2,
                core_medium = core_medium,
                sub_medium = sub_medium,
                wavelength_um = wavelength_um
            )

            sim : td.Simulation = waveguide_obj.sim

            mode_spec = td.ModeSpec(num_modes=4, target_neff=n_core_guess)
            plane = td.Box(center=(0, 0, h/2), size=(0, sim.size[1], sim.size[2]))

            solver = ModeSolver(simulation=sim, plane=plane, mode_spec=mode_spec, freqs=freqs)
            data = solver.solve()

            # Locate fundamental TE00 mode index at center frequency
            te00_idx = -1
            for m in range(len(data.n_eff.mode_index)):
                if get_te_fraction(data, m, f_idx=_index) > 0.5:
                    te00_idx = m
                    break

            if te00_idx == -1:
                print(f"  Height {h:.2f} μm (W={w:.2f} μm): Fundamental TE00 mode not found. Skipping.")
                continue

            # Single-Mode Verification (Check if 3rd mode or higher exceeds substrate index)
            neff_at_center = np.real(data.n_eff.isel(f=_index).values)
            n_higher = neff_at_center[2] if len(neff_at_center) > 2 else 0.0
            is_singlemode = bool(n_higher <= (n_sub_guess + 0.001))

            # Extract spectrum for target TE00 mode across sampled wavelengths
            neff_vals = np.real(data.n_eff.isel(mode_index=te00_idx).values)
            sorted_idx = np.argsort(wavelengths)
            srt_wvl = wavelengths[sorted_idx]
            srt_neff = neff_vals[sorted_idx]

            # Numerical Derivatives for Chromatic Dispersion Calculation
            dn_dlam = np.gradient(srt_neff, srt_wvl)
            d2n_dlam2 = np.gradient(dn_dlam, srt_wvl)

            # D = -(λ / c) * (d²n_eff / dλ²) [Converted to ps / (nm · km) via 1e4 factor]
            D_val = float(-(srt_wvl[_index] / C_UM_PS) * d2n_dlam2[_index] * 1e4)

            # Check documentation for a better description of the mathematics used here.
            # I wrote it all just for you!

            results_h.append(h)
            results_D.append(D_val)
            results_is_singlemode.append(is_singlemode)

            sm_status = "SINGLE MODE" if is_singlemode else "MULTIMODE"
            print(f"    [Dispersion] Height {h:.2f} μm: D = {D_val:.2f} ps/(nm·km) [{sm_status}]")

        if not results_h:
            logger.warning(f"   [Dispersion] No valid modes identified for width W = {w:.3f} μm across all heights.")
            continue

        # Plotting Optimization Space with Shaded Performance Regimes
        fig, ax = plt.subplots(figsize=(9, 6), dpi=200)
        ax.plot(results_h, results_D, 'o-', color='black', linewidth=2, label='Dispersion ($D$)')
        ax.axhline(0, color='blue', linestyle='--', linewidth=1.2, label='Zero Dispersion Line')

        # Shade region intervals based on dispersion sign and single-mode qualification
        for i in range(len(results_h) - 1):
            h_start, h_end = results_h[i], results_h[i+1]
            D_avg = (results_D[i] + results_D[i+1]) / 2.0
            is_sm = results_is_singlemode[i] and results_is_singlemode[i+1]

            if D_avg > 0:
                color_zone = 'lightgreen' if is_sm else 'salmon'
            else:
                color_zone = 'lightgray'

            ax.axvspan(h_start, h_end, color=color_zone, alpha=0.3)

        ax.set_xlabel('Waveguide Height ($\mu$m)')
        ax.set_ylabel('Dispersion Parameter $D$ [ps / (nm · km)]')
        ax.set_title(f'Geometric Optimization Space (Fixed W = {w:.3f} μm, λ = {wavelength_um:.3f} μm)', fontsize=12, fontweight='bold')

        legend_elements = [
            Patch(facecolor='lightgray', edgecolor='gray', alpha=0.5, label='Normal Dispersion ($D < 0$)'),
            Patch(facecolor='lightgreen', edgecolor='green', alpha=0.5, label='TARGET: Anomalous + Single Mode'),
            Patch(facecolor='salmon', edgecolor='red', alpha=0.5, label='Anomalous + Multimode Hazard')
        ]
        ax.legend(handles=legend_elements, loc='best')
        ax.grid(True, alpha=0.2)
        fig.tight_layout()

        if output_dir_name:
            path = Path(output_dir_name) / f"dispersion_metrics_W{w:.3f}um.png"
            plt.savefig(path, dpi=200, bbox_inches='tight')
            print(f"  --> Saved plot: {path}")

        if display_results:
            plt.show()
        else:
            plt.close(fig)

        all_results[w] = (results_h, results_D, results_is_singlemode)

    return all_results

