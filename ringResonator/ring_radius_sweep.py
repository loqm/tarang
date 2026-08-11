"""
ringSearches.ring_radius_sweep

> Includes a numerical experiment to compute intrinsic Q factor,
along with the FSR. This is done by simulating bends of 90 degrees and
180 degrees and computing the loss coefficient.
> Group index is computed using results from the numerical index.

Author: Esraaj Sarkar Gupta, Divyanshu Tiwari
Date: August, 2026
"""

import json
from pathlib import Path
from typing import Optional, Union, Sequence
import numpy as np
import logging

import matplotlib.pyplot as plt
import tidy3d as td

from geometries.resonator_arc import get_active_simulation

logger = logging.getLogger(__name__)


def compute_fsr_nm(n_group: float, radius_um: float, wavelength_um: float) -> float:
    """Analytical FSR (nm) for a ring resonator of given radius and group index."""
    L_um = 2 * np.pi * radius_um
    fsr_um = (wavelength_um ** 2) / (n_group * L_um)
    return fsr_um * 1000.0


def ring_radius_sweep(
    radii: Sequence[float],
    target_wavelength_um: float,

    wg_width: float,
    wg_height: float,

    base_output_dir: Path | str,

    core_medium: td.Medium,
    substrate_medium: td.Medium,
    cladding_medium: td.Medium = td.Medium(permittivity=1.0),
    
    scattering_loss_dm_cm: float = 2.0, # Emperical results

    fine_steps_per_wvl: int = 20,
    coarse_steps_per_wvl: int = 10,

    output_path: Optional[Union[str, Path]] = None,
    
    show: bool = True,
    save: bool = False,

    force_rerun: bool = False
) -> dict:
    """
    Constructs 90-degree and 180-degree arc simulations dynamically for a sweep of radii.
    Extracts pure bend loss (dB/cm), Total Loss, Q_intrinsic, and FSR.
    Returns a dictionary mapping radius to its fundamental physical properties.
    """
    base_output_dir = Path(base_output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = base_output_dir / "intrinsic_physics_manifest.json"
    
    results_manifest: dict = {}
    
    if manifest_path.exists() and not force_rerun:
        logger.info(f"[Physics] Cache hit at {manifest_path}. Bypassing simulation runs.")
        with open(manifest_path, "r") as f:
            results_manifest = json.load(f)
    else:
        logger.info(f"[Ring Radius Sweep] Initializing automated cutback sweep across {len(radii)} radii...")
        
        for radius in radii:
            try:
                # 1. Build active simulation objects dynamically
                arc_90 = get_active_simulation(
                    wavelength0=target_wavelength_um,
                    
                    radius=radius,
                    
                    wg_width=wg_width,
                    wg_height=wg_height,
                    
                    core_medium=core_medium,
                    substrate_medium=substrate_medium,
                    cladding_medium=cladding_medium,
                    
                    bend_angle_deg=90,
                    
                    fine_steps_per_wvl=fine_steps_per_wvl,
                    coarse_steps_per_wvl=coarse_steps_per_wvl
                )
                
                arc_180 = get_active_simulation(
                    wavelength0=target_wavelength_um,
                    
                    radius=radius,
                    
                    wg_width=wg_width,
                    wg_height=wg_height,
                    
                    core_medium=core_medium,
                    substrate_medium=substrate_medium,
                    cladding_medium=cladding_medium,
                    
                    bend_angle_deg=180,
                    
                    fine_steps_per_wvl=fine_steps_per_wvl,
                    coarse_steps_per_wvl=coarse_steps_per_wvl
                )
                
                # Documentation for the mathematics provided in the official documentation for TARANG
                # 2. Retrieve mode monitor results via resonatorArc method
                mon_90 = arc_90.get_mode_data(output_dir=base_output_dir, force_rerun=force_rerun)
                mon_180 = arc_180.get_mode_data(output_dir=base_output_dir, force_rerun=force_rerun)
                
                # 3. Process power transmission & losses
                T_90 = float(np.abs(mon_90.amps.sel(direction="-", mode_index=0).isel(f=1).item())**2)
                T_180 = float(np.abs(mon_180.amps.sel(direction="-", mode_index=0).isel(f=1).item())**2)
                
                loss_90_dB = -10 * np.log10(T_90)
                loss_180_dB = -10 * np.log10(T_180)
                
                L_90_cm = (0.5 * np.pi * radius) * 1e-4
                L_180_cm = (1.0 * np.pi * radius) * 1e-4
                
                fitted_loss_db_cm = (loss_180_dB - loss_90_dB) / (L_180_cm - L_90_cm)
                if fitted_loss_db_cm < 0: 
                    fitted_loss_db_cm = 0.0 
                
                total_loss_db_cm = fitted_loss_db_cm + scattering_loss_dm_cm
                
                # 4. Group index & Intrinsic Q derivation
                freqs = mon_90.amps.coords['f'].values
                n_effs = np.real(mon_90.n_eff.sel(mode_index=0).values)
                wavelengths = td.C_0 / freqs

                dn_dlam = (n_effs[0] - n_effs[-1]) / (wavelengths[0] - wavelengths[-1])
                n_g = n_effs[1] - target_wavelength_um * dn_dlam
                
                alpha_m = (total_loss_db_cm * 100) / 4.343 
                q_intrinsic = (2 * np.pi * n_g) / ((target_wavelength_um * 1e-6) * alpha_m)
                fsr = compute_fsr_nm(n_g, radius, target_wavelength_um)
                
                results_manifest[str(radius)] = {
                    "radius_um": radius,
                    "bend_loss_db_cm": fitted_loss_db_cm,
                    "total_loss_db_cm": total_loss_db_cm,
                    "q_intrinsic": float(q_intrinsic),
                    "n_group": float(n_g),
                    "fsr_nm": float(fsr)
                }
                logger.info(f"R={radius:4.1f} μm | Total Loss: {total_loss_db_cm:6.2f} dB/cm | Q_int: {q_intrinsic:.1e}")
                
            except Exception as e:
                logger.error(f"[Error] Execution failed for radius R={radius}: {e}", exc_info=True)

        with open(manifest_path, "w") as f:
            json.dump(results_manifest, f, indent=4)
            
    # ---- Sweeping Visualization Matrix ---- #
    if results_manifest and (show or save):
        radii_sorted = sorted([float(k) for k in results_manifest.keys()])
        total_loss = [results_manifest[str(r)]["total_loss_db_cm"] for r in radii_sorted]
        q_int = [results_manifest[str(r)]["q_intrinsic"] for r in radii_sorted]
        
        fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)
        
        c1 = 'tab:red'
        ax1.set_xlabel("Bend Radius (μm)", fontsize=12, fontweight='bold')
        ax1.set_ylabel("Total Prop Loss (dB/cm)", color=c1, fontsize=12, fontweight='bold')
        ax1.plot(radii_sorted, total_loss, 'o-', color=c1, linewidth=2, markersize=7)
        ax1.tick_params(axis='y', labelcolor=c1)
        ax1.grid(True, alpha=0.3)
        
        ax2 = ax1.twinx()
        c2 = 'tab:blue'
        ax2.set_ylabel("Intrinsic Q-Factor", color=c2, fontsize=12, fontweight='bold')
        ax2.plot(radii_sorted, q_int, 's--', color=c2, linewidth=2, markersize=7)
        ax2.tick_params(axis='y', labelcolor=c2)
        ax2.set_yscale('log')
        
        fig.suptitle("Radius Sweep: Loss & Intrinsic Q", fontsize=14, fontweight='bold')
        fig.tight_layout()
        
        if save:
            if output_path is None or str(output_path).strip() == "":
                out_p = base_output_dir / "bend_radius_sweep.png"
            else:
                out_p = Path(output_path)
                if out_p.is_dir() or not out_p.suffix:
                    out_p = out_p / "bend_radius_sweep.png"
            
            out_p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_p, dpi=300, bbox_inches="tight")
            logger.info(f"[Visualizer] Plot saved gracefully to: {out_p}")
            
        if show:
            plt.show()
        else:
            plt.close(fig)

    return results_manifest