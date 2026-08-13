"""
ringSearches/half_ring_gap_sweep.py

Author: Esraaj Sarkar Gupta
Date: 2026

Optimization:
    -> Completely autonomous. Generates its own intrinsic metrics from scratch.
    -> Bypasses argmin transmission hunting.
    -> Unwraps T_thru = cos^2(theta) to extract the monotonic coupling phase theta(gap).
    -> Fits theta(gap) to an exponential decay curve.
    -> Solves for the exact analytic critical gap.
    -> Self-caching. Future runs at the same radius just regenerate plots.
"""

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tidy3d as td
from tidy3d import web

# Fixed module import paths
from geometries.half_ring_resonator import get_active_simulation as get_half_ring_sim
from geometries.bend import get_active_simulation as get_bend_sim


def _extract_bend_geometry_kwargs(kwargs: dict) -> dict:
    """
    Normalizes geometry keyword arguments for geometries.bend module,
    which expects core_mat, cladding_mat, sub_mat.
    """
    bend_kwargs = dict(kwargs)
    core = kwargs.get("core_mat", kwargs.get("core_medium"))
    sub = kwargs.get("sub_mat", kwargs.get("substrate_medium", kwargs.get("substrate_mat")))
    clad = kwargs.get("cladding_mat", kwargs.get("top_cladding_medium", kwargs.get("cladding_medium")))

    if core is not None:
        bend_kwargs["core_mat"] = core
        bend_kwargs.pop("core_medium", None)
    if sub is not None:
        bend_kwargs["sub_mat"] = sub
        bend_kwargs.pop("substrate_medium", None)
        bend_kwargs.pop("substrate_mat", None)
    if clad is not None:
        bend_kwargs["cladding_mat"] = clad
        bend_kwargs.pop("top_cladding_medium", None)
        bend_kwargs.pop("cladding_medium", None)

    return bend_kwargs


def _extract_half_ring_geometry_kwargs(kwargs: dict) -> dict:
    """
    Normalizes geometry keyword arguments for geometries.half_ring_resonator module,
    which expects core_medium, substrate_medium, top_cladding_medium.
    """
    hr_kwargs = dict(kwargs)
    core = kwargs.get("core_medium", kwargs.get("core_mat"))
    sub = kwargs.get("substrate_medium", kwargs.get("sub_mat", kwargs.get("substrate_mat")))
    clad = kwargs.get("top_cladding_medium", kwargs.get("cladding_mat", kwargs.get("cladding_medium")))

    if core is not None:
        hr_kwargs["core_medium"] = core
        hr_kwargs.pop("core_mat", None)
    if sub is not None:
        hr_kwargs["substrate_medium"] = sub
        hr_kwargs.pop("sub_mat", None)
        hr_kwargs.pop("substrate_mat", None)
    if clad is not None:
        hr_kwargs["top_cladding_medium"] = clad
        hr_kwargs.pop("cladding_mat", None)
        hr_kwargs.pop("cladding_medium", None)

    return hr_kwargs


def compute_intrinsic_metrics(
    radius_um: float,
    target_wavelength_um: float,
    scattering_loss_db_cm: float,
    base_output_dir: Path,
    **geometry_kwargs
) -> dict:
    """
    Forces FDTD to calculate bending loss and group index so we don't have to rely on ghost files.
    """
    print(f"[Metrics] Simulating annular arcs for R={radius_um} um...")
    
    bend_kwargs = _extract_bend_geometry_kwargs(geometry_kwargs)

    # Generate the 90 and 180 degree arcs
    sim_90 = get_bend_sim(radius=radius_um, bend_angle_deg=90, wavelength0=target_wavelength_um, **bend_kwargs)
    sim_180 = get_bend_sim(radius=radius_um, bend_angle_deg=180, wavelength0=target_wavelength_um, **bend_kwargs)
    
    simulations = {
        "Bend_90": sim_90,
        "Bend_180": sim_180
    }
    
    batch = web.Batch(simulations=simulations, verbose=True)
    batch_data = batch.run(path_dir=str(base_output_dir / "bend_cache"))
    
    # Robustly extract transmission across directions ('+' vs '-') to handle monitor orientation conventions
    amp_90_plus = np.abs(batch_data["Bend_90"]['Through_Out'].amps.sel(direction="+", mode_index=0).item())**2
    amp_90_minus = np.abs(batch_data["Bend_90"]['Through_Out'].amps.sel(direction="-", mode_index=0).item())**2
    t_90 = float(max(amp_90_plus, amp_90_minus))

    amp_180_plus = np.abs(batch_data["Bend_180"]['Through_Out'].amps.sel(direction="+", mode_index=0).item())**2
    amp_180_minus = np.abs(batch_data["Bend_180"]['Through_Out'].amps.sel(direction="-", mode_index=0).item())**2
    t_180 = float(max(amp_180_plus, amp_180_minus))

    # Runtime sanity assertions for physical transmission
    assert 0.0 < t_90 <= 1.05, f"Unphysical transmission T_90 = {t_90:.4f}. Check monitor orientation/direction."
    assert 0.0 < t_180 <= 1.05, f"Unphysical transmission T_180 = {t_180:.4f}. Check monitor orientation/direction."

    loss_90_db = -10 * np.log10(t_90)
    loss_180_db = -10 * np.log10(t_180)
    
    l_90_cm = (np.pi / 2 * radius_um) * 1e-4
    l_180_cm = (np.pi * radius_um) * 1e-4
    
    alpha_bend_db_cm = (loss_180_db - loss_90_db) / (l_180_cm - l_90_cm)
    alpha_total_db_cm = alpha_bend_db_cm + scattering_loss_db_cm
    
    # Total power attenuation coefficient alpha_m [m^-1]
    alpha_m = (alpha_total_db_cm * 100) / 4.343
    
    # Calculate group index via mode solver on the 90-degree bend resonator object
    try:
        n_group = float(sim_90.get_group_index(
            target_neff=1.8,
            mode_index=0,
            use_bend_radius=True,
            local=True
        ))
    except Exception as e:
        print(f"[Error] Failed to calculate group index via mode solver: {e}")
        raise RuntimeError(f"Group index calculation failed for R={radius_um} um bend: {e}") from e
    
    q_intrinsic = (2 * np.pi * n_group) / ((target_wavelength_um * 1e-6) * alpha_m)
    
    print(f"[Metrics] Calculated Q_intrinsic: {q_intrinsic:.1f} | n_group: {n_group:.4f}")
    
    return {
        "q_intrinsic": q_intrinsic,
        "n_group": n_group,
        "bend_loss_db_cm": alpha_bend_db_cm,
        "total_loss_db_cm": alpha_total_db_cm
    }


def run_gap_optimization_batch(
    gap_sweep_array: np.ndarray,
    radius_um: float, 
    volume_um3: float,
    target_wavelength_um: float,
    scattering_loss_db_cm: float,
    base_output_dir: Path,
    **geometry_kwargs
) -> dict:
    
    base_output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = base_output_dir / "analytic_coupling_manifest.json"
    field_maps_dir = base_output_dir / "field_maps"
    field_maps_dir.mkdir(parents=True, exist_ok=True)
    
    # Separate keyword arguments for bend and half-ring simulations
    bend_geometry_kwargs = _extract_bend_geometry_kwargs(geometry_kwargs)
    half_ring_geometry_kwargs = _extract_half_ring_geometry_kwargs(geometry_kwargs)

    # --- CACHE HIT: BYPASS FDTD ---
    if manifest_path.exists():
        print(f"[Gap Solver] Cache hit at {manifest_path}. Bypassing FDTD compute, pulling existing data.")
        with open(manifest_path, "r") as f:
            cached_data = json.load(f)
            return cached_data["critical_metrics"]

    # --- PHASE 1: GENERATE INTRINSIC METRICS ---
    metrics = compute_intrinsic_metrics(
        radius_um=radius_um,
        target_wavelength_um=target_wavelength_um,
        scattering_loss_db_cm=scattering_loss_db_cm,
        base_output_dir=base_output_dir,
        **bend_geometry_kwargs
    )
    
    q_intrinsic = metrics["q_intrinsic"]
    n_group = metrics["n_group"]

    # --- PHASE 2: GAP SWEEP FDTD ---
    print(f"\n[Solver] Initializing FDTD batch for {len(gap_sweep_array)} gap configurations...")
    
    simulations = {}
    for gap in gap_sweep_array:
        sim_key = f"Gap_{gap:.3f}"
        simulations[sim_key] = get_half_ring_sim(
            radius=radius_um,
            gap=gap,
            wavelength0=target_wavelength_um,
            **half_ring_geometry_kwargs
        )
        
    batch = web.Batch(simulations=simulations, verbose=True)
    active_sim_data = batch.run(path_dir=str(base_output_dir / "gap_cache"))

    # --- PHASE 3: DATA EXTRACTION ---
    raw_data = []
    results_manifest = {}
    
    for sim_key, sim_data in active_sim_data.items():
        gap_val = float(sim_key.split("_")[1])
        
        try:
            amp_plus = np.abs(sim_data['Through_Out'].amps.sel(direction="+", mode_index=0).item())**2
            amp_minus = np.abs(sim_data['Through_Out'].amps.sel(direction="-", mode_index=0).item())**2
            T_thru = float(max(amp_plus, amp_minus))
            T_thru = max(0.0, min(1.0, T_thru)) # Clamp numerical noise
            
            raw_data.append((gap_val, T_thru))
            
            # Scrape ring monitors for logging
            amps_bot = sim_data['Ring_Mon_270_Bot'].amps.sel(direction="+", mode_index=0).item()
            results_manifest[sim_key] = {
                "gap_um": gap_val,
                "through_port_power": T_thru,
                "ring_power_270_bot": float(np.abs(amps_bot)**2)
            }
            
            # Save the E-field plot
            if 'coupling_field_map' in sim_data.simulation.monitors:
                field_data = sim_data['coupling_field_map']
                fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
                E_mag = np.sqrt(np.abs(field_data.Ex)**2 + np.abs(field_data.Ey)**2 + np.abs(field_data.Ez)**2)
                E_mag.isel(f=0).plot(x='x', y='y', ax=ax, cmap='magma', robust=True)
                ax.set_title(f"Field Intensity (R={radius_um}um, Gap={gap_val*1000:.0f}nm)", fontsize=10)
                ax.set_aspect('equal')
                fig.tight_layout()
                plt.savefig(field_maps_dir / f"field_map_{sim_key}.png")
                plt.close(fig)
                
        except Exception as e:
            print(f"[Warning] Data extraction failed for {sim_key}: {e}")

    # --- PHASE 4: ANALYTIC MATH ---
    raw_data.sort(key=lambda x: x[0])
    gaps = np.array([d[0] for d in raw_data])
    t_thrus = np.array([d[1] for d in raw_data])
    
    theta_raw = np.arccos(np.sqrt(t_thrus))
    
    # Phase folding correction
    peak_idx = np.argmax(theta_raw)
    valid_idx = np.zeros_like(theta_raw, dtype=bool)
    valid_idx[peak_idx:] = True
    valid_idx &= (theta_raw > 0.05) & (theta_raw < 1.5) 
    
    fit_gaps = gaps[valid_idx]
    fit_theta = theta_raw[valid_idx]

    # Exponential fit: ln(theta) = ln(theta_0) - gamma*d
    poly = np.polyfit(fit_gaps, np.log(fit_theta), 1)
    gamma = -poly[0]
    theta_0 = np.exp(poly[1])
    
    # Analytical target 
    L_um = 2 * np.pi * radius_um
    a = np.exp(-np.pi * n_group * L_um / (q_intrinsic * target_wavelength_um))
    
    target_cross_power = 1.0 - a**2
    theta_target = np.arcsin(np.sqrt(target_cross_power))
    
    critical_gap = -(1.0 / gamma) * np.log(theta_target / theta_0)
    
    q_loaded = q_intrinsic / 2.0
    fom = (q_loaded**2) / volume_um3
    
    critical_metrics = {
        "radius_um": radius_um,
        "critical_gap_um": critical_gap,
        "round_trip_a": a,
        "target_cross_power": target_cross_power,
        "q_intrinsic": q_intrinsic,
        "q_loaded": q_loaded,
        "volume_um3": volume_um3,
        "fom_raw": fom,
        "fit_params": {"theta_0": theta_0, "gamma": gamma},
        "bend_metrics": metrics
    }
    
    # --- PHASE 5: CREATE THE CACHE ---
    with open(manifest_path, "w") as f:
        json.dump({"critical_metrics": critical_metrics, "raw_sweep": results_manifest}, f, indent=4)
        
    # --- PHASE 6: PLOTTING ---
    smooth_gaps = np.linspace(min(gaps) - 0.05, max(gaps) + 0.05, 100)
    smooth_theta = theta_0 * np.exp(-gamma * smooth_gaps)
    smooth_t_thru = np.cos(smooth_theta)**2
    
    fig, ax1 = plt.subplots(figsize=(9, 6), dpi=300)
    color = 'tab:gray'
    ax1.set_xlabel("Coupling Gap ($\mu m$)", fontsize=12)
    ax1.set_ylabel("$T_{thru}$ (Raw FDTD Transmission)", color=color, fontsize=12)
    ax1.plot(gaps, t_thrus, 'ko', markersize=6, alpha=0.5, label='Raw $T_{thru}$')
    ax1.plot(smooth_gaps, smooth_t_thru, '--', color=color, alpha=0.5, label='Fitted $T_{thru}$')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(-0.05, 1.05)
    
    ax2 = ax1.twinx()
    color = 'tab:purple'
    ax2.set_ylabel("Integrated Coupling Phase $\\theta(d)$ [rad]", color=color, fontsize=12)
    ax2.plot(gaps, theta_raw, 's', color='tab:blue', markersize=8, label='Extracted $\\theta$')
    ax2.plot(smooth_gaps, smooth_theta, '-', color=color, linewidth=2.5, label=f'Fit: $\\theta_0 e^{{-\\gamma d}}$')
    ax2.axhline(theta_target, color='teal', linestyle=':', linewidth=2, label=f'Target $\\theta$ = {theta_target:.3f}')
    ax2.axvline(critical_gap, color='crimson', linestyle='--', linewidth=2, label=f'Analytic Critical Gap = {critical_gap*1000:.1f} nm')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yscale('log')
    
    fig.suptitle(f"Analytic Critical Coupling Solution (R={radius_um} $\mu m$)", fontsize=14)
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.88), bbox_transform=ax1.transAxes)
    ax1.grid(True, alpha=0.3)
    
    plt.savefig(base_output_dir / f"analytic_coupling_fit_R_{radius_um}.png", dpi=300)
    plt.close()
    
    print(f"[Analytic Solver] R={radius_um} | Crit Gap: {critical_gap:.3f} um | Q_loaded: {q_loaded:.1e}")
    return critical_metrics