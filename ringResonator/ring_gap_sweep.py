"""
ringSearches/ring_gap_sweep.py

Author: Esraaj Sarkar Gupta
Date: 2026

Optimization:
    -> Completely autonomous. Generates its own intrinsic metrics from scratch.
    -> Bypasses argmin transmission hunting.
    -> Unwraps T_thru = cos^2(theta) to extract the monotonic coupling phase theta(gap).
    -> Fits theta(gap) to an exponential decay curve.
    -> Solves for the exact analytic critical gap.
    -> Self-caching. Future runs at the same radius/architecture just regenerate plots.
    -> Sequential processing: Runs simulations one-by-one to avoid batch limits and saves HDF5 files.
    -> Architecture & Geometry-aware caching: Local cache files explicitly track geometry and arc type.
"""

import json
from pathlib import Path
from typing import Optional, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import tidy3d as td
from tidy3d import web

from geometries.half_ring_resonator import get_active_simulation as get_half_ring_sim
from geometries.partial_ring_resonator import get_active_simulation as get_partial_ring_sim
from geometries.resonator_arc import get_active_simulation as get_arc_sim


def _extract_geometry_kwargs(kwargs: dict) -> dict:
    """
    Normalizes geometry keyword arguments across resonator modules.
    Handles field aliasing (e.g. core_mat -> core_medium, cladding_mat -> top_cladding_medium).
    """
    clean_kwargs = dict(kwargs)
    core = kwargs.get("core_medium", kwargs.get("core_mat"))
    sub = kwargs.get("substrate_medium", kwargs.get("sub_mat", kwargs.get("substrate_mat")))
    clad = kwargs.get(
        "top_cladding_medium", 
        kwargs.get("cladding_medium", kwargs.get("cladding_mat"))
    )
    height = kwargs.get("wg_height", kwargs.get("wg_thickness"))

    # Cleanup aliases
    for key in ["core_mat", "sub_mat", "substrate_mat", "cladding_mat", "cladding_medium", "wg_thickness"]:
        clean_kwargs.pop(key, None)

    if core is not None:
        clean_kwargs["core_medium"] = core
    if sub is not None:
        clean_kwargs["substrate_medium"] = sub
    if clad is not None:
        clean_kwargs["top_cladding_medium"] = clad
    if height is not None:
        clean_kwargs["wg_height"] = height

    return clean_kwargs


def compute_intrinsic_metrics(
    radius_um: float,
    target_wavelength_um: float,
    scattering_loss_db_cm: float,
    base_output_dir: Path,
    fine_steps_per_wvl: int = 20,
    coarse_steps_per_wvl: int = 10,
    bend_angles_deg: Tuple[int, ...] = (90, 180, 270),
    **geometry_kwargs
) -> dict:
    """
    Calculates bending loss and group index using a multi-angle cutback
    regression over annular arcs (replaces the old 90-vs-180 two-point
    subtraction, which is extremely sensitive to any constant per-simulation
    offset since it divides by a tiny ~0.017 cm baseline).
    """
    print(f"[Metrics] Simulating annular arcs sequentially for R={radius_um} um...")

    arc_kwargs = _extract_geometry_kwargs(geometry_kwargs)
    # resonator_arc specifically expects cladding_medium instead of top_cladding_medium
    if "top_cladding_medium" in arc_kwargs:
        arc_kwargs["cladding_medium"] = arc_kwargs.pop("top_cladding_medium")

    w_val = arc_kwargs.get("wg_width", "NA")
    h_val = arc_kwargs.get("wg_height", "NA")
    geom_str = f"R{radius_um}_w{w_val}_h{h_val}_res{fine_steps_per_wvl}_{coarse_steps_per_wvl}"

    arc_cache_dir = base_output_dir / "arc_cache"
    arc_cache_dir.mkdir(parents=True, exist_ok=True)

    # --- PHASE 1: BUILD + RUN ALL ARC SIMULATIONS ---
    sim_objs = {}
    sim_data_dict = {}

    for angle in bend_angles_deg:
        name = f"Arc_{angle}_{geom_str}"
        arc_obj = get_arc_sim(
            radius=radius_um,
            bend_angle_deg=angle,
            wavelength0=target_wavelength_um,
            fine_steps_per_wvl=fine_steps_per_wvl,
            coarse_steps_per_wvl=coarse_steps_per_wvl,
            **arc_kwargs
        )
        sim_objs[angle] = arc_obj

        filepath = arc_cache_dir / f"{name}.hdf5"
        if filepath.exists():
            print(f"[Metrics] Loading cached FDTD data for {name} from {filepath}")
            sim_data_dict[angle] = td.SimulationData.from_file(str(filepath))
        else:
            print(f"[Metrics] Submitting job for {name}...")
            job = web.Job(simulation=arc_obj.sim, task_name=name)
            try:
                est_credits = job.estimate_cost()
                print(f"[Cost Estimate] Estimated cost for {name}: {est_credits:.4f} Flex Credits")
            except Exception as e:
                print(f"[Cost Estimate] Could not retrieve cost estimate for {name}: {e}")
            sim_data = job.run(path=str(filepath))
            sim_data_dict[angle] = sim_data

    # --- PHASE 2: LOCK A SINGLE, GLOBALLY-CONSISTENT PROPAGATION DIRECTION ---
    # Old code called max(amp_plus, amp_minus) independently per simulation.
    # That lets each sim silently pick a different "forward" direction, which
    # is fatal for any comparison (subtraction, regression) across them. Here
    # we sum power in each direction across every angle and commit to whichever
    # direction dominates in total -- a real transmitted signal should stay
    # dominant across all arc lengths, while a one-off flip from reflection or
    # mode mismatch at a single angle won't be enough to hijack the total.
    power_plus_total = 0.0
    power_minus_total = 0.0
    raw_amp_by_angle = {}

    for angle, sim_data in sim_data_dict.items():
        mon = sim_data['Mode Monitor']
        amp_data = mon.amps.isel(f=1) if ("f" in mon.amps.dims and mon.amps.sizes.get("f", 1) > 1) else mon.amps

        amp_plus = np.abs(amp_data.sel(direction="+", mode_index=0).item())**2
        amp_minus = np.abs(amp_data.sel(direction="-", mode_index=0).item())**2

        raw_amp_by_angle[angle] = (amp_plus, amp_minus)
        power_plus_total += amp_plus
        power_minus_total += amp_minus

    propagation_direction = "+" if power_plus_total >= power_minus_total else "-"
    print(f"[Metrics] Locked propagation direction = '{propagation_direction}' "
          f"(total power +: {power_plus_total:.4e}, -: {power_minus_total:.4e})")

    # --- PHASE 3: EXTRACT LOSS PER ANGLE USING THE FIXED DIRECTION ---
    arc_lengths_cm = []
    loss_db_values = []

    for angle in bend_angles_deg:
        amp_plus, amp_minus = raw_amp_by_angle[angle]
        t = amp_plus if propagation_direction == "+" else amp_minus

        assert 0.0 < t <= 1.05, f"Unphysical transmission T_{angle} = {t:.4f} (dir={propagation_direction})."

        loss_db = -10 * np.log10(t)
        length_cm = (np.deg2rad(angle) * radius_um) * 1e-4

        arc_lengths_cm.append(length_cm)
        loss_db_values.append(loss_db)

        print(f"[Metrics] Angle={angle}deg | L={length_cm*1e4:.2f}um | T={t:.4f} | Loss={loss_db:.3f} dB")

    arc_lengths_cm = np.array(arc_lengths_cm)
    loss_db_values = np.array(loss_db_values)

    # Linear regression: loss_dB(L) = alpha_bend * L + offset.
    # The offset absorbs constant per-simulation mismatch (taper coupling,
    # small reflections) instead of letting it leak into the slope, which a
    # 2-point difference over a short baseline can't do.
    slope, intercept = np.polyfit(arc_lengths_cm, loss_db_values, 1)

    alpha_bend_db_cm = max(0.0, float(slope))
    print(f"[Metrics] Cutback regression over {len(bend_angles_deg)} angles: "
          f"alpha_bend = {alpha_bend_db_cm:.4f} dB/cm (facet/taper offset ~ {intercept:.3f} dB)")

    alpha_total_db_cm = alpha_bend_db_cm + scattering_loss_db_cm
    alpha_m = (alpha_total_db_cm * 100) / 4.343

    # Group index solve uses any one of the arc sims as a reference geometry.
    ref_arc_obj = sim_objs[bend_angles_deg[0]]
    try:
        n_group = float(ref_arc_obj.get_group_index(
            target_neff=1.8,
            mode_index=0,
            use_bend_radius=True,
            local=True
        ))
    except Exception as e:
        print(f"[Error] Failed to calculate group index via mode solver: {e}")
        raise RuntimeError(f"Group index calculation failed for R={radius_um} um arc: {e}") from e

    q_intrinsic = (2 * np.pi * n_group) / ((target_wavelength_um * 1e-6) * alpha_m)

    print(f"[Metrics] Calculated Q_intrinsic: {q_intrinsic:.1f} | n_group: {n_group:.4f}")

    return {
        "q_intrinsic": q_intrinsic,
        "n_group": n_group,
        "bend_loss_db_cm": alpha_bend_db_cm,
        "total_loss_db_cm": alpha_total_db_cm,
        "cutback_fit": {
            "angles_deg": list(bend_angles_deg),
            "lengths_cm": arc_lengths_cm.tolist(),
            "loss_db": loss_db_values.tolist(),
            "facet_offset_db": float(intercept),
            "propagation_direction": propagation_direction,
        },
    }

def run_gap_optimization_batch(
    gap_sweep_array: np.ndarray,
    radius_um: float, 
    volume_um3: float,
    target_wavelength_um: float,
    scattering_loss_db_cm: float,
    base_output_dir: Path,
    architecture: str = "half_ring",
    span_angle: Optional[float] = None,
    fine_steps_per_wvl: int = 20,
    coarse_steps_per_wvl: int = 15,
    **geometry_kwargs
) -> dict:
    """
    Executes a gap optimization sweep for either a half_ring or partial_ring architecture.
    """
    base_output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = base_output_dir / "analytic_coupling_manifest.json"
    field_maps_dir = base_output_dir / "field_maps"
    field_maps_dir.mkdir(parents=True, exist_ok=True)
    gap_cache_dir = base_output_dir / "gap_cache"
    gap_cache_dir.mkdir(parents=True, exist_ok=True)
    
    clean_geometry_kwargs = _extract_geometry_kwargs(geometry_kwargs)

    # Determine Architecture & Cache Tags
    if span_angle is not None or architecture == "partial_ring":
        architecture = "partial_ring"
        span_angle = span_angle if span_angle is not None else 10.0
        arch_tag = f"partial_{span_angle:.1f}deg"
    else:
        architecture = "half_ring"
        arch_tag = "half_ring"

    w_val = clean_geometry_kwargs.get("wg_width", "NA")
    h_val = clean_geometry_kwargs.get("wg_height", "NA")
    geom_str = f"{arch_tag}_R{radius_um}_w{w_val}_h{h_val}_res{fine_steps_per_wvl}_{coarse_steps_per_wvl}"

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
        fine_steps_per_wvl=fine_steps_per_wvl,
        coarse_steps_per_wvl=coarse_steps_per_wvl,
        # bend_angles_deg=(90, 180, 270),  # uncomment to override the default
        **clean_geometry_kwargs
    )
    
    q_intrinsic = metrics["q_intrinsic"]
    n_group = metrics["n_group"]

    # --- PHASE 2: GAP SWEEP FDTD (SEQUENTIAL) ---
    print(f"\n[Solver] Initializing sequential FDTD runs ({architecture}) for {len(gap_sweep_array)} gaps...")
    
    active_sim_data = {}
    
    for gap in gap_sweep_array:
        sim_key = f"Gap_{gap:.3f}_{geom_str}"
        filepath = gap_cache_dir / f"{sim_key}.hdf5"
        
        if architecture == "partial_ring":
            resonator_obj = get_partial_ring_sim(
                radius=radius_um,
                gap=gap,
                wavelength0=target_wavelength_um,
                span_angle=span_angle,
                fine_steps_per_wvl=fine_steps_per_wvl,     
                coarse_steps_per_wvl=coarse_steps_per_wvl, 
                **clean_geometry_kwargs
            )
        else:
            resonator_obj = get_half_ring_sim(
                radius=radius_um,
                gap=gap,
                wavelength0=target_wavelength_um,
                fine_steps_per_wvl=fine_steps_per_wvl,     
                coarse_steps_per_wvl=coarse_steps_per_wvl, 
                **clean_geometry_kwargs
            )
        
        if filepath.exists():
            print(f"[Solver] Loading cached FDTD data for {sim_key}...")
            sim_data = td.SimulationData.from_file(str(filepath))
        else:
            print(f"[Solver] Submitting job for {sim_key}...")
            job = web.Job(simulation=resonator_obj.sim, task_name=sim_key)
            sim_data = job.run(path=str(filepath))
            
        active_sim_data[sim_key] = sim_data

    # --- PHASE 3: DATA EXTRACTION ---
    raw_data = []
    results_manifest = {}
    
    for sim_key, sim_data in active_sim_data.items():
        # Parse gap value out of the formatted key string
        gap_val = float(sim_key.split("_")[1])
        
        try:
            amp_plus = np.abs(sim_data['Through_Out'].amps.sel(direction="+", mode_index=0).item())**2
            amp_minus = np.abs(sim_data['Through_Out'].amps.sel(direction="-", mode_index=0).item())**2
            T_thru = float(max(amp_plus, amp_minus))
            T_thru = max(0.0, min(1.0, T_thru))
            
            raw_data.append((gap_val, T_thru))
            
            # Record internal track power if monitor exists
            ring_power_bot = 0.0
            if 'Ring_Mon_270_Bot' in sim_data:
                amps_bot = sim_data['Ring_Mon_270_Bot'].amps.sel(direction="+", mode_index=0).item()
                ring_power_bot = float(np.abs(amps_bot)**2)

            results_manifest[sim_key] = {
                "gap_um": gap_val,
                "through_port_power": T_thru,
                "ring_power_270_bot": ring_power_bot
            }
            
            if 'coupling_field_map' in sim_data:
                field_data = sim_data['coupling_field_map']
                fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
                E_mag = np.sqrt(np.abs(field_data.Ex)**2 + np.abs(field_data.Ey)**2 + np.abs(field_data.Ez)**2)
                E_mag.isel(f=0).plot(x='x', y='y', ax=ax, cmap='magma', robust=True)
                ax.set_title(f"Field Intensity ({arch_tag}, R={radius_um}um, Gap={gap_val*1000:.0f}nm)", fontsize=10)
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
    
    peak_idx = np.argmax(theta_raw)
    valid_idx = np.zeros_like(theta_raw, dtype=bool)
    valid_idx[peak_idx:] = True
    valid_idx &= (theta_raw > 0.05) & (theta_raw < 1.5) 
    
    fit_gaps = gaps[valid_idx]
    fit_theta = theta_raw[valid_idx]

    poly = np.polyfit(fit_gaps, np.log(fit_theta), 1)
    gamma = -poly[0]
    theta_0 = np.exp(poly[1])
    
    L_um = 2 * np.pi * radius_um
    a = np.exp(-np.pi * n_group * L_um / (q_intrinsic * target_wavelength_um))
    
    target_cross_power = 1.0 - a**2
    theta_target = np.arcsin(np.sqrt(target_cross_power))
    
    critical_gap = -(1.0 / gamma) * np.log(theta_target / theta_0)
    
    q_loaded = q_intrinsic / 2.0
    fom = (q_loaded**2) / volume_um3
    
    critical_metrics = {
        "architecture": architecture,
        "span_angle": span_angle,
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
    
    fig.suptitle(f"Analytic Critical Coupling ({arch_tag}, R={radius_um} $\mu m$)", fontsize=14)
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.88), bbox_transform=ax1.transAxes)
    ax1.grid(True, alpha=0.3)
    
    plt.savefig(base_output_dir / f"analytic_coupling_fit_{arch_tag}_R_{radius_um}.png", dpi=300)
    plt.close()
    
    print(f"[Analytic Solver] {arch_tag} | R={radius_um} | Crit Gap: {critical_gap:.3f} um | Q_loaded: {q_loaded:.1e}")
    return critical_metrics