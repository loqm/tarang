"""
geometries.partial_ring_resonator

Authors: Divyanshu Tiwari, Esraaj Sarkar Gupta
Date: 2026
"""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
import tidy3d as td

from ringResonator.ring_resonator import partialRingResonator


def get_passive_simulation(
    radius: float,
    gap: float,
    wg_width: float,
    wg_height: float,
    core_medium: td.Medium,
    substrate_medium: td.Medium,
    wavelength0: float,
    span_angle: float = 10.0,
    substrate_height: float = 4.0,
    top_cladding_medium: Optional[td.Medium] = None,
    top_cladding_height: float = 0.0,
    fine_steps_per_wvl: float = 20,
    coarse_steps_per_wvl: float = 10,
    wg_length: Optional[float] = None,
) -> partialRingResonator:
    """
    Constructs a partial-ring resonator where the arc sweeps +/- (span_angle / 2) 
    centered at the lowest point (closest to the bus waveguide).
    """
    air_medium = td.Medium(permittivity=1.0)

    if top_cladding_medium is not None and top_cladding_height >= wg_height:
        hole_medium = top_cladding_medium
    else:
        hole_medium = air_medium

    # -- Partial Ring Geometry Calculations -- #
    # Center of coupling is at -90 degrees (bottom of the ring)
    phi_center = -np.pi / 2.0
    phi_span = np.deg2rad(span_angle)
    phi_start = phi_center - phi_span / 2.0
    phi_end = phi_center + phi_span / 2.0

    # Generate polygon vertices for the arc
    N_pts = max(30, int(span_angle * 2))
    phi_vals = np.linspace(phi_start, phi_end, N_pts)
    
    r_out = radius + wg_width / 2.0
    r_in = radius - wg_width / 2.0
    
    outer_pts = [(r_out * np.cos(p), r_out * np.sin(p)) for p in phi_vals]
    inner_pts = [(r_in * np.cos(p), r_in * np.sin(p)) for p in reversed(phi_vals)]
    vertices = outer_pts + inner_pts

    # -- Domain Setup -- #
    y_wg_center: float = -(radius + wg_width + gap)
    
    # Calculate required X-span based on the edges of the arc
    x_max_arc = r_out * np.abs(np.cos(phi_start))
    
    if wg_length is None:
        # Tightly wrap the bus waveguide length around the chosen arc span
        wg_length = 2.0 * x_max_arc + 5.000

    y_min = y_wg_center - (wg_width / 2.0) - 1.500
    
    # Highest Y point of this specific arc
    y_max_arc = max([p[1] for p in vertices])
    y_max = y_max_arc + 1.500

    domain_size_x: float = wg_length
    domain_size_y: float = y_max - y_min
    sim_center_y: float = (y_max + y_min) / 2.0

    # -- Structures -- #
    substrate = td.Structure(
        geometry=td.Box(
            center=(0, 0, -substrate_height / 2.0),
            size=(td.inf, td.inf, substrate_height),
        ),
        medium=substrate_medium,
        name="Substrate",
    )

    structures = [substrate]

    if top_cladding_medium is not None and top_cladding_height > 0.0:
        cladding = td.Structure(
            geometry=td.Box(
                center=(0, 0, top_cladding_height / 2.0),
                size=(td.inf, td.inf, top_cladding_height),
            ),
            medium=top_cladding_medium,
            name="Top_Cladding",
        )
        structures.append(cladding)

    partial_ring = td.Structure(
        geometry=td.PolySlab(
            vertices=vertices,
            axis=2,
            slab_bounds=(0, wg_height)
        ),
        medium=core_medium,
        name=f"Partial_Ring_{span_angle}deg",
    )

    bus_waveguide = td.Structure(
        geometry=td.Box(
            center=(0, y_wg_center, wg_height / 2.0),
            size=(wg_length + 2.0, wg_width, wg_height),
        ),
        medium=core_medium,
        name="Bus_Waveguide",
    )

    structures.extend([partial_ring, bus_waveguide])

    # -- Run Time Optimization -- #
    arc_length = radius * phi_span
    max_path_length = wg_length + arc_length
    n_group_approx = 2.1
    single_pass_time = max_path_length / (td.C_0 / n_group_approx)
    calculated_run_time = max(5e-13, single_pass_time * 2.5)

    freq0 = td.C_0 / wavelength0
    try:
        n_core_approx = float(np.real(core_medium.n_complex(freq0)))
    except Exception:
        n_core_approx = 2.0

    dl_fine = wavelength0 / (n_core_approx * fine_steps_per_wvl)

    mesh_override = td.MeshOverrideStructure(
        geometry=td.Box(
            center=(0, sim_center_y, wg_height / 2.0),
            size=(domain_size_x, domain_size_y, wg_height + 0.500),
        ),
        dl=[dl_fine, dl_fine, dl_fine],
        enforce=True,
    )

    sim_passive = td.Simulation(
        center=(0.0, sim_center_y, wg_height / 2.0),
        size=(domain_size_x, domain_size_y, 3.000),
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=coarse_steps_per_wvl,
            wavelength=wavelength0,
            override_structures=[mesh_override],
        ),
        structures=structures,
        run_time=calculated_run_time,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(),
            y=td.Boundary.pml(),
            z=td.Boundary.pml(),
        ),
        medium=air_medium,
    )

    return partialRingResonator(
        sim=sim_passive,
        radius=radius,
        gap=gap,
        wg_width=wg_width,
        wg_height=wg_height,
        core_medium=core_medium,
        substrate_medium=substrate_medium,
        substrate_height=substrate_height,
        top_cladding_medium=top_cladding_medium if top_cladding_medium is not None else air_medium,
        top_cladding_height=top_cladding_height,
        fine_steps_per_wvl=fine_steps_per_wvl,
        coarse_steps_per_wvl=coarse_steps_per_wvl,
        wg_length=wg_length,
        wavelength0=wavelength0,
        span_angle=span_angle,
    )


def get_active_simulation(
    radius: float,
    gap: float,
    wg_width: float,
    wg_height: float,
    core_medium: td.Medium,
    substrate_medium: td.Medium,
    wavelength0: float,
    span_angle: float = 10.0,
    substrate_height: float = 4.0,
    top_cladding_medium: Optional[td.Medium] = None,
    top_cladding_height: float = 0.0,
    fine_steps_per_wvl: float = 10,
    coarse_steps_per_wvl: float = 8,
    wg_length: Optional[float] = None,
) -> partialRingResonator:
    """
    Constructs an active partial-ring simulation with mode sources and boundary monitors.
    """
    resonator_obj = get_passive_simulation(
        radius=radius,
        gap=gap,
        wg_width=wg_width,
        wg_height=wg_height,
        core_medium=core_medium,
        substrate_medium=substrate_medium,
        wavelength0=wavelength0,
        span_angle=span_angle,
        substrate_height=substrate_height,
        top_cladding_medium=top_cladding_medium,
        top_cladding_height=top_cladding_height,
        fine_steps_per_wvl=fine_steps_per_wvl,
        coarse_steps_per_wvl=coarse_steps_per_wvl,
        wg_length=wg_length,
    )

    sim_passive = resonator_obj.sim
    freq0 = td.C_0 / wavelength0

    try:
        n_eff = float(np.real(core_medium.n_complex(freq0)))
    except Exception:
        n_eff = 2.0

    phi_center = -np.pi / 2.0
    phi_span = np.deg2rad(span_angle)
    phi_start = phi_center - phi_span / 2.0
    phi_end = phi_center + phi_span / 2.0

    x_max_arc = (radius + wg_width / 2.0) * np.abs(np.cos(phi_start))
    if wg_length is None:
        wg_length = 2.0 * x_max_arc + 5.000

    y_wg_center: float = -(radius + wg_width + gap)
    mon_w = wg_width * 3.5
    mon_h = wg_height * 2.5

    source = td.ModeSource(
        center=(-wg_length / 2.0 + 1.0, y_wg_center, wg_height / 2.0),
        size=(0, mon_w, mon_h),
        direction="+",
        mode_spec=td.ModeSpec(num_modes=1, target_neff=n_eff),
        source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 15),
        name="Through_In",
    )

    monitor_thru = td.ModeMonitor(
        center=(wg_length / 2.0 - 1.0, y_wg_center, wg_height / 2.0),
        size=(0, mon_w, mon_h),
        freqs=[freq0],
        mode_spec=td.ModeSpec(num_modes=1, target_neff=n_eff),
        name="Through_Out",
    )

    # Dynamic normal determination based on angle to keep ModeSolver happy
    def get_orthogonal_size(phi: float) -> Tuple[float, float, float]:
        if np.abs(np.cos(phi)) < np.abs(np.sin(phi)):
            return (0, mon_w, mon_h) # X-normal (used near +/- 90 deg)
        return (mon_w, 0, mon_h)     # Y-normal (used near 0 / 180 deg)

    mon_ring_left = td.ModeMonitor(
        center=(radius * np.cos(phi_start), radius * np.sin(phi_start), wg_height / 2.0),
        size=get_orthogonal_size(phi_start),
        freqs=[freq0],
        mode_spec=td.ModeSpec(num_modes=1, target_neff=n_eff),
        name=f"Ring_Mon_Left_{np.rad2deg(phi_start):.1f}",
    )

    mon_ring_right = td.ModeMonitor(
        center=(radius * np.cos(phi_end), radius * np.sin(phi_end), wg_height / 2.0),
        size=get_orthogonal_size(phi_end),
        freqs=[freq0],
        mode_spec=td.ModeSpec(num_modes=1, target_neff=n_eff),
        name=f"Ring_Mon_Right_{np.rad2deg(phi_end):.1f}",
    )

    monitor_field = td.FieldMonitor(
        center=(0.0, sim_passive.center[1], wg_height / 2.0),
        size=(wg_length, sim_passive.size[1] - 1.0, 0),
        freqs=[freq0],
        name="coupling_field_map",
    )

    monitors_list = [
        monitor_thru,
        mon_ring_left,
        mon_ring_right,
        monitor_field,
    ]

    sim_active = sim_passive.updated_copy(sources=[source], monitors=monitors_list)

    resonator_obj.sim = sim_active
    return resonator_obj

get_active_sim = get_active_simulation

def get_optimal_plot_coords(sim: td.Simulation) -> Tuple[float, float, float]:
    x_coord = 0.0
    if sim.sources:
        y_coord = float(sim.sources[0].center[1])
        z_coord = float(sim.sources[0].center[2])
    else:
        y_coord = float(sim.center[1])
        z_coord = float(sim.center[2])

    print(f"[Visualizer] Locking onto partial coupling center: X={x_coord}, Y={y_coord}, Z={z_coord}")
    return x_coord, y_coord, z_coord