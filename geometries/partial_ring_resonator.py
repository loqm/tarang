"""
geometries.partial_ring_resonator.py

Authors: Esraaj Sarkar Gupta
Date: 2026

Constructs a partial ring resonator + waveguide simulation physically clipped
to a specified span angle centered at (0, -R), wrapped in a partialRingResonator.

This module defines a simulation with Absorber layers in the PML, specifically to
handle guided wave truncation. Light that leaves the end of a ring hits the PML at
oblique angles.
"""

from __future__ import annotations
from typing import Optional, Tuple
import warnings
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
    span_angle_rad: float,   # In Radians
    num_absorber_layers: int = 60,
    substrate_height: float = 4.0,
    top_cladding_medium: Optional[td.Medium] = None,
    top_cladding_height: float = 0.0,
    fine_steps_per_wvl: float = 20,
    coarse_steps_per_wvl: float = 15,
    wg_length: Optional[float] = None,
) -> partialRingResonator:
    """
    Constructs a passive simulation with the ring physically clipped to span_angle_rad.
    """

    span_angle_deg = np.rad2deg(span_angle_rad)

    if (span_angle_deg > 180.0):
        raise ValueError(f"Received a span angle of {span_angle_deg:.1f}°. Span angles > 180° are not supported.")
    elif (span_angle_deg == 180.0):
        warnings.warn(
            "For a 180° span angle, using the 'half_ring_resonator' module is recommended for better performance.",
            UserWarning,
            stacklevel=2,
        )

    air_medium = td.Medium(permittivity=1.0)
    hole_medium = (
        top_cladding_medium
        if (top_cladding_medium is not None and top_cladding_height >= wg_height)
        else air_medium
    )

    # -- Geometry Math & Physical Arc Truncation -- #
    half_arc = abs(span_angle_rad) / 2.0
    r_outer = radius + wg_width / 2.0
    r_inner = radius - wg_width / 2.0

    x_arc_max = r_outer * np.sin(half_arc)
    y_arc_min = -r_outer
    y_arc_max = -r_inner * np.cos(half_arc)

    # Physical Clipping Box isolating ONLY the span_angle_rad arc around (0, -R)
    clip_box = td.Box(
        center=(0.0, (y_arc_min + y_arc_max) / 2.0, wg_height / 2.0),
        size=(2.0 * x_arc_max, (y_arc_max - y_arc_min) + 0.001, wg_height + 0.400),
    )

    y_wg_center: float = -(radius + wg_width + gap)
    y_min = y_wg_center - (wg_width / 2.0) - 1.500
    y_max = y_arc_max + 1.500

    domain_size_x = max(4.0, 2.0 * x_arc_max + 3.0)
    if wg_length is not None:
        domain_size_x = max(domain_size_x, wg_length)

    bus_length = domain_size_x + 2.0
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

    ring_outer_geom = td.Cylinder(
        center=(0, 0, wg_height / 2.0),
        radius=r_outer,
        length=wg_height,
        axis=2,
    ) & clip_box

    ring_inner_geom = td.Cylinder(
        center=(0, 0, wg_height / 2.0),
        radius=r_inner,
        length=wg_height + 0.200,
        axis=2,
    ) & clip_box

    ring_outer = td.Structure(geometry=ring_outer_geom, medium=core_medium, name="Ring_Outer")
    ring_inner = td.Structure(geometry=ring_inner_geom, medium=hole_medium, name="Ring_Inner_Hole")

    bus_waveguide = td.Structure(
        geometry=td.Box(
            center=(0, y_wg_center, wg_height / 2.0),
            size=(bus_length, wg_width, wg_height),
        ),
        medium=core_medium,
        name="Bus_Waveguide",
    )

    structures.extend([ring_outer, ring_inner, bus_waveguide])

    # -- Grid & Run-time Optimization -- #
    max_path_length = domain_size_x + 2.0
    n_group_approx = 2.1
    calculated_run_time = max(5e-13, (max_path_length / (td.C_0 / n_group_approx)) * 3.0)

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
        # Specify absorber layers 
        boundary_spec=td.BoundarySpec(
            x=td.Boundary(
                plus=td.Absorber(num_layers=num_absorber_layers),
                minus=td.Absorber(num_layers=num_absorber_layers),
            ),
            y=td.Boundary(
                plus=td.Absorber(num_layers=num_absorber_layers),
                minus=td.Absorber(num_layers=num_absorber_layers),
            ),
            z=td.Boundary.pml(),
        ),
        medium=air_medium,
    )

    return partialRingResonator(
        span_angle_rad=span_angle_rad,
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
        wg_length=domain_size_x,
        wavelength0=wavelength0,
    )


def get_active_simulation(
    radius: float,
    gap: float,
    wg_width: float,
    wg_height: float,
    core_medium: td.Medium,
    substrate_medium: td.Medium,
    wavelength0: float,
    span_angle_rad: float = np.pi / 6.0,
    num_absorber_layers: int = 60,
    substrate_height: float = 4.0,
    top_cladding_medium: Optional[td.Medium] = None,
    top_cladding_height: float = 0.0,
    fine_steps_per_wvl: float = 20,
    coarse_steps_per_wvl: float = 15,
    wg_length: Optional[float] = None,
) -> partialRingResonator:
    """
    Constructs an active partial-ring simulation with mode sources and monitors.
    """
    resonator_obj = get_passive_simulation(
        radius=radius,
        gap=gap,
        wg_width=wg_width,
        wg_height=wg_height,
        core_medium=core_medium,
        substrate_medium=substrate_medium,
        wavelength0=wavelength0,
        span_angle_rad=span_angle_rad,
        num_absorber_layers=num_absorber_layers,
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

    domain_size_x = sim_passive.size[0]
    y_wg_center: float = -(radius + wg_width + gap)
    mon_w = wg_width * 3.5
    mon_h = wg_height * 2.5

    source = td.ModeSource(
        center=(-domain_size_x / 2.0 + 1.0, y_wg_center, wg_height / 2.0),
        size=(0, mon_w, mon_h),
        direction="+",
        mode_spec=td.ModeSpec(num_modes=1, target_neff=n_eff),
        source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 15),
        name="Through_In",
    )

    monitor_thru = td.ModeMonitor(
        center=(domain_size_x / 2.0 - 1.0, y_wg_center, wg_height / 2.0),
        size=(0, mon_w, mon_h),
        freqs=[freq0],
        mode_spec=td.ModeSpec(num_modes=1, target_neff=n_eff),
        name="Through_Out",
    )

    # Monitor along preserved arc segment
    theta_drop = (abs(span_angle_rad) / 2.0) * 0.75
    drop_x = radius * np.sin(theta_drop)
    drop_y = -radius * np.cos(theta_drop)

    mon_ring_drop = td.ModeMonitor(
        center=(drop_x, drop_y, wg_height / 2.0),
        size=(0, mon_w, mon_h),
        freqs=[freq0],
        mode_spec=td.ModeSpec(
            num_modes=1,
            target_neff=n_eff,
            angle_theta=theta_drop,
            bend_radius=radius,
            bend_axis=1,
        ),
        name="Ring_Drop_Out",
    )

    monitor_field = td.FieldMonitor(
        center=(0.0, sim_passive.center[1], wg_height / 2.0),
        size=(td.inf, td.inf, 0),
        freqs=[freq0],
        name="coupling_field_map",
    )

    sim_active = sim_passive.updated_copy(
        sources=[source],
        monitors=[monitor_thru, mon_ring_drop, monitor_field],
    )

    resonator_obj.sim = sim_active
    return resonator_obj


get_active_sim = get_active_simulation


def get_optimal_plot_coords(sim: td.Simulation) -> Tuple[float, float, float]:
    x_coord = 0.0
    z_coord = float(sim.sources[0].center[2]) if sim.sources else float(sim.center[2])
    y_coord = float(sim.center[1])

    print(
        f"[Visualizer] Locking All-Pass slice view onto coupling domain center: "
        f"X={x_coord:.3f}, Y={y_coord:.3f}, Z={z_coord:.3f}"
    )
    return x_coord, y_coord, z_coord