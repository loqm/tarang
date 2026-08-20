"""
geometries.resonator_arc.py

Authors: Esraaj Sarkar Gupta, Divyanshu Tiwari
Date: 2026
"""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
import tidy3d as td

from ringResonator.resonator_arc import resonatorArc

def get_passive_simulation(
    wavelength0: float,

    radius: float,

    wg_width: float,
    wg_height: float,

    core_medium: td.Medium,
    substrate_medium: td.Medium,
    cladding_medium: td.Medium = td.Medium(permittivity=1.0),   # Air

    bend_angle_deg: float = 90,

    fine_steps_per_wvl: int = 20,
    coarse_steps_per_wvl: int = 10
) -> resonatorArc:
    """
    Constructs a passive arc resonator simulation wrapped in a resonatorArc object.
    Explicitly handles 90°, 180°, and 270° bend geometries in isolated branches.
    """
    # -- Shared Ring & Substrate Geometry -- #
    geom_outer = td.Cylinder(
        center=(0, 0, wg_height / 2),
        radius=radius + wg_width / 2,
        length=wg_height,
        axis=2
    )
    geom_inner = td.Cylinder(
        center=(0, 0, wg_height / 2),
        radius=radius - wg_width / 2,
        length=wg_height + 0.200,
        axis=2
    )
    
    ring_outer = td.Structure(geometry=geom_outer, medium=core_medium, name="Ring Outer")
    ring_inner = td.Structure(geometry=geom_inner, medium=cladding_medium, name="Ring Inner Hole")
    annulus_geom = geom_outer - geom_inner

    substrate = td.Structure(
        geometry=td.Box(center=(0, 0, -1.000), size=(td.inf, td.inf, 2.000)),
        medium=substrate_medium, name="Substrate"
    )

    # -- Explicit Angle Branching -- #
    # Each branch defines: which structures cut the annulus down to the desired
    # arc, where the two straight port waveguides sit (each poking into a
    # region that was cut away, tangent to the ring at its connection point),
    # and the simulation domain box that contains the kept geometry with margin.
    if bend_angle_deg == 90:
        # 90 Arc: Keep top-right quadrant (x > 0, y > 0)
        cut_B = td.Structure(
            geometry=td.Box(center=(0, -radius, 0), size=(3 * radius, 2 * radius, 10.000)),
            medium=cladding_medium, name="Bottom Cut"
        )
        cut_L = td.Structure(
            geometry=td.Box(center=(-radius, radius / 2, 0), size=(2 * radius, 3 * radius, 10.000)),
            medium=cladding_medium, name="Left Cut"
        )
        wg_in = td.Structure(
            geometry=td.Box(center=(-2.000, radius, wg_height / 2), size=(4.000, wg_width, wg_height)),
            medium=core_medium, name="Waveguide In"
        )
        wg_out = td.Structure(
            geometry=td.Box(center=(radius, -2.000, wg_height / 2), size=(wg_width, 4.000, wg_height)),
            medium=core_medium, name="Waveguide Out"
        )
        structures = [ring_outer, ring_inner, substrate, cut_B, cut_L, wg_in, wg_out]
        domain_size_x = radius + 6.000
        domain_size_y = radius + 6.000
        center_x = radius / 2.0
        center_y = radius / 2.0

    elif bend_angle_deg == 180:
        # 180 Arc: Keep top half (y > 0)
        cut_B = td.Structure(
            geometry=td.Box(center=(0, -radius, 0), size=(3 * radius, 2 * radius, 10.000)),
            medium=cladding_medium, name="Bottom Cut"
        )
        wg_in = td.Structure(
            geometry=td.Box(center=(-radius, -2.000, wg_height / 2), size=(wg_width, 4.000, wg_height)),
            medium=core_medium, name="Waveguide In"
        )
        wg_out = td.Structure(
            geometry=td.Box(center=(radius, -2.000, wg_height / 2), size=(wg_width, 4.000, wg_height)),
            medium=core_medium, name="Waveguide Out"
        )
        structures = [ring_outer, ring_inner, substrate, cut_B, wg_in, wg_out]
        domain_size_x = 2 * radius + 6.000
        domain_size_y = radius + 6.000
        center_x = 0.0
        center_y = radius / 2.0

    elif bend_angle_deg == 270:
        # 270 Arc: Keep three quadrants (everything except x > 0, y < 0).
        # Both open ends of the arc -- angle 0 at (radius, 0) and angle 270
        # at (0, -radius) -- border that single removed quadrant, so both
        # port waveguides poke into it, tangent to the ring at their
        # respective connection points.
        cut_Q4 = td.Structure(
            geometry=td.Box(center=(radius, -radius, 0), size=(2 * radius, 2 * radius, 10.000)),
            medium=cladding_medium, name="Bottom-Right Cut"
        )
        wg_in = td.Structure(
            geometry=td.Box(center=(radius, -2.000, wg_height / 2), size=(wg_width, 4.000, wg_height)),
            medium=core_medium, name="Waveguide In"
        )
        wg_out = td.Structure(
            geometry=td.Box(center=(2.000, -radius, wg_height / 2), size=(4.000, wg_width, wg_height)),
            medium=core_medium, name="Waveguide Out"
        )
        structures = [ring_outer, ring_inner, substrate, cut_Q4, wg_in, wg_out]
        # Unlike 90/180, the kept geometry now reaches negative y (down to
        # the wg_out port at y = -radius) as well as positive x and y, so the
        # domain must be sized symmetrically around the origin on BOTH axes
        # instead of reusing the old radius/2-centered box that only worked
        # because 90/180 never went below y ~ 0.
        domain_size_x = 2 * radius + 6.000
        domain_size_y = 2 * radius + 6.000
        center_x = 0.0
        center_y = 0.0

    else:
        raise ValueError(
            f"[Error] Unsupported bend angle: {bend_angle_deg}°. "
            f"Only 90°, 180°, and 270° bends are explicitly supported."
        )

    # -- Grid Overrides & Solver Setup -- #
    try:
        freq0 = td.C_0 / wavelength0
        n_core_approx = float(np.real(core_medium.n_complex(freq0)))
    except Exception:
        n_core_approx = 2.0
        
    dl_fine = wavelength0 / (n_core_approx * fine_steps_per_wvl)

    mesh_override_ring = td.MeshOverrideStructure(geometry=annulus_geom, dl=[dl_fine] * 3, enforce=True)
    mesh_override_in = td.MeshOverrideStructure(geometry=wg_in.geometry, dl=[dl_fine] * 3, enforce=True)
    mesh_override_out = td.MeshOverrideStructure(geometry=wg_out.geometry, dl=[dl_fine] * 3, enforce=True)

    arc_length_um = (bend_angle_deg / 180.0 * np.pi * radius) + 6.0
    estimated_travel_time = (arc_length_um / (td.C_0 / n_core_approx)) * 2.5 
    
    sim_passive = td.Simulation(
        center=(center_x, center_y, -0.500),
        size=(domain_size_x, domain_size_y, 4.000),
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=coarse_steps_per_wvl, wavelength=wavelength0,
            override_structures=[mesh_override_ring, mesh_override_in, mesh_override_out]
        ),
        structures=structures,
        run_time=max(3e-12, estimated_travel_time),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
        medium=cladding_medium
    )
    
    return resonatorArc(
        sim=sim_passive,

        radius=radius,

        wg_width=wg_width,
        wg_height=wg_height,

        core_medium=core_medium,
        cladding_medium=cladding_medium,
        substrate_medium=substrate_medium,

        bend_angle_deg=bend_angle_deg,

        fine_steps_per_wvl=fine_steps_per_wvl,
        coarse_steps_per_wvl=coarse_steps_per_wvl,

        wavelength0=wavelength0
    )


def get_active_simulation(
    wavelength0: float,

    radius: float, 

    wg_width: float, 
    wg_height: float,

    core_medium: td.Medium, 
    substrate_medium: td.Medium,
    cladding_medium: td.Medium = td.Medium(permittivity=1.0),   # Air

    bend_angle_deg: float = 90, 

    fine_steps_per_wvl: int = 20, 
    coarse_steps_per_wvl: int = 10
) -> resonatorArc:  
    """
    Constructs an active simulation with mode sources and monitors for 90°, 180°, or 270° bends.
    """
    freq0 = td.C_0 / wavelength0
    df = freq0 * 0.02 

    try:
        n_eff = float(np.real(core_medium.n_complex(freq0)))
    except Exception:
        n_eff = 2.0

    resonator_obj = get_passive_simulation(
        wavelength0=wavelength0,

        radius=radius,

        wg_width=wg_width,
        wg_height=wg_height,

        core_medium=core_medium,
        cladding_medium=cladding_medium,
        substrate_medium=substrate_medium,

        bend_angle_deg=bend_angle_deg,

        fine_steps_per_wvl=fine_steps_per_wvl,
        coarse_steps_per_wvl=coarse_steps_per_wvl
    )
    sim_passive = resonator_obj.sim

    # -- Explicit Port Layout for Active Sources -- #
    if bend_angle_deg == 90:
        src_center = (-1.0, radius, wg_height / 2.0)
        src_size = (0, 3.0, 2.5)
        mon_center = (radius, -1.0, wg_height / 2.0)
        mon_size = (3.0, 0, 2.5)
    elif bend_angle_deg == 180:
        src_center = (-radius, -1.0, wg_height / 2.0)
        src_size = (3.0, 0, 2.5)
        mon_center = (radius, -1.0, wg_height / 2.0)
        mon_size = (3.0, 0, 2.5)
    elif bend_angle_deg == 270:
        # wg_in is the vertical strip at x = radius (propagates along +y,
        # entering the ring at (radius, 0)); wg_out is the horizontal strip
        # at y = -radius (propagates along +x, exiting at (0, -radius)).
        src_center = (radius, -1.0, wg_height / 2.0)
        src_size = (3.0, 0, 2.5)
        mon_center = (1.0, -radius, wg_height / 2.0)
        mon_size = (0, 3.0, 2.5)
    else:
        raise ValueError(
            f"[Error] Unsupported bend angle: {bend_angle_deg}°. "
            f"Only 90°, 180°, and 270° bends are explicitly supported."
        )

    source = td.ModeSource(
        center=src_center, size=src_size, direction="+",
        mode_spec=td.ModeSpec(num_modes=1, target_neff=n_eff),
        source_time=td.GaussianPulse(freq0=freq0, fwidth=freq0 / 10), name='Mode Source'
    )

    monitor = td.ModeMonitor(
        center=mon_center, size=mon_size, 
        freqs=[freq0 - df, freq0, freq0 + df], 
        mode_spec=td.ModeSpec(num_modes=1, target_neff=n_eff), name='Mode Monitor'
    )

    monitor_field = td.FieldMonitor(
        center=(sim_passive.center[0], sim_passive.center[1], wg_height / 2.0),
        size=(sim_passive.size[0], sim_passive.size[1], 0),
        freqs=[freq0],
        name='field_map'
    )

    sim_active = sim_passive.copy(update={"sources": [source], "monitors": [monitor, monitor_field]})
    resonator_obj.sim = sim_active
    return resonator_obj

# Try to be helpful
get_active_sim = get_active_simulation


def get_optimal_plot_coords(sim: td.Simulation) -> Tuple[float, float, float]:
    """
    Slicing helper focused on the critical waveguide core plane.
    """
    x_coord = 0.0
    y_coord = 0.0

    if sim.sources:
        z_coord = float(sim.sources[0].center[2])
    else:
        # Locate the core structure to extract true waveguide height
        core_struct = next(
            (s for s in sim.structures if s.name in ["Ring Outer", "Bus_Waveguide"]), 
            None
        )
        if core_struct:
            z_coord = float(core_struct.geometry.center[2])
        else:
            z_coord = float(sim.center[2])

    print(f"[Visualizer] Locking Bend slice view onto core center: X={x_coord}, Y={y_coord}, Z={z_coord}")
    return x_coord, y_coord, z_coord