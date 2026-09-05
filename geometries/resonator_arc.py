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
    coarse_steps_per_wvl: int = 10,
    num_absorber_layers: int = 60
) -> resonatorArc:
    """
    Constructs a passive arc resonator simulation wrapped in a resonatorArc object.
    Supports any angle between 0 and 180 degrees.
    Uses Absorber boundaries on X/Y to prevent PML structure extrusion artifacts
    on curved arc boundaries.
    """
    if not (0 < bend_angle_deg <= 180):
        raise ValueError(
            f"[Error] Unsupported bend angle: {bend_angle_deg}°. "
            f"Angle must be in the range (0, 180]."
        )

    theta_rad = np.deg2rad(bend_angle_deg)

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
    annulus_geom = geom_outer - geom_inner

    substrate = td.Structure(
        geometry=td.Box(center=(0, 0, -1.000), size=(td.inf, td.inf, 2.000)),
        medium=substrate_medium, name="Substrate"
    )

    # -- Dynamic Arc Generation via Boolean Subtraction -- #
    # 1. Remove everything in the x < 0 half-plane
    cut_left = td.Box(
        center=(-2 * radius, 0, wg_height / 2), 
        size=(4 * radius, 4 * radius, wg_height + 2.0)
    )
    core_geom = annulus_geom - cut_left

    # 2. Remove the wedge from the end angle to 180 degrees (if angle < 180)
    if bend_angle_deg < 180:
        cut_pts = [(0, 0), (2 * radius * np.sin(theta_rad), 2 * radius * np.cos(theta_rad))]
        if bend_angle_deg <= 90:
            cut_pts.append((2 * radius, -2 * radius))
        cut_pts.append((0, -2 * radius))
        
        cut_wedge = td.PolySlab(vertices=cut_pts, axis=2, slab_bounds=(-1.0, wg_height + 1.0))
        core_geom = core_geom - cut_wedge

    ring_arc = td.Structure(geometry=core_geom, medium=core_medium, name="Ring Arc")

    # -- Dynamic Waveguide Generation -- #
    wg_in_length = 4.0
    wg_in = td.Structure(
        geometry=td.Box(
            center=(-wg_in_length / 2, radius, wg_height / 2), 
            size=(wg_in_length, wg_width, wg_height)
        ),
        medium=core_medium, name="Waveguide In"
    )

    wg_out_length = 4.0
    w = wg_width
    # Base straight waveguide vertices along +X axis
    v1, v2, v3, v4 = (0, w/2), (0, -w/2), (wg_out_length, -w/2), (wg_out_length, w/2)
    
    def transform_pt(x, y):
        # Rotate by -theta_rad and translate to the end of the arc
        xr = x * np.cos(-theta_rad) - y * np.sin(-theta_rad)
        yr = x * np.sin(-theta_rad) + y * np.cos(-theta_rad)
        return xr + radius * np.sin(theta_rad), yr + radius * np.cos(theta_rad)
    
    wg_out_verts = [transform_pt(x, y) for x, y in [v1, v2, v3, v4]]
    wg_out = td.Structure(
        geometry=td.PolySlab(vertices=wg_out_verts, axis=2, slab_bounds=(0, wg_height)),
        medium=core_medium, name="Waveguide Out"
    )

    structures = [ring_arc, substrate, wg_in, wg_out]

    # -- Dynamic Domain Sizing -- #
    xs = [0, -wg_in_length, radius * np.sin(theta_rad)]
    ys = [radius + wg_width / 2, radius - wg_width / 2, radius * np.cos(theta_rad)]
    
    if bend_angle_deg >= 90:
        xs.append(radius + wg_width / 2)
        ys.append(0)
        
    for vx, vy in wg_out_verts:
        xs.append(vx)
        ys.append(vy)
        
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    pad = 3.0
    domain_size_x = (max_x - min_x) + 2 * pad
    domain_size_y = (max_y - min_y) + 2 * pad
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0

    # -- Grid Overrides & Solver Setup -- #
    try:
        freq0 = td.C_0 / wavelength0
        n_core_approx = float(np.real(core_medium.n_complex(freq0)))
    except Exception:
        n_core_approx = 2.0
        
    dl_fine = wavelength0 / (n_core_approx * fine_steps_per_wvl)

    mesh_override_ring = td.MeshOverrideStructure(geometry=ring_arc.geometry, dl=[dl_fine] * 3, enforce=True)
    mesh_override_in = td.MeshOverrideStructure(geometry=wg_in.geometry, dl=[dl_fine] * 3, enforce=True)
    mesh_override_out = td.MeshOverrideStructure(geometry=wg_out.geometry, dl=[dl_fine] * 3, enforce=True)

    arc_length_um = (bend_angle_deg / 180.0 * np.pi * radius) + wg_in_length + wg_out_length
    estimated_travel_time = (arc_length_um / (td.C_0 / n_core_approx)) * 2.5 
    
    boundary_spec = td.BoundarySpec(
        x=td.Boundary.absorber(num_layers=num_absorber_layers),
        y=td.Boundary.absorber(num_layers=num_absorber_layers),
        z=td.Boundary.pml()
    )

    sim_passive = td.Simulation(
        center=(center_x, center_y, -0.500),
        size=(domain_size_x, domain_size_y, 4.000),
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=coarse_steps_per_wvl, wavelength=wavelength0,
            override_structures=[mesh_override_ring, mesh_override_in, mesh_override_out]
        ),
        structures=structures,
        run_time=max(3e-12, estimated_travel_time),
        boundary_spec=boundary_spec,
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
    cladding_medium: td.Medium = td.Medium(permittivity=1.0),
    bend_angle_deg: float = 90, 
    fine_steps_per_wvl: int = 20, 
    coarse_steps_per_wvl: int = 10,
    num_absorber_layers: int = 60
) -> resonatorArc:  
    """
    Constructs an active simulation with mode sources and monitors dynamically 
    placed based on the arbitrary arc bend angle.
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
        coarse_steps_per_wvl=coarse_steps_per_wvl,
        num_absorber_layers=num_absorber_layers
    )
    sim_passive = resonator_obj.sim

    # -- Dynamic Port Layout -- #
    theta_rad = np.deg2rad(bend_angle_deg)
    
    # Source is always on wg_in, which is axis-aligned to +X
    src_center = (-2.0, radius, wg_height / 2.0)
    src_size = (0, 3.0, 2.5)

    # Monitor placed 2.0um along the oblique wg_out path
    mon_cx = radius * np.sin(theta_rad) + 2.0 * np.cos(-theta_rad)
    mon_cy = radius * np.cos(theta_rad) + 2.0 * np.sin(-theta_rad)
    mon_center = (mon_cx, mon_cy, wg_height / 2.0)

    # Because Tidy3D mode monitors must be axis-aligned, dynamically choose 
    # the normal orientation that minimizes obliqueness relative to propagation
    if 45 <= bend_angle_deg <= 135:
        mon_size = (3.0, 0, 2.5)  # Normal to Y
    else:
        mon_size = (0, 3.0, 2.5)  # Normal to X

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
        core_struct = next(
            (s for s in sim.structures if s.name in ["Ring Arc", "Waveguide In"]), 
            None
        )
        if core_struct:
            z_coord = float(core_struct.geometry.center[2])
        else:
            z_coord = float(sim.center[2])

    print(f"[Visualizer] Locking Bend slice view onto core center: X={x_coord}, Y={y_coord}, Z={z_coord}")
    return x_coord, y_coord, z_coord