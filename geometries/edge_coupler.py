"""
geometries.edge_coupler.py

Authors: Esraaj Sarkar Gupta, Divyanshu Tiwari
Date: 2026
"""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
import tidy3d as td

from edgeCoupler import edgeCoupler
from media.medium_helper import get_medium

def _estimate_run_time(sim_length_x: float, n_max: float = 4.0, num_round_trips: float = 10.0) -> float:
    """
    Estimate a run_time (s) sufficient for the pulse to traverse the simulation
    domain and decay, based on the domain's x-extent and an assumed worst-case
    group index.

    Args:
        sim_length_x: total x-extent of the simulation domain (um).
        n_max: assumed maximum group index of any mode/material in the domain,
            used as a conservative upper bound on propagation delay.
        num_round_trips: number of one-way transit times to allow for decay of
            reflections/resonances before the pulse is considered settled.
    """
    transit_time = (sim_length_x * 1e-6) * n_max / td.C_0  # convert um -> m
    return max(5e-12, num_round_trips * transit_time)


def get_passive_simulation(
        tip_width : float,
        taper_length : float,
        wg_width : float,
        wg_height : float,
        core_medium : td.Medium,
        substrate_medium : td.Medium,
        wavelength0 : float,
        top_cladding_medium : td.Medium = get_medium("air"),
        substrate_height : float = 4.0,
        top_cladding_height : float = 0.0,
        fine_steps_per_wvl : int = 20,
        coarse_steps_per_wvl : int = 15,
        settling_buffer_length : Optional[float] = None,
        symmetry : Optional[Tuple[int, int, int]] = None,
) -> edgeCoupler:
    
    if symmetry is None:
        symmetry = (0, 0, 0)
    
    # This will have to wait...
    elif symmetry != (0, 0, 0):
        raise NotImplementedError(
            f"symmetry={symmetry} requested, but symmetry support is not yet validated "
            "for this geometry (see notes on GaussianBeam parity and monitor-data scaling "
            "under symmetry). Pass symmetry=(0, 0, 0) or omit it for now."
        )
    
    # Some gaurdrails for incorrect symmetry (applicable later I'm sure)
    if symmetry[0] != 0:
        raise ValueError(
            "symmetry[0] (x-axis) must be 0: the taper geometry varies continuously "
            "along the propagation direction and has no x-mirror plane."
        )
    if symmetry[2] != 0 and not (
        top_cladding_medium == substrate_medium and top_cladding_height == substrate_height
    ):
        raise ValueError(
            "symmetry[2] (z-axis) requires top_cladding_medium == substrate_medium and "
            "top_cladding_height == substrate_height; otherwise the structure is not "
            "z-symmetric and tidy3d will mirror only the half you defined."
        )
    
    # Resolve settling buffer length dynamically if not supplied
    L_buffer = settling_buffer_length if settling_buffer_length is not None else max(5.0, 3.5 * wavelength0)
    substrate_length = taper_length + L_buffer + 4.0

    # 1. Substrate
    substrate = td.Structure(
        geometry=td.Box(
            center=(substrate_length / 2, 0, -wg_height / 2 - substrate_height / 2),
            size=(substrate_length, td.inf, substrate_height)
        ),
        medium=substrate_medium
    )

    # 2. Direct Taper
    vertices = [
        (0, -tip_width / 2),
        (taper_length, -wg_width / 2),
        (taper_length, wg_width / 2),
        (0, tip_width / 2)
    ]
    taper = td.Structure(
        geometry=td.PolySlab(
            vertices=vertices,
            axis=2, 
            slab_bounds=(-wg_height / 2, wg_height / 2)
        ),
        medium=core_medium
    )

    # 3. Straight Waveguide
    straight_wg = td.Structure(
        geometry=td.Box(
            center=(taper_length + L_buffer / 2, 0, 0),
            size=(L_buffer, wg_width, wg_height)
        ),
        medium=core_medium
    )

    structures = [substrate, taper, straight_wg]
    
    # Explicit finite top cladding layer
    if top_cladding_height > 0:
        top_cladding_layer = td.Structure(
            geometry=td.Box(
                center=(substrate_length / 2, 0, wg_height / 2 + top_cladding_height / 2),
                size=(substrate_length, td.inf, top_cladding_height)
            ),
            medium=top_cladding_medium
        )
        structures.append(top_cladding_layer)

    # Mesh Override Region around core
    freq0 = td.C_0 / wavelength0
    eps_core = core_medium.eps_model(freq0)
    n_core = float(np.real(np.sqrt(eps_core)))
    fine_dl = wavelength0 / (n_core * fine_steps_per_wvl)

    buffer_y = 0.5
    buffer_z = 0.5 
    override_geo = td.Box(
        center=(0, 0, 0),
        size=(td.inf, wg_width + 2 * buffer_y, wg_height + 2 * buffer_z)
    )
    
    mesh_override = td.MeshOverrideStructure(
        geometry=override_geo,
        dl=[fine_dl, fine_dl, fine_dl],
        enforce=True 
    )

    grid_spec = td.GridSpec.auto(
        wavelength=wavelength0, 
        min_steps_per_wvl=coarse_steps_per_wvl,
        override_structures=[mesh_override]
    )

    boundary_spec = td.BoundarySpec.all_sides(boundary=td.PML())

    sim = td.Simulation(
        size=(taper_length + L_buffer + 6.0, 6.0, 6.0),
        center=((taper_length + L_buffer) / 2, 0, 0),
        structures=structures,
        # Passive (geometry-only) simulation: no field decay to wait for, fixed short run_time is fine.
        run_time=5e-12, 
        medium=top_cladding_medium,
        grid_spec=grid_spec,
        boundary_spec=boundary_spec,
        symmetry=symmetry
    )

    return edgeCoupler(
        tip_width=tip_width, taper_length=taper_length, wg_width=wg_width, wg_height=wg_height,
        core_medium=core_medium, substrate_medium=substrate_medium, top_cladding_medium=top_cladding_medium,
        wavelength0=wavelength0, substrate_height=substrate_height, top_cladding_height=top_cladding_height,
        fine_steps_per_wvl=fine_steps_per_wvl, coarse_steps_per_wvl=coarse_steps_per_wvl,
        settling_buffer_length=L_buffer, substrate_length=substrate_length,
        sim=sim, in_coupling=False, out_coupling=False, symmetry=symmetry
    )


def get_active_simulation(
        tip_width : float,
        taper_length : float,
        wg_width : float,
        wg_height : float,
        core_medium : td.Medium,
        substrate_medium : td.Medium,
        wavelength0 : float,
        top_cladding_medium : td.Medium = get_medium("air"),
        substrate_height : float = 4.0,
        top_cladding_height : float = 0.0,
        fine_steps_per_wvl : int = 20,
        coarse_steps_per_wvl : int = 15,
        settling_buffer_length : Optional[float] = None,
        in_coupling : Optional[bool] = None,
        out_coupling : Optional[bool] = None,
        symmetry : Optional[Tuple[int, int, int]] = None,
) -> edgeCoupler:
    
    if symmetry is None:
        symmetry = (0, 0, 0)
        
    # Handle directional mode defaults and error validation
    if in_coupling is None and out_coupling is None:
        in_coupling = False
        out_coupling = True
    elif in_coupling is not None and out_coupling is None:
        out_coupling = not in_coupling
    elif in_coupling is None and out_coupling is not None:
        in_coupling = not out_coupling

    if in_coupling == out_coupling:
        raise ValueError(
            f"Invalid directional configuration: in_coupling={in_coupling}, out_coupling={out_coupling}. "
            "Exactly one of 'in_coupling' or 'out_coupling' must be True."
        )

    # Resolve settling buffer length dynamically if not supplied
    L_buffer = settling_buffer_length if settling_buffer_length is not None else max(5.0, 3.5 * wavelength0)
    substrate_length = taper_length + L_buffer + 4.0
    
    freq0 = td.C_0 / wavelength0
    fwidth = freq0 / 10.0

    # Build structures
    substrate = td.Structure(
        geometry=td.Box(
            center=(substrate_length / 2, 0, -wg_height / 2 - substrate_height / 2),
            size=(substrate_length, td.inf, substrate_height)
        ), medium=substrate_medium
    )
    taper = td.Structure(
        geometry=td.PolySlab(
            vertices=[(0, -tip_width / 2), (taper_length, -wg_width / 2), (taper_length, wg_width / 2), (0, tip_width / 2)],
            axis=2, slab_bounds=(-wg_height / 2, wg_height / 2)
        ), medium=core_medium
    )
    straight_wg = td.Structure(
        geometry=td.Box(
            center=(taper_length + L_buffer / 2, 0, 0),
            size=(L_buffer, wg_width, wg_height)
        ), medium=core_medium
    )

    structures = [substrate, taper, straight_wg]
    
    # Explicit finite top cladding layer
    if top_cladding_height > 0:
        top_cladding_layer = td.Structure(
            geometry=td.Box(
                center=(substrate_length / 2, 0, wg_height / 2 + top_cladding_height / 2),
                size=(substrate_length, td.inf, top_cladding_height)
            ),
            medium=top_cladding_medium
        )
        structures.append(top_cladding_layer)

    # Mesh Override Region around core
    eps_core = core_medium.eps_model(freq0)
    n_core = float(np.real(np.sqrt(eps_core)))
    fine_dl = wavelength0 / (n_core * fine_steps_per_wvl)

    buffer_y = 0.5
    buffer_z = 0.5 
    override_geo = td.Box(
        center=(0, 0, 0),
        size=(td.inf, wg_width + 2 * buffer_y, wg_height + 2 * buffer_z)
    )
    
    mesh_override = td.MeshOverrideStructure(
        geometry=override_geo,
        dl=[fine_dl, fine_dl, fine_dl],
        enforce=True 
    )

    grid_spec = td.GridSpec.auto(
        wavelength=wavelength0, 
        min_steps_per_wvl=coarse_steps_per_wvl,
        override_structures=[mesh_override]
    )

    # Configure sources and monitors based on direction
    if in_coupling:
        # Fiber -> Chip (+X direction)
        source = td.GaussianBeam(
            center=(-2.0, 0, 0),
            size=(0, 6.0, 6.0),
            source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
            direction='+',
            waist_radius=1.5, 
            pol_angle=0
        )
        monitor_source = td.FluxMonitor(center=(-1.0, 0, 0), size=(0, 6.0, 6.0), freqs=[freq0], name='flux_fiber')
        monitor_tip = td.FluxMonitor(center=(0.1, 0, 0), size=(0, tip_width + 1.0, wg_height + 1.0), freqs=[freq0], name='flux_captured')
        monitor_mode = td.ModeMonitor(
            center=(taper_length + L_buffer - 0.5, 0, 0),
            size=(0, wg_width + 2.0, wg_height + 2.0),
            freqs=[freq0],
            mode_spec=td.ModeSpec(num_modes=1, target_neff=2.0),
            name='mode_transmission'
        )
    else:
        # Chip -> Fiber (-X direction)
        source = td.ModeSource(
            center=(taper_length + L_buffer - 0.5, 0, 0),
            size=(0, wg_width + 2.0, wg_height + 2.0),
            source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
            direction='-',
            mode_spec=td.ModeSpec(num_modes=1, target_neff=2.0),
            mode_index=0
        )
        monitor_source = td.FluxMonitor(center=(taper_length + L_buffer - 1.0, 0, 0), size=(0, wg_width + 2.0, wg_height + 2.0), freqs=[freq0], name='flux_guided_input')
        monitor_tip = td.FluxMonitor(center=(0.1, 0, 0), size=(0, tip_width + 1.0, wg_height + 1.0), freqs=[freq0], name='flux_captured')
        monitor_mode = td.FluxMonitor(center=(-1.0, 0, 0), size=(0, 6.0, 6.0), freqs=[freq0], name='flux_fiber')

    monitor_field_xy = td.FieldMonitor(center=(taper_length / 2, 0, 0), size=(td.inf, td.inf, 0), freqs=[freq0], name='field_xy')
    monitor_field_xz = td.FieldMonitor(center=(taper_length / 2, 0, 0), size=(td.inf, 0, td.inf), freqs=[freq0], name='field_xz')

    boundary_spec = td.BoundarySpec.all_sides(boundary=td.PML())
    sim_length_x = taper_length + L_buffer + 6.0
    run_time = _estimate_run_time(sim_length_x, n_max=n_core)

    sim = td.Simulation(
        size=(taper_length + L_buffer + 6.0, 6.0, 6.0),
        center=((taper_length + L_buffer) / 2, 0, 0),
        structures=structures,
        sources=[source],
        monitors=[monitor_source, monitor_tip, monitor_mode, monitor_field_xy, monitor_field_xz],
        run_time=run_time,
        medium=top_cladding_medium,
        grid_spec=grid_spec,
        boundary_spec=boundary_spec,
        symmetry=symmetry
    )

    return edgeCoupler(
        tip_width=tip_width, taper_length=taper_length, wg_width=wg_width, wg_height=wg_height,
        core_medium=core_medium, substrate_medium=substrate_medium, top_cladding_medium=top_cladding_medium,
        wavelength0=wavelength0, substrate_height=substrate_height, top_cladding_height=top_cladding_height,
        fine_steps_per_wvl=fine_steps_per_wvl, coarse_steps_per_wvl=coarse_steps_per_wvl,
        settling_buffer_length=L_buffer, substrate_length=substrate_length,
        sim=sim, in_coupling=in_coupling, out_coupling=out_coupling, symmetry=symmetry
    )

def get_optimal_plot_coords(sim: td.Simulation) -> Tuple[float, float, float]:
    """
    Slicing helper focused on the critical point-coupling window (X=0 cross section).
    """
    x_coord = 0.0
    if sim.sources:
        y_coord = float(sim.sources[0].center[1])
        z_coord = float(sim.sources[0].center[2])
    else:
        y_coord = float(sim.center[1])
        z_coord = float(sim.center[2])

    return x_coord, y_coord, z_coord


# Problems with Symmetry
"""
1) The GaussianBeam's own parity isn't verified, and it's a distinct question from the mode's parity.
    The GaussianBeam guidance came from the out-coupling branch's ModeSource monitor. A mode solver
    with the imposed symmetry will naurally restrict itself to the compatible mode. However, the in-coupling
    branch is different: GaussianBeam is not solved for -- it's an explicitly specified free-space field
    profile. Whether or not the dominant field coponent is even or odd across y = 0 is entirely dependent on
    Tidy3D's convention for pol_angle/angle_theta and not on the waveguide mode its attempting to couple into.

    Temp Solution: Restrict in_coupling optimization fully.

2) IMPORTANT: Is the reported data already full-domain, or does it need manual scaling?
    Tidy3D's own FAQ on symmetry states plainly that if symmetries are used in the simulation, monitor data returned
    to you covers only the simulated (reduced) portion of the domain, not the symmetric extension, and gives an explicit
    example (mode volume) where you must manually multiply by the appropriate factor: 2, 4, or 8 depending on how many
    symmetry planes are active — to recover the full-domain quantity.

    This matters directly since all the efficiency computations are reliant on measurements from the mode
    monitors and flux monitors.


"""