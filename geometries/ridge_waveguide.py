"""
geometries.ridge_waveguide.py

This file defines and returns Tidy3D simulations of ridge waveguides on a
Bottom Oxide Layer, wrapped inside RidgeWaveguide objects.

Author: Esraaj Sarkar Gupta, Divyanshu Tiwari
Date: July 2026
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from waveguide.waveguide import RidgeWaveguide

import tidy3d as td
import numpy as np


def get_passive_simulation(
    width: float,
    height: float,
    length: float,

    core_medium: td.Medium,
    sub_medium: td.Medium,

    wavelength_um: float,

    substrate_height: float = 4.0,

    fine_min_steps_per_wvl: int = 20,
    coarse_min_steps_per_wvl: int = 10,

    margin_y: float = 2.0,  # Buffer space on each side of core in Y
    margin_z: float = 2.0,  # Air buffer above the core in Z

    run_time: float = 1e-12
) -> RidgeWaveguide:
    """
    Generates a Tidy3D simulation with independent control over core 
    and substrate dimensions, returning a wrapped RidgeWaveguide.
    """
    
    # Total domain dimensions:
    # Z-span covers substrate (below z=0), core (z=0 to h), and top air buffer
    sim_x = length
    sim_y = width + 2 * margin_y
    sim_z = substrate_height + height + margin_z
    sim_size = (sim_x, sim_y, sim_z)
    
    # Adjust Z-center so the origin (0,0,0) stays locked to the substrate-core interface
    z_center = (height + margin_z - substrate_height) / 2
    sim_center = (0, 0, z_center)

    # Core Waveguide (sits from z = 0 to z = height)
    core_geo = td.Box(
        center=(0, 0, height / 2),
        size=(td.inf, width, height)
    )
    core = td.Structure(geometry=core_geo, medium=core_medium)
    
    # Controlled Substrate Layer (sits from z = 0 down to z = -substrate_height)
    substrate_geo = td.Box(
        center=(0, 0, -substrate_height / 2), 
        size=(td.inf, td.inf, substrate_height)
    )
    substrate = td.Structure(geometry=substrate_geo, medium=sub_medium)
    
    # Mesh Override Region around core
    buffer_y = 0.5
    buffer_z = 0.5 

    freq0 = td.C_0 / wavelength_um
    eps_core = core_medium.eps_model(freq0)
    n_core = float(np.real(np.sqrt(eps_core)))
    
    fine_dl = wavelength_um / (n_core * fine_min_steps_per_wvl)

    override_geo = td.Box(
        center=(0, 0, height / 2),
        size=(td.inf, width + 2 * buffer_y, height + 2 * buffer_z)
    )
    
    mesh_override = td.MeshOverrideStructure(
        geometry=override_geo,
        dl=[fine_dl, fine_dl, fine_dl],
        enforce=True 
    )
    
    # Simulation Setup
    grid_spec = td.GridSpec.auto(
        wavelength=wavelength_um, 
        min_steps_per_wvl=coarse_min_steps_per_wvl,
        override_structures=[mesh_override]
    )
        
    boundary_spec = td.BoundarySpec(
        x=td.Boundary.pml(), 
        y=td.Boundary.pml(), 
        z=td.Boundary.pml()
    )
    
    sim = td.Simulation(
        center=sim_center,
        size=sim_size,
        grid_spec=grid_spec,
        structures=[substrate, core],
        boundary_spec=boundary_spec,
        run_time=run_time
    )

    from waveguide.waveguide import RidgeWaveguide

    return RidgeWaveguide(
        sim=sim,
        width=width,
        height=height,
        length=length,
        core_medium=core_medium,
        sub_medium=sub_medium,
        wavelength_um=wavelength_um,
        substrate_height=substrate_height,
        fine_min_steps_per_wvl=fine_min_steps_per_wvl,
        coarse_min_steps_per_wvl=coarse_min_steps_per_wvl,
        margin_y=margin_y,
        margin_z=margin_z,
        run_time=run_time
    )


def get_active_simulation(
    width: float,
    height: float,
    length: float,

    core_medium: td.Medium,
    sub_medium: td.Medium,

    wavelength_um: float,

    substrate_height: float = 4.0, 

    num_modes: int = 2,
    target_mode_index: int = 0,

    fine_min_steps_per_wvl: int = 20,
    coarse_min_steps_per_wvl: int = 10,

    margin_y: float = 2.0,
    margin_z: float = 2.0,

    run_time: float = 1e-12
) -> RidgeWaveguide:
    """
    Generates an active Tidy3D simulation with independent substrate control,
    returning a wrapped RidgeWaveguide.
    """
    
    sim_x = length
    sim_y = width + 2 * margin_y
    sim_z = substrate_height + height + margin_z
    sim_size = (sim_x, sim_y, sim_z)
    
    z_center = (height + margin_z - substrate_height) / 2
    sim_center = (0, 0, z_center)

    core_geo = td.Box(
        center=(0, 0, height / 2),
        size=(td.inf, width, height)
    )
    core = td.Structure(geometry=core_geo, medium=core_medium)
    
    substrate_geo = td.Box(
        center=(0, 0, -substrate_height / 2), 
        size=(td.inf, td.inf, substrate_height)
    )
    substrate = td.Structure(geometry=substrate_geo, medium=sub_medium)
    
    freq0 = td.C_0 / wavelength_um
    eps_core = core_medium.eps_model(freq0)
    n_core = float(np.real(np.sqrt(eps_core)))
    fine_dl = wavelength_um / (n_core * fine_min_steps_per_wvl)
                   
    buffer_y = 0.5
    buffer_z = 0.5 
    override_geo = td.Box(
        center=(0, 0, height / 2),
        size=(td.inf, width + 2 * buffer_y, height + 2 * buffer_z)
    )
    
    mesh_override = td.MeshOverrideStructure(
        geometry=override_geo,
        dl=[fine_dl, fine_dl, fine_dl],
        enforce=True 
    )
    
    pml_offset = 0.5 
    source_x = -length / 2 + pml_offset
    monitor_x = length / 2 - pml_offset
    
    plane_size = (0, sim_y, sim_z)
    
    fwidth = freq0 / 10 
    source_time = td.GaussianPulse(freq0=freq0, fwidth=fwidth)
    
    mode_spec = td.ModeSpec(num_modes=num_modes)
    
    mode_source = td.ModeSource(
        center=(source_x, 0, height / 2),
        size=plane_size,
        source_time=source_time,
        direction="+",
        mode_spec=mode_spec,
        mode_index=target_mode_index
    )
    
    mode_monitor = td.ModeMonitor(
        center=(monitor_x, 0, height / 2),
        size=plane_size,
        freqs=[freq0],
        mode_spec=mode_spec,
        name="output_mode_monitor"
    )
    
    grid_spec = td.GridSpec.auto(
        wavelength=wavelength_um, 
        min_steps_per_wvl=coarse_min_steps_per_wvl,
        override_structures=[mesh_override]
    )
        
    boundary_spec = td.BoundarySpec(
        x=td.Boundary.pml(), 
        y=td.Boundary.pml(), 
        z=td.Boundary.pml()
    )
    
    sim = td.Simulation(
        center=sim_center,
        size=sim_size,
        grid_spec=grid_spec,
        structures=[substrate, core],
        sources=[mode_source],
        monitors=[mode_monitor],
        boundary_spec=boundary_spec,
        run_time=run_time
    )

    from waveguide.waveguide import RidgeWaveguide

    return RidgeWaveguide(
        sim=sim,
        width=width,
        height=height,
        length=length,
        core_medium=core_medium,
        sub_medium=sub_medium,
        wavelength_um=wavelength_um,
        substrate_height=substrate_height,
        fine_min_steps_per_wvl=fine_min_steps_per_wvl,
        coarse_min_steps_per_wvl=coarse_min_steps_per_wvl,
        margin_y=margin_y,
        margin_z=margin_z,
        run_time=run_time
    )