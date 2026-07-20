"""
geometries.ridge_waveguide.py

This file defines and returns Tidy3D simulations of ridge waveguides on a
Bottom Oxide Layer.

Author: Esraaj Sarkar Gupta, Divyanshu Tiwari
Date: July 2026
"""

import tidy3d as td
import numpy as np

def get_passive_simulation(
    width: float,
    height: float,

    length: float,

    core_medium: td.Medium,
    sub_medium: td.Medium,

    wavelength_um: float,

    fine_min_steps_per_wvl: int = 20,
    coarse_min_steps_per_wvl : int = 10,

    margin_y: float = 2.0,  # Buffer space on each side of the core in Y
    margin_z: float = 2.0,  # Buffer space above/below the core in Z

    run_time : float = 1e-12
) -> td.Simulation:
    """
    Docstring for get_simulation
    
    :param width: Description
    :type width: float
    :param height: Description
    :type height: float
    :param length: Description
    :type length: float
    :param core_medium: Description
    :type core_medium: td.Medium
    :param sub_medium: Description
    :type sub_medium: td.Medium
    :param wl_0: Description
    :type wl_0: float
    :param fine_min_steps_per_wvl: Description
    :type fine_min_steps_per_wvl: int
    :param coarse_min_steps_per_wvl: Description
    :type coarse_min_steps_per_wvl: int
    :param margin_y: Description
    :type margin_y: float
    :param margin_z: Description
    :type margin_z: float
    :param run_time: Description
    :type run_time: float
    :return: Description
    :rtype: Any

    Generates a Tidy3D simulation with a fully dynamic domain size 
    calculated from the waveguide geometry.
    """
    
    # Dynamically calculate simulation dimensions
    sim_x = length
    sim_y = width + 2 * margin_y
    sim_z = height + 2 * margin_z
    sim_size = (sim_x, sim_y, sim_z)
    
    # Center the entire simulation box perfectly around the core waveguide
    sim_center = (0, 0, height / 2)

    # Define the Core Waveguide
    core_geo = td.Box(
        center=(0, 0, height / 2),
        size=(td.inf, width, height)
    )
    core = td.Structure(geometry=core_geo, medium=core_medium)
    
    # Dynamic Substrate
    # It starts at the core interface (z=0) and extends all the way down past the PML boundary
    sub_thickness = sim_z 
    substrate_geo = td.Box(
        center=(0, 0, -sub_thickness / 2), 
        size=(td.inf, td.inf, sub_thickness)
    )
    substrate = td.Structure(geometry=substrate_geo, medium=sub_medium)
    
    # Mesh Override Region (stays locked to core geometry)
    buffer_y = 0.5
    buffer_z = 0.5 

    # Computing the fine mesh geometry
    freq0 = td.C_0 / wl_0
    eps_core = core_medium.eps_model(freq0)
    n_core = float(np.real(np.sqrt(eps_core)))
    
    # Calculate explicit spatial step size (dl) required for the fine mesh region
    fine_dl = wl_0 / (n_core * fine_min_steps_per_wvl)
                   

    override_geo = td.Box(
        center=(0, 0, height / 2),
        size=(td.inf, width + 2 * buffer_y, height + 2 * buffer_z)
    )
    
    mesh_override = td.MeshOverrideStructure(
        geometry=override_geo,
        dl=[fine_dl, fine_dl, fine_dl],  # Dynamically computed fine grid
        enforce=True 
    )
    
    # -- Simulation Setup -- #
    grid_spec = td.GridSpec.auto(
        wavelength=wl_0, 
        min_steps_per_wvl=coarse_min_steps_per_wvl,
        override_structures=[mesh_override]
    )
        
    boundary_spec = td.BoundarySpec(
        x=td.Boundary.pml(), 
        y=td.Boundary.pml(), 
        z=td.Boundary.pml()
    )
    
    return td.Simulation(
        center=sim_center,
        size=sim_size,
        grid_spec=grid_spec,
        structures=[substrate, core],
        boundary_spec=boundary_spec,
        run_time=run_time
    )


def get_active_simulation(
    width: float,
    height: float,
    length: float,

    core_medium: td.Medium,
    sub_medium: td.Medium,

    wavelength_um: float,

    num_modes: int = 2,
    target_mode_index: int = 0,

    fine_min_steps_per_wvl: int = 20,
    coarse_min_steps_per_wvl: int = 10,

    margin_y: float = 2.0,
    margin_z: float = 2.0,

    run_time: float = 1e-12
) -> td.Simulation:
    """
    Docstring for get_active_simulation
    
    :param width: Description
    :type width: float
    :param height: Description
    :type height: float
    :param length: Description
    :type length: float
    :param core_medium: Description
    :type core_medium: td.Medium
    :param sub_medium: Description
    :type sub_medium: td.Medium
    :param wl_0: Description
    :type wl_0: float
    :param num_modes: Description
    :type num_modes: int
    :param target_mode_index: Description
    :type target_mode_index: int
    :param fine_min_steps_per_wvl: Description
    :type fine_min_steps_per_wvl: int
    :param coarse_min_steps_per_wvl: Description
    :type coarse_min_steps_per_wvl: int
    :param margin_y: Description
    :type margin_y: float
    :param margin_z: Description
    :type margin_z: float
    :param run_time: Description
    :type run_time: float
    :return: Description
    :rtype: Simulation

    Generates an active Tidy3D simulation with a Mode Source injected at the input 
    and a Mode Monitor recording at the output.
    """
    
    # Dynamically calculate simulation dimensions
    sim_x = length
    sim_y = width + 2 * margin_y
    sim_z = height + 2 * margin_z
    sim_size = (sim_x, sim_y, sim_z)
    
    # Center the entire simulation box perfectly around the core waveguide
    sim_center = (0, 0, height / 2)

    # Define Structures (Core & Substrate)
    core_geo = td.Box(
        center=(0, 0, height / 2),
        size=(td.inf, width, height)
    )
    core = td.Structure(geometry=core_geo, medium=core_medium)
    
    sub_thickness = sim_z 
    substrate_geo = td.Box(
        center=(0, 0, -sub_thickness / 2), 
        size=(td.inf, td.inf, sub_thickness)
    )
    substrate = td.Structure(geometry=substrate_geo, medium=sub_medium)
    
    # Dynamic Fine Mesh Math
    freq0 = td.C_0 / wl_0
    eps_core = core_medium.eps_model(freq0)
    n_core = float(np.real(np.sqrt(eps_core)))
    fine_dl = wl_0 / (n_core * fine_min_steps_per_wvl)
                   
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
    
    # Active Elements: Source and Monitor Positions
    # Place them offset from the PML edges along the X-axis (e.g., 0.5 µm inwards)
    pml_offset = 0.5 
    source_x = -length / 2 + pml_offset
    monitor_x = length / 2 - pml_offset
    
    # The source/monitor planes should span the entire Y-Z cross-section of the domain
    plane_size = (0, sim_y, sim_z)
    
    # Define a clean Gaussian pulse for the source time profile
    fwidth = freq0 / 10  # 10% bandwidth is standard for clean data
    source_time = td.GaussianPulse(freq0=freq0, fwidth=fwidth)
    
    # The mode specification blueprint
    mode_spec = td.ModeSpec(num_modes=num_modes)
    
    # Add the Mode Source at the input
    mode_source = td.ModeSource(
        center=(source_x, 0, height / 2),
        size=plane_size,

        source_time=source_time,
        direction="+",

        mode_spec=mode_spec,
        mode_index=target_mode_index
    )
    
    # Add the Mode Monitor at the output
    mode_monitor = td.ModeMonitor(
        center=(monitor_x, 0, height / 2),
        size=plane_size,

        freqs=[freq0],  # Track our target frequency
        mode_spec=mode_spec,
        name="output_mode_monitor"
    )
    
    # Final Setup Assembly
    grid_spec = td.GridSpec.auto(
        wavelength=wl_0, 
        min_steps_per_wvl=coarse_min_steps_per_wvl,
        override_structures=[mesh_override]
    )
        
    boundary_spec = td.BoundarySpec(
        x=td.Boundary.pml(), 
        y=td.Boundary.pml(), 
        z=td.Boundary.pml()
    )
    
    return td.Simulation(
        center=sim_center,
        size=sim_size,
        grid_spec=grid_spec,

        structures=[substrate, core],
        sources=[mode_source],
        monitors=[mode_monitor],
        
        boundary_spec=boundary_spec,
        run_time=run_time
    )