"""
tarang.geometries.ridge_waveguide.py

This file defines and returns Tidy3D simulations of ridge waveguides on a
Bottom Oxide Layer with optional Top Cladding, wrapped inside RidgeWaveguide objects.

Author: Esraaj Sarkar Gupta, Divyanshu Tiwari
Date: July 2026

Exposes: get_passive_simulation(), get_active_simulation().
"""

from __future__ import annotations
from tarang._compat import renamed_params
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from tarang.waveguide.waveguide import RidgeWaveguide

import tidy3d as td
import numpy as np


@renamed_params(sub_medium="box_medium", substrate_height="box_height", fine_min_steps_per_wvl="fine_steps_per_wvl", coarse_min_steps_per_wvl="coarse_steps_per_wvl")
def get_passive_simulation(
    width: float,
    height: float,
    length: float,

    core_medium: td.Medium,
    box_medium: td.Medium,

    wavelength_um: float,

    top_cladding_medium: Optional[td.Medium] = None,
    top_cladding_height: float = 0.0,
    
    box_height: float = 4.0,

    fine_steps_per_wvl: int = 20,
    coarse_steps_per_wvl: int = 15,

    margin_y: float = 2.0,
    margin_z: float = 2.0,

    sim_size_y: float | None = None,
    sim_size_z: float | None = None,

    run_time: float = 1e-12,
) -> RidgeWaveguide:
    """
    Generates a Tidy3D simulation with independent control over core,
    substrate, and optional top cladding dimensions, returning a wrapped RidgeWaveguide.
    """

    sim_x = length
    sim_y = sim_size_y if sim_size_y is not None else (width + 2 * margin_y)

    total_struct_height = height + top_cladding_height

    if sim_size_z is not None:
        sim_z = sim_size_z
        effective_margin_z = sim_size_z - box_height - total_struct_height
        if effective_margin_z <= 0:
            raise ValueError(
                f"sim_size_z ({sim_size_z:.3f} μm) must exceed "
                f"box_height + height + top_cladding_height ({box_height + total_struct_height:.3f} μm)."
            )
    else:
        sim_z = box_height + total_struct_height + margin_z
        effective_margin_z = margin_z

    sim_size = (sim_x, sim_y, sim_z)

    z_center = (total_struct_height + effective_margin_z - box_height) / 2
    sim_center = (0, 0, z_center)

    structures = []

    # 1. BOX layer (sits from z = -box_height to z = 0)
    substrate_geo = td.Box(
        center=(0, 0, -box_height / 2),
        size=(td.inf, td.inf, box_height),
    )
    substrate = td.Structure(
        geometry=substrate_geo,
        medium=box_medium,
    )
    structures.append(substrate)

    # 2. Top Cladding Layer (fills z = 0 up to z = height + top_cladding_height)
    # Note: Placed BEFORE core so core overrides it where they overlap.
    if top_cladding_medium is not None and top_cladding_height > 0:
        top_clad_geo = td.Box(
            center=(0, 0, total_struct_height / 2),
            size=(td.inf, td.inf, total_struct_height),
        )
        top_clad = td.Structure(
            geometry=top_clad_geo,
            medium=top_cladding_medium,
        )
        structures.append(top_clad)

    # 3. Core Waveguide (sits from z = 0 to z = height)
    core_geo = td.Box(
        center=(0, 0, height / 2),
        size=(td.inf, width, height),
    )
    core = td.Structure(
        geometry=core_geo,
        medium=core_medium,
    )
    structures.append(core)

    # Mesh Override Region around core
    buffer_y = 0.5
    buffer_z = 0.5

    freq0 = td.C_0 / wavelength_um
    eps_core = core_medium.eps_model(freq0)
    n_core = float(np.real(np.sqrt(eps_core)))

    fine_dl = wavelength_um / (n_core * fine_steps_per_wvl)

    override_geo = td.Box(
        center=(0, 0, height / 2),
        size=(td.inf, width + 2 * buffer_y, height + 2 * buffer_z),
    )

    mesh_override = td.MeshOverrideStructure(
        geometry=override_geo,
        dl=[fine_dl, fine_dl, fine_dl],
        enforce=True,
    )

    grid_spec = td.GridSpec.auto(
        wavelength=wavelength_um,
        min_steps_per_wvl=coarse_steps_per_wvl,
        override_structures=[mesh_override],
    )

    boundary_spec = td.BoundarySpec(
        x=td.Boundary.pml(),
        y=td.Boundary.pml(),
        z=td.Boundary.pml(),
    )

    sim = td.Simulation(
        center=sim_center,
        size=sim_size,
        grid_spec=grid_spec,
        structures=structures,
        boundary_spec=boundary_spec,
        run_time=run_time,
    )

    from tarang.waveguide.waveguide import RidgeWaveguide

    return RidgeWaveguide(
        sim=sim,
        width=width,
        height=height,
        length=length,
        core_medium=core_medium,
        box_medium=box_medium,
        top_cladding_medium=top_cladding_medium,
        top_cladding_height=top_cladding_height,
        wavelength_um=wavelength_um,
        box_height=box_height,
        fine_steps_per_wvl=fine_steps_per_wvl,
        coarse_steps_per_wvl=coarse_steps_per_wvl,
        margin_y=margin_y,
        margin_z=margin_z,
        run_time=run_time,
    )


@renamed_params(sub_medium="box_medium", substrate_height="box_height", fine_min_steps_per_wvl="fine_steps_per_wvl", coarse_min_steps_per_wvl="coarse_steps_per_wvl")
def get_active_simulation(
    width: float,
    height: float,
    length: float,

    core_medium: td.Medium,
    box_medium: td.Medium,

    wavelength_um: float,

    top_cladding_medium: Optional[td.Medium] = None,
    top_cladding_height: float = 0.0,

    box_height: float = 4.0,

    num_modes: int = 1,
    target_mode_index: int = 0,

    fine_steps_per_wvl: int = 20,
    coarse_steps_per_wvl: int = 15,

    margin_y: float = 2.0,
    margin_z: float = 2.0,

    sim_size_y: float | None = None,
    sim_size_z: float | None = None,
    
    run_time: float = 1e-12,
) -> RidgeWaveguide:
    """
    Generates an active Tidy3D simulation with mode sources/monitors and
    support for optional top cladding.
    """

    sim_x = length
    sim_y = sim_size_y if sim_size_y is not None else (width + 2 * margin_y)

    total_struct_height = height + top_cladding_height

    if sim_size_z is not None:
        sim_z = sim_size_z
        effective_margin_z = sim_size_z - box_height - total_struct_height
        if effective_margin_z <= 0:
            raise ValueError(
                f"sim_size_z ({sim_size_z:.3f} μm) must exceed "
                f"box_height + height + top_cladding_height ({box_height + total_struct_height:.3f} μm)."
            )
    else:
        sim_z = box_height + total_struct_height + margin_z
        effective_margin_z = margin_z

    sim_size = (sim_x, sim_y, sim_z)

    z_center = (total_struct_height + effective_margin_z - box_height) / 2
    sim_center = (0, 0, z_center)

    structures = []

    # Substrate
    substrate_geo = td.Box(
        center=(0, 0, -box_height / 2),
        size=(td.inf, td.inf, box_height),
    )
    substrate = td.Structure(
        geometry=substrate_geo,
        medium=box_medium,
    )
    structures.append(substrate)

    # Top Cladding
    if top_cladding_medium is not None and top_cladding_height > 0:
        top_clad_geo = td.Box(
            center=(0, 0, total_struct_height / 2),
            size=(td.inf, td.inf, total_struct_height),
        )
        top_clad = td.Structure(
            geometry=top_clad_geo,
            medium=top_cladding_medium,
        )
        structures.append(top_clad)

    # Core
    core_geo = td.Box(
        center=(0, 0, height / 2),
        size=(td.inf, width, height),
    )
    core = td.Structure(
        geometry=core_geo,
        medium=core_medium,
    )
    structures.append(core)

    freq0 = td.C_0 / wavelength_um
    eps_core = core_medium.eps_model(freq0)
    n_core = float(np.real(np.sqrt(eps_core)))
    fine_dl = wavelength_um / (n_core * fine_steps_per_wvl)

    buffer_y = 0.5
    buffer_z = 0.5
    override_geo = td.Box(
        center=(0, 0, height / 2),
        size=(td.inf, width + 2 * buffer_y, height + 2 * buffer_z),
    )

    mesh_override = td.MeshOverrideStructure(
        geometry=override_geo,
        dl=[fine_dl, fine_dl, fine_dl],
        enforce=True,
    )

    pml_offset = 0.5
    source_x = -length / 2 + pml_offset
    monitor_x = length / 2 - pml_offset

    plane_size = (0, sim_y, sim_z)

    fwidth = freq0 / 10
    source_time = td.GaussianPulse(
        freq0=freq0,
        fwidth=fwidth,
    )

    mode_spec = td.ModeSpec(num_modes=num_modes)

    mode_source = td.ModeSource(
        center=(source_x, 0, height / 2),
        size=plane_size,
        source_time=source_time,
        direction="+",
        mode_spec=mode_spec,
        mode_index=target_mode_index,
    )

    mode_monitor = td.ModeMonitor(
        center=(monitor_x, 0, height / 2),
        size=plane_size,
        freqs=[freq0],
        mode_spec=mode_spec,
        name="output_mode_monitor",
    )

    grid_spec = td.GridSpec.auto(
        wavelength=wavelength_um,
        min_steps_per_wvl=coarse_steps_per_wvl,
        override_structures=[mesh_override],
    )

    boundary_spec = td.BoundarySpec(
        x=td.Boundary.pml(),
        y=td.Boundary.pml(),
        z=td.Boundary.pml(),
    )

    sim = td.Simulation(
        center=sim_center,
        size=sim_size,
        grid_spec=grid_spec,
        structures=structures,
        sources=[mode_source],
        monitors=[mode_monitor],
        boundary_spec=boundary_spec,
        run_time=run_time,
    )

    from tarang.waveguide.waveguide import RidgeWaveguide

    return RidgeWaveguide(
        sim=sim,
        width=width,
        height=height,
        length=length,
        core_medium=core_medium,
        box_medium=box_medium,
        top_cladding_medium=top_cladding_medium,
        top_cladding_height=top_cladding_height,
        wavelength_um=wavelength_um,
        box_height=box_height,
        fine_steps_per_wvl=fine_steps_per_wvl,
        coarse_steps_per_wvl=coarse_steps_per_wvl,
        margin_y=margin_y,
        margin_z=margin_z,
        run_time=run_time,
    )
