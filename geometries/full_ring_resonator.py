"""
geometries.full_ring_resonator

Authors: Divyanshu Tiwari, Esraaj Sarkar Gupta
Date: 2026

Exposes: get_passive_simulation(), get_active_simulation(), get_optimal_plot_coords().
"""

from __future__ import annotations
from tarang import get_logger
from tarang._compat import renamed_params

from typing import Optional, Tuple

import numpy as np
import tidy3d as td

from ring_resonator.ring_resonator import RingResonator

logger = get_logger(__name__)


@renamed_params(substrate_medium="box_medium", substrate_height="box_height", wavelength0="wavelength_um")
def get_passive_simulation(
    radius: float,
    gap: float,

    wg_width: float,
    wg_height: float,

    core_medium: td.Medium,
    box_medium: td.Medium,

    wavelength_um: float,

    box_height: float = 4.0,

    top_cladding_medium: Optional[td.Medium] = None,
    top_cladding_height: float = 0.0,

    fine_steps_per_wvl: float = 20,
    coarse_steps_per_wvl: float = 15,

    wg_length: Optional[float] = None,
    run_time: Optional[float] = None,  # Duration in seconds; None keeps the automatic estimate.
) -> RingResonator:
    """
    Constructs a full-ring resonator passive simulation wrapped in a RingResonator object.
    Background simulation environment is set to air, with an optional explicit top cladding layer.
    """
    if wg_length is None:
        wg_length = 2 * radius + 5.000

    air_medium = td.Medium(permittivity=1.0)

    if top_cladding_medium is not None and top_cladding_height >= wg_height:
        hole_medium = top_cladding_medium
    else:
        hole_medium = air_medium

    # -- Domain Setup for Full Ring -- #
    y_wg_center: float = -(radius + wg_width + gap)

    # Y-bounds: Span from below the bus waveguide to above the top of the ring
    y_min = y_wg_center - (wg_width / 2.0) - 1.500
    y_max = (radius + wg_width / 2.0) + 1.500

    domain_size_x: float = wg_length
    domain_size_y: float = y_max - y_min
    sim_center_y: float = (y_max + y_min) / 2.0

    # -- Structures -- #
    substrate = td.Structure(
        geometry=td.Box(
            center=(0, 0, -box_height / 2.0),
            size=(td.inf, td.inf, box_height),
        ),
        medium=box_medium,
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

    ring_outer = td.Structure(
        geometry=td.Cylinder(
            center=(0, 0, wg_height / 2.0),
            radius=radius + wg_width / 2.0,
            length=wg_height,
            axis=2,
        ),
        medium=core_medium,
        name="Ring_Outer",
    )

    ring_inner = td.Structure(
        geometry=td.Cylinder(
            center=(0, 0, wg_height / 2.0),
            radius=radius - wg_width / 2.0,
            length=wg_height + 0.200,
            axis=2,
        ),
        medium=hole_medium,
        name="Ring_Inner_Hole",
    )

    bus_waveguide = td.Structure(
        geometry=td.Box(
            center=(0, y_wg_center, wg_height / 2.0),
            size=(wg_length + 2.0, wg_width, wg_height),
        ),
        medium=core_medium,
        name="Bus_Waveguide",
    )

    structures.extend([ring_outer, ring_inner, bus_waveguide])

    # -- Run Time Optimization (Cavity Lifetime for Resonant Recirculation) -- #
    ring_perimeter = 2.0 * np.pi * radius
    n_group_approx = 2.1
    round_trip_time = ring_perimeter / (td.C_0 / n_group_approx)
    calculated_run_time = max(
        1e-12,
        round_trip_time * 60.0,
    )

    freq0 = td.C_0 / wavelength_um
    try:
        n_core_approx = float(np.real(core_medium.n_complex(freq0)))
    except Exception:
        n_core_approx = 2.0

    dl_fine = wavelength_um / (n_core_approx * fine_steps_per_wvl)

    mesh_override = td.MeshOverrideStructure(
        geometry=td.Box(
            center=(0, sim_center_y, wg_height / 2.0),
            size=(domain_size_x, domain_size_y, wg_height + 0.500),
        ),
        dl=[dl_fine, dl_fine, dl_fine],
        enforce=True,
    )

    if run_time is None:
        run_time = calculated_run_time

    sim_passive = td.Simulation(
        center=(0.0, sim_center_y, wg_height / 2.0),
        size=(domain_size_x, domain_size_y, 3.000),
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=coarse_steps_per_wvl,
            wavelength=wavelength_um,
            override_structures=[mesh_override],
        ),
        structures=structures,
        run_time=run_time,
        boundary_spec=td.BoundarySpec(
            x=td.Boundary.pml(),
            y=td.Boundary.pml(),
            z=td.Boundary.pml(),
        ),
        medium=air_medium,
    )

    return RingResonator(
        sim=sim_passive,
        radius=radius,
        gap=gap,
        wg_width=wg_width,
        wg_height=wg_height,
        core_medium=core_medium,
        box_medium=box_medium,
        box_height=box_height,
        top_cladding_medium=(
            top_cladding_medium if top_cladding_medium is not None else air_medium
        ),
        top_cladding_height=top_cladding_height,
        fine_steps_per_wvl=fine_steps_per_wvl,
        coarse_steps_per_wvl=coarse_steps_per_wvl,
        wg_length=wg_length,
        wavelength_um=wavelength_um,
    )


@renamed_params(substrate_medium="box_medium", substrate_height="box_height", wavelength0="wavelength_um")
def get_active_simulation(
    radius: float,
    gap: float,

    wg_width: float,
    wg_height: float,

    core_medium: td.Medium,
    box_medium: td.Medium,

    wavelength_um: float,

    box_height: float = 4.0,

    top_cladding_medium: Optional[td.Medium] = None,
    top_cladding_height: float = 0.0,

    fine_steps_per_wvl: float = 20,
    coarse_steps_per_wvl: float = 15,

    wg_length: Optional[float] = None,
    run_time: Optional[float] = None,  # Duration in seconds; None keeps the automatic estimate.
) -> RingResonator:
    """Constructs an active full-ring simulation with mode sources and monitors across
    all cardinal points.
    """
    resonator_obj = get_passive_simulation(
        radius=radius,
        gap=gap,
        wg_width=wg_width,
        wg_height=wg_height,
        core_medium=core_medium,
        box_medium=box_medium,
        wavelength_um=wavelength_um,
        box_height=box_height,
        top_cladding_medium=top_cladding_medium,
        top_cladding_height=top_cladding_height,
        fine_steps_per_wvl=fine_steps_per_wvl,
        coarse_steps_per_wvl=coarse_steps_per_wvl,
        wg_length=wg_length,
        run_time=run_time,
    )

    sim_passive = resonator_obj.sim
    freq0 = td.C_0 / wavelength_um

    try:
        n_eff = float(np.real(core_medium.n_complex(freq0)))
    except Exception:
        n_eff = 2.0

    if wg_length is None:
        wg_length = 2 * radius + 5.000

    y_wg_center: float = -(radius + wg_width + gap)
    mon_w = wg_width * 3.5
    mon_h = wg_height * 2.5

    # ---- Channel Entry/Exit Ports ---- #
    source = td.ModeSource(
        center=(-wg_length / 2.0 + 1.0, y_wg_center, wg_height / 2.0),
        size=(0, mon_w, mon_h),
        direction="+",
        mode_spec=td.ModeSpec(
            num_modes=1,
            target_neff=n_eff,
        ),
        source_time=td.GaussianPulse(
            freq0=freq0,
            fwidth=freq0 / 15,
        ),
        name="Through_In",
    )

    monitor_thru = td.ModeMonitor(
        center=(wg_length / 2.0 - 1.0, y_wg_center, wg_height / 2.0),
        size=(0, mon_w, mon_h),
        freqs=[freq0],
        mode_spec=td.ModeSpec(
            num_modes=1,
            target_neff=n_eff,
        ),
        name="Through_Out",
    )

    # ---- Internal Track Audit Monitors (All 4 Cardinal Points) ---- #
    mon_ring_bot = td.ModeMonitor(
        center=(0.0, -radius, wg_height / 2.0),
        size=(0, mon_w, mon_h),
        freqs=[freq0],
        mode_spec=td.ModeSpec(
            num_modes=1,
            target_neff=n_eff,
        ),
        name="Ring_Mon_270_Bot",
    )

    mon_ring_right = td.ModeMonitor(
        center=(radius, 0.0, wg_height / 2.0),
        size=(mon_w, 0, mon_h),
        freqs=[freq0],
        mode_spec=td.ModeSpec(
            num_modes=1,
            target_neff=n_eff,
        ),
        name="Ring_Mon_0_Right",
    )

    mon_ring_top = td.ModeMonitor(
        center=(0.0, radius, wg_height / 2.0),
        size=(0, mon_w, mon_h),
        freqs=[freq0],
        mode_spec=td.ModeSpec(
            num_modes=1,
            target_neff=n_eff,
        ),
        name="Ring_Mon_90_Top",
    )

    mon_ring_left = td.ModeMonitor(
        center=(-radius, 0.0, wg_height / 2.0),
        size=(mon_w, 0, mon_h),
        freqs=[freq0],
        mode_spec=td.ModeSpec(
            num_modes=1,
            target_neff=n_eff,
        ),
        name="Ring_Mon_180_Left",
    )

    # ---- 2D Full Domain Field Map ---- #
    monitor_field = td.FieldMonitor(
        center=(0.0, sim_passive.center[1], wg_height / 2.0),
        size=(wg_length, sim_passive.size[1] - 1.0, 0),
        freqs=[freq0],
        name="coupling_field_map",
    )

    monitors_list = [
        monitor_thru,
        mon_ring_bot,
        mon_ring_right,
        mon_ring_top,
        mon_ring_left,
        monitor_field,
    ]

    sim_active = sim_passive.updated_copy(
        sources=[source],
        monitors=monitors_list,
    )

    resonator_obj.sim = sim_active
    return resonator_obj


get_active_sim = get_active_simulation
get_passive_sim = get_passive_simulation


def get_optimal_plot_coords(sim: td.Simulation) -> Tuple[float, float, float]:
    """Slicing helper focused on the critical point-coupling window (X=0 cross
    section).
    """
    x_coord = 0.0
    if sim.sources:
        y_coord = float(sim.sources[0].center[1])
        z_coord = float(sim.sources[0].center[2])
    else:
        y_coord = float(sim.center[1])
        z_coord = float(sim.center[2])

    logger.info(
        f"[Visualizer] Locking All-Pass slice view onto coupling center: X={x_coord}, Y={y_coord}, Z={z_coord}"
    )
    return x_coord, y_coord, z_coord
