"""
tarang.geometries.partial_ring_resonator.py

Authors: Esraaj Sarkar Gupta
Date: 2026

Constructs a partial ring resonator + waveguide simulation physically clipped
to a specified span angle centered at (0, -R), wrapped in a PartialRingResonator.

This module defines a simulation with Absorber layers in the PML, specifically to
handle guided wave truncation. Light that leaves the end of a ring hits the PML at
oblique angles.

Exposes: get_passive_simulation(), get_active_simulation(), get_optimal_plot_coords().
"""

from __future__ import annotations
from tarang import get_logger
from tarang._compat import renamed_params

from typing import Optional, Tuple
import warnings
import numpy as np
import tidy3d as td

from tarang.geometries._modes import core_target_neff

from tarang.ring_resonator.ring_resonator import PartialRingResonator

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

    span_angle_rad: float,  # In Radians
    num_absorber_layers: int = 60,

    box_height: float = 4.0,
    top_cladding_medium: Optional[td.Medium] = None,
    top_cladding_height: float = 0.0,

    fine_steps_per_wvl: float = 20,
    coarse_steps_per_wvl: float = 15,

    wg_length: Optional[float] = None,
    run_time: Optional[float] = None,  # Duration in seconds; None keeps the automatic estimate.
) -> PartialRingResonator:
    """Constructs a passive simulation with the ring physically clipped to
    span_angle_rad.
    """

    span_angle_deg = np.rad2deg(span_angle_rad)

    if span_angle_deg > 180.0:
        raise ValueError(
            f"Received a span angle of {span_angle_deg:.1f}°. Span angles > 180° are not supported."
        )
    elif span_angle_deg == 180.0:
        warnings.warn(
            "For a 180° span angle, using the 'half_ring_resonator' module is recommended for better performance.",
            UserWarning,
            stacklevel=3,  # caller of the @renamed_params wrapper
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

    domain_size_x = max(
        4.0,
        2.0 * x_arc_max + 3.0,
    )
    if wg_length is not None:
        domain_size_x = max(
            domain_size_x,
            wg_length,
        )

    bus_length = domain_size_x + 2.0
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

    ring_outer_geom = (
        td.Cylinder(
            center=(0, 0, wg_height / 2.0),
            radius=r_outer,
            length=wg_height,
            axis=2,
        )
        & clip_box
    )

    ring_inner_geom = (
        td.Cylinder(
            center=(0, 0, wg_height / 2.0),
            radius=r_inner,
            length=wg_height + 0.200,
            axis=2,
        )
        & clip_box
    )

    ring_outer = td.Structure(
        geometry=ring_outer_geom,
        medium=core_medium,
        name="Ring_Outer",
    )
    ring_inner = td.Structure(
        geometry=ring_inner_geom,
        medium=hole_medium,
        name="Ring_Inner_Hole",
    )

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
    calculated_run_time = max(
        5e-13,
        (max_path_length / (td.C_0 / n_group_approx)) * 3.0,
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

    return PartialRingResonator(
        span_angle_rad=span_angle_rad,
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
        wg_length=domain_size_x,
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

    span_angle_rad: float,
    num_absorber_layers: int = 60,

    box_height: float = 4.0,

    top_cladding_medium: Optional[td.Medium] = None,
    top_cladding_height: float = 0.0,

    fine_steps_per_wvl: float = 20,
    coarse_steps_per_wvl: float = 15,

    wg_length: Optional[float] = None,
    run_time: Optional[float] = None,  # Duration in seconds; None keeps the automatic estimate.
) -> PartialRingResonator:
    """Add bus excitation and coupling monitors to the passive partial ring.
    The span is the total angle; clipped ring ends require reflection checks.
    """
    if not np.isfinite(span_angle_rad) or not 0 < span_angle_rad <= np.pi:
        raise ValueError("span_angle_rad must be finite and in (0, pi].")
    for name, value in [
        ("radius", radius),
        ("wg_width", wg_width),
        ("wg_height", wg_height),
        ("wavelength_um", wavelength_um),
    ]:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if not np.isfinite(gap) or gap < 0 or radius <= wg_width / 2:
        raise ValueError("Require gap >= 0 and radius > wg_width / 2.")
    obj = get_passive_simulation(
        radius=radius,
        gap=gap,
        wg_width=wg_width,
        wg_height=wg_height,
        core_medium=core_medium,
        box_medium=box_medium,
        wavelength_um=wavelength_um,
        span_angle_rad=span_angle_rad,
        num_absorber_layers=num_absorber_layers,
        box_height=box_height,
        top_cladding_medium=top_cladding_medium,
        top_cladding_height=top_cladding_height,
        fine_steps_per_wvl=fine_steps_per_wvl,
        coarse_steps_per_wvl=coarse_steps_per_wvl,
        wg_length=wg_length,
        run_time=run_time,
    )
    sim = obj.sim
    frequency = td.C_0 / wavelength_um
    target_neff = core_target_neff(core_medium, frequency)
    mode_spec = td.ModeSpec(
        num_modes=1,
        target_neff=target_neff,
    )
    y_bus = -(radius + wg_width + gap)
    z_core = wg_height / 2
    mon_w, mon_h = 3.5 * wg_width, 2.5 * wg_height
    x_edge = sim.size[0] / 2
    # Ports lie outside the clipped ring's x extent, inside the domain.
    x_ring = (radius + wg_width / 2) * np.sin(span_angle_rad / 2)
    source_x = -(x_ring + 0.75 * (x_edge - x_ring))
    input_x = -(x_ring + 0.35 * (x_edge - x_ring))
    through_x = x_ring + 0.35 * (x_edge - x_ring)
    source = td.ModeSource(
        center=(source_x, y_bus, z_core),
        size=(0, mon_w, mon_h),
        direction="+",
        mode_spec=mode_spec,
        source_time=td.GaussianPulse(
            freq0=frequency,
            fwidth=frequency / 10,
        ),
        name="Bus_Source",
    )
    mon_h = min(
        mon_h,
        0.98 * sim.size[2],
    )
    monitors = [
        td.ModeMonitor(
            center=(input_x, y_bus, z_core),
            size=(0, mon_w, mon_h),
            freqs=[frequency],
            mode_spec=mode_spec,
            name="Through_In",
        ),
        td.ModeMonitor(
            center=(through_x, y_bus, z_core),
            size=(0, mon_w, mon_h),
            freqs=[frequency],
            mode_spec=mode_spec,
            name="Through_Out",
        ),
    ]
    # Diagnostic ring slices replace the absent 0°/180° ports of a half ring.
    # At beta from the bottom, the +x tangent is (cos(beta), sin(beta), 0).
    # For a yz mode plane, angle_phi=0 tilts toward +y, pi toward -y.
    for name, beta in [
        ("Ring_Out_Left", -0.4 * span_angle_rad),
        ("Ring_Mon_270_Bot", 0.0),
        ("Ring_Out_Right", 0.4 * span_angle_rad),
    ]:
        y_ring = -radius * np.cos(beta)
        y_lower, y_upper = (
            sim.center[1] - sim.size[1] / 2,
            sim.center[1] + sim.size[1] / 2,
        )
        ring_mon_w = min(
            mon_w / np.cos(beta),
            1.96 * (y_ring - y_lower),
            1.96 * (y_upper - y_ring),
        )
        monitors.append(
            td.ModeMonitor(
                center=(radius * np.sin(beta), y_ring, z_core),
                size=(0, ring_mon_w, mon_h),
                freqs=[frequency],
                mode_spec=td.ModeSpec(
                    num_modes=1,
                    target_neff=target_neff,
                    angle_theta=abs(beta),
                    angle_phi=0.0 if beta >= 0 else float(np.pi),
                ),
                name=name,
            )
        )
    monitors.append(
        td.FieldMonitor(
            center=(sim.center[0], sim.center[1], z_core),
            size=(sim.size[0], sim.size[1], 0),
            freqs=[frequency],
            name="coupling_field_map",
        )
    )
    obj.sim = sim.copy(
        update={
            "sources": [source],
            "monitors": monitors,
        }
    )
    warnings.warn(
        "The supplied partial-ring geometry ends inside the simulation domain. "
        "Check ring-port reflections and span convergence; its endpoints are not matched absorbers.",
        UserWarning,
        stacklevel=3,  # caller of the @renamed_params wrapper
    )
    return obj


get_active_sim = get_active_simulation
get_passive_sim = get_passive_simulation


def get_optimal_plot_coords(sim: td.Simulation) -> Tuple[float, float, float]:
    """Choose visualization slices through the waveguide core center.
    Return the x, y, and z coordinates for this simulation geometry.
    """
    x_coord = 0.0
    z_coord = float(sim.sources[0].center[2]) if sim.sources else float(sim.center[2])
    y_coord = float(sim.center[1])

    logger.info(
        f"[Visualizer] Locking All-Pass slice view onto coupling domain center: "
        f"X={x_coord:.3f}, Y={y_coord:.3f}, Z={z_coord:.3f}"
    )
    return x_coord, y_coord, z_coord
