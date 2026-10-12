"""
geometries.resonator_arc.py
Authors: Esraaj Sarkar Gupta, Divyanshu Tiwari
Date: 2026

Role: GEOMETRY BUILDER. Builds the Tidy3D Simulation for a waveguide bend (arc)
and returns it wrapped in the analysis class from ``ringResonator.resonator_arc``,
which runs the simulation and extracts bend loss. The two modules share a file
name on purpose: geometries/* builds, the device packages analyse.

Refactored to follow the mode-monitor bend-loss measurement pattern used in
Tidy3D's own "Euler Waveguide Bend" example notebook: a ModeSource launches
the fundamental mode into a straight lead, and a ModeMonitor placed close to
the bend reads the transmitted power directly from Tidy3D's built-in
source-normalized mode amplitudes -- no separate calibration run needed.

Reference: https://www.flexcompute.com/tidy3d/examples/notebooks/EulerWaveguideBend/
Unlike that notebook (a fixed 90 deg bend), this module supports an arbitrary
bend_angle_deg in (0, 180]. That means the output lead is a straight waveguide
rotated by an arbitrary angle, which is generally NOT aligned with the x or y
axis. Tidy3D monitor planes must stay axis-aligned, so instead of rotating the
plane we orient the *mode* via ModeSpec.angle_theta and angle_phi, choosing
whichever axis (x or y) is closer to the true propagation direction. This
keeps angle_theta <= 45 deg for any bend angle, and reduces to the reference
notebook's simple case (angle_theta = 0) exactly at bend_angle_deg = 90.

Exposes: get_passive_simulation(), get_active_simulation(), compute_arc_loss(),
get_optimal_plot_coords().
"""

from __future__ import annotations
from tarang import get_logger

from typing import Optional, Tuple
import warnings
import numpy as np
import tidy3d as td
from ringResonator.resonator_arc import resonatorArc
from media import get_medium

logger = get_logger(__name__)


def _transform_pt(
    x: float,
    y: float,
    theta_rad: float,
    radius: float,
) -> Tuple[float, float]:
    """Map output-lead coordinates into the global clockwise arc geometry.
    Measure local x from the bend exit and local y across the lead width.
    """
    xr = x * np.cos(-theta_rad) - y * np.sin(-theta_rad)
    yr = x * np.sin(-theta_rad) + y * np.cos(-theta_rad)
    return xr + radius * np.sin(theta_rad), yr + radius * np.cos(theta_rad)


def get_passive_simulation(
    wavelength0: float,

    radius: float,

    wg_width: float,
    wg_height: float,

    core_medium: td.Medium,
    substrate_medium: td.Medium,
    cladding_medium: td.Medium = get_medium("air"),

    bend_angle_deg: float = 90,

    fine_steps_per_wvl: int = 20,
    coarse_steps_per_wvl: int = 15,

    num_absorber_layers: int = 60,

    wg_in_length: float = 4.0,
    wg_out_length: float = 4.0,
    run_time: Optional[float] = None,  # Duration in seconds; None keeps the automatic estimate.
) -> resonatorArc:
    """Constructs a passive arc resonator simulation wrapped in a resonatorArc object.
    Include the bent core and straight leads without excitation or monitors.
    """
    if not (0 < bend_angle_deg <= 180):
        raise ValueError(
            f"[Error] Unsupported bend angle: {bend_angle_deg}°. "
            f"Angle must be in the range (0, 180]."
        )
    if not all(
        np.isfinite(v) and v > 0
        for v in (wavelength0, radius, wg_width, wg_height, wg_in_length, wg_out_length)
    ):
        raise ValueError(
            "Wavelength, radius, waveguide dimensions and lead lengths must be positive."
        )
    if radius <= wg_width / 2:
        raise ValueError("radius must exceed wg_width / 2.")
    theta_rad = np.deg2rad(bend_angle_deg)
    w = wg_width
    # -- Shared Ring & Substrate Geometry -- #
    geom_outer = td.Cylinder(
        center=(0, 0, wg_height / 2),
        radius=radius + w / 2,
        length=wg_height,
        axis=2,
    )
    geom_inner = td.Cylinder(
        center=(0, 0, wg_height / 2),
        radius=radius - w / 2,
        length=wg_height + 0.200,
        axis=2,
    )
    annulus_geom = geom_outer - geom_inner
    substrate = td.Structure(
        geometry=td.Box(
            center=(0, 0, -1.000),
            size=(td.inf, td.inf, 2.000),
        ),
        medium=substrate_medium,
        name="Substrate",
    )
    # -- Dynamic Arc Generation via Boolean Subtraction -- #
    cut_left = td.Box(
        center=(-2 * radius, 0, wg_height / 2),
        size=(4 * radius, 4 * radius, wg_height + 2.0),
    )
    core_geom = annulus_geom - cut_left
    if bend_angle_deg < 180:
        cut_pts = [
            (0, 0),
            (2 * radius * np.sin(theta_rad), 2 * radius * np.cos(theta_rad)),
        ]
        if bend_angle_deg <= 90:
            cut_pts.append((2 * radius, -2 * radius))
        cut_pts.append((0, -2 * radius))
        cut_wedge = td.PolySlab(
            vertices=cut_pts,
            axis=2,
            slab_bounds=(-1.0, wg_height + 1.0),
        )
        core_geom = core_geom - cut_wedge
    ring_arc = td.Structure(
        geometry=core_geom,
        medium=core_medium,
        name="Ring Arc",
    )
    # -- Waveguide Leads -- #
    wg_in = td.Structure(
        geometry=td.Box(
            center=(-wg_in_length / 2, radius, wg_height / 2),
            size=(wg_in_length, w, wg_height),
        ),
        medium=core_medium,
        name="Waveguide In",
    )
    wg_out_verts = [
        _transform_pt(
            x,
            y,
            theta_rad,
            radius,
        )
        for x, y in [
            (0, w / 2),
            (0, -w / 2),
            (wg_out_length, -w / 2),
            (wg_out_length, w / 2),
        ]
    ]
    wg_out = td.Structure(
        geometry=td.PolySlab(
            vertices=wg_out_verts,
            axis=2,
            slab_bounds=(0, wg_height),
        ),
        medium=core_medium,
        name="Waveguide Out",
    )
    # -- Dynamic Domain Sizing -- #
    xs = [0, -wg_in_length, radius * np.sin(theta_rad)]
    ys = [radius + w / 2, radius - w / 2, radius * np.cos(theta_rad)]
    if bend_angle_deg >= 90:
        xs.append(radius + w / 2)
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
    # -- Continue both leads through the absorbing boundaries. Their nominal
    # lengths still determine the domain and port positions; no exposed end
    # facet remains for oblique-incidence reflection. -- #
    extension = 2.0 * np.hypot(
        domain_size_x,
        domain_size_y,
    )
    extended_in_length = wg_in_length + extension
    wg_in = wg_in.updated_copy(
        geometry=td.Box(
            center=(-extended_in_length / 2, radius, wg_height / 2),
            size=(extended_in_length, w, wg_height),
        )
    )
    extended_out_length = wg_out_length + extension
    extended_verts = [
        _transform_pt(
            x,
            y,
            theta_rad,
            radius,
        )
        for x, y in [
            (0, w / 2),
            (0, -w / 2),
            (extended_out_length, -w / 2),
            (extended_out_length, w / 2),
        ]
    ]
    wg_out = wg_out.updated_copy(
        geometry=td.PolySlab(
            vertices=extended_verts,
            axis=2,
            slab_bounds=(0, wg_height),
        )
    )
    structures = [ring_arc, substrate, wg_in, wg_out]
    # -- Grid Overrides & Solver Setup -- #
    try:
        freq0 = td.C_0 / wavelength0
        n_core_approx = float(np.real(core_medium.n_complex(freq0)))
    except Exception:
        n_core_approx = 2.0
    dl_fine = wavelength0 / (n_core_approx * fine_steps_per_wvl)
    mesh_override_ring = td.MeshOverrideStructure(
        geometry=ring_arc.geometry,
        dl=[dl_fine] * 3,
        enforce=True,
    )
    mesh_override_in = td.MeshOverrideStructure(
        geometry=wg_in.geometry,
        dl=[dl_fine] * 3,
        enforce=True,
    )
    mesh_override_out = td.MeshOverrideStructure(
        geometry=wg_out.geometry,
        dl=[dl_fine] * 3,
        enforce=True,
    )
    arc_length_um = (
        (bend_angle_deg / 180.0 * np.pi * radius) + wg_in_length + wg_out_length
    )
    estimated_travel_time = (arc_length_um / (td.C_0 / n_core_approx)) * 2.5
    boundary_spec = td.BoundarySpec(
        x=td.Boundary.absorber(num_layers=num_absorber_layers),
        y=td.Boundary.absorber(num_layers=num_absorber_layers),
        z=td.Boundary.pml(),
    )
    # Fit the full modal aperture with 0.5 um clearance inside the z boundaries.
    port_height = max(
        2.5,
        wg_height + 1.0,
    )
    z_min = min(
        -2.5,
        wg_height / 2.0 - port_height / 2.0 - 0.5,
    )
    z_max = max(
        1.5,
        wg_height / 2.0 + port_height / 2.0 + 0.5,
    )
    if run_time is None:
        run_time = max(
            3e-12,
            estimated_travel_time,
        )

    sim_passive = td.Simulation(
        center=(center_x, center_y, (z_min + z_max) / 2.0),
        size=(domain_size_x, domain_size_y, z_max - z_min),
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=coarse_steps_per_wvl,
            wavelength=wavelength0,
            override_structures=[
                mesh_override_ring,
                mesh_override_in,
                mesh_override_out,
            ],
        ),
        structures=structures,
        run_time=run_time,
        boundary_spec=boundary_spec,
        medium=cladding_medium,
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
        wavelength0=wavelength0,
    )


def _output_mode_orientation(bend_angle_deg: float) -> Tuple[int, float, float, str]:
    """Return the port normal axis, modal angles, and extraction direction.
    Orient the positive-normal mode branch along the clockwise lead tangent.
    """
    if not np.isfinite(bend_angle_deg) or not 0 < bend_angle_deg <= 180:
        raise ValueError("bend_angle_deg must be finite and in (0, 180].")
    theta = np.deg2rad(bend_angle_deg)
    tangent = np.array([np.cos(theta), -np.sin(theta), 0.0])
    normal_axis = int(np.argmax(np.abs(tangent)))
    direction_sign = 1.0 if tangent[normal_axis] > 0 else -1.0
    forward_axis = direction_sign * tangent
    plane_axes = [axis for axis in range(3) if axis != normal_axis]
    angle_theta = float(
        np.arccos(
            np.clip(
                forward_axis[normal_axis],
                -1.0,
                1.0,
            )
        )
    )
    angle_phi = (
        float(
            np.arctan2(
                forward_axis[plane_axes[1]],
                forward_axis[plane_axes[0]],
            )
        )
        if np.linalg.norm(forward_axis[plane_axes]) > 1e-12
        else 0.0
    )
    return normal_axis, angle_theta, angle_phi, "+" if direction_sign > 0 else "-"


def _validate_port_positions(
    wg_in_length,
    wg_out_length,
    wg_width,
    bend_angle_deg,
    input_distance,
    output_distance,
    source_offset,
    min_source_spacing,
    junction_clearance,
):
    """Check that arc source and monitor planes lie in straight leads.
    Check nominal lead lengths, source spacing, and bend-junction clearances.
    """
    values = (
        wg_in_length,
        wg_out_length,
        wg_width,
        input_distance,
        output_distance,
        source_offset,
        min_source_spacing,
        junction_clearance,
    )
    if not all(np.isfinite(v) and v > 0 for v in values):
        raise ValueError(
            "Lead dimensions and port distances/clearances must be finite and positive."
        )
    if input_distance >= wg_in_length or output_distance >= wg_out_length:
        raise ValueError(
            "Each monitor distance must be smaller than its nominal lead length."
        )
    source_x = -wg_in_length + source_offset
    source_spacing = -input_distance - source_x
    if source_x >= 0 or source_spacing < min_source_spacing:
        raise ValueError(
            "Source must precede the input monitor. Increase wg_in_length or reduce "
            "input_monitor_distance_um/source_boundary_offset_um."
        )
    if input_distance < junction_clearance:
        raise ValueError("Input monitor is too close to the bend entrance.")
    normal, _, _, _ = _output_mode_orientation(bend_angle_deg)
    theta = np.deg2rad(bend_angle_deg)
    tangent = np.array([np.cos(theta), -np.sin(theta), 0.0])
    aperture = np.array(
        [0.0, 4.0 * wg_width, 0.0] if normal == 0 else [4.0 * wg_width, 0.0, 0.0]
    )
    half_longitudinal_span = float(
        np.dot(
            np.abs(tangent),
            aperture,
        )
        / 2
    )
    if output_distance - half_longitudinal_span < junction_clearance:
        raise ValueError(
            "Output aperture reaches too close to the bend exit. Increase "
            "output_monitor_distance_um and, if necessary, wg_out_length."
        )
    return {
        "source_x_um": source_x,
        "source_monitor_spacing_um": source_spacing,
        "output_aperture_clearance_um": output_distance - half_longitudinal_span,
    }


def get_active_simulation(
    wavelength0: float,

    radius: float,

    wg_width: float,
    wg_height: float,

    core_medium: td.Medium,
    substrate_medium: td.Medium,
    cladding_medium: td.Medium = get_medium("air"),

    bend_angle_deg: float = 90,

    fine_steps_per_wvl: int = 20,
    coarse_steps_per_wvl: int = 15,

    num_absorber_layers: int = 60,

    wg_in_length: float = 4.0,
    wg_out_length: float = 4.0,

    port_buffer_um: Optional[float] = None,
    *,
    input_monitor_distance_um: float = 2.0,
    output_monitor_distance_um: float = 3.0,
    source_boundary_offset_um: float = 1.0,
    min_source_monitor_spacing_um: float = 0.5,
    junction_clearance_um: float = 0.5,
    run_time: Optional[float] = None,  # Duration in seconds; None keeps the automatic estimate.
) -> resonatorArc:
    """Build an active arc with independently positioned sources and monitors.
    Validate lead clearances; see docs/API_NOTES.md for placement conventions.
    """
    freq0 = td.C_0 / wavelength0
    df = freq0 * 0.02
    if port_buffer_um is not None:
        if (
            input_monitor_distance_um,
            output_monitor_distance_um,
            source_boundary_offset_um,
        ) != (2.0, 3.0, 1.0):
            raise ValueError(
                "Use port_buffer_um or the independent positioning arguments, not both."
            )
        input_monitor_distance_um = output_monitor_distance_um = (
            source_boundary_offset_um
        ) = port_buffer_um
        warnings.warn(
            "port_buffer_um is deprecated; use independent monitor/source distances.",
            DeprecationWarning,
            stacklevel=2,
        )

    positions = _validate_port_positions(
        wg_in_length,
        wg_out_length,
        wg_width,
        bend_angle_deg,
        input_monitor_distance_um,
        output_monitor_distance_um,
        source_boundary_offset_um,
        min_source_monitor_spacing_um,
        junction_clearance_um,
    )
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
        num_absorber_layers=num_absorber_layers,
        wg_in_length=wg_in_length,
        wg_out_length=wg_out_length,
        run_time=run_time,
    )
    sim_passive = resonator_obj.sim
    theta_rad = np.deg2rad(bend_angle_deg)
    mon_w = wg_width * 4.0
    mon_h = max(
        2.5,
        wg_height + 1.0,
    )
    mode_spec_straight = td.ModeSpec(
        num_modes=1,
        target_neff=n_eff,
    )
    # -- Mode source: launched into the straight input lead -- #
    src_center = (positions["source_x_um"], radius, wg_height / 2.0)
    src_size = (0, mon_w, mon_h)
    source = td.ModeSource(
        center=src_center,
        size=src_size,
        direction="+",
        mode_spec=mode_spec_straight,
        source_time=td.GaussianPulse(
            freq0=freq0,
            fwidth=freq0 / 10,
        ),
        name="Mode Source",
    )
    # -- Input-side monitor: axis-aligned (wg_in runs along x), placed close
    # to the bend junction. -- #
    monitor_input = td.ModeMonitor(
        center=(-input_monitor_distance_um, radius, wg_height / 2.0),
        size=(0, mon_w, mon_h),
        freqs=[freq0 - df, freq0, freq0 + df],
        mode_spec=mode_spec_straight,
        name="Mode Monitor In",
    )
    # -- Output-side monitor: pick whichever axis (x or y) is closer to the
    # true propagation direction of the rotated output lead, so the required
    # ModeSpec.angle_theta correction never exceeds 45 deg. -- #
    normal_axis, angle_theta, angle_phi, output_direction = _output_mode_orientation(
        bend_angle_deg
    )
    out_x, out_y = _transform_pt(
        output_monitor_distance_um,
        0.0,
        theta_rad,
        radius,
    )
    mode_spec_out = td.ModeSpec(
        num_modes=1,
        target_neff=n_eff,
        angle_theta=angle_theta,
        angle_phi=angle_phi,
    )
    mon_out_size = (0, mon_w, mon_h) if normal_axis == 0 else (mon_w, 0, mon_h)
    monitor = td.ModeMonitor(
        center=(out_x, out_y, wg_height / 2.0),
        size=mon_out_size,
        freqs=[freq0 - df, freq0, freq0 + df],
        mode_spec=mode_spec_out,
        name="Mode Monitor",
    )
    monitor_field = td.FieldMonitor(
        center=(sim_passive.center[0], sim_passive.center[1], wg_height / 2.0),
        size=(sim_passive.size[0], sim_passive.size[1], 0),
        freqs=[freq0],
        name="field_map",
    )
    # Reject clipped modal apertures instead of silently changing normalization.
    lower, upper = sim_passive.bounds
    for port in (source, monitor_input, monitor):
        pmin, pmax = port.geometry.bounds
        for i, axis in enumerate("xyz"):
            if pmin[i] <= lower[i] or pmax[i] >= upper[i]:
                raise ValueError(
                    f"{port.name} reaches the {axis} simulation boundary: "
                    f"port [{pmin[i]:.6g}, {pmax[i]:.6g}] um, "
                    f"domain [{lower[i]:.6g}, {upper[i]:.6g}] um. "
                    "Enlarge the domain on this axis."
                )
    sim_active = sim_passive.updated_copy(
        sources=[source],
        monitors=[monitor_input, monitor, monitor_field],
    )
    resonator_obj.sim = sim_active
    # Input uses "+". Output uses the branch aligned with the rotated lead.
    resonator_obj.mode_direction = output_direction
    return resonator_obj


def compute_arc_loss(
    sim_data: td.SimulationData,
    resonator_obj: resonatorArc,
    mode_index: int = 0,
    straight_loss_db_per_cm: float = 0.0,
) -> dict:
    """Compute direct and input-normalized guided-mode insertion loss.
    Length-normalized loss includes junction effects; see docs/API_NOTES.md.
    """
    if not np.isfinite(straight_loss_db_per_cm) or straight_loss_db_per_cm < 0:
        raise ValueError("straight_loss_db_per_cm must be finite and nonnegative.")
    radius = float(resonator_obj.radius)
    theta = np.deg2rad(float(resonator_obj.bend_angle_deg))
    arc_length_cm = radius * theta * 1e-4
    if not np.isfinite(arc_length_cm) or arc_length_cm <= 0:
        raise ValueError("Arc length must be finite and positive.")
    input_data = sim_data["Mode Monitor In"]
    output_data = sim_data["Mode Monitor"]
    tangent = np.array([np.cos(theta), -np.sin(theta), 0.0])
    normal_axis = output_data.monitor.size.index(0.0)
    output_direction = "+" if tangent[normal_axis] > 0 else "-"
    a_in = input_data.amps.sel(
        direction="+",
        mode_index=mode_index,
    )
    a_out = output_data.amps.sel(
        direction=output_direction,
        mode_index=mode_index,
    )
    a_back = input_data.amps.sel(
        direction="-",
        mode_index=mode_index,
    )
    if not np.array_equal(
        a_in.f.values,
        a_out.f.values,
    ):
        raise ValueError("Input and output monitor frequency grids must match.")
    p_in, p_out = abs(a_in) ** 2, abs(a_out) ** 2
    if (
        not np.all(np.isfinite(p_in.values))
        or np.any(p_in.values <= 0)
        or not np.all(np.isfinite(p_out.values))
    ):
        raise ValueError("Modal powers must be finite, with nonzero input power.")
    # -- Direct method (matches the Euler Waveguide Bend example exactly):
    # trust the source's own normalization, no input monitor required. -- #
    with np.errstate(divide="ignore"):
        direct_insertion_loss_db = -10.0 * np.log10(p_out)
    # -- Ratio method: normalize against the measured input power instead. -- #
    transmission = p_out / p_in
    if np.any(transmission.values > 1.0 + 1e-3):
        warnings.warn(
            "Transmission exceeds unity. Check mode selection, aperture, mesh, "
            "run time and boundary reflections before interpreting loss.",
            UserWarning,
            stacklevel=2,
        )
    with np.errstate(divide="ignore"):
        insertion_loss_db = -10.0 * np.log10(transmission)
    arc_end = np.array(
        [radius * np.sin(theta), radius * np.cos(theta), output_data.monitor.center[2]]
    )
    input_lead_um = -input_data.monitor.center[0]
    output_lead_um = float(
        np.dot(
            np.array(output_data.monitor.center) - arc_end,
            tangent,
        )
    )
    if input_lead_um <= 0 or output_lead_um <= 0:
        raise ValueError("Expected input/output monitors on the straight leads.")
    straight_length_cm = (input_lead_um + output_lead_um) * 1e-4
    corrected_loss_db = insertion_loss_db - straight_loss_db_per_cm * straight_length_cm
    return {
        "transmission": transmission,
        "input_reflection": abs(a_back) ** 2 / p_in,
        "direct_insertion_loss_db": direct_insertion_loss_db,
        "insertion_loss_db": insertion_loss_db,
        "lead_corrected_insertion_loss_db": corrected_loss_db,
        "equivalent_arc_loss_db_per_cm": corrected_loss_db / arc_length_cm,
        "arc_length_cm": arc_length_cm,
        "straight_length_cm": straight_length_cm,
        "output_direction": output_direction,
    }


get_active_sim = get_active_simulation
get_passive_sim = get_passive_simulation


def get_optimal_plot_coords(sim: td.Simulation) -> Tuple[float, float, float]:
    """Slicing helper focused on the critical waveguide core plane.
    Return the x, y, and z coordinates for field and geometry slices.
    """
    x_coord = 0.0
    y_coord = 0.0
    if sim.sources:
        z_coord = float(sim.sources[0].center[2])
    else:
        core_struct = next(
            (s for s in sim.structures if s.name in ["Ring Arc", "Waveguide In"]),
            None,
        )
        if core_struct:
            z_coord = float(core_struct.geometry.center[2])
        else:
            z_coord = float(sim.center[2])
    logger.info(
        f"[Visualizer] Locking Bend slice view onto core center: X={x_coord}, Y={y_coord}, Z={z_coord}"
    )
    return x_coord, y_coord, z_coord
