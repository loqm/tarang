"""
tarang.geometries.grating_coupler.py

Defines the geometry and simulation parameters for the SiN grating coupler.
Optimized to terminate directly at the bottom of the SiO2 BOX layer to cut compute costs,
BUT properly supports a substrate layer when requested so your interference physics doesn't die.

Author: Divyanshu Tiwari, Esraaj Sarkar Gupta
Date: 2026

Exposes: get_active_simulation(), get_ideal_coords().
"""

from __future__ import annotations
from tarang import get_logger
from tarang._compat import renamed_params

from typing import TYPE_CHECKING
import numpy as np
import tidy3d as td

from tarang.grating_coupler.grating_coupler import GratingConfig, tilt_sign
from tarang.geometries._modes import core_target_neff

if TYPE_CHECKING:
    from tarang.grating_coupler.grating_coupler import GratingCoupler


logger = get_logger(__name__)
_NEGATIVE_THETA_NOTED = False


BOUNDARIES = ("pml", "stable_pml", "absorber")


def open_boundary(kind: str = "pml") -> td.Boundary:
    """The absorbing boundary of the open sides: 'pml' (Tidy3D's standard PML, 12 layers),
    'stable_pml' (40 layers, stable with dispersive media and fine meshes) or 'absorber'
    (adiabatic absorber, 40 layers)."""
    k = str(kind or "pml").strip().lower()
    if k == "pml":
        return td.Boundary.pml()
    if k == "stable_pml":
        return td.Boundary.stable_pml()
    if k == "absorber":
        return td.Boundary.absorber()
    raise ValueError(f"grating boundary {kind!r}: use one of {', '.join(BOUNDARIES)}")


def _note_negative_theta() -> None:
    """theta_deg was signed before v0.1.0 (always made negative); say once that the sign
    is now ignored and fiber_tilt sets the direction."""
    global _NEGATIVE_THETA_NOTED
    if not _NEGATIVE_THETA_NOTED:
        logger.info("[Simulation Builder] theta_deg is the fibre angle from vertical; its sign "
                    "is ignored. The direction is set by GratingConfig.fiber_tilt "
                    "('towards' or 'away' from the waveguide).")
        _NEGATIVE_THETA_NOTED = True


@renamed_params(substrate_medium="handle_medium", cladding_medium="top_cladding_medium")
def get_active_simulation(
    config: GratingConfig,

    core_medium: td.Medium,
    box_medium: td.Medium,

    handle_medium: td.Medium,
    top_cladding_medium: td.Medium | None = None,  # Optional: Defaults to Air if None

    is_3d: bool = False,

    fine_steps_per_wvl: int = 20,
    coarse_steps_per_wvl: int = 15,
    run_time: float = 5e-12,  # Simulation duration in seconds.
) -> GratingCoupler:
    """
    Constructs a Tidy3D simulation for the grating coupler and returns
    a wrapped GratingCoupler object.
    """
    freq0 = td.C_0 / config.wavelength_um
    half_band = float(config.bandwidth_um)
    if not 0 < half_band < 0.5 * config.wavelength_um:
        raise ValueError(f"bandwidth_um ({half_band}) must be > 0 and < half the wavelength")
    freqs = np.linspace(
        td.C_0 / (config.wavelength_um + half_band),
        td.C_0 / (config.wavelength_um - half_band),
        101,
    )
    fwidth = 0.5 * (np.max(freqs) - np.min(freqs))

    inf_eff = 1e3

    if not 0 < config.etch_depth <= config.core_height:
        raise ValueError(
            f"etch_depth ({config.etch_depth} um) must be > 0 and <= core_height "
            f"({config.core_height} um); etch_depth = core_height is a full etch."
        )
    full_etch = bool(np.isclose(config.etch_depth, config.core_height))

    # Default cladding to Air if not passed
    if top_cladding_medium is None:
        top_cladding_medium = td.Medium(permittivity=1.0)

    # -- 3D Check -- #
    if is_3d:  # 3D Simulation
        y_size = config.w_grating
        y_min = -config.w_grating / 2
        y_max = config.w_grating / 2

        y_boundary = open_boundary(getattr(config, "boundary", "pml"))

        sim_y_size = config.w_grating + 2 * config.buffer
        source_y_size = 2 * config.mfd

    else:  # 2D Simulation
        y_size = td.inf
        y_min = -inf_eff
        y_max = inf_eff

        y_boundary = td.Boundary.periodic()

        sim_y_size = 0
        source_y_size = td.inf

    # -- Fibre angle and orientation -- #
    # theta_deg is the angle from vertical; fiber_tilt sets the direction (see
    # GratingConfig). For a beam injected downwards, Tidy3D's angle_theta > 0 tilts the
    # propagation towards -x, i.e. towards the waveguide.
    if config.theta_deg < 0:
        _note_negative_theta()
    theta_deg = tilt_sign(config.fiber_tilt) * abs(config.theta_deg)

    # ---- Geometries ---- #
    # 1. The Buried Oxide (BOX) Layer
    box = td.Structure(
        geometry=td.Box.from_bounds(
            rmin=(-inf_eff, y_min, -config.box_height),
            rmax=(inf_eff, y_max, 0),
        ),
        medium=box_medium,
        name="BOX",
    )

    # 2. Core Slab
    slab_waveguide = td.Structure(
        geometry=td.Box.from_bounds(
            rmin=(-inf_eff, y_min, 0),
            rmax=(0, y_max, config.core_height),
        ),
        medium=core_medium,
        name="Slab_Waveguide",
    )

    # 3. Etched Floor
    etched_waveguide = td.Structure(
        geometry=td.Box.from_bounds(
            rmin=(0, y_min, 0),
            rmax=(inf_eff, y_max, config.core_height - config.etch_depth),
        ),
        medium=core_medium,
        name="Etched_Waveguide",
    )

    # 4. Grating Teeth: uniform (period, ff, n_teeth) or an explicit list (apodised)
    teeth = getattr(config, "teeth", None)
    spans = ([(float(a), float(b)) for a, b in teeth] if teeth else
             [(i * config.period, config.period * config.ff) for i in range(config.n_teeth)])
    # Sloped etched walls (v0.1.3): each tooth becomes a trapezoid in the (x, z) plane;
    # vertical walls (the default) keep the Box teeth of earlier versions exactly.
    from tarang.geometries import sidewall as sw
    wall = float(getattr(config, "sidewall_deg", 90.0))
    wall_ref = getattr(config, "sidewall_ref", "top")
    gratings: list = []
    for i, (x0_tooth, length) in enumerate(spans):
        tooth_center_x = x0_tooth + length / 2
        tooth_center_z = config.core_height - (config.etch_depth / 2)
        if sw.is_vertical(wall):
            geo = td.Box(
                center=(tooth_center_x, 0, tooth_center_z),
                size=(length, y_size, config.etch_depth),
            )
        else:   # the first tooth continues the waveguide: its left side is not etched
            geo = sw.tooth_geometry(x0_tooth, length, config.core_height - config.etch_depth,
                                    config.etch_depth, wall, wall_ref,
                                    y_size=None if not is_3d else y_size,
                                    vertical_left=abs(x0_tooth) < 1e-9)
        tooth = td.Structure(
            geometry=geo,
            medium=core_medium,
            name=f"Tooth_{i}",
        )
        gratings.append(tooth)

    # 5. Top Cladding (Air by default)
    cladding = td.Structure(
        geometry=td.Box.from_bounds(
            rmin=(-inf_eff, y_min, 0),
            rmax=(inf_eff, y_max, config.top_cladding_height),
        ),
        medium=top_cladding_medium,
        name="Cladding",
    )

    # A full etch leaves no core floor under the grooves (and Tidy3D warns about a
    # zero-thickness structure), so the etched floor is left out entirely.
    floor = [] if full_etch else [etched_waveguide]
    structures = [cladding, *floor, slab_waveguide, box] + gratings

    if teeth and len(spans) > 1:     # the last tooth plus its gap (the last local period)
        l_grating = spans[-1][0] + (spans[-1][0] - spans[-2][0])
    else:
        l_grating = config.n_teeth * config.period
    z_min_2d = -config.box_height - (
        config.handle_height if handle_medium is not None else config.buffer
    )

    # 6. Deep Silicon Substrate (If requested)
    if handle_medium is not None:
        substrate = td.Structure(
            geometry=td.Box.from_bounds(
                rmin=(-inf_eff, y_min, -config.box_height - config.handle_height),
                rmax=(inf_eff, y_max, -config.box_height),
            ),
            medium=handle_medium,
            name="Substrate",
        )
        structures.insert(
            0,
            substrate,
        )

    # ---- Source ---- #
    # source_x is where the beam centre reaches the top of the core; the source plane
    # sits higher, so its centre is shifted back along the beam by (height) * tan(theta).
    source_z = config.top_cladding_height + config.source_gap
    source_x_plane = config.source_x + (source_z - config.core_height) * np.tan(
        np.deg2rad(theta_deg))
    source = td.GaussianBeam(
        size=(2 * config.mfd, source_y_size, 0),
        center=[source_x_plane, 0, source_z],
        source_time=td.GaussianPulse(
            freq0=freq0,
            fwidth=fwidth,
        ),
        angle_theta=np.deg2rad(theta_deg),
        direction="-",
        waist_radius=config.mfd / 2,
        pol_angle=config.pol_angle,
        name="Fiber_Source",
    )

    # ---- Monitors ---- #
    mode_monitor = td.ModeMonitor(
        center=(-config.buffer / 2, 0, config.core_height / 2),
        size=(0, y_size, 6 * config.core_height),
        freqs=freqs,
        mode_spec=td.ModeSpec(
            num_modes=1,
            target_neff=core_target_neff(core_medium, freq0),
        ),
        name="mode monitor",
    )

    x_mon = td.FieldMonitor(
        center=(l_grating / 2, 0, config.core_height / 2),
        size=(0, sim_y_size, config.core_height + 3.0),
        freqs=[freq0],
        name="field_x_plane",
    )

    y_mon = td.FieldMonitor(
        center=(l_grating / 2, 0, config.core_height / 2),
        size=(l_grating + 2.0 * config.buffer, 0, config.core_height + 3.0),
        freqs=[freq0],
        name="field_y_plane",
    )

    z_mon = td.FieldMonitor(
        center=(l_grating / 2, 0, config.core_height / 2),
        size=(l_grating + 2.0 * config.buffer, sim_y_size, 0),
        freqs=[freq0],
        name="field_z_plane",
    )

    x_pad = 3.0 * config.wavelength_um if not is_3d else 0.0
    x_min_sim = -config.buffer - x_pad

    sim_box = td.Box.from_bounds(
        rmin=(x_min_sim, -sim_y_size / 2 if is_3d else 0, z_min_2d),
        rmax=(
            l_grating + config.buffer,
            sim_y_size / 2 if is_3d else 0,
            config.core_height + config.top_cladding_height + config.buffer,
        ),
    )

    n_core_approx = 2.0
    dl_fine = config.wavelength_um / (n_core_approx * fine_steps_per_wvl)

    mesh_override = td.MeshOverrideStructure(
        geometry=td.Box(
            center=(l_grating / 2, 0, config.core_height / 2),
            size=(
                l_grating + 2.0,
                sim_y_size,
                config.core_height + config.etch_depth + 1.0,
            ),
        ),
        dl=[dl_fine, dl_fine, dl_fine],
        enforce=True,
    )

    sim = td.Simulation(
        center=sim_box.center,
        size=sim_box.size,
        grid_spec=td.GridSpec.auto(
            min_steps_per_wvl=coarse_steps_per_wvl,
            wavelength=config.wavelength_um,
            override_structures=[mesh_override],
        ),
        structures=structures,
        sources=[source],
        monitors=[mode_monitor, x_mon, y_mon, z_mon],
        run_time=run_time,
        boundary_spec=td.BoundarySpec(
            x=open_boundary(getattr(config, "boundary", "pml")),
            y=y_boundary,
            z=open_boundary(getattr(config, "boundary", "pml")),
        ),
        medium=top_cladding_medium,
    )

    # Lazy import at runtime
    from tarang.grating_coupler.grating_coupler import GratingCoupler

    return GratingCoupler(
        sim=sim,
        config=config,
        core_medium=core_medium,
        box_medium=box_medium,
        handle_medium=handle_medium,
        top_cladding_medium=top_cladding_medium,
        is_3d=is_3d,
        fine_steps_per_wvl=fine_steps_per_wvl,
        coarse_steps_per_wvl=coarse_steps_per_wvl,
    )

get_active_sim = get_active_simulation

def get_ideal_coords(sim: td.Simulation) -> tuple[float, float, float]:
    """Choose slice coordinates through the grating and input region.
    Return the x, y, and z positions used for geometry visualization.
    """
    x_coord = float(sim.center[0])
    y_coord = 0.0
    wg_struct = next(
        (s for s in sim.structures if s.name == "Slab_Waveguide"),
        None,
    )
    z_coord = float(wg_struct.geometry.center[2]) if wg_struct else float(sim.center[2])
    return x_coord, y_coord, z_coord
