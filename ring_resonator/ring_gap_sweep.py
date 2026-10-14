"""
ring_resonator.ring_gap_sweep: gap sweeps using partial-ring geometry.

Original critical-coupling equations: Esraaj Sarkar Gupta, 2026.
The first monotonic coupling branch is selected using the legacy peak/filter
rule; intensity alone does not uniquely unwrap arbitrary coupling branches.

Exposes: ring_gap_sweep(), radius_gap_sweep().
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import tidy3d as td

from tarang import get_logger
from tarang._compat import renamed_params
from utilities.json_helpers import log_json, save_json as _save, snapshot as _snapshot
from utilities.sweep_helpers import cache_dir as _cache_dir, r_squared as _r_squared
from utilities.validation import (
    check_object as _check_object,
    positive as _positive,
    values as _values,
    resolve_span as _resolve_span,
)

logger = get_logger(__name__)
_emit = partial(
    log_json,
    logger=logger,
)


def _fit_coupling(
    gaps,
    transmissions,
    radius,
    qi,
    volume,
    wavelength,
    ng,
    *,
    theta_fit_bounds_rad=(0.05, 1.5),
    fit_after_peak=True,
    min_fit_points=2,
):
    """
    Fit exponential gap-dependent coupling on the selected valid branch.
    Return critical-gap predictions, quality factors, and fit diagnostics.
    """
    order = np.argsort(gaps)
    gaps = np.asarray(
        gaps,
        dtype=float,
    )[order]
    transmission = np.asarray(
        transmissions,
        dtype=float,
    )[order]
    theta = np.arccos(
        np.sqrt(
            np.clip(
                transmission,
                0,
                1,
            )
        )
    )
    lower, upper = theta_fit_bounds_rad
    valid = (
        np.arange(len(gaps)) >= np.argmax(theta)
        if fit_after_peak
        else np.ones(
            len(gaps),
            dtype=bool,
        )
    )
    valid &= (theta > lower) & (theta < upper)
    if np.count_nonzero(valid) < min_fit_points:
        raise ValueError(
            f"Need {min_fit_points} points within theta bounds "
            f"{theta_fit_bounds_rad}; extend the gap sweep or adjust fit arguments."
        )
    slope, intercept = np.polyfit(
        gaps[valid],
        np.log(theta[valid]),
        1,
    )
    gamma, theta_0 = float(-slope), float(np.exp(intercept))
    if not np.isfinite(gamma) or gamma <= 0 or not np.isfinite(theta_0):
        raise ValueError(
            "Coupling fit must have finite theta_0 and positive decay gamma."
        )
    length = 2 * np.pi * radius
    log_a = -np.pi * ng * length / (qi * wavelength)
    a = float(np.exp(log_a))
    cross_power = float(-np.expm1(2 * log_a))
    theta_target = float(np.arcsin(np.sqrt(cross_power)))
    critical_gap = float((intercept - np.log(theta_target)) / gamma)
    fit_min, fit_max = float(gaps[valid].min()), float(gaps[valid].max())
    q_loaded = qi / 2
    return {
        "radius_um": radius,
        "wavelength_um": wavelength,
        "n_group": ng,
        "critical_gap_um": critical_gap,
        "critical_gap_nonnegative": critical_gap >= 0,
        "critical_gap_extrapolated": not fit_min <= critical_gap <= fit_max,
        "round_trip_length_um": length,
        "round_trip_a": a,
        "round_trip_power_survival": a**2,
        "round_trip_loss_db": float(-20 * log_a / np.log(10)),
        "target_cross_power": cross_power,
        "target_through_power": a**2,
        "theta_target_rad": theta_target,
        "q_intrinsic": qi,
        "q_loaded": q_loaded,
        "q_coupling_at_critical": qi,
        "volume_um3": volume,
        "fom_raw": q_loaded**2 / volume,
        "fit_params": {
            "theta_0": theta_0,
            "gamma": gamma,
            "gamma_units": "1/um",
        },
        "fit_r_squared_log_theta": _r_squared(
            np.log(theta[valid]),
            slope * gaps[valid] + intercept,
        ),
        "fit_point_count": int(valid.sum()),
        "fit_gap_range_um": [fit_min, fit_max],
        "sorted_gaps_um": gaps.tolist(),
        "theta_raw_rad": theta.tolist(),
        "fit_mask": valid.tolist(),
    }


@renamed_params(substrate_medium="box_medium", substrate_height="box_height")
def ring_gap_sweep(
    gaps_sweep_list : list[float],  # Distinct nonnegative ring-to-bus gaps to simulate (um; at least two).

    output_dir : Path|str,  # Directory for reports, plots, and any simulation cache.

    radius_um : float,  # Ring radius for this gap sweep (um).

    q_intrinsic : float,  # Positive intrinsic ring quality factor (dimensionless).
    volume_um3 : float,  # Positive full-ring volume for the Q_loaded^2/V figure of merit (um^3).

    core_medium : td.Medium,  # Tidy3D material used for the waveguide core.
    box_medium : td.Medium,  # Tidy3D material beneath the waveguide.

    wg_width : float,  # Waveguide core width (um).
    wg_height : float,  # Waveguide core height (um).

    wavelength_um : float,  # Central vacuum wavelength (um).

    span_angle_deg: float | None = None,  # Total partial-ring span in degrees; defaults to 45 if neither angle is given.
    *,
    span_angle_rad: float | None = None,  # Alternative total span in radians (0 < span <= π); excludes span_angle_deg.

    n_group : float | None =None,  # Positive group index; None computes it with the mode solver.

    box_height : float = 4.0,  # BOX (oxide) thickness below the waveguide (um).

    top_cladding_medium : td.Medium | None = None,  # Optional upper cladding material; None leaves the air background.
    top_cladding_height : float = 0.0,              # Upper cladding thickness measured from z=0 (um); zero omits the layer.

    fine_steps_per_wvl : int = 20,      # Mesh resolution in the core region (steps per wavelength).
    coarse_steps_per_wvl : int = 15,    # Background mesh resolution (steps per wavelength).

    num_absorber_layers : int = 60,  # Number of absorbing boundary layers in the FDTD geometry.

    wg_length: float | None = None,  # Optional minimum bus/domain length along propagation (um); None sizes automatically.

    group_index_kwargs = None,  # Options passed to get_group_index, e.g. local=True or mode_index.

    force_rerun : bool = False,     # Rerun FDTD even when cached simulation data exist.
    verbose     : bool = False,     # Enable detailed simulation execution output.
    save_plots  : bool = True,      # Save analysis plots to output_dir.
    save_field_maps : bool = True,  # Save a field-map image for each simulated sweep point.

    partial_builder = None,  # Custom active partial-ring factory; None uses the standard builder.

    through_monitor : str = "Through_Out",  # Mode monitor from which through-port power is read.
    through_direction : str = "+",          # Modal propagation direction to read: forward (+) or backward (-).


    clip_through_power : bool = True,  # Clip normalized through power to [0, 1]; otherwise reject values above one.

    theta_fit_bounds_rad : tuple = (0.05, 1.5),  # Accepted coupling-angle interval for the exponential fit (radians).

    fit_after_peak : bool = True,  # Fit only points at or beyond the peak coupling angle in increasing gap order.
    min_fit_points : int = 2,      # Minimum number of accepted points needed for the coupling fit (at least two).

    field_monitor : str = "coupling_field_map",  # Field monitor used for saved field-map images.

    builder_kwargs = None,  # Extra geometry-builder options; must not duplicate explicit arguments.

    manifest_filename : str = "analytic_coupling_manifest.json",  # JSON report filename relative to output_dir.

    plot_sample_count : int = 200,  # Number of gap samples used to draw the fitted coupling curve (at least two).
    plot_dpi : int = 150,           # Resolution of displayed and saved figures (dots per inch).
):
    """Fit gap-dependent coupling and critical coupling for one ring radius.
    Return metrics and save raw fluxes; volume_um3 must be supplied explicitly.
    """
    gaps = _values("gaps_sweep_list", gaps_sweep_list, 2, True)
    
    radius = _positive("radius_um", radius_um)
    qi = _positive("q_intrinsic", q_intrinsic)
    volume = _positive("volume_um3", volume_um3)
    wavelength = _positive("wavelength_um", wavelength_um)

    span = _resolve_span(span_angle_deg, span_angle_rad)

    bounds = tuple(float(v) for v in theta_fit_bounds_rad)

    if (
        len(bounds) != 2
        or not all(np.isfinite(bounds))
        or not 0 <= bounds[0] < bounds[1] <= np.pi / 2
    ):
        raise ValueError(
            "theta_fit_bounds_rad must satisfy 0 <= lower < upper <= pi/2."
        )
    for name, value in [
        ("min_fit_points", min_fit_points),
        ("plot_sample_count", plot_sample_count),
    ]:
        if (
            isinstance(
                value,
                bool,
            )
            or not isinstance(
                value,
                (int, np.integer),
            )
            or value < 2
        ):
            raise ValueError(f"{name} must be an integer >= 2.")

    extra_builder_args = dict(builder_kwargs or {})

    _positive("wg_width", wg_width)
    _positive("wg_height", wg_height)

    if span > np.pi:
        raise ValueError("span_angle_rad must be in (0, pi].")
    if through_direction not in ("+", "-"):
        raise ValueError("through_direction must be + or -.")
    if n_group is not None:
        n_group = _positive(
            "n_group",
            n_group,
        )

    if partial_builder is None:
        from geometries.partial_ring_resonator import get_active_simulation

        partial_builder = get_active_simulation
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    path = output_dir / manifest_filename
    report = {
        "status": "running",
        "parameters": {
            "radius_um": radius,
            "gaps_um": gaps,
            "span_angle_rad": span,
            "span_angle_deg": float(np.rad2deg(span)),
            "q_intrinsic": qi,
            "volume_um3": volume,
            "n_group_supplied": n_group,
            "group_index_kwargs": dict(group_index_kwargs or {}),
            "through_monitor": through_monitor,
            "through_direction": through_direction,
            "normalization": "through modal power referenced to the normalized simulation source",
            "force_rerun": force_rerun,
            "verbose": verbose,
            "save_plots": save_plots,
            "save_field_maps": save_field_maps,
            "branch_filter_theta_rad": bounds,
            "fit_after_peak": fit_after_peak,
            "min_fit_points": min_fit_points,
            "clip_through_power": clip_through_power,
            "field_monitor": field_monitor,
            "builder_kwargs": extra_builder_args,
            "manifest_filename": manifest_filename,
            "plot_sample_count": plot_sample_count,
            "plot_dpi": plot_dpi,
        },
        "critical_metrics": {},
        "raw_sweep": {},
        "errors": [],
        "warnings": [
            "Short-arc truncation and span convergence must be checked before interpreting critical coupling."
        ],
    }
    _emit(report["parameters"])
    _save(
        path,
        report,
    )
    try:
        wrappers = []
        for gap in gaps:
            obj = partial_builder(
                radius=radius,
                gap=gap,
                wavelength_um=wavelength,
                span_angle_rad=span,
                wg_width=wg_width,
                wg_height=wg_height,
                core_medium=core_medium,
                box_medium=box_medium,
                box_height=box_height,
                top_cladding_medium=top_cladding_medium,
                top_cladding_height=top_cladding_height,
                fine_steps_per_wvl=fine_steps_per_wvl,
                coarse_steps_per_wvl=coarse_steps_per_wvl,
                num_absorber_layers=num_absorber_layers,
                wg_length=wg_length,
                **extra_builder_args,
            )
            _check_object(
                obj,
                radius=radius,
                gap=gap,
                span_angle_rad=span,
                wavelength_um=wavelength,
            )
            if through_monitor not in {m.name for m in obj.sim.monitors}:
                raise ValueError(f"Missing through monitor {through_monitor!r}.")
            wrappers.append(obj)
        if n_group is None:
            n_group = _positive(
                "n_group",
                wrappers[0].get_group_index(**dict(group_index_kwargs or {})),
            )
        report["parameters"]["n_group_used"] = n_group
        powers = []
        for obj in wrappers:
            cache = _cache_dir(
                output_dir,
                obj,
            )
            record = {
                "gap_um": obj.gap,
                "parameters": _snapshot(obj),
                "cache_dir": str(cache),
            }
            _emit(record)
            fluxes = obj.get_mode_fluxes(
                output_dir=cache,
                verbose=verbose,
                force_rerun=force_rerun,
            )
            # Use the specified propagation direction, not the method's dominant-flux shortcut.
            key = "flux_forward" if through_direction == "+" else "flux_backward"
            absolute_power = float(fluxes[through_monitor][key])
            power = absolute_power
            if not np.isfinite(power) or power < 0:
                raise ValueError("Through power must be finite and nonnegative.")
            if not clip_through_power and power > 1:
                raise ValueError(
                    "Normalized through power exceeds one with clipping disabled."
                )
            clipped = (
                float(
                    np.clip(
                        power,
                        0,
                        1,
                    )
                )
                if clip_through_power
                else power
            )
            record.update(
                through_port_power_absolute=absolute_power,
                through_port_power=clipped,
                through_port_power_unclipped=power,
                through_power_clipped=power != clipped,
                mode_fluxes=fluxes,
                missing_mode_monitors=[
                    m.name
                    for m in obj.sim.monitors
                    if hasattr(
                        m,
                        "mode_spec",
                    )
                    and m.name not in fluxes
                ],
            )
            if power != clipped:
                report["warnings"].append(
                    f"Gap {obj.gap}: T={power} clipped to [0,1] for legacy theta extraction."
                )
            powers.append(clipped)
            report["raw_sweep"][str(obj.gap)] = record
            _emit({k: v for k, v in record.items() if k != "parameters"})
            _save(
                path,
                report,
            )
            if save_field_maps:
                field_path = output_dir / "field_maps" / f"field_map_gap_{obj.gap}.png"
                try:
                    obj.plot_field(
                        output_dir=cache,
                        monitor_name=field_monitor,
                        output_path=field_path,
                        show=False,
                        save=True,
                        verbose=verbose,
                        force_rerun=False,
                        plot_dpi=plot_dpi,
                    )
                    record["field_map_path"] = str(field_path)
                except Exception as exc:
                    record["field_map_error"] = str(exc)
                    logger.warning(f"[Warning] Field plot failed: {exc}")
        metrics = _fit_coupling(
            gaps,
            powers,
            radius,
            qi,
            volume,
            wavelength,
            n_group,
            theta_fit_bounds_rad=bounds,
            fit_after_peak=fit_after_peak,
            min_fit_points=min_fit_points,
        )
        metrics.update(
            span_angle_rad=span,
            span_angle_deg=float(np.rad2deg(span)),
        )
        if metrics["critical_gap_extrapolated"]:
            report["warnings"].append(
                "Critical gap is outside the fitted gap interval."
            )
        if not metrics["critical_gap_nonnegative"]:
            report["warnings"].append(
                "Fit predicts a negative gap; no physical nonnegative critical gap was found."
            )
        report["critical_metrics"] = metrics
        _emit(metrics)
        for warning in report["warnings"]:
            logger.warning(f"[Warning] {warning}")
        _save(
            path,
            report,
        )
        if save_plots:
            try:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(dpi=plot_dpi)
                smooth = np.linspace(
                    min(gaps),
                    max(gaps),
                    plot_sample_count,
                )
                theta_fit = metrics["fit_params"]["theta_0"] * np.exp(
                    -metrics["fit_params"]["gamma"] * smooth
                )
                ax.scatter(
                    gaps,
                    powers,
                    label="Through power",
                )
                ax.plot(
                    smooth,
                    np.cos(theta_fit) ** 2,
                    label="Coupling fit",
                )
                ax.axvline(
                    metrics["critical_gap_um"],
                    ls="--",
                    label="Predicted critical gap",
                )
                ax.set(
                    xlabel="Gap (um)",
                    ylabel="Through power",
                    title=f"R = {radius:g} um, span = {np.rad2deg(span):g}°",
                )
                ax.legend()
                fig.savefig(
                    output_dir / f"analytic_coupling_fit_R_{radius}.png",
                    bbox_inches="tight",
                    dpi=plot_dpi,
                )
                plt.close(fig)
            except Exception as exc:
                report["warnings"].append(f"Fit plot failed: {exc}")
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        _save(
            path,
            report,
        )
    return report["critical_metrics"]


def radius_gap_sweep(
    radius_results,  # Radius-keyed physics results returned by either radius sweep function.
    gaps_sweep_list,  # Distinct nonnegative ring-to-bus gaps to simulate (um; at least two).
    output_dir,  # Directory for reports, plots, and any simulation cache.
    volume_um3_by_radius,  # Required full-ring volumes (um³): numeric/string radius mapping or callable.
    span_angle_deg: float | None = None,  # Total partial-ring span in degrees; defaults to 45 if neither angle is given.
    *,
    span_angle_rad: float | None = None,  # Alternative total span in radians (0 < span <= π); excludes span_angle_deg.
    manifest_filename="radius_gap_sweep_manifest.json",  # JSON report filename relative to output_dir.
    plot_dpi: int = 150,  # Resolution of displayed and saved figures (dots per inch).
    **geometry_kwargs,  # Additional ring_gap_sweep options, including materials and waveguide dimensions.
):
    """Run gap sweeps using the in-memory results returned by radius_sweep.
    Forward geometry settings and validate the wavelength for each radius.
    """
    span = _resolve_span(
        span_angle_deg,
        span_angle_rad,
    )
    output_dir = Path(output_dir)
    report = {
        "status": "running",
        "radius_results": radius_results,
        "parameters": {
            "span_angle_rad": span,
            "span_angle_deg": float(np.rad2deg(span)),
            "gaps_um": list(gaps_sweep_list),
            "geometry_kwargs": geometry_kwargs,
            "plot_dpi": plot_dpi,
        },
        "results": {},
        "errors": [],
    }
    gaps = report["parameters"]["gaps_um"]
    path = output_dir / manifest_filename
    _save(
        path,
        report,
    )
    try:
        if not radius_results:
            raise ValueError("radius_results must not be empty.")
        for physics in radius_results.values():
            radius = physics["radius_um"]
            if callable(volume_um3_by_radius):
                volume = volume_um3_by_radius(radius)
            else:
                volume = (
                    volume_um3_by_radius[radius]
                    if radius in volume_um3_by_radius
                    else volume_um3_by_radius[str(radius)]
                )
            kwargs = dict(geometry_kwargs)
            if "wavelength_um" in kwargs and not np.isclose(
                kwargs["wavelength_um"],
                physics["wavelength_um"],
            ):
                raise ValueError("Radius and gap sweep wavelengths must match.")
            kwargs.setdefault(
                "wavelength_um",
                physics["wavelength_um"],
            )
            result = ring_gap_sweep(
                gaps,
                output_dir / f"R_{radius}",
                radius,
                physics["q_intrinsic"],
                volume,
                span_angle_rad=span,
                n_group=physics["n_group"],
                plot_dpi=plot_dpi,
                **kwargs,
            )
            report["results"][str(radius)] = result
            _save(
                path,
                report,
            )
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        _save(
            path,
            report,
        )
    _emit(report["results"])
    return report["results"]
