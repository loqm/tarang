"""
ringResonator.ring_radius_sweep.py

FDTD cutback and supplied-loss analytical sweeps; JSON is an output, not a cache.

Author: Esraaj Sarkar Gupta, Divyanshu Tiwari.
The fitted slope assumes angle-independent junction/lead losses. Inspect R²
and residuals before interpreting that slope as distributed bend attenuation.

Exposes: radius_sweep(), radius_sweep_analytical(), plot_radius_summary().
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import tidy3d as td

from tarang import get_logger
from utilities.json_helpers import log_json, save_json as _save, snapshot as _snapshot
from utilities.sweep_helpers import cache_dir as _cache_dir, r_squared as _r_squared
from utilities.validation import (
    integer as _integer,
    check_object as _check_object,
    positive as _positive,
    values as _values,
    radius_value as _radius_value,
)

logger = get_logger(__name__)
_emit = partial(
    log_json,
    logger=logger,
)


def radius_sweep(
    radii_sweep_list : list[float],  # Distinct ring radii to evaluate (um).

    output_dir : Path|str,         # Directory for reports, plots, and any simulation cache.

    core_medium : td.Medium,        # Tidy3D medium used for the waveguide core.
    substrate_medium : td.Medium,   # Tidy3D medium for the substrate (BOX layer).

    wg_width : float,       # Waveguide core width (um).
    wg_height : float,      # Waveguide core height (um).

    wavelength_um : float,  # Central vacuum wavelength (um).

    scattering_loss_db_cm : float,  # Required nonnegative distributed scattering loss (dB/cm).

    cladding_medium : td.Medium = None,  # Upper cladding material; None uses air.

    fine_steps_per_wvl : int = 20,      # Mesh resolution in the core region (steps per wavelength).
    coarse_steps_per_wvl : int = 15,    # Background mesh resolution (steps per wavelength).

    # Distinct arc angles for the cutback fit (degrees, 0 < angle <= 180).
    bend_angles_sampled : tuple = (30.0, 45.0, 60.0, 75.0, 90.0),

    num_absorber_layers : int = 60,  # Number of absorbing boundary layers in the FDTD geometry.

    wg_in_length : float = 4.0,   # Straight input lead length before the bend (um).
    wg_out_length : float = 4.0,  # Straight output lead length after the bend (um).
    *,
    group_index_kwargs=None,  # Options passed to get_group_index, e.g. local=True or mode_index.

    force_rerun : bool = False,     # Rerun FDTD even when cached simulation data exist.
    verbose : bool = False,         # Enable detailed simulation execution output.

    arc_builder = None,  # Custom active arc factory; None uses the standard arc builder.
    clamp_negative_bend_loss : bool = True,  # Replace a negative fitted bend loss with zero before computing Q.

    db_to_neper_divisor : float = 4.343,  # Divide power loss in dB by this value to get exponential attenuation.

    input_monitor = "Mode Monitor In",  # Input mode monitor used to normalize arc transmission.
    output_monitor = "Mode Monitor",  # Output mode monitor used to measure arc transmission.
    field_monitor = "field_map",  # Field monitor used for saved field-map images.

    builder_kwargs = None,  # Extra geometry-builder options; must not duplicate explicit arguments.

    manifest_filename : str = "intrinsic_physics_manifest.json",  # JSON report filename relative to output_dir.

    plot_dpi : float = 150,  # Resolution of saved figures (dots per inch).

    # Volume (um^3) by numeric/string radius or callable(radius_um); None uses core volume.
    volume_um3_by_radius : float = None,
    volume_label = None,  # Description of the volume definition used in reports and plots.

    save_plots : bool = True,        # Save analysis plots to output_dir.
    save_field_maps : bool = False,  # Save a field-map image for each simulated sweep point.
    show_plots : bool = False,       # Display the radius summary figure interactively.

    annotate_summary : bool = True,  # Label radius summary datapoints with their values.
):
    """
    Fit bend attenuation and intrinsic quality factors across ring radii.
    Run or reuse multipoint arc simulations and return the saved physics report.
    """

    fine_steps_per_wvl = _positive(
        "fine_steps_per_wvl",
        fine_steps_per_wvl,
    )
    coarse_steps_per_wvl = _positive(
        "coarse_steps_per_wvl",
        coarse_steps_per_wvl,
    )
    num_absorber_layers = _integer(
        "num_absorber_layers",
        num_absorber_layers,
    )
    radii = _values(
        "radii_sweep_list",
        radii_sweep_list,
    )

    angles = _values(
        "bend_angles_sampled",
        bend_angles_sampled,
        2,
    )

    if any(a > 180 for a in angles):
        raise ValueError("Bend angles must be in (0, 180] degrees.")
    scattering = _positive(
        "scattering_loss_db_cm",
        scattering_loss_db_cm,
        True,
    )

    conversion = _positive(
        "db_to_neper_divisor",
        db_to_neper_divisor,
    )

    _positive(
        "plot_dpi",
        plot_dpi,
    )

    extra_builder_args = dict(builder_kwargs or {})
    if input_monitor == output_monitor or output_monitor == "Mode Monitor In":
        raise ValueError(
            "Input and output monitors must be distinct; Mode Monitor In is reserved for the input."
        )

    for name, value in [
        ("wg_width", wg_width),
        ("wg_height", wg_height),
        ("wavelength_um", wavelength_um),
        ("wg_in_length", wg_in_length),
        ("wg_out_length", wg_out_length),
    ]:
        _positive(
            name,
            value,
        )

    if cladding_medium is None:
        from media import get_medium
        cladding_medium = get_medium("air")

    if arc_builder is None:
        from geometries.resonator_arc import get_active_simulation
        arc_builder = get_active_simulation

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = output_dir / manifest_filename
    group_options = dict(group_index_kwargs or {})
    volumes = {
        str(r): _resolve_volume(
            r,
            wg_width,
            wg_height,
            volume_um3_by_radius,
        )
        for r in radii
    }
    resolved_volume_label = volume_label or (
        "supplied volume"
        if volume_um3_by_radius is not None
        else "geometric core volume"
    )
    report = {
        "status": "running",
        "parameters": {
            "radii_um": radii,
            "bend_angles_deg": angles,
            "scattering_loss_db_cm": scattering,
            "group_index_kwargs": group_options,
            "force_rerun": force_rerun,
            "verbose": verbose,
            "save_plots": save_plots,
            "fit_model": "loss_db = slope_db_cm * arc_length_cm + intercept_db",
            "db_to_neper_divisor": conversion,
            "clamp_negative_bend_loss": clamp_negative_bend_loss,
            "input_monitor": input_monitor,
            "output_monitor": output_monitor,
            "field_monitor": field_monitor,
            "save_field_maps": save_field_maps,
            "builder_kwargs": extra_builder_args,
            "manifest_filename": manifest_filename,
            "plot_dpi": plot_dpi,
            "volume_um3_by_radius": volumes,
            "volume_label": resolved_volume_label,
            "fom_model": "(q_intrinsic / 2)**2 / full_ring_volume_um3",
            "show_plots": show_plots,
            "annotate_summary": annotate_summary,
        },
        "results": {},
        "raw_sweep": {},
        "errors": [],
    }
    _emit(report["parameters"])
    _save(
        path,
        report,
    )
    try:
        for radius in radii:
            samples, wrappers = [], []
            for angle in angles:
                obj = arc_builder(
                    radius=radius,
                    bend_angle_deg=angle,
                    wavelength0=wavelength_um,
                    wg_width=wg_width,
                    wg_height=wg_height,
                    core_medium=core_medium,
                    substrate_medium=substrate_medium,
                    cladding_medium=cladding_medium,
                    fine_steps_per_wvl=fine_steps_per_wvl,
                    coarse_steps_per_wvl=coarse_steps_per_wvl,
                    num_absorber_layers=num_absorber_layers,
                    wg_in_length=wg_in_length,
                    wg_out_length=wg_out_length,
                    **extra_builder_args,
                )
                _check_object(
                    obj,
                    radius=radius,
                    bend_angle_deg=angle,
                    wavelength0=wavelength_um,
                )
                names = {m.name for m in obj.sim.monitors}
                if not {input_monitor, output_monitor} <= names:
                    raise ValueError(
                        f"Arc builder must provide {input_monitor!r} and {output_monitor!r}."
                    )
                if input_monitor != "Mode Monitor In":
                    if "Mode Monitor In" in names:
                        raise ValueError(
                            "Cannot alias input: a different Mode Monitor In already exists."
                        )
                    monitors = [
                        (
                            m.copy(
                                update={
                                    "name": "Mode Monitor In",
                                }
                            )
                            if m.name == input_monitor
                            else m
                        )
                        for m in obj.sim.monitors
                    ]
                    obj.sim = obj.sim.copy(
                        update={
                            "monitors": monitors,
                        }
                    )
                wrappers.append(obj)
            report["raw_sweep"][str(radius)] = samples
            for obj in wrappers:
                cache = _cache_dir(
                    output_dir,
                    obj,
                )
                sample = {
                    "parameters": _snapshot(obj),
                    "cache_dir": str(cache),
                }
                _emit(sample)
                # One forced execution per point. All later methods reuse its HDF5.
                loss = float(
                    obj.get_loss(
                        output_dir=cache,
                        per_cm=False,
                        monitor_name=output_monitor,
                        force_rerun=force_rerun,
                        verbose=verbose,
                    )
                )
                if not np.isfinite(loss):
                    raise ValueError(
                        "Nonfinite arc loss; cannot include this point in regression."
                    )
                length_cm = obj.radius * np.deg2rad(obj.bend_angle_deg) * 1e-4
                efficiency = float(10 ** (-loss / 10))
                sample.update(
                    bend_angle_deg=obj.bend_angle_deg,
                    arc_length_cm=length_cm,
                    arc_length_um=length_cm * 1e4,
                    loss_db=loss,
                    efficiency=efficiency,
                    efficiency_percent=100 * efficiency,
                    equivalent_arc_loss_db_cm=loss / length_cm,
                )
                if save_field_maps:
                    try:
                        field_path = (
                            output_dir
                            / "field_maps"
                            / f"R_{radius}um"
                            / f"angle_{obj.bend_angle_deg}deg.png"
                        )
                        field_path.parent.mkdir(
                            parents=True,
                            exist_ok=True,
                        )
                        obj.plot_field(
                            output_dir=cache,
                            monitor_name=field_monitor,
                            output_path=field_path,
                            show=False,
                            save=True,
                            verbose=verbose,
                            force_rerun=False,
                        )
                        sample["field_map_path"] = str(field_path)
                    except Exception as exc:
                        sample["field_map_error"] = str(exc)
                        logger.warning(f"[Warning] Field plot failed: {exc}")
                samples.append(sample)
                _emit({k: v for k, v in sample.items() if k != "parameters"})
                _save(
                    path,
                    report,
                )
            x = np.array([s["arc_length_cm"] for s in samples])
            y = np.array([s["loss_db"] for s in samples])
            slope, intercept = np.polyfit(
                x,
                y,
                1,
            )
            predicted = slope * x + intercept
            residual = y - predicted
            n_group = _positive(
                "n_group",
                wrappers[0].get_group_index(**group_options),
            )
            bend_loss = (
                max(
                    float(slope),
                    0.0,
                )
                if clamp_negative_bend_loss
                else float(slope)
            )
            total_loss = bend_loss + scattering
            alpha_m = total_loss * 100 / conversion
            qi = (
                None
                if alpha_m <= 0
                else 2 * np.pi * n_group / (wavelength_um * 1e-6 * alpha_m)
            )
            metrics = {
                "radius_um": radius,
                "wavelength_um": wavelength_um,
                "wg_width_um": wg_width,
                "wg_height_um": wg_height,
                "bend_loss_db_cm": bend_loss,
                "raw_fit_slope_db_cm": float(slope),
                "negative_slope_clamped": bool(slope < 0 and clamp_negative_bend_loss),
                "scattering_loss_db_cm": scattering,
                "total_loss_db_cm": total_loss,
                "alpha_power_per_m": alpha_m,
                "q_intrinsic": qi,
                "q_intrinsic_is_infinite": alpha_m == 0,
                "q_intrinsic_valid": alpha_m >= 0,
                "n_group": n_group,
                "fsr_nm": wrappers[0].compute_fsr_nm(n_group),
                "fsr_ghz": wrappers[0].compute_fsr_ghz(n_group),
                "fit_intercept_db": float(intercept),
                "fit_r_squared": _r_squared(
                    y,
                    predicted,
                ),
                "fit_rmse_db": float(np.sqrt(np.mean(residual**2))),
                "fit_slope_std_error_db_cm": (
                    float(
                        np.sqrt(
                            np.sum(residual**2)
                            / (len(x) - 2)
                            / np.sum((x - x.mean()) ** 2)
                        )
                    )
                    if len(x) > 2
                    else None
                ),
                "fit_point_count": len(x),
                "fit_residuals_db": residual.tolist(),
                "fit_predicted_loss_db": predicted.tolist(),
            }
            metrics.update(
                _fom_metrics(
                    qi,
                    volumes[str(radius)],
                    resolved_volume_label,
                )
            )
            if slope < 0 and clamp_negative_bend_loss:
                logger.warning(
                    "[Warning] Negative fitted attenuation clamped to zero; inspect fit diagnostics."
                )
            report["results"][str(radius)] = metrics
            _emit(metrics)
            _save(
                path,
                report,
            )
            if save_plots:
                try:
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots()
                    ax.scatter(
                        x * 1e4,
                        y,
                        label="Arc measurements",
                    )
                    order = np.argsort(x)
                    ax.plot(
                        x[order] * 1e4,
                        predicted[order],
                        label="Linear fit",
                    )
                    ax.set(
                        xlabel="Arc length (um)",
                        ylabel="Insertion loss (dB)",
                        title=f"R = {radius:g} um",
                    )
                    ax.legend()
                    cutback_path = (
                        output_dir / "cutback_plots" / f"R_{radius}um" / "loss_fit.png"
                    )
                    cutback_path.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    fig.savefig(
                        cutback_path,
                        bbox_inches="tight",
                        dpi=plot_dpi,
                    )
                    metrics["cutback_plot_path"] = str(cutback_path)
                    plt.close(fig)
                except Exception as exc:
                    report.setdefault(
                        "plot_errors",
                        [],
                    ).append(str(exc))
                    logger.warning(f"[Warning] Plot failed: {exc}")
        try:
            report["radius_summary"] = plot_radius_summary(
                report["results"],
                output_dir=output_dir,
                save=save_plots,
                show=show_plots,
                plot_dpi=plot_dpi,
                annotate=annotate_summary,
            )
        except Exception as exc:
            report.setdefault(
                "plot_errors",
                [],
            ).append(f"Radius summary: {exc}")
            logger.warning(f"[Warning] Radius summary failed: {exc}")
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
    return report["results"]


def radius_sweep_analytical(
    radii_sweep_list : list,  # Distinct ring radii to evaluate (um).

    output_dir : Path|str,  # Directory for the analytical report and radius summary outputs.

    core_medium : td.Medium,        # Tidy3D medium used for the waveguide core.
    substrate_medium : td.Medium,   # Tidy3D medium for the substrate (BOX layer).

    wg_width : float,       # Waveguide core width (um).
    wg_height : float,      # Waveguide core height (um).

    wavelength_um : float,  # Central vacuum wavelength (um).
    cladding_medium=None,  # Upper cladding material (TOX Layer); None uses air.

    fine_steps_per_wvl : int = 20,      # Mesh resolution in the core region (steps per wavelength).
    coarse_steps_per_wvl : int = 15,    # Background mesh resolution (steps per wavelength).

    *,

    total_loss_db_cm : float,  # Total loss including scattering (dB/cm): scalar, radius mapping, or callable.
    scattering_loss_db_cm : float,  # Required scattering component (dB/cm), already included in total_loss_db_cm.

    n_group = None,  # Group index: scalar, radius mapping, or callable; None uses the mode solver.

    group_index_kwargs = None,  # Options passed to get_group_index, e.g. local=True or mode_index.

    db_to_neper_divisor : float = 4.343,  # Divide power loss in dB by this value to get exponential attenuation.

    manifest_filename : str="intrinsic_physics_manifest.json",  # JSON report filename relative to output_dir.

    plot_dpi : int = 150,  # Resolution of saved figures (dots per inch).

    volume_um3_by_radius = None,  # Volume (um³) by numeric/string radius or callable(radius_um); None uses core volume.
    volume_label = None,  # Description of the volume definition used in reports and plots.

    save_plots : bool = True,   # Save analysis plots to output_dir.
    show_plots : bool = False,  # Display the radius summary figure interactively.

    annotate_summary : bool = True,  # Label radius summary datapoints with their values.
):
    """Calculate radius metrics from supplied total power loss without FDTD.

    total_loss_db_cm includes scattering_loss_db_cm; scattering is not added
    again. Both losses are required. Total loss and n_group accept a scalar,
    a mapping keyed by numeric/string radius, or a callable of radius in um.
    If n_group is omitted, use the existing cross-section mode solver with
    group_index_kwargs (which may include local=True).

    Return the same radius-keyed metrics and write the same manifest and
    summary outputs as radius_sweep. Unavailable fit diagnostics are None or
    empty lists; no cutback plots or field maps are generated.
    """

    fine_steps_per_wvl = _positive(
        "fine_steps_per_wvl",
        fine_steps_per_wvl,
    )
    coarse_steps_per_wvl = _positive(
        "coarse_steps_per_wvl",
        coarse_steps_per_wvl,
    )
    from .ring_resonator import BaseResonator

    radii = _values(
        "radii_sweep_list",
        radii_sweep_list,
    )
    scattering = _positive(
        "scattering_loss_db_cm",
        scattering_loss_db_cm,
        True,
    )
    conversion = _positive(
        "db_to_neper_divisor",
        db_to_neper_divisor,
    )
    wg_width = _positive(
        "wg_width",
        wg_width,
    )
    wg_height = _positive(
        "wg_height",
        wg_height,
    )
    wavelength_um = _positive(
        "wavelength_um",
        wavelength_um,
    )
    _positive(
        "plot_dpi",
        plot_dpi,
    )
    if cladding_medium is None:
        cladding_medium = td.Medium(permittivity=1.0)
    group_options = dict(group_index_kwargs or {})
    volumes = {
        str(r): _resolve_volume(
            r,
            wg_width,
            wg_height,
            volume_um3_by_radius,
        )
        for r in radii
    }
    losses = {
        str(r): _radius_value(
            "total_loss_db_cm",
            total_loss_db_cm,
            r,
            True,
        )
        for r in radii
    }
    if any(loss < scattering for loss in losses.values()):
        raise ValueError("total_loss_db_cm must be at least scattering_loss_db_cm.")
    groups = (
        {str(r): _radius_value(
            "n_group",
            n_group,
            r,
        ) for r in radii}
        if n_group is not None
        else None
    )
    resolved_volume_label = volume_label or (
        "supplied volume" if volume_um3_by_radius is not None else "geometric core volume"
    )
    output_dir = Path(output_dir)
    path = output_dir / manifest_filename
    report = {
        "status": "running",
        "parameters": {
            "method": "analytical",
            "radii_um": radii,
            "wavelength_um": wavelength_um,
            "wg_width_um": wg_width,
            "wg_height_um": wg_height,
            "total_loss_db_cm": losses,
            "scattering_loss_db_cm": scattering,
            "n_group": groups,
            "group_index_kwargs": group_options,
            "fit_model": None,
            "bend_angles_deg": [],
            "db_to_neper_divisor": conversion,
            "save_plots": save_plots,
            "manifest_filename": manifest_filename,
            "plot_dpi": plot_dpi,
            "volume_um3_by_radius": volumes,
            "volume_label": resolved_volume_label,
            "fom_model": "(q_intrinsic / 2)**2 / full_ring_volume_um3",
            "show_plots": show_plots,
            "annotate_summary": annotate_summary,
        },
        "results": {},
        "raw_sweep": {},
        "errors": [],
    }
    _emit(report["parameters"])
    _save(
        path,
        report,
    )
    try:
        for radius in radii:
            obj = BaseResonator(
                sim=None,
                radius=radius,
                gap=0.0,
                wg_width=wg_width,
                wg_height=wg_height,
                core_medium=core_medium,
                substrate_medium=substrate_medium,
                substrate_height=2.5,
                top_cladding_medium=cladding_medium,
                top_cladding_height=1.5,
                fine_steps_per_wvl=fine_steps_per_wvl,
                coarse_steps_per_wvl=coarse_steps_per_wvl,
                wg_length=0.0,
                wavelength0=wavelength_um,
            )
            group = (
                groups[str(radius)]
                if groups is not None
                else _positive(
                    "n_group",
                    obj.get_group_index(**group_options),
                )
            )
            total_loss = losses[str(radius)]
            alpha_m = total_loss * 100 / conversion
            qi = (
                None
                if alpha_m == 0
                else 2 * np.pi * group / (wavelength_um * 1e-6 * alpha_m)
            )
            metrics = {
                "radius_um": radius,
                "wavelength_um": wavelength_um,
                "wg_width_um": wg_width,
                "wg_height_um": wg_height,
                "bend_loss_db_cm": total_loss - scattering,
                "raw_fit_slope_db_cm": None,
                "negative_slope_clamped": False,
                "scattering_loss_db_cm": scattering,
                "total_loss_db_cm": total_loss,
                "alpha_power_per_m": alpha_m,
                "q_intrinsic": qi,
                "q_intrinsic_is_infinite": alpha_m == 0,
                "q_intrinsic_valid": True,
                "n_group": group,
                "fsr_nm": obj.compute_fsr_nm(group),
                "fsr_ghz": obj.compute_fsr_ghz(group),
                "fit_intercept_db": None,
                "fit_r_squared": None,
                "fit_rmse_db": None,
                "fit_slope_std_error_db_cm": None,
                "fit_point_count": 0,
                "fit_residuals_db": [],
                "fit_predicted_loss_db": [],
            }
            metrics.update(_fom_metrics(
                qi,
                volumes[str(radius)],
                resolved_volume_label,
            ))
            report["results"][str(radius)] = metrics
            report["raw_sweep"][str(radius)] = []
            _emit(metrics)
            _save(
                path,
                report,
            )
        try:
            report["radius_summary"] = plot_radius_summary(
                report["results"],
                output_dir=output_dir,
                save=save_plots,
                show=show_plots,
                plot_dpi=plot_dpi,
                annotate=annotate_summary,
            )
        except Exception as exc:
            report.setdefault(
                "plot_errors",
                [],
            ).append(f"Radius summary: {exc}")
            logger.warning(f"[Warning] Radius summary failed: {exc}")
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
    return report["results"]


def _resolve_volume(
    radius,
    width,
    height,
    volumes,
):
    """Resolve a positive full-ring volume in cubic micrometers.
    Use a supplied callable or mapping, or estimate the geometric core volume.
    """
    if volumes is None:
        volume = (
            2
            * np.pi
            * radius
            * _positive(
                "wg_width",
                width,
            )
            * _positive(
                "wg_height",
                height,
            )
        )
    elif callable(volumes):
        volume = volumes(radius)
    else:
        volume = volumes[radius] if radius in volumes else volumes[str(float(radius))]
    return _positive(
        "full_ring_volume_um3",
        volume,
    )


def _fom_metrics(
    q_intrinsic,
    volume,
    label,
):
    """Calculate loaded Q and the volume-normalized figure of merit.
    Assume critical coupling and report None for invalid intrinsic Q values.
    """
    valid = q_intrinsic is not None and np.isfinite(q_intrinsic) and q_intrinsic > 0
    q_loaded = float(q_intrinsic / 2) if valid else None
    fom = float(q_loaded**2 / volume) if valid else None
    if fom is not None and not np.isfinite(fom):
        fom = None
    return {
        "volume_um3": float(volume),
        "volume_label": label,
        "q_loaded_at_critical": q_loaded,
        "fom_at_critical": fom,
        "fom_assumption": "critical coupling; Q_loaded = Q_intrinsic / 2",
    }


def plot_radius_summary(
    radius_results,  # Radius-keyed physics results returned by either radius sweep function.
    output_dir,  # Directory for the summary datapoints JSON and optional figure.
    *,
    wg_width=None,  # Fallback core width (um) when results lack a stored volume.
    wg_height=None,  # Fallback core height (um) when results lack a stored volume.
    volume_um3_by_radius=None,  # Optional volume override (um³): radius mapping or callable; else reuse results.
    volume_label=None,  # Description of the volume definition used in reports and plots.
    save=True,  # Save the summary figure; summary JSON is always written.
    show=True,  # Display the summary figure interactively.
    annotate=True,  # Label plotted datapoints with their numeric values.
    plot_dpi=150,  # Resolution of saved figures (dots per inch).
    figure_size=(15, 4.8),  # Summary figure width and height (inches).
    figure_filename="radius_fom_fsr.png",  # Summary figure filename relative to output_dir.
    data_filename="radius_fom_fsr_datapoints.json",  # Summary datapoints JSON filename relative to output_dir.
):
    """Plot in-memory radius results and export summary datapoints without solving.
    Use supplied or geometric volumes to estimate Q_loaded²/volume at critical coupling.
    """
    if not radius_results:
        raise ValueError("radius_results must contain at least one radius.")
    _positive(
        "plot_dpi",
        plot_dpi,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    rows = []
    for metrics in radius_results.values():
        radius = _positive(
            "radius_um",
            metrics["radius_um"],
        )
        if volume_um3_by_radius is not None:
            volume = _resolve_volume(
                radius,
                None,
                None,
                volume_um3_by_radius,
            )
            label = volume_label or "supplied volume"
        elif metrics.get("volume_um3") is not None:
            volume = _positive(
                "volume_um3",
                metrics["volume_um3"],
            )
            label = volume_label or metrics.get(
                "volume_label",
                "stored volume",
            )
        else:
            width = wg_width if wg_width is not None else metrics.get("wg_width_um")
            height = wg_height if wg_height is not None else metrics.get("wg_height_um")
            if width is None or height is None:
                raise ValueError(
                    "Supply wg_width/wg_height or volume_um3_by_radius for older results."
                )
            volume = _resolve_volume(
                radius,
                width,
                height,
                None,
            )
            label = volume_label or "geometric core volume"
        rows.append(
            {
                "radius_um": radius,
                "wavelength_um": metrics.get("wavelength_um"),
                "q_intrinsic": metrics.get("q_intrinsic"),
                **_fom_metrics(
                    metrics.get("q_intrinsic"),
                    volume,
                    label,
                ),
                "fsr_nm": _positive(
                    "fsr_nm",
                    metrics["fsr_nm"],
                ),
                "fsr_ghz": _positive(
                    "fsr_ghz",
                    metrics["fsr_ghz"],
                ),
                "fit_r_squared": metrics.get("fit_r_squared"),
                "negative_slope_clamped": metrics.get(
                    "negative_slope_clamped",
                    False,
                ),
            }
        )
    rows.sort(key=lambda row: row["radius_um"])
    if len({row["radius_um"] for row in rows}) != len(rows):
        raise ValueError("Duplicate radius entries in radius_results.")
    finite = [row for row in rows if row["fom_at_critical"] is not None]
    best = (
        max(
            finite,
            key=lambda row: row["fom_at_critical"],
        )
        if finite
        else None
    )
    summary = {
        "fom_assumption": "Estimated critical coupling, not a gap-validated loaded Q.",
        "datapoints": rows,
        "largest_estimated_fom_radius_um": None if best is None else best["radius_um"],
        "largest_estimated_fom": None if best is None else best["fom_at_critical"],
        "selection_note": "Compare FSR and loss-fit quality; the maximum alone does not establish an optimal device.",
        "figure_path": None,
        "plot_settings": {
            "annotate": annotate,
            "plot_dpi": plot_dpi,
            "figure_size": figure_size,
            "save": save,
            "show": show,
        },
    }
    data_path = output_dir / data_filename
    _save(
        data_path,
        summary,
    )
    if save or show:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            1,
            3,
            figsize=figure_size,
            constrained_layout=True,
        )
        try:
            radii = np.array([r["radius_um"] for r in rows])
            configs = [
                (
                    "fom_at_critical",
                    "Estimated critical-coupling FOM",
                    "Q_loaded² / V (um⁻³)",
                    "#146c94",
                ),
                ("fsr_nm", "FSR in wavelength", "FSR (nm)", "#6c3baa"),
                ("fsr_ghz", "FSR in frequency", "FSR (GHz)", "#277a51"),
            ]
            for ax, (key, title, ylabel, color) in zip(
                axes,
                configs,
            ):
                values = np.array([np.nan if r[key] is None else r[key] for r in rows])
                ax.plot(
                    radii,
                    values,
                    "o-",
                    color=color,
                    linewidth=1.8,
                    markersize=6,
                )
                ax.set(
                    title=title,
                    xlabel="Ring radius (um)",
                    ylabel=ylabel,
                )
                ax.grid(
                    True,
                    alpha=0.2,
                )
                ax.margins(
                    x=0.12,
                    y=0.22,
                )
                if annotate:
                    for x, y in zip(
                        radii,
                        values,
                    ):
                        if np.isfinite(y):
                            ax.annotate(
                                f"{y:.3g}",
                                (x, y),
                                xytext=(0, 9),
                                textcoords="offset points",
                                ha="center",
                                fontsize=8,
                            )
            clamped = [r for r in finite if r["negative_slope_clamped"]]
            if clamped:
                axes[0].scatter(
                    [r["radius_um"] for r in clamped],
                    [r["fom_at_critical"] for r in clamped],
                    marker="x",
                    color="#cf6418",
                    s=80,
                    zorder=4,
                    label="Bend loss clamped",
                )
            if best is not None:
                axes[0].scatter(
                    [best["radius_um"]],
                    [best["fom_at_critical"]],
                    s=150,
                    facecolors="none",
                    edgecolors="#252525",
                    zorder=5,
                    label=f"Largest estimate: R={best['radius_um']:g} um",
                )
            else:
                axes[0].text(
                    0.5,
                    0.5,
                    "No finite positive-Q FOM values",
                    ha="center",
                    va="center",
                    transform=axes[0].transAxes,
                )
            if best is not None or clamped:
                axes[0].legend(
                    fontsize=8,
                    loc="best",
                )
            labels = sorted({r["volume_label"] for r in rows})
            fig.suptitle(
                "Radius comparison · Qi/2 assumption · " + ", ".join(labels),
                fontsize=12,
            )
            if save:
                figure_path = output_dir / figure_filename
                figure_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                fig.savefig(
                    figure_path,
                    dpi=plot_dpi,
                    bbox_inches="tight",
                )
                summary["figure_path"] = str(figure_path)
            if show:
                plt.show()
        finally:
            plt.close(fig)
            _save(
                data_path,
                summary,
            )
    _emit(summary)
    return summary
