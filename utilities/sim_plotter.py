"""
utilities.sim_plotter.py

General visualization utilities for Tidy3D simulation geometries,
supporting both 2D slices and full 3D multi-planar projections.

Author: Esraaj Sarkar Gupta
Date: 2026

Exposes: display_2d(), display_3d().
"""

from tarang import get_logger

from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import tidy3d as td

logger = get_logger(__name__)


def _extract_simulation(device) -> td.Simulation:
    """Extracts td.Simulation whether given a wrapper object or a raw Simulation.
    Raise TypeError when neither input form contains a supported simulation.
    """
    if isinstance(
        device,
        td.Simulation,
    ):
        return device
    if hasattr(
        device,
        "sim",
    ) and isinstance(
        device.sim,
        td.Simulation,
    ):
        return device.sim
    raise TypeError(
        f"Expected a Tidy3D Simulation or an object with a '.sim' attribute, but got {type(device).__name__}."
    )


def _validate_and_get_save_path(
    output_path: Optional[Union[str, Path]],
    default_filename: str,
) -> Path:
    """
    Ensures a valid output path exists when save=True.
    Handles directory paths cleanly by appending default_filename if no file is specified.
    """
    if output_path is None or str(output_path).strip() == "":
        raise ValueError(
            "Cannot save plot: 'save' is True, but no valid 'output_path' was provided."
        )

    path = Path(output_path)

    if path.is_dir() or str(output_path).endswith(("/", "\\")) or not path.suffix:
        path = path / default_filename

    if path.parent:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    return path


def display_2d(
    device,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "2D Structural Layout Verification",
    slice_axis: str = "y",
    slice_coord: float = 0.0,
    show: bool = True,
    save: bool = False,
    *,
    plot_dpi: int = 300,  # Resolution of displayed and saved figures (dots per inch).
) -> None:
    """
    Extracts a pristine 2D cross-sectional layout map of a simulation object
    along a single specified axis. Saves to disk only if save=True.
    """
    sim = _extract_simulation(device)

    fig, ax = plt.subplots(
        figsize=(11, 4),
        dpi=plot_dpi,
    )

    axis_clean = slice_axis.lower()
    if axis_clean == "x":
        sim.plot(
            x=slice_coord,
            ax=ax,
        )
        ax.set_xlabel(
            "Y (μm)",
            fontsize=10,
        )
        ax.set_ylabel(
            "Z (μm)",
            fontsize=10,
        )
    elif axis_clean == "y":
        sim.plot(
            y=slice_coord,
            ax=ax,
        )
        ax.set_xlabel(
            "X (μm)",
            fontsize=10,
        )
        ax.set_ylabel(
            "Z (μm)",
            fontsize=10,
        )
    elif axis_clean == "z":
        sim.plot(
            z=slice_coord,
            ax=ax,
        )
        ax.set_xlabel(
            "X (μm)",
            fontsize=10,
        )
        ax.set_ylabel(
            "Y (μm)",
            fontsize=10,
        )
    else:
        plt.close(fig)
        raise ValueError("Invalid slice axis. Choose from 'x', 'y', or 'z'.")

    ax.set_title(
        title,
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(
        True,
        alpha=0.2,
        linestyle="--",
    )

    fig.tight_layout()

    if save:
        path = _validate_and_get_save_path(
            output_path,
            "example_2d_simulation.png",
        )
        plt.savefig(
            path,
            dpi=plot_dpi,
            bbox_inches="tight",
        )
        logger.info(f"[Visualizer] 2D geometry layout map saved to: {path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def display_3d(
    device,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "3D Structural Layout Verification",
    coordinates: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    show: bool = True,
    save: bool = False,
    *,
    plot_dpi: int = 300,  # Resolution of displayed and saved figures (dots per inch).
) -> None:
    """
    Generates a comprehensive three-pane orthogonal projection sheet (XY, XZ, YZ slices)
    to visually audit internal volume. Saves to disk only if save=True.
    """
    sim = _extract_simulation(device)
    x_coord, y_coord, z_coord = coordinates

    fig, (ax_xy, ax_xz, ax_yz) = plt.subplots(
        1,
        3,
        figsize=(18, 5),
        dpi=plot_dpi,
    )
    fig.suptitle(
        title,
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    # Top-Down View (XY Plane)
    sim.plot(
        z=z_coord,
        ax=ax_xy,
    )
    ax_xy.set_title(
        f"Top-Down View (XY Plane at Z = {z_coord:.2f} μm)",
        fontsize=10,
    )
    ax_xy.set_xlabel("X (μm)")
    ax_xy.set_ylabel("Y (μm)")
    ax_xy.grid(
        True,
        alpha=0.2,
        linestyle="--",
    )

    # Longitudinal Side-View (XZ Plane)
    sim.plot(
        y=y_coord,
        ax=ax_xz,
    )
    ax_xz.set_title(
        f"Longitudinal View (XZ Plane at Y = {y_coord:.2f} μm)",
        fontsize=10,
    )
    ax_xz.set_xlabel("X (μm)")
    ax_xz.set_ylabel("Z (μm)")
    ax_xz.grid(
        True,
        alpha=0.2,
        linestyle="--",
    )

    # Transverse Cross-Section (YZ Plane)
    sim.plot(
        x=x_coord,
        ax=ax_yz,
    )
    ax_yz.set_title(
        f"Cross-Section View (YZ Plane at X = {x_coord:.2f} μm)",
        fontsize=10,
    )
    ax_yz.set_xlabel("Y (μm)")
    ax_yz.set_ylabel("Z (μm)")
    ax_yz.grid(
        True,
        alpha=0.2,
        linestyle="--",
    )

    fig.tight_layout()

    if save:
        path = _validate_and_get_save_path(
            output_path,
            "example_3d_simulation.png",
        )
        plt.savefig(
            path,
            dpi=plot_dpi,
            bbox_inches="tight",
        )
        logger.info(f"[Visualizer] 3D orthogonal layout sheet saved to: {path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
