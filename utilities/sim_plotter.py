"""
utilities.sim_plotter.py

General visualization utilities for Tidy3D simulation geometries, 
supporting both 2D slices and full 3D multi-planar projections.

Author: Esraaj Sarkar Gupta
Date: 2026
"""

from pathlib import Path
import matplotlib.pyplot as plt
import tidy3d as td
import logging

logger = logging.getLogger(__name__)


def save_simulation_layout_2d(
    sim: td.Simulation,
    output_path: Path,
    title: str = "2D Structural Layout Verification",
    slice_axis: str = "y",
    slice_coord: float = 0.0,
    show : bool = False,
    save : bool = True
) -> None:
    """
    Docstring for save_simulation_layout_2d
    
    :param sim: Description
    :type sim: td.Simulation
    :param output_path: Description
    :type output_path: Path
    :param title: Description
    :type title: str
    :param slice_axis: Description
    :type slice_axis: str
    :param slice_coord: Description
    :type slice_coord: float

    Extracts a pristine 2D cross-sectional layout map of a simulation object 
    along a single specified axis and saves it to the designated disk path.
    """
    # Ensure parent directories exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(11, 4), dpi=300)
    
    # Dynamically select the structural slicing plane
    if slice_axis.lower() == "x":
        sim.plot(x=slice_coord, ax=ax)
        ax.set_xlabel("Y (μm)", fontsize=10)
        ax.set_ylabel("Z (μm)", fontsize=10)
    elif slice_axis.lower() == "y":
        sim.plot(y=slice_coord, ax=ax)
        ax.set_xlabel("X (μm)", fontsize=10)
        ax.set_ylabel("Z (μm)", fontsize=10)
    elif slice_axis.lower() == "z":
        sim.plot(z=slice_coord, ax=ax)
        ax.set_xlabel("X (μm)", fontsize=10)
        ax.set_ylabel("Y (μm)", fontsize=10)
    else:
        raise ValueError("Invalid slice axis. Choose from 'x', 'y', or 'z'.")
        
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2, linestyle="--")
    
    fig.tight_layout()
    if save:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    logger.info(f"[Visualizer] 2D geometry layout map saved to: {output_path}")


def save_simulation_layout_3d(
    sim: td.Simulation,
    output_path: Path,
    title: str = "3D Structural Layout Verification",
    coordinates : tuple = (0.0, 0.0, 0.0),
    show : bool = False,
    save : bool = True
) -> None:
    """
    Docstring for save_simulation_layout_3d
    
    :param sim: Description
    :type sim: td.Simulation
    :param output_path: Description
    :type output_path: Path
    :param title: Description
    :type title: str
    :param coordinates: Description
    :type coordinates: list[float]

    Generates a comprehensive three-pane orthogonal projection sheet (XY, XZ, YZ slices)
    to visually audit the internal layout of a full 3D simulation volume.
    """ 
    # Ensure parent directories exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_coord = coordinates[0]; y_coord = coordinates[1]; z_coord = coordinates[2]
    
    # Create a 3-column canvas for the three primary planes
    fig, (ax_xy, ax_xz, ax_yz) = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    
    # Top-Down View (XY Plane cut at designated Z)
    sim.plot(z=z_coord, ax=ax_xy)
    ax_xy.set_title(f"Top-Down View (XY Plane at Z = {z_coord:.2f} μm)", fontsize=10)
    ax_xy.set_xlabel("X (μm)")
    ax_xy.set_ylabel("Y (μm)")
    ax_xy.grid(True, alpha=0.2, linestyle="--")
    
    # Longitudinal Side-View (XZ Plane cut at designated Y)
    sim.plot(y=y_coord, ax=ax_xz)
    ax_xz.set_title(f"Longitudinal View (XZ Plane at Y = {y_coord:.2f} μm)", fontsize=10)
    ax_xz.set_xlabel("X (μm)")
    ax_xz.set_ylabel("Z (μm)")
    ax_xz.grid(True, alpha=0.2, linestyle="--")
    
    # Transverse Cross-Section (YZ Plane cut at designated X)
    sim.plot(x=x_coord, ax=ax_yz)
    ax_yz.set_title(f"Cross-Section View (YZ Plane at X = {x_coord:.2f} μm)", fontsize=10)
    ax_yz.set_xlabel("Y (μm)")
    ax_yz.set_ylabel("Z (μm)")
    ax_yz.grid(True, alpha=0.2, linestyle="--")
    
    fig.tight_layout()
    if save:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    logger.info(f"[Visualizer] 3D orthogonal layout sheet saved to: {output_path}")