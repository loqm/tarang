"""
tarang.grating_coupler.grating_coupler

Defines the GratingConfig dataclass and GratingCoupler wrapper class.

Author: Esraaj Sarkar Gupta, Divyanshu Tiwari
Date: 2026

Exposes: GratingConfig, GratingCoupler, GratingCoupler.from_config(),
GratingCoupler.get_dir_name(), GratingCoupler.is_cached(),
GratingCoupler.get_efficiency(), GratingCoupler.get_spectrum(),
GratingCoupler.get_spectrum_summary(), GratingCoupler.get_fields(),
phase_matched_period(), spectrum_summary(), tilt_sign().
Set ``GratingCoupler.max_cost`` (FlexCredits) to cap cloud runs.
"""

from __future__ import annotations
from tarang import get_logger
from tarang._compat import RenamedAttributes, renamed_params
from tarang.utilities.cloud_run import CloudCachedDevice

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import tidy3d as td


logger = get_logger(__name__)


@dataclass
class GratingConfig(RenamedAttributes):
    """Grating-coupler geometry and excitation (lengths in um, angles in degrees).

    Layer stack, top to bottom: top cladding (top_cladding_height) / SiN core
    (core_height) / buried oxide (box_height) / optional Si handle
    (handle_height, used only when a handle_medium is given).

    Fibre orientation. The waveguide leaves the grating towards -x. ``theta_deg`` is the
    fibre angle from vertical (its sign is ignored) and ``fiber_tilt`` says which way the
    beam travels along the grating:

    * ``"towards"`` (default): the beam travels towards the waveguide; the fibre's upper
      end leans away from it. First-order phase matching:
      Lambda = lambda / (n_eff - n_cover sin(theta)). Two fibres feeding mirrored
      couplers form a V, as on a usual two-holder setup.
    * ``"away"``: the beam travels away from the waveguide;
      Lambda = lambda / (n_eff + n_cover sin(theta)). This was the only orientation
      before v0.1.0 (any theta was made negative).

    ``source_x`` is the beam centre where it reaches the top of the core (x = 0 is the
    start of the grating), so it stays the fibre position on the grating whatever the
    angle and source height. ``bandwidth_um`` is the half-width of the band recorded by
    the mode monitor (efficiency spectrum).
    """

    wavelength_um: float

    theta_deg: float
    mfd: float
    source_x: float
    source_gap: float

    # Layer stack
    core_height: float
    top_cladding_height: float  # Cladding above the core (was cladding_height)
    box_height: float  # Isolating oxide layer beneath core
    handle_height: float  # Deep silicon handle (was substrate_height)

    # Coupler geometry
    etch_depth: float
    period: float
    ff: float  # Fill Factor
    n_teeth: int
    w_grating: float

    # Source
    pol_angle: float

    buffer: float

    fiber_tilt: str = "towards"
    bandwidth_um: float = 0.1
    # Apodised grating: explicit teeth as ((x_start, length), ...) in um, first tooth at
    # x = 0; then period, ff and n_teeth only describe the design (n_teeth = len(teeth)).
    teeth: tuple | None = None

    _renamed_attrs = {"cladding_height": "top_cladding_height",
                      "substrate_height": "handle_height"}


GratingConfig.__init__ = renamed_params(
    cladding_height="top_cladding_height", substrate_height="handle_height"
)(GratingConfig.__init__)

FIBER_TILTS = ("towards", "away")


def tilt_sign(fiber_tilt: str) -> int:
    """+1 for "towards", -1 for "away" (sign of Tidy3D's ``angle_theta`` for a beam
    injected downwards; checked against the source's direction vector in the tests)."""
    tilt = str(fiber_tilt).strip().lower()
    if tilt not in FIBER_TILTS:
        raise ValueError(f"fiber_tilt must be one of {FIBER_TILTS}, not {fiber_tilt!r}")
    return 1 if tilt == "towards" else -1


def phase_matched_period(n_eff: float, wavelength_um: float, theta_deg: float,
                         fiber_tilt: str = "towards", n_cover: float = 1.0) -> float:
    """First-order grating period that couples a fibre at ``theta_deg`` (from vertical)
    into a mode of effective index ``n_eff``: lambda / (n_eff -+ n_cover sin(theta))."""
    s = n_cover * np.sin(np.deg2rad(abs(theta_deg)))
    return float(wavelength_um / (n_eff - tilt_sign(fiber_tilt) * s))


def spectrum_summary(wavelength_um, efficiency, target_um: float) -> dict:
    """Peak, value at the target and 1 dB / 3 dB bandwidths of an efficiency spectrum.

    Bandwidths are the width of the contiguous region around the peak within 1 dB / 3 dB
    of it; ``*_closed`` is False when that region reaches the end of the recorded band
    (the true bandwidth is then larger). ``peak_inside`` is False when the peak is at the
    edge of the band (the true peak lies outside it). The peak wavelength is refined
    between samples with a parabola; ``band_1db_lo_um`` / ``band_1db_hi_um`` (and 3 dB) are the band edges.
    """
    wl = np.asarray(wavelength_um, float)
    eta = np.asarray(efficiency, float)
    order = np.argsort(wl)
    wl, eta = wl[order], eta[order]
    i = int(np.argmax(eta))
    peak = float(eta[i])
    peak_wl = float(wl[i])
    if 0 < i < len(wl) - 1:
        # parabola through the highest sample and its neighbours: the peak wavelength to a
        # fraction of the sample spacing (the spectrum is sampled every ~2 nm)
        y0, y1, y2 = eta[i - 1], eta[i], eta[i + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            shift = 0.5 * (y0 - y2) / den
            peak_wl = float(wl[i] + shift * 0.5 * (wl[i + 1] - wl[i - 1]))
    out = dict(peak_wavelength_um=peak_wl, peak_efficiency=peak,
               peak_loss_db=float(-10 * np.log10(max(peak, 1e-12))),
               efficiency_at_target=float(np.interp(target_um, wl, eta)),
               peak_inside=bool(0 < i < len(wl) - 1))
    out["loss_db_at_target"] = float(-10 * np.log10(max(out["efficiency_at_target"], 1e-12)))
    for db in (1, 3):
        level = peak * 10 ** (-db / 10)
        lo = i
        while lo > 0 and eta[lo - 1] >= level:
            lo -= 1
        hi = i
        while hi < len(eta) - 1 and eta[hi + 1] >= level:
            hi += 1
        def edge(below: int, above: int) -> float:   # level crossing between two samples
            e0, e1 = eta[below], eta[above]
            t = 0.0 if e1 == e0 else (level - e0) / (e1 - e0)
            return float(wl[below] + t * (wl[above] - wl[below]))
        left = edge(lo - 1, lo) if lo > 0 else float(wl[0])
        right = edge(hi + 1, hi) if hi < len(eta) - 1 else float(wl[-1])
        out[f"bandwidth_{db}db_nm"] = (right - left) * 1e3
        out[f"bandwidth_{db}db_closed"] = bool(lo > 0 and hi < len(eta) - 1)
        out[f"band_{db}db_lo_um"], out[f"band_{db}db_hi_um"] = left, right
    return out


class GratingCoupler(CloudCachedDevice, RenamedAttributes):
    """
    Wraps a Tidy3D Simulation object for grating couplers, binding
    its GratingConfig parameters and material media directly to the object.

    Media: core_medium, box_medium (oxide under the core), handle_medium (Si
    wafer; None terminates the domain in the BOX), top_cladding_medium (None = air).
    """

    _renamed_attrs = {"substrate_medium": "handle_medium",
                      "cladding_medium": "top_cladding_medium"}

    @renamed_params(substrate_medium="handle_medium", cladding_medium="top_cladding_medium")
    def __init__(
        self,
        sim: td.Simulation,
        config: GratingConfig,
        core_medium: td.Medium,
        box_medium: td.Medium,
        handle_medium: td.Medium,
        top_cladding_medium: td.Medium | None = None,  # Defaults to "air" if None
        is_3d: bool = False,
        fine_steps_per_wvl: int = 20,
        coarse_steps_per_wvl: int = 15,
    ):
        """Initialize the grating device from its simulation and configuration.
        Store the output settings used for simulation and efficiency analysis.
        """
        self.sim = sim
        self.config = config

        self.core_medium = core_medium
        self.box_medium = box_medium
        self.handle_medium = handle_medium
        self.top_cladding_medium = top_cladding_medium

        self.is_3d = is_3d

        self.fine_steps_per_wvl = fine_steps_per_wvl
        self.coarse_steps_per_wvl = coarse_steps_per_wvl

    @classmethod
    @renamed_params(substrate_medium="handle_medium", cladding_medium="top_cladding_medium")
    def from_config(
        cls,
        config: GratingConfig,
        core_medium: td.Medium,
        box_medium: td.Medium,
        handle_medium: td.Medium | None = None,
        top_cladding_medium: td.Medium | None = None,
        is_3d: bool = False,
        fine_steps_per_wvl: int = 15,
        coarse_steps_per_wvl: int = 10,
    ) -> GratingCoupler:
        """Convenience factory method to build and wrap a simulation in one step.
        Build the active grating geometry without submitting a cloud simulation.
        """
        from tarang.geometries.grating_coupler import get_active_simulation

        return get_active_simulation(
            config=config,
            core_medium=core_medium,
            box_medium=box_medium,
            handle_medium=handle_medium,
            top_cladding_medium=top_cladding_medium,
            is_3d=is_3d,
            fine_steps_per_wvl=fine_steps_per_wvl,
            coarse_steps_per_wvl=coarse_steps_per_wvl,
        )

    # --------------------------------
    # Execution and Analysis Methods
    # --------------------------------
    _task_prefix = "gc"

    def _readable_name(self) -> str:
        """Short, human-readable cache stem; the full parameter set is stored in
        simulation.json beside the results and in the fingerprint."""
        cfg = self.config
        dim = "3D" if self.is_3d else "2D"
        if cfg.teeth:
            import hashlib
            sig = ",".join(f"{a:.4f}:{b:.4f}" for a, b in cfg.teeth)
            h = hashlib.sha256(sig.encode()).hexdigest()[:6]
            return (f"gc{dim}_apod{len(cfg.teeth)}_{h}_e{cfg.etch_depth:.4f}_"
                    f"sx{cfg.source_x:.4f}_{cfg.wavelength_um:.3f}um")
        return (
            f"gc{dim}_p{cfg.period:.4f}_ff{cfg.ff:.4f}_e{cfg.etch_depth:.4f}_"
            f"n{cfg.n_teeth}_sx{cfg.source_x:.4f}_{cfg.wavelength_um:.3f}um"
        )

    def _legacy_dir_name(self) -> str:
        """Directory name used before v0.1.0 (geometry only, no media). Results
        found there are reused only after their stored simulation is verified."""
        dim = "3D" if self.is_3d else "2D"
        cfg = self.config
        return (
            f"{dim}_p_{cfg.period:.4f}_dc_{cfg.ff:.4f}_e_{cfg.etch_depth:.4f}_"
            f"n_{cfg.n_teeth}_w_{cfg.w_grating:.4f}_ch_{cfg.core_height:.4f}_"
            f"clh_{cfg.top_cladding_height:.4f}_bh_{cfg.box_height:.4f}_sh_{cfg.handle_height:.4f}_"
            f"t_{cfg.theta_deg:.4f}_mfd_{cfg.mfd:.4f}_sx_{cfg.source_x:.4f}_sg_{cfg.source_gap:.4f}_"
            f"wl_{cfg.wavelength_um:.4f}_pol_{cfg.pol_angle:.4f}_b_{cfg.buffer:.4f}"
        )

    # get_dir_name(), is_cached() and _run_sim() come from CloudCachedDevice:
    # the cache key is the full simulation fingerprint (geometry, media, mesh,
    # sources, monitors), and every cloud run is cost-estimated first.

    def get_efficiency(
        self,
        output_dir: Path,
        verbose: bool = False,
    ) -> float:
        """
        Executes the simulation (or loads from cache) and extracts the coupling
        efficiency for the fundamental mode at the target wavelength.
        """
        sim_data = self._run_sim(
            output_dir,
            verbose=verbose,
        )

        logger.info(
            "[Analytics] Extracting scattering coefficients from 'mode monitor'..."
        )
        try:
            mode_amps_array = sim_data["mode monitor"].amps.sel(
                direction="-",
                mode_index=0,
            )
            target_freq = td.C_0 / self.config.wavelength_um

            efficiency = float(
                np.abs(
                    mode_amps_array.sel(
                        f=target_freq,
                        method="nearest",
                    )
                )
                ** 2
            )
            peak_loss = 10.0 * np.log10(efficiency + 1e-12)

            logger.info(f"[Analytics] Target mode audit completed.\n\
\tCoupling Efficiency: {efficiency*100.0:.2f}%\n\
\tInsertion Loss:      {peak_loss:.2f} dB")
            return efficiency

        except Exception as e:
            logger.error(f"[Analytics Error] Failed to parse mode data cleanly: {e}")
            return 0.0

    def get_spectrum(self, output_dir: Path, verbose: bool = False):
        """Run (or load from the cache) and return (wavelength_um, efficiency) of the
        fundamental mode coupled into the waveguide over the recorded band."""
        sim_data = self._run_sim(output_dir, verbose=verbose)
        amps = sim_data["mode monitor"].amps.sel(direction="-", mode_index=0)
        wl = td.C_0 / np.asarray(amps.f.values, float)
        eta = np.abs(np.asarray(amps.values)) ** 2
        order = np.argsort(wl)
        return wl[order], eta[order]

    def get_spectrum_summary(self, output_dir: Path, verbose: bool = False) -> dict:
        """``spectrum_summary`` of this coupler at its design wavelength."""
        wl, eta = self.get_spectrum(output_dir, verbose=verbose)
        return spectrum_summary(wl, eta, self.config.wavelength_um)

    def get_fields(
        self,
        output_dir: Path,
        show: bool = True,
        *,
        plot_dpi: int = 300,  # Resolution of displayed and saved figures (dots per inch).
    ) -> None:
        """
        Executes the simulation (or loads from cache) and plots spatial field
        distribution maps for valid 2D FieldMonitors.
        """
        cfg = self.config
        dim_str = "3D" if self.is_3d else "2D"

        param_dir_name = self.get_dir_name()
        target_dir = Path(output_dir) / param_dir_name

        sim_data = self._run_sim(
            output_dir,
            verbose=False,
        )
        target_freq = td.C_0 / cfg.wavelength_um

        logger.info(
            f"[Analytics] Processing spatial field distributions ({dim_str} mode)..."
        )

        for monitor in self.sim.monitors:
            if isinstance(
                monitor,
                td.ModeMonitor,
            ):
                logger.debug(
                    f"[Analytics] Skipping ModeMonitor '{monitor.name}' during spatial field plotting."
                )
                continue

            if isinstance(
                monitor,
                td.FieldMonitor,
            ):
                if not self.is_3d and monitor.name != "field_y_plane":
                    logger.info(
                        f"[Analytics] Skipping 3D-only slice '{monitor.name}' for 2D simulation."
                    )
                    continue

                non_zero_axes = sum(1 for axis_len in monitor.size if axis_len > 0)
                if non_zero_axes < 2:
                    logger.warning(
                        f"[Analytics] Skipping '{monitor.name}': degenerate slice with <2 spatial dimensions."
                    )
                    continue

                try:
                    fig, ax = plt.subplots(
                        figsize=(10, 5),
                        dpi=plot_dpi,
                    )

                    sim_data.plot_field(
                        field_monitor_name=monitor.name,
                        field_name="E",
                        val="abs^2",
                        f=target_freq,
                        ax=ax,
                    )

                    ax.set_title(
                        f"Field Intensity Map (|E|²): {monitor.name} ({dim_str})",
                        fontweight="bold",
                    )

                    save_path = target_dir / f"spatial_field_{monitor.name}.png"
                    plt.savefig(
                        save_path,
                        bbox_inches="tight",
                        dpi=plot_dpi,
                    )
                    logger.info(
                        f"[Analytics] Exported field profile slice: {save_path.name}"
                    )

                    if show:
                        plt.show()

                    plt.close(fig)

                except Exception as e:
                    logger.error(
                        f"[Analytics Error] Could not render field monitor plot '{monitor.name}': {e}"
                    )
