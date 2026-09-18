"""
waveguide.waveguide

Defines the RidgeWaveguide Class with self-contained modal analysis,
dispersion regime evaluation, and spatial confinement extraction.

Author: Esraaj Sarkar Gupta
Date: July, 2026

Exposes: WaveguideModeError, RidgeWaveguide, RidgeWaveguide.get_modes(),
RidgeWaveguide.get_dispersion(), RidgeWaveguide.get_dispersion_vs_wavelength(),
RidgeWaveguide.get_confinement().
"""

from __future__ import annotations
from tarang import get_logger
from utilities.validation import positive as _positive, values as _values, integer as _integer

from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import tidy3d as td
from tidy3d.plugins.mode import ModeSolver

# Speed of light in um/ps (from Tidy3D native um/s constant)
C_UM_PS = td.C_0 / 1e12  # Result: ~299.792458


logger = get_logger(__name__)


class WaveguideModeError(Exception):
    """Raised when mode solving fails or required mode conditions are not met."""

    pass


class RidgeWaveguide:
    def __init__(
        self,
        sim: td.Simulation,
        width: float,
        height: float,
        length: float,
        core_medium: td.Medium,
        sub_medium: td.Medium,
        wavelength_um: float,
        top_cladding_medium: Optional[td.Medium] = None,
        top_cladding_height: float = 0.0,
        substrate_height: float = 4.0,
        fine_min_steps_per_wvl: int = 20,
        coarse_min_steps_per_wvl: int = 10,
        margin_y: float = 2.0,  # Buffer space on each side of the core in Y
        margin_z: float = 2.0,  # Air/PML buffer space above the top cladding/core in Z
        run_time: float = 1e-12,
    ):
        """Initialize the device wrapper with its simulation and physical settings.
        Store the parameters used by subsequent analysis and geometry conversion.
        """
        self.sim = sim

        self.width = width
        self.height = height

        self.length = length

        self.core_medium = core_medium
        self.sub_medium = sub_medium

        self.top_cladding_medium = top_cladding_medium
        self.top_cladding_height = top_cladding_height

        self.wavelength_um = wavelength_um
        self.substrate_height = substrate_height

        self.fine_min_steps_per_wvl = fine_min_steps_per_wvl
        self.coarse_min_steps_per_wvl = coarse_min_steps_per_wvl

        self.margin_y = margin_y
        self.margin_z = margin_z

        self.run_time = run_time

    def __getattr__(
        self,
        item,
    ):
        """Delegate unknown attributes to the wrapped Tidy3D simulation.
        Preserve direct access to its geometry and solver settings.
        """
        return getattr(
            self.sim,
            item,
        )

    def _get_te_fraction(
        self,
        mode_data: Any,
        m_idx: int,
        f_idx: int = 0,
    ) -> float:
        """Computes TE polarization fraction for a waveguide propagating along X.
        Return the fraction of transverse electric intensity carried by Ey.
        """
        Ey = mode_data.Ey.isel(
            mode_index=m_idx,
            f=f_idx,
        ).values
        Ez = mode_data.Ez.isel(
            mode_index=m_idx,
            f=f_idx,
        ).values
        Iy = np.sum(np.abs(Ey) ** 2)
        Iz = np.sum(np.abs(Ez) ** 2)
        return float(Iy / (Iy + Iz + 1e-10))

    def get_modes(
        self,
        num_modes: int = 8,
    ) -> Dict[str, Any]:
        """Solve optical modes and return effective indices and polarization fractions.
        Include TE/TM classifications and guiding conditions in the result.
        """
        freq0 = td.C_0 / self.wavelength_um
        n_core = float(np.real(np.sqrt(self.core_medium.eps_model(freq0))))
        n_sub = float(np.real(np.sqrt(self.sub_medium.eps_model(freq0))))

        mode_plane = td.Box(
            center=(0, 0, self.height / 2),
            size=(0, self.width + 2.0, self.height + 2.0),
        )

        mode_spec = td.ModeSpec(
            num_modes=num_modes,
            target_neff=n_core,
        )
        solver = ModeSolver(
            simulation=self.sim,
            plane=mode_plane,
            mode_spec=mode_spec,
            freqs=[freq0],
        )

        logger.info(
            f"Solving top {num_modes} modes for W={self.width:.3f} μm, H={self.height:.3f} μm..."
        )
        try:
            mode_data = solver.solve()
        except Exception as e:
            raise WaveguideModeError(f"Tidy3D ModeSolver failed: {e}") from e

        modes_info = []
        for i in range(num_modes):
            n_eff = float(np.real(mode_data.n_eff[0, i]))
            te_frac = self._get_te_fraction(
                mode_data,
                m_idx=i,
                f_idx=0,
            )
            pol_label = "TE" if te_frac > 0.5 else "TM"
            is_guided = n_eff > n_sub

            modes_info.append(
                {
                    "mode_index": i,
                    "neff": n_eff,
                    "polarization": pol_label,
                    "te_fraction": te_frac,
                    "is_guided": is_guided,
                }
            )

        if not any(m["is_guided"] for m in modes_info):
            # raise WaveguideModeError(f"No guided modes found for W={self.width:.3f} μm, H={self.height:.3f} μm.")
            logger.error(
                f"No guided modes found for W={self.width:.3f} μm, H={self.height:.3f} μm."
            )

        return {
            "width_um": self.width,
            "height_um": self.height,
            "top_cladding_height_um": self.top_cladding_height,
            "wavelength_um": self.wavelength_um,
            "n_core": n_core,
            "n_sub": n_sub,
            "modes": modes_info,
            "raw_mode_data": mode_data,
        }

    def get_dispersion(
        self,
        delta_wvl_um: float = 0.1,
        num_points: int = 11,
    ) -> Dict[str, Any]:
        """Calculate chromatic dispersion D and classify its dispersion regime.
        Positive D is anomalous, negative D is normal, and near-zero D is reported.
        """
        freq0 = td.C_0 / self.wavelength_um
        n_core_guess = float(np.real(np.sqrt(self.core_medium.eps_model(freq0))))
        n_sub_guess = float(np.real(np.sqrt(self.sub_medium.eps_model(freq0))))

        mid_idx = num_points // 2
        wavelengths = np.linspace(
            self.wavelength_um - delta_wvl_um / 2,
            self.wavelength_um + delta_wvl_um / 2,
            num_points,
        )
        freqs = td.C_0 / wavelengths

        mode_spec = td.ModeSpec(
            num_modes=4,
            target_neff=n_core_guess,
        )
        plane = td.Box(
            center=(0, 0, self.height / 2),
            size=(0, self.sim.size[1], self.sim.size[2]),
        )

        solver = ModeSolver(
            simulation=self.sim,
            plane=plane,
            mode_spec=mode_spec,
            freqs=freqs,
        )
        try:
            data = solver.solve()
        except Exception as e:
            raise WaveguideModeError(f"Dispersion ModeSolver failed: {e}") from e

        # Isolate fundamental TE00 mode
        te00_idx = -1
        for m in range(len(data.n_eff.mode_index)):
            if (
                self._get_te_fraction(
                    data,
                    m_idx=m,
                    f_idx=mid_idx,
                )
                > 0.5
            ):
                te00_idx = m
                break

        if te00_idx == -1:
            raise WaveguideModeError(
                f"TE00 mode not found for W={self.width:.3f} μm, H={self.height:.3f} μm."
            )

        # Single-mode condition check
        neff_at_center = np.real(data.n_eff.isel(f=mid_idx).values)
        n_higher = neff_at_center[2] if len(neff_at_center) > 2 else 0.0
        is_singlemode = bool(n_higher <= (n_sub_guess + 0.001))

        # Numerical second-derivative across wavelength array
        neff_vals = np.real(data.n_eff.isel(mode_index=te00_idx).values)
        sorted_idx = np.argsort(wavelengths)
        srt_wvl = wavelengths[sorted_idx]
        srt_neff = neff_vals[sorted_idx]

        dn_dlam = np.gradient(
            srt_neff,
            srt_wvl,
        )
        d2n_dlam2 = np.gradient(
            dn_dlam,
            srt_wvl,
        )

        # D = - (lambda / c) * d2n/dlambda2 * 1e6 -> [ps / (nm * km)]
        D_val = float(-(srt_wvl[mid_idx] / C_UM_PS) * d2n_dlam2[mid_idx] * 1e6)

        # Classify the dispersion regime
        if D_val > 0.5:
            regime = "Anomalous Dispersion (D > 0)"
        elif D_val < -0.5:
            regime = "Normal Dispersion (D < 0)"
        else:
            regime = "Zero / Near-Zero Dispersion (D ~ 0)"

        logger.info(
            f"[Dispersion Engine] W={self.width:.3f} μm, H={self.height:.3f} μm, Clad_H={self.top_cladding_height:.3f} μm | "
            f"D = {D_val:.2f} ps/(nm·km) | Regime: {regime} | Single-Mode: {is_singlemode}"
        )

        return {
            "dispersion_ps_nm_km": D_val,
            "dispersion_regime": regime,
            "is_singlemode": is_singlemode,
            "wavelength_um": self.wavelength_um,
            "neff_fundamental": float(srt_neff[mid_idx]),
        }

    def get_dispersion_vs_wavelength(
        self,
        wvl_list: list,
        output_dir: Path,
        num_points: int = 11,
        delta_wvl_um: float = 0.05,
        show: bool = True,
        *,
        plot_dpi: int = 300,  # Resolution of displayed and saved figures (dots per inch).
    ) -> Dict[str, Any]:
        """
        Evaluates chromatic dispersion across a spectrum by sweeping target central wavelengths,
        calling get_dispersion() at each step, and generating a D vs. lambda plot.
        """

        wvl_list = _values("wvl_list", wvl_list)

        num_points = _integer(
            "num_points",
            num_points,
            minimum=3,
        )

        delta_wvl_um = _positive("delta_wvl_um", delta_wvl_um)
        plot_dpi = int(_positive("plot_dpi",plot_dpi))

        if min(wvl_list) <= delta_wvl_um / 2:
            raise ValueError("delta_wvl_um must keep every sampled wavelength positive.")
        import os
        import matplotlib.pyplot as plt

        d_vals = []
        regimes = []
        singlemode_flags = []
        neff_vals = []

        # Store original center wavelength to restore state afterwards
        original_wvl = self.wavelength_um

        wvl_start = wvl_list[0]
        wvl_end = wvl_list[-1]

        logger.info(
            f"[Dispersion Sweep] Sweeping {len(wvl_list)} points from {wvl_start:.3f} μm "
            f"to {wvl_end:.3f} μm for W={self.width:.3f} μm, H={self.height:.3f} μm, Clad_H={self.top_cladding_height:.3f} μm..."
        )

        try:
            for wvl in wvl_list:
                self.wavelength_um = float(wvl)
                res = self.get_dispersion(
                    delta_wvl_um=delta_wvl_um,
                    num_points=7,
                )

                d_vals.append(res["dispersion_ps_nm_km"])
                regimes.append(res["dispersion_regime"])
                singlemode_flags.append(res["is_singlemode"])
                neff_vals.append(res["neff_fundamental"])

        finally:
            # Always restore state
            self.wavelength_um = original_wvl

        d_vals = np.array(d_vals)

        # Build Plot
        fig, ax = plt.subplots(
            figsize=(8, 5),
            dpi=plot_dpi,
        )

        # Highlight Normal vs Anomalous regimes
        ax.axhline(
            0,
            color="gray",
            linestyle="--",
            linewidth=1.2,
            label="Zero Dispersion (ZDW)",
        )
        ax.plot(
            wvl_list,
            d_vals,
            color="#2b5c8f",
            marker="o",
            markersize=5,
            linewidth=2,
            label=f"TE00 Mode (W={self.width:.2f}μm, H={self.height:.2f}μm)",
        )

        # Visual shading for dispersion regimes
        ymin, ymax = ax.get_ylim()
        if ymax > 0:
            ax.fill_between(
                wvl_list,
                0,
                ymax,
                where=(d_vals > 0),
                color="red",
                alpha=0.08,
                label="Anomalous (D > 0)",
            )
        if ymin < 0:
            ax.fill_between(
                wvl_list,
                ymin,
                0,
                where=(d_vals < 0),
                color="blue",
                alpha=0.08,
                label="Normal (D < 0)",
            )

        ax.set_xlabel(
            "Wavelength $\\lambda$ (μm)",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_ylabel(
            "Dispersion $D$ [ps / (nm · km)]",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_title(
            f"Chromatic Dispersion Spectrum\n"
            f"SiN Waveguide ($W = {self.width:.3f}\\,\\mu\\mathrm{{m}}$, $H = {self.height:.3f}\\,\\mu\\mathrm{{m}}$, $H_{{clad}} = {self.top_cladding_height:.3f}\\,\\mu\\mathrm{{m}}$)",
            fontsize=12,
        )
        ax.grid(
            True,
            linestyle=":",
            alpha=0.6,
        )
        ax.legend(
            loc="best",
            frameon=True,
        )
        fig.tight_layout()

        # Handle Output Directory
        plot_path = None
        if output_dir:
            os.makedirs(
                output_dir,
                exist_ok=True,
            )
            filename = f"dispersion_W{self.width:.3f}_H{self.height:.3f}_Clad{self.top_cladding_height:.3f}_{wvl_start:.2f}-{wvl_end:.2f}um.png"
            plot_path = os.path.join(
                output_dir,
                filename,
            )
            fig.savefig(
                plot_path,
                dpi=plot_dpi,
            )
            logger.info(f"Plot successfully saved to: {plot_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)

        return {
            "wavelengths_um": (wvl_start, wvl_end),
            "dispersion_ps_nm_km": d_vals,
            "neff_array": np.array(neff_vals),
            "regimes": regimes,
            "is_singlemode_array": singlemode_flags,
            "plot_path": plot_path,
        }

    def get_confinement(
        self,
        sim_size_y: Optional[float] = None,
        sim_size_z: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculate the spatial confinement factor and effective mode area.
        for the fundamental TE00 mode on this waveguide geometry.
        """
        freq0 = td.C_0 / self.wavelength_um
        n_core_guess = float(np.real(np.sqrt(self.core_medium.eps_model(freq0))))

        # Determine evaluation cross-section domain
        plane_size_y = sim_size_y if sim_size_y is not None else float(self.sim.size[1])
        plane_size_z = sim_size_z if sim_size_z is not None else float(self.sim.size[2])

        mode_spec = td.ModeSpec(
            num_modes=6,
            target_neff=n_core_guess,
        )
        plane = td.Box(
            center=(0, 0, self.height / 2),
            size=(0, plane_size_y, plane_size_z),
        )

        solver = ModeSolver(
            simulation=self.sim,
            plane=plane,
            mode_spec=mode_spec,
            freqs=[freq0],
        )
        try:
            data = solver.solve()
        except Exception as e:
            raise WaveguideModeError(f"Confinement ModeSolver failed: {e}") from e

        te00_idx = -1
        for m in range(len(data.n_eff.mode_index)):
            if (
                self._get_te_fraction(
                    data,
                    m_idx=m,
                    f_idx=0,
                )
                > 0.5
            ):
                te00_idx = m
                break

        if te00_idx == -1:
            raise WaveguideModeError(
                f"TE00 mode not found for W={self.width:.3f} μm, H={self.height:.3f} μm."
            )

        Ex = data.Ex.isel(
            mode_index=te00_idx,
            f=0,
        ).values.squeeze()
        Ey = data.Ey.isel(
            mode_index=te00_idx,
            f=0,
        ).values.squeeze()
        Ez = data.Ez.isel(
            mode_index=te00_idx,
            f=0,
        ).values.squeeze()

        intensity = np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2

        y_vals = data.Ex.y.values
        z_vals = data.Ex.z.values
        Y, Z = np.meshgrid(
            y_vals,
            z_vals,
            indexing="ij",
        )

        core_mask = (np.abs(Y) <= self.width / 2) & (Z >= 0) & (Z <= self.height)

        gamma = float(np.sum(intensity[core_mask]) / np.sum(intensity))

        dy = float(y_vals[1] - y_vals[0])
        dz = float(z_vals[1] - z_vals[0])
        dA = dy * dz

        numerator = (np.sum(intensity) * dA) ** 2
        denominator = np.sum(intensity**2) * dA
        A_eff = float(numerator / denominator)

        logger.info(
            f"[Confinement Engine] W={self.width:.3f} μm, H={self.height:.3f} μm | "
            f"Γ = {gamma*100:.2f}% | A_eff = {A_eff:.3f} μm² | Evaluation Domain: {plane_size_y:.2f}x{plane_size_z:.2f} μm²"
        )

        return {
            "confinement_factor": gamma,
            "confinement_percent": gamma * 100.0,
            "effective_mode_area_um2": A_eff,
            "physical_core_area_um2": self.width * self.height,
            "eval_domain_um2": plane_size_y * plane_size_z,
        }
