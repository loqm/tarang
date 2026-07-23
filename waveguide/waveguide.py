"""
waveguide.waveguide

Defines the RidgeWaveguide Class with self-contained modal analysis,
dispersion regime evaluation, and spatial confinement extraction.

Author: Esraaj Sarkar Gupta
Date: July, 2026
"""

from __future__ import annotations
import logging
from typing import Dict, Any

import numpy as np
import tidy3d as td
from tidy3d.plugins.mode import ModeSolver

logger = logging.getLogger(__name__)

# Speed of light in um/ps (from Tidy3D native um/s constant)
C_UM_PS = td.C_0 / 1e12  # Result: ~299.792458


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

            substrate_height: float = 4.0,

            fine_min_steps_per_wvl: int = 20,
            coarse_min_steps_per_wvl: int = 10,

            margin_y: float = 2.0,  # Buffer space on each side of the core in Y
            margin_z: float = 2.0,  # Air buffer space above the core in Z

            run_time: float = 1e-12
    ):
        self.sim = sim
        
        self.width = width
        self.height = height

        self.length = length

        self.core_medium = core_medium
        self.sub_medium = sub_medium

        self.wavelength_um = wavelength_um
        self.substrate_height = substrate_height

        self.fine_min_steps_per_wvl = fine_min_steps_per_wvl
        self.coarse_min_steps_per_wvl = coarse_min_steps_per_wvl

        self.margin_y = margin_y
        self.margin_z = margin_z

        self.run_time = run_time

    def __getattr__(self, item):
        return getattr(self.sim, item)

    def _get_te_fraction(self, mode_data: Any, m_idx: int, f_idx: int = 0) -> float:
        """Computes TE polarization fraction for a waveguide propagating along X."""
        Ey = mode_data.Ey.isel(mode_index=m_idx, f=f_idx).values
        Ez = mode_data.Ez.isel(mode_index=m_idx, f=f_idx).values
        Iy = np.sum(np.abs(Ey)**2)
        Iz = np.sum(np.abs(Ez)**2)
        return float(Iy / (Iy + Iz + 1e-10))

    def get_modes(self, num_modes: int = 8) -> Dict[str, Any]:
        """
        Solves for the optical modes of this waveguide instance.
        
        Returns effective indices ($n_{\mathrm{eff}}$), polarization fractions, 
        TE/TM classifications, and guiding conditions.
        """
        freq0 = td.C_0 / self.wavelength_um
        n_core = float(np.real(np.sqrt(self.core_medium.eps_model(freq0))))
        n_sub = float(np.real(np.sqrt(self.sub_medium.eps_model(freq0))))

        mode_plane = td.Box(
            center=(0, 0, self.height / 2),
            size=(0, self.width + 2.0, self.height + 2.0)
        )

        mode_spec = td.ModeSpec(num_modes=num_modes, target_neff=n_core)
        solver = ModeSolver(
            simulation=self.sim,
            plane=mode_plane,
            mode_spec=mode_spec,
            freqs=[freq0]
        )

        logger.info(f"[Mode Engine] Solving top {num_modes} modes for W={self.width:.3f} μm, H={self.height:.3f} μm...")
        mode_data = solver.solve()

        modes_info = []
        for i in range(num_modes):
            n_eff = float(np.real(mode_data.n_eff[0, i]))
            te_frac = self._get_te_fraction(mode_data, m_idx=i, f_idx=0)
            pol_label = "TE" if te_frac > 0.5 else "TM"
            is_guided = n_eff > n_sub

            modes_info.append({
                "mode_index": i,
                "neff": n_eff,
                "polarization": pol_label,
                "te_fraction": te_frac,
                "is_guided": is_guided
            })

        return {
            "width_um": self.width,
            "height_um": self.height,
            "wavelength_um": self.wavelength_um,
            "n_core": n_core,
            "n_sub": n_sub,
            "modes": modes_info,
            "raw_mode_data": mode_data
        }

    def get_dispersion(self, delta_wvl_um: float = 0.1, num_points: int = 11) -> Dict[str, Any]:
        """
        Evaluates Chromatic Dispersion ($D$) and determines the dispersion regime.
        
        Regimes:
            - "Anomalous Dispersion (D > 0)"
            - "Normal Dispersion (D < 0)"
            - "Zero / Near-Zero Dispersion (D ~ 0)"
        """
        freq0 = td.C_0 / self.wavelength_um
        n_core_guess = float(np.real(np.sqrt(self.core_medium.eps_model(freq0))))
        n_sub_guess = float(np.real(np.sqrt(self.sub_medium.eps_model(freq0))))

        mid_idx = num_points // 2
        wavelengths = np.linspace(
            self.wavelength_um - delta_wvl_um / 2,
            self.wavelength_um + delta_wvl_um / 2,
            num_points
        )
        freqs = td.C_0 / wavelengths

        mode_spec = td.ModeSpec(num_modes=4, target_neff=n_core_guess)
        plane = td.Box(
            center=(0, 0, self.height / 2),
            size=(0, self.sim.size[1], self.sim.size[2])
        )

        solver = ModeSolver(simulation=self.sim, plane=plane, mode_spec=mode_spec, freqs=freqs)
        data = solver.solve()

        # Isolate fundamental TE00 mode
        te00_idx = -1
        for m in range(len(data.n_eff.mode_index)):
            if self._get_te_fraction(data, m_idx=m, f_idx=mid_idx) > 0.5:
                te00_idx = m
                break

        if te00_idx == -1:
            logger.warning(f"[Dispersion Engine] TE00 mode not found for W={self.width:.3f} μm, H={self.height:.3f} μm.")
            return {"error": "TE00 mode not found"}

        # Single-mode condition check
        neff_at_center = np.real(data.n_eff.isel(f=mid_idx).values)
        n_higher = neff_at_center[2] if len(neff_at_center) > 2 else 0.0
        is_singlemode = bool(n_higher <= (n_sub_guess + 0.001))

        # Numerical second-derivative across wavelength array
        neff_vals = np.real(data.n_eff.isel(mode_index=te00_idx).values)
        sorted_idx = np.argsort(wavelengths)
        srt_wvl = wavelengths[sorted_idx]
        srt_neff = neff_vals[sorted_idx]

        dn_dlam = np.gradient(srt_neff, srt_wvl)
        d2n_dlam2 = np.gradient(dn_dlam, srt_wvl)

        # D = - (lambda / c) * d2n/dlambda2 * 1e4 -> [ps / (nm * km)]
        D_val = float(-(srt_wvl[mid_idx] / C_UM_PS) * d2n_dlam2[mid_idx] * 1e4)

        # Classify the dispersion regime
        if D_val > 0.5:
            regime = "Anomalous Dispersion (D > 0)"
        elif D_val < -0.5:
            regime = "Normal Dispersion (D < 0)"
        else:
            regime = "Zero / Near-Zero Dispersion (D ~ 0)"

        logger.info(
            f"[Dispersion Engine] W={self.width:.3f} μm, H={self.height:.3f} μm | "
            f"D = {D_val:.2f} ps/(nm·km) | Regime: {regime} | Single-Mode: {is_singlemode}"
        )

        return {
            "dispersion_ps_nm_km": D_val,
            "dispersion_regime": regime,
            "is_singlemode": is_singlemode,
            "wavelength_um": self.wavelength_um,
            "neff_fundamental": float(srt_neff[mid_idx])
        }

    def get_confinement(self) -> Dict[str, Any]:
        """
        Calculates the spatial confinement factor ($\Gamma$) and effective mode area ($A_{\mathrm{eff}}$)
        for the fundamental TE00 mode on this waveguide geometry.
        """
        freq0 = td.C_0 / self.wavelength_um
        n_core_guess = float(np.real(np.sqrt(self.core_medium.eps_model(freq0))))

        mode_spec = td.ModeSpec(num_modes=6, target_neff=n_core_guess)
        plane = td.Box(
            center=(0, 0, self.height / 2),
            size=(0, self.sim.size[1], self.sim.size[2])
        )

        solver = ModeSolver(simulation=self.sim, plane=plane, mode_spec=mode_spec, freqs=[freq0])
        data = solver.solve()

        te00_idx = -1
        for m in range(len(data.n_eff.mode_index)):
            if self._get_te_fraction(data, m_idx=m, f_idx=0) > 0.5:
                te00_idx = m
                break

        if te00_idx == -1:
            logger.warning(f"[Confinement Engine] TE00 mode not found for W={self.width:.3f} μm, H={self.height:.3f} μm.")
            return {"error": "TE00 mode not found"}

        Ex = data.Ex.isel(mode_index=te00_idx, f=0).values.squeeze()
        Ey = data.Ey.isel(mode_index=te00_idx, f=0).values.squeeze()
        Ez = data.Ez.isel(mode_index=te00_idx, f=0).values.squeeze()

        intensity = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2

        y_vals = data.Ex.y.values
        z_vals = data.Ex.z.values
        Y, Z = np.meshgrid(y_vals, z_vals, indexing='ij')

        core_mask = (np.abs(Y) <= self.width / 2) & (Z >= 0) & (Z <= self.height)

        gamma = float(np.sum(intensity[core_mask]) / np.sum(intensity))

        dy = float(y_vals[1] - y_vals[0])
        dz = float(z_vals[1] - z_vals[0])
        dA = dy * dz

        numerator = (np.sum(intensity) * dA)**2
        denominator = np.sum(intensity**2) * dA
        A_eff = float(numerator / denominator)

        logger.info(
            f"[Confinement Engine] W={self.width:.3f} μm, H={self.height:.3f} μm | "
            f"Γ = {gamma*100:.2f}% | A_eff = {A_eff:.3f} μm²"
        )

        return {
            "confinement_factor": gamma,
            "confinement_percent": gamma * 100.0,
            "effective_mode_area_um2": A_eff,
            "physical_core_area_um2": self.width * self.height
        }