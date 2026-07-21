"""
waveguide.waveguide

Defines the RidgeWavguide Class

Author: Esraaj Sarkar Gupta
Date: July, 2026
"""

from __future__ import annotations
import tidy3d as td

class RidgeWaveguide:
    def __init__(
            self,
            sim : td.Simulation,

            width: float,
            height: float,

            length: float,

            core_medium: td.Medium,
            sub_medium: td.Medium,

            wavelength_um: float,

            fine_min_steps_per_wvl: int = 20,
            coarse_min_steps_per_wvl : int = 10,

            margin_y: float = 2.0,  # Buffer space on each side of the core in Y
            margin_z: float = 2.0,  # Buffer space above/below the core in Z

            run_time : float = 1e-12
    ):
        self.sim = sim
        
        self.width = width
        self.height = height

        self.length = length

        self.core_medium = core_medium
        self.sub_medium = sub_medium

        self.wavelength_um = wavelength_um

        self.fine_min_steps_per_wvl = fine_min_steps_per_wvl
        self.coarse_min_steps_per_wvl = coarse_min_steps_per_wvl

        self.margin_y = margin_y
        self.margin_z = margin_z

        self.run_time = run_time

    def __getattr__(self, item):
        return getattr(self.sim, item)
