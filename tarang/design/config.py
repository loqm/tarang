"""Settings of one design run, read from and written to a TOML file.

Every value has a default chosen from the wavelength, so a first run can accept the
defaults; the file written at the end (``design.toml``) repeats the run exactly. Values
left out of a file are asked for (interactive) or take their default (``--yes``).
"""
from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional


def _f(default=None, help: str = "", unit: str = ""):
    """Dataclass field with a description used for prompts and file comments."""
    if isinstance(default, (list, dict)):
        return field(default_factory=lambda d=default: type(d)(d), metadata={"help": help, "unit": unit})
    return field(default=default, metadata={"help": help, "unit": unit})


@dataclass
class Run:
    name: str = _f("design", "Name of this design (used for file names)")
    wavelength_um: float = _f(1.55, "Centre (pump) wavelength", "um")
    band_um: Optional[float] = _f(None, "Half-width of the band the materials are fitted over "
                                        "(default: 0.13 x wavelength)", "um")
    max_cost_per_run: float = _f(1.0, "Largest FlexCredit cost allowed for any single cloud run",
                                 "FlexCredits")
    max_total_cost: float = _f(10.0, "Budget for all cloud stages together", "FlexCredits")
    workers: int = _f(2, "Parallel processes for local mode solving (about 1 GB RAM each)")


@dataclass
class Materials:
    core: str = _f("si3n4", "Core material: a key in tarang/media/media.json or a path to your own "
                            "CSV (columns wl [um], n, k)")
    box: str = _f("sio2", "Buried oxide under the core (key or CSV path)")
    cladding: str = _f("air", "Top cladding (key or CSV path; air for an air-clad device)")
    handle: str = _f("si_franta", "Silicon wafer under the box, used for the grating coupler "
                                  "('none' to leave it out)")


@dataclass
class Stack:
    core_height_um: Optional[float] = _f(None, "Core film thickness if already fixed (e.g. the wafer "
                                               "you ordered); leave empty to map heights", "um")
    box_height_um: float = _f(4.0, "Buried oxide thickness", "um")
    cladding_height_um: float = _f(2.0, "Top cladding thickness (ignored for air)", "um")


@dataclass
class CrossSection:
    heights_um: list = _f([], "Core heights to map (default: around the fixed film, or "
                              "0.32-0.58 x wavelength)", "um")
    widths_um: list = _f([], "Core widths to map (default: 0.39-1.16 x wavelength)", "um")
    map_steps: int = _f(50, "Mesh steps per wavelength for the map (>= 50)")
    decision_steps: int = _f(70, "Mesh steps per wavelength for the chosen geometry (>= 70)")
    tolerance_width_um: float = _f(0.05, "Fabrication tolerance of the width (+/-)", "um")
    tolerance_height_um: float = _f(0.02, "Fabrication tolerance of the film thickness (+/-)", "um")
    height_um: Optional[float] = _f(None, "Chosen core height (picked from the map)", "um")
    width_um: Optional[float] = _f(None, "Chosen core width (picked from the map)", "um")


@dataclass
class Ring:
    fsr_ghz: float = _f(200.0, "Target free spectral range", "GHz")
    radius_um: Optional[float] = _f(None, "Ring radius (default: from the FSR target and the group "
                                          "index)", "um")
    scattering_loss_db_cm: float = _f(0.3, "Straight-waveguide scattering loss of your process "
                                           "(measured or cited)", "dB/cm")
    absorption_loss_db_cm: float = _f(0.1, "Material absorption loss", "dB/cm")
    bend_check: bool = _f(True, "Solve the bent waveguide locally to include bend loss (free)")


@dataclass
class Gap:
    gaps_um: list = _f([], "Ring-bus gaps to simulate (default: 0.19-0.52 x wavelength)", "um")
    span_angle_deg: float = _f(45.0, "Span of the partial ring simulated next to the bus", "deg")
    over_coupling: float = _f(1.5, "Also report the gap for this over-coupling ratio "
                                   "K = Q_int / Q_coupling (1 = critical)")
    fine_steps: int = _f(15, "FDTD mesh steps per wavelength in the waveguide")
    coarse_steps: int = _f(10, "FDTD mesh steps per wavelength elsewhere")


@dataclass
class Grating:
    etch: str = _f("full", "Grating etch: 'full' (teeth etched through the film, one lithography "
                           "step) or 'partial'")
    etch_depth_um: Optional[float] = _f(None, "Etch depth for a partial etch", "um")
    fiber_angle_deg: float = _f(10.0, "Fibre angle from vertical", "deg")
    mfd_um: float = _f(10.4, "Fibre mode-field diameter", "um")
    n_teeth: int = _f(20, "Number of grating periods")
    width_um: float = _f(12.0, "Grating width (transverse)", "um")
    periods_um: list = _f([], "Periods to try (default: Bragg estimate x 0.94-1.06)", "um")
    fill_factors: list = _f([0.4, 0.5, 0.6, 0.7], "Fill factors (tooth / period) to try")
    source_x_um: list = _f([], "Fibre positions along the grating to try (default: 0.3-0.7 x "
                               "MFD)", "um")
    check_3d: bool = _f(False, "Also simulate the best design in 3D (several times the 2D cost)")
    taper_length_um: float = _f(300.0, "Linear taper from the grating width to the waveguide", "um")


@dataclass
class Layout:
    bus_length_um: float = _f(400.0, "Straight bus length between the tapers", "um")
    tone: str = _f("structure", "'structure': polygons are the core; 'trench': polygons are the "
                                "etched trenches around it (positive resist)")
    trench_width_um: float = _f(3.0, "Trench width for tone = 'trench'", "um")
    waveguide_layer: list = _f([1, 0], "GDS [layer, datatype] of bus waveguides")
    ring_layer: list = _f([1, 0], "GDS [layer, datatype] of the ring")
    taper_layer: list = _f([1, 0], "GDS [layer, datatype] of the grating tapers")
    grating_layer: list = _f([1, 0], "GDS [layer, datatype] of the grating teeth")


SECTIONS = {"run": Run, "materials": Materials, "stack": Stack, "cross_section": CrossSection,
            "ring": Ring, "gap": Gap, "grating": Grating, "layout": Layout}


@dataclass
class DesignConfig:
    run: Run = field(default_factory=Run)
    materials: Materials = field(default_factory=Materials)
    stack: Stack = field(default_factory=Stack)
    cross_section: CrossSection = field(default_factory=CrossSection)
    ring: Ring = field(default_factory=Ring)
    gap: Gap = field(default_factory=Gap)
    grating: Grating = field(default_factory=Grating)
    layout: Layout = field(default_factory=Layout)
    given: set = field(default_factory=set, repr=False, compare=False)   # "section.key" read from a file

    # ---- file I/O ------------------------------------------------------------------- #
    @classmethod
    def load(cls, path: Path | str) -> "DesignConfig":
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        cfg = cls()
        for sec_name, sec_cls in SECTIONS.items():
            values = data.pop(sec_name, {})
            known = {f.name: f for f in fields(sec_cls)}
            unknown = set(values) - set(known)
            if unknown:
                raise ValueError(f"[{sec_name}] has unknown setting(s) {sorted(unknown)} in {path}")
            sec = getattr(cfg, sec_name)
            for key, value in values.items():
                setattr(sec, key, value)
                cfg.given.add(f"{sec_name}.{key}")
        if data:
            raise ValueError(f"Unknown section(s) {sorted(data)} in {path}")
        return cfg

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        lines = ["# TARANG design settings. Rerun exactly with:",
                 f"#     python -m tarang.design --config {path.name}",
                 "# Delete a line to be asked for it again; values without a line take their default.",
                 ""]
        for sec_name in SECTIONS:
            sec = getattr(self, sec_name)
            lines.append(f"[{sec_name}]")
            for f in fields(sec):
                value = getattr(sec, f.name)
                unit = f" [{f.metadata['unit']}]" if f.metadata.get("unit") else ""
                lines.append(f"# {f.metadata['help']}{unit}")
                if value is None or value == []:
                    lines.append(f"# {f.name} =")
                else:
                    lines.append(f"{f.name} = {_toml_value(value)}")
            lines.append("")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def section_dict(self, *names: str) -> dict[str, Any]:
        return {n: dataclasses.asdict(getattr(self, n)) for n in names}


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(float(v)) if isinstance(v, float) else str(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(f"Cannot write {v!r} to TOML")
