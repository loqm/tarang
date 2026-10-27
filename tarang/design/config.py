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
    goal: str = _f("comb_ring", "What to design: 'comb_ring' (ring resonator for frequency combs, "
                                "with its bus and couplers) or 'loss_test' (test waveguides for "
                                "propagation loss: cutback spirals with couplers)")
    coupling: str = _f("grating", "How light gets on and off the chip: 'grating' (grating "
                                  "couplers, fibre from above) or 'edge' (edge couplers at the "
                                  "chip facets, lensed fibre in the chip plane)")
    wavelength_um: float = _f(1.55, "Design wavelength (the pump)", "um")
    band_um: Optional[float] = _f(None, "Material-fit band half-width (default: 0.13 x "
                                        "wavelength)", "um")
    max_cost_per_run: float = _f(1.0, "Largest cost of any single cloud run",
                                 "FlexCredits")
    max_total_cost: float = _f(10.0, "Budget for all cloud stages together", "FlexCredits")
    workers: int = _f(2, "Parallel processes for local mode solving (about 1.5 GB RAM each)")
    figure_formats: list = _f(["png", "pdf"], "Figure files to write: png (raster), pdf and/or svg "
                                              "(vector, for papers and slides)")
    figure_dpi: int = _f(300, "Resolution of png figures", "dpi")
    figure_style: str = _f("screen", "Figure look: 'screen' (as drawn, for the GUI and "
                                     "slides) or 'paper' (8 pt text, thinner lines, 3.4 in "
                                     "wide for one column of panels, 7 in otherwise; written "
                                     "to figures_paper/)")
    figure_data: bool = _f(True, "Save each figure's plotted numbers (<name>_data.csv, maps "
                                 "in <name>_data.npz) and its labels, versions and settings "
                                 "(<name>_meta.json) next to it, to redraw it anywhere")
    sim_pictures: bool = _f(True, "Save a picture of the geometry of every cloud simulation "
                                  "(figures/sims/, named like its cloud folder)")


@dataclass
class Materials:
    core: str = _f("si3n4", "Core material: a key in tarang/media/media.json or a path to your own "
                            "CSV (columns wl [um], n, k)")
    box: str = _f("sio2", "Buried oxide under the core (key or CSV path)")
    cladding: str = _f("air", "Top cladding (key or CSV path; air for an air-clad device)")
    handle: str = _f("si_franta", "Silicon wafer under the BOX (grating coupler only; 'none' "
                                  "leaves it out)")


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
    mesh_ladder: list = _f([50, 60, 70, 80, 90], "Meshes (steps per wavelength) of the mesh "
                                                 "check of the chosen geometry: D is quoted as "
                                                 "their median +/- the largest distance from it")
    pump_mode: str = _f("TE00", "Mode the comb is pumped in (and test waveguides carry): 'TE00' "
                                "or 'TM00'. The map's targets, the check, the ring's dispersion "
                                "and the LLE inputs follow it. At short wavelengths TM00 is often "
                                "anomalous where TE00 is not (map 10)")
    snap_grid: bool = _f(True, "Put grid lines through the four core edges in every mode solve "
                               "(D varies about 4x less from mesh to mesh; false reproduces "
                               "designs made before v0.1.0)")
    tolerance_width_um: float = _f(0.05, "Fabrication tolerance of the width (+/-)", "um")
    tolerance_height_um: float = _f(0.02, "Fabrication tolerance of the film thickness (+/-)", "um")
    stop_after_map: bool = _f(False, "Stop after the map so a candidate can be chosen (the GUI "
                                     "uses this), then run the stage again")
    compare: list = _f([], "Candidates to compare before choosing, as \"HxW\" in um (e.g. "
                           "\"0.7x1.0\"): tolerance box, mode profiles, D over the band and a bend "
                           "each, side by side (local, free)")
    height_um: Optional[float] = _f(None, "Chosen core height (picked from the map)", "um")
    width_um: Optional[float] = _f(None, "Chosen core width (picked from the map)", "um")


@dataclass
class Ring:
    fsr_ghz: float = _f(200.0, "Target free spectral range", "GHz")
    radius_um: Optional[float] = _f(None, "Ring radius (default: from the FSR target and the group "
                                          "index)", "um")
    loss_preset: str = _f("typical_ebl", "Loss values to start from, each from a publication: "
                                         "typical_ebl, annealed_low_loss, state_of_the_art "
                                         "(see the assumptions table); your own values below "
                                         "replace them")
    scattering_loss_db_cm: Optional[float] = _f(None, "Straight-waveguide scattering loss "
                                                      "(default: from loss_preset)", "dB/cm")
    absorption_loss_db_cm: Optional[float] = _f(None, "Material absorption loss (default: from "
                                                      "loss_preset)", "dB/cm")
    bend_check: bool = _f(True, "Bend loss from a local bent-mode solve (free)")


@dataclass
class Gap:
    gaps_um: list = _f([], "Ring-bus gaps to simulate (default: 0.19-0.52 x wavelength)", "um")
    span_angle_deg: float = _f(45.0, "Span of the partial ring simulated next to the bus", "deg")
    over_coupling: float = _f(1.5, "Over-coupling ratio K = Q_int/Q_c for the second gap "
                                   "(1 = critical only)")
    reference: str = _f("ring_port", "How the coupled power is measured: 'ring_port' (power "
                                     "entering the ring arc; no extra run), 'bus_only' (through "
                                     "power relative to a run of the bus alone; one extra run) "
                                     "or 'source' (through power relative to the source; biased "
                                     "by numerical losses on the bus)")
    fine_steps: int = _f(15, "FDTD mesh steps per wavelength in the waveguide")
    coarse_steps: int = _f(10, "FDTD mesh steps per wavelength elsewhere")
    band_points: int = _f(11, "Wavelengths recorded at the ring ports (across +/-8 % of the "
                              "design frequency: coupling against wavelength; 1 = the design "
                              "wavelength only)")
    ring_inner_box_recess_um: float = _f(0.0, "BOX pocket inside the ring (air cut into the BOX "
                                              "by the partial-ring builder; 0 = flat BOX, 0.1 = "
                                              "runs made before Phase 12)", "um")


@dataclass
class Grating:
    etch: str = _f("full", "Grating etch: 'full' (teeth etched through the film, one lithography "
                           "step) or 'partial'")
    etch_depth_um: Optional[float] = _f(None, "Etch depth for a partial etch", "um")
    fiber_angle_deg: float = _f(10.0, "Fibre angle from vertical", "deg")
    fiber_tilt: str = _f("towards", "Which way the fibre beam travels along the grating: "
                                    "'towards' the waveguide (fibres on two holders form a V) or "
                                    "'away' from it")
    mfd_um: float = _f(10.4, "Fibre mode-field diameter", "um")
    n_teeth: int = _f(30, "Number of grating periods")
    width_um: float = _f(12.0, "Grating width (transverse)", "um")
    fill_factors: list = _f([], "Fill factors (tooth / period) to try (default: 0.5-0.9 for a "
                                "full etch, 0.4-0.8 for a partial etch)")
    detuning: list = _f([0.95, 1.0, 1.05], "Period multiples tried per fill factor (of its "
                                           "phase-matched period)")
    periods_um: list = _f([], "Fixed periods to try for every fill factor instead (default: "
                              "use detuning)", "um")
    source_x_um: list = _f([], "Fibre positions on the grating surface to try (default: five "
                               "positions across the grating)", "um")
    min_feature_um: float = _f(0.1, "Smallest tooth or groove your lithography can make", "um")
    bandwidth_um: float = _f(0.1, "Spectrum half-width around the wavelength", "um")
    retune: bool = _f(True, "Retune the period onto the wavelength (then refine fill factor "
                            "and fibre position)")
    centre_peak: bool = _f(True, "Centre the efficiency peak on the wavelength (False: keep the "
                                 "design with the highest efficiency at the wavelength)")
    check_3d: bool = _f(False, "Also simulate the best design in 3D (several times the 2D cost)")
    taper_length_um: float = _f(300.0, "Linear taper from the grating width to the waveguide", "um")


@dataclass
class Checks:
    band: bool = _f(True, "Coupling against wavelength (critical and over-coupled gaps; 2 "
                          "cloud runs, none if the gap runs recorded the band)")
    grating_extras: bool = _f(True, "Grating extras (back-reflection into the waveguide, fibre angle "
                                    "+/- tolerance, film thickness and tooth width +/- tolerance, "
                                    "and where the light goes; about 8 small 2D runs)")
    angle_tolerance_deg: float = _f(1.0, "Fibre angle tolerance of your holders (+/-)", "deg")
    litho_bias_um: float = _f(0.02, "Lithography width error of teeth and gaps (+/-)", "um")
    mesh_grating: bool = _f(True, "Grating mesh check (final grating again at 1.5x finer mesh; "
                                  "1 small 2D run)")
    mesh_gap: bool = _f(False, "Gap mesh check (critical gap again at 1.5x finer mesh; 1 3D "
                               "run, about 5x the cost of a gap run)")
    mode_families: bool = _f(True, "Ring mode families (other guided modes of the bent waveguide "
                                   "and how often they cross TE00; local, free)")


@dataclass
class Taper:
    grating_width_um: Optional[float] = _f(None, "Grating width to use (empty: the width that best "
                                                 "matches the fibre mode)", "um")
    widths_um: list = _f([], "Grating widths for the fibre-overlap curve (empty: 0.8-2.2 x MFD)",
                         "um")
    run_eme: bool = _f(True, "Also check the taper loss with Tidy3D 3D EME (one cloud run, priced "
                             "first; the free 2D model always runs)")
    lengths_um: list = _f([], "Taper lengths for the 3D EME (empty: 10-400 um)", "um")
    max_loss_db: float = _f(0.05, "Largest taper loss accepted when choosing the length", "dB")
    eme_cells: int = _f(120, "EME cells along the taper, graded (more = closer to a smooth "
                             "taper; 30 is not converged)")
    eme_modes: int = _f(20, "Modes per EME cell (even modes only; the wide end guides about 11)")
    use_in_layout: bool = _f(True, "Use this grating width and taper length in the layout")


@dataclass
class Apodise:
    run: bool = _f(False, "Also design an apodised grating (the uniform grating is always designed "
                          "and used unless this is on, checked in FDTD and better)")
    fill_factors: list = _f([], "Fill factors whose scattering strength is measured (empty: 0.5 "
                                "up to the largest the smallest feature allows)")
    verify: bool = _f(True, "Check the apodised design in 2D FDTD (5 small runs for the period "
                            "scale and fibre position, then 3 with the peak centred)")
    min_trench_um: Optional[float] = _f(None, "Narrowest trench between teeth that the full etch "
                                              "clears (empty: grating.min_feature_um)", "um")
    max_teeth: int = _f(60, "Most teeth of the apodised grating")
    prefer: str = _f("efficiency", "What 'better than the uniform grating' means: 'efficiency' "
                                   "(higher at the design wavelength) or 'bandwidth' (wider 1 dB "
                                   "band, at most 0.5 dB lower at the design wavelength)")
    ramp_check: bool = _f(False, "Also try the linear fill-factor ramp of Marchetti et al. (Sci. "
                                 "Rep. 7, 16670, 2017) as a second recipe (about 15 small 2D runs)")
    reflection_check: bool = _f(True, "Also simulate how much light the kept apodised grating "
                                      "reflects back into the waveguide (one small 2D run; the "
                                      "whole-device model uses it for the ripple)")
    use_in_layout: bool = _f(True, "Use the apodised grating in the layout when it beats the "
                                   "uniform one")


@dataclass
class Focus:
    run: bool = _f(False, "Also design a focusing grating (curved teeth focus the light into the "
                          "waveguide; a few-um slab replaces the long taper)")
    teeth: str = _f("best", "Teeth along the axis: 'uniform', 'apodised', or 'best' (apodised when "
                            "it was checked and beats the uniform grating)")
    tip_widths_um: list = _f([], "Waveguide tip widths to try at the focus (empty: 1-4 x the "
                                 "waveguide width)", "um")
    max_half_angle_deg: float = _f(20.0, "Largest half-angle of the light cone at the tip (wider "
                                         "cones are less well described by the model)", "deg")
    check_3d: bool = _f(False, "Check the focusing grating in 3D FDTD (one run, priced first; "
                               "several FlexCredits)")
    mesh_steps: int = _f(12, "Mesh cells per wavelength in the core for the 3D check (12: about "
                             "1 FC; 15 matches the 2D design, about twice the cost)")
    use_in_layout: bool = _f(True, "Make the focusing grating the main coupler (loss budget, "
                                   "layout figure); the GDS has both in any case")


@dataclass
class Edge:
    fiber_mfd_um: float = _f(3.5, "Mode-field diameter of the lensed fibre's spot (1/e^2 intensity "
                                  "diameter at the focus)", "um")
    mfd_tolerance_um: float = _f(0.5, "Spread of the spot size (+/-; datasheet tolerance)", "um")
    facet_widths_um: list = _f([], "Facet widths for the overlap map (empty: from the waveguide "
                                   "width to 2 x the MFD)", "um")
    facet_width_um: Optional[float] = _f(None, "Facet width used (empty: the recommended one)", "um")
    profiles: list = _f(["linear", "parabolic", "power"], "Taper shapes to compare: 'linear', "
                                                          "'parabolic' (width^2 grows linearly) and "
                                                          "'power' (width grows as s^m)")
    power_exponent: float = _f(0.6, "Exponent m of the 'power' taper, w = w_wg + (w_facet - w_wg) "
                                    "s^m with s = 0 at the waveguide (m < 1: fast near the "
                                    "waveguide, slow near the facet)")
    profile: Optional[str] = _f(None, "Taper shape used (empty: the one that reaches the accepted "
                                      "loss in the shortest length)")
    lengths_um: list = _f([], "Taper lengths for the 3D EME (empty: 5-300 um)", "um")
    taper_length_um: Optional[float] = _f(None, "Taper length used (empty: the shortest below the "
                                                "accepted loss)", "um")
    max_loss_db: float = _f(0.05, "Largest taper loss accepted when choosing the length", "dB")
    run_eme: bool = _f(True, "Check the taper loss with Tidy3D 3D EME (one cloud run per shape, "
                             "priced first; the free 2D model always runs)")
    eme_cells: int = _f(80, "EME cells along the taper, graded")
    eme_modes: int = _f(12, "Modes per EME cell (even modes only)")
    check_3d: bool = _f(True, "Simulate the facet in 3D FDTD (one small run, priced first): "
                              "coupling for any fibre offset, spot size and distance, and the facet "
                              "reflection")
    mesh_steps: int = _f(15, "Mesh cells per wavelength in the core for the 3D facet run")
    offset_um: float = _f(1.0, "Fibre offsets for the alignment tolerance (+/- in y and z)", "um")
    facet_straight_um: float = _f(20.0, "Straight section at the facet width between the chip "
                                        "edge and the taper (room for the dicing and polishing "
                                        "position error)", "um")
    facet_overhang_um: float = _f(10.0, "How far the waveguide continues past the chip-edge line "
                                        "into the dicing lane", "um")
    chip_length_um: float = _f(5000.0, "Chip length along the waveguides (facet to facet)", "um")
    chip_width_um: float = _f(5000.0, "Chip width across the waveguides", "um")
    dicing_lane_um: float = _f(100.0, "Width of the dicing lane drawn outside each facet", "um")
    pitch_um: float = _f(127.0, "Smallest spacing between neighbouring waveguides at a facet",
                         "um")
    chip_layer: list = _f([99, 0], "GDS [layer, datatype] of the chip outline (not etched)")
    lane_layer: list = _f([98, 0], "GDS [layer, datatype] of the dicing lanes (not etched)")


@dataclass
class Assumptions:
    n2_m2_w: float = _f(2.4e-19, "Kerr index of the core for the threshold estimate (Si3N4: "
                                 "Ikeda et al., Opt. Express 16, 12987 (2008))", "m^2/W")
    dn_dT_core_per_K: float = _f(2.45e-5, "Thermo-optic coefficient of the core (LPCVD Si3N4: "
                                          "Arbabi and Goddard, Opt. Lett. 38, 3878 (2013))", "1/K")
    dn_dT_box_per_K: float = _f(0.95e-5, "Thermo-optic coefficient of the BOX (SiOx: Arbabi and "
                                         "Goddard, Opt. Lett. 38, 3878 (2013))", "1/K")


@dataclass
class Layout:
    bus_length_um: float = _f(400.0, "Straight bus length between the tapers", "um")
    tone: str = _f("structure", "Polygon tone ('structure': polygons are the core; 'trench': "
                                "they are the etched trenches around it, positive resist)")
    trench_width_um: float = _f(3.0, "Trench width for tone = 'trench'", "um")
    waveguide_layer: list = _f([1, 0], "GDS [layer, datatype] of bus waveguides")
    ring_layer: list = _f([1, 0], "GDS [layer, datatype] of the ring")
    taper_layer: list = _f([1, 0], "GDS [layer, datatype] of the grating tapers")
    grating_layer: list = _f([1, 0], "GDS [layer, datatype] of the grating teeth")
    coupler: str = _f("auto", "Couplers drawn on the mask: 'auto' (the best one designed, with "
                              "the focusing grating beside it when designed), 'uniform', "
                              "'apodised', 'focusing', or 'all' (every kind designed, side by side)")
    reference: bool = _f(True, "Also draw a straight waveguide with the same couplers (no ring): "
                               "measuring it gives the coupler loss to divide out")
    gap_offsets_um: list = _f([], "Extra ring variants with the gap changed by these amounts "
                                  "(e.g. -0.05, 0.05: fabrication may shift the gap)", "um")


@dataclass
class Waveguides:
    lengths_cm: list = _f([0, 1, 2, 4], "Cutback lengths (0 = the straight reference; longer "
                                        "ones are drawn as spirals)", "cm")
    coupler_scatter_db: float = _f(0.3, "Scatter of the measured coupling from one structure to "
                                        "the next (fibre placement, cleanliness)", "dB")
    target_precision: float = _f(0.1, "Relative precision wanted on the measured loss (0.1 = 10 %)")
    bend_radius_um: Optional[float] = _f(None, "Smallest bend radius of the spirals (empty: the "
                                               "smallest whose bend loss is below a tenth of the "
                                               "expected propagation loss)", "um")
    bend_radii_um: list = _f([], "Bend radii to check (empty: 5-100 um, scaled with the "
                                 "wavelength)", "um")
    spiral_gap_um: float = _f(4.0, "Gap between neighbouring turns of a spiral", "um")


@dataclass
class Device:
    run: bool = _f(True, "Model the whole device (free, local): fibre-to-fibre transmission of "
                         "grating + taper + bus + ring + taper + grating from the stages' results")
    reflections: bool = _f(True, "Include the couplers' back-reflection (Fabry-Perot ripple "
                                 "between the two couplers: the gratings' reverse runs, or the "
                                 "facets of an edge-coupled chip)")
    zoom_fsr: float = _f(1.5, "Half-width of the close-up around the pump, in ring FSRs")


# Settings added after a run was made, with the value that reproduces that run exactly.
# If a run folder already has results for the stage and its design.toml lacks the setting,
# the legacy value is used (so saved simulations stay valid), and the stage's settings key
# leaves it out (so the stage is not repeated).
LEGACY = {"cross_section": {"snap_grid": False, "pump_mode": "TE00"},
          "gap": {"ring_inner_box_recess_um": 0.1, "band_points": 1},
          "apodise": {"prefer": "efficiency", "ramp_check": False},
          "focus": {"mesh_steps": 12}}

SECTIONS = {"run": Run, "materials": Materials, "stack": Stack, "cross_section": CrossSection,
            "ring": Ring, "gap": Gap, "grating": Grating, "checks": Checks, "taper": Taper,
            "apodise": Apodise, "focus": Focus, "edge": Edge, "waveguides": Waveguides,
            "layout": Layout, "device": Device, "assumptions": Assumptions}


@dataclass
class DesignConfig:
    run: Run = field(default_factory=Run)
    materials: Materials = field(default_factory=Materials)
    stack: Stack = field(default_factory=Stack)
    cross_section: CrossSection = field(default_factory=CrossSection)
    ring: Ring = field(default_factory=Ring)
    gap: Gap = field(default_factory=Gap)
    grating: Grating = field(default_factory=Grating)
    checks: Checks = field(default_factory=Checks)
    taper: Taper = field(default_factory=Taper)
    apodise: Apodise = field(default_factory=Apodise)
    focus: Focus = field(default_factory=Focus)
    edge: Edge = field(default_factory=Edge)
    waveguides: Waveguides = field(default_factory=Waveguides)
    layout: Layout = field(default_factory=Layout)
    device: Device = field(default_factory=Device)
    assumptions: Assumptions = field(default_factory=Assumptions)
    given: set = field(default_factory=set, repr=False, compare=False)   # "section.key" read from a file

    # ---- file I/O ------------------------------------------------------------------- #
    @classmethod
    def load(cls, path: Path | str) -> "DesignConfig":
        # utf-8-sig: Windows PowerShell 5 and some editors start UTF-8 files with a
        # byte-order mark, which the TOML reader would reject.
        data = tomllib.loads(Path(path).read_text(encoding="utf-8-sig"))
        cfg = cls()
        for sec_name, sec_cls in SECTIONS.items():
            values = data.pop(sec_name, {})
            known = {f.name: f for f in fields(sec_cls)}
            unknown = set(values) - set(known)
            if unknown:
                raise SettingsError(f"[{sec_name}] has unknown setting(s) {sorted(unknown)} in {path}")
            sec = getattr(cfg, sec_name)
            hints = _hints(sec_cls)
            for key, value in values.items():
                value = _typed(f"[{sec_name}] {key}", value, hints.get(key), path)
                setattr(sec, key, value)
                cfg.given.add(f"{sec_name}.{key}")
        if data:
            raise SettingsError(f"Unknown section(s) {sorted(data)} in {path}")
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


def _hints(sec_cls) -> dict:
    import typing
    return typing.get_type_hints(sec_cls)


def _typed(where: str, value, hint, path):
    """``value`` as the type its setting expects; a SettingsError naming the setting when it
    cannot be (a word where a number belongs, a number where a list belongs)."""
    import typing
    if hint is None:
        return value
    args = typing.get_args(hint)
    if typing.get_origin(hint) is typing.Union:          # Optional[X]
        hint = next(a for a in args if a is not type(None))
    want = {float: "a number", int: "a whole number", bool: "true or false",
            str: "text in quotes", list: "a list in [ ]"}.get(hint)
    if want is None:
        return value
    ok = value
    if hint is float:
        ok = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    elif hint is int:
        if isinstance(value, int) and not isinstance(value, bool):
            ok = value
        elif isinstance(value, float) and value.is_integer():
            ok = int(value)
        else:
            ok = None
    elif not isinstance(value, hint):
        ok = None
    if ok is None:
        raise SettingsError(f"{where} = {value!r} in {path}: expected {want}.")
    return ok


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


# ---- plausible values ---------------------------------------------------------------- #
class SettingsError(ValueError):
    """A setting that cannot be right (shown as a message, not a traceback)."""


#: lengths (µm) a design cannot sensibly go beyond; a value outside is nearly always a
#: unit slip (nanometres typed into a micrometre setting)
RANGES_UM: dict[str, tuple[float, float, str]] = {
    "stack.core_height_um": (0.02, 5.0, "Film thickness"),
    "stack.box_height_um": (0.1, 20.0, "BOX thickness"),
    "cross_section.height_um": (0.02, 5.0, "Core height"),
    "cross_section.width_um": (0.05, 20.0, "Core width"),
    "cross_section.heights_um": (0.02, 5.0, "Core heights to map"),
    "cross_section.widths_um": (0.05, 20.0, "Core widths to map"),
    "edge.fiber_mfd_um": (0.5, 20.0, "Lensed-fibre spot (MFD)"),
    "grating.mfd_um": (1.0, 30.0, "Fibre mode-field diameter"),
}


def range_problem(dotted: str, value) -> str:
    """'' if ``value`` (a number, a list of numbers or None) is plausible for the setting,
    otherwise why not, with the micrometre value meant if nanometres were typed."""
    if dotted not in RANGES_UM or value is None or value == []:
        return ""
    lo, hi, what = RANGES_UM[dotted]
    for v in (value if isinstance(value, (list, tuple)) else [value]):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return f"{what}: {v!r} is not a number in µm"
        if not lo <= v <= hi:
            hint = (f"; did you mean {v / 1000:g} µm ({v:g} nm)?" if v > hi and lo <= v / 1000 <= hi
                    else "")
            return f"{what} {v:g} µm is outside {lo:g}–{hi:g} µm (the setting is in µm){hint}"
    return ""


def implausible(cfg: "DesignConfig") -> dict[str, str]:
    """{"section.key": why} for every setting outside its plausible range."""
    out = {}
    for dotted in RANGES_UM:
        sec, key = dotted.split(".")
        msg = range_problem(dotted, getattr(getattr(cfg, sec), key))
        if msg:
            out[dotted] = msg
    pm = str(getattr(cfg.cross_section, "pump_mode", "TE00")).strip().upper()
    if pm not in ("TE00", "TM00"):
        out["cross_section.pump_mode"] = (f"Pump mode {cfg.cross_section.pump_mode!r}: use "
                                          "TE00 or TM00.")
    return out
