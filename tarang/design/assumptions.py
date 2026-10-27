"""Values the design assumes rather than computes, with where each comes from.

The loss presets are starting points taken from publications; your own measured values
replace them (``ring.scattering_loss_db_cm`` / ``ring.absorption_loss_db_cm``). The
table returned by :func:`table` is shown at the start of a run, written to the report
and to ``benchmark_pack/assumptions.csv``.
"""
from __future__ import annotations

from typing import Optional

#: name -> scattering and absorption loss (dB/cm), source and note.
LOSS_PRESETS: dict[str, dict] = {
    "typical_ebl": dict(
        scattering_db_cm=0.3, absorption_db_cm=0.1,
        short="Colación et al., Sci. Rep. (2025)",
        source="Colación, Rukh, Buck, Drake, Sci. Rep. (2025), doi:10.1038/s41598-025-30946-x",
        note="EBL-patterned, annealed LPCVD Si3N4 microrings (620 nm film) with intrinsic Q "
             "about 0.9-1.1 x 10^6, i.e. about 0.4 dB/cm at n_g = 2.2. The split between "
             "scattering and absorption is an assumption."),
    "annealed_low_loss": dict(
        scattering_db_cm=0.020, absorption_db_cm=0.001,
        short="Xuan et al., Optica 3, 1171 (2016)",
        source="Xuan et al., Optica 3, 1171 (2016), doi:10.1364/OPTICA.3.001171",
        note="600 nm LPCVD Si3N4, annealed at 1150-1200 °C, oxide-clad: intrinsic Q 1.7 x 10^7, "
             "loss about 2.1 dB/m in total (scattering and absorption). Split assumed."),
    "state_of_the_art": dict(
        scattering_db_cm=0.0067, absorption_db_cm=0.0013,
        short="Ji et al., Optica 4, 619 (2017)",
        source="Ji et al., Optica 4, 619 (2017), doi:10.1364/OPTICA.4.000619",
        note="730 nm Si3N4, oxide-clad: 0.8 dB/m total (upper bound), bulk absorption "
             "0.13 dB/m measured; intrinsic Q 3.7 x 10^7."),
}

N2_SOURCE = ("Ikeda, Saperstein, Alic, Fainman, Opt. Express 16, 12987 (2008); the value "
             "used by Ji et al., Optica 4, 619 (2017)")
DNDT_SOURCE = ("Arbabi and Goddard, Opt. Lett. 38, 3878 (2013), doi:10.1364/OL.38.003878: "
               "Si3N4 (2.45 ± 0.09)e-5 /K, SiOx (0.95 ± 0.10)e-5 /K")
MFD_SOURCE = "Corning SMF-28 single-mode fibre: MFD 10.4 ± 0.5 µm at 1550 nm (datasheet)"


def preset(name: str) -> dict:
    try:
        return LOSS_PRESETS[str(name).strip().lower()]
    except KeyError:
        raise ValueError(f"unknown loss preset {name!r}; choose one of "
                         f"{', '.join(LOSS_PRESETS)}") from None


def loss_source(cfg, which: str) -> str:
    """'preset <name>: <source>' if the value equals the preset's, else 'your value'."""
    p = preset(cfg.ring.loss_preset)
    value = getattr(cfg.ring, f"{which}_loss_db_cm")
    if value is None or abs(float(value) - p[f"{which}_db_cm"]) < 1e-12:
        return f"preset '{cfg.ring.loss_preset}': {p['source']}"
    return "your value (design.toml)"


def table(cfg, results: Optional[dict] = None) -> list[dict]:
    """Every assumed value: quantity, value, unit, source, note."""
    results = results or {}
    g, s, x, r = cfg.grating, cfg.stack, cfg.cross_section, cfg.ring
    p = preset(r.loss_preset)
    scat = r.scattering_loss_db_cm if r.scattering_loss_db_cm is not None else p["scattering_db_cm"]
    absl = r.absorption_loss_db_cm if r.absorption_loss_db_cm is not None else p["absorption_db_cm"]
    rows = [
        dict(quantity="scattering loss", value=scat, unit="dB/cm",
             source=loss_source(cfg, "scattering"), note=p["note"]),
        dict(quantity="absorption loss", value=absl, unit="dB/cm",
             source=loss_source(cfg, "absorption"),
             note="Unannealed LPCVD/PECVD nitride absorbs near 1520 nm (N-H bonds); use a "
                  "larger value for unannealed films."),
        dict(quantity="Kerr index n2 of the core", value=cfg.assumptions.n2_m2_w, unit="m^2/W",
             source=N2_SOURCE, note="Used only for the comparative threshold estimate (±~2x)."),
        dict(quantity="thermo-optic coefficient of the core", value=cfg.assumptions.dn_dT_core_per_K,
             unit="1/K", source=DNDT_SOURCE, note="For the resonance shift with temperature only."),
        dict(quantity="thermo-optic coefficient of the BOX", value=cfg.assumptions.dn_dT_box_per_K,
             unit="1/K", source=DNDT_SOURCE, note="PECVD SiOx; thermal oxide is close."),
        dict(quantity="fibre mode-field diameter", value=g.mfd_um, unit="um",
             source=MFD_SOURCE, note="Gaussian beam with its waist at the source plane."),
        dict(quantity="fibre angle from vertical", value=g.fiber_angle_deg, unit="deg",
             source="your setup", note=f"beam travels {g.fiber_tilt} the waveguide"),
        dict(quantity="width tolerance", value=x.tolerance_width_um, unit="um",
             source="your process", note="± around the chosen width"),
        dict(quantity="thickness tolerance", value=x.tolerance_height_um, unit="um",
             source="your process", note="± around the chosen film thickness"),
        dict(quantity="sidewalls", value="vertical", unit="",
             source="model limit", note="every simulation uses 90° sidewalls"),
        dict(quantity="BOX thickness", value=s.box_height_um, unit="um", source="your wafer",
             note="Si handle below the BOX in the grating simulation"),
        dict(quantity="smallest lithography feature", value=g.min_feature_um, unit="um",
             source="your process", note="grating teeth and grooves narrower than this are "
                                         "not simulated"),
    ]
    if str(getattr(cfg.run, "coupling", "grating") or "grating").lower() == "edge":
        e = cfg.edge
        drop = {"fibre mode-field diameter", "fibre angle from vertical", "smallest lithography feature"}
        rows = [r_ for r_ in rows if r_["quantity"] not in drop]
        rows.insert(3, dict(quantity="lensed-fibre spot (MFD)", value=e.fiber_mfd_um, unit="um",
                            source="your fibre's datasheet",
                            note=f"± {e.mfd_tolerance_um:g} µm; round Gaussian spot, waist at the "
                                 "working distance, TE"))
        rows.insert(4, dict(quantity="chip facet", value="flat", unit="", source="model limit",
                            note="cleaved or polished, perpendicular to the waveguide, no coating"))
        for r_ in rows:
            if r_["quantity"] == "BOX thickness":
                r_["note"] = "Si handle below the BOX in the facet mode solve"
    return rows
