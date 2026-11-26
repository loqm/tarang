"""Checks of the chosen materials against the design band, run in the materials stage.

Each check compares a published property (tarang/media/materials.json) or your setting
with the design: the measured transparency window, two-photon absorption across the band
gap, the sign of the thermo-optic coefficient, whether the Kerr index is known, whether
the loss presets (taken from Si3N4 rings) apply, and whether the core guides at all.

Every finding has a level: ``warning`` (the design may be wrong or a number is missing),
``note`` (worth knowing, nothing is wrong) or ``ok``. None of them stops the design; the
stage prints them, writes ``tables/materials_checks.csv`` and the report lists them.

Exposes: run_checks(), write_csv(), LEVELS.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

from tarang.media import materials

LEVELS = ("warning", "note", "ok")

#: user value and published value further apart than this (relative) are flagged
MISMATCH = 0.2


def _row(level: str, check: str, material: str, message: str) -> dict:
    assert level in LEVELS
    return dict(level=level, check=check, material=material, message=message)


def run_checks(cfg, band: tuple[float, float], n_at_wl: Optional[dict] = None) -> list[dict]:
    """Findings for ``cfg`` over the fitted ``band`` (µm). ``n_at_wl``: role -> real index
    at the design wavelength (from the stage's fits), for the guiding check."""
    from tarang.design.assumptions import MATERIAL_VALUES, material_value
    wl = float(cfg.run.wavelength_um)
    lo, hi = float(band[0]), float(band[1])
    core = str(cfg.materials.core)
    core_info = materials.info(core)
    out: list[dict] = []

    # ---- materials TARANG knows nothing about beyond n, k
    for role in ("core", "box", "cladding"):
        key = str(getattr(cfg.materials, role))
        if key.lower() not in {"air", "none"} and materials.info(key) is None:
            out.append(_row("note", "registry", key,
                            f"The {role} ({key}) has no entry in tarang/media/materials.json: "
                            "its Kerr index and thermo-optic coefficient come from your "
                            "settings ([assumptions]) only."))

    # ---- measured transparency window
    for role in ("core", "box", "cladding"):
        key = str(getattr(cfg.materials, role))
        inf = materials.info(key)
        if inf is None or inf.transparent_um is None:
            continue
        t_lo, t_hi = inf.transparent_um
        if lo < t_lo or hi > t_hi:
            out.append(_row("warning", "transparency", key,
                            f"The band {lo:g}-{hi:g} µm reaches outside the {role}'s "
                            f"transparency window {t_lo:g}-{t_hi:g} µm ({inf.transparent_source}): "
                            "expect absorption there that the index data may not show."))
        else:
            out.append(_row("ok", "transparency", key,
                            f"The band lies inside the {role}'s transparency window "
                            f"{t_lo:g}-{t_hi:g} µm."))

    # ---- band gap: one- and two-photon absorption in the core
    if core_info is not None and core_info.band_gap_ev:
        eg = float(core_info.band_gap_ev)
        edge1 = materials.H_EV_UM / eg
        edge2 = materials.two_photon_edge_um(eg)
        ph = materials.H_EV_UM / wl
        if wl <= edge1:
            out.append(_row("warning", "band gap", core,
                            f"The pump photon ({ph:.2f} eV at {wl:g} µm) is above the core's band "
                            f"gap ({eg:g} eV): linear absorption. Choose another core or a "
                            "longer wavelength."))
        elif wl < edge2:
            out.append(_row("warning", "two-photon absorption", core,
                            f"Two pump photons ({2 * ph:.2f} eV at {wl:g} µm) exceed the core's "
                            f"band gap ({eg:g} eV; {core_info.band_gap_source}): two-photon "
                            f"absorption below {edge2:.3f} µm adds a nonlinear loss (and free "
                            "carriers) that TARANG's threshold and LLE inputs do not include."))
        elif lo < edge2:
            out.append(_row("note", "two-photon absorption", core,
                            f"The pump is free of two-photon absorption, but the short end of the "
                            f"band (below {edge2:.3f} µm) is not ({eg:g} eV band gap)."))
        else:
            out.append(_row("ok", "two-photon absorption", core,
                            f"No two-photon absorption in the band (it starts below "
                            f"{edge2:.3f} µm for a {eg:g} eV band gap)."))
    elif core.lower() not in {"air", "none"}:
        out.append(_row("note", "two-photon absorption", core,
                        "Band gap of the core not recorded: two-photon absorption not checked."))

    # ---- Kerr index and thermo-optic coefficients: known? your value far from published?
    for name in ("n2", "dn_dT_core", "dn_dT_box"):
        setting, role, field_, what, unit = MATERIAL_VALUES[name]
        key = str(getattr(cfg.materials, role))
        value, source = material_value(cfg, name)
        if value is None:
            need = ("the threshold, gamma and the LLE inputs" if name == "n2"
                    else "the thermal shift of the resonances")
            out.append(_row("warning", what, key,
                            f"The {what} is {source}; {need} need it."))
            continue
        inf = materials.info(key)
        published = getattr(inf, field_) if inf else None
        mine = getattr(cfg.assumptions, setting)
        if mine is not None and published not in (None, 0.0) and \
                abs(float(mine) - float(published)) > MISMATCH * abs(float(published)):
            out.append(_row("warning", what, key,
                            f"Your [assumptions] {setting} = {float(mine):g} {unit} differs from "
                            f"the value published for {key} ({float(published):g} {unit}). Keep "
                            "it only if it was measured on this material."))

    # ---- sign of dn/dT: which side of a resonance is thermally stable
    dndt, _ = material_value(cfg, "dn_dT_core")
    if dndt is not None and dndt != 0:
        if dndt > 0:
            msg = ("dn/dT > 0: heating red-shifts the resonances. Tuning into a resonance from "
                   "the blue side is thermally self-stable (the thermal triangle); the "
                   "red-detuned soliton states are not, so reaching them needs a fast laser "
                   "scan, power kicking or an auxiliary laser.")
        else:
            msg = ("dn/dT < 0: heating blue-shifts the resonances, so the red-detuned side, "
                   "where solitons live, is thermally self-stable.")
        out.append(_row("note", "thermo-optic sign", core, msg))

    # ---- loss presets come from Si3N4 rings
    fam = core_info.family if core_info else ""
    r, m = cfg.ring, cfg.measured
    own_loss = (r.scattering_loss_db_cm is not None and r.absorption_loss_db_cm is not None) or \
        m.q_intrinsic is not None or m.propagation_loss_db_cm is not None
    if fam != "si3n4" and not own_loss:
        out.append(_row("warning", "loss presets", core,
                        "The loss presets (ring.loss_preset) were measured on Si3N4 rings. For "
                        f"{core}, give your losses (ring.scattering_loss_db_cm and "
                        "ring.absorption_loss_db_cm) or a measured Q_i or propagation loss "
                        "([measured]); until then Q_i and the threshold are Si3N4 estimates."))

    # ---- does the core guide at all?
    if n_at_wl:
        nc = n_at_wl.get("core")
        clad = [v for k, v in n_at_wl.items() if k in ("box", "cladding") and v is not None]
        if nc is not None and clad:
            nmax = max(clad)
            if nc <= nmax:
                out.append(_row("warning", "index contrast", core,
                                f"The core index {nc:.4f} is not above the cladding's {nmax:.4f}: "
                                "no guided mode. Check the materials."))
            else:
                out.append(_row("ok", "index contrast", core,
                                f"Core {nc:.4f} against {nmax:.4f} around it: contrast "
                                f"{nc - nmax:.3f}."))
    return out


def write_csv(rows: list[dict], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["level", "check", "material", "message"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return path
