"""Material properties beyond n and k, each with its source.

``media.json`` maps a material key to its index table; ``materials.json`` (this module's
registry) adds what a comb design needs besides the index: the Kerr index n2, the
thermo-optic coefficient dn/dT, the band gap and the measured transparency window.
A value that is not recorded is ``None``: TARANG then uses your value from the design
settings, and asks for one where the design needs it, instead of falling back silently
to another material's number.

A material given as your own CSV file has no entry here; its properties come from your
settings only.

Exposes: info(), known(), two_photon_edge_um(), H_EV_UM.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

REGISTRY_FILE = Path(__file__).with_name("materials.json")

#: h c / e in eV µm: a photon of wavelength λ (µm) has energy H_EV_UM / λ (eV)
H_EV_UM = 1.239842

_FIELDS = ("name", "n2_m2_w", "n2_source", "dn_dT_per_K", "dn_dT_source", "band_gap_ev",
           "band_gap_source", "transparent_um", "transparent_source", "family")


@dataclass(frozen=True)
class MaterialInfo:
    key: str
    name: str
    n2_m2_w: Optional[float]
    n2_source: str
    dn_dT_per_K: Optional[float]
    dn_dT_source: str
    band_gap_ev: Optional[float]
    band_gap_source: str
    transparent_um: Optional[tuple]
    transparent_source: str
    family: str


@lru_cache(maxsize=1)
def _registry() -> dict:
    data = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    data.pop("_about", None)
    for key, entry in data.items():
        missing = [f for f in _FIELDS if f not in entry]
        if missing:
            raise ValueError(f"{REGISTRY_FILE.name}: entry {key!r} lacks {missing}")
    return data


def known() -> list[str]:
    """Keys with an entry in the registry."""
    return sorted(_registry())


def info(key) -> Optional[MaterialInfo]:
    """The registry entry of a material key ('air' included); None for a CSV path or a key
    without an entry."""
    k = str(key or "").strip().lower()
    if k in {"none", ""} or k.endswith(".csv"):
        return None
    e = _registry().get(k)
    if e is None:
        return None
    tr = e["transparent_um"]
    return MaterialInfo(key=k, name=e["name"], n2_m2_w=e["n2_m2_w"], n2_source=e["n2_source"],
                        dn_dT_per_K=e["dn_dT_per_K"], dn_dT_source=e["dn_dT_source"],
                        band_gap_ev=e["band_gap_ev"], band_gap_source=e["band_gap_source"],
                        transparent_um=tuple(tr) if tr else None,
                        transparent_source=e["transparent_source"], family=e["family"])


def two_photon_edge_um(band_gap_ev: float) -> float:
    """Longest wavelength at which two photons together bridge the band gap
    (2 h c / λ >= E_g): two-photon absorption is possible below it."""
    return 2 * H_EV_UM / float(band_gap_ev)
