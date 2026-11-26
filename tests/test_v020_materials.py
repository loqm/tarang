"""v0.2.0, phase A: the material registry (tarang/media/materials.json) and the material
checks of the materials stage.

The registry replaces the Si3N4 values that used to be the defaults of every design, so a
Ta2O5 or GaP design no longer silently inherits Si3N4's Kerr index. For Si3N4 on SiO2 the
values are exactly the old defaults: every number of a Si3N4 design is unchanged."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "tarang" / "media"


# ---- the registry --------------------------------------------------------------------- #
def test_every_material_has_a_registry_entry_with_every_field():
    from tarang.media import materials
    database = json.loads((MEDIA / "media.json").read_text(encoding="utf-8"))
    for key in list(database) + ["air"]:
        inf = materials.info(key)
        assert inf is not None, f"{key} has no entry in materials.json"
        for name in ("n2_m2_w", "dn_dT_per_K", "band_gap_ev", "transparent_um"):
            value, source = getattr(inf, name), getattr(
                inf, name.replace("_m2_w", "").replace("_per_K", "").replace("_ev", "")
                .replace("_um", "") + "_source")
            if value not in (None, 0.0):
                assert source.strip(), f"{key}.{name} = {value} has no source"


def test_registry_is_shipped_with_the_package():
    import tomllib
    setup = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "materials.json" in setup["tool"]["setuptools"]["package-data"]["tarang.media"]


def test_unknown_and_csv_materials_have_no_entry():
    from tarang.media import materials
    assert materials.info("my_film.csv") is None
    assert materials.info("no_such_key") is None
    assert materials.info("SI3N4").key == "si3n4"                     # case as media.json


def test_two_photon_edge():
    from tarang.media.materials import two_photon_edge_um
    assert two_photon_edge_um(2.26) == pytest.approx(1.0972, abs=1e-4)   # GaP: 1.1 µm
    assert two_photon_edge_um(1.12) == pytest.approx(2.2140, abs=1e-4)   # Si


# ---- Si3N4 designs keep their numbers -------------------------------------------------- #
def test_si3n4_on_sio2_gets_exactly_the_old_defaults():
    from tarang.design.assumptions import material_value, n2
    from tarang.design.config import DesignConfig
    c = DesignConfig()
    assert (c.materials.core, c.materials.box) == ("si3n4", "sio2")
    assert c.assumptions.n2_m2_w is None                     # not fixed in the settings
    assert n2(c) == 2.4e-19                                   # v0.1.5's default, exactly
    assert material_value(c, "dn_dT_core")[0] == 2.45e-5
    assert material_value(c, "dn_dT_box")[0] == 0.95e-5
    assert "Ikeda" in material_value(c, "n2")[1]


def test_an_older_design_file_with_the_si3n4_values_reads_the_same(tmp_path):
    from tarang.design.assumptions import material_value
    from tarang.design.config import DesignConfig
    text = (ROOT / "examples" / "design_sin1550.toml").read_text(encoding="utf-8")
    p = tmp_path / "design.toml"
    p.write_text(text, encoding="utf-8")
    c = DesignConfig.load(p)
    value, source = material_value(c, "n2")
    assert value == 2.4e-19 and "Ikeda" in source           # equal to published: its source


def test_your_value_wins_and_is_labelled_as_yours():
    from tarang.design.assumptions import material_value, table
    from tarang.design.config import DesignConfig
    c = DesignConfig()
    c.assumptions.n2_m2_w = 2.5e-19
    assert material_value(c, "n2") == (2.5e-19, "your value (design.toml)")
    row = next(r for r in table(c) if r["quantity"] == "Kerr index n2 of the core")
    assert row["value"] == 2.5e-19 and row["source"] == "your value (design.toml)"


def test_tantala_core_takes_tantala_values_not_si3n4():
    from tarang.design.assumptions import material_value, n2
    from tarang.design.config import DesignConfig
    c = DesignConfig()
    c.materials.core = "ta2o5"
    assert n2(c) == 6.2e-19
    assert material_value(c, "dn_dT_core")[0] == 8.8e-6
    assert "OPTICA.411968" in material_value(c, "n2")[1]


def test_a_material_without_n2_asks_instead_of_guessing(tmp_path):
    from tarang.design.assumptions import n2, table
    from tarang.design.config import DesignConfig, SettingsError
    csv = tmp_path / "film.csv"
    csv.write_text("wl,n\n1.0,2.3\n2.0,2.2\n", encoding="utf-8")
    c = DesignConfig()
    c.materials.core = str(csv)
    with pytest.raises(SettingsError, match=r"\[assumptions\] n2_m2_w"):
        n2(c)
    row = next(r for r in table(c) if r["quantity"] == "Kerr index n2 of the core")
    assert row["value"] == "not set"
    c.assumptions.n2_m2_w = 1e-18
    assert n2(c) == 1e-18


def test_lle_inputs_use_the_core_materials_n2(tmp_path):
    """gamma = 2 pi n2 / (lambda A_eff) with the registry's n2 for the core."""
    from tarang.design.config import DesignConfig
    c = DesignConfig()
    c.materials.core = "ta2o5"
    from tarang.design.assumptions import n2
    gamma = 2 * math.pi * n2(c) / (1.064e-6 * 0.5e-12)
    assert gamma == pytest.approx(2 * math.pi * 6.2e-19 / (1.064e-6 * 0.5e-12))


# ---- the tantala index table ------------------------------------------------------------ #
def _black_sellmeier(wl):
    l2 = np.asarray(wl) ** 2
    return np.sqrt(1 + 0.033 * l2 / (l2 - 0.368 ** 2) + 3.212 * l2 / (l2 - 0.1639 ** 2)
                   + 3.747 * l2 / (l2 - 14.5 ** 2))


def test_tantala_table_is_black_et_al_sellmeier():
    data = np.genfromtxt(MEDIA / "data" / "Black_Ta2O5_Sellmeier.csv", delimiter=",", names=True)
    assert data["wl"][0] == pytest.approx(0.5) and data["wl"][-1] == pytest.approx(5.0)
    assert np.all(np.diff(data["wl"]) > 0) and len(data) == 451
    assert np.max(np.abs(data["n"] - _black_sellmeier(data["wl"]))) < 1e-9
    assert float(np.interp(1.55, data["wl"], data["n"])) == pytest.approx(2.0591, abs=1e-4)
    assert float(np.interp(1.064, data["wl"], data["n"])) == pytest.approx(2.0754, abs=1e-4)


@pytest.mark.slow
def test_tantala_fits_over_the_1064_band():
    from tarang.media.medium_helper import get_medium
    import tidy3d as td
    m = get_medium("ta2o5", wl_range=(0.93, 1.20))
    n = float(np.real(m.nk_model(td.C_0 / 1.064)[0]))
    assert n == pytest.approx(float(_black_sellmeier(1.064)), abs=2e-4)


# ---- the checks ----------------------------------------------------------------------- #
def _levels(findings, check):
    return [f["level"] for f in findings if f["check"] == check]


def test_si3n4_at_1550_has_no_warning():
    from tarang.design.config import DesignConfig
    from tarang.design.material_checks import run_checks
    f = run_checks(DesignConfig(), (1.35, 1.75), {"core": 1.996, "box": 1.444, "cladding": 1.0})
    assert [x for x in f if x["level"] == "warning"] == []
    assert _levels(f, "thermo-optic sign") == ["note"]
    assert _levels(f, "index contrast") == ["ok"]


def test_two_photon_absorption_is_flagged_below_the_edge():
    from tarang.design.config import DesignConfig
    from tarang.design.material_checks import run_checks
    c = DesignConfig()
    c.materials.core, c.assumptions.n2_m2_w, c.assumptions.dn_dT_core_per_K = "si_franta", 1e-18, 1e-4
    assert _levels(run_checks(c, (1.35, 1.75)), "two-photon absorption") == ["warning"]
    c.run.wavelength_um = 2.5                                  # above 2.214 µm: free
    assert _levels(run_checks(c, (2.3, 2.7)), "two-photon absorption") == ["ok"]
    c.run.wavelength_um = 2.3                                  # pump free, short end not
    assert _levels(run_checks(c, (2.0, 2.6)), "two-photon absorption") == ["note"]
    c.run.wavelength_um = 1.0                                  # above the gap itself
    msgs = [x["message"] for x in run_checks(c, (0.9, 1.1)) if x["check"] == "band gap"]
    assert msgs and "linear absorption" in msgs[0]


def test_band_outside_the_transparency_window():
    from tarang.design.config import DesignConfig
    from tarang.design.material_checks import run_checks
    c = DesignConfig()
    c.materials.core = "ta2o5"
    c.run.wavelength_um = 0.40
    assert _levels(run_checks(c, (0.30, 0.45)), "transparency") == ["warning"]
    c.run.wavelength_um = 1.064
    assert _levels(run_checks(c, (0.93, 1.20)), "transparency") == ["ok"]


def test_si3n4_loss_presets_are_flagged_for_other_cores():
    from tarang.design.config import DesignConfig
    from tarang.design.material_checks import run_checks
    c = DesignConfig()
    c.materials.core = "ta2o5"
    assert _levels(run_checks(c, (1.35, 1.75)), "loss presets") == ["warning"]
    c.ring.scattering_loss_db_cm, c.ring.absorption_loss_db_cm = 0.2, 0.01
    assert _levels(run_checks(c, (1.35, 1.75)), "loss presets") == []
    c = DesignConfig()
    c.materials.core = "ta2o5"
    c.measured.q_intrinsic = 1.5e6
    assert _levels(run_checks(c, (1.35, 1.75)), "loss presets") == []


def test_a_far_off_value_of_yours_is_flagged():
    from tarang.design.config import DesignConfig
    from tarang.design.material_checks import run_checks
    c = DesignConfig()
    c.materials.core, c.assumptions.n2_m2_w = "ta2o5", 2.4e-19     # an Si3N4 value left over
    msgs = [x["message"] for x in run_checks(c, (1.35, 1.75)) if x["level"] == "warning"]
    assert any("differs from the value published for ta2o5" in m for m in msgs)


def test_negative_dn_dT_says_the_soliton_side_is_stable():
    from tarang.design.config import DesignConfig
    from tarang.design.material_checks import run_checks
    c = DesignConfig()
    c.assumptions.dn_dT_core_per_K = -1e-5
    msg = next(x["message"] for x in run_checks(c, (1.35, 1.75)) if x["check"] == "thermo-optic sign")
    assert "red-detuned side, where solitons live, is thermally self-stable" in msg


def test_no_guided_mode_is_flagged():
    from tarang.design.config import DesignConfig
    from tarang.design.material_checks import run_checks
    f = run_checks(DesignConfig(), (1.35, 1.75), {"core": 1.40, "box": 1.444})
    assert _levels(f, "index contrast") == ["warning"]


def test_materials_stage_writes_the_checks(tmp_path, no_show):
    """The materials stage prints the findings, writes them to a table and keeps them for
    the report (a tantala core with Si3N4 loss presets: one warning)."""
    from tarang.design.__main__ import main
    from tarang.design.config import DesignConfig
    c = DesignConfig()
    c.materials.core, c.run.wavelength_um = "ta2o5", 1.064
    p = c.save(tmp_path / "design.toml")
    main(["--config", str(p), "--yes", "--only", "materials"])
    table = tmp_path / "tables" / "materials_checks.csv"
    assert table.exists()
    text = table.read_text(encoding="utf-8")
    assert "loss presets" in text and "transparency" in text
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    res = state["stages"]["materials"]["results"]
    assert any(x["check"] == "loss presets" for x in res["checks"])


# ---- GaP and AlGaAs tables ------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["Black_Ta2O5_Sellmeier.csv", "Wei_GaP_295K.csv",
                                  "Gehrsitz_Al0.20Ga0.80As_296K.csv"])
def test_every_formula_table_is_written_from_its_formula(tmp_path, name):
    """The shipped file is byte for byte what tarang.media.index_models writes."""
    from tarang.media.index_models import write
    fresh = write(name, tmp_path)
    assert fresh.read_bytes() == (MEDIA / "data" / name).read_bytes()


def test_gap_and_algaas_values():
    from tarang.media.index_models import e_gamma_algaas_ev, n_algaas_gehrsitz, n_gap_wei
    assert float(n_gap_wei(1.55)) == pytest.approx(3.0551, abs=1e-4)
    assert float(n_gap_wei(1.064)) == pytest.approx(3.1046, abs=1e-4)
    assert float(n_algaas_gehrsitz(1.55, 0.0)) == pytest.approx(3.3743, abs=1e-4)   # GaAs
    assert float(n_algaas_gehrsitz(1.55, 0.2)) == pytest.approx(3.2654, abs=1e-4)
    assert e_gamma_algaas_ev(0.0, 296.15) == pytest.approx(1.426, abs=1e-3)       # GaAs gap
    # Table IV against the sample fit of Table III (x = 0.176, 23 °C) at 780 nm: within the
    # stated quality of fit (Δn_max 4.4e-3 at x = 0.176)
    E2 = 1 / 0.78 ** 2
    R = 0.824 * 1.55e-3 / (0.724e-3 - E2) + 0.176 * 2.61e-3 / (1.331e-3 - E2)
    n3 = math.sqrt(3.1739 + 0.0622 / (1.855 - E2) + 53.3589 / (7.486 - E2) + R)
    assert abs(float(n_algaas_gehrsitz(0.78, 0.176)) - n3) < 4.4e-3


def test_gap_at_1064_is_flagged_for_two_photon_absorption():
    """GaP's 2.26 eV gap: two 1064 nm photons (2.33 eV) bridge it, 1550 nm ones do not."""
    from tarang.design.config import DesignConfig
    from tarang.design.material_checks import run_checks
    c = DesignConfig()
    c.materials.core, c.run.wavelength_um = "gap", 1.064
    assert _levels(run_checks(c, (0.93, 1.20)), "two-photon absorption") == ["warning"]
    c.run.wavelength_um = 1.55
    assert _levels(run_checks(c, (1.35, 1.75)), "two-photon absorption") == ["ok"]


def test_algaas_at_1550_is_free_of_two_photon_absorption_at_the_pump():
    from tarang.design.config import DesignConfig
    from tarang.design.material_checks import run_checks
    c = DesignConfig()
    c.materials.core = "al0.2ga0.8as"
    # 2 x 1.707 eV -> edge at 1.453 µm: pump free, the band's short end not
    assert _levels(run_checks(c, (1.35, 1.75)), "two-photon absorption") == ["note"]
    assert _levels(run_checks(c, (1.50, 1.60)), "two-photon absorption") == ["ok"]
