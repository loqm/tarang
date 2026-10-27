"""Phase 15B: the simulation figures of a comb paper. Dispersive-wave positions from the
zeros of D_int, A_eff and the Kerr coefficient over the band, the TE00/TM00 anomalous map,
the resonance shift with temperature, and the pyLLE proof script (outside the package)."""
from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_dispersive_waves_are_the_zeros_of_dint_away_from_the_pump():
    from tarang.waveguide.dispersion_mapping.broadband import dispersive_waves
    mu = np.arange(-50, 51)
    dint = 1e3 * (0.5 * mu ** 2 - 0.0005 * mu ** 4)            # zeros at mu = ±sqrt(1000)
    lam = 1550 - 0.8 * mu
    dw = dispersive_waves(mu, dint, lam)
    assert dw == pytest.approx([1550 - 0.8 * math.sqrt(1000), 1550 + 0.8 * math.sqrt(1000)],
                               abs=0.05)
    assert dispersive_waves(mu, 1e3 * mu ** 2.0, lam) == []      # pure anomalous: none
    assert dispersive_waves(mu, 1e3 * (mu ** 2 - 4), lam) == []  # |mu| = 2: the pump region


def test_thermal_shift_is_the_power_weighted_dn_dT_over_n_g():
    from tarang.design import physics
    lo, hi = physics.thermal_shift_pm_per_K(1.55, 2.2228, 0.8, 2.45e-5, 0.95e-5)
    assert lo == pytest.approx(1.55e6 / 2.2228 * 0.8 * 2.45e-5)
    assert hi == pytest.approx(1.55e6 / 2.2228 * (0.8 * 2.45e-5 + 0.2 * 0.95e-5))
    assert 13 < lo < hi < 16                                     # pm/K for 700 x 1000 nm


def test_thermo_optic_values_are_in_the_assumptions_with_their_source():
    from tarang.design import assumptions
    from tarang.design.config import DesignConfig
    rows = {r["quantity"]: r for r in assumptions.table(DesignConfig())}
    core = rows["thermo-optic coefficient of the core"]
    assert core["value"] == 2.45e-5 and "Arbabi" in core["source"]
    assert rows["thermo-optic coefficient of the BOX"]["value"] == 0.95e-5


def test_gamma_over_the_band(tmp_path, monkeypatch):
    from tarang import figures
    from tarang.waveguide.dispersion_mapping.broadband import plot_gamma
    monkeypatch.setitem(figures.SETTINGS, "formats", ["png"])
    monkeypatch.setitem(figures.SETTINGS, "dpi", 50)
    rec = dict(H=0.7, W=1.0, wl=[1.4, 1.5, 1.6, 1.7], neff_TE0=[1.75, 1.72, 1.69, 1.40],
               n_cut=[1.44, 1.44, 1.44, 1.44], aeff_TE0_um2=[0.80, 0.85, 0.90, 2.0])
    g = plot_gamma(rec, 2.4e-19, tmp_path / "gamma_H700_W1000.png")
    assert list(g["wl"]) == [1.4, 1.5, 1.6]                     # the unguided point is left out
    assert g["gamma"][1] == pytest.approx(2 * math.pi * 2.4e-19 / (1.5e-6 * 0.85e-12))
    assert (tmp_path / "gamma_H700_W1000.png").exists()
    assert plot_gamma(dict(rec, aeff_TE0_um2=None), 2.4e-19, tmp_path / "x.png") is None


def test_te_tm_anomalous_map(tmp_path):
    import os
    base = {"wl0": 0.78, "steps": 50, "core": "si3n4", "substrate": "sio2", "cladding": None,
            "box": 4.0, "n_shift": 0.0, "n_cut": 1.4537, "margin": 0.005, "si_handle": False,
            "pml": 0, "bend_radius": None, "loss_TE0_dB_cm": 0.0, "Aeff_TE0_um2": 0.2,
            "Gamma_TE0": 0.7, "beta2_TE0": 0.0, "ng_TE0": 2.1, "beta2_TM0": 0.0, "ng_TM0": 2.0,
            "fit_resid_TE0": 1e-6, "fit_resid_TM0": 1e-6, "neff_TE": [1.8, 1.5],
            "neff_TM": [1.78], "hybrid_TE": [False, False], "hybrid_TM": [False],
            "guided_TE_margin": 1, "guided_TM_margin": 1, "guided_counts_valid": True,
            "te0_gap_to_next": 0.05, "secs": 1.0}
    # the measured 780 nm values (H = 0.40, 0.45 um; W = 0.5, 0.6, 0.7 um)
    pts = {(0.40, 0.5): (379.2, 17.3, 1, 2), (0.40, 0.6): (215.4, 81.7, 2, 2),
           (0.40, 0.7): (65.2, 90.4, 3, 2), (0.45, 0.5): (480.7, 179.6, 1, 2),
           (0.45, 0.6): (283.4, 202.1, 3, 2), (0.45, 0.7): (121.3, 190.1, 3, 2),
           (0.35, 0.5): (209.2, -399.4, 1, 1), (0.35, 0.6): (99.0, -257.7, 2, 2),
           (0.35, 0.7): (-31.4, -207.8, 2, 2)}
    data = tmp_path / "m.jsonl"
    with open(data, "w", encoding="utf-8") as f:
        for (H, W), (dte, dtm, gte, gtm) in pts.items():
            f.write(json.dumps(dict(base, H=H, W=W, D_TE0=dte, D_TM0=dtm, guided_TE=gte,
                                    guided_TM=gtm)) + "\n")
    env = dict(os.environ, PYTHONPATH=str(ROOT), MPLBACKEND="Agg", TARANG_FIGURE_FORMATS="png",
               TARANG_FIGURE_DPI="50", TARANG_FIGURE_DATA="1")
    r = subprocess.run([sys.executable, "-m", "tarang.waveguide.dispersion_mapping.maps",
                        "--data", str(data), "--tm", "--plots-dir", str(tmp_path / "p")],
                       cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr
    meta = json.loads((tmp_path / "p" / "maps" / "map10_TE_TM_anomalous_meta.json")
                      .read_text(encoding="utf-8"))
    assert meta["panels"][0]["xlabel"] == "Width W (µm)"
    z = np.load(tmp_path / "p" / "maps" / "map10_TE_TM_anomalous_data.npz")
    cls = z["p0_0_z"]                    # 0 both normal, 1 TE, 2 TM, 3 both (rows: H)
    assert cls[0].tolist() == [1, 1, 0] and cls[1].tolist() == [3, 3, 3]


# ---- pyLLE proof script ------------------------------------------------------------------- #
def _script():
    import importlib.util
    p = ROOT / "validation" / "pylle_soliton" / "run_pylle.py"
    spec = importlib.util.spec_from_file_location("run_pylle", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_design(tmp_path):
    lle = tmp_path / "d" / "lle"
    lle.mkdir(parents=True)
    (lle / "pyLLE_dispersion.csv").write_text("100,1.9e14\n101,1.902e14\n", encoding="utf-8")
    over = dict(K=2.0, Q_c=5e5, Q_L=3.3e5, kappa_2pi_Hz=5.8e8, threshold_W=0.05)
    (lle / "lle_inputs.json").write_text(json.dumps(dict(
        pump_frequency_Hz=1.934e14, coupling={"critical": dict(over, K=1.0),
                                              "over_coupled_K2": over},
        pyLLE={"res": dict(R=107.5e-6, Qi=1e6, Qc=5e5, gamma=1.1,
                           dispfile="pyLLE_dispersion.csv")})), encoding="utf-8")
    return tmp_path / "d"


def test_pylle_inputs_come_from_the_design(tmp_path):
    m = _script()
    inp = m.load_inputs(_fake_design(tmp_path))
    res, sim = m.settings(inp)
    assert res["R"] == 107.5e-6 and res["Qc"] == 5e5 and res["γ"] == 1.1
    assert res["dispfile"].endswith("pyLLE_dispersion.csv")
    assert sim["Pin"] == [pytest.approx(4 * 0.05)]                # 4 x the over-coupled threshold
    kappa = 2 * math.pi * 5.8e8
    assert sim["δω_init"] == pytest.approx(2 * kappa / 2) and sim["δω_end"] < 0
    assert sim["μ_sim"] == [-200, 200]
    res2, sim2 = m.settings(inp, power_mw=150)
    assert sim2["Pin"] == [pytest.approx(0.15)]


def test_pylle_script_dry_run_and_missing_inputs(tmp_path):
    p = ROOT / "validation" / "pylle_soliton" / "run_pylle.py"
    d = _fake_design(tmp_path)
    env = {k: v for k, v in __import__("os").environ.items() if k != "PYTHONPATH"}
    env.update(PYTHONPATH=str(ROOT), PYTHONIOENCODING="cp1252")    # a Windows pipe, on any OS
    r = subprocess.run([sys.executable, str(p), str(d), "--dry-run"], capture_output=True,
                       text=True, encoding="cp1252", errors="replace", env=env)
    assert r.returncode == 0, r.stderr
    assert "pump 200 mW" in r.stdout
    r = subprocess.run([sys.executable, str(p), str(tmp_path / "nothing"), "--dry-run"],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode != 0 and "run the design's layout step first" in (r.stderr + r.stdout)


def test_soliton_step_is_on_the_plateau_after_the_chaotic_maximum():
    m = _script()
    p = np.concatenate([np.linspace(0, 10, 100), 10 + np.random.default_rng(0).random(100),
                        np.full(100, 2.0), np.zeros(100)])        # rise, chaos, soliton, lost
    k = m.soliton_step(p)
    assert 200 <= k < 300 and p[k] == 2.0


# ---- TM00 as the pump mode ---------------------------------------------------------------- #
def test_pump_fields_name_the_records_fields_of_each_mode():
    from tarang.waveguide.dispersion_mapping.core import pol_of, pump
    r = dict(D_TE0=300.0, D_TM0=150.0, ng_TE0=2.2, ng_TM0=2.1, neff_TE=[1.8], neff_TM=[1.7],
             guided_TE=2, guided_TM=1)
    te, tm = pump(r, "TE00"), pump(r, "tm00")
    assert (te["D"], te["ng"], te["guided"], te["other_D"]) == (300.0, 2.2, 2, 150.0)
    assert (tm["D"], tm["ng"], tm["guided"], tm["neff"]) == (150.0, 2.1, 1, [1.7])
    assert pol_of("TE00") == "TE" and pol_of("tm") == "TM"
    with pytest.raises(ValueError):
        pol_of("TE01x")


def _row(H, W, dte, dtm, nte, ntm, n_cut=1.4537):
    return dict(H=H, W=W, D_TE0=dte, D_TM0=dtm, ng_TE0=2.1, ng_TM0=2.0, neff_TE=nte,
                neff_TM=ntm, guided_TE=sum(n > n_cut for n in nte),
                guided_TM=sum(n > n_cut for n in ntm), n_cut=n_cut)


def test_tm00_targets_are_ranked_by_their_own_rule():
    from tarang.design.stages import _rank_targets, other_split
    rows = [_row(0.45, 0.6, 283.0, 202.0, [1.82, 1.70, 1.50], [1.808]),   # TM single, TE x3
            _row(0.45, 0.7, 121.0, 190.0, [1.85, 1.75, 1.55], [1.825]),
            _row(0.40, 0.6, 215.0, -5.0, [1.80, 1.60], [1.774]),           # TM normal
            _row(0.40, 0.7, 65.0, 90.0, [1.83, 1.70, 1.50], [1.791, 1.60])]  # TM two-mode
    tm = _rank_targets(rows, [0.40, 0.45], [0.6, 0.7], mode="TM00")
    assert [(c["H_um"], c["W_um"]) for c in tm] == [(0.45, 0.6), (0.45, 0.7)] or \
        [(c["H_um"], c["W_um"]) for c in tm] == [(0.45, 0.7), (0.45, 0.6)]
    assert "D_TM0" in tm[0] and "D_TE0" not in tm[0]
    # TM00 at 1.808 sits between TE modes 1.82 and 1.70: the nearest is 0.012 away
    assert other_split(rows[0], "TM00") == pytest.approx(0.012)
    assert other_split(rows[0], "TE00") == pytest.approx(0.012)
    te = _rank_targets(rows, [0.40, 0.45], [0.6, 0.7], mode="TE00")
    assert te == []                                    # every point is TE-multimode


def test_map_tool_judges_targets_by_the_pump_mode(tmp_path):
    import os
    base = {"wl0": 0.78, "steps": 50, "core": "si3n4", "substrate": "sio2", "cladding": None,
            "box": 4.0, "n_shift": 0.0, "margin": 0.005, "si_handle": False, "pml": 0,
            "bend_radius": None, "loss_TE0_dB_cm": 0.0, "Aeff_TE0_um2": 0.2, "Gamma_TE0": 0.7,
            "Aeff_TM0_um2": 0.25, "Gamma_TM0": 0.6, "beta2_TE0": -1.0, "beta2_TM0": -1.0,
            "fit_resid_TE0": 1e-6, "fit_resid_TM0": 1e-6, "guided_TE_margin": 1,
            "guided_TM_margin": 1, "guided_counts_valid": True, "te0_gap_to_next": 0.05,
            "secs": 1.0}
    rows = [_row(0.45, 0.6, 283.0, 202.0, [1.82, 1.70, 1.50], [1.808]),
            _row(0.45, 0.7, 121.0, 190.0, [1.85, 1.75, 1.55], [1.825]),
            _row(0.40, 0.6, 215.0, -5.0, [1.80, 1.60], [1.774]),
            _row(0.40, 0.7, 65.0, 90.0, [1.83, 1.70, 1.50], [1.791, 1.60])]
    data = tmp_path / "m.jsonl"
    with open(data, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(dict(base, **r, hybrid_TE=[False] * len(r["neff_TE"]),
                                    hybrid_TM=[False] * len(r["neff_TM"]))) + "\n")
    env = dict(os.environ, PYTHONPATH=str(ROOT), MPLBACKEND="Agg", TARANG_FIGURE_FORMATS="png",
               TARANG_FIGURE_DPI="40", TARANG_FIGURE_DATA="0")
    r = subprocess.run([sys.executable, "-m", "tarang.waveguide.dispersion_mapping.maps",
                        "--data", str(data), "--tm", "--pump", "TM00", "--plots-dir",
                        str(tmp_path / "p")], cwd=tmp_path, env=env, capture_output=True,
                       text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr
    tg = list(csv.DictReader(open(tmp_path / "p" / "target_geometries.csv", encoding="utf-8")))
    assert {(t["H_nm"], t["W_nm"]) for t in tg} == {("450", "600"), ("450", "700")}
    assert "D_TM0" in tg[0] and float(tg[0]["D_TM0"]) == 202.0


@pytest.mark.filterwarnings("ignore:The supplied partial-ring geometry ends inside")
def test_ring_gap_fdtd_launches_tm00_only_for_tm_designs():
    import tidy3d as td
    from tarang.geometries.partial_ring_resonator import get_active_simulation
    kw = dict(radius=20.0, gap=0.5, wg_width=0.6, wg_height=0.45,
              core_medium=td.Medium(permittivity=4.0), box_medium=td.Medium(permittivity=2.1),
              wavelength_um=0.78, span_angle_rad=0.6, fine_steps_per_wvl=8,
              coarse_steps_per_wvl=6)
    te = get_active_simulation(**kw).sim
    tm = get_active_simulation(**kw, mode_pol="tm").sim
    src_te = next(s for s in te.sources if isinstance(s, td.ModeSource))
    src_tm = next(s for s in tm.sources if isinstance(s, td.ModeSource))
    assert src_te.mode_spec.num_modes == 1                       # TE designs: unchanged
    assert src_tm.mode_spec.num_modes == 4
    assert src_tm.mode_spec.sort_spec.filter_key == "TM_fraction"
    assert all(m.mode_spec.sort_spec.filter_key == "TM_fraction"
               for m in tm.monitors if isinstance(m, td.ModeMonitor))


def test_pump_mode_setting_checks_and_keeps_old_keys(tmp_path):
    from tarang.design.config import DesignConfig, implausible
    from tarang.design.stages import gap_builder_options, pump_mode, settings_key
    c = DesignConfig()
    assert pump_mode(c) == "TE00" and "mode_pol" not in gap_builder_options(c)
    k = settings_key(c, "cross_section")
    c.cross_section.pump_mode = "TM00"
    assert settings_key(c, "cross_section") != k and gap_builder_options(c)["mode_pol"] == "tm"
    c.cross_section.pump_mode = "TE01"
    assert "cross_section.pump_mode" in implausible(c)


def test_tm00_design_stops_before_the_te_only_steps(tmp_path, capsys):
    from tarang.design import TE_ONLY_STAGES
    from tarang.design.__main__ import main
    from tarang.design.config import DesignConfig
    c = DesignConfig()
    c.cross_section.pump_mode = "TM00"
    p = c.save(tmp_path / "design.toml")
    assert main(["--config", str(p), "--yes", "--only", "grating"]) == 2
    out = capsys.readouterr().out
    assert "designed for TE00 only so far" in out and "--only materials" in out
    assert {"grating", "edge", "layout"} <= TE_ONLY_STAGES and "gap" not in TE_ONLY_STAGES


def test_gui_candidate_columns_follow_the_pump_mode():
    from tarang.gui import wizard as wz
    te = [dict(H_um=0.7, W_um=1.0, D_TE0=370.0, ng_TE0=2.22)]
    tm = [dict(H_um=0.45, W_um=0.6, D_TM0=202.0, ng_TM0=2.1)]
    assert "D_TE0" in wz.candidate_columns(te) and "D_TM0" in wz.candidate_columns(tm)
    assert wz.cand_D(te[0]) == 370.0 and wz.cand_D(tm[0]) == 202.0
    assert "cross_section.pump_mode" in wz.step("cross_section").fields
    assert wz.field_info(__import__("tarang.design.config", fromlist=["x"]).DesignConfig(),
                         "cross_section.pump_mode")["choices"] == ["TE00", "TM00"]


def test_broadband_records_of_either_mode_are_read_by_their_own_fields():
    from tarang.waveguide.dispersion_mapping.broadband import _f, _mode
    old = dict(neff_TE0=[1.7], aeff_TE0_um2=[0.7])
    new = dict(pump_mode="TM00", neff_TM0=[1.6], aeff_TM0_um2=[0.8])
    assert _mode(old) == "TE00" and _f(old, "neff") == [1.7]
    assert _mode(new) == "TM00" and _f(new, "aeff", "_um2") == [0.8]
