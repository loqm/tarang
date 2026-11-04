"""Phase 15A: figures for papers and a mesh band on D. Every figure is saved with the numbers
it plots (CSV, maps in an npz) and a JSON of its labels, versions and settings; the paper
style writes journal-sized copies to figures_paper/; the map tools get the same settings;
the mode solver's grid can be snapped to the core edges; the landscapes' legend keys no
longer take a colour of the colour map."""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest


@pytest.fixture
def figs(monkeypatch):
    from tarang import figures
    for k, v in dict(formats=["png"], dpi=60, style="screen", data=True).items():
        monkeypatch.setitem(figures.SETTINGS, k, v)
    return figures


def _plt():
    from tarang.design import visuals
    return visuals._plt()


def test_figure_is_saved_with_its_numbers_and_labels(tmp_path, figs):
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    x = np.array([1.50, 1.55, 1.60])
    ax[0].plot(x, [1.0, 2.0, 3.0], label="D TE00")
    ax[0].axhline(0.0)
    ax[0].axvline(1.55)
    ax[0].errorbar(x, [1, 1, 1], yerr=[0.1, 0.2, 0.3], fmt="o", label="measured")
    ax[0].bar([1, 2], [5, 6], label="counts")
    ax[0].scatter(x, [4, 5, 6], c=[7, 8, 9], label="map points")
    ax[0].set(xlabel="λ (µm)", ylabel="D (ps/(nm·km))", title="left")
    m = ax[1].pcolormesh(np.arange(4.0), np.arange(3.0), np.arange(6.0).reshape(2, 3))
    fig.colorbar(m, ax=ax[1])
    out = figs.save(fig, tmp_path / "figures" / "f1.png", meta={"stage": "demo"})
    assert out.name == "f1.png" and out.exists()
    rows = list(csv.DictReader(open(tmp_path / "figures" / "f1_data.csv", encoding="utf-8")))
    kinds = {(r["series"], r["kind"]) for r in rows}
    assert {("D TE00", "line"), ("measured", "errorbar"), ("counts", "bars"),
            ("map points", "points"), ("line 1", "hline"), ("line 2", "vline")} <= kinds
    err = [r for r in rows if r["kind"] == "errorbar"]
    assert [float(r["hi"]) - float(r["lo"]) for r in err] == pytest.approx([0.2, 0.4, 0.6])
    assert [float(r["z"]) for r in rows if r["kind"] == "points"] == [7, 8, 9]
    meta = json.loads((tmp_path / "figures" / "f1_meta.json").read_text(encoding="utf-8"))
    assert meta["stage"] == "demo" and meta["versions"]["tarang"]
    assert len(meta["panels"]) == 2                         # the colour bar is not a panel
    assert meta["panels"][0]["xlabel"] == "λ (µm)" and meta["panels"][0]["title"] == "left"
    z = np.load(tmp_path / "figures" / "f1_data.npz")
    assert z["p1_0_z"].shape == (2, 3)


def test_numbers_can_be_switched_off(tmp_path, figs, monkeypatch):
    monkeypatch.setitem(figs.SETTINGS, "data", False)
    plt = _plt()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    figs.save(fig, tmp_path / "g.png")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["g.png"]


def test_paper_style_is_journal_sized_and_keeps_the_screen_figures(tmp_path, figs, monkeypatch):
    monkeypatch.setitem(figs.SETTINGS, "style", "paper")
    plt = _plt()
    assert plt.rcParams["font.size"] == 8
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([0, 1], [0, 1])
    out = figs.save(fig, tmp_path / "run" / "figures" / "2_x" / "one.png", close=False)
    assert out == tmp_path / "run" / "figures_paper" / "2_x" / "one.png"
    assert fig.get_size_inches()[0] == pytest.approx(3.4)
    fig2, axs = plt.subplots(1, 3, figsize=(15, 4))
    figs.save(fig2, tmp_path / "two.png", close=False)       # no figures/ folder: same place
    assert (tmp_path / "two.png").exists() and fig2.get_size_inches()[0] == pytest.approx(7.0)
    plt.close("all")
    monkeypatch.setitem(figs.SETTINGS, "style", "screen")
    assert _plt().rcParams["font.size"] == 10


def test_settings_reach_the_tools_and_are_checked(figs, monkeypatch):
    figs.configure(formats=["png", "svg"], dpi=150, style="paper", data=False)
    env = figs.env()
    assert env == dict(TARANG_FIGURE_FORMATS="png,svg", TARANG_FIGURE_DPI="150",
                       TARANG_FIGURE_STYLE="paper", TARANG_FIGURE_DATA="0")
    with pytest.raises(ValueError):
        figs.configure(style="poster")
    with pytest.raises(ValueError):
        figs.configure(formats=["jpg"])


def test_design_settings_round_trip_and_leave_stage_keys_alone(tmp_path):
    from tarang.design.config import DesignConfig
    from tarang.design.stages import settings_key
    c = DesignConfig()
    assert (c.run.figure_style, c.run.figure_data) == ("screen", True)
    k = settings_key(c, "cross_section")
    c.run.figure_style, c.run.figure_data = "paper", False
    assert settings_key(c, "cross_section") == k               # figures only: nothing reruns
    c2 = DesignConfig.load(c.save(tmp_path / "design.toml"))
    assert (c2.run.figure_style, c2.run.figure_data) == ("paper", False)


def test_map_tool_writes_pdf_and_numbers_when_asked(tmp_path, monkeypatch):
    """dispersion_mapping.maps run as the design run starts it: formats from the env."""
    import os
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    data = tmp_path / "map.jsonl"
    base = {"wl0": 1.55, "steps": 50, "core": "si3n4", "substrate": "sio2", "cladding": None,
            "box": 4.0, "n_shift": 0.0, "n_cut": 1.44402, "margin": 0.005, "si_handle": False,
            "pml": 0, "bend_radius": None, "loss_TE0_dB_cm": 0.0, "Aeff_TE0_um2": 0.9,
            "Gamma_TE0": 0.8, "beta2_TE0": -100.0, "ng_TE0": 2.23, "D_TM0": -96.0,
            "beta2_TM0": 120.0, "ng_TM0": 2.2, "fit_resid_TE0": 1e-6, "fit_resid_TM0": 1e-6,
            "neff_TE": [1.70, 1.43], "neff_TM": [1.65], "hybrid_TE": [False, False],
            "hybrid_TM": [False], "guided_TE": 1, "guided_TM": 1, "guided_TE_margin": 1,
            "guided_TM_margin": 1, "guided_counts_valid": True, "te0_gap_to_next": 0.07,
            "secs": 1.0}
    with open(data, "w", encoding="utf-8") as f:
        for H in (0.7, 0.8):
            for W in (0.9, 1.0, 1.1):
                f.write(json.dumps(dict(base, H=H, W=W, D_TE0=round(400 * W - 300 * H, 1))) + "\n")
    env = dict(os.environ, PYTHONPATH=str(root), MPLBACKEND="Agg", TARANG_FIGURE_FORMATS="png,pdf",
               TARANG_FIGURE_DPI="60", TARANG_FIGURE_STYLE="screen", TARANG_FIGURE_DATA="1")
    r = subprocess.run([sys.executable, "-m", "tarang.waveguide.dispersion_mapping.maps",
                        "--data", str(data), "--plots-dir", str(tmp_path / "plots")],
                       cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr
    names = {p.name for p in (tmp_path / "plots").iterdir()}
    assert {"design_map.png", "design_map.pdf", "design_map_data.csv",
            "design_map_meta.json"} <= names


def test_grid_can_be_snapped_to_the_core_edges():
    from tarang.waveguide.dispersion_mapping import core
    for steps in (50, 70):
        g = core.build_sim(0.7, 1.0, steps=steps, snap=True).grid.boundaries
        assert min(abs(np.array(g.y) - 0.5)) < 1e-6 and min(abs(np.array(g.y) + 0.5)) < 1e-6
        assert min(abs(np.array(g.z) - 0.7)) < 1e-6 and min(abs(np.array(g.z))) < 1e-6
    g = core.build_sim(0.7, 1.0, steps=50, snap=False).grid.boundaries
    assert min(abs(np.array(g.z) - 0.7)) > 1e-3                # unsnapped: edge inside a cell


def test_landscape_legend_keys_are_neutral(tmp_path, figs):
    from tarang.design import landscapes
    rng = np.random.default_rng(0)
    pts = [dict(x=float(a), y=float(b), z=float(np.exp(-(a - 5) ** 2 - (b - 0.5) ** 2)))
           for a, b in zip(rng.uniform(0, 10, 20), rng.uniform(0, 1, 20))]
    import matplotlib.pyplot as plt
    captured = {}
    real = plt.Axes.legend

    def spy(self, *a, **kw):
        captured["handles"] = kw.get("handles")
        return real(self, *a, **kw)

    import unittest.mock as um
    with um.patch.object(plt.Axes, "legend", spy):
        landscapes.draw(tmp_path / "l.png", [dict(points=pts, title="t", xlabel="x", ylabel="y",
                                                   chosen=(5, 0.5))])
    h = captured["handles"]
    assert [x.get_label() for x in h] == ["simulated points", "chosen design"]
    assert h[0].get_markerfacecolor() == "0.6" and h[1].get_color() == "red"


# ---- the mesh band ------------------------------------------------------------------------ #
def test_mesh_band_is_the_median_and_the_largest_distance():
    from tarang.waveguide.dispersion_mapping.convergence import mesh_band
    snapped = {50: 373.99, 60: 374.03, 70: 376.81, 80: 373.77, 90: 370.88, 100: 373.67}
    b = mesh_band(snapped)
    assert b["value"] == pytest.approx(373.88) and b["band"] == pytest.approx(3.0)
    assert b["meshes"] == [50, 60, 70, 80, 90, 100]
    unsnapped = {50: 378.68, 60: 373.64, 70: 361.6, 80: 372.48, 90: 374.56}
    assert mesh_band(unsnapped)["band"] == pytest.approx(12.04)
    assert mesh_band({70: 1.0})["band"] is None


def test_convergence_plot_shows_the_band_and_writes_its_table(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    data = tmp_path / "conv.jsonl"
    with open(data, "w", encoding="utf-8") as f:
        for s, D, n, g in ((50, 373.99, 1.7092, 2.2228), (70, 376.81, 1.71118, 2.2215),
                           (90, 370.88, 1.7075, 2.2237)):
            f.write(json.dumps(dict(H=0.7, W=1.0, wl0=1.55, steps=s, snap=True, bend_radius=None,
                                    D_TE0=D, neff_TE=[n], ng_TE0=g, guided_TE=1)) + "\n")
    env = dict(os.environ, PYTHONPATH=str(root), MPLBACKEND="Agg", TARANG_FIGURE_FORMATS="png",
               TARANG_FIGURE_DPI="50")
    r = subprocess.run([sys.executable, "-m", "tarang.waveguide.dispersion_mapping.convergence",
                        "--data", str(data), "--plot-only", "--plots-dir", str(tmp_path / "p")],
                       cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr
    rows = list(csv.DictReader(open(tmp_path / "p" / "mesh_band.csv", encoding="utf-8")))
    assert len(rows) == 1 and rows[0]["snapped"] == "True"
    assert float(rows[0]["D"]) == pytest.approx(373.99) and float(rows[0]["D_band"]) == pytest.approx(3.11)
    meta = json.loads((tmp_path / "p" / "conv_H700_W1000_meta.json").read_text(encoding="utf-8"))
    assert any("median ± spread" in s for s in meta["panels"][0]["series"])


def test_snap_and_ladder_settings(tmp_path):
    from tarang.design.config import DesignConfig
    from tarang.design.stages import Context, settings_key
    c = DesignConfig()
    assert c.cross_section.snap_grid is True and c.cross_section.mesh_ladder == [50, 60, 70, 80, 90]
    k = settings_key(c, "cross_section")
    c.cross_section.mesh_ladder = [50, 70]
    assert settings_key(c, "cross_section") == k               # ladder: --redo to apply
    c.cross_section.snap_grid = False
    assert settings_key(c, "cross_section") != k               # snapping changes the numbers
    from tarang.design.ui import Prompter
    ctx = Context(DesignConfig(), Prompter(interactive=False), tmp_path)
    assert ctx.stack_args()[-3:] == ["--snap", "--snap-sides", 2]       # two-sided (v0.1.5)
    ctx.cfg.cross_section.snap_grid = False
    assert "--snap" not in ctx.stack_args()


def test_paper_style_moves_titles_to_the_caption_and_keeps_legends_readable(tmp_path, figs,
                                                                             monkeypatch):
    monkeypatch.setitem(figs.SETTINGS, "style", "paper")
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for k in range(4):
        axs[1].plot([0, 1], [k, k + 1], label=f"series {k}")
    axs[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
    axs[0].set_title("left panel")
    fig.suptitle("whole figure")
    for i in range(20):
        axs[0].text(0.5, 0.5, str(i))                 # dense in-cell annotations of a map
    figs.save(fig, tmp_path / "figures" / "p.png", close=False)
    meta = json.loads((tmp_path / "figures_paper" / "p_meta.json").read_text(encoding="utf-8"))
    assert meta["caption"] == {"figure": "whole figure", "(a)": "left panel"}   # v0.1.1: letters
    assert axs[0].get_title() == "" and not fig._suptitle.get_visible()
    assert not any(t.get_visible() for t in axs[0].texts if t.get_gid() != "tarang-letter")
    assert axs[1].get_legend()._ncols == 2
    sizes = [t.get_fontsize() for t in fig.findobj(plt.Text) if t.get_visible() and t.get_text()]
    assert min(sizes) >= 6.0 and max(sizes) <= 8.5
    plt.close(fig)
