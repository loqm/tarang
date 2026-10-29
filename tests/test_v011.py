"""v0.1.1: tidier paper-style figures (panel letters, legends off the data, notes and long
titles in the caption, font sizes by role), grating peaks at the band edge marked, the
grating efficiency against the fibre angle, wall time and version per stage, the 780 nm
example's height and width map, and the pyLLE script's soliton step, held single soliton
and D_int cross-check."""
from __future__ import annotations

import importlib.util
import io
import json
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def figs(monkeypatch):
    from tarang import figures
    for k, v in dict(formats=["png"], dpi=60, style="paper", data=True).items():
        monkeypatch.setitem(figures.SETTINGS, k, v)
    return figures


def _plt():
    from tarang.design import visuals
    return visuals._plt()


def _ctx(tmp_path, **cfg_changes):
    from tarang.design.config import DesignConfig
    from tarang.design.stages import Context
    from tarang.design.ui import Prompter
    cfg = DesignConfig()
    cfg.run.band_um = 0.2
    for dotted, value in cfg_changes.items():
        sec, key = dotted.split("__")
        setattr(getattr(cfg, sec), key, value)
        cfg.given.add(f"{sec}.{key}")
    ctx = Context(cfg, Prompter(interactive=False, out=io.StringIO()), tmp_path / "run")
    ctx.state["stages"] = {"cross_section": {"results": dict(height_um=0.7, width_um=1.0)}}
    return ctx


# ---- paper style ------------------------------------------------------------------------ #
def test_paper_panels_get_letters_in_reading_order_and_short_titles_stay(tmp_path, figs):
    plt = _plt()
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(2, 3)
    tall = fig.add_subplot(gs[:, 2])                  # a taller panel on the right
    a = fig.add_subplot(gs[0, 0])
    b = fig.add_subplot(gs[0, 1])
    c = fig.add_subplot(gs[1, 0])
    a.set_title("|Ex|")
    figs.keep_title(b)
    b.set_title("a long title kept on purpose")
    c.set_title("a long title for the caption")
    tall.set_title("tall")
    for ax in (a, b, c, tall):
        ax.plot([0, 1], [0, 1])
    figs.save(fig, tmp_path / "figures" / "letters.png", close=False)
    assert [ax for ax in figs._panels(fig)] == [a, b, tall, c]   # rows by overlap, then x
    letters = {ax: [t.get_text() for t in ax.texts if t.get_gid() == "tarang-letter"]
               for ax in (a, b, c, tall)}
    assert letters == {a: ["(a)"], b: ["(b)"], tall: ["(c)"], c: ["(d)"]}
    assert a.get_title() == "|Ex|" and b.get_title() == "a long title kept on purpose"
    assert c.get_title() == "" and tall.get_title() == "tall"
    meta = json.loads((tmp_path / "figures_paper" / "letters_meta.json").read_text(encoding="utf-8"))
    assert meta["caption"] == {"(d)": "a long title for the caption"}
    plt.close(fig)


def test_paper_font_sizes_follow_their_role(tmp_path, figs):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(3, 2))             # small: everything would grow
    ax.plot([0, 1], [0, 1], label="line")
    ax.set(xlabel="x label", ylabel="y label")
    ax.legend()
    ax.text(0.5, 0.2, "note", transform=ax.transAxes)
    figs.save(fig, tmp_path / "f.png", close=False)
    lo, hi = figs.PAPER_FONTS["tick"]
    assert all(lo <= t.get_fontsize() <= hi for t in ax.get_xticklabels() + ax.get_yticklabels())
    assert figs.PAPER_FONTS["label"][0] <= ax.xaxis.label.get_fontsize() <= figs.PAPER_FONTS["label"][1]
    assert ax.get_legend().get_texts()[0].get_fontsize() <= figs.PAPER_FONTS["legend"][1]
    plt.close(fig)


def _legend_hits(figs, ax):
    ax.figure.canvas.draw()
    r = ax.figure.canvas.get_renderer()
    return figs._clashes(ax, ax.get_legend().get_window_extent(r), r)


def test_a_legend_on_the_data_moves_to_a_free_corner(tmp_path, figs):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.linspace(0, 1, 200)
    ax.plot(x, 1 - x, label="falling")                 # upper left is crowded, lower left free
    ax.plot(x, 0.9 - 0.8 * x, label="also falling")
    ax.legend(loc="upper left")
    figs.save(fig, tmp_path / "corner.png", close=False)
    leg = ax.get_legend()
    assert _legend_hits(figs, ax) == 0
    assert leg.get_window_extent(fig.canvas.get_renderer()).overlaps(ax.bbox)   # still inside
    plt.close(fig)


def test_a_legend_anchored_a_rounding_error_below_its_panel_is_not_moved_under_it(tmp_path, figs):
    # on Windows the anchor of a legend inside its panel came back as y = -1e-16 in axes units
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([0, 1], [0, 1], label="rising")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -1e-12, 1, 1))
    figs.save(fig, tmp_path / "anchor.png", close=False)
    fig.canvas.draw()
    assert ax.get_legend().get_window_extent(fig.canvas.get_renderer()).overlaps(ax.bbox)
    plt.close(fig)


def test_a_legend_with_no_free_corner_goes_under_the_panel_clear_of_its_labels(tmp_path, figs):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 5))
    rng = np.random.default_rng(1)
    for k in range(4):                                 # data everywhere
        ax.plot(np.linspace(0, 1, 300), rng.random(300), label=f"noise {k}")
    ax.set(xlabel="wavelength (nm)")
    ax.legend(loc="center")
    figs.save(fig, tmp_path / "under.png", close=False)
    fig.canvas.draw()                                  # positions at the screen dpi again
    r = fig.canvas.get_renderer()
    leg = ax.get_legend().get_window_extent(r)
    assert leg.y1 <= ax.bbox.y0                        # under the panel
    assert leg.y1 <= ax.xaxis.label.get_window_extent(r).y0 + 0.5   # and under its x label
    plt.close(fig)


def test_notes_beside_a_panel_and_spilling_notes_go_to_the_caption(tmp_path, figs):
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for ax in axs:
        ax.plot([0, 1], [0, 1])
    axs[0].text(0.0, -0.3, "a long note written under the panel, explaining the dots",
                transform=axs[0].transAxes)
    axs[1].text(0.6, 0.5, "a note far too long to fit inside this small panel at any size",
                transform=axs[1].transAxes)
    axs[1].text(0.5, 0.5, "fits", transform=axs[1].transAxes)
    fig.text(0.5, 0.01, "figure-level remark")
    figs.save(fig, tmp_path / "figures" / "notes.png", close=False)
    meta = json.loads((tmp_path / "figures_paper" / "notes_meta.json").read_text(encoding="utf-8"))
    cap = meta["caption"]
    assert cap["notes"] == ["figure-level remark"]
    assert cap["note (a)"] == ["a long note written under the panel, explaining the dots"]
    assert cap["note (b)"] == ["a note far too long to fit inside this small panel at any size"]
    assert [t.get_text() for t in axs[1].texts if t.get_visible() and t.get_gid() != "tarang-letter"] \
        == ["fits"]
    plt.close(fig)


def test_a_legend_under_a_panel_wider_than_the_figure_gets_one_column(tmp_path, figs):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for k in range(4):
        ax.plot([0, 1], [k, k], label=f"a fairly long legend label number {k}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    figs.save(fig, tmp_path / "wide.png", close=False)
    assert fig.get_size_inches()[0] == pytest.approx(3.4)
    assert ax.get_legend()._ncols == 1
    plt.close(fig)


def test_legends_under_panels_of_a_row_fit_their_share_and_bars_count_as_data(tmp_path, figs):
    plt = _plt()
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for ax in axs:
        for k in range(2):
            ax.plot([0, 1], [k, k], label=f"a long legend label for curve {k}")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    figs.save(fig, tmp_path / "row.png", close=False)
    assert [ax.get_legend()._ncols for ax in axs] == [1, 1, 1]
    fig2, ax = plt.subplots(figsize=(8, 5))
    ax.bar([0, 1], [1.0, 0.4], width=0.9, label="bars")
    ax.legend(loc="upper left")                        # on the tall bar; upper right is free
    figs.save(fig2, tmp_path / "bars.png", close=False)
    assert _legend_hits(figs, ax) == 0
    plt.close("all")


def test_screen_style_is_untouched(tmp_path, figs, monkeypatch):
    monkeypatch.setitem(figs.SETTINGS, "style", "screen")
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].set_title("a long title that stays on screen")
    for ax in axs:
        ax.plot([0, 1], [0, 1], label="l")
    axs[0].legend(loc="center")
    figs.save(fig, tmp_path / "s.png", close=False)
    assert axs[0].get_title() == "a long title that stays on screen"
    assert not any(t.get_gid() == "tarang-letter" for ax in axs for t in ax.texts)
    plt.close(fig)


# ---- grating: peaks at the band edge ------------------------------------------------------ #
ROWS = [dict(step="A", dim="2D", period_um=1.10, fill_factor=0.7, source_x_um=5.2,
             efficiency_at_target=0.35, peak_efficiency=0.36, peak_wavelength_um=1.552,
             peak_inside=True),
        dict(step="A", dim="2D", period_um=1.34, fill_factor=0.5, source_x_um=5.2,
             efficiency_at_target=0.01, peak_efficiency=0.06, peak_wavelength_um=1.650,
             peak_inside="False"),                    # as read back from the CSV
        dict(step="A", dim="2D", period_um=1.20, fill_factor=0.6, source_x_um=5.2,
             efficiency_at_target=0.05, peak_efficiency=0.10, peak_wavelength_um=1.573,
             peak_inside=True)]


def test_band_edge_peaks_are_noted_and_marked(tmp_path, figs, monkeypatch):
    from tarang.design import landscapes, stages, visuals
    note = stages._band_edge_note(ROWS[:2] + [dict(ROWS[2], peak_inside=False)], 1.55, 0.1)
    assert note and note[0].startswith("2 of the 3 step-A designs peak at the end of the simulated "
                                       "band (1450-1650 nm)")
    assert stages._band_edge_note(ROWS[:1], 1.55, 0.1) == []
    assert stages._best_grating([dict(r, peak_inside=landscapes._true(r["peak_inside"]))
                                 for r in ROWS])["period_um"] == 1.10
    pts = landscapes.uniform_points(ROWS)
    assert [p["peak_inside"] for p in sorted(pts, key=lambda p: p["period_um"])] == [True, True, False]
    monkeypatch.setitem(figs.SETTINGS, "style", "screen")
    plt = _plt()
    shown = []
    real = visuals._save
    monkeypatch.setattr(visuals, "_save", lambda fig, path: shown.append(
        [t.get_text() for ax in fig.axes for t in ax.texts]) or real(fig, path))
    visuals.grating_landscape(ROWS, ROWS[0], tmp_path / "5b.png", 1.55)
    assert "≥6%\n≥1650" in shown[0] and "1%\n≥1650" in shown[0] and "36%\n1552" in shown[0]
    landscapes.draw_uniform(tmp_path / "5g.png", ROWS, ROWS[0], 1.55)
    plt.close("all")


# ---- grating: efficiency against the fibre angle ------------------------------------------ #
def test_grating_equation_moves_the_peak_with_the_angle():
    from tarang.design.checks import expected_peak_nm
    # λ = Λ (n_avg − sin θ) for a beam towards the waveguide: 12° -> 14° at Λ = 1.1 µm
    d = expected_peak_nm(1550.0, 1.1, 12.0, 14.0, "towards") - 1550.0
    assert d == pytest.approx(-1100 * (math.sin(math.radians(14)) - math.sin(math.radians(12))))
    assert d == pytest.approx(-37.4, abs=0.1)
    assert expected_peak_nm(1550.0, 1.1, 12.0, 14.0, "away") - 1550.0 == pytest.approx(-d)
    assert expected_peak_nm(1550.0, 1.1, 12.0, 12.0, "towards") == 1550.0


def test_angle_sweep_devices_skip_the_design_angle_and_keep_older_keys(tmp_path):
    from tarang.design import checks
    from tarang.design.stages import settings_key
    gc = dict(period_um=1.1, fill_factor=0.75, source_x_um=6.4, etch_depth_um=0.7,
              fiber_angle_deg=12.0, fiber_tilt="towards")
    ctx = _ctx(tmp_path, checks__angle_sweep_deg=[8.0, 12.0, 16.0, 8.0])
    devs = checks.angle_sweep_variants(ctx, gc)
    assert sorted(devs) == [8.0, 16.0]
    assert devs[16.0].config.theta_deg == 16.0 and devs[16.0].config.period == 1.1
    assert checks.angle_sweep_variants(_ctx(tmp_path), gc) == {}
    # the new setting left empty keeps the checks key of runs made before it existed
    plain = _ctx(tmp_path)
    key = settings_key(plain.cfg, "checks")
    plain.cfg.checks.angle_sweep_deg = [10.0]
    assert settings_key(plain.cfg, "checks") != key


def test_angle_sweep_rows_and_figure(tmp_path, monkeypatch, figs):
    from tarang.design import checks
    monkeypatch.setitem(figs.SETTINGS, "style", "screen")
    gc = dict(period_um=1.1, fill_factor=0.75, source_x_um=6.4, etch_depth_um=0.7,
              fiber_angle_deg=12.0, fiber_tilt="towards")
    ctx = _ctx(tmp_path, checks__angle_sweep_deg=[10.0, 14.0])
    wl = np.linspace(1.45, 1.65, 201)

    def spectrum(peak):
        return wl, 0.36 * np.exp(-((wl - peak) / 0.03) ** 2)

    class Dev:
        def __init__(self, ang):
            self.ang = ang

        def get_spectrum(self, folder):
            return spectrum(1.55 - 1.1 * (math.sin(math.radians(self.ang))
                                          - math.sin(math.radians(12.0))))
    rows = checks.angle_sweep_rows(ctx, gc, {10.0: Dev(10.0), 14.0: Dev(14.0)}, spectrum(1.55))
    assert [r["angle_deg"] for r in rows] == [10.0, 12.0, 14.0]
    assert [r["design_angle"] for r in rows] == [False, True, False]
    for r in rows:                                   # this toy obeys the grating equation
        assert r["peak_wavelength_nm"] == pytest.approx(r["expected_peak_nm"], abs=0.5)
    assert rows[1]["efficiency_at_target"] == pytest.approx(0.36, rel=1e-3)
    out = checks.draw_angle_sweep(ctx, gc, rows)
    assert Path(out).name == "7d_grating_angle.png" and Path(out).exists()


# ---- wall time and version per stage --------------------------------------------------- #
def test_stage_record_and_the_report_table():
    from tarang import __version__
    from tarang.design import stage_record
    from tarang.design.report import stage_times
    rec = stage_record(12.345)
    assert rec["seconds"] == 12.3 and rec["tarang"] == __version__ and "T" in rec["finished"]
    state = {"stages": {"cross_section": {"results": {}, "seconds": 600.0, "tarang": "0.1.1",
                                          "finished": "2026-10-29T10:00:00"},
                        "gap": {"results": {}, "seconds": 75.0, "tarang": "0.1.1",
                                "finished": "2026-10-29T10:05:00"},
                        "ring": {"results": {}}}}              # made before v0.1.1
    credits = [dict(stage="gap", estimate_FC="0.25"), dict(stage="gap", estimate_FC=0.5),
               dict(stage="gap", estimate_FC="?")]
    rows = stage_times(state, credits)
    assert rows == [dict(stage="cross_section", wall_time="10.0 min", credits_FC="0.000",
                         tarang="0.1.1", finished="2026-10-29 10:00:00"),
                    dict(stage="gap", wall_time="75 s", credits_FC="0.750", tarang="0.1.1",
                         finished="2026-10-29 10:05:00")]


# ---- examples -------------------------------------------------------------------------- #
def test_780_example_maps_heights_around_a_350_nm_film():
    from tarang.design.config import DesignConfig
    c = DesignConfig.load(ROOT / "examples" / "design_sin780.toml")
    assert c.cross_section.heights_um[0] == 0.3 and c.cross_section.heights_um[-1] == 0.45
    assert 0.35 in c.cross_section.heights_um
    assert c.cross_section.widths_um[0] == 0.4 and c.cross_section.widths_um[-1] == 0.9
    assert (c.grating.fiber_angle_deg, c.grating.mfd_um) == (12.0, 5.0)
    assert c.checks.angle_sweep_deg == [8.0, 10.0, 14.0, 16.0]


# ---- pyLLE script (no pyLLE needed) ------------------------------------------------------ #
def _pylle_script():
    spec = importlib.util.spec_from_file_location(
        "run_pylle", ROOT / "validation" / "pylle_soliton" / "run_pylle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_soliton_step_is_the_middle_of_the_steady_step():
    rp = _pylle_script()
    rng = np.random.default_rng(0)
    p = np.r_[np.zeros(100), np.linspace(1, 40, 100), 62.0, 40 + 10 * rng.random(50),
              np.linspace(56, 55, 100), np.linspace(50, 0, 5), np.zeros(50)]
    k = rp.soliton_step(p)
    assert 290 <= k <= 310                           # the steady step runs over 251-350


def test_normalised_pump_and_soliton_limit():
    rp = _pylle_script()
    fsr, radius, gamma = 200e9, 107.5e-6, 1.3
    k2, kex2 = 400e6, 160e6
    inp = dict(data=dict(dispersion=dict(FSR_Hz=fsr, D2_2pi_Hz=15e6), radius_m=radius,
                         gamma_per_W_m=gamma),
               regime=dict(kappa_2pi_Hz=k2, kappa_ex_2pi_Hz=kex2))
    n = rp.normalised(inp, 0.2)
    t_r = 1 / fsr
    theta, alpha = 2 * math.pi * kex2 * t_r, math.pi * k2 * t_r
    f2 = theta * 0.2 * gamma * 2 * math.pi * radius / alpha ** 3
    assert n["f2"] == pytest.approx(f2) and n["zeta_max"] == pytest.approx(math.pi ** 2 * f2 / 8)
    assert rp.normalised(dict(data={}, regime={}), 0.2) == {}


def test_pulse_width_interpolated_and_from_the_sech_soliton():
    rp = _pylle_script()
    theta = np.linspace(-np.pi, np.pi, 401)
    a = 9.85
    assert rp.fwhm(theta, 1 / np.cosh(a * (theta - 0.2)) ** 2) == pytest.approx(1.7627 / a, rel=5e-3)
    nrm = dict(kappa_2pi_Hz=496e6, D2_2pi_Hz=15.9e6, t_R_s=5e-12)
    # sech²(φ √(ζ κ/D2)): FWHM 1.7627 / √(ζ κ/D2) of the round trip
    want = 1.7627 / math.sqrt(3.11 * 496 / 15.9) / (2 * math.pi) * 5e-12 * 1e15
    assert rp.expected_fwhm_fs(nrm, 3.11) == pytest.approx(want)


# ---- literature comparison (no mode solves) ------------------------------------------- #
def _literature():
    spec = importlib.util.spec_from_file_location(
        "lit_compare", ROOT / "validation" / "literature" / "compare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_d2_from_beta2_and_the_fsr():
    lit = _literature()
    # the 1550 nm design: D = 362 ps/(nm km) -> β2 = -D λ²/(2π c), n_g 2.22, FSR 200 GHz
    beta2 = -362e-6 * (1.55e-6) ** 2 / (2 * math.pi * 299792458.0) * 1e27      # ps²/km
    assert lit.d2_2pi_MHz(beta2, 2.22, 200.0) == pytest.approx(15.7, abs=0.1)
    assert lit.d2_2pi_MHz(-beta2, 2.22, 200.0) < 0                  # normal GVD: D2 < 0


def test_papers_table_is_complete():
    lit = _literature()
    rows = lit.read_papers()
    kinds = {"measured", "simulated", "window", "sign"}
    assert len(rows) == 7 and len({r["id"] for r in rows}) == 7
    for r in rows:
        assert r["kind"] in kinds and r["mode"] in ("TE00", "TM00")
        assert r["cladding"] in ("air", "sio2", "oxide_sides_air_top")
        assert r["source"].startswith("arXiv:") and float(r["pump_nm"]) > 0
        if r["kind"] == "window":
            assert float(r["window_lo_um"]) < float(r["window_hi_um"])
        else:
            assert float(r["width_um"]) > 0 and float(r["fsr_GHz"]) > 0 and r["d2_2pi_MHz"]
