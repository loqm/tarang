"""v0.2.0, phase C: the full-vectorial nonlinear coefficient (Afshar and Monro, Opt. Express
17, 2298 (2009), Eq. 39) with each material's Kerr index, and its use in the LLE inputs and
the threshold."""
from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

pytest.importorskip("tidy3d")

from tarang.waveguide.dispersion_mapping import nonlinear as nl    # noqa: E402


def test_weak_guidance_reduces_to_the_scalar_form():
    """A large, weakly guiding silica core in silica: the vectorial gamma equals
    2 pi n2 / (lambda A_eff) (here within 1 %)."""
    r = nl.overlaps(4.0, 4.0, wl0=1.55, steps=20, box=4.0, core="sio2", substrate="sio2",
                    cladding="sio2", n_shift=0.005, cladding_height=4.0, num_modes=2)
    g, shares = nl.gamma_from(r["G_per_m3"], {k: 2.6e-20 for k in r["G_per_m3"]})
    assert g == pytest.approx(nl.scalar_gamma(2.6e-20, 1.55, r["aeff_um2"]), rel=0.01)
    assert sum(shares.values()) == pytest.approx(1.0)


def test_high_contrast_si3n4_gamma_exceeds_the_scalar_form():
    """Air-clad 0.7 x 1.2 um Si3N4 at 1550 nm: the field's longitudinal part and the n^2
    weighting raise gamma above the scalar estimate (Afshar and Monro's finding for
    sub-wavelength cores); silica's n2 adds about 0.1 %, air none."""
    r = nl.overlaps(0.7, 1.2, wl0=1.55, steps=30, snap=True, snap_sides=1)
    n2 = {"core": 2.4e-19, "box": 2.6e-20, "cladding": 0.0}
    g, shares = nl.gamma_from(r["G_per_m3"], n2)
    ratio = g / nl.scalar_gamma(2.4e-19, 1.55, r["aeff_um2"])
    assert 1.2 < ratio < 1.4
    assert shares["core"] > 0.99 and shares["cladding"] == 0.0 and 0 < shares["box"] < 0.01


def test_gamma_from_splits_by_material():
    g, sh = nl.gamma_from({"core": 2.0e18, "box": 1.0e17}, {"core": 1e-19, "box": 2e-20})
    assert g == pytest.approx(0.2 + 0.002)
    assert sh["core"] == pytest.approx(0.2 / 0.202)


# ---- the design: LLE inputs and threshold ------------------------------------------------ #
def _ctx(tmp_path, fake):
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from test_phase12 import _broadband, _ctx as ctx12
    ctx = ctx12(tmp_path)
    _broadband(ctx)
    return ctx


@pytest.fixture
def fake_overlaps(monkeypatch):
    calls = []

    def fake(H, W, **kw):
        calls.append((H, W, kw))
        return dict(mode="TE00", wl0=kw["wl0"], materials={"core": "si3n4", "box": "sio2",
                                                            "cladding": "air"},
                    G_per_m3={"core": 6.0e18, "box": 5.0e16, "cladding": 1.0e16},
                    aeff_um2=0.80, n_eff=1.75, sides=2)
    monkeypatch.setattr(nl, "overlaps", fake)
    return calls


def test_lle_inputs_use_the_vectorial_gamma(tmp_path, fake_overlaps):
    from tarang.design import lle
    ctx = _ctx(tmp_path, fake_overlaps)
    r = lle.export(ctx)
    data = json.loads((ctx.out / "lle" / "lle_inputs.json").read_text(encoding="utf-8"))
    g = 2.4e-19 * 6.0e18 + 2.6e-20 * 5.0e16            # air: n2 0
    assert data["gamma_model"] == "vectorial" and data["gamma_per_W_m"] == pytest.approx(g)
    assert data["pyLLE"]["res"]["gamma"] == pytest.approx(g)
    g_core = 2 * math.pi * 2.4e-19 / (1.55e-6 * 0.75e-12)     # the cross-section's A_eff
    assert data["gamma_core_only_per_W_m"] == pytest.approx(g_core)
    # threshold: n2 / A_eff replaced by gamma lambda / 2 pi, so it scales as 1 / gamma
    ctx.cfg.assumptions.gamma_model = "core"
    lle.export(ctx)
    core = json.loads((ctx.out / "lle" / "lle_inputs.json").read_text(encoding="utf-8"))
    assert data["coupling"]["critical"]["threshold_W"] == pytest.approx(
        core["coupling"]["critical"]["threshold_W"] * g_core / g)
    assert r["gamma_shares"]["core"] == pytest.approx(2.4e-19 * 6.0e18 / g)


def test_overlaps_are_solved_once_and_reused(tmp_path, fake_overlaps):
    from tarang.design import lle
    ctx = _ctx(tmp_path, fake_overlaps)
    lle.export(ctx)
    lle.export(ctx)
    assert len(fake_overlaps) == 1                       # second time from the saved file
    ctx.cfg.assumptions.n2_m2_w = 3.0e-19                # n2 needs no new solve
    data = lle.export(ctx)
    assert len(fake_overlaps) == 1
    assert data["gamma_per_W_m"] == pytest.approx(3.0e-19 * 6.0e18 + 2.6e-20 * 5.0e16)


def test_gamma_model_is_checked_and_only_reruns_the_local_end():
    from tarang.design.config import DesignConfig, implausible
    from tarang.design.stages import settings_key
    c = DesignConfig()
    c.assumptions.gamma_model = "scalar"
    assert "assumptions.gamma_model" in implausible(c)
    a, b = DesignConfig(), DesignConfig()
    b.assumptions.gamma_model = "core"
    for stage in ("materials", "cross_section", "ring", "gap", "grating", "focus"):
        assert settings_key(a, stage) == settings_key(b, stage)
    assert settings_key(a, "layout") != settings_key(b, "layout")


def test_report_threshold_uses_gamma():
    from tarang.design.report import threshold_mw
    xs = dict(aeff_um2=0.75, n_eff=1.75)
    ring = dict(q_intrinsic=1e6, radius_um=50.0, wavelength_um=1.55)
    p_core = threshold_mw(xs, ring, 2.4e-19)
    g_core = 2 * math.pi * 2.4e-19 / (1.55e-6 * 0.75e-12)
    assert threshold_mw(xs, ring, 2.4e-19, gamma=g_core) == pytest.approx(p_core)
    assert threshold_mw(xs, ring, 2.4e-19, gamma=1.3 * g_core) == pytest.approx(p_core / 1.3)
    assert threshold_mw(xs, {k: v for k, v in ring.items() if k != "wavelength_um"},
                        2.4e-19) is None                 # no silent 1550 nm any more
