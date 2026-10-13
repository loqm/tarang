"""Construct-and-validate tests for every geometry builder (Phase 4). Local only.

    python -m pytest tests/test_geometries.py -q

Each builder's Simulation must pass Tidy3D's own ``validate_pre_upload()``, the
check that runs on the server before a job starts. A geometry that fails it would
be rejected after upload and waste time. The tests also check that every monitor
the analysis methods read by default is actually created by the matching builder,
and the analytic FSR helpers.
"""
import numpy as np
import pytest
import tidy3d as td

pytestmark = pytest.mark.filterwarnings("ignore:The supplied partial-ring geometry")

WL = 1.55


def _m(n):
    return td.Medium(permittivity=n ** 2)


SIN, SIO2, SI, AIR = _m(1.996), _m(1.444), _m(3.476), _m(1.0)
MESH = dict(fine_steps_per_wvl=8, coarse_steps_per_wvl=6)
RING = dict(radius=10.0, gap=0.3, wg_width=0.8, wg_height=0.7, core_medium=SIN,
            box_medium=SIO2, wavelength_um=WL, **MESH)


def _ridge(active):
    import geometries.ridge_waveguide as g
    f = g.get_active_simulation if active else g.get_passive_simulation
    return f(width=0.9, height=0.7, length=3.0, core_medium=SIN, box_medium=SIO2,
             wavelength_um=WL, fine_steps_per_wvl=12, coarse_steps_per_wvl=8)


def _edge(active, out=True):
    import geometries.edge_coupler as g
    kw = dict(tip_width=0.2, taper_length=10.0, wg_width=0.8, wg_height=0.7, core_medium=SIN,
              box_medium=SIO2, wavelength_um=WL, top_cladding_medium=AIR, **MESH)
    if not active:
        return g.get_passive_simulation(**kw)
    return g.get_active_simulation(**kw, out_coupling=out, in_coupling=not out)


def _full(active):
    import geometries.full_ring_resonator as g
    return (g.get_active_simulation if active else g.get_passive_simulation)(**RING)


def _half(active):
    import geometries.half_ring_resonator as g
    return (g.get_active_simulation if active else g.get_passive_simulation)(**RING)


def _partial(active):
    import geometries.partial_ring_resonator as g
    return (g.get_active_simulation if active else g.get_passive_simulation)(
        span_angle_rad=np.pi / 4, **RING)


def _arc(active):
    import geometries.resonator_arc as g
    return (g.get_active_simulation if active else g.get_passive_simulation)(
        wavelength_um=WL, radius=10.0, wg_width=0.8, wg_height=0.7, core_medium=SIN,
        box_medium=SIO2, top_cladding_medium=AIR, bend_angle_deg=45.0, **MESH)


def _gc(is_3d=False, handle=False):
    from grating_coupler.grating_coupler import GratingConfig, GratingCoupler
    cfg = GratingConfig(wavelength_um=WL, theta_deg=-12.0, mfd=10.4, source_x=3.84, source_gap=2.0,
                        core_height=0.7, top_cladding_height=2.0, box_height=4.0, handle_height=2.0,
                        etch_depth=0.35, period=0.9, ff=0.5, n_teeth=12, w_grating=10.4,
                        pol_angle=np.pi / 2, buffer=3.0)
    return GratingCoupler.from_config(cfg, core_medium=SIN, box_medium=SIO2,
                                      handle_medium=SI if handle else None,
                                      top_cladding_medium=AIR, is_3d=is_3d, **MESH)


BUILDS = {
    "ridge_passive": lambda: _ridge(False), "ridge_active": lambda: _ridge(True),
    "edge_passive": lambda: _edge(False),
    "edge_out_coupling": lambda: _edge(True, out=True), "edge_in_coupling": lambda: _edge(True, out=False),
    "full_ring_passive": lambda: _full(False), "full_ring_active": lambda: _full(True),
    "half_ring_passive": lambda: _half(False), "half_ring_active": lambda: _half(True),
    "partial_ring_passive": lambda: _partial(False), "partial_ring_active": lambda: _partial(True),
    "arc_passive": lambda: _arc(False), "arc_active": lambda: _arc(True),
    "grating_2d": lambda: _gc(), "grating_2d_si_handle": lambda: _gc(handle=True),
    "grating_3d": lambda: _gc(is_3d=True),
}


@pytest.mark.parametrize("name", list(BUILDS))
def test_builder_passes_tidy3d_pre_upload_validation(name):
    dev = BUILDS[name]()
    passive = "passive" in name              # passive builds are for mode solving / plots
    dev.sim.validate_pre_upload(source_required=not passive)   # raises if Tidy3D would reject it
    if not passive:
        assert dev.sim.sources, f"{name}: an active simulation needs a source"


# Monitors that analysis methods read by name, per builder.
EXPECTED_MONITORS = {
    "edge_out_coupling": {"flux_fiber", "field_xy", "field_xz"},     # get_efficiency, plot_fields
    "edge_in_coupling": {"mode_transmission", "field_xy", "field_xz"},
    "full_ring_active": {"coupling_field_map"},                      # plot_field default
    "half_ring_active": {"coupling_field_map"},
    "partial_ring_active": {"coupling_field_map"},
    "arc_active": {"Mode Monitor", "Mode Monitor In", "field_map"},  # get_efficiency, plot_field
    "grating_2d": {"mode monitor"},                                  # get_efficiency
    "grating_3d": {"mode monitor"},
}


@pytest.mark.parametrize("name", list(EXPECTED_MONITORS))
def test_analysis_monitor_names_exist_in_builder(name):
    names = {m.name for m in BUILDS[name]().sim.monitors}
    missing = EXPECTED_MONITORS[name] - names
    assert not missing, f"{name}: analysis code reads {missing}, builder defines {sorted(names)}"


def test_grating_handle_layer_present_only_when_requested():
    def media(dev):
        return {s.medium for s in dev.sim.structures}
    assert SI in media(_gc(handle=True)) and SI not in media(_gc(handle=False))


def test_ring_conversions_keep_geometry():
    from ring_resonator import HalfRingResonator, PartialRingResonator, RingResonator
    full = _full(True)
    assert isinstance(full, RingResonator)
    half = full.to_half_ring()
    assert isinstance(half, HalfRingResonator) and isinstance(half.to_full_ring(), RingResonator)
    part = _partial(True)
    assert isinstance(part, PartialRingResonator)
    for dev in (half, part.to_half_ring(), part.to_full_ring()):
        assert (dev.radius, dev.gap, dev.wg_width) == (10.0, 0.3, 0.8)


def test_fsr_formulas():
    """FSR = lambda^2 / (n_g L) in wavelength and c / (n_g L) in frequency, L = 2 pi R."""
    ring = _full(False)
    n_g, R = 2.1, 10.0
    L_um = 2 * np.pi * R
    assert ring.compute_fsr_nm(n_g) == pytest.approx(WL ** 2 / (n_g * L_um) * 1e3, rel=1e-12)
    assert ring.compute_fsr_ghz(n_g) == pytest.approx(299792458e6 / (n_g * L_um) / 1e9, rel=1e-9)
    # Consistency: d(lambda) = lambda^2 d(nu) / c
    fsr_nm_from_ghz = WL ** 2 * ring.compute_fsr_ghz(n_g) * 1e9 / 299792458e6 * 1e3
    assert ring.compute_fsr_nm(n_g) == pytest.approx(fsr_nm_from_ghz, rel=1e-9)


def test_monitor_data_explains_what_is_missing():
    from utilities.cloud_run import MonitorDataError, monitor_data
    dev = _edge(True, out=True)
    data = td.SimulationData(simulation=dev.sim, data=())
    with pytest.raises(MonitorDataError, match="not defined in this simulation"):
        monitor_data(data, "no_such_monitor")
    with pytest.raises(MonitorDataError, match="contain no data"):
        monitor_data(data, "field_xy")       # defined, but this result has no data
    with pytest.raises(KeyError):            # old callers catching KeyError ...
        monitor_data(data, "field_xy")
    with pytest.raises(ValueError):          # ... or ValueError still work
        monitor_data(data, "field_xy")
