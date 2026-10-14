"""v0.1.0 API names and the deprecated pre-v0.1.0 aliases (Phase 3c). Local only.

    python -m pytest tests/test_api_compat.py -q

New names                       Old names (warn, removed in v0.2.0)
------------------------------  ------------------------------------------------
packages grating_coupler,       gratingCoupler, edgeCoupler, ringResonator
  edge_coupler, ring_resonator
classes EdgeCoupler,            edgeCoupler, halfringResonator, ringResonator,
  HalfRingResonator, ...          partialRingResonator, resonatorArc
box_medium / box_height         sub_medium, substrate_medium / substrate_height
  (oxide under the core)          (waveguide, edge coupler, rings, arc)
handle_medium / handle_height   substrate_medium / substrate_height
  (Si wafer, grating coupler)     (grating coupler only)
top_cladding_medium / _height   cladding_medium / cladding_height
wavelength_um                   wavelength0, wavelength0_um
fine/coarse_steps_per_wvl       fine/coarse_min_steps_per_wvl (waveguide)
"""
import ast
import importlib
import inspect
import pathlib
import warnings

import numpy as np
import pytest
import tidy3d as td

import utilities.cloud_run as cr

REPO = pathlib.Path(__file__).resolve().parents[1]

# The short partial ring used here ends inside the domain on purpose; TARANG warns about that.
pytestmark = pytest.mark.filterwarnings("ignore:The supplied partial-ring geometry")


def _m(n):
    return td.Medium(permittivity=n ** 2)


SIN, SIO2, SI, AIR = _m(1.996), _m(1.444), _m(3.476), _m(1.0)
fp = cr.sim_fingerprint


# ---- packages and classes -------------------------------------------------- #
@pytest.mark.parametrize("old,new,sub", [
    ("gratingCoupler", "grating_coupler", "nested_sampler"),
    ("edgeCoupler", "edge_coupler", "optimization_sweep"),
    ("ringResonator", "ring_resonator", "ring_gap_sweep"),
])
def test_old_package_names_alias_the_new_modules(old, new, sub):
    with pytest.warns(DeprecationWarning, match=new):
        old_pkg = importlib.import_module(old)
        importlib.reload(old_pkg)                    # warn even if imported earlier
    old_mod = importlib.import_module(f"{old}.{sub}")
    assert old_mod is importlib.import_module(f"{new}.{sub}")   # same object: isinstance, patching ok


@pytest.mark.parametrize("module,old,new", [
    ("edge_coupler", "edgeCoupler", "EdgeCoupler"),
    ("ring_resonator", "ringResonator", "RingResonator"),
    ("ring_resonator", "halfringResonator", "HalfRingResonator"),
    ("ring_resonator", "partialRingResonator", "PartialRingResonator"),
    ("ring_resonator", "resonatorArc", "ResonatorArc"),
    ("ring_resonator.ring_resonator", "partialRingResonator", "PartialRingResonator"),
])
def test_old_class_names_are_the_new_classes(module, old, new):
    mod = importlib.import_module(module)
    with pytest.warns(DeprecationWarning, match=new):
        cls = getattr(mod, old)
    assert cls is getattr(mod, new)


def test_ring_cache_labels_keep_pre_rename_names():
    """Cache folders and cloud task names embed these; renaming the classes must not change them."""
    from ring_resonator import HalfRingResonator, PartialRingResonator, RingResonator
    assert (HalfRingResonator._cache_label, RingResonator._cache_label,
            PartialRingResonator._cache_label) == ("halfringResonator", "ringResonator",
                                                   "partialRingResonator")


# ---- keyword arguments: old names build the identical simulation ----------- #
def _edge(**kw):
    import geometries.edge_coupler as g
    base = dict(tip_width=0.2, taper_length=10.0, wg_width=0.8, wg_height=0.8, core_medium=SIN,
                top_cladding_medium=AIR, fine_steps_per_wvl=8, coarse_steps_per_wvl=6, out_coupling=True)
    return g.get_active_simulation(**base, **kw)


def _ring(**kw):
    import geometries.partial_ring_resonator as g
    base = dict(radius=20.0, gap=0.3, wg_width=0.8, wg_height=0.8, core_medium=SIN,
                span_angle_rad=np.pi / 6, fine_steps_per_wvl=8, coarse_steps_per_wvl=6)
    return g.get_active_simulation(**base, **kw)


def _arc(**kw):
    import geometries.resonator_arc as g
    base = dict(radius=20.0, wg_width=0.8, wg_height=0.8, core_medium=SIN, bend_angle_deg=30.0,
                fine_steps_per_wvl=8, coarse_steps_per_wvl=6)
    return g.get_active_simulation(**base, **kw)


def _wg(**kw):
    import geometries.ridge_waveguide as g
    return g.get_passive_simulation(width=0.8, height=0.7, length=2.0, core_medium=SIN,
                                    wavelength_um=1.55, **kw)


def _gc_cfg(**kw):
    from grating_coupler.grating_coupler import GratingConfig
    base = dict(wavelength_um=1.55, theta_deg=-12.0, mfd=10.4, source_x=3.84, source_gap=2.0,
                core_height=0.8, box_height=4.0, etch_depth=0.5, period=0.82, ff=0.51, n_teeth=22,
                w_grating=10.4, pol_angle=np.pi / 2, buffer=3.0)
    return GratingConfig(**base, **kw)


def _gc(cfg, **kw):
    from grating_coupler.grating_coupler import GratingCoupler
    return GratingCoupler.from_config(cfg, core_medium=SIN, box_medium=SIO2,
                                      fine_steps_per_wvl=8, coarse_steps_per_wvl=6, **kw)


CASES = {
    "edge": (_edge, dict(box_medium=SIO2, box_height=3.0, wavelength_um=1.55),
             dict(substrate_medium=SIO2, substrate_height=3.0, wavelength0=1.55)),
    "ring": (_ring, dict(box_medium=SIO2, box_height=3.0, wavelength_um=1.55),
             dict(substrate_medium=SIO2, substrate_height=3.0, wavelength0=1.55)),
    "arc": (_arc, dict(box_medium=SIO2, top_cladding_medium=SIO2, wavelength_um=1.55),
            dict(substrate_medium=SIO2, cladding_medium=SIO2, wavelength0=1.55)),
    "waveguide": (_wg, dict(box_medium=SIO2, box_height=3.0, fine_steps_per_wvl=12, coarse_steps_per_wvl=8),
                  dict(sub_medium=SIO2, substrate_height=3.0, fine_min_steps_per_wvl=12,
                       coarse_min_steps_per_wvl=8)),
}


@pytest.mark.parametrize("name", list(CASES))
def test_old_keywords_build_identical_simulation(name):
    build, new_kw, old_kw = CASES[name]
    new = build(**new_kw)
    with pytest.warns(DeprecationWarning) as rec:
        old = build(**old_kw)
    assert fp(new.sim) == fp(old.sim)
    warned = " ".join(str(w.message) for w in rec)
    for o, n in zip(old_kw, new_kw):
        assert f"'{o}'" in warned and f"'{n}'" in warned


def test_grating_substrate_medium_still_means_the_si_handle():
    """The safety-critical case: in the grating coupler the old `substrate_medium` was
    the Si handle, not the oxide. It must map to handle_medium, never box_medium."""
    cfg_new = _gc_cfg(top_cladding_height=2.0, handle_height=8.0)
    with pytest.warns(DeprecationWarning):
        cfg_old = _gc_cfg(cladding_height=2.0, substrate_height=8.0)
    assert cfg_old == cfg_new
    new = _gc(cfg_new, handle_medium=SI, top_cladding_medium=AIR)
    with pytest.warns(DeprecationWarning) as rec:
        old = _gc(cfg_new, substrate_medium=SI, cladding_medium=AIR)
    msgs = " ".join(str(w.message) for w in rec)
    assert "'handle_medium'" in msgs and "'top_cladding_medium'" in msgs
    assert "'box_medium'" not in msgs
    assert fp(old.sim) == fp(new.sim)
    no_handle = _gc(cfg_new, top_cladding_medium=AIR)
    assert fp(no_handle.sim) != fp(new.sim)           # the handle really is in the simulation


def test_old_and_new_keyword_together_is_an_error():
    with pytest.raises(TypeError, match="box_medium"):
        _edge(box_medium=SIO2, substrate_medium=SIO2, wavelength_um=1.55)


# ---- attributes ------------------------------------------------------------ #
def test_old_attribute_names_forward_with_warning():
    edge = _edge(box_medium=SIO2, wavelength_um=1.55)
    with pytest.warns(DeprecationWarning, match="wavelength_um"):
        assert edge.wavelength0 == 1.55
    wg = _wg(box_medium=SIO2)
    with pytest.warns(DeprecationWarning, match="box_medium"):
        assert wg.sub_medium is wg.box_medium
    cfg = _gc_cfg(top_cladding_height=2.0, handle_height=8.0)
    with pytest.warns(DeprecationWarning, match="handle_height"):
        cfg.substrate_height = 6.0
    assert cfg.handle_height == 6.0
    with pytest.raises(AttributeError):
        edge.no_such_attribute


# ---- nested-sampler dictionaries ------------------------------------------- #
def test_nested_sampler_accepts_old_dictionary_keys(tmp_path):
    from grating_coupler.nested_sampler import GratingCouplerBayesianOptimizer as Opt
    fixed = dict(wavelength0_um=1.55, theta_deg=-12.0, mfd=10.4, source_gap=2.0, core_height=0.8,
                 cladding_height=2.0, box_height=4.0, substrate_height=8.0, w_grating=10.4, buffer=3.0)
    mats = dict(core_medium=SIN, box_medium=SIO2, sub_medium=None, cladding_medium=AIR)
    with pytest.warns(DeprecationWarning):
        opt = Opt(fixed_params=fixed, materials=mats, output_dir=tmp_path, show_corner_plots=False)
    assert opt.fixed_params["wavelength_um"] == 1.55 and opt.fixed_params["handle_height"] == 8.0
    assert opt.materials["top_cladding_medium"] is AIR and "sub_medium" not in opt.materials
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)   # the other old keys also warn
        with pytest.raises(ValueError, match="handle_height"):
            Opt(fixed_params={**fixed, "handle_height": 8.0}, materials=mats, output_dir=tmp_path / "b")


# ---- no old names left in the public API ------------------------------------ #
OLD = {"sub_medium", "substrate_medium", "cladding_medium", "substrate_height", "cladding_height",
       "wavelength0", "wavelength0_um", "fine_min_steps_per_wvl", "coarse_min_steps_per_wvl"}
PACKAGES = ("geometries", "edge_coupler", "grating_coupler", "ring_resonator", "waveguide")


def test_no_public_signature_uses_old_names():
    offenders = []
    for pkg in PACKAGES:
        for path in (REPO / pkg).glob("*.py"):           # dispersion_mapping has its own API
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and (not node.name.startswith("_") or node.name == "__init__"):
                    bad = [a.arg for a in node.args.args + node.args.kwonlyargs if a.arg in OLD]
                    if bad:
                        offenders.append(f"{path.relative_to(REPO)}:{node.name} {bad}")
    assert not offenders, offenders


def test_every_alias_points_at_a_real_parameter():
    """Guards against typos in @renamed_params mappings."""
    import geometries.edge_coupler, geometries.full_ring_resonator, geometries.grating_coupler  # noqa
    import geometries.half_ring_resonator, geometries.partial_ring_resonator  # noqa
    import geometries.resonator_arc, geometries.ridge_waveguide  # noqa
    import edge_coupler.edge_coupler, edge_coupler.optimization_sweep  # noqa
    import grating_coupler.grating_coupler, ring_resonator.ring_resonator  # noqa
    import ring_resonator.resonator_arc, ring_resonator.ring_gap_sweep, ring_resonator.ring_radius_sweep  # noqa
    import waveguide.waveguide, waveguide.mode_sweep, waveguide.dispersion_sweep, waveguide.confinement_sweep  # noqa
    import sys
    checked = 0
    for name, mod in list(sys.modules.items()):
        if not name.split(".")[0] in PACKAGES or mod is None:
            continue
        for obj in list(vars(mod).values()):
            funcs = [obj] if callable(obj) else []
            if inspect.isclass(obj) and obj.__module__ == name:
                funcs = [v for v in vars(obj).values() if callable(v) or isinstance(v, classmethod)]
            for f in funcs:
                f = getattr(f, "__func__", f)
                mapping = getattr(f, "__renamed_params__", None)
                if mapping:
                    params = inspect.signature(f).parameters
                    for old, new in mapping.items():
                        assert new in params and old not in params, (f.__qualname__, old, new)
                        checked += 1
    assert checked >= 60
