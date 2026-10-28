"""The "Your measured values" page of the GUI: one place for numbers from your own chips.

Everything TARANG assumes before a chip exists (losses from publications, the film and BOX
thickness ordered, the fibre's datasheet, n2 and dn/dT from the literature) can be replaced
here by what you measured. Each value goes into ``design.toml`` like any other setting:

* a measured intrinsic Q, or a cutback loss, goes to ``[measured]`` and replaces the assumed
  propagation loss in the ring step (``tarang.design.stages.measured_loss``);
* a measured coupler loss goes to ``[measured]`` and is shown next to the simulated one;
* a measured film thickness, BOX thickness, tolerance or fibre value replaces that setting
  itself and is listed in ``measured.keys``, so the report gives it your source.

Steps whose settings change show "settings changed: run again"; nothing runs by itself.
"""
from __future__ import annotations

from pathlib import Path

from tarang.gui import data, theme, wizard as wz

try:
    import streamlit as st
except ImportError:                    # the gui extra is missing; python -m tarang.gui says how
    st = None

#: (setting, label, unit, help) of the values that replace an existing setting
GEOMETRY = [
    ("stack.core_height_um", "Film thickness", "µm",
     "Thickness of your nitride film (ellipsometry or profilometer), e.g. 0.695."),
    ("stack.box_height_um", "BOX thickness", "µm", "Thickness of the oxide under the film."),
    ("cross_section.tolerance_width_um", "Width scatter ±", "µm",
     "How far the fabricated width spreads around the drawn one (SEM of several waveguides)."),
    ("cross_section.tolerance_height_um", "Film thickness scatter ±", "µm",
     "How far the film thickness varies across the wafer or chip."),
]
FIBRE_GRATING = [
    ("grating.mfd_um", "Fibre mode-field diameter", "µm", "From your fibre's datasheet or a measurement."),
    ("grating.fiber_angle_deg", "Fibre angle from vertical", "deg", "As set on your stage."),
]
FIBRE_EDGE = [
    ("edge.fiber_mfd_um", "Lensed-fibre spot (MFD)", "µm", "Spot size of your lensed fibre."),
]
MATERIAL = [
    ("assumptions.n2_m2_w", "Kerr index n2 of the film", "m²/W",
     "From a measurement on your film (e.g. a threshold or a four-wave-mixing fit), e.g. 2.5e-19."),
    ("assumptions.dn_dT_core_per_K", "Thermo-optic coefficient of the film", "1/K",
     "From the resonance shift with temperature of your rings, e.g. 2.45e-5."),
]


def _num(text: str, what: str):
    """'' -> None, else a float; ValueError naming the field."""
    t = str(text).strip().replace(",", ".")
    if not t:
        return None
    try:
        return float(t)
    except ValueError:
        raise ValueError(f"{what}: a number, or empty") from None


def _show(v) -> str:
    return "" if v is None else f"{v:g}"


def _get(cfg, dotted: str):
    sec, key = dotted.split(".")
    return getattr(getattr(cfg, sec), key)


def page(run: Path) -> None:
    cfg = data.load_config(run)
    m = cfg.measured
    st.subheader("Your measured values")
    st.caption("Until a chip is made, TARANG works with values from publications, your wafer order and "
               "datasheets. When you have measured something on your own chips or setup, enter it "
               "here: it replaces the assumed value, and the report lists it as measured, with the "
               "source you give. Leave a field empty when you have not measured it. Steps that use a "
               "changed value show 🔁 *settings changed: run again*; nothing runs by itself.")
    bad = {k: v for k, v in wz.problems(cfg).items() if k.startswith("measured.")}
    if bad:
        st.error("  \n".join(f"`{k}`: {v}" for k, v in bad.items()))
    values, marked = {}, set()
    with st.form(f"measured:{run}"):
        st.markdown(theme.heading("Where the values come from"), unsafe_allow_html=True)
        source = st.text_input("Source", value=str(m.source),
                               placeholder="e.g. chip A2, swept-laser scan and cutback, Nov 2026",
                               help="Shown as the source of every measured value in the report.")

        st.markdown(theme.heading("Loss of the ring and the waveguide"), unsafe_allow_html=True)
        kinds = ["not measured yet", "intrinsic Q of a ring", "cutback propagation loss"]
        now = 1 if m.q_intrinsic is not None else 2 if m.propagation_loss_db_cm is not None else 0
        kind = st.radio("What you measured", kinds, index=now, horizontal=True, key="mv:kind",
                        help="A ring's intrinsic Q holds every loss of the ring, the bend's included: "
                             "α = 2π n_g / (λ Q_i). A cutback gives the straight-waveguide loss.")
        a, b = st.columns(2)
        q_txt = a.text_input("Intrinsic Q (at the design wavelength)", value=_show(m.q_intrinsic),
                             placeholder="e.g. 1.2e6",
                             help="Q_i from a fit of a resonance (not the loaded Q): with the "
                                  "loaded Q_L and the dip transmission T0 of an under-coupled "
                                  "resonance, Q_i = 2 Q_L / (1 + √T0).")
        l_txt = b.text_input("Propagation loss (dB/cm)", value=_show(m.propagation_loss_db_cm),
                             placeholder="e.g. 0.35",
                             help="Slope of the fibre-to-fibre loss against waveguide length.")
        st.caption("Used in the Ring step instead of the loss preset; the absorption loss is kept "
                   "and the rest counts as scattering.")

        st.markdown(theme.heading("Couplers"), unsafe_allow_html=True)
        c_txt = st.text_input("Insertion loss per coupler (dB)", value=_show(m.coupler_loss_db),
                              placeholder="e.g. 5.8",
                              help="Half the fibre-to-fibre loss of the straight reference "
                                   "waveguide (its propagation loss is small). Shown next to the "
                                   "simulated value; the simulation itself is not changed.")

        groups = [("Film and fabrication", GEOMETRY),
                  ("Fibre", FIBRE_EDGE if wz._coupling(cfg) == "edge" else FIBRE_GRATING),
                  ("Material", MATERIAL)]
        texts = {}
        for title, rows in groups:
            st.markdown(theme.heading(title), unsafe_allow_html=True)
            for dotted, label, unit, hlp in rows:
                a, b = st.columns([2.2, 1])
                texts[dotted] = a.text_input(f"{label} ({unit})", value=_show(_get(cfg, dotted)),
                                             help=hlp, key=f"mv:{dotted}")
                if b.checkbox("measured", value=dotted in m.keys, key=f"mk:{dotted}",
                              help="Tick when the value on the left is your measurement (the "
                                   "report then gives it your source)."):
                    marked.add(dotted)
        saved = st.form_submit_button("Save measured values", type="primary")
    if not saved:
        return
    try:
        values["measured.source"] = source.strip()
        q = _num(q_txt, "Intrinsic Q")
        loss = _num(l_txt, "Propagation loss")
        values["measured.q_intrinsic"] = q if kind == kinds[1] else None
        values["measured.propagation_loss_db_cm"] = loss if kind == kinds[2] else None
        if kind == kinds[1] and q is None:
            raise ValueError("Intrinsic Q: give the value, or choose 'not measured yet'")
        if kind == kinds[2] and loss is None:
            raise ValueError("Propagation loss: give the value, or choose 'not measured yet'")
        values["measured.coupler_loss_db"] = _num(c_txt, "Insertion loss per coupler")
        for dotted, text in texts.items():
            v = _num(text, dotted)
            if v is None and dotted != "stack.core_height_um":
                if dotted in marked:
                    raise ValueError(f"{dotted}: ticked as measured but empty")
                continue                          # empty: keep the setting as it is
            values[dotted] = v
        shown = set(texts)
        values["measured.keys"] = sorted((set(m.keys) - shown) | {k for k in marked if k in shown})
    except ValueError as err:
        st.error(f"Not saved: {err}")
        return
    msg = wz.save_fields(run, values)
    if msg:
        st.error(f"Not saved: {msg}")
        return
    bad = {k: v for k, v in wz.problems(data.load_config(run)).items() if k.startswith("measured.")}
    if bad:
        st.warning("Saved, but: " + "  \n".join(f"`{k}`: {v}" for k, v in bad.items()))
    else:
        st.success("Saved in design.toml. Steps that use a changed value now show 🔁 settings "
                   "changed: run again (Guided design).")
