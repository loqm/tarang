"""Small schematic drawings (SVG) of each design option, for the guided GUI.

They are cartoons, not to scale: a ring next to its bus, cutback spirals, the three grating
couplers seen from above with the fibre's beam, an edge coupler with its lensed fibre, the
cross-section of the stack, and simple icons for the other steps. Each function returns an
SVG string drawn on its own light card, so it reads the same in a light or dark theme.
``as_img`` turns one into an <img> tag for Streamlit's markdown.
"""
from __future__ import annotations

import base64
import math

import numpy as np

W, H = 320, 190
CORE = "#d9822b"          # Si3N4
BOX = "#a9c9e8"           # SiO2
SI = "#9aa5b4"            # silicon handle
BEAM = "#e63946"          # light
GLASS = "#cfe3ea"         # fibre
INK = "#1f2937"
SOFT = "#64748b"
CARD = "#f7f9fc"


def _svg(body: str, title: str = "") -> str:
    t = (f'<text x="12" y="20" font-size="13" font-weight="600" fill="{INK}">{title}</text>'
         if title else "")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
            f'font-family="Helvetica, Arial, sans-serif">'
            f'<rect x="1" y="1" width="{W - 2}" height="{H - 2}" rx="14" fill="{CARD}" stroke="#d6dde6"/>'
            f'{t}{body}</svg>')


def _label(x, y, text, size=10.5, anchor="middle", color=SOFT):
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{color}">{text}</text>'


def _arrow(x1, y1, x2, y2, color=BEAM, width=2.2):
    a = math.atan2(y2 - y1, x2 - x1)
    h = 7
    p1 = (x2 - h * math.cos(a - 0.45), y2 - h * math.sin(a - 0.45))
    p2 = (x2 - h * math.cos(a + 0.45), y2 - h * math.sin(a + 0.45))
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" '
            f'stroke-width="{width}" stroke-linecap="round"/>'
            f'<polygon points="{x2:.1f},{y2:.1f} {p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}" fill="{color}"/>')


def _poly(pts, fill=CORE, stroke="none", sw=1.0, closed=True, extra=""):
    s = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    tag = "polygon" if closed else "polyline"
    return f'<{tag} points="{s}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {extra}/>'


def as_img(svg: str, width: int = W) -> str:
    """<img> tag with the SVG inline (base64), for st.markdown(..., unsafe_allow_html=True)."""
    b = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f'<img src="data:image/svg+xml;base64,{b}" width="{width}" style="max-width:100%"/>'


# ------------------------------------------------------------------ goals
def ring_bus() -> str:
    """A ring resonator above its bus waveguide; light in, comb out."""
    cx, cy, r = 160, 92, 46
    body = (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{CORE}" stroke-width="7"/>'
            f'<rect x="30" y="{cy + r + 9}" width="260" height="7" rx="3" fill="{CORE}"/>'
            + _arrow(36, cy + r + 30, 92, cy + r + 30)
            + _label(64, cy + r + 44, "pump in")
            + "".join(_arrow(230 + 12 * i, cy + r + 30, 244 + 12 * i, cy + r + 30,
                             color=c, width=2) for i, c in enumerate(("#e76f51", BEAM, "#9d0208")))
            + _label(256, cy + r + 44, "comb out")
            + _label(cx, cy + 4, "ring", 12, color=INK)
            + _label(cx + 24, cy + r + 6, "gap", 10))
    return _svg(body, "Ring resonator for a frequency comb")


def spirals() -> str:
    """Cutback structures: a straight reference and spirals of increasing length."""
    out = []
    xs = [52, 130, 215]
    turns = [0, 2, 3.5]
    for x, n in zip(xs, turns):
        if n == 0:
            out.append(f'<rect x="{x - 26}" y="96" width="52" height="4" fill="{CORE}"/>')
            continue
        t = np.linspace(0, 2 * math.pi * n, 400)
        r0, a = 4.0, 3.2 / math.pi
        r = r0 + a * t
        arm = np.stack([x + r * np.cos(t), 98 + r * np.sin(t)], axis=1)
        other = np.stack([2 * x - arm[:, 0], 2 * 98 - arm[:, 1]], axis=1)[::-1]
        pts = np.concatenate([other, arm])
        out.append(_poly(pts, fill="none", stroke=CORE, sw=2.2, closed=False))
    out.append(_label(52, 150, "reference"))
    out.append(_label(130, 150, "1 cm"))
    out.append(_label(215, 150, "4 cm"))
    out.append(_label(160, 172, "loss = slope of transmission against length", 10.5, color=INK))
    return _svg("".join(out), "Test waveguides: cutback spirals")


# ------------------------------------------------------------------ couplers
def _fibre_from_above(x, y_spot, angle_deg=12):
    """A cleaved fibre above the grating, tilted by ``angle_deg`` from vertical, its beam
    landing at (x, y_spot)."""
    a = math.radians(angle_deg)
    ux, uy = math.sin(a), -math.cos(a)               # along the fibre, upwards
    px, py = math.cos(a), math.sin(a)                # across it
    gap = 34                                        # beam length from the fibre end to the spot
    ex, ey = x + ux * gap, y_spot + uy * gap        # centre of the fibre's end face
    L, half = 44, 10
    body = [(ex - px * half, ey - py * half), (ex + px * half, ey + py * half),
            (ex + px * half + ux * L, ey + py * half + uy * L),
            (ex - px * half + ux * L, ey - py * half + uy * L)]
    core = [(ex - px * 2.5, ey - py * 2.5), (ex + px * 2.5, ey + py * 2.5),
            (ex + px * 2.5 + ux * L, ey + py * 2.5 + uy * L),
            (ex - px * 2.5 + ux * L, ey - py * 2.5 + uy * L)]
    return (_poly(body, fill=GLASS, stroke="#90a4ae") + _poly(core, fill="#e0f2f7")
            + _arrow(ex, ey, x, y_spot, width=2.6)
            + _label(ex + 22, ey - 22, f"fibre, {angle_deg}°", 10, "start"))


def _taper_and_slab(x0, x1, y, w0, w1, color=CORE):
    return _poly([(x0, y - w0 / 2), (x1, y - w1 / 2), (x1, y + w1 / 2), (x0, y + w0 / 2)], fill=color)


def grating_uniform() -> str:
    y = 120
    body = [_taper_and_slab(20, 120, y, 4, 46)]
    for i in range(11):
        body.append(f'<rect x="{126 + i * 14}" y="{y - 23}" width="9" height="46" fill="{CORE}"/>')
    body.append(_fibre_from_above(200, y - 26))
    body.append(_label(160, 178, "identical teeth: the light leaves with an exponential profile"))
    return _svg("".join(body), "Uniform grating coupler")


def grating_apodised() -> str:
    y = 120
    body = [_taper_and_slab(20, 120, y, 4, 46)]
    x = 126
    for i in range(11):
        ff = 0.45 + 0.35 * i / 10
        body.append(f'<rect x="{x:.1f}" y="{y - 23}" width="{14 * ff:.1f}" height="46" fill="{CORE}"/>')
        x += 14
    body.append(_fibre_from_above(200, y - 26))
    body.append(_label(160, 178, "teeth change along the grating: the light leaves as a Gaussian"))
    return _svg("".join(body), "Apodised grating coupler")


def grating_focusing() -> str:
    cx, cy = 60, 100
    body = [f'<rect x="12" y="{cy - 2}" width="40" height="4" fill="{CORE}"/>',
            _poly([(cx - 8, cy - 2), (cx, cy - 4), (cx, cy + 4), (cx - 8, cy + 2)])]
    for i in range(8):
        r0 = 58 + i * 10
        a = math.radians(27)
        t = np.linspace(-a, a, 40)
        outer = [(cx + (r0 + 6) * math.cos(v), cy + (r0 + 6) * math.sin(v)) for v in t]
        inner = [(cx + r0 * math.cos(v), cy + r0 * math.sin(v)) for v in t[::-1]]
        body.append(_poly(outer + inner))
    sector = [(cx, cy - 4), (cx + 58 * math.cos(0.47), cy - 58 * math.sin(0.47)),
              (cx + 58 * math.cos(0.47), cy + 58 * math.sin(0.47)), (cx, cy + 4)]
    body.insert(0, _poly(sector, fill="#f2c79b"))
    body.append(_label(160, 180, "curved teeth focus the light into the waveguide tip: no long taper"))
    return _svg("".join(body), "Focusing grating coupler")


def lensed_fibre(mfd: float = 3.5) -> str:
    body = (f'<rect x="150" y="80" width="150" height="40" fill="{GLASS}" stroke="#90a4ae"/>'
            f'<path d="M150,80 Q120,100 150,120 Z" fill="{GLASS}" stroke="#90a4ae"/>'
            f'<rect x="150" y="96" width="150" height="8" fill="#e0f2f7"/>'
            + _poly([(130, 90), (60, 98), (60, 102), (130, 110)], fill="#f8c4c8")
            + f'<ellipse cx="60" cy="100" rx="3" ry="9" fill="{BEAM}"/>'
            + _label(60, 126, f"spot, MFD {mfd:g} µm")
            + _label(225, 140, "lensed fibre"))
    return _svg(body, "Lensed fibre")


def edge_coupler() -> str:
    y = 100
    body = ['<rect x="12" y="40" width="190" height="122" fill="#eef4fb"/>',
            f'<line x1="202" y1="36" x2="202" y2="166" stroke="{INK}" stroke-dasharray="5,4"/>',
            f'<rect x="14" y="{y - 2}" width="80" height="4" fill="{CORE}"/>',
            _poly([(94, y - 2), (150, y - 12), (150, y + 12), (94, y + 2)]),
            f'<rect x="150" y="{y - 12}" width="52" height="24" fill="{CORE}"/>',
            f'<rect x="250" y="{y - 14}" width="66" height="28" fill="{GLASS}" stroke="#90a4ae"/>',
            f'<path d="M250,{y - 14} Q232,{y} 250,{y + 14} Z" fill="{GLASS}" stroke="#90a4ae"/>',
            _poly([(240, y - 9), (204, y - 4), (204, y + 4), (240, y + 9)], fill="#f8c4c8"),
            _label(52, y + 22, "waveguide"), _label(126, y + 30, "taper"),
            _label(176, y + 30, "wide facet"), _label(202, 30, "chip edge", 10),
            _label(282, y + 30, "lensed fibre"),
            _label(160, 180, "the waveguide widens to match the fibre spot at the facet")]
    return _svg("".join(body), "Edge coupler (wide facet)")


# ------------------------------------------------------------------ steps
def cross_section(Hc: float = 0.7, Wc: float = 1.0, box: float = 4.0, cladding: str = "air") -> str:
    s = 60.0                                 # px per um for the core
    s = min(s, 120 / max(Wc, 0.1), 70 / max(Hc, 0.1))
    base = 128
    body = [f'<rect x="20" y="{base + 18}" width="280" height="26" fill="{SI}"/>',
            f'<rect x="20" y="{base}" width="280" height="18" fill="{BOX}"/>']
    if str(cladding).lower() != "air":
        body.append(f'<rect x="20" y="{base - Hc * s - 22}" width="280" height="{Hc * s + 22}" fill="#cfe6d8" '
                    'opacity="0.8"/>')
    x0 = 160 - Wc * s / 2
    body.append(f'<rect x="{x0:.1f}" y="{base - Hc * s:.1f}" width="{Wc * s:.1f}" height="{Hc * s:.1f}" '
                f'fill="{CORE}"/>')
    body.append(_label(160, base - Hc * s - 8, f"W = {Wc * 1e3:.0f} nm", 11, color=INK))
    body.append(_label(x0 + Wc * s + 8, base - Hc * s / 2 + 4, f"H = {Hc * 1e3:.0f} nm", 11, "start", INK))
    body.append(_label(36, base + 13, f"BOX (SiO₂) {box:g} µm", 10, "start", INK))
    body.append(_label(36, base + 36, "Si handle", 10, "start", INK))
    body.append(_label(40, 40, "air" if str(cladding).lower() == "air" else str(cladding), 11, "start"))
    return _svg("".join(body), "Cross-section")


def materials() -> str:
    x = np.linspace(30, 290, 60)
    body = [_poly([(xi, 70 + 18 * math.exp(-(xi - 30) / 90)) for xi in x], fill="none", stroke=CORE, sw=3,
                  closed=False),
            _poly([(xi, 130 + 8 * math.exp(-(xi - 30) / 90)) for xi in x], fill="none", stroke="#4c8bc9", sw=3,
                  closed=False),
            _label(250, 64, "Si₃N₄  n ≈ 2.0", 10.5, color=INK), _label(250, 124, "SiO₂  n ≈ 1.44", 10.5, color=INK),
            _label(160, 172, "refractive index against wavelength, fitted for the solvers")]
    return _svg("".join(body), "Materials")


def gap_detail() -> str:
    cx, cy, r = 160, -140, 250                # the lower part of a large ring above the bus
    t = np.linspace(math.radians(62), math.radians(118), 60)
    ring = [(cx + r * math.cos(v), cy + r * math.sin(v)) for v in t]
    body = [_poly(ring, fill="none", stroke=CORE, sw=8, closed=False),
            f'<rect x="20" y="128" width="280" height="8" fill="{CORE}"/>',
            f'<line x1="160" y1="115" x2="160" y2="127" stroke="{INK}" stroke-width="1.2"/>',
            _label(172, 124, "gap", 11, "start", INK),
            _label(160, 170, "smaller gap: more light enters the ring (over-coupled)")]
    return _svg("".join(body), "Ring–bus gap")


def taper_picture() -> str:
    body = [f'<rect x="20" y="98" width="70" height="4" fill="{CORE}"/>',
            _poly([(90, 98), (250, 70), (250, 130), (90, 102)]),
            '<rect x="250" y="70" width="50" height="60" fill="#c96f1f"/>',
            _label(170, 150, "slow enough for the mode to follow (adiabatic)"),
            _label(275, 62, "grating", 10)]
    return _svg("".join(body), "Taper")


def checks_picture() -> str:
    body = []
    for i, (txt, ok) in enumerate((("fabrication tolerances", True), ("coupling across the band", True),
                                   ("mode families of the ring", True), ("fibre-to-fibre loss budget", True))):
        y = 52 + i * 30
        body.append(f'<circle cx="40" cy="{y}" r="9" fill="#2a9d8f"/>'
                    f'<path d="M35,{y} l4,4 l7,-8" stroke="white" stroke-width="2.2" fill="none"/>')
        body.append(_label(58, y + 4, txt, 11.5, "start", INK))
    return _svg("".join(body), "Checks")


def layout_chip(edge: bool = False) -> str:
    body = [f'<rect x="40" y="32" width="240" height="132" fill="#eef4fb" stroke="{INK}" stroke-dasharray="5,4"/>']
    for i in range(4):
        y = 56 + i * 28
        body.append(f'<rect x="{40 if edge else 70}" y="{y}" width="{240 if edge else 180}" height="3" fill="{CORE}"/>')
        body.append(f'<circle cx="160" cy="{y - 9}" r="8" fill="none" stroke="{CORE}" stroke-width="2.5"/>')
        if not edge:
            for x in (56, 254):
                body.append(f'<rect x="{x}" y="{y - 6}" width="12" height="15" fill="#c96f1f"/>')
    body.append(_label(160, 180, "GDS: every device, its couplers" + (" at the chip edges" if edge else "")))
    return _svg("".join(body), "Layout")


def device_spectrum() -> str:
    x = np.linspace(20, 300, 600)
    y = 70 + 4 * np.sin((x - 20) / 6)
    for c in (70, 140, 210, 280):
        y += 60 * np.exp(-((x - c) / 2.2) ** 2)
    body = [_poly(list(zip(x, y)), fill="none", stroke="#2563eb", sw=1.6, closed=False),
            _label(160, 170, "fibre-to-fibre transmission: every resonance and the ripple")]
    return _svg("".join(body), "Whole device")


def report_picture() -> str:
    body = [f'<rect x="110" y="34" width="100" height="130" rx="6" fill="white" stroke="{INK}"/>']
    for i in range(7):
        body.append(f'<rect x="124" y="{52 + i * 14}" width="{72 - (i % 3) * 14}" height="5" fill="#cbd5e1"/>')
    body.append(_label(160, 180, "report, tables, layout and the benchmark pack"))
    return _svg("".join(body), "Report")


ILLUSTRATIONS = {
    "comb_ring": ring_bus, "loss_test": spirals, "uniform": grating_uniform,
    "apodised": grating_apodised, "focusing": grating_focusing, "edge": edge_coupler,
    "lensed_fibre": lensed_fibre, "materials": materials, "gap": gap_detail, "taper": taper_picture,
    "checks": checks_picture, "device": device_spectrum, "report": report_picture,
}
