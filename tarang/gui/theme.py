"""The LOQM lab look for the TARANG GUI: a dark instrument theme.

The colours and fonts are those of the lab's software pages (PhoQuPy's Bench Builder):
a near-black blue background with faint violet and magenta glows, card panels, Space Grotesk
for text and IBM Plex Mono for labels and numbers, and a violet-magenta-amber gradient on
the product name.

Two parts:

* ``streamlit_flags()``: the ``--theme.*`` options ``python -m tarang.gui`` passes to
  Streamlit (colours, fonts, corners). Only options the installed Streamlit knows are
  passed, so an older Streamlit still starts.
* ``css()``: one style block for what the theme options cannot set (the gradient name,
  mono uppercase section labels, status pills, the background glows).

Fonts come from Google Fonts; without internet (a lab PC offline) the browser falls back to
the system fonts and nothing else changes. Figures stay white with black text: they are
journal figures, shown on dark cards.
"""
from __future__ import annotations

import html

#: colour tokens of the lab style
BG = "#080b18"
CARD = "#111834"
CARD2 = "#0e142c"
TEXT = "#edeff7"
MUTED = "#9ba3c4"
FAINT = "#6b7398"
VIOLET = "#7b61ff"
MAGENTA = "#ff4d8d"
AMBER = "#fdb528"
GREEN = "#2fd08b"
LINE = "rgba(255,255,255,.10)"
LINE2 = "rgba(255,255,255,.17)"
GRAD = f"linear-gradient(90deg,{VIOLET} 0%,{MAGENTA} 52%,{AMBER} 100%)"

SANS_URL = ("https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700"
            "&display=swap")
MONO_URL = "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&display=swap"
SANS = "'Space Grotesk', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
MONO = "'IBM Plex Mono', ui-monospace, Consolas, Menlo, monospace"

#: status -> (pill colour, text); the statuses of the guided steps
PILL = {"done": (GREEN, "done"), "skipped": (FAINT, "skipped"), "choose": (AMBER, "your choice"),
        "changed": (AMBER, "run again"), "running": (VIOLET, "running"), "failed": (MAGENTA, "failed"),
        "to do": (FAINT, "not run yet"), "waiting": (AMBER, "waiting"), "error": (MAGENTA, "error")}

THEME = {
    "theme.base": "dark",
    "theme.primaryColor": VIOLET,
    "theme.backgroundColor": BG,
    "theme.secondaryBackgroundColor": CARD,
    "theme.textColor": TEXT,
    "theme.linkColor": "#8fd6ff",
    "theme.borderColor": LINE2,
    "theme.baseRadius": "10px",
    "theme.greenColor": GREEN,
    "theme.redColor": MAGENTA,
    "theme.orangeColor": AMBER,
    "theme.yellowColor": AMBER,
    "theme.violetColor": VIOLET,
    # "<name>:<css url>" loads a Google font; the generic family after it is the fallback
    "theme.font": f"Space Grotesk:{SANS_URL}, sans-serif",
    "theme.headingFont": f"Space Grotesk:{SANS_URL}, sans-serif",
    "theme.codeFont": f"IBM Plex Mono:{MONO_URL}, monospace",
    "theme.sidebar.backgroundColor": CARD2,
    "theme.sidebar.secondaryBackgroundColor": CARD,
}


def _known_options() -> set:
    try:
        from streamlit import config
        return set(config.get_config_options())
    except Exception:                                  # noqa: BLE001  (no or odd Streamlit)
        return set()


def streamlit_flags(known: set | None = None) -> list:
    """``--theme.*`` command-line options for ``streamlit run``; options the installed
    Streamlit does not know are left out (it would refuse to start on them)."""
    known = _known_options() if known is None else known
    out = []
    for k, v in THEME.items():
        if k in known:
            out += [f"--{k}", v]
    return out


def active() -> bool:
    """True inside a Streamlit app started with the lab theme (not ``--plain``)."""
    try:
        import streamlit as st
        return str(st.get_option("theme.backgroundColor")).lower() == BG
    except Exception:                                  # noqa: BLE001
        return False


def heading(text: str) -> str:
    """Markdown for a section heading: the mono label in the lab theme, bold text otherwise."""
    return label(text) if active() else f"**{text}**"


def css() -> str:
    """Style block for ``st.markdown(..., unsafe_allow_html=True)``."""
    return f"""<style>
@import url('{SANS_URL}');
@import url('{MONO_URL}');
.stApp {{
  background: radial-gradient(1100px 600px at 85% -10%, rgba(123,97,255,.16), transparent 60%),
              radial-gradient(900px 520px at 5% 110%, rgba(255,77,141,.12), transparent 60%), {BG};
}}
.tg-brand {{ font-family:{SANS}; font-size:1.55rem; font-weight:700; letter-spacing:-.02em;
  line-height:1.1; margin:0 0 2px 0; }}
.tg-brand .g {{ background:{GRAD}; -webkit-background-clip:text; background-clip:text;
  color:transparent; }}
.tg-tag {{ font-family:{MONO}; font-size:.66rem; letter-spacing:.18em; text-transform:uppercase;
  color:{FAINT}; margin-bottom:10px; }}
.tg-label {{ font-family:{MONO}; font-size:.72rem; letter-spacing:.2em; text-transform:uppercase;
  color:{FAINT}; margin:14px 0 8px 0; display:flex; align-items:center; gap:9px; }}
.tg-label::before {{ content:""; width:18px; height:1px; background:{LINE2}; }}
.tg-pill {{ font-family:{MONO}; font-size:.72rem; border-radius:999px; padding:3px 11px;
  border:1px solid currentColor; display:inline-block; line-height:1.5; white-space:nowrap; }}
.tg-rule {{ height:2px; border:0; border-radius:2px; background:{GRAD}; opacity:.75;
  margin:4px 0 14px 0; }}
.tg-card {{ background:linear-gradient(180deg,{CARD},{CARD2}); border:1px solid {LINE};
  border-radius:16px; padding:14px 16px; }}
/* figures and drawings are white journal figures: give them a frame on the dark page */
[data-testid="stImage"] img {{ border-radius:10px; }}
code {{ font-family:{MONO}; }}
</style>"""


def brand(name: str = "TARANG", tagline: str = "") -> str:
    """The product name with the lab gradient on it, and a mono tagline."""
    t = f'<div class="tg-tag">{html.escape(tagline)}</div>' if tagline else ""
    return f'<div class="tg-brand"><span class="g">{html.escape(name)}</span></div>{t}'


def label(text: str) -> str:
    """A mono, uppercase section label with a short rule before it."""
    return f'<div class="tg-label">{html.escape(text)}</div>'


def pill(status: str, text: str | None = None) -> str:
    """A status badge: green done, amber waiting for you, magenta failed, violet running."""
    colour, default = PILL.get(status, (MUTED, status))
    return (f'<span class="tg-pill" style="color:{colour}">'
            f'{html.escape(text if text is not None else default)}</span>')


def rule() -> str:
    """A thin line in the lab gradient."""
    return '<hr class="tg-rule"/>'
