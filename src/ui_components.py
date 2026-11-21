"""
StockMind AI — "Neon Bull" Design System (src/ui_components.py)

Visual language derived from the StockMind bull logo:
  * Near-black navy terminal background with a faint world-map / grid texture
  * Neon bull-green as the primary accent (gains, active states, CTAs)
  * Teal-blue from the icon's rim as the secondary accent
  * Candle-red for losses, amber for caution
  * Soft glows, glass cards and a green→teal gradient rim (like the app icon)
"""

import base64
import functools
import os

import streamlit as st

from src.learning_content import (
    get_icon,
    ALGORITHM_EXPLANATIONS,
    CODE_EXPLAINERS,
)

# -----------------------------------------------------------------------------
# Brand palette (import these anywhere you need a hex value)
# -----------------------------------------------------------------------------
NEON_GREEN = "#2EE88A"
GREEN_BRIGHT = "#6CFFB0"
GREEN_DEEP = "#12B76A"
TEAL = "#1FB5E0"
CANDLE_RED = "#FF4D5E"
AMBER = "#F7B955"
TEXT_MUTED = "#7A948A"

_ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
LOGO_PATH = os.path.join(_ASSETS_DIR, "logo_128.png")  # small copy for inline embedding
LOGO_FULL_PATH = os.path.join(_ASSETS_DIR, "logo.png")
FAVICON_PATH = os.path.join(_ASSETS_DIR, "favicon.png")


@functools.lru_cache(maxsize=4)
def _img_b64(path: str) -> str:
    try:
        with open(path, "rb") as fh:
            return base64.b64encode(fh.read()).decode("ascii")
    except OSError:
        return ""


def logo_data_uri() -> str:
    b64 = _img_b64(LOGO_PATH)
    return f"data:image/png;base64,{b64}" if b64 else ""


# -----------------------------------------------------------------------------
# Global CSS
# -----------------------------------------------------------------------------
_ROOT_VARS = """
    --bg-app: #04070B;
    --bg-sidebar: #060B11;
    --bg-card: rgba(12, 21, 29, 0.82);
    --bg-card-solid: #0C151D;
    --bg-card-subtle: #0F1B24;
    --bg-input: #0B141C;
    --bg-guide: linear-gradient(135deg, rgba(46,232,138,0.10) 0%, rgba(31,181,224,0.08) 100%);
    --border-color: rgba(120, 200, 170, 0.14);
    --border-strong: rgba(46, 232, 138, 0.38);
    --border-hover: #2EE88A;
    --rim: linear-gradient(135deg, rgba(108,255,176,0.75) 0%, rgba(46,232,138,0.25) 40%, rgba(31,181,224,0.55) 100%);
    --text-primary: #E8F6EE;
    --text-secondary: #A9C2B5;
    --text-muted: #7A948A;
    --accent: #2EE88A;
    --accent-bright: #6CFFB0;
    --accent-deep: #12B76A;
    --accent-soft: rgba(46, 232, 138, 0.12);
    --teal: #1FB5E0;
    --teal-soft: rgba(31, 181, 224, 0.12);
    --blue-accent: #1FB5E0;
    --pos: #2EE88A;
    --neg: #FF4D5E;
    --neg-soft: rgba(255, 77, 94, 0.12);
    --amber: #F7B955;
    --glow: 0 0 0 1px rgba(46,232,138,0.18), 0 0 28px rgba(46,232,138,0.16);
    --shadow-card: 0 10px 30px rgba(0, 0, 0, 0.45);
    --shadow-hover: 0 14px 38px rgba(0, 0, 0, 0.55), 0 0 26px rgba(46,232,138,0.18);
    --font-display: 'Space Grotesk', 'Inter', sans-serif;
    --font-body: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
    --radius: 16px;
"""

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root { __ROOT__ }

/* ---------- App chrome ---------- */
#MainMenu, footer, [data-testid="stDecoration"],
[data-testid="stAppDeployButton"], .stDeployButton,
[data-testid="stToolbarActions"], [data-testid="stMainMenu"] { display: none !important; }
header[data-testid="stHeader"], [data-testid="stToolbar"] {
    background: transparent !important;
    pointer-events: none !important;
}
header[data-testid="stHeader"] { height: 3rem !important; }
header[data-testid="stHeader"] button,
[data-testid="stToolbar"] button,
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stExpandSidebarButton"] {
    pointer-events: auto !important;
    visibility: visible !important;
    opacity: 1 !important;
    cursor: pointer !important;
}
[data-testid="collapsedControl"], [data-testid="stExpandSidebarButton"] {
    background: var(--bg-card-solid) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.45) !important;
}
[data-testid="collapsedControl"] svg,
[data-testid="stExpandSidebarButton"] svg,
[data-testid="stSidebarCollapseButton"] svg { fill: var(--accent) !important; color: var(--accent) !important; }

html, body, [class*="css"], .stApp { font-family: var(--font-body); }
.stApp {
    color: var(--text-primary);
    background:
        radial-gradient(1100px 520px at 88% -8%, rgba(46,232,138,0.10), transparent 60%),
        radial-gradient(900px 480px at -10% 105%, rgba(31,181,224,0.08), transparent 60%),
        radial-gradient(rgba(46,232,138,0.055) 1px, transparent 1.2px) 0 0 / 22px 22px,
        var(--bg-app);
    background-attachment: fixed;
}
.block-container { padding-top: 2.2rem !important; padding-bottom: 2.5rem !important; max-width: 1320px; }
::selection { background: rgba(46,232,138,0.30); color: #fff; }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(46,232,138,0.18); border-radius: 10px; border: 2px solid transparent; background-clip: padding-box; }
::-webkit-scrollbar-thumb:hover { background: rgba(46,232,138,0.38); background-clip: padding-box; }

/* ---------- Typography ---------- */
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
label, [data-testid="stWidgetLabel"] p, [data-testid="stRadio"] label p, [data-testid="stCheckbox"] label p {
    color: var(--text-primary) !important;
}
[data-testid="stWidgetLabel"] p { color: var(--text-secondary) !important; font-weight: 600 !important; font-size: 0.82rem !important; letter-spacing: 0.01em; }
h1, h2, h3, h4, h5 { font-family: var(--font-display) !important; color: var(--text-primary) !important; letter-spacing: -0.02em; }
h3 { font-size: 1.18rem !important; }
h4 { font-size: 1.02rem !important; }
.stCaption, .stCaption p, small, [data-testid="stCaptionContainer"] { color: var(--text-muted) !important; }
code { background: var(--accent-soft) !important; color: var(--accent-bright) !important; padding: 2px 6px; border-radius: 6px; font-family: var(--font-mono); font-size: 0.85em; }
hr { border-color: var(--border-color) !important; margin: 0.9rem 0 !important; }
a { color: var(--accent) !important; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #07101A 0%, #050A10 55%, #04080C 100%) !important;
    border-right: 1px solid var(--border-color);
}
[data-testid="stSidebar"]::after {
    content: ""; position: absolute; top: 0; right: -1px; width: 1px; height: 100%;
    background: linear-gradient(180deg, transparent, rgba(46,232,138,0.55) 30%, rgba(31,181,224,0.45) 70%, transparent);
    pointer-events: none;
}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top: 0.4rem; }
.sb-section {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--text-muted) !important; margin: 14px 0 6px 2px; display: flex; align-items: center; gap: 6px;
}

/* Sidebar navigation: radio -> neon nav pills */
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] { gap: 3px; background: transparent !important; border: none !important; padding: 0 !important; width: 100% !important; box-shadow: none !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) { background: linear-gradient(90deg, rgba(46,232,138,0.18), rgba(46,232,138,0.04)) !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label {
    width: 100%; margin: 0 !important; padding: 9px 12px !important; border-radius: 11px;
    border: 1px solid transparent; transition: all .15s ease; cursor: pointer;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:hover {
    background: rgba(46,232,138,0.06); border-color: var(--border-color);
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(90deg, rgba(46,232,138,0.18), rgba(46,232,138,0.04));
    border-color: rgba(46,232,138,0.35);
    box-shadow: inset 3px 0 0 var(--accent), 0 0 18px rgba(46,232,138,0.10);
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) p {
    color: var(--accent-bright) !important; font-weight: 700 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label > div:first-of-type { display: none; }
[data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] > label p { font-weight: 600; font-size: 0.9rem; }

/* ---------- Main-area horizontal radios -> segmented control ---------- */
[data-testid="stRadio"] [role="radiogroup"] {
    background: var(--bg-card-subtle); border: 1px solid var(--border-color);
    padding: 4px; border-radius: 12px; gap: 4px; flex-wrap: wrap; width: fit-content;
}
[data-testid="stRadio"] [role="radiogroup"] > label {
    margin: 0 !important; padding: 6px 14px !important; border-radius: 9px; cursor: pointer; transition: all .15s ease;
}
[data-testid="stRadio"] [role="radiogroup"] > label > div:first-of-type { display: none; }
[data-testid="stRadio"] [role="radiogroup"] > label p { color: var(--text-secondary) !important; font-weight: 600; font-size: 0.86rem; }
[data-testid="stRadio"] [role="radiogroup"] > label:hover p { color: var(--text-primary) !important; }
[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(135deg, rgba(46,232,138,0.22), rgba(31,181,224,0.14));
    box-shadow: 0 0 0 1px rgba(46,232,138,0.40), 0 0 16px rgba(46,232,138,0.14);
}
[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) p { color: var(--accent-bright) !important; }

/* ---------- Form controls ---------- */
[data-baseweb="select"] > div {
    background: var(--bg-input) !important; border: 1px solid var(--border-color) !important;
    border-radius: 11px !important; color: var(--text-primary) !important; transition: border-color .15s, box-shadow .15s;
}
[data-baseweb="select"] > div:hover, [data-baseweb="select"] > div:focus-within {
    border-color: var(--border-strong) !important; box-shadow: 0 0 0 3px rgba(46,232,138,0.10) !important;
}
[data-baseweb="select"] span, [data-baseweb="select"] div, [data-baseweb="select"] input {
    color: var(--text-primary) !important; -webkit-text-fill-color: var(--text-primary) !important;
}
[data-baseweb="select"] svg { fill: var(--accent) !important; color: var(--accent) !important; }
[data-baseweb="popover"], [data-baseweb="popover"] > div, [data-baseweb="menu"], ul[role="listbox"] {
    background: #0A131B !important; border-color: var(--border-strong) !important; border-radius: 12px !important;
}
li[role="option"] { background: transparent !important; color: var(--text-primary) !important; }
li[role="option"]:hover, li[role="option"][aria-selected="true"] {
    background: var(--accent-soft) !important; color: var(--accent-bright) !important; font-weight: 600;
}
[data-baseweb="input"], [data-baseweb="base-input"] {
    background: var(--bg-input) !important; border: 1px solid var(--border-color) !important; border-radius: 11px !important;
}
[data-baseweb="input"]:focus-within { border-color: var(--border-strong) !important; box-shadow: 0 0 0 3px rgba(46,232,138,0.10) !important; }
[data-baseweb="input"] input, [data-baseweb="base-input"] input, .stTextInput input, .stNumberInput input {
    background: var(--bg-input) !important; color: var(--text-primary) !important;
    -webkit-text-fill-color: var(--text-primary) !important; caret-color: var(--accent) !important;
}
.stNumberInput input { font-family: var(--font-mono); }
input::placeholder { color: var(--text-muted) !important; -webkit-text-fill-color: var(--text-muted) !important; opacity: 1 !important; }
.stNumberInput button { background: var(--bg-card-subtle) !important; color: var(--accent) !important; border-color: var(--border-color) !important; }
.stNumberInput button svg { fill: var(--accent) !important; }
.stNumberInput button:hover { background: var(--accent-soft) !important; }

/* Sliders */
[data-testid="stSlider"] [role="slider"] { background: var(--accent) !important; box-shadow: 0 0 0 4px rgba(46,232,138,0.18), 0 0 14px rgba(46,232,138,0.55) !important; border: none !important; }
[data-testid="stSlider"] [data-testid="stThumbValue"] { color: var(--accent-bright) !important; font-family: var(--font-mono); font-weight: 700; }
[data-testid="stSlider"] [data-testid="stTickBarMin"], [data-testid="stSlider"] [data-testid="stTickBarMax"] { color: var(--text-muted) !important; font-family: var(--font-mono); }

/* Checkbox */
[data-testid="stCheckbox"] label span[role="checkbox"], [data-baseweb="checkbox"] > span:first-child { border-color: var(--border-strong) !important; }

/* Progress */
[data-testid="stProgress"] > div > div > div > div { background: linear-gradient(90deg, var(--teal), var(--accent)) !important; box-shadow: 0 0 12px rgba(46,232,138,0.45); }

/* ---------- Expanders, alerts ---------- */
[data-testid="stExpander"] {
    background: var(--bg-card) !important; border: 1px solid var(--border-color) !important;
    border-radius: 14px !important; margin-bottom: 10px; backdrop-filter: blur(8px);
}
[data-testid="stExpander"] details { border: none !important; }
[data-testid="stExpander"] summary:hover { background: rgba(46,232,138,0.04); }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p, [data-testid="stExpander"] summary span { color: var(--text-primary) !important; font-weight: 600; }
[data-testid="stExpander"] summary svg { fill: var(--accent) !important; color: var(--accent) !important; }
[data-testid="stAlert"] { background: var(--teal-soft) !important; border: 1px solid rgba(31,181,224,0.30) !important; border-radius: 14px !important; }
[data-testid="stAlert"] p { color: var(--text-primary) !important; }

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: var(--bg-card-subtle); padding: 5px; border-radius: 13px;
    border: 1px solid var(--border-color); overflow-x: auto;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px; padding: 8px 16px; height: auto; color: var(--text-muted) !important;
    font-weight: 600; background: transparent; transition: all .15s ease; white-space: nowrap;
}
.stTabs [data-baseweb="tab"] p { font-size: 0.87rem; }
.stTabs [data-baseweb="tab"]:hover { color: var(--text-primary) !important; background: rgba(46,232,138,0.05); }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(46,232,138,0.20), rgba(31,181,224,0.12)) !important;
    color: var(--accent-bright) !important; box-shadow: 0 0 0 1px rgba(46,232,138,0.35);
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 14px; }

/* ---------- Buttons ---------- */
.stButton > button, .stDownloadButton > button {
    position: relative; z-index: 2; cursor: pointer !important; border-radius: 11px;
    border: 1px solid var(--border-color) !important; background: var(--bg-card-solid) !important;
    color: var(--text-primary) !important; font-weight: 600; font-family: var(--font-body);
    transition: all .16s ease; min-height: 2.45rem;
}
.stButton > button p, .stDownloadButton > button p { color: var(--text-primary) !important; margin: 0 !important; font-weight: 600; }
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: var(--accent) !important; background: var(--accent-soft) !important;
    box-shadow: 0 0 18px rgba(46,232,138,0.18); transform: translateY(-1px);
}
.stButton > button:hover p, .stDownloadButton > button:hover p { color: var(--accent-bright) !important; }
.stButton > button:active { transform: translateY(0); }
.stButton > button[kind="primary"], .stButton > button[data-testid="baseButton-primary"], .stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #3CF29A 0%, #12B76A 100%) !important; border-color: transparent !important;
    box-shadow: 0 0 0 1px rgba(108,255,176,0.45), 0 6px 22px rgba(46,232,138,0.32);
}
.stButton > button[kind="primary"] p, .stButton > button[data-testid="baseButton-primary"] p, .stButton > button[data-testid="stBaseButton-primary"] p,
.stButton > button[kind="primary"]:hover p, .stButton > button[data-testid="baseButton-primary"]:hover p, .stButton > button[data-testid="stBaseButton-primary"]:hover p {
    color: #03140B !important; font-weight: 700;
}
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="baseButton-primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(135deg, #6CFFB0 0%, #2EE88A 100%) !important;
    box-shadow: 0 0 0 1px rgba(108,255,176,0.6), 0 8px 28px rgba(46,232,138,0.45);
}
.stDownloadButton > button { border-color: var(--border-strong) !important; }

[data-testid="stRadio"] label, [data-testid="stCheckbox"] label, [data-testid="stExpander"] summary, .stTabs [data-baseweb="tab"] { cursor: pointer !important; }

/* ---------- Charts & tables as glass cards ---------- */
[data-testid="stPlotlyChart"], [data-testid="stPyplot"], [data-testid="stImage"]:has(img[src*="pyplot"]) {
    background: var(--bg-card); border: 1px solid var(--border-color); border-radius: var(--radius);
    padding: 6px 8px 2px 8px; box-shadow: var(--shadow-card); margin-bottom: 8px; overflow: hidden;
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border: 1px solid var(--border-color); border-radius: 14px; overflow: hidden; box-shadow: var(--shadow-card);
}

/* ---------- Custom components ---------- */
.fin-card {
    position: relative; background: var(--bg-card); border: 1px solid var(--border-color);
    border-radius: var(--radius); padding: 16px 18px; margin-bottom: 12px; box-shadow: var(--shadow-card);
    backdrop-filter: blur(10px); transition: border-color .18s ease, box-shadow .18s ease, transform .18s ease;
}
.fin-card:hover { border-color: var(--border-strong); box-shadow: var(--shadow-hover); transform: translateY(-2px); }

.rim { position: relative; }
.rim::before {
    content: ""; position: absolute; inset: 0; border-radius: inherit; padding: 1px;
    background: var(--rim);
    -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
    -webkit-mask-composite: xor; mask-composite: exclude; pointer-events: none;
}

/* Brand block (sidebar) */
.brand { display: flex; align-items: center; gap: 12px; padding: 6px 2px 4px 2px; }
.brand img { width: 46px; height: 46px; flex-shrink: 0; border-radius: 14px; box-shadow: 0 0 0 1px rgba(46,232,138,0.35), 0 0 24px rgba(46,232,138,0.35); }
.brand-name {
    font-family: var(--font-display); font-weight: 700; font-size: 1.16rem; letter-spacing: 0; line-height: 1.05; white-space: nowrap;
    background: linear-gradient(90deg, #E8F6EE 0%, #6CFFB0 55%, #1FB5E0 100%);
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.brand-tag { font-size: 0.64rem; color: var(--text-muted); letter-spacing: 0.06em; white-space: nowrap; text-transform: uppercase; margin-top: 3px; }
.live-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 0 0 rgba(46,232,138,0.7); animation: pulse 2s infinite; vertical-align: middle; }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(46,232,138,0.6); } 70% { box-shadow: 0 0 0 8px rgba(46,232,138,0); } 100% { box-shadow: 0 0 0 0 rgba(46,232,138,0); } }
.status-chip {
    display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-top: 10px;
    padding: 8px 12px; border-radius: 11px; background: var(--accent-soft); border: 1px solid rgba(46,232,138,0.22);
    font-size: 0.76rem; color: var(--text-secondary); white-space: nowrap;
}
.status-chip b { color: var(--accent-bright); font-family: var(--font-mono); font-weight: 700; }

/* Ticker tape */
.tape { position: relative; overflow: hidden; border: 1px solid var(--border-color); border-radius: 12px;
    background: linear-gradient(90deg, rgba(12,21,29,0.95), rgba(12,21,29,0.75)); margin-bottom: 12px; height: 38px; }
.tape::before, .tape::after { content: ""; position: absolute; top: 0; width: 60px; height: 100%; z-index: 2; pointer-events: none; }
.tape::before { left: 0; background: linear-gradient(90deg, #0C151D, transparent); }
.tape::after { right: 0; background: linear-gradient(270deg, #0C151D, transparent); }
.tape-track { display: inline-flex; white-space: nowrap; animation: tape 48s linear infinite; height: 100%; align-items: center; }
.tape:hover .tape-track { animation-play-state: paused; }
.tape-item { display: inline-flex; align-items: center; gap: 8px; padding: 0 22px; font-size: 0.82rem; border-right: 1px solid var(--border-color); }
.tape-sym { font-weight: 700; color: var(--text-primary); letter-spacing: 0.02em; }
.tape-px { font-family: var(--font-mono); color: var(--text-secondary); }
.tape-up { color: var(--pos); font-family: var(--font-mono); font-weight: 700; }
.tape-dn { color: var(--neg); font-family: var(--font-mono); font-weight: 700; }
@keyframes tape { from { transform: translateX(0); } to { transform: translateX(-50%); } }

/* Hero */
.stock-hero {
    position: relative; overflow: hidden; border-radius: 20px; padding: 22px 26px; margin: 4px 0 18px 0;
    background:
        radial-gradient(600px 220px at 100% 0%, rgba(46,232,138,0.14), transparent 70%),
        radial-gradient(500px 200px at 0% 100%, rgba(31,181,224,0.10), transparent 70%),
        linear-gradient(180deg, rgba(14,24,33,0.95), rgba(9,16,23,0.95));
    border: 1px solid var(--border-color); box-shadow: var(--shadow-card);
    display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 18px;
}
.stock-hero::after {
    content: ""; position: absolute; right: -30px; bottom: -40px; width: 260px; height: 160px; opacity: 0.10; pointer-events: none;
    background: repeating-linear-gradient(90deg, var(--accent) 0 6px, transparent 6px 18px);
    -webkit-mask: linear-gradient(0deg, #000, transparent); mask: linear-gradient(0deg, #000, transparent);
    transform: skewY(-12deg);
}
.hero-left { display: flex; align-items: center; gap: 16px; z-index: 1; }
.hero-avatar {
    width: 54px; height: 54px; border-radius: 15px; display: grid; place-items: center;
    font-family: var(--font-display); font-weight: 700; font-size: 1.05rem; color: #03140B;
    background: linear-gradient(135deg, #6CFFB0, #12B76A); box-shadow: 0 0 22px rgba(46,232,138,0.35);
}
.stock-hero-name { font-family: var(--font-display); font-size: 1.28rem; font-weight: 700; color: var(--text-primary); letter-spacing: -0.02em; }
.stock-hero-price { font-family: var(--font-mono); font-size: 2.15rem; font-weight: 700; color: var(--text-primary); letter-spacing: -0.03em; line-height: 1.1; }
.chg-chip { display: inline-flex; align-items: center; gap: 6px; padding: 4px 11px; border-radius: 999px; font-family: var(--font-mono); font-weight: 700; font-size: 0.88rem; }
.chg-up { color: var(--accent-bright); background: var(--accent-soft); box-shadow: inset 0 0 0 1px rgba(46,232,138,0.30); }
.chg-dn { color: #FF8A95; background: var(--neg-soft); box-shadow: inset 0 0 0 1px rgba(255,77,94,0.30); }
.hero-stats { display: grid; grid-template-columns: repeat(4, minmax(86px, auto)); gap: 6px 22px; z-index: 1; }
.hero-stat-l { font-size: 0.68rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.10em; font-weight: 600; }
.hero-stat-v { font-family: var(--font-mono); font-size: 0.95rem; color: var(--text-primary); font-weight: 700; }
.hero-meta { font-size: 0.74rem; color: var(--text-muted); margin-top: 8px; display: flex; gap: 14px; flex-wrap: wrap; }

/* Section header */
.sec-head { display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 10px; margin: 2px 0 14px 0; }
.sec-eyebrow { font-size: 0.68rem; letter-spacing: 0.16em; text-transform: uppercase; color: var(--accent); font-weight: 700; margin-bottom: 4px; }
.sec-title { margin: 0; font-family: var(--font-display); font-weight: 700; font-size: 1.6rem; color: var(--text-primary); display: flex; align-items: center; letter-spacing: -0.02em; }
.sec-sub { margin: 4px 0 0 0; color: var(--text-muted); font-size: 0.9rem; }
.icon-chip {
    display: inline-grid; place-items: center; width: 36px; height: 36px; border-radius: 11px; margin-right: 12px;
    background: linear-gradient(135deg, rgba(46,232,138,0.20), rgba(31,181,224,0.12)); box-shadow: inset 0 0 0 1px rgba(46,232,138,0.35);
}
.icon-chip svg { margin: 0 !important; }

/* Guide card */
.simple-guide-card {
    background: var(--bg-guide); border: 1px solid rgba(46,232,138,0.18); border-radius: 14px;
    padding: 14px 18px 14px 20px; margin-bottom: 16px; position: relative; overflow: hidden;
}
.simple-guide-card::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: linear-gradient(180deg, var(--accent), var(--teal)); box-shadow: 0 0 12px var(--accent); }
.simple-guide-title { font-size: 0.9rem; font-weight: 700; color: var(--accent-bright); margin-bottom: 4px; display: flex; align-items: center; }
.simple-guide-body { font-size: 0.92rem; color: var(--text-primary); line-height: 1.55; }
.simple-guide-body strong { color: var(--accent-bright); }

/* KPI */
.kpi-card { overflow: hidden; }
.kpi-card::after { content: ""; position: absolute; left: 0; right: 0; top: 0; height: 2px; background: var(--kpi-c, var(--teal)); box-shadow: 0 0 14px var(--kpi-c, var(--teal)); opacity: .9; }
.kpi-title { font-size: 0.74rem; color: var(--text-muted); font-weight: 600; margin-bottom: 8px; display: flex; align-items: center; text-transform: uppercase; letter-spacing: 0.08em; }
.kpi-ico { display: inline-grid; place-items: center; width: 26px; height: 26px; border-radius: 8px; margin-right: 9px; background: var(--kpi-bg, var(--teal-soft)); }
.kpi-ico svg { margin: 0 !important; }
.kpi-value { font-family: var(--font-mono); font-size: 1.45rem; font-weight: 700; color: var(--text-primary); letter-spacing: -0.02em; line-height: 1.2; word-break: break-word; }
.kpi-sub-pos { font-size: 0.82rem; color: var(--pos); font-weight: 600; margin-top: 5px; }
.kpi-sub-neg { font-size: 0.82rem; color: var(--neg); font-weight: 600; margin-top: 5px; }
.kpi-sub-neutral { font-size: 0.82rem; color: var(--teal); font-weight: 500; margin-top: 5px; }

/* Module cards */
.mod-card { min-height: 148px; display: flex; flex-direction: column; gap: 6px; }
.mod-card .mod-title { font-family: var(--font-display); font-weight: 700; font-size: 1.0rem; color: var(--text-primary); display: flex; align-items: center; margin-top: 6px; }
.mod-card .mod-desc { font-size: 0.8rem; color: var(--text-muted); line-height: 1.45; }

/* Pills */
.ticker-badge {
    display: inline-block; background: var(--accent-soft); color: var(--accent-bright) !important; padding: 3px 10px;
    border-radius: 999px; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.04em; box-shadow: inset 0 0 0 1px rgba(46,232,138,0.28);
}
.mode-pill {
    display: inline-flex; align-items: center; gap: 6px; background: var(--bg-card-subtle); color: var(--text-secondary) !important;
    border: 1px solid var(--border-color); padding: 5px 12px; border-radius: 999px; font-size: 0.74rem; font-weight: 600;
}

/* Signal meter */
.gauge { display: flex; gap: 5px; margin: 10px 0 12px 0; }
.gauge span { flex: 1; height: 8px; border-radius: 999px; background: var(--bg-card-subtle); box-shadow: inset 0 0 0 1px var(--border-color); }
.gauge span.on { background: var(--seg-c); box-shadow: 0 0 10px var(--seg-c); }
.sig-row { display: flex; justify-content: space-between; align-items: center; gap: 10px; padding: 8px 0; border-top: 1px solid var(--border-color); font-size: 0.84rem; }
.sig-tag { font-weight: 700; font-size: 0.74rem; padding: 2px 9px; border-radius: 999px; white-space: nowrap; }
.sig-pos { color: var(--accent-bright); background: var(--accent-soft); }
.sig-neg { color: #FF8A95; background: var(--neg-soft); }

/* Tour step card */
.step-card h3 { margin-top: 8px !important; }

/* Footer */
.app-footer {
    margin-top: 40px; padding: 18px 22px; border: 1px solid var(--border-color); border-radius: 16px;
    background: var(--bg-card); display: flex; justify-content: space-between; align-items: center;
    flex-wrap: wrap; gap: 14px; font-size: 0.8rem; color: var(--text-muted);
}
.app-footer img { width: 28px; height: 28px; border-radius: 8px; vertical-align: middle; margin-right: 10px; box-shadow: 0 0 14px rgba(46,232,138,0.30); }
.app-footer a.deploy { color: var(--accent) !important; text-decoration: none; font-weight: 600; border: 1px solid var(--border-strong); padding: 5px 12px; border-radius: 9px; }
.app-footer a.deploy:hover { background: var(--accent-soft); }


@media (max-width: 900px) {
    .hero-stats { grid-template-columns: repeat(2, minmax(86px, auto)); }
    .stock-hero-price { font-size: 1.7rem; }
}
"""


def inject_theme_css(theme_mode: str = "Dark"):
    """Injects the Neon Bull design system. (theme_mode kept for backwards compatibility.)"""
    st.markdown(f"<style>{_CSS.replace('__ROOT__', _ROOT_VARS)}</style>", unsafe_allow_html=True)


def inject_glassmorphism_css():
    inject_theme_css("Dark")


# -----------------------------------------------------------------------------
# Brand & layout components
# -----------------------------------------------------------------------------
def render_sidebar_brand(market_label: str = "NSE · NASDAQ"):
    """Logo + gradient wordmark at the top of the sidebar."""
    uri = logo_data_uri()
    img = f'<img src="{uri}" alt="StockMind AI logo"/>' if uri else get_icon("stock", NEON_GREEN, 40)
    st.markdown(
        f"""
        <div class="brand">
            {img}
            <div>
                <div class="brand-name">STOCKMIND AI</div>
                <div class="brand-tag">AI Market Analytics</div>
            </div>
        </div>
        <div class="status-chip">
            <span><span class="live-dot"></span>&nbsp; Engine online</span>
            <b>{market_label}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_label(text: str, icon_name: str = ""):
    ico = get_icon(icon_name, NEON_GREEN, 12).replace("margin-right:8px", "margin-right:2px") if icon_name else ""
    st.markdown(f'<div class="sb-section">{ico}{text}</div>', unsafe_allow_html=True)


def render_ticker_tape(items: list):
    """Scrolling market ticker. items = [(symbol, price_str, change_pct), ...]."""
    if not items:
        return
    cells = ""
    for sym, px, chg in items:
        cls = "tape-up" if chg >= 0 else "tape-dn"
        arrow = "▲" if chg >= 0 else "▼"
        cells += (
            f'<span class="tape-item"><span class="tape-sym">{sym}</span>'
            f'<span class="tape-px">{px}</span><span class="{cls}">{arrow} {chg:+.2f}%</span></span>'
        )
    st.markdown(
        f'<div class="tape"><div class="tape-track">{cells}{cells}</div></div>',
        unsafe_allow_html=True,
    )


def render_stock_hero(meta: dict, curr: str, latest_close: float, change: float, change_pct: float,
                      stats: list, period_label: str):
    """Big price banner. stats = [(label, value_str), ...] (up to 8)."""
    up = change >= 0
    chip_cls = "chg-up" if up else "chg-dn"
    arrow = "▲" if up else "▼"
    initials = "".join(ch for ch in meta["ticker"].replace(".NS", "")[:3]).upper()
    stats_html = "".join(
        f'<div><div class="hero-stat-l">{lab}</div><div class="hero-stat-v">{val}</div></div>' for lab, val in stats
    )
    st.markdown(
        f"""
        <div class="stock-hero rim">
            <div class="hero-left">
                <div class="hero-avatar">{initials}</div>
                <div>
                    <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:4px;">
                        <span class="stock-hero-name">{meta['company_name']}</span>
                        <span class="ticker-badge">{meta['ticker']}</span>
                    </div>
                    <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
                        <span class="stock-hero-price">{curr}{latest_close:,.2f}</span>
                        <span class="chg-chip {chip_cls}">{arrow} {change:+,.2f} ({change_pct:+.2f}%)</span>
                        <span style="font-size:0.74rem;color:var(--text-muted);">1D</span>
                    </div>
                    <div class="hero-meta">
                        <span><span class="live-dot"></span>&nbsp; Last close · {meta['last_timestamp']}</span>
                        <span>{meta['source']}</span>
                        <span>Window · <b style="color:var(--text-secondary)">{period_label}</b></span>
                    </div>
                </div>
            </div>
            <div class="hero-stats">{stats_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, subtitle: str, icon_name: str, app_mode: str, eyebrow: str = ""):
    """Page title with glowing icon chip and mode pill."""
    icon_svg = get_icon(icon_name, NEON_GREEN, 19)
    eb = f'<div class="sec-eyebrow">{eyebrow}</div>' if eyebrow else ""
    mode_ico = "brain" if app_mode == "Pro Mode" else "compass"
    st.markdown(
        f"""
        <div class="sec-head">
            <div>
                {eb}
                <h2 class="sec-title"><span class="icon-chip">{icon_svg}</span>{title}</h2>
                <p class="sec-sub">{subtitle}</p>
            </div>
            <div><span class="mode-pill">{get_icon(mode_ico, TEAL, 13).replace('margin-right:8px', 'margin-right:0')}{app_mode}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_simple_guide(title: str, plain_explanation: str, analogy: str = ""):
    """Beginner-friendly explanation card with neon rail."""
    analogy_html = (
        f'<div style="margin-top:7px;color:var(--text-secondary);font-size:0.85rem;"><strong style="color:var(--teal)">Think of it like this:</strong> {analogy}</div>'
        if analogy
        else ""
    )
    st.markdown(
        f"""
        <div class="simple-guide-card">
            <div class="simple-guide-title">{get_icon('compass', NEON_GREEN, 16)}{title}</div>
            <div class="simple-guide-body">{plain_explanation}</div>
            {analogy_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


_STATUS_COLORS = {
    "pos": (NEON_GREEN, "rgba(46,232,138,0.14)"),
    "neg": (CANDLE_RED, "rgba(255,77,94,0.14)"),
    "neutral": (TEAL, "rgba(31,181,224,0.14)"),
}


def render_kpi_card(title: str, value: str, subtitle: str = "", status: str = "neutral", icon_name: str = "stock"):
    """KPI tile with status-coloured glow bar and icon chip."""
    sub_class = {"pos": "kpi-sub-pos", "neg": "kpi-sub-neg"}.get(status, "kpi-sub-neutral")
    c, bg = _STATUS_COLORS.get(status, _STATUS_COLORS["neutral"])
    icon_svg = get_icon(icon_name, c, 14)
    st.markdown(
        f"""
        <div class="fin-card kpi-card" style="--kpi-c:{c};--kpi-bg:{bg};">
            <div class="kpi-title"><span class="kpi-ico">{icon_svg}</span>{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="{sub_class}">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_module_card(tag: str, title: str, desc: str, icon_name: str):
    st.markdown(
        f"""
        <div class="fin-card mod-card">
            <div><span class="ticker-badge">{tag}</span></div>
            <div class="mod-title"><span class="icon-chip" style="width:30px;height:30px;margin-right:10px;">{get_icon(icon_name, NEON_GREEN, 15)}</span>{title}</div>
            <div class="mod-desc">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_algorithm_explainer(algo_key: str, app_mode: str):
    """Displays the 10-point beginner explanation or concise pro summary."""
    info = ALGORITHM_EXPLANATIONS.get(algo_key)
    if not info:
        return

    label = f"Learn Simply: How {algo_key} Works" if app_mode == "Beginner Mode" else f"Technical Summary: {algo_key}"
    with st.expander(label, expanded=False):
        if app_mode == "Beginner Mode":
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    f"""
                    - **What is it?** {info['what']}
                    - **Why do we need it?** {info['why_need']}
                    - **Why used here?** {info['why_project']}
                    - **How it works:** {info['how_works']}
                    - **Data it uses:** {info['data_used']}
                    """
                )
            with c2:
                st.markdown(
                    f"""
                    - **Inside the math:** {info['internal']}
                    - **What it produces:** {info['output']}
                    - **How to read results:** {info['interpret']}
                    - **Best strengths:** {info['advantages']}
                    - **Limits to remember:** {info['limitations']}
                    """
                )
        else:
            st.markdown(
                f"""
                - **Purpose:** {info['why_project']}
                - **Internal Mechanism:** {info['internal']}
                - **Executive Summary:** *"{info['faculty_note']}"*
                """
            )


def render_insights_script(
    section_title: str,
    what_about: str,
    why_included: str,
    what_implemented: str,
    results_observed: str,
    key_learning: str,
    app_mode: str,
):
    """Renders a concise executive walkthrough panel (auto-expanded in Pro Mode)."""
    expanded_default = app_mode == "Pro Mode"
    with st.expander(f"Executive Walkthrough — {section_title}", expanded=expanded_default):
        st.markdown(
            f"""
            1. **Overview:** {what_about}
            2. **Purpose:** {why_included}
            3. **Implementation:** {what_implemented}
            4. **Observed Results:** {results_observed}
            5. **Key Takeaway:** {key_learning}
            """
        )


# Backward-compatible alias
render_faculty_script = render_insights_script


def render_code_explainer(snippet_key: str):
    """Renders a clean Code Explanation panel with key Python/ML function breakdowns."""
    item = CODE_EXPLAINERS.get(snippet_key)
    if not item:
        return
    with st.expander(f"Inspect Python Implementation ({item['title']})", expanded=False):
        st.code(item["snippet"], language="python")
        for fn_name, fn_desc in item["functions"]:
            st.markdown(f"- **`{fn_name}`** — {fn_desc}")


def render_price_range_bar(low_val: float, high_val: float, current_val: float, currency: str, label: str = "Period Range"):
    """High–low range bar with a glowing position marker."""
    span = max(high_val - low_val, 1e-6)
    pct = max(0.0, min(100.0, ((current_val - low_val) / span) * 100.0))
    st.markdown(
        f"""
        <div class="fin-card" style="padding:16px 18px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
                <span style="font-weight:700;font-size:0.92rem;font-family:var(--font-display);">{get_icon('stock', NEON_GREEN, 15)}{label} Range</span>
                <span class="ticker-badge">{pct:.0f}% of range</span>
            </div>
            <div style="position:relative;height:8px;background:var(--bg-card-subtle);border-radius:999px;box-shadow:inset 0 0 0 1px var(--border-color);">
                <div style="width:{pct:.1f}%;height:100%;background:linear-gradient(90deg, {TEAL}, {NEON_GREEN});border-radius:999px;box-shadow:0 0 12px rgba(46,232,138,0.45);"></div>
                <div style="position:absolute;left:calc({pct:.1f}% - 8px);top:-5px;width:18px;height:18px;border-radius:50%;background:{GREEN_BRIGHT};border:3px solid #0C151D;box-shadow:0 0 0 2px {NEON_GREEN},0 0 16px {NEON_GREEN};"></div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:0.78rem;color:var(--text-muted);margin-top:12px;font-family:var(--font-mono);">
                <span>L {currency}{low_val:,.2f}</span>
                <span style="color:var(--text-primary);font-weight:700;">{currency}{current_val:,.2f}</span>
                <span>H {currency}{high_val:,.2f}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_signal_meter(close_p: float, sma20: float, sma50: float, rsi14: float, macd: float, macd_sig: float):
    """Technical signal summary with a 4-segment glowing gauge."""
    signals = [
        ("Short trend · SMA20", close_p > sma20, "Price above 20-day avg" if close_p > sma20 else "Price below 20-day avg"),
        ("Medium trend · SMA50", close_p > sma50, "Price above 50-day avg" if close_p > sma50 else "Price below 50-day avg"),
        ("Momentum · MACD", macd > macd_sig, "MACD above signal line" if macd > macd_sig else "MACD below signal line"),
        ("Mood · RSI-14", 30 <= rsi14 <= 68, "Healthy room to grow" if 30 <= rsi14 <= 68 else ("Overbought (>68)" if rsi14 > 68 else "Oversold (<30)")),
    ]
    bull_count = sum(1 for _, is_bull, _ in signals if is_bull)
    if bull_count >= 3:
        verdict, v_color = "Bullish momentum", NEON_GREEN
    elif bull_count == 2:
        verdict, v_color = "Neutral · wait & watch", AMBER
    else:
        verdict, v_color = "Bearish · weak trend", CANDLE_RED

    segs = "".join(f'<span class="{"on" if i < bull_count else ""}"></span>' for i in range(4))
    rows_html = ""
    for name, is_bull, desc in signals:
        tag_cls, tag = ("sig-pos", "Positive") if is_bull else ("sig-neg", "Caution")
        rows_html += (
            f'<div class="sig-row"><span><strong>{name}</strong> '
            f'<span style="color:var(--text-muted);">— {desc}</span></span>'
            f'<span class="sig-tag {tag_cls}">● {tag}</span></div>'
        )

    st.markdown(
        f"""
        <div class="fin-card" style="--seg-c:{v_color};">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;">
                <span style="font-weight:700;font-size:0.95rem;font-family:var(--font-display);">{get_icon('sliders', v_color, 16)}Technical Signal Meter</span>
                <span style="color:{v_color};font-weight:700;font-size:0.86rem;font-family:var(--font-mono);">{bull_count}/4 · {verdict}</span>
            </div>
            <div class="gauge">{segs}</div>
            {rows_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_app_footer():
    """Branded footer."""
    uri = logo_data_uri()
    img = f'<img src="{uri}" alt=""/>' if uri else ""
    st.markdown(
        f"""
        <div class="app-footer">
            <div style="display:flex;align-items:center;flex-wrap:wrap;gap:4px;">
                {img}<strong style="color:var(--text-primary);font-family:var(--font-display);">STOCKMIND AI</strong>
                <span>&nbsp;·&nbsp; Market data via Yahoo Finance may be delayed. AI forecasts are statistical estimates, not financial advice.</span>
            </div>
            <div style="display:flex;align-items:center;gap:14px;">
                <span style="color:var(--pos);font-weight:600;"><span class="live-dot"></span>&nbsp; Engine online</span>
                <a class="deploy" href="https://share.streamlit.io/" target="_blank">Deploy to Cloud ↗</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
