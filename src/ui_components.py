"""
Modern Fintech UI Components (src/ui_components.py)
Production-grade fintech design system with Light, Dark, and System theme support.
High-contrast styling across all inputs, selectboxes, dropdown menus, expanders, and charts.
"""

import streamlit as st
from src.learning_content import (
    get_icon,
    ALGORITHM_EXPLANATIONS,
    CODE_EXPLAINERS,
)


_LIGHT_VARS = """
    --bg-app: #F8FAFC;
    --bg-sidebar: #FFFFFF;
    --bg-card: #FFFFFF;
    --bg-card-subtle: #F1F5F9;
    --bg-guide: linear-gradient(135deg, #ECFDF5 0%, #EFF6FF 100%);
    --border-color: #CBD5E1;
    --border-hover: #00B386;
    --text-primary: #0F172A;
    --text-secondary: #334155;
    --text-muted: #64748B;
    --accent: #00B386;
    --accent-soft: rgba(0, 179, 134, 0.14);
    --blue-accent: #2563EB;
    --pos: #009B72;
    --neg: #DC2626;
    --shadow-card: 0 2px 10px rgba(15, 23, 42, 0.06);
    --shadow-hover: 0 6px 20px rgba(15, 23, 42, 0.10);
    --input-bg: #FFFFFF;
    --alert-bg: #EFF6FF;
"""

_DARK_VARS = """
    --bg-app: #0B0E14;
    --bg-sidebar: #11151E;
    --bg-card: #131722;
    --bg-card-subtle: #1B202E;
    --bg-guide: linear-gradient(135deg, rgba(0, 179, 134, 0.14) 0%, rgba(37, 99, 235, 0.14) 100%);
    --border-color: #2D3446;
    --border-hover: #00C896;
    --text-primary: #F8FAFC;
    --text-secondary: #CBD5E1;
    --text-muted: #94A3B8;
    --accent: #00C896;
    --accent-soft: rgba(0, 200, 150, 0.16);
    --blue-accent: #38BDF8;
    --pos: #00C896;
    --neg: #F87171;
    --shadow-card: 0 4px 16px rgba(0, 0, 0, 0.30);
    --shadow-hover: 0 8px 24px rgba(0, 0, 0, 0.42);
    --input-bg: #1A1F2E;
    --alert-bg: #172033;
"""


def inject_theme_css(theme_mode: str = "System"):
    """Injects high-contrast fintech CSS supporting Light, Dark, and automatic OS System mode."""
    t = (theme_mode or "System").lower()
    if t == "light":
        root_block = f":root {{ {_LIGHT_VARS} }}"
    elif t == "dark":
        root_block = f":root {{ {_DARK_VARS} }}"
    else:
        root_block = (
            f":root {{ {_LIGHT_VARS} }}\n"
            f"@media (prefers-color-scheme: dark) {{ :root {{ {_DARK_VARS} }} }}"
        )

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        {root_block}

        /* Clean standalone app chrome: prevent invisible fixed stHeader from blocking top buttons */
        #MainMenu {{display: none !important;}}
        footer {{display: none !important;}}
        [data-testid="stDecoration"] {{display: none !important;}}
        [data-testid="stAppDeployButton"], .stDeployButton, [data-testid="stToolbar"] {{
            display: none !important;
        }}
        header[data-testid="stHeader"] {{
            background: transparent !important;
            pointer-events: none !important;
            height: 2.5rem !important;
        }}
        /* Keep sidebar open/close toggle clickable even though header bar is pass-through */
        header[data-testid="stHeader"] button,
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapseButton"] {{
            pointer-events: auto !important;
        }}

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        .stApp {{
            background-color: var(--bg-app);
            color: var(--text-primary);
        }}
        .block-container {{
            padding-top: 2.6rem !important;
            padding-bottom: 2.5rem !important;
            max-width: 1260px;
        }}

        /* Sidebar */
        [data-testid="stSidebar"] {{
            background-color: var(--bg-sidebar);
            border-right: 1px solid var(--border-color);
        }}

        /* Global high-contrast typography */
        .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
        h1, h2, h3, h4, h5, label,
        [data-testid="stWidgetLabel"] p,
        [data-testid="stRadio"] label p,
        [data-testid="stCheckbox"] label p {{
            color: var(--text-primary) !important;
        }}
        .stCaption, .stCaption p, small {{
            color: var(--text-muted) !important;
        }}
        code {{
            background-color: var(--bg-card-subtle) !important;
            color: var(--blue-accent) !important;
            padding: 2px 6px;
            border-radius: 5px;
        }}

        /* High-contrast Form Controls */
        [data-baseweb="select"] > div {{
            background-color: var(--input-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 10px !important;
            color: var(--text-primary) !important;
        }}
        [data-baseweb="select"] span,
        [data-baseweb="select"] div,
        [data-baseweb="select"] input {{
            color: var(--text-primary) !important;
            -webkit-text-fill-color: var(--text-primary) !important;
        }}
        [data-baseweb="select"] svg {{
            fill: var(--text-primary) !important;
            color: var(--text-primary) !important;
        }}

        /* Dropdown Popover Menu & Options */
        [data-baseweb="popover"],
        [data-baseweb="popover"] > div,
        [data-baseweb="menu"],
        ul[role="listbox"] {{
            background-color: var(--bg-card) !important;
            border-color: var(--border-color) !important;
        }}
        li[role="option"] {{
            background-color: var(--bg-card) !important;
            color: var(--text-primary) !important;
        }}
        li[role="option"]:hover,
        li[role="option"][aria-selected="true"] {{
            background-color: var(--accent-soft) !important;
            color: var(--accent) !important;
            font-weight: 600;
        }}

        /* Text & Number Inputs */
        [data-baseweb="input"],
        [data-baseweb="base-input"] {{
            background-color: var(--input-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 10px !important;
        }}
        [data-baseweb="input"] input,
        [data-baseweb="base-input"] input,
        .stTextInput input,
        .stNumberInput input {{
            background-color: var(--input-bg) !important;
            color: var(--text-primary) !important;
            -webkit-text-fill-color: var(--text-primary) !important;
            caret-color: var(--accent) !important;
        }}
        .stTextInput input::placeholder,
        .stNumberInput input::placeholder,
        input::placeholder {{
            color: var(--text-muted) !important;
            -webkit-text-fill-color: var(--text-muted) !important;
            opacity: 1 !important;
        }}
        .stNumberInput button {{
            background-color: var(--bg-card-subtle) !important;
            color: var(--text-primary) !important;
            border-color: var(--border-color) !important;
        }}
        .stNumberInput button svg {{
            fill: var(--text-primary) !important;
        }}

        /* Expanders */
        [data-testid="stExpander"] {{
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 12px !important;
            margin-bottom: 10px;
        }}
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary p,
        [data-testid="stExpander"] summary span {{
            color: var(--text-primary) !important;
            font-weight: 600;
        }}
        [data-testid="stExpander"] summary svg {{
            fill: var(--text-primary) !important;
            color: var(--text-primary) !important;
        }}

        /* Alerts */
        [data-testid="stAlert"] {{
            background-color: var(--alert-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 12px !important;
        }}
        [data-testid="stAlert"] p {{
            color: var(--text-primary) !important;
        }}

        /* Sliders */
        [data-testid="stSlider"] [data-testid="stThumbValue"],
        [data-testid="stSlider"] [data-testid="stTickBarMin"],
        [data-testid="stSlider"] [data-testid="stTickBarMax"] {{
            color: var(--text-primary) !important;
        }}

        /* Elevated Fintech Card */
        .fin-card {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 16px 18px;
            margin-bottom: 12px;
            box-shadow: var(--shadow-card);
            transition: border-color 0.15s ease, box-shadow 0.15s ease, transform 0.15s ease;
        }}
        .fin-card:hover {{
            border-color: var(--border-hover);
            box-shadow: var(--shadow-hover);
            transform: translateY(-1px);
        }}

        /* Hero Stock Price Banner */
        .stock-hero {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 18px 22px;
            margin-bottom: 18px;
            box-shadow: var(--shadow-card);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 14px;
        }}
        .stock-hero-name {{
            font-size: 1.20rem;
            font-weight: 800;
            color: var(--text-primary);
            letter-spacing: -0.02em;
        }}
        .stock-hero-price {{
            font-size: 1.95rem;
            font-weight: 800;
            color: var(--text-primary);
            letter-spacing: -0.03em;
            line-height: 1.1;
        }}

        /* Friendly Guide Banner */
        .simple-guide-card {{
            background: var(--bg-guide);
            border: 1px solid var(--border-color);
            border-left: 4px solid var(--accent);
            border-radius: 12px;
            padding: 14px 18px;
            margin-bottom: 16px;
        }}
        .simple-guide-title {{
            font-size: 0.90rem;
            font-weight: 700;
            color: var(--accent);
            margin-bottom: 4px;
            display: flex;
            align-items: center;
        }}
        .simple-guide-body {{
            font-size: 0.92rem;
            color: var(--text-primary);
            line-height: 1.5;
        }}

        /* KPI Text */
        .kpi-title {{
            font-size: 0.78rem;
            color: var(--text-muted);
            font-weight: 600;
            margin-bottom: 6px;
            display: flex;
            align-items: center;
        }}
        .kpi-value {{
            font-size: 1.50rem;
            font-weight: 800;
            color: var(--text-primary);
            letter-spacing: -0.02em;
        }}
        .kpi-sub-pos {{ font-size: 0.84rem; color: var(--pos); font-weight: 600; margin-top: 4px; }}
        .kpi-sub-neg {{ font-size: 0.84rem; color: var(--neg); font-weight: 600; margin-top: 4px; }}
        .kpi-sub-neutral {{ font-size: 0.84rem; color: var(--blue-accent); font-weight: 500; margin-top: 4px; }}

        /* Pill Badges */
        .ticker-badge {{
            display: inline-block;
            background: var(--accent-soft);
            color: var(--accent) !important;
            padding: 3px 10px;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 700;
        }}
        .mode-pill {{
            display: inline-block;
            background: var(--bg-card-subtle);
            color: var(--text-secondary) !important;
            border: 1px solid var(--border-color);
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 600;
        }}

        /* Segmented Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 6px;
            background-color: var(--bg-card-subtle);
            padding: 5px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px;
            padding: 7px 14px;
            color: var(--text-muted) !important;
            font-weight: 600;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: var(--bg-card) !important;
            color: var(--text-primary) !important;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        }}

        /* Crisp Interactive Buttons (Secondary & Primary Active Pills) */
        .stButton > button, .stDownloadButton > button {{
            position: relative;
            z-index: 2;
            cursor: pointer !important;
            border-radius: 10px;
            border: 1px solid var(--border-color) !important;
            background-color: var(--bg-card) !important;
            color: var(--text-primary) !important;
            font-weight: 600;
            transition: border-color 0.14s ease, background-color 0.14s ease, color 0.14s ease;
        }}
        .stButton > button p, .stDownloadButton > button p {{
            color: var(--text-primary) !important;
            margin: 0 !important;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            border-color: var(--accent) !important;
            color: var(--accent) !important;
            background-color: var(--accent-soft) !important;
        }}
        .stButton > button:hover p, .stDownloadButton > button:hover p {{
            color: var(--accent) !important;
        }}

        /* Active Primary Pill (e.g., Selected Watchlist Stock) */
        .stButton > button[kind="primary"],
        .stButton > button[data-testid="baseButton-primary"],
        .stButton > button[data-testid="stBaseButton-primary"] {{
            background-color: var(--accent) !important;
            border-color: var(--accent) !important;
            color: #FFFFFF !important;
        }}
        .stButton > button[kind="primary"] p,
        .stButton > button[data-testid="baseButton-primary"] p,
        .stButton > button[data-testid="stBaseButton-primary"] p,
        .stButton > button[kind="primary"]:hover p,
        .stButton > button[data-testid="baseButton-primary"]:hover p,
        .stButton > button[data-testid="stBaseButton-primary"]:hover p {{
            color: #FFFFFF !important;
        }}
        .stButton > button[kind="primary"]:hover,
        .stButton > button[data-testid="baseButton-primary"]:hover,
        .stButton > button[data-testid="stBaseButton-primary"]:hover {{
            background-color: var(--pos) !important;
            border-color: var(--pos) !important;
            color: #FFFFFF !important;
        }}

        /* Ensure all interactive controls have clean pointer hitboxes */
        [data-testid="stRadio"] label,
        [data-testid="stCheckbox"] label,
        [data-testid="stExpander"] summary,
        .stTabs [data-baseweb="tab"] {{
            cursor: pointer !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_glassmorphism_css():
    inject_theme_css("Dark")


def render_section_header(title: str, subtitle: str, icon_name: str, app_mode: str):
    """Renders a clean, high-contrast page title with Lucide SVG icon and mode pill."""
    icon_svg = get_icon(icon_name, "#00B386", 22)
    st.markdown(
        f"""
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;margin-bottom:12px;gap:8px;">
            <div>
                <h2 style="margin:0;font-weight:800;font-size:1.45rem;color:var(--text-primary);display:flex;align-items:center;">
                    {icon_svg}{title}
                </h2>
                <p style="margin:3px 0 0 0;color:var(--text-muted);font-size:0.90rem;">{subtitle}</p>
            </div>
            <div><span class="mode-pill">{app_mode}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_simple_guide(title: str, plain_explanation: str, analogy: str = ""):
    """Renders an ultra-clear, beginner-friendly explanation card."""
    analogy_html = (
        f'<div style="margin-top:6px;color:var(--text-secondary);font-size:0.86rem;"><strong>Think of it like this:</strong> {analogy}</div>'
        if analogy
        else ""
    )
    st.markdown(
        f"""
        <div class="simple-guide-card">
            <div class="simple-guide-title">{get_icon('compass', '#00B386', 16)}{title}</div>
            <div class="simple-guide-body">{plain_explanation}</div>
            {analogy_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(title: str, value: str, subtitle: str = "", status: str = "neutral", icon_name: str = "stock"):
    """Renders a clean, theme-aware fintech KPI card."""
    sub_class = "kpi-sub-pos" if status == "pos" else ("kpi-sub-neg" if status == "neg" else "kpi-sub-neutral")
    icon_color = "#00B386" if status == "pos" else ("#EB5B3C" if status == "neg" else "#0EA5E9")
    icon_svg = get_icon(icon_name, icon_color, 15)
    st.markdown(
        f"""
        <div class="fin-card">
            <div class="kpi-title">{icon_svg}{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="{sub_class}">{subtitle}</div>
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
    """Renders a Groww/Yahoo Finance-style visual High-Low price range bar."""
    span = max(high_val - low_val, 1e-6)
    pct = max(0.0, min(100.0, ((current_val - low_val) / span) * 100.0))
    st.markdown(
        f"""
        <div class="fin-card" style="padding:14px 18px;">
            <div style="display:flex;justify-content:space-between;font-size:0.80rem;color:var(--text-muted);margin-bottom:8px;">
                <span><strong>{label} Low:</strong> {currency}{low_val:,.2f}</span>
                <span><strong>Current Position:</strong> {pct:.0f}%</span>
                <span><strong>{label} High:</strong> {currency}{high_val:,.2f}</span>
            </div>
            <div style="position:relative;height:8px;background:var(--bg-card-subtle);border-radius:999px;overflow:visible;">
                <div style="width:{pct:.1f}%;height:100%;background:linear-gradient(90deg, #0EA5E9, #00B386);border-radius:999px;"></div>
                <div style="position:absolute;left:calc({pct:.1f}% - 7px);top:-4px;width:15px;height:15px;border-radius:50%;background:var(--accent);border:2px solid var(--bg-card);box-shadow:0 0 0 2px var(--accent-soft);"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_signal_meter(close_p: float, sma20: float, sma50: float, rsi14: float, macd: float, macd_sig: float):
    """Renders a TradingView-style Technical Signal Summary card in plain English."""
    signals = [
        ("Short Trend (20-Day Avg)", close_p > sma20, "Price is above 20-day average" if close_p > sma20 else "Price is below 20-day average"),
        ("Medium Trend (50-Day Avg)", close_p > sma50, "Price is above 50-day average" if close_p > sma50 else "Price is below 50-day average"),
        ("Momentum Speed (MACD)", macd > macd_sig, "MACD is above signal line" if macd > macd_sig else "MACD is below signal line"),
        ("Market Mood (RSI-14)", 30 <= rsi14 <= 68, "Healthy room to grow" if 30 <= rsi14 <= 68 else ("Overbought (>68)" if rsi14 > 68 else "Oversold (<30)")),
    ]
    bull_count = sum(1 for _, is_bull, _ in signals if is_bull)
    if bull_count >= 3:
        verdict, v_color = "Positive Momentum (Bullish)", "#00B386"
    elif bull_count == 2:
        verdict, v_color = "Mixed / Wait & Watch (Neutral)", "#F59E0B"
    else:
        verdict, v_color = "Cautious / Weak Trend (Bearish)", "#EB5B3C"

    rows_html = ""
    for name, is_bull, desc in signals:
        dot_c = "#00B386" if is_bull else "#EB5B3C"
        tag = "Positive" if is_bull else "Caution"
        rows_html += (
            f'<div style="display:flex;justify-content:space-between;align-items:center;padding:5px 0;border-bottom:1px solid var(--border-color);font-size:0.84rem;">'
            f'<span><strong>{name}</strong> <span style="color:var(--text-muted);">— {desc}</span></span>'
            f'<span style="color:{dot_c};font-weight:700;">● {tag}</span></div>'
        )

    st.markdown(
        f"""
        <div class="fin-card">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                <span style="font-weight:700;font-size:0.95rem;color:var(--text-primary);">{get_icon('sliders', v_color, 16)}Technical Signal Meter</span>
                <span style="color:{v_color};font-weight:800;font-size:0.88rem;">{bull_count}/4 Positive • {verdict}</span>
            </div>
            {rows_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_app_footer():
    """Renders a sleek SaaS footer with Deploy/Cloud Share moved cleanly to the bottom."""
    st.markdown(
        """
        <div style="margin-top:36px;padding:16px 20px;border-top:1px solid var(--border-color);display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;font-size:0.80rem;color:var(--text-muted);">
            <div>
                <strong style="color:var(--text-primary);">STOCKMIND AI</strong> &nbsp;&bull;&nbsp;
                Market data via Yahoo Finance may be delayed. AI forecasts are statistical estimates, not financial advice.
            </div>
            <div style="display:flex;align-items:center;gap:14px;">
                <span style="color:var(--pos);font-weight:600;">● Engine Online</span>
                <a href="https://share.streamlit.io/" target="_blank" style="color:var(--accent);text-decoration:none;font-weight:600;border:1px solid var(--border-color);padding:4px 10px;border-radius:8px;">Deploy to Cloud ↗</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
