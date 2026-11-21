"""
STOCKMIND AI – Intelligent Stock Market Analytics Platform
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

from src.data_collection import (
    INDIAN_STOCKS,
    INTERNATIONAL_STOCKS,
    fetch_stock_data,
    fetch_multi_stock_behavior_dataset,
    resolve_ticker_symbol,
)
from src.preprocessing import clean_stock_data, chronological_train_test_split
from src.feature_engineering import (
    FEATURE_COLUMNS,
    INDICATOR_EXPLANATIONS,
    prepare_modeling_dataset,
)
from src.regression import (
    run_all_regression_models,
    run_multivariate_linear_regression,
    run_logistic_regression_direction,
)
from src.dimensionality_reduction import (
    apply_low_variance_filter,
    apply_high_correlation_filter,
    apply_forward_backward_selection,
    apply_mrmr_selection,
    run_subspace_projections,
    compare_before_after_reduction,
)
from src.clustering import run_stock_clustering_suite
from src.reinforcement_learning import (
    run_k_armed_bandit_simulation,
    run_rl_trading_simulation,
)
from src.neural_network import (
    BIOLOGICAL_VS_ARTIFICIAL_TABLE,
    calculate_single_neuron_output,
    run_mlp_and_perceptron_classification,
)
from src.prediction import PREDICTION_MODELS, generate_stock_predictions
from src.visualization import (
    plot_candlestick_with_ma,
    plot_rsi_chart,
    plot_macd_chart,
    plot_volume_chart,
    plot_regression_predictions,
    plot_regression_metrics_comparison,
    plot_gradient_descent_curve,
    plot_correlation_heatmap,
    plot_pca_scree,
    plot_2d_projection,
    plot_elbow_and_silhouette,
    plot_cluster_scatter,
    create_matplotlib_dendrogram,
    plot_rl_portfolio_performance,
    plot_activation_functions,
    plot_mlp_loss_curves,
    plot_confusion_matrix_chart,
    plot_future_forecast_chart,
    plot_stock_comparison_chart,
    plot_investment_growth_chart,
)
from src.learning_content import (
    get_icon,
    GLOSSARY_ITEMS,
    GUIDED_TOUR_STEPS,
    get_pipeline_svg_diagram,
    get_ann_svg_diagram,
    get_rl_svg_diagram,
)
from src.ui_components import (
    NEON_GREEN,
    FAVICON_PATH,
    inject_theme_css,
    render_sidebar_brand,
    render_sidebar_label,
    render_ticker_tape,
    render_stock_hero,
    render_module_card,
    render_section_header,
    render_simple_guide,
    render_kpi_card,
    render_algorithm_explainer,
    render_insights_script,
    render_code_explainer,
    render_price_range_bar,
    render_signal_meter,
    render_app_footer,
)

try:
    _PAGE_ICON = Image.open(FAVICON_PATH)
except Exception:
    _PAGE_ICON = "📈"

st.set_page_config(
    page_title="STOCKMIND AI – Intelligent Stock Market Analytics",
    page_icon=_PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

MAIN_NAV = [
    "Overview",
    "Compare & Returns",
    "Price Forecast",
    "AI Analytics Studio",
    "Charts & Data",
    "Learn & Glossary",
    "Model Benchmarks",
]

LAB_NAMES = [
    "Price Models (Regression)",
    "Feature Lab (PCA & Selection)",
    "Market Clusters",
    "Strategy Simulator (RL)",
    "Neural Net (MLP)",
]

ALL_STOCKS = {**INDIAN_STOCKS, **INTERNATIONAL_STOCKS}
STOCK_LABELS = [f"{k} — {v.split(' (')[0]}" for k, v in ALL_STOCKS.items()]
TICKER_TO_LABEL = {k: f"{k} — {v.split(' (')[0]}" for k, v in ALL_STOCKS.items()}
QUICK_WATCHLIST = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "AAPL", "NVDA"]
TAPE_TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS",
                "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA", "META"]


# =============================================================================
# SINGLE-CLICK STATE CALLBACKS
# =============================================================================
def _init_state():
    defaults = {
        "active_nav": "Overview",
        "active_lab": LAB_NAMES[0],
        "active_ticker": "RELIANCE.NS",
        "stock_dropdown": TICKER_TO_LABEL["RELIANCE.NS"],
        "custom_ticker_input": "",
        "period_choice": "2y",
        "chart_style": "Area",
        "theme_mode": "Dark",
        "app_mode": "Beginner Mode",
        "train_split_pct": 80,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _on_select_watchlist(ticker: str):
    st.session_state.active_ticker = ticker
    if ticker in TICKER_TO_LABEL:
        st.session_state.stock_dropdown = TICKER_TO_LABEL[ticker]
    st.session_state.custom_ticker_input = ""


def _on_dropdown_change():
    picked = st.session_state.stock_dropdown.split(" — ")[0]
    st.session_state.active_ticker = picked
    st.session_state.custom_ticker_input = ""


def _on_custom_ticker():
    raw = st.session_state.custom_ticker_input.strip().upper()
    if raw:
        resolved = resolve_ticker_symbol(raw)
        st.session_state.active_ticker = resolved
        if resolved in TICKER_TO_LABEL:
            st.session_state.stock_dropdown = TICKER_TO_LABEL[resolved]


def _on_open_studio_lab(lab_name: str):
    st.session_state.active_nav = "AI Analytics Studio"
    st.session_state.active_lab = lab_name


# =============================================================================
# CACHED COMPUTATION LAYER
# =============================================================================
@st.cache_data(ttl=900, show_spinner=False)
def load_and_prepare_stock(ticker: str, period: str):
    raw_df, meta = fetch_stock_data(ticker, period=period, use_cache=True)
    if raw_df is None or meta["status"] != "ok":
        return None, None, meta, None, None, None, None
    clean_df, clean_report = clean_stock_data(raw_df)
    modeling_df, latest_row, full_feat_df = prepare_modeling_dataset(clean_df)
    return raw_df, clean_df, meta, clean_report, modeling_df, latest_row, full_feat_df


@st.cache_data(ttl=900, show_spinner=False)
def load_ticker_tape(tickers: tuple):
    """Latest close + 1D % change for the scrolling market tape."""
    items = []
    for sym in tickers:
        try:
            df, meta = fetch_stock_data(sym, period="6mo", use_cache=True)
        except Exception:
            continue
        if df is None or len(df) < 2:
            continue
        last, prev = float(df["Close"].iloc[-1]), float(df["Close"].iloc[-2])
        cur = "₹" if sym.endswith(".NS") else "$"
        items.append((sym.replace(".NS", ""), f"{cur}{last:,.2f}", (last / prev - 1.0) * 100.0 if prev else 0.0))
    return items


@st.cache_data(ttl=1800, show_spinner=False)
def load_multi_stock_dataset(period: str = "1y"):
    return fetch_multi_stock_behavior_dataset(period=period)


@st.cache_data(ttl=900, show_spinner=False)
def cached_regression(ticker: str, period: str, train_ratio: float, ridge_a: float, lasso_a: float, poly_d: int):
    _, _, _, _, mod_df, _, _ = load_and_prepare_stock(ticker, period)
    X_tr, X_te, y_tr, y_te, _, _ = chronological_train_test_split(mod_df, FEATURE_COLUMNS, "Target_Next_Close", train_ratio)
    _, _, y_dir_tr, y_dir_te, _, _ = chronological_train_test_split(mod_df, FEATURE_COLUMNS, "Target_Direction", train_ratio)
    reg = run_all_regression_models(X_tr, X_te, y_tr, y_te, ridge_alpha=ridge_a, lasso_alpha=lasso_a, poly_degree=poly_d)
    mv = run_multivariate_linear_regression(mod_df, FEATURE_COLUMNS, train_ratio=train_ratio)
    log = run_logistic_regression_direction(X_tr, X_te, y_dir_tr, y_dir_te)
    return reg, mv, log


@st.cache_data(ttl=900, show_spinner=False)
def cached_dim_reduction(ticker: str, period: str, train_ratio: float, var_t: float, corr_t: float, n_sel: int):
    _, _, _, _, mod_df, _, _ = load_and_prepare_stock(ticker, period)
    X_tr, X_te, y_tr, y_te, _, _ = chronological_train_test_split(mod_df, FEATURE_COLUMNS, "Target_Next_Close", train_ratio)
    _, _, y_trd_tr, y_trd_te, _, _ = chronological_train_test_split(mod_df, FEATURE_COLUMNS, "Target_Trend_Class", train_ratio)
    lv = apply_low_variance_filter(X_tr, threshold=var_t)
    hc = apply_high_correlation_filter(X_tr, threshold=corr_t)
    fb = apply_forward_backward_selection(X_tr, y_tr, n_features=n_sel)
    mrmr = apply_mrmr_selection(X_tr, y_tr, n_features=n_sel)
    proj = run_subspace_projections(X_tr, X_te, y_trd_tr, y_trd_te)
    ba = compare_before_after_reduction(X_tr, X_te, y_tr, y_te, hc["selected"], mrmr["selected"], fb["forward_selected"])
    return lv, hc, fb, mrmr, proj, ba


@st.cache_data(ttl=900, show_spinner=False)
def cached_clustering(k_choice: int, dist_choice: str):
    multi_df = load_multi_stock_dataset("1y")
    return run_stock_clustering_suite(multi_df, n_clusters=k_choice, distance_metric=dist_choice)


@st.cache_data(ttl=900, show_spinner=False)
def cached_rl(ticker: str, period: str, algo: str, episodes: int, alpha: float, gamma: float):
    _, _, _, _, mod_df, _, _ = load_and_prepare_stock(ticker, period)
    rl_res = run_rl_trading_simulation(mod_df, algorithm=algo, episodes=episodes, alpha=alpha, gamma=gamma, initial_capital=100000.0)
    bandit_res = run_k_armed_bandit_simulation(mod_df)
    return rl_res, bandit_res


@st.cache_data(ttl=900, show_spinner=False)
def cached_mlp(ticker: str, period: str, train_ratio: float, hidden_tuple: tuple, act: str, epochs: int):
    _, _, _, _, mod_df, _, _ = load_and_prepare_stock(ticker, period)
    X_tr, X_te, y_dir_tr, y_dir_te, _, _ = chronological_train_test_split(mod_df, FEATURE_COLUMNS, "Target_Direction", train_ratio)
    return run_mlp_and_perceptron_classification(X_tr, X_te, y_dir_tr, y_dir_te, hidden_layers=hidden_tuple, activation=act, max_epochs=epochs)


@st.cache_data(ttl=900, show_spinner=False)
def cached_prediction(ticker: str, period: str, train_ratio: float, model_name: str, use_mrmr: bool, horizon: int):
    _, _, _, _, mod_df, latest_row, _ = load_and_prepare_stock(ticker, period)
    X_tr, _, y_tr, _, _, _ = chronological_train_test_split(mod_df, FEATURE_COLUMNS, "Target_Next_Close", train_ratio)
    feats = apply_mrmr_selection(X_tr, y_tr, n_features=8)["selected"] if use_mrmr else FEATURE_COLUMNS
    return generate_stock_predictions(mod_df, latest_row, feats, selected_model_name=model_name, train_ratio=train_ratio, forecast_horizon_days=horizon)


def main():
    _init_state()

    theme_mode = "Dark"
    inject_theme_css(theme_mode)

    with st.sidebar:
        render_sidebar_brand()

        render_sidebar_label("Navigate", "dashboard")
        st.radio("Navigation", MAIN_NAV, key="active_nav", label_visibility="collapsed")
        selected_nav = st.session_state.active_nav

        render_sidebar_label("Stock", "database")
        st.selectbox(
            "Popular Stocks",
            STOCK_LABELS,
            key="stock_dropdown",
            on_change=_on_dropdown_change,
            label_visibility="collapsed",
        )
        st.text_input(
            "Custom Symbol",
            key="custom_ticker_input",
            on_change=_on_custom_ticker,
            placeholder="Type any symbol (e.g. ITC, WIPRO, META) + Enter",
            label_visibility="collapsed",
        )

        st.caption("Pick from 30 popular NSE & US stocks, or type any symbol and press Enter.")

        render_sidebar_label("Settings", "sliders")
        with st.expander("Preferences", expanded=False):
            st.radio("Experience Mode", ["Beginner Mode", "Pro Mode"], key="app_mode")
            st.slider("Train / Test Split (%)", 70, 90, key="train_split_pct", step=5)

        st.caption("STOCKMIND AI · Real-time analytics engine")

    app_mode = st.session_state.app_mode
    train_ratio = st.session_state.train_split_pct / 100.0
    active_ticker = st.session_state.active_ticker
    period_choice = st.session_state.period_choice

    # Scrolling market tape
    render_ticker_tape(load_ticker_tape(tuple(TAPE_TICKERS)))

    # 1-Click Active Watchlist Pills (Clearly highlights active stock & switches immediately!)
    w_cols = st.columns(len(QUICK_WATCHLIST))
    for i, sym in enumerate(QUICK_WATCHLIST):
        with w_cols[i]:
            is_selected = sym == active_ticker
            label = f"● {sym.replace('.NS', '')}" if is_selected else sym.replace(".NS", "")
            st.button(
                label,
                key=f"wl_{sym}",
                on_click=_on_select_watchlist,
                args=(sym,),
                type="primary" if is_selected else "secondary",
                width="stretch",
            )

    # Load Data (Cached)
    raw_df, clean_df, meta, clean_report, modeling_df, latest_row, full_feat_df = load_and_prepare_stock(
        active_ticker, period_choice
    )
    if clean_df is None or meta["status"] != "ok":
        st.error(meta.get("error", "Could not find this stock symbol."))
        st.info("Click any company pill above or pick from the 30 popular Indian & Global stocks in the sidebar.")
        render_app_footer()
        return

    curr = "₹" if "INR" in meta["currency"] else "$"
    latest_close = float(clean_df["Close"].iloc[-1])
    prev_close = float(clean_df["Close"].iloc[-2]) if len(clean_df) > 1 else latest_close
    daily_change = latest_close - prev_close
    daily_change_pct = (daily_change / prev_close) * 100.0

    def _fmt_vol(v: float) -> str:
        for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
            if v >= div:
                return f"{v / div:.2f}{unit}"
        return f"{v:,.0f}"

    last_row = clean_df.iloc[-1]
    avg_vol20 = float(clean_df["Volume"].tail(20).mean())
    hero_stats = [
        ("Open", f"{curr}{float(last_row['Open']):,.2f}"),
        ("Prev close", f"{curr}{prev_close:,.2f}"),
        ("Day high", f"{curr}{float(last_row['High']):,.2f}"),
        ("Day low", f"{curr}{float(last_row['Low']):,.2f}"),
        ("Volume", _fmt_vol(float(last_row["Volume"]))),
        ("Avg vol 20D", _fmt_vol(avg_vol20)),
        (f"{period_choice.upper()} high", f"{curr}{float(clean_df['High'].max()):,.0f}"),
        (f"{period_choice.upper()} low", f"{curr}{float(clean_df['Low'].min()):,.0f}"),
    ]
    render_stock_hero(meta, curr, latest_close, daily_change, daily_change_pct, hero_stats, period_choice.upper())

    X_train, X_test, y_train, y_test, dates_train, dates_test = chronological_train_test_split(
        modeling_df, FEATURE_COLUMNS, "Target_Next_Close", train_ratio=train_ratio
    )

    # =========================================================================
    # 1. OVERVIEW
    # =========================================================================
    if selected_nav == "Overview":
        pred_summary = cached_prediction(active_ticker, period_choice, train_ratio, "Ridge Regression (L2 Regularized)", False, 5)
        latest_rsi = float(full_feat_df["RSI_14"].iloc[-1])
        latest_sma20 = float(full_feat_df["SMA_20"].iloc[-1])
        latest_sma50 = float(full_feat_df["SMA_50"].iloc[-1])
        latest_macd = float(full_feat_df["MACD"].iloc[-1])
        latest_macd_sig = float(full_feat_df["MACD_Signal"].iloc[-1])
        mood_label = "Overbought (>70)" if latest_rsi > 70 else ("Oversold (<30)" if latest_rsi < 30 else "Balanced")
        dir_word = "rise slightly" if pred_summary["expected_change"] >= 0 else "dip slightly"

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            render_kpi_card("Next-Day AI Estimate", f"{curr}{pred_summary['next_day_predicted_close']:,.2f}", f"{pred_summary['expected_change']:+.2f} ({pred_summary['expected_change_pct']:+.2f}%)", "pos" if pred_summary["expected_change"] >= 0 else "neg", "target")
        with c2:
            render_kpi_card("Model Fit (Test R²)", f"{pred_summary['test_metrics']['R2']*100:.1f}%", f"Avg error: ±{curr}{pred_summary['test_metrics']['MAE']:.2f}", "neutral", "award")
        with c3:
            render_kpi_card("Momentum (RSI-14)", f"{latest_rsi:.0f} / 100", mood_label, "neg" if latest_rsi > 70 else ("pos" if latest_rsi < 30 else "neutral"), "sliders")
        with c4:
            hi_w = float(clean_df["High"].max())
            lo_w = float(clean_df["Low"].min())
            render_kpi_card(f"{period_choice.upper()} High / Low", f"{curr}{hi_w:,.0f}", f"Low: {curr}{lo_w:,.0f}", "neutral", "stock")

        # Inline Chart Controls
        ctrl_l, ctrl_r = st.columns([2, 1])
        with ctrl_l:
            st.radio("Time Range", ["6mo", "1y", "2y", "5y"], key="period_choice", horizontal=True)
        with ctrl_r:
            st.radio("Chart Style", ["Area", "Candlestick"], key="chart_style", horizontal=True)

        st.plotly_chart(
            plot_candlestick_with_ma(full_feat_df, active_ticker, theme=theme_mode, chart_style=st.session_state.chart_style),
            width="stretch",
        )

        # Groww / TradingView Signature Visual Widgets: Range Bar + Technical Signal Meter
        sig_l, sig_r = st.columns([1, 1.15])
        with sig_l:
            render_price_range_bar(lo_w, hi_w, latest_close, curr, label=f"{period_choice.upper()} Price")
            render_simple_guide(
                f"Plain-English Snapshot for {meta['ticker']}",
                f"Share price is <strong>{curr}{latest_close:,.2f}</strong>. Based on recent trends, the AI estimates it may {dir_word} toward "
                f"<strong>{curr}{pred_summary['next_day_predicted_close']:,.2f}</strong> next session.",
                "Just like a weather forecast, AI estimates are built from past patterns—never guaranteed!",
            )
        with sig_r:
            render_signal_meter(latest_close, latest_sma20, latest_sma50, latest_rsi, latest_macd, latest_macd_sig)

        st.markdown(
            f'<div class="sec-head" style="margin-top:18px;"><div><div class="sec-eyebrow">5 ML engines</div>'
            f'<h3 class="sec-title" style="font-size:1.25rem;"><span class="icon-chip">{get_icon("layers", NEON_GREEN, 17)}</span>AI Analytics Modules</h3></div></div>',
            unsafe_allow_html=True,
        )
        u_cols = st.columns(5)
        module_cards = [
            ("Regression", "Price Models", "10 price models + Up/Down direction", "regression", LAB_NAMES[0]),
            ("Features", "Feature Lab", "Feature selection, mRMR, PCA & LDA", "layers", LAB_NAMES[1]),
            ("Clusters", "Market Groups", "Group 12 stocks by risk & return", "cluster", LAB_NAMES[2]),
            ("Simulation", "Strategy Bot", "Virtual-capital RL trading simulator", "bot", LAB_NAMES[3]),
            ("Deep Learning", "Neural Net", "Neuron calculator & MLP classifier", "brain", LAB_NAMES[4]),
        ]
        for idx, (tag, title, desc, icon_n, target_lab) in enumerate(module_cards):
            with u_cols[idx]:
                render_module_card(tag, title, desc, icon_n)
                st.button(
                    f"Open {title} →",
                    key=f"open_mod_{idx}",
                    on_click=_on_open_studio_lab,
                    args=(target_lab,),
                    width="stretch",
                )

    # =========================================================================
    # 2. COMPARE & RETURNS (GROWW / YAHOO FINANCE POPULAR FEATURES!)
    # =========================================================================
    elif selected_nav == "Compare & Returns":
        render_section_header(
            "Compare Stocks & What-If Returns Calculator",
            "Compare any two companies side by side or test how an investment would have grown over time.",
            "stock",
            app_mode, eyebrow=selected_nav,
        )

        t_cmp, t_calc = st.tabs(["Head-to-Head Stock Comparison", "What-If Investment Calculator"])

        with t_cmp:
            render_simple_guide(
                "How Stock Comparison Works",
                "Both stocks start at <strong>0%</strong> on Day 1 so you can fairly compare which company grew faster — even if one stock costs ₹500 and the other costs ₹3,500!",
                "Like comparing two runners starting from the exact same starting line.",
            )
            default_rival = "TCS.NS" if active_ticker != "TCS.NS" else "INFY.NS"
            rival_idx = list(ALL_STOCKS.keys()).index(default_rival) if default_rival in ALL_STOCKS else 1
            rival_label = st.selectbox(f"Compare {active_ticker} against:", STOCK_LABELS, index=rival_idx)
            rival_ticker = rival_label.split(" — ")[0]

            _, clean_df2, meta2, _, _, _, feat_df2 = load_and_prepare_stock(rival_ticker, period_choice)
            if clean_df2 is not None:
                st.plotly_chart(plot_stock_comparison_chart(clean_df, active_ticker, clean_df2, rival_ticker, theme=theme_mode), width="stretch")

                def _quick_stats(df_c, df_f, sym_name):
                    ret_pct = ((float(df_c["Close"].iloc[-1]) / float(df_c["Close"].iloc[0])) - 1.0) * 100.0
                    vol_pct = float(df_c["Close"].pct_change().std() * np.sqrt(252) * 100.0)
                    dd_pct = float(((df_c["Close"] - df_c["Close"].cummax()) / df_c["Close"].cummax() * 100.0).min())
                    rsi_now = float(df_f["RSI_14"].iloc[-1])
                    return {
                        "Stock": sym_name,
                        f"Total Return ({period_choice.upper()}) %": round(ret_pct, 2),
                        "Yearly Risk / Volatility %": round(vol_pct, 2),
                        "Biggest Drop (Max Drawdown %)": round(dd_pct, 2),
                        "Current Mood (RSI-14)": round(rsi_now, 1),
                    }

                st.dataframe(pd.DataFrame([_quick_stats(clean_df, full_feat_df, active_ticker), _quick_stats(clean_df2, feat_df2, rival_ticker)]), width="stretch")

        with t_calc:
            render_simple_guide(
                "What-If Returns Calculator",
                f"Pick any starting amount below to see what it would be worth today if you had invested in <strong>{active_ticker}</strong> at the start of the <strong>{period_choice.upper()}</strong> window.",
                "Just like Groww's returns calculator — great for understanding real profit, loss, and risk!",
            )
            default_amt = 10000 if curr == "₹" else 1000
            step_amt = 5000 if curr == "₹" else 500
            invested = st.number_input(f"Amount Invested ({curr})", min_value=100, max_value=10000000, value=default_amt, step=step_amt)

            start_p = float(clean_df["Close"].iloc[0])
            end_p = float(clean_df["Close"].iloc[-1])
            shares_bought = invested / start_p
            final_val = shares_bought * end_p
            profit = final_val - invested
            profit_pct = (profit / invested) * 100.0

            ic1, ic2, ic3, ic4 = st.columns(4)
            with ic1:
                render_kpi_card("Money Put In", f"{curr}{invested:,.0f}", f"On {clean_report['start_date']}", "neutral", "database")
            with ic2:
                render_kpi_card("Value Today", f"{curr}{final_val:,.2f}", f"On {clean_report['end_date']}", "pos" if profit >= 0 else "neg", "stock")
            with ic3:
                render_kpi_card("Total Profit / Loss", f"{curr}{profit:+,.2f}", f"{profit_pct:+.2f}% Return", "pos" if profit >= 0 else "neg", "target")
            with ic4:
                render_kpi_card("Shares Owned", f"{shares_bought:.2f}", f"Bought at {curr}{start_p:,.2f}", "neutral", "award")

            st.plotly_chart(plot_investment_growth_chart(clean_df, active_ticker, float(invested), curr, theme=theme_mode), width="stretch")

    # =========================================================================
    # 3. PRICE FORECAST
    # =========================================================================
    elif selected_nav == "Price Forecast":
        render_section_header("AI Price Forecast", f"Next-day and multi-day statistical projections for {active_ticker}.", "target", app_mode, eyebrow=selected_nav)
        render_simple_guide(
            "How this forecast works",
            f"Today's technical indicators feed into your chosen model to estimate <strong>{active_ticker}'s next closing price</strong>. The shaded green region marks the 95% confidence band.",
            "Estimates are sharpest on Day 1 and naturally widen over longer horizons.",
        )

        col_m1, col_m2, col_m3 = st.columns([2, 1, 1])
        with col_m1:
            chosen_pred_model = st.selectbox("Select Prediction Model", PREDICTION_MODELS, index=0)
        with col_m2:
            horizon_days = st.slider("Forecast Horizon (Days)", 3, 10, 5, 1)
        with col_m3:
            use_mrmr = st.checkbox("Use Top 8 mRMR Features", value=False)

        pred_out = cached_prediction(active_ticker, period_choice, train_ratio, chosen_pred_model, use_mrmr, horizon_days)

        pk1, pk2, pk3, pk4 = st.columns(4)
        with pk1:
            render_kpi_card("Latest Close", f"{curr}{pred_out['latest_close']:,.2f}", pred_out["latest_date"], "neutral", "stock")
        with pk2:
            render_kpi_card("Next-Day Estimate", f"{curr}{pred_out['next_day_predicted_close']:,.2f}", f"{pred_out['expected_change']:+.2f} ({pred_out['expected_change_pct']:+.2f}%)", "pos" if pred_out["expected_change"] >= 0 else "neg", "target")
        with pk3:
            render_kpi_card("Signal", "Bullish" if pred_out["expected_change"] >= 0 else "Bearish", pred_out["signal"], "pos" if pred_out["expected_change"] >= 0 else "neg", "regression")
        with pk4:
            render_kpi_card("Test R² & MAE", f"{pred_out['test_metrics']['R2']:.4f}", f"Avg error: ±{curr}{pred_out['test_metrics']['MAE']:.2f}", "neutral", "award")

        st.plotly_chart(plot_future_forecast_chart(clean_df, pred_out["future_projection_df"], active_ticker, chosen_pred_model, theme=theme_mode), width="stretch")

        t_tbl, t_hist = st.tabs(["Multi-Day Estimate Table", "Historical Backtest"])
        with t_tbl:
            st.dataframe(pred_out["future_projection_df"], width="stretch")
        with t_hist:
            st.plotly_chart(plot_regression_predictions(pred_out["test_comparison_df"]["Date"], pred_out["test_comparison_df"]["Actual_Next_Close"], pred_out["test_comparison_df"]["Predicted_Next_Close"], chosen_pred_model, active_ticker, theme=theme_mode), width="stretch")
            st.dataframe(pred_out["test_comparison_df"].tail(15), width="stretch")

    # =========================================================================
    # 4. AI ANALYTICS STUDIO (ALL 5 ML MODULES)
    # =========================================================================
    elif selected_nav == "AI Analytics Studio":
        render_section_header("AI Analytics Studio", "Interactive modeling suite covering Regression, Dimensionality Reduction, Clustering, RL, and Neural Networks.", "brain", app_mode, eyebrow=selected_nav)
        st.radio("Select Module", LAB_NAMES, key="active_lab", horizontal=True, label_visibility="collapsed")
        selected_lab = st.session_state.active_lab

        if selected_lab == LAB_NAMES[0]:
            render_simple_guide(
                "Price Models (Regression) vs Direction (Logistic Classification)",
                "Regression predicts an exact price number. Logistic Regression predicts only whether tomorrow's price will go <strong>Up (1)</strong> or <strong>Down (0)</strong>.",
                "Regression = guessing an exact score (88/100). Classification = guessing Pass or Fail.",
            )
            with st.expander("Model Hyperparameters (Ridge, Lasso & Polynomial)", expanded=False):
                cp1, cp2, cp3 = st.columns(3)
                with cp1:
                    ridge_a = st.slider("Ridge L2 Penalty (alpha)", 0.01, 20.0, 1.0, 0.25)
                with cp2:
                    lasso_a = st.slider("Lasso L1 Penalty (alpha)", 0.001, 2.0, 0.05, 0.01)
                with cp3:
                    poly_d = st.radio("Polynomial Degree", [2, 3], horizontal=True)

            reg_res, mv_df, log_res = cached_regression(active_ticker, period_choice, train_ratio, ridge_a, lasso_a, poly_d)
            metrics_df = reg_res["metrics_df"]

            t1, t2, t3, t4 = st.tabs(["Predictions & Scores", "OLS vs Gradient Descent", "Direction & Multi-Target", "How It Works"])
            with t1:
                chosen_m = st.selectbox("Select Model to Plot", list(reg_res["predictions"].keys()), index=5)
                st.plotly_chart(plot_regression_predictions(dates_test, y_test.values, reg_res["predictions"][chosen_m]["test_pred"], chosen_m, active_ticker, theme=theme_mode), width="stretch")
                st.plotly_chart(plot_regression_metrics_comparison(metrics_df, theme=theme_mode), width="stretch")
                st.dataframe(metrics_df[["Model", "Train_RMSE", "Test_RMSE", "Train_R2", "Test_R2", "Bias_Variance_Diagnosis"]], width="stretch")
            with t2:
                co, cc = st.columns(2)
                with co:
                    og = reg_res["ols_gd_demo"]
                    st.markdown(f"**OLS Closed-Form:** Slope `{og['OLS']['Slope']}`, Intercept `{og['OLS']['Intercept']}`  \n**Gradient Descent:** Slope `{og['Gradient_Descent']['Slope']}`, Intercept `{og['Gradient_Descent']['Intercept']}`")
                    st.plotly_chart(plot_gradient_descent_curve(og["loss_history"], theme=theme_mode), width="stretch")
                with cc:
                    st.dataframe(reg_res["coef_df"], width="stretch", height=340)
            with t3:
                lc1, lc2, lc3, lc4 = st.columns(4)
                with lc1:
                    render_kpi_card("Logistic Train Acc", f"{log_res['train_accuracy']:.1f}%", "Binary Up/Down", "neutral", "regression")
                with lc2:
                    render_kpi_card("Logistic Test Acc", f"{log_res['test_accuracy']:.1f}%", "Unseen Days", "pos", "target")
                with lc3:
                    render_kpi_card("Precision", f"{log_res['precision']:.1f}%", "Upward Calls", "neutral", "award")
                with lc4:
                    render_kpi_card("F1-Score", f"{log_res['f1_score']:.1f}%", "Balance", "neutral", "award")
                st.markdown("#### Multivariate Regression (Next Close, High & Low Simultaneously)")
                st.dataframe(mv_df, width="stretch")
            with t4:
                for k in ["Simple Linear Regression", "Multiple Linear Regression", "Polynomial Regression", "Ridge Regression", "Lasso Regression", "Decision Tree & Model Tree Regression", "Logistic Regression"]:
                    render_algorithm_explainer(k, app_mode)
                render_code_explainer("regression")

        elif selected_lab == LAB_NAMES[1]:
            render_simple_guide(
                "Feature Lab: Keeping the Smartest Clues (Selection, PCA & LDA)",
                "Filters out duplicate indicators and compresses 19 features into clean 2D maps — fitted strictly on training data to prevent leakage.",
                "Packing light for a trip: keep only the best unique items and leave duplicates behind.",
            )
            with st.expander("Filter & Feature Count Controls", expanded=False):
                cf1, cf2, cf3 = st.columns(3)
                with cf1:
                    var_t = st.slider("Low Variance Cutoff", 0.005, 0.050, 0.015, 0.005)
                with cf2:
                    corr_t = st.slider("High Correlation Cutoff", 0.70, 0.98, 0.85, 0.02)
                with cf3:
                    n_sel = st.slider("Features to Select (Wrapper & mRMR)", 4, 12, 6, 1)

            lv, hc, fb, mrmr, proj, ba = cached_dim_reduction(active_ticker, period_choice, train_ratio, var_t, corr_t, n_sel)
            t1, t2, t3, t4 = st.tabs(["Selected vs Removed", "2D Projections (PCA, LDA, ICA, MDS, Isomap)", "Before vs After", "How It Works"])
            with t1:
                st.dataframe(pd.DataFrame([
                    {"Method": "Low Variance Filter", "Kept": len(lv["selected"]), "Selected": ", ".join(lv["selected"]), "Removed": ", ".join(lv["removed"]) or "None"},
                    {"Method": "High Correlation Filter", "Kept": len(hc["selected"]), "Selected": ", ".join(hc["selected"]), "Removed": ", ".join(hc["removed"]) or "None"},
                    {"Method": "Forward Selection (SFS)", "Kept": len(fb["forward_selected"]), "Selected": ", ".join(fb["forward_selected"]), "Removed": ", ".join(fb["forward_removed"])},
                    {"Method": "Backward Elimination (SBS)", "Kept": len(fb["backward_selected"]), "Selected": ", ".join(fb["backward_selected"]), "Removed": ", ".join(fb["backward_removed"])},
                    {"Method": "mRMR", "Kept": len(mrmr["selected"]), "Selected": ", ".join(mrmr["selected"]), "Removed": ", ".join(mrmr["removed"])},
                ]), width="stretch")
                st.plotly_chart(plot_correlation_heatmap(hc["corr_matrix"], theme=theme_mode), width="stretch")
            with t2:
                st.plotly_chart(plot_pca_scree(proj["pca_variance_df"], theme=theme_mode), width="stretch")
                m_pick = st.selectbox("Select 2D Projection", ["PCA (Unsupervised)", "LDA (Supervised)", "ICA", "Factor Analysis", "MDS", "Isomap"])
                m_map = {"PCA (Unsupervised)": proj["pca_2d_df"], "LDA (Supervised)": proj["lda_2d_df"], "ICA": proj["ica_2d_df"], "Factor Analysis": proj["fa_2d_df"], "MDS": proj["mds_2d_df"], "Isomap": proj["isomap_2d_df"]}
                st.plotly_chart(plot_2d_projection(m_map[m_pick], m_pick, theme=theme_mode), width="stretch")
            with t3:
                st.dataframe(ba, width="stretch")
                st.dataframe(proj["fa_loadings_df"], width="stretch")
            with t4:
                render_algorithm_explainer("Feature Selection (Filters, Wrappers & mRMR)", app_mode)
                render_algorithm_explainer("PCA vs LDA vs ICA & Manifolds", app_mode)
                render_code_explainer("dim_reduction")

        elif selected_lab == LAB_NAMES[2]:
            render_simple_guide(
                "Market Clusters: Grouping 12 Stocks by Risk & Return",
                "Clustering groups similar equities (e.g., Calm Blue-Chips vs High-Growth Movers) to reveal market structure — it does not directly predict tomorrow's price.",
                "Like grouping athletes into sprinters and marathon runners based on their stats.",
            )
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                k_choice = st.slider("Number of Groups (k)", 2, 6, 3, 1)
            with cc2:
                viz_algo = st.selectbox("Clustering Algorithm", ["KMeans_Cluster", "KMedoids_Cluster", "Agglomerative_Cluster", "Divisive_Cluster", "BIRCH_Cluster", "Chameleon_Cluster"])
            with cc3:
                dist_choice = st.selectbox("Distance Metric", ["euclidean", "manhattan", "cosine"])

            clust = cached_clustering(k_choice, dist_choice)
            kc1, kc2, kc3 = st.columns(3)
            with kc1:
                render_kpi_card("Hopkins Tendency", f"{clust['hopkins_statistic']:.2f}", "> 0.65 = Clear Clusters", "pos", "cluster")
            with kc2:
                render_kpi_card("K-Medoids Centers", ", ".join(clust["medoid_tickers"]), "Representative Stocks", "neutral", "stock")
            with kc3:
                bk = clust["k_eval_df"].sort_values("Silhouette_Score", ascending=False).iloc[0]
                render_kpi_card("Optimal k (Silhouette)", f"k = {int(bk['k'])}", f"Score: {bk['Silhouette_Score']:.2f}", "pos", "award")

            t1, t2, t3, t4 = st.tabs(["Risk-Return Map", "Elbow & Quality Metrics", "Dendrogram & DIANA", "How It Works"])
            with t1:
                st.plotly_chart(plot_cluster_scatter(clust["clustered_stocks_df"], viz_algo, f"Market Behavior Groups ({viz_algo}, k={k_choice})", theme=theme_mode), width="stretch")
                st.dataframe(clust["clustered_stocks_df"][["Ticker", "Company", "Annual_Return_Pct", "Annual_Volatility_Pct", "Market_Behavior_Group"]], width="stretch")
            with t2:
                st.plotly_chart(plot_elbow_and_silhouette(clust["k_eval_df"], theme=theme_mode), width="stretch")
                st.dataframe(clust["algo_comparison_df"], width="stretch")
            with t3:
                st.pyplot(create_matplotlib_dendrogram(clust["linkage_matrix"], list(clust["clustered_stocks_df"]["Ticker"].values), theme=theme_mode))
                st.dataframe(clust["diana_splits_df"], width="stretch")
            with t4:
                render_algorithm_explainer("Clustering (K-Means, K-Medoids, Hierarchical, BIRCH, Chameleon)", app_mode)
                render_code_explainer("clustering")

        elif selected_lab == LAB_NAMES[3]:
            render_simple_guide(
                "Strategy Simulator: Virtual-Capital Reinforcement Learning",
                f"A simulated agent starts with <strong>{curr}100,000 virtual capital</strong> (no real money) and learns when to <strong>Buy</strong>, <strong>Sell</strong>, or <strong>Hold</strong>.",
                "Like learning a video game: good moves earn points, bad moves lose points, and the bot improves over rounds.",
            )
            rc1, rc2 = st.columns([2, 1])
            with rc1:
                rl_algo = st.selectbox("Learning Paradigm", ["Model-Based RL (Dyna-Q)", "Model-Free TD (Q-Learning)"])
            with rc2:
                rl_ep = st.slider("Training Episodes", 5, 35, 15, 5)
            with st.expander("Agent Hyperparameters (Alpha & Gamma)", expanded=False):
                ra1, ra2 = st.columns(2)
                with ra1:
                    rl_a = st.slider("Learning Rate (alpha)", 0.05, 0.50, 0.15, 0.05)
                with ra2:
                    rl_g = st.slider("Discount Factor (gamma)", 0.70, 0.99, 0.92, 0.02)

            rl_res, bandit_res = cached_rl(active_ticker, period_choice, rl_algo, rl_ep, rl_a, rl_g)
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                render_kpi_card("Simulated Portfolio", f"{curr}{rl_res['final_portfolio_value']:,.2f}", f"Start: {curr}100,000", "pos" if rl_res["final_return_pct"] >= 0 else "neg", "bot")
            with k2:
                render_kpi_card("Simulated Return", f"{rl_res['final_return_pct']:+.2f}%", f"Buy & Hold: {rl_res['buy_hold_return_pct']:+.2f}%", "pos" if rl_res["final_return_pct"] >= rl_res["buy_hold_return_pct"] else "neutral", "stock")
            with k3:
                render_kpi_card("Cumulative Reward", f"{rl_res['cumulative_reward']:+.1f}", f"{len(modeling_df)} trading steps", "pos" if rl_res["cumulative_reward"] >= 0 else "neg", "award")
            with k4:
                ac = rl_res["action_counts"]
                render_kpi_card("Actions Taken", f"B:{ac['Buy']} | S:{ac['Sell']} | H:{ac['Hold']}", "Buy / Sell / Hold", "neutral", "sliders")

            t1, t2, t3 = st.tabs(["Portfolio Curve", "Policy Table & K-Armed Bandit", "How It Works"])
            with t1:
                st.plotly_chart(plot_rl_portfolio_performance(rl_res["trajectory_df"], rl_algo, theme=theme_mode), width="stretch")
            with t2:
                st.markdown("#### K-Armed Bandit (Single-State 4-Strategy Test)")
                st.dataframe(bandit_res["bandit_summary_df"], width="stretch")
                st.markdown("#### Learned State-Action Policy (Q-Table)")
                st.dataframe(rl_res["policy_df"], width="stretch", height=300)
            with t3:
                st.markdown(get_rl_svg_diagram(), unsafe_allow_html=True)
                render_algorithm_explainer("Reinforcement Learning (Bandit, TD Q-Learning & Model-Based Dyna-Q)", app_mode)

        elif selected_lab == LAB_NAMES[4]:
            render_simple_guide(
                "Neural Net (MLP): Layered Pattern Classifier",
                "Connected layers of artificial neurons analyze today's indicators and vote on whether tomorrow's closing price will go <strong>Up (1)</strong> or <strong>Down (0)</strong>.",
                "Like a team of voters weighing sunshine, clouds, and wind before raising their hands.",
            )
            t1, t2, t3, t4 = st.tabs(["MLP Classifier", "Single Neuron Calculator", "Biological vs Artificial NN", "How It Works"])
            with t1:
                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    arch_opts = {"Small (16, 8)": (16, 8), "Medium (32, 16)": (32, 16), "Large (64, 32)": (64, 32)}
                    arch_lbl = st.selectbox("Network Size", list(arch_opts.keys()), index=1)
                    h_tuple = arch_opts[arch_lbl]
                with mc2:
                    act_c = st.selectbox("Activation Function", ["relu", "logistic", "tanh"], index=0)
                with mc3:
                    ep_c = st.slider("Training Epochs", 30, 150, 80, 10)

                nn_res = cached_mlp(active_ticker, period_choice, train_ratio, h_tuple, act_c, ep_c)
                mk1, mk2, mk3, mk4 = st.columns(4)
                with mk1:
                    render_kpi_card("Single-Layer (SLP)", f"{nn_res['slp_test_accuracy']:.1f}%", "Linear threshold", "neutral", "brain")
                with mk2:
                    render_kpi_card("MLP Train Acc", f"{nn_res['mlp_train_accuracy']:.1f}%", arch_lbl, "neutral", "layers")
                with mk3:
                    render_kpi_card("MLP Test Acc", f"{nn_res['mlp_test_accuracy']:.1f}%", act_c.upper(), "pos", "target")
                with mk4:
                    render_kpi_card("F1-Score", f"{nn_res['mlp_f1']:.1f}%", f"Precision: {nn_res['mlp_precision']:.1f}%", "pos", "award")

                cl, cm = st.columns([1.3, 1])
                with cl:
                    st.plotly_chart(plot_mlp_loss_curves(nn_res["loss_history_df"], theme=theme_mode), width="stretch")
                with cm:
                    st.plotly_chart(plot_confusion_matrix_chart(nn_res["confusion_matrix"], "Confusion Matrix", theme=theme_mode), width="stretch")
            with t2:
                nc1, nc2, nc3, nc4 = st.columns(4)
                with nc1:
                    x1 = st.number_input("Input x1", value=0.85, step=0.1)
                    w1 = st.number_input("Weight w1", value=1.20, step=0.1)
                with nc2:
                    x2 = st.number_input("Input x2", value=-0.40, step=0.1)
                    w2 = st.number_input("Weight w2", value=-0.75, step=0.1)
                with nc3:
                    x3 = st.number_input("Input x3", value=0.60, step=0.1)
                    w3 = st.number_input("Weight w3", value=0.90, step=0.1)
                with nc4:
                    b_val = st.number_input("Bias (b)", value=-0.15, step=0.05)

                n_out = calculate_single_neuron_output([x1, x2, x3], [w1, w2, w3], b_val)
                nk1, nk2, nk3, nk4 = st.columns(4)
                with nk1:
                    render_kpi_card("Net Input (z)", f"{n_out['net_input_z']:.3f}", "sum(x*w) + b", "neutral", "brain")
                with nk2:
                    render_kpi_card("Binary Sigmoid", f"{n_out['binary_sigmoid_output']:.3f}", "Range (0, 1)", "pos", "target")
                with nk3:
                    render_kpi_card("Bipolar Sigmoid", f"{n_out['bipolar_sigmoid_output']:.3f}", "Range (-1, +1)", "neutral", "sliders")
                with nk4:
                    render_kpi_card("ReLU Output", f"{n_out['relu_output']:.3f}", n_out["direction_interpretation"], "pos" if n_out["step_perceptron_output"] == 1 else "neg", "award")
                st.plotly_chart(plot_activation_functions(n_out["curves_df"], n_out["net_input_z"], theme=theme_mode), width="stretch")
            with t3:
                st.markdown(get_ann_svg_diagram("(32, 16)", "RELU"), unsafe_allow_html=True)
                st.dataframe(BIOLOGICAL_VS_ARTIFICIAL_TABLE, width="stretch")
            with t4:
                render_algorithm_explainer("Neural Networks (Single Neuron, Perceptron & MLP Backpropagation)", app_mode)
                render_code_explainer("neural_network")

    # =========================================================================
    # 5. CHARTS & DATA
    # =========================================================================
    elif selected_nav == "Charts & Data":
        render_section_header("Charts, Indicators & Raw Data", f"Full technical analysis, clean CSV export, and data pipeline for {active_ticker}.", "database", app_mode, eyebrow=selected_nav)
        t_ind, t_raw, t_pipe = st.tabs(["Technical Charts", "Raw Data & CSV", "Data Pipeline"])
        with t_ind:
            st.plotly_chart(plot_candlestick_with_ma(full_feat_df, active_ticker, theme=theme_mode, chart_style="Candlestick"), width="stretch")
            ca, cb = st.columns(2)
            with ca:
                st.plotly_chart(plot_macd_chart(full_feat_df, active_ticker, theme=theme_mode), width="stretch")
            with cb:
                st.plotly_chart(plot_rsi_chart(full_feat_df, active_ticker, theme=theme_mode), width="stretch")
            st.plotly_chart(plot_volume_chart(full_feat_df, active_ticker, theme=theme_mode), width="stretch")
            with st.expander("Indicator Explanations", expanded=False):
                for ind_name, ind_desc in INDICATOR_EXPLANATIONS.items():
                    st.markdown(f"- **{ind_name}:** {ind_desc}")
        with t_raw:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                render_kpi_card("Trading Days", f"{clean_report['final_rows']}", f"Window: {period_choice.upper()}", "neutral", "database")
            with c2:
                render_kpi_card("First Date", clean_report["start_date"], "Start", "neutral", "stock")
            with c3:
                render_kpi_card("Latest Date", clean_report["end_date"], "End", "neutral", "stock")
            with c4:
                render_kpi_card("Missing Fixed", f"{clean_report['missing_values_before']} → {clean_report['missing_values_after']}", "Forward-fill", "pos", "sliders")
            st.dataframe(clean_df.tail(100), width="stretch")
            st.download_button(f"Download {active_ticker} CSV", clean_df.to_csv(index=False).encode("utf-8"), f"{active_ticker}_cleaned.csv", "text/csv")
        with t_pipe:
            st.markdown(get_pipeline_svg_diagram(), unsafe_allow_html=True)
            dup_c = int(raw_df.duplicated(subset=["Date"]).sum()) if "Date" in raw_df.columns else 0
            st.info(f"Live Audit ({active_ticker}): Raw {raw_df.shape} → {dup_c} duplicates & {clean_report['missing_values_before']} missing cleaned → {len(FEATURE_COLUMNS)} features → {len(X_train)} train days & {len(X_test)} test days.")
            render_code_explainer("preprocessing")

    # =========================================================================
    # 6. LEARN & GLOSSARY
    # =========================================================================
    elif selected_nav == "Learn & Glossary":
        render_section_header("Learning Center & Glossary", "Step-by-step guided tour, searchable financial/ML dictionary, and evaluation guide.", "book", app_mode, eyebrow=selected_nav)
        t_tour, t_dict, t_eval = st.tabs(["13-Step Guided Tour", "Word Dictionary", "Evaluation Metrics Guide"])
        with t_tour:
            step_idx = st.slider("Select Step (1 to 13)", 1, 13, 1, 1) - 1
            title, expl, why_imp, impl, tip = GUIDED_TOUR_STEPS[step_idx]
            st.markdown(
                f"""
                <div class="fin-card step-card rim">
                    <span class="ticker-badge">Step {step_idx + 1} of 13</span>
                    <h3 style="margin-top:8px;">{title}</h3>
                    <p><strong>Simple Explanation:</strong> {expl}</p>
                    <p style="color:var(--text-secondary);"><strong>Why it matters:</strong> {why_imp}</p>
                    <p style="color:var(--accent);font-weight:600;">Try it: {tip}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress((step_idx + 1) / 13.0)
        with t_dict:
            cs1, cs2 = st.columns([2, 1])
            with cs1:
                sq = st.text_input("Search any term (e.g. RSI, Ridge, PCA, Sigmoid)", value="").strip().lower()
            with cs2:
                cf = st.selectbox("Category", ["All"] + sorted(list({g["Category"] for g in GLOSSARY_ITEMS})))
            items = [g for g in GLOSSARY_ITEMS if (cf == "All" or g["Category"] == cf) and (sq == "" or sq in g["Term"].lower() or sq in g["Simple"].lower())]
            for item in items:
                with st.expander(f"{item['Term']} ({item['Category']})", expanded=bool(sq)):
                    st.markdown(f"- **Simple Meaning:** {item['Simple']}\n- **Technical Meaning:** {item['Technical']}\n- **Used in Platform:** {item['Project_Use']}")
        with t_eval:
            reg_d, _, _ = cached_regression(active_ticker, period_choice, train_ratio, 1.0, 0.05, 2)
            rr = reg_d["metrics_df"][reg_d["metrics_df"]["Model"].str.contains("Ridge")].iloc[0]
            ov = reg_d["metrics_df"][reg_d["metrics_df"]["Model"].str.contains("Unconstrained")].iloc[0]
            c1, c2 = st.columns(2)
            with c1:
                render_simple_guide("Grading Price Predictions (MAE, RMSE, R²)", f"On {active_ticker}, Ridge's average error (<strong>MAE</strong>) is <strong>{curr}{rr['Test_MAE']}</strong> and <strong>R² = {rr['Test_R2']}</strong> (closer to 1.0 is better).", "MAE measures how many rupees or dollars off the estimate was on an average day.")
            with c2:
                render_simple_guide("Understanding Overfitting", f"An unconstrained Decision Tree got <strong>Train R² = {ov['Train_R2']}</strong> (memorized noise) but dropped to <strong>Test R² = {ov['Test_R2']}</strong> on unseen days.", "Like memorizing a practice quiz instead of learning the concepts.")

    # =========================================================================
    # 7. MODEL BENCHMARKS
    # =========================================================================
    elif selected_nav == "Model Benchmarks":
        render_section_header("Model Benchmarks & Architecture", "Side-by-side performance comparison across all predictive models and analytics modules.", "award", app_mode, eyebrow=selected_nav)
        t_sc, t_arch = st.tabs(["Model Leaderboard", "Platform Architecture"])
        with t_sc:
            reg_all, _, log_res = cached_regression(active_ticker, period_choice, train_ratio, 1.0, 0.05, 2)
            nn_res = cached_mlp(active_ticker, period_choice, train_ratio, (32, 16), "relu", 80)
            st.plotly_chart(plot_regression_metrics_comparison(reg_all["metrics_df"], theme=theme_mode), width="stretch")
            st.dataframe(reg_all["metrics_df"][["Model", "Train_RMSE", "Test_RMSE", "Train_R2", "Test_R2", "Bias_Variance_Diagnosis"]], width="stretch")
            st.subheader("Directional Classifiers (Up vs Down)")
            st.dataframe(pd.DataFrame([
                {"Classifier": "Logistic Regression", "Train_Acc_%": log_res["train_accuracy"], "Test_Acc_%": log_res["test_accuracy"], "F1_%": log_res["f1_score"]},
                {"Classifier": "Single-Layer Perceptron", "Train_Acc_%": nn_res["slp_train_accuracy"], "Test_Acc_%": nn_res["slp_test_accuracy"], "F1_%": None},
                {"Classifier": "Multilayer Perceptron (MLP)", "Train_Acc_%": nn_res["mlp_train_accuracy"], "Test_Acc_%": nn_res["mlp_test_accuracy"], "F1_%": nn_res["mlp_f1"]},
            ]), width="stretch")
        with t_arch:
            st.markdown(
                """
                <div class="fin-card">
                    <h4 style="margin-top:0;color:var(--accent-bright) !important;">STOCKMIND AI Platform Overview</h4>
                    <p>STOCKMIND AI is an end-to-end quantitative stock analytics platform combining Yahoo Finance market ingestion, leakage-free chronological time-series preprocessing, 19 technical indicators, and five modular machine learning engines.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(pd.DataFrame([
                {"Module": "Price Models (Regression)", "Engine": "src/regression.py", "Capabilities": "OLS, Gradient Descent, Simple/Multiple/Multivariate/Polynomial, Ridge, Lasso, Decision Trees, Model Tree, Logistic"},
                {"Module": "Feature Lab (Dim. Reduction)", "Engine": "src/dimensionality_reduction.py", "Capabilities": "Low Variance, High Correlation, SFS, SBS, mRMR, Factor Analysis, PCA, LDA, ICA, MDS, Isomap"},
                {"Module": "Market Clusters", "Engine": "src/clustering.py", "Capabilities": "Hopkins Tendency, K-Means, K-Medoids (PAM), Agglomerative, DIANA, BIRCH, Chameleon, Elbow, Silhouette"},
                {"Module": "Strategy Simulator (RL)", "Engine": "src/reinforcement_learning.py", "Capabilities": "K-Armed Bandit, 18-State Market Environment, Model-Free Q-Learning, Model-Based Dyna-Q"},
                {"Module": "Neural Net (MLP)", "Engine": "src/neural_network.py", "Capabilities": "Single Neuron Sigmoid/ReLU Calculator, Single-Layer Perceptron, MLP Backpropagation Classifier"},
            ]), width="stretch")

    # Clean SaaS Footer (with Deploy to Cloud moved neatly to bottom)
    render_app_footer()


if __name__ == "__main__":
    main()
