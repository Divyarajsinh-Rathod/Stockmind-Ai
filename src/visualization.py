"""
Theme-Aware Financial Visualization Module (src/visualization.py)
Generates crisp, TradingView/Groww-inspired Plotly charts and Matplotlib dendrograms
that automatically adapt to Light, Dark, and System theme modes.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram


GREEN_BULL = "#2EE88A"      # neon bull green (logo)
GREEN_BRIGHT = "#6CFFB0"
RED_BEAR = "#FF4D5E"        # candle red (logo)
BLUE_ACCENT = "#1FB5E0"     # teal rim (logo)
CYAN_ACCENT = "#1FB5E0"
AMBER_ACCENT = "#F7B955"
PURPLE_ACCENT = "#A78BFA"
PINK_ACCENT = "#F472B6"
FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"
MONO_FAMILY = "JetBrains Mono, ui-monospace, monospace"

# Neon colour scales for heatmaps
NEON_DIVERGING = [[0.0, "#FF4D5E"], [0.25, "#7A2333"], [0.5, "#0C151D"], [0.75, "#12704A"], [1.0, "#2EE88A"]]
NEON_SEQUENTIAL = [[0.0, "#0C151D"], [0.35, "#0E4D4F"], [0.7, "#17A57A"], [1.0, "#6CFFB0"]]


def _rgba(hex_c: str, a: float) -> str:
    h = hex_c.lstrip("#")
    return f"rgba({int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}, {a})"


def _get_theme_tokens(theme: str = "Dark") -> dict:
    """Single neon-dark token set (theme arg kept for backwards compatibility)."""
    return {
        "paper_bg": "rgba(0,0,0,0)",
        "plot_bg": "rgba(0,0,0,0)",
        "grid": "rgba(120, 200, 170, 0.08)",
        "axis": "rgba(120, 200, 170, 0.18)",
        "text": "#DCEFE5",
        "subtext": "#7A948A",
        "legend_bg": "rgba(0,0,0,0)",
        "line_primary": BLUE_ACCENT,
        "mpl_bg": "#0C151D",
        "hover_bg": "#0A131B",
    }


def _apply_theme(fig: go.Figure, title: str = "", height: int = 420, theme: str = "Dark") -> go.Figure:
    tk = _get_theme_tokens(theme)
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>" if title else "", font=dict(size=14, color=tk["text"], family="Space Grotesk, " + FONT_FAMILY), x=0.01, y=0.97),
        paper_bgcolor=tk["paper_bg"],
        plot_bgcolor=tk["plot_bg"],
        font=dict(color=tk["subtext"], family=FONT_FAMILY, size=12),
        height=height,
        hovermode="x unified",
        hoverlabel=dict(bgcolor=tk["hover_bg"], bordercolor=GREEN_BULL, font=dict(color=tk["text"], family=MONO_FAMILY, size=12)),
        margin=dict(l=44, r=18, t=58, b=36),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0, bgcolor=tk["legend_bg"], font=dict(size=11, color=tk["text"])),
        colorway=[GREEN_BULL, BLUE_ACCENT, AMBER_ACCENT, PURPLE_ACCENT, RED_BEAR, PINK_ACCENT],
    )
    fig.update_xaxes(showgrid=True, gridcolor=tk["grid"], zeroline=False, linecolor=tk["axis"], tickfont=dict(color=tk["subtext"]), title_font=dict(color=tk["subtext"]))
    fig.update_yaxes(showgrid=True, gridcolor=tk["grid"], zeroline=False, linecolor=tk["axis"], tickfont=dict(color=tk["subtext"], family=MONO_FAMILY), title_font=dict(color=tk["subtext"]))
    fig.update_annotations(font=dict(color=tk["text"], size=11))
    return fig


def _glow_line(fig: go.Figure, x, y, color: str, name: str, width: float = 2.4, fill: bool = False, **kw):
    """Draws a soft neon glow underlay + crisp line (optionally with gradient-like fill)."""
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=_rgba(color, 0.18), width=width * 4), hoverinfo="skip", showlegend=False, **kw))
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", name=name, line=dict(color=color, width=width),
        fill="tozeroy" if fill else None, fillcolor=_rgba(color, 0.10) if fill else None, **kw,
    ))


def plot_candlestick_with_ma(df: pd.DataFrame, ticker: str, theme: str = "Dark", chart_style: str = "Area") -> go.Figure:
    """Plots Groww/Robinhood-style clean Area line or TradingView Candlestick overlaid with SMA_20 & SMA_50."""
    tk = _get_theme_tokens(theme)
    fig = go.Figure()

    if chart_style == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=df["Date"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="OHLC",
                increasing=dict(line=dict(color=GREEN_BULL, width=1.2), fillcolor=GREEN_BULL),
                decreasing=dict(line=dict(color=RED_BEAR, width=1.2), fillcolor=RED_BEAR),
            )
        )
    else:
        is_up = float(df["Close"].iloc[-1]) >= float(df["Close"].iloc[0])
        line_c = GREEN_BULL if is_up else RED_BEAR
        _glow_line(fig, df["Date"], df["Close"], line_c, "Close Price", width=2.4, fill=True)
        lo, hi = float(df["Low"].min()), float(df["High"].max())
        pad = (hi - lo) * 0.06
        fig.update_yaxes(range=[lo - pad, hi + pad])

    if "SMA_20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["SMA_20"],
                mode="lines",
                name="20-Day Avg (SMA)",
                line=dict(color=tk["line_primary"], width=1.5),
            )
        )
    if "SMA_50" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["SMA_50"],
                mode="lines",
                name="50-Day Avg (SMA)",
                line=dict(color=AMBER_ACCENT, width=1.6, dash="dot"),
            )
        )
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig = _apply_theme(fig, f"{ticker} · Price Action", height=450, theme=theme)
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[dict(count=1, label="1M", step="month", stepmode="backward"),
                     dict(count=3, label="3M", step="month", stepmode="backward"),
                     dict(count=6, label="6M", step="month", stepmode="backward"),
                     dict(step="all", label="ALL")],
            bgcolor="rgba(46,232,138,0.08)", activecolor="rgba(46,232,138,0.30)",
            bordercolor="rgba(46,232,138,0.25)", borderwidth=1, font=dict(color="#DCEFE5", size=11),
            x=0, y=1.0, yanchor="bottom",
        ),
        showspikes=True, spikecolor="rgba(46,232,138,0.5)", spikethickness=1, spikedash="dot", spikemode="across",
    )
    fig.update_layout(legend=dict(y=1.08), margin=dict(t=72))
    return fig


def plot_rsi_chart(df: pd.DataFrame, ticker: str, theme: str = "Dark") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["RSI_14"],
            mode="lines",
            name="RSI (14)",
            line=dict(color=PURPLE_ACCENT, width=2.0),
        )
    )
    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(31,181,224,0.06)", line_width=0)
    fig.add_hline(y=70, line_dash="dash", line_color=RED_BEAR, annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color=GREEN_BULL, annotation_text="Oversold (30)")
    fig.update_yaxes(range=[0, 100])
    return _apply_theme(fig, f"{ticker} — Momentum Mood (RSI-14)", height=310, theme=theme)


def plot_macd_chart(df: pd.DataFrame, ticker: str, theme: str = "Dark") -> go.Figure:
    tk = _get_theme_tokens(theme)
    fig = go.Figure()
    hist_colors = [GREEN_BULL if v >= 0 else RED_BEAR for v in df["MACD_Hist"].fillna(0)]
    fig.add_trace(go.Bar(x=df["Date"], y=df["MACD_Hist"], name="Histogram", marker_color=hist_colors, opacity=0.6))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], mode="lines", name="MACD", line=dict(color=tk["line_primary"], width=1.8)))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD_Signal"], mode="lines", name="Signal (9-EMA)", line=dict(color=AMBER_ACCENT, width=1.8)))
    return _apply_theme(fig, f"{ticker} — Trend Speed (MACD)", height=310, theme=theme)


def plot_volume_chart(df: pd.DataFrame, ticker: str, theme: str = "Dark") -> go.Figure:
    fig = go.Figure()
    colors = [GREEN_BULL if c >= o else RED_BEAR for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df["Date"], y=df["Volume"], name="Volume", marker_color=colors, opacity=0.75))
    if "Volume_SMA_20" in df.columns:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Volume_SMA_20"], mode="lines", name="20-Day Avg", line=dict(color=AMBER_ACCENT, width=1.8)))
    return _apply_theme(fig, f"{ticker} — Shares Traded (Volume)", height=310, theme=theme)


def plot_regression_predictions(dates_test, y_test, y_pred, model_name: str, ticker: str, theme: str = "Dark") -> go.Figure:
    tk = _get_theme_tokens(theme)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates_test, y=y_test, mode="lines", name="Actual Price", line=dict(color=tk["line_primary"], width=2.2)))
    fig.add_trace(go.Scatter(x=dates_test, y=y_pred, mode="lines", name=f"AI Estimate ({model_name})", line=dict(color=GREEN_BULL, width=2.0, dash="dot")))
    return _apply_theme(fig, f"{ticker} — Actual vs AI Predicted Price", height=390, theme=theme)


def plot_regression_metrics_comparison(metrics_df: pd.DataFrame, theme: str = "Dark") -> go.Figure:
    tk = _get_theme_tokens(theme)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=metrics_df["Model"], y=metrics_df["Train_R2"], name="Study Score (Train R²)", marker_color=tk["line_primary"]))
    fig.add_trace(go.Bar(x=metrics_df["Model"], y=metrics_df["Test_R2"], name="Real Test Score (Test R²)", marker_color=GREEN_BULL))
    fig.update_layout(barmode="group", xaxis_tickangle=-22)
    return _apply_theme(fig, "Regression Models — Study vs Real Test Accuracy (R²)", height=410, theme=theme)


def plot_gradient_descent_curve(loss_df: pd.DataFrame, theme: str = "Dark") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=loss_df["Epoch"], y=loss_df["MSE_Loss"], mode="lines", name="MSE Loss", line=dict(color=AMBER_ACCENT, width=2.4)))
    return _apply_theme(fig, "Gradient Descent — Error Dropping Over Steps", height=330, theme=theme)


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, theme: str = "Dark") -> go.Figure:
    fig = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale=NEON_DIVERGING, zmin=-1.0, zmax=1.0, aspect="auto")
    return _apply_theme(fig, "Clue Similarity Map (Correlation Heatmap)", height=490, theme=theme)


def plot_pca_scree(pca_var_df: pd.DataFrame, theme: str = "Dark") -> go.Figure:
    tk = _get_theme_tokens(theme)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=pca_var_df["Component"], y=pca_var_df["Explained_Variance_Pct"], name="Single PC (%)", marker_color=tk["line_primary"]), secondary_y=False)
    fig.add_trace(go.Scatter(x=pca_var_df["Component"], y=pca_var_df["Cumulative_Variance_Pct"], mode="lines+markers", name="Total Info Kept (%)", line=dict(color=GREEN_BULL, width=2.4)), secondary_y=True)
    fig.add_hline(y=90, line_dash="dash", line_color=AMBER_ACCENT, secondary_y=True, annotation_text="90% Info")
    return _apply_theme(fig, "PCA — How Much Information Each Component Keeps", height=370, theme=theme)


def plot_2d_projection(df_2d: pd.DataFrame, method_title: str, theme: str = "Dark") -> go.Figure:
    color_map = {"Bullish (Up)": GREEN_BULL, "Neutral (Flat)": AMBER_ACCENT, "Bearish (Down)": RED_BEAR}
    fig = px.scatter(df_2d, x="Dim1", y="Dim2", color="Market_Regime", color_discrete_map=color_map, opacity=0.85)
    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="#04070B")))
    return _apply_theme(fig, f"2D Map — {method_title}", height=380, theme=theme)


def plot_elbow_and_silhouette(k_eval_df: pd.DataFrame, theme: str = "Dark") -> go.Figure:
    tk = _get_theme_tokens(theme)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Elbow Curve (Lower = Tighter)", "Silhouette Score (Higher = Cleaner)"))
    fig.add_trace(go.Scatter(x=k_eval_df["k"], y=k_eval_df["WCSS_Inertia"], mode="lines+markers", name="WCSS", line=dict(color=tk["line_primary"], width=2.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=k_eval_df["k"], y=k_eval_df["Silhouette_Score"], mode="lines+markers", name="Silhouette", line=dict(color=GREEN_BULL, width=2.2)), row=1, col=2)
    return _apply_theme(fig, "Finding the Best Number of Stock Groups (k)", height=360, theme=theme)


def plot_cluster_scatter(clustered_df: pd.DataFrame, cluster_col: str, title: str, theme: str = "Dark") -> go.Figure:
    df_plot = clustered_df.copy()
    df_plot["Team"] = "Group " + df_plot[cluster_col].astype(str)
    fig = px.scatter(
        df_plot,
        x="Annual_Volatility_Pct",
        y="Annual_Return_Pct",
        color="Team",
        text="Ticker",
        hover_data=["Company", "Market", "Sharpe_Ratio", "Avg_RSI"],
        color_discrete_sequence=[GREEN_BULL, CYAN_ACCENT, AMBER_ACCENT, PURPLE_ACCENT, RED_BEAR, PINK_ACCENT],
    )
    fig.update_traces(textposition="top center", marker=dict(size=14, line=dict(width=1.5, color="#04070B")), textfont=dict(color="#DCEFE5", size=11))
    fig.update_xaxes(title_text="Risk / Volatility (%)")
    fig.update_yaxes(title_text="Yearly Return (%)")
    return _apply_theme(fig, title, height=430, theme=theme)


def create_matplotlib_dendrogram(linkage_matrix: np.ndarray, labels: list, theme: str = "Dark"):
    tk = _get_theme_tokens(theme)
    bg = tk["mpl_bg"]
    txt = "#DCEFE5"
    fig, ax = plt.subplots(figsize=(10, 4.0), facecolor=bg)
    ax.set_facecolor(bg)
    from scipy.cluster.hierarchy import set_link_color_palette
    set_link_color_palette([GREEN_BULL, CYAN_ACCENT, AMBER_ACCENT, PURPLE_ACCENT, PINK_ACCENT])
    dendrogram(
        linkage_matrix,
        labels=labels,
        leaf_rotation=25,
        leaf_font_size=10,
        color_threshold=0.7 * np.max(linkage_matrix[:, 2]),
        above_threshold_color="#5E7A70",
        ax=ax,
    )
    ax.grid(axis="y", color=(0.47, 0.78, 0.67, 0.10), linewidth=0.8)
    ax.set_title("Stock Family Tree (Agglomerative Dendrogram)", color=txt, fontsize=12, pad=10)
    ax.set_xlabel("Stock Ticker", color=txt, fontsize=10)
    ax.set_ylabel("Distance", color=txt, fontsize=10)
    ax.tick_params(axis="x", colors=txt)
    ax.tick_params(axis="y", colors=txt)
    for spine in ax.spines.values():
        spine.set_color((0.47, 0.78, 0.67, 0.25))
    plt.tight_layout()
    return fig


def plot_rl_portfolio_performance(traj_df: pd.DataFrame, algorithm_name: str, theme: str = "Dark") -> go.Figure:
    tk = _get_theme_tokens(theme)
    fig = go.Figure()
    _glow_line(fig, traj_df["Date"], traj_df["RL_Portfolio_Value"], GREEN_BULL, f"AI Bot ({algorithm_name})", width=2.4)
    fig.add_trace(go.Scatter(x=traj_df["Date"], y=traj_df["Buy_And_Hold_Value"], mode="lines", name="Just Buy & Hold", line=dict(color=tk["line_primary"], width=1.8, dash="dash")))
    buys = traj_df[traj_df["Action"] == "Buy"].iloc[::4]
    sells = traj_df[traj_df["Action"] == "Sell"].iloc[::4]
    fig.add_trace(go.Scatter(x=buys["Date"], y=buys["RL_Portfolio_Value"], mode="markers", name="Buy Move", marker=dict(symbol="triangle-up", size=8, color=GREEN_BULL)))
    fig.add_trace(go.Scatter(x=sells["Date"], y=sells["RL_Portfolio_Value"], mode="markers", name="Sell Move", marker=dict(symbol="triangle-down", size=8, color=RED_BEAR)))
    return _apply_theme(fig, f"Play-Money Portfolio — {algorithm_name}", height=410, theme=theme)


def plot_activation_functions(curves_df: pd.DataFrame, current_z: float, theme: str = "Dark") -> go.Figure:
    tk = _get_theme_tokens(theme)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curves_df["z"], y=curves_df["Binary_Sigmoid"], mode="lines", name="Binary Sigmoid [0, 1]", line=dict(color=GREEN_BULL, width=2.3)))
    fig.add_trace(go.Scatter(x=curves_df["z"], y=curves_df["Bipolar_Sigmoid"], mode="lines", name="Bipolar Sigmoid [-1, +1]", line=dict(color=tk["line_primary"], width=2.3)))
    fig.add_trace(go.Scatter(x=curves_df["z"], y=np.clip(curves_df["ReLU"], 0, 2.0), mode="lines", name="ReLU (Clipped at 2)", line=dict(color=AMBER_ACCENT, width=2.0, dash="dot")))
    fig.add_vline(x=float(np.clip(current_z, -6.0, 6.0)), line_dash="dash", line_color=RED_BEAR, annotation_text=f"z = {current_z:.2f}")
    return _apply_theme(fig, "Activation Curves (Binary vs Bipolar Sigmoid vs ReLU)", height=360, theme=theme)


def plot_mlp_loss_curves(loss_df: pd.DataFrame, theme: str = "Dark") -> go.Figure:
    tk = _get_theme_tokens(theme)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=loss_df["Epoch"], y=loss_df["Training_Loss"], mode="lines", name="Study Error (Train)", line=dict(color=tk["line_primary"], width=2.2)))
    fig.add_trace(go.Scatter(x=loss_df["Epoch"], y=loss_df["Validation_Loss"], mode="lines", name="Check Error (Validation)", line=dict(color=AMBER_ACCENT, width=2.2, dash="dash")))
    return _apply_theme(fig, "MLP Learning Progress (Lower Error = Better)", height=350, theme=theme)


def plot_confusion_matrix_chart(cm: np.ndarray, title: str = "Confusion Matrix", theme: str = "Dark") -> go.Figure:
    labels = ["Down (0)", "Up (1)"]
    fig = px.imshow(cm, x=labels, y=labels, text_auto=True, color_continuous_scale=NEON_SEQUENTIAL, labels=dict(x="AI Guessed", y="Actually Happened", color="Days"))
    return _apply_theme(fig, title, height=350, theme=theme)


def plot_future_forecast_chart(recent_df: pd.DataFrame, future_df: pd.DataFrame, ticker: str, model_name: str, theme: str = "Dark") -> go.Figure:
    tk = _get_theme_tokens(theme)
    hist_tail = recent_df.iloc[-45:].copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_tail["Date"], y=hist_tail["Close"], mode="lines", name="Past 45 Days", line=dict(color=tk["line_primary"], width=2.4)))

    f_dates = pd.to_datetime(future_df["Estimated_Date"])
    connect_dates = [hist_tail["Date"].iloc[-1]] + list(f_dates)
    connect_prices = [float(hist_tail["Close"].iloc[-1])] + list(future_df["Estimated_Close"])

    fig.add_trace(go.Scatter(x=connect_dates, y=connect_prices, mode="lines+markers", name=f"AI Estimate ({model_name})", line=dict(color=GREEN_BULL, width=2.5, dash="dash"), marker=dict(size=7, symbol="diamond")))
    fig.add_trace(
        go.Scatter(
            x=list(f_dates) + list(f_dates)[::-1],
            y=list(future_df["Upper_95_Band"]) + list(future_df["Lower_95_Band"])[::-1],
            fill="toself",
            fillcolor="rgba(46, 232, 138, 0.14)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Normal Wiggle Room (95%)",
        )
    )
    return _apply_theme(fig, f"{ticker} — Recent History & Next-Week AI Estimate", height=410, theme=theme)


def plot_stock_comparison_chart(df1: pd.DataFrame, ticker1: str, df2: pd.DataFrame, ticker2: str, theme: str = "Dark") -> go.Figure:
    """Compares two stocks on a normalized percentage growth basis (Start = 0%), like Yahoo Finance & TradingView."""
    tk = _get_theme_tokens(theme)
    s1 = ((df1["Close"] / float(df1["Close"].iloc[0])) - 1.0) * 100.0
    s2 = ((df2["Close"] / float(df2["Close"].iloc[0])) - 1.0) * 100.0

    fig = go.Figure()
    _glow_line(fig, df1["Date"], s1, GREEN_BULL, f"{ticker1} (% Return)", width=2.4)
    fig.add_trace(go.Scatter(x=df2["Date"], y=s2, mode="lines", name=f"{ticker2} (% Return)", line=dict(color=tk["line_primary"], width=2.4)))
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(169,194,181,0.45)")
    fig.update_yaxes(ticksuffix="%")
    return _apply_theme(fig, f"Head-to-Head Growth: {ticker1} vs {ticker2}", height=390, theme=theme)


def plot_investment_growth_chart(df: pd.DataFrame, ticker: str, invested_amount: float, currency: str, theme: str = "Dark") -> go.Figure:
    """Groww-style What-If investment value curve showing how an invested amount grew over the window."""
    values = invested_amount * (df["Close"] / float(df["Close"].iloc[0]))
    is_profit = float(values.iloc[-1]) >= invested_amount
    line_c = GREEN_BULL if is_profit else RED_BEAR
    fill_c = _rgba(line_c, 0.12)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=values, mode="lines", name="Portfolio Value", line=dict(color=line_c, width=2.4), fill="tozeroy", fillcolor=fill_c))
    fig.add_hline(y=invested_amount, line_dash="dash", line_color="rgba(169,194,181,0.45)", annotation_text=f"Invested: {currency}{invested_amount:,.0f}")
    return _apply_theme(fig, f"If You Invested {currency}{invested_amount:,.0f} in {ticker}", height=360, theme=theme)
