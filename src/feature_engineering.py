"""
Feature Engineering Module (src/feature_engineering.py)
Computes technical indicators and shifted prediction targets without future data leakage.
"""

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "SMA_10",
    "SMA_20",
    "EMA_12",
    "EMA_26",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "Daily_Return",
    "Price_Change_Pct",
    "Volatility_20",
    "Volume_Ratio",
    "High_Low_Spread",
    "Lag_Close_1",
    "Lag_Return_1",
]

INDICATOR_EXPLANATIONS = {
    "SMA_10 & SMA_20 (Simple Moving Average)": (
        "Calculates the unweighted mean of the previous 10 or 20 closing prices. "
        "It smooths out day-to-day noise to reveal the underlying short-term and medium-term price trend."
    ),
    "EMA_12 & EMA_26 (Exponential Moving Average)": (
        "Gives higher weight to recent trading days so it reacts faster to sudden price changes than a Simple Moving Average."
    ),
    "RSI_14 (Relative Strength Index)": (
        "A momentum oscillator ranging from 0 to 100 that measures the speed and magnitude of recent gains versus losses over 14 days. "
        "Values above 70 typically indicate overbought conditions, while values below 30 indicate oversold conditions."
    ),
    "MACD (Moving Average Convergence Divergence)": (
        "Computed as EMA(12) minus EMA(26), alongside a 9-day EMA Signal Line. "
        "When MACD crosses above its Signal Line, it indicates upward bullish momentum."
    ),
    "Daily_Return (%)": (
        "The percentage change from the previous trading day's closing price to today's closing price: "
        "((Close_t - Close_{t-1}) / Close_{t-1}) * 100."
    ),
    "Volatility_20 (Rolling Volatility)": (
        "The 20-day rolling standard deviation of daily returns. Higher volatility indicates larger price swings and higher market risk."
    ),
    "Price_Change_Pct (%)": (
        "Intraday percentage movement from market Open to Close on day t: ((Close_t - Open_t) / Open_t) * 100."
    ),
    "Volume_Ratio (Trading Volume Analysis)": (
        "Compares today's trading volume against its 20-day average volume (Volume_t / Volume_SMA_20). "
        "A ratio above 1.5 indicates unusually strong buying or selling participation."
    ),
}


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Calculates the Relative Strength Index (RSI) over a specified rolling window."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def create_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers technical indicators using strictly historical/current day (t) values,
    and creates next-day targets (t+1) by shifting backward.

    Leakage Prevention Guarantee:
    - Every feature at row t uses ONLY prices/volumes from day <= t.
    - Target_Next_Close at row t is Close_{t+1} = df['Close'].shift(-1).
    - Target_Direction at row t is 1 if Close_{t+1} > Close_t else 0.
    """
    feat_df = df.copy()

    close = feat_df["Close"]
    open_p = feat_df["Open"]
    high = feat_df["High"]
    low = feat_df["Low"]
    volume = feat_df["Volume"]

    # 1. Moving Averages (SMA & EMA)
    feat_df["SMA_10"] = close.rolling(window=10).mean()
    feat_df["SMA_20"] = close.rolling(window=20).mean()
    feat_df["SMA_50"] = close.rolling(window=50).mean()
    feat_df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    feat_df["EMA_26"] = close.ewm(span=26, adjust=False).mean()

    # 2. MACD and Signal Line
    feat_df["MACD"] = feat_df["EMA_12"] - feat_df["EMA_26"]
    feat_df["MACD_Signal"] = feat_df["MACD"].ewm(span=9, adjust=False).mean()
    feat_df["MACD_Hist"] = feat_df["MACD"] - feat_df["MACD_Signal"]

    # 3. Relative Strength Index (14-day)
    feat_df["RSI_14"] = compute_rsi(close, window=14)

    # 4. Returns, Intraday Spread, and Rolling Volatility
    feat_df["Daily_Return"] = close.pct_change() * 100.0
    feat_df["Price_Change_Pct"] = ((close - open_p) / open_p) * 100.0
    feat_df["High_Low_Spread"] = ((high - low) / low) * 100.0
    feat_df["Volatility_20"] = feat_df["Daily_Return"].rolling(window=20).std()

    # 5. Trading Volume Analysis
    feat_df["Volume_SMA_20"] = volume.rolling(window=20).mean()
    feat_df["Volume_Ratio"] = volume / feat_df["Volume_SMA_20"].replace(0, np.nan)
    feat_df["Volume_Change_Pct"] = volume.pct_change() * 100.0

    # 6. Lagged Features (strictly past values t-1, t-2)
    feat_df["Lag_Close_1"] = close.shift(1)
    feat_df["Lag_Close_2"] = close.shift(2)
    feat_df["Lag_Return_1"] = feat_df["Daily_Return"].shift(1)

    # 7. Next-Trading-Day Targets (t+1)
    feat_df["Target_Next_Close"] = close.shift(-1)
    feat_df["Target_Next_High"] = high.shift(-1)
    feat_df["Target_Next_Low"] = low.shift(-1)
    feat_df["Target_Direction"] = (feat_df["Target_Next_Close"] > close).astype(int)

    # Multi-class trend label for LDA demonstration (Down = 0, Flat = 1, Up = 2)
    next_ret_pct = ((feat_df["Target_Next_Close"] - close) / close) * 100.0
    feat_df["Target_Trend_Class"] = np.where(
        next_ret_pct > 0.35, 2, np.where(next_ret_pct < -0.35, 0, 1)
    )

    # Replace any infinite values from division
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan)

    return feat_df


def prepare_modeling_dataset(df: pd.DataFrame):
    """
    Returns the modeling DataFrame (with historical rows having valid features and next-day target),
    plus the latest trading day's feature row (used to predict tomorrow's unknown close price).
    """
    feat_df = create_technical_features(df)

    # The very last row has valid features at day T, but Target_Next_Close is NaN (tomorrow's price)
    latest_feature_row = feat_df.dropna(subset=FEATURE_COLUMNS).iloc[[-1]].copy()

    # Historical rows with known next-day close price for training and testing
    modeling_df = feat_df.dropna(subset=FEATURE_COLUMNS + ["Target_Next_Close"]).reset_index(drop=True)

    return modeling_df, latest_feature_row, feat_df
