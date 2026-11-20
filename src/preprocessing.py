"""
Data Preprocessing Module (src/preprocessing.py)
Handles missing value cleaning, chronological time-series splitting, and leakage-free feature scaling.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def clean_stock_data(df: pd.DataFrame):
    """
    Cleans raw OHLCV stock data without introducing future data leakage.

    Steps:
    1. Standardizes Date column and sorts chronologically.
    2. Converts price and volume columns to numeric types.
    3. Forward-fills missing values (using only past prices) and drops any invalid initial rows.

    Returns:
        tuple: (cleaned_df: pd.DataFrame, cleaning_report: dict)
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty or None.")

    work_df = df.copy()
    initial_rows = len(work_df)

    work_df["Date"] = pd.to_datetime(work_df["Date"]).dt.tz_localize(None)
    work_df = work_df.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)

    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing_before = int(work_df[numeric_cols].isna().sum().sum())

    for col in numeric_cols:
        work_df[col] = pd.to_numeric(work_df[col], errors="coerce")

    # Replace non-positive prices with NaN
    for price_col in ["Open", "High", "Low", "Close"]:
        work_df.loc[work_df[price_col] <= 0, price_col] = np.nan

    # Forward-fill uses strictly past observations (avoids looking into the future)
    work_df[numeric_cols] = work_df[numeric_cols].ffill()
    work_df = work_df.dropna(subset=numeric_cols).reset_index(drop=True)

    missing_after = int(work_df[numeric_cols].isna().sum().sum())
    final_rows = len(work_df)

    report = {
        "initial_rows": initial_rows,
        "final_rows": final_rows,
        "rows_removed": initial_rows - final_rows,
        "missing_values_before": missing_before,
        "missing_values_after": missing_after,
        "start_date": work_df["Date"].iloc[0].strftime("%Y-%m-%d") if final_rows > 0 else "N/A",
        "end_date": work_df["Date"].iloc[-1].strftime("%Y-%m-%d") if final_rows > 0 else "N/A",
    }
    return work_df, report


def chronological_train_test_split(df: pd.DataFrame, feature_cols: list, target_col: str, train_ratio: float = 0.8):
    """
    Splits a time-series DataFrame strictly in chronological order (no random shuffling).

    Why chronological splitting is required:
    Random train_test_split leaks future market regimes into the training set.
    Chronological splitting trains on past dates (first 80%) and evaluates on unseen future dates (last 20%).
    """
    valid_df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
    n_total = len(valid_df)
    if n_total < 30:
        raise ValueError(f"Insufficient valid rows ({n_total}) for train-test splitting. Need at least 30 rows.")

    split_idx = int(n_total * train_ratio)
    train_df = valid_df.iloc[:split_idx].copy()
    test_df = valid_df.iloc[split_idx:].copy()

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].copy()

    dates_train = train_df["Date"].copy()
    dates_test = test_df["Date"].copy()

    return X_train, X_test, y_train, y_test, dates_train, dates_test


def scale_features_leakage_free(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Fits StandardScaler ONLY on X_train, then transforms both X_train and X_test.
    Returns scaled DataFrames preserving column names and index, plus the fitted scaler.
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    return X_train_scaled, X_test_scaled, scaler
