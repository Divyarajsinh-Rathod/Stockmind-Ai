"""
Stock Price Prediction Engine Module (src/prediction.py)
Coordinates end-to-end data preprocessing, feature selection, model training,
next-trading-day closing price estimation, and multi-day scenario projections.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from src.regression import evaluate_regression_metrics, SimpleLinearModelTree


PREDICTION_MODELS = [
    "Ridge Regression (L2 Regularized)",
    "Multiple Linear Regression (OLS)",
    "Lasso Regression (L1 Regularized)",
    "Model Tree (Piecewise Linear)",
    "Decision Tree Regressor (Pruned)",
    "MLP Neural Network Regressor",
    "Naive Baseline (Close_t)",
]


def build_model_instance(model_name: str):
    if model_name == "Multiple Linear Regression (OLS)":
        return LinearRegression()
    elif model_name == "Lasso Regression (L1 Regularized)":
        return Lasso(alpha=0.05, max_iter=5000, random_state=42)
    elif model_name == "Model Tree (Piecewise Linear)":
        return SimpleLinearModelTree(max_depth=2, min_samples_leaf=20)
    elif model_name == "Decision Tree Regressor (Pruned)":
        return DecisionTreeRegressor(max_depth=4, min_samples_leaf=10, random_state=42)
    elif model_name == "MLP Neural Network Regressor":
        return MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
    else:
        return Ridge(alpha=1.0, random_state=42)


def generate_stock_predictions(
    modeling_df: pd.DataFrame,
    latest_feature_row: pd.DataFrame,
    feature_cols: list,
    selected_model_name: str = "Ridge Regression (L2 Regularized)",
    train_ratio: float = 0.8,
    forecast_horizon_days: int = 5,
):
    """
    Trains the selected model on chronological historical data, evaluates on the test split,
    predicts the next trading day's closing price from today's known features (no leakage),
    and projects an educational multi-day trajectory with uncertainty bands.
    """
    valid_df = modeling_df.dropna(subset=feature_cols + ["Target_Next_Close"]).reset_index(drop=True)
    split_idx = int(len(valid_df) * train_ratio)

    train_df = valid_df.iloc[:split_idx].copy()
    test_df = valid_df.iloc[split_idx:].copy()

    X_train = train_df[feature_cols]
    y_train = train_df["Target_Next_Close"].values.astype(float)
    X_test = test_df[feature_cols]
    y_test = test_df["Target_Next_Close"].values.astype(float)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if selected_model_name == "Naive Baseline (Close_t)":
        y_train_pred = X_train["Close"].values
        y_test_pred = X_test["Close"].values
        latest_close = float(latest_feature_row["Close"].iloc[0])
        next_day_pred = latest_close
    else:
        model = build_model_instance(selected_model_name)
        model.fit(X_train_s, y_train)
        y_train_pred = model.predict(X_train_s)
        y_test_pred = model.predict(X_test_s)

        # Fit on 100% of available historical labeled rows to generate tomorrow's forecast
        full_scaler = StandardScaler()
        X_full_s = full_scaler.fit_transform(valid_df[feature_cols])
        y_full = valid_df["Target_Next_Close"].values.astype(float)
        final_model = build_model_instance(selected_model_name)
        final_model.fit(X_full_s, y_full)

        X_latest_s = full_scaler.transform(latest_feature_row[feature_cols])
        next_day_pred = float(final_model.predict(X_latest_s)[0])

    train_metrics = evaluate_regression_metrics(y_train, y_train_pred)
    test_metrics = evaluate_regression_metrics(y_test, y_test_pred)

    latest_close = float(latest_feature_row["Close"].iloc[0])
    latest_date = pd.to_datetime(latest_feature_row["Date"].iloc[0])
    expected_change = next_day_pred - latest_close
    expected_change_pct = (expected_change / latest_close) * 100.0

    # Residual standard error on test set for confidence interval estimation
    residual_std = float(np.std(y_test - y_test_pred))

    # Multi-day iterative scenario projection (educational estimate)
    future_dates = pd.bdate_range(start=latest_date + pd.Timedelta(days=1), periods=forecast_horizon_days)
    daily_step_pct = np.clip(expected_change_pct / 100.0, -0.025, 0.025)
    future_rows = []
    curr_p = latest_close

    for d_idx, f_date in enumerate(future_dates):
        if d_idx == 0:
            curr_p = next_day_pred
        else:
            # Damped mean-reverting step based on predicted daily drift
            curr_p = curr_p * (1.0 + daily_step_pct * (0.75 ** d_idx))
        band = 1.96 * residual_std * np.sqrt(d_idx + 1)
        future_rows.append({
            "Trading_Day": f"Day +{d_idx + 1}",
            "Estimated_Date": f_date.strftime("%Y-%m-%d"),
            "Estimated_Close": round(float(curr_p), 2),
            "Lower_95_Band": round(float(max(0.01, curr_p - band)), 2),
            "Upper_95_Band": round(float(curr_p + band), 2),
        })

    comparison_df = pd.DataFrame({
        "Date": test_df["Date"].values,
        "Current_Day_Close": np.round(test_df["Close"].values, 2),
        "Actual_Next_Close": np.round(y_test, 2),
        "Predicted_Next_Close": np.round(y_test_pred, 2),
        "Prediction_Error": np.round(y_test - y_test_pred, 2),
    })

    return {
        "selected_model": selected_model_name,
        "latest_date": latest_date.strftime("%Y-%m-%d"),
        "latest_close": round(latest_close, 2),
        "next_day_predicted_close": round(next_day_pred, 2),
        "expected_change": round(expected_change, 2),
        "expected_change_pct": round(expected_change_pct, 2),
        "signal": "BULLISH (Expected Increase)" if expected_change >= 0 else "BEARISH (Expected Decrease)",
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "residual_std": round(residual_std, 2),
        "test_comparison_df": comparison_df,
        "future_projection_df": pd.DataFrame(future_rows),
    }
