"""
Unit 1: Regression Methods Module (src/regression.py)
Implements all MAI511-2 Unit 1 syllabus algorithms:
1. Simple Linear Regression
2. Ordinary Least Squares (OLS) Closed-Form Estimation
3. Gradient Descent (Step-by-step Batch Gradient Descent + SGDRegressor)
4. Multiple Linear Regression
5. Multivariate Linear Regression (Predicting Close, High, Low simultaneously)
6. Polynomial Regression (Degrees 2 and 3)
7. Ridge Regression (L2 Regularization)
8. Lasso Regression (L1 Regularization)
9. Decision Tree Regression (Pruned vs Unconstrained for Bias-Variance / Overfitting)
10. Model Trees (Decision Tree with Linear Regression Models at Leaf Nodes)
11. Logistic Regression (Classification: Next-Day Price Increase vs Decrease)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate_regression_metrics(y_true, y_pred) -> dict:
    """Calculates standard regression metrics: MAE, MSE, RMSE, and R2 Score."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": round(mae, 4), "MSE": round(mse, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}


def demonstrate_ols_and_gradient_descent(X_train_close: np.ndarray, y_train: np.ndarray, lr: float = 0.05, epochs: int = 150):
    """
    Demonstrates Ordinary Least Squares (OLS) closed-form matrix solution vs
    iterative Batch Gradient Descent optimization on standardized Simple Linear Regression.
    """
    x_mean, x_std = X_train_close.mean(), X_train_close.std() + 1e-8
    y_mean, y_std = y_train.mean(), y_train.std() + 1e-8

    x_norm = (X_train_close - x_mean) / x_std
    y_norm = (y_train - y_mean) / y_std

    # 1. Ordinary Least Squares (OLS) Closed-Form: beta = (X^T X)^(-1) X^T y
    X_design = np.column_stack([np.ones_like(x_norm), x_norm])
    ols_beta = np.linalg.pinv(X_design.T @ X_design) @ (X_design.T @ y_norm)
    ols_intercept_norm, ols_slope_norm = float(ols_beta[0]), float(ols_beta[1])

    # Convert normalized parameters back to original price scale
    ols_slope = ols_slope_norm * (y_std / x_std)
    ols_intercept = y_mean - ols_slope * x_mean

    # 2. Batch Gradient Descent
    b0, b1 = 0.0, 0.0
    n = float(len(x_norm))
    loss_history = []
    param_history = []

    for epoch in range(epochs):
        y_hat = b0 + b1 * x_norm
        error = y_hat - y_norm
        mse_loss = float(np.mean(error ** 2))
        loss_history.append({"Epoch": epoch + 1, "MSE_Loss": round(mse_loss, 6)})

        grad_b0 = (2.0 / n) * np.sum(error)
        grad_b1 = (2.0 / n) * np.sum(error * x_norm)

        b0 -= lr * grad_b0
        b1 -= lr * grad_b1

        if epoch % 10 == 0 or epoch == epochs - 1:
            slope_orig = b1 * (y_std / x_std)
            intercept_orig = y_mean - slope_orig * x_mean
            param_history.append({
                "Epoch": epoch + 1,
                "Slope": round(float(slope_orig), 5),
                "Intercept": round(float(intercept_orig), 4),
                "Normalized_MSE": round(mse_loss, 6),
            })

    gd_slope = b1 * (y_std / x_std)
    gd_intercept = y_mean - gd_slope * x_mean

    return {
        "OLS": {"Intercept": round(float(ols_intercept), 4), "Slope": round(float(ols_slope), 5)},
        "Gradient_Descent": {"Intercept": round(float(gd_intercept), 4), "Slope": round(float(gd_slope), 5)},
        "loss_history": pd.DataFrame(loss_history),
        "param_history": pd.DataFrame(param_history),
    }


class SimpleLinearModelTree:
    """
    Model Tree implementation (Unit 1 Syllabus Topic 12):
    Unlike a standard Decision Tree Regressor that predicts a constant average value in each leaf,
    a Model Tree fits a local Ridge/Linear Regression model inside each leaf node.
    """

    def __init__(self, max_depth: int = 2, min_samples_leaf: int = 20):
        self.tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)
        self.leaf_models = {}

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.tree.fit(X, y)
        leaf_ids = self.tree.apply(X)
        for leaf_id in np.unique(leaf_ids):
            mask = leaf_ids == leaf_id
            local_model = Ridge(alpha=1.0)
            local_model.fit(X[mask], y[mask])
            self.leaf_models[leaf_id] = local_model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        leaf_ids = self.tree.apply(X)
        preds = np.zeros(len(X))
        for i, leaf_id in enumerate(leaf_ids):
            model = self.leaf_models.get(leaf_id)
            if model is not None:
                preds[i] = float(model.predict(X[i : i + 1])[0])
            else:
                preds[i] = float(self.tree.predict(X[i : i + 1])[0])
        return preds


def run_all_regression_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    ridge_alpha: float = 1.0,
    lasso_alpha: float = 0.05,
    poly_degree: int = 2,
):
    """
    Trains and evaluates all Unit 1 continuous regression models on chronological train/test splits.
    Fits scalers strictly on X_train to prevent data leakage.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train_arr = y_train.values.astype(float)
    y_test_arr = y_test.values.astype(float)

    results = []
    predictions = {}
    trained_models = {}

    def _record_model(name: str, y_tr_pred, y_te_pred, description: str, model_obj=None):
        tr_m = evaluate_regression_metrics(y_train_arr, y_tr_pred)
        te_m = evaluate_regression_metrics(y_test_arr, y_te_pred)
        r2_gap = round(tr_m["R2"] - te_m["R2"], 4)
        if tr_m["R2"] < 0.50 and te_m["R2"] < 0.50:
            fit_status = "Underfitting (High Bias)"
        elif r2_gap > 0.12:
            fit_status = "Overfitting (High Variance)"
        else:
            fit_status = "Balanced Fit (Good Generalization)"

        results.append({
            "Model": name,
            "Train_MAE": tr_m["MAE"],
            "Test_MAE": te_m["MAE"],
            "Train_MSE": tr_m["MSE"],
            "Test_MSE": te_m["MSE"],
            "Train_RMSE": tr_m["RMSE"],
            "Test_RMSE": te_m["RMSE"],
            "Train_R2": tr_m["R2"],
            "Test_R2": te_m["R2"],
            "R2_Gap": r2_gap,
            "Bias_Variance_Diagnosis": fit_status,
            "Academic_Note": description,
        })
        predictions[name] = {
            "train_pred": y_tr_pred,
            "test_pred": y_te_pred,
        }
        if model_obj is not None:
            trained_models[name] = model_obj

    # 0. Baseline Persistence Model (Predicts Tomorrow Close = Today Close)
    baseline_tr = X_train["Close"].values
    baseline_te = X_test["Close"].values
    _record_model(
        "0. Naive Baseline (Close_t)",
        baseline_tr,
        baseline_te,
        "Reference benchmark: assumes tomorrow's closing price equals today's closing price.",
    )

    # 1. Simple Linear Regression (Single feature: Close)
    slr = LinearRegression()
    slr.fit(X_train[["Close"]].values, y_train_arr)
    _record_model(
        "1. Simple Linear Regression",
        slr.predict(X_train[["Close"]].values),
        slr.predict(X_test[["Close"]].values),
        "Uses only today's Close price (1 feature) via Ordinary Least Squares.",
        slr,
    )

    # 2. Multiple Linear Regression (All technical features via OLS)
    mlr = LinearRegression()
    mlr.fit(X_train_scaled, y_train_arr)
    _record_model(
        "2. Multiple Linear Regression (OLS)",
        mlr.predict(X_train_scaled),
        mlr.predict(X_test_scaled),
        "Uses all engineered technical indicators simultaneously via OLS estimation.",
        (mlr, scaler),
    )

    # 3. Gradient Descent Linear Regression (SGDRegressor)
    sgd = SGDRegressor(max_iter=2000, tol=1e-4, penalty=None, learning_rate="invscaling", eta0=0.01, random_state=42)
    sgd.fit(X_train_scaled, y_train_arr)
    _record_model(
        "3. Gradient Descent Regressor (SGD)",
        sgd.predict(X_train_scaled),
        sgd.predict(X_test_scaled),
        "Optimizes linear weights iteratively using Stochastic Gradient Descent.",
        (sgd, scaler),
    )

    # 4. Polynomial Regression (Nonlinear expansion on key momentum/price features)
    poly_cols = ["Close", "SMA_10", "RSI_14", "MACD", "Volatility_20"]
    poly_scaler = StandardScaler()
    X_tr_p = poly_scaler.fit_transform(X_train[poly_cols])
    X_te_p = poly_scaler.transform(X_test[poly_cols])
    poly_trans = PolynomialFeatures(degree=poly_degree, include_bias=False)
    X_tr_poly = poly_trans.fit_transform(X_tr_p)
    X_te_poly = poly_trans.transform(X_te_p)
    poly_reg = Ridge(alpha=1.0)
    poly_reg.fit(X_tr_poly, y_train_arr)
    _record_model(
        f"4. Polynomial Regression (Degree={poly_degree})",
        poly_reg.predict(X_tr_poly),
        poly_reg.predict(X_te_poly),
        f"Captures nonlinear feature interactions using degree-{poly_degree} polynomial terms.",
        (poly_reg, poly_scaler, poly_trans, poly_cols),
    )

    # 5. Ridge Regression (L2 Regularization)
    ridge = Ridge(alpha=ridge_alpha, random_state=42)
    ridge.fit(X_train_scaled, y_train_arr)
    _record_model(
        f"5. Ridge Regression (L2, alpha={ridge_alpha})",
        ridge.predict(X_train_scaled),
        ridge.predict(X_test_scaled),
        "Shrinks correlated indicator weights using L2 penalty to reduce variance.",
        (ridge, scaler),
    )

    # 6. Lasso Regression (L1 Regularization)
    lasso = Lasso(alpha=lasso_alpha, max_iter=5000, random_state=42)
    lasso.fit(X_train_scaled, y_train_arr)
    _record_model(
        f"6. Lasso Regression (L1, alpha={lasso_alpha})",
        lasso.predict(X_train_scaled),
        lasso.predict(X_test_scaled),
        "Applies L1 penalty to drive redundant feature coefficients to exact zero.",
        (lasso, scaler),
    )

    # 7. Decision Tree Regression (Pruned, max_depth=4)
    dt_pruned = DecisionTreeRegressor(max_depth=4, min_samples_leaf=10, random_state=42)
    dt_pruned.fit(X_train_scaled, y_train_arr)
    _record_model(
        "7. Decision Tree Regressor (Pruned, depth=4)",
        dt_pruned.predict(X_train_scaled),
        dt_pruned.predict(X_test_scaled),
        "Axis-aligned recursive partitioning with depth control to prevent overfitting.",
        (dt_pruned, scaler),
    )

    # 8. Decision Tree Regression (Unconstrained - Overfitting Demonstration)
    dt_overfit = DecisionTreeRegressor(max_depth=None, min_samples_split=2, random_state=42)
    dt_overfit.fit(X_train_scaled, y_train_arr)
    _record_model(
        "8. Decision Tree Regressor (Unconstrained)",
        dt_overfit.predict(X_train_scaled),
        dt_overfit.predict(X_test_scaled),
        "Demonstrates overfitting: memorizes training data (R2~1.0) but degrades on unseen test data.",
        (dt_overfit, scaler),
    )

    # 9. Model Tree (Decision Tree with Linear Regression in Leaves)
    model_tree = SimpleLinearModelTree(max_depth=2, min_samples_leaf=20)
    model_tree.fit(X_train_scaled, y_train_arr)
    _record_model(
        "9. Model Tree (Piecewise Linear Leaves)",
        model_tree.predict(X_train_scaled),
        model_tree.predict(X_test_scaled),
        "Combines decision tree regime splitting with local linear regression models in each leaf.",
        (model_tree, scaler),
    )

    # Extract feature coefficients comparison (MLR vs Ridge vs Lasso)
    coef_df = pd.DataFrame({
        "Feature": X_train.columns,
        "OLS_Multiple_Linear": np.round(mlr.coef_, 4),
        "Ridge_L2_Coef": np.round(ridge.coef_, 4),
        "Lasso_L1_Coef": np.round(lasso.coef_, 4),
    })

    # OLS vs Gradient Descent demonstration on Close -> Target_Next_Close
    ols_gd_demo = demonstrate_ols_and_gradient_descent(X_train["Close"].values, y_train_arr)

    return {
        "metrics_df": pd.DataFrame(results),
        "predictions": predictions,
        "trained_models": trained_models,
        "coef_df": coef_df,
        "ols_gd_demo": ols_gd_demo,
    }


def run_multivariate_linear_regression(df_modeling: pd.DataFrame, feature_cols: list, train_ratio: float = 0.8):
    """
    Demonstrates Multivariate Linear Regression (Unit 1 Syllabus Topic 6):
    Predicts multiple continuous output variables simultaneously:
    Y = [Target_Next_Close, Target_Next_High, Target_Next_Low]
    """
    target_cols = ["Target_Next_Close", "Target_Next_High", "Target_Next_Low"]
    valid_df = df_modeling.dropna(subset=feature_cols + target_cols).reset_index(drop=True)
    split_idx = int(len(valid_df) * train_ratio)

    X_tr = valid_df[feature_cols].iloc[:split_idx]
    X_te = valid_df[feature_cols].iloc[split_idx:]
    Y_tr = valid_df[target_cols].iloc[:split_idx].values
    Y_te = valid_df[target_cols].iloc[split_idx:].values

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    mv_model = LinearRegression()
    mv_model.fit(X_tr_s, Y_tr)
    Y_pred = mv_model.predict(X_te_s)

    rows = []
    for idx, t_name in enumerate(target_cols):
        m = evaluate_regression_metrics(Y_te[:, idx], Y_pred[:, idx])
        rows.append({
            "Simultaneous_Output_Target": t_name,
            "Test_MAE": m["MAE"],
            "Test_RMSE": m["RMSE"],
            "Test_R2": m["R2"],
        })
    return pd.DataFrame(rows)


def run_logistic_regression_direction(X_train: pd.DataFrame, X_test: pd.DataFrame, y_dir_train: pd.Series, y_dir_test: pd.Series):
    """
    Implements Logistic Regression (Unit 1 Syllabus Topic 13) strictly for binary classification:
    Predicts whether the next trading day's closing price will Increase (1) or Decrease (0).
    """
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_tr_s, y_dir_train.values)

    tr_pred = clf.predict(X_tr_s)
    te_pred = clf.predict(X_te_s)
    te_prob = clf.predict_proba(X_te_s)[:, 1]

    cm = confusion_matrix(y_dir_test.values, te_pred)
    return {
        "train_accuracy": round(float(accuracy_score(y_dir_train.values, tr_pred)) * 100.0, 2),
        "test_accuracy": round(float(accuracy_score(y_dir_test.values, te_pred)) * 100.0, 2),
        "precision": round(float(precision_score(y_dir_test.values, te_pred, zero_division=0)) * 100.0, 2),
        "recall": round(float(recall_score(y_dir_test.values, te_pred, zero_division=0)) * 100.0, 2),
        "f1_score": round(float(f1_score(y_dir_test.values, te_pred, zero_division=0)) * 100.0, 2),
        "confusion_matrix": cm,
        "test_probabilities": te_prob,
        "test_predictions": te_pred,
    }
