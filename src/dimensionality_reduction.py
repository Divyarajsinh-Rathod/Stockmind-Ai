"""
Unit 2: Dimensionality Reduction & Feature Selection Module (src/dimensionality_reduction.py)
Implements all MAI511-2 Unit 2 syllabus algorithms without data leakage:
1. Low Variance Filter
2. High Correlation Filter
3. Forward Feature Selection (SFS)
4. Backward Feature Elimination (SBS)
5. Maximum Relevance Minimum Redundancy (mRMR)
6. Factor Analysis (FA)
7. Principal Component Analysis (PCA - Unsupervised)
8. Linear Discriminant Analysis (LDA - Supervised)
9. Independent Component Analysis (ICA)
10. Multidimensional Scaling (MDS)
11. Isomap (Manifold Learning)
"""

import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold, SequentialFeatureSelector, f_regression
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import MDS, Isomap
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score


def apply_low_variance_filter(X_train: pd.DataFrame, threshold: float = 0.015):
    """
    Applies Low Variance Filter on MinMax-normalized training features (fit strictly on X_train).
    """
    minmax = MinMaxScaler()
    X_tr_norm = pd.DataFrame(minmax.fit_transform(X_train), columns=X_train.columns)
    variances = X_tr_norm.var()

    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X_tr_norm)
    mask = selector.get_support()

    selected = list(X_train.columns[mask])
    removed = list(X_train.columns[~mask])

    var_table = pd.DataFrame({
        "Feature": X_train.columns,
        "Normalized_Variance": np.round(variances.values, 5),
        "Status": ["Selected" if m else "Removed (Low Variance)" for m in mask],
    }).sort_values("Normalized_Variance", ascending=False).reset_index(drop=True)

    return {"selected": selected, "removed": removed, "table": var_table}


def apply_high_correlation_filter(X_train: pd.DataFrame, threshold: float = 0.85):
    """
    Identifies and removes redundant features whose pairwise absolute Pearson correlation
    exceeds the threshold (computed strictly on X_train).
    """
    corr_matrix = X_train.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    removed = []
    dropped_pairs = []
    for col in upper_tri.columns:
        high_corr_Series = upper_tri[col][upper_tri[col] > threshold]
        if not high_corr_Series.empty:
            removed.append(col)
            for idx_name, val in high_corr_Series.items():
                dropped_pairs.append({
                    "Removed_Feature": col,
                    "Correlated_With": idx_name,
                    "Abs_Correlation": round(float(val), 4),
                })

    selected = [c for c in X_train.columns if c not in removed]
    return {
        "selected": selected,
        "removed": removed,
        "corr_matrix": X_train.corr().round(3),
        "dropped_pairs_df": pd.DataFrame(dropped_pairs),
    }


def apply_forward_backward_selection(X_train: pd.DataFrame, y_train: pd.Series, n_features: int = 6):
    """
    Demonstrates Wrapper Methods: Forward Feature Selection and Backward Feature Elimination
    using Ridge Regression and cross-validation on training data only.
    """
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    n_select = min(max(2, n_features), X_train.shape[1] - 1)

    estimator = Ridge(alpha=1.0)

    # Forward Selection
    sfs_forward = SequentialFeatureSelector(
        estimator, n_features_to_select=n_select, direction="forward", cv=3
    )
    sfs_forward.fit(X_tr_s, y_train.values)
    forward_selected = list(X_train.columns[sfs_forward.get_support()])
    forward_removed = [c for c in X_train.columns if c not in forward_selected]

    # Backward Elimination
    sfs_backward = SequentialFeatureSelector(
        estimator, n_features_to_select=n_select, direction="backward", cv=3
    )
    sfs_backward.fit(X_tr_s, y_train.values)
    backward_selected = list(X_train.columns[sfs_backward.get_support()])
    backward_removed = [c for c in X_train.columns if c not in backward_selected]

    return {
        "forward_selected": forward_selected,
        "forward_removed": forward_removed,
        "backward_selected": backward_selected,
        "backward_removed": backward_removed,
    }


def apply_mrmr_selection(X_train: pd.DataFrame, y_train: pd.Series, n_features: int = 6):
    """
    Implements Maximum Relevance Minimum Redundancy (mRMR - Unit 2 Syllabus Topic 11):
    Greedily selects features that maximize F-statistic relevance with Target_Next_Close
    while minimizing average absolute correlation (redundancy) with already-selected features.
    """
    f_scores, _ = f_regression(X_train.fillna(0), y_train.values)
    f_scores = np.nan_to_num(f_scores, nan=0.0)
    relevance = pd.Series(f_scores, index=X_train.columns)
    corr_matrix = X_train.corr().abs().fillna(0.0)

    selected = []
    remaining = list(X_train.columns)
    selection_log = []

    # Step 1: Pick highest relevance feature first
    first_feat = relevance.idxmax()
    selected.append(first_feat)
    remaining.remove(first_feat)
    selection_log.append({
        "Step": 1,
        "Selected_Feature": first_feat,
        "Relevance_F_Score": round(float(relevance[first_feat]), 2),
        "Avg_Redundancy_Corr": 0.0,
        "mRMR_Score": round(float(relevance[first_feat]), 2),
    })

    # Step 2..k: Maximize Relevance / Redundancy Quotient (FCQ)
    while len(selected) < min(n_features, len(X_train.columns)) and remaining:
        best_feat = None
        best_score = -np.inf
        best_red = 0.0

        for cand in remaining:
            rel = float(relevance[cand])
            red = float(corr_matrix.loc[cand, selected].mean())
            score = rel / max(red, 0.05)
            if score > best_score:
                best_score = score
                best_feat = cand
                best_red = red

        selected.append(best_feat)
        remaining.remove(best_feat)
        selection_log.append({
            "Step": len(selected),
            "Selected_Feature": best_feat,
            "Relevance_F_Score": round(float(relevance[best_feat]), 2),
            "Avg_Redundancy_Corr": round(best_red, 4),
            "mRMR_Score": round(float(best_score), 2),
        })

    return {
        "selected": selected,
        "removed": remaining,
        "mrmr_table": pd.DataFrame(selection_log),
    }


def run_subspace_projections(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_trend_train: pd.Series,
    y_trend_test: pd.Series,
    n_components: int = 3,
):
    """
    Demonstrates Unsupervised vs Supervised Dimensionality Reduction techniques:
    - PCA (Unsupervised orthogonal variance maximization)
    - LDA (Supervised class-separation maximization using 3 trend classes: Down, Flat, Up)
    - ICA (Independent Component Analysis for non-Gaussian source separation)
    - Factor Analysis (Latent common market factors)
    - MDS & Isomap (Distance-preserving & nonlinear geodesic manifold embeddings)
    All transformations are fit strictly on X_train (or subsample for manifold visualization).
    """
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    # 1. PCA (Unsupervised)
    pca_full = PCA(random_state=42)
    pca_full.fit(X_tr_s)
    exp_var_df = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(len(pca_full.explained_variance_ratio_))],
        "Explained_Variance_Pct": np.round(pca_full.explained_variance_ratio_ * 100.0, 2),
        "Cumulative_Variance_Pct": np.round(np.cumsum(pca_full.explained_variance_ratio_) * 100.0, 2),
    })

    pca_2d = PCA(n_components=2, random_state=42)
    X_tr_pca2 = pca_2d.fit_transform(X_tr_s)

    # 2. LDA (Supervised - requires class labels y_trend_train)
    n_classes = len(np.unique(y_trend_train))
    lda_comps = min(2, n_classes - 1, X_tr_s.shape[1])
    lda = LinearDiscriminantAnalysis(n_components=lda_comps)
    X_tr_lda = lda.fit_transform(X_tr_s, y_trend_train.values)
    if X_tr_lda.shape[1] == 1:
        X_tr_lda = np.column_stack([X_tr_lda[:, 0], np.zeros(len(X_tr_lda))])

    # 3. ICA (Independent Component Analysis)
    ica = FastICA(n_components=2, random_state=42, max_iter=500)
    X_tr_ica = ica.fit_transform(X_tr_s)

    # 4. Factor Analysis (Latent Factors)
    fa = FactorAnalysis(n_components=2, random_state=42)
    X_tr_fa = fa.fit_transform(X_tr_s)
    fa_loadings = pd.DataFrame(
        np.round(fa.components_.T, 3),
        index=X_train.columns,
        columns=["Latent_Factor_1 (Trend/Level)", "Latent_Factor_2 (Momentum/Volatility)"],
    ).reset_index().rename(columns={"index": "Feature"})

    # 5. MDS & Isomap (Evaluated on most recent 160 training points for fast interactive rendering)
    sub_n = min(160, len(X_tr_s))
    X_sub = X_tr_s[-sub_n:]
    y_sub = y_trend_train.values[-sub_n:]

    mds = MDS(n_components=2, random_state=42, normalized_stress="auto", n_init=2, max_iter=150)
    X_sub_mds = mds.fit_transform(X_sub)

    isomap = Isomap(n_neighbors=8, n_components=2)
    X_sub_iso = isomap.fit_transform(X_sub)

    label_map = {0: "Bearish (Down)", 1: "Neutral (Flat)", 2: "Bullish (Up)"}
    full_labels = [label_map.get(int(v), "Neutral") for v in y_trend_train.values]
    sub_labels = [label_map.get(int(v), "Neutral") for v in y_sub]

    return {
        "pca_variance_df": exp_var_df,
        "pca_2d_df": pd.DataFrame({"Dim1": X_tr_pca2[:, 0], "Dim2": X_tr_pca2[:, 1], "Market_Regime": full_labels}),
        "lda_2d_df": pd.DataFrame({"Dim1": X_tr_lda[:, 0], "Dim2": X_tr_lda[:, 1], "Market_Regime": full_labels}),
        "ica_2d_df": pd.DataFrame({"Dim1": X_tr_ica[:, 0], "Dim2": X_tr_ica[:, 1], "Market_Regime": full_labels}),
        "fa_2d_df": pd.DataFrame({"Dim1": X_tr_fa[:, 0], "Dim2": X_tr_fa[:, 1], "Market_Regime": full_labels}),
        "fa_loadings_df": fa_loadings,
        "mds_2d_df": pd.DataFrame({"Dim1": X_sub_mds[:, 0], "Dim2": X_sub_mds[:, 1], "Market_Regime": sub_labels}),
        "isomap_2d_df": pd.DataFrame({"Dim1": X_sub_iso[:, 0], "Dim2": X_sub_iso[:, 1], "Market_Regime": sub_labels}),
    }


def compare_before_after_reduction(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    corr_selected: list,
    mrmr_selected: list,
    forward_selected: list,
):
    """
    Evaluates Ridge Regression performance BEFORE vs AFTER dimensionality reduction
    to prove why removing noisy/redundant features improves generalization.
    """
    comparisons = []

    def _eval_subset(label: str, cols: list):
        scaler = StandardScaler()
        t0 = time.perf_counter()
        X_tr = scaler.fit_transform(X_train[cols])
        X_te = scaler.transform(X_test[cols])
        model = Ridge(alpha=1.0)
        model.fit(X_tr, y_train.values)
        pred_tr = model.predict(X_tr)
        pred_te = model.predict(X_te)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        comparisons.append({
            "Feature_Set_Strategy": label,
            "Num_Features": len(cols),
            "Train_RMSE": round(float(np.sqrt(mean_squared_error(y_train.values, pred_tr))), 4),
            "Test_RMSE": round(float(np.sqrt(mean_squared_error(y_test.values, pred_te))), 4),
            "Test_R2": round(float(r2_score(y_test.values, pred_te)), 4),
            "Compute_Time_ms": round(elapsed_ms, 2),
        })

    _eval_subset("1. All Original Features (Unreduced)", list(X_train.columns))
    if corr_selected:
        _eval_subset("2. High Correlation Filtered Subset", corr_selected)
    if forward_selected:
        _eval_subset("3. Forward Feature Selection Subset", forward_selected)
    if mrmr_selected:
        _eval_subset("4. mRMR Selected Subset", mrmr_selected)

    # PCA Reduced (top 5 components)
    scaler = StandardScaler()
    t0 = time.perf_counter()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
    pca = PCA(n_components=min(5, X_train.shape[1]), random_state=42)
    X_tr_p = pca.fit_transform(X_tr_s)
    X_te_p = pca.transform(X_te_s)
    model = Ridge(alpha=1.0)
    model.fit(X_tr_p, y_train.values)
    pred_tr = model.predict(X_tr_p)
    pred_te = model.predict(X_te_p)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    comparisons.append({
        "Feature_Set_Strategy": "5. PCA Compressed (Top 5 Orthogonal Components)",
        "Num_Features": pca.n_components_,
        "Train_RMSE": round(float(np.sqrt(mean_squared_error(y_train.values, pred_tr))), 4),
        "Test_RMSE": round(float(np.sqrt(mean_squared_error(y_test.values, pred_te))), 4),
        "Test_R2": round(float(r2_score(y_test.values, pred_te)), 4),
        "Compute_Time_ms": round(elapsed_ms, 2),
    })

    return pd.DataFrame(comparisons)
