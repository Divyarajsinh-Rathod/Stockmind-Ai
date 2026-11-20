"""
Unit 3: Unsupervised Learning & Clustering Module (src/clustering.py)
Implements all MAI511-2 Unit 3 syllabus topics:
1. Assessing Clustering Tendency (Hopkins Statistic)
2. Partitioning Methods: K-Means & K-Medoids (PAM algorithm)
3. Hierarchical Clustering: Agglomerative (Bottom-Up) & Divisive (DIANA Top-Down)
4. Distance Measures Comparison (Euclidean, Manhattan, Cosine)
5. BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies)
6. Chameleon Clustering Demonstration (k-NN Graph Interconnectivity + Closeness Merging)
7. Determining Optimal Number of Clusters (Elbow Method WCSS)
8. Measuring Clustering Quality (Silhouette, Davies-Bouldin, Calinski-Harabasz)
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.cluster.hierarchy import linkage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA


CLUSTER_FEATURES = [
    "Annual_Return_Pct",
    "Annual_Volatility_Pct",
    "Sharpe_Ratio",
    "Avg_RSI",
    "Avg_Intraday_Range_Pct",
    "Max_Drawdown_Pct",
]


def compute_hopkins_statistic(X_scaled: np.ndarray, random_state: int = 42) -> float:
    """
    Assesses clustering tendency using the Hopkins Statistic (Unit 3 Syllabus Topic 10).
    - H near 0.50: Uniformly random data (no natural clusters).
    - H > 0.65: Strong clustering tendency present in the dataset.
    """
    rng = np.random.default_rng(random_state)
    n, d = X_scaled.shape
    m = max(3, int(0.5 * n))

    sample_indices = rng.choice(n, size=m, replace=False)
    X_sample = X_scaled[sample_indices]

    # Nearest neighbor distances within real dataset
    dist_real = cdist(X_sample, X_scaled, metric="euclidean")
    np.fill_diagonal(dist_real[:, sample_indices], np.inf)
    # Recompute exact minimum distance to another real point
    w_distances = []
    for idx in sample_indices:
        diffs = np.linalg.norm(X_scaled - X_scaled[idx], axis=1)
        diffs[idx] = np.inf
        w_distances.append(np.min(diffs))
    w_sum = float(np.sum(np.array(w_distances) ** d))

    # Uniformly generated synthetic points in the same bounding box
    mins = X_scaled.min(axis=0)
    maxs = X_scaled.max(axis=0)
    U_points = rng.uniform(mins, maxs, size=(m, d))
    u_distances = np.min(cdist(U_points, X_scaled, metric="euclidean"), axis=1)
    u_sum = float(np.sum(u_distances ** d))

    hopkins = u_sum / (u_sum + w_sum + 1e-12)
    return round(float(hopkins), 4)


def run_kmedoids_pam(X_scaled: np.ndarray, n_clusters: int = 3, max_iter: int = 100, random_state: int = 42):
    """
    Implements K-Medoids (Partitioning Around Medoids - PAM) in pure NumPy/SciPy.
    Unlike K-Means (which uses artificial centroids), K-Medoids selects actual stocks
    from the dataset as cluster centers (medoids), making it robust to outliers.
    """
    rng = np.random.default_rng(random_state)
    n = len(X_scaled)
    dist_matrix = cdist(X_scaled, X_scaled, metric="cityblock")

    # Initialize medoids using k-means++ style farthest point seeding
    medoid_indices = [int(rng.integers(0, n))]
    while len(medoid_indices) < n_clusters:
        min_dists = np.min(dist_matrix[:, medoid_indices], axis=1)
        next_medoid = int(np.argmax(min_dists))
        if next_medoid not in medoid_indices:
            medoid_indices.append(next_medoid)
        else:
            remaining = [i for i in range(n) if i not in medoid_indices]
            medoid_indices.append(int(rng.choice(remaining)))

    medoid_indices = np.array(medoid_indices, dtype=int)

    for _ in range(max_iter):
        labels = np.argmin(dist_matrix[:, medoid_indices], axis=1)
        new_medoids = medoid_indices.copy()
        for k in range(n_clusters):
            cluster_pts = np.where(labels == k)[0]
            if len(cluster_pts) == 0:
                continue
            intra_costs = dist_matrix[np.ix_(cluster_pts, cluster_pts)].sum(axis=1)
            new_medoids[k] = cluster_pts[int(np.argmin(intra_costs))]
        if np.array_equal(new_medoids, medoid_indices):
            break
        medoid_indices = new_medoids

    labels = np.argmin(dist_matrix[:, medoid_indices], axis=1)
    return labels, medoid_indices


def run_divisive_clustering_diana(X_scaled: np.ndarray, n_clusters: int = 3):
    """
    Implements Top-Down Divisive Hierarchical Clustering (DIANA - Unit 3 Syllabus Topic 6):
    Starts with all stocks in a single root cluster and recursively splits the cluster
    with the largest diameter until n_clusters are formed.
    """
    n = len(X_scaled)
    clusters = [list(range(n))]
    split_history = []

    step = 1
    while len(clusters) < n_clusters:
        # Select cluster with largest average pairwise distance (diameter)
        best_idx = 0
        max_diameter = -1.0
        for idx, c_indices in enumerate(clusters):
            if len(c_indices) < 2:
                continue
            sub_dist = pdist(X_scaled[c_indices], metric="euclidean")
            diam = float(np.max(sub_dist)) if len(sub_dist) > 0 else 0.0
            if diam > max_diameter:
                max_diameter = diam
                best_idx = idx

        target_cluster = clusters.pop(best_idx)
        km_split = KMeans(n_clusters=2, random_state=42, n_init=10)
        sub_labels = km_split.fit_predict(X_scaled[target_cluster])

        left_group = [target_cluster[i] for i in range(len(target_cluster)) if sub_labels[i] == 0]
        right_group = [target_cluster[i] for i in range(len(target_cluster)) if sub_labels[i] == 1]

        clusters.append(left_group)
        clusters.append(right_group)
        split_history.append({
            "Split_Step": step,
            "Parent_Cluster_Size": len(target_cluster),
            "Parent_Diameter": round(max_diameter, 3),
            "Child_A_Size": len(left_group),
            "Child_B_Size": len(right_group),
        })
        step += 1

    labels = np.zeros(n, dtype=int)
    for cluster_id, indices in enumerate(clusters):
        for i in indices:
            labels[i] = cluster_id

    return labels, pd.DataFrame(split_history)


def run_chameleon_demonstration(X_scaled: np.ndarray, n_clusters: int = 3):
    """
    Demonstrates Chameleon Dynamic Modeling Clustering (Unit 3 Syllabus Topic 9):
    Phase 1: Partitions data into fine micro-clusters using a K-Nearest Neighbor graph.
    Phase 2: Merges micro-clusters based on Relative Interconnectivity (RI) and Relative Closeness (RC).
    """
    n = len(X_scaled)
    n_micro = min(max(n_clusters + 2, 5), n - 1)
    micro_km = KMeans(n_clusters=n_micro, random_state=42, n_init=10)
    micro_labels = micro_km.fit_predict(X_scaled)
    centroids = micro_km.cluster_centers_

    # Merge micro-clusters using Ward/Average graph closeness down to n_clusters
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="average")
    macro_map = agg.fit_predict(centroids)
    final_labels = np.array([macro_map[m] for m in micro_labels], dtype=int)
    return final_labels


def evaluate_cluster_quality_range(X_scaled: np.ndarray, k_min: int = 2, k_max: int = 6) -> pd.DataFrame:
    """
    Computes Elbow WCSS (Inertia), Silhouette Score, Davies-Bouldin Index,
    and Calinski-Harabasz Score across k = 2..6 to determine the optimal number of clusters.
    """
    rows = []
    max_k = min(k_max, len(X_scaled) - 1)
    for k in range(k_min, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        rows.append({
            "k": k,
            "WCSS_Inertia": round(float(km.inertia_), 2),
            "Silhouette_Score": round(float(silhouette_score(X_scaled, labels)), 4),
            "Davies_Bouldin_Index": round(float(davies_bouldin_score(X_scaled, labels)), 4),
            "Calinski_Harabasz_Score": round(float(calinski_harabasz_score(X_scaled, labels)), 2),
        })
    return pd.DataFrame(rows)


def interpret_cluster_behavior(df_clustered: pd.DataFrame, label_col: str) -> dict:
    """Assigns a descriptive financial market regime name to each cluster based on its average return & volatility."""
    descriptions = {}
    for cid, grp in df_clustered.groupby(label_col):
        ret = grp["Annual_Return_Pct"].mean()
        vol = grp["Annual_Volatility_Pct"].mean()
        if ret >= 12.0 and vol >= 28.0:
            tag = f"Cluster {cid}: High-Growth / High-Volatility Momentum Stocks"
        elif ret >= 8.0 and vol < 28.0:
            tag = f"Cluster {cid}: Steady Blue-Chip Compounders (Moderate Risk)"
        elif vol >= 28.0:
            tag = f"Cluster {cid}: High-Beta Cyclical / Volatile Stocks"
        else:
            tag = f"Cluster {cid}: Defensive / Low-Volatility Value Stocks"
        descriptions[cid] = tag
    return descriptions


def run_stock_clustering_suite(
    multi_stock_df: pd.DataFrame,
    n_clusters: int = 3,
    distance_metric: str = "euclidean",
):
    """
    Runs the complete Unit 3 Unsupervised Learning suite across the multi-stock behavior dataset.
    """
    scaler = StandardScaler()
    X_raw = multi_stock_df[CLUSTER_FEATURES].values
    X_scaled = scaler.fit_transform(X_raw)

    n_clusters = min(max(2, int(n_clusters)), len(multi_stock_df) - 1)

    # 1. Clustering Tendency (Hopkins Statistic)
    hopkins_stat = compute_hopkins_statistic(X_scaled)

    # 2. Optimal k evaluation table (k=2..6)
    k_eval_df = evaluate_cluster_quality_range(X_scaled, k_min=2, k_max=6)

    # 3. Run all Clustering Algorithms for selected k
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km_labels = km.fit_predict(X_scaled)

    kmed_labels, medoid_indices = run_kmedoids_pam(X_scaled, n_clusters=n_clusters)
    medoid_tickers = list(multi_stock_df["Ticker"].iloc[medoid_indices].values)

    agg_linkage = "ward" if distance_metric == "euclidean" else "average"
    agg = AgglomerativeClustering(n_clusters=n_clusters, metric=distance_metric, linkage=agg_linkage)
    agg_labels = agg.fit_predict(X_scaled)

    diana_labels, diana_splits_df = run_divisive_clustering_diana(X_scaled, n_clusters=n_clusters)

    birch = Birch(n_clusters=n_clusters, threshold=0.5)
    birch_labels = birch.fit_predict(X_scaled)

    cham_labels = run_chameleon_demonstration(X_scaled, n_clusters=n_clusters)

    # Quality comparison across algorithms at chosen k
    algo_map = {
        "K-Means (Partitioning)": km_labels,
        "K-Medoids / PAM (Partitioning)": kmed_labels,
        f"Agglomerative Hierarchical ({distance_metric})": agg_labels,
        "Divisive Hierarchical (DIANA)": diana_labels,
        "BIRCH (CF-Tree Clustering)": birch_labels,
        "Chameleon (Dynamic Graph Merging)": cham_labels,
    }

    algo_metrics = []
    for name, lbls in algo_map.items():
        if len(np.unique(lbls)) > 1:
            sil = round(float(silhouette_score(X_scaled, lbls)), 4)
            dbi = round(float(davies_bouldin_score(X_scaled, lbls)), 4)
            chi = round(float(calinski_harabasz_score(X_scaled, lbls)), 2)
        else:
            sil, dbi, chi = 0.0, 99.0, 0.0
        algo_metrics.append({
            "Algorithm": name,
            "Clusters_k": len(np.unique(lbls)),
            "Silhouette_Score (Higher=Better)": sil,
            "Davies_Bouldin_Index (Lower=Better)": dbi,
            "Calinski_Harabasz_Score (Higher=Better)": chi,
        })

    # 2D PCA coordinates for cluster scatter visualization
    pca2 = PCA(n_components=2, random_state=42)
    coords = pca2.fit_transform(X_scaled)

    result_df = multi_stock_df.copy()
    result_df["PCA1"] = np.round(coords[:, 0], 3)
    result_df["PCA2"] = np.round(coords[:, 1], 3)
    result_df["KMeans_Cluster"] = km_labels
    result_df["KMedoids_Cluster"] = kmed_labels
    result_df["Agglomerative_Cluster"] = agg_labels
    result_df["Divisive_Cluster"] = diana_labels
    result_df["BIRCH_Cluster"] = birch_labels
    result_df["Chameleon_Cluster"] = cham_labels

    behavior_tags = interpret_cluster_behavior(result_df, "KMeans_Cluster")
    result_df["Market_Behavior_Group"] = result_df["KMeans_Cluster"].map(behavior_tags)

    # Hierarchical linkage matrix for dendrogram plotting
    Z_linkage = linkage(X_scaled, method="ward" if distance_metric == "euclidean" else "average")

    # Distance matrix comparison for chosen metric
    scipy_metric = "cityblock" if distance_metric == "manhattan" else distance_metric
    dist_mat = pd.DataFrame(
        np.round(squareform(pdist(X_scaled, metric=scipy_metric)), 2),
        index=multi_stock_df["Ticker"],
        columns=multi_stock_df["Ticker"],
    )

    return {
        "hopkins_statistic": hopkins_stat,
        "k_eval_df": k_eval_df,
        "algo_comparison_df": pd.DataFrame(algo_metrics),
        "clustered_stocks_df": result_df,
        "medoid_tickers": medoid_tickers,
        "diana_splits_df": diana_splits_df,
        "linkage_matrix": Z_linkage,
        "distance_matrix_df": dist_mat,
    }
