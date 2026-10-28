"""
cluster.py - Clustering: KMeans, KMedoids, Agglomerative.

Outputs:
  models/cluster_kmeans.pkl
  models/cluster_kmedoids.pkl
  reports/cluster_profiles.json
  reports/figures/cluster_elbow.png
  reports/figures/cluster_silhouette.png
  reports/figures/cluster_dendrogram.png
  reports/figures/cluster_pca3d_*.html  (Plotly, read by dashboard)
  data/processed/cluster_labels.csv
"""

import json
import logging
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

try:
    from sklearn_extra.cluster import KMedoids
    HAS_KMEDOIDS = True
except ImportError:
    HAS_KMEDOIDS = False
    logging.getLogger(__name__).warning(
        "scikit-learn-extra not available; KMedoids will be skipped."
    )

from config import (
    CLUSTER_K_RANGE,
    DATA_PROCESSED_DIR,
    FIGURE_DPI,
    FIGURES_DIR,
    KMEDOIDS_PKL,
    CLUSTER_PKL,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
)

logger = logging.getLogger(__name__)

CLUSTER_LABELS_CSV = DATA_PROCESSED_DIR / "cluster_labels.csv"
CLUSTER_PCA_CSV    = DATA_PROCESSED_DIR / "cluster_pca.csv"


def _risk_label(fraud_rate: float) -> str:
    if fraud_rate >= 0.10:
        return "High Risk"
    elif fraud_rate >= 0.03:
        return "Medium Risk"
    else:
        return "Low Risk"


def choose_k_elbow_silhouette(X: np.ndarray) -> int:
    """Elbow + silhouette to choose optimal k."""
    inertias, silhouettes = [], []
    ks = list(CLUSTER_K_RANGE)
    for k in ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        if k > 1:
            silhouettes.append(silhouette_score(X, labels, sample_size=5000,
                                                random_state=RANDOM_STATE))
        else:
            silhouettes.append(0.0)

    # Plot elbow
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ks, inertias, "bo-")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia")
    axes[0].set_title("KMeans Elbow Curve")

    axes[1].plot(ks, silhouettes, "rs-")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("KMeans Silhouette Scores")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cluster_elbow.png", dpi=FIGURE_DPI)
    plt.close(fig)

    # Best k: highest silhouette
    best_k = ks[int(np.argmax(silhouettes))]
    logger.info("Optimal k by silhouette: %d", best_k)
    return best_k


def run_kmeans(X: np.ndarray, k: int) -> KMeans:
    logger.info("KMeans (k=%d) …", k)
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    km.fit(X)
    joblib.dump(km, CLUSTER_PKL)
    return km


def run_kmedoids(X: np.ndarray, k: int):
    if not HAS_KMEDOIDS:
        return None
    logger.info("KMedoids (k=%d) …", k)
    km = KMedoids(n_clusters=k, random_state=RANDOM_STATE)
    km.fit(X)
    joblib.dump(km, KMEDOIDS_PKL)
    return km


def run_agglomerative(X_small: np.ndarray) -> np.ndarray:
    """Agglomerative clustering + dendrogram on a small subset."""
    logger.info("Agglomerative clustering (n=%d) …", len(X_small))
    k = min(5, len(X_small) - 1)
    agg = AgglomerativeClustering(n_clusters=k)
    labels = agg.fit_predict(X_small)

    # Dendrogram on up to 200 samples
    n_dendro = min(200, len(X_small))
    Z = linkage(X_small[:n_dendro], method="ward")
    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(Z, ax=ax, no_labels=True, color_threshold=0.7 * max(Z[:, 2]))
    ax.set_title("Hierarchical Clustering Dendrogram (ward linkage)")
    ax.set_xlabel("Samples"); ax.set_ylabel("Distance")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cluster_dendrogram.png", dpi=FIGURE_DPI)
    plt.close(fig)
    return labels


def profile_clusters(df_samp: pd.DataFrame, labels: np.ndarray, algo: str) -> dict:
    """Compute fraud rate, mean amount, night share per cluster."""
    df_samp = df_samp.copy()
    df_samp["cluster"] = labels
    profiles = {}
    for c in sorted(df_samp["cluster"].unique()):
        grp = df_samp[df_samp["cluster"] == c]
        fr  = grp["isFraud"].mean() if "isFraud" in grp else 0.0
        am  = grp["amount"].mean()  if "amount" in grp else 0.0
        ns  = grp.get("isNight", pd.Series(0, index=grp.index)).mean()
        profiles[int(c)] = {
            "cluster"    : int(c),
            "n"          : int(len(grp)),
            "fraud_rate" : round(float(fr), 4),
            "mean_amount": round(float(am), 2),
            "night_share": round(float(ns), 4),
            "risk_label" : _risk_label(float(fr)),
        }
        logger.info("  Cluster %d: n=%d  fraud=%.2f%%  mean_amount=%.0f  risk=%s",
                    c, len(grp), fr * 100, am, _risk_label(float(fr)))
    return profiles


def run_clustering(
    df_engineered: pd.DataFrame,
    X_scaled: np.ndarray = None,
    scaler = None,
    features: list = None,
) -> None:
    """
    Full clustering pipeline.

    Uses up to 50k samples (all fraud + sample of legit) to stay fast.
    Saves cluster labels and PCA-reduced coordinates for dashboard.
    """
    t0 = time.time()

    # Reset index to ensure positional indexing works
    df_engineered = df_engineered.reset_index(drop=True)

    # Sample for clustering
    rng = np.random.default_rng(RANDOM_STATE)
    fraud_idx = np.where(df_engineered["isFraud"].values == 1)[0]
    legit_idx = np.where(df_engineered["isFraud"].values == 0)[0]
    max_legit = 20_000
    if len(legit_idx) > max_legit:
        legit_idx = rng.choice(legit_idx, max_legit, replace=False)
    idx = np.concatenate([fraud_idx, legit_idx])
    rng.shuffle(idx)

    df_cl = df_engineered.iloc[idx].reset_index(drop=True)
    if scaler is not None and features is not None:
        valid_cols = [c for c in features if c in df_cl.columns]
        X_cl = scaler.transform(df_cl[valid_cols].values)
    elif X_scaled is not None:
        X_cl = X_scaled[idx]
    else:
        raise ValueError("Either (scaler, features) or X_scaled must be provided to run_clustering.")

    # ── Choose k ──────────────────────────────────────────────────────────────
    best_k = choose_k_elbow_silhouette(X_cl)

    # ── KMeans ────────────────────────────────────────────────────────────────
    km_model  = run_kmeans(X_cl, best_k)
    km_labels = km_model.labels_

    # ── KMedoids ──────────────────────────────────────────────────────────────
    kmed_labels = None
    if HAS_KMEDOIDS:
        kmed_model  = run_kmedoids(X_cl, best_k)
        kmed_labels = kmed_model.labels_

    # ── Agglomerative (small subset) ──────────────────────────────────────────
    agg_n = min(2000, len(X_cl))
    agg_labels = run_agglomerative(X_cl[:agg_n])

    # ── Profiles ──────────────────────────────────────────────────────────────
    km_profiles = profile_clusters(df_cl, km_labels, "kmeans")
    profiles    = {"kmeans": km_profiles}
    if kmed_labels is not None:
        kmed_profiles = profile_clusters(df_cl, kmed_labels, "kmedoids")
        profiles["kmedoids"] = kmed_profiles

    out_prof = REPORTS_DIR / "cluster_profiles.json"
    with open(out_prof, "w") as f:
        json.dump(profiles, f, indent=2)
    logger.info("Cluster profiles → %s", out_prof)

    # ── PCA for 3D dashboard scatter ──────────────────────────────────────────
    pca   = PCA(n_components=3, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_cl)
    pca_df = pd.DataFrame(X_pca, columns=["pc1", "pc2", "pc3"])
    pca_df["kmeans_label"]  = km_labels
    pca_df["kmedoids_label"] = kmed_labels if kmed_labels is not None else km_labels
    pca_df["isFraud"] = df_cl["isFraud"].values
    pca_df["amount"]  = df_cl["amount"].values if "amount" in df_cl else 0
    CLUSTER_PCA_CSV.parent.mkdir(parents=True, exist_ok=True)
    pca_df.to_csv(CLUSTER_PCA_CSV, index=False)

    # Save cluster labels aligned to original sample indices
    label_df = pd.DataFrame({"orig_idx": idx, "kmeans_label": km_labels})
    if kmed_labels is not None:
        label_df["kmedoids_label"] = kmed_labels
    label_df.to_csv(CLUSTER_LABELS_CSV, index=False)

    logger.info("Clustering done in %.1f s. Labels → %s", time.time() - t0, CLUSTER_LABELS_CSV)
