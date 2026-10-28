"""
features.py - Feature selection: SelectKBest and RandomForest importance.

Produces reports/feature_ranking.csv with columns:
    feature, skb_score, rf_importance, mean_rank
"""

import logging
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

from config import FEATURE_RANK_CSV, RANDOM_STATE

logger = logging.getLogger(__name__)


def select_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list,
    k: int = 15,
) -> pd.DataFrame:
    """
    Rank features by SelectKBest (f_classif) and RandomForest importance.

    Parameters
    ----------
    X_train : np.ndarray  (already scaled)
    y_train : np.ndarray
    feature_names : list of str
    k : int  - number of top features to highlight

    Returns
    -------
    pd.DataFrame with feature rankings, also saved to FEATURE_RANK_CSV.
    """
    t0 = time.time()
    logger.info("Running feature selection (SelectKBest + RF importance) …")

    # ── SelectKBest with f_classif ────────────────────────────────────────────
    skb = SelectKBest(score_func=f_classif, k="all")
    skb.fit(X_train, y_train)
    skb_scores = skb.scores_

    # ── Mutual information as a cross-check ───────────────────────────────────
    if len(X_train) > 20_000:
        rng_mi = np.random.default_rng(RANDOM_STATE)
        mi_idx = rng_mi.choice(len(X_train), size=20_000, replace=False)
        mi_scores = mutual_info_classif(X_train[mi_idx], y_train[mi_idx], random_state=RANDOM_STATE)
    else:
        mi_scores = mutual_info_classif(X_train, y_train, random_state=RANDOM_STATE)

    # ── RandomForest importance ───────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    rf_imp = rf.feature_importances_

    # ── Combine and rank ──────────────────────────────────────────────────────
    ranking = pd.DataFrame({
        "feature"       : feature_names,
        "skb_score"     : skb_scores,
        "mi_score"      : mi_scores,
        "rf_importance" : rf_imp,
    })

    # Rank each metric (1 = best) and compute mean rank
    for col in ["skb_score", "mi_score", "rf_importance"]:
        ranking[f"{col}_rank"] = ranking[col].rank(ascending=False).astype(int)

    ranking["mean_rank"] = (
        ranking["skb_score_rank"]
        + ranking["mi_score_rank"]
        + ranking["rf_importance_rank"]
    ) / 3
    ranking = ranking.sort_values("mean_rank").reset_index(drop=True)
    ranking["overall_rank"] = ranking.index + 1

    FEATURE_RANK_CSV.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(FEATURE_RANK_CSV, index=False)
    logger.info(
        "Feature ranking saved to %s  (%.1f s)", FEATURE_RANK_CSV, time.time() - t0
    )
    logger.info("Top %d features:\n%s", k, ranking[["feature", "mean_rank"]].head(k).to_string())

    return ranking
