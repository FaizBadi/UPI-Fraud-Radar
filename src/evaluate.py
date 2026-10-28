"""
evaluate.py - Model evaluation: metrics, CV, bootstrap CIs, Bayes demo.

Outputs:
  reports/metrics.json
  reports/model_comparison.csv
  reports/bayes_demo.json
  models/best_model.pkl  (chosen by recall @ precision >= 0.80)
"""

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.utils import resample

from config import (
    BAYES_JSON,
    BEST_MODEL_PKL,
    FEATURE_LIST_PKL,
    METRICS_JSON,
    MODEL_COMP_CSV,
    MODELS_DIR,
    RANDOM_STATE,
    SCALER_PKL,
    BOOTSTRAP_N,
    CV_FOLDS,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_proba(model, X):
    """Return probability of class 1 (fraud) for any sklearn estimator."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        df = model.decision_function(X)
        # min-max scale to [0,1]
        mn, mx = df.min(), df.max()
        if mx > mn:
            return (df - mn) / (mx - mn)
        return np.zeros(len(df))
    else:
        return model.predict(X).astype(float)


def evaluate_model(name: str, model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Compute all evaluation metrics for one model."""
    if isinstance(model, IsolationForest):
        return {}   # IsolationForest is handled separately

    if name == "KNN" and len(X_test) > 10_000:
        rng = np.random.default_rng(RANDOM_STATE)
        fraud_idx = np.where(y_test == 1)[0]
        legit_idx = np.where(y_test == 0)[0]
        n_fraud = min(len(fraud_idx), 2_000)
        n_legit = min(len(legit_idx), 10_000 - n_fraud)
        samp_fraud = rng.choice(fraud_idx, size=n_fraud, replace=False)
        samp_legit = rng.choice(legit_idx, size=n_legit, replace=False)
        sub_idx = np.concatenate([samp_fraud, samp_legit])
        rng.shuffle(sub_idx)
        X_eval, y_eval = X_test[sub_idx], y_test[sub_idx]
    else:
        X_eval, y_eval = X_test, y_test

    y_pred  = model.predict(X_eval)
    y_proba = _get_proba(model, X_eval)

    prec    = precision_score(y_eval, y_pred, zero_division=0)
    rec     = recall_score(y_eval, y_pred, zero_division=0)
    f1      = f1_score(y_eval, y_pred, zero_division=0)
    pr_auc  = average_precision_score(y_eval, y_proba)
    roc_auc = roc_auc_score(y_eval, y_proba)
    cm      = confusion_matrix(y_eval, y_pred).tolist()

    result = {
        "name"     : name,
        "precision": round(prec, 4),
        "recall"   : round(rec,  4),
        "f1"       : round(f1,   4),
        "pr_auc"   : round(pr_auc,  4),
        "roc_auc"  : round(roc_auc, 4),
        "confusion_matrix": cm,
    }
    logger.info(
        "  %-22s  Prec=%.3f  Rec=%.3f  F1=%.3f  PR-AUC=%.3f  ROC-AUC=%.3f",
        name, prec, rec, f1, pr_auc, roc_auc,
    )
    return result


def cross_validate_model(name: str, model, X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """5-fold StratifiedKFold CV on representative training sample."""
    if isinstance(model, IsolationForest):
        return {}
    logger.info("  CV for %s …", name)
    max_cv = 5_000 if name == "KNN" else 15_000
    if len(y_train) > max_cv:
        rng = np.random.default_rng(RANDOM_STATE)
        fraud_idx = np.where(y_train == 1)[0]
        legit_idx = np.where(y_train == 0)[0]
        n_fraud = min(len(fraud_idx), max_cv // 4)
        n_legit = min(len(legit_idx), max_cv - n_fraud)
        samp_fraud = rng.choice(fraud_idx, size=n_fraud, replace=False)
        samp_legit = rng.choice(legit_idx, size=n_legit, replace=False)
        cv_idx = np.concatenate([samp_fraud, samp_legit])
        rng.shuffle(cv_idx)
        X_cv, y_cv = X_train[cv_idx], y_train[cv_idx]
    else:
        X_cv, y_cv = X_train, y_train

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["precision", "recall", "f1", "average_precision"]
    cv_res = cross_validate(model, X_cv, y_cv, cv=skf,
                            scoring=scoring, n_jobs=-1)
    return {
        f"cv_{k.replace('test_', '')}": round(float(np.mean(v)), 4)
        for k, v in cv_res.items() if k.startswith("test_")
    }


def bootstrap_ci(model, X_test: np.ndarray, y_test: np.ndarray, n: int = BOOTSTRAP_N):
    """Bootstrap 95% CIs for recall and precision of ONE model (vectorized predictions)."""
    recalls, precs = [], []
    rng = np.random.default_rng(RANDOM_STATE)
    yp_all = model.predict(X_test)
    for _ in range(n):
        idx = rng.choice(len(y_test), size=len(y_test), replace=True)
        yb  = y_test[idx]
        if yb.sum() == 0:
            continue
        yp = yp_all[idx]
        recalls.append(recall_score(yb, yp, zero_division=0))
        precs.append(precision_score(yb, yp, zero_division=0))
    def _ci(arr):
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return round(float(lo), 4), round(float(hi), 4)
    return {
        "bootstrap_recall_ci95"   : _ci(recalls),
        "bootstrap_precision_ci95": _ci(precs),
        "bootstrap_recall_mean"   : round(float(np.mean(recalls)), 4),
        "bootstrap_precision_mean": round(float(np.mean(precs)), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Bayes demo
# ─────────────────────────────────────────────────────────────────────────────

def bayes_demo(df_raw: pd.DataFrame, gnb_model, scaler, features: list) -> dict:
    """
    Manual conditional probability computation:
        P(Fraud | isNight, origEmptied, type in [TRANSFER, CASH_OUT])
    vs GaussianNB prediction probability for the same subset.

    df_raw should contain the engineered features (before scaling).
    """
    try:
        # Subset: night, origEmptied, TRANSFER or CASH_OUT
        mask = (
            (df_raw.get("isNight", pd.Series(0, index=df_raw.index)) == 1) &
            (df_raw.get("origEmptied", pd.Series(0, index=df_raw.index)) == 1) &
            (
                (df_raw.get("type_TRANSFER", pd.Series(0, index=df_raw.index)) == 1) |
                (df_raw.get("type_CASH_OUT", pd.Series(0, index=df_raw.index)) == 1)
            )
        )
        subset = df_raw[mask]
        if len(subset) == 0:
            logger.warning("Bayes demo: no rows matching conditions.")
            return {}

        # Manual Bayes: count-based estimate
        p_fraud_given_conditions = subset["isFraud"].mean() if "isFraud" in subset else None

        # GaussianNB prediction on the same subset
        feat_cols = [f for f in features if f in subset.columns]
        X_sub     = scaler.transform(subset[feat_cols].values) if feat_cols else None
        gnb_prob  = float(gnb_model.predict_proba(X_sub)[:, 1].mean()) if X_sub is not None else None

        result = {
            "conditions"               : "isNight=1 AND origEmptied=1 AND type in [TRANSFER, CASH_OUT]",
            "n_matching_rows"          : int(len(subset)),
            "manual_P_fraud"           : round(float(p_fraud_given_conditions), 4),
            "gnb_mean_predicted_prob"  : round(gnb_prob, 4) if gnb_prob is not None else None,
            "explanation": (
                "Manual P(Fraud | conditions) = count(fraud & conditions) / count(conditions). "
                "GaussianNB models P(X|class) as Gaussian and applies Bayes theorem to estimate "
                "P(class|X). Both numbers should be close if the Gaussian assumption holds."
            ),
        }
        BAYES_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(BAYES_JSON, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Bayes demo saved: manual=%.4f  GNB=%.4f",
                    result["manual_P_fraud"], result["gnb_mean_predicted_prob"] or 0)
        return result
    except Exception as e:
        logger.error("Bayes demo failed: %s", e)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(models: dict, data: dict, df_engineered: pd.DataFrame) -> dict:
    """
    Evaluate every model, run CV, compute bootstrap CIs for best model,
    select best model, save all reports.

    Selection criterion: highest recall where precision >= 0.80.
    Fallback: best F1 if no model reaches precision 0.80.
    """
    X_test   = data["X_test_sample"]
    y_test   = data["y_test_sample"]
    X_train  = data["X_train_sample"]
    y_train  = data["y_train_sample"]

    all_metrics = {}
    rows = []

    logger.info("=" * 60)
    logger.info("MODEL EVALUATION  (test sample n=%d, fraud=%d)", len(y_test), y_test.sum())
    logger.info("=" * 60)

    for name, model in models.items():
        if isinstance(model, IsolationForest):
            continue
        m = evaluate_model(name, model, X_test, y_test)
        if not m:
            continue
        cv = cross_validate_model(name, model, X_train, y_train)
        m.update(cv)
        all_metrics[name] = m
        rows.append(m)

    # ── Select best model ─────────────────────────────────────────────────────
    comp_df = pd.DataFrame(rows)
    # Filter by precision >= 0.80
    eligible = comp_df[comp_df["precision"] >= 0.80]
    if len(eligible) > 0:
        best_row = eligible.loc[eligible["recall"].idxmax()]
        selection_method = "highest recall @ precision >= 0.80"
    else:
        best_row = comp_df.loc[comp_df["f1"].idxmax()]
        selection_method = "best F1 (no model reached precision >= 0.80)"

    best_name  = best_row["name"]
    best_model = models[best_name]
    logger.info("\nBest model: %s (%s)", best_name, selection_method)

    # ── Bootstrap CIs for best model ─────────────────────────────────────────
    logger.info("Bootstrap CIs for %s (n=%d) …", best_name, BOOTSTRAP_N)
    ci = bootstrap_ci(best_model, X_test, y_test, n=BOOTSTRAP_N)
    all_metrics[best_name].update(ci)
    all_metrics[best_name]["is_best"] = True
    logger.info(
        "  Recall 95%% CI: %s   Precision 95%% CI: %s",
        ci["bootstrap_recall_ci95"], ci["bootstrap_precision_ci95"]
    )

    # ── Save best model ───────────────────────────────────────────────────────
    scaler   = data["scaler"]
    features = data["features"]
    joblib.dump(best_model, BEST_MODEL_PKL)
    joblib.dump(scaler,     SCALER_PKL)
    joblib.dump(features,   FEATURE_LIST_PKL)
    logger.info("Best model saved -> %s", BEST_MODEL_PKL)

    # ── Save reports ──────────────────────────────────────────────────────────
    from config import REPORTS_DIR as _REPORTS_DIR
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(METRICS_JSON, "w") as f:
        json.dump({"best_model": best_name,
                   "selection_method": selection_method,
                   "models": all_metrics}, f, indent=2)

    comp_df.to_csv(MODEL_COMP_CSV, index=False)
    logger.info("Reports saved: %s  |  %s", METRICS_JSON, MODEL_COMP_CSV)

    # ── Bayes demo ────────────────────────────────────────────────────────────
    if "GaussianNB" in models:
        bayes_demo(
            df_engineered,
            models["GaussianNB"],
            scaler,
            features,
        )

    return all_metrics
