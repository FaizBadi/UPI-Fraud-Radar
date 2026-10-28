"""
neural_net.py - MLPClassifier with three activation functions.

Trains MLPClassifier(hidden_layer_sizes=(64,32)) for activations:
  'relu', 'logistic' (sigmoid), 'tanh'

Outputs:
  models/mlp_relu.pkl
  models/mlp_logistic.pkl
  models/mlp_tanh.pkl
  reports/figures/mlp_loss_curves.png
  reports/mlp_comparison.json
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
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier

from config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR, RANDOM_STATE, FIGURE_DPI

logger = logging.getLogger(__name__)

COLORS = {"relu": "#00E5FF", "logistic": "#FF3B5C", "tanh": "#FFB020"}


def train_mlp(activation: str, X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """Train a single MLP with specified activation."""
    logger.info("  Training MLP (%s) …", activation)
    t0 = time.time()
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation=activation,
        max_iter=200,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=False,
    )
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    logger.info("    Done: %d iterations in %.1f s", model.n_iter_, elapsed)
    path = MODELS_DIR / f"mlp_{activation}.pkl"
    joblib.dump(model, path)
    return model


def plot_loss_curves(mlp_models: dict) -> None:
    """Save a single figure with loss curves for all three activations."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for act, model in mlp_models.items():
        curve = model.loss_curve_
        ax.plot(curve, color=COLORS[act], lw=2, label=f"activation={act}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("MLP Loss Curves by Activation Function")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGURES_DIR / "mlp_loss_curves.png"
    fig.savefig(out, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Loss curves saved → %s", out)


def run_neural_net(data: dict) -> dict:
    """
    Train MLP for relu, logistic, tanh; evaluate; plot loss curves.

    Uses the SMOTE-augmented sample for training (same as LR/RF).
    """
    X_train = data["X_train_smote"]
    y_train = data["y_train_smote"]
    X_test  = data["X_test_sample"]
    y_test  = data["y_test_sample"]

    if len(y_train) > 50_000:
        rng = np.random.default_rng(RANDOM_STATE)
        fraud_idx = np.where(y_train == 1)[0]
        legit_idx = np.where(y_train == 0)[0]
        n_half = 25_000
        samp_f = rng.choice(fraud_idx, size=min(n_half, len(fraud_idx)), replace=False)
        samp_l = rng.choice(legit_idx, size=min(n_half, len(legit_idx)), replace=False)
        sub_idx = np.concatenate([samp_f, samp_l])
        rng.shuffle(sub_idx)
        X_train, y_train = X_train[sub_idx], y_train[sub_idx]

    activations = ["relu", "logistic", "tanh"]
    mlp_models  = {}
    results     = {}

    logger.info("Training MLP classifiers (3 activations) …")
    for act in activations:
        model = train_mlp(act, X_train, y_train)
        mlp_models[act] = model

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        results[act] = {
            "activation" : act,
            "n_iter"     : model.n_iter_,
            "final_loss" : round(float(model.loss_curve_[-1]), 6),
            "precision"  : round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall"     : round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1"         : round(f1_score(y_test, y_pred, zero_division=0), 4),
            "pr_auc"     : round(average_precision_score(y_test, y_proba), 4),
            "roc_auc"    : round(roc_auc_score(y_test, y_proba), 4),
        }
        logger.info(
            "  %-10s  Prec=%.3f  Rec=%.3f  F1=%.3f  PR-AUC=%.3f",
            act, results[act]["precision"], results[act]["recall"],
            results[act]["f1"], results[act]["pr_auc"],
        )

    plot_loss_curves(mlp_models)

    out = REPORTS_DIR / "mlp_comparison.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("MLP comparison saved → %s", out)

    # Save the best MLP (by F1) under a canonical name for the dashboard
    best_act   = max(results, key=lambda a: results[a]["f1"])
    best_mlp   = mlp_models[best_act]
    joblib.dump(best_mlp, MODELS_DIR / "mlp_best.pkl")
    logger.info("Best MLP activation: %s  →  models/mlp_best.pkl", best_act)

    return results
