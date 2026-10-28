"""
train.py - Train all classifiers specified in the brief.

Models trained:
  - LogisticRegression   (with class_weight; MLE explained in comment)
  - DecisionTreeClassifier
  - RandomForestClassifier  ← saved as best_model.pkl if it wins
  - KNeighborsClassifier    (on sample, not full data)
  - LinearSVC + CalibratedClassifierCV  (fast SVM with probabilities)
  - GaussianNB
  - IsolationForest         (anomaly score)

All models saved to models/<name>.pkl.
"""

import logging
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from config import MODELS_DIR, RANDOM_STATE

logger = logging.getLogger(__name__)

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _save(model, name: str):
    path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(model, path)
    logger.info("  Saved %s -> %s", name, path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# LOGISTIC REGRESSION
# MLE explanation: LogisticRegression maximises the log-likelihood
#   ℓ(β) = Σ [ y_i log σ(β·x_i) + (1-y_i) log(1-σ(β·x_i)) ]
# where σ is the sigmoid function.  scikit-learn minimises the negative
# log-likelihood (cross-entropy) with an L2 penalty (default C=1).
# class_weight='balanced' scales each sample's contribution so that
# the minority class (fraud) receives proportionally more weight,
# countering the effect of class imbalance.
# ─────────────────────────────────────────────────────────────────────────────

def train_logistic(X_train, y_train):
    logger.info("Training LogisticRegression …")
    t0 = time.time()
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="lbfgs",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    _save(model, "logistic_regression")
    logger.info("  Done in %.1f s", time.time() - t0)
    return model


def train_decision_tree(X_train, y_train):
    logger.info("Training DecisionTreeClassifier …")
    t0 = time.time()
    model = DecisionTreeClassifier(
        max_depth=12,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    _save(model, "decision_tree")
    logger.info("  Done in %.1f s", time.time() - t0)
    return model


def train_random_forest(X_train, y_train):
    logger.info("Training RandomForestClassifier (200 trees) …")
    t0 = time.time()
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    _save(model, "random_forest")
    logger.info("  Done in %.1f s", time.time() - t0)
    return model


def train_knn(X_train, y_train):
    """KNN on the working sample (subsampled to 20k rows for tractable O(N*d) inference)."""
    logger.info("Training KNeighborsClassifier …")
    t0 = time.time()
    max_knn = 20_000
    if len(y_train) > max_knn:
        fraud_idx = np.where(y_train == 1)[0]
        legit_idx = np.where(y_train == 0)[0]
        rng = np.random.default_rng(RANDOM_STATE)
        n_fraud = min(len(fraud_idx), max_knn // 4)
        n_legit = min(len(legit_idx), max_knn - n_fraud)
        samp_fraud = rng.choice(fraud_idx, size=n_fraud, replace=False)
        samp_legit = rng.choice(legit_idx, size=n_legit, replace=False)
        sub_idx = np.concatenate([samp_fraud, samp_legit])
        rng.shuffle(sub_idx)
        X_train, y_train = X_train[sub_idx], y_train[sub_idx]
    model = KNeighborsClassifier(n_neighbors=7, n_jobs=-1)
    model.fit(X_train, y_train)
    _save(model, "knn")
    logger.info("  Done in %.1f s", time.time() - t0)
    return model


def train_svm(X_train, y_train):
    """
    LinearSVC + CalibratedClassifierCV so we get predict_proba.
    LinearSVC is much faster than kernel SVC on large feature sets.
    """
    logger.info("Training SVM (LinearSVC + CalibratedClassifierCV) …")
    t0 = time.time()
    base = LinearSVC(
        class_weight="balanced",
        max_iter=3000,
        random_state=RANDOM_STATE,
    )
    model = CalibratedClassifierCV(base, cv=3)
    model.fit(X_train, y_train)
    _save(model, "svm")
    logger.info("  Done in %.1f s", time.time() - t0)
    return model


def train_naive_bayes(X_train, y_train):
    logger.info("Training GaussianNB …")
    t0 = time.time()
    model = GaussianNB()
    model.fit(X_train, y_train)
    _save(model, "naive_bayes")
    logger.info("  Done in %.1f s", time.time() - t0)
    return model


def train_isolation_forest(X_train):
    """IsolationForest for anomaly scoring (unsupervised)."""
    logger.info("Training IsolationForest …")
    t0 = time.time()
    model = IsolationForest(
        n_estimators=100,
        contamination=0.005,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train)
    _save(model, "iso_forest")
    logger.info("  Done in %.1f s", time.time() - t0)
    return model


def train_all(data: dict) -> dict:
    """
    Train every classifier and return a dict of {name: model}.

    Uses SMOTE-augmented sample for: LR, DT, RF (these are fast enough).
    Uses plain sample for: KNN, SVM, NB (to respect MAX_LEGIT_TRAIN budget).
    IsolationForest trains on the plain sample (unlabelled, unsupervised).
    """
    X_smote   = data["X_train_smote"]
    y_smote   = data["y_train_smote"]
    X_samp    = data["X_train_sample"]
    y_samp    = data["y_train_sample"]

    models = {}
    models["LogisticRegression"]  = train_logistic(X_smote, y_smote)
    models["DecisionTree"]         = train_decision_tree(X_smote, y_smote)
    models["RandomForest"]         = train_random_forest(X_smote, y_smote)
    models["KNN"]                  = train_knn(X_samp, y_samp)
    models["SVM"]                  = train_svm(X_samp, y_samp)
    models["GaussianNB"]           = train_naive_bayes(X_smote, y_smote)
    models["IsolationForest"]      = train_isolation_forest(X_samp)

    return models
