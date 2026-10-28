"""
preprocess.py - Data cleaning, outlier reporting, feature engineering,
                train/test split, scaling, and SMOTE oversampling.

Pipeline steps (run in order by run_all.py):
  1. clean(df)            → cleaned df
  2. engineer_features(df)→ df with new columns
  3. split_and_scale(df)  → X_train, X_test, y_train, y_test, scaler, features
"""

import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_PROCESSED_DIR,
    DEMO_FLAG_FILE,
    FEATURE_LIST_PKL,
    MAX_LEGIT_TEST,
    MAX_LEGIT_TRAIN,
    PROCESSED_CSV,
    RANDOM_STATE,
    SCALER_PKL,
    TEST_SIZE,
    TRANSACTION_TYPES,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic data quality steps:
    - Report missing values (do NOT impute; PaySim has none, but we handle the
      general case by forward-filling numeric columns and flagging).
    - Drop exact duplicates.
    - Verify dtypes and coerce if needed.
    - Compute and REPORT (but do NOT remove) outliers by IQR and Z-score for
      `amount`.  Large amounts are often fraud, so removing them would lose
      signal.

    Returns cleaned DataFrame.
    """
    t0 = time.time()
    logger.info("Cleaning data: %d rows …", len(df))

    # ── Missing values ────────────────────────────────────────────────────────
    missing = df.isnull().sum()
    if missing.any():
        logger.warning("Missing values found:\n%s", missing[missing > 0])
        # Fill numeric with median (keep non-numeric as-is)
        num_cols = df.select_dtypes(include="number").columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    else:
        logger.info("No missing values.")

    # ── Duplicates ────────────────────────────────────────────────────────────
    n_dup = df.duplicated().sum()
    if n_dup:
        logger.warning("Dropping %d duplicate rows.", n_dup)
        df = df.drop_duplicates().reset_index(drop=True)

    # ── Dtype coercion ────────────────────────────────────────────────────────
    df["step"]           = df["step"].astype(int)
    df["isFraud"]        = df["isFraud"].astype(int)
    df["isFlaggedFraud"] = df["isFlaggedFraud"].astype(int)
    for col in ["amount", "oldbalanceOrg", "newbalanceOrig",
                "oldbalanceDest", "newbalanceDest"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # ── Outlier report (IQR and Z-score on `amount`) ──────────────────────────
    q1, q3 = df["amount"].quantile(0.25), df["amount"].quantile(0.75)
    iqr    = q3 - q1
    n_iqr  = ((df["amount"] < q1 - 1.5 * iqr) | (df["amount"] > q3 + 1.5 * iqr)).sum()
    z      = (df["amount"] - df["amount"].mean()) / df["amount"].std()
    n_z    = (z.abs() > 3).sum()
    logger.info(
        "Outlier report for 'amount': IQR-flagged=%d (%.2f%%), Z>3=%d (%.2f%%)"
        " — NOT removed (large amounts can be fraud).",
        n_iqr, 100 * n_iqr / len(df),
        n_z,   100 * n_z   / len(df),
    )

    logger.info("Cleaned: %d rows remain (%.1f s).", len(df), time.time() - t0)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all engineered features specified in the brief, then one-hot
    encode `type` and drop high-cardinality ID columns.

    New columns (all derived from existing PaySim columns only):
        hour, day, isNight, balanceDiffOrig, balanceDiffDest,
        origEmptied, errorOrig, errorDest, amountToBalanceRatio,
        type_CASH_IN, type_CASH_OUT, type_DEBIT, type_PAYMENT, type_TRANSFER
    """
    logger.info("Engineering features …")

    # Time features
    df["hour"]    = (df["step"] - 1) % 24
    df["day"]     = (df["step"] - 1) // 24
    df["isNight"] = (df["hour"] < 8).astype(int)

    # Balance difference features
    df["balanceDiffOrig"] = df["oldbalanceOrg"]  - df["newbalanceOrig"]
    df["balanceDiffDest"] = df["newbalanceDest"]  - df["oldbalanceDest"]

    # Binary flag: sender balance was emptied
    df["origEmptied"] = (
        (df["newbalanceOrig"] == 0) & (df["oldbalanceOrg"] > 0)
    ).astype(int)

    # Bookkeeping error features (detects mismatch / money creation)
    df["errorOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    df["errorDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]

    # Ratio of amount to sender's balance
    df["amountToBalanceRatio"] = df["amount"] / (df["oldbalanceOrg"] + 1.0)

    # One-hot encode `type`; ensure all PaySim types are present
    for t in TRANSACTION_TYPES:
        df[f"type_{t}"] = (df["type"] == t).astype(int)

    # Drop high-cardinality ID columns and original `type`
    df = df.drop(columns=["nameOrig", "nameDest", "type", "isFlaggedFraud"],
                 errors="ignore")

    logger.info("Feature engineering done: %d columns.", len(df.columns))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SPLIT, SCALE, SMOTE
# ─────────────────────────────────────────────────────────────────────────────

def split_and_scale(df: pd.DataFrame):
    """
    Stratified 80/20 train-test split → fit StandardScaler on train only →
    apply class_weight-compatible sampling for slow models → SMOTE on training
    sample.

    Returns
    -------
    dict with keys:
        X_train_full, X_test_full, y_train_full, y_test_full   (full sets)
        X_train_smote, y_train_smote                            (SMOTE-augmented sample)
        X_train_sample, y_train_sample                          (balanced sample, no SMOTE)
        X_test_sample, y_test_sample                            (test evaluation sample)
        scaler, features
    """
    target_col = "isFraud"
    # drop target, step (raw time index), day (derived from step - both are leaky if used raw)
    drop_cols  = [c for c in [target_col, "step", "day"] if c in df.columns]
    features   = [c for c in df.columns if c not in drop_cols]
    # Re-check target is in df
    if target_col not in df.columns:
        raise ValueError("'isFraud' column missing from dataframe!")

    X = df[features].values
    y = df[target_col].values


    logger.info("Split: %d total rows, %d features.", len(X), len(features))

    # ── Stratified 80/20 split ────────────────────────────────────────────────
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("Train: %d rows  |  Test: %d rows", len(X_train_f), len(X_test_f))

    # ── Fit scaler on TRAIN only ──────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_f)
    X_test_scaled  = scaler.transform(X_test_f)

    # ── Working sample for slow models (keep all fraud + up to MAX_LEGIT_TRAIN legit) ─
    train_fraud_idx = np.where(y_train_f == 1)[0]
    train_legit_idx = np.where(y_train_f == 0)[0]
    rng = np.random.default_rng(RANDOM_STATE)
    if len(train_legit_idx) > MAX_LEGIT_TRAIN:
        sampled_legit = rng.choice(train_legit_idx, MAX_LEGIT_TRAIN, replace=False)
    else:
        sampled_legit = train_legit_idx
    sample_idx = np.concatenate([train_fraud_idx, sampled_legit])
    rng.shuffle(sample_idx)
    X_train_samp = X_train_scaled[sample_idx]
    y_train_samp = y_train_f[sample_idx]
    logger.info("Train sample: %d rows (fraud=%d, legit=%d)",
                len(y_train_samp), y_train_samp.sum(),
                (y_train_samp == 0).sum())

    # ── Test evaluation sample (natural fraud ratio) ──────────────────────────
    test_fraud_idx = np.where(y_test_f == 1)[0]
    test_legit_idx = np.where(y_test_f == 0)[0]
    if len(test_legit_idx) > MAX_LEGIT_TEST:
        sampled_test_legit = rng.choice(test_legit_idx, MAX_LEGIT_TEST, replace=False)
    else:
        sampled_test_legit = test_legit_idx
    test_idx = np.concatenate([test_fraud_idx, sampled_test_legit])
    rng.shuffle(test_idx)
    X_test_samp = X_test_scaled[test_idx]
    y_test_samp = y_test_f[test_idx]
    logger.info("Test sample: %d rows (fraud=%d, legit=%d)",
                len(y_test_samp), y_test_samp.sum(),
                (y_test_samp == 0).sum())

    # ── SMOTE on TRAINING SAMPLE only ────────────────────────────────────────
    smote = SMOTE(random_state=RANDOM_STATE)
    X_smote, y_smote = smote.fit_resample(X_train_samp, y_train_samp)
    logger.info("After SMOTE: %d rows (fraud=%d, legit=%d)",
                len(y_smote), y_smote.sum(), (y_smote == 0).sum())

    # ── Persist artefacts ─────────────────────────────────────────────────────
    joblib.dump(scaler,   SCALER_PKL)
    joblib.dump(features, FEATURE_LIST_PKL)
    logger.info("Scaler and feature list saved.")

    return {
        "X_train_full"   : X_train_scaled,
        "X_test_full"    : X_test_scaled,
        "y_train_full"   : y_train_f,
        "y_test_full"    : y_test_f,
        "X_train_smote"  : X_smote,
        "y_train_smote"  : y_smote,
        "X_train_sample" : X_train_samp,
        "y_train_sample" : y_train_samp,
        "X_test_sample"  : X_test_samp,
        "y_test_sample"  : y_test_samp,
        "scaler"         : scaler,
        "features"       : features,
    }


def save_processed(df: pd.DataFrame) -> None:
    """Save the feature-engineered dataframe for caching (preserving all fraud rows)."""
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if len(df) > 300_000:
        fraud_df = df[df["isFraud"] == 1]
        legit_df = df[df["isFraud"] == 0].sample(300_000 - len(fraud_df), random_state=RANDOM_STATE)
        save_df = pd.concat([fraud_df, legit_df]).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
        save_df.to_csv(PROCESSED_CSV, index=False)
        logger.info("Processed data (300k stratified sample with 100%% frauds) saved to %s", PROCESSED_CSV)
    else:
        df.to_csv(PROCESSED_CSV, index=False)
        logger.info("Processed data saved to %s", PROCESSED_CSV)
