"""
predict.py - Load best model and score one transaction.

Usage:
    python predict.py --type TRANSFER --amount 50000 --hour 2
                      --old_balance_orig 51000 --old_balance_dest 0
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Add src/ to path so imports work when called from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    BEST_MODEL_PKL,
    FEATURE_LIST_PKL,
    ISO_FOREST_PKL,
    SCALER_PKL,
    TRANSACTION_TYPES,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_artifacts():
    """Load model, scaler, feature list. Raises FileNotFoundError if missing."""
    for p in [BEST_MODEL_PKL, SCALER_PKL, FEATURE_LIST_PKL]:
        if not p.exists():
            raise FileNotFoundError(
                f"Artifact not found: {p}\n"
                "Run `python run_all.py [--demo]` first to train models."
            )
    model    = joblib.load(BEST_MODEL_PKL)
    scaler   = joblib.load(SCALER_PKL)
    features = joblib.load(FEATURE_LIST_PKL)
    iso      = joblib.load(ISO_FOREST_PKL) if ISO_FOREST_PKL.exists() else None
    return model, scaler, features, iso


def build_feature_row(
    txn_type: str,
    amount: float,
    hour: int,
    old_balance_orig: float,
    new_balance_orig: float,
    old_balance_dest: float,
    new_balance_dest: float,
) -> dict:
    """Compute all derived features for a single transaction."""
    is_night         = int(hour < 8)
    balance_diff_o   = old_balance_orig - new_balance_orig
    balance_diff_d   = new_balance_dest - old_balance_dest
    orig_emptied     = int(new_balance_orig == 0 and old_balance_orig > 0)
    error_orig       = new_balance_orig + amount - old_balance_orig
    error_dest       = old_balance_dest + amount - new_balance_dest
    amount_to_bal    = amount / (old_balance_orig + 1.0)

    row = {
        "amount"              : amount,
        "oldbalanceOrg"       : old_balance_orig,
        "newbalanceOrig"      : new_balance_orig,
        "oldbalanceDest"      : old_balance_dest,
        "newbalanceDest"      : new_balance_dest,
        "hour"                : hour,
        "isNight"             : is_night,
        "balanceDiffOrig"     : balance_diff_o,
        "balanceDiffDest"     : balance_diff_d,
        "origEmptied"         : orig_emptied,
        "errorOrig"           : error_orig,
        "errorDest"           : error_dest,
        "amountToBalanceRatio": amount_to_bal,
    }
    # One-hot type
    for t in TRANSACTION_TYPES:
        row[f"type_{t}"] = int(txn_type == t)

    return row


def score_transaction(
    txn_type: str,
    amount: float,
    hour: int,
    old_balance_orig: float,
    new_balance_orig: float = 0.0,
    old_balance_dest: float = 0.0,
    new_balance_dest: float = 0.0,
) -> dict:
    """
    Score a single transaction and return probability + anomaly score.

    Parameters follow the PaySim column schema; new balances default to 0.
    """
    model, scaler, features, iso = load_artifacts()

    row  = build_feature_row(
        txn_type, amount, hour,
        old_balance_orig, new_balance_orig,
        old_balance_dest, new_balance_dest,
    )
    df   = pd.DataFrame([row])
    # Align to feature list (fill missing with 0)
    for f in features:
        if f not in df.columns:
            df[f] = 0
    X = scaler.transform(df[features].values)

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X)[0, 1])
    elif hasattr(model, "decision_function"):
        s  = model.decision_function(X)[0]
        prob = float(1 / (1 + np.exp(-s)))   # sigmoid
    else:
        prob = float(model.predict(X)[0])

    # Anomaly score (-1 = anomaly in IsolationForest; convert to 0-1)
    anomaly_score = None
    if iso is not None:
        raw = iso.score_samples(X)[0]
        anomaly_score = round(float(1 - (raw - (-0.5)) / 0.5), 4)
        anomaly_score = max(0.0, min(1.0, anomaly_score))

    risk_level = "LOW" if prob < 0.3 else ("MEDIUM" if prob < 0.7 else "HIGH")
    action     = {"LOW": "Allow", "MEDIUM": "Verify via OTP", "HIGH": "Block and alert"}[risk_level]

    result = {
        "fraud_probability": round(prob, 4),
        "risk_level"       : risk_level,
        "action"           : action,
        "anomaly_score"    : anomaly_score,
        "features_used"    : row,
    }

    # SHAP explanation (optional)
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]   # class 1
        else:
            sv = shap_vals[0]
        shap_df = pd.DataFrame({"feature": features, "shap": sv})
        shap_df = shap_df.reindex(shap_df["shap"].abs().sort_values(ascending=False).index)
        result["top_contributions"] = shap_df.head(5)[["feature", "shap"]].to_dict(orient="records")
    except Exception:
        # Fallback: feature importance × feature deviation from mean
        try:
            if hasattr(model, "feature_importances_"):
                imp  = model.feature_importances_
                devs = np.abs(X[0] - 0)   # scaled, so 0 is mean
                scores = imp * devs
                idx = np.argsort(scores)[::-1][:5]
                result["top_contributions"] = [
                    {"feature": features[i], "shap": float(scores[i])} for i in idx
                ]
        except Exception:
            pass

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Score a single UPI transaction.")
    parser.add_argument("--type",             default="TRANSFER",
                        choices=TRANSACTION_TYPES)
    parser.add_argument("--amount",           type=float, default=50_000)
    parser.add_argument("--hour",             type=int,   default=2)
    parser.add_argument("--old_balance_orig", type=float, default=51_000)
    parser.add_argument("--new_balance_orig", type=float, default=0)
    parser.add_argument("--old_balance_dest", type=float, default=0)
    parser.add_argument("--new_balance_dest", type=float, default=0)
    args = parser.parse_args()

    result = score_transaction(
        txn_type         = args.type,
        amount           = args.amount,
        hour             = args.hour,
        old_balance_orig = args.old_balance_orig,
        new_balance_orig = args.new_balance_orig,
        old_balance_dest = args.old_balance_dest,
        new_balance_dest = args.new_balance_dest,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
