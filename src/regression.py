"""
regression.py - Three regression analyses:
  1. Simple linear regression:  amount ~ oldbalanceOrg
  2. Multiple linear regression: predict amount from all numeric features
  3. Polynomial regression (degree 3): fraud_rate ~ hour_of_day

Outputs:
  reports/regression_results.json
  reports/figures/reg_simple.png
  reports/figures/reg_polynomial.png
"""

import json
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from config import FIGURES_DIR, RANDOM_STATE, REPORTS_DIR, FIGURE_DPI

logger = logging.getLogger(__name__)


def simple_linear_regression(df: pd.DataFrame) -> dict:
    """
    Simple linear regression: amount ~ oldbalanceOrg.
    Uses a 10k sample for scatter plot clarity.
    """
    logger.info("Simple linear regression: amount ~ oldbalanceOrg …")
    sample = df.sample(min(10_000, len(df)), random_state=RANDOM_STATE)
    X = sample[["oldbalanceOrg"]].values
    y = sample["amount"].values

    reg = LinearRegression()
    reg.fit(X, y)
    y_pred = reg.predict(X)
    r2  = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    logger.info("  R²=%.4f  MSE=%.2f", r2, mse)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X, y, alpha=0.3, s=8, color="#4C6EF5", label="Data (sample)")
    x_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    ax.plot(x_line, reg.predict(x_line), color="#E03131", lw=2, label=f"Fit (R²={r2:.3f})")
    ax.set_xlabel("Sender Old Balance (₹)")
    ax.set_ylabel("Transaction Amount (₹)")
    ax.set_title("Simple Linear Regression: Amount vs Sender Balance")
    ax.legend()
    ax.set_xscale("symlog")
    ax.set_yscale("symlog")
    fig.tight_layout()
    out = FIGURES_DIR / "reg_simple.png"
    fig.savefig(out, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("  Saved %s", out)

    return {"type": "simple", "r2": round(r2, 4), "mse": round(mse, 2),
            "coef": round(float(reg.coef_[0]), 6),
            "intercept": round(float(reg.intercept_), 2)}


def multiple_linear_regression(df: pd.DataFrame, feature_cols: list) -> dict:
    """Multiple linear regression predicting `amount`."""
    logger.info("Multiple linear regression: predicting amount …")
    valid_feats = [c for c in feature_cols if c in df.columns and c != "amount"]
    if len(valid_feats) < 2:
        logger.warning("Too few features for multiple regression.")
        return {}
    sub = df[valid_feats + ["amount"]].dropna().sample(
        min(50_000, len(df)), random_state=RANDOM_STATE
    )
    X = sub[valid_feats].values
    y = sub["amount"].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    reg = LinearRegression()
    reg.fit(X_tr, y_tr)
    y_pred = reg.predict(X_te)
    r2  = r2_score(y_te, y_pred)
    mse = mean_squared_error(y_te, y_pred)
    logger.info("  Multiple regression  R²=%.4f  MSE=%.2f", r2, mse)
    coef_df = pd.DataFrame({"feature": valid_feats, "coef": reg.coef_})
    coef_df = coef_df.reindex(coef_df["coef"].abs().sort_values(ascending=False).index)
    return {"type": "multiple", "r2": round(r2, 4), "mse": round(mse, 2),
            "n_features": len(valid_feats)}


def polynomial_regression_fraud_vs_hour(df: pd.DataFrame) -> dict:
    """
    Polynomial regression (degree 3): fraud_rate ~ hour_of_day.
    Groups by hour, computes fraud rate, fits a degree-3 polynomial.
    """
    logger.info("Polynomial regression: fraud_rate ~ hour …")
    if "hour" not in df.columns:
        df = df.copy()
        df["hour"] = (df["step"] - 1) % 24

    hourly = (
        df.groupby("hour")["isFraud"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "fraud_rate", "count": "n"})
        .reset_index()
    )
    X = hourly[["hour"]].values
    y = hourly["fraud_rate"].values

    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_poly = poly.fit_transform(X)
    reg = LinearRegression()
    reg.fit(X_poly, y)
    y_pred = reg.predict(X_poly)
    r2 = r2_score(y, y_pred)
    logger.info("  Polynomial regression  R²=%.4f", r2)

    # Smooth curve
    x_smooth = np.linspace(0, 23, 200).reshape(-1, 1)
    y_smooth  = reg.predict(poly.transform(x_smooth))

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(hourly["hour"], hourly["fraud_rate"] * 100,
           color="#4C6EF5", alpha=0.6, label="Actual fraud rate %")
    ax.plot(x_smooth, y_smooth * 100, color="#E03131", lw=2,
            label=f"Polynomial fit (degree=3, R²={r2:.3f})")
    ax.axvspan(-0.5, 7.5, color="navy", alpha=0.12, label="Night hours (00–07)")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Fraud Rate (%)")
    ax.set_title("Polynomial Regression: Fraud Rate vs Hour of Day")
    ax.set_xticks(range(24))
    ax.legend()
    fig.tight_layout()
    out = FIGURES_DIR / "reg_polynomial.png"
    fig.savefig(out, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("  Saved %s", out)

    return {"type": "polynomial", "degree": 3, "r2": round(r2, 4),
            "hourly_fraud_rate": hourly[["hour", "fraud_rate"]].to_dict(orient="records")}


def run_regression(df: pd.DataFrame, feature_cols: list) -> None:
    """Run all three regression analyses and save results."""
    results = {}
    results["simple"]     = simple_linear_regression(df)
    results["multiple"]   = multiple_linear_regression(df, feature_cols)
    results["polynomial"] = polynomial_regression_fraud_vs_hour(df)

    out = REPORTS_DIR / "regression_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Regression results saved → %s", out)
