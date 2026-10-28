"""
make_figures.py - Generate the six required documentation PNGs at 200 DPI.

All figures are derived from the CLEANED + ENGINEERED data (no pie charts).
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from config import FIGURE_DPI, FIGURES_DIR, REPORTS_DIR

logger = logging.getLogger(__name__)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor"  : "#F8F9FA",
    "axes.grid"       : True,
    "grid.alpha"      : 0.4,
    "font.size"       : 11,
})


def g31_boxplot_amount_by_fraud(df: pd.DataFrame) -> None:
    """3.1 Box Plot - Transaction Amount by Fraud Class (log y-axis)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    legit_amounts = df.loc[df["isFraud"] == 0, "amount"].clip(lower=1)
    fraud_amounts = df.loc[df["isFraud"] == 1, "amount"].clip(lower=1)

    bp = ax.boxplot(
        [legit_amounts, fraud_amounts],
        patch_artist=True,
        notch=True,
        widths=0.5,
    )
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Legitimate", "Fraud"])
    colors = ["#4C6EF5", "#E03131"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color("white")
        median.set_linewidth(2)

    ax.set_yscale("log")
    ax.set_title("3.1 Box Plot - Transaction Amount by Fraud Class", fontsize=13, fontweight="bold")
    ax.set_ylabel("Transaction Amount (₹) — log scale")
    ax.set_xlabel("Class")
    fig.tight_layout()
    out = FIGURES_DIR / "g31_boxplot_amount_by_fraud.png"
    fig.savefig(out, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved %s", out)


def g32_hist_amount(df: pd.DataFrame) -> None:
    """3.2 Histogram - Distribution of Transaction Amount (Side-by-Side Subplots)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    legit_df = df.loc[df["isFraud"] == 0]
    fraud_df = df.loc[df["isFraud"] == 1]

    # Subplot 1: Legitimate Transactions
    bins_legit = np.logspace(
        np.log10(max(legit_df["amount"].min(), 1)),
        np.log10(legit_df["amount"].max() + 1),
        50,
    )
    ax1.hist(legit_df["amount"], bins=bins_legit, color="#4C6EF5", alpha=0.75, edgecolor="white")
    ax1.set_xscale("log")
    ax1.set_xlabel("Transaction Amount (₹) — log scale", fontweight="bold")
    ax1.set_ylabel("Transaction Count", fontweight="bold")
    ax1.set_title(f"Legitimate Transactions (Total: {len(legit_df):,})", fontsize=12, fontweight="bold", color="#1A365D")
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Fraudulent Transactions
    bins_fraud = np.logspace(
        np.log10(max(fraud_df["amount"].min(), 1)),
        np.log10(fraud_df["amount"].max() + 1),
        50,
    )
    ax2.hist(fraud_df["amount"], bins=bins_fraud, color="#E03131", alpha=0.8, edgecolor="white")
    ax2.set_xscale("log")
    ax2.set_xlabel("Transaction Amount (₹) — log scale", fontweight="bold")
    ax2.set_ylabel("Transaction Count", fontweight="bold")
    ax2.set_title(f"Fraudulent Transactions (Total: {len(fraud_df):,})", fontsize=12, fontweight="bold", color="#9B1C1C")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("3.2 Histogram — Distribution of Transaction Amount (Legitimate vs Fraud)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = FIGURES_DIR / "g32_hist_amount.png"
    fig.savefig(out, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved %s", out)


def g33_line_volume_by_hour(df: pd.DataFrame) -> None:
    """3.3 Line Plot - Transaction Volume by Hour of Day."""
    hourly = df.groupby("hour").size().reset_index(name="count")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hourly["hour"], hourly["count"], "o-", color="#4C6EF5",
            lw=2, markersize=5, label="Total transactions")
    ax.axvspan(-0.5, 7.5, color="navy", alpha=0.1, label="Night hours (00–07)")
    ax.set_xticks(range(24))
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Transaction Count")
    ax.set_title("3.3 Line Plot - Transaction Volume by Hour of Day", fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    out = FIGURES_DIR / "g33_line_volume_by_hour.png"
    fig.savefig(out, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved %s", out)


def g51_bar_fraud_rate_by_type(df: pd.DataFrame) -> None:
    """5.1 Bar Plot - Fraud Rate by Transaction Type."""
    # Reconstruct type from one-hot columns
    type_cols = [c for c in df.columns if c.startswith("type_")]
    if not type_cols:
        logger.warning("No type_ columns found; skipping g51.")
        return
    type_series = df[type_cols].idxmax(axis=1).str.replace("type_", "", regex=False)
    df2 = df.copy()
    df2["type"] = type_series

    by_type = df2.groupby("type")["isFraud"].agg(["mean", "sum", "count"])
    by_type = by_type.sort_values("mean", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(by_type.index, by_type["mean"] * 100, color="#E03131", alpha=0.75, edgecolor="white")
    ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=10)
    ax.set_ylabel("Fraud Rate (%)")
    ax.set_xlabel("Transaction Type")
    ax.set_title("5.1 Bar Plot - Fraud Rate by Transaction Type", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = FIGURES_DIR / "g51_bar_fraud_rate_by_type.png"
    fig.savefig(out, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved %s", out)


def g52_line_fraud_rate_by_hour(df: pd.DataFrame) -> None:
    """5.2 Line Plot - Fraud Rate by Hour of Day (night shaded)."""
    if "hour" not in df.columns:
        df = df.copy(); df["hour"] = (df["step"] - 1) % 24
    hourly = df.groupby("hour")["isFraud"].mean().reset_index(name="fraud_rate")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hourly["hour"], hourly["fraud_rate"] * 100, "o-",
            color="#E03131", lw=2, markersize=5, label="Fraud rate %")
    ax.axvspan(-0.5, 7.5, color="navy", alpha=0.12, label="Night hours (00–07)")
    ax.set_xticks(range(24))
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Fraud Rate (%)")
    ax.set_title("5.2 Line Plot - Fraud Rate by Hour of Day", fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    out = FIGURES_DIR / "g52_line_fraud_rate_by_hour.png"
    fig.savefig(out, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved %s", out)


def g53_heatmap_correlation(df: pd.DataFrame) -> None:
    """5.3 Heatmap - Correlation Matrix of Cleaned Features."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    # Remove step (leaky)
    num_cols = [c for c in num_cols if c not in ["step"]]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(num_cols)))
    ax.set_yticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(num_cols, fontsize=8)
    # Annotate cells
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                    fontsize=6, color="black" if abs(corr.values[i,j]) < 0.7 else "white")
    ax.set_title("5.3 Heatmap - Correlation Matrix of Cleaned Features", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = FIGURES_DIR / "g53_heatmap_correlation.png"
    fig.savefig(out, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved %s", out)


def write_data_summary(df: pd.DataFrame) -> None:
    """Write data_summary.txt with shape, fraud stats, describe, missing values."""
    n_fraud    = int(df["isFraud"].sum())
    pct_fraud  = df["isFraud"].mean() * 100
    n_missing  = int(df.isnull().sum().sum())
    desc       = df["amount"].describe()

    lines = [
        "=" * 70,
        "DATA SUMMARY",
        "=" * 70,
        f"Shape          : {df.shape}",
        f"Fraud count    : {n_fraud:,}",
        f"Fraud pct      : {pct_fraud:.4f}%",
        f"Legit count    : {len(df) - n_fraud:,}",
        f"Missing values : {n_missing}",
        "",
        "--- amount describe ---",
        str(desc),
        "",
        "--- Class balance ---",
        str(df["isFraud"].value_counts()),
        "",
        "--- Column dtypes ---",
        str(df.dtypes),
    ]

    out = REPORTS_DIR / "data_summary.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    logger.info("Data summary written → %s", out)


def make_all_figures(df: pd.DataFrame) -> None:
    """Generate all 6 required documentation figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Generating documentation figures …")
    g31_boxplot_amount_by_fraud(df)
    g32_hist_amount(df)
    g33_line_volume_by_hour(df)
    g51_bar_fraud_rate_by_type(df)
    g52_line_fraud_rate_by_hour(df)
    g53_heatmap_correlation(df)
    write_data_summary(df)
    logger.info("All figures done.")
