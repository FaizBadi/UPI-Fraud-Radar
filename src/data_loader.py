"""
data_loader.py - Load the PaySim dataset or generate a synthetic demo version.

Usage:
    df = load_data(demo=False)   # tries real CSV; raises FileNotFoundError
    df = load_data(demo=True)    # generates & saves DEMO_CSV, returns it
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DEMO_CSV,
    DEMO_FRAUD_RATE,
    DEMO_FRAUD_TYPES,
    DEMO_LEGIT_TYPES,
    DEMO_ROWS,
    RAW_CSV,
    RANDOM_STATE,
)

logger = logging.getLogger(__name__)

# ── Column schema ─────────────────────────────────────────────────────────────
PAYSIM_COLS = [
    "step", "type", "amount",
    "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest",
    "isFraud", "isFlaggedFraud",
]


def _missing_dataset_message() -> str:
    return (
        "\n"
        "=" * 70 + "\n"
        "  DATASET NOT FOUND\n"
        "=" * 70 + "\n"
        f"  Expected: {RAW_CSV}\n\n"
        "  To download the real dataset:\n"
        "    1) Go to https://www.kaggle.com/datasets/ealaxi/paysim1\n"
        "    2) Download 'PS_20174392719_1491204439457_log.csv'\n"
        "    3) Place it in:  data/raw/\n\n"
        "  To run on synthetic demo data instead:\n"
        "    python run_all.py --demo\n"
        "=" * 70 + "\n"
    )


def generate_demo_data(n_rows: int = DEMO_ROWS, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Generate a PaySim-like synthetic dataset (~300 k rows).

    Key design choices that mirror PaySim fraud patterns:
    - Fraud ONLY in TRANSFER and CASH_OUT (as in real PaySim).
    - Fraud rate ≈ 0.5 %.
    - Fraudulent transactions mostly empty sender's balance
      (oldbalanceOrg ≈ amount, newbalanceOrig ≈ 0).
    - Legitimate traffic is suppressed 00-07 (night hours).
    - Fraud transactions are spread uniformly over 24 hours.
    """
    rng = np.random.default_rng(seed)
    logger.info("Generating synthetic demo data (%d rows) …", n_rows)
    t0 = time.time()

    n_fraud = int(n_rows * DEMO_FRAUD_RATE)
    n_legit = n_rows - n_fraud

    # ── Legitimate transactions ───────────────────────────────────────────────
    # Day/hour distribution: suppress 00-07, peak 09-18
    legit_steps = []
    while len(legit_steps) < n_legit:
        s = rng.integers(1, 745)           # 31 days * 24 h = 744 steps
        h = (s - 1) % 24
        # 5x less traffic during night hours
        if h < 8 and rng.random() < 0.80:
            continue
        legit_steps.append(s)
    legit_steps = np.array(legit_steps[:n_legit])

    legit_types  = rng.choice(DEMO_LEGIT_TYPES, size=n_legit,
                              p=[0.25, 0.05, 0.20, 0.30, 0.20])
    legit_amount = rng.lognormal(mean=8.0, sigma=1.5, size=n_legit).clip(1, 5e6)
    legit_old_o  = rng.lognormal(mean=9.0, sigma=1.8, size=n_legit).clip(0, 1e7)
    legit_new_o  = np.maximum(0.0, legit_old_o - legit_amount * rng.uniform(0.1, 1.0, n_legit))
    legit_old_d  = rng.lognormal(mean=8.5, sigma=1.8, size=n_legit).clip(0, 1e7)
    legit_new_d  = legit_old_d + legit_amount * rng.uniform(0.5, 1.0, n_legit)

    legit_df = pd.DataFrame({
        "step"           : legit_steps,
        "type"           : legit_types,
        "amount"         : legit_amount.round(2),
        "nameOrig"       : [f"C{i:09d}" for i in rng.integers(1e8, 9e8, n_legit)],
        "oldbalanceOrg"  : legit_old_o.round(2),
        "newbalanceOrig" : legit_new_o.round(2),
        "nameDest"       : [f"C{i:09d}" for i in rng.integers(1e8, 9e8, n_legit)],
        "oldbalanceDest" : legit_old_d.round(2),
        "newbalanceDest" : legit_new_d.round(2),
        "isFraud"        : 0,
        "isFlaggedFraud" : 0,
    })

    # ── Fraudulent transactions ───────────────────────────────────────────────
    # Spread uniformly over 24 hours
    fraud_steps  = rng.integers(1, 745, size=n_fraud)
    fraud_types  = rng.choice(DEMO_FRAUD_TYPES, size=n_fraud)
    fraud_amount = rng.lognormal(mean=9.5, sigma=1.2, size=n_fraud).clip(100, 5e6)
    # Mostly empty sender balance (80 % of time)
    drain_mask   = rng.random(n_fraud) < 0.80
    fraud_old_o  = np.where(drain_mask,
                            fraud_amount * rng.uniform(0.95, 1.05, n_fraud),
                            rng.lognormal(9, 1.5, n_fraud))
    fraud_new_o  = np.where(drain_mask, 0.0,
                            np.maximum(0, fraud_old_o - fraud_amount))
    fraud_old_d  = rng.lognormal(8.0, 2.0, n_fraud).clip(0, 1e6)
    fraud_new_d  = fraud_old_d  # receiver balance unchanged in fraud (mule account)

    fraud_df = pd.DataFrame({
        "step"           : fraud_steps,
        "type"           : fraud_types,
        "amount"         : fraud_amount.round(2),
        "nameOrig"       : [f"C{i:09d}" for i in rng.integers(1e8, 9e8, n_fraud)],
        "oldbalanceOrg"  : fraud_old_o.round(2),
        "newbalanceOrig" : fraud_new_o.round(2),
        "nameDest"       : [f"M{i:09d}" for i in rng.integers(1e8, 9e8, n_fraud)],
        "oldbalanceDest" : fraud_old_d.round(2),
        "newbalanceDest" : fraud_new_d.round(2),
        "isFraud"        : 1,
        "isFlaggedFraud" : 0,
    })

    df = (
        pd.concat([legit_df, fraud_df], ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    df["step"] = df["step"].astype(int)

    DEMO_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DEMO_CSV, index=False)
    elapsed = time.time() - t0
    logger.info("Demo data generated: %d rows (%.2f%% fraud) in %.1f s  →  %s",
                len(df), df["isFraud"].mean() * 100, elapsed, DEMO_CSV)
    return df


def load_data(demo: bool = False) -> pd.DataFrame:
    """
    Load the PaySim CSV (or generate demo).

    Parameters
    ----------
    demo : bool
        If True, generate / reload synthetic demo data regardless of whether
        the real CSV exists.

    Returns
    -------
    pd.DataFrame
        Raw PaySim-schema dataframe.
    """
    if demo:
        if DEMO_CSV.exists():
            logger.info("Loading existing demo data from %s", DEMO_CSV)
            df = pd.read_csv(DEMO_CSV)
        else:
            df = generate_demo_data()
        return df

    if not RAW_CSV.exists():
        print(_missing_dataset_message())
        raise FileNotFoundError(
            f"Real dataset not found at {RAW_CSV}. "
            "Run with --demo flag or place the CSV as described above."
        )

    logger.info("Loading real PaySim data from %s …", RAW_CSV)
    t0 = time.time()
    df = pd.read_csv(RAW_CSV)
    logger.info("Loaded %d rows in %.1f s", len(df), time.time() - t0)
    return df
