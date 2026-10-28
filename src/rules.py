"""
rules.py - Association rule mining with mlxtend Apriori.

Steps:
  1. Bin numeric features (amount → Low/Medium/High, hour → Night/Day)
  2. Add boolean flags: origEmptied, type_TRANSFER, type_CASH_OUT, isFraud
  3. Run Apriori + association_rules
  4. Filter consequents containing isFraud=1
  5. Sort by lift, save reports/fraud_rules.csv

Output:
  reports/fraud_rules.csv  (support, confidence, lift, antecedents, consequents)
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from config import (
    APRIORI_MIN_CONF,
    APRIORI_MIN_SUP,
    FRAUD_RULES_CSV,
    RANDOM_STATE,
    REPORTS_DIR,
)

logger = logging.getLogger(__name__)


def _bin_amount(series: pd.Series) -> pd.Series:
    q33, q66 = series.quantile(0.33), series.quantile(0.66)
    return pd.cut(series, bins=[-np.inf, q33, q66, np.inf],
                  labels=["amount_Low", "amount_Medium", "amount_High"])


def _bin_hour(series: pd.Series) -> pd.Series:
    return series.apply(lambda h: "hour_Night" if h < 8 else "hour_Day")


def build_transaction_list(df: pd.DataFrame) -> list:
    """
    Convert each row into a set of string items for mlxtend.

    Items used (only PaySim-derived):
        amount_Low / amount_Medium / amount_High
        hour_Night / hour_Day
        origEmptied
        type_TRANSFER
        type_CASH_OUT
        isFraud
    """
    df = df.reset_index(drop=True)
    rows = []
    amount_binned = _bin_amount(df["amount"])
    hour_col      = df["hour"] if "hour" in df.columns else pd.Series(12, index=df.index)
    hour_binned   = _bin_hour(hour_col)

    # Pre-extract columns as numpy arrays for speed
    orig_emptied   = df["origEmptied"].values       if "origEmptied"   in df.columns else np.zeros(len(df), int)
    type_transfer  = df["type_TRANSFER"].values     if "type_TRANSFER" in df.columns else np.zeros(len(df), int)
    type_cash_out  = df["type_CASH_OUT"].values     if "type_CASH_OUT" in df.columns else np.zeros(len(df), int)
    is_fraud       = df["isFraud"].values           if "isFraud"       in df.columns else np.zeros(len(df), int)

    for i in range(len(df)):
        items = [str(amount_binned.iloc[i]), str(hour_binned.iloc[i])]

        if orig_emptied[i]  == 1: items.append("origEmptied")
        if type_transfer[i] == 1: items.append("type_TRANSFER")
        if type_cash_out[i] == 1: items.append("type_CASH_OUT")
        if is_fraud[i]      == 1: items.append("isFraud")

        rows.append(items)
    return rows


def run_apriori(df_engineered: pd.DataFrame) -> pd.DataFrame:
    """
    Run Apriori on a stratified sample of the engineered dataframe.

    Keeps all fraud rows + up to 20k legit rows to make Apriori tractable.
    """
    t0 = time.time()
    logger.info("Association rule mining (Apriori) …")

    # Sample
    rng = np.random.default_rng(RANDOM_STATE)
    fraud_mask = df_engineered["isFraud"] == 1
    fraud_df   = df_engineered[fraud_mask]
    legit_df   = df_engineered[~fraud_mask]

    max_legit = 20_000
    if len(legit_df) > max_legit:
        legit_df = legit_df.sample(max_legit, random_state=RANDOM_STATE)

    sample = pd.concat([fraud_df, legit_df]).reset_index(drop=True)
    logger.info("  Apriori sample: %d rows (fraud=%d)", len(sample), fraud_mask.sum())

    # Build transactions
    transactions = build_transaction_list(sample)

    # One-hot encode
    te = TransactionEncoder()
    te_array = te.fit_transform(transactions)
    te_df = pd.DataFrame(te_array, columns=te.columns_)

    # Apriori
    frequent_items = apriori(te_df, min_support=APRIORI_MIN_SUP, use_colnames=True)
    if len(frequent_items) == 0:
        logger.warning("No frequent itemsets at min_support=%.3f, retrying with 0.001 …", APRIORI_MIN_SUP)
        frequent_items = apriori(te_df, min_support=0.001, use_colnames=True)

    if len(frequent_items) == 0:
        logger.warning("No frequent itemsets found.")
        save_df = pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        save_df.to_csv(FRAUD_RULES_CSV, index=False)
        return save_df

    rules = association_rules(frequent_items, metric="confidence",
                              min_threshold=APRIORI_MIN_CONF)
    logger.info("  Total rules: %d", len(rules))

    if len(rules) == 0:
        save_df = pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        save_df.to_csv(FRAUD_RULES_CSV, index=False)
        return save_df

    # Filter rules where consequent contains 'isFraud'
    rules["has_fraud_consequent"] = rules["consequents"].apply(
        lambda cs: "isFraud" in cs
    )
    fraud_rules = rules[rules["has_fraud_consequent"]].copy()
    if len(fraud_rules) == 0:
        fraud_rules = rules.copy() # fallback to any high-confidence rules

    fraud_rules = fraud_rules.sort_values("lift", ascending=False).reset_index(drop=True)

    # Human-readable columns
    fraud_rules["antecedents_str"]  = fraud_rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    fraud_rules["consequents_str"]  = fraud_rules["consequents"].apply(lambda s: ", ".join(sorted(s)))

    out_cols = ["antecedents_str", "consequents_str", "support", "confidence", "lift"]
    save_df  = fraud_rules[out_cols].rename(columns={
        "antecedents_str": "antecedents",
        "consequents_str": "consequents",
    })

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    save_df.to_csv(FRAUD_RULES_CSV, index=False)
    logger.info(
        "Fraud rules: %d found  →  %s  (%.1f s)",
        len(save_df), FRAUD_RULES_CSV, time.time() - t0,
    )

    # Log top 5
    for _, row in save_df.head(5).iterrows():
        logger.info(
            "  IF [%s] THEN [%s]  sup=%.4f  conf=%.3f  lift=%.2f",
            row["antecedents"], row["consequents"],
            row["support"], row["confidence"], row["lift"],
        )

    return save_df
