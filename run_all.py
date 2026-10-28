"""
run_all.py - Master pipeline runner for UPI Fraud Radar.

Usage:
    python run_all.py          # requires real CSV in data/raw/
    python run_all.py --demo   # generates synthetic demo data and runs pipeline
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# ── Add src/ to path ──────────────────────────────────────────────────────────
SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

# ── Now import project modules ────────────────────────────────────────────────
from config import (
    DATA_PROCESSED_DIR,
    DEMO_FLAG_FILE,
    PROCESSED_CSV,
    REPORTS_DIR,
    FIGURES_DIR,
    MODELS_DIR,
)
from data_loader   import load_data
from preprocess    import clean, engineer_features, split_and_scale, save_processed
from features      import select_features
from train         import train_all
from evaluate      import run_evaluation
from regression    import run_regression
from neural_net    import run_neural_net
from cluster       import run_clustering
from rules         import run_apriori
from make_figures  import make_all_figures

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(__file__).parent / "pipeline.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("run_all")


def banner(msg: str) -> None:
    logger.info("")
    logger.info("=" * 65)
    logger.info("  %s", msg)
    logger.info("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="UPI Fraud Radar — Full Pipeline")
    parser.add_argument("--demo", action="store_true",
                        help="Use synthetic demo data instead of real PaySim CSV")
    args = parser.parse_args()

    total_start = time.time()
    banner("UPI FRAUD RADAR — PIPELINE START" + (" [DEMO MODE]" if args.demo else ""))

    # ── Mark demo mode ────────────────────────────────────────────────────────
    if args.demo:
        DEMO_FLAG_FILE.parent.mkdir(parents=True, exist_ok=True)
        DEMO_FLAG_FILE.write_text("demo")
        logger.info("DEMO MODE: outputs will be marked as DEMO DATA in the dashboard.")
    else:
        if DEMO_FLAG_FILE.exists():
            DEMO_FLAG_FILE.unlink()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Load data
    # ─────────────────────────────────────────────────────────────────────────
    banner("STEP 1 — Loading data")
    df_raw = load_data(demo=args.demo)
    logger.info("Raw data shape: %s", df_raw.shape)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Clean
    # ─────────────────────────────────────────────────────────────────────────
    banner("STEP 2 — Cleaning")
    df_clean = clean(df_raw)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: Feature engineering
    # ─────────────────────────────────────────────────────────────────────────
    banner("STEP 3 — Feature Engineering")
    df_feat = engineer_features(df_clean)
    save_processed(df_feat)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: Documentation figures (use df_feat)
    # ─────────────────────────────────────────────────────────────────────────
    banner("STEP 4 — Documentation Figures")
    make_all_figures(df_feat)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5: Split / scale / SMOTE
    # ─────────────────────────────────────────────────────────────────────────
    banner("STEP 5 — Split, Scale, SMOTE")
    data = split_and_scale(df_feat)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6: Feature selection
    # ─────────────────────────────────────────────────────────────────────────
    banner("STEP 6 — Feature Selection")
    select_features(
        data["X_train_sample"],
        data["y_train_sample"],
        data["features"],
    )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 7: Train all classifiers
    # ─────────────────────────────────────────────────────────────────────────
    banner("STEP 7 — Training Classifiers")
    models = train_all(data)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 8: Evaluate + select best model
    # ─────────────────────────────────────────────────────────────────────────
    banner("STEP 8 — Evaluation")
    all_metrics = run_evaluation(models, data, df_feat)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 9: Regression
    # ─────────────────────────────────────────────────────────────────────────
    banner("STEP 9 — Regression Analysis")
    run_regression(df_feat, data["features"])

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 10: Neural network (MLP)
    # ─────────────────────────────────────────────────────────────────────────
    banner("STEP 10 — Neural Network (MLP)")
    run_neural_net(data)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 11: Clustering
    # ─────────────────────────────────────────────────────────────────────────
    banner("STEP 11 — Clustering")
    run_clustering(df_feat, scaler=data["scaler"], features=data["features"])

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 12: Association rules
    # ─────────────────────────────────────────────────────────────────────────
    banner("STEP 12 — Association Rules (Apriori)")
    run_apriori(df_feat)

    # ─────────────────────────────────────────────────────────────────────────
    # DONE
    # ─────────────────────────────────────────────────────────────────────────
    total = time.time() - total_start
    banner(f"PIPELINE COMPLETE in {total/60:.1f} minutes")

    logger.info("\nFiles created:")
    for d in [MODELS_DIR, REPORTS_DIR, FIGURES_DIR, DATA_PROCESSED_DIR]:
        for f in sorted(d.iterdir()):
            if f.is_file():
                logger.info("  %s", f)

    logger.info("\nNext step:")
    logger.info("  streamlit run app/app.py")

    if args.demo:
        logger.info("\nNOTE: Pipeline ran in DEMO MODE on synthetic data.")
        logger.info("To use real PaySim data:")
        logger.info("  1) Download from https://www.kaggle.com/datasets/ealaxi/paysim1")
        logger.info("  2) Place in data/raw/PS_20174392719_1491204439457_log.csv")
        logger.info("  3) Run: python run_all.py")


if __name__ == "__main__":
    main()
