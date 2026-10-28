"""
config.py - Central configuration for UPI Fraud Radar project.
All paths and constants are defined here; no hard-coded paths elsewhere.
"""

from pathlib import Path

# ── Root directory (project root, one level above src/) ──────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_RAW_DIR       = ROOT / "data" / "raw"
DATA_PROCESSED_DIR = ROOT / "data" / "processed"
RAW_CSV            = DATA_RAW_DIR / "PS_20174392719_1491204439457_log.csv"
DEMO_CSV           = DATA_RAW_DIR / "demo_paysim.csv"
PROCESSED_CSV      = DATA_PROCESSED_DIR / "processed_features.csv"
DEMO_FLAG_FILE     = DATA_PROCESSED_DIR / "is_demo.flag"   # written when demo is used

# ── Model / report paths ───────────────────────────────────────────────────────
MODELS_DIR         = ROOT / "models"
REPORTS_DIR        = ROOT / "reports"
FIGURES_DIR        = REPORTS_DIR / "figures"

BEST_MODEL_PKL     = MODELS_DIR / "best_model.pkl"
SCALER_PKL         = MODELS_DIR / "scaler.pkl"
FEATURE_LIST_PKL   = MODELS_DIR / "feature_list.pkl"
ISO_FOREST_PKL     = MODELS_DIR / "iso_forest.pkl"
CLUSTER_PKL        = MODELS_DIR / "cluster_kmeans.pkl"
KMEDOIDS_PKL       = MODELS_DIR / "cluster_kmedoids.pkl"

METRICS_JSON       = REPORTS_DIR / "metrics.json"
MODEL_COMP_CSV     = REPORTS_DIR / "model_comparison.csv"
FEATURE_RANK_CSV   = REPORTS_DIR / "feature_ranking.csv"
FRAUD_RULES_CSV    = REPORTS_DIR / "fraud_rules.csv"
BAYES_JSON         = REPORTS_DIR / "bayes_demo.json"
DATA_SUMMARY_TXT   = REPORTS_DIR / "data_summary.txt"

# ── Split / sample sizes ───────────────────────────────────────────────────────
TEST_SIZE          = 0.20          # 80/20 stratified split
RANDOM_STATE       = 42
MAX_LEGIT_TRAIN    = 300_000       # max legitimate rows kept for slow models
MAX_LEGIT_TEST     = 100_000       # max legitimate rows in test evaluation sample
CV_FOLDS           = 5
BOOTSTRAP_N        = 1_000

# ── Demo generation parameters ────────────────────────────────────────────────
DEMO_ROWS          = 300_000
DEMO_FRAUD_RATE    = 0.005         # 0.5 %
DEMO_FRAUD_TYPES   = ["TRANSFER", "CASH_OUT"]
DEMO_LEGIT_TYPES   = ["PAYMENT", "DEBIT", "TRANSFER", "CASH_OUT", "CASH_IN"]

# ── Cluster / rule parameters ──────────────────────────────────────────────────
CLUSTER_K_RANGE    = range(2, 9)   # k values for elbow method
APRIORI_MIN_SUP    = 0.005
APRIORI_MIN_CONF   = 0.30

# ── Figure DPI ─────────────────────────────────────────────────────────────────
FIGURE_DPI         = 200

# ── Transaction types in PaySim ───────────────────────────────────────────────
TRANSACTION_TYPES  = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# ── Ensure all output directories exist ───────────────────────────────────────
for _d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
