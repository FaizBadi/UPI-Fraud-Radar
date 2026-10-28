# 🛡️ UPI Fraud Radar — Transaction Anomaly Detection & Fraud Pattern Mining

A full-stack machine learning project for detecting fraudulent UPI/mobile-payment transactions using the PaySim synthetic dataset. Covers the complete ML pipeline with a stunning cyber-security themed dashboard.

---

## 📁 Folder Structure

```
upi-fraud-radar/
  data/raw/                 ← PaySim CSV (download from Kaggle, see below)
  data/processed/           ← Engineered features, cluster labels, PCA coords
  src/
    config.py               ← All paths & constants (edit sample sizes here)
    data_loader.py          ← Load real PaySim or generate synthetic demo
    preprocess.py           ← Clean, feature engineer, split, scale, SMOTE
    features.py             ← SelectKBest + RandomForest importance ranking
    train.py                ← LR, DT, RF, KNN, SVM, GaussianNB, IsolationForest
    evaluate.py             ← Precision/Recall/F1/PR-AUC, CV, bootstrap, Bayes
    regression.py           ← Simple, multiple, polynomial regression
    neural_net.py           ← MLP (relu / sigmoid / tanh) + loss curves
    cluster.py              ← KMeans, KMedoids, Agglomerative + dendrogram
    rules.py                ← Apriori association rules → fraud patterns
    make_figures.py         ← 6 documentation PNGs at 200 DPI
    predict.py              ← Score a single transaction (CLI + library)
  models/                   ← Saved .pkl files (best_model, scaler, etc.)
  reports/                  ← metrics.json, model_comparison.csv, feature_ranking.csv
  reports/figures/          ← PNG documentation graphs
  app/
    app.py                  ← Master unified single-page FinTech operations dashboard
    style.py                ← Custom CSS design system
    components.py           ← Reusable UI components
  run_all.py                ← Master pipeline runner
  requirements.txt
  README.md
  .gitignore
```

---

## 📦 Dataset

**PaySim** — Synthetic mobile money transactions (Kaggle).

| | |
|---|---|
| Source | https://www.kaggle.com/datasets/ealaxi/paysim1 |
| Filename | `PS_20174392719_1491204439457_log.csv` |
| Rows | ~6.36 million |
| Columns | step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud |

**Placement steps:**
1. Create a Kaggle account and accept dataset terms.
2. Download the CSV (≈ 470 MB).
3. Place it at `data/raw/PS_20174392719_1491204439457_log.csv`.
4. Run `python run_all.py` (without `--demo`).

> **No CSV? No problem.** Use `--demo` mode (see below) to test the entire project on 300k synthetic rows.

---

## ⚙️ Setup

```bash
# 1. Create virtual environment (Python 3.10+)
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the Pipeline

### Demo mode (no Kaggle download required)
```bash
python run_all.py --demo
```
Generates ~300k synthetic rows, runs the full pipeline (~10–15 minutes on a laptop), and marks all outputs as **DEMO DATA** in the dashboard.

### Real data mode (after placing the CSV)
```bash
python run_all.py
```

### Launch the unified dashboard
```bash
streamlit run app/app.py
```
Open http://localhost:8501 in your browser. All operational telemetry, 1-click test attack samples, real-time risk triage, animated radar scanner, model leaderboard, and syndicate rules are unified on **ONE single continuous page**.

### Score a single transaction (CLI)
```bash
python src/predict.py --type TRANSFER --amount 50000 --hour 2 \
                      --old_balance_orig 51000 --new_balance_orig 0
```

---

## 📊 Models & Metrics

| Model | Notes |
|---|---|
| **LogisticRegression** | MLE-based, class_weight=balanced |
| **DecisionTreeClassifier** | max_depth=12, balanced |
| **RandomForestClassifier** | 200 trees, often best |
| **KNN** | k=7, on working sample |
| **SVM (LinearSVC + Calibrated)** | Fast linear SVM with probabilities |
| **GaussianNB** | Bayesian baseline |
| **MLPClassifier** | Hidden (64,32); relu / logistic / tanh compared |
| **IsolationForest** | Anomaly score signal |
| **KMeans / KMedoids** | Cluster profiling |
| **Agglomerative** | Dendrogram visualization |
| **Apriori** | Fraud pattern rule mining |

**Best model selection:** highest Recall where Precision ≥ 0.80; fallback to best F1.

**No accuracy as primary metric** — we use Precision, Recall, F1, PR-AUC, ROC-AUC.

---

## 🎨 Dashboard Pages

| # | Page | What it shows |
|---|---|---|
| 1 | **Radar Home** | Animated CSS radar, 4 KPI cards, threat level |
| 2 | **Transaction Lab** | Single-transaction risk gauge + SHAP explanation |
| 3 | **Live Stream** | Simulated real-time feed with rolling probability chart |
| 4 | **Model Arena** | Race bars, polar comparison, confusion matrix, bootstrap CIs |
| 5 | **Cluster Galaxy** | 3D PCA scatter, cluster profiles, dendrogram |
| 6 | **Pattern Rules** | Apriori rules as cards + bubble chart |
| 7 | **Fraud Clock** | 24-hour polar bar, hour×type heatmap, polynomial curve |
| 8 | **About** | Overview, pipeline diagram, syllabus mapping |

---

## 📐 Feature Engineering

All features are derived from the original PaySim columns:

| Feature | Formula |
|---|---|
| `hour` | `(step - 1) % 24` |
| `day` | `(step - 1) // 24` |
| `isNight` | `hour < 8` |
| `balanceDiffOrig` | `oldbalanceOrg - newbalanceOrig` |
| `balanceDiffDest` | `newbalanceDest - oldbalanceDest` |
| `origEmptied` | `newbalanceOrig == 0 AND oldbalanceOrg > 0` |
| `errorOrig` | `newbalanceOrig + amount - oldbalanceOrg` |
| `errorDest` | `oldbalanceDest + amount - newbalanceDest` |
| `amountToBalanceRatio` | `amount / (oldbalanceOrg + 1)` |
| `type_*` | One-hot encoded transaction type |