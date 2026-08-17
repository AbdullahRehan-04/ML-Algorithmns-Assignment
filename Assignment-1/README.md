# Assignment 1 — Supervised ML Benchmark

A complete, reproducible classification benchmark on a **telecom customer
churn** dataset: seven algorithms trained and compared under identical
preprocessing, with the required experiments (KNN-k, tree depth, DT vs RF,
RF vs boosting, SVM scaling, class imbalance) run as isolated, documented
comparisons.

## 1. Dataset

`data/telecom_churn.csv` — **6,025 rows** (6,000 after de-duplication), 14
features + binary target `churn`.

The dataset is **synthetically generated** (`src/generate_data.py`) rather
than downloaded, so it contains no real or confidential customer data, but
the feature-to-target relationships are built in on purpose (contract type,
tenure, support calls, internet service, etc. genuinely drive churn
probability, with noise layered on top) so that model comparisons are
meaningful rather than random.

| Type | Features |
|---|---|
| Numerical | `tenure_months`, `monthly_charges`, `total_charges`, `age`, `num_support_calls` |
| Categorical | `contract_type`, `internet_service`, `payment_method`, `tech_support`, `paperless_billing`, `has_partner`, `has_dependents`, `senior_citizen` |
| Target | `churn` (0/1, ~33% positive class — moderately imbalanced) |

The raw file also contains **25 duplicate rows** and **missing values** in
`total_charges`, `age`, and `tech_support`, mirroring real-world data quality
issues that the pipeline must handle.

## 2. Project structure

```
assignment_1_supervised/
├── data/
│   └── telecom_churn.csv
├── notebooks/
│   └── assignment_1_analysis.ipynb   # walkthrough notebook (calls src/)
├── src/
│   ├── generate_data.py   # builds the synthetic dataset
│   ├── data_prep.py       # inspection, cleaning, splitting, preprocessing pipeline
│   ├── utils.py           # metrics, timing, plotting helpers
│   ├── train.py           # trains & evaluates all 7 required models
│   └── experiments.py     # runs all required comparison experiments
├── results/
│   ├── results_table.csv               # every model x every split x every metric
│   ├── test_summary_ranked.csv         # test-set only, ranked by F1
│   ├── confusion_matrices/*.png        # one per model, test set
│   ├── roc_curves_test.png
│   ├── training_time_comparison.png
│   ├── inference_time_comparison.png
│   ├── test_f1_comparison.png
│   ├── experiments_summary.json
│   └── experiment_plots/               # CSVs + PNGs for each required experiment
├── requirements.txt
├── README.md
└── observations.md         # model-selection reasoning & written conclusions
```

## 3. Setup

```bash
cd Assignment-1
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Running the project

Run everything from the `assignment_1_supervised/` root, in order:

```bash
# 1. Generate the dataset (already included in data/, but reproducible)
python src/generate_data.py

# 2. Train all 7 models, evaluate on train/val/test, save results + plots
python src/train.py

# 3. Run the required comparison experiments (KNN-k, tree depth, etc.)
python src/experiments.py
```

Or open `notebooks/assignment_1_analysis.ipynb` for an interactive
walkthrough that calls the same `src/` functions and displays all plots
inline.

All randomness is controlled by `RANDOM_SEED = 42`, set consistently across
data generation, splitting, and every model, so results are reproducible.

## 5. What each script does

- **`generate_data.py`** — builds the synthetic dataset with realistic
  feature/target relationships, injects missing values and duplicates.
- **`data_prep.py`** — `load_and_inspect()` prints shape/dtypes/target
  distribution/nulls/duplicates and drops duplicates + the ID column;
  `split_data()` creates a **60/20/20 stratified** train/val/test split with
  a fixed seed; `build_preprocessor()` returns a `ColumnTransformer` (median
  imputation + `StandardScaler` for numeric, most-frequent imputation +
  one-hot encoding for categorical) used inside every model's `Pipeline` so
  the **test set is never fit on**.
- **`utils.py`** — timing wrappers, the full metric set (accuracy,
  precision, recall, F1, ROC-AUC, confusion matrix), and plotting helpers.
- **`train.py`** — trains Logistic Regression, KNN, Decision Tree, Random
  Forest, Gradient Boosting, XGBoost, and SVM, each inside an identical
  preprocessing pipeline; evaluates on train/val/test; saves the results
  table, confusion matrices, ROC curves, and timing bar charts.
- **`experiments.py`** — runs every explicitly required experiment as an
  isolated comparison (see `observations.md` for the findings).

## 6. Key results (test set, ranked by F1)

See `results/test_summary_ranked.csv` for exact numbers and
`observations.md` for full interpretation. Logistic Regression had the best
F1 and ROC-AUC on the held-out test set; the more complex ensemble/boosting
models (XGBoost, Random Forest) overfit visibly (near-perfect training
scores that did not transfer to validation/test).
