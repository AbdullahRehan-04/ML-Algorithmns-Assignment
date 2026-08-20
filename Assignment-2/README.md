# Customer Segmentation — KMeans, DBSCAN & PCA

An unsupervised-learning project that groups customers based on behavioral
features. The goal is to **understand how each clustering algorithm behaves**
on real customer data — not to assume every discovered cluster automatically
represents a real business segment.

**Dataset:** [Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis) (Kaggle)

## Project structure

```
Assignment-2/
├── data/
│   └── marketing_campaign.csv          <- put the downloaded Kaggle CSV here
├── data_preparation.py                 <- data-preparation script (deliverable)
├── notebooks/
│   └── customer_segmentation_analysis.ipynb   <- KMeans, DBSCAN, PCA analysis
├── outputs/                            <- generated plots + CSVs land here
├── observations.md                     <- written answers to required questions
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## How to run

1. **Get the dataset.** Download it from Kaggle (login required) — it comes
   as `marketing_campaign.csv`, semicolon-separated. Place it at:
   `data/marketing_campaign.csv`

2. **Run data preparation:**
   ```bash
   python data_preparation.py
   ```
   This inspects the raw data, handles missing/invalid values, engineers the
   behavioral features used for clustering, scales them, and writes:
   - `data/prepared_data.csv` (cleaned + engineered, unscaled — used for
     profiling clusters in real units, e.g. actual dollars/days)
   - `data/prepared_data_scaled.csv` (the same rows, standardized — fed into
     KMeans/DBSCAN/PCA)

3. **Run the analysis notebook:**
   ```bash
   jupyter notebook notebooks/customer_segmentation_analysis.ipynb
   ```
   Run all cells top to bottom. It reads the two prepared CSVs and produces
   every remaining deliverable (KMeans elbow/silhouette, DBSCAN sweep, PCA
   variance, visualizations, and cluster profile tables), saving plots and
   CSVs into `outputs/`.

## Features used for clustering

The raw dataset's ~28 columns aren't behavioral metrics by themselves — several
are combined/engineered in `data_preparation.py` into:

| Feature | Meaning |
|---|---|
| `Income` | Household income |
| `Recency` | Days since last purchase |
| `Total_Spend` | Sum of spend across all product categories |
| `Total_Purchases` | Sum of purchases across all channels (web, store, catalog, deals) |
| `Age` | Derived from birth year |
| `Customer_Tenure_Days` | Days since the customer enrolled |
| `NumWebVisitsMonth` | Website visits per month |
| `Family_Size` | Kids + teens + partner status |

## Data preparation summary

- Dropped rows with missing `Income` (~1% of data) rather than imputing —
  income is a core feature and a small drop is safer than guessing it.
- Removed rows with clearly invalid `Year_Birth` (e.g. 1899) and an extreme
  `Income` outlier (666,666 vs. a normal range of ~$5K–$120K).
- Dropped identifier/non-behavioral columns (`ID`, raw birth year, enrollment
  date, and the constant `Z_CostContact`/`Z_Revenue` columns) before
  clustering, since these carry no meaningful distance signal.
- Scaled all clustering features with `StandardScaler` — see `observations.md`
  for why this matters for distance-based algorithms.

## Results summary (this run, on the real dataset)

- 2,240 raw customers → **2,212** after cleaning (dropped 24 missing-Income
  rows + a handful of birth-year/income outliers)
- **KMeans: K=4** selected via elbow analysis (silhouette alone peaked at a
  less useful K=2) — four balanced, interpretable segments (527–582
  customers each)
- **DBSCAN: eps=1.3, min_samples=10** → 3 clusters, **19.7% of customers
  flagged as noise** — useful for outlier detection, not as a standalone
  segmentation scheme (one dominant cluster held 99% of non-noise points)
- **PCA:** first 2 components explain **54.6%** of total variance

Full numbers, per-cluster profiles, and the reasoning behind each choice are
in `observations.md`.

## Deliverables checklist

- [x] Data-preparation notebook/script — `data_preparation.py`
- [x] KMeans inertia and silhouette analysis — notebook §2
- [x] DBSCAN experiments (eps/min_samples sweep, noise detection) — notebook §4–5
- [x] PCA explained-variance analysis — notebook §6
- [x] Cluster visualizations — notebook §7, saved to `outputs/`
- [x] Cluster profile table — notebook §3/9, saved to `outputs/`
- [x] README and observations — this file + `observations.md`
