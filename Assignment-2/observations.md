# Observations

Results below come from actually running the pipeline on the real Kaggle
"Customer Personality Analysis" dataset (`marketing_campaign.csv`, 2,240 raw
rows → 2,212 after cleaning).

## How did scaling change the clustering result?

Without scaling, `Income` (range ≈ $1,730–$162,397) and `Total_Spend` (range
≈ $0–$2,525) would dominate the Euclidean distance calculation entirely,
next to features like `Family_Size` (range ~1–5) or `NumWebVisitsMonth`
(range ~0–20). Unscaled KMeans would essentially reduce to sorting customers
by income/spend and largely ignore recency, tenure, and engagement. After
applying `StandardScaler` (mean 0, std 1 per feature — confirmed in notebook
§1), every feature contributes proportionally, and the resulting clusters
mix spending power, recency, tenure, and family structure rather than being
a disguised income ranking.

## How was the final K selected?

The raw silhouette-vs-K sweep (notebook §2) actually peaks at **K=2**
(silhouette ≈ 0.272), which is a real signal — one dominant axis (income/
spend) creates a clean two-way split — but it's too coarse to be useful for
segmentation. Silhouette then declines steadily from K=3 onward (0.201 →
0.165 → 0.158 → ...), with no second local maximum. So instead of maximizing
silhouette blindly, we used the **elbow in the inertia curve**: the
percentage drop in inertia from K=3→4 is 9.2%, dropping further to 6.0% at
K=4→5 and continuing to flatten afterward — the curve visibly bends around
**K=4**. We selected **K=4** as the practical balance between a compact model
and enough granularity for four distinguishable business segments.

## What characteristics distinguish each KMeans cluster?

From the cluster profile table (`outputs/cluster_profile_kmeans.csv`):

| Cluster | Income | Total_Spend | Total_Purchases | Age | Family_Size | Count | Read |
|---|---|---|---|---|---|---|---|
| 0 | $59.6K | $912 | 22.3 | 49 | 2.7 | 534 | Solid mid-to-high spenders, above-average purchase count |
| 1 | $29.1K | $102 | 7.9 | 37 | 2.6 | 582 | Lower income, low spend — budget-conscious / lower-engagement segment |
| 2 | $77.7K | $1,325 | 20.3 | 46 | **1.8** | 527 | Highest income & spend, smaller households, low web visits (2.3/mo) — likely in-store/high-value shoppers who don't need to browse online much |
| 3 | $44.4K | $173 | 10.0 | 49 | **3.2** | 569 | Larger families, moderate income, low spend per capita — newer accounts (tenure 194 days, notably shorter than other clusters) |

Recency is fairly flat across all four clusters (48–52 days), so it isn't a
strong differentiator here — spend, income, and family size do most of the
separating work.

## How sensitive was DBSCAN to eps and min_samples?

Very. The sweep across `eps ∈ {0.5,...,2.0}` × `min_samples ∈ {3,5,8,10}`
(notebook §4) showed the familiar pattern: small `eps` fragments the data
into many tiny clusters with heavy noise, while large `eps` collapses almost
everything into one giant cluster. The configuration selected —
**eps = 1.3, min_samples = 10** — was the best silhouette among configs
producing a business-comparable number of clusters (3), but note its
silhouette (0.032) is far lower than KMeans' (0.165 at K=4), and the cluster
sizes are extremely unbalanced: one cluster holds 1,763 of the 1,776
non-noise points, with two tiny 6–7-point clusters. This is a real,
important finding in itself (see final question below).

## Which points were considered noise, and why might that be useful?

**436 of 2,212 customers (19.7%)** were flagged as noise at the selected
configuration — a notably large fraction. Looking at the two small non-noise
DBSCAN clusters found alongside the noise, they represent oddities like very
high spend combined with very low recency (cluster 2: Recency ≈ 9 days,
Total_Spend ≈ $1,759) or unusually high income with very low family size
and very low web visits (cluster 1). The noise pool itself is likely full of
similarly atypical customers — extreme spenders, brand-new low-activity
accounts, or unusual combinations that don't fit any dense region. In
practice, businesses might route these ~20% of customers to manual review or
a separate "unclassified — investigate" bucket rather than lumping them into
one of the four main KMeans segments' messaging.

## How much variance did the first two PCA components explain?

**54.6%** (notebook §6). This is a moderate amount — a majority, but well
under a number like 80% that would let us fully trust the 2D scatter plot.
Nearly half the original 8-dimensional variance is compressed away in the
PCA projection used for visualization, so two points that look adjacent in
the PCA plot could genuinely differ on a dimension (e.g. `Family_Size` or
`NumWebVisitsMonth`) that isn't well represented in PC1/PC2. The plots in
§7 should be read as a helpful summary, not proof of exact cluster shape.

## Which clustering algorithm was more useful for this dataset, and why?

**KMeans (K=4) was clearly more useful for actual segmentation** here. It
produced four reasonably balanced clusters (527–582 customers each) with
interpretable, distinct profiles (see table above) and a meaningfully higher
silhouette score (0.165 vs. DBSCAN's 0.032). DBSCAN, by contrast, collapsed
almost all non-noise customers (1,763 of 1,776) into a single large cluster
and flagged nearly 1 in 5 customers as noise — useful as an **outlier-
detection pass** to surface unusual customers worth investigating
individually, but not useful on its own as a segmentation scheme for this
dataset. The most practical workflow: run DBSCAN first to flag the ~20% of
atypical customers for separate handling, then use KMeans's four segments to
drive marketing strategy for the remaining ~80%.
