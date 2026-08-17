# Observations — Assignment 1: Supervised ML Benchmark

All numbers below are pulled directly from `results/results_table.csv`,
`results/test_summary_ranked.csv`, and `results/experiment_plots/*.csv`
after running `src/train.py` and `src/experiments.py` with `RANDOM_SEED=42`.

## 1. Test-set results (all 7 models)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train time (s) | Inference (ms/sample) |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.707 | 0.578 | 0.380 | **0.458** | **0.742** | 0.033 | 0.0068 |
| KNN (k=5) | 0.663 | 0.481 | 0.393 | 0.433 | 0.622 | 0.020 | 0.0239 |
| SVM (RBF) | 0.703 | 0.576 | 0.337 | 0.425 | 0.716 | 2.300 | 0.1368 |
| Gradient Boosting | 0.692 | 0.544 | 0.344 | 0.422 | 0.726 | 0.597 | 0.0078 |
| XGBoost | 0.661 | 0.476 | 0.372 | 0.418 | 0.688 | 0.109 | 0.0076 |
| Random Forest | 0.693 | 0.549 | 0.329 | 0.411 | 0.726 | 1.280 | 0.0446 |
| Decision Tree | 0.684 | 0.532 | 0.276 | 0.363 | 0.723 | **0.031** | **0.0060** |

(Bold = best in column among the seven.)

## 2. Required observations

**Which model performed best on training data?**
XGBoost, by a wide margin — train F1 = 0.985, train accuracy = 0.991, train
ROC-AUC = 0.9996. This is close to memorization of the training set rather
than genuine signal.

**Which model performed best on validation data?**
Logistic Regression — validation F1 = 0.479, the highest of all seven
models on that split.

**Which model generalized best to the final test set?**
Logistic Regression again. Its train → test F1 barely moved (0.450 → 0.458,
effectively no gap) and it had the highest test ROC-AUC (0.742). In
contrast, XGBoost's F1 collapsed from 0.985 (train) to 0.418 (test), and
Random Forest's fell from 0.793 (train) to 0.411 (test) — both are large,
telltale generalization gaps.

**Which algorithm was most interpretable?**
Logistic Regression: every prediction reduces to a weighted sum of
coefficients, each with a clear sign and magnitude that maps to "this
feature increases/decreases churn odds." The Decision Tree is a close
second — a shallow tree can be drawn as an explicit if/then rule set — but
Logistic Regression's coefficients are simpler to communicate to
non-technical stakeholders without visual aids.

**Which algorithm was fastest at inference?**
Decision Tree — 0.0060 ms/sample, marginally faster than Logistic
Regression (0.0068 ms) and XGBoost (0.0076 ms). All three of these are
essentially O(depth) or O(features) lookups at prediction time. SVM is by
far the slowest (0.137 ms/sample) because RBF-kernel prediction requires
computing a kernel distance to every stored support vector.

**Which model would you choose if explainability were a requirement?**
**Logistic Regression.** It gives per-feature coefficients that can be
directly reported (e.g., "month-to-month contracts increase churn odds
by X%"), which is exactly the kind of statement a retention team or
regulator would want. A shallow Decision Tree is the fallback if the
audience prefers visual decision rules over coefficients.

**Which model would you choose if predictive performance were the primary
objective?**
On this benchmark, **Logistic Regression** again — it has the best test F1
*and* the best test ROC-AUC of all seven models, even before tuning. This
is a direct consequence of the dataset: churn was generated as a
(noisy) **linear** function of the features, so a linear model is
well-matched to the true decision boundary and the added flexibility of
tree ensembles mainly buys overfitting rather than better fit. In
Assignment 3, hyperparameter tuning (depth/estimator limits,
regularization, learning rate) is applied specifically to close this gap
for a boosting model, since properly regularized gradient boosting would
be expected to at least match Logistic Regression on this kind of tabular
problem.

**Did any model show signs of high bias or high variance?**
Yes, both are visible:
- **High variance (overfitting):** XGBoost (train F1 0.985 → test F1
  0.418), Random Forest (0.793 → 0.411), and to a lesser extent KNN with
  k=5 (train F1 0.610 → test F1 0.433). All three fit training-set noise
  that does not exist in unseen data.
- **High bias:** best illustrated in the dedicated depth experiment below
  — an overly constrained Decision Tree (`max_depth=2`) has limited
  capacity, though on this dataset it happened to still capture most of
  the usable signal (see below).

## 3. Required experiments — results and interpretation

### KNN: comparing values of K
| k | Train F1 | Validation F1 |
|---|---|---|
| 3 | 0.687 | 0.379 |
| 5 | 0.610 | 0.385 |
| 11 | 0.498 | 0.381 |
| 25 | 0.407 | 0.311 |
| 51 | 0.289 | 0.261 |

Small k (3) memorizes local noise (high train F1, mediocre validation F1 —
classic high variance). As k grows past ~11, both train and validation F1
decline steadily — the decision boundary becomes too smooth/global (high
bias). k=5 gives close to the best validation F1 in this range, which is
why it was used as the representative KNN model in the main comparison.

### Decision Tree: shallow vs. deep (also the underfit/overfit demonstration)
| Tree | Train F1 | Val F1 | Train Acc | Val Acc | Train−Val Acc gap |
|---|---|---|---|---|---|
| Shallow (depth=2) | 0.470 | **0.521** | 0.699 | 0.722 | −0.023 |
| Moderate (depth=6) | 0.435 | 0.369 | 0.731 | 0.693 | +0.039 |
| Deep (depth=None) | **1.000** | 0.436 | **1.000** | 0.624 | **+0.376** |

The unrestricted tree (`depth=None`) is the clearest **overfitting** example
in the whole benchmark: it perfectly memorizes the training set (F1 = 1.0)
but validation accuracy drops 37.6 points versus training accuracy — a
textbook high-variance signature. The `depth=2` tree is the intended
**underfitting** example (very limited capacity), though in this dataset it
happened to land close to the strongest split (`contract_type`) early,
so its validation score is competitive — a useful reminder that "shallow"
and "underfit" are not always synonymous; capacity has to be judged
against how much true signal exists to capture, not depth alone.

### Decision Tree vs. Random Forest
| Model | Val F1 | Val Accuracy | Val ROC-AUC |
|---|---|---|---|
| Decision Tree | 0.369 | 0.693 | 0.714 |
| Random Forest | 0.425 | 0.703 | 0.733 |

Averaging many de-correlated trees (bagging) reduces the variance of a
single Decision Tree: Random Forest improves validation F1 by ~15% relative
and ROC-AUC by ~2 points, at the cost of ~40x longer training time and no
loss of validation performance — the classic bagging trade-off (variance
down, bias roughly unchanged, cost up).

### Random Forest vs. a boosting model (Gradient Boosting)
| Model | Val F1 | Val ROC-AUC |
|---|---|---|
| Random Forest | 0.425 | 0.733 |
| Gradient Boosting | 0.441 | 0.739 |

Gradient Boosting edges out Random Forest on both metrics. Boosting builds
trees sequentially to correct the previous ensemble's residual errors
(bias reduction), whereas bagging trains trees independently and averages
them (variance reduction) — here, boosting's error-correcting approach
generalizes marginally better on this dataset, though the margin is small
enough that either would be a defensible baseline pending tuning.

### SVM: with vs. without feature scaling
| Setup | Val F1 | Val Accuracy | Val ROC-AUC |
|---|---|---|---|
| SVM with scaling | 0.392 | 0.703 | 0.717 |
| SVM without scaling | **0.000** | 0.673 | 0.513 |

This is the starkest result in the whole benchmark. Without scaling, the
RBF kernel's distance computation is dominated by `total_charges` (values
in the hundreds/thousands) while `senior_citizen` (0/1) becomes
essentially invisible — the model collapses to predicting the majority
class for every sample (F1 = 0, ROC-AUC ≈ 0.51, i.e. random). This
directly demonstrates why **any distance- or margin-based algorithm (KNN,
SVM) requires feature scaling**, while tree-based models (Decision Tree,
Random Forest, boosting) are scale-invariant by construction (they split
on thresholds per feature, not on distances).

### Effect of class imbalance (~33% positive class)
| Setup | Val Precision | Val Recall | Val F1 | Val Accuracy |
|---|---|---|---|---|
| Logistic Regression (default) | 0.608 | 0.395 | 0.479 | 0.719 |
| Logistic Regression (`class_weight='balanced'`) | 0.496 | **0.755** | **0.599** | 0.669 |
| Random Forest (default) | 0.576 | 0.337 | 0.425 | 0.703 |
| Random Forest (`class_weight='balanced'`) | 0.525 | 0.640 | 0.577 | 0.693 |

With the default (unweighted) loss, both models favor the majority
(no-churn) class — recall on the churn class is only 33–40%, meaning the
majority of actual churners are missed. Re-weighting the loss to penalize
minority-class errors more heavily (`class_weight='balanced'`) roughly
doubles recall (churners correctly caught) for both models, at the cost of
some precision and overall accuracy. For a churn-prevention use case where
missing an at-risk customer is more costly than a false alarm, the
balanced setting is very likely the better business trade-off despite the
lower raw accuracy — a good example of why accuracy alone is a poor metric
for an imbalanced classification target.

## 4. Model-selection reasoning (final conclusion)

If forced to ship one model from this benchmark as-is: **Logistic
Regression**, because it is simultaneously the best generalizing model
(smallest train/test gap), the top performer on both validation and test
F1/ROC-AUC, the fastest to train, and the most explainable — a rare case
where the "best" model on every axis is also the simplest one. This is a
direct consequence of the (roughly linear) way churn probability was
constructed in this dataset; on a dataset with genuinely non-linear
feature interactions, a properly tuned boosting model would be the
expected winner. That gap-closing tuning is exactly what Assignment 3
performs on a chosen model via `GridSearchCV`/`RandomizedSearchCV`.

If the business objective specifically prioritizes **catching as many
churners as possible** (recall) over overall accuracy, the
`class_weight='balanced'` Logistic Regression variant (recall 0.755,
F1 0.599 on validation) would be the more appropriate choice, since it
directly targets the imbalanced-class trade-off discussed above.

## 5. Unsuccessful / notable experiments worth documenting

- **SVM without scaling** is included deliberately as a "failed"
  configuration (F1 = 0) to make the scaling requirement concrete rather
  than asserted — see the SVM section above.
- **Unrestricted-depth Decision Tree** is likewise a deliberately "bad"
  configuration used only to illustrate overfitting, not as a candidate
  for deployment.
- Default (unweighted) models on this ~33%-positive-class dataset
  systematically under-predict the minority class; this was only visible
  by explicitly inspecting per-class recall, not by looking at accuracy
  alone — accuracy differences between default and balanced settings were
  only 2–5 points, masking a 2x swing in recall.

## 6. Answers to selected review questions

**Why should the test set not be used while tuning models?**
Any decision made by looking at test-set performance (choosing
hyperparameters, model family, feature engineering) leaks information from
the test set into the model-selection process, so the final reported test
score no longer reflects true out-of-sample performance — it becomes an
optimistic, biased estimate. The test set must be touched exactly once, at
the very end.

**What is data leakage and where could it occur in this project?**
Data leakage is any situation where information that would not be
available at genuine prediction time influences training. In this project
the main risk is fitting the `SimpleImputer`/`StandardScaler`/
`OneHotEncoder` on the full dataset (or on train+validation combined)
before splitting — that would let statistics from validation/test rows
(e.g., the median used for imputation) leak into training. This is why
`build_preprocessor()` is fit only inside each `Pipeline.fit(X_train, ...)`
call, never on the full dataset.

**Why do KNN and SVM commonly require feature scaling?**
Both are distance/margin-based: KNN classifies by literal Euclidean (or
similar) distance to neighbors, and SVM's RBF kernel is a function of
squared distance between points. Unscaled features with larger numeric
ranges (e.g., `total_charges` in the hundreds vs. `senior_citizen` as 0/1)
dominate the distance calculation, as demonstrated empirically in the SVM
scaling experiment above (F1 dropped from 0.392 to 0.000).

**How is Random Forest different from a single Decision Tree?**
Random Forest trains many Decision Trees on bootstrapped samples of the
data (bagging) with an additional random subset of features considered at
each split, then averages/votes their predictions. This de-correlates the
trees' individual errors, reducing the variance of a single (often
overfit) Decision Tree without a corresponding cost in bias.

**What is the conceptual difference between bagging and boosting?**
Bagging (Random Forest) trains independent models in parallel on
resampled data and averages them to reduce variance. Boosting (Gradient
Boosting, XGBoost) trains models sequentially, where each new model
focuses on the residual errors of the current ensemble, primarily
reducing bias — at higher risk of overfitting if not regularized, as seen
with XGBoost's near-perfect training fit in this benchmark.

**When could Logistic Regression be preferable to XGBoost even with a
slightly lower score?** — and here it did not even have a lower score:
Logistic Regression outperformed XGBoost on every test metric while being
~3x faster to train and trivially explainable. More generally, Logistic
Regression is preferable when the relationship is close to linear, when
the model needs to be explained to regulators/stakeholders, when the
dataset is small (fewer parameters to overfit), or when inference latency
budgets are tight.
