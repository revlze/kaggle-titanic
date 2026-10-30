# Titanic experiment notes — 2026-10-30

I summarize ten completed local runs and eight of my Kaggle submissions here.
I keep the resolved starting configurations, effective final parameters, fold
scores, versions, source hashes and local submission hashes in the
[machine-readable snapshot](experiment-results.json). My current config uses a
single tuned KNN model; I use each run's saved config to describe earlier experiments.

## What the results support

My fixed logistic-regression baseline and its tuned version share my highest
recorded public accuracy, **0.77272**. None of my other submissions in this
comparison exceeds it. I keep this simple baseline as my reference, without
claiming that logistic regression is generally superior on Titanic.

With a single tuned XGBoost model, I reached **83.56% ± 3.24 pp** local CV and
**0.75837** public accuracy. With a five-seed soft ensemble, I reached
**82.87% ± 2.62 pp** and **0.75598**, respectively. I tuned the two procedures
separately and selected different parameters, so I cannot attribute their
difference to averaging alone.
Equal public scores also need not imply identical predictions.

With tuned K-nearest neighbors, I reached **81.32% ± 3.13 pp** local CV and
**0.75837** public accuracy (submission **57799934**). My final model uses 23
neighbors, uniform weights and Minkowski distance with `p=2` (Euclidean), after numeric
scaling. The separate final selection CV is **82.58%**, which selected parameters
and is not an independent quality estimate.

![My eight submissions: local CV mean and fold SD versus Kaggle public accuracy.](figures/accuracy-comparison.svg)

Both series use accuracy, but their evaluation populations differ. Error bars
show local fold SD with `ddof=0`; they are not confidence intervals. There are no
public-score error bars, private scores or significance estimates in my results.
The [PNG version](figures/accuracy-comparison.png) contains the same plotted values.

## Source and matching boundary

I recorded these scores using `kaggle competitions submissions -c titanic` on
**2026-10-30**. All eight of my submissions below were marked
`SubmissionStatus.COMPLETE`; every private score was blank. I copied the scores
and timestamps directly from the CLI output and left its timezone unspecified.

| Submission ref | CLI date | Description | Public accuracy |
|---|---|---|---:|
| 57799934 | 2026-10-30 08:52:04.537000 | `knn_1_tuned` | 0.75837 |
| 57798445 | 2026-10-30 07:57:07.790000 | `lgbm_1_tuned` | 0.76076 |
| 57797886 | 2026-10-30 07:38:36.543000 | `xgb_1_tuned` | 0.75837 |
| 57797655 | 2026-10-30 07:32:17.703000 | `xgb_5seeds_tuned` | 0.75598 |
| 57764463 | 2026-10-29 08:13:10.070000 | `lr_v1_tuned` | 0.77272 |
| 57764134 | 2026-10-29 08:00:46.317000 | `gb_v1` | 0.76076 |
| 57764095 | 2026-10-29 07:59:23.090000 | `lr_v1` | 0.77272 |
| 57760541 | 2026-10-29 05:51:48.140000 | `gb_v1_optuna 29oct` | 0.75837 |

I matched each submission description to a unique saved local experiment. I
associate `gb_v1_optuna 29oct` with my **20-trial** `gb_v1_optuna` run; the 50-
and 60-trial runs are separate experiments. These earlier matches use saved metadata.
For KNN, my successful submit command names
`artifacts/knn_1_tuned/20261030T085106_102763Z/submission.csv` and returned
reference 57799934. This links that submission directly to the local run.
I keep local CSV hashes for traceability; the CLI history does not verify those
hashes against the uploaded files. All eight submissions used the filename `submission.csv`.

I save local runs under `artifacts/<experiment>/<run_id>/`:

| Experiment identifier | Exact run ID | Associated submission |
|---|---|---:|
| `lr_v1` | `20261029T075746_383922Z` | 57764095 |
| `lr_v1_tuned` | `20261029T081116_571264Z` | 57764463 |
| `gb_v1` | `20261029T054427_429190Z` | 57764134 |
| `gb_v1_optuna` | `20261029T054822_685707Z` | 57760541 |
| `xgb_1_tuned` | `20261030T073735_087383Z` | 57797886 |
| `xgb_5seeds_tuned` | `20261030T072714_855205Z` | 57797655 |
| `lgbm_1_tuned` | `20261030T075559_571494Z` | 57798445 |
| `knn_1_tuned` | `20261030T085106_102763Z` | 57799934 |
| `gb_optuna_50trials` | `20261029T060848_617478Z` | Not recorded |
| `gb_optuna_60trials` | `20261029T070239_722427Z` | Not recorded |

I keep `artifacts/` outside Git. I publish selected run metadata in the JSON
snapshot; model checkpoints, full OOF files and trial tables stay local.
Reproduction also requires the original Titanic CSVs.

## Common local evaluation

I use the same labeled data, `base_title_family` features, seed **1027309**
(`0xFACED`), 712 development rows, 179 excluded previously viewed holdout rows
and five outer-fold splits across all ten runs. I checked that the saved data
hashes, development/holdout indices and outer-fold identities match exactly.
The saved test-data hashes also match.

| Procedure | Parameter choice | Quality estimate |
|---|---|---|
| Fixed | YAML parameters | Five-fold stratified CV on development |
| Tuned | Optuna on three inner folds of each outer train | Accuracy on the untouched outer validation fold |
| Final fit | Separate development-only search, if enabled; fresh refit on all 891 labels | No new independent score after refitting |

Each development passenger has one OOF prediction. Imputation, encoding and any
scaling are learned inside the relevant training fold. I enable scaling for
logistic regression and KNN, and disable it for my boosting runs. The final tuned
pipelines use a separate three-fold selection search on development, then fit all 891 labeled
rows, including the former holdout. The test CSV is used for inference only.

| Model / setup | OOF accuracy, % | Final selection CV, % | Completed searches × trials |
|---|---:|---:|---|
| Logistic regression (baseline) | 82.72 | Not tuned | 0 |
| Logistic regression + Optuna | 82.30 | 83.14 | 6 × 60 |
| Gradient boosting (baseline) | 82.87 | Not tuned | 0 |
| Gradient boosting + Optuna | 83.01 | 83.99 | 6 × 20 |
| XGBoost + Optuna | 83.57 | 84.69 | 6 × 60 |
| XGBoost ensemble (5 models) | 82.87 | 84.27 | 6 × 60 |
| LightGBM + Optuna | 82.58 | 84.69 | 6 × 60 |
| K-nearest neighbors + Optuna | 81.32 | 82.58 | 6 × 60 |

The six searches are five outer-fold searches plus one separate final selection
search. Every saved trial table contains the configured number of `COMPLETE`
trials. The final selection score chose parameters and is not an independent
accuracy estimate. Pooled OOF accuracy and mean fold accuracy can differ slightly
because fold sizes are not identical.

I also completed two further GB runs. I have no recorded public scores for them
in this comparison, so I report only their local results:

| Model / setup | Local CV, % ± fold SD | OOF accuracy, % | Final selection CV, % |
|---|---:|---:|---:|
| Gradient boosting + Optuna (50 trials) | 83.57 ± 3.26 | 83.57 | 84.13 |
| Gradient boosting + Optuna (60 trials) | 83.85 ± 3.14 | 83.85 | 84.27 |

I cannot conclude from these runs that increasing trials improves public accuracy.
I stratify by row rather than grouping by family or ticket, and I reuse the same
development folds for model and feature comparisons. This limits what I can
claim about a selected winner. My historical holdout is not a fresh final test.

## Implemented versus measured

I can configure logistic regression, gradient boosting, CatBoost, XGBoost,
LightGBM and K-nearest neighbors through the YAML runner. I completed a full KNN
run and submitted it to Kaggle, using a single model with numeric scaling. Seed
ensembles for models with random seeds support soft voting, hard voting and stacking.
Of the ensemble methods, I have a full local run and public score only for the
**five-seed XGBoost soft ensemble** in this comparison. I have no full comparative
results for CatBoost, hard voting or stacking in this table.

Soft/hard voting uses sklearn VotingClassifier. Stacking trains its meta-model
on OOF probabilities with preprocessing inside every base pipeline. Its own OOF
folds are needed even without Optuna. The current ensemble code does not draw
bootstrap samples for whole models; row subsampling inside a boosting library
is a separate mechanism.

## Verification and reproduction limits

I checked the saved OOF row identities, absence of the excluded holdout, per-fold
accuracies, mean/SD, pooled OOF accuracy, completed trial counts, and every local
submission's 418-row ordering and binary labels. I assembled this comparison from
saved runs and the CLI history above, without retraining models, reloading
checkpoints for inference or making new submissions. I copied checkpoint-consistency
flags from the original run reports and have not reverified them for this comparison.

To reproduce a historical setup, use that run's resolved `configuration` in the
JSON snapshot rather than today's mutable `config.yaml`. Keep its data, split,
preprocessing, ensemble and search settings together, then use the existing
runner described in the [README](../README.md#run-and-configure). The snapshot
records library versions and hashes; it does not bundle the original environment
or guarantee identical results across platforms and later dependency versions.
