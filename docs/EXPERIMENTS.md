# Titanic experiment notes — 2026-10-30

This update connects ten completed local runs with eight author-reported Kaggle
submissions. The [machine-readable snapshot](experiment-results.json) retains
the resolved starting configurations, effective final parameters, fold scores,
versions, source hashes and local submission hashes. The working config is a
single tuned KNN model; it does not describe every earlier experiment.

## What the results support

The fixed logistic-regression baseline and its tuned version share the highest
reported public accuracy, **0.77272**. None of the supplied alternative submissions
exceeds it. This supports keeping the simple baseline as a reference; it does
not prove that logistic regression is generally superior on Titanic.

The tuned single XGBoost run records **83.56% ± 3.24 pp** local CV and **0.75837**
public accuracy. Its five-seed soft ensemble records **82.87% ± 2.62 pp** and
**0.75598**, respectively. These are separately tuned procedures with different
selected parameters, not an isolated comparison of averaging with fixed models.
Equal public scores also need not imply identical predictions.

Tuned K-nearest neighbors records **81.32% ± 3.13 pp** local CV and **0.75837**
public accuracy (submission **57799934**). Its final model uses 23 neighbors,
uniform weights and Minkowski distance with `p=2` (Euclidean), after numeric
scaling. The separate final selection CV is **82.58%**, which selected parameters
and is not an independent quality estimate.

![Eight submissions: local CV mean and fold SD versus author-reported public accuracy.](figures/accuracy-comparison.svg)

Both series use accuracy, but their evaluation populations differ. Error bars
show local fold SD with `ddof=0`; they are not confidence intervals. There are no
public-score error bars, private scores or significance estimates in the evidence.
The [PNG version](figures/accuracy-comparison.png) contains the same plotted values.

## Source and matching boundary

The project author supplied the output of
`kaggle competitions submissions -c titanic` on **2026-10-30**. The eight rows
below were all marked `SubmissionStatus.COMPLETE`; every private score was blank.
These scores were transcribed, not independently fetched during this update.
Timestamps are preserved as displayed; their timezone is not inferred.

| Submission ref | CLI date, as supplied | Description | Public accuracy |
|---|---|---|---:|
| 57799934 | 2026-10-30 08:52:04.537000 | `knn_1_tuned` | 0.75837 |
| 57798445 | 2026-10-30 07:57:07.790000 | `lgbm_1_tuned` | 0.76076 |
| 57797886 | 2026-10-30 07:38:36.543000 | `xgb_1_tuned` | 0.75837 |
| 57797655 | 2026-10-30 07:32:17.703000 | `xgb_5seeds_tuned` | 0.75598 |
| 57764463 | 2026-10-29 08:13:10.070000 | `lr_v1_tuned` | 0.77272 |
| 57764134 | 2026-10-29 08:00:46.317000 | `gb_v1` | 0.76076 |
| 57764095 | 2026-10-29 07:59:23.090000 | `lr_v1` | 0.77272 |
| 57760541 | 2026-10-29 05:51:48.140000 | `gb_v1_optuna 29oct` | 0.75837 |

Each description has a unique matching saved local experiment. The GB description
`gb_v1_optuna 29oct` is associated with the **20-trial** `gb_v1_optuna` run, not
the later 50- or 60-trial runs. Uploaded CSV bytes were not supplied, so these
associations are based on metadata rather than proof that Kaggle scored a local
file with the recorded hash. For KNN, the author also supplied the successful
submit command naming
`artifacts/knn_1_tuned/20261030T085106_102763Z/submission.csv` and the returned
reference 57799934. This identifies the local run explicitly; remote uploaded
bytes were not independently verified. All supplied filenames were `submission.csv`.

Local run directories follow `artifacts/<experiment>/<run_id>/`:

| Experiment | Exact run ID | Associated submission |
|---|---|---:|
| `lr_v1` | `20261029T075746_383922Z` | 57764095 |
| `lr_v1_tuned` | `20261029T081116_571264Z` | 57764463 |
| `gb_v1` | `20261029T054427_429190Z` | 57764134 |
| `gb_v1_optuna` | `20261029T054822_685707Z` | 57760541 |
| `xgb_1_tuned` | `20261030T073735_087383Z` | 57797886 |
| `xgb_5seeds_tuned` | `20261030T072714_855205Z` | 57797655 |
| `lgbm_1_tuned` | `20261030T075559_571494Z` | 57798445 |
| `knn_1_tuned` | `20261030T085106_102763Z` | 57799934 |
| `gb_optuna_50trials` | `20261029T060848_617478Z` | Not provided |
| `gb_optuna_60trials` | `20261029T070239_722427Z` | Not provided |

`artifacts/` is ignored by Git. Its paths identify local evidence, not downloadable
checkpoints or working GitHub links. The documentation snapshot keeps
the selected metadata available without publishing models, full OOF files or
trial tables. The raw data must still be obtained separately.

## Common local evaluation

All ten runs have the same labeled-data hash, `base_title_family` features,
seed **1027309** (`0xFACED`), 712 development indices, 179 excluded previously
viewed holdout indices and exact five outer-fold identities. The recorded test
data hashes also agree. These equalities were checked directly rather than
inferred from similarly named protocols.

| Procedure | Parameter choice | Quality estimate |
|---|---|---|
| Fixed | YAML parameters | Five-fold stratified CV on development |
| Tuned | Optuna on three inner folds of each outer train | Accuracy on the untouched outer validation fold |
| Final fit | Separate development-only search, if enabled; fresh refit on all 891 labels | No new independent score after refitting |

Each development passenger has one OOF prediction. Imputation, encoding and any
scaling are learned inside the relevant training fold. Logistic regression and KNN use
scaling; the recorded boosting runs do not. The final tuned pipelines use a
separate three-fold selection search on development, then fit all 891 labeled
rows, including the former holdout. The test CSV is used for inference only.

| Experiment | OOF accuracy, % | Final selection CV, % | Completed searches × trials |
|---|---:|---:|---|
| `lr_v1` | 82.72 | Not tuned | 0 |
| `lr_v1_tuned` | 82.30 | 83.14 | 6 × 60 |
| `gb_v1` | 82.87 | Not tuned | 0 |
| `gb_v1_optuna` | 83.01 | 83.99 | 6 × 20 |
| `xgb_1_tuned` | 83.57 | 84.69 | 6 × 60 |
| `xgb_5seeds_tuned` | 82.87 | 84.27 | 6 × 60 |
| `lgbm_1_tuned` | 82.58 | 84.69 | 6 × 60 |
| `knn_1_tuned` | 81.32 | 82.58 | 6 × 60 |

The six searches are five outer-fold searches plus one separate final selection
search. Every saved trial table contains the configured number of `COMPLETE`
trials. The final selection score chose parameters and is not an independent
accuracy estimate. Pooled OOF accuracy and mean fold accuracy can differ slightly
because fold sizes are not identical.

Two further completed GB runs have no associated score in the supplied CLI
snapshot. They remain local evidence, not missing leaderboard values to infer:

| Experiment | Local CV, % ± fold SD | OOF accuracy, % | Final selection CV, % |
|---|---:|---:|---:|
| `gb_optuna_50trials` | 83.57 ± 3.26 | 83.57 | 84.13 |
| `gb_optuna_60trials` | 83.85 ± 3.14 | 83.85 | 84.27 |

These runs do not establish that increasing trials improves public accuracy.
The protocol is stratified by row rather than grouped by family or ticket, and
repeated model/feature comparisons on the same development folds limit claims
about a selected winner. The historical holdout is not a fresh final test.

## Implemented versus measured

The YAML runner supports LogisticRegression, sklearn GradientBoosting, CatBoost,
XGBoost, LightGBM and KNeighborsClassifier. KNN has a completed full run and
reported public score here, using a single model with numeric scaling. Seed ensembles for models
with random seeds support soft voting, hard voting and stacking.
Only the **five-seed XGBoost soft ensemble** has a full local run and associated
public score in this snapshot. CatBoost, hard voting and stacking have prior
implementation smoke checks, not full comparative results in this table.

Soft/hard voting uses sklearn VotingClassifier. Stacking trains its meta-model
on OOF probabilities with preprocessing inside every base pipeline. Its own OOF
folds are needed even without Optuna. The current ensemble code does not draw
bootstrap samples for whole models; row subsampling inside a boosting library
is a separate mechanism.

## Verification and reproduction limits

This documentation pass checked the saved OOF row identities, absence of the
excluded holdout, per-fold accuracies, mean/SD, pooled OOF accuracy, completed
trial counts, and every local submission's 418-row ordering and binary labels.
It did not fit models, load checkpoints for inference, submit predictions, or
independently retrieve Kaggle results. The saved checkpoint-consistency flags
are copied from the original run reports, not reverified here.

Earlier implementation work recorded 19 passing tests and short fixed/nested,
voting/stacking and LightGBM checks. Those are execution evidence, not the full
experiment results above. Tests were not rerun in this documentation pass because
they perform fits; the current test sources remain under `tests/`.

To reproduce a historical setup, use that run's resolved `configuration` in the
JSON snapshot rather than today's mutable `config.yaml`. Keep its data, split,
preprocessing, ensemble and search settings together, then use the existing
runner described in the [README](../README.md#run-and-configure). The snapshot
records library versions and hashes; it does not bundle the original environment
or guarantee identical results across platforms and later dependency versions.

The [2026-10-28 evaluation audit](../EVALUATION_AUDIT.md) explains earlier
selection/epoch-accounting problems. Its old scores, commands and seven-test
count are historical; they should not be combined with these later runs as one
evaluation or described as current reproduction instructions.
