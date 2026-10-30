# Titanic experiments — 2026-10-30

Ten local runs and eight Kaggle submissions. The main table and plot are in the
[README](../README.md#recorded-results); full run details are in the
[result snapshot](experiment-results.json).

## Notes

I keep logistic regression as my baseline: **0.77272** public accuracy with and
without Optuna. None of my other recorded submissions exceeded it.

Single XGBoost: **83.56% ± 3.24 pp** CV, **0.75837** public.
Five-seed soft ensemble: **82.87% ± 2.62 pp** CV, **0.75598** public.
I tuned them separately, so this does not isolate the effect of averaging.

Tuned KNN: **81.32% ± 3.13 pp** CV, **0.75837** public. Final settings:
23 neighbors, uniform weights, Euclidean distance and numeric scaling.

## Extra local runs

| Model / setup | CV accuracy, % ± fold SD | Final selection CV, % |
|---|---:|---:|
| Gradient boosting + Optuna (50 trials) | 83.57 ± 3.26 | 84.13 |
| Gradient boosting + Optuna (60 trials) | 83.85 ± 3.14 | 84.27 |

## Validation

Same `base_title_family` features, seed `0xFACED`, 712 development rows and
179 excluded, previously viewed holdout rows. Fixed models use stratified
5-fold CV; tuned models use 5 outer / 3 inner folds. Preprocessing is fitted
inside each training fold.

Tuned runs also use a separate 3-fold final selection search, then refit on all
891 labeled rows. The selection score is not an independent accuracy estimate.
CV spread is fold SD, not a confidence interval.

I split by row, not by family or ticket, and reuse the same development folds
across experiments. The old holdout is not a fresh final test.

## Source and matching boundary


Exact run/submission IDs, starting configs, final parameters, fold scores,
versions and local CSV hashes are in the [JSON snapshot](experiment-results.json).
Earlier submissions are matched by description; the KNN submit command also
records the exact local file. These records do not verify uploaded-file hashes.
Raw data, models and full run outputs stay outside Git.

For a saved setup, use its config from the snapshot and the
[run instructions](../README.md#run-and-configure).
