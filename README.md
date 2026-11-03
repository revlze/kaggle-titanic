# Titanic

Titanic survival predictions.

**Best public accuracy**: 0.78229 — stacking of Gradient Boosting, Logistic Regression,
Random Forest and Decision Tree with a Logistic Regression meta-model, Optuna,
and title, family-size, extended ticket and cabin features.

## Recorded results

Fixed models use 5-fold CV; tuned models use nested CV(idk, if this necessary): 5 outer / 3 inner folds.
The best stacking model uses 3 OOF folds to train its meta-model.

<details>
<summary>Table of results</summary>

| Model / setup | Optuna trials / search | Local CV, % ± SD | Public accuracy |
|---|---:|---:|---:|
| Logistic regression (baseline) | 0 | 82.72 ± 2.15 | 0.77272 |
| Logistic regression + Optuna | 60 | 82.30 ± 2.44 | 0.77272 |
| Gradient boosting (baseline) | 0 | 82.86 ± 1.71 | 0.76076 |
| Gradient boosting + Optuna | 20 | 83.00 ± 2.10 | 0.75837 |
| XGBoost + Optuna | 60 | 83.56 ± 3.24 | 0.75837 |
| XGBoost ensemble (5 models) | 60 | 82.87 ± 2.62 | 0.75598 |
| LightGBM + Optuna | 60 | 82.58 ± 2.81 | 0.76076 |
| K-nearest neighbors + Optuna | 60 | 81.32 ± 3.13 | 0.75837 |
| Logistic regression + ticket | 0 | 82.72 ± 2.89 | 0.77751 |
| LightGBM + ticket | 0 | 83.71 ± 4.24 | 0.77033 |
| Logistic regression + ticket/cabin | 0 | 81.31 ± 2.30 | 0.77751 |
| LightGBM + ticket/cabin | 0 | 83.42 ± 2.79 | 0.76794 |
| LightGBM + ticket/cabin + Optuna | 60 | 83.42 ± 1.95 | 0.76794 |
| Gradient boosting + ticket/cabin (5 models, hard vote) | 0 | 83.43 ± 0.96 | 0.76794 |
| Gradient boosting + ticket/cabin + Optuna (5 models, hard vote) | 10 | 83.00 ± 2.56 | 0.77990 |
| Stacking (GB + LR + RF + DT) + Optuna | 20 | 83.43 ± 1.24 | **0.78229** |

</details>

## Run and configure

Example configurations are in [configs/](configs/).

To use the selected best parameters:

```bash
cp configs/best_model.yaml config.yaml
```

Place the Titanic CSVs at `data/train.csv` and `data/test.csv`, then edit if needed
[config.yaml](config.yaml) and run:

```bash
uv sync --python 3.14
uv run python run.py
```

Each run creates `artifacts/<exp_name>/<UTC timestamp>/`; earlier runs are preserved.

<details>
<summary>Configuration options</summary>

| Setting | Purpose / default |
|---|---|
| `base.exp_name`, `base.seed` | Run label and existing seed `0xFACED` |
| `run.mode` | `full`, `evaluate`, `train`, or `predict` |
| `run.model_path` | Saved `model.joblib` or `model.pt`, required for `predict` |
| `data.train_path`, `data.test_path` | CSV locations |
| `features.feature_set` | Current: `base_title_family_extticket_extcabin` |
| `preprocessing` | One-hot encoding; enable scaling for logistic regression and KNN |
| `model.name`, `model.params` | Classifier (or `ensemble`) and native constructor parameters |
| `ensemble` | Member algorithms/parameters, soft/hard voting or stacking, OOF folds and meta-model |
| `validation` | Train fraction, evaluation folds and inner search folds |
| `tuning` | Optuna enable flag, trial budget, concurrent trials and parameter distributions |
| `training`, `loss`, `optimizer`, `scheduler` | DNN epochs, batch size, device and training settings |
| `metric.name` | `accuracy`, the supported primary metric |
| `output.dir` | `artifacts` |

`validation.train_size: 0.8` reserves 80% of the labeled CSV for training and CV;
the remaining 20% is the local `test` split. DNNs use `validation.val_size: 0.2`
to split each training portion into `train` and `val` for epoch selection.
The unlabeled Kaggle `test.csv` is loaded separately as `kaggle_test` for inference.

```yaml
run:
  mode: predict
  model_path: artifacts/gb_v1/<run_id>/model.joblib
```

For a single classifier, edit `model.name` and `model.params`. If Optuna is enabled,
replace `tuning.search_space` with parameters for that model too.

For classical models, `tuning.n_jobs` sets the number of concurrent Optuna trials:
`1` runs sequentially, and `-1` uses the CPU count. Missing settings default to `1`.
Inner CV and stacking use `n_jobs=1`; keep individual model workers at `1` when
parallelizing trials. Parallel search may produce different results with the same seed.

For seed ensembles, set `ensemble.n_models` and choose `soft`, `hard` or `stacking`
in `ensemble.method`. Without a member list, these are copies of one model family.

For different algorithms or different settings of one algorithm, use
`model.name: ensemble`, `model.params: {}` and an `ensemble.members` list of
`name`/`params` entries; see `configs/ensemble.yaml`. The member count comes from
the list. Each member has its own preprocessing pipeline and may override
`preprocessing` (for example, `scale: true` for logistic regression).
With `tuning.enabled: false`, the configured parameters are used directly.
To tune an explicit ensemble, put `search_space` inside each member and leave
`tuning.search_space: {}`. One trial samples a full parameter combination for the
ensemble and is scored by its mean CV accuracy. Seed ensembles also use separate
parameter paths for each member. Reports save every member's final parameters.
For stacking, `ensemble.meta_model.search_space` also tunes the meta-model
in the same study; its selected parameters are saved in the ensemble report.
Custom DNNs still use one model per run.

</details>

DNN examples: [MyModel](configs/dnn.yaml) and [MyModelEmb](configs/dnn_embedding.yaml).
`custom/Name` loads a PyTorch model from `src/models.py`; input sizes are inferred automatically.

Base features: `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`, `Sex`, `Embarked`,
plus `Title`, `FamilySize`, `TicketString`, `TicketNum`, `CabinLet`, `CabinNum`, `CabinCount`. Preprocessing is fitted inside each training fold.

`TicketString`, `TicketNum`: ticket prefix and numeric suffix.

`CabinLet`, `CabinNum`, `CabinCount`: unique cabin letters, mean cabin number and
count of numbered cabin entries, calculated within each passenger's row.

## Outputs

Check `summary.md` in the run folder for accuracy and final parameters.
`metrics.json` has the same scores. CV accuracy is the evaluation result;
the final Optuna score is used for parameter selection.
Local test accuracy is measured on the reserved rows before the full-data refit
and saved in `test.json` and `test_predictions.csv`.
`evaluate` runs CV and local test evaluation without a full-data refit.
`train` skips CV, evaluates the local test and refits on all labeled rows.
`full` runs all stages and creates Kaggle predictions; `predict` loads a saved
model and creates predictions without training.

`submission.csv` is ready for Kaggle. Classical models use `model.joblib`;
DNNs use `model.pt` plus `preprocess.joblib` in the same run folder.

Earlier experiments are in `notebooks/`.
