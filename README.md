# Titanic

Titanic survival predictions.

**Best public accuracy**: 0.77990 — five Gradient Boosting models with hard voting, Optuna and the aggregated features family-size; title; extended ticket and cabin features.

## Recorded results

Local CV is **mean accuracy (%) ± fold SD (percentage points)**. Fixed models use
5-fold CV; tuned models use nested cv: 5 outer / 3 inner folds (idk, maybe nested CV is not needed here).

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
| Gradient boosting + ticket/cabin + Optuna (5 models, hard vote) | 10 | 83.00 ± 2.56 | **0.77990** |

## Run and configure

Check example [configs/](configs/). 

Using configs:
```bash
cp configs/best_model.yaml config.yaml
```

Optuna is off in the examples; set `tuning.enabled: true` to turn on.

DNN examples: [MyModel](configs/config_dnn.yaml) and [MyModelEmb](configs/config_dnn_embedding.yaml).
`custom/Name` loads a PyTorch model from `src/models.py`; input sizes are inferred automatically.

Place the data Titanic at `data/train.csv` and `data/test.csv`, then edit
[config.yaml](config.yaml) and run:

```bash
uv sync --python 3.14
uv run python run.py
```

Each run creates `artifacts/<exp_name>/<UTC timestamp>/`; earlier runs are preserved.

| Setting | Purpose / default |
|---|---|
| `base.exp_name`, `base.seed` | Run label and existing seed `0xFACED` |
| `run.mode` | `full`, `evaluate`, `train`, or `predict` |
| `run.model_path` | Saved `model.joblib` or `model.pt`, required for `predict` |
| `data.train_path`, `data.test_path` | CSV locations |
| `features.feature_set` | Current: `base_title_family_extticket_extcabin` |
| `preprocessing` | One-hot encoding; enable scaling for logistic regression and KNN |
| `model.name`, `model.params` | Classifier and native constructor parameters |
| `ensemble` | Member count, soft/hard voting or stacking, OOF folds and meta-model |
| `validation` | Train fraction, evaluation folds and inner search folds |
| `tuning` | Optuna enable flag, trial budget and parameter distributions |
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

To change the model, edit `model.name` and `model.params`. If Optuna is enabled,
replace `tuning.search_space` with parameters for that model too.

For seed ensembles, set `ensemble.n_models` and choose `soft`, `hard` or `stacking`
in `ensemble.method`. Members use the same model family. KNN and custom DNNs use one model.

Base features: `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`, `Sex`, `Embarked`,
plus `Title`, `FamilySize`, `TicketString`, `TicketNum`, `CabinLet`, `CabinNum`, `CabinCount`. Preprocessing is fitted inside each training fold.

`TicketString`, `TicketNum`: ticket prefix/number and cabin letters

`CabinLet`, `CabinNum`, `CabinCount`: mean cabin number and count of numbered
cabin entries, calculated within each passenger's row.

## Outputs

Check `summary.md` in the run folder for accuracy and final parameters.
`metrics.json` has the same scores. CV accuracy is the evaluation result;
the final Optuna score is used for parameter selection.
Local test accuracy is measured on the reserved rows before the full-data refit
and saved in `test.json` and `test_predictions.csv`.
`evaluate` runs CV and test evaluation; `full` then refits on all labeled rows.

`submission.csv` is ready for Kaggle. Classical models use `model.joblib`;
DNNs use `model.pt` plus `preprocess.joblib` in the same run folder.

Earlier experiments are in `notebooks/`.
