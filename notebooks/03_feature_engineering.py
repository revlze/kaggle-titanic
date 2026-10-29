import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Feature engineering и предобработка

    Проверяем отдельные гипотезы относительно `base` на одинаковых
    наблюдениях и CV-фолдах, с фиксированными параметрами моделей.
    `Title` передаёт информацию об обращении и социальном статусе;
    `FamilySize` — о числе родственников на борту. Возвращение исходных
    `Ticket` и `Cabin` проверяется отдельно, поскольку тоже меняет вход.

    Все признаки вычисляются по одной строке. Статистики imputer, scaler
    и encoder обучаются внутри train-фолда. Одинаковая функция применяется
    к train и Kaggle test. `RUN_MODELS=True` запускает сравнение.
    """)
    return


@app.cell
def _():
    import os
    import random
    import copy
    import json
    import hashlib
    from datetime import datetime, timezone
    import joblib

    # Обход падения OpenMP на macOS; задаём до импорта ML-библиотек.
    os.environ["OMP_NUM_THREADS"] = "1"

    import torch
    import optuna
    import numpy as np
    import torch.nn as nn
    import seaborn as sns
    import polars as pl
    import matplotlib.pyplot as plt

    from torch.utils.data import DataLoader, random_split, TensorDataset

    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, make_scorer
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.base import clone
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier

    from tqdm import tqdm
    from pathlib import Path

    from utils import set_seed

    return (
        CatBoostClassifier,
        ColumnTransformer,
        LogisticRegression,
        OneHotEncoder,
        OrdinalEncoder,
        Pipeline,
        SimpleImputer,
        StandardScaler,
        StratifiedKFold,
        accuracy_score,
        clone,
        datetime,
        hashlib,
        json,
        np,
        pl,
        set_seed,
        timezone,
        tqdm,
        train_test_split,
    )


@app.cell
def _(datetime, set_seed, timezone):
    SEED = 0xFACED
    RUN_MODELS = True
    SAVE_RESULTS = False
    RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    set_seed(SEED)
    return RUN_ID, RUN_MODELS, SAVE_RESULTS, SEED


@app.cell
def _(mo):
    base_path = mo.notebook_location().parent
    data_path = base_path / 'data/titanic'
    results_path = base_path / 'notebooks/results'
    return data_path, results_path


@app.cell
def _(data_path, pl):
    train_dataset = pl.read_csv(data_path / 'train.csv')
    test_dataset = pl.read_csv(data_path / 'test.csv')
    return (train_dataset,)


@app.cell
def _():
    from features import make_features, make_preprocess
    return make_features, make_preprocess


@app.cell
def _(SEED, StratifiedKFold, np, train_dataset, train_test_split):
    y = train_dataset['Survived']
    train_i, val_i = train_test_split(
        np.arange(len(train_dataset)), train_size=0.8, stratify=y, random_state=SEED,
    )
    y_train, y_val = y[train_i], y[val_i]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    return cv, train_i, val_i, y_train, y_val


@app.cell
def _(accuracy_score, clone, np, pl, tqdm):
    def scorer_accuracy(y_true, y_pred):
            y_pred = (y_pred > 0.5).astype(int)
            return accuracy_score(y_true, y_pred)

    def fit_single_model(model_name, model, cv, X_train, y_train):
        print(f"Started cv for model: {model_name}")

        scores = {
            "model": model_name,
            "train_acc": [],
            "cv_acc": [],
            "fold_indices": [],
        }

        for train_i, val_i in tqdm(
                                    cv.split(X_train, y_train),
                                    total=cv.get_n_splits(),
                                    leave=False
                                ):
            X_train_fold, y_train_fold = X_train[train_i], y_train[train_i]
            X_val_fold, y_val_fold = X_train[val_i], y_train[val_i]

            fold_model = clone(model)
            fold_model.fit(X_train_fold, y_train_fold)
            scores["fold_indices"].append(val_i.tolist())

            pred_train = fold_model.predict(X_train_fold)
            pred_val = fold_model.predict(X_val_fold)

            scores["train_acc"].append(
                scorer_accuracy(y_train_fold, pred_train)
            )
   
            scores["cv_acc"].append(
                scorer_accuracy(y_val_fold, pred_val)
            )


        result = {
            "model": scores["model"],

            "train_acc": np.round(np.mean(scores["train_acc"]), 4),

            "cv_acc": float(np.mean(scores["cv_acc"])),
            "cv_std": float(np.std(scores["cv_acc"])),
            "cv_fold_scores": scores["cv_acc"],
            "cv_val_indices": scores["fold_indices"],
        }

        print(f"Ended cv for model: {model_name}")

        return result


    def start_fits(models, params, cv, X_train, y_train, X_val, y_val):
        results = []

        for model_name, model in models.items():
            model = clone(model)
            if params is not None:
                model.set_params(**params[model_name])
            result = fit_single_model(
                model_name,
                model,
                cv,
                X_train,
                y_train
            )

            # Feature selection must not inspect the previously viewed holdout.
            result["score_role"] = "exploratory_feature_cv"


            results.append(result)

        return pl.DataFrame(results)

    return (start_fits,)


@app.cell
def _(
    RUN_ID,
    SAVE_RESULTS,
    SEED,
    data_path,
    hashlib,
    json,
    pl,
    results_path,
    train_dataset,
    train_i,
    val_i,
):
    def save_experiments(results, name, parameters=None, feature_set='base',
                         score_role='exploratory_feature_cv', extra=None):
        """Сохранить метрики, конфигурацию и точные разбиения завершённого запуска."""
        from datetime import datetime, timezone
        from importlib.metadata import version
        if results is None or not SAVE_RESULTS:
            return None
        results_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        path = results_path / f'{timestamp}_{name}.json'
        record = {
            'run_id': path.stem, 'session_id': RUN_ID, 'name': name, 'feature_set': feature_set,
            'seed': SEED, 'metric': 'accuracy', 'score_role': score_role,
            'protocol': 'stratified_80_20_then_5fold_shuffle_v1',
            'train_indices': train_i.tolist(), 'holdout_indices': val_i.tolist(),
            'train_passenger_ids': train_dataset[train_i]['PassengerId'].to_list(),
            'holdout_passenger_ids': train_dataset[val_i]['PassengerId'].to_list(),
            'train_sha256': hashlib.sha256((data_path / 'train.csv').read_bytes()).hexdigest(),
            'parameters': parameters, 'extra': extra,
            'versions': {package: version(package) for package in ['marimo', 'scikit-learn', 'polars', 'numpy', 'torch', 'optuna', 'catboost', 'lightgbm', 'xgboost']},
            'results': results.to_dicts() if isinstance(results, pl.DataFrame) else results,
        }
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        return path

    return (save_experiments,)


@app.cell
def _(CatBoostClassifier, LogisticRegression, SEED):
    feature_variants = ['base', 'base_title', 'base_family', 'base_title_family', 'base_ticket_cabin', 'all']
    # Одинаковые параметры внутри каждой модели позволяют выделить эффект признаков.
    feature_models = {
        'LogisticRegression': LogisticRegression(max_iter=2000, random_state=SEED),
        'CatBoost': CatBoostClassifier(iterations=300, verbose=False, random_seed=SEED,
                                      thread_count=1, allow_writing_files=False),
    }
    return feature_models, feature_variants


@app.cell
def _(
    Pipeline,
    RUN_MODELS,
    clone,
    cv,
    feature_models,
    feature_variants,
    make_features,
    make_preprocess,
    pl,
    save_experiments,
    start_fits,
    train_dataset,
    train_i,
    val_i,
    y_train,
    y_val,
):
    feature_results = None
    if RUN_MODELS:
        _records = []
        for _variant in feature_variants:
            _X, _numeric, _categorical = make_features(train_dataset, _variant)
            _models = {
                name: Pipeline([('preprocess', make_preprocess(_numeric, _categorical)),
                                ('model', clone(model))])
                for name, model in feature_models.items()
            }
            _table = start_fits(_models, None, cv, _X[train_i], y_train, _X[val_i], y_val)
            _table = _table.with_columns(pl.lit(_variant).alias('features'))
            _records.append(_table)
            save_experiments(_table, f'features_{_variant}', feature_set=_variant,
                             parameters={name: model.get_params() for name, model in feature_models.items()})
        feature_results = pl.concat(_records)
    return (feature_results,)


@app.cell
def _(feature_results):
    feature_results.drop(['cv_fold_scores', 'cv_val_indices', 'cv_std']) if feature_results is not None else None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Изменение относительно базовых признаков

    Разность считается по тем же фолдам и отдельно для каждой модели.
    Положительное среднее на пяти фолдах само по себе не доказывает
    устойчивое улучшение; посмотри на отдельные разности. Holdout не используется.
    """)
    return


@app.cell
def _(feature_models, feature_results, np, pl):
    feature_deltas = None
    if feature_results is not None:
        _rows = []
        for _model in feature_models:
            _baseline = feature_results.filter((pl.col('model') == _model) & (pl.col('features') == 'base')).to_dicts()[0]
            for _row in feature_results.filter(pl.col('model') == _model).to_dicts():
                _rows.append({
                    'model': _model, 'features': _row['features'],
                    'cv_acc': _row['cv_acc'], 'cv_std': _row['cv_std'],
                    'delta_cv': _row['cv_acc'] - _baseline['cv_acc'],
                    'paired_fold_deltas': (np.array(_row['cv_fold_scores']) - np.array(_baseline['cv_fold_scores'])).tolist(),
                })
        feature_deltas = pl.DataFrame(_rows)
    feature_deltas
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Отдельная проверка предобработки

    `RUN_PREPROCESS=True` сравнивает scaling и one-hot/ordinal на `base`.
    Остальные параметры фиксированы. Ordinal-коды не задают естественный
    порядок категорий; возможное ухудшение линейной модели интерпретируется
    с учётом этого ограничения. Default baseline уже включает imputation;
    отдельного измеренного сравнения способов заполнения пропусков пока нет.
    """)
    return


@app.cell
def _():
    RUN_PREPROCESS = False
    return (RUN_PREPROCESS,)


@app.cell
def _(
    Pipeline,
    RUN_PREPROCESS,
    clone,
    cv,
    feature_models,
    make_features,
    make_preprocess,
    pl,
    save_experiments,
    start_fits,
    train_dataset,
    train_i,
    val_i,
    y_train,
    y_val,
):
    preprocess_results = None
    if RUN_PREPROCESS:
        _records = []
        for _encoding, _scale in [('onehot', True), ('onehot', False), ('ordinal', True)]:
            _X, _numeric, _categorical = make_features(train_dataset, 'base')
            _models = {name: Pipeline([
                ('preprocess', make_preprocess(_numeric, _categorical, _encoding, _scale)),
                ('model', clone(model)),
            ]) for name, model in feature_models.items()}
            _table = start_fits(_models, None, cv, _X[train_i], y_train, _X[val_i], y_val)
            _label = f'{_encoding}_scale_{_scale}'
            _table = _table.with_columns(pl.lit(_label).alias('preprocessing'))
            _records.append(_table)
            save_experiments(_table, f'preprocess_{_label}', feature_set='base',
                             extra={'encoding': _encoding, 'scale': _scale})
        preprocess_results = pl.concat(_records)
    preprocess_results
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Выводы

    В исходном ноутбуке `Title`, `FamilySize`, исходные `Ticket` и `Cabin`
    были подключены к общей предобработке. Отдельной сохранённой абляции
    нет, поэтому утверждать пользу конкретного признака пока нельзя.

    Новое сравнение разделяет эти изменения. После запуска укажи для
    каждого оставленного признака его мотивацию, baseline, разницу по
    фолдам и решение: оставить, исключить или перепроверить. Нейтральные
    и неудачные результаты тоже сохраняются в журнале.
    """)
    return


if __name__ == "__main__":
    app.run()
