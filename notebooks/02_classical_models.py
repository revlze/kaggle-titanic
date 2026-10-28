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
    # Классические модели: baseline и настройка

    Сравниваем модели при одном наборе признаков. `FEATURE_SET='base'`
    соответствует семи исходным признакам. Primary metric — accuracy.
    Сначала фиксируем stratified holdout 20%, затем выполняем 5-fold CV
    только на оставшихся 80%; preprocessing обучается внутри каждого фолда.

    Для запуска baseline включи `RUN_MODELS`. Подбор включается отдельно
    через `OPTUNA_SEARCH`; `N_TRIALS` задаёт бюджет на модель. Открытие
    ноутбука само по себе не запускает поиск. Итоги находятся в конце.
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
    from sklearn.linear_model import (
        LinearRegression,
        Lasso,
        Ridge,
        ElasticNet,
        LogisticRegression,
    )
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, make_scorer
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.base import clone
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier

    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier

    from tqdm import tqdm
    from pathlib import Path

    from final_proj.utils import set_seed

    return (
        CatBoostClassifier,
        ColumnTransformer,
        DecisionTreeClassifier,
        ElasticNet,
        GradientBoostingClassifier,
        KNeighborsClassifier,
        LGBMClassifier,
        Lasso,
        LinearRegression,
        LogisticRegression,
        OneHotEncoder,
        OrdinalEncoder,
        Pipeline,
        RandomForestClassifier,
        Ridge,
        SimpleImputer,
        StackingClassifier,
        StandardScaler,
        StratifiedKFold,
        VotingClassifier,
        XGBClassifier,
        accuracy_score,
        clone,
        datetime,
        hashlib,
        json,
        make_scorer,
        np,
        optuna,
        pl,
        set_seed,
        timezone,
        tqdm,
        train_test_split,
    )


@app.cell
def _(datetime, set_seed, timezone):
    SEED = 0xFACED
    FEATURE_SET = "base"
    RUN_MODELS = False
    OPTUNA_SEARCH = False
    N_TRIALS = 20
    SAVE_RESULTS = False
    RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    set_seed(SEED)
    return (
        FEATURE_SET,
        N_TRIALS,
        OPTUNA_SEARCH,
        RUN_ID,
        RUN_MODELS,
        SAVE_RESULTS,
        SEED,
    )


@app.cell
def _(mo):
    base_path = mo.notebook_location().parent
    data_path = base_path / "data/titanic"
    results_path = base_path / "notebooks/results"
    return data_path, results_path


@app.cell
def _(data_path, pl):
    train_dataset = pl.read_csv(data_path / "train.csv")
    test_dataset = pl.read_csv(data_path / "test.csv")
    return test_dataset, train_dataset


@app.cell
def _(
    ColumnTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    Pipeline,
    SimpleImputer,
    StandardScaler,
    pl,
):
    feature_sets = {
        "base": [],
        "base_title": ["Title"],
        "base_family": ["FamilySize"],
        "base_title_family": ["Title", "FamilySize"],
        "base_ticket_cabin": ["Ticket", "Cabin"],
        "all": ["Title", "FamilySize", "Ticket", "Cabin"],
    }

    def make_features(df, feature_set="base"):
        """Создать построчные признаки одинаково для train и test, без таргета."""
        extra = feature_sets[feature_set]
        numeric_columns = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
        categorical_columns = ["Sex", "Embarked"]
        if "Title" in extra:
            df = df.with_columns(
                pl.col("Name").str.extract(r",\s*([^.]+)\.", 1).alias("Title")
            )
        if "FamilySize" in extra:
            df = df.with_columns(
                (pl.col("SibSp") + pl.col("Parch") + 1).alias("FamilySize")
            )
            numeric_columns += ["FamilySize"]
        categorical_columns += [c for c in ["Cabin", "Ticket", "Title"] if c in extra]
        return (
            df.select(numeric_columns + categorical_columns),
            numeric_columns,
            categorical_columns,
        )

    def make_preprocess(
        numeric_columns, categorical_columns, encoding="onehot", scale=True
    ):
        """Вернуть unfitted preprocessing: fit вызывается внутри train-фолда."""
        numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale:
            numeric_steps.append(("scaler", StandardScaler()))
        numeric = Pipeline(numeric_steps)
        if encoding == "onehot":
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        elif encoding == "ordinal":
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
        else:
            raise ValueError(f"Unknown encoding: {encoding}")
        categorical = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(missing_values=None, strategy="most_frequent"),
                ),
                ("encoder", encoder),
            ]
        )
        return ColumnTransformer(
            [
                ("numeric", numeric, numeric_columns),
                ("categorical", categorical, categorical_columns),
            ]
        )

    return make_features, make_preprocess


@app.cell
def _(
    FEATURE_SET,
    SEED,
    StratifiedKFold,
    make_features,
    make_preprocess,
    np,
    test_dataset,
    train_dataset,
    train_test_split,
):
    X, numeric_columns, categorical_columns = make_features(train_dataset, FEATURE_SET)
    X_test, _, _ = make_features(test_dataset, FEATURE_SET)
    y = train_dataset["Survived"]
    train_i, val_i = train_test_split(
        np.arange(len(X)),
        train_size=0.8,
        stratify=y,
        random_state=SEED,
    )
    X_train, X_val = X[train_i], X[val_i]
    y_train, y_val = y[train_i], y[val_i]
    preprocess = make_preprocess(numeric_columns, categorical_columns)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    return X_train, X_val, cv, preprocess, train_i, val_i, y_train, y_val


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
            cv.split(X_train, y_train), total=cv.get_n_splits(), leave=False
        ):
            X_train_fold, y_train_fold = X_train[train_i], y_train[train_i]
            X_val_fold, y_val_fold = X_train[val_i], y_train[val_i]

            fold_model = clone(model)
            fold_model.fit(X_train_fold, y_train_fold)
            scores["fold_indices"].append(val_i.tolist())

            pred_train = fold_model.predict(X_train_fold)
            pred_val = fold_model.predict(X_val_fold)

            scores["train_acc"].append(scorer_accuracy(y_train_fold, pred_train))

            scores["cv_acc"].append(scorer_accuracy(y_val_fold, pred_val))

        result = {
            "model": scores["model"],
            "train_acc": np.round(np.mean(scores["train_acc"]), 4),
            "cv_acc": float(np.mean(scores["cv_acc"])),

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
            result = fit_single_model(model_name, model, cv, X_train, y_train)

            model.fit(X_train, y_train)

            pred_val = model.predict(X_val)

            result["val_acc"] = np.round(scorer_accuracy(y_val, pred_val), 4)

            results.append(result)

        return pl.DataFrame(results)

    return scorer_accuracy, start_fits


@app.cell
def _(SEED, optuna, pl, start_fits):
    def start_optuna(
        models,
        param_distributions,
        cv,
        scoring,
        X_train,
        y_train,
        random_state=SEED,
        n_trials=100,
        n_jobs=1,
    ):
        searches = {}
        optuna_params = {name: {} for name in models}
        for name, model in models.items():
            print(f"Started Optuna for model: {name}", flush=True)
            search = optuna.integration.OptunaSearchCV(
                estimator=model,
                param_distributions=param_distributions[name],
                cv=cv,
                verbose=0,
                scoring=scoring,
                n_trials=n_trials if param_distributions[name] else 1,
                random_state=random_state,
                n_jobs=n_jobs,
            )

            search.fit(X_train, y_train)
            print(f"Ended Optuna for model: {name}", flush=True)

            searches[name] = search

            for param_name, param_value in search.best_params_.items():
                optuna_params[name][param_name] = param_value

        return optuna_params

    def fit_optuna(
        models,
        param_distributions,
        cv,
        scoring,
        X_train,
        y_train,
        X_val,
        y_val,
        random_state=SEED,
        n_trials=100,
        n_jobs=1,
    ):
        params = start_optuna(
            models,
            param_distributions,
            cv,
            scoring,
            X_train,
            y_train,
            random_state,
            n_trials,
            n_jobs,
        )
        results = start_fits(models, params, cv, X_train, y_train, X_val, y_val)

        results = results.with_columns(pl.lit("selection_cv").alias("score_role"))
        return results, params

    return (fit_optuna,)


@app.cell
def _(
    FEATURE_SET,
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
    def save_experiments(
        results,
        name,
        parameters=None,
        feature_set=FEATURE_SET,
        score_role="fixed_configuration_cv",
        extra=None,
    ):
        """Сохранить метрики, конфигурацию и точные разбиения завершённого запуска."""
        from datetime import datetime, timezone
        from importlib.metadata import version

        if results is None or not SAVE_RESULTS:
            return None
        results_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        path = results_path / f"{timestamp}_{name}.json"
        record = {
            "run_id": path.stem,
            "session_id": RUN_ID,
            "name": name,
            "feature_set": feature_set,
            "seed": SEED,
            "metric": "accuracy",
            "score_role": score_role,
            "protocol": "stratified_80_20_then_5fold_shuffle_v1",
            "train_indices": train_i.tolist(),
            "holdout_indices": val_i.tolist(),
            "train_passenger_ids": train_dataset[train_i]["PassengerId"].to_list(),
            "holdout_passenger_ids": train_dataset[val_i]["PassengerId"].to_list(),
            "train_sha256": hashlib.sha256(
                (data_path / "train.csv").read_bytes()
            ).hexdigest(),
            "parameters": parameters,
            "extra": extra,
            "versions": {
                package: version(package)
                for package in [
                    "marimo",
                    "scikit-learn",
                    "polars",
                    "numpy",
                    "torch",
                    "optuna",
                    "catboost",
                    "lightgbm",
                    "xgboost",
                ]
            },
            "results": results.to_dicts()
            if isinstance(results, pl.DataFrame)
            else results,
        }
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
        return path

    return (save_experiments,)


@app.cell
def _(
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Pipeline,
    Ridge,
    SEED,
    optuna,
    preprocess,
):
    param_distributions_lin = {
        "LinearRegression": {},
        "Lasso": {
            "model__alpha": optuna.distributions.FloatDistribution(1e-4, 200, log=True),
        },
        "Ridge": {
            "model__alpha": optuna.distributions.FloatDistribution(1e-4, 200, log=True),
        },
        "ElasticNet": {
            "model__alpha": optuna.distributions.FloatDistribution(1e-4, 200, log=True),
            "model__l1_ratio": optuna.distributions.FloatDistribution(0.0, 1.0),
        },
        "LogisticRegression": {},
    }

    models_lin = {
        "LinearRegression": Pipeline(
            [
                ("preprocess", preprocess),
                ("model", LinearRegression()),
            ]
        ),
        "Lasso": Pipeline(
            [
                ("preprocess", preprocess),
                ("model", Lasso(max_iter=10000, random_state=SEED)),
            ]
        ),
        "Ridge": Pipeline(
            [
                ("preprocess", preprocess),
                ("model", Ridge()),
            ]
        ),
        "ElasticNet": Pipeline(
            [
                ("preprocess", preprocess),
                ("model", ElasticNet(max_iter=10000, random_state=SEED)),
            ]
        ),
        "LogisticRegression": Pipeline(
            [
                ("preprocess", preprocess),
                ("model", LogisticRegression(max_iter=2000, random_state=SEED)),
            ]
        ),
    }
    return models_lin, param_distributions_lin


@app.cell
def _(KNeighborsClassifier, Pipeline, optuna, preprocess):
    models_knn = {
        "KNeighborsClassifier": Pipeline(
            [
                ("preprocess", preprocess),
                ("model", KNeighborsClassifier()),
            ]
        ),
    }
    # n_neighbors, weight, metric
    param_distributions_knn = {
        "KNeighborsClassifier": {
            "model__n_neighbors": optuna.distributions.IntDistribution(1, 100),
            "model__weights": optuna.distributions.CategoricalDistribution(
                ["uniform", "distance"]
            ),
            "model__metric": optuna.distributions.CategoricalDistribution(
                [
                    "cityblock",
                    "cosine",
                    "euclidean",
                    "l1",
                    "l2",
                    "manhattan",
                    "nan_euclidean",
                    "minkowski",
                ]
            ),
        },
    }
    return models_knn, param_distributions_knn


@app.cell
def _(DecisionTreeClassifier, Pipeline, SEED, optuna, preprocess):
    model_tree = {
        "DecisionTreeClassifier": Pipeline(
            [
                ("preprocess", preprocess),
                ("model", DecisionTreeClassifier(random_state=SEED)),
            ]
        )
    }

    param_distributions_tree = {
        "DecisionTreeClassifier": {
            "model__criterion": optuna.distributions.CategoricalDistribution(
                ["gini", "entropy", "log_loss"]
            ),
            "model__min_samples_split": optuna.distributions.IntDistribution(2, 10),
            "model__min_samples_leaf": optuna.distributions.IntDistribution(1, 10),
        }
    }
    return model_tree, param_distributions_tree


@app.cell
def _(Pipeline, RandomForestClassifier, SEED, optuna, preprocess):
    model_rf = {
        "RandomForestClassifier": Pipeline(
            [
                ("preprocess", preprocess),
                ("model", RandomForestClassifier(n_jobs=1, random_state=SEED)),
            ]
        )
    }

    param_distributions_rf = {
        "RandomForestClassifier": {
            "model__n_estimators": optuna.distributions.IntDistribution(10, 200),
            "model__max_depth": optuna.distributions.IntDistribution(2, 60),
            "model__min_samples_split": optuna.distributions.IntDistribution(2, 20),
            "model__min_samples_leaf": optuna.distributions.IntDistribution(1, 10),
        }
    }
    return model_rf, param_distributions_rf


@app.cell
def _(
    CatBoostClassifier,
    GradientBoostingClassifier,
    LGBMClassifier,
    Pipeline,
    SEED,
    XGBClassifier,
    optuna,
    preprocess,
):
    model_gb = {
        "GradientBoostingClassifier": Pipeline(
            [
                ("preprocess", preprocess),
                ("model", GradientBoostingClassifier(random_state=SEED)),
            ]
        ),
        "LGBMClassifier": Pipeline(
            [
                ("preprocess", preprocess),
                ("model", LGBMClassifier(n_jobs=1, verbosity=-1, random_state=SEED)),
            ]
        ),
        "XGBClassifier": Pipeline(
            [
                ("preprocess", preprocess),
                ("model", XGBClassifier(n_jobs=1, random_state=SEED)),
            ]
        ),
        "CatBoost": Pipeline(
            [
                ("preprocess", preprocess),
                (
                    "model",
                    CatBoostClassifier(
                        verbose=False,
                        random_seed=SEED,
                        thread_count=1,
                        allow_writing_files=False,
                    ),
                ),
            ]
        ),
    }

    param_distributions_gb = {
        "GradientBoostingClassifier": {
            "model__loss": optuna.distributions.CategoricalDistribution(
                ["log_loss", "exponential"]
            ),
            "model__learning_rate": optuna.distributions.FloatDistribution(1e-4, 1e-1),
            "model__n_estimators": optuna.distributions.IntDistribution(10, 200),
            "model__min_samples_split": optuna.distributions.IntDistribution(2, 50),
            "model__min_samples_leaf": optuna.distributions.IntDistribution(1, 50),
            "model__max_depth": optuna.distributions.IntDistribution(1, 10),
        },
        "LGBMClassifier": {
            "model__num_leaves": optuna.distributions.IntDistribution(3, 100),
            "model__max_depth": optuna.distributions.IntDistribution(1, 20),
            "model__learning_rate": optuna.distributions.FloatDistribution(1e-4, 1e-1),
            "model__n_estimators": optuna.distributions.IntDistribution(100, 1000),
            "model__reg_alpha": optuna.distributions.FloatDistribution(1e-3, 1),
            "model__reg_lambda": optuna.distributions.FloatDistribution(1e-3, 1),
        },
        "XGBClassifier": {
            "model__eta": optuna.distributions.FloatDistribution(1e-4, 1),
            "model__gamma": optuna.distributions.IntDistribution(0, 100),
            "model__max_depth": optuna.distributions.IntDistribution(1, 32),
        },
        "CatBoost": {
            "model__iterations": optuna.distributions.IntDistribution(200, 800),
            "model__depth": optuna.distributions.IntDistribution(3, 7),
            "model__learning_rate": optuna.distributions.FloatDistribution(0.01, 0.15),
            "model__l2_leaf_reg": optuna.distributions.FloatDistribution(1, 10),
        },
    }
    return model_gb, param_distributions_gb


@app.cell
def _(
    LogisticRegression,
    SEED,
    StackingClassifier,
    VotingClassifier,
    cv,
    model_gb,
    model_rf,
    models_lin,
):
    from sklearn.linear_model import RidgeClassifier
    models_to_ens = {
        'GradientBoostingClassifier': model_gb['GradientBoostingClassifier'],
        'RandomForestClassifier': model_rf['RandomForestClassifier'],
        'LogisticRegression': models_lin['LogisticRegression'],
    }

    models_ens = {
        'Voting_soft': VotingClassifier(
            estimators=list(models_to_ens.items()),
            voting='soft',
            n_jobs=-1,
        ),
        'Voting_hard': VotingClassifier(
            estimators=list(models_to_ens.items()),
            voting='hard',
            n_jobs=-1,
        ),
        'Stacking_log': StackingClassifier(
            estimators=list(models_to_ens.items()),
            final_estimator=LogisticRegression(max_iter=2000, random_state=SEED, penalty='l2'),
            cv=cv,
            n_jobs=-1,
        ),
        'Stacking_ridge': StackingClassifier(
            estimators=list(models_to_ens.items()),
            final_estimator=RidgeClassifier(),
            cv=cv,
            n_jobs=-1,
        )
    }
    return (models_ens,)


@app.cell
def _(X_train, X_val, cv, models_ens, start_fits, y_train, y_val):
    res_ens = start_fits(models_ens, None, cv, X_train, y_train, X_val, y_val)
    res_ens
    return (res_ens,)


@app.cell
def _(res_ens):
    res_ens
    return


@app.cell
def _(
    model_gb,
    model_rf,
    model_tree,
    models_knn,
    models_lin,
    param_distributions_gb,
    param_distributions_knn,
    param_distributions_lin,
    param_distributions_rf,
    param_distributions_tree,
):
    models = models_lin | models_knn | model_tree | model_rf | model_gb
    param_distributions = (
        param_distributions_lin
        | param_distributions_knn
        | param_distributions_tree
        | param_distributions_rf
        | param_distributions_gb
    )
    # Для первого поиска достаточно нескольких выбранных моделей.
    models_to_tune = [
        "Ridge",
        "KNeighborsClassifier",
        "RandomForestClassifier",
        "GradientBoostingClassifier",
    ]
    return models, models_to_tune, param_distributions


@app.cell
def _(
    RUN_MODELS,
    X_train,
    X_val,
    cv,
    models,
    save_experiments,
    start_fits,
    y_train,
    y_val,
):
    results_default = None
    if RUN_MODELS:
        results_default = start_fits(models, None, cv, X_train, y_train, X_val, y_val)
        save_experiments(
            results_default,
            "classical_default",
            parameters={
                name: model.named_steps["model"].get_params()
                for name, model in models.items()
            },
            extra={"encoding": "onehot", "scale": True},
        )
    results_default
    return (results_default,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Настройка параметров

    Здесь сохранён первоначальный подход OptunaSearchCV. Его `cv_acc`
    после выбора параметров оценивается на тех же фолдах и помечается
    `selection_cv`. Это оптимистичная оценка подбора. Holdout в поиск не
    передаётся; при многократном ручном выборе по нему он тоже становится
    exploratory. Для независимой оценки всего поиска потребуется nested CV.
    """)
    return


@app.cell
def _(
    N_TRIALS,
    OPTUNA_SEARCH,
    X_train,
    X_val,
    cv,
    fit_optuna,
    make_scorer,
    models,
    models_to_tune,
    param_distributions,
    save_experiments,
    scorer_accuracy,
    y_train,
    y_val,
):
    optuna_results = None
    optuna_params = None
    if OPTUNA_SEARCH:
        _selected = {name: models[name] for name in models_to_tune}
        optuna_results, optuna_params = fit_optuna(
            _selected,
            param_distributions,
            cv,
            make_scorer(scorer_accuracy),
            X_train,
            y_train,
            X_val,
            y_val,
            n_trials=N_TRIALS,
            n_jobs=1,
        )
        save_experiments(
            optuna_results, "classical_optuna", optuna_params, score_role="selection_cv"
        )
    optuna_results
    return (optuna_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Сводка текущего запуска
    """)
    return


@app.cell
def _(optuna_results, pl, results_default):
    current_comparison = (
        pl.concat(
            [
                _table.with_columns(pl.lit(_name).alias("experiment"))
                for _name, _table in [
                    ("default", results_default),
                    ("tuned", optuna_results),
                ]
                if _table is not None
            ],
            how="diagonal",
        )
        if results_default is not None or optuna_results is not None
        else None
    )
    current_comparison
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Исторические результаты значимых запусков
    """)
    return


@app.cell
def _(pl):
    historical_classical = [
        {
            "snapshot": "notebooks",
            "stage": "default",
            "model": "LinearRegression",
            "train_acc": 0.993,
            "cv_acc": 0.7879,
            "val_acc": 0.8492,
        },
        {
            "snapshot": "notebooks",
            "stage": "default",
            "model": "Lasso",
            "train_acc": 0.6166,
            "cv_acc": 0.6166,
            "val_acc": 0.6145,
        },
        {
            "snapshot": "notebooks",
            "stage": "default",
            "model": "Ridge",
            "train_acc": 0.9888,
            "cv_acc": 0.8385,
            "val_acc": 0.838,
        },
        {
            "snapshot": "notebooks",
            "stage": "default",
            "model": "ElasticNet",
            "train_acc": 0.6166,
            "cv_acc": 0.6166,
            "val_acc": 0.6145,
        },
        {
            "snapshot": "notebooks",
            "stage": "default",
            "model": "LogisticRegression",
            "train_acc": 0.8876,
            "cv_acc": 0.8371,
            "val_acc": 0.8268,
        },
        {
            "snapshot": "notebooks",
            "stage": "optuna",
            "model": "LinearRegression",
            "train_acc": 0.993,
            "cv_acc": 0.7879,
            "val_acc": 0.8492,
        },
        {
            "snapshot": "notebooks",
            "stage": "optuna",
            "model": "Lasso",
            "train_acc": 0.9916,
            "cv_acc": 0.8399,
            "val_acc": 0.8492,
        },
        {
            "snapshot": "notebooks",
            "stage": "optuna",
            "model": "Ridge",
            "train_acc": 0.9923,
            "cv_acc": 0.8427,
            "val_acc": 0.8324,
        },
        {
            "snapshot": "notebooks",
            "stage": "optuna",
            "model": "ElasticNet",
            "train_acc": 0.9919,
            "cv_acc": 0.8427,
            "val_acc": 0.838,
        },
        {
            "snapshot": "notebooks",
            "stage": "optuna",
            "model": "LogisticRegression",
            "train_acc": 0.8876,
            "cv_acc": 0.8371,
        },
        {
            "snapshot": "notebooks",
            "stage": "optuna",
            "model": "KNeighborsClassifier",
            "train_acc": 0.9975,
            "cv_acc": 0.8357,
            "val_acc": 0.838,
        },
        {
            "snapshot": "notebooks",
            "stage": "default",
            "model": "KNeighborsClassifier",
            "train_acc": 0.8578,
            "cv_acc": 0.8062,
            "val_acc": 0.8324,
        },
        {
            "snapshot": "notebooks",
            "stage": "optuna",
            "model": "DecisionTreeClassifier",
            "train_acc": 0.8838,
            "cv_acc": 0.8259,
            "val_acc": 0.8156,
        },
        {
            "snapshot": "notebooks",
            "stage": "default",
            "model": "DecisionTreeClassifier",
            "train_acc": 0.9975,
            "cv_acc": 0.8203,
            "val_acc": 0.8156,
        },
        {
            "snapshot": "notebooks",
            "stage": "optuna",
            "model": "RandomForestClassifier",
            "train_acc": 0.9449,
            "cv_acc": 0.8301,
            "val_acc": 0.8436,
        },
        {
            "snapshot": "notebooks",
            "stage": "optuna",
            "model": "GradientBoostingClassifier",
            "train_acc": 0.9182,
            "cv_acc": 0.8582,
            "val_acc": 0.8268,
        },
        {
            "snapshot": "notebooks",
            "stage": "optuna",
            "model": "LGBMClassifier",
            "train_acc": 0.9287,
            "cv_acc": 0.8525,
            "val_acc": 0.7989,





        },
        {
            "snapshot": "notebooks",
            "stage": "optuna",
            "model": "XGBClassifier",
            "train_acc": 0.9031,
            "cv_acc": 0.8484,
            "val_acc": 0.8156,





        },
        {
            "snapshot": "notebooks",
            "stage": "optuna",
            "model": "CatBoost",
            "train_acc": 0.9958,
            "cv_acc": 0.8483,
            "val_acc": 0.8436,





        },
        {
            "snapshot": "older_src",
            "stage": "unknown",
            "model": "LinearRegression",
            "train_acc": 0.7985,
            "train_f1": 0.7252,
            "cv_acc": 0.7964,
            "cv_f1": 0.725,
            "val_acc": 0.7933,
            "val_f1": 0.7176,





        },
        {
            "snapshot": "older_src",
            "stage": "unknown",
            "model": "Lasso",
            "train_acc": 0.6166,
            "train_f1": 0.0,
            "cv_acc": 0.6166,
            "cv_f1": 0.0,
            "val_acc": 0.6145,
            "val_f1": 0.0,





        },
        {
            "snapshot": "older_src",
            "stage": "unknown",
            "model": "Ridge",
            "train_acc": 0.7988,
            "train_f1": 0.7255,
            "cv_acc": 0.7964,
            "cv_f1": 0.725,
            "val_acc": 0.7933,
            "val_f1": 0.7176,





        },
        {
            "snapshot": "older_src",
            "stage": "unknown",
            "model": "ElasticNet",
            "train_acc": 0.6166,
            "train_f1": 0.0,
            "cv_acc": 0.6166,
            "cv_f1": 0.0,
            "val_acc": 0.6145,
            "val_f1": 0.0,





        },
        {
            "snapshot": "older_src",
            "stage": "unknown",
            "model": "LogisticRegression",
            "train_acc": 0.8006,
            "train_f1": 0.7282,
            "cv_acc": 0.7936,
            "cv_f1": 0.7213,
            "val_acc": 0.8101,
            "val_f1": 0.7424,





        },
        {
            "snapshot": "older_src",
            "stage": "unknown",
            "model": "KNeighborsClassifier",
            "train_acc": 0.8525,
            "train_f1": 0.7976,
            "cv_acc": 0.8034,
            "cv_f1": 0.7259,
            "val_acc": 0.7877,
            "val_f1": 0.7246,





        },
        {
            "snapshot": "older_src",
            "stage": "unknown",
            "model": "DecisionTreeClassifier",
            "train_acc": 0.9838,
            "train_f1": 0.9785,
            "cv_acc": 0.7935,
            "cv_f1": 0.7202,
            "val_acc": 0.7765,
            "val_f1": 0.7015,


        },
    ]
    pl.DataFrame(historical_classical).drop("train_f1", "cv_f1", "val_f1")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    По сохранённому снимку default-дерево имело train accuracy 0.9975,
    CV 0.8203 и holdout 0.8156: большой разрыв указывает на переобучение.
    После подбора train снизился до 0.8838, CV составил 0.8259,
    holdout остался 0.8156. На этом holdout улучшения нет.

    У KNN записаны default CV 0.8062 / holdout 0.8324 и tuned CV
    0.8357 / holdout 0.8380. Tuned CV участвовал в выборе параметров;
    разница holdout соответствует примерно одному пассажиру из 179.

    Среди записанных tuned-бустингов GradientBoosting имеет CV 0.8582,
    но holdout 0.8268; у CatBoost CV 0.8483 и holdout 0.8436. Ранжирование
    зависит от разбиения. Эти наблюдения не устанавливают лучшую модель.

    После новых запусков сравнивай одинаковый `FEATURE_SET`, протокол,
    train/CV разрыв, среднее и разброс. Параметры всех испытаний не следует
    описывать отдельными выводами: один поиск отвечает на один вопрос.
    """)
    return


if __name__ == "__main__":
    app.run()
