import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Imports
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import os
    import random

    from copy import deepcopy

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
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier

    from tqdm import tqdm
    from pathlib import Path

    from final_proj.utils import set_seed

    return (
        CatBoostClassifier,
        ColumnTransformer,
        DataLoader,
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
        Path,
        Pipeline,
        RandomForestClassifier,
        Ridge,
        SimpleImputer,
        StandardScaler,
        StratifiedKFold,
        TensorDataset,
        XGBClassifier,
        accuracy_score,
        clone,
        make_scorer,
        nn,
        np,
        optuna,
        pl,
        plt,
        set_seed,
        torch,
        tqdm,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Config
    """)
    return


@app.cell
def _(set_seed):
    SEED = 0xFACED
    OPTUNA_SEARCH = True
    DEVICE = "cpu"
    set_seed(SEED)
    return DEVICE, OPTUNA_SEARCH, SEED


@app.cell
def _(Path):
    base_path = Path("/users/platon/Documents/final_proj/")
    checkpoint_dir = base_path / "checkpoints/"
    data_path = base_path / "data/titanic/"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir, data_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## data
    """)
    return


@app.cell
def _(data_path, pl):
    train_dataset = pl.read_csv(data_path / "train.csv")
    test_dataset = pl.read_csv(data_path / "test.csv")
    return test_dataset, train_dataset


@app.cell
def _(test_dataset, train_dataset):
    # убираем ненужные фичи
    train_set = train_dataset.drop(["PassengerId", "Name", "Ticket", "Cabin"])
    test_set = test_dataset.drop(["PassengerId", "Name", "Ticket", "Cabin"])

    train_set = train_dataset.drop(["PassengerId", "Name"])
    test_set = test_dataset.drop(["PassengerId", "Name"])
    return (train_set,)


@app.cell
def _(train_set):
    X, y = train_set.drop("Survived"), train_set["Survived"]
    return (y,)


@app.cell
def _():
    # X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, stratify=y, random_state=SEED)
    # X_test = test_set
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Feature Engeeniring
    """)
    return


@app.cell
def _(SEED, pl, test_dataset, train_dataset, train_test_split, y):
    train_set_agg = train_dataset.with_columns(
        pl.col("Name").str.extract(r",\s*([^.]+)\.", 1).alias("Title")
    )
    train_set_agg = train_set_agg.with_columns(
        (pl.col("SibSp") + pl.col("Parch") + 1).alias("FamilySize")
    )

    train_set_agg = train_set_agg.drop(["PassengerId", "Name"])
    test_set_agg = test_dataset.drop(["PassengerId", "Name"])

    X_agg, y_agg = train_set_agg.drop("Survived"), train_set_agg["Survived"]

    X_train, X_val, y_train, y_val = train_test_split(
        X_agg, y_agg, train_size=0.8, stratify=y, random_state=SEED
    )
    X_test = test_set_agg
    return X_agg, X_train, X_val, train_set_agg, y_agg, y_train, y_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### новые стоблцы

    - title — это спрасенные обращения из колонки 'Name'
    - familysize — это SibSp + Parch + 1, просто общее кол-во семьи на борту
    """)
    return


@app.cell
def _(train_set_agg):
    train_set_agg
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## preprocess
    """)
    return


@app.cell
def _(
    ColumnTransformer,
    OneHotEncoder,
    Pipeline,
    SimpleImputer,
    StandardScaler,
):
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(missing_values=None, strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        [
            (
                "numeric",
                numeric,
                ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize"],
            ),
            (
                "categorical",
                categorical,
                ["Sex", "Embarked", "Cabin", "Ticket", "Title"],
            ),
        ]
    )
    return numeric, preprocess


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Base sklearn
    """)
    return


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
        }

        for train_i, val_i in tqdm(
            cv.split(X_train, y_train), total=cv.get_n_splits(), leave=False
        ):
            X_train_fold, y_train_fold = X_train[train_i], y_train[train_i]
            X_val_fold, y_val_fold = X_train[val_i], y_train[val_i]

            model.fit(X_train_fold, y_train_fold)

            pred_train = model.predict(X_train_fold)
            pred_val = model.predict(X_val_fold)

            scores["train_acc"].append(scorer_accuracy(y_train_fold, pred_train))

            scores["cv_acc"].append(scorer_accuracy(y_val_fold, pred_val))

        result = {
            "model": scores["model"],
            "train_acc": np.round(np.mean(scores["train_acc"]), 4),
            "cv_acc": np.round(np.mean(scores["cv_acc"]), 4),
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
def _(SEED, optuna, start_fits):
    def start_optuna(
        models,
        param_distributions,
        cv,
        scoring,
        X_train,
        y_train,
        random_state=SEED,
        n_trials=100,
        n_jobs=-1,
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
        n_jobs=-1,
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

        return results, params

    return (fit_optuna,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## linear regression
    """)
    return


@app.cell
def _(
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Pipeline,
    Ridge,
    StratifiedKFold,
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
                ("model", Lasso()),
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
                ("model", ElasticNet()),
            ]
        ),
        "LogisticRegression": Pipeline(
            [
                ("preprocess", preprocess),
                ("model", LogisticRegression()),
            ]
        ),
    }

    cv = StratifiedKFold(n_splits=5)
    return cv, models_lin, param_distributions_lin


@app.cell
def _(X_train, X_val, cv, models_lin, start_fits, y_train, y_val):
    results_lin = start_fits(models_lin, None, cv, X_train, y_train, X_val, y_val)

    results_lin
    return


@app.cell
def _():
    return


@app.cell
def _(
    OPTUNA_SEARCH,
    X_train,
    X_val,
    cv,
    fit_optuna,
    make_scorer,
    models_lin,
    param_distributions_lin,
    scorer_accuracy,
    y_train,
    y_val,
):
    if OPTUNA_SEARCH:
        optuna_results_lin, optuna_params_lin = fit_optuna(
            models_lin,
            param_distributions_lin,
            cv,
            make_scorer(scorer_accuracy),
            X_train,
            y_train,
            X_val,
            y_val,
        )

    optuna_results_lin
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## KNN
    """)
    return


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
def _(
    OPTUNA_SEARCH,
    X_train,
    X_val,
    cv,
    fit_optuna,
    make_scorer,
    models_knn,
    param_distributions_knn,
    scorer_accuracy,
    y_train,
    y_val,
):
    if OPTUNA_SEARCH:
        optuna_results_knn, optuna_params_knn = fit_optuna(
            models_knn,
            param_distributions_knn,
            cv,
            make_scorer(scorer_accuracy),
            X_train,
            y_train,
            X_val,
            y_val,
        )

    optuna_results_knn
    return


@app.cell
def _(X_train, X_val, cv, models_knn, start_fits, y_train, y_val):
    default_result_knn = start_fits(
        models_knn, None, cv, X_train, y_train, X_val, y_val
    )
    default_result_knn
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## DecisionTreeClassifier
    """)
    return


@app.cell
def _(DecisionTreeClassifier, Pipeline, optuna, preprocess):
    model_tree = {
        "DecisionTreeClassifier": Pipeline(
            [
                ("preprocess", preprocess),
                ("model", DecisionTreeClassifier()),
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
def _(
    OPTUNA_SEARCH,
    X_train,
    X_val,
    cv,
    fit_optuna,
    make_scorer,
    model_tree,
    param_distributions_tree,
    scorer_accuracy,
    y_train,
    y_val,
):
    if OPTUNA_SEARCH:
        optuna_results_tree, optuna_params_tree = fit_optuna(
            model_tree,
            param_distributions_tree,
            cv,
            make_scorer(scorer_accuracy),
            X_train,
            y_train,
            X_val,
            y_val,
        )
    optuna_results_tree, optuna_params_tree
    return


@app.cell
def _(X_train, X_val, cv, model_tree, start_fits, y_train, y_val):
    start_fits(model_tree, None, cv, X_train, y_train, X_val, y_val)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    видно, что обычное дерево переобучилось
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## RandomForest
    """)
    return


@app.cell
def _(Pipeline, RandomForestClassifier, optuna, preprocess):
    model_rf = {
        "RandomForestClassifier": Pipeline(
            [("preprocess", preprocess), ("model", RandomForestClassifier(n_jobs=-1))]
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
    OPTUNA_SEARCH,
    X_train,
    X_val,
    cv,
    fit_optuna,
    make_scorer,
    model_rf,
    param_distributions_rf,
    scorer_accuracy,
    y_train,
    y_val,
):
    if OPTUNA_SEARCH:
        optuna_results_rf, optuna_params_rf = fit_optuna(
            model_rf,
            param_distributions_rf,
            cv,
            make_scorer(scorer_accuracy),
            X_train,
            y_train,
            X_val,
            y_val,
        )
    optuna_results_rf, optuna_params_rf
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Boostings
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _(
    CatBoostClassifier,
    GradientBoostingClassifier,
    LGBMClassifier,
    Pipeline,
    XGBClassifier,
    optuna,
    preprocess,
):
    model_gb = {
        "GradientBoostingClassifier": Pipeline(
            [
                ("preprocess", preprocess),
                ("model", GradientBoostingClassifier()),
            ]
        ),
        "LGBMClassifier": Pipeline(
            [
                ("preprocess", preprocess),
                ("model", LGBMClassifier(n_jobs=1, verbosity=-1)),
            ]
        ),
        "XGBClassifier": Pipeline(
            [
                ("preprocess", preprocess),
                ("model", XGBClassifier(n_jobs=1)),
            ]
        ),
        "CatBoost": Pipeline(
            [
                ("preporcess", preprocess),
                ("model", CatBoostClassifier(verbose=False)),
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
    OPTUNA_SEARCH,
    X_train,
    X_val,
    cv,
    fit_optuna,
    make_scorer,
    model_gb,
    param_distributions_gb,
    scorer_accuracy,
    y_train,
    y_val,
):

    if OPTUNA_SEARCH:
        optuna_results_gb, optuna_params_gb = fit_optuna(
            model_gb,
            param_distributions_gb,
            cv,
            make_scorer(scorer_accuracy),
            X_train,
            y_train,
            X_val,
            y_val,
        )

    optuna_results_gb, optuna_params_gb
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## DNN
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### torch preprocess
    """)
    return


@app.cell
def _(DataLoader, TensorDataset, clone, torch):
    def get_loader(X_train, y_train, X_val, y_val, preprocess):
        nn_preprocess = clone(preprocess)
        X_train_np = nn_preprocess.fit_transform(X_train)
        X_val_np = nn_preprocess.transform(X_val)

        X_train_np = X_train_np.toarray()
        X_val_np = X_val_np.toarray()

        X_train_t = torch.as_tensor(X_train_np, dtype=torch.float32)
        y_train_t = torch.tensor(y_train.to_numpy(), dtype=torch.float32).unsqueeze(1)

        X_val_t = torch.as_tensor(X_val_np, dtype=torch.float32)
        y_val_t = torch.tensor(y_val.to_numpy(), dtype=torch.float32).unsqueeze(1)

        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=64,
            shuffle=True,
        )

        val_loader = DataLoader(
            TensorDataset(X_val_t, y_val_t),
            batch_size=64,
        )

        return train_loader, val_loader

    return (get_loader,)


@app.cell
def _(X_train, X_val, get_loader, preprocess, y_train, y_val):
    train_loader, val_loader = get_loader(X_train, y_train, X_val, y_val, preprocess)
    return train_loader, val_loader


@app.cell
def _(train_loader):
    inp_dim = train_loader.dataset.tensors[0].shape[1]
    return (inp_dim,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### dnn model
    """)
    return


@app.cell
def _(nn):
    class MLPModel(nn.Module):
        def __init__(self, inp, out):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(inp, 25), nn.ReLU(), nn.Linear(25, out)
            )

        def forward(self, x):
            return self.model(x)

    return (MLPModel,)


@app.cell
def MyModel(nn):
    class MyModel(nn.Module):
        def __init__(self, inp, out, hidden_sizes=[500, 200, 50], dropout=0.0):
            super().__init__()
            self.inp = inp
            self.out = out
            self.hidden_sizes = [inp] + hidden_sizes
            self.dropout = dropout
            self._build_model()

        def _build_model(self):
            layers = []
            for hs1, hs2 in zip(self.hidden_sizes, self.hidden_sizes[1:]):
                layers.extend(
                    [
                        nn.Linear(hs1, hs2, bias=False),
                        nn.BatchNorm1d(hs2),
                        nn.LeakyReLU(),
                        nn.Dropout(self.dropout),
                    ]
                )
            layers.append(nn.Linear(self.hidden_sizes[-1], self.out))
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)

    return (MyModel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Setup dnn
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Что пробовал:
    ### устал их пробовать

    opt:
    - [ ] Adadelta
    - [ ] Adafactor
    - [ ] Adagrad
    - [x] Adam - нормал
    - [ ] AdamW
    - [ ] SparseAdam
    - [ ] Adamax
    - [ ] ASGD
    - [ ] LBFGS
    - [ ] Muon - Muon only supports 2D parameters whereas we found a parameter with size: torch.Size([500])
    - [ ] NAdam
    - [x] RAdam — нормал,
    - [x] RMSprop — нормал, но надо с парамтрами играться
    - [x] Rprop - нормал, стабильненько идет график accuracy
    - [x] SGD - нормал, momentum=0.9
    """)
    return


@app.cell
def _(MyModel, inp_dim, nn, torch):
    model2 = MyModel(inp_dim, 1)
    loss_model2 = nn.BCEWithLogitsLoss()
    opt2 = torch.optim.AdamW(model2.parameters(), lr=3e-4)
    sched2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt2,
    )
    return loss_model2, model2, opt2, sched2


@app.cell
def _(MLPModel, inp_dim, nn, torch):
    model1 = MLPModel(inp_dim, 1)
    loss_model1 = nn.BCEWithLogitsLoss()
    opt1 = torch.optim.Adam(model1.parameters(), lr=0.01)
    return loss_model1, model1, opt1


@app.cell
def _(MyModel, nn, torch):
    # Setup for cv and future cfg system
    model_cls = MyModel
    model_params = {
        "out": 1,
        "hidden_sizes": [500, 200, 50],
        "dropout": 0.0,
    }

    loss_model = nn.BCEWithLogitsLoss()

    optimizer_cls = torch.optim.AdamW
    optimizer_params = {"lr": 3e-4}

    scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = {}
    return (
        loss_model,
        model_cls,
        model_params,
        optimizer_cls,
        optimizer_params,
        scheduler_cls,
        scheduler_params,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Обучение dnn
    """)
    return


@app.cell
def _(SEED, epoch_train, get_loader, np, optuna, set_seed, tqdm):
    def cv_train(
        model_cls,
        model_params,
        loss_model,
        opt_cls,
        opt_params,
        sched_params,
        cv,
        X,
        y,
        preprocess,
        sched_cls=None,
        epoches=15,
        trial=None,
    ):
        results = {
            "config": {
                "model_cls": model_cls,
                "model_params": model_params,
                "loss": loss_model,
                "opt_cls": opt_cls,
                "opt_params": opt_params,
                "sched_cls": sched_cls,
                "sched_params": sched_params,
            },
            "folds": [],
            "train_loss": {},
            "train_acc": {},
            "val_loss": {},
            "val_acc": {},
            "metrics": {},
            "summary": None,
        }

        fold_scores = []

        for fold, (train_idx, val_idx) in tqdm(
            enumerate(cv.split(X, y), start=1), total=cv.get_n_splits(), leave=False
        ):
            set_seed(SEED + fold)

            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            fold_train_loader, fold_val_loader = get_loader(
                X_train, y_train, X_val, y_val, preprocess
            )

            inp = fold_train_loader.dataset.tensors[0].shape[1]
            fold_model = model_cls(inp=inp, **model_params)

            fold_opt = opt_cls(fold_model.parameters(), **opt_params)
            fold_sched = (
                None if sched_cls is None else sched_cls(fold_opt, **sched_params)
            )

            res = epoch_train(
                fold_model,
                loss_model,
                fold_opt,
                fold_train_loader,
                fold_val_loader,
                sched=fold_sched,
                epoches=epoches,
            )

            fold_scores.append(res["val_acc"][-1])

            if trial is not None and fold >= 3:
                trial.report(np.mean(fold_scores), step=fold)

                if trial.should_prune():
                    raise optuna.TrialPruned()

            results["folds"].append(
                {
                    "fold": fold,
                    "input_dim": inp,
                    "best_epoch": int(np.argmax(res["val_acc"]) + 1),
                    "best_val_acc": max(res["val_acc"]),
                    **res,
                }
            )

        # Aggregate metrics across folds
        for metric in ("train_loss", "train_acc", "val_loss", "val_acc"):
            values = np.array([fold[metric] for fold in results["folds"]])

            results[metric] = {
                "mean": values.mean(axis=0).tolist(),
                "std": values.std(axis=0).tolist(),
            }

            final_values = values[:, -1]

            results["metrics"][metric] = {
                "mean": float(final_values.mean()),
                "std": float(final_values.std()),
            }

        mean_acc = results["metrics"]["val_acc"]["mean"]
        std_acc = results["metrics"]["val_acc"]["std"]

        results["summary"] = (
            f"CV accuracy: {mean_acc:.4f} ± {std_acc:.4f} "
            f"({len(results['folds'])} folds, last epoch)"
        )

        return results

    return (cv_train,)


@app.cell
def _(DEVICE, torch, tqdm):
    def single_train(model, loss_model, opt, train_loader):
        model.train()

        train_loss = 0
        correct = 0

        def closure():
            opt.zero_grad()
            pred = model(x)
            loss = loss_model(pred, target)
            loss.backward()
            return loss

        train_loop = tqdm(train_loader, leave=False)
        for x, target in train_loop:
            x = x.to(DEVICE)
            target = target.to(DEVICE)

            pred = model(x)
            loss = loss_model(pred, target)

            opt.zero_grad()
            loss.backward()
            if isinstance(opt, torch.optim.LBFGS):
                opt.step(closure)
            else:
                opt.step()

            train_loss += loss.item() * len(x)
            correct += ((pred >= 0) == target).sum().item()

        return train_loss / len(train_loader.dataset), correct / len(
            train_loader.dataset
        )

    def single_eval(model, loss_model, val_loader):
        model.eval()

        val_loss = 0
        correct = 0

        with torch.no_grad():
            for x, target in val_loader:
                x = x.to(DEVICE)
                target = target.to(DEVICE)

                pred = model(x)
                loss = loss_model(pred, target)

                val_loss += loss.item() * len(x)
                correct += ((pred >= 0) == target).sum().item()

        return val_loss / len(val_loader.dataset), correct / len(val_loader.dataset)

    return single_eval, single_train


@app.cell
def _(DEVICE, optuna, single_eval, single_train, torch):
    def epoch_train(
        model,
        loss_model,
        opt,
        train_loader,
        val_loader,
        sched=None,
        epoches=15,
        trial=None,
    ):
        model.to(DEVICE)
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        summary = []

        for epoch in range(epoches):
            single_train_loss, single_train_acc = single_train(
                model, loss_model, opt, train_loader
            )
            single_val_loss, single_val_acc = single_eval(model, loss_model, val_loader)

            if sched is not None:
                if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    sched.step(single_val_loss)
                else:
                    sched.step()

            if trial is not None:
                trial.report(single_val_acc, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            train_loss.append(single_train_loss)
            train_acc.append(single_train_acc)
            val_loss.append(single_val_loss)
            val_acc.append(single_val_acc)
            summary.append(
                f"[{epoch + 1}/{epoches}],train_loss={round(single_train_loss, 4)}|\
                train_acc={round(single_train_acc, 4)}|val_loss={round(single_val_loss, 4)}|\
                val_acc={round(single_val_acc, 4)}"
            )

        return {
            "model": model,
            "loss": loss_model,
            "opt": opt,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "summary": summary,
        }

    return (epoch_train,)


@app.cell
def _(epoch_train, plt):
    def plot_res(result, result2=None):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        results = [("Model 1", result)]
        if result2 is not None:
            results.append(("Model 2", result2))

        for name, res in results:
            epochs = range(1, len(res["train_loss"]) + 1)
            for i, metric in enumerate(("loss", "acc")):
                (line,) = ax[i].plot(
                    epochs, res[f"train_{metric}"], label=f"{name}: train"
                )
                ax[i].plot(
                    epochs,
                    res[f"val_{metric}"],
                    "--",
                    color=line.get_color(),
                    label=f"{name}: validation",
                )

        for axis, title in zip(ax, ("Loss", "Accuracy")):
            axis.set(title=title, xlabel="Epoch", ylabel=title)
            axis.grid(alpha=0.3)
            axis.legend()

        fig.tight_layout()
        return fig

    def compare_dnn(
        model1,
        loss_model1,
        opt1,
        model2,
        loss_model2,
        opt2,
        train_loader,
        val_loader,
        sched2=None,
        epoches=15,
        train_loader2=None,
        val_loader2=None,
        sched1=None,
    ):
        if train_loader2 is None:
            train_loader2 = train_loader
        if val_loader2 is None:
            val_loader2 = val_loader

        results1 = epoch_train(
            model1,
            loss_model1,
            opt1,
            train_loader,
            val_loader,
            sched=sched1,
            epoches=epoches,
        )
        results2 = epoch_train(
            model2,
            loss_model2,
            opt2,
            train_loader2,
            val_loader2,
            sched=sched2,
            epoches=epoches,
        )

        fig = plot_res(results1, results2)
        return results1, results2, fig

    return (compare_dnn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### прост тест кроссвалидации
    """)
    return


@app.cell
def _(
    X_agg,
    cv,
    cv_train,
    loss_model,
    model_cls,
    model_params,
    np,
    optimizer_cls,
    optimizer_params,
    plt,
    preprocess,
    scheduler_cls,
    scheduler_params,
    y_agg,
):
    results_cv = cv_train(
        model_cls,
        model_params,
        loss_model,
        optimizer_cls,
        optimizer_params,
        scheduler_params,
        cv,
        X_agg,
        y_agg,
        preprocess,
        scheduler_cls,
    )

    for type in ['train_acc', 'val_acc']:
        mean = np.array(results_cv[type]["mean"])
        std = np.array(results_cv[type]["std"])
        epochs = np.arange(1, len(mean) + 1)

        _fig, _ax = plt.subplots(figsize=(10, 5))

        _ax.plot(epochs, mean, label=type, color="royalblue")
        _ax.fill_between(
            epochs,
            mean - std,
            mean + std,
            color="royalblue",
            alpha=0.2,
        )

        _ax.set(xlabel="Epoch", ylabel="Accuracy")
        _ax.legend()
        _ax.grid(alpha=0.3)

    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Сравнение dnn
    """)
    return


@app.cell
def _(
    compare_dnn,
    loss_model1,
    loss_model2,
    model1,
    model2,
    opt1,
    opt2,
    sched2,
    train_loader,
    val_loader,
):
    EPOCHES = 10
    results1, results2, fig = compare_dnn(
        model1,
        loss_model1,
        opt1,
        model2,
        loss_model2,
        opt2,
        train_loader,
        val_loader,
        sched2=sched2,
        epoches=EPOCHES,
    )
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## optuna dnn
    """)
    return


@app.cell
def _(checkpoint_dir, cv_train, nn, torch):
    def run_trial(
        trial,
        model_cls,
        cv,
        X,
        y,
        preprocess,
        name,
        model_params,
        epochs=15,
        sched_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
        sched_params=None,
    ):
        optimizer_name = trial.suggest_categorical(
            "optimizer", ["AdamW", "Adam", "SGD"]
        )

        if optimizer_name == "SGD":
            lr = trial.suggest_float("lr_sgd", 1e-3, 1e-1, log=True)
            optimizer_params = {
                "lr": lr,
                "momentum": trial.suggest_float("momentum", 0.0, 0.95),
            }
        else:
            lr = trial.suggest_float("lr_adam", 1e-4, 1e-2, log=True)
            optimizer_params = {"lr": lr}

        optimizer_cls = getattr(torch.optim, optimizer_name)
        loss_fn = nn.BCEWithLogitsLoss()
        sched_params = sched_params or {}

        results = cv_train(
            model_cls=model_cls,
            model_params=model_params,
            loss_model=loss_fn,
            opt_cls=optimizer_cls,
            opt_params=optimizer_params,
            sched_cls=sched_cls,
            sched_params=sched_params,
            cv=cv,
            X=X,
            y=y,
            preprocess=preprocess,
            epoches=epochs,
            trial=trial,
        )

        accuracy = results["metrics"]["val_acc"]["mean"]

        save_dir = checkpoint_dir / f"optuna_{name}"
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_params": model_params,
                "optimizer_name": optimizer_name,
                "optimizer_params": optimizer_params,
                "scheduler_name": (
                    sched_cls.__name__ if sched_cls is not None else None
                ),
                "scheduler_params": sched_params,
                "accuracy": accuracy,
                "metrics": results["metrics"],
                "history": {
                    metric: results[metric]
                    for metric in ("train_loss", "train_acc", "val_loss", "val_acc")
                },
                "folds": [
                    {
                        "fold": fold["fold"],
                        "best_epoch": fold["best_epoch"],
                        "best_val_acc": fold["best_val_acc"],
                        "final_val_acc": fold["val_acc"][-1],
                    }
                    for fold in results["folds"]
                ],
                "trial_params": trial.params,
                "summary": results["summary"],
            },
            save_dir / f"trial_{trial.number}.pt",
        )

        return accuracy

    return (run_trial,)


@app.cell
def _(MyModel, X_train, cv, preprocess, run_trial, y_train):
    def objective_mlp(trial, name=""):
        hidden_sizes = [
            trial.suggest_categorical("hidden_1", [250, 500]),
            trial.suggest_categorical("hidden_2", [100, 200]),
            trial.suggest_categorical("hidden_3", [25, 50]),
        ]
        dropout = trial.suggest_float("dropout", 0.0, 0.3)

        model_params = {
            "out": 1,
            "hidden_sizes": hidden_sizes,
            "dropout": dropout,
        }
        model = MyModel(**model_params)

        return run_trial(
            trial=trial,
            model_cls=MyModel,
            cv=cv,
            X=X_train,
            y=y_train,
            preprocess=preprocess,
            name=name,
            model_params=model_params,
        )

    return (objective_mlp,)


@app.cell
def _(
    MyModel,
    MyModelEmb,
    X_train,
    cat_slice,
    cv,
    num_slice,
    preprocess,
    run_trial,
    y_train,
):
    def objective_emb(trial, name=""):
        hidden_sizes = [
            trial.suggest_categorical("hidden_1", [250, 500]),
            trial.suggest_categorical("hidden_2", [100, 200]),
            trial.suggest_categorical("hidden_3", [25, 50]),
        ]
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        sex_emb_dim = trial.suggest_categorical("sex_emb_dim", [1, 2, 3])
        embarked_emb_dim = trial.suggest_categorical("embarked_emb_dim", [1, 2, 3])
        cabin_emb_dim = trial.suggest_categorical("cabin_emb_dim", [1, 2, 3])
        ticket_emb_dim = trial.suggest_categorical("ticket_emb_dim", [1, 2, 3])
        title_emb_dim = trial.suggest_categorical("title_emb_dim", [1, 2, 3])

        n_num = num_slice.stop - num_slice.start
        model_params = {
            "inp": n_num
            + sex_emb_dim
            + embarked_emb_dim
            + cabin_emb_dim
            + ticket_emb_dim
            + title_emb_dim,
            "out": 1,
            "hidden_sizes": hidden_sizes,
            "dropout": dropout,
            "sex_emb_dim": sex_emb_dim,
            "embarked_emb_dim": embarked_emb_dim,
            "cabin_emb_dim": cabin_emb_dim,
            "ticket_emb_dim": ticket_emb_dim,
            "title_emb_dim": title_emb_dim,
        }
        model = MyModelEmb(num_slice=num_slice, cat_slice=cat_slice, **model_params)

        return run_trial(
            trial=trial,
            model_cls=MyModel,
            cv=cv,
            X=X_train,
            y=y_train,
            preprocess=preprocess,
            name=name,
            model_params=model_params,
        )

    return (objective_emb,)


@app.cell
def _(DEVICE, OPTUNA_SEARCH, checkpoint_dir, objective_mlp, optuna, torch):
    best_model = None
    if OPTUNA_SEARCH:
        name = "default_optim_v1"
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective_mlp(trial, name=name), n_trials=100)

        best_model = torch.load(
            checkpoint_dir / f"optuna_{name}" / f"trial_{study.best_trial.number}.pt",
            map_location=DEVICE,
            weights_only=True,
        )
        print(best_model["accuracy"])
    return (best_model,)


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## embedding dnn
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### preprocess
    """)
    return


@app.cell
def _(ColumnTransformer, OrdinalEncoder, Pipeline, SimpleImputer, np, numeric):
    categorical_embedding = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    missing_values=None,
                    strategy="most_frequent",
                ),
            ),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    dtype=np.int64,
                ),
            ),
        ]
    )

    preprocess_embedding = ColumnTransformer(
        [
            (
                "numeric",
                numeric,
                ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize"],
            ),
            (
                "categorical",
                categorical_embedding,
                ["Sex", "Embarked", "Cabin", "Ticket", "Title"],
            ),
        ]
    )
    return (preprocess_embedding,)


@app.cell
def _(X_train, X_val, clone, preprocess_embedding, torch, y_train, y_val):
    nn_preprocess_embedding = clone(preprocess_embedding)
    X_train_np_emb = nn_preprocess_embedding.fit_transform(X_train)
    X_val_np_emb = nn_preprocess_embedding.transform(X_val)

    num_slice = nn_preprocess_embedding.output_indices_["numeric"]
    cat_slice = nn_preprocess_embedding.output_indices_["categorical"]

    X_train_t_emb = torch.as_tensor(X_train_np_emb, dtype=torch.float32)
    y_train_t_emb = torch.tensor(y_train.to_numpy(), dtype=torch.float32).unsqueeze(1)

    X_val_t_emb = torch.as_tensor(X_val_np_emb, dtype=torch.float32)
    y_val_t_emb = torch.tensor(y_val.to_numpy(), dtype=torch.float32).unsqueeze(1)
    return (
        X_train_t_emb,
        X_val_t_emb,
        cat_slice,
        num_slice,
        y_train_t_emb,
        y_val_t_emb,
    )


@app.cell
def _(X_train):
    X_train["Title"].unique().shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### loaders
    """)
    return


@app.cell
def _(
    DataLoader,
    TensorDataset,
    X_train_t_emb,
    X_val_t_emb,
    y_train_t_emb,
    y_val_t_emb,
):
    train_loader_emb = DataLoader(
        TensorDataset(X_train_t_emb, y_train_t_emb),
        batch_size=64,
        shuffle=True,
    )

    val_loader_emb = DataLoader(
        TensorDataset(X_val_t_emb, y_val_t_emb),
        batch_size=64,
    )
    return train_loader_emb, val_loader_emb


@app.cell
def _(X_train):
    X_train["Ticket"].unique().shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### model
    """)
    return


@app.cell
def _(nn, torch):
    class MyModelEmb(nn.Module):
        def __init__(
            self,
            inp,
            out,
            num_slice,
            cat_slice,
            hidden_sizes=[500, 200, 50],
            dropout=0.0,
            sex_emb_dim=2,
            embarked_emb_dim=2,
            cabin_emb_dim=2,
            ticket_emb_dim=2,
            title_emb_dim=2,
        ):
            super().__init__()
            self.inp = inp
            self.out = out
            self.hidden_sizes = [inp] + hidden_sizes
            self.dropout = dropout
            self._build_model()
            self.sex_emb = nn.Embedding(3, sex_emb_dim)
            self.embarked_emb = nn.Embedding(4, embarked_emb_dim)
            self.cabin_emb = nn.Embedding(124, cabin_emb_dim)
            self.ticket_emb = nn.Embedding(562, ticket_emb_dim)
            self.title_emb = nn.Embedding(16, title_emb_dim)
            self.num_slice = num_slice
            self.cat_slice = cat_slice

        def _build_model(self):
            layers = []
            for hs1, hs2 in zip(self.hidden_sizes, self.hidden_sizes[1:]):
                layers.extend(
                    [
                        nn.Linear(hs1, hs2, bias=False),
                        nn.BatchNorm1d(hs2),
                        nn.LeakyReLU(),
                        nn.Dropout(self.dropout),
                    ]
                )
            layers.append(nn.Linear(self.hidden_sizes[-1], self.out))
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            x_num = x[:, self.num_slice]
            x_cat = x[:, self.cat_slice].to(torch.int32) + 1
            sex_vector = self.sex_emb(x_cat[:, 0])
            embarked_vector = self.embarked_emb(x_cat[:, 1])
            cabin_vector = self.cabin_emb(x_cat[:, 2])
            ticket_vector = self.ticket_emb(x_cat[:, 3])
            title_vector = self.title_emb(x_cat[:, 4])
            x = torch.cat(
                [
                    x_num,
                    sex_vector,
                    embarked_vector,
                    cabin_vector,
                    ticket_vector,
                    title_vector,
                ],
                dim=1,
            )
            return self.model(x)

    return (MyModelEmb,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### setup model
    """)
    return


@app.cell
def _(MyModelEmb, cat_slice, nn, num_slice, torch):
    model_emb = MyModelEmb(6 + 5 * 2, 1, num_slice, cat_slice)
    loss_model_emb = nn.BCEWithLogitsLoss()
    opt_emb = torch.optim.AdamW(model_emb.parameters(), lr=0.01)
    sched_emb = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_emb)
    return loss_model_emb, model_emb, opt_emb, sched_emb


@app.cell
def _(
    epoch_train,
    loss_model_emb,
    model_emb,
    opt_emb,
    sched_emb,
    train_loader_emb,
    val_loader_emb,
):
    results = epoch_train(
        model_emb, loss_model_emb, opt_emb, train_loader_emb, val_loader_emb, sched_emb
    )
    max(results["val_acc"])
    return


@app.cell
def _(DEVICE, OPTUNA_SEARCH, checkpoint_dir, objective_emb, optuna, torch):
    best_model_emb = None
    if OPTUNA_SEARCH:
        name_exp = "emb_optim_v1"
        study_emb = optuna.create_study(direction="maximize")
        study_emb.optimize(
            lambda trial: objective_emb(trial, name=name_exp), n_trials=100
        )

        best_model_emb = torch.load(
            checkpoint_dir
            / f"optuna_{name_exp}"
            / f"trial_{study_emb.best_trial.number}.pt",
            map_location=DEVICE,
            weights_only=True,
        )
        print(best_model_emb["accuracy"])
    return (best_model_emb,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compare dnn default with dnn embbedding
    """)
    return


@app.cell
def _(
    DEVICE,
    MyModel,
    MyModelEmb,
    best_model,
    best_model_emb,
    cat_slice,
    compare_dnn,
    mo,
    nn,
    num_slice,
    torch,
    train_loader,
    train_loader_emb,
    val_loader,
    val_loader_emb,
):
    mo.stop(
        best_model is None or best_model_emb is None,
        mo.md("Для сравнения запусти обе ячейки Optuna с `OPTUNA_SEARCH = True`."),
    )
    # Checkpoint и loaders должны соответствовать текущему preprocessing.
    _ckpt_mlp = best_model
    _ckpt_emb = best_model_emb

    _model_mlp = MyModel(**_ckpt_mlp["model_params"]).to(DEVICE)
    _model_mlp.load_state_dict(_ckpt_mlp["model_state_dict"])

    _model_emb = MyModelEmb(
        num_slice=num_slice,
        cat_slice=cat_slice,
        **_ckpt_emb["model_params"],
    ).to(DEVICE)
    _model_emb.load_state_dict(_ckpt_emb["model_state_dict"])

    _loss_mlp = nn.BCEWithLogitsLoss()
    _loss_emb = nn.BCEWithLogitsLoss()

    _opt_mlp = getattr(torch.optim, _ckpt_mlp.get("optimizer_name", "AdamW"))(
        _model_mlp.parameters(),
        **_ckpt_mlp.get("optimizer_params", {"lr": _ckpt_mlp["lr"]}),
    )
    _opt_emb = getattr(torch.optim, _ckpt_emb.get("optimizer_name", "AdamW"))(
        _model_emb.parameters(),
        **_ckpt_emb.get("optimizer_params", {"lr": _ckpt_emb["lr"]}),
    )

    _sched_mlp = torch.optim.lr_scheduler.ReduceLROnPlateau(_opt_mlp)
    _sched_emb = torch.optim.lr_scheduler.ReduceLROnPlateau(_opt_emb)

    compare_results_mlp, compare_results_emb, compare_fig = compare_dnn(
        _model_mlp,
        _loss_mlp,
        _opt_mlp,
        _model_emb,
        _loss_emb,
        _opt_emb,
        train_loader,
        val_loader,
        sched1=_sched_mlp,
        sched2=_sched_emb,
        train_loader2=train_loader_emb,
        val_loader2=val_loader_emb,
        epoches=30,
    )

    compare_fig
    return


if __name__ == "__main__":
    app.run()
