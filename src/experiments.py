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
    import os
    import random

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

    from tqdm import tqdm
    from pathlib import Path

    from utils import set_seed

    return (
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
        f1_score,
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
    OPTUNA_SEARCH = False
    DEVICE = 'cpu'
    set_seed(SEED)
    return DEVICE, OPTUNA_SEARCH, SEED


@app.cell
def _(Path):
    base_path = Path('/users/platon/Documents/final_proj/')
    checkpoint_dir = base_path / 'checkpoints/'
    data_path = base_path / 'data/titanic/'

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
    train_dataset = pl.read_csv(data_path / 'train.csv')
    test_dataset = pl.read_csv(data_path / 'test.csv')
    return test_dataset, train_dataset


@app.cell
def _(test_dataset, train_dataset):
    # убираем ненужные фичи
    train_set= train_dataset.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'])
    test_set = test_dataset.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'])
    return test_set, train_set


@app.cell
def _(train_set):
    X, y = train_set.drop('Survived'), train_set['Survived']
    return X, y


@app.cell
def _(SEED, X, test_set, train_test_split, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, stratify=y, random_state=SEED)
    X_test = test_set
    return X_train, X_val, y_train, y_val


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
    numeric = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    categorical = Pipeline([
        ("imputer", SimpleImputer(missing_values=None, strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer([
        ('numeric', numeric, ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']),
        ('categorical', categorical, ['Sex', 'Embarked']),
    ])
    return numeric, preprocess


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Base
    """)
    return


@app.cell
def _(accuracy_score, clone, f1_score, np, pl, tqdm):
    def scorer_accuracy(y_true, y_pred):
            y_pred = (y_pred > 0.5).astype(int)
            return accuracy_score(y_true, y_pred)

    def scorer_f1(y_true, y_pred):
            y_pred = (y_pred > 0.5).astype(int)
            return f1_score(y_true, y_pred)

    def fit_single_model(model_name, model, cv, X_train, y_train):
        print(f"Started cv for model: {model_name}")

        scores = {
            "model": model_name,
            "train_acc": [],
            "train_f1": [],
            "cv_acc": [],
            "cv_f1": [],
        }

        for train_i, val_i in tqdm(
                                    cv.split(X_train, y_train),
                                    total=cv.get_n_splits(),
                                    leave=False
                                ):
            X_train_fold, y_train_fold = X_train[train_i], y_train[train_i]
            X_val_fold, y_val_fold = X_train[val_i], y_train[val_i]

            model.fit(X_train_fold, y_train_fold)

            pred_train = model.predict(X_train_fold)
            pred_val = model.predict(X_val_fold)

            scores["train_acc"].append(
                scorer_accuracy(y_train_fold, pred_train)
            )
            scores["train_f1"].append(
                scorer_f1(y_train_fold, pred_train)
            )

            scores["cv_acc"].append(
                scorer_accuracy(y_val_fold, pred_val)
            )
            scores["cv_f1"].append(
                scorer_f1(y_val_fold, pred_val)
            )

        result = {
            "model": scores["model"],

            "train_acc": np.round(np.mean(scores["train_acc"]), 4),
            "train_f1": np.round(np.mean(scores["train_f1"]), 4),

            "cv_acc": np.round(np.mean(scores["cv_acc"]), 4),
            "cv_f1": np.round(np.mean(scores["cv_f1"]), 4),
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

            model.fit(X_train, y_train)

            pred_val = model.predict(X_val)

            result["val_acc"] = np.round(scorer_accuracy(y_val, pred_val), 4)
            result["val_f1"] = np.round(scorer_f1(y_val, pred_val), 4)

            results.append(result)

        return pl.DataFrame(results)

    return scorer_accuracy, start_fits


@app.cell
def _(SEED, optuna, start_fits):
    def start_optuna(models, param_distributions, cv, scoring, X_train, y_train, random_state=SEED, n_trials=100, n_jobs=-1):
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
                n_jobs=n_jobs
            )

            search.fit(X_train, y_train)
            print(f"Ended Optuna for model: {name}", flush=True)

            searches[name] = search

            for param_name, param_value in search.best_params_.items():
                optuna_params[name][param_name] = param_value

        return optuna_params


    def fit_optuna(models, param_distributions, cv, scoring, X_train, y_train, X_val, y_val, random_state=SEED, n_trials=100, n_jobs=-1):
        params = start_optuna(
            models,
            param_distributions,
            cv,
            scoring,
            X_train, y_train,
            random_state,
            n_trials,
            n_jobs
        )
        results = start_fits(
            models,
            params,
            cv,
            X_train, y_train,
            X_val, y_val
        )

        return results, params

    return (fit_optuna,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## линейная регрессия
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
            "model__alpha": optuna.distributions.FloatDistribution(
                1e-4, 200, log=True
            ),
        },

        "Ridge": {
            "model__alpha": optuna.distributions.FloatDistribution(
                1e-4, 200, log=True
            ),
        },

        "ElasticNet": {
            "model__alpha": optuna.distributions.FloatDistribution(
                1e-4, 200, log=True
            ),
            "model__l1_ratio": optuna.distributions.FloatDistribution(
                0.0, 1.0
            ),
        },
        'LogisticRegression': {

        }
    }

    models_lin = {
        'LinearRegression': Pipeline([
            ('preprocess', preprocess),
            ('model', LinearRegression()),
        ]),
        'Lasso': Pipeline([
            ('preprocess', preprocess),
            ('model', Lasso()),
        ]),
        'Ridge': Pipeline([
            ('preprocess', preprocess),
            ('model', Ridge()),
        ]),
        'ElasticNet': Pipeline([
            ('preprocess', preprocess),
            ('model', ElasticNet()),
        ]),
        'LogisticRegression': Pipeline([
            ('preprocess', preprocess),
            ('model', LogisticRegression()),
        ])
    }


    cv = StratifiedKFold(n_splits=5)
    return cv, models_lin, param_distributions_lin


@app.cell
def _(X_train, X_val, cv, models_lin, start_fits, y_train, y_val):
    results_lin = start_fits(
        models_lin,
        None,
        cv,
        X_train,
        y_train,
        X_val,
        y_val
    )

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
        optuna_results_lin,optuna_params_lin = fit_optuna(
            models_lin, param_distributions_lin, cv, make_scorer(scorer_accuracy),
            X_train, y_train, X_val, y_val
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
        'KNeighborsClassifier': Pipeline([
            ('preprocess', preprocess),
            ('model', KNeighborsClassifier()),
        ]),
    }
    # n_neighbors, weight, metric
    param_distributions_knn = {
        'KNeighborsClassifier': {
            'model__n_neighbors': optuna.distributions.IntDistribution(1, 100),
            'model__weights': optuna.distributions.CategoricalDistribution(['uniform', 'distance']),
            'model__metric': optuna.distributions.CategoricalDistribution(
                [
                    'cityblock', 'cosine', 'euclidean', 'l1', 'l2',
                    'manhattan', 'nan_euclidean', 'minkowski'
                ]
            )
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
            models_knn, param_distributions_knn, cv, make_scorer(scorer_accuracy),
            X_train, y_train, X_val, y_val)

    optuna_results_knn
    return


@app.cell
def _(X_train, X_val, cv, models_knn, start_fits, y_train, y_val):
    default_result_knn = start_fits(models_knn, None, cv, X_train, y_train, X_val, y_val)
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
        'DecisionTreeClassifier': Pipeline([
            ('preprocess', preprocess),
            ('model', DecisionTreeClassifier()),
        ])
    }

    param_distributions_tree = {
        'DecisionTreeClassifier': {
            'model__criterion': optuna.distributions.CategoricalDistribution(['gini', 'entropy', 'log_loss']),
            'model__min_samples_split': optuna.distributions.IntDistribution(2, 10),
            'model__min_samples_leaf': optuna.distributions.IntDistribution(1, 10)
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
            X_train, y_train, X_val, y_val
        )
    optuna_results_tree, optuna_params_tree
    return


@app.cell
def _(X_train, X_val, cv, model_tree, start_fits, y_train, y_val):
    start_fits(model_tree, None, cv,
                X_train, y_train, X_val, y_val)
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
        'RandomForestClassifier': Pipeline([
            ('preprocess', preprocess),
            ('model', RandomForestClassifier(n_jobs=-1))
        ])
    }

    param_distributions_rf = {
        'RandomForestClassifier': {
            'model__n_estimators': optuna.distributions.IntDistribution(10, 200),
            'model__max_depth': optuna.distributions.IntDistribution(2, 60),
            'model__min_samples_split': optuna.distributions.IntDistribution(2, 20),
            'model__min_samples_leaf': optuna.distributions.IntDistribution(1, 10),
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
            model_rf, param_distributions_rf,
            cv, make_scorer(scorer_accuracy),
            X_train, y_train, X_val, y_val
        )
    optuna_results_rf, optuna_params_rf
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Boostings
    """)
    return


@app.cell
def _(
    GradientBoostingClassifier,
    LGBMClassifier,
    Pipeline,
    XGBClassifier,
    optuna,
    preprocess,
):
    model_gb = {
        'GradientBoostingClassifier': Pipeline([
            ('preprocess', preprocess),
            ('model', GradientBoostingClassifier()),
        ]),
        'LGBMClassifier': Pipeline([
            ('preprocess', preprocess),
            ('model', LGBMClassifier(n_jobs=1, verbosity=-1)),
        ]),
        'XGBClassifier': Pipeline([
            ('preprocess', preprocess),
            ('model', XGBClassifier(n_jobs=1)),
        ]),
    }

    param_distributions_gb = {
        'GradientBoostingClassifier': {
            'model__loss': optuna.distributions.CategoricalDistribution(['log_loss', 'exponential']),
            'model__learning_rate': optuna.distributions.FloatDistribution(1e-4, 1e-1),
            'model__n_estimators': optuna.distributions.IntDistribution(10, 200),
            'model__min_samples_split': optuna.distributions.IntDistribution(2, 50),
            'model__min_samples_leaf': optuna.distributions.IntDistribution(1, 50),
            'model__max_depth': optuna.distributions.IntDistribution(1, 10)
        },
        'LGBMClassifier': {
            'model__num_leaves': optuna.distributions.IntDistribution(3, 100),
            'model__max_depth': optuna.distributions.IntDistribution(1, 20),
            'model__learning_rate': optuna.distributions.FloatDistribution(1e-4, 1e-1),
            'model__n_estimators': optuna.distributions.IntDistribution(100, 1000),
            'model__reg_alpha': optuna.distributions.FloatDistribution(1e-3, 1),
            'model__reg_lambda': optuna.distributions.FloatDistribution(1e-3, 1),
        },
        'XGBClassifier': {
            'model__eta': optuna.distributions.FloatDistribution(1e-4, 1),
            'model__gamma': optuna.distributions.IntDistribution(0, 100),
            'model__max_depth': optuna.distributions.IntDistribution(1, 32),
        }
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
        optuna_results_gb, optuna_params_gb = fit_optuna(model_gb, param_distributions_gb, cv, make_scorer(scorer_accuracy),
            X_train, y_train, X_val, y_val)

    try:
        optuna_results_gb, optuna_params_gb
    except:
        print('turn OPTUNA_SEARCH to True, for enable optuna search')
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
def _(X_train, X_val, clone, preprocess, torch, y_train, y_val):
    nn_preprocess = clone(preprocess)
    X_train_np = nn_preprocess.fit_transform(X_train)
    X_val_np = nn_preprocess.transform(X_val)

    X_train_t = torch.as_tensor(X_train_np, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.to_numpy(), 
    dtype=torch.float32).unsqueeze(1)

    X_val_t = torch.as_tensor(X_val_np, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.to_numpy(), 
    dtype=torch.float32).unsqueeze(1)
    return X_train_t, X_val_t, y_train_t, y_val_t


@app.cell
def _(DataLoader, TensorDataset, X_train_t, X_val_t, y_train_t, y_val_t):
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
                    nn.Linear(inp, 25),
                    nn.ReLU(),
                    nn.Linear(25, out),
                    nn.Sigmoid()
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
                    nn.Dropout(self.dropout)
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

    opt:
    - [] Adadelta
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
def _(X_train_t):
    X_train_t.shape[1]
    return


@app.cell
def _(MyModel, X_train_t, nn, torch):
    model2 = MyModel(X_train_t.shape[1], 1)
    loss_model2 = nn.BCEWithLogitsLoss()
    opt2 = torch.optim.AdamW(model2.parameters(), lr=3e-4)
    sched2 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt2, )
    return loss_model2, model2, opt2, sched2


@app.cell
def _(MLPModel, X_train_t, nn, torch):
    model1 = MLPModel(X_train_t.shape[1], 1)
    loss_model1 = nn.BCEWithLogitsLoss()
    opt1 = torch.optim.Adam(model1.parameters(), lr=0.01)
    return loss_model1, model1, opt1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Обучение dnn
    """)
    return


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
            correct += ((pred >= 0.5) == target).sum().item()


        return train_loss / len(train_loader.dataset), correct / len(train_loader.dataset)

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
                correct += ((pred >= 0.5) == target).sum().item()


        return val_loss / len(val_loader.dataset), correct / len(val_loader.dataset)

    return single_eval, single_train


@app.cell
def _(DEVICE, optuna, single_eval, single_train, torch):
    def epoch_train(model, loss_model, opt, train_loader, val_loader, sched=None, epoches=15, trial=None):
        model.to(DEVICE)
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        summary = []
        for epoch in range(epoches):
            single_train_loss, single_train_acc = single_train(model, loss_model, opt, train_loader)
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
                f'[{epoch + 1}/{epoches}],train_loss={round(single_train_loss, 4)}|\
                train_acc={round(single_train_acc, 4)}|val_loss={round(single_val_loss, 4)}|\
                val_acc={round(single_val_acc, 4)}'
            )

        return {
            'model': model,
            'loss': loss_model,
            'opt': opt,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'summary': summary,
        }

    return (epoch_train,)


@app.cell
def _(epoch_train, plt):
    def plot_res(result, result2=None):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        results = [('Model 1', result)]
        if result2 is not None:
            results.append(('Model 2', result2))

        for name, res in results:
            epochs = range(1, len(res['train_loss']) + 1)
            for i, metric in enumerate(('loss', 'acc')):
                line, = ax[i].plot(epochs, res[f'train_{metric}'], label=f'{name}: train')
                ax[i].plot(epochs, res[f'val_{metric}'], '--',
                           color=line.get_color(), label=f'{name}: validation')

        for axis, title in zip(ax, ('Loss', 'Accuracy')):
            axis.set(title=title, xlabel='Epoch', ylabel=title)
            axis.grid(alpha=0.3)
            axis.legend()

        fig.tight_layout()
        return fig


    def compare_dnn(model1, loss_model1, opt1,
                    model2, loss_model2, opt2,
                    train_loader, val_loader, sched2=None, epoches=15):
        results1 = epoch_train(model1, loss_model1, opt1, train_loader, val_loader, epoches=epoches)
        results2 = epoch_train(model2, loss_model2, opt2, train_loader, val_loader, sched=sched2, epoches=epoches)
        fig = plot_res(results1, results2)
        return results1, results2, fig


    return (compare_dnn,)


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
    results1, results2, fig = compare_dnn(model1, loss_model1, opt1, model2, loss_model2, opt2, train_loader, val_loader, sched2=sched2, epoches=EPOCHES)
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### optuna dnn
    """)
    return


@app.cell
def _(
    MyModel,
    checkpoint_dir,
    epoch_train,
    nn,
    num_slice,
    torch,
    train_loader,
    val_loader,
):
    def objective(trial, use_embedding=False, name=''):
        optuna_dir = checkpoint_dir / f'optuna_{name}'
        optuna_dir.mkdir(exist_ok=True, parents=True)

        hidden_sizes = [
        trial.suggest_categorical("hidden_1", [250, 500]),
        trial.suggest_categorical("hidden_2", [100, 200]),
        trial.suggest_categorical("hidden_3", [25, 50]),
        ]
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        if use_embedding:
            sex_emb_dim = trial.suggest_categorical("sex_emb_dim", [1, 2, 3])
            embarked_emb_dim = trial.suggest_categorical(
                "embarked_emb_dim", [1, 2, 3]
            )

        n_num = num_slice.stop - num_slice.start
        inp = n_num + sex_emb_dim + embarked_emb_dim

        inp = 10
        out = 1

        model = MyModel(
            inp,
            out,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        )

        loss_model = nn.BCEWithLogitsLoss()
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)

        result = epoch_train(
            model,
            loss_model,
            opt,
            train_loader,
            val_loader,
            sched,
            15,
            trial     
        )

        accuracy = result['val_acc'][-1]
        if use_embedding:
            torch.save(
            {
                "model_state_dict": model.state_dict(),
                "inp": inp,
                "out": out,
                "hidden_sizes": hidden_sizes,
                "dropout": dropout,
                'sex_emb_dim': sex_emb_dim,
                'embarked_emb_dim': embarked_emb_dim,
                "accuracy": accuracy,
            },
            optuna_dir / f"trial_{trial.number}.pt",
        )
        else:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "inp": inp,
                    "out": out,
                    "hidden_sizes": hidden_sizes,
                    "dropout": dropout,
                    "accuracy": accuracy,
                },
                optuna_dir / f"trial_{trial.number}.pt",
            )

        return accuracy

    return (objective,)


app._unparsable_cell(
    r"""
    if OPTUNA_SEARCH:
        study = optuna.create_study(direction='maximize')
        study.optimize(labmda trial: objective(trial, name='default'), n_trials=100)

        best_model = torch.load(checkpoint_dir / f'trial_{study.best_trial.number}.pt', map_location=DEVICE, weights_only=True)
        print(best_model['accuracy'])
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Add embedding to dnn
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
    categorical_embedding = Pipeline([
        ("imputer", SimpleImputer(
            missing_values=None,
            strategy="most_frequent",
        )),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int64,
        )),
    ])

    preprocess_embedding = ColumnTransformer([
        ('numeric', numeric, ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']),
        ('categorical', categorical_embedding, ['Sex', 'Embarked']),
    ])
    return (preprocess_embedding,)


@app.cell
def _(X_train, X_val, clone, preprocess_embedding, torch, y_train, y_val):
    nn_preprocess_embedding = clone(preprocess_embedding)
    X_train_np_emb = nn_preprocess_embedding.fit_transform(X_train)
    X_val_np_emb = nn_preprocess_embedding.transform(X_val)

    num_slice = nn_preprocess_embedding.output_indices_["numeric"]
    cat_slice = nn_preprocess_embedding.output_indices_["categorical"]

    X_train_t_emb = torch.as_tensor(
        X_train_np_emb, dtype=torch.float32
    )
    y_train_t_emb = torch.tensor(y_train.to_numpy(), 
    dtype=torch.float32).unsqueeze(1)


    X_val_t_emb = torch.as_tensor(
        X_val_np_emb, dtype=torch.float32
    )
    y_val_t_emb = torch.tensor(y_val.to_numpy(),
    dtype=torch.float32).unsqueeze(1)
    return cat_slice, num_slice


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### loaders
    """)
    return


@app.cell
def _(DataLoader, TensorDataset, X_train_t, X_val_t, y_train_t, y_val_t):
    train_loader_emb = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=64,
        shuffle=True,
    )

    val_loader_emb = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=64,
    )
    return train_loader_emb, val_loader_emb


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### model
    """)
    return


@app.cell
def _(nn, torch):
    class MyModelEmb(nn.Module):
        def __init__(self, inp, out, num_slice, cat_slice, hidden_sizes=[500, 200, 50], dropout=0.0, sex_emb_dim=2, embarked_emb_dim=2):
            super().__init__()
            self.inp = inp
            self.out = out
            self.hidden_sizes = [inp] + hidden_sizes
            self.dropout = dropout
            self._build_model()
            self.sex_emb = nn.Embedding(3, sex_emb_dim)
            self.embarked_emb = nn.Embedding(4, embarked_emb_dim)
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
                    nn.Dropout(self.dropout)
                    ]
                )
            layers.append(nn.Linear(self.hidden_sizes[-1], self.out))
            self.model = nn.Sequential(*layers) 

        def forward(self, x):
            x_num = x[:, self.num_slice]
            x_cat = x[:, self.cat_slice].to(torch.int32) + 1
            sex_vector = self.sex_emb(x_cat[:, 0])
            embarked_vector = self.embarked_emb(x_cat[:, 1])
            x = torch.cat([x_num, sex_vector, embarked_vector], dim=1)
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
    model_emb = MyModelEmb(5 + 2 * 2, 1, num_slice, cat_slice)
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
    results = epoch_train(model_emb, loss_model_emb, opt_emb, train_loader_emb, val_loader_emb, sched_emb)
    max(results['val_acc'])
    return


@app.cell
def _(DEVICE, checkpoint_dir, objective, optuna, torch):
    if True:
        name_exp = 'emb_v1'
        study_emb = optuna.create_study(direction='maximize')
        study_emb.optimize(lambda trial: objective(trial, use_embedding=True, name=name_exp), n_trials=100)
    

        best_model_emb = torch.load(checkpoint_dir / f'optuna_{name_exp}' / f'trial_{study_emb.best_trial.number}.pt', map_location=DEVICE, weights_only=True)
        print(best_model_emb['accuracy'])
    return (best_model_emb,)


@app.cell
def _(best_model_emb):
    print(best_model_emb['accuracy'])
    print(best_model_emb)
    return


if __name__ == "__main__":
    app.run()
