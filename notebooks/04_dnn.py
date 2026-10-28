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
    # DNN: архитектура, обучение и embeddings

    Перенесены MLPModel, MyModel, MyModelEmb и функции обучения из исходного
    ноутбука. Набор признаков фиксирован: семь базовых признаков, FamilySize,
    Title и исходные Cabin/Ticket. У ordinary и embedding веток одни строки.

    `RUN_DNN` включает обычное сравнение, `RUN_ABLATIONS` — выбранные
    отдельные проверки, `OPTUNA_SEARCH` — два поиска. По умолчанию они
    выключены. `USE_CV=False` выбирает holdout, `USE_CV=True` — 5-fold CV
    на тренировочных 80%, как у классических моделей. Primary metric —
    accuracy: лучшая эпоха для holdout, последняя эпоха каждого фолда для
    CV, как в исходном `cv_train`. Подбор Optuna использует тот же режим.
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

    from final_proj.utils import set_seed

    return (
        ColumnTransformer,
        DataLoader,
        OneHotEncoder,
        OrdinalEncoder,
        Path,
        Pipeline,
        SimpleImputer,
        StandardScaler,
        StratifiedKFold,
        TensorDataset,
        clone,
        copy,
        datetime,
        hashlib,
        joblib,
        json,
        nn,
        np,
        optuna,
        pl,
        plt,
        set_seed,
        timezone,
        torch,
        tqdm,
        train_test_split,
    )


@app.cell
def _(datetime, set_seed, timezone):
    SEED = 0xFACED
    DEVICE = 'cpu'
    USE_CV = True
    RUN_DNN = False
    RUN_ABLATIONS = False
    OPTUNA_SEARCH = True
    N_TRIALS = 20
    SAVE_RESULTS = False
    RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    set_seed(SEED)
    return (
        DEVICE,
        N_TRIALS,
        OPTUNA_SEARCH,
        RUN_ABLATIONS,
        RUN_DNN,
        RUN_ID,
        SAVE_RESULTS,
        SEED,
        USE_CV,
    )


@app.cell
def _(RUN_ID, mo):
    base_path = mo.notebook_location().parent
    data_path = base_path / 'data/titanic'
    checkpoint_dir = base_path / 'checkpoints' / RUN_ID
    results_path = base_path / 'notebooks/results'
    return base_path, checkpoint_dir, data_path, results_path


@app.cell
def _(data_path, pl):
    train_dataset = pl.read_csv(data_path / 'train.csv')
    test_dataset = pl.read_csv(data_path / 'test.csv')
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
    def make_features(df):
        """Создать фиксированные входные признаки для сравнения двух DNN."""
        df = df.with_columns(
            pl.col('Name').str.extract(r',\s*([^.]+)\.', 1).alias('Title'),
            (pl.col('SibSp') + pl.col('Parch') + 1).alias('FamilySize'),
        )
        numeric_columns = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize']
        categorical_columns = ['Sex', 'Embarked', 'Cabin', 'Ticket', 'Title']
        return df.select(numeric_columns + categorical_columns), numeric_columns, categorical_columns

    def make_preprocess(numeric_columns, categorical_columns, encoding='onehot', scale=True):
        """Вернуть unfitted preprocessing: fit вызывается внутри train-фолда."""
        numeric_steps = [('imputer', SimpleImputer(strategy='median'))]
        if scale:
            numeric_steps.append(('scaler', StandardScaler()))
        numeric = Pipeline(numeric_steps)
        if encoding == 'onehot':
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        elif encoding == 'ordinal':
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        else:
            raise ValueError(f'Unknown encoding: {encoding}')
        categorical = Pipeline([
            ('imputer', SimpleImputer(missing_values=None, strategy='most_frequent')),
            ('encoder', encoder),
        ])
        return ColumnTransformer([
            ('numeric', numeric, numeric_columns),
            ('categorical', categorical, categorical_columns),
        ])

    return make_features, make_preprocess


@app.cell
def _(
    SEED,
    StratifiedKFold,
    make_features,
    make_preprocess,
    np,
    test_dataset,
    train_dataset,
    train_test_split,
):
    X, numeric_columns, categorical_columns = make_features(train_dataset)
    X_test, _, _ = make_features(test_dataset)
    y = train_dataset['Survived']
    train_i, val_i = train_test_split(
        np.arange(len(X)), train_size=0.8, stratify=y, random_state=SEED,
    )
    X_train, X_val = X[train_i], X[val_i]
    y_train, y_val = y[train_i], y[val_i]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    preprocess = make_preprocess(numeric_columns, categorical_columns)
    return (
        X_train,
        X_val,
        categorical_columns,
        cv,
        numeric_columns,
        preprocess,
        train_i,
        val_i,
        y_train,
        y_val,
    )


@app.cell
def _(
    RUN_ID,
    SAVE_RESULTS,
    SEED,
    USE_CV,
    data_path,
    hashlib,
    json,
    pl,
    results_path,
    train_dataset,
    train_i,
    val_i,
):
    def save_experiments(results, name, parameters=None,
                         score_role='fixed_configuration_cv', extra=None):
        """Сохранить метрики, конфигурацию и точные разбиения завершённого запуска."""
        from datetime import datetime, timezone
        from importlib.metadata import version
        if results is None or not SAVE_RESULTS:
            return None
        results_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        path = results_path / f'{timestamp}_{name}.json'
        record = {
            'run_id': path.stem, 'session_id': RUN_ID, 'name': name, 'feature_set': 'all',
            'seed': SEED, 'metric': 'accuracy', 'score_role': score_role,
            'protocol': ('stratified_80_20_holdout_5fold_cv_last_epoch_v1' if USE_CV
                         else 'stratified_80_20_holdout_epoch_selection_v1'),
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Две независимые ветки предобработки
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
    return X_train_t, X_val_t, nn_preprocess, y_train_t, y_val_t


@app.cell
def _(
    DataLoader,
    SEED,
    TensorDataset,
    X_train_t,
    X_val_t,
    torch,
    y_train_t,
    y_val_t,
):
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=64,
    )
    return train_loader, val_loader


@app.cell
def _(nn):
    class MLPModel(nn.Module):
        def __init__(self, inp, out):
            super().__init__()
            self.model = nn.Sequential(
                    nn.Linear(inp, 25),
                    nn.ReLU(),
                    nn.Linear(25, out)
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


@app.cell
def _(categorical_columns, make_preprocess, numeric_columns):
    preprocess_embedding = make_preprocess(numeric_columns, categorical_columns, encoding='ordinal')
    return (preprocess_embedding,)


@app.cell
def _(X_train, X_val, clone, preprocess_embedding, torch, y_train, y_val):
    nn_preprocess_embedding = clone(preprocess_embedding)
    X_train_np_emb = nn_preprocess_embedding.fit_transform(X_train)
    X_val_np_emb = nn_preprocess_embedding.transform(X_val)

    cat_sizes = [len(values) + 1 for values in nn_preprocess_embedding.named_transformers_["categorical"].named_steps["encoder"].categories_]

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
    return (
        X_train_t_emb,
        X_val_t_emb,
        cat_sizes,
        cat_slice,
        nn_preprocess_embedding,
        num_slice,
        y_train_t_emb,
        y_val_t_emb,
    )


@app.cell
def _(
    DataLoader,
    SEED,
    TensorDataset,
    X_train_t_emb,
    X_val_t_emb,
    torch,
    y_train_t_emb,
    y_val_t_emb,
):
    train_loader_emb = DataLoader(
        TensorDataset(X_train_t_emb, y_train_t_emb),
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    val_loader_emb = DataLoader(
        TensorDataset(X_val_t_emb, y_val_t_emb),
        batch_size=64,
    )
    return train_loader_emb, val_loader_emb


@app.cell
def _(nn, torch):
    class MyModelEmb(nn.Module):
        def __init__(self, inp, out, num_slice, cat_slice, hidden_sizes=[500, 200, 50], dropout=0.0, sex_emb_dim=2, embarked_emb_dim=2, cabin_emb_dim=2, ticket_emb_dim=2, title_emb_dim=2, cat_sizes=None):
            super().__init__()
            if cat_sizes is None:
                raise ValueError("Передай размеры словарей из train encoder через cat_sizes")
            self.inp = inp
            self.out = out
            self.hidden_sizes = [inp] + hidden_sizes
            self.dropout = dropout
            self._build_model()
            self.sex_emb = nn.Embedding(cat_sizes[0], sex_emb_dim)
            self.embarked_emb = nn.Embedding(cat_sizes[1], embarked_emb_dim)
            self.cabin_emb = nn.Embedding(cat_sizes[2], cabin_emb_dim)
            self.ticket_emb = nn.Embedding(cat_sizes[3], ticket_emb_dim)
            self.title_emb = nn.Embedding(cat_sizes[4], title_emb_dim)
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
            cabin_vector = self.cabin_emb(x_cat[:, 2])
            ticket_vector = self.ticket_emb(x_cat[:, 3])
            title_vector = self.title_emb(x_cat[:, 4])
            x = torch.cat([x_num, sex_vector, embarked_vector, cabin_vector, ticket_vector, title_vector], dim=1)
            return self.model(x)

    return (MyModelEmb,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Размер входа one-hot берётся из тензора. Размеры embedding-словарей
    получены из категорий train encoder, плюс один индекс для неизвестного.
    Срезы numeric/categorical берутся из output_indices_. Обе ветки обучают
    preprocessing только на X_train; X_val получает transform. В режиме
    CV preprocessing, входной размер и embedding-словари создаются заново
    внутри каждого фолда.

    `epoch_train` теперь восстанавливает веса лучшей эпохи. Сохраняемая
    accuracy относится именно к этим весам, а не к последней эпохе.
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
            correct += ((pred >= 0) == target).sum().item()


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
                correct += ((pred >= 0) == target).sum().item()


        return val_loss / len(val_loader.dataset), correct / len(val_loader.dataset)

    return single_eval, single_train


@app.cell
def _(DEVICE, copy, optuna, single_eval, single_train, torch):
    def epoch_train(model, loss_model, opt, train_loader, val_loader, sched=None, epoches=15, trial=None):
        model.to(DEVICE)
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        summary = []
        best_accuracy = -1.0
        best_state = None
        best_epoch = None
        for epoch in range(epoches):
            single_train_loss, single_train_acc = single_train(model, loss_model, opt, train_loader)
            single_val_loss, single_val_acc = single_eval(model, loss_model, val_loader)

            if single_val_acc > best_accuracy:
                best_accuracy = single_val_acc
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1

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

        model.load_state_dict(best_state)
        return {
            'model': model,
            'best_epoch': best_epoch,
            'accuracy': best_accuracy,
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
def _(DataLoader, TensorDataset, clone, torch):
    def get_loader(X_train, y_train, X_val, y_val, preprocess, return_preprocess=False):
        nn_preprocess = clone(preprocess)
        X_train_np = nn_preprocess.fit_transform(X_train)
        X_val_np = nn_preprocess.transform(X_val)

        if hasattr(X_train_np, 'toarray'):
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

        if return_preprocess:
            return train_loader, val_loader, nn_preprocess
        return train_loader, val_loader

    return (get_loader,)


@app.cell
def _(MyModelEmb, SEED, epoch_train, get_loader, np, optuna, set_seed, tqdm):
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

            fold_train_loader, fold_val_loader, nn_preprocess = get_loader(
                X_train, y_train, X_val, y_val, preprocess, return_preprocess=True
            )

            inp = fold_train_loader.dataset.tensors[0].shape[1]
            fold_model_params = model_params.copy()
            fold_model_params.pop('inp', None)
            if model_cls is MyModelEmb:
                num_slice = nn_preprocess.output_indices_['numeric']
                cat_slice = nn_preprocess.output_indices_['categorical']
                inp = num_slice.stop - num_slice.start + sum(
                    model_params.get(f'{column}_emb_dim', 2)
                    for column in ['sex', 'embarked', 'cabin', 'ticket', 'title']
                )
                fold_model_params.update({
                    'num_slice': num_slice,
                    'cat_slice': cat_slice,
                    'cat_sizes': [len(values) + 1 for values in
                                  nn_preprocess.named_transformers_['categorical']
                                  .named_steps['encoder'].categories_],
                })
            fold_model = model_cls(inp=inp, **fold_model_params)

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
def _(SEED, epoch_train, plt):
    def plot_res(result, result2=None):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        results = [('Model 1', result)]
        if result2 is not None:
            results.append(('Model 2', result2))

        for name, res in results:
            history = res['train_loss']
            epochs = range(1, len(history['mean'] if isinstance(history, dict) else history) + 1)
            for i, metric in enumerate(('loss', 'acc')):
                train_values, val_values = res[f'train_{metric}'], res[f'val_{metric}']
                if isinstance(train_values, dict):
                    line, = ax[i].plot(epochs, train_values['mean'], label=f'{name}: train')
                    ax[i].plot(epochs, val_values['mean'], '--',
                               color=line.get_color(), label=f'{name}: validation')
                    for values in [train_values, val_values]:
                        ax[i].fill_between(
                            epochs,
                            [mean - std for mean, std in zip(values['mean'], values['std'])],
                            [mean + std for mean, std in zip(values['mean'], values['std'])],
                            color=line.get_color(), alpha=0.15,
                        )
                else:
                    line, = ax[i].plot(epochs, train_values, label=f'{name}: train')
                    ax[i].plot(epochs, val_values, '--',
                               color=line.get_color(), label=f'{name}: validation')

        for axis, title in zip(ax, ('Loss', 'Accuracy')):
            axis.set(title=title, xlabel='Epoch', ylabel=title)
            axis.grid(alpha=0.3)
            axis.legend()

        fig.tight_layout()
        return fig


    def compare_dnn(model1, loss_model1, opt1,
                    model2, loss_model2, opt2,
                    train_loader, val_loader, sched2=None, epoches=15,
                    train_loader2=None, val_loader2=None, sched1=None):
        if train_loader2 is None:
            train_loader2 = train_loader
        if val_loader2 is None:
            val_loader2 = val_loader

        train_loader.generator.manual_seed(SEED)
        results1 = epoch_train(
            model1, loss_model1, opt1, train_loader, val_loader,
            sched=sched1, epoches=epoches,
        )
        train_loader2.generator.manual_seed(SEED)
        results2 = epoch_train(
            model2, loss_model2, opt2, train_loader2, val_loader2,
            sched=sched2, epoches=epoches,
        )

        fig = plot_res(results1, results2)
        return results1, results2, fig

    return compare_dnn, plot_res


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Простая MLP и глубокая сеть
    """)
    return


@app.cell
def _(
    MLPModel,
    MyModel,
    RUN_DNN,
    SEED,
    USE_CV,
    X_train,
    X_train_t,
    compare_dnn,
    cv,
    cv_train,
    nn,
    plot_res,
    preprocess,
    save_experiments,
    set_seed,
    torch,
    train_loader,
    val_loader,
    y_train,
):
    results1, results2, fig = None, None, None
    if RUN_DNN:
        if USE_CV:
            results1 = cv_train(
                MLPModel, {'out': 1}, nn.BCEWithLogitsLoss(),
                torch.optim.Adam, {'lr': 0.01}, {}, cv, X_train, y_train,
                preprocess, epoches=10,
            )
            results2 = cv_train(
                MyModel, {'out': 1, 'hidden_sizes': [500, 200, 50], 'dropout': 0.0},
                nn.BCEWithLogitsLoss(), torch.optim.AdamW, {'lr': 3e-4}, {},
                cv, X_train, y_train, preprocess,
                sched_cls=torch.optim.lr_scheduler.ReduceLROnPlateau, epoches=10,
            )
            fig = plot_res(results1, results2)
            _records = [{'model': _name,
                         'cv_acc': _res['metrics']['val_acc']['mean'],
                         'cv_std': _res['metrics']['val_acc']['std'],
                         'cv_fold_scores': [_fold['val_acc'][-1] for _fold in _res['folds']],
                         'train_acc': _res['metrics']['train_acc']['mean'],
                         'history': {k: _res[k] for k in ['train_loss', 'train_acc', 'val_loss', 'val_acc']}}
                        for _name, _res in [('MLPModel', results1), ('MyModel', results2)]]
        else:
            set_seed(SEED)
            model1 = MLPModel(X_train_t.shape[1], 1)
            loss_model1 = nn.BCEWithLogitsLoss()
            opt1 = torch.optim.Adam(model1.parameters(), lr=0.01)
            set_seed(SEED)
            model2 = MyModel(X_train_t.shape[1], 1)
            loss_model2 = nn.BCEWithLogitsLoss()
            opt2 = torch.optim.AdamW(model2.parameters(), lr=3e-4)
            sched2 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt2)
            results1, results2, fig = compare_dnn(
                model1, loss_model1, opt1, model2, loss_model2, opt2,
                train_loader, val_loader, sched2=sched2, epoches=10,
            )
            _records = [{'model': _name, 'val_acc': _res['accuracy'],
                         'best_epoch': _res['best_epoch'],
                         'train_acc': _res['train_acc'][_res['best_epoch'] - 1],
                         'history': {k: _res[k] for k in ['train_loss', 'train_acc', 'val_loss', 'val_acc']}}
                        for _name, _res in [('MLPModel', results1), ('MyModel', results2)]]
        save_experiments(_records, 'dnn_comparison',
                         score_role='fixed_configuration_cv_last_epoch' if USE_CV else 'best_epoch_holdout',
                         extra={'epochs': 10, 'MLPModel': {'optimizer': 'Adam', 'lr': 0.01},
                                'MyModel': {'hidden_sizes': [500, 200, 50], 'dropout': 0.0,
                                            'optimizer': 'AdamW', 'lr': 3e-4, 'scheduler': 'plateau'}})
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Здесь одновременно различаются глубина, BatchNorm, activation,
    optimizer, LR и scheduler. Это сравнение двух полных конфигураций;
    по нему нельзя выделить вклад одного BatchNorm или оптимизатора.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Контролируемые проверки обучения
    """)
    return


@app.cell
def _(
    MyModel,
    RUN_ABLATIONS,
    SEED,
    USE_CV,
    X_train,
    X_train_t,
    cv,
    cv_train,
    epoch_train,
    nn,
    pl,
    preprocess,
    save_experiments,
    set_seed,
    torch,
    train_loader,
    val_loader,
    y_train,
):
    ablation_setups = {
        'Adam': {'optimizer': 'Adam', 'lr': 3e-4, 'dropout': 0.0, 'scheduler': 'none'},
        'AdamW': {'optimizer': 'AdamW', 'lr': 3e-4, 'dropout': 0.0, 'scheduler': 'none'},
        'dropout_01': {'optimizer': 'AdamW', 'lr': 3e-4, 'dropout': 0.1, 'scheduler': 'none'},
        'dropout_03': {'optimizer': 'AdamW', 'lr': 3e-4, 'dropout': 0.3, 'scheduler': 'none'},
        'cosine': {'optimizer': 'AdamW', 'lr': 3e-4, 'dropout': 0.0, 'scheduler': 'cosine'},
        'plateau': {'optimizer': 'AdamW', 'lr': 3e-4, 'dropout': 0.0, 'scheduler': 'plateau'},
    }
    selected_ablations = ['AdamW', 'dropout_01', 'cosine']
    ablation_results = None
    if RUN_ABLATIONS:
        _rows = []
        for _name in selected_ablations:
            _params = ablation_setups[_name]
            if USE_CV:
                _scheduler_cls, _scheduler_params = None, {}
                if _params['scheduler'] == 'cosine':
                    _scheduler_cls = torch.optim.lr_scheduler.CosineAnnealingLR
                    _scheduler_params = {'T_max': 15}
                elif _params['scheduler'] == 'plateau':
                    _scheduler_cls = torch.optim.lr_scheduler.ReduceLROnPlateau
                _res = cv_train(
                    MyModel, {'out': 1, 'dropout': _params['dropout']}, nn.BCEWithLogitsLoss(),
                    getattr(torch.optim, _params['optimizer']), {'lr': _params['lr']},
                    _scheduler_params, cv, X_train, y_train, preprocess,
                    sched_cls=_scheduler_cls, epoches=15,
                )
                _rows.append({'experiment': _name, 'model': 'MyModel',
                              'cv_acc': _res['metrics']['val_acc']['mean'],
                              'cv_std': _res['metrics']['val_acc']['std'],
                              'cv_fold_scores': [_fold['val_acc'][-1] for _fold in _res['folds']],
                              'params': _params})
            else:
                set_seed(SEED)
                train_loader.generator.manual_seed(SEED)
                _model = MyModel(X_train_t.shape[1], 1, dropout=_params['dropout'])
                _optimizer = getattr(torch.optim, _params['optimizer'])(_model.parameters(), lr=_params['lr'])
                _scheduler = None
                if _params['scheduler'] == 'cosine':
                    _scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(_optimizer, T_max=15)
                elif _params['scheduler'] == 'plateau':
                    _scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(_optimizer)
                _res = epoch_train(_model, nn.BCEWithLogitsLoss(), _optimizer,
                                   train_loader, val_loader, _scheduler, epoches=15)
                _rows.append({'experiment': _name, 'model': 'MyModel', 'val_acc': _res['accuracy'],
                              'best_epoch': _res['best_epoch'], 'params': _params})
        ablation_results = pl.DataFrame(_rows)
        save_experiments(_rows, 'dnn_ablations',
                         score_role='fixed_configuration_cv_last_epoch' if USE_CV else 'best_epoch_holdout',
                         extra={'hidden_sizes': [500, 200, 50], 'batch_size': 64, 'epochs': 15})
    ablation_results
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Сохранённые заметки об оптимизаторах
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Это исходные качественные заметки автора. «Нормал» без параметров,
    seed и чисел не считается измеренным сравнением. Ошибка Muon — ошибка
    совместимости с формой параметров, а не доказательство низкого качества.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Обычная DNN и embedding: отдельные Optuna studies
    """)
    return


@app.cell
def _(
    SEED,
    USE_CV,
    X_train,
    checkpoint_dir,
    cv,
    cv_train,
    epoch_train,
    nn,
    torch,
    y_train,
):
    def run_trial(
        trial,
        model,
        train_loader,
        val_loader,
        name,
        model_params,
        epochs=15,
        model_cls=None,
        preprocess=None,
        sched_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
        sched_params=None,
    ):
        optimizer_name = trial.suggest_categorical(
            "optimizer", ["AdamW", "Adam", "SGD"]
        )

        # Для SGD пробуем более высокий learning rate, чем для Adam/AdamW.
        if optimizer_name == "SGD":
            lr = trial.suggest_float("lr_sgd", 1e-3, 1e-1, log=True)
            optimizer_params = {
                "lr": lr,
                "momentum": trial.suggest_float("momentum", 0.0, 0.95),
            }
        else:
            lr = trial.suggest_float("lr_adam", 1e-4, 1e-2, log=True)
            optimizer_params = {"lr": lr}

        loss_fn = nn.BCEWithLogitsLoss()
        optimizer_class = getattr(torch.optim, optimizer_name)
        sched_params = sched_params or {}
        if USE_CV:
            result = cv_train(
                model_cls=model_cls,
                model_params=model_params,
                loss_model=loss_fn,
                opt_cls=optimizer_class,
                opt_params=optimizer_params,
                sched_cls=sched_cls,
                sched_params=sched_params,
                cv=cv,
                X=X_train,
                y=y_train,
                preprocess=preprocess,
                epoches=epochs,
                trial=trial,
            )
            accuracy = result['metrics']['val_acc']['mean']
        else:
            optimizer = optimizer_class(model.parameters(), **optimizer_params)
            scheduler = None if sched_cls is None else sched_cls(optimizer, **sched_params)
            result = epoch_train(
                model,
                loss_fn,
                optimizer,
                train_loader,
                val_loader,
                sched=scheduler,
                epoches=epochs,
                trial=trial,
            )
            accuracy = result["accuracy"]

        save_dir = checkpoint_dir / f"optuna_{name}"
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_params": model_params,
            "optimizer_name": optimizer_name,
            "optimizer_params": optimizer_params,
            "scheduler_name": sched_cls.__name__ if sched_cls is not None else None,
            "scheduler_params": sched_params,
            "lr": lr,
            "accuracy": accuracy,
            "feature_set": "all",
            "seed": SEED,
            "validation": "cv" if USE_CV else "holdout",
            "history": {key: result[key] for key in ["train_loss", "train_acc", "val_loss", "val_acc"]},
            "trial_params": trial.params,
        }
        if USE_CV:
            checkpoint.update({
                'metrics': result['metrics'],
                'folds': [
                    {
                        'fold': fold['fold'],
                        'best_epoch': fold['best_epoch'],
                        'best_val_acc': fold['best_val_acc'],
                        'final_val_acc': fold['val_acc'][-1],
                    }
                    for fold in result['folds']
                ],
                'summary': result['summary'],
            })
        else:
            checkpoint.update({
                'model_state_dict': model.state_dict(),
                'best_epoch': result['best_epoch'],
            })
        torch.save(checkpoint, save_dir / f"trial_{trial.number}.pt")

        return accuracy

    return (run_trial,)


@app.cell
def _(
    MyModel,
    SEED,
    USE_CV,
    X_train_t,
    preprocess,
    run_trial,
    set_seed,
    train_loader,
    val_loader,
):
    def objective_mlp(trial, name=''):
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
        model = None
        if not USE_CV:
            model_params['inp'] = X_train_t.shape[1]
            set_seed(SEED)
            train_loader.generator.manual_seed(SEED)
            model = MyModel(**model_params)

        return run_trial(
            trial, model, train_loader, val_loader, name, model_params,
            model_cls=MyModel, preprocess=preprocess,
        )

    return (objective_mlp,)


@app.cell
def _(
    MyModelEmb,
    SEED,
    USE_CV,
    cat_sizes,
    cat_slice,
    num_slice,
    preprocess_embedding,
    run_trial,
    set_seed,
    train_loader_emb,
    val_loader_emb,
):
    def objective_emb(trial, name=''):
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
            "out": 1,
            "hidden_sizes": hidden_sizes,
            "dropout": dropout,
            "sex_emb_dim": sex_emb_dim,
            "embarked_emb_dim": embarked_emb_dim,
            "cabin_emb_dim": cabin_emb_dim,
            "ticket_emb_dim": ticket_emb_dim,
            'title_emb_dim': title_emb_dim,
        }
        model = None
        if not USE_CV:
            model_params['cat_sizes'] = cat_sizes
            model_params['inp'] = n_num + sex_emb_dim + embarked_emb_dim + cabin_emb_dim + ticket_emb_dim + title_emb_dim
            set_seed(SEED)
            train_loader_emb.generator.manual_seed(SEED)
            model = MyModelEmb(
                num_slice=num_slice, cat_slice=cat_slice, **model_params
            )

        return run_trial(
            trial, model, train_loader_emb, val_loader_emb, name, model_params,
            model_cls=MyModelEmb, preprocess=preprocess_embedding,
        )

    return (objective_emb,)


@app.cell
def _(
    DEVICE,
    N_TRIALS,
    OPTUNA_SEARCH,
    SEED,
    USE_CV,
    checkpoint_dir,
    datetime,
    joblib,
    nn_preprocess,
    objective_mlp,
    optuna,
    timezone,
    torch,
):
    best_model = None
    if OPTUNA_SEARCH:
        name = 'default_optim_v1_' + ('cv_' if USE_CV else 'holdout_') + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(lambda trial: objective_mlp(trial, name=name), n_trials=N_TRIALS)

        best_model = torch.load(checkpoint_dir / f'optuna_{name}' /f'trial_{study.best_trial.number}.pt', map_location=DEVICE, weights_only=True)
        if not USE_CV:
            joblib.dump(nn_preprocess, checkpoint_dir / f'optuna_{name}' / 'preprocess.joblib')
        study.trials_dataframe().to_csv(checkpoint_dir / f'optuna_{name}' / 'trials.csv', index=False)
        print(best_model['accuracy'])
    return (best_model,)


@app.cell
def _(
    DEVICE,
    N_TRIALS,
    OPTUNA_SEARCH,
    SEED,
    USE_CV,
    checkpoint_dir,
    datetime,
    joblib,
    nn_preprocess_embedding,
    objective_emb,
    optuna,
    timezone,
    torch,
):
    best_model_emb = None
    if OPTUNA_SEARCH:
        name_exp = 'emb_optim_v1_' + ('cv_' if USE_CV else 'holdout_') + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        study_emb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
        study_emb.optimize(lambda trial: objective_emb(trial, name=name_exp), n_trials=N_TRIALS)


        best_model_emb = torch.load(checkpoint_dir / f'optuna_{name_exp}' / f'trial_{study_emb.best_trial.number}.pt', map_location=DEVICE, weights_only=True)
        if not USE_CV:
            joblib.dump(nn_preprocess_embedding, checkpoint_dir / f'optuna_{name_exp}' / 'preprocess.joblib')
        study_emb.trials_dataframe().to_csv(checkpoint_dir / f'optuna_{name_exp}' / 'trials.csv', index=False)
        print(best_model_emb['accuracy'])
    return (best_model_emb,)


@app.cell
def _(
    DEVICE,
    MyModel,
    MyModelEmb,
    USE_CV,
    best_model,
    best_model_emb,
    cat_slice,
    nn,
    num_slice,
    pl,
    save_experiments,
    single_eval,
    val_loader,
    val_loader_emb,
):
    frozen_comparison = None
    if best_model is not None and best_model_emb is not None:
        if USE_CV:
            frozen_comparison = pl.DataFrame([
                {'model': _name, 'cv_acc': _checkpoint['accuracy'],
                 'cv_std': _checkpoint['metrics']['val_acc']['std'],
                 'cv_fold_scores': [_fold['final_val_acc'] for _fold in _checkpoint['folds']]}
                for _name, _checkpoint in [('MyModel', best_model), ('MyModelEmb', best_model_emb)]
            ])
        else:
            _mlp = MyModel(**best_model['model_params']).to(DEVICE)
            _mlp.load_state_dict(best_model['model_state_dict'])
            _emb = MyModelEmb(num_slice=num_slice, cat_slice=cat_slice,
                              **best_model_emb['model_params']).to(DEVICE)
            _emb.load_state_dict(best_model_emb['model_state_dict'])
            frozen_comparison = pl.DataFrame([
                {'model': 'MyModel', 'recorded_accuracy': best_model['accuracy'],
                 'evaluated_accuracy': single_eval(_mlp, nn.BCEWithLogitsLoss(), val_loader)[1],
                 'best_epoch': best_model['best_epoch']},
                {'model': 'MyModelEmb', 'recorded_accuracy': best_model_emb['accuracy'],
                 'evaluated_accuracy': single_eval(_emb, nn.BCEWithLogitsLoss(), val_loader_emb)[1],
                 'best_epoch': best_model_emb['best_epoch']},
            ])
        save_experiments(frozen_comparison, 'dnn_optuna_comparison',
                         parameters={'MyModel': best_model['model_params'],
                                     'MyModelEmb': best_model_emb['model_params']},
                         score_role='selection_cv_last_epoch' if USE_CV else 'best_epoch_and_trial_holdout')
    frozen_comparison
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Загруженные модели только оцениваются. Дополнительного обучения после
    загрузки нет. Каждый новый запуск поиска получает уникальную папку;
    прежние checkpoint-папки не перезаписываются. В holdout preprocessing
    сохраняется рядом с соответствующей study. В CV, как в исходном коде,
    trial сохраняет метрики и истории фолдов; сравнение читает эти оценки.
    Порог логита равен нулю, loss — BCEWithLogitsLoss.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Исторические результаты DNN
    """)
    return


@app.cell
def _(pl):
    historical_dnn = [{'study_folder': 'optuna_default', 'saved_trials': 71, 'best_recorded_val_acc': 0.8547486033519553, 'input_dims': [10], 'best_trial': 'checkpoints/optuna_default/trial_25.pt', 'best_metadata': {'model_params': {'inp': 10, 'out': 1, 'hidden_sizes': [250, 200, 50], 'dropout': 0.004543658399719457}, 'lr': 0.0011838403271264722, 'accuracy': 0.8547486033519553}}, {'study_folder': 'optuna_default_optim_v1', 'saved_trials': 80, 'best_recorded_val_acc': 0.9050279329608939, 'input_dims': [10, 694, 710], 'best_trial': 'checkpoints/optuna_default_optim_v1/trial_55.pt', 'best_metadata': {'model_params': {'inp': 710, 'out': 1, 'hidden_sizes': [500, 100, 50], 'dropout': 0.006857054419362352}, 'optimizer_name': 'Adam', 'optimizer_params': {'lr': 0.0009862442232437746}, 'lr': 0.0009862442232437746, 'accuracy': 0.9050279329608939}}, {'study_folder': 'optuna_emb_optim_v1', 'saved_trials': 88, 'best_recorded_val_acc': 0.8659217877094972, 'input_dims': [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20], 'best_trial': 'checkpoints/optuna_emb_optim_v1/trial_32.pt', 'best_metadata': {'model_params': {'inp': 19, 'out': 1, 'hidden_sizes': [500, 200, 50], 'dropout': 0.070857423101546, 'sex_emb_dim': 3, 'embarked_emb_dim': 3, 'cabin_emb_dim': 3, 'ticket_emb_dim': 2, 'title_emb_dim': 2}, 'optimizer_name': 'Adam', 'optimizer_params': {'lr': 0.007530305285915081}, 'lr': 0.007530305285915081, 'accuracy': 0.8659217877094972}}, {'study_folder': 'optuna_emb_v1', 'saved_trials': 82, 'best_recorded_val_acc': 0.8491620111731844, 'input_dims': [7, 8, 9, 10, 11], 'best_trial': 'checkpoints/optuna_emb_v1/trial_33.pt', 'best_metadata': {'model_params': {'inp': 11, 'out': 1, 'hidden_sizes': [500, 100, 50], 'dropout': 0.011510672019888854, 'sex_emb_dim': 3, 'embarked_emb_dim': 3}, 'lr': 0.0027597401997224743, 'accuracy': 0.8491620111731844}}]
    pl.DataFrame([{k: v for k, v in r.items() if k not in ("best_metadata", "input_dims")} for r in historical_dnn])
    return


@app.cell
def _(Path, base_path, torch):
    path = Path("checkpoints/optuna_default_optim_v1/trial_55.pt")

    model_best = torch.load(base_path / path, weights_only=False)

    model_best
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    В старых папках сохранены 71/80/88/82 завершённых trial-файла;
    это не обязательно полные studies. Лучший записанный holdout ordinary
    `default_optim_v1` — 0.9050, embedding `emb_optim_v1` — 0.8659.
    Они являются максимумами выбора по эпохам и trials. В старом коде
    сохранялись веса последней эпохи, поэтому эти числа не удостоверяют
    accuracy загруженной модели. В ordinary-папке смешаны входы 10, 694,
    710; наборы признаков и preprocessing полностью не восстановлены.

    ## Выводы

    Сложная сеть, embeddings и другой optimizer требуют измеренного
    сравнения: преимущество не следует из архитектуры. Старые результаты
    подтверждают факт экспериментов и записанные оценки, но не устойчивое
    превосходство DNN над бустингами или embeddings над one-hot.
    Для финального выбора нужен единый внешний протокол оценки; текущие
    DNN-запуски используют выбранный режим holdout/CV; оценки поиска Optuna
    участвуют в подборе параметров и остаются исследовательскими.
    """)
    return


if __name__ == "__main__":
    app.run()
