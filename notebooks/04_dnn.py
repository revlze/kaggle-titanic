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
    выключены. `USE_CV=False` использует внутреннее разбиение тренировочных
    80%, `USE_CV=True` — 5 внешних фолдов с отдельным stopping split внутри
    каждого outer train. Исходный 20% holdout не управляет обучением.
    Optuna всегда использует nested CV 5×3. Существующие большие MLP и
    embeddings здесь исследуются на `all`; единое сравнение малой MLP и
    классики на `base_title_family` находится в 05. Метрика — accuracy.
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
    OPTUNA_SEARCH = False
    N_TRIALS = 20
    SAVE_RESULTS = True
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
    results_path = base_path / 'notebooks/results/dnn'
    return base_path, checkpoint_dir, data_path, results_path


@app.cell
def _(data_path, pl):
    train_dataset = pl.read_csv(data_path / 'train.csv')
    test_dataset = pl.read_csv(data_path / 'test.csv')
    return test_dataset, train_dataset


@app.cell
def _():
    from features import make_features as _shared_features, make_preprocess

    def make_features(df):
        return _shared_features(df, 'all')

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
    fit_i, stop_i = train_test_split(
        np.arange(len(X_train)), test_size=0.2, stratify=y_train, random_state=SEED,
    )
    X_fit, X_stop = X_train[fit_i], X_train[stop_i]
    y_fit, y_stop = y_train[fit_i], y_train[stop_i]
    preprocess = make_preprocess(numeric_columns, categorical_columns)
    return (
        X_train,
        X_val,
        X_fit,
        X_stop,
        y_fit,
        y_stop,
        fit_i,
        stop_i,
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
            'protocol': ('stratified_80_20_outer5_internal_stop_v2' if USE_CV
                         else 'stratified_80_20_development_internal_stop_v2'),
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
def _(X_fit, X_stop, clone, preprocess, torch, y_fit, y_stop):
    nn_preprocess = clone(preprocess)
    X_train_np = nn_preprocess.fit_transform(X_fit)
    X_val_np = nn_preprocess.transform(X_stop)


    X_train_t = torch.as_tensor(X_train_np, dtype=torch.float32)
    y_train_t = torch.tensor(y_fit.to_numpy(),
    dtype=torch.float32).unsqueeze(1)

    X_val_t = torch.as_tensor(X_val_np, dtype=torch.float32)
    y_val_t = torch.tensor(y_stop.to_numpy(),
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
def _():
    from models import MyModel
    return (MyModel,)


@app.cell
def _(categorical_columns, make_preprocess, numeric_columns):
    preprocess_embedding = make_preprocess(numeric_columns, categorical_columns, encoding='ordinal')
    return (preprocess_embedding,)


@app.cell
def _(X_fit, X_stop, clone, preprocess_embedding, torch, y_fit, y_stop):
    nn_preprocess_embedding = clone(preprocess_embedding)
    X_train_np_emb = nn_preprocess_embedding.fit_transform(X_fit)
    X_val_np_emb = nn_preprocess_embedding.transform(X_stop)

    cat_sizes = [len(values) + 1 for values in nn_preprocess_embedding.named_transformers_["categorical"].named_steps["encoder"].categories_]

    num_slice = nn_preprocess_embedding.output_indices_["numeric"]
    cat_slice = nn_preprocess_embedding.output_indices_["categorical"]

    X_train_t_emb = torch.as_tensor(
        X_train_np_emb, dtype=torch.float32
    )
    y_train_t_emb = torch.tensor(y_fit.to_numpy(),
    dtype=torch.float32).unsqueeze(1)


    X_val_t_emb = torch.as_tensor(
        X_val_np_emb, dtype=torch.float32
    )
    y_val_t_emb = torch.tensor(y_stop.to_numpy(),
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
def _():
    from models import MyModelEmb
    return (MyModelEmb,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Размер входа one-hot берётся из тензора. Размеры embedding-словарей
    получены из категорий train encoder, плюс один индекс для неизвестного.
    Срезы numeric/categorical берутся из output_indices_. Обе ветки обучают
    preprocessing только на X_fit; внутренний X_stop получает transform. В режиме
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
def _(DEVICE):
    from training import train_epochs

    def epoch_train(model, loss_model, opt, train_loader, val_loader, sched=None, epoches=15, trial=None):
        # val_loader is an internal stopping split, never the 20% holdout.
        return train_epochs(model, loss_model, opt, train_loader, val_loader,
                            sched, epoches, trial, device=DEVICE)

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
def _(DEVICE, MyModelEmb, SEED, np, train_test_split):
    from training import fit_dnn, predict_dnn

    def cv_train(model_cls, model_params, loss_model, opt_cls, opt_params,
                 sched_params, cv, X, y, preprocess, sched_cls=None, epoches=15, trial=None):
        if trial is not None:
            raise ValueError('Для подбора используй run_cv_experiment; cv_train оценивает фиксированную конфигурацию.')
        results = {'folds': [], 'metrics': {}, 'summary': None}
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
            fit_idx, stop_idx = train_test_split(
                train_idx, test_size=0.2, stratify=np.asarray(y)[train_idx], random_state=SEED + fold,
            )
            configuration = {
                'model_params': model_params, 'optimizer_name': opt_cls.__name__,
                'optimizer_params': opt_params,
                'scheduler_name': None if sched_cls is None else sched_cls.__name__,
                'scheduler_params': sched_params,
            }
            fitted = fit_dnn(
                model_cls, configuration, preprocess, X[fit_idx], y[fit_idx], X[stop_idx], y[stop_idx],
                seed=SEED + fold, epochs=epoches, device=DEVICE,
                embedding_columns=['sex', 'embarked', 'cabin', 'ticket', 'title'] if model_cls is MyModelEmb else None,
            )
            predictions, _ = predict_dnn(fitted['model'], fitted['preprocess'], X[val_idx], DEVICE)
            outer_acc = float(np.mean(predictions == np.asarray(y)[val_idx]))
            results['folds'].append({
                **fitted, 'fold': fold, 'outer_val_acc': outer_acc,
                'fit_indices': fit_idx.tolist(), 'stop_indices': stop_idx.tolist(),
                'outer_val_indices': val_idx.tolist(),
            })
        for metric in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
            values = np.asarray([fold[metric] for fold in results['folds']])
            # Curves refer to INTERNAL stopping validation, not outer predictions.
            results[metric] = {'mean': values.mean(axis=0).tolist(), 'std': values.std(axis=0).tolist()}
            results['metrics'][metric] = {'mean': float(values[:, -1].mean()), 'std': float(values[:, -1].std())}
        outer_scores = [fold['outer_val_acc'] for fold in results['folds']]
        results['metrics']['outer_val_acc'] = {'mean': float(np.mean(outer_scores)), 'std': float(np.std(outer_scores))}
        results['summary'] = f"Outer CV accuracy: {np.mean(outer_scores):.4f} ± {np.std(outer_scores):.4f}"
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
                               color=line.get_color(), label=f'{name}: internal validation')
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
                               color=line.get_color(), label=f'{name}: internal validation')

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
                         'cv_acc': _res['metrics']['outer_val_acc']['mean'],
                         'cv_std': _res['metrics']['outer_val_acc']['std'],
                         'cv_fold_scores': [_fold['outer_val_acc'] for _fold in _res['folds']],
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
                         score_role='outer_cv_fixed_configuration' if USE_CV else 'internal_epoch_selection',
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
                              'cv_acc': _res['metrics']['outer_val_acc']['mean'],
                              'cv_std': _res['metrics']['outer_val_acc']['std'],
                              'cv_fold_scores': [_fold['outer_val_acc'] for _fold in _res['folds']],
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
                         score_role='outer_cv_fixed_configuration' if USE_CV else 'internal_epoch_selection',
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
def _():
    from evaluation import run_cv_experiment
    from models import suggest_notebook_dnn
    return run_cv_experiment, suggest_notebook_dnn


@app.cell
def _(
    DEVICE, MyModel, MyModelEmb, N_TRIALS, OPTUNA_SEARCH, SEED,
    pl, preprocess, preprocess_embedding, results_path,
    run_cv_experiment, suggest_notebook_dnn, train_dataset,
):
    frozen_comparison = None
    if OPTUNA_SEARCH:
        _specs = {
            'MyModel': {'kind': 'dnn', 'model_cls': MyModel, 'preprocess': preprocess,
                        'suggest_params': lambda trial: suggest_notebook_dnn(trial)},
            'MyModelEmb': {'kind': 'dnn', 'model_cls': MyModelEmb, 'preprocess': preprocess_embedding,
                           'embedding_columns': ['sex', 'embarked', 'cabin', 'ticket', 'title'],
                           'suggest_params': lambda trial: suggest_notebook_dnn(trial, embedding=True)},
        }
        _run = run_cv_experiment(
            train_dataset, _specs, results_path / 'nested_all', feature_set='all',
            seed=SEED, n_trials=N_TRIALS, epochs=15, device=DEVICE,
        )
        frozen_comparison = pl.DataFrame(_run['results'])
    frozen_comparison
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Внешние предсказания получены после загрузки сохранённых весов и
    preprocessing. `best_epoch` и `best_val_acc` относятся к внутреннему
    stopping split; `last_epoch` — число выполненных эпох, `outer_val_acc`
    — accuracy сохранённой модели на внешнем фолде. Подбор, scheduler и
    выбор весов не видят outer validation. Старые файлы не перезаписываются.
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

    model_best = torch.load(base_path / path, weights_only=False) if (base_path / path).is_file() else None

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
    Для контролируемого сравнения используй 05. Текущие Optuna-запуски
    используют nested CV 5×3; inner selection scores и outer accuracy
    сохраняются отдельно. Старые holdout/CV-оценки с ними напрямую не сравниваются.
    """)
    return


if __name__ == "__main__":
    app.run()
