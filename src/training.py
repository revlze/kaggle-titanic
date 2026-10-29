"""Train on fit rows, choose weights on internal validation, predict separately."""

import copy
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import optuna
import polars as pl
import torch
from omegaconf import OmegaConf
from optuna_integration.sklearn import OptunaSearchCV
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

import utils
from data import split_development
from features import make_features


def tune_classical(estimator, distributions, X, y, cv, *, seed, n_trials):
    """Search only the rows supplied by the caller; clone preprocessing for every fit."""
    if n_trials < 1:
        raise ValueError("tuning.n_trials must be positive")
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.NopPruner(),
    )
    search = OptunaSearchCV(
        clone(estimator), distributions, cv=cv, scoring="accuracy", n_trials=n_trials,
        n_jobs=1, refit=True, study=study, error_score="raise",
    )
    search.fit(X, y)
    return search


def fit_classical(train_dataset, estimator, config, run_dir, distributions=None):
    """Select on development if requested, then refit a fresh pipeline on ALL labels."""
    run_dir = Path(run_dir)
    X, _, _ = make_features(train_dataset, config.features.feature_set)
    y = train_dataset["Survived"].to_numpy()
    best_params = {}
    selection = {"score_role": "fixed_parameters", "accuracy": None}
    if config.tuning.enabled:
        train_i, holdout_i = split_development(y, config.base.seed, config.validation.development_size)
        inner = list(StratifiedKFold(
            n_splits=config.validation.inner_splits, shuffle=True, random_state=config.base.seed,
        ).split(X[train_i], y[train_i]))
        search = tune_classical(
            estimator, distributions, X[train_i], y[train_i], inner,
            seed=config.base.seed, n_trials=config.tuning.n_trials,
        )
        best_params = search.best_params_
        search.study_.trials_dataframe().to_csv(run_dir / "final_trials.csv", index=False)
        selection = {
            "score_role": "development_selection_cv", "accuracy": float(search.best_score_),
            "best_params": best_params, "development_indices": train_i.tolist(),
            "excluded_holdout_indices": holdout_i.tolist(),
            "splits": [{"fit_source_indices": train_i[a].tolist(),
                        "validation_source_indices": train_i[b].tolist()} for a, b in inner],
        }
    # Do not reuse a fitted outer model or select parameters by outer accuracy.
    fitted = clone(estimator).set_params(**best_params)
    fitted.fit(X, y)
    artifact = {
        "pipeline": fitted, "feature_set": config.features.feature_set,
        "feature_columns": X.columns, "model_name": config.model.name,
        "configuration": OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
    }
    model_path = run_dir / "model.joblib"
    joblib.dump(artifact, model_path)
    restored = joblib.load(model_path)["pipeline"]
    assert np.array_equal(fitted.predict(X), restored.predict(X))
    assert np.array_equal(fitted.predict_proba(X), restored.predict_proba(X))
    # Keep row identities and split indices out of the readable training report.
    selection_splits = {key: selection.pop(key) for key in [
        "development_indices", "excluded_holdout_indices", "splits",
    ] if key in selection}
    (run_dir / "training_splits.json").write_text(json.dumps({
        "fit_passenger_ids": train_dataset["PassengerId"].to_list(),
        "selection": selection_splits,
    }, indent=2))
    report = {
        "model": config.model.name, "fit_role": "final_refit_all_labeled_rows",
        "fitted_rows": len(y), "splits_path": "training_splits.json",
        "selection": selection, "parameters": fitted.named_steps["model"].get_params(),
        "checkpoint_predictions_verified": True,
    }
    (run_dir / "training.json").write_text(json.dumps(report, indent=2))
    return model_path


def predict_classical(model_path, test_dataset, run_dir):
    """Use the saved feature specification and fitted transforms, without fitting."""
    artifact = joblib.load(model_path)
    X, _, _ = make_features(test_dataset, artifact["feature_set"])
    if X.columns != artifact["feature_columns"]:
        raise ValueError("Inference features differ from the saved training columns")
    model = artifact["pipeline"]
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, list(model.classes_).index(1)]
    if (predictions.shape != (len(test_dataset),) or not np.isin(predictions, [0, 1]).all()
            or not np.isfinite(probabilities).all() or not ((0 <= probabilities) & (probabilities <= 1)).all()):
        raise ValueError("Invalid prediction shape, labels or probabilities")
    submission = test_dataset.select("PassengerId").with_columns(
        pl.Series("Survived", predictions, dtype=pl.Int64),
    )
    run_dir = Path(run_dir)
    submission.write_csv(run_dir / "submission.csv")
    submission.with_columns(pl.Series("p_survived", probabilities)).write_csv(run_dir / "predictions.csv")
    (run_dir / "inference.json").write_text(json.dumps({
        "model_path": str(Path(model_path).resolve()), "model": artifact["model_name"],
        "feature_set": artifact["feature_set"], "feature_columns": X.columns,
        "rows": len(submission), "training_configuration": artifact["configuration"],
        "test_dataframe_sha256": hashlib.sha256(test_dataset.write_csv().encode()).hexdigest(),
    }, indent=2))
    return submission


def predict_loader(model, loss_model, loader, device="cpu"):
    model.eval()
    loss_sum, probabilities, targets = 0.0, [], []
    with torch.no_grad():
        for x, target in loader:
            logits = model(x.to(device))
            loss_sum += loss_model(logits, target.to(device)).item() * len(x)
            probabilities.extend(torch.sigmoid(logits).cpu().numpy().ravel())
            targets.extend(target.numpy().ravel())
    probabilities = np.asarray(probabilities)
    predictions = (probabilities >= 0.5).astype(int)
    return loss_sum / len(targets), accuracy_score(targets, predictions)


def train_epochs(model, loss_model, opt, train_loader, val_loader, sched=None,
                 epoches=15, trial=None, device="cpu"):
    """val_loader is INTERNAL validation, never an outer evaluation fold."""
    model.to(device)
    history = {name: [] for name in ["train_loss", "train_acc", "val_loss", "val_acc"]}
    best_accuracy, best_state, best_epoch = -1.0, None, None
    for epoch in range(1, epoches + 1):
        model.train()
        loss_sum, correct = 0.0, 0
        for x, target in train_loader:
            x, target = x.to(device), target.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_model(logits, target)
            loss.backward()
            if isinstance(opt, torch.optim.LBFGS):
                def closure():
                    opt.zero_grad()
                    closure_loss = loss_model(model(x), target)
                    closure_loss.backward()
                    return closure_loss
                opt.step(closure)
            else:
                opt.step()
            loss_sum += loss.item() * len(x)
            correct += ((logits.detach() >= 0) == target).sum().item()
        val_loss, val_acc = predict_loader(model, loss_model, val_loader, device)
        history["train_loss"].append(loss_sum / len(train_loader.dataset))
        history["train_acc"].append(correct / len(train_loader.dataset))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if val_acc > best_accuracy:
            best_accuracy, best_epoch = val_acc, epoch
            best_state = copy.deepcopy(model.state_dict())
        if sched is not None:
            if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sched.step(val_loss)
            else:
                sched.step()
        if trial is not None:
            import optuna
            trial.report(val_acc, epoch - 1)
            if trial.should_prune():
                raise optuna.TrialPruned()
    model.load_state_dict(best_state)
    model.eval()
    # Check the actual restored weights, rather than trusting a recorded maximum.
    restored_accuracy = predict_loader(model, loss_model, val_loader, device)[1]
    assert restored_accuracy == best_accuracy
    return {
        "model": model, "loss": loss_model, "opt": opt,
        "accuracy": restored_accuracy, "best_val_acc": restored_accuracy,
        "best_epoch": best_epoch, "weights_epoch": best_epoch,
        "last_epoch": epoches, "last_val_acc": history["val_acc"][-1],
        "summary": [], **history,
    }


def build_dnn(model_cls, model_params, preprocessing, input_dim, embedding_columns=None):
    params = model_params.copy()
    params["inp"] = input_dim
    embedding_metadata = None
    if embedding_columns:
        numeric = preprocessing.output_indices_["numeric"]
        categorical = preprocessing.output_indices_["categorical"]
        sizes = [len(c) + 1 for c in preprocessing.named_transformers_["categorical"].named_steps["encoder"].categories_]
        params["inp"] = numeric.stop - numeric.start + sum(
            params.get(f"{column}_emb_dim", 2) for column in embedding_columns
        )
        embedding_metadata = {
            "num_slice": [numeric.start, numeric.stop],
            "cat_slice": [categorical.start, categorical.stop],
            "cat_sizes": sizes,
        }
    return restore_dnn(model_cls, params, embedding_metadata), params, embedding_metadata


def restore_dnn(model_cls, model_params, embedding_metadata=None):
    params = model_params.copy()
    if embedding_metadata:
        params.update({
            "num_slice": slice(*embedding_metadata["num_slice"]),
            "cat_slice": slice(*embedding_metadata["cat_slice"]),
            "cat_sizes": embedding_metadata["cat_sizes"],
        })
    return model_cls(**params)


def fit_dnn(model_cls, configuration, preprocessing, X_fit, y_fit, X_stop, y_stop,
            seed, epochs=15, embedding_columns=None, device="cpu"):
    """Only fit/stop data enter training; the caller keeps outer validation aside."""
    utils.set_seed(seed)
    fitted_preprocess = clone(preprocessing)
    x_fit = torch.as_tensor(fitted_preprocess.fit_transform(X_fit), dtype=torch.float32)
    x_stop = torch.as_tensor(fitted_preprocess.transform(X_stop), dtype=torch.float32)
    targets_fit = torch.as_tensor(np.array(y_fit, copy=True).reshape(-1, 1), dtype=torch.float32)
    targets_stop = torch.as_tensor(np.array(y_stop, copy=True).reshape(-1, 1), dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(x_fit, targets_fit), batch_size=64, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    stop_loader = DataLoader(TensorDataset(x_stop, targets_stop), batch_size=64)
    model, params, embedding_metadata = build_dnn(
        model_cls, configuration["model_params"], fitted_preprocess, x_fit.shape[1], embedding_columns,
    )
    # Preserve the notebook trial format, using the shared YAML-style getters.
    config = OmegaConf.create({
        "loss": {"name": configuration.get("loss_name", "BCEWithLogitsLoss"),
                 "params": configuration.get("loss_params", {})},
        "optimizer": {"name": configuration["optimizer_name"],
                      "params": configuration["optimizer_params"]},
        "scheduler": {"name": configuration.get("scheduler_name"),
                      "params": configuration.get("scheduler_params", {})},
    })
    optimizer = utils.get_optimizer(config, model)
    scheduler = utils.get_scheduler(config, optimizer)
    loss_model = utils.get_loss(config)
    result = train_epochs(model, loss_model, optimizer, train_loader, stop_loader,
                          scheduler, epoches=epochs, device=device)
    result.update(preprocess=fitted_preprocess, model_params=params,
                  embedding_metadata=embedding_metadata, configuration=configuration)
    return result


def predict_dnn(model, preprocessing, X, device="cpu"):
    """Prediction has no labels, optimizer, scheduler or checkpoint selection."""
    model.to(device)
    model.eval()
    values = torch.as_tensor(preprocessing.transform(X), dtype=torch.float32)
    with torch.no_grad():
        probabilities = np.concatenate([
            torch.sigmoid(model(batch.to(device))).cpu().numpy().ravel()
            for batch in values.split(64)
        ])
    return (probabilities >= 0.5).astype(int), probabilities
