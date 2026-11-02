"""Train models, select DNN weights on val, evaluate on test and predict."""

import copy
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import optuna
import torch
from omegaconf import OmegaConf
from optuna_integration.sklearn import OptunaSearchCV
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from torch.utils.data import DataLoader, TensorDataset

import utils
from data import TitanicData
from features import make_features
from models import build_dnn, restore_dnn


def tune_classical(estimator, distributions, X, y, cv, *, seed, n_trials):
    """Tune a classical pipeline by mean CV accuracy on the supplied rows.

    Args:
        estimator: Unfitted pipeline, optionally containing a voting/stacking model.
        distributions: Optuna distributions keyed by sklearn parameter paths.
        X: Features used for parameter selection.
        y: Target labels aligned with X.
        cv: Fit/validation index pairs relative to X and y.
        seed: Seed for the Optuna TPE sampler.
        n_trials: Positive number of parameter configurations to evaluate.

    Returns:
        A search result exposing best_estimator_, best_params_, best_score_
        and study_. The best estimator is refitted on all supplied rows.

    Every CV fit clones preprocessing. Ensemble members share sampled settings
    while retaining their assigned seeds. best_score_ is a selection score.
    """
    if n_trials < 1:
        raise ValueError("tuning.n_trials must be positive")

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.NopPruner(),
    )

    model = estimator.named_steps["model"]

    if isinstance(model, (VotingClassifier, StackingClassifier)):
        # Sample one shared setting, then apply it to every member of the ensemble.
        first_name = model.estimators[0][0]

        def shared_params(params):
            """Expand first-member parameter paths to every ensemble member."""
            return {key.replace(f"model__{first_name}__", f"model__{name}__", 1): value
                    for key, value in params.items() for name, _ in model.estimators}

        def objective(trial):
            """Sample shared member settings and return their mean CV accuracy."""
            params = {}
            for name, distribution in distributions.items():
                if isinstance(distribution, optuna.distributions.CategoricalDistribution):
                    params[name] = trial.suggest_categorical(name, distribution.choices)
                elif isinstance(distribution, optuna.distributions.IntDistribution):
                    params[name] = trial.suggest_int(name, distribution.low, distribution.high,
                                                     step=distribution.step, log=distribution.log)
                else:
                    params[name] = trial.suggest_float(name, distribution.low, distribution.high,
                                                       step=distribution.step, log=distribution.log)
            candidate = clone(estimator).set_params(**shared_params(params))
            scores = cross_val_score(candidate, X, y, cv=cv, scoring="accuracy", n_jobs=1, error_score="raise")
            trial.set_user_attr("inner_fold_scores", scores.tolist())
            return float(np.mean(scores))

        study.optimize(objective, n_trials=n_trials)
        best_params = shared_params(study.best_params)
        fitted = clone(estimator).set_params(**best_params).fit(X, y)
        return SimpleNamespace(best_estimator_=fitted, best_params_=best_params,
                               best_score_=study.best_value, study_=study)

    search = OptunaSearchCV(
        clone(estimator), distributions, cv=cv, scoring="accuracy", n_trials=n_trials,
        n_jobs=1, refit=True, study=study, error_score="raise",
    )
    search.fit(X, y)
    return search


def fit_classical(data: TitanicData, estimator, config, run_dir, distributions=None, *, refit_all=True):
    """Select parameters on train, score test and optionally refit all rows.

    Args:
        data: Shared TitanicData instance with the same split used by CV.
        estimator: Unfitted classical pipeline or ensemble.
        config: Feature, validation and tuning settings.
        run_dir: Existing directory for the model and training reports.
        distributions: Optuna parameter distributions when tuning is enabled.
        refit_all: Fit and save the final full-data model after test evaluation.

    Returns:
        Path to model.joblib, containing the fitted pipeline, feature columns
        and training configuration, or None when refit_all is False.
        Test scores and predictions are saved before the optional final refit.

    With tuning disabled, use the configured parameters directly. The final
    refit includes the test rows after selection and test evaluation.
    """
    run_dir = Path(run_dir)
    train_dataset = data.train_dataset
    X, y = data.X, data.y
    X_train, y_train = data.X_train, data.y_train
    X_test, y_test = data.X_test, data.y_test
    train_i, test_i = data.train_i, data.test_i

    best_params = {}
    selection = {"score_role": "fixed_parameters", "accuracy": None}

    if config.tuning.enabled:
        inner = list(StratifiedKFold(
            n_splits=config.validation.inner_splits, shuffle=True, random_state=config.base.seed,
        ).split(X_train, y_train))

        search = tune_classical(
            estimator, distributions, X_train, y_train, inner,
            seed=config.base.seed, n_trials=config.tuning.n_trials,
        )

        best_params = search.best_params_
        fitted = search.best_estimator_

        utils.save_trials(search.study_, run_dir / "final_trials.csv")

        selection = {
            "score_role": "train_selection_cv", "accuracy": float(search.best_score_),
            "best_params": best_params, "train_indices": train_i.tolist(),
            "excluded_test_indices": test_i.tolist(),
            "splits": [{"train_source_indices": train_i[a].tolist(),
                        "val_source_indices": train_i[b].tolist()} for a, b in inner],
        }

    else:
        fitted = clone(estimator).fit(X_train, y_train)

    predictions, probabilities = utils.predict_classifier(fitted, X_test)
    test_accuracy = float(accuracy_score(y_test, predictions))

    utils.save_test_results(
        run_dir, config, train_dataset, train_i, test_i,
        predictions, probabilities, test_accuracy, selection,
    )

    # return none if refit on full train_dataset is not needed
    if not refit_all:
        return None

    # Do not reuse a fitted outer model or select parameters by outer accuracy.
    fitted = clone(estimator).set_params(**best_params)
    fitted.fit(X, y)

    return utils.save_classical_model(fitted, X, train_dataset, config, run_dir, selection)

def predict_classical(model_path, kaggle_test, run_dir):
    """Load a classical pipeline and predict in Kaggle test row order.

    Args:
        model_path: Path to a model.joblib artifact from fit_classical.
        kaggle_test: Raw passenger DataFrame for inference.
        run_dir: Existing directory for submission, predictions and provenance.

    Returns:
        A Polars DataFrame with PassengerId and predicted Survived labels.
        Feature extraction follows the artifact; fitted preprocessing is reused.
    """
    artifact = joblib.load(model_path)
    X, _, _ = make_features(kaggle_test, artifact["feature_set"])

    if X.columns != artifact["feature_columns"]:
        raise ValueError("Inference features differ from the saved training columns")

    model = artifact["pipeline"]
    predictions, probabilities = utils.predict_classifier(model, X)

    return utils.save_predictions(artifact, predictions, probabilities, kaggle_test, run_dir, model_path,
                            utils.get_model_report(model.named_steps["model"])["ensemble"])


def tune_dnn(spec, X, y, cv, *, seed, n_trials, epochs, device="cpu", batch_size=64, val_size=0.2):
    """Tune a DNN using CV accuracy with separate internal validation subsets.

    Args:
        spec: Model class/name, preprocessing and suggest_params trial sampler.
        X: Features available to this parameter search.
        y: Target labels aligned with X.
        cv: Inner fit/validation index pairs relative to X and y.
        seed: Seed for the sampler and internal validation splits.
        n_trials: Number of configurations to evaluate.
        epochs: Training epoch budget for each fold.
        device: PyTorch execution device.
        batch_size: Requested training batch size.
        val_size: Fraction of each inner training fold reserved for validation.

    Returns:
        An Optuna study whose trials record configuration, inner fold scores
        and selected epochs. Inner validation labels only score predictions;
        preprocessing and epoch selection use each inner fold's train/val rows.
    """
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.NopPruner())

    def objective(trial):
        """Return mean inner-CV accuracy for one sampled DNN configuration."""
        configuration = spec['suggest_params'](trial)
        trial.set_user_attr('configuration', configuration)
        scores, best_epochs = [], []
        for fold, (train, val) in enumerate(cv, 1):
            X_train, X_val, y_train, y_val = train_test_split(
                X[train], y[train], test_size=val_size,
                stratify=y[train], random_state=seed + fold,
            )
            fitted = fit_dnn(spec['model_cls'], configuration, spec['preprocess'],
                             X_train, y_train, X_val, y_val, seed=seed + fold, epochs=epochs,
                             embedding_columns=spec.get('embedding_columns'), device=device, batch_size=batch_size)
            predictions, _ = predict_dnn(fitted['model'], fitted['preprocess'], X[val], device)
            scores.append(float(accuracy_score(y[val], predictions)))
            best_epochs.append(fitted['best_epoch'])
        trial.set_user_attr('inner_fold_scores', scores)
        trial.set_user_attr('inner_best_epochs', best_epochs)

        return float(np.mean(scores))

    study.optimize(objective, n_trials=n_trials)
    return study


def fit_custom_dnn(data: TitanicData, spec, config, run_dir, *, refit_all=True):
    """Select DNN settings/epochs, evaluate test and optionally refit all rows.

    Args:
        data: Shared TitanicData instance with the same split used by CV.
        spec: Custom-DNN specification from utils.get_dnn_spec.
        config: Feature, validation, tuning and training settings.
        run_dir: Existing directory for model.pt, preprocessing and training reports.
        refit_all: Fit and save the final full-data model after test evaluation.

    Returns:
        Path to model.pt. The companion preprocess.joblib stores the fitted
        transformer required for inference. Return None when refit_all is False.

    An internal train split selects the epoch count and learning-rate
    schedule. Fresh train and full-data fits replay both without validation;
    the test is scored by the train-only model.
    """
    run_dir = Path(run_dir)
    train_dataset = data.train_dataset
    X, y = data.X, data.y
    X_test, y_test = data.X_test, data.y_test
    train_i, test_i = data.train_i, data.test_i

    device, epochs, batch_size = config.training.device, config.training.epochs, config.training.batch_size
    val_size = OmegaConf.select(config, 'validation.val_size', default=0.2)
    configuration = spec['configuration']

    selection = {'score_role': 'fixed_parameters', 'accuracy': None}
    splits = []

    if config.tuning.enabled:
        cv = list(StratifiedKFold(n_splits=config.validation.inner_splits, shuffle=True,
                                  random_state=config.base.seed).split(data.X_train, data.y_train))
        study = tune_dnn(spec, data.X_train, data.y_train, cv, seed=config.base.seed,
                         n_trials=config.tuning.n_trials, epochs=epochs, device=device,
                         batch_size=batch_size, val_size=val_size)

        configuration = study.best_trial.user_attrs['configuration']

        utils.save_trials(study, run_dir/'final_trials.csv')

        selection = {'score_role': 'train_selection_cv', 'accuracy': float(study.best_value),
                     'best_params': study.best_params}
        splits = [{'train_source_indices': train_i[a].tolist(),
                   'val_source_indices': train_i[b].tolist()} for a, b in cv]

    X_train, X_val, y_train, y_val, train_indices, val_indices = train_test_split(
        data.X_train, data.y_train, train_i, test_size=val_size,
        stratify=data.y_train, random_state=config.base.seed,
    )

    selected = fit_dnn(spec['model_cls'], configuration, spec['preprocess'], X_train, y_train, X_val, y_val,
                       seed=config.base.seed, epochs=epochs, embedding_columns=spec.get('embedding_columns'),
                       device=device, batch_size=batch_size)

    fitted = fit_dnn(
        spec['model_cls'], configuration, spec['preprocess'], data.X_train, data.y_train, None, None,
        seed=config.base.seed, epochs=selected['best_epoch'],
        embedding_columns=spec.get('embedding_columns'), device=device, batch_size=batch_size,
        learning_rates=selected['learning_rates'][:selected['best_epoch']],
    )

    predictions, probabilities = predict_dnn(fitted['model'], fitted['preprocess'], X_test, device)
    test_accuracy = float(accuracy_score(y_test, predictions))

    utils.save_test_results(
        run_dir, config, train_dataset, train_i, test_i,
        predictions, probabilities, test_accuracy, selection, epochs=selected['best_epoch'],
    )

    if not refit_all:
        return None

    # Freeze both epoch count and scheduler decisions before adding any other labels.
    fitted = fit_dnn(spec['model_cls'], configuration, spec['preprocess'], X, y, None, None,
                     seed=config.base.seed, epochs=selected['best_epoch'],
                     embedding_columns=spec.get('embedding_columns'), device=device, batch_size=batch_size,
                     learning_rates=selected['learning_rates'][:selected['best_epoch']])

    return utils.save_custom_dnn(
        fitted, selected, X, train_dataset, config, run_dir, selection,
        train_i, test_i, splits, train_indices, val_indices,
    )

def predict_custom_dnn(model_path, kaggle_test, run_dir, device='cpu'):
    """Restore a custom DNN and its preprocessing and write Kaggle predictions.

    Args:
        model_path: Path to model.pt with its companion preprocessing artifact.
        kaggle_test: Raw passenger DataFrame whose feature columns must match training.
        run_dir: Existing directory for submission, probabilities and provenance.
        device: PyTorch device used for inference.

    Returns:
        A Polars DataFrame with PassengerId and predicted Survived labels.
    """
    artifact = torch.load(model_path, map_location='cpu', weights_only=True)
    X, _, _ = make_features(kaggle_test, artifact['feature_set'])

    if X.columns != artifact['feature_columns']:
        raise ValueError('Inference features differ from the saved training columns')

    model = restore_dnn(artifact['model_name'], artifact['model_params'], artifact['embedding_metadata'])
    model.load_state_dict(artifact['model_state_dict'])

    preprocessing = joblib.load(Path(model_path).parent/artifact['preprocess_path'])
    predictions, probabilities = predict_dnn(model, preprocessing, X, device)

    return utils.save_predictions(artifact, predictions, probabilities, kaggle_test, run_dir, model_path, {'n_models': 1})


def predict_loader(model, loss_model, loader, device="cpu"):
    """Evaluate a labeled binary-classification loader without gradients.

    Args:
        model: DNN returning one raw logit per row.
        loss_model: Scalar loss applied to logits and binary targets.
        loader: Batches of feature tensors and column-shaped target tensors.
        device: Device already holding the model.

    Returns:
        (loss, accuracy): Sample-weighted batch loss and classification accuracy,
        using sigmoid probabilities thresholded at 0.5.
    """
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
                 epoches=15, trial=None, device="cpu", learning_rates=None):
    """Train a binary DNN and optionally restore its best internal-validation weights.

    Args:
        model: PyTorch module returning one logit per row.
        loss_model: Loss function accepting logits and binary targets.
        opt: Optimizer bound to the model's parameters.
        train_loader: Batches used for gradient updates.
        val_loader: Internal validation loader, or None for a final fixed-epoch refit.
        sched: Optional scheduler stepped after internal validation.
        epoches: Number of training epochs, preserving the existing argument name.
        trial: Optional Optuna trial for epoch reporting and pruning.
        device: PyTorch execution device.
        learning_rates: Optional per-epoch, per-optimizer-group rates to replay.

    Returns:
        A dictionary with the trained model, optimizer, loss, metric histories,
        selected/last epoch numbers and used learning rates. With validation,
        restore the earliest weights achieving the highest validation accuracy.
        Without validation, retain the final weights and leave accuracy unset.

    val_loader must contain internal validation rows, not an outer evaluation fold.
    """
    model.to(device)

    history = {name: [] for name in ["train_loss", "train_acc", "val_loss", "val_acc"]}
    best_accuracy, best_state, best_epoch = -1.0, None, None

    used_learning_rates = []
    for epoch in range(1, epoches + 1):
        if learning_rates is not None:
            for group, rate in zip(opt.param_groups, learning_rates[epoch - 1]):
                group['lr'] = rate
        used_learning_rates.append([group['lr'] for group in opt.param_groups])
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
                    """Recompute the current batch loss and gradients for LBFGS."""
                    opt.zero_grad()
                    closure_loss = loss_model(model(x), target)
                    closure_loss.backward()
                    return closure_loss
                opt.step(closure)
            else:
                opt.step()
            loss_sum += loss.item() * len(x)
            correct += ((logits.detach() >= 0) == target).sum().item()
        history["train_loss"].append(loss_sum / len(train_loader.dataset))
        history["train_acc"].append(correct / len(train_loader.dataset))
        if val_loader is None:
            continue  # Final refit uses a frozen epoch count and learning-rate schedule.
        val_loss, val_acc = predict_loader(model, loss_model, val_loader, device)
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
    if val_loader is None:
        model.eval()
        return {
            'model': model, 'loss': loss_model, 'opt': opt, 'accuracy': None,
            'best_epoch': epoches, 'weights_epoch': epoches, 'last_epoch': epoches,
            'learning_rates': used_learning_rates, **history,
        }
    model.load_state_dict(best_state)
    model.eval()
    return {
        "model": model, "loss": loss_model, "opt": opt,
        "accuracy": best_accuracy, "best_val_acc": best_accuracy,
        "best_epoch": best_epoch, "weights_epoch": best_epoch,
        "last_epoch": epoches, "last_val_acc": history["val_acc"][-1],
        "summary": [], "learning_rates": used_learning_rates, **history,
    }


def fit_dnn(model_cls, configuration, preprocessing, X_train, y_train, X_val, y_val,
            seed, epochs=15, embedding_columns=None, device="cpu", batch_size=64, learning_rates=None):
    """Fit preprocessing and a fresh DNN using train and optional val rows.

    Args:
        model_cls: PyTorch module class or configured model name.
        configuration: Model, loss, optimizer and scheduler settings.
        preprocessing: Unfitted transformer cloned and fitted only on X_train.
        X_train: Features used to learn preprocessing and model weights.
        y_train: Binary labels aligned with X_train.
        X_val: Internal validation features, or None for a fixed-epoch refit.
        y_val: Binary labels for X_val, or None when validation is disabled.
        seed: Seed for initialization, shuffled batches and supported operations.
        epochs: Positive training epoch budget.
        embedding_columns: Optional ordered names of embedded categorical features.
        device: PyTorch execution device.
        batch_size: Requested batch size, adjusted to avoid a singleton tail.
        learning_rates: Optional per-epoch learning-rate schedule for final refitting.

    Returns:
        The train_epochs result extended with fitted preprocessing, resolved model
        parameters, embedding metadata and configuration. Outer validation stays
        with the caller and is never passed to this training function.
    """
    utils.set_seed(seed)

    fitted_preprocess = clone(preprocessing)

    x_train = torch.as_tensor(fitted_preprocess.fit_transform(X_train), dtype=torch.float32)

    targets_train = torch.as_tensor(np.array(y_train, copy=True).reshape(-1, 1), dtype=torch.float32)

    if epochs < 1 or batch_size < 1:
        raise ValueError('training.epochs and training.batch_size must be positive')

    # Merge a singleton tail into a larger batch for BatchNorm-based architectures.
    while batch_size > 1 and len(x_train) % batch_size == 1:
        batch_size += 1

    train_loader = DataLoader(TensorDataset(x_train, targets_train), batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = None
    if X_val is not None:
        x_val = torch.as_tensor(fitted_preprocess.transform(X_val), dtype=torch.float32)
        targets_val = torch.as_tensor(np.array(y_val, copy=True).reshape(-1, 1), dtype=torch.float32)
        val_loader = DataLoader(TensorDataset(x_val, targets_val), batch_size=batch_size)

    model, params, embedding_metadata = build_dnn(
        model_cls, configuration["model_params"], fitted_preprocess, x_train.shape[1], embedding_columns,
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
    scheduler = utils.get_scheduler(config, optimizer) if val_loader is not None else None
    loss_model = utils.get_loss(config)

    result = train_epochs(model, loss_model, optimizer, train_loader, val_loader,
                          scheduler, epoches=epochs, device=device, learning_rates=learning_rates)
    result.update(preprocess=fitted_preprocess, model_params=params,
                  embedding_metadata=embedding_metadata, configuration=configuration)

    return result


def predict_dnn(model, preprocessing, X, device="cpu"):
    """Predict binary labels and survival probabilities with fitted DNN artifacts.

    Args:
        model: Trained PyTorch module producing one logit per row.
        preprocessing: Fitted transformer applied to X without refitting.
        X: Feature DataFrame in the model's expected column layout.
        device: PyTorch device used for batches of up to 64 rows.

    Returns:
        (predictions, probabilities): Arrays of shape (len(X),) in input row order.
        Probabilities are sigmoid outputs; values at least 0.5 receive label 1.
        Inference runs in evaluation mode with gradients disabled.
    """
    model.to(device)
    model.eval()
    values = torch.as_tensor(preprocessing.transform(X), dtype=torch.float32)
    with torch.no_grad():
        probabilities = np.concatenate([
            torch.sigmoid(model(batch.to(device))).cpu().numpy().ravel()
            for batch in values.split(64)
        ])
    return (probabilities >= 0.5).astype(int), probabilities
