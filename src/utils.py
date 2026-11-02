"""Model configuration, artifact persistence and experiment reporting utilities."""

import os
import random
import copy
import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import polars as pl

import torch
import torch.nn as nn
import numpy as np
import torchvision
import optuna
from omegaconf import OmegaConf
from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from features import make_features, make_preprocess
from data import TitanicData, resolve_path

def set_seed(seed=42):
    """Seed Python, NumPy and PyTorch and request deterministic torch operations.

    Also configure available CUDA backends, set PYTHONHASHSEED for subsequent
    Python processes and print the selected seed. Unsupported deterministic
    operations produce warnings rather than errors.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f'Setted seed to {seed}')

def get_optimizer(config, model):
    """Construct the configured optimizer over trainable model parameters.

    Args:
        config: Configuration with optimizer.name and optimizer.params.
        model: PyTorch module whose requires_grad parameters are optimized.

    Returns:
        A new torch.optim optimizer bound to the model's trainable parameters.
    """
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = getattr(torch.optim, config.optimizer.name)(parameters, **config.optimizer.params)
    return optimizer

def get_loss(config):
    """Instantiate a torch.nn loss using config.loss.name and loss.params."""
    return getattr(nn, config.loss.name)(**config.loss.params)

def get_scheduler(config, optimizer):
    """Construct a learning-rate scheduler for the supplied optimizer.

    Read its name and constructor parameters from config.scheduler.
    Return None when the scheduler section or its name is null.
    """
    if config.scheduler is None or config.scheduler.name is None:
        return None
    return getattr(torch.optim.lr_scheduler, config.scheduler.name)(
        optimizer, **config.scheduler.params,
    )

def get_model(config, *, load_pretrained=True, **runtime_params):
    """Instantiate a classical classifier, custom DNN or torchvision model.

    Args:
        config: Configuration containing model.name, model.params and the
            base seed when constructing a classical classifier.
        load_pretrained: Allow configured torchvision weights when True;
            force weights=None when False.
        **runtime_params: Constructor parameters overriding model.params,
            including input sizes inferred by the DNN training code.

    Returns:
        A newly constructed model. Names starting with custom/ resolve to
        classes in models.py; other non-classifier names resolve in torchvision.
    """
    params = OmegaConf.to_container(config.model.params, resolve=True)
    params.update(runtime_params)
    classifiers = {
        'GradientBoostingClassifier': GradientBoostingClassifier,
        'LogisticRegression': LogisticRegression,
        'XGBClassifier': XGBClassifier,
        'CatBoostClassifier': CatBoostClassifier,
        'LGBMClassifier': LGBMClassifier,
        'KNeighborsClassifier': KNeighborsClassifier,
    }
    if config.model.name in classifiers:
        if config.model.name == 'CatBoostClassifier':
            # Use the sklearn seed alias so the ensemble can assign distinct seeds.
            if 'random_seed' in params:
                params['random_state'] = params.pop('random_seed')
            params.setdefault('verbose', False)
            params.setdefault('allow_writing_files', False)
        if config.model.name != 'KNeighborsClassifier':
            params.setdefault('random_state', config.base.seed)
        model = classifiers[config.model.name](**params)
    elif config.model.name.startswith('custom/'):
        # Import here because models.py also exposes the original helper names.
        import models
        model = getattr(models, config.model.name.split('/')[-1])(**params)
    else:
        if not load_pretrained:
            params['weights'] = None
        model = getattr(torchvision.models, config.model.name)(**params)

    return model

def get_pipeline(config, data: TitanicData):
    """Build an unfitted classical pipeline or ensemble from the configuration.

    Args:
        config: Feature, preprocessing, model and optional ensemble settings.
        data: Shared TitanicData instance supplying numeric/categorical columns.

    Returns:
        A sklearn Pipeline for one classifier, soft/hard voting or stacking.
        Stacking members contain their own preprocessing so each OOF fit learns
        its transforms from its training rows. Ensemble seeds use base.seed.

    Raises:
        ValueError: The model, ensemble size, voting method or stacking folds
            are unsupported. KNN supports a single model only.
    """
    if config.model.name not in {
        'GradientBoostingClassifier', 'LogisticRegression', 'CatBoostClassifier', 'XGBClassifier', 'LGBMClassifier',
        'KNeighborsClassifier',
    }:
        raise ValueError(f'Unsupported classical model: {config.model.name}')
    model = get_model(config)
    n_models = OmegaConf.select(config, 'ensemble.n_models', default=1)
    if type(n_models) is not int or n_models < 1:
        raise ValueError('ensemble.n_models must be a positive integer')
    if config.model.name == 'KNeighborsClassifier' and n_models > 1:
        raise ValueError('KNeighborsClassifier has no random seed; use ensemble.n_models: 1 instead of identical copies')
    method = OmegaConf.select(config, 'ensemble.method', default='soft')
    if method not in {'soft', 'hard', 'stacking'}:
        raise ValueError('ensemble.method must be soft, hard or stacking')
    numeric, categorical = data.numeric_columns, data.categorical_columns
    preprocess = make_preprocess(numeric, categorical, **config.preprocessing)
    if n_models > 1:
        seeds = np.random.RandomState(config.base.seed).randint(np.iinfo(np.int32).max, size=n_models)
        members = []
        for index, seed in enumerate(seeds):
            member = clone(model).set_params(random_state=int(seed))
            if method == 'stacking':
                # cross_val_predict must fit preprocessing inside each stacking fold.
                member = Pipeline([('preprocess', clone(preprocess)), ('model', member)])
            members.append((f'member_{index}', member))
        if method == 'stacking':
            folds = OmegaConf.select(config, 'ensemble.stacking_folds', default=3)
            if type(folds) is not int or folds < 2:
                raise ValueError('ensemble.stacking_folds must be an integer >= 2')
            meta_model = OmegaConf.select(config, 'ensemble.meta_model', default=OmegaConf.create({
                'name': 'LogisticRegression', 'params': {'C': 1.0, 'max_iter': 2000},
            }))
            meta_config = OmegaConf.create({'base': {'seed': config.base.seed},
                                           'model': OmegaConf.to_container(meta_model, resolve=True)})
            model = StackingClassifier(
                estimators=members, final_estimator=get_model(meta_config), stack_method='predict_proba',
                cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.base.seed), n_jobs=1,
            )
            return Pipeline([('model', model)])
        model = VotingClassifier(estimators=members, voting=method, n_jobs=1)
    return Pipeline([
        ('preprocess', preprocess),
        ('model', model),
    ])

def get_dnn_spec(config, data: TitanicData):
    """Build a validated custom-DNN specification for training and evaluation.

    Args:
        config: Custom model, preprocessing, loss, optimizer and tuning settings.
        data: Shared TitanicData instance supplying numeric/categorical columns.

    Returns:
        A dictionary containing the model name, fixed training configuration,
        unfitted preprocessing, optional embedding columns and a trial sampler.

    Raises:
        ValueError: The custom model, binary output/loss, embedding layout,
            ensemble size or tuning parameter definitions are unsupported.
    """
    import models
    model_class = getattr(models, config.model.name.removeprefix('custom/'))
    if not isinstance(model_class, type) or not issubclass(model_class, nn.Module):
        raise ValueError('custom/ must name a torch.nn.Module in models.py')
    if OmegaConf.select(config, 'ensemble.n_models', default=1) != 1:
        raise ValueError('Custom DNNs currently support ensemble.n_models: 1')
    if config.loss.name != 'BCEWithLogitsLoss':
        raise ValueError('Binary DNNs use BCEWithLogitsLoss and one output logit')
    numeric, categorical = data.numeric_columns, data.categorical_columns
    embedding_columns = None
    if issubclass(model_class, models.MyModelEmb):
        if categorical != ['Sex', 'Embarked', 'Cabin', 'Ticket', 'Title'] or config.preprocessing.encoding != 'ordinal':
            raise ValueError('MyModelEmb requires features.feature_set: all and preprocessing.encoding: ordinal')
        embedding_columns = [name.lower() for name in categorical]
    scheduler = OmegaConf.to_container(config.scheduler, resolve=True) if config.scheduler is not None else {}
    configuration = {
        'model_params': OmegaConf.to_container(config.model.params, resolve=True),
        'loss_name': config.loss.name, 'loss_params': OmegaConf.to_container(config.loss.params, resolve=True),
        'optimizer_name': config.optimizer.name,
        'optimizer_params': OmegaConf.to_container(config.optimizer.params, resolve=True),
        'scheduler_name': scheduler.get('name'), 'scheduler_params': scheduler.get('params', {}),
    }
    if configuration['model_params'].get('out', 1) != 1:
        raise ValueError('Binary DNNs require model.params.out: 1')
    space = OmegaConf.to_container(config.tuning.search_space, resolve=True)
    sections = {'model': 'model_params', 'optimizer': 'optimizer_params', 'scheduler': 'scheduler_params'}
    for key, settings in space.items():
        section, separator, parameter = key.partition('.')
        if not separator or section not in sections or not parameter:
            raise ValueError(f'DNN tuning key must be model.*, optimizer.* or scheduler.*: {key}')
        if settings['type'] not in {'int', 'float', 'categorical'}:
            raise ValueError(f'Unknown distribution type: {settings["type"]}')
        if key in {'model.inp', 'model.out'}:
            raise ValueError('DNN input/output sizes are fixed by preprocessing and binary classification')
    if config.tuning.enabled and not space:
        raise ValueError('Optuna requires a non-empty tuning.search_space')

    def suggest_params(trial):
        """Return a copy of the base DNN settings with sampled YAML parameters.

        trial supplies values for model.*, optimizer.* and scheduler.* keys;
        the base configuration captured by get_dnn_spec remains unchanged.
        """
        candidate = copy.deepcopy(configuration)
        for key, settings in space.items():
            section, parameter = key.split('.', 1)
            settings = settings.copy()
            kind = settings.pop('type')
            candidate[sections[section]][parameter] = getattr(trial, f'suggest_{kind}')(key, **settings)
        return candidate

    return {
        'kind': 'dnn', 'model_cls': config.model.name, 'configuration': configuration,
        'preprocess': make_preprocess(numeric, categorical, **config.preprocessing),
        'embedding_columns': embedding_columns, 'suggest_params': suggest_params,
    }

def get_model_report(model):
    """Describe a fitted classifier's parameters and ensemble composition.

    Args:
        model: Fitted classifier or fitted bagging, voting or stacking ensemble.

    Returns:
        A dictionary with parameters and ensemble metadata. Ensemble parameters
        come from the first member; individual random seeds are listed separately.
        Stacking metadata also includes the fitted meta-model and OOF fold count.
    """
    if isinstance(model, (BaggingClassifier, VotingClassifier, StackingClassifier)):
        members = [member.named_steps['model'] if isinstance(member, Pipeline) else member
                   for member in model.estimators_]
        parameters = members[0].get_params()
        # Each fitted member has its own seed; there is no single model seed.
        parameters.pop('random_state', None)
        method = 'stacking' if isinstance(model, StackingClassifier) else getattr(model, 'voting', 'soft')
        ensemble = {
            'n_models': len(members), 'method': method,
            'aggregation': {'soft': 'mean_probability', 'hard': 'majority_vote', 'stacking': 'meta_model'}[method],
            'random_states': [member.get_params()['random_state'] for member in members],
        }
        if method == 'stacking':
            ensemble.update(stacking_folds=model.cv.n_splits, meta_model={
                'name': type(model.final_estimator_).__name__, 'params': model.final_estimator_.get_params(),
            })
        return {'parameters': parameters, 'ensemble': ensemble}
    return {'parameters': model.get_params(), 'ensemble': {'n_models': 1}}

def predict_classifier(model, X):
    """Predict class labels and optional probabilities for the positive class.

    Args:
        model: Fitted classifier or pipeline exposing predict and classes_.
        X: Feature matrix accepted by the fitted model.

    Returns:
        (predictions, probabilities): One-dimensional arrays in X row order.
        probabilities is None when predict_proba is unavailable, as with hard
        voting; otherwise it contains probabilities for class 1.
    """
    predictions = model.predict(X)
    probabilities = None
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)[:, list(model.classes_).index(1)]
    return predictions, probabilities

def get_search_space(config, estimator):
    """Translate YAML tuning ranges into validated Optuna distributions.

    Args:
        config: Configuration containing tuning.search_space and model.name.
        estimator: Unfitted pipeline defining valid model parameter names.

    Returns:
        A dictionary keyed by sklearn parameter paths. Voting/stacking paths
        target the first member; tune_classical shares each setting across members.

    Raises:
        ValueError: The space is empty, a parameter/type is unknown, or an
            ensemble search attempts to override its assigned member seeds.
    """
    distributions = {
        'int': optuna.distributions.IntDistribution,
        'float': optuna.distributions.FloatDistribution,
        'categorical': optuna.distributions.CategoricalDistribution,
    }
    space = {}
    model = estimator.named_steps['model']
    if isinstance(model, (VotingClassifier, StackingClassifier)):
        name, base_model = model.estimators[0]
        prefix = f'model__{name}__'
        if isinstance(base_model, Pipeline):
            base_model = base_model.named_steps['model']
            prefix += 'model__'
    else:
        base_model = model.estimator if isinstance(model, BaggingClassifier) else model
        prefix = 'model__estimator__' if isinstance(model, BaggingClassifier) else 'model__'
    known_parameters = set(base_model.get_params())
    # CatBoost lists only explicitly supplied parameters in get_params().
    if isinstance(base_model, CatBoostClassifier):
        import inspect
        known_parameters.update(inspect.signature(CatBoostClassifier).parameters)
    for name, settings in OmegaConf.to_container(config.tuning.search_space, resolve=True).items():
        key = prefix + name
        if name not in known_parameters:
            raise ValueError(f'Unknown tuning parameter for {config.model.name}: {name}')
        if isinstance(model, (BaggingClassifier, VotingClassifier, StackingClassifier)) and name in {'random_state', 'random_seed'}:
            raise ValueError('Ensemble seeds come from base.seed; do not tune random_state/random_seed')
        kind = settings.pop('type')
        if kind not in distributions:
            raise ValueError(f'Unknown distribution type: {kind}')
        space[key] = distributions[kind](**settings)
    if not space:
        raise ValueError('Optuna requires a non-empty tuning.search_space')
    return space


def write_run_summary(run_dir):
    """Collect saved run artifacts into metrics.json and a readable summary.md.

    Args:
        run_dir: Run directory containing manifest.json, config.yaml and any
            completed evaluation, training or inference reports.

    Returns:
        The metrics dictionary written to disk. Missing or unfinished stages
        leave their scores unset. CV, test and final selection scores are
        recorded separately; this function performs no model fitting or selection.
    """
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    config = OmegaConf.load(run_dir / "config.yaml")
    evaluation_path = run_dir / manifest.get("evaluation", "evaluation/results.json")
    evaluation = json.loads(evaluation_path.read_text()) if evaluation_path.exists() else {}
    training_path = run_dir / "training.json"
    training = json.loads(training_path.read_text()) if training_path.exists() else {}
    test_path = run_dir / manifest.get("test", "test.json")
    test = json.loads(test_path.read_text()) if test_path.exists() else {}
    inference_path = run_dir / "inference.json"
    inference = json.loads(inference_path.read_text()) if inference_path.exists() else {}
    score = evaluation["results"][0] if evaluation.get("status") == "complete" else {}
    selection = training.get("selection", {})
    parameters = training.get("parameters", {})
    # Show configured and tuned parameters; full constructor defaults stay in training.json.
    parameter_names = set(config.model.params) | {
        name.rsplit("__", 1)[-1].rsplit('.', 1)[-1] for name in selection.get("best_params", {})
    }
    final_parameters = {name: value for name, value in parameters.items() if name in parameter_names}
    training_config = inference.get("training_configuration", {}) if manifest["mode"] == "predict" else config
    ensemble = training.get("ensemble") or inference.get("ensemble") or {
        "n_models": OmegaConf.select(OmegaConf.create(training_config), "ensemble.n_models", default=1),
        "method": OmegaConf.select(OmegaConf.create(training_config), "ensemble.method", default="soft"),
    }
    metrics = {
        "experiment": manifest["exp_name"], "status": manifest["status"],
        "run_kind": manifest["run_kind"],
        "model": training.get("model") or score.get("model") or inference.get("model") or manifest.get("model"),
        "evaluation_method": ("nested_cv" if evaluation.get("tuning_enabled") else "fixed_cv") if score else None,
        "cv_accuracy_mean": score.get("outer_accuracy_mean"),
        "cv_accuracy_std": score.get("outer_accuracy_std"),
        "cv_fold_accuracies": score.get("outer_fold_scores", []),
        "oof_accuracy": score.get("outer_oof_accuracy"),
        "test_accuracy": test.get("accuracy"),
        "test_rows": test.get("rows"),
        "test_train_rows": test.get("train_rows"),
        "evaluation_rows": len(evaluation.get("train_indices", [])) if score else None,
        "final_selection_cv_accuracy": selection.get("accuracy"),
        "trials_per_search": config.tuning.n_trials if config.tuning.enabled else 0,
        "final_model_fit_rows": training.get("fitted_rows"),
        "final_model_parameters": final_parameters or None,
        "ensemble": ensemble,
        "probabilities_available": training.get("probabilities_available",
                                                inference.get("probabilities_available", score.get("probabilities_available"))),
        "prediction_rows": inference.get("rows"),
    }
    if str(metrics['model']).startswith('custom/'):
        metrics.update(final_training_epochs=training.get('epochs'), final_optimizer=training.get('optimizer'))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    lines = [
        f"# {manifest['exp_name']}", "",
        f"Model: **{metrics['model'] or 'not loaded'}**. Status: **{manifest['status']}**. "
        f"Run kind: **{manifest['run_kind']}**.", "",
    ]
    if manifest["run_kind"] == "smoke":
        lines += ["**SMOKE: execution check only; do not use these scores to choose a model.**", ""]
    if ensemble["n_models"] > 1:
        method = ensemble.get("method", "soft")
        description = {"soft": "equal averaging of class probabilities",
                       "hard": "majority vote of class labels", "stacking": "meta-model trained on OOF probabilities"}[method]
        lines += [f"Ensemble: **{ensemble['n_models']} models**, **{method}**: {description}. "
                  "CV evaluates the complete ensemble.", ""]
        if method == "hard":
            lines += ["Hard voting provides class labels only; `p_survived` is empty in prediction/OOF CSVs.", ""]
        if ensemble.get("meta_model"):
            lines += [f"Stacking: {ensemble['stacking_folds']} OOF folds; meta-model: "
                      f"`{ensemble['meta_model']['name']}`. Preprocessing is fitted inside each OOF fold.", ""]
        if ensemble.get("random_states"):
            lines += [f"Member seeds: `{ensemble['random_states']}`.", ""]
    if score:
        lines += [
            f"## CV accuracy: {metrics['cv_accuracy_mean']:.2%} ± {metrics['cv_accuracy_std'] * 100:.2f} pp", "",
            "This is the quality estimate on validation folds within train. The ± value is fold SD, "
            "not a confidence interval.", "",
            f"Protocol: **{metrics['evaluation_method']}**, {evaluation['cv']['outer_folds']} evaluation folds, "
            f"{metrics['evaluation_rows']} train rows.", "",
        ]
        if evaluation.get("tuning_enabled"):
            lines += [f"Optuna: {evaluation['cv']['inner_folds']} inner folds, "
                      f"{evaluation['n_trials_per_outer_fold']} trials per outer fold.", ""]
        lines += ["| Evaluation fold | Accuracy |", "|---|---|"]
        lines += [f"| {fold} | {accuracy:.2%} |" for fold, accuracy in enumerate(metrics["cv_fold_accuracies"], 1)]
        lines += ["", f"Pooled OOF accuracy: **{metrics['oof_accuracy']:.2%}**.", ""]
    else:
        lines += ["CV accuracy: **not evaluated in this run**.", ""]
    if test:
        lines += [
            f"## Local test accuracy: {metrics['test_accuracy']:.2%}", "",
            f"The model was fitted on **{metrics['test_train_rows']} train rows** and scored on "
            f"**{metrics['test_rows']} test rows** before any full-data refit. "
            "This is the previously viewed exploratory test.", "",
        ]
    if training:
        lines += ["## Final model", ""]
        if 'epochs' in training:
            lines += [f"Epochs: **{training['epochs']}**. The epoch count and learning-rate schedule were "
                      "selected on internal validation, then replayed during a fresh full-data refit.", "",
                      "Optimizer:", "", "```yaml", OmegaConf.to_yaml(OmegaConf.create(training['optimizer'])).rstrip(),
                      "```", ""]
        if metrics["final_selection_cv_accuracy"] is not None:
            lines += [
                f"Final Optuna selection CV accuracy: **{metrics['final_selection_cv_accuracy']:.2%}**.", "",
                "This score selected the final parameters; it is not an independent quality estimate.", "",
            ]
        else:
            lines += ["Parameters came directly from the YAML; final Optuna search was disabled.", ""]
        lines += [
            f"The saved model was refitted on **{metrics['final_model_fit_rows']} labeled rows**. "
            "It has no separate accuracy on unseen data after that refit. "
            "Use the CV estimate above to assess the training procedure; Kaggle test labels are unavailable.", "",
            "Selected parameters:", "", "```yaml",
            OmegaConf.to_yaml(OmegaConf.create({"model": {"name": metrics["model"], "params": final_parameters}})).rstrip(),
            "```", "",
        ]
    else:
        lines += ["Final model training: **not performed or not yet completed in this run**.", ""]
    if metrics["prediction_rows"] is not None:
        lines += [f"Prediction output: **{metrics['prediction_rows']} rows** in [submission.csv](submission.csv).", ""]
    if manifest.get("error"):
        lines += [f"Run error: `{manifest['error']}`", ""]
    lines += ["## Files", "", "- [metrics.json](metrics.json): compact scores on the 0–1 scale."]
    for path, label in [
        ("training.json", "Final parameter selection and fitted model parameters"),
        ("training_splits.json", "Technical row identities and selection splits"),
        ("test.json", "Train-only model accuracy on test"),
        ("test_predictions.csv", "Test labels and predictions"),
        ("final_trials.csv", "Trials of the separate final Optuna search"),
        ("evaluation/results.json", "Detailed evaluation protocol and fold parameters"),
        ("evaluation/oof.csv", "Out-of-fold predictions"),
        ("model.joblib", "Final saved model"),
        ("model.pt", "Final PyTorch weights and architecture"),
        ("preprocess.joblib", "Fitted preprocessing for the PyTorch model"),
        ("config.yaml", "Resolved run configuration"),
    ]:
        if (run_dir / path).exists():
            lines.append(f"- [{path}]({path}): {label}.")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")
    return metrics



def plot_comparison(summary, oof, run_kind="full", feature_set="base_title_family", inner_folds=3):
    """Plot mean outer accuracy, fold SD and individual OOF fold accuracies.

    Args:
        summary: Per-model records with model name and outer accuracy mean/SD.
        oof: Polars table containing model, outer_fold, y_true and y_pred.
        run_kind: Run label displayed in the figure title.
        feature_set: Feature variant displayed in the title.
        inner_folds: Inner search fold count; zero labels ordinary fixed CV.

    Returns:
        A matplotlib Figure with accuracies on the 0-1 scale. The caller saves
        and closes it. Error bars show fold SD, not confidence intervals.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    names = [row["model"] for row in summary]
    means = [row["outer_accuracy_mean"] for row in summary]
    stds = [row["outer_accuracy_std"] for row in summary]
    ax.errorbar(range(len(names)), means, yerr=stds, fmt="o", capsize=6,
                label="Mean ± fold SD (not a confidence interval)", color="black")
    for i, name in enumerate(names):
        scores = oof.filter(pl.col("model") == name).group_by("outer_fold").agg(
            (pl.col("y_true") == pl.col("y_pred")).mean().alias("accuracy"),
        )["accuracy"].to_list()
        ax.scatter([i] * len(scores), scores, color="tab:blue", alpha=0.6)
    folds = oof["outer_fold"].n_unique()
    protocol = f"{folds} outer / {inner_folds} inner folds" if inner_folds else f"{folds}-fold fixed-parameter CV"
    ax.set(xticks=range(len(names)), xticklabels=names, ylabel="Outer CV accuracy",
           title=f"{run_kind.upper()}: {protocol}; {feature_set}")
    ax.tick_params(axis="x", labelrotation=15)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig



def save_predictions(artifact, predictions, probabilities, kaggle_test, run_dir, model_path, ensemble):
    """Validate and save ordered predictions for classical and PyTorch models.

    Args:
        artifact: Saved feature specification, model name and training configuration.
        predictions: One binary label per Kaggle test row, with shape (n_rows,).
        probabilities: Class-1 probabilities with the same shape, or None.
        kaggle_test: Passenger DataFrame supplying IDs and row order.
        run_dir: Existing directory for submission.csv, predictions.csv and inference.json.
        model_path: Model artifact path recorded in inference provenance.
        ensemble: Member count and optional voting/stacking metadata.

    Returns:
        A Polars DataFrame with PassengerId and integer Survived predictions.

    Raises:
        ValueError: Label/probability shapes or values are invalid.
    """
    if (predictions.shape != (len(kaggle_test),) or not np.isin(predictions, [0, 1]).all()
            or (probabilities is not None and (
                probabilities.shape != (len(kaggle_test),) or not np.isfinite(probabilities).all()
                or not ((0 <= probabilities) & (probabilities <= 1)).all()))):
        raise ValueError("Invalid prediction shape, labels or probabilities")
    submission = kaggle_test.select("PassengerId").with_columns(
        pl.Series("Survived", predictions, dtype=pl.Int64),
    )
    run_dir = Path(run_dir)
    submission.write_csv(run_dir / "submission.csv")
    submission.with_columns(pl.Series("p_survived", probabilities if probabilities is not None
                                      else [None] * len(submission), dtype=pl.Float64)).write_csv(run_dir / "predictions.csv")
    (run_dir / "inference.json").write_text(json.dumps({
        "model_path": str(Path(model_path).resolve()), "model": artifact["model_name"],
        "ensemble": ensemble,
        "probabilities_available": probabilities is not None,
        "feature_set": artifact["feature_set"], "feature_columns": artifact["feature_columns"],
        "rows": len(submission), "training_configuration": artifact["configuration"],
        "kaggle_test_dataframe_sha256": hashlib.sha256(kaggle_test.write_csv().encode()).hexdigest(),
    }, indent=2))
    return submission


def create_run(config, root, train_dataset, resolved):
    """Create a run directory, snapshot its configuration and write initial reports.

    Return (run_dir, manifest). resolved is the fully resolved configuration;
    train_dataset may be None for prediction-only runs.
    """
    mode = config.run.mode
    source_root = Path(__file__).resolve().parent.parent
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = resolve_path(config.output.dir, root) / config.base.exp_name / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(OmegaConf.create(resolved), run_dir / "config.yaml", resolve=True)
    manifest = {
        "run_id": run_id, "exp_name": config.base.exp_name, "mode": mode,
        "run_kind": config.base.run_kind, "status": "running", "seed": config.base.seed,
        "model": config.model.name if mode != "predict" else None,
        "feature_set": config.features.feature_set if mode != "predict" else None,
        "summary": "summary.md", "metrics": "metrics.json",
        "versions": {name: version(name) for name in ["numpy", "polars", "scikit-learn", "torch", "omegaconf", "optuna", "joblib", "catboost", "xgboost", "lightgbm"]},
        "source_sha256": {str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in [(source_root / "run.py"), *(source_root / "src").glob("*.py")]},
    }
    if train_dataset is not None:
        manifest["dataframe_sha256"] = hashlib.sha256(train_dataset.write_csv().encode()).hexdigest()
    save_run_state(run_dir, manifest)
    return run_dir, manifest


def save_run_state(run_dir, manifest):
    """Persist the current run manifest, refresh summary.md and return metrics."""
    run_dir = Path(run_dir)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return write_run_summary(run_dir)


def print_run_summary(run_dir, metrics):
    """Print test/selection scores and locations of the saved run reports."""
    if metrics["test_accuracy"] is not None:
        print(f"Local test accuracy: {metrics['test_accuracy']:.4f} "
              f"({metrics['test_rows']} rows; model fitted on {metrics['test_train_rows']} train rows)")
    if metrics["final_selection_cv_accuracy"] is not None:
        print(f"Final Optuna selection CV accuracy: {metrics['final_selection_cv_accuracy']:.4f} (selection score)")
    print(f"Report: {run_dir / 'summary.md'}")
    print(f"Metrics: {run_dir / 'metrics.json'}")
    print(f"Results: {run_dir}")


def create_evaluation_run(train_dataset, model_specs, results_path, train_i,
                          test_i, outer_splits, *, feature_set, seed, n_trials,
                          epochs, run_kind, device, tune, train_size,
                          inner_splits, val_size, output_dir=None):
    """Create an evaluation directory and save its protocol and row provenance.

    Return (run_dir, manifest). Row indices refer to train_dataset; outer_splits
    index the train subset. This function records splits without fitting.
    """
    n_splits = len(outer_splits)
    y_train = train_dataset["Survived"].to_numpy()[train_i]
    evaluation_kind = "nested" if tune else "fixed"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + f"_{evaluation_kind}_{run_kind}"
    run_dir = Path(output_dir) if output_dir is not None else Path(results_path) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    score_role = "unbiased_outer_cv" if tune else "fixed_configuration_cv"
    manifest = {
        "run_id": run_id, "run_kind": run_kind, "status": "running",
        "metric": "accuracy", "score_role": score_role,
        "selection_score_role": "inner_selection_cv" if tune else None,
        "feature_set": feature_set, "seed": seed, "n_trials_per_outer_fold": n_trials if tune else 0,
        "tuning_enabled": tune, "train_size": train_size,
        "dnn_max_epochs": epochs, "device": device,
        "protocol": f"stratified_{train_size:g}_{evaluation_kind}_outer{n_splits}_inner{inner_splits if tune else 0}_v1",
        "cv": {"outer_folds": n_splits, "inner_folds": inner_splits if tune else 0, "shuffle": True,
               "outer_dnn_internal_validation_fraction": val_size, "std_ddof": 0},
        "test_role": "previously_viewed_exploratory_test",
        "train_indices": train_i.tolist(), "test_indices": test_i.tolist(),
        "train_passenger_ids": train_dataset[train_i]["PassengerId"].to_list(),
        "test_passenger_ids": train_dataset[test_i]["PassengerId"].to_list(),
        "dataframe_sha256": hashlib.sha256(train_dataset.write_csv().encode()).hexdigest(),
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ["data.py", "evaluation.py", "features.py", "models.py", "training.py", "utils.py"]},
        "versions": {package: version(package) for package in [
            "numpy", "polars", "scikit-learn", "torch", "optuna", "optuna-integration", "joblib", "matplotlib", "marimo",
        ]},
        "outer_splits": [], "models": list(model_specs),
    }
    for fold, (train, val) in enumerate(outer_splits, start=1):
        inner = (list(StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed + fold)
                      .split(train, y_train[train])) if tune else [])
        manifest["outer_splits"].append({
            "outer_fold": fold, "train_source_indices": train_i[train].tolist(),
            "val_source_indices": train_i[val].tolist(),
            "inner_splits": [{"train_source_indices": train_i[train[a]].tolist(),
                              "val_source_indices": train_i[train[b]].tolist()} for a, b in inner],
        })
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return run_dir, manifest


def save_evaluation_results(run_dir, manifest, summary, fold_records, oof):
    """Save OOF predictions, reports and the plot; return the completed results."""
    oof.write_csv(run_dir / "oof.csv")
    pl.DataFrame(summary).drop("outer_fold_scores").write_csv(run_dir / "comparison.csv")
    figure = plot_comparison(summary, oof, manifest["run_kind"], manifest["feature_set"], inner_folds=manifest["cv"]["inner_folds"])
    figure.savefig(run_dir / "outer_accuracy.png", dpi=150)
    import matplotlib.pyplot as plt
    plt.close(figure)
    manifest.update(status="complete")
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    result = {**manifest, "run_dir": str(run_dir.resolve()), "results": summary, "folds": fold_records}
    (run_dir / "results.json").write_text(json.dumps(result, indent=2))
    return result


def save_trials(study, path):
    """Write an Optuna study's trial table to the supplied CSV path."""
    study.trials_dataframe().to_csv(path, index=False)


def save_test_results(run_dir, config, train_dataset, train_i, test_i,
                         predictions, probabilities, accuracy, selection, *, epochs=None):
    """Save test accuracy, predictions and the train-only fit provenance.

    train_i/test_i index the original dataset. selection contains settings
    chosen using train rows; optional epochs records the DNN epoch count.
    """
    run_dir = Path(run_dir)
    test = train_dataset[test_i]
    test.select("PassengerId", pl.col("Survived").alias("y_true")).with_columns(
        pl.Series("y_pred", predictions, dtype=pl.Int64),
        pl.Series("p_survived", probabilities if probabilities is not None
                  else [None] * len(test), dtype=pl.Float64),
    ).write_csv(run_dir / "test_predictions.csv")
    report = {
        "model": config.model.name, "metric": "accuracy", "accuracy": accuracy,
        "score_role": "test_accuracy", "train_rows": len(train_i), "rows": len(test_i),
        "train_source_indices": train_i.tolist(), "test_source_indices": test_i.tolist(),
        "selection": {key: selection[key] for key in ["score_role", "accuracy", "best_params"] if key in selection},
    }
    if epochs is not None:
        report["epochs"] = epochs
    (run_dir / "test.json").write_text(json.dumps(report, indent=2))


def save_classical_fold(fitted, run_dir, record):
    """Save a fold pipeline and add checkpoint/ensemble details to record."""
    saved_path = run_dir / record["model"] / f"fold_{record['outer_fold']}.joblib"
    joblib.dump(fitted, saved_path)
    report = get_model_report(fitted.named_steps["model"])
    record.update(checkpoint=str(saved_path.relative_to(run_dir)),
                  estimator_params=report["parameters"], ensemble=report["ensemble"])


def save_dnn_fold(fitted, model_cls, run_dir, record, *, train_indices,
                  val_indices, study=None):
    """Save a DNN fold and preprocessing and extend its provenance record.

    train_indices/val_indices refer to original dataset rows.
    """
    model_dir = run_dir / record["model"]
    fold, fold_seed = record["outer_fold"], record["seed"]
    configuration = fitted["configuration"]
    saved_path = model_dir / f"fold_{fold}.pt"
    preprocess_path = model_dir / f"fold_{fold}_preprocess.joblib"
    checkpoint = {
        "model_state_dict": fitted["model"].state_dict(),
        "model_class": model_cls if isinstance(model_cls, str) else model_cls.__name__,
        "model_params": fitted["model_params"], "embedding_metadata": fitted["embedding_metadata"],
        "configuration": configuration, "seed": fold_seed,
        "best_epoch": fitted["best_epoch"], "weights_epoch": fitted["weights_epoch"],
        "last_epoch": fitted["last_epoch"], "best_val_acc": fitted["best_val_acc"],
        "last_val_acc": fitted["last_val_acc"],
        "validation_role": "internal_outer_train_validation",
    }
    torch.save(checkpoint, saved_path)
    joblib.dump(fitted["preprocess"], preprocess_path)
    record.update(
        inner_selection_accuracy=float(study.best_value) if study is not None else None,
        best_params=study.best_params if study is not None else {},
        configuration=configuration, checkpoint=str(saved_path.relative_to(run_dir)),
        preprocessing=str(preprocess_path.relative_to(run_dir)),
        train_source_indices=train_indices.tolist(), val_source_indices=val_indices.tolist(),
        best_epoch=fitted["best_epoch"], weights_epoch=fitted["weights_epoch"],
        last_epoch=fitted["last_epoch"], best_val_acc=fitted["best_val_acc"],
        last_val_acc=fitted["last_val_acc"],
        internal_validation_history={k: fitted[k] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]},
    )


def record_oof_predictions(oof_rows, record, val, train_i, train_dataset,
                           y_train, predictions, probabilities):
    """Append one fold's predictions with original passenger and row identities."""
    name, fold = record["model"], record["outer_fold"]
    for position, prediction, probability in zip(val, predictions, probabilities if probabilities is not None
                                                else [None] * len(val)):
        source = int(train_i[position])
        oof_rows.append({"model": name, "train_index": int(position), "source_index": source,
                         "PassengerId": int(train_dataset[source, "PassengerId"]),
                         "y_true": int(y_train[position]), "outer_fold": fold,
                         "y_pred": int(prediction), "p_survived": float(probability) if probability is not None else None})


def save_classical_model(fitted, X, train_dataset, config, run_dir, selection):
    """Save a final classical model and write its training reports.

    Return the model.joblib path. Split provenance is moved out of selection
    into training_splits.json, preserving the compact training report.
    """
    run_dir = Path(run_dir)
    y = train_dataset["Survived"].to_numpy()
    artifact = {
        "pipeline": fitted, "feature_set": config.features.feature_set,
        "feature_columns": X.columns, "model_name": config.model.name,
        "configuration": OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
    }
    model_path = run_dir / "model.joblib"
    joblib.dump(artifact, model_path)
    # Keep row identities and split indices out of the readable training report.
    selection_splits = {key: selection.pop(key) for key in [
        "train_indices", "excluded_test_indices", "splits",
    ] if key in selection}
    (run_dir / "training_splits.json").write_text(json.dumps({
        "train_passenger_ids": train_dataset["PassengerId"].to_list(),
        "selection": selection_splits,
    }, indent=2))
    report = {
        "model": config.model.name, "fit_role": "final_refit_all_labeled_rows",
        "fitted_rows": len(y), "splits_path": "training_splits.json",
        "selection": selection, **get_model_report(fitted.named_steps["model"]),
        "probabilities_available": hasattr(fitted, "predict_proba"),
    }
    (run_dir / "training.json").write_text(json.dumps(report, indent=2))
    return model_path


def save_custom_dnn(fitted, selected, X, train_dataset, config, run_dir, selection,
                    train_i, test_i, splits, train_indices, val_indices):
    """Save a final DNN, preprocessing and training reports.

    Return the model.pt path. selected supplies the frozen validation decisions;
    train_i/test_i and train_indices/val_indices refer to original dataset rows.
    """
    run_dir = Path(run_dir)
    y = train_dataset["Survived"].to_numpy()
    configuration = fitted["configuration"]
    artifact = {
        'model_name': config.model.name, 'model_params': fitted['model_params'],
        'embedding_metadata': fitted['embedding_metadata'],
        'model_state_dict': {key: value.detach().cpu() for key, value in fitted['model'].state_dict().items()},
        'preprocess_path': 'preprocess.joblib', 'feature_set': config.features.feature_set,
        'feature_columns': X.columns, 'configuration': OmegaConf.to_container(config, resolve=True),
    }
    model_path = run_dir/'model.pt'
    torch.save(artifact, model_path)
    joblib.dump(fitted['preprocess'], run_dir/'preprocess.joblib')
    (run_dir/'training_splits.json').write_text(json.dumps({
        'train_passenger_ids': train_dataset['PassengerId'].to_list(),
        'selection': {'train_indices': train_i.tolist(), 'excluded_test_indices': test_i.tolist(),
                      'splits': splits, 'train_source_indices': train_indices.tolist(), 'val_source_indices': val_indices.tolist()},
    }, indent=2))
    (run_dir/'training.json').write_text(json.dumps({
        'model': config.model.name, 'fit_role': 'final_refit_all_labeled_rows', 'fitted_rows': len(y),
        'splits_path': 'training_splits.json', 'selection': selection, 'parameters': fitted['model_params'],
        'optimizer': {'name': configuration['optimizer_name'], 'params': configuration['optimizer_params']},
        'scheduler': {'name': configuration['scheduler_name'], 'params': configuration['scheduler_params']},
        'epochs': selected['best_epoch'], 'learning_rates': fitted['learning_rates'],
        'validation': {'score_role': 'internal_validation_accuracy', 'accuracy': selected['best_val_acc']},
        'ensemble': {'n_models': 1}, 'probabilities_available': True,
    }, indent=2))
    return model_path


def controlled_model_specs(train_dataset, seed):
    """Build unfitted model specifications for the controlled nested-CV comparison.

    Args:
        train_dataset: Raw DataFrame used to determine base_title_family columns.
        seed: Random seed for the classical classifier constructors.

    Returns:
        Specifications for logistic regression, random forest, Gradient Boosting
        and SmallMLP, including separate preprocessing and Optuna search settings.
    """
    from sklearn.ensemble import RandomForestClassifier
    from models import SmallMLP, comparison_search_spaces, suggest_small_mlp

    _, numeric, categorical = make_features(train_dataset, "base_title_family")
    preprocessing = make_preprocess(numeric, categorical)
    estimators = {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=seed),
        "RandomForestClassifier": RandomForestClassifier(n_jobs=1, random_state=seed),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=seed),
    }
    spaces = comparison_search_spaces()
    specs = {name: {"kind": "classical", "estimator": Pipeline([
        ("preprocess", clone(preprocessing)), ("model", estimator),
    ]), "distributions": spaces[name]} for name, estimator in estimators.items()}
    specs["MLP"] = {"kind": "dnn", "model_cls": SmallMLP, "preprocess": clone(preprocessing),
                     "suggest_params": suggest_small_mlp}
    return specs
