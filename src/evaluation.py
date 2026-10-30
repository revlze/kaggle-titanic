"""Shared 80/20 split, nested CV, saved-model predictions and OOF artifacts."""

import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import optuna
import polars as pl
import torch
from omegaconf import OmegaConf
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from features import make_features, make_preprocess
from models import SmallMLP, comparison_search_spaces, suggest_small_mlp
from training import fit_dnn, predict_dnn, restore_dnn, tune_classical
from data import split_development
from utils import get_model_report, predict_classifier


def write_run_summary(run_dir):
    """Collect saved scores into one readable report; never fit or select a model."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    config = OmegaConf.load(run_dir / "config.yaml")
    evaluation_path = run_dir / manifest.get("evaluation", "evaluation/results.json")
    evaluation = json.loads(evaluation_path.read_text()) if evaluation_path.exists() else {}
    training_path = run_dir / "training.json"
    training = json.loads(training_path.read_text()) if training_path.exists() else {}
    inference_path = run_dir / "inference.json"
    inference = json.loads(inference_path.read_text()) if inference_path.exists() else {}
    score = evaluation["results"][0] if evaluation.get("status") == "complete" else {}
    selection = training.get("selection", {})
    parameters = training.get("parameters", {})
    # Show configured and tuned parameters; full constructor defaults stay in training.json.
    parameter_names = set(config.model.params) | {
        name.rsplit("__", 1)[-1] for name in selection.get("best_params", {})
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
            "This is the quality estimate on held-out development folds. The ± value is fold SD, "
            "not a confidence interval.", "",
            f"Protocol: **{metrics['evaluation_method']}**, {evaluation['cv']['outer_folds']} evaluation folds, "
            f"{metrics['evaluation_rows']} development rows.", "",
        ]
        if evaluation.get("tuning_enabled"):
            lines += [f"Optuna: {evaluation['cv']['inner_folds']} inner folds, "
                      f"{evaluation['n_trials_per_outer_fold']} trials per outer fold.", ""]
        lines += ["| Evaluation fold | Accuracy |", "|---|---|"]
        lines += [f"| {fold} | {accuracy:.2%} |" for fold, accuracy in enumerate(metrics["cv_fold_accuracies"], 1)]
        lines += ["", f"Pooled OOF accuracy: **{metrics['oof_accuracy']:.2%}**.", ""]
    else:
        lines += ["CV accuracy: **not evaluated in this run**.", ""]
    if training:
        lines += ["## Final model", ""]
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
        ("final_trials.csv", "Trials of the separate final Optuna search"),
        ("evaluation/results.json", "Detailed evaluation protocol and fold parameters"),
        ("evaluation/oof.csv", "Out-of-fold predictions"),
        ("model.joblib", "Final saved model"),
        ("config.yaml", "Resolved run configuration"),
    ]:
        if (run_dir / path).exists():
            lines.append(f"- [{path}]({path}): {label}.")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")
    return metrics


def make_cv_plan(y, seed, development_size=0.8, n_splits=5):
    train_i, holdout_i = split_development(y, seed, development_size)
    development_y = np.asarray(y)[train_i]
    outer = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(train_i, development_y))
    assert not set(train_i) & set(holdout_i)
    assert len(set(train_i) | set(holdout_i)) == len(y)
    assert sorted(np.concatenate([val for _, val in outer]).tolist()) == list(range(len(train_i)))
    for fit, val in outer:
        assert not set(fit) & set(val)
    return train_i, holdout_i, outer


def validate_oof(oof, train_dataset, train_i, holdout_i, outer_splits, model_names):
    """Validate coverage, source-row identity, labels, folds and probabilities."""
    assert set(oof["model"].to_list()) == set(model_names)
    for name in model_names:
        rows = oof.filter(pl.col("model") == name)
        assert len(rows) == len(train_i)
        assert rows["source_index"].n_unique() == len(train_i)
        assert set(rows["source_index"].to_list()) == set(train_i)
        assert not set(rows["source_index"].to_list()) & set(holdout_i)
        for fold, (_, val) in enumerate(outer_splits, start=1):
            part = rows.filter(pl.col("outer_fold") == fold).sort("development_index")
            assert part["development_index"].to_list() == val.tolist()
            source = train_i[val]
            assert part["source_index"].to_list() == source.tolist()
            assert part["PassengerId"].to_list() == train_dataset[source]["PassengerId"].to_list()
            assert part["y_true"].to_list() == train_dataset[source]["Survived"].to_list()
        if rows["p_survived"].null_count():
            assert rows["p_survived"].null_count() == len(rows)
        else:
            probabilities = rows["p_survived"].to_numpy()
            assert np.isfinite(probabilities).all()
            assert ((probabilities >= 0) & (probabilities <= 1)).all()
        assert set(rows["y_pred"].to_list()) <= {0, 1}


def plot_comparison(summary, oof, run_kind="full", feature_set="base_title_family", inner_folds=3):
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


def run_cv_experiment(train_dataset, model_specs, results_path, *, feature_set,
                          seed, n_trials=20, epochs=30, run_kind="full", device="cpu",
                          tune=True, development_size=0.8, n_splits=5, inner_splits=3,
                          output_dir=None):
    """Only the development set enters evaluation fits or hyperparameter search.

    Classical specs contain estimator/distributions; DNN specs contain
    model_cls/preprocess/suggest_params and optional embedding_columns.
    The larger notebook DNNs can use the same evaluation without changing
    their architectures. Holdout labels are used only to stratify the split.
    With tune=False, classical estimators run ordinary CV without inner folds.
    Existing notebook calls retain the nested-CV defaults.
    """
    if (tune and n_trials < 1) or epochs < 1:
        raise ValueError("n_trials and epochs must be positive")
    if not tune and any(spec["kind"] != "classical" for spec in model_specs.values()):
        raise ValueError("Fixed-parameter evaluation here supports classical estimators only")
    X, _, _ = make_features(train_dataset, feature_set)
    y = train_dataset["Survived"].to_numpy()
    train_i, holdout_i, outer_splits = make_cv_plan(y, seed, development_size, n_splits)
    X_dev, y_dev = X[train_i], y[train_i]
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
        "tuning_enabled": tune, "development_size": development_size,
        "dnn_max_epochs": epochs, "device": device,
        "protocol": f"stratified_{development_size:g}_{evaluation_kind}_outer{n_splits}_inner{inner_splits if tune else 0}_v1",
        "cv": {"outer_folds": n_splits, "inner_folds": inner_splits if tune else 0, "shuffle": True,
               "outer_dnn_internal_validation_fraction": 0.2, "std_ddof": 0},
        "holdout_role": "previously_viewed_exploratory_holdout_not_evaluated",
        "train_indices": train_i.tolist(), "holdout_indices": holdout_i.tolist(),
        "train_passenger_ids": train_dataset[train_i]["PassengerId"].to_list(),
        "holdout_passenger_ids": train_dataset[holdout_i]["PassengerId"].to_list(),
        "dataframe_sha256": hashlib.sha256(train_dataset.write_csv().encode()).hexdigest(),
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ["data.py", "evaluation.py", "features.py", "models.py", "training.py", "utils.py"]},
        "versions": {package: version(package) for package in [
            "numpy", "polars", "scikit-learn", "torch", "optuna", "optuna-integration", "joblib", "matplotlib", "marimo",
        ]},
        "outer_splits": [], "models": list(model_specs),
    }
    for fold, (fit, val) in enumerate(outer_splits, start=1):
        inner = (list(StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed + fold)
                      .split(fit, y_dev[fit])) if tune else [])
        manifest["outer_splits"].append({
            "outer_fold": fold, "fit_source_indices": train_i[fit].tolist(),
            "validation_source_indices": train_i[val].tolist(),
            "inner_splits": [{"fit_source_indices": train_i[fit[a]].tolist(),
                              "validation_source_indices": train_i[fit[b]].tolist()} for a, b in inner],
        })
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    oof_rows, fold_records, summary = [], [], []
    for name, spec in model_specs.items():
        model_dir = run_dir / name
        model_dir.mkdir()
        model_scores, selection_scores = [], []
        for fold, (fit, val) in enumerate(outer_splits, start=1):
            fold_seed = seed + fold
            X_fit, y_fit = X_dev[fit], y_dev[fit]
            inner = (list(StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=fold_seed)
                          .split(fit, y_fit)) if tune else [])
            study = None
            record = {"model": name, "outer_fold": fold, "seed": fold_seed,
                      "selection_score_role": "inner_selection_cv" if tune else None, "score_role": score_role}
            if spec["kind"] == "classical":
                if tune:
                    search = tune_classical(
                        spec["estimator"], spec["distributions"], X_fit, y_fit, inner,
                        seed=fold_seed, n_trials=n_trials,
                    )
                    fitted = search.best_estimator_
                    study = search.study_
                    record.update(inner_selection_accuracy=float(search.best_score_), best_params=search.best_params_)
                else:
                    fitted = clone(spec["estimator"]).fit(X_fit, y_fit)
                    record.update(inner_selection_accuracy=None, best_params={})
                saved_path = model_dir / f"fold_{fold}.joblib"
                joblib.dump(fitted, saved_path)
                # Outer predictions come directly from the serialized estimator.
                restored = joblib.load(saved_path)
                predictions, probabilities = predict_classifier(restored, X_dev[val])
                original_predictions, original_probabilities = predict_classifier(fitted, X_dev[val])
                assert np.array_equal(predictions, original_predictions)
                assert np.array_equal(probabilities, original_probabilities)
                model_report = get_model_report(restored.named_steps["model"])
                record.update(checkpoint=str(saved_path.relative_to(run_dir)),
                              estimator_params=model_report["parameters"], ensemble=model_report["ensemble"])
            elif spec["kind"] == "dnn":
                study = optuna.create_study(
                    direction="maximize", sampler=optuna.samplers.TPESampler(seed=fold_seed),
                    pruner=optuna.pruners.NopPruner(),
                )
                def objective(trial):
                    configuration = spec["suggest_params"](trial)
                    trial.set_user_attr("configuration", configuration)
                    scores, best_epochs = [], []
                    for inner_fold, (a, b) in enumerate(inner, start=1):
                        fitted = fit_dnn(
                            spec["model_cls"], configuration, spec["preprocess"],
                            X_fit[a], y_fit[a], X_fit[b], y_fit[b],
                            seed=fold_seed * 10 + inner_fold, epochs=epochs,
                            embedding_columns=spec.get("embedding_columns"), device=device,
                        )
                        inner_predictions, _ = predict_dnn(fitted["model"], fitted["preprocess"], X_fit[b], device)
                        scores.append(float(accuracy_score(y_fit[b], inner_predictions)))
                        best_epochs.append(fitted["best_epoch"])
                    trial.set_user_attr("inner_best_epochs", best_epochs)
                    trial.set_user_attr("inner_fold_scores", scores)
                    return float(np.mean(scores))

                study.optimize(objective, n_trials=n_trials)
                configuration = study.best_trial.user_attrs["configuration"]
                # The outer training rows supply BOTH fit and stopping subsets.
                a, b = train_test_split(np.arange(len(fit)), test_size=0.2,
                                        stratify=y_fit, random_state=fold_seed)
                assert not set(fit[a]) & set(val)
                assert not set(fit[b]) & set(val)
                fitted = fit_dnn(
                    spec["model_cls"], configuration, spec["preprocess"],
                    X_fit[a], y_fit[a], X_fit[b], y_fit[b], seed=fold_seed,
                    epochs=epochs, embedding_columns=spec.get("embedding_columns"), device=device,
                )
                saved_path = model_dir / f"fold_{fold}.pt"
                preprocess_path = model_dir / f"fold_{fold}_preprocess.joblib"
                checkpoint = {
                    "model_state_dict": fitted["model"].state_dict(),
                    "model_class": spec["model_cls"].__name__,
                    "model_params": fitted["model_params"], "embedding_metadata": fitted["embedding_metadata"],
                    "configuration": configuration, "seed": fold_seed,
                    "best_epoch": fitted["best_epoch"], "weights_epoch": fitted["weights_epoch"],
                    "last_epoch": fitted["last_epoch"], "best_val_acc": fitted["best_val_acc"],
                    "last_val_acc": fitted["last_val_acc"],
                    "validation_role": "internal_outer_train_validation",
                }
                torch.save(checkpoint, saved_path)
                joblib.dump(fitted["preprocess"], preprocess_path)
                loaded = torch.load(saved_path, map_location=device, weights_only=True)
                restored = restore_dnn(spec["model_cls"], loaded["model_params"], loaded["embedding_metadata"])
                restored.load_state_dict(loaded["model_state_dict"])
                restored_preprocess = joblib.load(preprocess_path)
                stop_predictions, _ = predict_dnn(restored, restored_preprocess, X_fit[b], device)
                assert accuracy_score(y_fit[b], stop_predictions) == loaded["best_val_acc"]
                predictions, probabilities = predict_dnn(restored, restored_preprocess, X_dev[val], device)
                original_predictions, original_probabilities = predict_dnn(fitted["model"], fitted["preprocess"], X_dev[val], device)
                assert np.array_equal(predictions, original_predictions)
                assert np.array_equal(probabilities, original_probabilities)
                record.update(
                    inner_selection_accuracy=float(study.best_value), best_params=study.best_params,
                    configuration=configuration, checkpoint=str(saved_path.relative_to(run_dir)),
                    preprocessing=str(preprocess_path.relative_to(run_dir)),
                    fit_source_indices=train_i[fit[a]].tolist(), stop_source_indices=train_i[fit[b]].tolist(),
                    best_epoch=loaded["best_epoch"], weights_epoch=loaded["weights_epoch"],
                    last_epoch=loaded["last_epoch"], best_val_acc=loaded["best_val_acc"],
                    last_val_acc=loaded["last_val_acc"],
                    internal_validation_history={k: fitted[k] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]},
                )
            else:
                raise ValueError(f"Unknown model kind: {spec['kind']}")
            outer_accuracy = float(accuracy_score(y_dev[val], predictions))
            record["outer_val_acc"] = outer_accuracy
            record["checkpoint_predictions_verified"] = True
            fold_records.append(record)
            model_scores.append(outer_accuracy)
            if tune:
                selection_scores.append(record["inner_selection_accuracy"])
                study.trials_dataframe().to_csv(model_dir / f"fold_{fold}_trials.csv", index=False)
            for position, prediction, probability in zip(val, predictions, probabilities if probabilities is not None
                                                        else [None] * len(val)):
                source = int(train_i[position])
                oof_rows.append({"model": name, "development_index": int(position), "source_index": source,
                                 "PassengerId": int(train_dataset[source, "PassengerId"]),
                                 "y_true": int(y_dev[position]), "outer_fold": fold,
                                 "y_pred": int(prediction), "p_survived": float(probability) if probability is not None else None})
        model_oof = pl.DataFrame([r for r in oof_rows if r["model"] == name])
        summary.append({"model": name, "score_role": score_role, "run_kind": run_kind,
                        "outer_accuracy_mean": float(np.mean(model_scores)),
                        "outer_accuracy_std": float(np.std(model_scores)), "outer_fold_scores": model_scores,
                        "outer_oof_accuracy": float(accuracy_score(model_oof["y_true"], model_oof["y_pred"])),
                        "probabilities_available": model_oof["p_survived"].null_count() == 0,
                        "mean_inner_selection_accuracy": float(np.mean(selection_scores)) if tune else None})
    oof = pl.DataFrame(oof_rows).with_columns(pl.col("p_survived").cast(pl.Float64)).sort(["model", "development_index"])
    validate_oof(oof, train_dataset, train_i, holdout_i, outer_splits, list(model_specs))
    oof.write_csv(run_dir / "oof.csv")
    pl.DataFrame(summary).drop("outer_fold_scores").write_csv(run_dir / "comparison.csv")
    figure = plot_comparison(summary, oof, run_kind, feature_set, inner_folds=inner_splits if tune else 0)
    figure.savefig(run_dir / "outer_accuracy.png", dpi=150)
    import matplotlib.pyplot as plt
    plt.close(figure)
    manifest.update(status="complete", oof_verified=True)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    result = {**manifest, "run_dir": str(run_dir.resolve()), "results": summary, "folds": fold_records}
    (run_dir / "results.json").write_text(json.dumps(result, indent=2))
    return result


def controlled_model_specs(train_dataset, seed):
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
