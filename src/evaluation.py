"""Stratified fixed/nested CV and OOF accuracy evaluation."""

import numpy as np
import polars as pl
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split


from data import TitanicData
from training import fit_dnn, predict_dnn, tune_classical, tune_dnn
import utils
from utils import predict_classifier


def run_cv_experiment(data: TitanicData, model_specs, results_path, *,
                          n_trials=20, epochs=30, run_kind="full", device="cpu",
                          tune=True, n_splits=5, inner_splits=3,
                          output_dir=None, batch_size=64, val_size=0.2):
    """Only the train set enters evaluation fits or hyperparameter search.

    Classical specs contain estimator/distributions; DNN specs contain
    model_cls/preprocess/suggest_params and optional embedding_columns;
    fixed DNN evaluation takes parameters from spec['configuration'].
    The larger notebook DNNs can use the same evaluation without changing
    their architectures. This stage reserves test rows; the fitting stage
    evaluates them after CV and before the optional full-data refit.
    With tune=False, models run ordinary CV without inner search folds.
    DNNs keep an internal validation split separate from evaluation folds.
    Existing notebook calls retain the nested-CV defaults.

    Args:
        data: Shared TitanicData instance containing the fixed train/test split.
        model_specs: Mapping from model names to classical or DNN specifications.
        results_path: Parent directory for a generated evaluation run directory.
        n_trials: Optuna trials per model and outer fold when tune is True.
        epochs: Maximum DNN training epochs for each fit.
        run_kind: full/smoke label; actual budgets are supplied by the caller.
        device: PyTorch device used for DNN fits and predictions.
        tune: Run nested CV with inner parameter search when True; otherwise
            evaluate the fixed model configurations with ordinary outer CV.
        n_splits: Number of stratified outer evaluation folds.
        inner_splits: Number of inner folds used for parameter selection.
        output_dir: Exact new evaluation directory, overriding results_path/run_id.
        batch_size: Requested DNN batch size.
        val_size: Fraction of DNN training rows reserved for internal validation.

    Returns:
        A results dictionary containing the protocol, model summaries, fold records
        and run directory. Also save OOF predictions, checkpoints, search tables
        when tuning, JSON reports and an accuracy plot in the evaluation directory.
    """
    if (tune and n_trials < 1) or epochs < 1:
        raise ValueError("n_trials and epochs must be positive")
    
    train_dataset = data.train_dataset
    X, y = data.X_train, data.y_train
    train_i, test_i = data.train_i, data.test_i
    seed, feature_set = data.seed, data.feature_set

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    outer_splits = list(cv.split(X, y))

    run_dir, manifest = utils.create_evaluation_run(
        train_dataset, model_specs, results_path, train_i, test_i, outer_splits,
        feature_set=feature_set, seed=seed, n_trials=n_trials, epochs=epochs,
        run_kind=run_kind, device=device, tune=tune, train_size=data.train_size,
        inner_splits=inner_splits, val_size=val_size, output_dir=output_dir,
    )

    score_role = manifest["score_role"]

    oof_rows, fold_records, summary = [], [], []

    for name, spec in model_specs.items():
        model_dir = run_dir / name
        model_dir.mkdir(parents=True)
        model_scores, selection_scores = [], []
        for fold, (train, val) in enumerate(outer_splits, start=1):
            fold_seed = seed + fold
            X_train, y_train = X[train], y[train]
            inner = (list(StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=fold_seed)
                          .split(X_train, y_train)) if tune else [])
            study = None

            record = {"model": name, "outer_fold": fold, "seed": fold_seed,
                      "selection_score_role": "inner_selection_cv" if tune else None, "score_role": score_role}
            
            if spec["kind"] == "classical":
                if tune:
                    search = tune_classical(
                        spec["estimator"], spec["distributions"], X_train, y_train, inner,
                        seed=fold_seed, n_trials=n_trials,
                    )
                    fitted = search.best_estimator_
                    study = search.study_
                    record.update(inner_selection_accuracy=float(search.best_score_), best_params=search.best_params_)
                else:
                    fitted = clone(spec["estimator"]).fit(X_train, y_train)
                    record.update(inner_selection_accuracy=None, best_params={})
                utils.save_classical_fold(fitted, run_dir, record)
                predictions, probabilities = predict_classifier(fitted, X[val])
            elif spec["kind"] == "dnn":
                if tune:
                    study = tune_dnn(spec, X_train, y_train, inner, seed=fold_seed, n_trials=n_trials,
                                     epochs=epochs, device=device, batch_size=batch_size, val_size=val_size)
                    configuration = study.best_trial.user_attrs['configuration']
                else:
                    configuration = spec['configuration']
                # Split only the outer train rows into train and val for epoch selection.
                X_train, X_val, y_train, y_val, train_fold_i, val_fold_i = train_test_split(
                    X_train, y_train, train, test_size=val_size,
                    stratify=y_train, random_state=fold_seed,
                )
                fitted = fit_dnn(
                    spec["model_cls"], configuration, spec["preprocess"],
                    X_train, y_train, X_val, y_val, seed=fold_seed,
                    epochs=epochs, embedding_columns=spec.get("embedding_columns"), device=device, batch_size=batch_size,
                )
                utils.save_dnn_fold(
                    fitted, spec["model_cls"], run_dir, record,
                    train_indices=train_i[train_fold_i], val_indices=train_i[val_fold_i],
                    study=study,
                )
                predictions, probabilities = predict_dnn(fitted["model"], fitted["preprocess"], X[val], device)
            else:
                raise ValueError(f"Unknown model kind: {spec['kind']}")
            
            outer_accuracy = float(accuracy_score(y[val], predictions))
            record["outer_val_acc"] = outer_accuracy
            fold_records.append(record)
            model_scores.append(outer_accuracy)

            if tune:
                selection_scores.append(record["inner_selection_accuracy"])
                utils.save_trials(study, model_dir / f"fold_{fold}_trials.csv")
            utils.record_oof_predictions(
                oof_rows, record, val, train_i, train_dataset, y, predictions, probabilities,
            )

        model_oof = pl.DataFrame([r for r in oof_rows if r["model"] == name])

        summary.append({"model": name, "score_role": score_role, "run_kind": run_kind,
                        "outer_accuracy_mean": float(np.mean(model_scores)),
                        "outer_accuracy_std": float(np.std(model_scores)), "outer_fold_scores": model_scores,
                        "outer_oof_accuracy": float(accuracy_score(model_oof["y_true"], model_oof["y_pred"])),
                        "probabilities_available": model_oof["p_survived"].null_count() == 0,
                        "mean_inner_selection_accuracy": float(np.mean(selection_scores)) if tune else None})
        
    oof = pl.DataFrame(oof_rows).with_columns(pl.col("p_survived").cast(pl.Float64)).sort(["model", "train_index"])

    return utils.save_evaluation_results(run_dir, manifest, summary, fold_records, oof)
