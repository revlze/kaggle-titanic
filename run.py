"""Run one Titanic experiment using config.yaml; no command-line options."""

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from omegaconf import OmegaConf

from data import TitanicData, load_data, resolve_path
from evaluation import run_cv_experiment
import utils
from training import fit_classical, predict_classical, fit_custom_dnn, predict_custom_dnn


def run_pipeline(config, root=ROOT):
    """Run one configured experiment; also usable from notebooks."""
    resolved = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    mode = config.run.mode
    if mode not in {"full", "evaluate", "train", "predict"}:
        raise ValueError("run.mode must be full, evaluate, train or predict")
    if config.metric.name != "accuracy":
        raise ValueError("This pipeline uses metric.name: accuracy")
    if config.base.run_kind not in {"full", "smoke"}:
        raise ValueError("base.run_kind must be full or smoke")
    if mode == "predict" and not config.run.model_path:
        raise ValueError("Set run.model_path to a saved model.joblib or model.pt for predict mode")

    train_dataset, data, estimator, distributions = None, None, None, None
    is_dnn = config.model.name.startswith('custom/')
    device = OmegaConf.select(config, 'training.device', default='cpu')
    if mode != "predict":
        utils.set_seed(config.base.seed)
        train_dataset = load_data(resolve_path(config.data.train_path, root))
        data = TitanicData(
            train_dataset, feature_set=config.features.feature_set, seed=config.base.seed,
            development_size=config.validation.development_size,
        )
        if is_dnn:
            model_spec = utils.get_dnn_spec(config, data)
        else:
            estimator = utils.get_pipeline(config, data)
            if config.tuning.enabled:
                distributions = utils.get_search_space(config, estimator)
            model_spec = {'kind': 'classical', 'estimator': estimator, 'distributions': distributions}

    run_dir, manifest = utils.create_run(config, root, train_dataset, resolved)
    try:
        if mode in {"full", "evaluate"}:
            print(f"Evaluating {config.model.name}; Optuna enabled: {config.tuning.enabled}")
            result = run_cv_experiment(
                data, {config.model.name: model_spec}, run_dir,
                tune=config.tuning.enabled, n_trials=config.tuning.n_trials,
                n_splits=config.validation.n_splits, inner_splits=config.validation.inner_splits,
                epochs=OmegaConf.select(config, 'training.epochs', default=30), device=device,
                batch_size=OmegaConf.select(config, 'training.batch_size', default=64),
                stopping_size=OmegaConf.select(config, 'validation.stopping_size', default=0.2),
                run_kind=config.base.run_kind, output_dir=run_dir / "evaluation",
            )
            manifest["evaluation"] = "evaluation/results.json"
            # Publish CV scores before the separate final search starts.
            utils.save_run_state(run_dir, manifest)
            score = result["results"][0]
            print(f"CV accuracy: {score['outer_accuracy_mean']:.4f} ± {score['outer_accuracy_std']:.4f} (fold SD)")
        if mode in {"full", "evaluate", "train"}:
            print("Evaluating the development-trained model on holdout")
            refit_all = mode != "evaluate"
            model_path = (fit_custom_dnn(data, model_spec, config, run_dir, refit_all=refit_all) if is_dnn
                          else fit_classical(data, estimator, config, run_dir, distributions, refit_all=refit_all))
            manifest["holdout"] = "holdout.json"
            if model_path is not None:
                manifest["model_path"] = model_path.name
        if mode in {"full", "predict"}:
            if mode == "predict":
                model_path = resolve_path(config.run.model_path, root)
                manifest["model_path"] = str(model_path.resolve())
            # Test data enter the workflow only after training, or in predict mode.
            test_dataset = load_data(resolve_path(config.data.test_path, root), training=False)
            submission = (predict_custom_dnn(model_path, test_dataset, run_dir, device)
                          if model_path.suffix == '.pt' else predict_classical(model_path, test_dataset, run_dir))
            manifest["submission"] = "submission.csv"
            print(f"Submission: {len(submission)} rows")
        manifest["status"] = "complete"
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        metrics = utils.save_run_state(run_dir, manifest)
    utils.print_run_summary(run_dir, metrics)
    return run_dir


def main():
    run_pipeline(OmegaConf.load(ROOT / "config.yaml"))


if __name__ == "__main__":
    main()
