"""Run one Titanic experiment using config.yaml; no command-line options."""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from omegaconf import OmegaConf

from data import load_data, resolve_path
from evaluation import run_cv_experiment, write_run_summary
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

    train_dataset, estimator, distributions = None, None, None
    is_dnn = config.model.name.startswith('custom/')
    device = OmegaConf.select(config, 'training.device', default='cpu')
    if mode != "predict":
        utils.set_seed(config.base.seed)
        train_dataset = load_data(resolve_path(config.data.train_path, root))
        if is_dnn:
            model_spec = utils.get_dnn_spec(config, train_dataset)
        else:
            estimator = utils.get_pipeline(config, train_dataset)
            if config.tuning.enabled:
                distributions = utils.get_search_space(config, estimator)
            model_spec = {'kind': 'classical', 'estimator': estimator, 'distributions': distributions}

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
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in [Path(__file__), *(ROOT / "src").glob("*.py")]},
    }
    if train_dataset is not None:
        manifest["dataframe_sha256"] = hashlib.sha256(train_dataset.write_csv().encode()).hexdigest()
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    write_run_summary(run_dir)
    try:
        if mode in {"full", "evaluate"}:
            print(f"Evaluating {config.model.name}; Optuna enabled: {config.tuning.enabled}")
            result = run_cv_experiment(
                train_dataset, {config.model.name: model_spec}, run_dir,
                feature_set=config.features.feature_set, seed=config.base.seed,
                tune=config.tuning.enabled, n_trials=config.tuning.n_trials,
                development_size=config.validation.development_size,
                n_splits=config.validation.n_splits, inner_splits=config.validation.inner_splits,
                epochs=OmegaConf.select(config, 'training.epochs', default=30), device=device,
                batch_size=OmegaConf.select(config, 'training.batch_size', default=64),
                stopping_size=OmegaConf.select(config, 'validation.stopping_size', default=0.2),
                run_kind=config.base.run_kind, output_dir=run_dir / "evaluation",
            )
            manifest["evaluation"] = "evaluation/results.json"
            # Publish CV scores before the separate final search starts.
            manifest_path.write_text(json.dumps(manifest, indent=2))
            write_run_summary(run_dir)
            score = result["results"][0]
            print(f"CV accuracy: {score['outer_accuracy_mean']:.4f} ± {score['outer_accuracy_std']:.4f} (fold SD)")
        if mode in {"full", "train"}:
            print("Training final pipeline on all labeled rows")
            model_path = (fit_custom_dnn(train_dataset, model_spec, config, run_dir) if is_dnn
                          else fit_classical(train_dataset, estimator, config, run_dir, distributions))
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
        manifest_path.write_text(json.dumps(manifest, indent=2))
        metrics = write_run_summary(run_dir)
    if metrics["final_selection_cv_accuracy"] is not None:
        print(f"Final Optuna selection CV accuracy: {metrics['final_selection_cv_accuracy']:.4f} (selection score)")
    print(f"Report: {run_dir / 'summary.md'}")
    print(f"Metrics: {run_dir / 'metrics.json'}")
    print(f"Results: {run_dir}")
    return run_dir


def main():
    run_pipeline(OmegaConf.load(ROOT / "config.yaml"))


if __name__ == "__main__":
    main()
