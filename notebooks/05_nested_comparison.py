import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import marimo as mo
    import polars as pl
    from evaluation import controlled_model_specs, plot_comparison, run_cv_experiment
    return controlled_model_specs, mo, pl, plot_comparison, run_cv_experiment


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Контролируемое сравнение: nested CV

    LogisticRegression, RandomForest, GradientBoosting и небольшая MLP
    получают `base_title_family` и одинаковые внешние разбиения.
    Primary metric — accuracy. Исходное stratified-разделение 80/20 сохранено.
    На тренировочных 80% выполняются 5 внешних и 3 внутренних фолда.

    Optuna использует только внутренние фолды. При финальном обучении DNN
    stopping split выделяется из outer train; outer validation используется
    только для предсказаний сохранённой модели. Прежний holdout уже
    просматривался и не участвует в выборе или оценке этого эксперимента.

    Для проверки протокола включи `RUN_SMOKE`. Для полноценного сравнения
    включи `RUN_FULL`. Smoke-метрики не предназначены для выбора модели.
    Оба запуска по умолчанию выключены.
    """)
    return


@app.cell
def _():
    SEED = 0xFACED
    RUN_SMOKE = False
    RUN_FULL = False
    N_TRIALS = 20
    MAX_EPOCHS = 30
    return MAX_EPOCHS, N_TRIALS, RUN_FULL, RUN_SMOKE, SEED


@app.cell
def _(mo, pl):
    base_path = mo.notebook_location().parent
    train_dataset = pl.read_csv(base_path / "data/titanic/train.csv")
    results_path = base_path / "notebooks/results/nested_comparison"
    return results_path, train_dataset


@app.cell
def _(
    MAX_EPOCHS, N_TRIALS, RUN_FULL, RUN_SMOKE, SEED,
    controlled_model_specs, results_path, run_cv_experiment, train_dataset,
):
    comparison_run = None
    if RUN_FULL and RUN_SMOKE:
        raise ValueError("Выбери один режим: RUN_FULL или RUN_SMOKE")
    if RUN_FULL or RUN_SMOKE:
        comparison_run = run_cv_experiment(
            train_dataset, controlled_model_specs(train_dataset, SEED), results_path,
            feature_set="base_title_family", seed=SEED,
            n_trials=N_TRIALS if RUN_FULL else 1,
            epochs=MAX_EPOCHS if RUN_FULL else 2,
            run_kind="full" if RUN_FULL else "smoke",
        )
    return (comparison_run,)


@app.cell
def _(comparison_run, pl):
    _table = None if comparison_run is None else pl.DataFrame(comparison_run["results"]).drop("outer_fold_scores")
    _table
    return


@app.cell
def _(comparison_run, pl, plot_comparison):
    _figure = None
    if comparison_run is not None:
        _figure = plot_comparison(
            comparison_run["results"], pl.read_csv(comparison_run["run_dir"] + "/oof.csv"),
            comparison_run["run_kind"], comparison_run["feature_set"],
        )
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `outer_accuracy_mean/std` — среднее и стандартное отклонение внешней
    accuracy; SD не является доверительным интервалом.
    `mean_inner_selection_accuracy` относится к подбору и не заменяет outer CV.
    По одному максимуму внешней accuracy также нельзя гарантировать
    устойчивое превосходство победителя среди нескольких алгоритмов.

    В папке запуска сохраняются JSON протокола и метрик, `oof.csv`,
    `comparison.csv`, график, trial-таблицы и модели каждого outer-фолда.
    OOF покрывает все 712 development-строк ровно один раз для каждой модели;
    строки 20% holdout в OOF отсутствуют. При изменении исходных данных
    проверка использует фактический размер development-части.

    `best_val_acc` и `best_epoch` относятся к внутреннему stopping split.
    `weights_epoch` совпадает с `best_epoch`; `last_epoch` — выполненные эпохи.
    `outer_val_acc` рассчитана по предсказаниям загруженных весов, без дообучения.
    DNN оставляет 20% outer train для stopping, поэтому не обучается на этих
    строках градиентными шагами; это часть заранее заданного протокола.
    """)
    return


if __name__ == "__main__":
    app.run()
