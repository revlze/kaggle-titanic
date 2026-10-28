import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Titanic: исследование данных

    Цель — понять качество данных и обосновать preprocessing и признаки.
    Здесь используется размеченный train. Kaggle test не используется
    для выбора модели, признаков или параметров. Описательные связи
    с Survived не устанавливают причин выживания.
    """)
    return


@app.cell
def _():
    import polars as pl
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    return np, pl, plt, sns


@app.cell
def _(mo, pl):
    base_path = mo.notebook_location().parent
    train_dataset = pl.read_csv(base_path / 'data/titanic/train.csv')
    return (train_dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Качество данных
    """)
    return


@app.cell
def _(pl, train_dataset):
    train_quality = pl.DataFrame({
        'metric': ['rows', 'features_without_target', 'missing_values', 'duplicated_rows', 'unique_passenger_ids'],
        'value': [train_dataset.height, train_dataset.width - 1,
                  train_dataset.null_count().sum_horizontal().item(),
                  train_dataset.is_duplicated().sum(), train_dataset['PassengerId'].n_unique()],
    })
    train_quality
    return


@app.cell
def _(pl, train_dataset):
    schema = pl.DataFrame({'feature': train_dataset.columns,
                           'dtype': [str(t) for t in train_dataset.dtypes]})
    schema
    return


@app.cell
def _(train_dataset):
    train_dataset.describe()
    return


@app.cell
def _(pl, train_dataset):
    constant_columns = [name for name in train_dataset.columns if train_dataset[name].n_unique() <= 1]
    pl.DataFrame({'constant_feature': constant_columns}, schema={'constant_feature': pl.String})
    return


@app.cell
def _(pl, train_dataset):
    suspicious = train_dataset.filter(
        (pl.col('Age') < 0) | (pl.col('Fare') < 0)
        | ~pl.col('Pclass').is_in([1, 2, 3])
        | ~pl.col('Survived').is_in([0, 1])
    )
    suspicious
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Проверяются отрицательные Age/Fare и недопустимые значения Pclass/Survived.
    Пустая таблица означает отсутствие именно этих нарушений, а не всех
    возможных проблем. PassengerId — идентификатор; его уникальность
    учитывается отдельно от дубликатов целых строк.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Пропуски
    """)
    return


@app.cell
def _(pl, train_dataset):
    train_nulls = train_dataset.null_count().transpose(
        include_header=True, header_name='feature', column_names=['null_count'],
    ).filter(pl.col('null_count') > 0).with_columns(
        (pl.col('null_count') / train_dataset.height).alias('null_fraction')
    )
    train_nulls
    return (train_nulls,)


@app.cell
def _(plt, train_nulls):
    fig_missing, _ax = plt.subplots(figsize=(7, 3))
    _ax.bar(train_nulls['feature'].to_list(), train_nulls['null_count'].to_list())
    _ax.set(title='Missing values in train', ylabel='Count')
    fig_missing.tight_layout()
    fig_missing
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Age можно заполнить медианой, Embarked — модой как понятным baseline.
    Эти значения должны оцениваться только на обучающем фолде.
    Большая доля пропусков Cabin не доказывает бесполезность столбца:
    исходная категория, наличие каюты и палуба могут содержать разный сигнал.
    Их пользу нужно проверять экспериментом, а не по числу пропусков.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Баланс классов
    """)
    return


@app.cell
def _(pl, train_dataset):
    target_balance = train_dataset.group_by('Survived').len().sort('Survived').with_columns(
        (pl.col('len') / train_dataset.height).alias('fraction')
    )
    target_balance
    return (target_balance,)


@app.cell
def _(pl, plt, target_balance):
    fig_target, _ax = plt.subplots(figsize=(5, 3))
    _ax.bar(target_balance['Survived'].cast(pl.String).to_list(), target_balance['len'].to_list())
    _ax.set(title='Target balance', xlabel='Survived', ylabel='Passengers')
    fig_target.tight_layout()
    fig_target
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Классы несбалансированы. Accuracy необходимо сравнивать с правилом
    «всегда предсказывать наиболее частый класс». Для разбиений используем
    стратификацию; близкая к доле большинства accuracy не гарантирует
    полезных предсказаний положительного класса.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Категории и выживаемость
    """)
    return


@app.cell
def _(pl, train_dataset):
    category_survival = {
        name: train_dataset.group_by(name).agg(
            pl.len().alias('passengers'),
            pl.col('Survived').mean().alias('survival_rate'),
        ).sort(name)
        for name in ['Sex', 'Pclass', 'Embarked']
    }
    category_survival
    return (category_survival,)


@app.cell
def _(category_survival, pl, plt):
    fig_categories, _axes = plt.subplots(1, 3, figsize=(13, 4))
    for _ax, (_name, _table) in zip(_axes, category_survival.items()):
        _ax.bar(_table[_name].cast(pl.String).fill_null('missing').to_list(),
                _table['survival_rate'].to_list())
        _ax.set(title=_name, ylabel='Survival fraction', ylim=(0, 1))
        for _i, (_count, _rate) in enumerate(zip(_table['passengers'], _table['survival_rate'])):
            _ax.annotate(f'n={_count}', (_i, _rate), xytext=(0, -14 if _rate > 0.9 else 3),
                         textcoords='offset points', ha='center',
                         color='white' if _rate > 0.9 else 'black')
    fig_categories.tight_layout()
    fig_categories
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    У женщин наблюдается большая доля выживших. Для Pclass меньший код
    означает более высокий класс обслуживания; сравнивать следует доли,
    а не только абсолютное число выживших. Аналогично большее число
    погибших среди отплывших из Southampton само по себе не означает
    более высокий риск: численность групп различается.
    Для пропущенного Embarked группа состоит всего из двух пассажиров;
    её долю выживших нельзя считать устойчивой оценкой.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Возраст, стоимость билета и возможные выбросы
    """)
    return


@app.cell
def _(np, plt, sns, train_dataset):
    fig_numeric, _axes = plt.subplots(2, 2, figsize=(12, 7))
    for _name, _ax in zip(['Age', 'Fare'], _axes[0]):
        sns.histplot(data=train_dataset.to_dict(as_series=False), x=_name, hue='Survived',
                     bins=25, element='step', ax=_ax)
    sns.boxplot(data=train_dataset.to_dict(as_series=False), x='Survived', y='Age', ax=_axes[1, 0])
    sns.histplot(np.log1p(train_dataset['Fare'].to_numpy()), bins=30, ax=_axes[1, 1])
    _axes[1, 1].set(xlabel='log1p(Fare)', title='Fare on logarithmic scale (EDA only)')
    fig_numeric.tight_layout()
    fig_numeric
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    У Fare длинный правый хвост: log1p(Fare) — мотивированная гипотеза
    для будущего эксперимента, здесь логарифм используется только на графике.
    Пассажиры за усами boxplot Age не становятся ошибочными наблюдениями.
    Удаление редких, но допустимых значений без сравнения метрик не обосновано.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Связи численных признаков
    """)
    return


@app.cell
def _(plt, sns, train_dataset):
    numeric_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']
    complete_numeric = train_dataset.select(numeric_features).drop_nulls()
    fig_corr, _ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(complete_numeric.corr().to_numpy(), annot=True, fmt='.2f',
                xticklabels=numeric_features, yticklabels=numeric_features,
                cmap='coolwarm', center=0, ax=_ax)
    fig_corr.tight_layout()
    fig_corr
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Корреляции рассчитаны на строках с заполненным Age; такой срез может
    отличаться от всего train. Sex и Embarked в числовой heatmap отсутствуют,
    поэтому по ней нельзя заключать, что с таргетом связан только Pclass.
    Связи SibSp/Parch и Pclass/Fare мотивируют анализ избыточности, но не
    автоматическое удаление признаков. FamilySize проверяется отдельно.

    ## Выводы для моделирования

    Начальный preprocessing: median Age, mode Embarked, one-hot категорий,
    scaling численных признаков для чувствительных к масштабу моделей.
    Все обучаемые преобразования включаются в Pipeline внутри CV.
    Новые признаки проверяются относительно семи исходных, без одновременной
    настройки модели. Начальные гипотезы — Title, FamilySize и отдельная
    проверка исходных Ticket/Cabin. Целевая метрика сравнений — accuracy.
    """)
    return


if __name__ == "__main__":
    app.run()
