import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _():
    from torch.utils.data import Dataset, DataLoader, random_split
    import torch
    import numpy as np
    import polars as pl
    from pathlib import Path
    import seaborn as sns
    import matplotlib.pyplot as plt

    return Dataset, Path, pl, plt, sns


@app.cell
def _(Path):
    base_path = Path('/Users/platon/Documents/final_proj')
    return (base_path,)


@app.cell
def _(base_path, pl):
    train_dataset = pl.read_csv(base_path / 'data/titanic/train.csv')
    test_dataset = pl.read_csv(base_path / 'data/titanic/test.csv')
    return (train_dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### хз надо или нет, просто попробовал
    """)
    return


@app.cell
def _(Dataset):
    class TitanicSet(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index):
            item = self.dataset[index]
            target_col = 'Survived'
            target = item[target_col]
            features = item.drop(target_col)
            return features, target

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### качество данных
    """)
    return


@app.cell
def _(pl, train_dataset):
    train_quality = pl.DataFrame({
        "metric": [
            "features",
            "missing_values",
            "duplicated_rows",
        ],
        "value": [
            train_dataset.shape[1],
            train_dataset.null_count().sum_horizontal().item(),
            train_dataset.is_duplicated().sum(),
        ],
    })
    train_quality
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - короче 12 фичей
    - 866 пропусков
    """)
    return


@app.cell
def _(pl, train_dataset):
    train_nulls = train_dataset.null_count().transpose(
        include_header=True,
        header_name="feature",
        column_names=["null_count"],
    ).filter(pl.col('null_count') > 0)
    return (train_nulls,)


@app.cell
def _(pl, train_dataset):
    pl.DataFrame(train_dataset.schema)
    return


@app.cell
def _(train_dataset):
    train_dataset.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    пропуски:
    - есть в возрасте, их наверно просто достаточно заполнить медианным значением по age
    - есть в номере кабины, где-то 77% пропусков, не знаю что сделать, это категориальный, наверно можно убрать, нужно посмотреть на график с таргетом
    - еще есть в порте отправления, но их всего два, думаю можно просто кинуть самую частую категорию просто туда
    """)
    return


@app.cell
def _(train_nulls):
    train_nulls
    return


@app.cell
def _(plt, sns, train_nulls):
    sns.barplot(train_nulls, x="feature", y="null_count")
    plt.title('Missing values')
    return


@app.cell
def _(sns, train_dataset):
    sns.countplot(data=train_dataset, x='Survived')
    return


@app.cell
def _(train_dataset):
    train_dataset.schema
    return


@app.cell
def _(pl, train_dataset):
    numeric_features = []
    for feat_name, type_feat in train_dataset.schema.items():
        if type_feat != pl.String:
            numeric_features.append(feat_name)
    return (numeric_features,)


@app.cell
def _(numeric_features, plt, sns, train_dataset):
    for _feat_name in numeric_features:
        if _feat_name != 'Survived':
            sns.histplot(train_dataset, x=_feat_name, 
            hue='Survived', multiple="dodge",shrink=0.8,
            bins=10) 
            plt.title(f'Distribituion of {_feat_name}')       
            plt.show()
            plt.tight_layout()
    return


@app.cell
def _(sns, train_dataset):
    sns.histplot(train_dataset, x='Sex', 
            hue='Survived', multiple="dodge",shrink=0.8,
            bins=10) 
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    видно короче что женщин больше выжило
    """)
    return


@app.cell
def _(numeric_features, sns, train_dataset):
    sns.heatmap(train_dataset[numeric_features].drop_nulls().corr(), annot=True,
    xticklabels=numeric_features, yticklabels=numeric_features, cmap='coolwarm',
    center=0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    вообще по признакам не прям видно сильную корреляцию, что-то есть между parch и sibsp, а еще есть между pclass и fare, но дропать их не буду, стоит наверно проверить потом с ними и без
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    с таргетом связь какая-то есть только у Pclass, получается чем выше класс, тем выше выживаемость
    а еще есть с fare связь, типо я так понимаю, чем от стоимости билета выжиываемость как-то зависит
    """)
    return


@app.cell
def _(sns, train_dataset):
    sns.boxplot(data=train_dataset, x="Survived", y="Age")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    тут видно какие-то выбросы сверху, ну наверно это просто старенькие люди, которых немного было и так, поэтому всё норм
    """)
    return


@app.cell
def _(sns, train_dataset):
    sns.countplot(data=train_dataset, x="Pclass", hue="Survived")
    return


@app.cell
def _(sns, train_dataset):
    sns.countplot(data=train_dataset, x="Embarked", hue="Survived")
    return


@app.cell
def _(mo):
    mo.md(r"""
    еще видно, что много народу померло, котрые отплывали с Southampton, скорее всего просто там и больше гораздо село, но всё же довольно большой разброс в примерно 200 человек
    """)
    return


if __name__ == "__main__":
    app.run()
