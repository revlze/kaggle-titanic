"""Load Titanic CSVs and reproduce the shared development/holdout split."""

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split


def resolve_path(path, root):
    """Configured relative paths start at the project root, not the current cwd."""
    path = Path(path)
    return path if path.is_absolute() else Path(root) / path


def load_data(path, *, training=True):
    df = pl.read_csv(path)
    required = {"PassengerId", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch",
                "Ticket", "Fare", "Cabin", "Embarked"}
    if training:
        required.add("Survived")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing Titanic columns: {sorted(missing)}")
    if not len(df):
        raise ValueError("The dataset is empty")
    ids = df["PassengerId"]
    if not ids.dtype.is_integer() or ids.null_count() or ids.n_unique() != len(df):
        raise ValueError("PassengerId must contain unique, non-null integer identifiers")
    if training:
        target = df["Survived"]
        if target.null_count() or set(target.to_list()) != {0, 1}:
            raise ValueError("Survived must contain both binary classes 0 and 1, without nulls")
    elif "Survived" in df.columns:
        raise ValueError("Inference CSV must not contain Survived")
    return df


def split_development(y, seed, development_size=0.8):
    return train_test_split(
        np.arange(len(y)), train_size=development_size, stratify=y, random_state=seed,
    )
