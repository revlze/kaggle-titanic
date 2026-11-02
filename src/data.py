"""Load Titanic data and keep one shared train/test split."""

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from features import make_features


class TitanicData:
    """Store labeled features and one train/test split for a run.

    CV and training share this instance and its X_train/y_train, X_test/y_test
    and original row indices. Feature extraction is row-wise; preprocessing
    statistics are learned later inside model fits.

    X_test/y_test are labeled rows reserved from train.csv. Kaggle test.csv
    is loaded separately for inference.
    """

    def __init__(self, train_dataset, *, feature_set, seed, train_size=0.8):
        """Build features and create the stratified split once.

        Args:
            train_dataset: Raw labeled passenger DataFrame.
            feature_set: Feature variant passed to make_features.
            seed: Random seed for the train/test split.
            train_size: Fraction of rows assigned to train.
        """
        self.train_dataset = train_dataset
        self.feature_set = feature_set
        self.seed = seed
        self.train_size = train_size
        self.X, self.numeric_columns, self.categorical_columns = make_features(train_dataset, feature_set)
        self.y = train_dataset["Survived"].to_numpy()
        (self.X_train, self.X_test, self.y_train, self.y_test,
         self.train_i, self.test_i) = train_test_split(
            self.X, self.y, np.arange(len(self.y)), train_size=train_size,
            stratify=self.y, random_state=seed,
        )


def resolve_path(path, root):
    """Resolve a configured path relative to the project root.

    Args:
        path: Absolute or relative filesystem path.
        root: Base directory for relative paths.

    Returns:
        A pathlib.Path; absolute inputs are preserved.
    """
    path = Path(path)
    return path if path.is_absolute() else Path(root) / path


def load_data(path, *, training=True):
    """Read a Titanic CSV and validate its columns, passenger IDs and target.

    Args:
        path: CSV file to read into a Polars DataFrame.
        training: Require both binary Survived classes when True; require
            Survived to be absent when False.

    Returns:
        The loaded DataFrame with its original row order and feature values.

    Raises:
        ValueError: Required columns are missing, the dataset is empty,
            passenger IDs are invalid, or the target violates training mode.
    """
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
