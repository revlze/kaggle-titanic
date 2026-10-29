"""Row-wise Titanic features and unfitted preprocessing shared by experiments."""

import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def make_features(df, feature_set="base"):
    variants = {
        "base": [],
        "base_title": ["Title"],
        "base_family": ["FamilySize"],
        "base_title_family": ["Title", "FamilySize"],
        "base_ticket_cabin": ["Ticket", "Cabin"],
        "all": ["Title", "FamilySize", "Ticket", "Cabin"],
    }
    extra = variants[feature_set]
    numeric = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    categorical = ["Sex", "Embarked"]
    if "Title" in extra:
        df = df.with_columns(
            pl.col("Name").str.extract(r",\s*([^.]+)\.", 1).alias("Title")
        )
    if "FamilySize" in extra:
        df = df.with_columns(
            (pl.col("SibSp") + pl.col("Parch") + 1).alias("FamilySize")
        )
        numeric += ["FamilySize"]
    categorical += [c for c in ["Cabin", "Ticket", "Title"] if c in extra]
    return df.select(numeric + categorical), numeric, categorical


def make_preprocess(numeric_columns, categorical_columns, encoding="onehot", scale=True):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    if encoding == "onehot":
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    elif encoding == "ordinal":
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    else:
        raise ValueError(f"Unknown encoding: {encoding}")
    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), numeric_columns),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(missing_values=None, strategy="most_frequent")),
            ("encoder", encoder),
        ]), categorical_columns),
    ])
