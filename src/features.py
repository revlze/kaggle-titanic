"""Row-wise Titanic features and unfitted preprocessing shared by experiments."""

import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def make_features(df: pl.DataFrame, feature_set="base"):
    """Select a feature variant and derive its row-wise Titanic features.

    Args:
        df: Passenger DataFrame containing the raw columns for the variant.
        feature_set: Key in variants, selecting title/family, ticket or cabin
            features in addition to the base columns.

    Returns:
        (X, numeric, categorical): A Polars feature DataFrame and the column
        lists for each preprocessing branch. Pclass is categorical. X contains
        numeric columns first and excludes PassengerId and Survived.

    Cabin mean/count features use entries within one passenger's row.
    The older all variant includes raw Ticket/Cabin instead of parsed blocks.
    """
    variants = {
        "base": [],
        "base_title": ["Title"],
        "base_family": ["FamilySize"],
        'base_extticket': ['TicketString', 'TicketNum'],
        'base_extcabin': ['CabinLet', 'CabinNum', 'CabinCount'],
        "base_title_family": ["Title", "FamilySize",],
        "base_title_family_extticket": ["Title", "FamilySize", 'TicketString', 'TicketNum'],
        "base_title_family_extticket_extcabin": ["Title", "FamilySize",
                                                  'TicketString', 'TicketNum',
                                                  'CabinLet', 'CabinNum', 'CabinCount'
                                                ],
        "base_ticket_cabin": ["Ticket", "Cabin"],
        "all": ["Title", "FamilySize", "Ticket", "Cabin"],
    }
    extra = variants[feature_set]
    numeric = [ "Age", "SibSp", "Parch", "Fare"]
    categorical = ["Pclass", "Sex", "Embarked"]
    if "Title" in extra:
        df = df.with_columns(
            pl.col("Name").str.extract(r",\s*([^.]+)\.", 1).alias("Title")
        )
    if "FamilySize" in extra:
        df = df.with_columns(
            (pl.col("SibSp") + pl.col("Parch") + 1).alias("FamilySize")
        )

    if all(i in extra for i in ['TicketString', 'TicketNum']):
        df = df.with_columns(
            pl.col('Ticket')
            .str.extract(r"^(.*?)\s+\d+$", 1)
            .fill_null("NONE")
            .alias("TicketString"),

            pl.col("Ticket")
            .str.extract(r"(\d+)$", 1)
            .cast(pl.Int64, strict=False)
            .alias("TicketNum"),
        )

    if all(i in extra for i in ['CabinLet', 'CabinNum', 'CabinCount']):
        df = df.with_columns(
            pl.col('Cabin')
            .str.extract_all(r"[A-Z]+")
            .list.unique(maintain_order=True)
            .list.join("")
            .alias("CabinLet"),

            pl.col("Cabin")
            .str.extract_all(r"\d+")
            .list.eval(pl.element().cast(pl.Int64))
            .list.mean()
            .alias('CabinNum'),

            pl.col("Cabin")
            .str.count_matches(r"\b[A-Z]+\d+\b")
            .alias("CabinCount"),
        )


    categorical += [c for c in ["Cabin", "Ticket", "Title", 'TicketString', 'CabinLet'] if c in extra]
    numeric += [n for n in ['FamilySize', 'TicketNum', 'CabinNum', 'CabinCount'] if n in extra]

    return df.select(numeric + categorical), numeric, categorical


def make_preprocess(numeric_columns, categorical_columns, encoding="onehot", scale=True):
    """Build an unfitted imputation, encoding and optional scaling transformer.

    Args:
        numeric_columns: Columns imputed with their training medians.
        categorical_columns: Columns imputed with their training modes.
        encoding: onehot ignores unknown categories; ordinal maps them to -1.
        scale: Apply StandardScaler to the numeric branch when True.

    Returns:
        A ColumnTransformer producing numeric values before encoded categories.
        Fit it on each training split to learn preprocessing statistics.

    Raises:
        ValueError: encoding is neither onehot nor ordinal.
    """
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
