"""Neural architectures and notebook Optuna search spaces."""

import optuna
import torch
from omegaconf import OmegaConf
from torch import nn

import utils


def build_dnn(model_cls, model_params, preprocessing, input_dim, embedding_columns=None):
    """Infer input sizes from fitted preprocessing and instantiate a fresh DNN.

    Args:
        model_cls: PyTorch module class or configured model name.
        model_params: Architecture settings; inp is inferred here.
        preprocessing: Fitted ColumnTransformer with output slices and categories.
        input_dim: Width of its transformed feature matrix.
        embedding_columns: Ordered category names used for embedding width settings.

    Returns:
        (model, params, embedding_metadata): The new model, resolved constructor
        parameters and optional numeric/category slices and vocabulary sizes.
        Embedding vocabularies reserve one extra slot for unknown categories.
    """
    params = model_params.copy()
    params["inp"] = input_dim
    embedding_metadata = None
    if embedding_columns:
        numeric = preprocessing.output_indices_["numeric"]
        categorical = preprocessing.output_indices_["categorical"]
        sizes = [len(c) + 1 for c in preprocessing.named_transformers_["categorical"].named_steps["encoder"].categories_]
        params["inp"] = numeric.stop - numeric.start + sum(
            params.get(f"{column}_emb_dim", 2) for column in embedding_columns
        )
        embedding_metadata = {
            "num_slice": [numeric.start, numeric.stop],
            "cat_slice": [categorical.start, categorical.stop],
            "cat_sizes": sizes,
        }
    return restore_dnn(model_cls, params, embedding_metadata), params, embedding_metadata


def restore_dnn(model_cls, model_params, embedding_metadata=None):
    """Instantiate a DNN architecture from constructor and embedding metadata.

    Args:
        model_cls: PyTorch module class or configured model name.
        model_params: Saved constructor parameters, including inferred input size.
        embedding_metadata: Optional serialized slices and category vocabulary sizes.

    Returns:
        A newly initialized module. The caller loads checkpoint weights into it.
    """
    params = model_params.copy()
    if embedding_metadata:
        params.update({
            "num_slice": slice(*embedding_metadata["num_slice"]),
            "cat_slice": slice(*embedding_metadata["cat_slice"]),
            "cat_sizes": embedding_metadata["cat_sizes"],
        })
    if isinstance(model_cls, str):
        return utils.get_model(OmegaConf.create({'model': {'name': model_cls, 'params': {}}}), **params)
    return model_cls(**params)


class SmallMLP(nn.Module):
    """Small ReLU MLP with dropout for the controlled model comparison."""

    def __init__(self, inp, out=1, hidden_sizes=(32,), dropout=0.1):
        """Initialize the dense network.

        Args:
            inp: Number of preprocessed input features.
            out: Number of output logits; binary classification uses one.
            hidden_sizes: Non-empty sequence of hidden layer widths.
            dropout: Drop probability after each hidden activation.
        """
        super().__init__()
        layers = []
        for before, after in zip([inp, *hidden_sizes], hidden_sizes):
            layers.extend([nn.Linear(before, after), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_sizes[-1], out))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """Map a (batch, inp) feature tensor to (batch, out) raw logits."""
        return self.model(x)


def comparison_search_spaces():
    """Return Optuna spaces for logistic regression, random forest and GB.

    The dictionary is keyed by classifier name. Each value maps sklearn
    pipeline parameter names, prefixed with model__, to Optuna distributions.
    """
    return {
        "LogisticRegression": {
            "model__C": optuna.distributions.FloatDistribution(1e-3, 100, log=True),
        },
        "RandomForestClassifier": {
            "model__n_estimators": optuna.distributions.IntDistribution(10, 200),
            "model__max_depth": optuna.distributions.IntDistribution(2, 60),
            "model__min_samples_split": optuna.distributions.IntDistribution(2, 20),
            "model__min_samples_leaf": optuna.distributions.IntDistribution(1, 10),
        },
        "GradientBoostingClassifier": {
            "model__loss": optuna.distributions.CategoricalDistribution(["log_loss", "exponential"]),
            "model__learning_rate": optuna.distributions.FloatDistribution(1e-4, 1e-1),
            "model__n_estimators": optuna.distributions.IntDistribution(10, 200),
            "model__min_samples_split": optuna.distributions.IntDistribution(2, 50),
            "model__min_samples_leaf": optuna.distributions.IntDistribution(1, 50),
            "model__max_depth": optuna.distributions.IntDistribution(1, 10),
        },
    }


def suggest_small_mlp(trial):
    """Sample the small MLP architecture, dropout and AdamW settings.

    Args:
        trial: Optuna trial used to suggest parameter values.

    Returns:
        A configuration dictionary with model_params, optimizer settings and
        a ReduceLROnPlateau scheduler specification, ready for fit_dnn.
    """
    architecture = trial.suggest_categorical("architecture", ["32", "64_32"])
    return {
        "model_params": {
            "out": 1,
            "hidden_sizes": [32] if architecture == "32" else [64, 32],
            "dropout": trial.suggest_float("dropout", 0.1, 0.3),
        },
        "optimizer_name": "AdamW",
        "optimizer_params": {
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True),
        },
        "scheduler_name": "ReduceLROnPlateau",
        "scheduler_params": {"patience": 3, "factor": 0.5},
    }


def suggest_notebook_dnn(trial, embedding=False):
    """Sample the original notebook DNN and optimizer configuration.

    Args:
        trial: Optuna trial used for architecture and optimizer suggestions.
        embedding: Also sample embedding widths for the five categorical slots.

    Returns:
        Model, optimizer and scheduler settings in the fit_dnn format.
        This search preserves the larger notebook architecture.
    """
    params = {
        "out": 1,
        "hidden_sizes": [
            trial.suggest_categorical("hidden_1", [250, 500]),
            trial.suggest_categorical("hidden_2", [100, 200]),
            trial.suggest_categorical("hidden_3", [25, 50]),
        ],
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
    }
    if embedding:
        for column in ["sex", "embarked", "cabin", "ticket", "title"]:
            params[f"{column}_emb_dim"] = trial.suggest_categorical(f"{column}_emb_dim", [1, 2, 3])
    optimizer = trial.suggest_categorical("optimizer", ["AdamW", "Adam", "SGD"])
    if optimizer == "SGD":
        opt_params = {
            "lr": trial.suggest_float("lr_sgd", 1e-3, 1e-1, log=True),
            "momentum": trial.suggest_float("momentum", 0.0, 0.95),
        }
    else:
        opt_params = {"lr": trial.suggest_float("lr_adam", 1e-4, 1e-2, log=True)}
    return {
        "model_params": params,
        "optimizer_name": optimizer,
        "optimizer_params": opt_params,
        "scheduler_name": "ReduceLROnPlateau",
        "scheduler_params": {},
    }


class MyModel(nn.Module):
    """Notebook MLP with BatchNorm, LeakyReLU and dropout hidden blocks."""

    def __init__(self, inp, out, hidden_sizes=[500, 200, 50], dropout=0.0):
        """Initialize the MLP for an already encoded feature matrix.

        Args:
            inp: Number of preprocessed input features.
            out: Number of output logits; binary classification uses one.
            hidden_sizes: Widths of the hidden layers.
            dropout: Drop probability within each hidden block.
        """
        super().__init__()
        self.inp = inp
        self.out = out
        self.hidden_sizes = [inp] + hidden_sizes
        self.dropout = dropout
        self._build_model()

    def _build_model(self):
        """Store the hidden blocks and final linear output layer in self.model."""
        layers = []
        for hs1, hs2 in zip(self.hidden_sizes, self.hidden_sizes[1:]):
            layers.extend(
                [
                nn.Linear(hs1, hs2, bias=False),
                nn.BatchNorm1d(hs2),
                nn.LeakyReLU(),
                nn.Dropout(self.dropout)
                ]
            )
        layers.append(nn.Linear(self.hidden_sizes[-1], self.out))
        self.model = nn.Sequential(*layers) 

    def forward(self, x):
        """Map a (batch, inp) feature tensor to (batch, out) raw logits."""
        return self.model(x)


class MyModelEmb(nn.Module):
    """Notebook MLP combining numeric features with five category embeddings.

    Categorical inputs must follow the fixed order Sex, Embarked, Cabin,
    Ticket, Title and contain ordinal codes from the fitted encoder.
    """

    def __init__(self, inp, out, num_slice, cat_slice, hidden_sizes=[500, 200, 50], dropout=0.0, sex_emb_dim=2, embarked_emb_dim=2, cabin_emb_dim=2, ticket_emb_dim=2, title_emb_dim=2, cat_sizes=None):
        """Initialize the dense network and category embedding tables.

        Args:
            inp: Numeric feature count plus the sum of embedding widths.
            out: Number of output logits; binary classification uses one.
            num_slice: Slice selecting numeric columns in the encoded input.
            cat_slice: Slice selecting the five ordered categorical columns.
            hidden_sizes: Widths of the dense hidden layers.
            dropout: Drop probability within each hidden block.
            sex_emb_dim: Width of the Sex embedding.
            embarked_emb_dim: Width of the Embarked embedding.
            cabin_emb_dim: Width of the Cabin embedding.
            ticket_emb_dim: Width of the Ticket embedding.
            title_emb_dim: Width of the Title embedding.
            cat_sizes: Five vocabulary sizes, each including an unknown slot.
        """
        super().__init__()
        if cat_sizes is None:
            raise ValueError("Передай размеры словарей из train encoder через cat_sizes")
        self.inp = inp
        self.out = out
        self.hidden_sizes = [inp] + hidden_sizes
        self.dropout = dropout
        self._build_model()
        self.sex_emb = nn.Embedding(cat_sizes[0], sex_emb_dim)
        self.embarked_emb = nn.Embedding(cat_sizes[1], embarked_emb_dim)
        self.cabin_emb = nn.Embedding(cat_sizes[2], cabin_emb_dim)
        self.ticket_emb = nn.Embedding(cat_sizes[3], ticket_emb_dim)
        self.title_emb = nn.Embedding(cat_sizes[4], title_emb_dim)
        self.num_slice = num_slice
        self.cat_slice = cat_slice

    def _build_model(self):
        """Build self.model for concatenated numeric features and embeddings."""
        layers = []
        for hs1, hs2 in zip(self.hidden_sizes, self.hidden_sizes[1:]):
            layers.extend(
                [
                nn.Linear(hs1, hs2, bias=False),
                nn.BatchNorm1d(hs2),
                nn.LeakyReLU(),
                nn.Dropout(self.dropout)
                ]
            )
        layers.append(nn.Linear(self.hidden_sizes[-1], self.out))
        self.model = nn.Sequential(*layers) 

    def forward(self, x):
        """Embed ordinal categories, concatenate numeric values and return logits.

        x contains numeric and ordinal columns in the fitted transformer's
        output order. Adding one maps the encoder's unknown code -1 to slot 0.
        The returned tensor has shape (batch, out).
        """
        x_num = x[:, self.num_slice]
        x_cat = x[:, self.cat_slice].to(torch.int32) + 1
        sex_vector = self.sex_emb(x_cat[:, 0])
        embarked_vector = self.embarked_emb(x_cat[:, 1])
        cabin_vector = self.cabin_emb(x_cat[:, 2])
        ticket_vector = self.ticket_emb(x_cat[:, 3])
        title_vector = self.title_emb(x_cat[:, 4])
        x = torch.cat([x_num, sex_vector, embarked_vector, cabin_vector, ticket_vector, title_vector], dim=1)
        return self.model(x)

