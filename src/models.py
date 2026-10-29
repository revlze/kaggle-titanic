"""Neural architectures and notebook Optuna search spaces."""

import optuna
import torch
from torch import nn

# Keep previous imports working; config-based construction now lives in utils.py.
from utils import get_pipeline as make_classical_pipeline, get_search_space as make_search_space


class SmallMLP(nn.Module):
    def __init__(self, inp, out=1, hidden_sizes=(32,), dropout=0.1):
        super().__init__()
        layers = []
        for before, after in zip([inp, *hidden_sizes], hidden_sizes):
            layers.extend([nn.Linear(before, after), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_sizes[-1], out))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def comparison_search_spaces():
    """Keep the existing RF/GB ranges; add a modest logistic-regression search."""
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
    """Preserve the original notebook search, separate from the small MLP."""
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
    def __init__(self, inp, out, hidden_sizes=[500, 200, 50], dropout=0.0):
        super().__init__()
        self.inp = inp
        self.out = out
        self.hidden_sizes = [inp] + hidden_sizes
        self.dropout = dropout
        self._build_model()

    def _build_model(self):
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
        return self.model(x)


class MyModelEmb(nn.Module):
    def __init__(self, inp, out, num_slice, cat_slice, hidden_sizes=[500, 200, 50], dropout=0.0, sex_emb_dim=2, embarked_emb_dim=2, cabin_emb_dim=2, ticket_emb_dim=2, title_emb_dim=2, cat_sizes=None):
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
        x_num = x[:, self.num_slice]
        x_cat = x[:, self.cat_slice].to(torch.int32) + 1
        sex_vector = self.sex_emb(x_cat[:, 0])
        embarked_vector = self.embarked_emb(x_cat[:, 1])
        cabin_vector = self.cabin_emb(x_cat[:, 2])
        ticket_vector = self.ticket_emb(x_cat[:, 3])
        title_vector = self.title_emb(x_cat[:, 4])
        x = torch.cat([x_num, sex_vector, embarked_vector, cabin_vector, ticket_vector, title_vector], dim=1)
        return self.model(x)

