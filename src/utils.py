import os
import random

import torch
import torch.nn as nn
import numpy as np
import torchvision
import optuna
from omegaconf import OmegaConf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from features import make_features, make_preprocess

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f'Setted seed to {seed}')

def get_optimizer(config, model):
    '''Get PyTorch optimizer'''
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = getattr(torch.optim, config.optimizer.name)(parameters, **config.optimizer.params)
    return optimizer

def get_loss(config):
    '''Get PyTorch loss from config.loss.name and config.loss.params.'''
    return getattr(nn, config.loss.name)(**config.loss.params)

def get_scheduler(config, optimizer):
    '''Get PyTorch scheduler; a null scheduler disables it.'''
    if config.scheduler is None or config.scheduler.name is None:
        return None
    return getattr(torch.optim.lr_scheduler, config.scheduler.name)(
        optimizer, **config.scheduler.params,
    )

def get_model(config, *, load_pretrained=True):
    '''Get a configured sklearn classifier, custom PyTorch model or torchvision model.'''
    params = OmegaConf.to_container(config.model.params, resolve=True)
    classifiers = {
        'GradientBoostingClassifier': GradientBoostingClassifier,
        'LogisticRegression': LogisticRegression,
    }
    if config.model.name in classifiers:
        params.setdefault('random_state', config.base.seed)
        model = classifiers[config.model.name](**params)
    elif config.model.name.startswith('custom/'):
        # Import here because models.py also exposes the original helper names.
        import models
        model = getattr(models, config.model.name.split('/')[-1])(**params)
    else:
        if not load_pretrained:
            params['weights'] = None
        model = getattr(torchvision.models, config.model.name)(**params)

    return model

def get_pipeline(config, train_dataset):
    '''Build the classical pipeline using the existing config-based model getter.'''
    if config.model.name not in {'GradientBoostingClassifier', 'LogisticRegression'}:
        raise ValueError(f'Unsupported classical model: {config.model.name}')
    model = get_model(config)
    _, numeric, categorical = make_features(train_dataset, config.features.feature_set)
    return Pipeline([
        ('preprocess', make_preprocess(numeric, categorical, **config.preprocessing)),
        ('model', model),
    ])

def get_search_space(config, estimator):
    '''Translate YAML ranges to the Optuna distributions used in the notebooks.'''
    distributions = {
        'int': optuna.distributions.IntDistribution,
        'float': optuna.distributions.FloatDistribution,
        'categorical': optuna.distributions.CategoricalDistribution,
    }
    space = {}
    for name, settings in OmegaConf.to_container(config.tuning.search_space, resolve=True).items():
        key = f'model__{name}'
        if key not in estimator.get_params():
            raise ValueError(f'Unknown tuning parameter for {config.model.name}: {name}')
        kind = settings.pop('type')
        if kind not in distributions:
            raise ValueError(f'Unknown distribution type: {kind}')
        space[key] = distributions[kind](**settings)
    if not space:
        raise ValueError('Optuna requires a non-empty tuning.search_space')
    return space
