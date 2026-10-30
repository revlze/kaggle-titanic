import os
import random

import torch
import torch.nn as nn
import numpy as np
import torchvision
import optuna
from omegaconf import OmegaConf
from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

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
        'XGBClassifier': XGBClassifier,
        'CatBoostClassifier': CatBoostClassifier,
        'LGBMClassifier': LGBMClassifier,
    }
    if config.model.name in classifiers:
        if config.model.name == 'CatBoostClassifier':
            # Use the sklearn seed alias so the ensemble can assign distinct seeds.
            if 'random_seed' in params:
                params['random_state'] = params.pop('random_seed')
            params.setdefault('verbose', False)
            params.setdefault('allow_writing_files', False)
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
    if config.model.name not in {
        'GradientBoostingClassifier', 'LogisticRegression', 'CatBoostClassifier', 'XGBClassifier', 'LGBMClassifier',
    }:
        raise ValueError(f'Unsupported classical model: {config.model.name}')
    model = get_model(config)
    n_models = OmegaConf.select(config, 'ensemble.n_models', default=1)
    if type(n_models) is not int or n_models < 1:
        raise ValueError('ensemble.n_models must be a positive integer')
    method = OmegaConf.select(config, 'ensemble.method', default='soft')
    if method not in {'soft', 'hard', 'stacking'}:
        raise ValueError('ensemble.method must be soft, hard or stacking')
    _, numeric, categorical = make_features(train_dataset, config.features.feature_set)
    preprocess = make_preprocess(numeric, categorical, **config.preprocessing)
    if n_models > 1:
        seeds = np.random.RandomState(config.base.seed).randint(np.iinfo(np.int32).max, size=n_models)
        members = []
        for index, seed in enumerate(seeds):
            member = clone(model).set_params(random_state=int(seed))
            if method == 'stacking':
                # cross_val_predict must fit preprocessing inside each stacking fold.
                member = Pipeline([('preprocess', clone(preprocess)), ('model', member)])
            members.append((f'member_{index}', member))
        if method == 'stacking':
            folds = OmegaConf.select(config, 'ensemble.stacking_folds', default=3)
            if type(folds) is not int or folds < 2:
                raise ValueError('ensemble.stacking_folds must be an integer >= 2')
            meta_model = OmegaConf.select(config, 'ensemble.meta_model', default=OmegaConf.create({
                'name': 'LogisticRegression', 'params': {'C': 1.0, 'max_iter': 2000},
            }))
            meta_config = OmegaConf.create({'base': {'seed': config.base.seed},
                                           'model': OmegaConf.to_container(meta_model, resolve=True)})
            model = StackingClassifier(
                estimators=members, final_estimator=get_model(meta_config), stack_method='predict_proba',
                cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.base.seed), n_jobs=1,
            )
            return Pipeline([('model', model)])
        model = VotingClassifier(estimators=members, voting=method, n_jobs=1)
    return Pipeline([
        ('preprocess', preprocess),
        ('model', model),
    ])

def get_model_report(model):
    '''Return JSON-ready base parameters and the actual fitted ensemble seeds.'''
    if isinstance(model, (BaggingClassifier, VotingClassifier, StackingClassifier)):
        members = [member.named_steps['model'] if isinstance(member, Pipeline) else member
                   for member in model.estimators_]
        parameters = members[0].get_params()
        # Each fitted member has its own seed; there is no single model seed.
        parameters.pop('random_state', None)
        method = 'stacking' if isinstance(model, StackingClassifier) else getattr(model, 'voting', 'soft')
        ensemble = {
            'n_models': len(members), 'method': method,
            'aggregation': {'soft': 'mean_probability', 'hard': 'majority_vote', 'stacking': 'meta_model'}[method],
            'random_states': [member.get_params()['random_state'] for member in members],
        }
        if method == 'stacking':
            ensemble.update(stacking_folds=model.cv.n_splits, meta_model={
                'name': type(model.final_estimator_).__name__, 'params': model.final_estimator_.get_params(),
            })
        return {'parameters': parameters, 'ensemble': ensemble}
    return {'parameters': model.get_params(), 'ensemble': {'n_models': 1}}

def predict_classifier(model, X):
    '''Return labels and optional survival probabilities; hard voting has no probabilities.'''
    predictions = model.predict(X)
    probabilities = None
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)[:, list(model.classes_).index(1)]
    return predictions, probabilities

def get_search_space(config, estimator):
    '''Translate YAML ranges to the Optuna distributions used in the notebooks.'''
    distributions = {
        'int': optuna.distributions.IntDistribution,
        'float': optuna.distributions.FloatDistribution,
        'categorical': optuna.distributions.CategoricalDistribution,
    }
    space = {}
    model = estimator.named_steps['model']
    if isinstance(model, (VotingClassifier, StackingClassifier)):
        name, base_model = model.estimators[0]
        prefix = f'model__{name}__'
        if isinstance(base_model, Pipeline):
            base_model = base_model.named_steps['model']
            prefix += 'model__'
    else:
        base_model = model.estimator if isinstance(model, BaggingClassifier) else model
        prefix = 'model__estimator__' if isinstance(model, BaggingClassifier) else 'model__'
    known_parameters = set(base_model.get_params())
    # CatBoost lists only explicitly supplied parameters in get_params().
    if isinstance(base_model, CatBoostClassifier):
        import inspect
        known_parameters.update(inspect.signature(CatBoostClassifier).parameters)
    for name, settings in OmegaConf.to_container(config.tuning.search_space, resolve=True).items():
        key = prefix + name
        if name not in known_parameters:
            raise ValueError(f'Unknown tuning parameter for {config.model.name}: {name}')
        if isinstance(model, (BaggingClassifier, VotingClassifier, StackingClassifier)) and name in {'random_state', 'random_seed'}:
            raise ValueError('Ensemble seeds come from base.seed; do not tune random_state/random_seed')
        kind = settings.pop('type')
        if kind not in distributions:
            raise ValueError(f'Unknown distribution type: {kind}')
        space[key] = distributions[kind](**settings)
    if not space:
        raise ValueError('Optuna requires a non-empty tuning.search_space')
    return space
