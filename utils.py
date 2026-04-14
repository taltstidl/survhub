"""
SurvBoard utility functions.
"""
import json
from pathlib import Path


def load_config(model_name, dataset_name, tuned, fold):
    configs_path = Path('configs', model_name + '-tuned' if tuned else model_name)
    single_path = configs_path / f'{dataset_name}_{fold:02d}.json'
    if single_path.exists():
        with open(single_path, 'r') as f:
            return json.load(f)[0]
    multi_path = configs_path / f'{dataset_name}.json'
    if multi_path.exists():
        with open(multi_path, 'r') as f:
            return json.load(f)[fold - 1]
    raise FileNotFoundError(f'No config file at {single_path} or {multi_path}')


def is_tfm(model_name):
    return model_name == 'tabpfn' or model_name == 'popsicl'


def is_risk_model(model_name, model):
    if model_name == 'coxnet':
        return True
    if model_name == 'rsf':
        return True
    if model_name == 'gbse':
        if hasattr(model, 'best_estimator_'):
            params = model.best_estimator_.get_params()
        else:
            params = model.get_params()
        return params['loss'] == 'coxph'
    if model_name == 'ssvm':
        if hasattr(model, 'best_estimator_'):
            params = model.best_estimator_.get_params()
        else:
            params = model.get_params()
        return params['rank_ratio'] == 1.0
    if model_name == 'deepsurv':
        return True
    if model_name == 'rankdeepsurv':
        return False
    if model_name == 'deepweisurv1' or model_name == 'deepweisurv2':
        return False
    if model_name == 'deephit':
        return False
    if model_name == 'tabpfn':
        return False
    if model_name == 'popsicl':
        return True
    raise ValueError('Unknown model {}'.format(model_name))


def style_boxplot(bp, colors):
    # Sprinkles a bit more color onto the default box plot style
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.33)
        patch.set_edgecolor(color)
    for i, color in enumerate(colors):
        bp['whiskers'][i * 2].set_color(color)
        bp['whiskers'][i * 2 + 1].set_color(color)
        bp['caps'][i * 2].set_color(color)
        bp['caps'][i * 2 + 1].set_color(color)
        bp['medians'][i].set_color(color)
        bp['fliers'][i].set_markerfacecolor(color)
        bp['fliers'][i].set_markeredgecolor('none')
        bp['fliers'][i].set_markersize(2)
