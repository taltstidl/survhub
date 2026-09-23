"""
SurvBoard utility functions.
"""
import numpy as np

model_metadata = {
    'coxnet': {'is_tfm': False, 'is_risk': True},
    'rsf': {'is_tfm': False, 'is_risk': True},
    'gbse': {'is_tfm': False, 'is_risk': lambda p: p['loss'] == 'coxph'},
    'ssvm': {'is_tfm': False, 'is_risk': lambda p: p['rank_ratio'] == 1.0},
    'deepsurv': {'is_tfm': False, 'is_risk': True},
    'rankdeepsurv': {'is_tfm': False, 'is_risk': False},
    'deepweisurv': {'is_tfm': False, 'is_risk': False},
    'deephit': {'is_tfm': False, 'is_risk': False},
    'tabpfn': {'is_tfm': True, 'is_risk': False},
    'popsicl': {'is_tfm': True, 'is_risk': True},
}


def is_tfm(model_name):
    assert model_name in model_metadata.keys(), f'No metadata for {model_name}'
    return model_metadata[model_name]['is_tfm']


def is_risk_model(model_name, model):
    assert model_name in model_metadata.keys(), f'No metadata for {model_name}'
    is_risk = model_metadata[model_name]['is_risk']
    if callable(is_risk):
        p = model.best_estimator_.get_params() if hasattr(model, 'best_estimator_') else model.get_params()
        is_risk = is_risk(p)
    return is_risk


def is_time_independent(model):
    return hasattr(model, 'predict') and callable(model.predict)


def is_time_dependent(model):
    has_cumulative_hazard = hasattr(model, 'predict_cumulative_hazard_function') and callable(
        model.predict_cumulative_hazard_function)
    has_survival = hasattr(model, 'predict_survival_function') and callable(model.predict_survival_function)
    return has_cumulative_hazard and has_survival


def prepare_eval_dataset(y_train, y_test):
    # IPCW weighting requires that t <= max(y_train.t | y_train.e == True) for all t in y_test and eval_times
    # Otherwise the probability of censoring is undefined
    horizon = y_train[y_train['event']]['time'].max()
    # Truncate and artificially recensor test data beyond the horizon
    y_test_masked = y_test.copy()
    mask = y_test['time'] > horizon
    y_test_masked['event'][mask] = False
    y_test_masked['time'][mask] = horizon
    # Construct evaluation times constrained by first event time and last observed time
    min_time = y_test_masked[y_test_masked['event']]['time'].min()
    max_time = y_test_masked['time'].max() - 1e-8
    eval_times = np.linspace(min_time, max_time, 10)
    return y_test_masked, eval_times


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
