"""
SurvBoard utility functions.
"""

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
    if model_name == 'deepweisurv':
        return False
    if model_name == 'deephit':
        return False
    if model_name == 'tabpfn':
        return False
    if model_name == 'popsicl':
        return True
    raise ValueError('Unknown model {}'.format(model_name))