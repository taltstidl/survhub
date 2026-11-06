"""
SurvBoard custom models.
"""
from typing import Literal

import numpy as np
import torch
from sklearn.base import BaseEstimator
from sklearn.utils.validation import validate_data, check_is_fitted
from sksurv.base import SurvivalAnalysisMixin
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.svm import FastKernelSurvivalSVM
from sksurv.util import check_array_survival
from survivalpfn import SurvivalEstimator


class SurvBoardRandomSurvivalForest(RandomSurvivalForest):
    def fit(self, X, y, **kwargs):
        # If bootstrap is False, sksurv will raise ValueError if max_samples is also set
        if not self.bootstrap:
            self.max_samples = None
        return super().fit(X, y, **kwargs)


class SurvBoardGradientBoostingSurvivalAnalysis(GradientBoostingSurvivalAnalysis):
    def fit(self, X, y, **kwargs):
        if self.loss == 'squared' or self.loss == 'ipcwls':
            # Does not support zero survival times
            event_field, time_field = y.dtype.names
            y_time = y[time_field]
            mask = y_time == 0
            X, y = X[~mask], y[~mask]
        return super().fit(X, y, **kwargs)


class SurvBoardFastKernelSurvivalSVM(FastKernelSurvivalSVM):
    def fit(self, X, y):
        # Does not support zero survival times
        event_field, time_field = y.dtype.names
        y_time = y[time_field]
        mask = y_time == 0
        return super().fit(X[~mask], y[~mask])


class SurvivalPFNFunction:
    def __init__(self, dist, index, device, kind: Literal['survival', 'cumulative_hazard']='survival'):
        self.dist = dist
        self.index = index
        self.device = device
        self.kind = kind

    def __call__(self, t):
        t = torch.as_tensor(t, dtype=torch.float32, device=self.device)
        if self.kind == 'survival':
            return self.dist.survival_function(t)[self.index]
        else:
            return self.dist.cumulative_hazard(t)[self.index]


class SurvHubSurvivalPFN(BaseEstimator, SurvivalAnalysisMixin):
    is_tfm_ = True

    def __init__(self, device=None):
        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def fit(self, X, y):
        # Validate and extract data
        X, y = validate_data(self, X, y)
        y_event, y_time = check_array_survival(X, y)

        self.model_ = SurvivalEstimator(device=self.device)
        self.model_.fit(X=X, T=y_time.copy(), delta=y_event)

    def predict(self, X):
        check_is_fitted(self)
        dist = self.model_.predict_event_distribution(X)
        times = dist.median().detach().cpu().numpy()
        return times

    def predict_cumulative_hazard_function(self, X):
        check_is_fitted(self)
        dist = self.model_.predict_event_distribution(X)
        n_samples = X.shape[0]
        funcs = np.empty(n_samples, dtype=object)
        for i in range(n_samples):
            funcs[i] = SurvivalPFNFunction(dist, i, device=self.device, kind='cumulative_hazard')
        return funcs

    def predict_survival_function(self, X):
        check_is_fitted(self)
        dist = self.model_.predict_event_distribution(X)
        n_samples = X.shape[0]
        funcs = np.empty(n_samples, dtype=object)
        for i in range(n_samples):
            funcs[i] = SurvivalPFNFunction(dist, i, device=self.device, kind='survival')
        return funcs
