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
            return self.dist.survival_function(t)[self.index].cpu().numpy()
        else:
            return self.dist.cumulative_hazard(t)[self.index].cpu().numpy()


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

    @torch.no_grad()
    def predict(self, X):
        check_is_fitted(self)
        dist = self.model_.predict_event_distribution(X)
        times = dist.median().detach().cpu().numpy()
        return times

    @torch.no_grad()
    def predict_cumulative_hazard_function(self, X):
        check_is_fitted(self)
        dist = self.model_.predict_event_distribution(X)
        n_samples = X.shape[0]
        funcs = np.empty(n_samples, dtype=object)
        for i in range(n_samples):
            funcs[i] = SurvivalPFNFunction(dist, i, device=self.device, kind='cumulative_hazard')
        return funcs

    @torch.no_grad()
    def predict_survival_function(self, X):
        check_is_fitted(self)
        dist = self.model_.predict_event_distribution(X)
        n_samples = X.shape[0]
        funcs = np.empty(n_samples, dtype=object)
        for i in range(n_samples):
            funcs[i] = SurvivalPFNFunction(dist, i, device=self.device, kind='survival')
        return funcs


class FixedFunction:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __call__(self, t):
        # Allow only fixed outputs at the same sampling times
        assert np.allclose(t, self.x)
        return self.y


class SurvHubTFMAdapter(BaseEstimator, SurvivalAnalysisMixin):
    # Based on https://github.com/dain20000222/TabularFM4Survival/blob/main/static/static_tabpfn_final.py
    is_tfm_ = True

    def __init__(self, tfm: Literal['tabpfn', 'tabicl'], eval_times: np.ndarray, random_state=None, device=None):
        assert tfm in ['tabpfn', 'tabicl'], f'Unsupported tabular foundation model {tfm}'
        self.tfm = tfm
        self._eval_times = eval_times
        self.eval_times = list(eval_times)
        self.random_state = random_state
        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _construct_train(self, X, y_event, y_time):
        # Censored samples are only observed till the censoring time, shape (n_times, n_samples)
        observed = y_event[:, None] | (self._eval_times <= y_time[:, None])
        event_happened = y_event[:, None] & (y_time[:, None] <= self._eval_times)

        rows, cols = np.nonzero(observed)
        X_tfm = np.column_stack([X[rows], self._eval_times[cols]])
        y_tfm = event_happened[rows, cols].astype(int)
        return X_tfm, y_tfm

    def _predict_survival(self, X):
        check_is_fitted(self)
        n_samples, n_times = X.shape[0], self._eval_times.shape[0]
        X_tfm = np.column_stack([np.repeat(X, n_times, axis=0), np.tile(self._eval_times, n_samples)])
        event_class_index = list(self.model_.classes_).index(1)
        p_event = self.model_.predict_proba(X_tfm)[:, event_class_index].reshape(n_samples, n_times)
        return np.minimum.accumulate(1.0 - p_event, axis=-1)

    def fit(self, X, y):
        # Validate and extract data
        X, y = validate_data(self, X, y)
        y_event, y_time = check_array_survival(X, y)
        X_tfm, y_tfm = self._construct_train(X, y_event, y_time)
        if self.tfm == 'tabpfn':
            from tabpfn import TabPFNClassifier
            self.model_ = TabPFNClassifier(ignore_pretraining_limits=True,
                                           random_state=self.random_state, device=self.device)
        if self.tfm == 'tabicl':
            from tabicl import TabICLClassifier
            self.model_ = TabICLClassifier(random_state=self.random_state, device=self.device)
        self.model_.fit(X_tfm, y_tfm)

    @torch.no_grad()
    def predict(self, X):
        # Calculate risk scores as average of (1 - survival) across multiple eval times
        return np.mean(1.0 - self._predict_survival(X), axis=-1)

    @torch.no_grad()
    def predict_cumulative_hazard_function(self, X):
        s = self._predict_survival(X)
        ch = -np.log(np.clip(s, 1e-8, 1.0))
        n_samples = X.shape[0]
        funcs = np.empty(n_samples, dtype=object)
        for i in range(n_samples):
            funcs[i] = FixedFunction(self._eval_times, ch[i])
        return funcs

    @torch.no_grad()
    def predict_survival_function(self, X):
        s = self._predict_survival(X)
        n_samples = X.shape[0]
        funcs = np.empty(n_samples, dtype=object)
        for i in range(n_samples):
            funcs[i] = FixedFunction(self._eval_times, s[i])
        return funcs
