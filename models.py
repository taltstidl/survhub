"""
SurvBoard custom models.
"""
from sklearn.base import BaseEstimator
from sklearn.utils.validation import validate_data, check_is_fitted
from sksurv.base import SurvivalAnalysisMixin
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import check_array_survival
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion


class SurvBoardRandomSurvivalForest(RandomSurvivalForest):
    def fit(self, X, y, **kwargs):
        # If bootstrap is False, sksurv will raise ValueError if max_samples is also set
        if not self.bootstrap:
            self.max_samples = None
        return super().fit(X, y, **kwargs)


class SurvBoardTabPFN(SurvivalAnalysisMixin, BaseEstimator):
    def fit(self, X, y):
        # Validate and extract data
        X, y = validate_data(self, X, y)
        y_event, y_time = check_array_survival(X, y)

        # Train TabPFN regressor only with observed events
        X_observed, y_observed = X[y_event], y_time[y_event]
        self.tabpfn_ = TabPFNRegressor.create_default_for_version(ModelVersion.V2)
        self.tabpfn_.fit(X_observed, y_observed)

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X)
        return self.tabpfn_.predict(X)
