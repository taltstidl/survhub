"""
SurvBoard benchmarking script.
"""
import argparse
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from optuna_integration.sklearn import OptunaSearchCV
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.svm import FastKernelSurvivalSVM
from sksurv.util import Surv
from tabicl import TabICLSurver
from torch_survival.models import DeepSurv, DeepHit, RankDeepSurv, DeepWeiSurv

from models import SurvBoardRandomSurvivalForest, SurvBoardTabPFN
from utils import is_risk_model, is_tfm


def load_coxnet(y_event, seed, tuned=True):
    estimator = CoxnetSurvivalAnalysis()
    if tuned:
        params = {
            'alpha_min_ratio': FloatDistribution(1e-5, 1e0, log=True),
            'l1_ratio': FloatDistribution(0.0, 1.0),
        }
        folds = StratifiedKFold(n_splits=5).split(np.arange(y_event.shape[0]), y_event)
        estimator = OptunaSearchCV(estimator, params, cv=folds, n_trials=50, random_state=seed)
    return estimator


def load_rsf(y_event, seed, tuned=True):
    estimator = SurvBoardRandomSurvivalForest(n_estimators=50, random_state=seed)
    if tuned:
        # Taken from TabArena https://arxiv.org/pdf/2506.16791
        params = {
            'max_features': FloatDistribution(0.4, 1.0),
            'max_samples': FloatDistribution(0.5, 1.0),
            'min_samples_split': IntDistribution(2, 4, log=True),
            'bootstrap': CategoricalDistribution([True, False]),
            # 'min_impurity_decrease': FloatDistribution(1e-5, 1e-3, log=True),
        }
        folds = StratifiedKFold(n_splits=5).split(np.arange(y_event.shape[0]), y_event)
        estimator = OptunaSearchCV(estimator, params, cv=folds, n_trials=50, random_state=seed)
    return estimator


def load_gbse(y_event, seed, tuned=True):
    estimator = GradientBoostingSurvivalAnalysis(n_estimators=50, random_state=seed)
    if tuned:
        params = {
            'loss': CategoricalDistribution(['coxph', 'squared']),
            'learning_rate': FloatDistribution(1e-3, 1e0, log=True),
            'subsample': CategoricalDistribution([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            'max_features': FloatDistribution(0.4, 1.0),
            'min_samples_split': IntDistribution(2, 4, log=True),
        }
        folds = StratifiedKFold(n_splits=5).split(np.arange(y_event.shape[0]), y_event)
        estimator = OptunaSearchCV(estimator, params, cv=folds, n_trials=50, random_state=seed)
    return estimator


def load_ssvm(y_event, seed, tuned=False):
    estimator = FastKernelSurvivalSVM(random_state=seed)
    if tuned:
        params = {
            'alpha': FloatDistribution(1e-5, 1e5),
            'rank_ratio': FloatDistribution(0.0, 1.0),
            'fit_intercept': CategoricalDistribution([True, False]),
            'kernel': CategoricalDistribution(['linear', 'rbf']),
            'gamma': FloatDistribution(1e-5, 1e1, log=True),
        }
        folds = StratifiedKFold(n_splits=5).split(np.arange(y_event.shape[0]), y_event)
        estimator = OptunaSearchCV(estimator, params, cv=folds, n_trials=50, random_state=seed)
    return estimator


def load_deepsurv(y_event, seed, tuned=True):
    # Internally tuned using Optuna
    return DeepSurv(random_state=seed, device='cuda')


def load_deephit(y_event, seed, tuned=True):
    # Internally tuned using Optuna
    return DeepHit(random_state=seed, device='cuda')


def load_rankdeepsurv(y_event, seed, tuned=True):
    # Internally tuned using Optuna
    n_epochs = int(math.ceil(2000 / y_event.shape[0]))
    print('Training with', n_epochs, 'epochs')
    return RankDeepSurv(random_state=seed, device='cuda', n_epochs=n_epochs)


def load_deepweisurv(y_event, seed, tuned=True):
    return DeepWeiSurv(random_state=seed, device='cuda')


def load_tabpfn(y_event, seed, tuned=True):
    return SurvBoardTabPFN()


def load_popsicl(y_event, seed, tuned=True):
    return TabICLSurver()


def evaluate_model(model_name, dataset_name, tuned):
    # Load dataset
    data_path = Path('data', 'export', dataset_name, 'data.csv')
    df = pd.read_csv(data_path)
    df['event'] = df['event'].astype(bool)
    # Encode dataset
    if is_tfm(model_name):
        enc_cat = Pipeline(steps=[
            ('ore', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-1))])
    else:
        enc_cat = Pipeline(steps=[('ohe', OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore'))])
    sel_cat = make_column_selector(pattern='^fac\\_')
    enc_num = Pipeline(steps=[('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    sel_num = make_column_selector(pattern='^num\\_')
    enc_df = ColumnTransformer(transformers=[('ord', enc_cat, sel_cat), ('s', enc_num, sel_num)])
    # Perform 5-fold cross validation (repeated five times with different seeds)
    scores, fit_times, predict_times = [], [], []
    configs = []
    for seed in [1, 2, 3, 4, 5]:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for train_idx, test_idx in cv.split(np.arange(df.shape[0]), df['event'].to_numpy()):
            # Split into training and testing
            X_train = enc_df.fit_transform(df.iloc[train_idx, :])
            X_test = enc_df.transform(df.iloc[test_idx, :])
            y_train = Surv.from_dataframe('event', 'time', df.iloc[train_idx, :])
            y_test = Surv.from_dataframe('event', 'time', df.iloc[test_idx, :])
            # Load and score model
            # y_event needed to properly compute nested folds stratified by event
            model = globals()['load_{}'.format(model_name)](y_train['event'], seed, tuned)
            start_time = time.perf_counter()
            model.fit(X_train, y_train)
            fit_time = time.perf_counter() - start_time
            start_time = time.perf_counter()
            predictions = model.predict(X_test)
            predict_time = time.perf_counter() - start_time
            if is_risk_model(model_name, model):
                score = concordance_index_censored(y_test['event'], y_test['time'], predictions)[0]
            else:
                score = concordance_index_censored(y_test['event'], y_test['time'], -predictions)[0]
            # Collect all metrics
            scores.append(score)
            fit_times.append(fit_time)
            predict_times.append(predict_time)
            # Extract hyperparameter configuration
            config = {}
            if hasattr(model, 'best_estimator_'):
                config = model.best_estimator_.get_params()
            if hasattr(model, 'get_optuna_params'):
                config = model.get_optuna_params()
            configs.append(config)
    # Save results
    results = {
        'model': model_name + '-tuned' if tuned else model_name,
        'dataset': dataset_name,
        'c_index': np.mean(scores),
        'c_indices': scores,
        'fit_time': np.mean(fit_times),
        'fit_times': fit_times,
        'predict_time': np.mean(predict_times),
        'predict_times': predict_times,
    }
    results_path = Path('results', model_name + '-tuned' if tuned else model_name)
    results_path.mkdir(parents=True, exist_ok=True)
    with (results_path / (dataset_name + '.json')).open('w') as f:
        json.dump(results, f)
    # Save hyperparameter configurations
    if any(configs):
        configs_path = Path('configs', model_name + '-tuned' if tuned else model_name)
        configs_path.mkdir(parents=True, exist_ok=True)
        with (configs_path / (dataset_name + '.json')).open('w') as f:
            json.dump(configs, f)


def main():
    summary_df = pd.read_csv(Path('summary.csv'))
    datasets = [str(i) for i in range(summary_df.shape[0])] + summary_df['name'].tolist()
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Survboard benchmarking script')
    models = ['coxnet', 'rsf', 'gbse', 'ssvm', 'deepsurv', 'rankdeepsurv', 'deepweisurv', 'deephit', 'tabpfn',
              'popsicl']
    parser.add_argument('-m', '--model', choices=models, required=True)
    parser.add_argument('-d', '--dataset', choices=datasets, required=True)
    parser.add_argument('--tuned', default=False, action='store_true', help='Use tuned hyperparameters')
    parser.add_argument('--small-only', default=False, action='store_true', help='Only evaluate small datasets')
    args = parser.parse_args()
    # dataset_names = ['ovarian', 'glioma', 'Bergamaschi']
    # Train and evaluate model
    optuna.logging.disable_default_handler()
    warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)
    dataset = summary_df.iloc[int(args.dataset)]['name'] if args.dataset.isdigit() else args.dataset
    evaluate_model(args.model, dataset, args.tuned)


if __name__ == '__main__':
    main()
