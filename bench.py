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
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc
from sksurv.util import Surv
from tabicl import TabICLSurver
from torch_survival.models import DeepSurv, DeepHit, RankDeepSurv, DeepWeiSurv

from models import SurvBoardRandomSurvivalForest, SurvBoardGradientBoostingSurvivalAnalysis, \
    SurvBoardFastKernelSurvivalSVM, SurvBoardTabPFN
from utils import is_risk_model, is_tfm, load_config


def load_coxnet(y_event, seed, tuned=True, params=None):
    estimator = CoxnetSurvivalAnalysis()
    if tuned:
        if params is not None:
            return CoxnetSurvivalAnalysis(**params)
        params = {
            'alpha_min_ratio': FloatDistribution(1e-5, 1e0, log=True),
            'l1_ratio': FloatDistribution(0.0, 1.0),
        }
        folds = StratifiedKFold(n_splits=5).split(np.arange(y_event.shape[0]), y_event)
        estimator = OptunaSearchCV(estimator, params, cv=folds, n_trials=50, random_state=seed)
    return estimator


def load_rsf(y_event, seed, tuned=True, params=None):
    estimator = SurvBoardRandomSurvivalForest(n_estimators=50, random_state=seed)
    if tuned:
        if params is not None:
            return SurvBoardRandomSurvivalForest(**params)
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


def load_gbse(y_event, seed, tuned=True, params=None):
    estimator = SurvBoardGradientBoostingSurvivalAnalysis(n_estimators=50, random_state=seed)
    if tuned:
        if params is not None:
            return SurvBoardGradientBoostingSurvivalAnalysis(**params)
        params = {
            'loss': CategoricalDistribution(['coxph', 'squared']),
            'learning_rate': FloatDistribution(1e-3, 1e-1, log=True),
            'subsample': CategoricalDistribution([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            'max_features': FloatDistribution(0.4, 1.0),
            'min_samples_split': IntDistribution(2, 4, log=True),
        }
        folds = StratifiedKFold(n_splits=5).split(np.arange(y_event.shape[0]), y_event)
        estimator = OptunaSearchCV(estimator, params, cv=folds, n_trials=50, random_state=seed)
    return estimator


def load_ssvm(y_event, seed, tuned=False, params=None):
    estimator = SurvBoardFastKernelSurvivalSVM(random_state=seed)
    if tuned:
        if params is not None:
            return SurvBoardFastKernelSurvivalSVM(**params)
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


def load_deepweisurv1(y_event, seed, tuned=True):
    return DeepWeiSurv(n_dists=1, random_state=seed, device='cuda')


def load_deepweisurv2(y_event, seed, tuned=True):
    return DeepWeiSurv(n_dists=2, random_state=seed, device='cuda')


def load_tabpfn(y_event, seed, tuned=True):
    return SurvBoardTabPFN()


def load_popsicl(y_event, seed, tuned=True):
    return TabICLSurver()


def evaluate_model(model_name, dataset_name, tuned, fold=None):
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
    scores_harrell_c, scores_uno_c, scores_auc, fit_times, predict_times = [], [], [], [], []
    configs = []
    experiment_i = 0
    for seed in [1, 2, 3, 4, 5]:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for train_idx, test_idx in cv.split(np.arange(df.shape[0]), df['event'].to_numpy()):
            # Check whether this is the correct experiment to run, if given
            experiment_i += 1
            if fold is not None and experiment_i != fold:
                continue
            # Split into training and testing
            X_train = enc_df.fit_transform(df.iloc[train_idx, :])
            X_test = enc_df.transform(df.iloc[test_idx, :])
            y_train = Surv.from_dataframe('event', 'time', df.iloc[train_idx, :])
            y_test = Surv.from_dataframe('event', 'time', df.iloc[test_idx, :])
            # Load and score model
            # y_event needed to properly compute nested folds stratified by event
            params = load_config(model_name, dataset_name, tuned, experiment_i, missing_ok=not tuned)
            model = globals()['load_{}'.format(model_name)](y_train['event'], seed, tuned, params)
            start_time = time.perf_counter()
            model.fit(X_train, y_train)
            fit_time = time.perf_counter() - start_time
            start_time = time.perf_counter()
            predictions = model.predict(X_test)
            predict_time = time.perf_counter() - start_time
            # Compute all relevant metrics
            risk_scores = predictions if is_risk_model(model_name, model) else -predictions
            harrell_c = concordance_index_censored(y_test['event'], y_test['time'], risk_scores)[0]
            horizon = np.quantile(y_train['time'], 0.95)
            uno_c = concordance_index_ipcw(y_train, y_test, risk_scores, tau=horizon)[0]
            mask = y_test['time'] > horizon
            y_test['event'][mask] = False
            y_test['time'][mask] = horizon
            times = np.quantile(np.unique(y_test['time']), np.linspace(0.05, 0.95, 10))
            _, auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, times)
            # Collect all metrics
            scores_harrell_c.append(harrell_c)
            scores_uno_c.append(uno_c)
            scores_auc.append(auc)
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
        'harrell_c': np.mean(scores_harrell_c),
        'harrell_cs': scores_harrell_c,
        'uno_c': np.mean(scores_uno_c),
        'uno_cs': scores_uno_c,
        'auc': np.mean(scores_auc),
        'aucs': scores_auc,
        'fit_time': np.mean(fit_times),
        'fit_times': fit_times,
        'predict_time': np.mean(predict_times),
        'predict_times': predict_times,
    }
    json_file = f'{dataset_name}.json' if fold is None else f'{dataset_name}_{fold:02d}.json'
    results_path = Path('results', model_name + '-tuned' if tuned else model_name)
    results_path.mkdir(parents=True, exist_ok=True)
    full_path = results_path / json_file
    if full_path.exists():
        with full_path.open('r') as f:
            existing_results = json.load(f)
        results = results | existing_results
    with full_path.open('w') as f:
        json.dump(results, f)
    # Save hyperparameter configurations
    if any(configs):
        configs_path = Path('configs', model_name + '-tuned' if tuned else model_name)
        configs_path.mkdir(parents=True, exist_ok=True)
        with (configs_path / json_file).open('w') as f:
            json.dump(configs, f)


def main():
    summary_df = pd.read_csv(Path('summary.csv'))
    datasets = [str(i) for i in range(summary_df.shape[0])] + summary_df['name'].tolist()
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Survboard benchmarking script')
    models = ['coxnet', 'rsf', 'gbse', 'ssvm', 'deepsurv', 'rankdeepsurv', 'deepweisurv1', 'deepweisurv2', 'deephit',
              'tabpfn', 'popsicl']
    parser.add_argument('-m', '--model', choices=models, required=True)
    parser.add_argument('-d', '--dataset', choices=datasets, required=True)
    parser.add_argument('-i', '--fold', choices=range(1, 26), type=int)
    parser.add_argument('--tuned', default=False, action='store_true', help='Use tuned hyperparameters')
    parser.add_argument('--small-only', default=False, action='store_true', help='Only evaluate small datasets')
    args = parser.parse_args()
    # dataset_names = ['ovarian', 'glioma', 'Bergamaschi']
    # Train and evaluate model
    optuna.logging.disable_default_handler()
    warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)
    dataset = summary_df.iloc[int(args.dataset)]['name'] if args.dataset.isdigit() else args.dataset
    evaluate_model(args.model, dataset, args.tuned, args.fold)


if __name__ == '__main__':
    main()
