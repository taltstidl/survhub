"""
SurvBoard benchmarking script.
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
from optuna_integration.sklearn import OptunaSearchCV
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.svm import FastKernelSurvivalSVM
from sksurv.util import Surv


def load_cph(y_event, seed, tuned=True):
    estimator = CoxPHSurvivalAnalysis()
    if tuned:
        params = {
            'alpha': FloatDistribution(1e-4, 1e1, log=True),
        }
        folds = StratifiedKFold(n_splits=5).split(np.arange(y_event.shape[0]), y_event)
        estimator = OptunaSearchCV(estimator, params, cv=folds, n_trials=50, random_state=seed)
    return estimator


def load_rsf(y_event, seed, tuned=True):
    estimator = RandomSurvivalForest(n_estimators=50, random_state=seed)
    if tuned:
        # Taken from TabArena https://arxiv.org/pdf/2506.16791
        params = {
            'max_features': FloatDistribution(0.4, 1.0),
            'max_samples': FloatDistribution(0.5, 1.0),
            'min_samples_split': IntDistribution(2, 4, log=True),
            'bootstrap': CategoricalDistribution([True, False]),
            #'min_impurity_decrease': FloatDistribution(1e-5, 1e-3, log=True),
        }
        folds = StratifiedKFold(n_splits=5).split(np.arange(y_event.shape[0]), y_event)
        estimator = OptunaSearchCV(estimator, params, cv=folds, n_trials=50, random_state=seed)
    return estimator


def load_ssvm(y_event, seed, tuned=False):
    estimator = FastKernelSurvivalSVM(random_state=seed)
    if tuned:
        params = {
            'alpha': FloatDistribution(1e-5, 1e1),
            'rank_ratio': FloatDistribution(0.0, 1.0),
            'fit_intercept': CategoricalDistribution([True, False]),
            'kernel': CategoricalDistribution(['linear', 'rbf']),
            'gamma': FloatDistribution(1e-5, 1e1, log=True),
        }
        folds = StratifiedKFold(n_splits=5).split(np.arange(y_event.shape[0]), y_event)
        estimator = OptunaSearchCV(estimator, params, cv=folds, n_trials=50, random_state=seed)
    return estimator


def evaluate_model(model_name, dataset_name, tuned):
    # Load dataset
    data_path = Path('data', 'export', dataset_name, 'data.csv')
    df = pd.read_csv(data_path)
    # Encode dataset
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
            score = concordance_index_censored(y_test['event'], y_test['time'], predictions)[0]
            # Collect all metrics
            scores.append(score)
            fit_times.append(fit_time)
            predict_times.append(predict_time)
            # Extract hyperparameter configuration
            config = {}
            if hasattr(model, 'best_estimator_'):
                config = model.best_estimator_.get_params()
            configs.append(config)
    # Prepare results as dict
    result = {
        'model': model_name,
        'dataset': dataset_name,
        'c_index': np.mean(scores),
        'fit_time': np.mean(fit_times),
        'predict_time': np.mean(predict_times),
        **{'c_index_{}'.format(i): s for i, s in enumerate(scores)},
        **{'fit_time_{}'.format(i): t for i, t in enumerate(fit_times)},
        **{'predict_time_{}'.format(i): t for i, t in enumerate(predict_times)},
    }
    # Save hyperparameter configurations
    if any(configs):
        configs_path = Path('configs', model_name + '-tuned' if tuned else model_name)
        configs_path.mkdir(parents=True, exist_ok=True)
        with (configs_path / (dataset_name + '.json')).open('w') as f:
            json.dump(configs, f)
    return result


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Survboard benchmarking script')
    models = ['cph', 'rsf', 'ssvm', 'deepsurv', 'rankdeepsurv', 'deepweisurv', 'dpwte', 'deephit', 'tabpfn', 'popsicl']
    parser.add_argument('-m', '--model', choices=models, required=True)
    parser.add_argument('--tuned', default=False, action='store_true', help='Use tuned hyperparameters')
    parser.add_argument('--small-only', default=False, action='store_true', help='Only evaluate small datasets')
    parser.add_argument('--parallel', default=False, action='store_true', help='Only for CPU models')
    args = parser.parse_args()
    # If present, load existing results file
    results_path = Path('{}.csv'.format(args.model))
    results = pd.read_csv(results_path).to_dict('records') if results_path.exists() else []
    # Iterate over all datasets for processing
    summary_df = pd.read_csv(Path('summary.csv'))
    summary_df = summary_df[~summary_df['name'].isin(r['dataset'] for r in results)]
    # dataset_names = summary_df['dataset'].tolist()
    dataset_names = ['ovarian', 'glioma', 'Bergamaschi']
    if args.parallel:
        with ProcessPoolExecutor() as executor:
            jobs = [executor.submit(evaluate_model, args.model, name, args.tuned) for name in dataset_names]
            for job in as_completed(jobs):
                results.append(job.result())
                pd.DataFrame(results).to_csv(results_path, index=False)
    else:
        for dataset_name in dataset_names:
            results.append(evaluate_model(args.model, dataset_name, args.tuned))
            pd.DataFrame(results).to_csv(results_path, index=False)


if __name__ == '__main__':
    main()
