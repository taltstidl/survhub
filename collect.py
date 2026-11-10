import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_result(results_dir, model_name, dataset_name):
    # Check whether single results file is available (either ran as such or previously compiled)
    result_path = results_dir / model_name / f'{dataset_name}.json'
    if result_path.exists():
        return json.load(result_path.open())
    # Check whether all separate results are available
    result_paths = [results_dir / model_name / f'{dataset_name}_{i:02d}.json' for i in range(1, 26)]
    if all(result_path.exists() for result_path in result_paths):
        fold_results = [json.load(result_path.open()) for result_path in result_paths]
        # Compile and save single results file
        scores = [r['c_index'] for r in fold_results]
        fit_times = [r['fit_time'] for r in fold_results]
        predict_times = [r['predict_time'] for r in fold_results]
        results = {
            'model': fold_results[0]['model'],
            'dataset': fold_results[0]['dataset'],
            'c_index': np.mean(scores),
            'c_indices': scores,
            'fit_time': np.mean(fit_times),
            'fit_times': fit_times,
            'predict_time': np.mean(predict_times),
            'predict_times': predict_times,
        }
        with result_path.open('w') as f:
            json.dump(results, f)
        return results
    print(f'No results found for {model_name} and {dataset_name}.')
    return None


def collect_results(results_dir, models, datasets):
    results_df = []
    for dataset in datasets.itertuples():
        for model in models:
            results = load_result(results_dir, model, dataset.name)
            if results is None:
                continue
            # Add results for each fold
            for i in range(25):
                results_df.append({
                    'dataset': dataset.name,
                    'model': model,
                    'fold': i,
                    'c_index': results['c_indices'][i],
                    'fit_time': results['fit_times'][i],
                    'predict_time': results['predict_times'][i],
                })
    return pd.DataFrame(results_df)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser('SurvBoard results collection script.')
    parser.add_argument('--results-dir', type=Path, default='server/results')
    parser.add_argument('--configs-dir', type=Path, default='server/configs')
    args = parser.parse_args()

    # Load dataset metadata
    summary_df = pd.read_csv('summary.csv')
    summary_df['n_cols'] = summary_df['n_num_cols'] + summary_df['n_cat_cols']

    # Collect results for small datasets only
    print('Processing small datasets...')
    # 'gbse', 'gbse-tuned'
    models_small = ['coxnet', 'coxnet-tuned', 'rsf', 'rsf-tuned', 'gbse', 'gbse-tuned', 'ssvm', 'ssvm-tuned',
                    'deepsurv', 'rankdeepsurv', 'deepweisurv1', 'deepweisurv2', 'deephit', 'tabpfn', 'popsicl']
    datasets_small = summary_df[(summary_df['n_samples'] <= 1000) & (summary_df['n_cols'] <= 100)]
    results_df = collect_results(args.results_dir, models_small, datasets_small)
    results_df.to_csv('results_small.csv', index=False)

    # Collect results for large datasets only
    print('Processing large datasets...')
    models_large = ['coxnet', 'coxnet-tuned', 'rsf', 'gbse', 'ssvm',
                    'deepsurv', 'rankdeepsurv', 'deepweisurv1', 'deepweisurv2', 'deephit']
    datasets_large = summary_df[(summary_df['n_samples'] > 1000) | (summary_df['n_cols'] > 100)]
    results_df = collect_results(args.results_dir, models_large, datasets_large)
    results_df.to_csv('results_large.csv', index=False)


if __name__ == '__main__':
    main()
