import argparse
import json
import math
from pathlib import Path


def status_symbol(model_path: Path, dataset) -> str:
    multi_paths = [model_path / f'{dataset}_{i + 1:02d}.json' for i in range(25)]
    if not all(path.exists() for path in multi_paths):
        return '░'  # dataset not finished
    has_ti_metrics, has_td_metrics = [], []
    for path in multi_paths:
        metrics = json.load(path.open())['metrics']
        has_ti_metrics.append(
            all((m in metrics) and not math.isnan(metrics[m]) for m in ['harrell_c', 'uno_c', 'static_auc']))
        has_td_metrics.append(
            all((m in metrics) and not math.isnan(metrics[m]) for m in ['dynamic_auc', 'ibs', 'antolini_c']))
    has_ti_metrics, has_td_metrics = all(has_ti_metrics), all(has_td_metrics)
    if has_ti_metrics and has_td_metrics:
        return '\033[1;36;48m█\033[1;37;0m'  # both time-independent and time-dependent metrics available
    elif has_ti_metrics:
        return '\033[1;34;48m█\033[1;37;0m'  # only time-independent metrics available
    elif has_td_metrics:
        return '\033[1;32;48m█\033[1;37;0m'  # only time-dependent metrics available
    else:
        return '░'  # some metrics missing


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser('SurvBoard status report script.')
    parser.add_argument('--report-missing', default=False, action='store_true')
    parser.add_argument('--metric', default='none', choices=['none', 'harrell_c', 'uno_c', 'auc'])
    args = parser.parse_args()

    # Create and validate paths
    results_path = Path('results')
    if not results_path.exists():
        print('Results folder not found')
        return
    summary_path = Path('summary.csv')
    if not results_path.exists():
        print('Summary sheet not found')
        return

    # Retrieve list of datasets
    # This implicitly assumes that no datasets include a comma in their name
    datasets = list(l.split(',')[0] for l in open(summary_path))[1:]  # for header
    num_datasets = len(datasets)

    # Check status of each model (subdirectory of results)
    for model_path in results_path.iterdir():
        bar = ''.join(status_symbol(model_path, dataset) for dataset in datasets)
        bar_clean = ''.join(b for b in bar if b in {'░', '█'})
        datasets_finished = [d for d, b in zip(datasets, bar_clean) if b != '░']
        num_datasets_finished = len(datasets_finished)
        print(model_path.name.ljust(28), bar, f'({num_datasets_finished}/{num_datasets})')
        if args.report_missing:
            datasets_missing = set(datasets) - set(datasets_finished)
            print('Missing datasets: ' + ', '.join(sorted(datasets_missing)))


if __name__ == '__main__':
    main()
