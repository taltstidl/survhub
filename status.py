import argparse
import json
from pathlib import Path


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
    print('Number of datasets: {}'.format(num_datasets))

    # Check status of each model (subdirectory of results)
    for model_path in results_path.iterdir():
        datasets_finished = []
        for dataset in datasets:
            # Option 1: all folds were run in a single experiment
            single_path = model_path / f'{dataset}.json'
            if single_path.exists():
                if args.metric == 'none' or args.metric in json.load(single_path.open()):
                    datasets_finished.append(dataset)
            # Option 2: folds were separate experiments
            multi_paths = [model_path / f'{dataset}_{i + 1:02d}.json' for i in range(25)]
            if all(path.exists() for path in multi_paths):
                if args.metric == 'none' or all(args.metric in json.load(path.open()) for path in multi_paths):
                    datasets_finished.append(dataset)
        # datasets_finished = set(re.sub(r'_\d{2}$', '', l.stem) for l in model_path.iterdir())
        num_datasets_finished = len(datasets_finished)
        bar = '█' * num_datasets_finished + '░' * (num_datasets - num_datasets_finished)
        print(model_path.name.ljust(28), bar)
        if args.report_missing:
            datasets_missing = set(datasets) - set(datasets_finished)
            print('Missing datasets: ' + ', '.join(sorted(datasets_missing)))


if __name__ == '__main__':
    main()
